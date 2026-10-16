#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""実機試行の bag で、押し込み直前の補正 (press_in_refine) をやり直す。

hover で止まっている区間 (ログの最後の press_in_refine から押し込み開始
まで) の画像で実機ノードと同じく手だけを検出し、3 フレームずつ掌を選んで
``spik.refine_post_process`` で押し込み姿勢を解き直す。補正量の上限
(``REFINE_MAX_*``) は外して、必要だった補正量 (目標の位置・回転の変化) と
IK が解けるかを出し、実機ノードの上限で採用されるか (``*``) も付ける。
手の選び方は次の条件で比べる。

  node  実機ノードと同じ (PRESS_IN_REFINE_* の距離・score・法線)
  flip  node + 法線が 135 度超 (掌と甲の取り違え) なら裏返して使う
  wide  flip + 距離 350mm 以内

``--warmup approach`` (既定) は実機ノードと同じく接近の開始から Hands を
走らせて追跡を温める。``--warmup 0`` は補正のときだけ Hands を呼ぶ
(2026-10-16 までの実機ノード)。

入力は試行ごとの bag と、その試行の ``person_NN.log`` (同じディレクトリに
``palms/``・``handshakes/``・``motions/`` がある)。roscore は不要。

Usage
-----
    python3 tools/ros/replay_press_in_refine.py \\
        data/real/2026-10-16/trial6_side_by_side/person_06.log \\
        data/real/2026-10-16/trial6_side_by_side/*.bag
"""

import argparse
import datetime
import json
import math
import os
import re
import sys

import numpy as np

import rosbag
import rospy
import tf2_py

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, 'scripts')
for _path in (os.path.join(_SCRIPTS_DIR, 'ros'), _SCRIPTS_DIR,
              os.path.join(_REPO_ROOT, 'src')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_camera_pipeline_test as rcpt  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, transform_to_matrix)

spik = rcpt.spik
epp = rcpt.epp

COLOR_TOPIC = '/camera/color/image_raw/decompressed'
DEPTH_TOPIC = '/camera/depth/image_raw/decompressed'
INFO_TOPIC = '/camera/color/camera_info'
LOG_TIME_RE = re.compile(r'^(\d\d):(\d\d):(\d\d\.\d+) ')
SAVE_RE = re.compile(r'(\d{8}_\d{6})\.json に保存')
FLIP_MIN_ANGLE_DEG = 135.0

VARIANTS = dict(
    node=dict(max_distance=rcpt.PRESS_IN_REFINE_MAX_HAND_DISTANCE, flip=False),
    flip=dict(max_distance=rcpt.PRESS_IN_REFINE_MAX_HAND_DISTANCE, flip=True),
    wide=dict(max_distance=0.35, flip=True),
)
# 実機ノードが補正を採用する上限 (計測のため下で外す)。
NODE_MAX_POSITION_CHANGE = spik.REFINE_MAX_POSITION_CHANGE
NODE_MAX_ROTATION_CHANGE = spik.REFINE_MAX_ROTATION_CHANGE


def log_time(line, date):
    m = LOG_TIME_RE.match(line)
    if m is None:
        return None
    base = datetime.datetime.strptime(date, '%Y-%m-%d')
    return (base + datetime.timedelta(
        hours=int(m.group(1)), minutes=int(m.group(2)),
        seconds=float(m.group(3)))).timestamp()


def parse_log(path, date):
    """保存名・最後の press_in_refine の時刻と待ち時間・押し込み開始の時刻。"""
    name = None
    refine = None
    segments = []
    with open(path) as f:
        for line in f:
            t = log_time(line, date)
            m = SAVE_RE.search(line)
            if m and name is None:
                name = m.group(1) + '.json'
            if '"event": "press_in_refine"' in line:
                rec = json.loads(line[line.index('{'):])
                refine = (t, rec)
            elif '[debug][segment] waypoint数=' in line:
                segments.append(t)
    if refine is None:
        raise RuntimeError('{} に press_in_refine がありません'.format(path))
    press_start = min(t for t in segments if t > refine[0])
    return name, refine, segments[0], press_start


def load_tf(bag, t0, t1):
    buf = tf2_py.BufferCore(rospy.Duration(t1 - t0 + 60.0))
    for topic, msg, _ in bag.read_messages(topics=['/tf_static']):
        for tr in msg.transforms:
            buf.set_transform_static(tr, 'bag')
    for topic, msg, _ in bag.read_messages(
            topics=['/tf'], start_time=rospy.Time.from_sec(t0),
            end_time=rospy.Time.from_sec(t1)):
        for tr in msg.transforms:
            buf.set_transform(tr, 'bag')
    return buf


def iter_frames(bag, t0, t1):
    """bag の記録時刻 [t0, t1] の (color, depth, info, カラーの記録時刻) の組。"""
    colors, color_times, depths, info = [], [], [], None
    for topic, msg, t in bag.read_messages(
            topics=[COLOR_TOPIC, DEPTH_TOPIC, INFO_TOPIC],
            start_time=rospy.Time.from_sec(t0),
            end_time=rospy.Time.from_sec(t1)):
        if topic == INFO_TOPIC:
            info = msg
        elif topic == COLOR_TOPIC:
            colors.append(msg)
            color_times.append(t.to_sec())
        else:
            depths.append(msg)
    if info is None:
        for _, msg, _ in bag.read_messages(topics=[INFO_TOPIC]):
            info = msg
            break
    stamps = np.array([c.header.stamp.to_sec() for c in colors])
    for depth in depths:
        i = int(np.argmin(np.abs(stamps - depth.header.stamp.to_sec())))
        if abs(stamps[i] - depth.header.stamp.to_sec()) <= 0.05:
            yield colors[i], depth, info, color_times[i]


def flip_palm(palm):
    """掌と甲を取り違えた掌を裏返す (指先方向はそのまま)。"""
    flipped = dict(palm)
    for key in ('y_axis', 'z_axis'):
        flipped[key] = [-float(v) for v in palm[key]]
    flipped['rot'] = np.column_stack([
        palm['x_axis'], flipped['y_axis'], flipped['z_axis']]).tolist()
    return flipped


def select_hand(palm_estimator, frames, side, expected_position,
                expected_normal, max_distance, flip):
    """実機ノードの ``_select_offered_hand`` と同じ選び方 (``flip`` なら
    裏返った手も使う)。``(palm or None, distances, flipped)``。"""
    chosen, distances, flips, cameras = [], [], [], []
    for hands, camera_position in frames:
        best = None
        for hand in hands:
            joints = epp.hand_joint_positions(hand, side)
            palm = palm_estimator.estimate_palm(
                joints, side, camera_position, expected_normal)
            if palm is None or hand['score'] < rcpt.PRESS_IN_REFINE_MIN_HAND_SCORE:
                continue
            dist = float(np.linalg.norm(
                np.asarray(palm['position']) - expected_position))
            angle = math.degrees(math.acos(np.clip(np.dot(
                palm['y_axis'], expected_normal), -1.0, 1.0)))
            if angle <= rcpt.PRESS_IN_REFINE_MAX_NORMAL_ANGLE_DEG:
                is_flipped = False
            elif flip and angle >= FLIP_MIN_ANGLE_DEG:
                is_flipped = True
            else:
                continue
            if best is None or dist < best[0]:
                best = (dist, joints, is_flipped)
        if best is not None and best[0] <= max_distance:
            distances.append(best[0])
            chosen.append(best[1])
            flips.append(best[2])
            cameras.append(camera_position)
    if not chosen:
        return None, distances, False
    names = [n for n in chosen[0]
             if sum(n in j for j in chosen) * 2 >= len(chosen)]
    joints = {n: np.median([j[n] for j in chosen if n in j], axis=0).tolist()
              for n in names}
    palm = palm_estimator.estimate_palm(
        joints, side, np.median(cameras, axis=0), expected_normal)
    flipped = sum(flips) * 2 > len(flips)
    if palm is not None and flipped:
        palm = flip_palm(palm)
    return palm, distances, flipped


def place_at(robot, joint_names, waypoint):
    robot.newcoords(rcpt.Coordinates())
    rcpt.apply_waypoint_pose(robot, joint_names, [waypoint], 0)


def run(args):
    date = re.search(r'(\d{4})(\d{2})(\d{2})_\d{6}|(\d{4}-\d{2}-\d{2})',
                     args.log)
    if args.date:
        date = args.date
    elif date.group(4):
        date = date.group(4)
    else:
        date = '{}-{}-{}'.format(date.group(1), date.group(2), date.group(3))
    data_dir = os.path.dirname(os.path.abspath(args.log))
    name, (t_refine, refine_rec), approach_start, press_start = parse_log(
        args.log, date)
    with open(os.path.join(data_dir, 'palms', name)) as f:
        palms = json.load(f)
    with open(os.path.join(data_dir, 'handshakes', name)) as f:
        result = json.load(f)
    with open(os.path.join(data_dir, 'motions', name)) as f:
        motion = json.load(f)
    side = result['offered_hand']
    planned_palm = palms[side]
    joint_names = motion['joint_names']
    hover_wp = motion['waypoints'][-1]

    node_args = rcpt.build_arg_parser().parse_args([])
    planner = rcpt.HandshakePlanner(node_args)
    robot = planner.robot
    spik.attach_camera_optical_coords(robot)
    # 上限を外して必要な補正量を測る。
    spik.REFINE_MAX_POSITION_CHANGE = float('inf')
    spik.REFINE_MAX_ROTATION_CHANGE = float('inf')
    estimator = PeoplePoseEstimator(
        use_hand=True,
        min_detection_confidence=node_args.min_detection_confidence,
        min_tracking_confidence=node_args.min_tracking_confidence,
        min_visibility=node_args.min_visibility,
        min_joints=node_args.min_joints,
        max_z_diff=node_args.max_z_diff,
        min_body_size=node_args.min_body_size,
        max_body_size=node_args.max_body_size,
        max_limb_length=node_args.max_limb_length,
        max_hand_segment_length=node_args.max_hand_segment_length,
        max_hand_reach=node_args.max_hand_reach,
        depth_patch_size=node_args.depth_patch_size)
    palm_estimator = epp.PalmPoseEstimator()

    place_at(robot, joint_names, hover_wp)
    base_rot = robot.base_link.worldrot().copy()
    base_pos = robot.base_link.worldpos().copy()
    expected_position = base_rot.T @ (
        np.asarray(planned_palm['position']) - base_pos)
    expected_normal = base_rot.T @ np.asarray(planned_palm['y_axis'])

    t0 = t_refine - refine_rec.get('wait_time', 0.0) - args.pre
    t1 = press_start
    # 追跡を温め始める時刻 (approach なら実機ノードと同じく接近の開始)。
    t_warm = (approach_start if args.warmup == 'approach'
              else t0 - float(args.warmup))
    bag = rosbag.Bag(args.bag)
    buf = load_tf(bag, min(t_warm, t0) - 5.0, t1 + 5.0)
    frames = []
    for color_msg, depth_msg, info, t_color in iter_frames(
            bag, min(t_warm, t0), t1):
        try:
            c2b = transform_to_matrix(buf.lookup_transform_core(
                'base_link', color_msg.header.frame_id,
                color_msg.header.stamp).transform)
        except (tf2_py.LookupException, tf2_py.ConnectivityException,
                tf2_py.ExtrapolationException):
            continue
        color = imgmsg_to_ndarray(color_msg, desired_encoding='bgr8')
        depth = PeoplePoseEstimator.depth_to_meters(
            imgmsg_to_ndarray(depth_msg), encoding=depth_msg.encoding)
        hands = estimator.estimate_hands_3d(
            color, depth, CameraIntrinsics.from_matrix(info.K),
            output_transform=c2b)
        # warmup 区間は Hands の追跡を温めるだけで評価には使わない。
        if t_color < t0:
            continue
        frames.append((hands, c2b[:3, 3]))
        if args.verbose:
            for hand in hands:
                joints = epp.hand_joint_positions(hand, side)
                palm = palm_estimator.estimate_palm(
                    joints, side, c2b[:3, 3], expected_normal)
                if palm is None:
                    print('    frame {} {} score {:.2f}: 掌を推定できない'.format(
                        len(frames) - 1, hand['side'], hand['score']))
                    continue
                print('    frame {} {} score {:.2f}{}: 距離 {:.0f}mm 法線 {:.0f}deg'
                      .format(len(frames) - 1, hand['side'], hand['score'],
                              ' (推定深度)' if palm['from_model_depth']
                              else ' (真横、親指 {:+.0f}mm)'.format(
                                  palm['edge_on_thumb_depth'] * 1e3)
                              if palm['edge_on'] else '',
                              np.linalg.norm(np.asarray(palm['position'])
                                             - expected_position) * 1e3,
                              math.degrees(math.acos(np.clip(np.dot(
                                  palm['y_axis'], expected_normal), -1, 1)))))
    print('[replay] {} ({}手/{}腕): hover 区間 {:.2f}s に {} フレーム、'
          '実機の結果 {}'.format(
              os.path.basename(args.log), side, result['robot_arm'],
              t1 - t0, len(frames), refine_rec['reason']))

    records = []
    for start in range(0, max(len(frames) - 2, 0), args.stride):
        group = frames[start:start + rcpt.PRESS_IN_REFINE_FRAMES]
        row = dict(start=start)
        for key, variant in VARIANTS.items():
            palm, distances, flipped = select_hand(
                palm_estimator, group, side, expected_position,
                expected_normal, variant['max_distance'], variant['flip'])
            if palm is None:
                row[key] = None
                continue
            place_at(robot, joint_names, hover_wp)
            observed = spik.transform_palm(palm, base_rot, base_pos)
            info = spik.refine_post_process(
                robot, result['robot_arm'], observed, result['turn_deg'],
                result['post_process'])
            row[key] = dict(
                reason=info['reason'], flipped=flipped,
                accepted=bool(
                    info['post_process'] is not None
                    and info['position_change'] <= NODE_MAX_POSITION_CHANGE
                    and info['rotation_change'] <= NODE_MAX_ROTATION_CHANGE),
                hand_distance=float(np.median(distances)),
                position_change=info['position_change'],
                rotation_change_deg=math.degrees(info['rotation_change']),
                palm_shift=(np.asarray(observed['position'])
                            - np.asarray(planned_palm['position'])).tolist())
        records.append(row)
        cells = []
        for key in VARIANTS:
            r = row[key]
            cells.append('{:>8}:{:>26}'.format(key, '-' if r is None else
                         '{}{}{} d{:3.0f} {:3.0f}mm/{:3.0f}deg'.format(
                             '*' if r['accepted'] else '', r['reason'],
                             '(F)' if r['flipped'] else '',
                             r['hand_distance'] * 1e3,
                             r['position_change'] * 1e3,
                             r['rotation_change_deg'])))
        print('  f{:03d} '.format(start) + ' '.join(cells), flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(dict(log=args.log, bag=args.bag, side=side,
                           robot_arm=result['robot_arm'],
                           real_reason=refine_rec['reason'],
                           n_frames=len(frames), groups=records), f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('log', help='試行の person_NN.log')
    parser.add_argument('bag', help='その試行を含む bag')
    parser.add_argument('--date', default=None,
                        help='ログの日付 YYYY-MM-DD (パスから取れないとき)')
    parser.add_argument('--pre', type=float, default=0.3,
                        help='press_in_refine の待ち始めより何秒前から使うか')
    parser.add_argument('--warmup', default='approach',
                        help='評価区間の何秒前から Hands を走らせて追跡を温め'
                             'るか。approach (既定) なら接近の開始から (実機'
                             'ノードと同じ)、0 なら評価区間だけ (変更前の'
                             '実機ノードと同じ)')
    parser.add_argument('--stride', type=int, default=3,
                        help='3 フレームの組をずらす間隔 [フレーム]')
    parser.add_argument('--out', default=None, help='結果の JSON')
    parser.add_argument('--verbose', action='store_true',
                        help='フレームごとの手の候補を出す')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
