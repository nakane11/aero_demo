#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""rosbag から、差し出し判定によらず骨格・掌位置姿勢・骨格重畳画像を抽出する。

既定は ``--sample-interval`` 秒おきに機械的にサンプリング (判定結果は掌 JSON
に参考として残すだけ)。``--single-sample`` では bag ごとに 1 フレーム
(``<bag>.json`` の ``trigger_stamp`` に最も近いもの、無ければ中央) を保存する。

出力: ``<output-dir>/{skeletons,palms,images}/<bag名>[_<連番>].{json,png}``
(palms には label_offer_images.py が human_label を書く)。roscore 不要。

Usage
-----
    python3 tools/ros/extract_skeletons_from_bag.py \
        --bag session1.bag --output-dir /tmp/offer_dataset

    # record_palm_offer_clips.py が切り出した bag から 1 本 1 サンプル
    python3 tools/ros/extract_skeletons_from_bag.py \
        --bag palm_offer_clips/*.bag --output-dir /tmp/offer_dataset \
        --single-sample
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

import genpy
import rosbag
import tf2_ros

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, 'scripts')
_PKG_SRC_DIR = os.path.join(_REPO_ROOT, 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import json_io  # noqa: E402
from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo import skeleton_filters  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, transform_to_matrix)

import estimate_palm_poses as epp  # noqa: E402
from record_palm_offer_clips import (  # noqa: E402
    _FALLBACK_ROBOT_HAND_POSITION)


def load_tf_buffer(bag_path):
    """bag 内の全 ``/tf``/``/tf_static`` を読み込んだ ``BufferCore`` (キャッシュ 1 時間)。"""
    buf = tf2_ros.BufferCore(genpy.Duration(3600))
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _t in bag.read_messages(topics=['/tf', '/tf_static']):
            for tr in msg.transforms:
                try:
                    if topic == '/tf_static':
                        buf.set_transform_static(tr, 'bag')
                    else:
                        buf.set_transform(tr, 'bag')
                except Exception:
                    pass
    return buf


def collect_synced_frames(bag_path, color_topic, depth_topic, info_topic,
                          slop):
    """color 基準に最も近い depth を対応させた (color, depth, info) の時刻順リスト。

    camera_info は同期条件にせず、最も近い時刻のものを添える。
    """
    color_msgs, depth_msgs, info_msgs = [], [], []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _t in bag.read_messages(
                topics=[color_topic, depth_topic, info_topic]):
            if topic == color_topic:
                color_msgs.append(msg)
            elif topic == depth_topic:
                depth_msgs.append(msg)
            elif topic == info_topic:
                info_msgs.append(msg)

    def _closest(msgs, stamp, used):
        best_idx, best_dt = None, slop
        for i, m in enumerate(msgs):
            if i in used:
                continue
            dt = abs(m.header.stamp.to_sec() - stamp)
            if dt <= best_dt:
                best_idx, best_dt = i, dt
        return best_idx

    info_stamps = [m.header.stamp.to_sec() for m in info_msgs]
    used_depth = set()
    frames = []
    for color_msg in color_msgs:
        stamp = color_msg.header.stamp.to_sec()
        di = _closest(depth_msgs, stamp, used_depth)
        if di is None or not info_msgs:
            continue
        used_depth.add(di)
        ii = min(range(len(info_msgs)),
                 key=lambda i: abs(info_stamps[i] - stamp))
        frames.append((color_msg, depth_msgs[di], info_msgs[ii]))
    frames.sort(key=lambda f: f[0].header.stamp.to_sec())
    return frames


def _lookup_robot_position(buf, base_frame, hand_frame, stamp):
    """ロボット手先位置 (TF、引けなければフォールバック)。"""
    try:
        transform = buf.lookup_transform_core(base_frame, hand_frame, stamp)
    except Exception:
        return np.asarray(_FALLBACK_ROBOT_HAND_POSITION, dtype=np.float64)
    t = transform.transform.translation
    return np.array([t.x, t.y, t.z], dtype=np.float64)


def _load_trigger_stamp(bag_path):
    """``<bag_stem>.json`` の ``trigger_stamp`` (無ければ None)。"""
    meta_path = os.path.splitext(bag_path)[0] + '.json'
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get('trigger_stamp')


def _pick_representative_index(frames, trigger_stamp):
    """代表フレームの index (``trigger_stamp`` に最も近いもの、無ければ中央)。"""
    if not frames:
        return None
    if trigger_stamp is None:
        return len(frames) // 2
    return min(range(len(frames)),
              key=lambda i: abs(
                  frames[i][0].header.stamp.to_sec() - trigger_stamp))


def _save_frame(joint_positions, person_joints_2d, color, palm_estimator,
                buf, args, stamp, out_dirs, name):
    """1 フレーム分の骨格・掌・重畳画像を保存する。

    判定に使ったロボット手先位置 (base_link) を ``robot_position`` に残す
    (tune_offer_selector.py が再計算に使う)。
    """
    robot_position = _lookup_robot_position(
        buf, args.base_frame, args.robot_hand_frame, stamp)
    palm_estimator.offered_hand_selector.robot_position = robot_position
    palms = palm_estimator.estimate(joint_positions)
    palms['robot_position'] = [float(v) for v in robot_position]

    json_io.save_json(
        os.path.join(out_dirs['skeletons'], name + '.json'),
        dict(skeleton=dict(
            joint_positions={k: list(v)
                             for k, v in joint_positions.items()},
            height=0.0)))
    # 既存の human_label は残る。
    epp.save_json(palms, os.path.join(out_dirs['palms'], name + '.json'))
    overlay = skeleton_drawing.draw_skeleton_overlay(
        color, person_joints_2d, offered_side=palms['offered_hand'])
    cv2.imwrite(os.path.join(out_dirs['images'], name + '.png'), overlay)


def process_bag(bag_path, args, pose_estimator, joint_smoother,
                palm_estimator, out_dirs):
    """``bag_path`` のフレームを選んで保存し、保存数を返す。"""
    buf = load_tf_buffer(bag_path)
    frames = collect_synced_frames(
        bag_path, args.color_topic, args.depth_topic,
        args.camera_info_topic, args.sync_slop)
    print('{}: {} フレームを同期できました。'.format(bag_path, len(frames)))
    if not frames:
        print('  -> 同期できたフレームが無いためスキップします。')
        return 0

    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    if args.single_sample:
        trigger_stamp = _load_trigger_stamp(bag_path)
        target_indices = {_pick_representative_index(frames, trigger_stamp)}
    else:
        target_indices = None  # サンプリング間隔で都度判定する

    n_saved = 0
    n_no_tf = 0
    n_no_person = 0
    last_saved_stamp = None
    for idx, (color_msg, depth_msg, info_msg) in enumerate(frames):
        stamp_sec = color_msg.header.stamp.to_sec()
        try:
            transform = buf.lookup_transform_core(
                args.base_frame, color_msg.header.frame_id,
                color_msg.header.stamp)
        except Exception:
            camera_to_base = None
            n_no_tf += 1
        else:
            camera_to_base = transform_to_matrix(transform.transform)

        joint_positions = None
        person_joints_2d = None
        color = None
        if camera_to_base is not None:
            color = imgmsg_to_ndarray(color_msg, desired_encoding='bgr8')
            depth_raw = imgmsg_to_ndarray(depth_msg)
            depth_m = PeoplePoseEstimator.depth_to_meters(
                depth_raw, encoding=depth_msg.encoding)
            intrinsics = CameraIntrinsics.from_matrix(info_msg.K)
            people, joints_2d = pose_estimator.estimate_3d(
                color, depth_m, intrinsics, output_transform=camera_to_base)
            joint_positions = people[0] if people else None
            person_joints_2d = joints_2d[0] if joints_2d else None
            if joint_positions is not None:
                # 保存しないフレームも平滑化には通す (速度推定が飛ばないように)。
                joint_positions = joint_smoother.update(
                    joint_positions, t=stamp_sec)

        if target_indices is not None:
            should_save = idx in target_indices
        else:
            should_save = (last_saved_stamp is None
                          or stamp_sec - last_saved_stamp
                             >= args.sample_interval)
        if not should_save:
            continue
        if joint_positions is None:
            n_no_person += 1
            continue

        name = (bag_name if args.single_sample
                else '{}_{:05d}'.format(bag_name, idx))
        _save_frame(joint_positions, person_joints_2d, color, palm_estimator,
                   buf, args, color_msg.header.stamp, out_dirs, name)
        n_saved += 1
        last_saved_stamp = stamp_sec

    print('  -> 保存 {} 件 (TF 未解決で除外 {} 件, 人物未検出で除外 {} 件)'
         .format(n_saved, n_no_tf, n_no_person))
    return n_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--bag', type=str, nargs='+', required=True,
        help='入力 bag (複数・glob 可)。')
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='出力先 (skeletons/palms/images を作る)。')
    parser.add_argument('--color-topic', type=str,
                        default='/camera/color/image_raw/decompressed')
    parser.add_argument('--depth-topic', type=str,
                        default='/camera/depth/image_raw/decompressed')
    parser.add_argument('--camera-info-topic', type=str,
                        default='/camera/color/camera_info')
    parser.add_argument('--base-frame', type=str, default='base_link')
    parser.add_argument(
        '--sync-slop', type=float, default=0.1,
        help='color/depth 同期の最大時刻差 [s]。')
    parser.add_argument(
        '--sample-interval', type=float, default=0.5,
        help='サンプリング間隔 [s] (--single-sample なしのとき)。')
    parser.add_argument(
        '--single-sample', action='store_true',
        help='bag ごとに代表 1 フレームだけ保存する (切り出し済みクリップ向け)。')
    # --- PeoplePoseEstimator ---
    parser.add_argument('--min-detection-confidence', type=float, default=0.5)
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5)
    parser.add_argument('--min-visibility', type=float, default=0.5)
    parser.add_argument('--min-joints', type=int, default=6)
    parser.add_argument('--max-z-diff', type=float, default=1.0)
    parser.add_argument('--min-body-size', type=float, default=0.3)
    parser.add_argument('--max-body-size', type=float, default=2.5)
    parser.add_argument('--max-limb-length', type=float, default=0.7)
    parser.add_argument('--max-hand-segment-length', type=float, default=0.12)
    parser.add_argument('--max-hand-reach', type=float, default=0.22)
    parser.add_argument('--depth-patch-size', type=int, default=3)
    # --- 関節位置の平滑化 ---
    parser.add_argument('--joint-smoothing-mincutoff', type=float, default=0.5)
    parser.add_argument('--joint-smoothing-beta', type=float, default=0.3)
    parser.add_argument('--joint-smoothing-dcutoff', type=float, default=1.0)
    # --- 差し出し手判定 (estimate_palm_poses.OfferedHandSelector) ---
    parser.add_argument('--offer-score-min', type=float, default=0.65)
    parser.add_argument('--robot-hand-frame', type=str,
                        default='r_eef_grasp_link')
    parser.add_argument('--max-person-distance', type=float, default=3.0)
    args = parser.parse_args()

    bag_paths = []
    for pattern in args.bag:
        matched = sorted(glob.glob(pattern))
        bag_paths.extend(matched if matched else [pattern])
    if not bag_paths:
        print('--bag に一致する bag ファイルが見つかりません。')
        return

    out_dirs = {name: os.path.join(args.output_dir, name)
               for name in ('skeletons', 'palms', 'images')}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    pose_estimator = PeoplePoseEstimator(
        use_hand=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        min_visibility=args.min_visibility,
        min_joints=args.min_joints,
        max_z_diff=args.max_z_diff,
        min_body_size=args.min_body_size,
        max_body_size=args.max_body_size,
        max_limb_length=args.max_limb_length,
        max_hand_segment_length=args.max_hand_segment_length,
        max_hand_reach=args.max_hand_reach,
        depth_patch_size=args.depth_patch_size)

    max_distance = (None if args.max_person_distance <= 0
                   else args.max_person_distance)
    offered_hand_selector = epp.OfferedHandSelector(
        robot_position=None, score_min=args.offer_score_min,
        max_distance=max_distance)
    palm_estimator = epp.PalmPoseEstimator(offered_hand_selector)

    total_saved = 0
    try:
        for bag_path in bag_paths:
            # 平滑化は bag ごとに独立。
            joint_smoother = skeleton_filters.OneEuroFilter(
                mincutoff=args.joint_smoothing_mincutoff,
                beta=args.joint_smoothing_beta,
                dcutoff=args.joint_smoothing_dcutoff)
            total_saved += process_bag(
                bag_path, args, pose_estimator, joint_smoother,
                palm_estimator, out_dirs)
    finally:
        pose_estimator.close()

    print('\n合計 {} フレームを {} に保存しました。'.format(
        total_saved, args.output_dir))
    print('次は label_offer_images.py で images/ を見ながら palms/ に '
         'human_label を付けてください。')


if __name__ == '__main__':
    main()
