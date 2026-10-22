#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""実機で保存した試行の骨格・掌 JSON を、実機ノードと同じ手順で解き直す。

``run_camera_pipeline_test.py`` が試行ごとに保存する ``skeletons/`` と
``palms/`` (``--save-dir`` か実機ログの ``logs/``) を読み、ノードと同じ
``HandshakePlanner`` で IK -> 軌道計画 -> 横並び移動の計画を行う。
骨格検出の揺れが入らず毎回同じ結果になるので、変更前後の比較に使う
(docs/test_data.md)。roscore・カメラ・実機は不要。

試行ごとの到達段階 (``stage``):
  no_palm        掌 JSON に差し出した手が無い
  ik_fail        IK が解けない
  no_press       押し込み姿勢 (押し付け+視線) が解けない
  motion_ng      軌道が干渉検証に通らない
  motion_ok      軌道までは通るが横並び移動が通らない
  transition_ok  横並び移動まで通る
保存時の ``handshakes/``・``motions/`` があれば、そのときの段階
(``saved_stage``) も並べて出す。

ノードのオプション (``--torso-surface-offset`` など) はそのまま渡せる。
カメラ光軸は TF を引かず既定値を使う。

Usage
-----
    python3 tools/ros/resolve_saved_trials.py data/real/2026-10-13/logs
    python3 tools/ros/resolve_saved_trials.py \\
        data/real/2026-10-13/logs data/real/2026-10-15/logs \\
        --out /tmp/resolve.json --torso-surface-offset 0.10
"""

import argparse
import collections
import json
import os
import sys
import time
import zlib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, 'scripts')
for _path in (os.path.join(_SCRIPTS_DIR, 'ros'), _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import run_camera_pipeline_test as rcpt  # noqa: E402

spik = rcpt.spik
phm = rcpt.phm

STAGES = ('no_palm', 'ik_fail', 'no_press', 'motion_ng', 'motion_ok',
          'transition_ok')


def trial_stage(result, motion):
    """IK 結果 ``result`` と軌道 ``motion`` から到達段階を返す。"""
    if not result['solved']:
        return 'ik_fail'
    if result.get('post_process') is None or motion is None:
        return 'no_press'
    if not rcpt.motion_verified(motion):
        return 'motion_ng'
    transition = motion.get('transition')
    if transition is not None and transition['verified']:
        return 'transition_ok'
    return 'motion_ok'


def load_saved_stage(data_dir, name):
    """保存時の ``handshakes/``・``motions/`` から段階を求める (無ければ None)。"""
    handshake_path = os.path.join(data_dir, 'handshakes', name)
    if not os.path.exists(handshake_path):
        return None
    with open(handshake_path) as f:
        result = json.load(f)
    motion = None
    motion_path = os.path.join(data_dir, 'motions', name)
    if os.path.exists(motion_path):
        with open(motion_path) as f:
            motion = json.load(f)
    return trial_stage(result, motion)


def approach_heading_metrics(motion, palm_position):
    """接近 (lead-in + 軌道) の台車の yaw の変化量 [度] と、台車の正面から見た
    掌 ``palm_position`` の方位の絶対値の最大 [度]。"""
    waypoints = (list(motion.get('lead_in_waypoints') or [])
                 + list(motion['waypoints']))
    yaws = np.array([wp['base_yaw'] for wp in waypoints])
    xy = np.array([wp['base_position'][:2] for wp in waypoints])
    bearing = (np.arctan2(palm_position[1] - xy[:, 1],
                          palm_position[0] - xy[:, 0]) - yaws)
    bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
    return dict(
        approach_yaw_change_deg=float(np.degrees(yaws[-1] - yaws[0])),
        approach_max_palm_bearing_deg=float(np.degrees(
            np.max(np.abs(bearing)))))


def resolve_trial(planner, args, data_dir, name, attempt):
    """1 試行を解き直して記録 (dict) を返す。"""
    joint_positions = spik.load_skeleton_json(
        os.path.join(data_dir, 'skeletons', name))
    with open(os.path.join(data_dir, 'palms', name)) as f:
        palms = json.load(f)
    record = dict(
        data_dir=data_dir, name=name,
        saved_stage=load_saved_stage(data_dir, name))
    offered_hand = palms.get('offered_hand')
    if offered_hand not in ('L', 'R') or palms.get(offered_hand) is None:
        record['stage'] = 'no_palm'
        return record
    # 乱数の消費が前の試行の解き直し回数に依存しないよう試行ごとに固定する。
    np.random.seed(zlib.crc32(name.encode()))
    t0 = time.time()
    planned = planner.plan(joint_positions, palms, offered_hand, attempt)
    result, motion = planned['result'], planned['motion']
    record.update(
        hand=offered_hand, robot_arm=planned['robot_arm'],
        stage=trial_stage(result, motion),
        time=time.time() - t0)
    if motion is not None:
        record['motion_kind'] = motion['kind']
        record['motion_min_distance'] = float(
            min(motion['waypoint_min_distances']))
        record.update(approach_heading_metrics(
            motion, palms[offered_hand]['position']))
        transition = motion.get('transition')
        if transition is not None:
            record['transition_fraction'] = transition['fraction']
            record['transition_reason'] = transition['reason']
            record['transition_time'] = transition.get('compute_time')
        record['motion'] = dict(motion, robot_arm=planned['robot_arm'])
    return record


def resolve_transition_only(planner, args, data_dir, name):
    """保存時の押し込み姿勢 (``handshakes/``) から横並び移動だけを解き直した
    記録を返す (保存時に押し込みまで解けていなければ None)。IK・軌道の揺れを
    除いて横並び移動の変更を比べるため。"""
    path = os.path.join(data_dir, 'handshakes', name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        result = json.load(f)
    post = result.get('post_process')
    if not result.get('solved') or post is None:
        return None
    joint_positions = spik.load_skeleton_json(
        os.path.join(data_dir, 'skeletons', name))
    offset = spik.human_translation_offset(
        joint_positions, front_distance=args.human_front_distance)
    translated_joints = spik.translate_joint_positions(
        joint_positions, offset)
    collision_joints = spik.shift_torso_joints_from_surface(
        translated_joints, offset, args.torso_surface_offset)
    # untranslate_result の逆。
    post = dict(post)
    for key in ('target_position', 'base_position'):
        post[key] = [post[key][0] + offset[0], post[key][1] + offset[1]] + \
            list(post[key][2:])
    np.random.seed(zlib.crc32(name.encode()))
    t0 = time.time()
    transition = rcpt.sbs.plan_transition(
        planner.robot, result['robot_arm'], result['offered_hand'], post,
        result['turn_deg'], translated_joints, collision_joints,
        planner.verification_pairs, planner.collision_pairs,
        planner.base_limits,
        approach=planner.side_by_side_approach == 'sync')
    motion = dict(robot_arm=result['robot_arm'], waypoints=[],
                  transition=transition)
    if planner.refine_robot is not None and transition['verified']:
        # ノードが動作中に裏で解く計画 (寄せあり) も解き、ノードと同じ基準で
        # 実行する方を選ぶ (_refined_transition_waypoints)。
        refined = rcpt.sbs.refine_transition(
            planner.refine_robot, result['robot_arm'],
            result['offered_hand'], post, result['turn_deg'],
            translated_joints, collision_joints,
            planner.refine_verification_pairs, planner.refine_collision_pairs,
            planner.base_limits, transition)
        motion['transition_refined'] = refined
        motion['transition_executed'] = (
            'refined' if refined.get('verified')
            and refined['fraction'] >= transition['fraction'] else 'secured')
        rcpt.untranslate_motion(dict(waypoints=[], transition=refined),
                                offset)
    executed = motion.get('transition_refined') \
        if motion.get('transition_executed') == 'refined' else transition
    rcpt.untranslate_motion(motion, offset)
    return dict(
        data_dir=data_dir, name=name, saved_stage=load_saved_stage(
            data_dir, name),
        hand=result['offered_hand'], robot_arm=result['robot_arm'],
        stage='transition_ok' if transition['verified'] else 'motion_ok',
        time=time.time() - t0, transition_fraction=executed['fraction'],
        transition_reason=executed['reason'],
        transition_time=transition.get('compute_time'),
        transition_executed=motion.get('transition_executed'), motion=motion)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n')[0],
        epilog='これ以外のオプションは run_camera_pipeline_test.py に渡す。')
    parser.add_argument(
        'data_dirs', nargs='+',
        help='skeletons/ と palms/ を持つディレクトリ (複数可)。')
    parser.add_argument('--out', type=str, default=None,
                        help='試行ごとの記録を JSON で保存するパス。')
    parser.add_argument('--motion-dir', type=str, default=None,
                        help='解き直した軌道 (横並び移動を含む) を試行ごとの '
                             'JSON で保存するディレクトリ。')
    parser.add_argument('--transition-only', action='store_true',
                        help='保存時の押し込み姿勢から横並び移動だけを解き'
                             '直す (保存時に押し込みまで解けた試行だけ)。')
    rcpt.sbs.add_human_arm_arguments(parser)
    my_args, node_argv = parser.parse_known_args()
    rcpt.sbs.apply_human_arm_arguments(my_args)
    args = rcpt.build_arg_parser().parse_args(node_argv)

    planner = rcpt.HandshakePlanner(args)
    spik.attach_camera_optical_coords(planner.robot)
    # ノードと同じ初期姿勢から warmup する (バッチ IK の初回呼び出しの
    # 可動域の扱いもノードと揃えるため)。
    phm.arms_down_angles(planner.robot, planner.robot.joint_list)
    planner.warmup()

    trials = [(d, name) for d in my_args.data_dirs
              for name in sorted(os.listdir(os.path.join(d, 'skeletons')))
              if name.endswith('.json')]
    records = []
    for i, (data_dir, name) in enumerate(trials):
        if my_args.transition_only:
            record = resolve_transition_only(planner, args, data_dir, name)
            if record is None:
                continue
        else:
            record = resolve_trial(planner, args, data_dir, name, i + 1)
        motion = record.pop('motion', None)
        if my_args.motion_dir and motion is not None:
            # 別のディレクトリの同じ名前と区別するため、親の名前を前に付ける。
            os.makedirs(my_args.motion_dir, exist_ok=True)
            rcpt.json_io.save_json(os.path.join(
                my_args.motion_dir, '{}_{}'.format(
                    os.path.basename(os.path.normpath(data_dir)), name)),
                motion)
        records.append(record)
        print('[{}/{}] {} {}: {} (保存時 {})'.format(
            i + 1, len(trials), data_dir, name, record['stage'],
            record['saved_stage']), flush=True)

    counts = collections.Counter(r['stage'] for r in records)
    print('[summary] {} 試行: {}'.format(len(records), ', '.join(
        '{} {}'.format(s, counts[s]) for s in STAGES if counts[s])))
    changed = [r for r in records
               if r['saved_stage'] is not None and r['stage'] != r['saved_stage']]
    if changed:
        print('[summary] 保存時と段階が変わった試行:')
        for r in changed:
            print('  {} {}: {} -> {}'.format(
                r['data_dir'], r['name'], r['saved_stage'], r['stage']))
    if my_args.out:
        with open(my_args.out, 'w') as f:
            json.dump(records, f, indent=1, ensure_ascii=False)
        print('[summary] {} に保存しました。'.format(my_args.out))


if __name__ == '__main__':
    main()
