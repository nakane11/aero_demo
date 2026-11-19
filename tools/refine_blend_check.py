#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""押し込み直前の解き直し (``refine_post_process``) を、脚を固定する角度 (``REFINE_FIXED_LEGS``) の
'hover' (前の形) と 'planned' (計画の押し込み後) で比べる小さなテスト。

develop の tools/refine_blend_check.py (ユニット I・V4) を master の形に書き直したもの (unified の系統は無い)。
人ごとに計画 (hover + 押し込み後) を solve_palm_ik の main と同じ前処理で解き、計画時の掌から向きを傾け・位置を
ずらした「検出し直した掌」を作って、実機ノードと同じく hover の姿勢にロボットを置いてから
``refine_post_process(..., planned_palm=計画時の掌)`` を、同じ計画のまま両方の形で呼ぶ。成否 (reason)・
向きを計画へ寄せた割合 (blend)・脚の段階 (legs)・時間と、解き直した押し込み後の膝・足首・腰ピッチ・腰ヨーの
計画の押し込み後からのずれ [度] を出す。

    python3 tools/refine_blend_check.py --out /tmp/refine.json [--people human_000,human_006]
    python3 tools/refine_blend_check.py --summary /tmp/refine.json
"""

import argparse
import json
import math
import os
import sys
import time
import zlib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
_SCRIPTS_DIR = os.path.join(_ROOT, 'scripts')
for path in (_SCRIPTS_DIR, os.path.join(_ROOT, 'src')):
    if path not in sys.path:
        sys.path.insert(0, path)

# jax の import より前に設定する (実機ノードと同じ)。
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR',
                      os.path.expanduser('~/.cache/jax_compilation_cache'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')

import numpy as np  # noqa: E402

import solve_palm_ik as spik  # noqa: E402
from skrobot.coordinates.math import rotation_matrix  # noqa: E402

# (向きの傾き [deg]、位置のずれ [m]): 検出し直した掌の、計画時の掌からの変化。
PERTURBATIONS = ((0.0, 0.0), (20.0, 0.02), (40.0, 0.02), (55.0, 0.03))
# 比べる形: --compare fixed_legs は ``spik.REFINE_FIXED_LEGS`` の hover / planned、--compare pin_narrow は
# ``spik.PIN_LEG_NARROW`` の off / on (2 段目の脚の幅を腰が上がる側にも効かせるか)。
COMPARES = {
    'fixed_legs': (('hover', dict(REFINE_FIXED_LEGS='hover')),
                   ('planned', dict(REFINE_FIXED_LEGS='planned'))),
    'pin_narrow': (('pin_off', dict(PIN_LEG_NARROW=False)),
                   ('pin_on', dict(PIN_LEG_NARROW=True))),
}
# ずれを記録する関節。
WATCH_JOINTS = ('knee_joint', 'ankle_joint', 'waist_p_joint', 'waist_y_joint')
# develop の docs/unified_solver/smoke_set.txt の 12 人 (合成 seed0_ik_targets)。
DEFAULT_PEOPLE = ('human_000', 'human_006', 'human_092', 'human_014', 'human_105',
                  'human_052', 'human_120', 'human_228', 'human_289', 'human_232',
                  'human_004', 'human_270')


def perturbed_palm(palm, tilt_deg, shift):
    """掌の法線 (y 軸) を x 軸まわりに ``tilt_deg`` 傾け、位置を上へ ``shift`` ずらした掌。"""
    out = dict(palm)
    rot = rotation_matrix(math.radians(tilt_deg), np.asarray(palm['x_axis'], dtype=np.float64))
    for key in ('x_axis', 'y_axis', 'z_axis'):
        if key in palm:
            out[key] = (rot @ np.asarray(palm[key], dtype=np.float64)).tolist()
    out['position'] = (np.asarray(palm['position'], dtype=np.float64)
                       + np.array([0.0, 0.0, shift])).tolist()
    return out


def setup():
    robot = spik.Aero(use_hand=False)
    spik.restrict_elbow_range(robot)
    spik.restrict_leg_range(robot)
    spik.restrict_waist_range(robot)
    spik.restrict_neck_range(robot)
    spik.lock_fixed_joints(robot)
    spik.apply_collision_model(robot)
    spik.apply_hand_box(robot)
    spik.other_hand_points('r')
    collision_pairs = spik.load_collision_pairs(
        os.path.join(_SCRIPTS_DIR, 'collision_pairs.json'), robot)
    verification_pairs = spik.build_verification_pairs_for_model(
        robot, spik.DEFAULT_COLLISION_VERIFY_MODEL)
    base_limits = [spik.DEFAULT_BASE_X_RANGE, spik.DEFAULT_BASE_Y_RANGE,
                   spik.DEFAULT_BASE_YAW_RANGE]
    ik_kwargs = dict(self_collision=True, collision_pairs=collision_pairs,
                     verification_pairs=verification_pairs)
    t0 = time.time()
    np.random.seed(0)
    spik._warmup_batch_ik(robot, base_limits, ik_kwargs)
    print('[refine-check] warmup {:.1f} 秒'.format(time.time() - t0))
    return robot, base_limits, ik_kwargs


def plan_person(robot, base_limits, ik_kwargs, input_set, name):
    """solve_palm_ik の main と同じ前処理で hover + 押し込み後を解く。``(palm, robot_arm, result, picked)``。"""
    palms = spik.load_palm_json(os.path.join(input_set, 'palms', name + '.json'))
    hand = palms.get('offered_hand')
    palm = palms.get(hand) if hand in ('L', 'R') else None
    if palm is None:
        return None
    robot_arm = spik.DEFAULT_ROBOT_ARM[hand]
    joints = spik.load_skeleton_json(os.path.join(input_set, 'skeletons', name + '.json'))
    offset = spik.human_translation_offset(joints, front_distance=spik.HUMAN_FRONT_DISTANCE)
    joints = spik.translate_joint_positions(joints, offset)
    collision_joints = spik.shift_torso_joints_from_surface(joints, offset, 0.0)
    obstacles = spik.human_body_obstacles(collision_joints)
    palm = spik.translate_palm(palm, offset)
    limits = base_limits
    side_sign = spik.offered_hand_side_sign(hand, joints, palm)
    if side_sign is not None:
        limits = [limits[0], spik.restrict_base_y_range_to_hand_side(limits[1], side_sign),
                  limits[2]]
    human_yaw = spik.human_facing_yaw(joints)
    if human_yaw is not None:
        limits = [limits[0], limits[1], spik.restrict_base_yaw_range_to_human_facing(
            limits[2], human_yaw,
            margin=math.radians(spik.DEFAULT_BASE_YAW_FACING_MARGIN_DEG))]
    standing_xy = spik.human_standing_xy(joints)
    standing_x = None if standing_xy is None else float(standing_xy[0])
    np.random.seed(zlib.crc32((name + '.json').encode()) % (2 ** 32))
    picked, ik_time, sel_time, limits, _ = spik.solve_person_ik_side_by_side(
        robot, palm, hand, robot_arm, obstacles, limits, standing_x,
        joint_positions=collision_joints, placement_joint_positions=joints,
        **ik_kwargs)
    if picked is None or picked[3] is None:
        return palm, robot_arm, None, picked
    rots = spik.palm_to_target_rots(palm, hand, robot_arm)
    turn_index, angle_vector, base_pose, post = picked
    result = spik.solved_result(robot, robot_arm, rots[turn_index], turn_index,
                                angle_vector, base_pose, limits, post, ik_time,
                                sel_time, hand, palm)
    return palm, robot_arm, result, picked


def joint_deviation_deg(post, planned_post):
    """解き直した押し込み後 ``post`` の、計画の押し込み後からのずれ [度] (``WATCH_JOINTS``)。"""
    if post is None:
        return None
    got = dict(zip(post['joint_names'], post['joint_angle_vector']))
    ref = dict(zip(planned_post['joint_names'], planned_post['joint_angle_vector']))
    return {name: math.degrees(got[name] - ref[name]) for name in WATCH_JOINTS
            if name in got and name in ref}


def run(args):
    robot, base_limits, ik_kwargs = setup()
    people = (args.people.split(',') if args.people else list(DEFAULT_PEOPLE))
    records = []
    for name in people:
        planned = plan_person(robot, base_limits, ik_kwargs, args.input_set, name)
        if planned is None:
            continue
        palm, robot_arm, result, picked = planned
        if result is None:
            print('[refine-check] {}: 計画 (押し込み後) が解けないため飛ばします。'.format(name))
            records.append(dict(person=name, planned=False))
            continue
        _, angle_vector, base_pose, _ = picked
        planned_post = result['post_process']
        hover_dev = joint_deviation_deg(
            dict(joint_names=result['joint_names'],
                 joint_angle_vector=result['joint_angle_vector']), planned_post)
        for tilt, shift in PERTURBATIONS:
            observed = perturbed_palm(palm, tilt, shift)
            for mode, settings in COMPARES[args.compare]:
                for key, value in settings.items():
                    setattr(spik, key, value)
                # 実機ノードと同じく hover の姿勢から解き直す。
                robot.angle_vector(angle_vector)
                robot.newcoords(base_pose)
                hover_waist_z = float(robot.waist_link.worldpos()[2])
                info = spik.refine_post_process(
                    robot, robot_arm, observed, result['turn_deg'], planned_post,
                    planned_palm=palm)
                # 解けたときロボットは解き直した押し込み後の姿勢 (台車の平行移動は水平だけ)。
                waist_rise = (None if info['post_process'] is None
                              else float(robot.waist_link.worldpos()[2]) - hover_waist_z)
                rec = dict(person=name, planned=True, mode=mode, tilt_deg=tilt, shift_m=shift,
                           waist_rise_m=waist_rise,
                           reason=info['reason'], blend=info.get('blend'),
                           legs=info.get('legs'),
                           compute_time=float(info['compute_time']),
                           position_change=float(info['position_change']),
                           rotation_change_deg=math.degrees(info['rotation_change']),
                           deviation_deg=joint_deviation_deg(info['post_process'], planned_post),
                           hover_deviation_deg=hover_dev)
                records.append(rec)
                dev = rec['deviation_deg'] or {}
                from_hover = {j: dev[j] - hover_dev[j] for j in dev if j in hover_dev}
                print('[refine-check] {} 傾き {:>4.0f}° ずれ {:.0f}mm {:<7}: {:<12} blend {} legs {} '
                      '{:.2f} 秒 計画から waist_p {} knee {} ankle {} / hover から knee {} ankle {} 腰の高さ {}'
                      .format(
                          name, tilt, shift * 1e3, mode, rec['reason'], rec['blend'], rec['legs'],
                          rec['compute_time'],
                          *(['{:+.1f}'.format(dev[j]) if j in dev else '-'
                             for j in ('waist_p_joint', 'knee_joint', 'ankle_joint')]
                            + ['{:+.1f}'.format(from_hover[j]) if j in from_hover else '-'
                               for j in ('knee_joint', 'ankle_joint')]
                            + ['-' if waist_rise is None else '{:+.1f}cm'.format(waist_rise * 100)])))
    with open(args.out, 'w') as f:
        json.dump(dict(compare=args.compare, records=records), f, indent=1)
    print('[refine-check] {} 件を {} に保存しました。'.format(len(records), args.out))
    summary(records, args.compare)


def summary(records, compare='fixed_legs'):
    for mode, _ in COMPARES[compare]:
        rs = [r for r in records if r.get('planned') and r.get('mode') == mode]
        if not rs:
            continue
        ok = [r for r in rs if r['reason'] in ('ok', 'ok_arm_only')]
        by_reason, by_legs = {}, {}
        for r in rs:
            by_reason[r['reason']] = by_reason.get(r['reason'], 0) + 1
        for r in ok:
            by_legs[r['legs']] = by_legs.get(r['legs'], 0) + 1
        times = [r['compute_time'] for r in rs]
        print('{}: 計画あり {} 人、解き直し {} 件中 成功 {} ({})、脚の段 {}、時間 平均 {:.2f} 秒・最大 {:.2f} 秒'
              .format(mode, len({r['person'] for r in rs}), len(rs), len(ok), by_reason, by_legs,
                      float(np.mean(times)), float(np.max(times))))
        for joint in WATCH_JOINTS:
            vals = [abs(r['deviation_deg'][joint]) for r in ok
                    if r.get('deviation_deg') and joint in r['deviation_deg']]
            if vals:
                print('  |{} の計画からのずれ| 中央 {:.1f}°・最大 {:.1f}°'.format(
                    joint, float(np.median(vals)), float(np.max(vals))))
        limited = [r for r in ok if r['legs'] == 'limited']
        for joint in ('knee_joint', 'ankle_joint'):
            vals = [abs(r['deviation_deg'][joint] - r['hover_deviation_deg'][joint]) for r in limited
                    if joint in (r.get('deviation_deg') or {})
                    and joint in (r.get('hover_deviation_deg') or {})]
            if vals:
                print('  limited の段: |{} の hover からの動き| 中央 {:.1f}°・最大 {:.1f}°'.format(
                    joint, float(np.median(vals)), float(np.max(vals))))
        rises = [r['waist_rise_m'] for r in limited if r.get('waist_rise_m') is not None]
        if rises:
            print('  limited の段: 腰の高さの hover からの変化 最小 {:+.1f}・最大 {:+.1f} cm'.format(
                min(rises) * 100, max(rises) * 100))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--people', default=None,
                        help='人物名のカンマ区切り (既定は develop の smoke_set の 12 人)。')
    parser.add_argument('--input-set', default=os.path.join(
        _ROOT, 'data', 'synthetic', 'seed0_ik_targets'))
    parser.add_argument('--compare', choices=sorted(COMPARES), default='fixed_legs',
                        help='比べる切り替え (既定 fixed_legs)。')
    parser.add_argument('--out')
    parser.add_argument('--summary', help='保存した JSON をまとめ直す。')
    args = parser.parse_args()
    if args.summary:
        data = json.load(open(args.summary))
        summary(data['records'], data.get('compare', 'fixed_legs'))
        return
    if not args.out:
        parser.error('--out が必要です (まとめ直すときは --summary)。')
    run(args)


if __name__ == '__main__':
    main()
