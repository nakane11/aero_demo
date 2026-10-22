#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""保存した横並び移動の計画 (``motions/*.json`` の ``transition``) で、ロボットの
前腕と人の前腕・手の近さと、人の手をロボットの手に固定したときの人の手首の角度を
測る。

- 線分の距離: 人の前腕 (肘 -> 手首) とロボットの肘 (``{arm}_elbow_link``) -> 手先、
  人の手 (手首 -> 掌 + 指先 ``HUMAN_FINGER_LENGTH``) とロボットの肘 -> 手首
  (``{arm}_hand_yaw_link``) の中心線どうしの最短距離。
- 検証の距離: ``TransitionChecker.forearm_clearances`` (ロボットの前腕・肘・手首の
  表面から、人の前腕・手の甲の円柱まで)。
- 手首の角度: ``side_by_side_transition.human_wrist_angles`` (押し込み時 s=0 を除く
  全 waypoint の最大)。

Usage
-----
    python3 tools/measure_transition_human_arm.py <dir>/motions [...] \\
        [--out x.json]
"""

import argparse
import json
import math
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import numpy as np  # noqa: E402
from skrobot.models import Aero  # noqa: E402

import side_by_side_transition as sbs  # noqa: E402
import solve_palm_ik as spik  # noqa: E402


def segment_distance(p0, p1, q0, q1):
    """線分 ``p0``-``p1`` と ``q0``-``q1`` の最短距離 (細かく刻んで求める)。"""
    p0, p1, q0, q1 = (np.asarray(v, dtype=np.float64)
                      for v in (p0, p1, q0, q1))
    t = np.linspace(0.0, 1.0, 101)[:, np.newaxis]
    points = p0 + t * (p1 - p0)
    return float(np.min(spik.segment_points_distance(q0, q1, points)))


def load_robot():
    robot = Aero(use_hand=False)
    spik.restrict_elbow_range(robot)
    spik.restrict_leg_range(robot)
    spik.restrict_waist_range(robot)
    spik.restrict_neck_range(robot)
    spik.lock_fixed_joints(robot)
    spik.apply_collision_model(robot)
    spik.apply_hand_box(robot)
    return robot


def measure(robot, checkers, motion):
    """1 人の横並び移動の計画を測った dict (計画が無ければ None)。"""
    transition = motion.get('transition')
    if motion.get('transition_executed') == 'refined':
        # 実機で実行した裏の計画 (寄せあり)。
        transition = motion['transition_refined']
    if not transition or not transition.get('verified'):
        return None
    robot_arm = motion['robot_arm']
    hand = transition['hand']
    if (robot_arm, hand) not in checkers:
        checkers[(robot_arm, hand)] = sbs.TransitionChecker(
            robot_arm, hand, checkers['pairs'])
    checker = checkers[(robot_arm, hand)]
    press = transition['press_arm']
    press_forearm = np.subtract(press['wrist'], press['elbow'])
    press_normal = np.asarray(press['palm_rot'])[:, 1]
    turn_before = transition['turn_before']
    press_angles = sbs.human_wrist_angles(
        press['elbow'], press['wrist'], press['palm_rot'], hand,
        press_forearm, press_normal)
    approach = transition.get('approach') or {}
    n_stage1 = len(transition['waypoints']) - approach.get('n_waypoints', 0)
    n_total = max(1, int(round(n_stage1 / max(transition['fraction'],
                                              1e-6))))
    end = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    elbow_link = getattr(robot, '{}_elbow_link'.format(robot_arm))
    wrist_link = getattr(robot, '{}_hand_yaw_link'.format(robot_arm))
    rows = []
    for k, wp in enumerate(transition['waypoints']):
        # 腕を下ろす動きの割合 (寄せは 1)
        s = wp['human_arm'].get('s', min(1.0, (k + 1) / float(n_total)))
        sbs._place(robot, wp['joint_angle_vector'],
                   (wp['base_position'][0], wp['base_position'][1],
                    wp['base_yaw']))
        arm = wp['human_arm']
        palm_pos = np.asarray(arm['palm_position'])
        palm_rot = np.asarray(arm['palm_rot'])
        tip = palm_pos + palm_rot[:, 0] * sbs.HUMAN_FINGER_LENGTH
        # 人の前腕とはロボットの肘 -> 手先、人の手とは (手先が掌に触れて
        # いるので) ロボットの肘 -> 手首の線分で測る。
        r0, r1 = elbow_link.worldpos(), end.worldpos()
        fore = segment_distance(r0, r1, arm['elbow'], arm['wrist'])
        hand_d = segment_distance(r0, wrist_link.worldpos(), arm['wrist'],
                                  tip)
        angle = sbs._angle_between(np.subtract(r1, r0),
                                   np.subtract(arm['wrist'], arm['elbow']))
        clear = checker.forearm_clearances(robot, arm, cull_distance=0.5)
        hand_rot = sbs.hand_on_robot(end.worldrot(), turn_before, robot_arm)
        wrist = sbs.human_wrist_angles(arm['elbow'], arm['wrist'], hand_rot,
                                       hand, press_forearm, press_normal)
        rows.append(dict(
            line_forearm=fore, line_hand=hand_d, forearm_angle=angle,
            clear_forearm=clear[0], clear_hand=clear[1],
            turn_change=abs(math.degrees(sbs._wrap(math.radians(
                wp['turn_deg'] - turn_before)))),
            violation=sbs.wrist_violation(
                wrist, sbs.wrist_limits(press_angles, s)), **wrist))
    palm_z = [wp['human_arm']['palm_position'][2]
              for wp in transition['waypoints']]
    laterals = transition.get('laterals') or []
    lateral = {}
    if laterals:
        # 一度離れてから寄せで戻った量 (台車の横の距離の最大 - 終点)。
        lateral = dict(lateral_before=transition.get('lateral_before'),
                       lateral_max=max(laterals), lateral_after=laterals[-1],
                       lateral_overshoot=max(laterals) - laterals[-1])
    press_clear = transition.get('press_forearm_clearance')
    return dict(
        press_clear=(min(press_clear) if press_clear else None),
        palm_z_press=press['palm_position'][2], palm_z_after=palm_z[-1],
        palm_drop=press['palm_position'][2] - palm_z[-1],
        arm_deg=transition.get('arm_deg'),
        abduction_deg=transition.get('abduction_deg'), **lateral,
        hand=hand, robot_arm=robot_arm, fraction=transition['fraction'],
        n_waypoints=len(rows), turn_before=turn_before,
        turn_after=transition.get('turn_after'),
        press_wrist=press_angles,
        min_line=min(min(r['line_forearm'], r['line_hand']) for r in rows),
        min_line_forearm=min(r['line_forearm'] for r in rows),
        min_line_hand=min(r['line_hand'] for r in rows),
        forearm_angle=[min(r['forearm_angle'] for r in rows),
                       max(r['forearm_angle'] for r in rows)],
        min_clear_forearm=min(r['clear_forearm'] for r in rows),
        min_clear_hand=min(r['clear_hand'] for r in rows),
        max_flexion=max(r['flexion_deg'] for r in rows),
        max_extension=-min(r['flexion_deg'] for r in rows),
        max_radial=max(r['deviation_deg'] for r in rows),
        max_ulnar=-min(r['deviation_deg'] for r in rows),
        max_twist=max(abs(r['twist_deg']) for r in rows),
        max_turn_change=max(r['turn_change'] for r in rows),
        n_violations=sum(r['violation'] is not None for r in rows),
        compute_time=transition.get('compute_time'))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('motion_dirs', nargs='+')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    robot = load_robot()
    checkers = dict(pairs=spik.build_verification_pairs_for_model(
        robot, spik.DEFAULT_COLLISION_VERIFY_MODEL))
    records = []
    for motion_dir in args.motion_dirs:
        for name in sorted(os.listdir(motion_dir)):
            if not name.endswith('.json'):
                continue
            with open(os.path.join(motion_dir, name)) as f:
                motion = json.load(f)
            if motion.get('transition') is None:
                continue
            record = measure(robot, checkers, motion) or dict(
                fraction=0.0, reason=motion['transition'].get('reason'))
            record.update(dir=motion_dir, name=name)
            records.append(record)
            if 'min_line' in record:
                print('{} {}: {:.0f}% 線分 {:.1f} cm (前腕 {:.1f}/手 {:.1f})、'
                      '検証 前腕 {:.1f}/甲 {:.1f} cm、前腕の角度 {:.0f}-{:.0f}、'
                      '握り {:+.0f} -> {:+.0f} (最大変化 {:.0f})、手首 掌屈 '
                      '{:.0f}/背屈 {:.0f}/撓屈 {:.0f}/尺屈 {:.0f}/ひねり {:.0f}'
                      ' (押し込み時 {:+.0f}/{:+.0f})、超過 {}/{}'.format(
                          os.path.basename(os.path.dirname(motion_dir)), name,
                          record['fraction'] * 100, record['min_line'] * 100,
                          record['min_line_forearm'] * 100,
                          record['min_line_hand'] * 100,
                          record['min_clear_forearm'] * 100,
                          record['min_clear_hand'] * 100,
                          *record['forearm_angle'], record['turn_before'],
                          record['turn_after'], record['max_turn_change'],
                          record['max_flexion'], record['max_extension'],
                          record['max_radial'], record['max_ulnar'],
                          record['max_twist'],
                          record['press_wrist']['flexion_deg'],
                          record['press_wrist']['deviation_deg'],
                          record['n_violations'], record['n_waypoints']))
            else:
                print('{} {}: 横並びしない ({})'.format(
                    motion_dir, name, record['reason']))
    done = [r for r in records if r.get('fraction', 0.0) >= 1.0]
    print('[summary] 横並び 100%: {}/{}'.format(len(done), len(records)))
    measured = [r for r in records if 'min_line' in r]
    if measured:
        def stats(key):
            values = np.array([r[key] for r in measured])
            return '中央値 {:.1f} / 最小 {:.1f} / 最大 {:.1f}'.format(
                np.median(values), values.min(), values.max())
        for key, label, scale in (
                ('min_line', '線分の最小距離 [cm]', 100),
                ('min_clear_forearm', '検証の前腕までの距離 [cm]', 100),
                ('min_clear_hand', '検証の手の甲までの距離 [cm]', 100)):
            values = np.array([r[key] for r in measured]) * scale
            print('[summary] {}: 中央値 {:.1f} / 最小 {:.1f} / 最大 {:.1f}'
                  .format(label, np.median(values), values.min(),
                          values.max()))
        for key, label in (('max_flexion', '掌屈'), ('max_extension', '背屈'),
                           ('max_radial', '撓屈'), ('max_ulnar', '尺屈'),
                           ('max_twist', 'ひねり'),
                           ('max_turn_change', '握りの向きの変化')):
            print('[summary] {} の最大 [度]: {}'.format(label, stats(key)))
        drops = np.array([r['palm_drop'] for r in measured]) * 100
        print('[summary] 掌を下ろした量 [cm]: 中央値 {:.1f} / 最小 {:.1f} / '
              '最大 {:.1f}'.format(np.median(drops), drops.min(), drops.max()))
        with_lat = [r for r in measured if 'lateral_after' in r]
        if with_lat:
            for key, label in (('lateral_after', '終点の台車の横の距離 [m]'),
                               ('lateral_overshoot',
                                '一度離れてから戻った量 [m]')):
                values = np.array([r[key] for r in with_lat])
                print('[summary] {}: 中央値 {:.2f} / 最大 {:.2f}'.format(
                    label, np.median(values), values.max()))
        print('[summary] 手首の目安を超える waypoint がある人: {}/{}'.format(
            sum(r['n_violations'] > 0 for r in measured), len(measured)))
        times = [r['compute_time'] for r in measured if r.get('compute_time')]
        if times:
            print('[summary] 計画時間 [秒]: 平均 {:.2f} / 最大 {:.2f}'.format(
                np.mean(times), np.max(times)))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(records, f, indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
