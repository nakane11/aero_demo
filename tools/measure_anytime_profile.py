#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""探索を最初の解で止めずに続けたとき、解の良さがいつ・どれだけ良くなるかを測る
(探索の続けどきを決めるための性能曲線)。

通常の探索は、hover の候補を「ロボットの関節の曲げ量・台車のずれ」のコスト順に
調べ、押し込みまで通った最初の候補で止める。横並び移動はその 1 つにだけ解く。
ここでは候補を ``--max-candidates`` まで止めずに調べ、押し込みまで通った候補の
先頭 ``--max-transitions`` 個に横並び移動 (寄せまで) を解いて、次を記録する。

- b: 横並びまで進めた割合 ``fraction``、寄せた後の横の距離 ``lateral_after``
- a: 横並び移動中の人の腕の LUBA (Kee & Karwowski 2001) の不快度スコアの合計
  (成分ごとに経路での最悪の段階のスコア)
- 押し込みが最後まで解けない候補の、押し込みの経路で進めた割合と、そこから
  人の掌 (押し込みの最終目標) までの残りの距離

時刻は、候補を順に「検証・押し込み → (通れば) 横並び移動」と調べたときの経過時間
(バッチ IK を含む) に直して出す。x の窓は通常の探索と同じ順に解き、押し込みまで
通った候補が ``--max-transitions`` 個そろったら次の窓は解かない。

Usage
-----
    python3 tools/measure_anytime_profile.py \\
        [--input-set data/synthetic/seed0_ik_targets] [--out out.json] \\
        [--max-candidates 40] [--max-transitions 6] [--limit N]
"""

import argparse
import json
import math
import os
import sys
import time
import zlib

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
_SCRIPTS_DIR = os.path.join(_ROOT_DIR, 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import numpy as np  # noqa: E402

import side_by_side_transition as sbs  # noqa: E402
import solve_palm_ik as spik  # noqa: E402


def luba_component_score(key, value):
    """LUBA の成分 ``key`` の角度 ``value`` [度] (正負は sbs と同じ) のスコア。"""
    stages = sbs.LUBA_STANDING_SCORES[key][0 if value >= 0.0 else 1]
    if stages is None:
        stages = sbs.LUBA_STANDING_SCORES[key][0]
    for end, score in stages:
        if abs(value) <= end:
            return score
    return stages[-1][1]


def luba_score(human_max):
    """``sbs.human_arm_max`` の結果から LUBA のスコアの合計と成分ごとの値。"""
    if not human_max:
        return None, {}
    parts = {}
    for key, (low, high) in (human_max.get('joints') or {}).items():
        parts[key] = max(luba_component_score(key, low),
                         luba_component_score(key, high))
    if 'flexion_deg' in human_max:
        parts['wrist_flexion'] = max(
            luba_component_score('wrist_flexion', human_max['flexion_deg']),
            luba_component_score('wrist_flexion',
                                 -human_max['extension_deg']))
        parts['wrist_deviation'] = max(
            luba_component_score('wrist_deviation', human_max['radial_deg']),
            luba_component_score('wrist_deviation',
                                 -human_max['ulnar_deg']))
    return float(sum(parts.values())), parts


def load_robot():
    robot = spik.Aero(use_hand=False)
    spik.restrict_elbow_range(robot)
    spik.restrict_leg_range(robot)
    spik.restrict_waist_range(robot)
    spik.restrict_neck_range(robot)
    spik.lock_fixed_joints(robot)
    spik.apply_collision_model(robot)
    spik.apply_hand_box(robot)
    spik.other_hand_points('r')
    return robot


def ik_kwargs_for(collision_pairs, verification_pairs, max_candidates):
    """``solve_palm_ik.main`` の既定と同じ ``solve_person_ik`` の引数。"""
    return dict(
        attempts_per_pose=spik.DEFAULT_ATTEMPTS_PER_POSE,
        collision_weight=spik.DEFAULT_COLLISION_WEIGHT,
        collision_margin=spik.DEFAULT_COLLISION_MARGIN,
        self_collision=collision_pairs is not None,
        collision_pairs=collision_pairs,
        self_collision_weight=None,
        self_collision_margin=spik.DEFAULT_SELF_COLLISION_MARGIN,
        collision_ik_stop=spik.DEFAULT_COLLISION_IK_STOP,
        collision_ik_thre=spik.DEFAULT_COLLISION_IK_THRE,
        collision_ik_rthre=spik.DEFAULT_COLLISION_IK_RTHRE,
        collision_joint_limit_margin_ratio=(
            spik.DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO),
        verification_pairs=verification_pairs,
        collision_verify_tolerance=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE,
        post_process_ik_stop=spik.DEFAULT_POST_PROCESS_IK_STOP,
        post_process_thre=spik.DEFAULT_POST_PROCESS_IK_THRE,
        post_process_rthre=spik.DEFAULT_POST_PROCESS_IK_RTHRE,
        post_process_max_candidates=max_candidates,
        hover_human_clearance=spik.DEFAULT_HOVER_HUMAN_CLEARANCE)


def press_partial(palm, rot, failure):
    """押し込みが解けなかった候補の、進めた割合と人の掌までの残りの距離 [m]。
    失敗した点 ``point`` (1 から) の手前の点まで進めたとする (最終目標で失敗
    したら、その手前の途中の点まで)。"""
    n = failure['n_points']
    reached = (failure['point'] - 1) / float(n)
    end = spik.press_in_target_position(palm, rot, 1.0)
    here = spik.press_in_target_position(palm, rot, reached)
    hover = spik.press_in_target_position(palm, rot, 0.0)
    return dict(stage=failure['stage'], point=failure['point'], n_points=n,
                reached=reached,
                remaining=float(np.linalg.norm(end - here)),
                hover_distance=float(np.linalg.norm(end - hover)))


def setup_transition_robot():
    """横並び移動用のロボット (``plan_handshake_motion.py`` と同じく IK とは別の
    モデルで、台車の全可動域でバッチ IK を温める。狭い可動域で先に解くと以後の
    結果が壊れる)。``(robot, collision_pairs, verification_pairs,
    base_limits)``。"""
    robot = load_robot()
    verification_pairs = spik.build_verification_pairs_for_model(robot)
    pairs = spik.load_collision_pairs(
        os.path.join(_SCRIPTS_DIR, 'collision_pairs.json'), robot)
    base_limits = [tuple(spik.DEFAULT_BASE_X_RANGE),
                   tuple(spik.DEFAULT_BASE_Y_RANGE),
                   tuple(spik.DEFAULT_BASE_YAW_RANGE)]
    for robot_arm, hand in (('l', 'R'), ('r', 'L')):
        picked, _, _ = spik.solve_person_ik(
            robot, spik._WARMUP_PALM, hand, robot_arm,
            spik.human_body_obstacles({}), base_limits=base_limits,
            collision_pairs=pairs, joint_positions={},
            verification_pairs=verification_pairs)
        sbs.warmup_batch_ik(
            robot, robot_arm, hand,
            picked[1] if picked is not None else robot.angle_vector(),
            pairs, base_limits)
    return robot, pairs, verification_pairs, base_limits


def measure_person(robot, name, palm_data, joint_positions, collision_pairs,
                   verification_pairs, base_limits, args, tr):
    hand = palm_data.get('offered_hand')
    palm = palm_data.get(hand) if hand in ('L', 'R') else None
    if palm is None:
        return None
    robot_arm = spik.DEFAULT_ROBOT_ARM[hand]
    np.random.seed(zlib.crc32(name.encode()))

    offset = spik.human_translation_offset(
        joint_positions, front_distance=spik.HUMAN_FRONT_DISTANCE)
    translated = spik.translate_joint_positions(joint_positions, offset)
    collision_joints = spik.shift_torso_joints_from_surface(
        translated, offset, 0.0)
    obstacles = ([] if collision_pairs is None
                 else spik.human_body_obstacles(collision_joints))
    palm = spik.translate_palm(palm, offset)

    person_limits = base_limits
    side = spik.offered_hand_side_sign(hand, translated, palm)
    if side is not None:
        person_limits = [person_limits[0],
                         spik.restrict_base_y_range_to_hand_side(
                             person_limits[1], side),
                         person_limits[2]]
    human_yaw = spik.human_facing_yaw(translated)
    if human_yaw is not None:
        person_limits = [person_limits[0], person_limits[1],
                         spik.restrict_base_yaw_range_to_human_facing(
                             person_limits[2], human_yaw,
                             margin=math.radians(
                                 spik.DEFAULT_BASE_YAW_FACING_MARGIN_DEG))]
    standing = spik.human_standing_xy(translated)
    standing_x = None if standing is None else float(standing[0])
    rots = spik.palm_to_target_rots(palm, hand, robot_arm)
    turn_degs = spik.turn_candidates_deg(hand, palm)
    ik_kwargs = ik_kwargs_for(collision_pairs, verification_pairs,
                              args.max_candidates)

    margins = list(spik.DEFAULT_BASE_X_STANDING_MARGINS)
    if standing_x is None:
        margins = [-1.0]
    # 探索の時計 (検証・押し込みと横並び移動を順に行ったときの経過時間)。
    clock = 0.0
    events = []
    n_press_ok = 0
    for window, x_margin in enumerate(margins):
        if n_press_ok >= args.max_transitions:
            break
        limits = [spik.restrict_base_x_range_to_human_standing(
            person_limits[0], standing_x, margin=x_margin),
            person_limits[1], person_limits[2]]
        collect = []
        t0 = time.time()
        _, ik_time, _ = spik.solve_person_ik(
            robot, palm, hand, robot_arm, obstacles, base_limits=limits,
            front_offset_weight=spik.DEFAULT_FRONT_OFFSET_WEIGHT,
            facing_yaw_weight=spik.DEFAULT_FACING_YAW_WEIGHT,
            joint_positions=collision_joints,
            placement_joint_positions=translated,
            collect=collect, exhaustive=True, **ik_kwargs)
        clock += ik_time
        events.append(dict(kind='batch_ik', window=window, x_margin=x_margin,
                           clock=clock, n_examined=len(collect)))
        previous = t0 + ik_time
        for rec in collect:
            clock += rec['time'] - previous
            previous = rec['time']
            ev = dict(kind='candidate', window=window, x_margin=x_margin,
                      clock=clock, cost=rec['cost'],
                      turn_deg=float(turn_degs[rec['turn_index']]),
                      outcome=rec['outcome'])
            if rec['outcome'] == 'press_fail' and rec.get('press_failure'):
                ev['press'] = press_partial(
                    palm, rots[rec['turn_index']], rec['press_failure'])
            if rec['outcome'] == 'press_ok':
                post = rec['post_result']
                yaw = spik._twisted_shoulder_yaw_deg(
                    robot, robot_arm, post['joint_angle_vector'])
                ev['twisted_shoulder_yaw'] = yaw
                if n_press_ok < args.max_transitions:
                    tr_robot, tr_pairs, tr_verify, tr_limits = tr
                    if args.stage1_first:
                        # 寄せの前の段階 (approach=False、実機ノードの動く前の
                        # 計画と同じ) だけの時間と、終点の人との横の距離。
                        ts = time.time()
                        stage1 = sbs.plan_transition(
                            tr_robot, robot_arm, hand, post,
                            float(turn_degs[rec['turn_index']]), translated,
                            collision_joints, tr_verify, tr_pairs, tr_limits,
                            approach=False,
                            time_limit=args.transition_time_limit)
                        t_stage1 = time.time() - ts
                        clock += t_stage1
                        laterals = stage1.get('laterals') or []
                        s1_score, _ = luba_score(stage1.get('human_max'))
                        ev.update(
                            stage1_time=t_stage1,
                            stage1_verified=bool(stage1.get('verified')),
                            stage1_fraction=float(
                                stage1.get('fraction') or 0.0),
                            stage1_lateral=(float(laterals[-1])
                                            if laterals else None),
                            stage1_luba=s1_score)
                    ts = time.time()
                    transition = sbs.plan_transition(
                        tr_robot, robot_arm, hand, post,
                        float(turn_degs[rec['turn_index']]), translated,
                        collision_joints, tr_verify, tr_pairs, tr_limits,
                        approach=True,
                        time_limit=args.transition_time_limit)
                    t_transition = time.time() - ts
                    clock += t_transition
                    approach = transition.get('approach') or {}
                    score, parts = luba_score(transition.get('human_max'))
                    ev.update(
                        clock=clock, transition_time=t_transition,
                        verified=bool(transition.get('verified')),
                        fraction=float(transition.get('fraction') or 0.0),
                        lateral_before=approach.get('lateral_before'),
                        lateral_after=approach.get('lateral_after'),
                        luba=score, luba_parts=parts,
                        reason=transition.get('reason'),
                        timed_out=bool(transition.get('timed_out')))
                n_press_ok += 1
            events.append(ev)
    return dict(name=name, hand=hand, robot_arm=robot_arm, events=events)


def quality_key(ev):
    """解の良さの並べ方 (小さいほど良い): 横並びまで進めたか → 寄せた後の横の
    距離 → LUBA。"""
    full = ev.get('verified') and ev.get('fraction', 0.0) >= 1.0
    lateral = ev.get('lateral_after')
    return (0 if full else 1, -ev.get('fraction', 0.0),
            lateral if lateral is not None else 9.9,
            ev.get('luba') if ev.get('luba') is not None else 99.0)


def summarize(results):
    lines = []
    solved = [r for r in results
              if any('fraction' in e for e in r['events'])]
    lines.append('対象 {} 人、押し込みまで通った候補がある人 {} 人'.format(
        len(results), len(solved)))
    improved = 0
    gains_lat, gains_luba, first_t, best_t = [], [], [], []
    for r in solved:
        evs = [e for e in r['events'] if 'fraction' in e]
        first = evs[0]
        best = min(evs, key=quality_key)
        first_t.append(first['clock'])
        best_t.append(best['clock'])
        if quality_key(best) < quality_key(first):
            improved += 1
            if (first.get('lateral_after') is not None
                    and best.get('lateral_after') is not None):
                gains_lat.append(first['lateral_after']
                                 - best['lateral_after'])
            if first.get('luba') is not None and best.get('luba') is not None:
                gains_luba.append(first['luba'] - best['luba'])
        lines.append(
            '  {:<16} 候補 {} 個: 最初 {:.2f}s ({}) → 最良 {:.2f}s ({})'.format(
                r['name'], len(evs), first['clock'], _q_text(first),
                best['clock'], _q_text(best)))
    if solved:
        lines.append('最初の解より良い解が後ろにあった人: {}/{}'.format(
            improved, len(solved)))
        lines.append('最初の解の時刻 平均 {:.2f}s・最良の解の時刻 平均 {:.2f}s'
                     .format(np.mean(first_t), np.mean(best_t)))
        if gains_lat:
            lines.append('  寄せた後の横の距離の改善 平均 {:.3f} m (最大 {:.3f})'
                         .format(np.mean(gains_lat), np.max(gains_lat)))
        if gains_luba:
            lines.append('  LUBA の改善 平均 {:.2f} (最大 {:.2f})'.format(
                np.mean(gains_luba), np.max(gains_luba)))
    # 押し込みが最後まで解けない候補 (hover は通った) だけの人。
    hover_only = [r for r in results
                  if not any(e.get('outcome') == 'press_ok'
                             for e in r['events'])
                  and any(e.get('outcome') == 'press_fail'
                          for e in r['events'])]
    lines.append('hover は通るが押し込みが最後まで解けない人: {} 人'.format(
        len(hover_only)))
    for r in hover_only:
        presses = [e['press'] for e in r['events'] if 'press' in e]
        best = max(presses, key=lambda p: p['reached'])
        lines.append('  {:<16} 進めた割合 最大 {:.2f} (残り {:.3f} m / hover '
                     'から {:.3f} m、{} で失敗)'.format(
                         r['name'], best['reached'], best['remaining'],
                         best['hover_distance'], best['stage']))
    # 押し込みまで通った人も含めた、押し込み失敗の候補の進めた割合の分布。
    reached = [e['press']['reached'] for r in results for e in r['events']
               if 'press' in e]
    if reached:
        hist = np.histogram(reached, bins=[0, 0.125, 0.25, 0.5, 0.75, 1.01])
        lines.append('押し込みが解けない候補 {} 個の進めた割合の分布: {}'.format(
            len(reached), ', '.join(
                '{:.2f}〜{:.2f}: {}'.format(lo, hi, n)
                for lo, hi, n in zip(hist[1][:-1], hist[1][1:], hist[0]))))
    return '\n'.join(lines)


def _q_text(ev):
    return '横並び {:.0f}%{}{}'.format(
        100.0 * ev.get('fraction', 0.0),
        '' if ev.get('lateral_after') is None
        else '・寄せ後 {:.2f}m'.format(ev['lateral_after']),
        '' if ev.get('luba') is None else '・LUBA {:.0f}'.format(ev['luba']))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--input-set', default=os.path.join(
        _ROOT_DIR, 'data', 'synthetic', 'seed0_ik_targets'))
    parser.add_argument('--out', default=None)
    parser.add_argument('--max-candidates', type=int, default=40,
                        help='1 つの x の窓で調べる候補の数 (既定 40)。')
    parser.add_argument('--max-transitions', type=int, default=6,
                        help='横並び移動を解く押し込みまで通った候補の数。')
    parser.add_argument('--transition-time-limit', type=float, default=None)
    parser.add_argument('--stage1-first', action='store_true',
                        help='寄せまで解く前に寄せない横並び移動 '
                        '(approach=False) も解き、時間と横の距離を記録する。')
    parser.add_argument('--limit', type=int, default=None,
                        help='先頭のこの人数だけ (確認用)。')
    parser.add_argument('--only', nargs='*', default=None)
    args = parser.parse_args()

    palms_dir = os.path.join(args.input_set, 'palms')
    skeletons_dir = os.path.join(args.input_set, 'skeletons')
    files = spik.iter_palm_files(palms_dir)
    if args.only:
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] in args.only]
    if args.limit:
        files = files[:args.limit]

    robot = load_robot()
    collision_pairs = spik.load_collision_pairs(
        os.path.join(_SCRIPTS_DIR, 'collision_pairs.json'), robot)
    verification_pairs = spik.build_verification_pairs_for_model(robot)
    base_limits = [spik.DEFAULT_BASE_X_RANGE,
                   (-spik.BASE_Y_MOVABLE_HALF_RANGE,
                    spik.BASE_Y_MOVABLE_HALF_RANGE),
                   spik.DEFAULT_BASE_YAW_RANGE]
    spik._warmup_batch_ik(robot, base_limits, ik_kwargs_for(
        collision_pairs, verification_pairs, args.max_candidates))
    tr = setup_transition_robot()

    results = []
    warmed = False
    for i, path in enumerate(files):
        name = os.path.splitext(os.path.basename(path))[0]
        skeleton = os.path.join(skeletons_dir, os.path.basename(path))
        if not os.path.exists(skeleton):
            continue
        palm_data = spik.load_palm_json(path)
        joints = spik.load_skeleton_json(skeleton)
        if not warmed:
            # 横並び移動の jit を温める (1 人目の時刻に入れない)。
            measure_person(robot, name, palm_data, joints, collision_pairs,
                           verification_pairs, base_limits, args, tr)
            warmed = True
        t0 = time.time()
        r = measure_person(robot, name, palm_data, joints, collision_pairs,
                           verification_pairs, base_limits, args, tr)
        if r is None:
            continue
        r['wall'] = time.time() - t0
        results.append(r)
        print('[{}/{}] {} {:.1f}s 候補 {}'.format(
            i + 1, len(files), name, r['wall'],
            sum(1 for e in r['events'] if e['kind'] == 'candidate')),
            flush=True)
        if args.out:
            with open(args.out, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=1,
                          default=float)
    print(summarize(results))


if __name__ == '__main__':
    main()
