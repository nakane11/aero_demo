#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""掌を押し込んだ後、掌を合わせたまま台車を動かして、人とさらに横並びに
なる位置へ移る区間 (横並び移動) を計画する。

押し込み (``solve_palm_ik`` の後処理姿勢) までは従来どおりで、台車は人の
立ち位置付近の窓 (``solve_person_ik_side_by_side``) で解いているが、人の
手の位置に縛られるため、人より前に出たり人の方へ斜めを向いたりすることが
残る (docs/handshake_base_placement.md)。押し込んだ後は、

- 掌 (押し込み目標の位置・高さ・向き) は空間に固定したまま、
- 台車を、人の立ち位置との前後ずれが小さい位置 (最優先) で、向きが人の
  正面方向に近い位置へ動かし (``solve_goal``)、
- 腕・首は各 waypoint で掌に押し付け・掌を見るよう解き直す (``_press_ik``)。

移動先は、押し込み目標に対して台車も動かす干渉回避付きバッチ IK で求める。
台車の前後は人の立ち位置 ± ``DEFAULT_GOAL_X_MARGINS`` の窓 (狭い方から、
解けるまで広げる)、向きは人の正面方向 ±30 度、左右は差し出した手の側に
絞り、窓の中では従来と同じコスト (曲げ量 + 前方ずれ + 向きのずれ) の小さい
解を選ぶ。掌と台車の距離は変わってよい。バッチの形 (目標 3 つ・初期値の数・
干渉ペア) を ``solve_person_ik`` と同じにして、JAX のコンパイル結果を共有
する (目標は同じものを 3 つ並べる)。

経路は台車の位置・向きを直線的に補間し、腕は直前の waypoint の解から解き
進める。途中で腕が届かない・干渉する waypoint があれば、その手前までで
止める。

検証は人体との距離 (6 cm) と自己干渉。ロボットの差し出す腕の肘から先と、人の
差し出した手・前腕は手を合わせているので距離を見ない (押し込み姿勢の検証と
同じ考え方)。
"""

import math
import time

import numpy as np
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import normalize_mask
from skrobot.coordinates.math import rpy_matrix

import solve_palm_ik as spik


# 移動先の台車の前後位置を、人の立ち位置からどれだけずらしてよいか [m]。
# 狭い方から試し、解けなければ広げる。
DEFAULT_GOAL_X_MARGINS = (0.05, 0.15, 0.3, 0.5)

# 移動先のコストで、人の正面方向の前方ずれ [m] に掛ける重み。前後位置を
# 最優先にするため、押し込み姿勢の候補選び (DEFAULT_FRONT_OFFSET_WEIGHT)
# より大きくする。x の窓 (DEFAULT_GOAL_X_MARGINS) はワールドの x に沿って
# いるので、人が斜めを向いているときの前後ずれはこのコストで抑える。
GOAL_FRONT_OFFSET_WEIGHT = 100.0

# 移動先の候補 (コストの安い順) のうち、押し込み IK・検証・経路まで試す数。
MAX_GOAL_CANDIDATES = 5

# 1 waypoint あたりの台車の移動量の上限 (位置 [m]・向き [rad])。
MAX_BASE_STEP = 0.05
MAX_YAW_STEP = math.radians(5.0)

# 移動先のコストが、押し込み時よりこれ以上下がらなければ動かない。
MIN_COST_IMPROVEMENT = 0.5

# 隣り合う waypoint の間で、関節角がこれより大きく変わったら IK が別の解へ
# 飛んだとみなして、そこで止める [rad]。
MAX_JOINT_STEP = math.radians(20.0)

# 移動先までのうち動けた割合がこれに満たなければ、横並び移動はしない。
MIN_FRACTION = 0.3


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def placement_cost(base, standing_xy, facing,
                   front_offset_weight=GOAL_FRONT_OFFSET_WEIGHT,
                   facing_yaw_weight=spik.DEFAULT_FACING_YAW_WEIGHT):
    """台車 ``base`` ``(x, y, yaw)`` の横並びからのずれのコスト (前方ずれ
    [m] と向きのずれ [rad] の絶対値の重み付き和、曲げ量は含まない)。"""
    front = float(np.dot(np.asarray(base[:2]) - standing_xy, facing))
    yaw = _wrap(base[2] - math.atan2(facing[1], facing[0]))
    return front_offset_weight * abs(front) + facing_yaw_weight * abs(yaw)


def _place(robot, angle_vector, base):
    robot.angle_vector(np.asarray(angle_vector, dtype=np.float64))
    robot.newcoords(Coordinates(pos=[float(base[0]), float(base[1]), 0.0],
                                rot=rpy_matrix(float(base[2]), 0.0, 0.0)))


def _press_ik(robot, robot_arm, target, gaze_point):
    """台車をいまの位置に固定し、腕を押し込み目標 ``target`` に、カメラの
    光軸を ``gaze_point`` に向ける (``solve_palm_ik.solve_post_process`` と
    同じ IK)。視線が解けなければ首はそのままで腕だけを解く。腕が解けな
    ければ ``False`` (ロボットは呼び出し前の姿勢に戻る)。"""
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    gaze_coords = spik.camera_optical_coords(robot)
    stop = spik.DEFAULT_POST_PROCESS_IK_STOP
    thre = spik.DEFAULT_POST_PROCESS_IK_THRE
    rthre = spik.DEFAULT_POST_PROCESS_IK_RTHRE
    gaze_rthre = spik.DEFAULT_POST_PROCESS_GAZE_IK_RTHRE
    result = robot.inverse_kinematics(
        target_coords=[target, spik._gaze_target(gaze_coords, gaze_point)],
        move_target=[move_target, gaze_coords],
        link_list=[whole_body.link_list, robot.head.link_list],
        position_mask=[normalize_mask(True), normalize_mask(False)],
        rotation_mask=[normalize_mask(True), normalize_mask('xy')],
        stop=stop, thre=[thre, thre], rthre=[rthre, gaze_rthre],
        revert_if_fail=True)
    if result is not False:
        spik._reaim_gaze(robot, gaze_coords, gaze_point, stop, gaze_rthre)
        return True
    result = robot.inverse_kinematics(
        target, move_target=move_target, link_list=whole_body.link_list,
        stop=stop, thre=thre, rthre=rthre, revert_if_fail=True)
    return result is not False


def _below_elbow(link, robot_arm):
    """``link`` が ``robot_arm`` の腕の肘から先 (手・指を含む) か。"""
    elbow = '{}_elbow_joint'.format(robot_arm)
    while link is not None:
        joint = getattr(link, 'joint', None)
        if joint is not None and joint.name == elbow:
            return True
        link = getattr(link, 'parent_link', None)
    return False


class TransitionChecker(object):
    """横並び移動の各姿勢の干渉検証 (自己干渉の貫通・人体からの距離)。
    ロボットの差し出す腕の肘から先と、人の差し出した手・前腕
    (``offered_hand_obstacle_indices``) の組は、手を合わせているので
    見ない。"""

    def __init__(self, robot_arm, hand, verification_pairs, joint_positions,
                 clearance=spik.DEFAULT_HOVER_HUMAN_CLEARANCE,
                 tolerance=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE):
        self.clearance = clearance
        self.tolerance = tolerance
        self.self_pairs = spik.self_collision_pairs(verification_pairs)
        self.obstacles = (spik.human_body_obstacles(joint_positions)
                          if joint_positions else None)
        offered = spik.offered_hand_obstacle_indices(hand)
        source = spik.human_clearance_pairs(verification_pairs)
        self.clearance_pairs = None
        if source is not None:
            pairs = [pair for pair in source
                     if not (isinstance(pair[1], int) and pair[1] in offered
                             and _below_elbow(pair[0], robot_arm))]
            self.clearance_pairs = (
                source.with_pairs(pairs)
                if isinstance(source, spik.VerificationPairs) else pairs)

    def check(self, robot):
        """``robot`` の今の姿勢を検証し、問題があればその説明、無ければ
        ``None`` を返す。"""
        dist, pair = spik.collision_pairs_min_distance(
            robot, self.self_pairs, None, return_pair=True)
        if dist < -self.tolerance:
            return '自己干渉 {:.3f} m ({} x {})'.format(
                -dist, *spik.collision_pair_name(pair))
        if not self.obstacles or not self.clearance_pairs:
            return None
        clearances = spik.human_obstacle_clearances(
            robot, self.clearance_pairs, self.obstacles,
            cull_distance=self.clearance)
        if not clearances:
            return None
        index = min(clearances, key=clearances.get)
        if clearances[index] < self.clearance:
            return '人体 ({}) まで {:.3f} m'.format(
                spik.human_obstacle_names()[index], clearances[index])
        return None


def _batch_ik(robot, robot_arm, hand, target, seed_av, collision_joints,
              collision_pairs, base_limits, attempts_per_pose):
    """押し込み目標 ``target`` に対して台車も動かす干渉回避付きバッチ IK
    (``solve_person_ik`` と同じ形: 目標 3 つ (同じものを並べる)・初期値・
    干渉ペア)。人の差し出した手・前腕・上腕は手を合わせているので障害物
    から外す (遠くのダミーにする、個数は変えない)。差し出さない腕・首の
    初期値は ``seed_av``。収束した解 ``[(angle_vector, (x, y, yaw)),
    ...]`` を返す。"""
    obstacles = spik.human_body_obstacles(collision_joints)
    removed = spik.offered_hand_obstacle_indices(hand) | {
        spik.human_obstacle_names().index('{0}Shoulder-{0}Elbow'.format(hand))}
    obstacles = [spik._dummy_cylinder(o.radius) if i in removed else o
                 for i, o in enumerate(obstacles)]
    pairs = None
    if collision_pairs:
        links_by_name = {link.name: link for link in
                         list(robot.link_list)
                         + list(getattr(robot, 'extra_collision_links', []))}
        offered = sorted(spik.offered_hand_obstacle_indices(hand))
        pairs = list(collision_pairs) + [
            (links_by_name[name.format(robot_arm)], i)
            for name in spik.OFFERED_HAND_PENALTY_LINKS
            if name.format(robot_arm) in links_by_name for i in offered]
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    # バッチ IK は台車をワールド原点から動かす (base_limits もその前提の
    # 絶対座標)。
    _place(robot, seed_av, (0.0, 0.0, 0.0))
    restore = spik.restrict_joint_range_margin(
        whole_body.link_list,
        spik.DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO)
    try:
        angle_vectors, base_poses, success_flags, _ = \
            robot.batch_inverse_kinematics(
                target_coords=[target] * spik.NUM_TURN_CANDIDATES,
                move_target=getattr(robot,
                                    '{}arm_end_coords'.format(robot_arm)),
                link_list=whole_body.link_list,
                position_mask=True, rotation_mask=True,
                stop=spik.DEFAULT_COLLISION_IK_STOP,
                thre=spik.DEFAULT_COLLISION_IK_THRE,
                rthre=spik.DEFAULT_COLLISION_IK_RTHRE,
                initial_angles='current',
                attempts_per_pose=attempts_per_pose,
                return_all_attempts=True,
                backend='jax',
                use_base='planar', base_limits=base_limits,
                collision_obstacles=obstacles,
                collision_weight=spik.DEFAULT_COLLISION_WEIGHT,
                collision_margin=spik.DEFAULT_COLLISION_MARGIN,
                self_collision=pairs is not None,
                collision_pairs=pairs,
                self_collision_margin=spik.DEFAULT_SELF_COLLISION_MARGIN,
                collision_geometry=spik.DEFAULT_IK_COLLISION_GEOMETRY)
    finally:
        restore()
    solutions = []
    for index, ok in enumerate(success_flags):
        if not ok:
            continue
        base_pose = base_poses[index]
        rot = np.asarray(base_pose.worldrot())
        solutions.append((
            np.asarray(angle_vectors[index], dtype=np.float64),
            (float(base_pose.worldpos()[0]), float(base_pose.worldpos()[1]),
             math.atan2(rot[1, 0], rot[0, 0]))))
    return solutions


def solve_goal_candidates(robot, robot_arm, hand, target, seed_av,
                          translated_joints, collision_joints,
                          collision_pairs, base_limits,
                          x_margins=DEFAULT_GOAL_X_MARGINS,
                          attempts_per_pose=spik.DEFAULT_ATTEMPTS_PER_POSE):
    """移動先の台車の候補を、前後の窓の狭い方から (候補が出るまで) バッチ
    IK で解き、コスト (曲げ量 + 前方ずれ + 向きのずれ) の安い順に返す。

    Returns
    -------
    (candidates, x_margin)
        ``candidates`` は ``(cost, angle_vector, base (x, y, yaw))`` の
        リスト、``x_margin`` は候補が出た窓。
    """
    standing = spik.human_standing_xy(translated_joints)
    facing = spik.human_facing_direction(translated_joints)
    human_yaw = math.atan2(facing[1], facing[0])
    # 台車の可動域 (ワールドの x/y に沿った箱) で人の前後を絞れるよう、
    # 人の立ち位置まわりに -human_yaw 回して、人の正面方向が +x の座標系で
    # 解く (解いた台車の位置姿勢は元の座標系に戻す)。
    to_local = _rotation_about(-human_yaw, standing)
    to_world = _rotation_about(human_yaw, standing)
    local_joints = {name: to_local.transform_point(p)
                    for name, p in translated_joints.items()}
    local_collision_joints = {name: to_local.transform_point(p)
                              for name, p in collision_joints.items()}
    local_target = Coordinates(
        pos=to_local.transform_point(target.worldpos()),
        rot=to_local.rot @ target.worldrot())
    limits = list(base_limits)
    side_sign = spik.offered_hand_side_sign(hand, local_joints, None)
    if side_sign is not None:
        limits[1] = spik.restrict_base_y_range_to_hand_side(
            limits[1], side_sign)
    limits[2] = spik.restrict_base_yaw_range_to_human_facing(
        limits[2], 0.0,
        margin=math.radians(spik.DEFAULT_BASE_YAW_FACING_MARGIN_DEG))
    bend_cost_indices = spik._joint_bend_cost_indices(robot, robot_arm)
    for x_margin in x_margins:
        window = [spik.restrict_base_x_range_to_human_standing(
            limits[0], standing[0], x_margin), limits[1], limits[2]]
        candidates = []
        for av, base in _batch_ik(
                robot, robot_arm, hand, local_target, seed_av,
                local_collision_joints, collision_pairs, window,
                attempts_per_pose):
            xy = to_world.transform_point([base[0], base[1], 0.0])[:2]
            base = (float(xy[0]), float(xy[1]), base[2] + human_yaw)
            candidates.append(
                (spik._joint_bend_cost_from_vector(av, bend_cost_indices)
                 + placement_cost(base, standing, facing), av, base))
        if candidates:
            candidates.sort(key=lambda c: c[0])
            return candidates, x_margin
    return [], None


class _rotation_about(object):
    """``center_xy`` を通る鉛直軸まわりに ``theta`` 回す変換。"""

    def __init__(self, theta, center_xy):
        c, s = math.cos(theta), math.sin(theta)
        self.rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.center = np.array([center_xy[0], center_xy[1], 0.0])

    def transform_point(self, point):
        point = np.asarray(point, dtype=np.float64)
        return (self.rot @ (point - self.center) + self.center).tolist()


def plan_path(robot, robot_arm, target, start_av, start_base, goal_base,
              goal_av, checker):
    """押し込み姿勢から移動先の台車 ``goal_base`` まで、台車を直線的に
    補間しながら、腕・首を直前の解から解き進めた waypoint を返す。解けない
    waypoint は、直前の解に移動先の姿勢 ``goal_av`` への補間を 1 区間分
    足した初期値でも試す。それでもだめならその手前までで止める。

    Returns
    -------
    (waypoints, fraction, reason)
        ``fraction`` は移動先までのうち進めた割合、``reason`` は止めた
        理由 (最後まで進めたら ``None``)。
    """
    start_base = np.asarray(start_base, dtype=np.float64)
    goal_base = np.asarray(goal_base, dtype=np.float64)
    delta = goal_base - start_base
    n = max(2, int(math.ceil(max(
        np.linalg.norm(delta[:2]) / MAX_BASE_STEP,
        abs(delta[2]) / MAX_YAW_STEP))))
    start_av = np.asarray(start_av, dtype=np.float64)
    step_av = (np.asarray(goal_av, dtype=np.float64) - start_av) / n
    prev = start_av
    waypoints = []
    reason = None
    for k in range(1, n + 1):
        base = start_base + delta * k / n
        for seed in (prev, prev + step_av):
            _place(robot, seed, base)
            if _press_ik(robot, robot_arm, target, target.worldpos()):
                break
        else:
            reason = '{}/{} で腕が掌に届かない'.format(k, n)
            break
        av = robot.angle_vector().copy()
        if float(np.max(np.abs(av - prev))) > MAX_JOINT_STEP:
            reason = '{}/{} で関節角が飛ぶ'.format(k, n)
            break
        problem = checker.check(robot)
        if problem is not None:
            reason = '{}/{} で{}'.format(k, n, problem)
            break
        waypoints.append(dict(
            base_position=[float(base[0]), float(base[1]), 0.0],
            base_yaw=float(base[2]),
            joint_angle_vector=[float(v) for v in av]))
        prev = av
    return waypoints, len(waypoints) / float(n), reason


def plan_transition(robot, robot_arm, hand, post, translated_joints,
                    collision_joints, verification_pairs, collision_pairs,
                    base_limits,
                    attempts_per_pose=spik.DEFAULT_ATTEMPTS_PER_POSE):
    """押し込み姿勢 ``post`` (``solve_post_process`` の結果 dict) から、
    掌を固定したまま台車を横並びの位置へ動かす waypoint を計画する.

    ``translated_joints`` は立ち位置・向きの判定用、``collision_joints``
    は干渉判定用 (体幹をずらした) の骨格 (どちらも IK と同じ座標系)。
    ``base_limits`` は台車の可動域全体 (人ごとに絞る前のもの)。

    Returns
    -------
    dict
        ``verified`` (横並び移動をするか)、``waypoints`` (押し込み姿勢の
        次から移動先まで、``plan_handshake_motion`` と同じ形)、
        ``joint_names``、``fraction`` (移動先までのうち進めた割合)、
        ``x_margin`` (移動先を解いた前後の窓)、``cost_before``/
        ``cost_after`` (配置のコスト)、``reason`` (止めた・しなかった理由)、
        ``placement_before``/``placement_after``
        (``base_placement_metrics``)、``compute_time`` [秒]。
    """
    t0 = time.time()
    result = dict(verified=False, waypoints=[],
                  joint_names=list(post['joint_names']), fraction=0.0,
                  x_margin=None, reason=None,
                  placement_before=None, placement_after=None)
    start_base = np.array([post['base_position'][0],
                           post['base_position'][1], post['base_yaw']])
    start_av = np.asarray(post['joint_angle_vector'], dtype=np.float64)
    standing = spik.human_standing_xy(translated_joints)
    facing = spik.human_facing_direction(translated_joints)
    if standing is None or facing is None:
        result.update(reason='人の立ち位置・向きが骨格から求まらない',
                      compute_time=time.time() - t0)
        return result
    result['placement_before'] = spik.base_placement_metrics(
        translated_joints, post['base_position'], post['base_yaw'])
    cost_before = placement_cost(start_base, standing, facing)
    result['cost_before'] = cost_before

    target = Coordinates(pos=list(post['target_position']),
                         rot=np.asarray(post['target_rot']))
    candidates, x_margin = solve_goal_candidates(
        robot, robot_arm, hand, target, start_av, translated_joints,
        collision_joints, collision_pairs, base_limits,
        attempts_per_pose=attempts_per_pose)
    result['x_margin'] = x_margin
    checker = TransitionChecker(robot_arm, hand, verification_pairs,
                                collision_joints)
    reasons = []
    best = None
    tried = 0
    for _, goal_av, goal_base in candidates:
        if tried >= MAX_GOAL_CANDIDATES:
            break
        # yaw を押し込み時から近い回り方の値にする。
        goal_base = np.array([goal_base[0], goal_base[1], start_base[2]
                              + _wrap(goal_base[2] - start_base[2])])
        if cost_before - placement_cost(goal_base, standing, facing) \
                < MIN_COST_IMPROVEMENT:
            reasons.append('移動先が押し込み時と変わらない')
            continue
        # 移動先の姿勢そのものを、首も含めて押し込み姿勢に解き直して検証
        # してから経路を解く。
        _place(robot, goal_av, goal_base)
        if not _press_ik(robot, robot_arm, target, target.worldpos()):
            reasons.append('移動先で押し込み IK が解けない')
            continue
        problem = checker.check(robot)
        if problem is not None:
            reasons.append('移動先で{}'.format(problem.split(' ')[0]))
            continue
        tried += 1
        waypoints, fraction, why = plan_path(
            robot, robot_arm, target, start_av, start_base, goal_base,
            robot.angle_vector().copy(), checker)
        if why is not None:
            reasons.append('経路の{}'.format(why))
        if best is None or fraction > best[1]:
            best = (waypoints, fraction)
        if fraction >= 1.0:
            break
    if not candidates:
        reasons.append('移動先の IK が解けない')
    if best is None or best[1] < MIN_FRACTION:
        result.update(reason=', '.join(sorted(set(reasons))) or None,
                      cost_after=cost_before, compute_time=time.time() - t0)
        return result
    waypoints, fraction = best
    last = waypoints[-1]
    end = (last['base_position'][0], last['base_position'][1],
           last['base_yaw'])
    result.update(
        verified=True, waypoints=waypoints, fraction=fraction,
        reason=None if fraction >= 1.0 else ', '.join(sorted(set(reasons))),
        cost_after=placement_cost(end, standing, facing),
        placement_after=spik.base_placement_metrics(
            translated_joints, last['base_position'], last['base_yaw']),
        compute_time=time.time() - t0)
    return result


def transition_summary(transition):
    """``plan_transition`` の結果の 1 行の説明。``fraction``/``x_margin``
    が無い古い形式の JSON も表示できるようにする (ビューアの描画が例外で
    途中で止まると、ロボットが前の waypoint の姿勢のまま残るため)。"""
    if not transition['verified']:
        return 'しない ({})'.format(transition.get('reason'))
    before = transition['placement_before'] or {}
    after = transition['placement_after'] or {}
    text = ('前方ずれ {:+.2f} -> {:+.2f} m、向きのずれ {:+.0f} -> {:+.0f} 度、'
            '方位 {:+.0f} -> {:+.0f} 度 (移動先の {:.0f}%、窓 ±{} m、'
            '{:.2f} 秒)'.format(
                before.get('front_offset', float('nan')),
                after.get('front_offset', float('nan')),
                before.get('yaw_offset_deg', float('nan')),
                after.get('yaw_offset_deg', float('nan')),
                before.get('bearing_deg', float('nan')),
                after.get('bearing_deg', float('nan')),
                transition.get('fraction', float('nan')) * 100.0,
                transition.get('x_margin'),
                transition.get('compute_time', float('nan'))))
    if transition.get('reason'):
        text += ' / 途中で止めた: {}'.format(transition['reason'])
    return text
