#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``solve_palm_ik.py`` の握手姿勢 (hover 目標) へ至る接近軌道を干渉回避
付きで計画し、JSON に保存する。

台車は人間の手を中心に公転+自転して最終位置へ寄り、最後は pre-touch 姿勢
から掌の法線方向にまっすぐ近づく。まず最適化なしの軌道を厳密形状で検証し、
通らないときだけ jaxls で最適化する (jaxls は ``pip install
"git+https://github.com/brentyi/jaxls.git"`` が別途必要)。押し込み
(``post_process``) の区間は計画・検証しない。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo estimate_palm_poses.py
    rosrun aero_demo solve_palm_ik.py
    rosrun aero_demo plan_handshake_motion.py
"""

import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PKG_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import json_io  # noqa: E402  (パス追加後に import)

# jax の永続コンパイルキャッシュ (jax の import 前に設定する必要がある)。
os.environ.setdefault(
    'JAX_COMPILATION_CACHE_DIR',
    os.path.expanduser('~/.cache/jax_compilation_cache'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.planner.trajectory_optimization.problem import (  # noqa: E402
    TrajectoryProblem)
from skrobot.planner.trajectory_optimization.solvers import (  # noqa: E402
    create_solver)
from skrobot.planner.trajectory_optimization.trajectory import (  # noqa: E402
    interpolate_trajectory)

import side_by_side_transition as sbs  # noqa: E402
import solve_palm_ik as spik  # noqa: E402  (パス追加後に import)

DEFAULT_N_WAYPOINTS = 20

# waypoint 間の時間刻み [秒] (コストの正規化用で、実機の再生速度ではない)。
DEFAULT_DT = 0.2

DEFAULT_MAX_ITERATIONS = 60

# 干渉コストが働き始める距離 [m] (人体 / 自己干渉)。
DEFAULT_COLLISION_ACTIVATION_DISTANCE = spik.DEFAULT_HOVER_HUMAN_CLEARANCE
DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE = (
    spik.DEFAULT_SELF_COLLISION_MARGIN)

DEFAULT_SMOOTHNESS_WEIGHT = 1.0
DEFAULT_ACCELERATION_WEIGHT = 1.0

# ロボットの初期台車姿勢 [x, y, yaw] (IK と同じ座標系)。
INITIAL_BASE_POSE = (0.0, 0.0, 0.0)

# 接近開始位置を置く円の半径 (手から最終台車位置までの距離) への上乗せ [m]。
DEFAULT_APPROACH_DISTANCE = 0.3  # [m]

# 初期位置がこの距離以内なら接近開始位置を置かず初期位置から計画する [m]。
MIN_APPROACH_DISTANCE = 0.05  # [m]

# 回り込み用の接近開始位置候補の角度の刻みと上限。
APPROACH_CANDIDATE_ANGLE_STEP = math.radians(30.0)  # [rad]
APPROACH_CANDIDATE_MAX_ANGLE = math.radians(120.0)  # [rad]

# orbit_tangent_start の方位角の探索刻みと向きのずれの許容幅。
ORBIT_TANGENT_SEARCH_STEP = math.radians(0.5)  # [rad]
ORBIT_TANGENT_MAX_MISMATCH = math.radians(3.0)  # [rad]

# lead-in は人の立ち位置からこの距離以内の waypoint だけ検証する [m]。
LEAD_IN_CHECK_RADIUS = 1.0  # [m]

# lead-in の waypoint 間隔 (並進 / 回頭)。
LEAD_IN_STEP = 0.1  # [m]
LEAD_IN_ANGLE_STEP = math.radians(10.0)  # [rad]

# pre-touch 姿勢を掌の法線方向へ引き戻す距離 [m]。
DEFAULT_PRETOUCH_STANDOFF = 0.25  # [m]

# 軌道のうち pre-touch 姿勢までに使う割合。
DEFAULT_PRETOUCH_SPLIT = 0.75

# pre-touch 姿勢の IK (干渉回避なし) の収束条件。
DEFAULT_PRETOUCH_IK_STOP = 50
DEFAULT_PRETOUCH_IK_THRE = 0.01  # [m]
DEFAULT_PRETOUCH_IK_RTHRE = math.radians(5.0)  # [rad]

# 自己干渉の許容貫通量 [m]。
DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE = (
    spik.DEFAULT_COLLISION_VERIFY_TOLERANCE)  # [m]

# 押し込み前 (lead-in・接近軌道・hover) にロボットと人体の間に空ける距離 [m]。
DEFAULT_HUMAN_CLEARANCE = spik.DEFAULT_HOVER_HUMAN_CLEARANCE  # [m]

# 接近区間のこの割合以降で首を押し込み姿勢の角度へ補間する。
HEAD_GAZE_BLEND_START = 0.5


def human_body_cylinder_obstacles(joint_positions):
    """``solve_palm_ik.human_body_obstacles`` の円柱を ``add_collision_cost``
    の ``world_obstacles`` 形式 (``type='cylinder'``) に変換する。

    個数は人物によらず ``len(human_obstacle_names())`` で一定 (jit キャッシュ
    のため)。
    """
    obstacles = []
    for cyl in spik.human_body_obstacles(joint_positions):
        obstacles.append(dict(
            type='cylinder',
            center=[float(v) for v in cyl.worldpos()],
            rotation=[[float(v) for v in row] for row in cyl.worldrot()],
            radius=float(cyl.radius),
            half_height=float(cyl.height) / 2.0,
        ))
    return obstacles


def arms_down_angles(robot, joint_list):
    """腕を体の横に下ろした姿勢 (``reset_pose`` の肘を 0 度に伸ばしたもの) を
    ``joint_list`` の順で返す。"""
    robot.reset_pose()
    for side in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(side)).joint_angle(0.0)
    return np.array([
        float(np.clip(joint.joint_angle(), joint.min_angle, joint.max_angle))
        for joint in joint_list])


def handshake_base_goal(handshake):
    """``handshake`` の最終台車姿勢 ``[x, y, yaw]``。"""
    return np.array([handshake['base_position'][0],
                     handshake['base_position'][1],
                     handshake['base_yaw']])


def wrap_angle(angle):
    """``angle`` [rad] を [-π, π) に正規化する。"""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def approach_direction(human_xy, base_goal):
    """人間の立ち位置から最終台車位置へ向かう水平単位ベクトル。"""
    direction = np.array([base_goal[0] - human_xy[0],
                          base_goal[1] - human_xy[1]], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    return np.array([-1.0, 0.0]) if norm < 1e-6 else direction / norm


def orbit_center_xy(handshake):
    """公転の中心 (IK の目標位置 ``target_position`` の水平位置)。"""
    return np.asarray(handshake['target_position'][:2], dtype=np.float64)


def orbit_direction(center_xy, initial_base_pose, human_xy, base_goal):
    """中心 ``center_xy`` から初期位置へ向かう水平単位ベクトル (重なる場合は
    ``approach_direction``)。"""
    direction = (np.asarray(initial_base_pose[:2], dtype=np.float64)
                 - center_xy)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return approach_direction(human_xy, base_goal)
    return direction / norm


def orbit_base_start(base_goal, center_xy, direction, distance,
                     initial_base_pose=None):
    """中心から ``direction`` へ半径 (最終台車位置までの距離 + ``distance``)
    離れ、中心を向く接近開始位置 ``[x, y, yaw]``。

    初期位置が既にこの半径以内なら (後退しないよう) 初期位置を返す。
    """
    center_xy = np.asarray(center_xy, dtype=np.float64)
    radius = float(np.linalg.norm(base_goal[:2] - center_xy)) + distance
    if initial_base_pose is not None:
        initial = np.asarray(initial_base_pose, dtype=np.float64)
        if (float(np.linalg.norm(initial[:2] - center_xy))
                <= radius + MIN_APPROACH_DISTANCE):
            return initial.copy()
    xy = center_xy + direction * radius
    return np.array([xy[0], xy[1],
                     math.atan2(-direction[1], -direction[0])])


def orbit_tangent_start(base_goal, center_xy, avoid_xy, distance,
                        initial_base_pose):
    """lead-in の直進が公転経路 (``orbit_base_path``) の出だしの接線になる
    接近開始位置 ``[x, y, yaw]`` (yaw は進行方向)。

    初期位置が既に円の内側か、向きが揃う位置が無ければ ``None``。
    """
    center = np.asarray(center_xy, dtype=np.float64)
    initial = np.asarray(initial_base_pose, dtype=np.float64)
    r1 = float(np.linalg.norm(np.asarray(base_goal[:2]) - center))
    r0 = r1 + distance
    if float(np.linalg.norm(initial[:2] - center)) \
            <= r0 + MIN_APPROACH_DISTANCE:
        return None
    min_cos = math.cos(ORBIT_TANGENT_MAX_MISMATCH)
    best_per_side = {}  # 公転の向き (+1/-1) -> (cos_mismatch, 経路長, p0, travel)
    for theta0 in np.arange(-math.pi, math.pi, ORBIT_TANGENT_SEARCH_STEP):
        e_r = np.array([math.cos(theta0), math.sin(theta0)])
        e_t = np.array([-e_r[1], e_r[0]])
        p0 = center + r0 * e_r
        travel = p0 - initial[:2]
        travel_len = float(np.linalg.norm(travel))
        sweep = orbit_sweep(np.array([p0[0], p0[1], 0.0]), base_goal,
                            center, avoid_xy)
        if sweep is None or travel_len < 1e-6:
            continue
        dtheta = sweep[1]
        # 経路の出だしの向き (半径方向 r1 - r0 と接線方向 r0 * dtheta の比)。
        v =(r1 - r0) * e_r + r0 * dtheta * e_t
        cos_mismatch = float(np.dot(v, travel)) / (
            float(np.linalg.norm(v)) * travel_len)
        if cos_mismatch < min_cos:
            continue
        # 公転の向きごとに最もよく揃うものを残し、最後に経路長が短い方を選ぶ。
        side = 1 if dtheta >= 0.0 else -1
        length = travel_len + 0.5 * (r0 + r1) * abs(dtheta)
        if side not in best_per_side \
                or cos_mismatch > best_per_side[side][0]:
            best_per_side[side] = (cos_mismatch, length, p0, travel)
    if not best_per_side:
        return None
    _, _, p0, travel = min(best_per_side.values(), key=lambda b: b[1])
    return np.array([p0[0], p0[1], math.atan2(travel[1], travel[0])])


def orbit_sweep(base_start, base_goal, center_xy, avoid_xy):
    """``base_start`` から ``base_goal`` への公転・自転の量を返す。

    Returns
    -------
    (theta0, sweep, r0, r1, dyaw)
        始点の方位角 [rad]、公転角 [rad] (正で反時計回り)、始点・終点の
        半径 [m]、yaw の変化量 [rad]。半径がほぼ 0 なら ``None``。

    公転は ``avoid_xy`` (人の立ち位置) の方位を通らない向きに回る
    (中心と重なるなら近い方)。
    """
    center_xy = np.asarray(center_xy, dtype=np.float64)
    rel0 = np.asarray(base_start[:2], dtype=np.float64) - center_xy
    rel1 = np.asarray(base_goal[:2], dtype=np.float64) - center_xy
    r0, r1 = float(np.linalg.norm(rel0)), float(np.linalg.norm(rel1))
    if r0 < 1e-6 or r1 < 1e-6:
        return None
    theta0 = math.atan2(rel0[1], rel0[0])
    theta1 = math.atan2(rel1[1], rel1[0])
    ccw = (theta1 - theta0) % (2 * math.pi)
    sweep = wrap_angle(theta1 - theta0)
    rel_h = np.asarray(avoid_xy[:2], dtype=np.float64) - center_xy
    if float(np.linalg.norm(rel_h)) > 1e-6:
        theta_h = math.atan2(rel_h[1], rel_h[0])
        passes_human = (theta_h - theta0) % (2 * math.pi) < ccw
        sweep = ccw - 2 * math.pi if passes_human else ccw
    dyaw = sweep + wrap_angle(
        float(base_goal[2]) - float(base_start[2]) - sweep)
    return theta0, sweep, r0, r1, dyaw


def orbit_unwrap_start(base_start, base_goal, center_xy, avoid_xy):
    """``unwrap_start_yaw`` の公転版 (``orbit_sweep`` の自転量に合わせる)。"""
    sweep = orbit_sweep(base_start, base_goal, center_xy, avoid_xy)
    if sweep is None:
        return unwrap_start_yaw(base_start, base_goal)
    base_start = np.array(base_start, dtype=np.float64)
    base_start[2] = float(base_goal[2]) - sweep[4]
    return base_start


def orbit_base_path(base_start, base_goal, center_xy, avoid_xy, n):
    """中心 ``center_xy`` の周りを公転+自転する台車の経路 ``(n, 3)``。

    方位角・半径・yaw を ``1 - (1 - s)^2`` で補間するアルキメデス螺旋
    (終点で減速)。始点の yaw は ``orbit_unwrap_start`` 済みである前提。
    公転が決まらなければ直線補間。
    """
    s = np.linspace(0.0, 1.0, n)
    sweep = orbit_sweep(base_start, base_goal, center_xy, avoid_xy)
    if sweep is None:
        return np.stack([np.linspace(base_start[i], base_goal[i], n)
                         for i in range(3)], axis=1)
    theta0, dtheta, r0, r1, dyaw = sweep
    turn = 1.0 - (1.0 - s) ** 2
    theta = theta0 + dtheta * turn
    radius = r1 + (r0 - r1) * (1.0 - turn)
    path = np.stack([center_xy[0] + radius * np.cos(theta),
                     center_xy[1] + radius * np.sin(theta),
                     float(base_goal[2]) - dyaw * (1.0 - turn)], axis=1)
    path[0] = base_start
    path[-1] = base_goal
    return path


def approach_start_candidates(base_goal, human_xy, distance,
                              initial_base_pose, orbit_center):
    """接近開始位置の候補 ``[(角度 [rad], [x, y, yaw]), ...]``。

    先頭は角度 0 (``orbit_tangent_start``、無ければ ``orbit_base_start``)。
    残りはそれを手の周りに回した位置 (yaw は初期位置からの進行方向) で、
    台車の経路長が短い順。人の背後・横からの回り込み用。
    """
    human_xy = np.asarray(human_xy[:2], dtype=np.float64)
    center = np.asarray(orbit_center[:2], dtype=np.float64)
    tangent = orbit_tangent_start(
        base_goal, center, human_xy, distance, initial_base_pose)
    if tangent is not None:
        direction = tangent[:2] - center
        direction /= float(np.linalg.norm(direction))
        zero = (0.0, tangent)
    else:
        direction = orbit_direction(
            center, initial_base_pose, human_xy, base_goal)
        zero = (0.0, orbit_base_start(
            base_goal, center, direction, distance,
            initial_base_pose=initial_base_pose))
    radius = float(np.linalg.norm(base_goal[:2] - center)) + distance
    candidates = []
    n_steps = int(round(APPROACH_CANDIDATE_MAX_ANGLE
                        / APPROACH_CANDIDATE_ANGLE_STEP))
    for k in range(1, n_steps + 1):
        for sign in (1.0, -1.0):
            angle = sign * k * APPROACH_CANDIDATE_ANGLE_STEP
            c, s = math.cos(angle), math.sin(angle)
            rotated = np.array([c * direction[0] - s * direction[1],
                                s * direction[0] + c * direction[1]])
            xy = center + rotated * radius
            # 進行方向を向く (初期位置とほぼ重なるときは手の方)。
            travel = xy - np.asarray(initial_base_pose[:2], dtype=np.float64)
            if float(np.linalg.norm(travel)) > MIN_APPROACH_DISTANCE:
                yaw = math.atan2(travel[1], travel[0])
            else:
                yaw = math.atan2(-rotated[1], -rotated[0])
            candidates.append((angle, np.array([xy[0], xy[1], yaw])))

    initial_xy = np.asarray(initial_base_pose[:2], dtype=np.float64)

    def path_length(candidate):
        start = candidate[1]
        return (float(np.linalg.norm(start[:2] - initial_xy))
                + float(np.linalg.norm(base_goal[:2] - start[:2])))
    return [zero] + sorted(candidates, key=path_length)


def build_lead_in_waypoints(initial_base_pose, first_waypoint, joint_names,
                            start_joint_angles=None):
    """初期台車姿勢から ``first_waypoint`` までの lead-in の waypoint 列
    (``first_waypoint`` 自身は含まない)。

    その場回転で終点の向きになってから直進する。関節角は
    ``start_joint_angles`` (``{関節名: 角度}``、省略時は終点のまま) から
    線形補間する。
    """
    end_base = np.array([first_waypoint['base_position'][0],
                         first_waypoint['base_position'][1],
                         first_waypoint['base_yaw']])
    start_base = unwrap_start_yaw(initial_base_pose, end_base)
    delta = end_base - start_base
    n_rotate = int(math.ceil(abs(delta[2]) / LEAD_IN_ANGLE_STEP))
    n_translate = int(math.ceil(np.linalg.norm(delta[:2]) / LEAD_IN_STEP))
    n = n_rotate + n_translate
    if n == 0:
        return []
    end_vec = np.asarray(first_waypoint['joint_angle_vector'],
                         dtype=np.float64)
    if start_joint_angles is None:
        start_vec = end_vec
    else:
        start_vec = np.array([
            start_joint_angles.get(name, angle)
            for name, angle in zip(joint_names, end_vec)])
    waypoints = []
    for i in range(n):
        t = float(i) / n
        if i < n_rotate:
            base = start_base + np.array(
                [0.0, 0.0, delta[2] * float(i) / n_rotate])
        else:
            u = float(i - n_rotate) / n_translate
            base = np.array([start_base[0] + delta[0] * u,
                             start_base[1] + delta[1] * u, end_base[2]])
        waypoints.append(dict(
            base_position=[float(base[0]), float(base[1]), 0.0],
            base_yaw=float(base[2]),
            joint_angle_vector=[
                float(v) for v in start_vec + (end_vec - start_vec) * t],
        ))
    return waypoints


def verify_lead_in(robot, joint_names, lead_in, verification_pairs,
                   joint_positions, human_xy, obstacle_cache):
    """人から ``LEAD_IN_CHECK_RADIUS`` 以内の lead-in waypoint を検証し、
    (最小距離のリスト, 人体距離の余裕のリスト) を返す (未検証は ``None``)。"""
    near = [i for i, wp in enumerate(lead_in)
            if math.hypot(wp['base_position'][0] - human_xy[0],
                          wp['base_position'][1] - human_xy[1])
            <= LEAD_IN_CHECK_RADIUS]
    distances = [None] * len(lead_in)
    clearances = [None] * len(lead_in)
    if near:
        near_waypoints = [lead_in[i] for i in near]
        checked = verify_waypoints(
            robot, joint_names, near_waypoints,
            verification_pairs, joint_positions,
            obstacle_cache=obstacle_cache)
        checked_clearances = verify_human_clearance(
            robot, joint_names, near_waypoints, verification_pairs,
            obstacle_cache)
        for i, d, c in zip(near, checked, checked_clearances):
            distances[i] = float(d)
            clearances[i] = float(c)
    return distances, clearances


def unwrap_start_yaw(base_start, base_goal):
    """``base_start`` の yaw を ``base_goal`` から ±π 以内の等価な角度にした
    コピー (線形補間で遠回りしないため。終点側は変えない)。"""
    base_start = np.array(base_start, dtype=np.float64)
    base_start[2] = base_goal[2] + wrap_angle(base_start[2] - base_goal[2])
    return base_start


def build_start_and_goal(robot, robot_arm, handshake, base_start, orbit):
    """始点 (腕を下ろした姿勢 + ``base_start``) と終点 (``handshake``) の
    関節角・台車姿勢を ``{robot_arm}arm_whole_body`` の関節順で求める。

    始点の yaw は ``orbit`` (``(公転の中心, 避ける位置)``) で
    ``orbit_unwrap_start`` により調整する。

    Returns
    -------
    (link_list, joint_list, q_start, base_start, q_goal, base_goal)
        呼び出し後、``robot`` の台車はワールド原点 (``TrajectoryProblem``
        は構築時の台車姿勢からの差分を扱うため、軌道の台車列をそのまま
        ワールド座標として読める)、腕は始点の姿勢になる。
    """
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    link_list = whole_body.link_list
    joint_list = [link.joint for link in link_list]

    q_start = arms_down_angles(robot, joint_list)

    name_to_angle = dict(zip(handshake['joint_names'],
                             handshake['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    q_goal = np.array([j.joint_angle() for j in joint_list])
    base_goal = handshake_base_goal(handshake)
    base_start = orbit_unwrap_start(base_start, base_goal, *orbit)

    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    for joint, angle in zip(joint_list, q_start):
        joint.joint_angle(float(angle))
    # 差し出さない腕は軌道全体で IK 結果の姿勢 (指先が台車にかからない
    # よう曲げたもの) に固定する。
    other = 'l' if robot_arm == 'r' else 'r'
    for joint in robot.joint_list:
        if joint.name.startswith(other + '_') and joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    return link_list, joint_list, q_start, base_start, q_goal, base_goal


def build_problem(robot, robot_arm, link_list, n_waypoints, dt,
                  world_obstacles, collision_link_list,
                  collision_activation_distance,
                  self_collision_activation_distance,
                  smoothness_weight, acceleration_weight,
                  collision_weight=100.0, self_collision_weight=100.0):
    """``TrajectoryProblem`` を組み立てる (始点・終点の値は呼び出し側が渡す)。

    干渉コストは近似なので、収束しても干渉が無い保証はない。経路は必ず
    ``verify_waypoints`` で検証すること。"""
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    problem = TrajectoryProblem(
        robot_model=robot, link_list=link_list, n_waypoints=n_waypoints,
        dt=dt, move_target=whole_body.end_coords, n_base_dof=3)
    problem.add_smoothness_cost(weight=smoothness_weight)
    problem.add_acceleration_cost(weight=acceleration_weight)
    problem.add_joint_limit_constraint()
    # add_self_collision_cost は add_collision_cost が作る primitives を使う
    # ので、この順で呼ぶ。
    problem.add_collision_cost(
        collision_link_list, world_obstacles,
        weight=collision_weight,
        activation_distance=collision_activation_distance,
        as_constraint=True)
    problem.add_self_collision_cost(
        weight=self_collision_weight,
        activation_distance=self_collision_activation_distance,
        as_constraint=True)
    problem.set_fixed_endpoints(start=True, end=True)
    return problem


def base_interpolation(base_start, base_goal, n, orbit=None):
    """台車の補間 ``(n, 3)`` (``orbit`` があれば公転+自転、無ければ直線)。"""
    if orbit is not None:
        return orbit_base_path(base_start, base_goal, orbit[0], orbit[1], n)
    return np.stack([np.linspace(base_start[i], base_goal[i], n)
                     for i in range(3)], axis=1)


def build_initial_trajectory(q_start, base_start, q_goal, base_goal,
                             n_waypoints, orbit=None):
    """始点・終点を補間した初期軌道 ``(n_waypoints, n_joints + 3)``。"""
    q_interp = interpolate_trajectory(q_start, q_goal, n_waypoints)
    base_interp = base_interpolation(
        base_start, base_goal, n_waypoints, orbit=orbit)
    traj = np.hstack([q_interp, base_interp])
    n_joints = len(q_start)
    traj[0, :n_joints] = q_start
    traj[0, n_joints:] = base_start
    traj[-1, :n_joints] = q_goal
    traj[-1, n_joints:] = base_goal
    return traj


# 結果 JSON の ``kind`` の表示名。
KIND_LABELS = {
    'pretouch': 'pre-touch 経由 (最適化なし)',
    'linear': '線形補間のみ (最適化なし)',
    'optimized': '軌道最適化',
}


def palm_normal_direction(handshake, joint_positions):
    """人間の掌の外向き法線 (掌の中心から IK の目標位置への向き)。
    掌の中心が求まらなければ ``None``。"""
    side = handshake.get('offered_hand')
    names = ['{}Hand{}'.format(side, index)
             for index in spik.HAND_PALM_LANDMARKS]
    if not all(name in joint_positions for name in names):
        return None
    palm_center = np.mean(
        [joint_positions[name] for name in names], axis=0)
    direction = np.asarray(handshake['target_position']) - palm_center
    norm = float(np.linalg.norm(direction))
    return None if norm < 1e-6 else direction / norm


def solve_pretouch_pose(robot, robot_arm, link_list, joint_list, handshake,
                        q_goal, base_goal, normal, standoff):
    """終点の手先を掌の法線方向へ ``standoff`` [m] 引き戻した pre-touch 姿勢を
    台車固定・干渉回避なしの IK で解き、``joint_list`` 順の関節角を返す
    (失敗時 ``None``)。"""
    target_position = (np.asarray(handshake['target_position'])
                       + normal * standoff)
    robot.newcoords(Coordinates(
        pos=[float(base_goal[0]), float(base_goal[1]), 0.0],
        rot=rpy_matrix(float(base_goal[2]), 0.0, 0.0)))
    for joint, angle in zip(joint_list, q_goal):
        joint.joint_angle(float(angle))
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    result = robot.inverse_kinematics(
        target_coords=Coordinates(
            pos=target_position.tolist(),
            rot=np.asarray(handshake['target_rot'], dtype=np.float64)),
        move_target=whole_body.end_coords, link_list=link_list,
        stop=DEFAULT_PRETOUCH_IK_STOP, thre=DEFAULT_PRETOUCH_IK_THRE,
        rthre=DEFAULT_PRETOUCH_IK_RTHRE, revert_if_fail=True)
    if result is False:
        return None
    return np.array([joint.joint_angle() for joint in joint_list])


def build_pretouch_trajectory(q_start, base_start, q_pre, q_goal, base_goal,
                              n_waypoints, split_ratio, orbit=None):
    """始点 → pre-touch 姿勢 → 終点 の 2 区間の軌道。後半は台車を動かさない。"""
    n_joints = len(q_start)
    split = max(2, min(n_waypoints - 1, int(n_waypoints * split_ratio)))
    traj = np.zeros((n_waypoints, n_joints + 3))
    traj[:split, :n_joints] = interpolate_trajectory(q_start, q_pre, split)
    traj[split - 1:, :n_joints] = interpolate_trajectory(
        q_pre, q_goal, n_waypoints - split + 1)
    traj[:split, n_joints:] = base_interpolation(
        base_start, base_goal, split, orbit=orbit)
    traj[split:, n_joints:] = base_goal
    return traj


def trajectory_waypoints(robot, joint_list, trajectory):
    """``trajectory`` の各行を JSON 用の waypoint (台車姿勢・全身の関節角) に
    する。``joint_list`` 以外の関節は ``robot`` の現在値のまま。"""
    n_joints = len(joint_list)
    waypoints = []
    for row in trajectory:
        for joint, angle in zip(joint_list, row[:n_joints]):
            joint.joint_angle(float(angle))
        bx, by, byaw = row[n_joints:n_joints + 3]
        robot.newcoords(Coordinates(
            pos=[float(bx), float(by), 0.0],
            rot=rpy_matrix(float(byaw), 0.0, 0.0)))
        waypoints.append(dict(
            base_position=[float(v) for v in robot.base_link.worldpos()],
            base_yaw=float(byaw),
            joint_angle_vector=[float(v) for v in robot.angle_vector()],
        ))
    return waypoints


def build_obstacle_cache(joint_positions):
    """人体の障害物と表面サンプルを 1 人分まとめて作る
    (``verify_waypoints`` の ``obstacle_cache``)。"""
    obstacle_links = spik.human_body_obstacles(joint_positions) \
        if joint_positions else None
    obstacle_samples = (
        [spik.cylinder_surface_samples(o) for o in obstacle_links]
        if obstacle_links else None)
    return obstacle_links, obstacle_samples


_SELF_PAIRS_CACHE = {}


def _self_pairs_cached(verification_pairs):
    # 自己干渉の組は verification_pairs ごとに 1 回だけ作る (計算計画を持つため)。
    if verification_pairs is None:
        return None
    key = id(verification_pairs)
    cached = _SELF_PAIRS_CACHE.get(key)
    if cached is None or cached[0] is not verification_pairs:
        cached = (verification_pairs,
                  spik.self_collision_pairs(verification_pairs))
        _SELF_PAIRS_CACHE[key] = cached
    return cached[1]


def verify_waypoints(robot, joint_names, waypoints, verification_pairs,
                     joint_positions, obstacle_cache=None, robot_arm=None):
    """各 waypoint の自己干渉の組の最小距離 [m] (負なら貫通) のリスト。

    人体との距離は ``verify_human_clearance`` で別に見る。``robot_arm`` を
    渡すと、差し出さない手の指先と台車の箱の距離も含める (検証モデルは
    指なしのため)。
    """
    if obstacle_cache is not None:
        obstacle_links, obstacle_samples = obstacle_cache
    else:
        obstacle_links, obstacle_samples = build_obstacle_cache(
            joint_positions)
    verification_pairs = _self_pairs_cached(verification_pairs)
    distances = []
    for wp in waypoints:
        name_to_angle = dict(zip(joint_names, wp['joint_angle_vector']))
        for joint in robot.joint_list:
            if joint.name in name_to_angle:
                joint.joint_angle(name_to_angle[joint.name])
        robot.newcoords(Coordinates(
            pos=wp['base_position'],
            rot=rpy_matrix(wp['base_yaw'], 0.0, 0.0)))
        dist = spik.collision_pairs_min_distance(
            robot, verification_pairs, joint_positions,
            obstacle_links=obstacle_links, obstacle_samples=obstacle_samples)
        if robot_arm is not None:
            dist = min(dist, spik.other_hand_base_clearance(robot, robot_arm))
        distances.append(dist)
    return distances


def _apply_waypoint(robot, joint_names, wp):
    name_to_angle = dict(zip(joint_names, wp['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.newcoords(Coordinates(
        pos=wp['base_position'],
        rot=rpy_matrix(wp['base_yaw'], 0.0, 0.0)))


def verify_human_clearance(robot, joint_names, waypoints, verification_pairs,
                           obstacle_cache):
    """各 waypoint の人体との最短距離から ``DEFAULT_HUMAN_CLEARANCE`` を
    引いた余裕 [m] のリスト (0 以上なら通過、障害物が無ければ ``inf``)。"""
    obstacle_links = obstacle_cache[0]
    if not obstacle_links or not waypoints:
        return [float('inf')] * len(waypoints)
    verification_pairs = spik.human_clearance_pairs(verification_pairs)
    margins = []
    for wp in waypoints:
        _apply_waypoint(robot, joint_names, wp)
        per_obstacle = spik.human_obstacle_clearances(
            robot, verification_pairs, obstacle_links,
            cull_distance=DEFAULT_HUMAN_CLEARANCE)
        margins.append(min(per_obstacle.values(), default=float('inf'))
                       - DEFAULT_HUMAN_CLEARANCE)
    return margins


def motion_passes(distances, clearance_margins):
    """貫通の許容量と人体との距離の両方を満たすか (``None`` は無視)。"""
    return (all(d is None or d >= -DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE
                for d in distances)
            and all(c is None or c >= 0.0 for c in clearance_margins))


def motion_margin(candidate):
    """候補の比較用の余裕 [m] (貫通・人体距離の悪い方、0 以上なら通過)。"""
    return min(
        min(candidate['waypoint_min_distances'])
        + DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE,
        min(candidate['waypoint_human_clearance_margins']))


def not_planned_result(reason):
    """経路計画の対象外だった人物の結果 dict。"""
    return dict(planned=False, not_planned_reason=reason)


def perturb_initial_trajectory(initial_traj, n_joints, rng, scale):
    """中間 waypoint の関節角にガウスノイズを足した warm start (台車は揺らさ
    ない)。局所解を抜けるリトライ用。"""
    traj = initial_traj.copy()
    noise = rng.normal(scale=scale, size=(traj.shape[0] - 2, n_joints))
    traj[1:-1, :n_joints] += noise
    return traj


def current_other_arm_posture(robot_arm, handshake):
    """差し出さない腕の姿勢に最も近い ``OTHER_ARM_POSTURES_DEG`` の添字。"""
    other = 'l' if robot_arm == 'r' else 'r'
    angles = dict(zip(handshake['joint_names'], handshake['joint_angle_vector']))
    shoulder_p = math.degrees(angles['{}_shoulder_p_joint'.format(other)])
    elbow = math.degrees(angles['{}_elbow_joint'.format(other)])
    errors = [abs(shoulder_p - sp) + abs(elbow - el)
              for sp, el in spik.OTHER_ARM_POSTURES_DEG]
    return int(np.argmin(errors))


def handshake_with_other_arm_posture(robot, robot_arm, handshake, posture,
                                     verification_pairs, joint_positions):
    """差し出さない腕だけを ``OTHER_ARM_POSTURES_DEG[posture]`` に差し替えた
    ``handshake`` のコピー。hover・押し込み姿勢の検証に通らなければ ``None``。
    """
    tolerance = spik.DEFAULT_COLLISION_VERIFY_TOLERANCE
    self_pairs = _self_pairs_cached(verification_pairs)
    new = copy.deepcopy(handshake)

    def apply(result):
        name_to_angle = dict(zip(result['joint_names'],
                                 result['joint_angle_vector']))
        for joint in robot.joint_list:
            if joint.name in name_to_angle:
                joint.joint_angle(name_to_angle[joint.name])
        robot.newcoords(Coordinates(
            pos=result['base_position'],
            rot=rpy_matrix(result['base_yaw'], 0.0, 0.0)))
        spik.apply_other_arm_posture(robot, robot_arm, posture)
        result['joint_names'] = [j.name for j in robot.joint_list]
        result['joint_angle_vector'] = [float(v)
                                        for v in robot.angle_vector()]

    apply(new)
    if spik.collision_pairs_min_distance(
            robot, self_pairs, joint_positions) < -tolerance:
        return None
    if joint_positions:
        clearances = spik.human_obstacle_clearances(
            robot, spik.human_clearance_pairs(verification_pairs),
            spik.human_body_obstacles(joint_positions),
            cull_distance=DEFAULT_HUMAN_CLEARANCE)
        if min(clearances.values(), default=float('inf')) \
                < DEFAULT_HUMAN_CLEARANCE:
            return None
    post = new.get('post_process')
    if post is not None:
        apply(post)
        if spik.collision_pairs_min_distance(
                robot, self_pairs, joint_positions) < -tolerance:
            return None
        post['other_arm_posture'] = list(spik.OTHER_ARM_POSTURES_DEG[posture])
    return new


def other_hand_min_clearance(robot, robot_arm, motion):
    """経路上の差し出さない手の指先と台車の箱の最短距離 [m]。"""
    joint_names = motion['joint_names']
    clearance = float('inf')
    for wp in motion['waypoints']:
        name_to_angle = dict(zip(joint_names, wp['joint_angle_vector']))
        for joint in robot.joint_list:
            if joint.name in name_to_angle:
                joint.joint_angle(name_to_angle[joint.name])
        clearance = min(clearance,
                        spik.other_hand_base_clearance(robot, robot_arm))
    return clearance


def plan_person_motion(robot, robot_arm, handshake, joint_positions, human_xy,
                       args, verification_pairs, solver,
                       initial_base_pose=INITIAL_BASE_POSE):
    """1 人分の握手動作の軌道を計画し、結果 dict を返す。

    差し出さない手の指先が台車に入って通らない場合は、その腕を 1 段ずつ
    曲げて計画し直す。通れば ``handshake`` も破壊的に書き換え、
    ``other_arm_posture_replanned`` に姿勢 [deg] を入れる。
    """
    start_time = time.time()
    tolerance = DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE
    motion = _plan_person_motion_once(
        robot, robot_arm, handshake, joint_positions, human_xy, args,
        verification_pairs, solver, initial_base_pose=initial_base_pose)
    posture = current_other_arm_posture(robot_arm, handshake)
    while (not (motion['verified'] and motion['lead_in_verified'])
           and posture + 1 < len(spik.OTHER_ARM_POSTURES_DEG)
           and other_hand_min_clearance(robot, robot_arm, motion)
           < -tolerance):
        posture += 1
        candidate = handshake_with_other_arm_posture(
            robot, robot_arm, handshake, posture, verification_pairs,
            joint_positions)
        if candidate is None:
            continue
        retry = _plan_person_motion_once(
            robot, robot_arm, candidate, joint_positions, human_xy, args,
            verification_pairs, solver, initial_base_pose=initial_base_pose)
        if retry['verified'] and retry['lead_in_verified']:
            handshake.clear()
            handshake.update(candidate)
            retry['other_arm_posture_replanned'] = list(
                spik.OTHER_ARM_POSTURES_DEG[posture])
            motion = retry
            break
    motion['compute_time'] = time.time() - start_time
    return motion


def _plan_person_motion_once(robot, robot_arm, handshake, joint_positions,
                             human_xy, args, verification_pairs, solver,
                             initial_base_pose=INITIAL_BASE_POSE):
    """1 人分の軌道を計画する (差し出さない腕の差し替えなし)。

    まず角度 0 の接近開始位置で最適化込みで計画し、lead-in かその先が
    通らなければ ``approach_start_candidates`` の残りを経路長順に最適化
    なしで試す。結果には lead-in の検証結果 (``lead_in_*``)、採用した
    候補の角度 ``approach_angle`` [rad] などを入れる。

    ``solver`` は jit キャッシュを効かせるため人物間で使い回すこと。
    """
    start_time = time.time()
    obstacle_cache = build_obstacle_cache(joint_positions)

    approach_distance = getattr(args, 'approach_distance', None)
    if approach_distance is None:
        approach_distance = DEFAULT_APPROACH_DISTANCE
    base_goal = handshake_base_goal(handshake)
    orbit =(orbit_center_xy(handshake),
            np.asarray(human_xy[:2], dtype=np.float64))
    candidates = approach_start_candidates(
        base_goal, human_xy, approach_distance, initial_base_pose,
        orbit_center=orbit[0])

    def check_lead_in(base_start):
        # lead-in は始点だけで決まるので、軌道計画の前に安く検証する。
        _,joint_list, q_start, base_start, _, _ = build_start_and_goal(
            robot, robot_arm, handshake, base_start, orbit=orbit)
        first_waypoint = trajectory_waypoints(
            robot, joint_list, [np.concatenate([q_start, base_start])])[0]
        joint_names = [j.name for j in robot.joint_list]
        lead_in = build_lead_in_waypoints(
            initial_base_pose, first_waypoint, joint_names)
        distances, clearances = verify_lead_in(
            robot, joint_names, lead_in, verification_pairs,
            joint_positions, human_xy, obstacle_cache)
        verified = motion_passes(distances, clearances)
        return lead_in, distances, clearances, verified

    zero,others = candidates[0], candidates[1:]
    zero_lead_in = check_lead_in(zero[1])
    zero_motion = None
    if zero_lead_in[-1]:
        zero_motion = _plan_from_start(
            robot, robot_arm, handshake, joint_positions, zero[1], args,
            verification_pairs, solver, obstacle_cache, orbit=orbit)
    best = None
    n_tried = 1
    if zero_motion is not None and zero_motion['verified']:
        best = (zero, zero_lead_in, zero_motion)
    else:
        # 残りの候補は (遅いので) 最適化なしで試す。
        first_lead_in_ok = None
        for candidate in others:
            n_tried += 1
            lead_in_check = check_lead_in(candidate[1])
            if not lead_in_check[-1]:
                continue
            if first_lead_in_ok is None:
                first_lead_in_ok = (candidate, lead_in_check)
            motion = _plan_from_start(
                robot, robot_arm, handshake, joint_positions, candidate[1],
                args, verification_pairs, solver, obstacle_cache,
                optimize=False, orbit=orbit)
            if motion['verified']:
                best = (candidate, lead_in_check, motion)
                break
        if best is None and zero_motion is None \
                and first_lead_in_ok is not None:
            # lead-in が通った最短の候補で最適化まで行う。
            candidate, lead_in_check = first_lead_in_ok
            best = (candidate, lead_in_check, _plan_from_start(
                robot, robot_arm, handshake, joint_positions, candidate[1],
                args, verification_pairs, solver, obstacle_cache, orbit=orbit))
        if best is None:
            # どの候補も通らなければ角度 0 の結果を返す。
            if zero_motion is None:
                zero_motion = _plan_from_start(
                    robot, robot_arm, handshake, joint_positions, zero[1],
                    args, verification_pairs, solver, obstacle_cache, orbit=orbit)
            best = (zero, zero_lead_in, zero_motion)

    ((angle, _), (lead_in, lead_in_distances, lead_in_clearances,
                  lead_in_verified), motion) = best
    motion['head_gaze_blended'] = blend_head_to_post_process(
        robot, motion, handshake)
    motion['approach_distance'] = approach_distance
    motion['approach_angle'] = float(angle)
    motion['approach_candidates_tried'] = n_tried
    motion['lead_in_waypoints'] = lead_in
    motion['lead_in_min_distances'] = lead_in_distances
    motion['lead_in_human_clearance_margins'] = lead_in_clearances
    motion['lead_in_verified'] = lead_in_verified
    motion['compute_time'] = time.time() - start_time
    return motion


def blend_head_to_post_process(robot, motion, handshake):
    """接近区間の ``HEAD_GAZE_BLEND_START`` 以降で首の角度を押し込み姿勢の
    値へ線形補間する (破壊的。hover 時点で掌が画角に入るように)。

    書き換えた waypoint は検証し直さない。書き換えたら True を返す。
    """
    post = handshake.get('post_process')
    waypoints = motion.get('waypoints')
    if post is None or not waypoints:
        return False
    joint_names = motion['joint_names']
    post_angles = dict(zip(post['joint_names'], post['joint_angle_vector']))
    head_indices = [joint_names.index(link.joint.name)
                    for link in robot.head.link_list
                    if link.joint.name in post_angles
                    and link.joint.name in joint_names]
    if not head_indices:
        return False
    target = np.array([post_angles[joint_names[i]] for i in head_indices])
    hover = np.array([waypoints[-1]['joint_angle_vector'][i]
                      for i in head_indices])
    if np.allclose(target, hover):
        return False

    n = len(waypoints)
    start = min(int(n * HEAD_GAZE_BLEND_START), n - 1)
    for k in range(start, n):
        t = (k - start + 1) / float(n - start)
        vec = waypoints[k]['joint_angle_vector']
        for i, goal in zip(head_indices, target):
            vec[i] = float(vec[i] + (goal - vec[i]) * t)
    return True


def _plan_from_start(robot, robot_arm, handshake, joint_positions, base_start,
                     args, verification_pairs, solver, obstacle_cache, orbit,
                     optimize=True):
    """台車の始点 ``base_start`` から 1 人分の軌道を計画する。

    pre-touch 経由 → 線形補間の順に検証し、通ればそれを返す。通らなければ
    jaxls で最適化し、warm start を揺らして ``args.motion_attempts`` 回まで
    解き直す (全滅なら最も余裕の大きい候補を ``verified: false`` で返す)。
    ``optimize`` が偽なら最適化しない。
    """
    link_list, joint_list, q_start, base_start, q_goal, base_goal = \
        build_start_and_goal(robot, robot_arm, handshake, base_start,
                             orbit=orbit)
    n_joints = len(q_start)

    def make_candidate(trajectory, kind, attempt=None, cost=None,
                       solve_time=0.0):
        waypoints = trajectory_waypoints(robot, joint_list, trajectory)
        joint_names = [j.name for j in robot.joint_list]
        distances = verify_waypoints(
            robot, joint_names, waypoints, verification_pairs,
            joint_positions, obstacle_cache=obstacle_cache,
            robot_arm=robot_arm)
        clearances = verify_human_clearance(
            robot, joint_names, waypoints, verification_pairs,
            obstacle_cache)
        return dict(
            planned=True,
            kind=kind,
            optimized=kind == 'optimized',
            verified=motion_passes(distances, clearances),
            attempt=attempt,
            cost=cost,
            n_waypoints=args.n_waypoints,
            dt=DEFAULT_DT,
            robot_arm=robot_arm,
            joint_names=joint_names,
            waypoints=waypoints,
            waypoint_min_distances=[float(d) for d in distances],
            waypoint_human_clearance_margins=[float(c) for c in clearances],
            solve_time=solve_time,
        )

    # 試した候補ごとの合否 (結果の candidates_tried、デバッグ用)。
    tried = []

    def record(candidate):
        dists = candidate['waypoint_min_distances']
        margins = candidate['waypoint_human_clearance_margins']
        tried.append(dict(
            kind=candidate['kind'], attempt=candidate['attempt'],
            verified=candidate['verified'],
            min_distance=float(min(dists)),
            min_distance_waypoint=int(np.argmin(dists)),
            human_clearance_margin=float(min(margins)),
            human_clearance_waypoint=int(np.argmin(margins))))
        candidate['candidates_tried'] = tried
        candidate['pretouch_status'] = pretouch_status
        return candidate

    initial_traj = build_initial_trajectory(
        q_start, base_start, q_goal, base_goal, args.n_waypoints,
        orbit=orbit)
    candidates = []
    normal = palm_normal_direction(handshake, joint_positions)
    pretouch_status = 'no_palm_normal'
    if normal is not None:
        q_pre = solve_pretouch_pose(
            robot, robot_arm, link_list, joint_list, handshake, q_goal,
            base_goal, normal, DEFAULT_PRETOUCH_STANDOFF)
        pretouch_status = 'ik_failed'
        if q_pre is not None:
            pretouch_status = 'built'
            candidates.append(('pretouch', build_pretouch_trajectory(
                q_start, base_start, q_pre, q_goal, base_goal,
                args.n_waypoints, DEFAULT_PRETOUCH_SPLIT, orbit=orbit)))
    candidates.append(('linear', initial_traj))

    force_optimize = getattr(args, 'force_optimize', False)
    best = None
    best_trajectory = None
    for kind, trajectory in candidates:
        candidate = record(make_candidate(trajectory, kind))
        if candidate['verified'] and not force_optimize:
            return candidate
        if best is None or motion_margin(candidate) > motion_margin(best):
            best = candidate
            best_trajectory = trajectory
        if candidate['verified']:
            break
    if not optimize:
        return best

    # 問題を組む前に台車を原点・腕を始点に戻す (solve_pretouch_pose が
    # 台車を動かしているため。build_start_and_goal 参照)。
    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    for joint, angle in zip(joint_list, q_start):
        joint.joint_angle(float(angle))

    world_obstacles = human_body_cylinder_obstacles(joint_positions)
    collision_link_list = spik.collision_link_list_for_arm(robot)
    problem = build_problem(
        robot, robot_arm, link_list, args.n_waypoints, DEFAULT_DT,
        world_obstacles, collision_link_list,
        getattr(args, 'collision_activation_distance',
                DEFAULT_COLLISION_ACTIVATION_DISTANCE),
        DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE,
        DEFAULT_SMOOTHNESS_WEIGHT, DEFAULT_ACCELERATION_WEIGHT,
        collision_weight=100.0,
        self_collision_weight=100.0)
    rng = np.random.RandomState(0)

    # 既に verified な warm start (force_optimize 時のみ) は 1 回だけ解く。
    max_attempts = args.motion_attempts if not best['verified'] else 1

    for attempt in range(max_attempts):
        warm_start = best_trajectory if attempt == 0 \
            else perturb_initial_trajectory(
                best_trajectory, n_joints, rng,
                0.3)
        solve_start = time.time()
        result = solver.solve(problem, warm_start)
        candidate = record(make_candidate(
            result.trajectory, 'optimized', attempt, float(result.cost),
            time.time() - solve_start))
        if candidate['verified']:
            best = candidate
            break
        if motion_margin(candidate) > motion_margin(best):
            best = candidate
    return best


def _warmup_solver(robot, solver, args):
    """左右両腕の軌道最適化を 1 回ずつ解き、jaxls の jit コンパイルを
    済ませる。cache key は障害物の個数しか見ないので空の骨格で足りる。"""
    world_obstacles = human_body_cylinder_obstacles({})
    for robot_arm in ('l', 'r'):
        warmup_start = time.time()
        robot.newcoords(Coordinates())
        robot.base_link.newcoords(Coordinates())
        link_list = getattr(
            robot, '{}arm_whole_body'.format(robot_arm)).link_list
        joint_list = [link.joint for link in link_list]
        q_start = arms_down_angles(robot, joint_list)
        q_goal = q_start
        base_start = np.array([0.0, 0.0, 0.0])
        base_goal = np.array([0.3, 0.0, 0.1])
        collision_link_list = spik.collision_link_list_for_arm(robot)
        problem = build_problem(
            robot, robot_arm, link_list, args.n_waypoints, DEFAULT_DT,
            world_obstacles, collision_link_list,
            getattr(args, 'collision_activation_distance',
                    DEFAULT_COLLISION_ACTIVATION_DISTANCE),
            DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE,
            DEFAULT_SMOOTHNESS_WEIGHT, DEFAULT_ACCELERATION_WEIGHT,
            collision_weight=100.0, self_collision_weight=100.0)
        initial_traj = build_initial_trajectory(
            q_start, base_start, q_goal, base_goal, args.n_waypoints)
        solver.solve(problem, initial_traj)
        print('[warmup] {}腕: 軌道最適化のトレース/コンパイル {:.1f} 秒'
              .format(robot_arm, time.time() - warmup_start))
    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())


def main():
    parser = argparse.ArgumentParser(
        description='握手姿勢へ至る干渉回避付きの軌道を計画し JSON に保存する。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='solve_palm_ik.py の出力 JSON のディレクトリ。')
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='骨格 JSON のディレクトリ (--input-dir と同じファイル名で対応)。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_motion_poses'),
        help='軌道 JSON の保存先ディレクトリ。')
    parser.add_argument(
        '--approach-distance', type=float,
        default=DEFAULT_APPROACH_DISTANCE,
        help='接近開始位置の円の半径への上乗せ [m] (既定 {})。'.format(
            DEFAULT_APPROACH_DISTANCE))
    parser.add_argument(
        '--initial-base-pose', type=float, nargs=3,
        default=list(INITIAL_BASE_POSE), metavar=('X', 'Y', 'YAW'),
        help='ロボットの初期台車姿勢 [m, m, rad] (IK と同じ座標系)。')
    parser.add_argument(
        '--n-waypoints', type=int, default=DEFAULT_N_WAYPOINTS,
        help='軌道の waypoint 数 (既定 {})。'.format(DEFAULT_N_WAYPOINTS))
    parser.add_argument(
        '--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
        help='jaxls の最大反復回数 (既定 {})。'.format(
            DEFAULT_MAX_ITERATIONS))
    parser.add_argument(
        '--motion-attempts', type=int, default=3,
        help='warm start を揺らして最適化を解き直す最大回数 (既定 3)。')
    parser.add_argument(
        '--force-optimize', action='store_true',
        help='最適化なしの軌道が通っても必ず jaxls で最適化する (計測用)。')
    parser.add_argument(
        '--collision-verify-model', choices=spik.COLLISION_VERIFY_MODELS,
        default=spik.DEFAULT_COLLISION_VERIFY_MODEL,
        help='事後検証のモデル (solve_palm_ik.py と同じ意味)。')
    parser.add_argument(
        '--collision-activation-distance', type=float,
        default=DEFAULT_COLLISION_ACTIVATION_DISTANCE,
        help='人体との干渉コストが働き始める距離 [m] (既定 {})。'.format(
            DEFAULT_COLLISION_ACTIVATION_DISTANCE))
    parser.add_argument(
        '--torso-surface-offset', type=float, default=0.0,
        help='体幹の関節を体の奥へずらす距離 [m] (solve_palm_ik.py と同じ値、'
            '実カメラなら {})。'.format(spik.DEFAULT_TORSO_SURFACE_OFFSET))
    parser.add_argument(
        '--side-by-side-transition', action='store_true',
        help='押し込み後の横並び移動も計画し transition に保存する。')
    parser.add_argument(
        '--collision-pairs', type=str,
        default=os.path.join(_THIS_DIR, 'collision_pairs.json'),
        help='横並び移動のバッチ IK で使う干渉ペア。')
    args = parser.parse_args()

    files = json_io.iter_json_files(args.input_dir)
    if not files:
        print('{} に握手姿勢 JSON が見つかりません。先に solve_palm_ik.py '
              'を実行してください。'.format(args.input_dir))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    robot = Aero(use_hand=False)
    spik.restrict_elbow_range(robot)
    spik.restrict_leg_range(robot)
    spik.restrict_waist_range(robot)
    spik.restrict_neck_range(robot)
    spik.lock_fixed_joints(robot)
    spik.apply_collision_model(robot)
    spik.apply_hand_box(robot)
    # 指の点群を先に作っておく (指ありモデルの読み込みに約 1 秒)。
    spik.other_hand_points('r')
    verification_pairs = spik.build_verification_pairs_for_model(
        robot, args.collision_verify_model)
    print('[collision-verify] 軌道の事後検証は {} モデルの {} 組で行います。'
          .format(args.collision_verify_model, len(verification_pairs)))
    # jit キャッシュを効かせるためソルバーは人物間で使い回す。
    solver = create_solver('jaxls', max_iterations=args.max_iterations,
                           verbose=False)
    # skrobot の JaxBackend は最初に作られた時に x64 を有効にする (バッチ IK
    # が作る)。warmup をその前に走らせると float32 でコンパイルされ、本番の
    # float64 の solve で各腕の初回に約 9 秒の再コンパイルが起きる。
    import jax
    jax.config.update('jax_enable_x64', True)
    _warmup_solver(robot, solver, args)

    transition_pairs = None
    base_limits = [tuple(spik.DEFAULT_BASE_X_RANGE),
                   tuple(spik.DEFAULT_BASE_Y_RANGE),
                   tuple(spik.DEFAULT_BASE_YAW_RANGE)]
    if args.side_by_side_transition:
        if os.path.exists(args.collision_pairs):
            transition_pairs = spik.load_collision_pairs(
                args.collision_pairs, robot)
        # 最初のバッチ IK は台車の全可動域で解く (狭い可動域で最初に呼ぶと
        # 以後の結果が壊れる)。横並び移動用のソルバーも同様。
        for robot_arm, hand in (('l', 'R'), ('r', 'L')):
            picked, _, _ = spik.solve_person_ik(
                robot, spik._WARMUP_PALM, hand, robot_arm,
                spik.human_body_obstacles({}), base_limits=base_limits,
                collision_pairs=transition_pairs, joint_positions={},
                verification_pairs=verification_pairs)
            sbs.warmup_batch_ik(
                robot, robot_arm, hand,
                picked[1] if picked is not None else robot.angle_vector(),
                transition_pairs, base_limits)

    n_optimized = n_verified = n_total = n_not_planned = 0
    n_transition = n_transition_verified = 0
    for i, path in enumerate(files):
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        handshake = json.load(open(path))
        # 押し込み姿勢が解けなかった人は計画しない。
        if (not handshake.get('target') or not handshake.get('solved')
                or handshake.get('post_process') is None):
            reason = ('not_target' if not handshake.get('target')
                      else 'ik_not_solved' if not handshake.get('solved')
                      else 'no_press')
            json_io.save_json(out_path, not_planned_result(reason))
            n_not_planned += 1
            print('[{}/{}] {} -> {} (not planned: {})'.format(
                i + 1, len(files), os.path.basename(path), out_path, reason))
            continue

        skeleton_path = os.path.join(args.skeleton_dir,
                                     os.path.basename(path))
        joint_positions = spik.load_skeleton_json(skeleton_path)
        offset = spik.human_translation_offset(
            joint_positions, front_distance=spik.HUMAN_FRONT_DISTANCE)
        joint_positions = spik.translate_joint_positions(
            joint_positions, offset)
        # 立ち位置が求まらなければ公称位置 (Aero の前方) を使う。
        human_xy = spik.human_standing_xy(joint_positions)
        if human_xy is None:
            human_xy = np.array([spik.HUMAN_FRONT_DISTANCE, 0.0])

        # 干渉判定用に体幹を体の奥へずらした骨格。
        collision_joints = spik.shift_torso_joints_from_surface(
            joint_positions, offset, args.torso_surface_offset)
        result = plan_person_motion(
            robot, handshake['robot_arm'], handshake, collision_joints,
            human_xy, args, verification_pairs, solver,
            initial_base_pose=np.array(args.initial_base_pose))
        n_total += 1
        n_optimized += int(result['optimized'])
        n_verified += int(result['verified'])
        post = handshake.get('post_process')
        if args.side_by_side_transition and post is not None:
            transition = sbs.plan_transition(
                robot, handshake['robot_arm'], handshake['offered_hand'],
                post, handshake['turn_deg'], joint_positions,
                collision_joints, verification_pairs, transition_pairs,
                base_limits)
            result['transition'] = transition
            n_transition += 1
            n_transition_verified += int(transition['verified'])
            print('  [transition] 横並び移動: {}'.format(
                sbs.transition_summary(transition)))
        json_io.save_json(out_path, result)
        if result.get('other_arm_posture_replanned') is not None:
            # 差し替えた腕の姿勢を IK 結果の JSON にも反映する。
            json_io.save_json(path, handshake)
            print('  [other-arm] 差し出さない腕を {} deg に曲げて計画し直し、'
                  '{} も更新しました。'.format(
                      result['other_arm_posture_replanned'], path))
        print('[{}/{}] {} -> {} (verified={}, lead_in_verified={}, '
              'min_dist={:.4f} m, {}, approach_angle={:.0f} 度, '
              '{:.1f} 秒)'.format(
                  i + 1, len(files), os.path.basename(path), out_path,
                  result['verified'], result['lead_in_verified'],
                  min(result['waypoint_min_distances']),
                  KIND_LABELS.get(result['kind'], result['kind']),
                  math.degrees(result['approach_angle']),
                  result['compute_time']))

    print('{}/{} verified (うち最適化まで要した人数 {} / '
          '対象外・IK失敗・押し込み失敗 {} 人)。'.format(
              n_verified, n_total, n_optimized, n_not_planned))
    if args.side_by_side_transition:
        print('横並び移動: {}/{} verified。'.format(
            n_transition_verified, n_transition))


if __name__ == '__main__':
    main()
