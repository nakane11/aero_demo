#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``solve_palm_ik.py`` の握手姿勢 (hover 目標) へ至る接近軌道を干渉回避
付きで計画し、JSON に保存する。

台車は人の手を中心に公転+自転して最終位置へ寄り、腕は関節の線形補間で
hover 姿勢へ上げる。まず最適化なしの軌道を厳密形状で検証し、通らないときだけ
jaxls (別途 pip install が必要) で最適化する。押し込み (``post_process``)
は計画・検証しない。

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
import threading
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SYNTHETIC_DIR = os.path.join(_THIS_DIR, '..', 'data', 'synthetic')
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

# waypoint 間の時間刻み [秒] (コスト正規化用。実機の速度ではない)。
DEFAULT_DT = 0.2

DEFAULT_MAX_ITERATIONS = 60

# 干渉コストが働き始める距離 [m] (人体 / 自己干渉)。
DEFAULT_COLLISION_ACTIVATION_DISTANCE = spik.DEFAULT_COLLISION_MARGIN
DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE = (
    spik.DEFAULT_SELF_COLLISION_MARGIN)

DEFAULT_SMOOTHNESS_WEIGHT = 1.0
DEFAULT_ACCELERATION_WEIGHT = 1.0

# 初期台車姿勢 [x, y, yaw] (IK と同じ座標系)。
INITIAL_BASE_POSE = (0.0, 0.0, 0.0)

# 接近開始位置の円の半径 (手から最終台車位置まで) への上乗せ [m]。
DEFAULT_APPROACH_DISTANCE = 0.3  # [m]

# 初期位置がこの距離以内なら接近開始位置を置かず初期位置から計画する [m]。
MIN_APPROACH_DISTANCE = 0.05  # [m]

# 回り込み用の接近開始位置候補の角度の刻みと上限。
APPROACH_CANDIDATE_ANGLE_STEP = math.radians(30.0)  # [rad]
APPROACH_CANDIDATE_MAX_ANGLE = math.radians(120.0)  # [rad]

# orbit_tangent_start の方位角の探索刻みと向きのずれの許容幅。
ORBIT_TANGENT_SEARCH_STEP = math.radians(0.5)  # [rad]
ORBIT_TANGENT_MAX_MISMATCH = math.radians(3.0)  # [rad]

# 螺旋の自転を最短の向きで回ると、台車の正面から見た人・掌の方位がこの角度を
# 超える (首ヨーの可動域 約 ±120 度で追えず、カメラの視界から外れる) なら、
# 逆向きの方が超え方が小さいときだけ逆向きに回る (``_spin_toward_human``)。
# 方位は螺旋を ORBIT_SPIN_BEARING_SAMPLES 点で調べる。
ORBIT_SPIN_MAX_BEARING = math.radians(110.0)  # [rad]
ORBIT_SPIN_BEARING_SAMPLES = 41

# lead-in は人からこの距離以内の waypoint だけ検証する [m]。
LEAD_IN_CHECK_RADIUS = 1.0  # [m]

# lead-in の waypoint 間隔 (並進 / 回頭)。
LEAD_IN_STEP = 0.1  # [m]
LEAD_IN_ANGLE_STEP = math.radians(10.0)  # [rad]

# 自己干渉の許容貫通量 [m]。
DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE = (
    spik.DEFAULT_COLLISION_VERIFY_TOLERANCE)  # [m]

# 押し込み前 (lead-in・接近軌道) の検証での人体との距離 [m]。
# 横並び移動 (TRANSITION_HUMAN_CLEARANCE) と揃える。
DEFAULT_HUMAN_CLEARANCE = 0.05  # [m]

# 接近区間のこの割合以降で首を押し込み姿勢の角度へ補間する。
HEAD_GAZE_BLEND_START = 0.5


def human_body_cylinder_obstacles(
        joint_positions, hand=None,
        activation_distance=DEFAULT_COLLISION_ACTIVATION_DISTANCE):
    """``solve_palm_ik.human_body_obstacles`` の円柱を ``add_collision_cost``
    の ``world_obstacles`` 形式 (``type='cylinder'``) に変換する。

    ``hand`` (差し出した手) を渡すと、その掌・指は遠くのダミーに置き換え、
    前腕は ``activation_distance`` で ``OFFERED_HAND_CLEARANCE`` の距離に
    なるよう細らせる。

    個数は人物によらず一定 (jit キャッシュのため)。
    """
    cylinders = spik.human_body_obstacles(joint_positions)
    if hand is not None:
        forearm = spik.human_obstacle_names().index(
            '{0}Elbow-{0}Wrist'.format(hand))
        for i in spik.offered_hand_obstacle_indices(hand):
            cylinders[i] = (
                spik.offered_hand_penalty_obstacle(
                    cylinders[i], activation_distance) if i == forearm
                else spik._dummy_cylinder(cylinders[i].radius))
    obstacles = []
    for cyl in cylinders:
        obstacles.append(dict(
            type='cylinder',
            center=[float(v) for v in cyl.worldpos()],
            rotation=[[float(v) for v in row] for row in cyl.worldrot()],
            radius=float(cyl.radius),
            half_height=float(cyl.height) / 2.0,
        ))
    return obstacles


def arms_down_angles(robot, joint_list):
    """腕を下ろした姿勢 (``reset_pose`` の肘を 0 度にしたもの) を
    ``joint_list`` 順で返す。"""
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
    """公転の中心 (掌の水平位置)。"""
    center = handshake.get('palm_position', handshake['target_position'])
    return np.asarray(center[:2], dtype=np.float64)


def orbit_direction(center_xy, initial_base_pose, human_xy, base_goal):
    """中心から初期位置へ向かう水平単位ベクトル (重なれば ``approach_direction``)。"""
    direction = (np.asarray(initial_base_pose[:2], dtype=np.float64)
                 - center_xy)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return approach_direction(human_xy, base_goal)
    return direction / norm


def orbit_base_start(base_goal, center_xy, direction, distance,
                     initial_base_pose=None):
    """中心から ``direction`` へ (最終台車位置までの距離 + ``distance``) 離れ、
    中心を向く接近開始位置 ``[x, y, yaw]``。

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
    接近開始位置 ``[x, y, yaw]``。初期位置が円の内側か、揃う位置が無ければ
    ``None``。
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
                            center, avoid_xy, spin_toward_human=False)
        if sweep is None or travel_len < 1e-6:
            continue
        dtheta = sweep[1]
        # 経路の出だしの向き。
        v =(r1 - r0) * e_r + r0 * dtheta * e_t
        cos_mismatch = float(np.dot(v, travel)) / (
            float(np.linalg.norm(v)) * travel_len)
        if cos_mismatch < min_cos:
            continue
        # 公転の向きごとに最も揃うものを残し、経路長が短い方を選ぶ。
        side = 1 if dtheta >= 0.0 else -1
        length = travel_len + 0.5 * (r0 + r1) * abs(dtheta)
        if side not in best_per_side \
                or cos_mismatch > best_per_side[side][0]:
            best_per_side[side] = (cos_mismatch, length, p0, travel)
    if not best_per_side:
        return None
    _, _, p0, travel = min(best_per_side.values(), key=lambda b: b[1])
    return np.array([p0[0], p0[1], math.atan2(travel[1], travel[0])])


def orbit_sweep(base_start, base_goal, center_xy, avoid_xy,
                spin_toward_human=True):
    """``base_start`` から ``base_goal`` への公転・自転の量を返す。

    Returns
    -------
    (theta0, sweep, r0, r1, dyaw)
        始点の方位角 [rad]、公転角 [rad] (正で反時計回り)、始点・終点の
        半径 [m]、yaw の変化量 [rad]。半径がほぼ 0 なら ``None``。

    公転は ``avoid_xy`` (人の位置) の方位を通らない向きに回る。自転は最短の
    向きだが、``spin_toward_human`` なら人・掌が背中側を通るときだけ逆向きに
    する (``_spin_toward_human``)。
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
    if spin_toward_human and abs(dyaw) > 1e-6:
        dyaw = _spin_toward_human(theta0, sweep, r0, r1, dyaw, base_goal,
                                  center_xy, avoid_xy)
    return theta0, sweep, r0, r1, dyaw


def _orbit_max_bearing(theta0, sweep, r0, r1, dyaw, base_goal, center_xy,
                       points):
    """公転+自転 (``orbit_base_path`` と同じ補間) の間の、台車の正面から見た
    ``points`` (水平位置の列) の方位の絶対値の最大 [rad]。"""
    turn = 1.0 - (1.0 - np.linspace(0.0, 1.0, ORBIT_SPIN_BEARING_SAMPLES)) ** 2
    theta = theta0 + sweep * turn
    radius = r1 + (r0 - r1) * (1.0 - turn)
    x = center_xy[0] + radius * np.cos(theta)
    y = center_xy[1] + radius * np.sin(theta)
    yaw = float(base_goal[2]) - dyaw * (1.0 - turn)
    worst = 0.0
    for p in points:
        bearing = np.arctan2(p[1] - y, p[0] - x) - yaw
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        worst = max(worst, float(np.max(np.abs(bearing))))
    return worst


def _spin_toward_human(theta0, sweep, r0, r1, dyaw, base_goal, center_xy,
                       avoid_xy):
    """自転の向き: 最短の向き (``dyaw``) で人 (``avoid_xy``) か掌 (公転の中心)
    の方位が ``ORBIT_SPIN_MAX_BEARING`` を超える (首で追えず視界から外れる)
    なら、逆向きに 1 周ずらした向きの方が方位の最大が小さいときだけそちらに
    する。向かい合って始めると、人の左手なら反時計回り、右手なら時計回り
    (回る途中で正面が人の方を通る) になる。"""
    points = [np.asarray(avoid_xy[:2], dtype=np.float64), center_xy]
    shortest = _orbit_max_bearing(theta0, sweep, r0, r1, dyaw, base_goal,
                                  center_xy, points)
    if shortest <= ORBIT_SPIN_MAX_BEARING:
        return dyaw
    other = dyaw - math.copysign(2 * math.pi, dyaw)
    if _orbit_max_bearing(theta0, sweep, r0, r1, other, base_goal,
                          center_xy, points) < shortest:
        return other
    return dyaw


def orbit_unwrap_start(base_start, base_goal, center_xy, avoid_xy):
    """``unwrap_start_yaw`` の公転版。"""
    sweep = orbit_sweep(base_start, base_goal, center_xy, avoid_xy)
    if sweep is None:
        return unwrap_start_yaw(base_start, base_goal)
    base_start = np.array(base_start, dtype=np.float64)
    base_start[2] = float(base_goal[2]) - sweep[4]
    return base_start


def orbit_base_path(base_start, base_goal, center_xy, avoid_xy, n):
    """中心 ``center_xy`` の周りを公転+自転する台車の経路 ``(n, 3)``。

    方位角・半径・yaw を ``1 - (1 - s)^2`` で補間する螺旋 (終点で減速)。
    始点 yaw は ``orbit_unwrap_start`` 済み前提。公転が決まらなければ直線。
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
    残りはそれを手の周りに回した位置で、台車の経路長が短い順 (回り込み用)。
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
            # 進行方向を向く (重なるときは手の方)。
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

    その場回転してから直進する。関節角は ``start_joint_angles``
    (``{関節名: 角度}``、省略時は終点のまま) から線形補間する。
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
                   joint_positions, human_xy, obstacle_cache, hand=None):
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
            obstacle_cache, hand=hand)
        for i, d, c in zip(near, checked, checked_clearances):
            distances[i] = float(d)
            clearances[i] = float(c)
    return distances, clearances


def unwrap_start_yaw(base_start, base_goal):
    """``base_start`` の yaw を ``base_goal`` から ±π 以内にしたコピー
    (遠回り防止)。"""
    base_start = np.array(base_start, dtype=np.float64)
    base_start[2] = base_goal[2] + wrap_angle(base_start[2] - base_goal[2])
    return base_start


def build_start_and_goal(robot, robot_arm, handshake, base_start, orbit):
    """始点 (腕を下ろした姿勢 + ``base_start``) と終点 (``handshake``) の
    関節角・台車姿勢を ``{robot_arm}arm_whole_body`` の関節順で求める。

    始点の yaw は ``orbit`` (``(公転の中心, 避ける位置)``) で調整する。

    Returns
    -------
    (link_list, joint_list, q_start, base_start, q_goal, base_goal)
        呼び出し後、``robot`` の台車はワールド原点 (軌道の台車列をそのまま
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
    # 差し出さない腕は軌道全体で IK 結果の姿勢に固定する。
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
    """``TrajectoryProblem`` を組み立てる。

    干渉コストは近似なので、経路は必ず ``verify_waypoints`` で検証すること。"""
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    problem = TrajectoryProblem(
        robot_model=robot, link_list=link_list, n_waypoints=n_waypoints,
        dt=dt, move_target=whole_body.end_coords, n_base_dof=3)
    problem.add_smoothness_cost(weight=smoothness_weight)
    problem.add_acceleration_cost(weight=acceleration_weight)
    problem.add_joint_limit_constraint()
    # add_self_collision_cost は add_collision_cost の primitives を使うのでこの順。
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
    'linear': '線形補間のみ (最適化なし)',
    'optimized': '軌道最適化',
}


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
    """人体の障害物と表面サンプルを 1 人分作る (``obstacle_cache``)。"""
    obstacle_links = spik.human_body_obstacles(joint_positions) \
        if joint_positions else None
    obstacle_samples = (
        [spik.cylinder_surface_samples(o) for o in obstacle_links]
        if obstacle_links else None)
    return obstacle_links, obstacle_samples


_SELF_PAIRS_CACHE = {}


def _self_pairs_cached(verification_pairs):
    # 自己干渉の組は verification_pairs ごとに 1 回だけ作る。
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
    渡すと、差し出さない手の指先と台車の箱の距離も含める。
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
                           obstacle_cache, hand=None):
    """各 waypoint の人体との最短距離から ``DEFAULT_HUMAN_CLEARANCE`` (差し
    出した手 ``hand`` の手まわりは ``OFFERED_HAND_CLEARANCE``) を引いた余裕
    [m] のリスト (0 以上で通過、障害物が無ければ ``inf``)。"""
    obstacle_links = obstacle_cache[0]
    if not obstacle_links or not waypoints:
        return [float('inf')] * len(waypoints)
    verification_pairs = spik.human_clearance_pairs(verification_pairs)
    thresholds = spik.human_clearance_thresholds(
        hand, DEFAULT_HUMAN_CLEARANCE)
    margins = []
    for wp in waypoints:
        _apply_waypoint(robot, joint_names, wp)
        per_obstacle = spik.human_obstacle_clearances(
            robot, verification_pairs, obstacle_links,
            cull_distance=DEFAULT_HUMAN_CLEARANCE)
        margins.append(spik.clearance_shortfall(per_obstacle, thresholds)[1])
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
        thresholds = spik.human_clearance_thresholds(
            handshake.get('offered_hand'), DEFAULT_HUMAN_CLEARANCE)
        if spik.clearance_shortfall(clearances, thresholds)[1] < 0.0:
            return None
    post = new.get('post_process')
    if post is not None:
        # 押し込み途中の姿勢も差し替える (検証は自己干渉のみ)。
        path = []
        for angle_vector in post.get('press_in_path', []):
            step = dict(post, joint_angle_vector=angle_vector)
            apply(step)
            if spik.collision_pairs_min_distance(
                    robot, self_pairs, joint_positions) < -tolerance:
                return None
            path.append(step['joint_angle_vector'])
        if 'press_in_path' in post:
            post['press_in_path'] = path
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
                       initial_base_pose=INITIAL_BASE_POSE, deadline=None):
    """1 人分の握手動作の軌道を計画し、結果 dict を返す。

    差し出さない手の指先が台車に入って通らない場合は、その腕を 1 段ずつ
    曲げて計画し直す。通れば ``handshake`` を破壊的に書き換え、
    ``other_arm_posture_replanned`` に姿勢 [deg] を入れる。
    ``deadline`` (``time.time()`` の値) を過ぎたら残りの候補・姿勢を試さない。
    """
    start_time = time.time()
    tolerance = DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE
    motion = _plan_person_motion_once(
        robot, robot_arm, handshake, joint_positions, human_xy, args,
        verification_pairs, solver, initial_base_pose=initial_base_pose,
        deadline=deadline)
    posture = current_other_arm_posture(robot_arm, handshake)
    while (not (motion['verified'] and motion['lead_in_verified'])
           and (deadline is None or time.time() < deadline)
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
            verification_pairs, solver, initial_base_pose=initial_base_pose,
            deadline=deadline)
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
                             initial_base_pose=INITIAL_BASE_POSE,
                             deadline=None):
    """1 人分の軌道を計画する (差し出さない腕の差し替えなし)。

    まず角度 0 の接近開始位置で最適化込みで計画し、通らなければ
    ``approach_start_candidates`` の残りを経路長順に最適化なしで試す
    (``deadline`` を過ぎたら残りの候補と最適化を試さない)。
    結果に lead-in の検証結果 (``lead_in_*``) と採用角度 ``approach_angle``
    [rad] を入れる。``solver`` は jit キャッシュのため人物間で使い回すこと。
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
        # lead-in は始点だけで決まるので計画前に検証する。
        _,joint_list, q_start, base_start, _, _ = build_start_and_goal(
            robot, robot_arm, handshake, base_start, orbit=orbit)
        first_waypoint = trajectory_waypoints(
            robot, joint_list, [np.concatenate([q_start, base_start])])[0]
        joint_names = [j.name for j in robot.joint_list]
        lead_in = build_lead_in_waypoints(
            initial_base_pose, first_waypoint, joint_names)
        distances, clearances = verify_lead_in(
            robot, joint_names, lead_in, verification_pairs,
            joint_positions, human_xy, obstacle_cache,
            hand=handshake.get('offered_hand'))
        verified = motion_passes(distances, clearances)
        return lead_in, distances, clearances, verified

    def out_of_time():
        return deadline is not None and time.time() >= deadline

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
        # 残りの候補は最適化なしで試す。
        first_lead_in_ok = None
        for candidate in others:
            if out_of_time():
                break
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
                and first_lead_in_ok is not None and not out_of_time():
            # lead-in が通った最短の候補で最適化まで行う。
            candidate, lead_in_check = first_lead_in_ok
            best = (candidate, lead_in_check, _plan_from_start(
                robot, robot_arm, handshake, joint_positions, candidate[1],
                args, verification_pairs, solver, obstacle_cache, orbit=orbit))
        if best is None:
            # どれも通らなければ角度 0 の結果を返す (最適化なし)。
            if zero_motion is None:
                zero_motion = _plan_from_start(
                    robot, robot_arm, handshake, joint_positions, zero[1],
                    args, verification_pairs, solver, obstacle_cache,
                    optimize=False, orbit=orbit)
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
    hover 用の視線 (``post['hover_head']``、無ければ最終姿勢の値) へ線形補間する
    (破壊的。hover 時に掌が画角に入るように)。

    書き換えた waypoint は検証し直さない。書き換えたら True。
    """
    post = handshake.get('post_process')
    waypoints = motion.get('waypoints')
    if post is None or not waypoints:
        return False
    joint_names = motion['joint_names']
    post_angles = dict(zip(post['joint_names'], post['joint_angle_vector']))
    # hover の視線は、最終姿勢 (顔が手を向く) ではなくカメラが手を捉える視線にする。
    post_angles.update(post.get('hover_head') or {})
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

    線形補間を検証し、通ればそれを返す。通らなければ jaxls で最適化する
    (検証に通らなければ余裕の大きい方を ``verified: false`` で返す)。
    ``optimize`` が偽なら最適化しない。
    """
    link_list, joint_list, q_start, base_start, q_goal, base_goal = \
        build_start_and_goal(robot, robot_arm, handshake, base_start,
                             orbit=orbit)

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
            obstacle_cache, hand=handshake.get('offered_hand'))
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

    # 試した候補ごとの合否 (candidates_tried)。
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
        return candidate

    initial_traj = build_initial_trajectory(
        q_start, base_start, q_goal, base_goal, args.n_waypoints,
        orbit=orbit)
    best = record(make_candidate(initial_traj, 'linear'))
    if (best['verified'] and not getattr(args, 'force_optimize', False)) \
            or not optimize:
        return best

    # 検証で台車・腕が動いたので、台車を原点・腕を始点に戻す。
    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    for joint, angle in zip(joint_list, q_start):
        joint.joint_angle(float(angle))

    activation_distance = getattr(args, 'collision_activation_distance',
                                  DEFAULT_COLLISION_ACTIVATION_DISTANCE)
    world_obstacles = human_body_cylinder_obstacles(
        joint_positions, hand=handshake.get('offered_hand'),
        activation_distance=activation_distance)
    collision_link_list = spik.collision_link_list_for_arm(robot)
    problem = build_problem(
        robot, robot_arm, link_list, args.n_waypoints, DEFAULT_DT,
        world_obstacles, collision_link_list,
        activation_distance,
        DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE,
        DEFAULT_SMOOTHNESS_WEIGHT, DEFAULT_ACCELERATION_WEIGHT,
        collision_weight=100.0,
        self_collision_weight=100.0)
    solve_start = time.time()
    result = solver.solve(problem, initial_traj)
    candidate = record(make_candidate(
        result.trajectory, 'optimized', 0, float(result.cost),
        time.time() - solve_start))
    if candidate['verified'] or motion_margin(candidate) > motion_margin(best):
        best = candidate
    return best


def _warmup_solver(robot, solver, args):
    """左右両腕の軌道最適化を 1 回ずつ解き jit コンパイルを済ませる
    (cache key は障害物の個数のみなので空の骨格で足りる)。"""
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


def transition_start_time(robot, motion, handshake, initial_base_pose):
    """実機ノードで、計画が終わってから横並び移動を始めるまでの時間 [秒] の
    見積もり。表示ツールのシミュレーション表示と同じ ``PlanTimeline`` で、
    lead-in・接近・押し込みを実機と同じ区間時間で動かし、うなずき・hover での
    補正・押し込み後の待ちは ``handshake_stages.SIMULATED_DURATIONS`` の固定値と
    する (実機ノードの裏の計画は、計画の直後からこの時刻までに終われば使われる)。
    ``robot`` の姿勢は書き換わる。"""
    from aero_demo import handshake_stages
    from aero_demo.handshake_timeline import PlanTimeline
    import handshake_viewer_common as viewer
    names = list(motion['joint_names'])
    display, n_approach = viewer.build_display_waypoints(
        dict(motion, transition=None), handshake)
    start_angles = dict(zip([joint.name for joint in robot.joint_list],
                            arms_down_angles(robot, robot.joint_list)))
    prepend = build_lead_in_waypoints(
        list(initial_base_pose), display[0], names,
        start_joint_angles=start_angles)
    timeline = PlanTimeline(
        prepend + display, len(prepend), n_approach, 0, names,
        {joint.name: joint for joint in robot.joint_list})
    # n_transition=0 の timeline は押し込み後の「握る」までなので、横並び移動の
    # 開始待ちを足す。
    return timeline.duration + handshake_stages.SIMULATED_DURATIONS[
        handshake_stages.STAGE_TRANSITION_WAIT[0]]


def main():
    parser = argparse.ArgumentParser(
        description='握手姿勢へ至る干渉回避付きの軌道を計画し JSON に保存する。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_SYNTHETIC_DIR, 'random_handshake_poses'),
        help='solve_palm_ik.py の出力 JSON のディレクトリ。')
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_SYNTHETIC_DIR, 'random_human_poses'),
        help='骨格 JSON のディレクトリ (--input-dir と同じファイル名で対応)。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_SYNTHETIC_DIR, 'random_motion_poses'),
        help='軌道 JSON の保存先ディレクトリ。')
    parser.add_argument(
        '--wait-file', type=str, default=None,
        help='warmup 後、このファイルができるまで待ってから --input-dir を読む。')
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
        help='押し込み後の横並び移動も計画し transition に保存する。実機ノード '
             'と同じく、動く前に確保する計画 (寄せなし) を締め切りの残りで解き '
             '(transition)、寄せと残りの掌の向きは裏の計画と同じ '
             'refine_transition で解く (transition_refined、実行する方は '
             'transition_executed)。')
    parser.add_argument(
        '--no-refine-budget', action='store_true',
        help='裏の計画 (refine_transition) を、実機ノードで横並び移動を始める '
             '時刻 (transition_start_time) で打ち切らず最後まで解く。')
    parser.add_argument(
        '--plan-time-limit', type=float, default=spik.PLAN_TIME_LIMIT,
        help='1 人の計画全体 (IK -> 軌道計画 -> 確保する横並び移動) の時間の '
             '上限 [秒] (実機ノードと同じ、既定 {})。IK の時間は握手姿勢 JSON '
             'の collision_ik_time + candidate_selection_time を引く。0 以下なら'
             '上限なし。'.format(spik.PLAN_TIME_LIMIT))
    parser.add_argument(
        '--collision-pairs', type=str,
        default=os.path.join(_THIS_DIR, 'collision_pairs.json'),
        help='横並び移動のバッチ IK で使う干渉ペア。')
    parser.add_argument(
        '--goal-leg-rise', choices=sbs.GOAL_LEG_RISE_MODES,
        default=sbs.GOAL_LEG_RISE,
        help='横並び移動で腰を押し込み終了時より上げるか (off: 制限なし、'
            'strict: 上げない、fallback: 候補がなくなるときだけ制限なし、'
            '既定 {})。'.format(sbs.GOAL_LEG_RISE))
    parser.add_argument(
        '--goal-crouch-weight', type=float, default=sbs.GOAL_CROUCH_WEIGHT,
        help='横並び移動先のしゃがみ量の重み (既定 {})。'.format(
            sbs.GOAL_CROUCH_WEIGHT))
    parser.add_argument(
        '--goal-bow-weight', type=float, default=sbs.GOAL_BOW_WEIGHT,
        help='横並び移動先の腰の前傾 [rad] の二乗の重み (既定 {})。'.format(
            sbs.GOAL_BOW_WEIGHT))
    parser.add_argument(
        '--leg-rise-allow', type=float, default=None,
        help='strict のとき横並びで腰が上がってよい量 [m] (既定は上げない)。')
    parser.add_argument(
        '--leg-low-margin', type=float, default=None,
        help='足首の上限・膝の下限を全域幅のこの割合だけ狭める '
            '(solve_palm_ik.py と同じにする)。')
    sbs.add_human_arm_arguments(parser)
    args = parser.parse_args()
    sbs.apply_human_arm_arguments(args)
    if args.leg_low_margin is not None:
        spik.LEG_LOW_SIDE_MARGIN_RATIO = args.leg_low_margin
    if args.leg_rise_allow is not None:
        sbs.GOAL_LEG_RISE_TOLERANCE = args.leg_rise_allow
        sbs.PATH_LEG_RISE_TOLERANCE = args.leg_rise_allow
        spik.LEG_RISE_PIN_TOLERANCE = 2.0 * args.leg_rise_allow
    sbs.GOAL_LEG_RISE = args.goal_leg_rise
    sbs.GOAL_CROUCH_WEIGHT = args.goal_crouch_weight
    sbs.GOAL_BOW_WEIGHT = args.goal_bow_weight

    files = json_io.iter_json_files(args.input_dir)
    if not files and args.wait_file is None:
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
    # 指の点群を先に作る。
    spik.other_hand_points('r')
    verification_pairs = spik.build_verification_pairs_for_model(
        robot, args.collision_verify_model)
    print('[collision-verify] 軌道の事後検証は {} モデルの {} 組で行います。'
          .format(args.collision_verify_model, len(verification_pairs)))
    # jit キャッシュのためソルバーは人物間で使い回す。
    solver = create_solver('jaxls', max_iterations=args.max_iterations,
                           verbose=False)
    # JaxBackend は作成時に x64 を有効にする。warmup が先だと float32 で
    # コンパイルされ、本番で再コンパイルが起きる。
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
        # 最初のバッチ IK は台車の全可動域で解く (狭いと以後の結果が壊れる)。
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

    if args.wait_file is not None:
        # 入力の完成を待つ。
        wait_t0 = time.time()
        while not os.path.exists(args.wait_file):
            time.sleep(0.1)
        print('[wait] 入力の握手姿勢 JSON の完成を {:.1f} 秒待ちました。'
              .format(time.time() - wait_t0))
        files = json_io.iter_json_files(args.input_dir)
        if not files:
            print('{} に握手姿勢 JSON が見つかりません。'.format(
                args.input_dir))
            return

    n_optimized = n_verified = n_total = n_not_planned = 0
    n_transition = n_transition_verified = 0
    for i, path in enumerate(files):
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        handshake = json.load(open(path))
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
        # 立ち位置が求まらなければ公称位置。
        human_xy = spik.human_standing_xy(joint_positions)
        if human_xy is None:
            human_xy = np.array([spik.HUMAN_FRONT_DISTANCE, 0.0])

        # 計画全体の締め切り (実機ノードと同じ)。IK にかかった時間を引く。
        deadline = None
        if args.plan_time_limit > 0:
            deadline = time.time() + args.plan_time_limit - (
                (handshake.get('collision_ik_time') or 0.0)
                + (handshake.get('candidate_selection_time') or 0.0))
            if time.time() >= deadline:
                json_io.save_json(out_path,
                                  not_planned_result('plan_timed_out'))
                n_not_planned += 1
                print('[{}/{}] {} -> {} (not planned: IK で計画の時間の上限 '
                      '{:.1f} 秒を過ぎた)'.format(
                          i + 1, len(files), os.path.basename(path), out_path,
                          args.plan_time_limit))
                continue

        # 干渉判定用に体幹をずらした骨格。
        collision_joints = spik.shift_torso_joints_from_surface(
            joint_positions, offset, args.torso_surface_offset)
        result = plan_person_motion(
            robot, handshake['robot_arm'], handshake, collision_joints,
            human_xy, args, verification_pairs, solver,
            initial_base_pose=np.array(args.initial_base_pose),
            deadline=deadline)
        n_total += 1
        n_optimized += int(result['optimized'])
        n_verified += int(result['verified'])
        post = handshake.get('post_process')
        if (args.side_by_side_transition and post is not None
                and result['verified'] and result['lead_in_verified']):
            # 実機ノードと同じく、軌道が通った人だけ。動く前に確保する計画
            # (寄せなし) を締め切りの残りで解く。
            common = (robot, handshake['robot_arm'],
                      handshake['offered_hand'], post, handshake['turn_deg'],
                      joint_positions, collision_joints, verification_pairs,
                      transition_pairs, base_limits)
            transition = sbs.plan_transition(
                *common, approach=False,
                time_limit=(None if deadline is None
                            else max(0.0, deadline - time.time())))
            result['transition'] = transition
            n_transition += 1
            n_transition_verified += int(transition['verified'])
            print('  [transition] 横並び移動 (確保): {}'.format(
                sbs.transition_summary(transition)))
            if transition['verified']:
                # 実機ノードが動作中に裏で解く計画 (寄せと残りの掌の向き)。
                # 実機ノードと同じく、横並び移動を始める時刻 (計画の直後から
                # transition_start_time) までに終わらなければ止めて確保した
                # 計画を使う。実行するのは確保した計画以上に進めた方
                # (_refined_transition_waypoints と同じ)。
                budget = (None if args.no_refine_budget else
                          transition_start_time(
                              robot, result, handshake,
                              args.initial_base_pose))
                stop = threading.Event()
                timer = None
                if budget is not None:
                    timer = threading.Timer(budget, stop.set)
                    timer.daemon = True
                    timer.start()
                refine_t0 = time.time()
                try:
                    refined = sbs.refine_transition(*common, transition,
                                                    stop=stop)
                except sbs.TransitionStopped:
                    refined = None
                finally:
                    if timer is not None:
                        timer.cancel()
                result['transition_refine'] = dict(
                    budget=budget, time=time.time() - refine_t0,
                    in_time=refined is not None)
                if refined is not None:
                    result['transition_refined'] = refined
                result['transition_executed'] = (
                    'refined' if refined is not None
                    and refined.get('verified')
                    and refined['fraction'] >= transition['fraction']
                    else 'secured')
                print('  [transition] 横並び移動 (裏の計画、{:.1f} 秒、横並びの'
                      '開始まで {}): {}'.format(
                          time.time() - refine_t0,
                          '-' if budget is None
                          else '{:.1f} 秒'.format(budget),
                          '間に合わず確保した計画' if refined is None
                          else sbs.transition_summary(refined)))
        json_io.save_json(out_path, result)
        if result.get('other_arm_posture_replanned') is not None:
            # 差し替えた姿勢を IK 結果の JSON にも反映する。
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
