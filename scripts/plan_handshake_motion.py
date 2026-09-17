#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``solve_palm_ik.py`` が出力した握手姿勢 (最終姿勢 1 点) を目標として、
そこへ至る「最後の接近」の軌道 (waypoint 列) を干渉回避付きで生成し、
JSON として保存する。

計画するのは人間の近くまで来てからの区間だけ (遠方からの長距離走行は
ナビゲーションの仕事とし対象外)。軌道の始点は ``build_start_and_goal``
が次のように決める:

* 台車: 最終台車位置から見て人間の反対方向へ ``--approach-distance``
  [m] 下がった位置。向きは最終姿勢と同じ。
* 腕: 肩は ``Aero.reset_pose`` のまま、肘を伸ばして体の横に自然に
  下ろした姿勢 (``arms_down_angles`` 参照)。

終点の手前には **pre-touch 姿勢** を挟む: 目標手先姿勢を人間の掌の法線
方向へ ``--pretouch-standoff`` [m] 引き戻した位置を通常のヤコビアン法
IK で解き、軌道を「始点 → pre-touch 姿勢 → 終点」の 2 区間に分ける
(``build_pretouch_trajectory``)。最後の接近を法線方向の直線にすることで
手先が掌を通り抜けないようにする。

終点は干渉回避付き IK が収束した「掌の少し手前」の位置
(``solve_palm_ik.TARGET_HOVER_OFFSET``) までで、実際に人間に触れる
位置ではない。掌へわずかにめり込む位置まで詰める後処理判定
(``solve_palm_ik.solve_post_process``、結果は握手姿勢 JSON の
``post_process`` キー) はここでは経路として計画・検証しない。表示上
必要であれば ``view_handshake_motion.py`` が ``post_process`` を
直接読んで最後に描き足す。

対象にするのは ``solve_palm_ik.py`` の出力のうち ``target`` かつ
``solved`` が ``true`` の人物だけ (それ以外は ``planned: false`` の
JSON をそのまま書き出す)。

障害物 (人体) には ``solve_palm_ik.human_body_obstacles`` と同じ
``Cylinder`` ジオメトリを ``TrajectoryProblem.add_collision_cost`` の
``world_obstacles`` に ``'cylinder'`` 型として渡す
(``human_body_cylinder_obstacles`` 参照、scikit-robot fork の
``jaxls`` バックエンドのみ対応)。ロボット自身の干渉ジオメトリ
(``collision_link_list``) も ``apply_collision_model`` が差し替えた
box/cylinder/sphere のプリミティブ近似 (``solve_palm_ik.py`` と同じ) を
そのまま使う。

最適化中のコストは warm start を厳密解に近づけるためのもので、収束が
実際に干渉を解消した保証にはならない。``plan_person_motion`` は必ず
``solve_palm_ik.collision_pairs_min_distance`` (厳密な ``collision_
mesh`` を使う事後検証) で経路上の全 waypoint を検証し、``verified``
フラグに反映する。まず最適化を掛けずに幾何的な構成だけで作った軌道
(pre-touch 経由の線形補間) をこの厳密検証に通し、通れば最適化しない。
干渉が残ったときだけ ``jaxls`` で最適化し、それでも通らなければ warm
start を揺らして ``--motion-attempts`` 回まで解き直す。

台車を含む複数 waypoint の最適化は ``jaxls`` バックエンド
(``create_solver('jaxls')``) でのみ実装されている (台車の自由度を扱える
唯一のバックエンド)。``pip install "git+https://github.com/brentyi/
jaxls.git"`` が別途必要 (PyPI には無い)。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo estimate_palm_poses.py
    rosrun aero_demo solve_palm_ik.py
    rosrun aero_demo plan_handshake_motion.py

(いずれも --input-dir/--output-dir を省略すると、scripts/ 直下の
random_human_poses/ -> random_palm_poses/ -> random_handshake_poses/ ->
random_motion_poses/ を共通の入出力先として自動的につながる)
"""

import argparse
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

# jax の永続コンパイルキャッシュ (solve_palm_ik.py と同じ設定。jax を
# import する前に指定する必要がある)。
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

import solve_palm_ik as spik  # noqa: E402  (パス追加後に import)

# 1 人分の軌道の waypoint 数 (始点・終点を含む)。
DEFAULT_N_WAYPOINTS = 20

# waypoint 間の時間刻み [秒] (軌道の速度/加速度コストの正規化に使うだけで、
# 実機の再生速度そのものではない -- 再生時は用途に応じて調整してよい)。
DEFAULT_DT = 0.2

# jaxls (Levenberg-Marquardt + Augmented Lagrangian) の最大反復回数。
DEFAULT_MAX_ITERATIONS = 60

# 人体障害物との干渉コストが働き始める距離 [m] / 自己干渉のそれ [m]。
# solve_palm_ik.py の DEFAULT_COLLISION_MARGIN/DEFAULT_SELF_COLLISION_
# MARGIN と揃えてある。
DEFAULT_COLLISION_ACTIVATION_DISTANCE = spik.DEFAULT_COLLISION_MARGIN
DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE = (
    spik.DEFAULT_SELF_COLLISION_MARGIN)

# 軌道の滑らかさ (速度) / 加速度コストの重み。
DEFAULT_SMOOTHNESS_WEIGHT = 1.0
DEFAULT_ACCELERATION_WEIGHT = 1.0

# 軌道の始点で、最終台車位置から人間の反対方向へどれだけ下がるか [m]
# (モジュール docstring / ``approach_base_start`` 参照)。これより遠方は
# ナビゲーションの担当なので計画・検証しない。
DEFAULT_APPROACH_DISTANCE = 1.0  # [m]

# pre-touch 姿勢を、目標手先位置から掌の法線方向へどれだけ引き戻すか [m]
# (モジュール docstring / ``build_pretouch_trajectory`` 参照)。0.10/0.15 で
# は手先が掌の脇をかすめる人物があり、0.25 で試した 3 人全員が厳密検証を
# 通った。
DEFAULT_PRETOUCH_STANDOFF = 0.25  # [m]

# 軌道全体のうち、pre-touch 姿勢に到達するまでに使う割合。残りが掌の法線
# 方向に沿った最終接近になる。
DEFAULT_PRETOUCH_SPLIT = 0.75

# pre-touch 姿勢を解く通常のヤコビアン法 IK (干渉回避なし) の収束条件。
# solve_palm_ik.py の後処理判定 (solve_post_process) と同程度に緩めてある
# -- 事前段階のバッチ IK 自体の収束閾値がこれより緩いため。
DEFAULT_PRETOUCH_IK_STOP = 50
DEFAULT_PRETOUCH_IK_THRE = 0.01  # [m]
DEFAULT_PRETOUCH_IK_RTHRE = math.radians(5.0)  # [rad]

# 経路上の waypoint の事後検証で許容する最大貫通量 [m]。solve_palm_ik.py
# 自身の最終姿勢の判定 (``DEFAULT_COLLISION_VERIFY_TOLERANCE`` = 1 mm) は
# 「実際に人間に触れる/押し付ける」姿勢そのものの判定なので厳しいが、
# 経路上の通過点はそこまで厳密でなくてよいとして 1 cm まで許容する。
DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE = 0.01  # [m]


def human_body_cylinder_obstacles(joint_positions):
    """``solve_palm_ik.human_body_obstacles`` が返す全 ``Cylinder`` (体幹・
    頭部・四肢・掌・指、関節欠損はダミーで埋め済み) を、``TrajectoryProblem.
    add_collision_cost`` の ``world_obstacles`` が受け付ける ``'cylinder'``
    形式 (``{'type': 'cylinder', 'center', 'rotation', 'radius',
    'half_height'}``) に変換する。``solve_palm_ik.py``/``view_handshake_
    poses.py`` 等が実際の干渉回避・画面表示に使うのと全く同じ ``Cylinder``
    プリミティブ (掌のように骨の線分では表せない部位も、実物と同じ向き・
    大きさの円柱) から作るので、形状の食い違いは生じない。

    常に ``len(human_obstacle_names())`` 個のシリンダーを返す (人物に
    よらず一定 -- モジュール docstring 参照)。
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
    """腕を体の横に自然に下ろした姿勢を、``joint_list`` の順で返す。

    ``Aero.reset_pose`` は肘を -135 度まで曲げるが、これだと手先が
    肩の高さまで上がった「構え」の姿勢になるため、肩の角度はそのままに
    肘だけ伸ばす (0 度) ことで自然に下ろした姿勢にする。
    """
    robot.reset_pose()
    for side in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(side)).joint_angle(0.0)
    return np.array([
        float(np.clip(joint.joint_angle(), joint.min_angle, joint.max_angle))
        for joint in joint_list])


def approach_base_start(base_goal, human_xy, distance):
    """接近開始位置の台車姿勢 ``[x, y, yaw]`` を返す。

    最終台車位置から見て人間の立ち位置の反対方向 (人間を中心とした半径
    方向の外向き) へ ``distance`` [m] 下がった位置。向きは最終姿勢と同じ
    -- ここまで来る移動と向き合わせはナビゲーションの担当という前提
    (モジュール docstring 参照)。
    """
    direction = np.array([base_goal[0] - human_xy[0],
                          base_goal[1] - human_xy[1]], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    # 台車の目標が人間の立ち位置と重なることは実際には無いが、0 除算だけ
    # は避けて後方 (-x) へ下がる向きにしておく。
    direction = np.array([-1.0, 0.0]) if norm < 1e-6 else direction / norm
    return np.array([base_goal[0] + direction[0] * distance,
                     base_goal[1] + direction[1] * distance,
                     base_goal[2]])


def build_start_and_goal(robot, robot_arm, handshake, human_xy,
                         approach_distance):
    """始点 (腕を下ろした姿勢 + 接近開始位置) と終点 (``handshake`` =
    ``solve_palm_ik.py`` の出力 JSON) の関節角ベクトル・台車位置姿勢を、
    ``{robot_arm}arm_whole_body`` の関節順序 (``joint_list``) で求める。

    Returns
    -------
    (link_list, joint_list, q_start, base_start, q_goal, base_goal)
        ``base_*`` は ワールド座標の ``[x, y, yaw]`` (台車は常に水平なので
        z=0 は陽に持たない)。呼び出し後、``robot`` は台車をワールド原点・
        単位姿勢に、腕を始点の姿勢にした状態になっている --
        ``TrajectoryProblem`` は構築時点の台車姿勢を基準にして軌道の台車
        3 自由度をそこからの差分として扱うので、基準を原点にしておけば
        軌道の台車列をそのままワールド座標として読み書きできる
        (``trajectory_waypoints``/``verify_waypoints`` がそう解釈する)。
        最適化対象でない関節 (反対側の腕・腰・首) もこの姿勢で固定される。
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
    base_goal = np.array([handshake['base_position'][0],
                          handshake['base_position'][1],
                          handshake['base_yaw']])
    base_start = approach_base_start(base_goal, human_xy, approach_distance)

    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    for joint, angle in zip(joint_list, q_start):
        joint.joint_angle(float(angle))
    # 対象アーム以外も腕を下ろした姿勢に
    for side in ('r', 'l'):
        if side != robot_arm:
            elbow_joint = getattr(robot, '{}_elbow_joint'.format(side))
            elbow_joint.joint_angle(elbow_joint.max_angle)
    return link_list, joint_list, q_start, base_start, q_goal, base_goal


def build_problem(robot, robot_arm, link_list, n_waypoints, dt,
                  world_obstacles, collision_link_list,
                  collision_activation_distance,
                  self_collision_activation_distance,
                  smoothness_weight, acceleration_weight,
                  collision_weight=100.0, self_collision_weight=100.0):
    """``TrajectoryProblem`` を組み立てる (始点・終点はまだ固定するだけで
    値は入れない -- 呼び出し側が初期軌道・境界値を渡す)。

    ``collision_weight``/``self_collision_weight`` は ``as_constraint=True``
    (既定, Augmented Lagrangian の等式/不等式制約として扱う) のときは
    ペナルティ項の初期重みにしかならず、収束を保証しない -- 経路上の
    干渉は必ず ``verify_waypoints`` の厳密形状による事後検証で確認する
    こと (``solve_palm_ik.py`` の ``pick_verified_candidate`` と同じ理由:
    最適化中の box/cylinder 同士の距離は交互射影による近似なので、これが
    0 に収束していても実メッシュでは接触が残ることがある)。"""
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    problem = TrajectoryProblem(
        robot_model=robot, link_list=link_list, n_waypoints=n_waypoints,
        dt=dt, move_target=whole_body.end_coords, n_base_dof=3)
    problem.add_smoothness_cost(weight=smoothness_weight)
    problem.add_acceleration_cost(weight=acceleration_weight)
    problem.add_joint_limit_constraint()
    problem.add_collision_cost(
        collision_link_list, world_obstacles,
        weight=collision_weight,
        activation_distance=collision_activation_distance,
        as_constraint=True)
    # add_collision_cost が内部で collision_link_list の各リンクの
    # link.collision_primitive (apply_collision_model が差し替えた
    # box/cylinder/sphere, solve_palm_ik.py と全く同じ形状) から
    # problem.collision_primitives を組み立てる (モジュール docstring
    # 参照)。add_self_collision_cost はその結果を直後に参照する。
    problem.add_self_collision_cost(
        weight=self_collision_weight,
        activation_distance=self_collision_activation_distance,
        as_constraint=True)
    problem.set_fixed_endpoints(start=True, end=True)
    return problem


def build_initial_trajectory(q_start, base_start, q_goal, base_goal,
                             n_waypoints):
    """始点・終点を線形補間した初期軌道 (``(n_waypoints, n_joints + 3)``)
    を作る (jaxls ソルバーへの warm start)。"""
    q_interp = interpolate_trajectory(q_start, q_goal, n_waypoints)
    base_interp = np.stack(
        [np.linspace(base_start[i], base_goal[i], n_waypoints)
         for i in range(3)], axis=1)
    traj = np.hstack([q_interp, base_interp])
    n_joints = len(q_start)
    traj[0, :n_joints] = q_start
    traj[0, n_joints:] = base_start
    traj[-1, :n_joints] = q_goal
    traj[-1, n_joints:] = base_goal
    return traj


# 採用した軌道の作り方 (結果 JSON の ``kind``) の表示名。
KIND_LABELS = {
    'pretouch': 'pre-touch 経由 (最適化なし)',
    'linear': '線形補間のみ (最適化なし)',
    'optimized': '軌道最適化',
}


def palm_normal_direction(handshake, joint_positions):
    """人間の掌の法線方向 (掌から外向き = ロボットが寄ってくる向き)。

    ``solve_palm_ik.palm_target_position`` は「目標位置 = 掌の位置 +
    法線 * ``TARGET_HOVER_OFFSET``」としているので、掌の中心 (干渉回避の
    掌カプセルと同じ ``solve_palm_ik.HAND_PALM_LANDMARKS`` の平均) から
    IK の目標位置へ向かう向きがそのまま法線になる。手のランドマークが
    欠けていて掌の中心が求まらない場合は ``None``。
    """
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
    """pre-touch 姿勢 (終点の手先姿勢を掌の法線方向へ ``standoff`` [m]
    引き戻した位置・向きは終点と同じ) の関節角を、台車を最終位置に置いた
    まま通常のヤコビアン法 IK (干渉回避なし) で解く。

    Returns
    -------
    numpy.ndarray or None
        ``joint_list`` の順の関節角。IK が収束しなければ ``None``。
    """
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
                              n_waypoints, split_ratio):
    """始点 → pre-touch 姿勢 → 終点 の 2 区間からなる軌道を作る。

    前半 (``split_ratio`` まで) で台車を最終位置まで動かしながら腕を
    pre-touch 姿勢へ持ち上げ、後半で掌の法線方向に沿ってまっすぐ終点へ
    寄る (台車は動かさない)。モジュール docstring 参照。
    """
    n_joints = len(q_start)
    split = max(2, min(n_waypoints - 1, int(n_waypoints * split_ratio)))
    traj = np.zeros((n_waypoints, n_joints + 3))
    traj[:split, :n_joints] = interpolate_trajectory(q_start, q_pre, split)
    traj[split - 1:, :n_joints] = interpolate_trajectory(
        q_pre, q_goal, n_waypoints - split + 1)
    traj[:split, n_joints:] = np.stack(
        [np.linspace(base_start[i], base_goal[i], split) for i in range(3)],
        axis=1)
    traj[split:, n_joints:] = base_goal
    return traj


def trajectory_waypoints(robot, joint_list, trajectory):
    """最適化結果 ``trajectory`` (``(n_waypoints, n_joints + 3)``) の各行
    を ``robot`` に反映し、JSON に保存する waypoint (台車位置姿勢・全身の
    関節角) のリストにする。

    ``joint_list`` (計画対象の腕の関節) 以外の関節は ``robot`` の現在の値
    (``build_start_and_goal`` が入れた腕を下ろした姿勢) のまま変えない --
    軌道はこれらの関節を動かさないので一定のはず。
    """
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
    """``human_body_obstacles``/``cylinder_surface_samples`` を 1 回だけ
    計算し、``verify_waypoints`` に ``obstacle_cache`` として渡せる形
    (``(obstacle_links, obstacle_samples)``) にまとめる。

    どちらも ``joint_positions`` だけで決まり人物の姿勢が変わらない限り
    不変なので、1 人分の ``plan_person_motion`` 呼び出し全体
    (pre-touch/線形補間の初期候補 + ``--motion-attempts`` 回の最適化
    リトライ、それぞれが ``verify_waypoints`` を 1 回呼ぶ) を通して
    ここで 1 回だけ構築したものを使い回す (``verify_waypoints`` 単体では
    従来通り waypoint ごとの使い回ししかできず、候補・リトライをまたいだ
    再構築の無駄までは防げなかったため)。
    """
    obstacle_links = spik.human_body_obstacles(joint_positions) \
        if joint_positions else None
    obstacle_samples = (
        [spik.cylinder_surface_samples(o) for o in obstacle_links]
        if obstacle_links else None)
    return obstacle_links, obstacle_samples


def verify_waypoints(robot, joint_names, waypoints, verification_pairs,
                     joint_positions, obstacle_cache=None):
    """各 waypoint を ``robot`` に反映し、厳密な形状による事後検証
    (``solve_palm_ik.collision_pairs_min_distance``。IK 側の事後検証と
    同じ関数) で最小距離を計測する。

    Returns
    -------
    list of float
        waypoint ごとの最小距離 [m] (負なら貫通)。
    """
    # human_body_obstacles は joint_positions だけで決まり、軌道上の全
    # waypoint を通して人体の姿勢は変わらないため、waypoint ごとに作り
    # 直さずここで 1 回だけ構築して使い回す。同じ理由で、各障害物の表面
    # サンプル (cylinder_surface_samples、collision_pairs_min_distance が
    # 円柱の逆向き貫通判定に使う) も obstacle_links と同様に姿勢に依存
    # しないため、ここで 1 回だけ計算して使い回す。呼び出し側 (``plan_
    # person_motion``) が候補・リトライをまたいで使い回せるよう事前計算
    # 済みの ``obstacle_cache`` (``build_obstacle_cache`` の戻り値) を
    # 渡してきた場合はそれを使い、未指定ならここで 1 回だけ構築する。
    if obstacle_cache is not None:
        obstacle_links, obstacle_samples = obstacle_cache
    else:
        obstacle_links, obstacle_samples = build_obstacle_cache(
            joint_positions)
    distances = []
    for wp in waypoints:
        name_to_angle = dict(zip(joint_names, wp['joint_angle_vector']))
        for joint in robot.joint_list:
            if joint.name in name_to_angle:
                joint.joint_angle(name_to_angle[joint.name])
        robot.newcoords(Coordinates(
            pos=wp['base_position'],
            rot=rpy_matrix(wp['base_yaw'], 0.0, 0.0)))
        distances.append(spik.collision_pairs_min_distance(
            robot, verification_pairs, joint_positions,
            obstacle_links=obstacle_links, obstacle_samples=obstacle_samples))
    return distances


def not_planned_result(reason):
    """経路計画の対象外だった人物 (IK 未対象/IK 失敗) のための結果 dict."""
    return dict(planned=False, not_planned_reason=reason)


def perturb_initial_trajectory(initial_traj, n_joints, rng, scale):
    """warm start をランダムに揺らした軌道を作る (始点・終点は変えない)。

    最適化中に使う干渉コストは box/cylinder 同士の交互射影による近似の
    ため局所解に落ちて経路上の厳密な干渉を見逃すことがある (モジュール
    docstring / ``build_problem``
    参照)。``solve_palm_ik.py`` が 1 目標につき多数の初期値から並列に IK
    を解いて最初に厳密検証を通った解を採用するのと同じ発想で、warm start
    を変えた ``--motion-attempts`` 回のリトライのうち最初に厳密検証を
    通った軌道を採用する (``plan_person_motion`` 参照)。中間 waypoint の
    腕関節角だけにガウスノイズを足す (台車の経路は概ね直進で問題ないため
    揺らさない)。
    """
    traj = initial_traj.copy()
    noise = rng.normal(scale=scale, size=(traj.shape[0] - 2, n_joints))
    traj[1:-1, :n_joints] += noise
    return traj


def plan_person_motion(robot, robot_arm, handshake, joint_positions, human_xy,
                       args, verification_pairs, solver):
    """1 人分の握手動作の軌道を計画し、結果 dict を返す。

    まず最適化を掛けずに、幾何的に構成した軌道を厳密検証に通す:
    pre-touch 姿勢を経由するもの (``build_pretouch_trajectory``、掌を
    通り抜けない最終接近になる) を優先し、次に始点と終点を単純に線形補間
    しただけのもの (``build_initial_trajectory``、pre-touch の IK が
    解けなかった場合の保険) を試す。どちらかが干渉なしならそのまま採用
    する -- 最適化中の干渉コストは box/cylinder 同士の交互射影による
    近似なので、無理に最適化すると厳密検証上の余裕を削ってしまうことが
    ある (実測で確認)。

    どちらも干渉が残った場合だけ ``jaxls`` で最適化し、``--motion-
    attempts`` 回まで warm start を揺らして
    (``perturb_initial_trajectory``) 解き直す。最初に厳密検証を通った
    ものを採用し、全て失敗したら経路上の最小距離
    (``min(waypoint_min_distances)``) が最も良かった (最も貫通が浅い)
    候補を ``verified: false`` のまま採用する
    (``solve_palm_ik.pick_verified_candidate`` のフォールバックと同じ
    考え方)。

    ``args.force_optimize`` (既定 False) が真の場合は、pre-touch/線形
    補間の候補が厳密検証に通っていても早期 return せず、必ず ``jaxls``
    の最適化ループまで進む (warm start にはそれらの候補のうち最も貫通が
    浅いものを使う)。軌道最適化そのものの計算時間を単独で計測したい
    とき用 (通常の運用では最適化を経ずに済むケースがほとんどのため、
    ``--force-optimize`` を付けないと ``optimized`` な候補が得られない)。

    ``solver`` は ``verification_pairs`` と同様、呼び出し側 (``main``)
    が人物ループの外で1回だけ作って使い回す ``JaxlsSolver`` インスタンス
    (人物・試行ごとに作り直すと jaxls の JIT キャッシュ
    (``JaxlsSolver._cached_problem``) が効かず、初回コンパイルが人物ごとに
    走ってしまうため)。
    """
    start_time = time.time()
    link_list, joint_list, q_start, base_start, q_goal, base_goal = \
        build_start_and_goal(robot, robot_arm, handshake, human_xy,
                             DEFAULT_APPROACH_DISTANCE)
    n_joints = len(q_start)
    # pre-touch/線形補間の初期候補 + 最適化リトライのすべてが同じ人物
    # (joint_positions 不変) を検証するので、build_obstacle_cache を
    # ここで 1 回だけ呼んで使い回す (verify_waypoints のモジュール docstring
    # 参照)。
    obstacle_cache = build_obstacle_cache(joint_positions)

    def make_candidate(trajectory, kind, attempt=None, cost=None,
                       solve_time=0.0):
        waypoints = trajectory_waypoints(robot, joint_list, trajectory)
        joint_names = [j.name for j in robot.joint_list]
        distances = verify_waypoints(
            robot, joint_names, waypoints, verification_pairs,
            joint_positions, obstacle_cache=obstacle_cache)
        return dict(
            planned=True,
            kind=kind,
            optimized=kind == 'optimized',
            verified=min(distances) >= -DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE,
            attempt=attempt,
            cost=cost,
            n_waypoints=args.n_waypoints,
            dt=DEFAULT_DT,
            robot_arm=robot_arm,
            approach_distance=DEFAULT_APPROACH_DISTANCE,
            joint_names=joint_names,
            waypoints=waypoints,
            waypoint_min_distances=[float(d) for d in distances],
            solve_time=solve_time,
        )

    initial_traj = build_initial_trajectory(
        q_start, base_start, q_goal, base_goal, args.n_waypoints)
    candidates = []
    normal = palm_normal_direction(handshake, joint_positions)
    if normal is not None:
        q_pre = solve_pretouch_pose(
            robot, robot_arm, link_list, joint_list, handshake, q_goal,
            base_goal, normal, DEFAULT_PRETOUCH_STANDOFF)
        if q_pre is not None:
            candidates.append(('pretouch', build_pretouch_trajectory(
                q_start, base_start, q_pre, q_goal, base_goal,
                args.n_waypoints, DEFAULT_PRETOUCH_SPLIT)))
    candidates.append(('linear', initial_traj))

    force_optimize = getattr(args, 'force_optimize', False)
    best = None
    best_trajectory = None
    for kind, trajectory in candidates:
        candidate = make_candidate(trajectory, kind)
        if candidate['verified'] and not force_optimize:
            candidate['compute_time'] = time.time() - start_time
            return candidate
        if best is None or (min(candidate['waypoint_min_distances'])
                            > min(best['waypoint_min_distances'])):
            best = candidate
            best_trajectory = trajectory
        if candidate['verified']:
            # force_optimize でもここで打ち切ってよい: candidates は
            # pre-touch -> 線形補間の優先順で並んでおり (モジュール
            # docstring 参照)、既に厳密検証を通った候補が見つかった
            # 時点でそれ以上に「良い」warm start は原理上ありえない
            # (verify_waypoints は 1 候補あたり n_waypoints 回の厳密形状
            # 干渉チェックを伴う重い処理なので、後続候補の検証は無駄)。
            break

    # solve_pretouch_pose が台車を最終位置へ動かしているので、最適化問題を
    # 組む前に基準の姿勢 (台車=ワールド原点, 腕=始点) に戻す
    # (build_start_and_goal の docstring 参照)。
    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    for joint, angle in zip(joint_list, q_start):
        joint.joint_angle(float(angle))

    world_obstacles = human_body_cylinder_obstacles(joint_positions)
    collision_link_list = spik.collision_link_list_for_arm(robot, robot_arm)
    problem = build_problem(
        robot, robot_arm, link_list, args.n_waypoints, DEFAULT_DT,
        world_obstacles, collision_link_list,
        DEFAULT_COLLISION_ACTIVATION_DISTANCE,
        DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE,
        DEFAULT_SMOOTHNESS_WEIGHT, DEFAULT_ACCELERATION_WEIGHT,
        collision_weight=100.0,
        self_collision_weight=100.0)
    rng = np.random.RandomState(0)

    # warm start を揺らした ``--motion-attempts`` 回のリトライ
    # (perturb_initial_trajectory) は、``best`` (pre-touch/線形補間の候補)
    # がまだ厳密検証を通っていない (verified: False -- 干渉が残っている)
    # ときに局所解を抜け出すためのもの。``best`` が既に verified: True
    # なら、そもそも改善する必要が無い候補を warm start にして何度も
    # 揺らして解き直す意味が無く (``force_optimize`` で早期 return せず
    # ここまで来た場合のみ起こる)、1 回だけ最適化を試す
    # (``force_optimize`` は軌道最適化の計算時間そのものを計測したい
    # ときに使うので、最適化を全く試さずに終わらせるわけにはいかない --
    # モジュール docstring 参照)。
    max_attempts = args.motion_attempts if not best['verified'] else 1

    for attempt in range(max_attempts):
        warm_start = best_trajectory if attempt == 0 \
            else perturb_initial_trajectory(
                best_trajectory, n_joints, rng,
                0.3)
        solve_start = time.time()
        result = solver.solve(problem, warm_start)
        candidate = make_candidate(
            result.trajectory, 'optimized', attempt, float(result.cost),
            time.time() - solve_start)
        if candidate['verified']:
            best = candidate
            break
        if (min(candidate['waypoint_min_distances'])
                > min(best['waypoint_min_distances'])):
            best = candidate

    best['compute_time'] = time.time() - start_time
    return best


def _warmup_solver(robot, solver, args):
    """人物ループに入る前に、左右両腕分の軌道最適化問題を一度ずつ
    ``solver`` に解かせ、jaxls の JIT トレース/コンパイルを済ませておく
    (``solve_palm_ik.py`` の ``_warmup_batch_ik`` と同じ考え方)。

    ``JaxlsSolver`` はコンパイル済み問題を腕ごと (``joint_limits_lower``/
    ``upper`` が cache key に含まれるため 'l'/'r' で自動的に別エントリに
    なる) にキャッシュして使い回す (``_cache`` 参照) が、人物ループの中で
    初めてその腕を最適化する人物がこのトレース/コンパイルを肩代わりして
    しまうと、その人物だけ計算時間が数秒〜数十秒突出してしまう。事前に
    ここで両腕分済ませておけば、人物ループ中の ``solver.solve()`` は
    どちらの腕でも最初からキャッシュヒットする。

    障害物の中身 (人体の姿勢) は cache key に含まれない
    (``JaxlsSolver._make_cache_key`` の ``obstacle_key`` はシリンダー
    個数だけを見る) ので、実際の人物の骨格ではなく空の ``joint_positions``
    (``human_body_obstacles`` がダミーで埋めた、個数だけ実物と同じ
    シリンダー列) で十分。始点・終点の関節角/台車位置も適当な値でよく、
    ここでの計算結果 (収束したかどうか) 自体は使わずに捨てる。
    """
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
        collision_link_list = spik.collision_link_list_for_arm(
            robot, robot_arm)
        problem = build_problem(
            robot, robot_arm, link_list, args.n_waypoints, DEFAULT_DT,
            world_obstacles, collision_link_list,
            DEFAULT_COLLISION_ACTIVATION_DISTANCE,
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
        description='solve_palm_ik.py が出力した握手姿勢 (最終姿勢 1 点) '
                    'を目標に、ロボットの初期姿勢からそこへ至る干渉回避 '
                    '付きの軌道 (台車移動+腕の動き) を計画し、waypoint 列 '
                    'を JSON として保存する。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='solve_palm_ik.py の出力 JSON のディレクトリ (既定は '
            'solve_palm_ik.py の既定の出力先と同じ random_handshake_'
            'poses/)。')
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='人体の全身関節位置を持つ骨格 JSON (generate_random_human_'
            'poses.py の出力, --input-dir と同じファイル名で対応させる) '
            'のディレクトリ (既定 random_human_poses/)。干渉回避の障害物 '
            '(この人物の身体) を作るのに使う。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_motion_poses'),
        help='軌道 JSON の保存先ディレクトリ (既定 random_motion_poses/。'
            '入力と同じファイル名で保存する)。')
    parser.add_argument(
        '--n-waypoints', type=int, default=DEFAULT_N_WAYPOINTS,
        help='軌道の waypoint 数 (始点・終点を含む。既定 {})。'.format(
            DEFAULT_N_WAYPOINTS))
    parser.add_argument(
        '--max-iterations', type=int, default=DEFAULT_MAX_ITERATIONS,
        help='軌道最適化 (jaxls) の最大反復回数 (既定 {})。'.format(
            DEFAULT_MAX_ITERATIONS))
    parser.add_argument(
        '--motion-attempts', type=int, default=3,
        help='線形補間だけでは干渉が残った場合に、warm start を変えて '
            '厳密検証に通るまで最適化を解き直す最大回数 (既定 3)。'
            'box/cylinder 同士の交互射影による干渉コストは局所解に落ちて '
            '経路上の干渉を見逃すことがある (perturb_initial_trajectory '
            '参照)。全て失敗したら最も貫通が浅い軌道を verified: false の '
            'まま採用する。')
    parser.add_argument(
        '--force-optimize', action='store_true',
        help='pre-touch/線形補間の候補が事後検証に通っていても早期 '
            'return せず、必ず jaxls の軌道最適化まで実行する (既定は '
            'オフ -- 通常運用では最適化を経ずに済むケースがほとんどで、 '
            '付けないと optimized な候補を得られない場合が多い)。'
            '軌道最適化そのものの計算時間を単独で計測したいときに使う '
            '(plan_person_motion のモジュール docstring 参照)。')
    parser.add_argument(
        '--no-warmup', action='store_true',
        help='人物ループに入る前の左右両腕分の軌道最適化ウォームアップ '
            '(``_warmup_solver``) をスキップする (既定はウォームアップ '
            'する)。スキップすると、人物ループ中に初めてその腕を最適化 '
            'する人物がトレース/コンパイルを肩代わりして突出して遅く '
            'なる。')
    args = parser.parse_args()

    files = json_io.iter_json_files(args.input_dir)
    if not files:
        print('{} に握手姿勢 JSON が見つかりません。先に solve_palm_ik.py '
              'を実行してください。'.format(args.input_dir))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    robot = Aero(use_hand=False)
    spik.restrict_elbow_range(robot)
    spik.apply_collision_model(robot)
    # 事後検証 (verify_waypoints) の総当たりペアは、ロボットの構造だけで
    # 決まり人物ごとの姿勢に依存しないので人物ループの外で 1 回だけ作る
    # (solve_palm_ik.py の main と同じ理由。robot_arm 引数は結果に
    # 影響しないプレースホルダ)。
    verification_pairs = spik.build_collision_verification_pairs(robot, 'r')
    # jaxls ソルバーも人物ループの外で 1 個だけ作って使い回す。人物・試行
    # ごとに作り直すと JaxlsSolver の JIT キャッシュ (_cached_problem) が
    # 3 回の motion-attempts リトライの間しか効かず、次の人物では毎回
    # 初回コンパイルが走ってしまう (実測で人物あたり最大 23 秒)。
    solver = create_solver('jaxls', max_iterations=DEFAULT_MAX_ITERATIONS,
                           verbose=False)
    if not args.no_warmup:
        _warmup_solver(robot, solver, args)

    n_optimized = n_verified = n_total = n_not_planned = 0
    for i, path in enumerate(files):
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        handshake = json.load(open(path))
        if not handshake.get('target') or not handshake.get('solved'):
            reason = ('not_target' if not handshake.get('target')
                      else 'ik_not_solved')
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
        # 接近開始位置は人間の立ち位置を基準に決める (approach_base_start)。
        # 骨格から立ち位置が求まらない場合は、平行移動が行われていないので
        # solve_palm_ik.py と同じ公称位置 (Aero の前方) を使う。
        human_xy = spik.human_standing_xy(joint_positions)
        if human_xy is None:
            human_xy = np.array([spik.HUMAN_FRONT_DISTANCE, 0.0])

        result = plan_person_motion(
            robot, handshake['robot_arm'], handshake, joint_positions,
            human_xy, args, verification_pairs, solver)
        n_total += 1
        n_optimized += int(result['optimized'])
        n_verified += int(result['verified'])
        json_io.save_json(out_path, result)
        print('[{}/{}] {} -> {} (verified={}, min_dist={:.4f} m, '
              '{}, {:.1f} 秒)'.format(
                  i + 1, len(files), os.path.basename(path), out_path,
                  result['verified'],
                  min(result['waypoint_min_distances']),
                  KIND_LABELS.get(result['kind'], result['kind']),
                  result['compute_time']))

    print('{}/{} verified (うち最適化まで要した人数 {} / '
          '対象外・IK失敗 {} 人)。'.format(
              n_verified, n_total, n_optimized, n_not_planned))


if __name__ == '__main__':
    main()
