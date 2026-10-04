#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""掌の位置姿勢 JSON (``estimate_palm_poses.py`` の出力) から、Aero が手を
繋ぐ全身 IK (台車移動を含む) を人物ごとにバッチで解き、結果を JSON に保存する
オフライン処理。``offered_hand`` が null の人物は ``target: false`` の JSON
だけを書く。

人物は立ち位置が Aero の前方 ``HUMAN_FRONT_DISTANCE`` に来るよう平行移動
してから解く。人体 (骨格を円柱で近似) と自己干渉を回避し、収束した候補は
厳密形状での事後検証と後処理判定 (``solve_post_process``) を通ったものだけ
を採用する (``pick_verified_candidate``)。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo estimate_palm_poses.py
    rosrun aero_demo solve_palm_ik.py

(既定の入出力先は scripts/ 直下の random_human_poses/ ->
random_palm_poses/ -> random_handshake_poses/)
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
import zlib

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

from skrobot.coordinates import CascadedCoords  # noqa: E402
from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import matrix2ypr, normalize_mask  # noqa: E402
from skrobot.coordinates.math import quaternion2matrix  # noqa: E402
from skrobot.model import Link  # noqa: E402
from skrobot.model import RobotModel  # noqa: E402
from skrobot.model.primitives import Cylinder  # noqa: E402
from skrobot.models import Aero  # noqa: E402

from aero_demo.collision_model import build_collision_model_urdf  # noqa: E402

# 掌の +Y (甲->掌) まわりの回転角 [deg] で、ロボットの手首側 (grasp_link の
# -X) が人間の親指側/小指側に来るもの。差し出す手の左右で符号が逆。
THUMB_SIDE_DEG_BY_HAND = {'R': 90.0, 'L': -90.0}
PINKY_SIDE_DEG_BY_HAND = {'R': -90.0, 'L': 90.0}

# 掌の法線の z 成分がこれを超えれば上向き、-これ未満なら下向き、間は横向き。
PALM_VERTICAL_THRESHOLD = math.sin(math.radians(30.0))


def turn_candidates_deg(hand, palm):
    """差し出す手 ``hand`` ('R'/'L') と掌 ``palm`` から、試す向き [deg] を
    優先順に返す。掌が上向きなら親指側、下向き (甲が上) なら小指側、横向き
    なら 0 度を最優先にする。"""
    thumb_deg = THUMB_SIDE_DEG_BY_HAND[hand]
    pinky_deg = PINKY_SIDE_DEG_BY_HAND[hand]
    normal_z = float(np.asarray(palm['y_axis'], dtype=np.float64)[2])
    if normal_z > PALM_VERTICAL_THRESHOLD:
        return (thumb_deg, 0.0, pinky_deg)
    if normal_z < -PALM_VERTICAL_THRESHOLD:
        return (pinky_deg, 0.0, thumb_deg)
    return (0.0, thumb_deg, pinky_deg)


# ``turn_candidates_deg`` が返す候補数。
NUM_TURN_CANDIDATES = 3

# 人の手に対して使うロボットの腕 (同じ方向を向いて反対側の手で繋ぐ)。
DEFAULT_ROBOT_ARM = {'L': 'r', 'R': 'l'}

# 人物をこの距離だけ Aero (原点) の前方 (+x) に平行移動してから解く。
HUMAN_FRONT_DISTANCE = 3.0  # [m]

# 台車の可動域の半幅 (人物の立ち位置中心)。
BASE_X_MOVABLE_HALF_RANGE = 3.0  # [m]
BASE_Y_MOVABLE_HALF_RANGE = 3.0  # [m]

# 台車の [x, y, yaw] の既定の (下限, 上限) (IK 開始位置 = 原点基準)。
DEFAULT_BASE_X_RANGE = (HUMAN_FRONT_DISTANCE - BASE_X_MOVABLE_HALF_RANGE,
                        HUMAN_FRONT_DISTANCE + BASE_X_MOVABLE_HALF_RANGE)
DEFAULT_BASE_Y_RANGE = (-BASE_Y_MOVABLE_HALF_RANGE, BASE_Y_MOVABLE_HALF_RANGE)
DEFAULT_BASE_YAW_RANGE = (-math.pi / 2.0, math.pi / 2.0)

# 1 目標姿勢あたりの初期値の数 (0 番目は seed_arm_pose、残りは一様乱数)。
DEFAULT_ATTEMPTS_PER_POSE = 512

# 干渉回避付きバッチ IK の反復回数・収束閾値。
DEFAULT_COLLISION_IK_STOP = 80
DEFAULT_COLLISION_IK_THRE = 0.03  # [m]
DEFAULT_COLLISION_IK_RTHRE = math.radians(8.0)  # [rad]

# 人体との干渉ペナルティの重みとマージン [m] (マージンは事後検証・軌道計画と同じ)。
DEFAULT_COLLISION_WEIGHT = 10.0
DEFAULT_COLLISION_MARGIN = 0.06

# 自己干渉ペナルティのマージン [m] (重みの既定は collision_weight と同じ)。
DEFAULT_SELF_COLLISION_MARGIN = 0.02

# バッチ IK の干渉ペナルティでのロボット側の形状: 'primitive' (事後検証と
# 同じ箱・円柱・球) または 'spheres' (skrobot 既定の球近似)。
DEFAULT_IK_COLLISION_GEOMETRY = 'primitive'

# IK の目標を掌から法線方向に浮かせる距離 [m] (目標は手の内部の点なので、
# 表面間距離はこれより 2〜3 cm 短い)。
TARGET_HOVER_OFFSET = 0.10  # [m]

# hover 姿勢でロボットと人体 (全身) の間に空ける距離 [m]。
DEFAULT_HOVER_HUMAN_CLEARANCE = DEFAULT_COLLISION_MARGIN  # [m]

# 後処理判定の目標の掌からのオフセット [m] (負 = 掌にわずかにめり込む)。
POST_PROCESS_TARGET_HOVER_OFFSET = -0.01  # [m]

# 後処理判定の腕 IK (ヤコビアン法、干渉回避・台車なし) の反復回数・収束閾値。
DEFAULT_POST_PROCESS_IK_STOP = 40
DEFAULT_POST_PROCESS_IK_THRE = 0.01  # [m]
DEFAULT_POST_PROCESS_IK_RTHRE = math.radians(5.0)  # [rad]

# 押し込み直前の最終補正で、目標の変化がこれを超えたら補正しない (誤検出対策)。
REFINE_MAX_POSITION_CHANGE = 0.10  # [m]
REFINE_MAX_ROTATION_CHANGE = math.radians(30.0)  # [rad]

# pick_verified_candidate が事後検証・後処理を試す候補数の上限 (None = 無制限)。
DEFAULT_POST_PROCESS_MAX_CANDIDATES = None

# 後処理判定の視線 IK (首 3 関節、rotation_mask='xy') の反復回数・収束閾値。
DEFAULT_POST_PROCESS_GAZE_IK_STOP = 40
DEFAULT_POST_PROCESS_GAZE_IK_RTHRE = math.radians(5.0)  # [rad]

# head_link -> camera_link (launch/decompress.launch の head_to_camera_link と
# 同じ値、四元数は (x, y, z, w))。head_end_coords の +Z は実機カメラの光軸と
# 11.3 度ずれているため、視線 IK はこちらの光軸を使う。
HEAD_TO_CAMERA_LINK_POS = (0.06700, 0.01038, 0.18521)  # [m]
HEAD_TO_CAMERA_LINK_QUAT_XYZW = (0.032565, -0.0105949, -0.012757, 0.999332)
# camera_link -> optical frame の回転 (REP 103、列が optical の各軸)。
CAMERA_LINK_TO_OPTICAL_ROT = np.array([[0.0, 0.0, 1.0],
                                       [-1.0, 0.0, 0.0],
                                       [0.0, -1.0, 0.0]])

# 肘の下限 [deg] (0 度 = 伸ばした状態)。曲げきった不自然な解を避ける。
ELBOW_MIN_ANGLE_DEG = -120.0

# 脚 (ankle: 0〜90 度、knee: -90〜0 度) の腰が低くなる側を削る比率。
# 実機の押し込みで足首が指令に追従できないことがあるため。
LEG_LOW_SIDE_MARGIN_RATIO = 0.1

# 腰ピッチの上限 [deg]。URDF は 34.95 度だが実機は 32.96 度で止まる (ハード上限)。
WAIST_P_MAX_ANGLE_DEG = 32.5

# 首ピッチの上限 [deg]。URDF は 55 度だが実機は 45.44 度で止まる。
NECK_P_MAX_ANGLE_DEG = 44.5

# 干渉回避付きバッチ IK だけに適用する関節可動域の上下マージン比率。
DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO = 0.1

# 候補選択の「関節の曲げ量コスト」(関節角 [rad] の重み付き二乗和) の対象
# 関節 ({arm}_{接尾辞}_joint) と重み。ヨー軸・首・台車は含めない。
JOINT_BEND_COST_JOINTS = {
    'shoulder_p': 0.3,
    'shoulder_r': 1.5,
    'elbow': 1.0,
    'wrist_p': 1.0,
    'wrist_r': 3.0,
}


def _joint_bend_cost_indices(robot, robot_arm):
    """曲げ量コストの各関節の ``robot.angle_vector()`` 中のインデックスと
    重みのリストを返す。"""
    index_by_name = {joint.name: i for i, joint in enumerate(robot.joint_list)}
    return [(index_by_name['{}_{}_joint'.format(robot_arm, suffix)], weight)
           for suffix, weight in JOINT_BEND_COST_JOINTS.items()]


def _joint_bend_cost_from_vector(angle_vector, bend_cost_indices):
    """``angle_vector`` の曲げ量コスト (小さいほど自然な姿勢) を返す。"""
    cost = 0.0
    for index, weight in bend_cost_indices:
        cost += weight * float(angle_vector[index]) ** 2
    return cost

# 人体の干渉回避用の円柱: (関節名 A, 関節名 B, 半径 [m])。
HUMAN_COLLISION_SEGMENTS = (
    ('Nose', 'Neck', 0.10),
    ('Neck', 'RShoulder', 0.09),
    ('Neck', 'LShoulder', 0.09),
    ('RShoulder', 'RElbow', 0.06),
    ('LShoulder', 'LElbow', 0.06),
    ('RElbow', 'RWrist', 0.04),
    ('LElbow', 'LWrist', 0.04),
    ('RShoulder', 'RHip', 0.13),
    ('LShoulder', 'LHip', 0.13),
    ('RHip', 'LHip', 0.13),
    ('RHip', 'RKnee', 0.09),
    ('RKnee', 'RAnkle', 0.06),
    ('LHip', 'LKnee', 0.09),
    ('LKnee', 'LAnkle', 0.06),
)
# 胴体の 3 本の円柱の半径を決める幅の関節 (事後検証が円柱前提なので Box は
# 使わない)。半径は前後の厚みで、実測の肩幅・腰幅から概算する。
_TORSO_SEGMENT_WIDTH_JOINTS = {
    ('RShoulder', 'RHip'): ('RShoulder', 'LShoulder'),
    ('LShoulder', 'LHip'): ('RShoulder', 'LShoulder'),
    ('RHip', 'LHip'): ('RHip', 'LHip'),
}
# 体幹の前後の厚み / 左右の幅 (経験則)。
TORSO_DEPTH_TO_WIDTH_RATIO = 0.55
# 衣服等の安全マージン [m]。
TORSO_RADIUS_MARGIN = 0.03
# 誤検出対策の半径のクリップ範囲 [m]。
MIN_TORSO_RADIUS = 0.08
MAX_TORSO_RADIUS = 0.20


def _torso_segment_radius(name_a, name_b, default_radius, joint_positions):
    """胴体の辺なら実測の肩幅/腰幅から概算した半径を、それ以外や幅を
    測れないときは ``default_radius`` を返す。"""
    width_joints = _TORSO_SEGMENT_WIDTH_JOINTS.get((name_a, name_b))
    if width_joints is None:
        return default_radius
    joint_a, joint_b = width_joints
    if joint_a not in joint_positions or joint_b not in joint_positions:
        return default_radius
    width = float(np.linalg.norm(
        np.asarray(joint_positions[joint_a], dtype=np.float64)
        - np.asarray(joint_positions[joint_b], dtype=np.float64)))
    radius = 0.5 * width * TORSO_DEPTH_TO_WIDTH_RATIO + TORSO_RADIUS_MARGIN
    return float(np.clip(radius, MIN_TORSO_RADIUS, MAX_TORSO_RADIUS))


# 実カメラの骨格の関節は体の表面の点なので、干渉判定では体幹の関節を
# カメラから離れる水平方向へずらす (合成データの関節は体内にあるので不要)。
TORSO_SURFACE_JOINTS = ('Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip')
DEFAULT_TORSO_SURFACE_OFFSET = 0.07  # [m]


def shift_torso_joints_from_surface(joint_positions, viewpoint_xy,
                                    offset=DEFAULT_TORSO_SURFACE_OFFSET):
    """体幹の関節を ``viewpoint_xy`` (カメラの x/y、骨格と同じ座標系) から
    離れる水平方向へ ``offset`` [m] ずらしたコピーを返す (干渉判定用)。"""
    if offset <= 0.0 or not joint_positions:
        return joint_positions
    viewpoint_xy = np.asarray(viewpoint_xy, dtype=np.float64)[:2]
    shifted = dict(joint_positions)
    for name in TORSO_SURFACE_JOINTS:
        if name not in shifted:
            continue
        pos = np.array(shifted[name], dtype=np.float64)
        direction = pos[:2] - viewpoint_xy
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        pos[:2] += direction / norm * offset
        shifted[name] = pos.tolist()
    return shifted


# 検出できなかった関節を親関節から鉛直に埋めるための
# (親関節名, 子関節名, z オフセット [m]、負が下)。親->子の順に並べる。
_CHAIN_FALLBACK_OFFSETS = (
    ('Neck', 'Nose', 0.12),         # 頭は首の上
    ('RShoulder', 'RElbow', -0.30),  # 上腕
    ('LShoulder', 'LElbow', -0.30),
    ('RElbow', 'RWrist', -0.25),     # 前腕
    ('LElbow', 'LWrist', -0.25),
    ('RShoulder', 'RHip', -0.50),   # 体幹の側面 (肩から腰)
    ('LShoulder', 'LHip', -0.50),
    ('RHip', 'RKnee', -0.40),        # 大腿
    ('LHip', 'LKnee', -0.40),
    ('RKnee', 'RAnkle', -0.40),      # 下腿
    ('LKnee', 'LAnkle', -0.40),
)


def _fill_missing_joints_straight_down(joint_positions):
    """欠けた関節を ``_CHAIN_FALLBACK_OFFSETS`` で親関節から補い、前腕の
    手首端を掌の円柱の表面まで延ばしたコピーを返す。"""
    filled = dict(joint_positions)
    for parent_name, child_name, z_offset in _CHAIN_FALLBACK_OFFSETS:
        if parent_name in filled and child_name not in filled:
            parent_pos = np.asarray(filled[parent_name], dtype=np.float64)
            filled[child_name] = parent_pos + np.array([0.0, 0.0, z_offset])
    # 骨格の手首と掌の landmark は別の推定器由来なので、前腕の端を掌の円柱に
    # 接する位置へ置き直す (点が足りなければ Hand0 で代用)。
    for side in ('R', 'L'):
        wrist_name = '{}Wrist'.format(side)
        elbow_name = '{}Elbow'.format(side)
        palm_names = ['{}Hand{}'.format(side, idx)
                     for idx in HAND_PALM_LANDMARKS]
        available = [np.asarray(filled[name], dtype=np.float64)
                    for name in palm_names if name in filled]
        if len(available) >= PALM_OBSTACLE_MIN_POINTS:
            palm_center = np.mean(available, axis=0)
            if elbow_name in filled:
                elbow_pos = np.asarray(filled[elbow_name], dtype=np.float64)
                direction = palm_center - elbow_pos
                dist = np.linalg.norm(direction)
                if dist > 1e-6:
                    filled[wrist_name] = (
                        palm_center
                        - (direction / dist) * HAND_PALM_RADIUS)
                else:
                    filled[wrist_name] = palm_center
            else:
                filled[wrist_name] = palm_center
        elif '{}Hand0'.format(side) in filled:
            filled[wrist_name] = filled['{}Hand0'.format(side)]
    return filled

# 手の干渉回避用ジオメトリ (MediaPipe 形式の {R,L}Hand0..20)。掌は平たい
# 円柱、各指は付け根から指先までの細い円柱で近似する。
HAND_PALM_LANDMARKS = (0, 5, 9, 13, 17)  # 手首 + 4 指の付け根 (MCP)
HAND_FINGER_LANDMARKS = (
    (1, 4),    # 親指: CMC -> 指先
    (5, 8),    # 人差し指: MCP -> 指先
    (9, 12),   # 中指: MCP -> 指先
    (13, 16),  # 薬指: MCP -> 指先
    (17, 20),  # 小指: MCP -> 指先
)
# 各指の名前 (--collision-pairs JSON で使う)。
HAND_FINGER_LABELS = ('thumb', 'index', 'middle', 'ring', 'pinky')
HAND_PALM_RADIUS = 0.05  # [m] 掌の円柱の半径
HAND_PALM_HEIGHT = 0.02  # [m] 掌の円柱の厚み (平たくする)
HAND_FINGER_RADIUS = 0.008  # [m] 指の円柱の半径 (細くする)

# 掌の円柱を置くのに必要な landmark 数 (palm_plane.MIN_PALM_POINTS と同じ)。
PALM_OBSTACLE_MIN_POINTS = 3

# 障害物の個数を固定にする (JAX の再コンパイル回避) ためのダミーの距離 [m]。
DUMMY_OBSTACLE_DISTANCE = 100.0  # [m]


def _turn_about_y(rot, turn_deg):
    """``rot`` の局所 +X/+Z を、+Y 軸まわりに ``turn_deg`` 度だけ回す
    (+Y はそのまま)。"""
    x_axis, y_axis, z_axis = rot[:, 0], rot[:, 1], rot[:, 2]
    phi = math.radians(turn_deg)
    turned_x = math.cos(phi) * x_axis + math.sin(phi) * z_axis
    turned_z = -math.sin(phi) * x_axis + math.cos(phi) * z_axis
    return np.column_stack([turned_x, y_axis, turned_z])


def _correct_grasp_frame(rot, arm):
    """左腕用に +Y/+Z を反転する (局所 +X まわり 180 度).

    URDF の都合で ``l_eef_grasp_link`` の +Y は右と逆に掌->甲を向くため、
    右腕用 (+Y=甲->掌) の ``rot`` を左腕で使うにはこの補正が必須。
    """
    if arm != 'l':
        return rot
    return np.column_stack([rot[:, 0], -rot[:, 1], -rot[:, 2]])


def palm_to_target_rots(palm, hand, robot_arm):
    """掌 (1 手分) から手先 (``{arm}_eef_grasp_link``, +X=指方向,
    +Y=甲->掌) の目標姿勢の候補を ``turn_candidates_deg`` の順に返す。
    指方向は鏡写しにせず、+Y は掌の法線の逆向き。"""
    return [palm_target_rot(palm, deg, robot_arm)
           for deg in turn_candidates_deg(hand, palm)]


def palm_target_rot(palm, turn_deg, robot_arm):
    """向き ``turn_deg`` [度] の目標姿勢を 1 つだけ返す。"""
    x_axis = np.asarray(palm['x_axis'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    y_axis = -normal
    z_axis = np.cross(x_axis, y_axis)
    base_rot = np.column_stack([x_axis, y_axis, z_axis])
    return _correct_grasp_frame(_turn_about_y(base_rot, turn_deg), robot_arm)


def palm_target_position(palm):
    """掌から ``TARGET_HOVER_OFFSET`` だけ浮かせた位置 (法線方向) を IK の
    目標位置として返す。"""
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    return position + normal * TARGET_HOVER_OFFSET


def human_standing_xy(joint_positions):
    """人物の立ち位置 (x, y) を返す (腰の中点、無ければ Neck、無ければ None)."""
    hips = [joint_positions[name] for name in ('RHip', 'LHip')
           if name in joint_positions]
    if hips:
        xy = np.mean(np.asarray(hips, dtype=np.float64), axis=0)[:2]
    elif 'Neck' in joint_positions:
        xy = np.asarray(joint_positions['Neck'], dtype=np.float64)[:2]
    else:
        return None
    return xy


def human_translation_offset(joint_positions, front_distance=HUMAN_FRONT_DISTANCE):
    """立ち位置を ``(front_distance, 0)`` に移す平行移動量 (dx, dy) を返す
    (求まらなければ (0, 0))."""
    person_xy = human_standing_xy(joint_positions)
    if person_xy is None:
        return (0.0, 0.0)
    target_xy = np.array([front_distance, 0.0])
    offset = target_xy - person_xy
    return (float(offset[0]), float(offset[1]))


def translate_joint_positions(joint_positions, offset):
    """全関節位置を x/y だけ ``offset`` 平行移動したコピーを返す."""
    dx, dy = offset
    if dx == 0.0 and dy == 0.0:
        return joint_positions
    translated = {}
    for name, pos in joint_positions.items():
        pos = np.array(pos, dtype=np.float64)
        pos[0] += dx
        pos[1] += dy
        translated[name] = pos.tolist()
    return translated


def translate_palm(palm, offset):
    """掌の ``position`` を x/y だけ ``offset`` 平行移動したコピーを返す."""
    dx, dy = offset
    if dx == 0.0 and dy == 0.0:
        return palm
    translated = dict(palm)
    position = list(palm['position'])
    position[0] += dx
    position[1] += dy
    translated['position'] = position
    return translated


def transform_palm(palm, rot, pos):
    """掌を剛体変換 ``x -> rot @ x + pos`` で移したコピーを返す。"""
    rot = np.asarray(rot, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    transformed = dict(palm)
    transformed['position'] = (
        rot @ np.asarray(palm['position'], dtype=np.float64) + pos).tolist()
    for key in ('x_axis', 'y_axis', 'z_axis'):
        if palm.get(key) is not None:
            transformed[key] = (
                rot @ np.asarray(palm[key], dtype=np.float64)).tolist()
    if palm.get('rot') is not None:
        transformed['rot'] = (
            rot @ np.asarray(palm['rot'], dtype=np.float64)).tolist()
    return transformed


def human_facing_direction(joint_positions):
    """人物の正面方向 (xy 単位ベクトル) を右肩->左肩 (無ければ腰) と鉛直上
    向きの外積で返す。求まらなければ None。"""
    def get(name):
        v = joint_positions.get(name)
        return None if v is None else np.asarray(v, dtype=np.float64)

    for a, b in (('RShoulder', 'LShoulder'), ('RHip', 'LHip')):
        pa, pb = get(a), get(b)
        if pa is None or pb is None:
            continue
        right_to_left = pb - pa
        right_to_left[2] = 0.0
        if np.linalg.norm(right_to_left) < 1e-6:
            continue
        forward = np.cross(right_to_left, np.array([0.0, 0.0, 1.0]))
        return forward[:2] / np.linalg.norm(forward[:2])
    return None


def human_facing_yaw(joint_positions):
    """``human_facing_direction`` を world yaw [rad] (atan2) に変換する。
    求まらなければ None。"""
    direction = human_facing_direction(joint_positions)
    if direction is None:
        return None
    return math.atan2(direction[1], direction[0])


def offered_hand_side_sign(human_hand, joint_positions, palm):
    """差し出した手 (手首、無ければ掌) が立ち位置から見て y のどちら側に
    あるかを +1.0/-1.0 で返す (求まらない・ほぼ正面なら None)。"""
    if joint_positions is None:
        return None
    center_xy = human_standing_xy(joint_positions)
    if center_xy is None:
        return None
    wrist_name = '{}Wrist'.format(human_hand)
    if wrist_name in joint_positions:
        hand_xy = np.asarray(
            joint_positions[wrist_name], dtype=np.float64)[:2]
    elif palm is not None:
        hand_xy = np.asarray(palm['position'], dtype=np.float64)[:2]
    else:
        return None
    diff_y = float(hand_xy[1] - center_xy[1])
    if abs(diff_y) < 1e-6:
        return None
    return 1.0 if diff_y > 0.0 else -1.0


def restrict_base_y_range_to_hand_side(base_y_range, side_sign):
    """台車の y 範囲を ``side_sign`` の側 (人物は y=0) に制限して返す。
    ``side_sign`` が None か、制限すると空になる場合はそのまま返す。"""
    if side_sign is None:
        return base_y_range
    lo, hi = base_y_range
    if side_sign > 0.0:
        restricted_lo = max(lo, 0.0)
        return (restricted_lo, hi) if restricted_lo <= hi else base_y_range
    restricted_hi = min(hi, 0.0)
    return (lo, restricted_hi) if lo <= restricted_hi else base_y_range


# 台車の x を人の立ち位置 ± この幅 [m] に絞る窓。狭い順に試す (負は絞らない)。
DEFAULT_BASE_X_STANDING_MARGINS = (0.15, 0.3, -1.0)

# 候補の並べ替えで曲げ量コストに足す、台車の人の正面方向へのずれ [m] の重み。
DEFAULT_FRONT_OFFSET_WEIGHT = 30.0

# 同じく、台車の向きの人の正面方向からのずれ [rad] の重み。
DEFAULT_FACING_YAW_WEIGHT = 30.0


def restrict_base_x_range_to_human_standing(base_x_range, standing_x,
                                            margin):
    """台車の x 範囲を ``standing_x ± margin`` との積集合に絞って返す (人が
    ほぼ ±x を向いている前提)。None・負の margin・空集合ならそのまま返す。"""
    if standing_x is None or margin is None or margin < 0.0:
        return base_x_range
    lo, hi = base_x_range
    restricted_lo = max(lo, standing_x - margin)
    restricted_hi = min(hi, standing_x + margin)
    if restricted_lo > restricted_hi:
        return base_x_range
    return (restricted_lo, restricted_hi)


# 台車の yaw を人の正面方向 ± この角度 [deg] に絞る。
DEFAULT_BASE_YAW_FACING_MARGIN_DEG = 30.0


def restrict_base_yaw_range_to_human_facing(base_yaw_range, human_yaw,
                                            margin=math.radians(
                                                DEFAULT_BASE_YAW_FACING_MARGIN_DEG)):
    """台車の yaw 範囲を ``human_yaw ± margin`` [rad] で置き換えて返す
    (積集合ではない。``human_yaw`` が None ならそのまま)。"""
    if human_yaw is None:
        return base_yaw_range
    return (human_yaw - margin, human_yaw + margin)


def seed_arm_pose(robot, robot_arm):
    """バッチ IK の attempt 0 に使う種の姿勢をロボットに作る (台車は原点).

    ``reset_pose()`` は台車を戻さないので ``robot``/``base_link`` の両方を
    単位姿勢に戻す。使わない腕は肘を伸ばして体の横に下ろす。
    """
    robot.reset_pose()
    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    other_arm = 'l' if robot_arm == 'r' else 'r'
    getattr(robot, '{}_elbow_joint'.format(other_arm)).joint_angle(0.0)
    mirror = 1.0 if robot_arm == 'l' else -1.0
    getattr(robot, '{}_shoulder_p_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_shoulder_r_joint'.format(robot_arm)) \
        .joint_angle(0.8 * mirror)
    getattr(robot, '{}_shoulder_y_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_elbow_joint'.format(robot_arm)).joint_angle(-1.0)
    getattr(robot, '{}_wrist_y_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_wrist_p_joint'.format(robot_arm)).joint_angle(0.087)
    getattr(robot, '{}_wrist_r_joint'.format(robot_arm)).joint_angle(0.0)


# 差し出さない腕の姿勢の候補 (shoulder_p, elbow) [deg] を曲げの小さい順に。
# 腰が低いと指先が台車前方の箱に入るため、入らない最初の候補に差し替える。
OTHER_ARM_POSTURES_DEG = (
    (-14.0, 0.0),
    (-20.0, -30.0),
    (-30.0, -45.0),
    (-40.0, -60.0),
)
# 差し出さない手 (指先を含む) と台車の箱の間に最低限空ける距離 [m]。
OTHER_ARM_BASE_CLEARANCE = 0.02

# side ('r'/'l') -> 指ありモデルの手・指の表面サンプル (hand_yaw_link 座標系)
_OTHER_HAND_POINTS_CACHE = {}


def other_hand_points(side):
    """指ありモデルの ``side`` 側の手と指の表面サンプルを
    ``{side}_hand_yaw_link`` 座標系で返す (指なしモデルに載せて指先を求める
    ため。左手の取り付け位置が URDF 間で違うので hand_link 基準にしない)。
    """
    cached = _OTHER_HAND_POINTS_CACHE.get(side)
    if cached is not None:
        return cached
    # 指ありモデルの読み込みが重いので左右まとめて作る。
    from aero_demo.aero_urdf_setup import load_aero
    model = load_aero(use_hand=True)
    apply_collision_model(model)
    model.reset_pose()
    for s in ('r', 'l'):
        hand_yaw = getattr(model, '{}_hand_yaw_link'.format(s))
        points = []
        for link in model.link_list:
            if link.collision_mesh is None or \
                    not link.name.startswith(s + '_'):
                continue
            if 'feetech' not in link.name and link.name != s + '_hand_link':
                continue
            world = (link_surface_samples(link) @ link.worldrot().T
                     + link.worldpos())
            points.append((world - hand_yaw.worldpos()) @ hand_yaw.worldrot())
        _OTHER_HAND_POINTS_CACHE[s] = np.vstack(points)
    return _OTHER_HAND_POINTS_CACHE[side]


def other_hand_base_clearance(robot, robot_arm):
    """差し出さない手 (指先を含む) と台車の箱の最短距離 [m] を返す (入り
    込んでいれば負)。"""
    side = 'l' if robot_arm == 'r' else 'r'
    hand_yaw = getattr(robot, '{}_hand_yaw_link'.format(side))
    world = (other_hand_points(side) @ hand_yaw.worldrot().T
             + hand_yaw.worldpos())
    boxes = [robot.wheel_base_link] + [
        link for link in getattr(robot, 'extra_collision_links', [])
        if link.parent_link is robot.wheel_base_link]
    clearance = float('inf')
    for box in boxes:
        prim = getattr(box, 'collision_primitive', None)
        if prim is None or prim.get('type') != 'box':
            continue
        rot = box.worldrot() @ np.asarray(prim['rotation'])
        center = box.worldpos() + box.worldrot() @ np.asarray(prim['center'])
        q = np.abs((world - center) @ rot) - np.asarray(prim['half_extents'])
        dist = (np.linalg.norm(np.maximum(q, 0.0), axis=1)
                + np.minimum(q.max(axis=1), 0.0))
        clearance = min(clearance, float(dist.min()))
    return clearance


def apply_other_arm_posture(robot, robot_arm, posture_index):
    """差し出さない腕を ``OTHER_ARM_POSTURES_DEG[posture_index]`` にする。"""
    side = 'l' if robot_arm == 'r' else 'r'
    shoulder_p, elbow = OTHER_ARM_POSTURES_DEG[posture_index]
    getattr(robot, '{}_shoulder_p_joint'.format(side)).joint_angle(
        math.radians(shoulder_p))
    getattr(robot, '{}_elbow_joint'.format(side)).joint_angle(
        math.radians(elbow))


def select_other_arm_posture(robot, robot_arm):
    """現在の姿勢で差し出さない手が台車の箱から ``OTHER_ARM_BASE_CLEARANCE``
    以上離れる最初の ``OTHER_ARM_POSTURES_DEG`` の添字を返す (無ければ
    None)。腕は最後に試した姿勢のままになる。"""
    for index in range(len(OTHER_ARM_POSTURES_DEG)):
        apply_other_arm_posture(robot, robot_arm, index)
        if other_hand_base_clearance(robot, robot_arm) \
                >= OTHER_ARM_BASE_CLEARANCE:
            return index
    return None


def with_other_arm_posture(robot, angle_vector, robot_arm, posture_index):
    """``angle_vector`` の差し出さない腕だけを ``posture_index`` の姿勢に
    差し替えた関節角ベクトルを返す (``robot`` はその姿勢になる)。"""
    robot.angle_vector(angle_vector)
    apply_other_arm_posture(robot, robot_arm, posture_index)
    return robot.angle_vector().copy()


def restrict_elbow_range(robot):
    """両肘の下限を ``ELBOW_MIN_ANGLE_DEG`` にする (``*_whole_body`` にも
    効く)。"""
    min_angle = math.radians(ELBOW_MIN_ANGLE_DEG)
    for arm in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(arm)).min_angle = min_angle


def restrict_leg_range(robot):
    """ankle の上限と knee の下限 (腰が低くなる側) を全域幅の
    ``LEG_LOW_SIDE_MARGIN_RATIO`` だけ狭める。URDF の可動域を
    ``_urdf_range`` に覚えるので 2 回呼んでも重ならない。"""
    for joint, low_side_is_max in ((robot.ankle_joint, True),
                                   (robot.knee_joint, False)):
        if not hasattr(joint, '_urdf_range'):
            joint._urdf_range = (joint.min_angle, joint.max_angle)
        lo, hi = joint._urdf_range
        margin = LEG_LOW_SIDE_MARGIN_RATIO * (hi - lo)
        if low_side_is_max:
            joint.max_angle = hi - margin
        else:
            joint.min_angle = lo + margin
        joint.joint_angle(min(max(joint.joint_angle(), joint.min_angle),
                              joint.max_angle))


def restrict_waist_range(robot):
    """``waist_p_joint`` の上限 (前傾側) を ``WAIST_P_MAX_ANGLE_DEG`` まで
    狭める。"""
    _restrict_max_angle(robot.waist_p_joint, WAIST_P_MAX_ANGLE_DEG)


def restrict_neck_range(robot):
    """``neck_p_joint`` の上限 (下向き側) を ``NECK_P_MAX_ANGLE_DEG`` まで
    狭める。"""
    _restrict_max_angle(robot.neck_p_joint, NECK_P_MAX_ANGLE_DEG)


def _restrict_max_angle(joint, max_angle_deg):
    """``joint`` の上限を ``max_angle_deg`` まで狭める (URDF より広げない)。"""
    if not hasattr(joint, '_urdf_range'):
        joint._urdf_range = (joint.min_angle, joint.max_angle)
    joint.max_angle = min(joint._urdf_range[1], math.radians(max_angle_deg))
    joint.joint_angle(min(max(joint.joint_angle(), joint.min_angle),
                          joint.max_angle))


def lock_fixed_joints(robot):
    """実機にモータが無い関節 (``{r,l}_hand_y_joint``) の可動域を 0 に潰す.

    バッチ IK・軌道計画は ``link_list`` の関節を全部動かすため必須。あわせて
    手の取り付け位置を指ありモデルに合わせ、IK の手先を指先側へずらす。
    """
    for name in robot._FIXED_JOINT_NAMES:
        joint = getattr(robot, name)
        joint.min_angle = 0.0
        joint.max_angle = 0.0
        joint.joint_angle(0.0)
    align_hand_mount_with_hand_model(robot)
    shift_grasp_point(robot)


# IK の手先を eef_grasp_link から局所 +X (指先側) にずらす量 [m]。
GRASP_POINT_OFFSET_X = 0.04


def shift_grasp_point(robot):
    """``{r,l}arm_end_coords`` を親から局所 +X に ``GRASP_POINT_OFFSET_X``
    [m] の位置に置き直す (何度呼んでも重ならない)。"""
    for side in ('r', 'l'):
        end_coords = getattr(robot, '{}arm_end_coords'.format(side), None)
        if end_coords is None:
            continue
        local = end_coords.copy_coords()
        local.translation = np.array([GRASP_POINT_OFFSET_X, 0.0, 0.0])
        end_coords.newcoords(local)


# {r,l}_hand_y_joint の origin [m] (指ありの URDF の値)。指なしの URDF は
# 左だけ z=0 で、手先が 4 cm ずれるため合わせる。
HAND_Y_JOINT_ORIGIN = (0.0, 0.0, -0.04)


def align_hand_mount_with_hand_model(robot):
    """``{r,l}_hand_y_joint`` の取り付け位置を ``HAND_Y_JOINT_ORIGIN`` に
    合わせる (既に同じなら何もしない)。"""
    for side in ('r', 'l'):
        joint = getattr(robot, '{}_hand_y_joint'.format(side), None)
        if joint is None:
            continue
        origin = np.asarray(HAND_Y_JOINT_ORIGIN, dtype=np.float64)
        if np.allclose(joint.default_coords.translation, origin):
            continue
        # default_coords を変えても子リンクは動かないので両方書き換える。
        # 子リンクは newcoords で (translation 直接代入はキャッシュが古いまま)。
        joint.default_coords.translation = origin
        local = joint.child_link.copy_coords()
        local.translation = origin.copy()
        joint.child_link.newcoords(local)


def restrict_joint_range_margin(link_list, margin_ratio):
    """``link_list`` の各関節の可動域を上下とも全域幅の ``margin_ratio``
    だけ一時的に狭め、元に戻す関数を返す (可動域が無限の関節は対象外)。"""
    joints = []
    seen_ids = set()
    for link in link_list:
        joint = link.joint
        if joint is None or id(joint) in seen_ids:
            continue
        seen_ids.add(id(joint))
        joints.append(joint)

    originals = []
    for joint in joints:
        lo, hi = float(joint.min_angle), float(joint.max_angle)
        width = hi - lo
        if not np.isfinite(lo) or not np.isfinite(hi) or width <= 0:
            continue
        originals.append((joint, lo, hi))
        margin = width * margin_ratio
        joint.min_angle = lo + margin
        joint.max_angle = hi - margin

    def restore():
        for joint, lo, hi in originals:
            joint.min_angle = lo
            joint.max_angle = hi

    return restore


def _cylinder_between(p0, p1, radius):
    """``p0``-``p1`` を軸 (ローカル +Z) とする ``Cylinder`` を作る。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    diff = p1 - p0
    height = float(np.linalg.norm(diff))
    if height < 1e-6:
        height = 1e-6
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = diff / height
    seed = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return Cylinder(radius=radius, height=height,
                    pos=((p0 + p1) / 2.0).tolist(), rot=rot)


def _palm_obstacle(points):
    """掌 (``HAND_PALM_LANDMARKS`` の 5 点) を近似する平たい ``Cylinder``
    を作る (軸は掌面の法線)。"""
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    wrist, index_mcp, pinky_mcp = points[0], points[1], points[-1]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    norm = np.linalg.norm(normal)
    z_axis = normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    seed = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return Cylinder(radius=HAND_PALM_RADIUS, height=HAND_PALM_HEIGHT,
                    pos=center.tolist(), rot=rot)


def _palm_obstacle_partial(points):
    """検出できた一部の掌 landmark から SVD 平面フィットで大まかな掌の
    ``Cylinder`` を作る。点が足りない・ほぼ一直線なら None。"""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < PALM_OBSTACLE_MIN_POINTS:
        return None
    center = points.mean(axis=0)
    centered = points - center
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    if len(s) < 3 or s[0] < 1e-9 or (s[1] / s[0]) < 0.15:
        return None
    z_axis = vt[2]
    seed = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return Cylinder(radius=HAND_PALM_RADIUS, height=HAND_PALM_HEIGHT,
                    pos=center.tolist(), rot=rot)


def _dummy_cylinder(radius):
    """欠けた部位の代わりに遠くに置くダミー ``Cylinder``。"""
    return Cylinder(radius=radius, height=1e-3,
                    pos=[DUMMY_OBSTACLE_DISTANCE] * 3)


def human_body_obstacles(joint_positions):
    """骨格の関節位置から全身 (差し出した手を含む) の障害物 ``Cylinder``
    のリストを作る。個数は常に一定 (``human_obstacle_names`` と同じ順) で、
    欠けた部位はダミーで埋める。"""
    joint_positions = _fill_missing_joints_straight_down(joint_positions)
    obstacles = []
    for name_a, name_b, default_radius in HUMAN_COLLISION_SEGMENTS:
        if name_a in joint_positions and name_b in joint_positions:
            radius = _torso_segment_radius(
                name_a, name_b, default_radius, joint_positions)
            obstacles.append(_cylinder_between(
                joint_positions[name_a], joint_positions[name_b], radius))
        else:
            obstacles.append(_dummy_cylinder(default_radius))
    for side in ('R', 'L'):
        palm_names = ['{}Hand{}'.format(side, idx)
                     for idx in HAND_PALM_LANDMARKS]
        available = [joint_positions[name] for name in palm_names
                    if name in joint_positions]
        if len(available) == len(palm_names):
            obstacles.append(_palm_obstacle(available))
        else:
            partial = _palm_obstacle_partial(available)
            obstacles.append(
                partial if partial is not None
                else _dummy_cylinder(HAND_PALM_RADIUS))
        for base_idx, tip_idx in HAND_FINGER_LANDMARKS:
            base_name = '{}Hand{}'.format(side, base_idx)
            tip_name = '{}Hand{}'.format(side, tip_idx)
            if base_name in joint_positions and tip_name in joint_positions:
                obstacles.append(_cylinder_between(
                    joint_positions[base_name], joint_positions[tip_name],
                    HAND_FINGER_RADIUS))
            else:
                obstacles.append(_dummy_cylinder(HAND_FINGER_RADIUS))
    return obstacles


def offered_hand_obstacle_indices(hand):
    """差し出された手 (掌・指) と同じ側の前腕の障害物の添字の集合。"""
    forearm = '{0}Elbow-{0}Wrist'.format(hand)
    return {i for i, name in enumerate(human_obstacle_names())
            if name.startswith('{}_'.format(hand)) or name == forearm}


# 差し出された手・前腕とのペナルティを掛けるロボット側のリンク ({} は腕)。
OFFERED_HAND_PENALTY_LINKS = ('{}_hand_link', '{}_thumb_box_link',
                              '{}_hand_yaw_link', '{}_forearm_link')


def human_obstacle_names():
    """``human_body_obstacles`` と同じ順序の名前のリスト (--collision-pairs
    の人体側の名前)。"""
    names = ['{}-{}'.format(name_a, name_b)
            for name_a, name_b, _ in HUMAN_COLLISION_SEGMENTS]
    for side in ('R', 'L'):
        names.append('{}_palm'.format(side))
        for label in HAND_FINGER_LABELS:
            names.append('{}_{}'.format(side, label))
    return names


def segment_points_distance(p0, p1, points):
    """線分 ``p0``-``p1`` と各点 ``points`` (N, 3) の最短距離 (N,)。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    d = p1 - p0
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        t = np.zeros(len(points))
    else:
        t = np.clip((points - p0) @ d / denom, 0.0, 1.0)
    closest = p0 + t[:, np.newaxis] * d
    return np.linalg.norm(points - closest, axis=1)


def human_capsules(joint_positions):
    """``human_body_obstacles`` と同じ順序のカプセル (p0, p1, 半径) の
    リストと名前のリストを返す。"""
    joint_positions = _fill_missing_joints_straight_down(joint_positions)
    caps = []
    names = []
    dummy = np.array([DUMMY_OBSTACLE_DISTANCE] * 3)
    for name_a, name_b, default_radius in HUMAN_COLLISION_SEGMENTS:
        if name_a in joint_positions and name_b in joint_positions:
            p0 = np.asarray(joint_positions[name_a], dtype=np.float64)
            p1 = np.asarray(joint_positions[name_b], dtype=np.float64)
            radius = _torso_segment_radius(
                name_a, name_b, default_radius, joint_positions)
        else:
            p0 = p1 = dummy
            radius = default_radius
        caps.append((p0, p1, radius))
        names.append('{}-{}'.format(name_a, name_b))
    for side in ('R', 'L'):
        palm_names = ['{}Hand{}'.format(side, idx)
                     for idx in HAND_PALM_LANDMARKS]
        available = [joint_positions[name] for name in palm_names
                    if name in joint_positions]
        if len(available) >= PALM_OBSTACLE_MIN_POINTS:
            center = np.mean(available, axis=0)
        else:
            center = dummy
        caps.append((center, center, HAND_PALM_RADIUS))
        names.append('{}_palm'.format(side))
        for (base_idx, tip_idx), label in zip(
                HAND_FINGER_LANDMARKS, HAND_FINGER_LABELS):
            base_name = '{}Hand{}'.format(side, base_idx)
            tip_name = '{}Hand{}'.format(side, tip_idx)
            if base_name in joint_positions and tip_name in joint_positions:
                p0 = np.asarray(joint_positions[base_name], dtype=np.float64)
                p1 = np.asarray(joint_positions[tip_name], dtype=np.float64)
            else:
                p0 = p1 = dummy
            caps.append((p0, p1, HAND_FINGER_RADIUS))
            names.append('{}_{}'.format(side, label))
    return caps, names


# 事後検証で許す自己干渉の貫通深さ [m] (人体との組は距離で見る)。
DEFAULT_COLLISION_VERIFY_TOLERANCE = 0.0  # [m]

# 円柱の表面サンプルの周方向・軸方向の点数。
CYLINDER_SAMPLES_N_THETA = 16
CYLINDER_SAMPLES_N_HEIGHT = 5


def cylinder_surface_samples(obstacle):
    """円柱の表面をワールド座標でサンプルし、``(points, center, radius)``
    (``center``/``radius`` は足切り用の包含球) を返す。"""
    n_theta = CYLINDER_SAMPLES_N_THETA
    n_height = CYLINDER_SAMPLES_N_HEIGHT
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    heights = np.linspace(-obstacle.height / 2.0, obstacle.height / 2.0,
                          n_height)
    ring = np.column_stack([obstacle.radius * np.cos(thetas),
                            obstacle.radius * np.sin(thetas),
                            np.zeros(n_theta)])
    side = (np.tile(ring, (n_height, 1))
            + np.repeat(np.column_stack(
                [np.zeros(n_height), np.zeros(n_height), heights]),
                n_theta, axis=0))
    caps = np.array([[0.0, 0.0, obstacle.height / 2.0],
                     [0.0, 0.0, -obstacle.height / 2.0]])
    local_pts = np.vstack([side, caps])
    center = obstacle.worldpos()
    radius = math.hypot(obstacle.radius, obstacle.height / 2.0)
    return local_pts @ obstacle.worldrot().T + center, center, radius


# id(link) -> link_collision_shape の結果。
_LINK_COLLISION_SHAPE_CACHE = {}


def link_collision_shape(link):
    """``link.collision_mesh`` (凸プリミティブ) をローカル座標の
    ``(bounds, normals, offsets, radius)`` で返す (キャッシュする)。

    ``normals @ p - offsets`` が全て 0 以下なら ``p`` は内部。``bounds``
    (AABB) と ``radius`` (原点中心の包含球) は足切り用。
    """
    cached = _LINK_COLLISION_SHAPE_CACHE.get(id(link))
    if cached is not None:
        return cached
    mesh = link.collision_mesh
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_points = verts[faces[:, 0]]
    # 面の向きが保証されないので、法線を重心から外向きに揃える。
    outward = np.einsum(
        'ij,ij->i', normals, verts[faces].mean(axis=1) - verts.mean(axis=0))
    normals = normals * np.where(outward < 0.0, -1.0, 1.0)[:, np.newaxis]
    offsets = np.einsum('ij,ij->i', normals, face_points)
    radius = float(np.linalg.norm(verts, axis=1).max())
    shape = ((verts.min(axis=0), verts.max(axis=0)), normals, offsets, radius)
    _LINK_COLLISION_SHAPE_CACHE[id(link)] = shape
    return shape


def obstacle_into_link_depth(samples, link, shape):
    """円柱の表面サンプルがリンクの形状に入り込んだ深さ [m] (正 = 貫通) を
    返す。リンクの頂点だけの判定では面の途中の貫通を見逃すため、円柱側から
    も見る。"""
    points, obstacle_center, obstacle_radius = samples
    (lo, hi), normals, offsets, link_radius = shape
    # ホットパスなので np.linalg.norm ではなく math.sqrt で足切りする。
    center_diff = obstacle_center - link.worldpos()
    center_dist = math.sqrt(
        center_diff[0] * center_diff[0] + center_diff[1] * center_diff[1]
        + center_diff[2] * center_diff[2])
    if center_dist > link_radius + obstacle_radius:
        return -float('inf')
    local_pts = (points - link.worldpos()) @ link.worldrot()
    bbox_depth = np.minimum(local_pts - lo, hi - local_pts).min(axis=1)
    inside_bbox = bbox_depth > 0.0
    if not inside_bbox.any():
        return float(bbox_depth.max())
    plane_dist = local_pts[inside_bbox] @ normals.T - offsets
    return float(-plane_dist.max(axis=1).min())


# id(link) -> link_surface_samples の結果。
_LINK_SURFACE_SAMPLES_CACHE = {}

# 自己干渉判定の表面サンプルの間隔 [m] と 1 リンクあたりの点数の上限。
SELF_COLLISION_SAMPLE_SPACING = 0.01
SELF_COLLISION_MAX_SAMPLES = 2000


def link_surface_samples(link):
    """``link.collision_mesh`` の表面をほぼ等間隔にサンプルした点 (ローカル
    座標、決定的) を返す。"""
    cached = _LINK_SURFACE_SAMPLES_CACHE.get(id(link))
    if cached is not None:
        return cached
    import trimesh
    mesh = link.collision_mesh
    spacing = max(SELF_COLLISION_SAMPLE_SPACING,
                  math.sqrt(2.0 * float(mesh.area)
                            / SELF_COLLISION_MAX_SAMPLES))
    verts, _ = trimesh.remesh.subdivide_to_size(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces), max_edge=spacing)
    points = np.asarray(verts, dtype=np.float64)
    _LINK_SURFACE_SAMPLES_CACHE[id(link)] = points
    return points


def _obb_separated(link_a, link_b):
    """2 リンクの包む直方体 (OBB) 同士が分離軸判定で離れているか。"""
    (lo_a, hi_a) = link_collision_shape(link_a)[0]
    (lo_b, hi_b) = link_collision_shape(link_b)[0]
    rot_a, rot_b = link_a.worldrot(), link_b.worldrot()
    half_a, half_b = (hi_a - lo_a) / 2.0, (hi_b - lo_b) / 2.0
    center_a = link_a.worldpos() + rot_a @ ((hi_a + lo_a) / 2.0)
    center_b = link_b.worldpos() + rot_b @ ((hi_b + lo_b) / 2.0)
    # 分離軸の候補: 面の法線 6 本と辺同士の外積 9 本。
    cross = np.cross(rot_a.T[:, np.newaxis, :],
                     rot_b.T[np.newaxis, :, :]).reshape(9, 3)
    axes = np.vstack([rot_a.T, rot_b.T, cross])
    extent_a = np.abs(axes @ rot_a) @ half_a
    extent_b = np.abs(axes @ rot_b) @ half_b
    return bool(np.any(np.abs(axes @ (center_b - center_a))
                       > extent_a + extent_b + 1e-9))


def _obb_separated_batch(lo, hi, rots, positions, index_a, index_b):
    """``_obb_separated`` のバッチ版。``lo``/``hi``: (L, 3)、``rots``:
    (L, 3, 3)、``positions``: (L, 3)、``index_a``/``index_b``: (J,)。
    (J,) の bool を返す。"""
    rot_a, rot_b = rots[index_a], rots[index_b]
    half_a = (hi[index_a] - lo[index_a]) / 2.0
    half_b = (hi[index_b] - lo[index_b]) / 2.0
    center_a = positions[index_a] + np.einsum(
        'jmn,jn->jm', rot_a, (hi[index_a] + lo[index_a]) / 2.0)
    center_b = positions[index_b] + np.einsum(
        'jmn,jn->jm', rot_b, (hi[index_b] + lo[index_b]) / 2.0)
    axes_a = rot_a.transpose(0, 2, 1)
    axes_b = rot_b.transpose(0, 2, 1)
    cross = np.cross(axes_a[:, :, np.newaxis, :],
                     axes_b[:, np.newaxis, :, :]).reshape(-1, 9, 3)
    axes = np.concatenate([axes_a, axes_b, cross], axis=1)
    extent_a = np.einsum('jkm,jm->jk', np.abs(axes @ rot_a), half_a)
    extent_b = np.einsum('jkm,jm->jk', np.abs(axes @ rot_b), half_b)
    gap = np.abs(np.einsum('jkm,jm->jk', axes, center_b - center_a))
    return np.any(gap > extent_a + extent_b + 1e-9, axis=1)


# id(link) -> _link_sample_chunks の結果。
_LINK_SAMPLE_CHUNKS_CACHE = {}
SELF_COLLISION_CHUNK_SIZE = 128


def _link_sample_chunks(link):
    """表面サンプルを ``SELF_COLLISION_CHUNK_SIZE`` 点以下の塊に分け、
    ``(塊の点のリスト, 中心 (C, 3), 包む球の半径 (C,))`` を返す。"""
    cached = _LINK_SAMPLE_CHUNKS_CACHE.get(id(link))
    if cached is not None:
        return cached
    chunks = []
    stack = [link_surface_samples(link)]
    while stack:
        pts = stack.pop()
        if len(pts) <= SELF_COLLISION_CHUNK_SIZE:
            chunks.append(pts)
            continue
        axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        order = np.argsort(pts[:, axis], kind='stable')
        half = len(pts) // 2
        stack += [pts[order[:half]], pts[order[half:]]]
    centers = np.array([(c.max(axis=0) + c.min(axis=0)) / 2.0
                        for c in chunks])
    radii = np.array([np.linalg.norm(c - center, axis=1).max()
                      for c, center in zip(chunks, centers)])
    cached = (chunks, centers, radii)
    _LINK_SAMPLE_CHUNKS_CACHE[id(link)] = cached
    return cached


def _link_samples_into_link_depth(src, dst):
    """``src`` の表面サンプルが ``dst`` の凸形状に入り込んだ最大の深さ [m]
    (無ければ 0)。"""
    (lo, hi), normals, offsets, _ = link_collision_shape(dst)
    chunks, centers, radii = _link_sample_chunks(src)
    # src のローカル座標 -> dst のローカル座標
    rot = dst.worldrot().T @ src.worldrot()
    trans = dst.worldrot().T @ (src.worldpos() - dst.worldpos())
    local_centers = centers @ rot.T + trans
    reach = np.all((local_centers + radii[:, np.newaxis] > lo)
                   & (local_centers - radii[:, np.newaxis] < hi), axis=1)
    depth = 0.0
    for i in np.nonzero(reach)[0]:
        local_pts = chunks[i] @ rot.T + trans
        inside_bbox = np.all((local_pts > lo) & (local_pts < hi), axis=1)
        if not inside_bbox.any():
            continue
        plane_dist = local_pts[inside_bbox] @ normals.T - offsets
        inside = plane_dist.max(axis=1) < 0.0
        if inside.any():
            depth = max(depth, float(-plane_dist[inside].max(axis=1).min()))
    return depth


def _penetration_depth(link_a, link_b):
    """両方向の表面サンプルによる貫通深さ (OBB 判定の後に使う)。"""
    return max(_link_samples_into_link_depth(link_a, link_b),
               _link_samples_into_link_depth(link_b, link_a))


def self_collision_depth(link_a, link_b):
    """ロボットの 2 リンクの貫通深さ [m] (無ければ 0) を、表面サンプルが
    相手の内部に入り込んだ深さの両方向の大きい方で返す (サンプル間隔の
    半分程度まで過小評価しうる)。"""
    if _obb_separated(link_a, link_b):
        return 0.0
    return _penetration_depth(link_a, link_b)


def collision_pair_name(pair):
    """組 ``(Link, Link)``/``(Link, int)`` を名前の組にする。"""
    link_a, other = pair
    if isinstance(other, (int, np.integer)):
        return (link_a.name, human_obstacle_names()[other])
    return (link_a.name, other.name)


def collision_pairs_min_distance(robot, collision_pairs, joint_positions,
                                 obstacle_links=None, obstacle_samples=None,
                                 return_pair=False):
    """``collision_pair_distances`` の最小値 [m] (負なら貫通、組が無ければ
    inf) を返す。``return_pair`` なら ``(距離, 組)`` を返す。"""
    if not collision_pairs:
        return (float('inf'), None) if return_pair else float('inf')
    dists = collision_pair_distances(
        robot, collision_pairs, joint_positions,
        obstacle_links=obstacle_links, obstacle_samples=obstacle_samples)
    index = int(np.argmin(dists))
    if not return_pair:
        return dists[index]
    return dists[index], collision_pairs[index]


def collision_pair_distances(robot, collision_pairs, joint_positions,
                             obstacle_links=None, obstacle_samples=None):
    """現在の姿勢での ``collision_pairs`` (``(Link, Link)``/``(Link, int)``)
    の組ごとの距離 [m] のリストを返す (負は貫通、人体が無ければ人体との組
    は inf)。正の値は貫通判定用の粗い近似。

    人体側は表示と同じ ``human_body_obstacles`` の円柱を使い、リンクの頂点
    が円柱に入る深さと円柱の表面がリンクに入る深さ (頂点だけでは面の途中
    の貫通を見逃す) の大きい方を採る。``obstacle_links``/
    ``obstacle_samples`` は人体が同じままループで呼ぶときの事前計算。
    """
    if not collision_pairs:
        return []
    # 別モデルで判定するペア (VerificationPairs) なら姿勢を robot に合わせる。
    sync_from = getattr(collision_pairs, 'sync_from', None)
    if sync_from is not None:
        sync_from(robot)
    if obstacle_links is None:
        obstacle_links = human_body_obstacles(joint_positions) \
            if joint_positions else None
    plan = _pair_plan(collision_pairs)
    dists = np.empty(len(collision_pairs))
    world_vertices_by_link = {}
    samples_by_obstacle = {}

    def _world_vertices(link):
        if link not in world_vertices_by_link:
            local = np.asarray(link.collision_mesh.vertices, dtype=np.float64)
            world_vertices_by_link[link] = (
                local @ link.worldrot().T + link.worldpos())
        return world_vertices_by_link[link]

    def _samples(index):
        if obstacle_samples is not None:
            return obstacle_samples[index]
        if index not in samples_by_obstacle:
            samples_by_obstacle[index] = cylinder_surface_samples(
                obstacle_links[index])
        return samples_by_obstacle[index]

    # 包含球同士の隙間 (真の距離以下) を全組まとめて求め、正の組は厳密判定を
    # 省いてその値を使う。
    links = plan['links']
    positions = np.array([link.worldpos() for link in links])
    radii = plan['radii']

    self_k, self_a, self_b = plan['self_k'], plan['self_a'], plan['self_b']
    if len(self_k):
        diff = positions[self_b] - positions[self_a]
        margins = (np.sqrt(np.einsum('ij,ij->i', diff, diff))
                   - radii[self_a] - radii[self_b])
        dists[self_k] = margins
        rigid = _rigid_self_pairs(collision_pairs, plan)
        near = np.nonzero(~(margins > 0.0))[0]
        if len(near):
            # OBB の分離判定と、片方が箱 (頂点 8 個) の組の頂点間距離をまとめて計算。
            lo, hi, box_verts, padded_verts, n_verts = _pair_plan_boxes(plan)
            rots = np.array([link.worldrot() for link in links])
            near_a, near_b = self_a[near], self_b[near]
            separated = _obb_separated_batch(
                lo, hi, rots, positions, near_a, near_b)
            a_is_box = n_verts[near_a] == 8
            b_is_box = n_verts[near_b] == 8
            box_pair = a_is_box | b_is_box
            box_dist = np.full(len(near), np.nan)
            box_side = np.where(a_is_box, near_a, near_b)
            other_side = np.where(a_is_box, near_b, near_a)
            # 箱同士と、片方だけ箱 (相手は NaN で埋めた頂点) に分けて計算。
            for group, other_verts in ((a_is_box & b_is_box, box_verts),
                                       (a_is_box ^ b_is_box, padded_verts)):
                if not group.any():
                    continue
                box_dist[group] = _min_vertex_distances(
                    box_verts, other_verts, rots, positions,
                    box_side[group], other_side[group])
        for i, j in enumerate(near):
            link_a, link_b = links[self_a[j]], links[self_b[j]]
            key = (link_a, link_b)
            if rigid[j] and key in _RIGID_PAIR_DISTANCE_CACHE:
                dists[self_k[j]] = _RIGID_PAIR_DISTANCE_CACHE[key]
                continue
            # 貫通していれば -深さ、なければ頂点同士の最短距離 (過大な近似)。
            depth = 0.0 if separated[i] else _penetration_depth(
                link_a, link_b)
            if depth > 0.0:
                dist = -depth
            elif box_pair[i]:
                dist = float(box_dist[i])
            else:
                verts_a = _world_vertices(link_a)
                verts_b = _world_vertices(link_b)
                dist = float(np.linalg.norm(
                    verts_a[:, np.newaxis, :]
                    - verts_b[np.newaxis, :, :], axis=-1).min())
            if rigid[j]:
                _RIGID_PAIR_DISTANCE_CACHE[key] = dist
            dists[self_k[j]] = dist

    human_k, human_a, human_o = (plan['human_k'], plan['human_a'],
                                 plan['human_o'])
    if len(human_k):
        if obstacle_links is None:
            dists[human_k] = float('inf')
        else:
            if obstacle_samples is not None:
                centers = np.array([s[1] for s in obstacle_samples])
                obstacle_radii = np.array([s[2] for s in obstacle_samples])
            else:
                centers = np.array([o.worldpos() for o in obstacle_links])
                obstacle_radii = np.array([
                    math.hypot(o.radius, o.height / 2.0)
                    for o in obstacle_links])
            diff = centers[human_o] - positions[human_a]
            margins = (np.sqrt(np.einsum('ij,ij->i', diff, diff))
                       - radii[human_a] - obstacle_radii[human_o])
            dists[human_k] = margins
            for j in np.nonzero(~(margins > 0.0))[0]:
                link_a = links[human_a[j]]
                shape_a = link_collision_shape(link_a)
                obstacle = obstacle_links[human_o[j]]
                samples = _samples(human_o[j])
                verts_a = _world_vertices(link_a)
                local_pts = ((verts_a - obstacle.worldpos())
                            @ obstacle.worldrot())
                radial = np.sqrt(local_pts[:, 0] ** 2 + local_pts[:, 1] ** 2)
                axial = np.abs(local_pts[:, 2])
                depth = float(np.minimum(
                    obstacle.radius - radial,
                    obstacle.height / 2.0 - axial).max())
                depth = max(depth, obstacle_into_link_depth(
                    samples, link_a, shape_a))
                dists[human_k[j]] = -depth
    return dists.tolist()


# (Link, Link) -> 相対姿勢が変わらない自己干渉の組の距離。
_RIGID_PAIR_DISTANCE_CACHE = {}


def _pair_plan(collision_pairs):
    """距離計算をまとめて行うための添字の配列とリンクの包含球の半径の dict
    (``VerificationPairs`` ならキャッシュする)。"""
    plan = getattr(collision_pairs, '_plan', None)
    if plan is not None and plan['n'] == len(collision_pairs):
        return plan
    links = []
    index_of = {}

    def link_index(link):
        if link not in index_of:
            index_of[link] = len(links)
            links.append(link)
        return index_of[link]

    self_rows, human_rows = [], []
    for k, (link_a, other) in enumerate(collision_pairs):
        if isinstance(other, int):
            human_rows.append((k, link_index(link_a), other))
        else:
            self_rows.append((k, link_index(link_a), link_index(other)))

    def columns(rows):
        arr = np.array(rows, dtype=np.int64).reshape(-1, 3)
        return arr[:, 0], arr[:, 1], arr[:, 2]

    self_k, self_a, self_b = columns(self_rows)
    human_k, human_a, human_o = columns(human_rows)
    plan = dict(
        n=len(collision_pairs), links=links,
        radii=np.array([link_collision_shape(link)[3] for link in links]),
        self_k=self_k, self_a=self_a, self_b=self_b,
        human_k=human_k, human_a=human_a, human_o=human_o)
    if isinstance(collision_pairs, VerificationPairs):
        collision_pairs._plan = plan
    return plan


def _ragged_arange(starts, counts):
    """``np.concatenate([np.arange(s, s + c) for s, c in zip(starts,
    counts)])`` をループなしで作る。"""
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    return (np.repeat(starts - offsets, counts)
            + np.arange(int(np.sum(counts))))


def _human_pair_plan(plan):
    """``human_obstacle_clearances`` 用に ``plan`` へキャッシュする値 (人体と
    組むリンクの通し番号、詰め直した ``local_a``、表面サンプルの塊を並べた
    配列とその範囲。座標はリンクのローカル)。"""
    human_plan = plan.get('human_plan')
    if human_plan is not None:
        return human_plan
    link_ids, local_a = np.unique(plan['human_a'], return_inverse=True)
    spacing = np.zeros(len(plan['links']))
    chunk_centers, chunk_radii, chunk_points, chunk_count = [], [], [], []
    for i in link_ids:
        spacing[i] = _link_surface_sample_spacing(plan['links'][i])
        chunks, centers, radii = _link_sample_chunks(plan['links'][i])
        chunk_points += chunks
        chunk_centers.append(centers)
        chunk_radii.append(radii)
        chunk_count.append(len(chunks))
    chunk_count = np.array(chunk_count)
    chunk_point_count = np.array([len(c) for c in chunk_points])
    human_plan = dict(
        link_ids=link_ids, local_a=local_a.reshape(-1),
        obstacles=np.unique(plan['human_o']).tolist(), spacing=spacing,
        chunk_centers=np.vstack(chunk_centers),
        chunk_radii=np.concatenate(chunk_radii),
        points=np.vstack(chunk_points),
        chunk_count=chunk_count,
        chunk_start=np.r_[0, np.cumsum(chunk_count)[:-1]],
        chunk_point_count=chunk_point_count,
        chunk_point_start=np.r_[0, np.cumsum(chunk_point_count)[:-1]])
    plan['human_plan'] = human_plan
    return human_plan


def _min_vertex_distances(verts_a, verts_b, rots, positions, index_a,
                          index_b):
    """組ごとの頂点同士の最短距離 (J,) を返す。``verts_a``/``verts_b`` は
    リンクごとのローカルの頂点 (L, V, 3) で、NaN は無視する。"""
    rot_a, rot_b = rots[index_a], rots[index_b]
    world_a = (np.einsum('jvn,jmn->jvm', verts_a[index_a], rot_a)
               + positions[index_a][:, np.newaxis, :])
    local_a = np.einsum('jvm,jmn->jvn',
                        world_a - positions[index_b][:, np.newaxis, :], rot_b)
    local_b = verts_b[index_b]
    sq = (np.einsum('jan,jan->ja', local_a, local_a)[:, :, np.newaxis]
          + np.einsum('jbn,jbn->jb', local_b, local_b)[:, np.newaxis, :]
          - 2.0 * np.einsum('jan,jbn->jab', local_a, local_b))
    return np.sqrt(np.maximum(
        np.fmin.reduce(sq.reshape(len(sq), -1), axis=1), 0.0))


def _pair_plan_boxes(plan):
    """``plan`` のリンクごとの ``(lo, hi, 箱の頂点 (L, 8, 3), NaN で埋めた
    全頂点 (L, V, 3), 頂点数 (L,))`` (``plan`` にキャッシュ)。"""
    boxes = plan.get('boxes')
    if boxes is not None:
        return boxes
    links = plan['links']
    bounds = [link_collision_shape(link)[0] for link in links]
    verts = [np.asarray(link.collision_mesh.vertices, dtype=np.float64)
             for link in links]
    n_verts = np.array([len(v) for v in verts])
    padded = np.full((len(links), n_verts.max(), 3), np.nan)
    box_verts = np.full((len(links), 8, 3), np.nan)
    for i, v in enumerate(verts):
        padded[i, :len(v)] = v
        if len(v) == 8:
            box_verts[i] = v
    boxes = (np.array([b[0] for b in bounds]),
             np.array([b[1] for b in bounds]), box_verts, padded, n_verts)
    plan['boxes'] = boxes
    return boxes


def _rigid_self_pairs(collision_pairs, plan):
    """自己干渉の組ごとに、相対姿勢が常に同じか (bool の配列)。別モデルで
    ``sync_from`` されない関節 (指) だけでつながる組が True。"""
    rigid = plan.get('rigid')
    if rigid is not None:
        return rigid
    rigid = np.zeros(len(plan['self_k']), dtype=bool)
    joint_map = getattr(collision_pairs, '_joint_map', None)
    if getattr(collision_pairs, 'model', None) is not None and joint_map:
        movable = {model_joint for model_joint, _ in joint_map}

        def rigid_root(link):
            # 動かされる関節の子リンクまで親をたどる。
            while True:
                if getattr(link, 'joint', None) in movable:
                    return link
                parent = getattr(link, 'parent_link', None)
                if parent is None:
                    return link
                link = parent

        roots = [rigid_root(link) for link in plan['links']]
        rigid = np.array([roots[a] is roots[b] for a, b in
                          zip(plan['self_a'], plan['self_b'])], dtype=bool)
    plan['rigid'] = rigid
    return rigid


def _link_surface_sample_spacing(link):
    """``link_surface_samples`` のサンプル間隔 [m]。"""
    return max(SELF_COLLISION_SAMPLE_SPACING,
               math.sqrt(2.0 * float(link.collision_mesh.area)
                         / SELF_COLLISION_MAX_SAMPLES))


def points_cylinder_distance(points, obstacle):
    """点群 (N, 3) と円柱の符号付き距離 [m] (N,) (負なら内部)。"""
    local = (points - obstacle.worldpos()) @ obstacle.worldrot()
    dr = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2) - obstacle.radius
    da = np.abs(local[:, 2]) - obstacle.height / 2.0
    outside = np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(da, 0.0) ** 2)
    return np.where((dr > 0.0) | (da > 0.0), outside, np.maximum(dr, da))


def human_obstacle_clearances(robot, collision_pairs, obstacle_links,
                              cull_distance):
    """人体の障害物ごとのロボットとの最短距離 [m] を ``{添字: 距離}`` で返す.

    リンク表面のサンプルから円柱までの距離をサンプル間隔分だけ安全側に
    見積もる。``cull_distance`` 以上離れた組は下界で済ませる。深い貫通は
    正しく測れないので、貫通判定は ``collision_pair_distances`` で行うこと。
    """
    sync_from = getattr(collision_pairs, 'sync_from', None)
    if sync_from is not None:
        sync_from(robot)
    plan = _pair_plan(collision_pairs)
    human_a, human_o = plan['human_a'], plan['human_o']
    if not len(human_a):
        return {}
    human_plan = _human_pair_plan(plan)
    links = plan['links']
    positions = np.array([links[i].worldpos()
                          for i in human_plan['link_ids']])
    centers = np.array([o.worldpos() for o in obstacle_links])
    obstacle_radii = np.array([math.hypot(o.radius, o.height / 2.0)
                               for o in obstacle_links])
    diff = centers[human_o] - positions[human_plan['local_a']]
    gaps = (np.sqrt(np.einsum('ij,ij->i', diff, diff))
            - plan['radii'][human_a] - obstacle_radii[human_o])
    near = np.nonzero(gaps < cull_distance)[0]
    if len(near):
        obstacle_rots = np.array([o.worldrot() for o in obstacle_links])
        obstacle_r = np.array([o.radius for o in obstacle_links])
        obstacle_half_h = np.array([o.height / 2.0 for o in obstacle_links])
        rots = np.array([links[i].worldrot() for i in human_plan['link_ids']])

        def cylinder_distances(local_points, a, o):
            # 点ごとに別の円柱での points_cylinder_distance。
            world = (np.einsum('nij,nj->ni', rots[a], local_points)
                     + positions[a])
            local = np.einsum('ni,nij->nj', world - centers[o],
                              obstacle_rots[o])
            dr = np.sqrt(local[:, 0] ** 2 + local[:, 1] ** 2) - obstacle_r[o]
            da = np.abs(local[:, 2]) - obstacle_half_h[o]
            outside = np.sqrt(np.maximum(dr, 0.0) ** 2
                              + np.maximum(da, 0.0) ** 2)
            return np.where((dr > 0.0) | (da > 0.0), outside,
                            np.maximum(dr, da))

        # 「組 x 塊」を一度に計算する。塊の点の距離は「中心の距離 ± 包む球の
        # 半径」に収まるので、下限が組の上限の最小を超える塊は省ける。
        a = human_plan['local_a'][near]
        o = human_o[near]
        n_chunks = human_plan['chunk_count'][a]
        pair_of_chunk = np.repeat(np.arange(len(near)), n_chunks)
        chunk = _ragged_arange(human_plan['chunk_start'][a], n_chunks)
        chunk_a, chunk_o = a[pair_of_chunk], o[pair_of_chunk]
        center_dist = cylinder_distances(
            human_plan['chunk_centers'][chunk], chunk_a, chunk_o)
        radii = human_plan['chunk_radii'][chunk]
        lower = center_dist - radii
        chunk_offsets = np.r_[0, np.cumsum(n_chunks)[:-1]]
        upper = np.minimum.reduceat(center_dist + radii, chunk_offsets)
        # 表面上のどの点もサンプル点から spacing / sqrt(3) 以内にある。
        margin = human_plan['spacing'][human_a[near]] / math.sqrt(3.0)
        # cull_distance 未満になり得ない塊も省く (組の値は塊の下限の最小)。
        keep = ((lower <= upper[pair_of_chunk])
                & (lower - margin[pair_of_chunk] < cull_distance))
        gaps[near] = np.minimum.reduceat(lower, chunk_offsets) - margin
        if keep.any():
            kept, kept_pair = chunk[keep], pair_of_chunk[keep]
            n_points = human_plan['chunk_point_count'][kept]
            pair_of_point = np.repeat(kept_pair, n_points)
            point = _ragged_arange(human_plan['chunk_point_start'][kept],
                                   n_points)
            dist = cylinder_distances(human_plan['points'][point],
                                      a[pair_of_point], o[pair_of_point])
            # 点は組の順に並んでいる。残った組は下限を実際の最小値で置き換える。
            point_offsets = np.r_[
                0, np.flatnonzero(np.diff(pair_of_point)) + 1]
            pairs = pair_of_point[point_offsets]
            gaps[near[pairs]] = (np.minimum.reduceat(dist, point_offsets)
                                 - margin[pairs])
    per_obstacle = np.full(len(obstacle_links), np.inf)
    np.minimum.at(per_obstacle, human_o, gaps)
    return {int(i): float(per_obstacle[i]) for i in human_plan['obstacles']}


_HAND_BOX_CACHE = {}

# apply_hand_box が足す親指の箱のリンク名 ({} は腕)。
HAND_THUMB_BOX_LINK = '{}_thumb_box_link'

# 手のリンク自身の箱とは別の箱にする部位 (リンク名に含まれる文字列, 箱の
# リンク名)。残り (掌・4 本の指) は手のリンク自身の箱。
HAND_SPLIT_BOXES = ((('thumb',), HAND_THUMB_BOX_LINK),)


def hand_box_primitives(robot):
    """指なしの ``robot`` の手のリンクの代わりに使う、指ありモデルの手と指
    を包む箱の一覧 ``[(手のリンク, 名前, primitive dict), ...]`` を返す.

    名前が None の箱 (掌+4 本の指) は手のリンク自身の形状を置き換え、名前
    付きの箱 (親指) は干渉専用のリンクにする。``robot`` ごとにキャッシュ。
    """
    cached = _HAND_BOX_CACHE.get(id(robot))
    if cached is not None:
        return cached
    from aero_demo.aero_urdf_setup import load_aero
    model = load_aero(use_hand=True)
    apply_collision_model(model)
    movable = {joint.name for joint in robot.joint_list}

    def rigid_root(link):
        # 指なしのロボットにある関節 (= 動かす関節) の子リンクまでたどる。
        while True:
            joint = getattr(link, 'joint', None)
            if joint is not None and joint.name in movable:
                return link
            parent = getattr(link, 'parent_link', None)
            if parent is None:
                return link
            link = parent

    groups = {}
    for link in collision_link_list_for_arm(model):
        groups.setdefault(rigid_root(link), []).append(link)
    robot_links = {link.name: link
                   for link in collision_link_list_for_arm(robot)}

    def box_around(root, members):
        points = np.vstack([
            (np.asarray(m.collision_mesh.vertices) @ m.worldrot().T
             + m.worldpos() - root.worldpos()) @ root.worldrot()
            for m in members])
        lo, hi = points.min(axis=0), points.max(axis=0)
        # 手首側 (+z) は元の手の形状の端まで (手首と常に重なるのを防ぐ)。
        original = np.asarray(
            robot_links[root.name].collision_mesh.vertices)
        hi[2] = min(hi[2], original[:, 2].max())
        return dict(type='box', center=(lo + hi) / 2.0, rotation=np.eye(3),
                    half_extents=(hi - lo) / 2.0)

    boxes = []
    for root, members in groups.items():
        # 指がぶら下がっているリンクだけ。
        if root.name not in robot_links or all(
                m.name in robot_links for m in members):
            continue
        # 横に張り出す親指は別の箱にする。
        side = root.name.split('_')[0]
        parent = robot_links[root.name]
        rest = list(members)
        split = []
        for part, link_name in HAND_SPLIT_BOXES:
            group = [m for m in rest if any(k in m.name for k in part)]
            if group:
                rest = [m for m in rest if m not in group]
                split.append((parent, link_name.format(side),
                              box_around(root, group)))
        boxes.append((parent, None, box_around(root, rest)))
        boxes.extend(split)
    _HAND_BOX_CACHE[id(robot)] = boxes
    return boxes


def apply_hand_box(robot):
    """指なしの ``robot`` の手を ``hand_box_primitives`` の箱で表す (親指は
    ``extra_collision_links`` に追加)。skrobot が干渉形状をキャッシュする
    ので、IK を解く前に 1 回だけ呼ぶこと。"""
    import trimesh
    extra_links = list(getattr(robot, 'extra_collision_links', []))
    for parent, name, prim in hand_box_primitives(robot):
        transform = np.eye(4)
        transform[:3, 3] = prim['center']
        mesh = trimesh.creation.box(
            extents=2.0 * np.asarray(prim['half_extents']),
            transform=transform)
        if name is None:
            link = parent
        else:
            link = Link(name=name)
            link.newcoords(parent.copy_worldcoords())
            parent.assoc(link)
            link.add_parent_link(parent)
            extra_links.append(link)
        link.collision_mesh = mesh
        link.collision_primitive = prim
    robot.extra_collision_links = extra_links


def apply_collision_model(robot):
    """``robot`` の各リンクの ``collision_mesh`` をプリミティブ近似形状
    (``build_collision_model_urdf`` が生成・キャッシュ) に差し替え、干渉
    モデルにだけあるリンクを ``robot.extra_collision_links`` に足す."""
    collision_urdf_path = build_collision_model_urdf(robot.urdf_path)

    collision_robot = RobotModel()
    collision_robot.load_urdf_file(
        str(collision_urdf_path), include_mimic_joints=False)
    collision_links_by_name = {
        link.name: link for link in collision_robot.link_list}

    n_replaced = 0
    for link in robot.link_list:
        collision_link = collision_links_by_name.get(link.name)
        mesh = (getattr(collision_link, 'collision_mesh', None)
               if collision_link is not None else None)
        if mesh is not None:
            link.collision_mesh = mesh
            # 軌道最適化が同じ形状を使えるようプリミティブのパラメータも渡す。
            link.collision_primitive = getattr(
                collision_link, 'collision_primitive', None)
            n_replaced += 1

    # 干渉モデルにだけあるリンク (台車前方の箱など) は、link_list に入れず
    # 親リンクに assoc した干渉専用のリンクにする (parent_link も設定する)。
    robot_links_by_name = {link.name: link for link in robot.link_list}
    extra_links = [link for link in getattr(robot, 'extra_collision_links',
                                            [])]
    extra_names = {link.name for link in extra_links}
    for collision_link in collision_robot.link_list:
        mesh = getattr(collision_link, 'collision_mesh', None)
        if (collision_link.name in robot_links_by_name
                or collision_link.name in extra_names or mesh is None):
            continue
        src_parent = collision_link.parent_link
        parent = (robot_links_by_name.get(src_parent.name)
                  if src_parent is not None else None)
        if parent is None:
            raise ValueError(
                '{} のリンク {!r} の親リンクがロボットに見つかりません。'
                .format(collision_urdf_path, collision_link.name))
        relative = src_parent.copy_worldcoords().inverse_transformation() \
            .transform(collision_link.copy_worldcoords())
        link = Link(name=collision_link.name)
        link.newcoords(parent.copy_worldcoords().transform(relative))
        parent.assoc(link)
        link.add_parent_link(parent)
        link.collision_mesh = mesh
        link.collision_primitive = getattr(
            collision_link, 'collision_primitive', None)
        extra_links.append(link)
        extra_names.add(link.name)
    robot.extra_collision_links = extra_links

    print('[collision-model] {} 個のリンクの干渉ジオメトリを ({}) から '
          '差し替えました{}。'.format(
              n_replaced, collision_urdf_path,
              ' (干渉専用のリンク {} を追加)'.format(
                  ', '.join(link.name for link in extra_links))
              if extra_links else ''))


def collision_link_list_for_arm(robot):
    """干渉ジオメトリを持つ全身のリンクの一覧を返す.

    干渉専用のリンクは親リンクの直後に並べる (``ignore_adjacent`` で常に
    重なる親子の組が除かれるようにするため)。
    """
    extras_by_parent = {}
    for link in getattr(robot, 'extra_collision_links', []):
        extras_by_parent.setdefault(link.parent_link, []).append(link)
    links = []
    for link in robot.link_list:
        if getattr(link, 'collision_mesh', None) is not None:
            links.append(link)
        links.extend(extras_by_parent.get(link, []))
    return links


def load_collision_pairs(path, robot):
    """``--collision-pairs`` の JSON (``[[名前A, 名前B], ...]``) を
    ``(Link, Link)``/``(Link, 人体の障害物の添字)`` のリストにする。
    名前 A は常にロボットのリンク名。"""
    with open(path) as f:
        pair_names = json.load(f)
    links_by_name = {link.name: link for link in
                     list(robot.link_list)
                     + list(getattr(robot, 'extra_collision_links', []))}
    obstacle_index_by_name = {
        name: idx for idx, name in enumerate(human_obstacle_names())}
    pairs = []
    for name_a, name_b in pair_names:
        if name_a not in links_by_name:
            raise ValueError(
                '{} (--collision-pairs) に含まれるリンク名 {!r} が ロボット'
                'に見つかりません。'.format(path, name_a))
        link_a = links_by_name[name_a]
        if name_b in links_by_name:
            pairs.append((link_a, links_by_name[name_b]))
        elif name_b in obstacle_index_by_name:
            pairs.append((link_a, obstacle_index_by_name[name_b]))
        else:
            raise ValueError(
                '{} (--collision-pairs) に含まれる名前 {!r} が、ロボットの '
                'リンク名にも human_obstacle_names() の人体セグメント名にも '
                '一致しません。'.format(path, name_b))
    return pairs


# 自己干渉の事後検証から除く、関節のつながりが近い組の段数 (干渉ジオメトリを
# 持つリンクで数える。近似形状が関節付近で重なるだけで実機では当たらない)。
SELF_COLLISION_IGNORE_LINK_DISTANCE = 3

# 段数では除けないが、実機では当たらないので除く組 ({} は腕)。
SELF_COLLISION_IGNORE_PAIRS = (
    ('{}_forearm_link', '{}_feetech_thumb0_link'),
)


def _collision_parent(link, collision_links):
    parent = link.parent_link
    while parent is not None and parent not in collision_links:
        parent = parent.parent_link
    return parent


def self_collision_ignored(robot, collision_link_list):
    """自己干渉の事後検証から除く組 (``frozenset`` の集合) を返す: つながり
    が ``SELF_COLLISION_IGNORE_LINK_DISTANCE`` 段以内の組、既定の姿勢で既に
    貫通している組、``SELF_COLLISION_IGNORE_PAIRS``。``robot`` の姿勢は
    戻す。"""
    collision_links = set(collision_link_list)
    ignore_names = {frozenset((a.format(side), b.format(side)))
                    for a, b in SELF_COLLISION_IGNORE_PAIRS
                    for side in ('r', 'l')}
    parent_of = {link: _collision_parent(link, collision_links)
                 for link in collision_link_list}

    def ancestors(link):
        chain = [link]
        while parent_of.get(chain[-1]) is not None:
            chain.append(parent_of[chain[-1]])
        return chain

    ancestor_depth = {link: {a: i for i, a in enumerate(ancestors(link))}
                      for link in collision_link_list}
    ignored = set()
    for link_a, link_b in itertools.combinations(collision_link_list, 2):
        depth_a, depth_b = ancestor_depth[link_a], ancestor_depth[link_b]
        common = [depth_a[a] + depth_b[a] for a in depth_a if a in depth_b]
        if ((common
             and min(common) <= SELF_COLLISION_IGNORE_LINK_DISTANCE)
                or frozenset((link_a.name, link_b.name)) in ignore_names):
            ignored.add(frozenset((link_a, link_b)))

    saved_av = robot.angle_vector().copy()
    saved_coords = robot.copy_worldcoords()
    try:
        try:
            robot.reset_pose()
        except NotImplementedError:
            robot.init_pose()
        for link_a, link_b in itertools.combinations(collision_link_list, 2):
            key = frozenset((link_a, link_b))
            if key not in ignored and self_collision_depth(
                    link_a, link_b) > DEFAULT_COLLISION_VERIFY_TOLERANCE:
                ignored.add(key)
    finally:
        robot.angle_vector(saved_av)
        robot.newcoords(saved_coords)
    return ignored


def build_collision_verification_pairs(robot, robot_arm):
    """事後検証用の総当たりのペア (``load_collision_pairs`` と同じ形式) を
    作る: ``self_collision_ignored`` を除いた自己干渉の全組と、全リンク x
    全人体セグメント。ロボットの構造だけで決まる。"""
    collision_link_list = collision_link_list_for_arm(robot)
    ignored = self_collision_ignored(robot, collision_link_list)
    pairs = [(link_a, link_b) for link_a, link_b in
             itertools.combinations(collision_link_list, 2)
             if frozenset((link_a, link_b)) not in ignored]
    for link in collision_link_list:
        for obstacle_index in range(len(human_obstacle_names())):
            pairs.append((link, obstacle_index))
    return pairs


class VerificationPairs(list):
    """事後検証のペアのリストに判定用のモデルを添えたもの。``model`` が
    None でなければペアのリンクは ``model`` のもので、判定前に
    ``sync_from`` で姿勢を IK のロボットに合わせる。"""

    def __init__(self, pairs, model=None):
        super().__init__(pairs)
        self.model = model
        self._joint_map = None
        # 人体から離れている量を別のモデルで測る組 (human_clearance_pairs)。
        self.clearance_pairs = None

    def with_pairs(self, pairs):
        """同じ ``model`` で、ペアだけを差し替えたものを返す。"""
        result = VerificationPairs(pairs, self.model)
        result._joint_map = self._joint_map
        return result

    def sync_from(self, robot):
        if self.model is None or self.model is robot:
            return
        if self._joint_map is None:
            by_name = {joint.name: joint for joint in robot.joint_list}
            self._joint_map = [(joint, by_name[joint.name])
                               for joint in self.model.joint_list
                               if joint.name in by_name]
        for model_joint, robot_joint in self._joint_map:
            model_joint.joint_angle(robot_joint.joint_angle())
        # 台車は robot と base_link のどちらで与えられることもあるので、
        # base_link のワールド座標に合わせる。
        self.model.newcoords(robot.base_link.copy_worldcoords())


def self_collision_pairs(verification_pairs):
    """``verification_pairs`` のうち自己干渉の組だけ (None ならそのまま)。
    人体との貫通は ``human_obstacle_clearances`` の距離で捕まえる。"""
    if verification_pairs is None:
        return None
    pairs = [pair for pair in verification_pairs
             if not isinstance(pair[1], int)]
    if isinstance(verification_pairs, VerificationPairs):
        return verification_pairs.with_pairs(pairs)
    return pairs


def human_clearance_pairs(verification_pairs):
    """人体との距離を測る組 ('mixed' なら指ありモデルの組)。"""
    return getattr(verification_pairs, 'clearance_pairs', None) \
        or verification_pairs


COLLISION_VERIFY_MODELS = ('mixed', 'nohand', 'hand')
DEFAULT_COLLISION_VERIFY_MODEL = 'mixed'


def build_verification_pairs_for_model(
        robot, verify_model=DEFAULT_COLLISION_VERIFY_MODEL):
    """``verify_model`` で事後検証する ``VerificationPairs`` を作る:
    'nohand' は ``robot`` (指なし+手の箱)、'hand' は指ありモデル、'mixed'
    は自己干渉を ``robot`` で、人体との距離を指ありモデルで測る。"""
    if verify_model in ('nohand', 'mixed'):
        pairs = VerificationPairs(
            build_collision_verification_pairs(robot, 'r'))
        if verify_model == 'mixed':
            hand = build_verification_pairs_for_model(robot, 'hand')
            pairs.clearance_pairs = hand.with_pairs(
                [pair for pair in hand if isinstance(pair[1], int)])
        return pairs
    if verify_model != 'hand':
        raise ValueError('verify_model は {} のどれかです: {!r}'.format(
            COLLISION_VERIFY_MODELS, verify_model))
    from aero_demo.aero_urdf_setup import load_aero
    model = load_aero(use_hand=True)
    apply_collision_model(model)
    return VerificationPairs(
        build_collision_verification_pairs(model, 'r'), model)


def attach_camera_optical_coords(robot, pos=None, rot=None):
    """``robot.head_link`` に実機カメラの光軸を +Z とする座標
    ``robot.camera_optical_coords`` を取り付けて返す.

    ``pos``/``rot`` は head_link から見た位置 [m]・回転行列 (省略時は
    ``HEAD_TO_CAMERA_LINK_*`` と ``CAMERA_LINK_TO_OPTICAL_ROT`` から作る)。
    """
    if pos is None:
        pos = HEAD_TO_CAMERA_LINK_POS
    if rot is None:
        qx, qy, qz, qw = HEAD_TO_CAMERA_LINK_QUAT_XYZW
        rot = quaternion2matrix([qw, qx, qy, qz], normalize=True).dot(
            CAMERA_LINK_TO_OPTICAL_ROT)
    coords = getattr(robot, 'camera_optical_coords', None)
    if coords is None:
        coords = CascadedCoords(parent=robot.head_link,
                                name='camera_optical_coords')
        robot.camera_optical_coords = coords
    coords.newcoords(Coordinates(pos=list(pos), rot=np.asarray(rot)))
    return coords


def camera_optical_coords(robot):
    """``robot.camera_optical_coords`` を返す (無ければ既定値で取り付ける)。"""
    coords = getattr(robot, 'camera_optical_coords', None)
    if coords is None:
        coords = attach_camera_optical_coords(robot)
    return coords


def _gaze_target(gaze_coords, position):
    """``gaze_coords`` の +Z を ``position`` へ向ける視線 IK の目標座標。
    rotation_mask='xy' はひねりが大きいと収束しないので、現在の向きから
    最小回転で向けた姿勢にする。"""
    return Coordinates(
        pos=list(position), rot=gaze_coords.worldrot()).align_axis_to_direction(
            np.asarray(position) - gaze_coords.worldpos())


def _gaze_error(gaze_coords, position):
    """``gaze_coords`` の +Z と ``position`` への方向のなす角 [rad]。"""
    direction = np.asarray(position) - gaze_coords.worldpos()
    direction = direction / np.linalg.norm(direction)
    return float(math.acos(np.clip(
        np.dot(gaze_coords.worldrot()[:, 2], direction), -1.0, 1.0)))


# _reaim_gaze の最大繰り返し回数。
POST_PROCESS_GAZE_REAIM_ROUNDS = 3


def _reaim_gaze(robot, gaze_coords, position, stop, rthre,
                rounds=POST_PROCESS_GAZE_REAIM_ROUNDS):
    """腕・首の同時 IK の後、首だけで視線を ``position`` へ向け直す.

    同時 IK の視線目標は解く前の頭の位置で固定され、解く間に頭が動く分
    ずれるため。目標を作り直して最大 ``rounds`` 回繰り返し、悪化したら戻す。
    """
    head_joints = [link.joint for link in robot.head.link_list]
    for _ in range(rounds):
        error = _gaze_error(gaze_coords, position)
        if error < rthre:
            return
        before = [joint.joint_angle() for joint in head_joints]
        robot.inverse_kinematics(
            _gaze_target(gaze_coords, position), move_target=gaze_coords,
            link_list=robot.head.link_list, position_mask=False,
            rotation_mask='xy', stop=stop, rthre=rthre,
            revert_if_fail=False)
        if _gaze_error(gaze_coords, position) >= error:
            for joint, angle in zip(head_joints, before):
                joint.joint_angle(angle)
            return


def solve_post_process(robot, robot_arm, palm, target_rot,
                       stop=DEFAULT_POST_PROCESS_IK_STOP,
                       thre=DEFAULT_POST_PROCESS_IK_THRE,
                       rthre=DEFAULT_POST_PROCESS_IK_RTHRE,
                       gaze_ik_stop=DEFAULT_POST_PROCESS_GAZE_IK_STOP,
                       gaze_ik_rthre=DEFAULT_POST_PROCESS_GAZE_IK_RTHRE,
                       gaze=True):
    """後処理判定: 台車を動かさずに、腕で掌から
    ``POST_PROCESS_TARGET_HOVER_OFFSET`` の位置 (わずかにめり込む) へ
    ``target_rot`` で押し付けるのと、カメラの光軸を掌へ向ける
    (``gaze=False`` なら省く) のを同時に解く.

    ``robot`` の現在の姿勢を初期値に書き換える (失敗時は元に戻る)。両方
    収束したら結果 dict、そうでなければ None を返す。
    """
    start_time = time.time()
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    target_pos = position + normal * POST_PROCESS_TARGET_HOVER_OFFSET
    target_coords = Coordinates(pos=target_pos.tolist(), rot=target_rot)

    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    head_move_target = camera_optical_coords(robot)
    head_target = _gaze_target(head_move_target, position)

    try:
        if gaze:
            # タスクごとのマスクは normalize_mask 済みのリストで渡す (そう
            # しないと 1 つのマスク指定と誤解釈される)。
            result = robot.inverse_kinematics(
                target_coords=[target_coords, head_target],
                move_target=[move_target, head_move_target],
                link_list=[whole_body.link_list, robot.head.link_list],
                position_mask=[normalize_mask(True), normalize_mask(False)],
                rotation_mask=[normalize_mask(True), normalize_mask('xy')],
                stop=max(stop, gaze_ik_stop),
                thre=[thre, thre], rthre=[rthre, gaze_ik_rthre],
                revert_if_fail=True)
        else:
            result = robot.inverse_kinematics(
                target_coords, move_target=move_target,
                link_list=whole_body.link_list,
                stop=stop, thre=thre, rthre=rthre, revert_if_fail=True)
    except Exception as e:
        print('  [post-process] 押し付け/視線 IK で例外が発生したため '
              '棄却します: {}'.format(e))
        return None
    if result is False:
        return None
    if gaze:
        _reaim_gaze(robot, head_move_target, position,
                    gaze_ik_stop, gaze_ik_rthre)

    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    return dict(
        target_position=[float(v) for v in target_pos],
        target_rot=[[float(v) for v in row] for row in target_rot],
        hand_position=[float(v) for v in move_target.worldpos()],
        hand_rot=[[float(v) for v in row] for row in move_target.worldrot()],
        base_position=[float(v) for v in robot.base_link.worldpos()],
        base_yaw=float(yaw),
        joint_names=[j.name for j in robot.joint_list],
        joint_angle_vector=[float(v) for v in robot.angle_vector()],
        compute_time=time.time() - start_time,
    )


def refine_post_process(robot, robot_arm, palm, turn_deg, planned_post):
    """hover 到達後、検出し直した掌 ``palm`` で押し込み姿勢を解き直す最終
    補正.

    ``robot`` は hover 目標の姿勢にしておき、``palm`` はそのワールド系で
    表す。``planned_post`` は計画時の ``result['post_process']``。
    ``post_process``/``position_change`` [m]/``rotation_change`` [rad]/
    ``reason`` ('ok'/'ok_arm_only'/'too_large'/'ik_failed')/
    ``compute_time`` の dict を返す。
    """
    start_time = time.time()
    target_rot = palm_target_rot(palm, turn_deg, robot_arm)
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    target_pos = position + normal * POST_PROCESS_TARGET_HOVER_OFFSET
    position_change = float(np.linalg.norm(
        target_pos - np.asarray(planned_post['target_position'])))
    rel_rot = np.asarray(planned_post['target_rot']).T @ target_rot
    rotation_change = float(math.acos(
        np.clip((np.trace(rel_rot) - 1.0) / 2.0, -1.0, 1.0)))
    info = dict(post_process=None, position_change=position_change,
                rotation_change=rotation_change)
    if (position_change > REFINE_MAX_POSITION_CHANGE
            or rotation_change > REFINE_MAX_ROTATION_CHANGE):
        info.update(reason='too_large', compute_time=time.time() - start_time)
        return info
    post = solve_post_process(robot, robot_arm, palm, target_rot)
    reason = 'ok'
    if post is None:
        # 首が可動域の端で視線 IK が収束しないことが多いので、首は計画時の
        # 角度にして腕だけで解き直す。
        planned_angles = dict(zip(planned_post['joint_names'],
                                  planned_post['joint_angle_vector']))
        for link in robot.head.link_list:
            if link.joint.name in planned_angles:
                link.joint.joint_angle(planned_angles[link.joint.name])
        post = solve_post_process(robot, robot_arm, palm, target_rot,
                                  gaze=False)
        reason = 'ok_arm_only' if post is not None else 'ik_failed'
    info.update(post_process=post, reason=reason,
                compute_time=time.time() - start_time)
    return info


def simulate_final_correction(robot, robot_arm, result, palm, angle_vector,
                              base_pose, trials, slip_xy, slip_yaw, rng):
    """hover 目標で台車が一様乱数 (x/y ±``slip_xy`` [m]、yaw ±``slip_yaw``
    [rad]) だけスリップしたと仮定して ``refine_post_process`` を ``trials``
    回解き、各回の結果 (``post_process`` を除き ``slip`` を付けたもの) の
    リストを返す。"""
    robot.angle_vector(angle_vector)
    robot.newcoords(base_pose)
    base_rot = robot.base_link.worldrot().copy()
    base_pos = robot.base_link.worldpos().copy()
    records = []
    for _ in range(trials):
        dx, dy = rng.uniform(-slip_xy, slip_xy, size=2)
        dyaw = rng.uniform(-slip_yaw, slip_yaw)
        c, s = math.cos(dyaw), math.sin(dyaw)
        slip_rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # スリップ後の台車から見た掌を hover 目標のワールド系へ (B S^-1 B^-1)。
        rot = base_rot @ slip_rot.T @ base_rot.T
        pos = (base_pos - rot @ base_pos
               - base_rot @ slip_rot.T @ np.array([dx, dy, 0.0]))
        observed = transform_palm(palm, rot, pos)
        robot.angle_vector(angle_vector)
        robot.newcoords(base_pose)
        info = refine_post_process(
            robot, robot_arm, observed, result['turn_deg'],
            result['post_process'])
        info.pop('post_process')
        info['slip'] = [float(dx), float(dy), float(dyaw)]
        records.append(info)
    return records


def _base_front_offset(base_pose, standing_xy, facing):
    """台車が人の立ち位置から人の正面方向 ``facing`` にどれだけ前 [m] に
    あるか (後ろなら負)。"""
    base_xy = np.asarray(base_pose.worldpos(), dtype=np.float64)[:2]
    return float(np.dot(base_xy - standing_xy, facing))


def _base_yaw_offset(base_pose, facing):
    """台車の +x 軸の人の正面方向からのずれ [rad] (左回りが正)。"""
    x_axis = np.asarray(base_pose.worldrot(), dtype=np.float64)[:2, 0]
    return float(math.atan2(
        facing[0] * x_axis[1] - facing[1] * x_axis[0],
        facing[0] * x_axis[0] + facing[1] * x_axis[1]))


def base_placement_metrics(joint_positions, base_position, base_yaw):
    """最終の台車位置と人の位置関係の指標 dict を返す (求まらなければ None).

    ``bearing_deg``: 台車から見た人の方向 (正面 0、左回りが正)。
    ``front_offset``: 台車が人の正面方向にどれだけ前か [m]。
    ``yaw_offset_deg``: 台車の向きのずれ (人のいる側へ回る向きが正)。
    """
    standing_xy = human_standing_xy(joint_positions)
    facing = human_facing_direction(joint_positions)
    if standing_xy is None or facing is None:
        return None
    base_xy = np.asarray(base_position, dtype=np.float64)[:2]
    d = standing_xy - base_xy
    c, s = math.cos(-base_yaw), math.sin(-base_yaw)
    bearing = math.atan2(s * d[0] + c * d[1], c * d[0] - s * d[1])
    yaw_offset = base_yaw - math.atan2(facing[1], facing[0])
    yaw_offset = math.atan2(math.sin(yaw_offset), math.cos(yaw_offset))
    return dict(
        bearing_deg=math.degrees(bearing),
        front_offset=float(np.dot(base_xy - standing_xy, facing)),
        yaw_offset_deg=math.degrees(yaw_offset) * (
            1.0 if bearing > 0.0 else -1.0))


def pick_verified_candidate(robot, success_flags, angle_vectors, base_poses,
                            verification_pairs, joint_positions,
                            collision_verify_tolerance,
                            robot_arm, palm, hand, rots,
                            attempts_per_pose=DEFAULT_ATTEMPTS_PER_POSE,
                            post_process_ik_stop=DEFAULT_POST_PROCESS_IK_STOP,
                            post_process_thre=DEFAULT_POST_PROCESS_IK_THRE,
                            post_process_rthre=DEFAULT_POST_PROCESS_IK_RTHRE,
                            post_process_max_candidates=(
                                DEFAULT_POST_PROCESS_MAX_CANDIDATES),
                            front_offset_weight=DEFAULT_FRONT_OFFSET_WEIGHT,
                            facing_yaw_weight=DEFAULT_FACING_YAW_WEIGHT,
                            hover_human_clearance=(
                                DEFAULT_HOVER_HUMAN_CLEARANCE),
                            placement_joint_positions=None):
    """バッチ IK の候補 (添字 = 向き * ``attempts_per_pose`` + 初期値) から、
    収束・事後検証 (自己干渉の貫通なし、人体から ``hover_human_clearance``
    以上) ・後処理判定をすべて通る候補をコストの低い順に探す.

    コストは関節の曲げ量に、台車の人の正面方向へのずれ x
    ``front_offset_weight`` と向きのずれ x ``facing_yaw_weight`` を足した
    もの (立ち位置・向きは ``placement_joint_positions`` から求める)。
    後処理だけ失敗した場合は、事後検証を通った最小コストの候補を
    ``post_process_result=None`` で採用する。

    ``(turn_index, angle_vector, base_pose, post_process_result)`` または
    None を返す。``robot`` は最後に調べた候補の姿勢のままになる。
    """
    turn_degs = turn_candidates_deg(hand, palm)
    bend_cost_indices = _joint_bend_cost_indices(robot, robot_arm)

    standing_xy = facing = None
    if placement_joint_positions is None:
        placement_joint_positions = joint_positions
    if (front_offset_weight > 0.0 or facing_yaw_weight > 0.0) \
            and placement_joint_positions:
        standing_xy = human_standing_xy(placement_joint_positions)
        facing = human_facing_direction(placement_joint_positions)

    def candidate_cost(candidate_index):
        cost = _joint_bend_cost_from_vector(
            angle_vectors[candidate_index], bend_cost_indices)
        if standing_xy is not None and facing is not None:
            base_pose = base_poses[candidate_index]
            if front_offset_weight > 0.0:
                cost += front_offset_weight * abs(_base_front_offset(
                    base_pose, standing_xy, facing))
            if facing_yaw_weight > 0.0:
                cost += facing_yaw_weight * abs(_base_yaw_offset(
                    base_pose, facing))
        return cost

    # 収束した候補をコスト昇順に (同着は添字順)。
    candidates = sorted(
        ((candidate_cost(candidate_index), candidate_index)
         for candidate_index, ok in enumerate(success_flags) if ok),
        key=lambda item: (item[0], item[1]))

    # 人体の障害物と表面サンプルは候補ループの外で 1 回だけ作る。
    obstacle_links = human_body_obstacles(joint_positions) \
        if joint_positions else None
    obstacle_samples = (
        [cylinder_surface_samples(o) for o in obstacle_links]
        if obstacle_links else None)

    # 差し出さない腕 (後で姿勢を差し替える) に関わる組とそれ以外に分け、
    # それ以外は後処理の前に検証して安く落とす。
    other_prefix = ('l' if robot_arm == 'r' else 'r') + '_'

    def _subset(keep, source=verification_pairs):
        if source is None:
            return None
        pairs = [pair for pair in source if keep(pair)]
        if isinstance(source, VerificationPairs):
            return source.with_pairs(pairs)
        return pairs

    def _involves_other_arm(pair):
        link_a, other = pair
        return link_a.name.startswith(other_prefix) or (
            not isinstance(other, int) and other.name.startswith(other_prefix))

    early_verification_pairs = _subset(
        lambda pair: not _involves_other_arm(pair))
    other_arm_verification_pairs = _subset(_involves_other_arm)
    self_verification_pairs = _subset(
        lambda pair: not isinstance(pair[1], int))
    # 人体との距離を検証しない場合だけ、人体との組も貫通で見る。
    if hover_human_clearance is not None and obstacle_links:
        early_penetration_pairs = self_collision_pairs(
            early_verification_pairs)
        other_arm_penetration_pairs = self_collision_pairs(
            other_arm_verification_pairs)
    else:
        early_penetration_pairs = early_verification_pairs
        other_arm_penetration_pairs = other_arm_verification_pairs
    clearance_source = human_clearance_pairs(verification_pairs)
    early_clearance_pairs = _subset(
        lambda pair: not _involves_other_arm(pair), clearance_source)
    other_arm_clearance_pairs = _subset(_involves_other_arm, clearance_source)

    def _too_close_to_human(pairs, label):
        if (hover_human_clearance is None or not obstacle_links
                or pairs is None):
            return False
        clearances = human_obstacle_clearances(
            robot, pairs, obstacle_links,
            cull_distance=hover_human_clearance)
        if not clearances:
            return False
        index = min(clearances, key=clearances.get)
        if clearances[index] >= hover_human_clearance:
            return False
        print('  [hover-clearance] {} の候補は人体 ({}) まで {:.3f} m しか '
              '離れていないため棄却します。'.format(
                  label, human_obstacle_names()[index], clearances[index]))
        return True

    fallback = None
    examined = 0
    for cost, candidate_index in candidates:
        if (post_process_max_candidates is not None
                and examined >= post_process_max_candidates):
            break
        examined += 1
        turn_index = candidate_index // attempts_per_pose
        attempt_index = candidate_index % attempts_per_pose
        label = 'turn={:.0f}deg/初期値 {} (bend_cost={:.4f})'.format(
            turn_degs[turn_index], attempt_index, cost)
        robot.angle_vector(angle_vectors[candidate_index])
        robot.newcoords(base_poses[candidate_index])
        hover_av = robot.angle_vector().copy()
        hover_waist_z = float(robot.waist_link.worldpos()[2])
        min_dist, pair = collision_pairs_min_distance(
            robot, early_penetration_pairs, joint_positions,
            obstacle_links=obstacle_links, obstacle_samples=obstacle_samples,
            return_pair=True)
        if min_dist < -collision_verify_tolerance:
            print('  [collision-verify] {} の候補は IK は収束'
                  'したが、事後検証で {:.4f} m 貫通 ({} x {}) していたため'
                  '棄却します。'.format(label, min_dist,
                                       *collision_pair_name(pair)))
            continue
        if _too_close_to_human(early_clearance_pairs, label):
            continue
        # 押し込みを解き、腰が低い方の姿勢で差し出さない腕の姿勢を決めて
        # hover・押し込みの両方に差し替える。
        post_result = solve_post_process(
            robot, robot_arm, palm, rots[turn_index],
            stop=post_process_ik_stop, gaze_ik_stop=post_process_ik_stop,
            thre=post_process_thre, rthre=post_process_rthre)
        if post_result is None or \
                float(robot.waist_link.worldpos()[2]) >= hover_waist_z:
            robot.angle_vector(hover_av)
        posture = select_other_arm_posture(robot, robot_arm)
        if posture is None:
            print('  [other-arm] {} の候補は差し出さない手の指先が台車に'
                  'かからない腕の姿勢が無いため棄却します。'.format(label))
            continue
        hover_av = with_other_arm_posture(robot, hover_av, robot_arm, posture)
        min_dist, pair = collision_pairs_min_distance(
            robot, other_arm_penetration_pairs, joint_positions,
            obstacle_links=obstacle_links, obstacle_samples=obstacle_samples,
            return_pair=True)
        if min_dist < -collision_verify_tolerance:
            print('  [collision-verify] {} の候補は差し出さない腕を差し替えた'
                  '姿勢で {:.4f} m 貫通 ({} x {}) していたため棄却します。'
                  .format(label, min_dist, *collision_pair_name(pair)))
            continue
        if _too_close_to_human(other_arm_clearance_pairs, label):
            continue
        if fallback is None:
            fallback = (turn_index, hover_av, base_poses[candidate_index],
                        cost)
        if post_result is not None:
            # 押し込み姿勢は人に触れるので自己干渉だけを検証する。
            press_av = with_other_arm_posture(
                robot, post_result['joint_angle_vector'], robot_arm, posture)
            press_dist, pair = collision_pairs_min_distance(
                robot, self_verification_pairs, joint_positions,
                obstacle_links=obstacle_links,
                obstacle_samples=obstacle_samples, return_pair=True)
            if press_dist < -collision_verify_tolerance:
                print('  [collision-verify] {} の候補は押し込み姿勢で {:.4f} m '
                      '自己干渉 ({} x {}) していたため、後処理判定を失敗扱いに'
                      'します。'.format(label, press_dist,
                                       *collision_pair_name(pair)))
                post_result = None
            else:
                post_result = dict(
                    post_result,
                    joint_angle_vector=[float(v) for v in press_av],
                    other_arm_posture=list(OTHER_ARM_POSTURES_DEG[posture]))
        if post_result is not None:
            return (turn_index, hover_av, base_poses[candidate_index],
                    post_result)
        print('  [post-process] {} の候補は干渉検証を通過した '
              'が、後処理判定 (押し付け/視線 IK) には失敗したため、次に '
              '曲げ量コストが低い候補を試します。'.format(label))

    if fallback is None:
        return None
    turn_index, angle_vector, base_pose, fallback_cost = fallback
    print('  [post-process] 調べた範囲の候補で後処理判定に全て失敗 '
          'した (または上限に達した) ため、turn={:.0f}deg '
          '(bend_cost={:.4f}) の候補を後処理前の解として採用します。'
          .format(turn_degs[turn_index], fallback_cost))
    return (turn_index, angle_vector, base_pose, None)


def solve_person_ik(robot, palm, hand, robot_arm, collision_obstacles,
                    attempts_per_pose=DEFAULT_ATTEMPTS_PER_POSE,
                    base_limits=None,
                    collision_weight=DEFAULT_COLLISION_WEIGHT,
                    collision_margin=DEFAULT_COLLISION_MARGIN,
                    self_collision=True,
                    collision_pairs=None,
                    self_collision_weight=None,
                    self_collision_margin=DEFAULT_SELF_COLLISION_MARGIN,
                    collision_ik_stop=DEFAULT_COLLISION_IK_STOP,
                    collision_ik_thre=DEFAULT_COLLISION_IK_THRE,
                    collision_ik_rthre=DEFAULT_COLLISION_IK_RTHRE,
                    collision_joint_limit_margin_ratio=(
                        DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO),
                    joint_positions=None,
                    verification_pairs=None,
                    collision_verify_tolerance=(
                        DEFAULT_COLLISION_VERIFY_TOLERANCE),
                    post_process_ik_stop=DEFAULT_POST_PROCESS_IK_STOP,
                    post_process_thre=DEFAULT_POST_PROCESS_IK_THRE,
                    post_process_rthre=DEFAULT_POST_PROCESS_IK_RTHRE,
                    post_process_max_candidates=(
                        DEFAULT_POST_PROCESS_MAX_CANDIDATES),
                    front_offset_weight=DEFAULT_FRONT_OFFSET_WEIGHT,
                    facing_yaw_weight=DEFAULT_FACING_YAW_WEIGHT,
                    n_turn_candidates=None,
                    hover_human_clearance=DEFAULT_HOVER_HUMAN_CLEARANCE,
                    collision_geometry=DEFAULT_IK_COLLISION_GEOMETRY,
                    placement_joint_positions=None):
    """1 人分の全ての向き x 初期値を、人体と ``collision_pairs`` の自己干渉
    を回避するバッチ IK (jax) でまとめて解き、``pick_verified_candidate``
    で 1 つ選ぶ.

    ``palm``/``joint_positions`` は人物を前方に平行移動済みのもの。収束判定
    は干渉を見ないので、``verification_pairs`` (総当たり) で事後検証する。
    ``collision_pairs`` が None/空なら干渉回避なし (None は事後検証もなし)。
    ``collision_joint_limit_margin_ratio`` はバッチ IK の間だけ適用する。
    ``n_turn_candidates`` は向きの候補を先頭から何個解くか。

    ``(picked, collision_ik_time, candidate_selection_time)`` を返す (時間
    は秒)。
    """
    seed_arm_pose(robot, robot_arm)
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    target_pos = palm_target_position(palm)
    rots = palm_to_target_rots(palm, hand, robot_arm)[:n_turn_candidates]
    target_coords = [Coordinates(pos=target_pos.tolist(), rot=rot)
                     for rot in rots]
    # 人体の障害物が無いときは、人体との組 (添字が範囲外になる) を除く。
    effective_collision_pairs = collision_pairs
    effective_self_collision = self_collision
    effective_collision_obstacles = collision_obstacles
    effective_collision_link_list = None
    if not collision_pairs:
        # 干渉回避を切るとヤコビアン法になり人ごとに再コンパイルするので、
        # リンク 1 つだけの自己干渉 (コスト 0) で同じ勾配降下法にする。
        effective_collision_pairs = None
        effective_self_collision = True
        effective_collision_obstacles = []
        effective_collision_link_list = [robot.body_link]
    elif collision_pairs is not None and not collision_obstacles:
        effective_collision_pairs = [
            (link_a, other) for link_a, other in collision_pairs
            if not isinstance(other, int)]
        if not effective_collision_pairs:
            # 空リストだと ValueError になるので自己干渉を無効にする。
            effective_self_collision = False
    else:
        # 差し出された手・前腕とロボットの手先側のリンクの組を足す。
        offered = sorted(offered_hand_obstacle_indices(hand))
        links_by_name = {link.name: link for link in
                         list(robot.link_list)
                         + list(getattr(robot, 'extra_collision_links', []))}
        effective_collision_pairs = list(collision_pairs) + [
            (links_by_name[name.format(robot_arm)], i)
            for name in OFFERED_HAND_PENALTY_LINKS
            if name.format(robot_arm) in links_by_name for i in offered]
    restore_joint_range = restrict_joint_range_margin(
        whole_body.link_list, collision_joint_limit_margin_ratio)
    collision_ik_start = time.time()
    try:
        # 全初期値の解を候補にする (並びは目標優先)。
        angle_vectors, base_poses, success_flags, _ = \
            robot.batch_inverse_kinematics(
                target_coords=target_coords,
                move_target=move_target,
                link_list=whole_body.link_list,
                position_mask=True, rotation_mask=True,
                stop=collision_ik_stop,
                thre=collision_ik_thre,
                rthre=collision_ik_rthre,
                initial_angles='current',
                attempts_per_pose=attempts_per_pose,
                return_all_attempts=True,
                backend='jax',
                use_base='planar', base_limits=base_limits,
                collision_link_list=effective_collision_link_list,
                collision_obstacles=effective_collision_obstacles,
                collision_weight=collision_weight,
                collision_margin=collision_margin,
                self_collision=effective_self_collision,
                collision_pairs=effective_collision_pairs,
                self_collision_weight=self_collision_weight,
                self_collision_margin=self_collision_margin,
                collision_geometry=collision_geometry)
    finally:
        restore_joint_range()
    collision_ik_time = time.time() - collision_ik_start
    effective_verification_pairs = verification_pairs
    if verification_pairs is not None and not collision_obstacles:
        self_pairs = [(link_a, other) for link_a, other in verification_pairs
                      if not isinstance(other, int)]
        effective_verification_pairs = (
            verification_pairs.with_pairs(self_pairs)
            if isinstance(verification_pairs, VerificationPairs)
            else self_pairs)
    candidate_selection_start = time.time()
    picked = pick_verified_candidate(
        robot, success_flags, angle_vectors, base_poses,
        effective_verification_pairs, joint_positions,
        collision_verify_tolerance, robot_arm, palm, hand, rots,
        attempts_per_pose=attempts_per_pose,
        post_process_ik_stop=post_process_ik_stop,
        post_process_thre=post_process_thre,
        post_process_rthre=post_process_rthre,
        post_process_max_candidates=post_process_max_candidates,
        front_offset_weight=front_offset_weight,
        facing_yaw_weight=facing_yaw_weight,
        hover_human_clearance=hover_human_clearance,
        placement_joint_positions=placement_joint_positions)
    candidate_selection_time = time.time() - candidate_selection_start
    return picked, collision_ik_time, candidate_selection_time


def solve_person_ik_side_by_side(robot, palm, hand, robot_arm,
                                 collision_obstacles, base_limits,
                                 standing_x,
                                 x_margins=DEFAULT_BASE_X_STANDING_MARGINS,
                                 **kwargs):
    """台車の x を ``standing_x ± x_margins`` の窓 (負は絞らない) に絞って
    狭い順に ``solve_person_ik`` を解き、後処理まで解けた最初の結果を返す.

    ``(picked, collision_ik_time, candidate_selection_time, base_limits,
    x_margin)`` を返す (時間は合計、窓は最後に試したもの)。
    """
    if standing_x is None or not x_margins:
        x_margins = (-1.0,)
    collision_ik_time = candidate_selection_time = 0.0
    # どの窓でも後処理まで通らなければ、最初に見つかった後処理前の解を返す。
    fallback = None
    for x_margin in x_margins:
        person_base_limits = [
            restrict_base_x_range_to_human_standing(
                base_limits[0], standing_x, margin=x_margin),
            base_limits[1], base_limits[2]]
        picked, ik_time, selection_time = solve_person_ik(
            robot, palm, hand, robot_arm, collision_obstacles,
            base_limits=person_base_limits, **kwargs)
        collision_ik_time += ik_time
        candidate_selection_time += selection_time
        if picked is not None and picked[3] is not None:
            break
        if picked is not None and fallback is None:
            fallback = (picked, person_base_limits, x_margin)
        print('  [base-x] 立ち位置 ±{} m では後処理まで解けませんでした。'
              .format(x_margin if x_margin >= 0.0 else 'inf'))
    else:
        if fallback is not None:
            picked, person_base_limits, x_margin = fallback
    return (picked, collision_ik_time, candidate_selection_time,
            person_base_limits, x_margin)


def base_movable_region(base_limits):
    """``base_limits`` をワールド座標の台車の可動域の dict にする。"""
    x_range, y_range, yaw_range = base_limits
    return dict(
        x_range=[float(x_range[0]), float(x_range[1])],
        y_range=[float(y_range[0]), float(y_range[1])],
        yaw_range=[float(yaw_range[0]), float(yaw_range[1])],
    )


def solved_result(robot, robot_arm, target_pos, target_rot, turn_index,
                  angle_vector, base_pose, base_limits, post_process_result,
                  collision_ik_time, candidate_selection_time, hand, palm):
    """採用した解をロボットに反映して結果 dict を組む (後処理の結果は
    ``post_process`` キーに入れる)。"""
    robot.angle_vector(angle_vector)
    robot.newcoords(base_pose)
    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    result = dict(
        target=True,
        solved=True,
        turn_deg=turn_candidates_deg(hand, palm)[turn_index],
        target_position=[float(v) for v in target_pos],
        target_rot=[[float(v) for v in row] for row in target_rot],
        hand_position=[float(v) for v in hand_coords.worldpos()],
        hand_rot=[[float(v) for v in row] for row in hand_coords.worldrot()],
        base_position=[float(v) for v in robot.base_link.worldpos()],
        base_yaw=float(yaw),
        base_movable_region=base_movable_region(base_limits),
        joint_names=[j.name for j in robot.joint_list],
        joint_angle_vector=[float(v) for v in robot.angle_vector()],
        collision_ik_time=collision_ik_time,
        candidate_selection_time=candidate_selection_time,
    )
    result['post_process'] = post_process_result
    return result


def unsolved_result(robot, robot_arm, target_pos, target_rot, base_limits,
                    collision_ik_time, candidate_selection_time, hand, palm):
    """解けなかった人物の結果 dict (種の姿勢、向きは最後の候補)。"""
    seed_arm_pose(robot, robot_arm)
    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    return dict(
        target=True,
        solved=False,
        turn_deg=turn_candidates_deg(hand, palm)[-1],
        target_position=[float(v) for v in target_pos],
        target_rot=[[float(v) for v in row] for row in target_rot],
        hand_position=[float(v) for v in hand_coords.worldpos()],
        hand_rot=[[float(v) for v in row] for row in hand_coords.worldrot()],
        base_position=[float(v) for v in robot.base_link.worldpos()],
        base_yaw=float(yaw),
        base_movable_region=base_movable_region(base_limits),
        joint_names=[j.name for j in robot.joint_list],
        joint_angle_vector=[float(v) for v in robot.angle_vector()],
        collision_ik_time=collision_ik_time,
        candidate_selection_time=candidate_selection_time,
    )


def not_target_result(offered_hand, reason):
    """IK の対象外の人物の結果 dict (``reason``: 'no_offered_hand'/'no_palm')。"""
    return dict(target=False, solved=False, offered_hand=offered_hand,
                robot_arm=None, not_target_reason=reason)


def load_palm_json(path):
    """``estimate_palm_poses.save_json`` が保存した 1 人分の JSON を読む."""
    with open(path) as f:
        return json.load(f)


def save_json(result, path):
    """IK の結果 dict を JSON として保存する."""
    json_io.save_json(path, result)


iter_palm_files = json_io.iter_json_files
load_skeleton_json = json_io.load_skeleton_json

# バッチ IK のウォームアップ用のダミー掌 (値に意味は無い)。
_WARMUP_PALM = dict(
    position=[0.5, 0.0, 1.0],
    x_axis=[1.0, 0.0, 0.0],
    y_axis=[0.0, 1.0, 0.0],
)


def _warmup_batch_ik(robot, base_limits, ik_kwargs):
    """人物ループの前にダミー目標で ``solve_person_ik`` を解き、jax の
    トレース/コンパイルを済ませる (左右の腕ごと、結果は捨てる)。
    ``ik_kwargs`` は人物ループと同じものを渡すこと (違うと別のコンパイル)。
    """
    collision_obstacles = human_body_obstacles({})
    for robot_arm, hand in (('l', 'R'), ('r', 'L')):
        t0 = time.time()
        solve_person_ik(
            robot, _WARMUP_PALM, hand, robot_arm, collision_obstacles,
            base_limits=base_limits, joint_positions={}, **ik_kwargs)
        print('[warmup] {}腕: バッチIKのトレース/コンパイル {:.1f} 秒'
              .format(robot_arm, time.time() - t0))


def main():
    global TARGET_HOVER_OFFSET
    parser = argparse.ArgumentParser(
        description='掌の位置姿勢 JSON から、手を繋ぐ全身 IK (台車移動を含む) '
                    'を人物ごとに解いて JSON に保存する。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_palm_poses'),
        help='掌の位置姿勢 JSON の入力ディレクトリ。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='IK の結果 JSON の保存先 (入力と同じファイル名)。')
    parser.add_argument(
        '--attempts-per-pose', type=int,
        default=DEFAULT_ATTEMPTS_PER_POSE,
        help='1 つの目標姿勢に対して振る初期値の数 (既定 {})。'.format(
            DEFAULT_ATTEMPTS_PER_POSE))
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='骨格 JSON のディレクトリ (--input-dir と同じファイル名で対応)。'
            '人体の障害物を作るのに使う。')
    parser.add_argument(
        '--collision-pairs', type=str,
        default=os.path.join(_THIS_DIR, 'collision_pairs.json'),
        help='干渉回避でチェックする組の JSON (tools/build_collision_pairs.py '
            'が生成)。無ければ干渉回避と事後検証を無効にする (0 組なら干渉'
            '回避だけ無効)。')
    parser.add_argument(
        '--collision-verify-model', choices=COLLISION_VERIFY_MODELS,
        default=DEFAULT_COLLISION_VERIFY_MODEL,
        help='事後検証のモデル。mixed (既定) は自己干渉を指なし+手の箱、人体'
            'との距離を指ありで判定。nohand は両方箱、hand は両方指あり。')
    parser.add_argument(
        '--post-process-max-candidates', type=int,
        default=DEFAULT_POST_PROCESS_MAX_CANDIDATES,
        help='事後検証・後処理を試す候補数の上限 (既定・0 以下は無制限)。')
    parser.add_argument(
        '--final-correction-trials', type=int, default=0,
        help='押し込み直前の最終補正を台車のスリップを乱数で与えてこの回数'
            'だけ解き直し、結果の final_correction に記録する (既定 0)。')
    parser.add_argument(
        '--final-correction-slip', type=float, nargs=2, default=[0.03, 3.0],
        metavar=('XY', 'YAW_DEG'),
        help='--final-correction-trials のスリップの範囲 (±XY [m]、'
            '±YAW_DEG [度])。')
    parser.add_argument(
        '--base-x-standing-margins', type=float, nargs='+',
        default=list(DEFAULT_BASE_X_STANDING_MARGINS),
        help='台車の x を人の立ち位置 ±この幅 [m] に絞る。先頭から順に試す。'
            '負は絞らない (既定 {})。'
            .format(' '.join(str(m) for m in DEFAULT_BASE_X_STANDING_MARGINS)))
    parser.add_argument(
        '--front-offset-weight', type=float,
        default=DEFAULT_FRONT_OFFSET_WEIGHT,
        help='候補のコストに足す、台車の人の正面方向へのずれ [m] の重み '
            '(既定 {})。'.format(DEFAULT_FRONT_OFFSET_WEIGHT))
    parser.add_argument(
        '--facing-yaw-weight', type=float,
        default=DEFAULT_FACING_YAW_WEIGHT,
        help='候補のコストに足す、台車の向きのずれ [rad] の重み (既定 {})。'
            .format(DEFAULT_FACING_YAW_WEIGHT))
    # 以下は tools/grid_search_collision_ik.py でパラメータを振るためのもの。
    parser.add_argument(
        '--collision-ik-stop', type=int, default=DEFAULT_COLLISION_IK_STOP,
        help='干渉回避付きバッチ IK の反復回数 (既定 {})。'.format(
            DEFAULT_COLLISION_IK_STOP))
    parser.add_argument(
        '--collision-weight', type=float, default=DEFAULT_COLLISION_WEIGHT,
        help='バッチ IK の人体との干渉ペナルティの重み (既定 {})。'.format(
            DEFAULT_COLLISION_WEIGHT))
    parser.add_argument(
        '--collision-margin', type=float, default=DEFAULT_COLLISION_MARGIN,
        help='バッチ IK の人体との干渉ペナルティが効き始める距離 [m] '
            '(既定 {})。'.format(DEFAULT_COLLISION_MARGIN))
    parser.add_argument(
        '--self-collision-weight', type=float, default=None,
        help='バッチ IK の自己干渉ペナルティの重み (既定は '
            '--collision-weight と同じ)。')
    parser.add_argument(
        '--self-collision-margin', type=float,
        default=DEFAULT_SELF_COLLISION_MARGIN,
        help='バッチ IK の自己干渉ペナルティが効き始める距離 [m] (既定 {})。'
            .format(DEFAULT_SELF_COLLISION_MARGIN))
    parser.add_argument(
        '--turn-candidates', type=int, default=None,
        help='バッチに載せる向きの候補数 (turn_candidates_deg の先頭から、'
            '既定は全部の {})。'.format(NUM_TURN_CANDIDATES))
    parser.add_argument(
        '--base-y-half-range', type=float, default=BASE_Y_MOVABLE_HALF_RANGE,
        help='台車の y の可動域の半幅 [m] (既定 {})。'.format(
            BASE_Y_MOVABLE_HALF_RANGE))
    parser.add_argument(
        '--post-process-ik-stop', type=int,
        default=DEFAULT_POST_PROCESS_IK_STOP,
        help='後処理 IK (押し込み・視線) の反復回数の上限 (既定 {})。'.format(
            DEFAULT_POST_PROCESS_IK_STOP))
    parser.add_argument(
        '--post-process-thre', type=float,
        default=DEFAULT_POST_PROCESS_IK_THRE,
        help='後処理 IK の腕の位置の収束閾値 [m] (既定 {})。'.format(
            DEFAULT_POST_PROCESS_IK_THRE))
    parser.add_argument(
        '--post-process-rthre', type=float,
        default=math.degrees(DEFAULT_POST_PROCESS_IK_RTHRE),
        help='後処理 IK の腕の姿勢の収束閾値 [deg] (既定 {:.1f})。'.format(
            math.degrees(DEFAULT_POST_PROCESS_IK_RTHRE)))
    parser.add_argument(
        '--hover-human-clearance', type=float,
        default=DEFAULT_HOVER_HUMAN_CLEARANCE,
        help='hover 姿勢でロボットと人体の間に空ける距離 [m] (既定 {}、'
            '負で無効)。'.format(DEFAULT_HOVER_HUMAN_CLEARANCE))
    parser.add_argument(
        '--target-hover-offset', type=float, default=TARGET_HOVER_OFFSET,
        help='IK の目標を掌から法線方向に浮かせる距離 [m] (既定 {})。'
            .format(TARGET_HOVER_OFFSET))
    parser.add_argument(
        '--torso-surface-offset', type=float, default=0.0,
        help='干渉判定用に体幹の関節をカメラから離れる水平方向へずらす距離 '
            '[m]。実カメラの骨格では {} を指定する (既定 0)。'.format(
                DEFAULT_TORSO_SURFACE_OFFSET))
    args = parser.parse_args()
    TARGET_HOVER_OFFSET = args.target_hover_offset

    files = iter_palm_files(args.input_dir)
    if not files:
        print('{} に掌の位置姿勢 JSON が見つかりません。先に '
              'estimate_palm_poses.py を実行してください。'.format(
                  args.input_dir))
        return

    np.random.seed(0)
    os.makedirs(args.output_dir, exist_ok=True)
    robot = Aero(use_hand=False)
    restrict_elbow_range(robot)
    restrict_leg_range(robot)
    restrict_waist_range(robot)
    restrict_neck_range(robot)
    lock_fixed_joints(robot)
    apply_collision_model(robot)
    apply_hand_box(robot)
    other_hand_points('r')  # 指ありモデルの読み込みを先に済ませる

    collision_pairs = None
    verification_pairs = None
    if os.path.exists(args.collision_pairs):
        collision_pairs = load_collision_pairs(args.collision_pairs, robot)
        print('[collision-pairs] {} 組の干渉ペアを {} から読み込みました。'
              .format(len(collision_pairs), args.collision_pairs))
        # 事後検証は総当たりの組で行う。
        verification_pairs = build_verification_pairs_for_model(
            robot, args.collision_verify_model)
        print('[collision-verify] 事後検証は {} モデルの {} 組で行います。'
              .format(args.collision_verify_model, len(verification_pairs)))
    else:
        print('[collision-pairs] {} が見つからないため、干渉回避 (自己干渉'
              '・人体との干渉の両方) を無効にして IK を解きます。'.format(
                  args.collision_pairs))

    base_limits = [DEFAULT_BASE_X_RANGE,
                   (-args.base_y_half_range, args.base_y_half_range),
                   DEFAULT_BASE_YAW_RANGE]

    # 人物ごとに変わらない solve_person_ik の引数 (warmup と共通)。
    ik_kwargs = dict(
        attempts_per_pose=args.attempts_per_pose,
        collision_weight=args.collision_weight,
        collision_margin=args.collision_margin,
        self_collision=(collision_pairs is not None),
        collision_pairs=collision_pairs,
        self_collision_weight=args.self_collision_weight,
        self_collision_margin=args.self_collision_margin,
        collision_ik_stop=args.collision_ik_stop,
        collision_ik_thre=DEFAULT_COLLISION_IK_THRE,
        collision_ik_rthre=DEFAULT_COLLISION_IK_RTHRE,
        collision_joint_limit_margin_ratio=(
            DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO),
        verification_pairs=verification_pairs,
        collision_verify_tolerance=DEFAULT_COLLISION_VERIFY_TOLERANCE,
        post_process_ik_stop=args.post_process_ik_stop,
        post_process_thre=args.post_process_thre,
        post_process_rthre=math.radians(args.post_process_rthre),
        post_process_max_candidates=(
            args.post_process_max_candidates
            if args.post_process_max_candidates
            and args.post_process_max_candidates > 0 else None),
        n_turn_candidates=args.turn_candidates,
        hover_human_clearance=(args.hover_human_clearance
                               if args.hover_human_clearance >= 0.0 else None))

    # IK 対象が 1 人もいなければウォームアップしない。
    has_target = any(
        load_palm_json(path).get('offered_hand') in ('L', 'R')
        for path in files)
    if has_target:
        _warmup_batch_ik(robot, base_limits, ik_kwargs)

    n_solved = 0
    n_total = 0
    n_not_target = 0
    for i, path in enumerate(files):
        # 結果が処理順に依存しないよう、人物ごとにシードを決め直す。
        np.random.seed(zlib.crc32(os.path.basename(path).encode()))
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        palms = load_palm_json(path)
        human_hand = palms.get('offered_hand')
        palm = palms.get(human_hand) if human_hand in ('L', 'R') else None
        if palm is None:
            reason = ('no_palm' if human_hand in ('L', 'R')
                      else 'no_offered_hand')
            save_json(not_target_result(human_hand, reason), out_path)
            n_not_target += 1
            print('[{}/{}] {} -> {} (not target: {})'.format(
                i + 1, len(files), os.path.basename(path), out_path, reason))
            continue
        robot_arm = DEFAULT_ROBOT_ARM[human_hand]

        # 骨格があれば人物を前方 HUMAN_FRONT_DISTANCE へ平行移動し、身体を
        # 障害物にする (無ければ人体との干渉回避なし)。
        skeleton_path = os.path.join(args.skeleton_dir,
                                     os.path.basename(path))
        if os.path.exists(skeleton_path):
            joint_positions = load_skeleton_json(skeleton_path)
            offset = human_translation_offset(
                joint_positions, front_distance=HUMAN_FRONT_DISTANCE)
            joint_positions = translate_joint_positions(
                joint_positions, offset)
            # 干渉判定用の骨格 (カメラは平行移動後 offset の位置)。
            collision_joints = shift_torso_joints_from_surface(
                joint_positions, offset, args.torso_surface_offset)
            collision_obstacles = (
                [] if collision_pairs is None
                else human_body_obstacles(collision_joints))
            palm = translate_palm(palm, offset)
        else:
            print('  {} に骨格 JSON が無いため、この人物は人体との干渉回避 '
                  'なしで解きます。'.format(skeleton_path))
            collision_obstacles = []
            joint_positions = collision_joints = None

        # 台車の y を差し出した手の側に、yaw を人の正面方向付近に絞る。
        person_base_limits = base_limits
        side_sign = offered_hand_side_sign(
            human_hand, joint_positions, palm)
        if side_sign is not None:
            person_base_limits = [
                base_limits[0],
                restrict_base_y_range_to_hand_side(
                    base_limits[1], side_sign),
                base_limits[2]]

        if joint_positions is not None:
            human_yaw = human_facing_yaw(joint_positions)
            if human_yaw is not None:
                person_base_limits = [
                    person_base_limits[0],
                    person_base_limits[1],
                    restrict_base_yaw_range_to_human_facing(
                        person_base_limits[2], human_yaw,
                        margin=math.radians(DEFAULT_BASE_YAW_FACING_MARGIN_DEG))]

        standing_xy = (None if joint_positions is None
                       else human_standing_xy(joint_positions))
        standing_x = None if standing_xy is None else float(standing_xy[0])

        target_pos = palm_target_position(palm)
        rots = palm_to_target_rots(palm, human_hand, robot_arm)
        (picked, collision_ik_time, candidate_selection_time,
         person_base_limits, x_margin) = solve_person_ik_side_by_side(
            robot, palm, human_hand, robot_arm, collision_obstacles,
            person_base_limits, standing_x,
            x_margins=args.base_x_standing_margins,
            front_offset_weight=args.front_offset_weight,
            facing_yaw_weight=args.facing_yaw_weight,
            joint_positions=collision_joints,
            placement_joint_positions=joint_positions, **ik_kwargs)
        if picked is None:
            result = unsolved_result(
                robot, robot_arm, target_pos, rots[-1], person_base_limits,
                collision_ik_time, candidate_selection_time, human_hand,
                palm)
        else:
            turn_index, angle_vector, base_pose, post_process_result \
                = picked
            result = solved_result(
                robot, robot_arm, target_pos, rots[turn_index],
                turn_index, angle_vector, base_pose,
                person_base_limits, post_process_result, collision_ik_time,
                candidate_selection_time, human_hand, palm)
            if (args.final_correction_trials > 0
                    and post_process_result is not None):
                # 共有の np.random を進めないよう別の乱数で。
                result['final_correction'] = simulate_final_correction(
                    robot, robot_arm, result, palm, angle_vector, base_pose,
                    args.final_correction_trials,
                    args.final_correction_slip[0],
                    math.radians(args.final_correction_slip[1]),
                    np.random.RandomState(i))
        result['offered_hand'] = human_hand
        result['robot_arm'] = robot_arm
        result['base_x_standing_margin'] = x_margin
        n_total += 1
        n_solved += int(result['solved'])
        save_json(result, out_path)
        print('[{}/{}] {} -> {} ({}Hand -> {}arm, {})'.format(
            i + 1, len(files), os.path.basename(path), out_path,
            human_hand, robot_arm,
            'solved' if result['solved'] else 'NOT solved'))

    print('{}/{} solved (対象外 {} 人: 掌推定の offered_hand が null 等)。'
          .format(n_solved, n_total, n_not_target))


if __name__ == '__main__':
    main()
