#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人間の掌の位置姿勢 JSON (``estimate_palm_poses.py`` の出力) を入力とし、
Aeroが全身 IK (台車移動を含む) を解いて手を繋ぐ姿勢を求め、その結果
(関節角・台車位置・手先姿勢) を JSON として保存する。``rospy`` は import
せず、保存済みの JSON を読んで 1 回だけ IK を解くオフライン処理。

対象にするのは、掌推定が「人がこの手を差し出している」と判定した手
(掌 JSON の ``offered_hand`` が ``'L'``/``'R'``) を持つ人物だけ。
``offered_hand`` が ``null`` の人物は IK を解かず、入力と同じファイル名で
``target: false`` の JSON (IK の結果は持たない) を書き出す。使うロボットの
腕は既定で人間の手の反対側 (``--robot-arm`` で上書き可)。

Aero は常にワールド原点・台車位置固定で IK を開始する (``seed_arm_pose``
参照) ため、人物側 (骨格の全関節位置・掌の目標位置) を x/y 方向に平行移動
し、その人物の立ち位置がちょうど Aero の前方 ``HUMAN_FRONT_DISTANCE`` [m]
に来るようにしてから IK を解く (``human_translation_offset``/
``translate_joint_positions``/``translate_palm`` 参照)。台車の移動範囲は
``--base-x-range``/``--base-y-range``/``--base-yaw-range`` で指定する。

人体を障害物とした干渉回避も行う。``--skeleton-dir`` から人物ごとの全身の
関節位置を読み、体幹・頭部・四肢を ``skrobot.model.primitives.Cylinder``
で近似し、``batch_inverse_kinematics`` の ``collision_obstacles`` に渡す。
干渉を避ける対象のロボットリンクは台車・胴体・頭部・両腕を含む全身
(``collision_link_list_for_arm`` 参照)。ロボット側の干渉ジオメトリは実
メッシュではなく、``view_aero_collision_model.py`` と同じ方法で生成・
キャッシュした box/cylinder/sphere のプリミティブ近似形状を使う
(``apply_collision_model`` 参照)。IK の目標位置は掌から ``TARGET_HOVER_
OFFSET`` だけ浮かせ、目標そのものが人体の干渉回避ジオメトリと重ならない
ようにしている。``batch_inverse_kinematics`` の収束判定はこの干渉ペナル
ティの残差を見ないため、``solve_person_ik`` は収束した候補について採用前
に必ず ``collision_pairs_min_distance`` で厳密な形状による事後検証を行い、
実際に貫通したままの解は棄却する (``--collision-verify-tolerance`` 参照)。

さらに、干渉検証まで通った候補についても、``TARGET_HOVER_OFFSET`` で
浮かせた目標に届いただけでは実際に人間に掌を押し付けられる保証は無い
ため、``pick_verified_candidate`` は候補ごとに最後に後処理判定
(``solve_post_process``) を行う。台車を動かさない通常のヤコビアン法 IK
で、腕を目標位置が掌へわずかにめり込む位置 (``POST_PROCESS_TARGET_HOVER_
OFFSET``) まで詰め直すのと、ロボットが人間の差し出している手を見るよう
首を向けるのを 1 回の IK 呼び出しで同時に解く。後処理判定に失敗した候補
は棄却し、次の候補を試す。全ての候補で後処理判定に失敗した場合のみ、
干渉検証を通過した最初の候補を後処理前のまま (``post_process`` キーを
``null`` にして) 採用するフォールバックを行う (``pick_verified_candidate``
参照)。

人体だけでなく、ロボット自身のリンク同士の干渉 (自己干渉) も既定で回避
する (``batch_inverse_kinematics`` の ``self_collision=True``、``--no-
self-collision`` で無効化できる)。チェックする組み合わせは常に
``--collision-pairs`` (JSON: 2 要素の名前のリストのリスト。
``build_collision_pairs.py`` が ``analyze_collision_pairs.py`` の出力から
生成する) で明示的に指定した組み合わせだけに限る (``load_collision_
pairs`` 参照)。このファイルが既定のパスに無ければ、干渉回避を丸ごと
無効にして通常のヤコビアン法の IK にフォールバックする。

向きを 0/±90 度 (``TURN_CANDIDATES_DEG``) ずらした目標を、それぞれ
``--attempts-per-pose`` 個の初期値から解き、そうしてできる全候補を順に
試して、干渉回避付きバッチ IK が解けて (収束・事後の干渉検証を通過して)
かつ後処理判定にも成功した最初のものを採用する (``pick_verified_
candidate`` 参照)。``return_all_attempts=True`` を渡し、初期値ごとの解を
集約させず全て候補として受け取る (``solve_person_ik`` 参照)。IK は
``batch_inverse_kinematics`` で人物 1 人ごとに 1 回解く (その人の全向き
× 全初期値を 1 バッチにまとめる。``collision_obstacles`` は 1 回の
バッチ呼び出し全体で 1 つの集合しか渡せないため)。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo estimate_palm_poses.py
    rosrun aero_demo solve_palm_ik.py

(いずれも --input-dir/--output-dir を省略すると、scripts/ 直下の
random_human_poses/ -> random_palm_poses/ -> random_handshake_poses/ を
共通の入出力先として自動的につながる)
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

# jax の永続コンパイルキャッシュを有効にする (干渉回避付きバッチ IK の
# 初回コンパイルは数分かかるため)。jax を import する前に環境変数で
# 指定する必要がある。既に設定済みならユーザーの指定を優先し上書きしない。
os.environ.setdefault(
    'JAX_COMPILATION_CACHE_DIR',
    os.path.expanduser('~/.cache/jax_compilation_cache'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import matrix2ypr, normalize_mask  # noqa: E402
from skrobot.model import RobotModel  # noqa: E402
from skrobot.model.primitives import Cylinder  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.planner.trajectory_optimization.collision import (  # noqa: E402
    create_self_collision_pairs)

from view_aero_collision_model import build_collision_model_urdf  # noqa: E402

# 掌のローカル +Y (甲->掌方向) まわりにこの角度ずつ向きをずらした候補を
# 順に試し、IK が解けた最初のものを採用する。
TURN_CANDIDATES_DEG = (0.0, 90.0, -90.0)

# 人間の手 (掌 JSON の offered_hand) に対して既定で使うロボットの腕
# (--robot-arm auto)。向かい合わず、人間と同じ方向を向いて反対側の手で
# 繋ぐ想定なので、人の手とは反対側の腕を使う。
DEFAULT_ROBOT_ARM = {'L': 'r', 'R': 'l'}

# Aero は常にワールド原点で IK を開始するため、人物側をこの距離だけ
# Aero の前方 (+x) に平行移動してから IK を解く。
HUMAN_FRONT_DISTANCE = 3.0  # [m]

# 台車の既定の移動可能領域の半幅 (人物の立ち位置を中心とした前後・左右
# それぞれの距離)。
BASE_X_MOVABLE_HALF_RANGE = 3.0  # [m]
BASE_Y_MOVABLE_HALF_RANGE = 3.0  # [m]

# 台車 (use_base='planar' の仮想関節) の既定の移動範囲。IK 開始時の台車
# 位置 (ワールド原点) を基準にした [x, y, yaw] の (下限, 上限)。
DEFAULT_BASE_X_RANGE = (HUMAN_FRONT_DISTANCE - BASE_X_MOVABLE_HALF_RANGE,
                        HUMAN_FRONT_DISTANCE + BASE_X_MOVABLE_HALF_RANGE)
DEFAULT_BASE_Y_RANGE = (-BASE_Y_MOVABLE_HALF_RANGE, BASE_Y_MOVABLE_HALF_RANGE)
DEFAULT_BASE_YAW_RANGE = (-math.pi / 2.0, math.pi / 2.0)

# 1 目標姿勢あたりに振る初期値の数 (バッチ IK の attempts_per_pose)。
# 初期値 0 は seed_arm_pose の種の姿勢、残りは関節範囲・台車の可動範囲の
# 一様乱数。
DEFAULT_ATTEMPTS_PER_POSE = 512

# 干渉回避 (collision_obstacles) 付きバッチ IK の収束判定。
DEFAULT_COLLISION_IK_STOP = 80
DEFAULT_COLLISION_IK_THRE = 0.03  # [m]
DEFAULT_COLLISION_IK_RTHRE = math.radians(8.0)  # [rad]

# 干渉回避ペナルティの重み・マージン (skrobot の batch_inverse_kinematics
# の既定値と同じ)。
DEFAULT_COLLISION_WEIGHT = 10.0
DEFAULT_COLLISION_MARGIN = 0.05

# 自己干渉回避ペナルティのマージン (skrobot の既定値と同じ)。重みは既定で
# collision_weight と同じ値を使う。
DEFAULT_SELF_COLLISION_MARGIN = 0.02

# IK のターゲットを掌からどれだけ浮かせるか (法線方向) [m]。
TARGET_HOVER_OFFSET = 0.08  # [m]

# 後処理判定 (solve_post_process) で使う、掌からのオフセット [m]。負の値
# (掌の内側にわずかにめり込む位置) にすることで、実際に掌へ手を近づけ
# られる姿勢が取れるかを検証する。
POST_PROCESS_TARGET_HOVER_OFFSET = -0.01  # [m]

# 後処理判定の腕 IK (通常のヤコビアン法, 干渉回避なし・台車なし) の最大
# 反復回数。腕・視線は 1 つのループで一緒に反復されるため、実効上限は
# DEFAULT_POST_PROCESS_GAZE_IK_STOP との max になる。
DEFAULT_POST_PROCESS_IK_STOP = 40

# 後処理判定の腕 IK の収束閾値 (位置[m]/姿勢[rad])。事前段階の干渉回避
# 付きバッチ IK の閾値 (DEFAULT_COLLISION_IK_THRE/_RTHRE) がこれより緩い
# ため、skrobot 既定の厳しい基準では後処理が失敗しやすく、それに合わせて
# 緩めてある。
DEFAULT_POST_PROCESS_IK_THRE = 0.01  # [m]
DEFAULT_POST_PROCESS_IK_RTHRE = math.radians(5.0)  # [rad]

# 後処理判定の視線 IK (人間の手を見る首 3 関節) の最大反復回数・収束閾値
# (姿勢のみ, rotation_mask='xy' で視線軸まわりの回転は見ない)。
DEFAULT_POST_PROCESS_GAZE_IK_STOP = 40
DEFAULT_POST_PROCESS_GAZE_IK_RTHRE = math.radians(5.0)  # [rad]

# 肘関節 ({r,l}_elbow_joint) の可動域制限 [deg]。0 度が腕をまっすぐ伸ばした
# 状態、負方向が肘を曲げる方向。可動域いっぱいまで曲げた IK 解は前腕と
# 上腕が平行に近づく不自然な姿勢になりやすいため、下限だけこの値まで
# 緩める (restrict_elbow_range 参照)。上限 (0 度) は変更しない。
ELBOW_MIN_ANGLE_DEG = -120.0

# 干渉回避付きバッチ IK (solve_person_ik) だけに適用する、関節可動域の
# 上下マージン比率 (restrict_joint_range_margin 参照)。
DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO = 0.1

# 人体の干渉回避用ジオメトリ: (骨格の関節名 A, 関節名 B, 半径[m]) の
# タプルの並び。BODY_JOINT_NAMES (generate_random_human_poses.py) の
# 関節を結ぶ主要な骨を円柱 (Capsule) で近似する。
HUMAN_COLLISION_SEGMENTS = (
    ('Nose', 'Neck', 0.10),
    ('Neck', 'RShoulder', 0.09),
    ('Neck', 'LShoulder', 0.09),
    ('RShoulder', 'RElbow', 0.06),
    ('LShoulder', 'LElbow', 0.06),
    ('RElbow', 'RWrist', 0.04),
    ('LElbow', 'LWrist', 0.04),
    ('Neck', 'RHip', 0.13),
    ('Neck', 'LHip', 0.13),
    ('RHip', 'LHip', 0.13),
    ('RHip', 'RKnee', 0.09),
    ('RKnee', 'RAnkle', 0.06),
    ('LHip', 'LKnee', 0.09),
    ('LKnee', 'LAnkle', 0.06),
)

# 手 (指先まで) の干渉回避用ジオメトリ。骨格の関節位置は手首までしか無く、
# 手首より先の指はカバーされないため、MediaPipe 形式の手のランドマーク
# ({R,L}Hand0..{R,L}Hand20, 0 が手首, 1-4/5-8/9-12/13-16/17-20 がそれぞれ
# 親指/人差し指/中指/薬指/小指) を使う。掌は手首と 4 本の指の付け根 (MCP)
# を囲む平たい Cylinder、各指は付け根から指先までを結ぶ細い Cylinder で
# 近似する。
HAND_PALM_LANDMARKS = (0, 5, 9, 13, 17)  # 手首 + 4 指の付け根 (MCP)
HAND_FINGER_LANDMARKS = (
    (1, 4),    # 親指: CMC -> 指先
    (5, 8),    # 人差し指: MCP -> 指先
    (9, 12),   # 中指: MCP -> 指先
    (13, 16),  # 薬指: MCP -> 指先
    (17, 20),  # 小指: MCP -> 指先
)
# HAND_FINGER_LANDMARKS の各指に対応する名前 (--collision-pairs JSON で
# 人体側のオブジェクトを指すのに使う。analyze_collision_pairs.py の
# _FINGER_LABELS と同じ順序)。
HAND_FINGER_LABELS = ('thumb', 'index', 'middle', 'ring', 'pinky')
HAND_PALM_RADIUS = 0.05  # [m] 掌の円柱の半径
HAND_PALM_HEIGHT = 0.02  # [m] 掌の円柱の厚み (平たくする)
HAND_FINGER_RADIUS = 0.008  # [m] 指の円柱の半径 (細くする)

# human_body_obstacles が返す障害物の個数を人物によらず常に固定にする
# ため (JAX の再コンパイルを避けるため)、骨格の関節が欠けている骨・手を
# 埋めるダミー障害物をこの距離だけ離れた位置に置く [m]。
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
    """左腕用に +Y/+Z を反転する.

    ``l_eef_grasp_link`` は ``r_eef_grasp_link`` に対して +X (指方向)
    まわりに 180 度ずれている (URDF が左右ミラーで作られているため) ので、
    右腕用に組んだ ``rot`` を左腕で使うにはこの補正が要る。
    """
    if arm != 'l':
        return rot
    return np.column_stack([rot[:, 0], -rot[:, 1], -rot[:, 2]])


def palm_to_target_rots(palm, robot_arm):
    """掌の位置姿勢 JSON (``estimate_palm_poses.PalmPoseEstimator`` の
    出力の 1 手分) から、ロボットの手先座標系 (``{arm}_eef_grasp_link``,
    +X=指方向, +Y=甲->掌方向, +Z=+X×+Y) で表した目標姿勢の候補群
    (``TURN_CANDIDATES_DEG`` の数だけ) を返す。

    向かい合う握手ではなく、人間と同じ方向を向いて反対側の手で繋ぐ想定
    なので、指方向 (+X) は鏡写しにせずそのまま使う。ロボットの +Y は
    掌の法線 (``y_axis``) の逆向き (``-y_axis``、人間の掌に正対する向き)。
    """
    x_axis = np.asarray(palm['x_axis'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    y_axis = -normal
    z_axis = np.cross(x_axis, y_axis)
    base_rot = np.column_stack([x_axis, y_axis, z_axis])
    return [_correct_grasp_frame(_turn_about_y(base_rot, deg), robot_arm)
           for deg in TURN_CANDIDATES_DEG]


def palm_target_position(palm):
    """掌から ``TARGET_HOVER_OFFSET`` だけ浮かせた位置 (法線方向) を IK の
    目標位置として返す。"""
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    return position + normal * TARGET_HOVER_OFFSET


def human_standing_xy(joint_positions):
    """人物の立ち位置 (x, y) の目安を骨格の関節位置から求める.

    骨盤 (``RHip``/``LHip`` の中点) を優先し、両方無ければ ``Neck``、
    それも無ければ ``None`` (平行移動は行わない) を返す。
    """
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
    """人物をちょうど Aero の前方 ``front_distance`` [m] に置くための
    平行移動量 (dx, dy) を返す.

    Aero は常にワールド原点、向き +x で IK を開始するので、人物の立ち位置
    (``human_standing_xy``) が ``(front_distance, 0.0)`` に一致するように
    移動する量を返す。立ち位置が骨格から求まらないときは ``(0.0, 0.0)``
    (平行移動なし) を返す。
    """
    person_xy = human_standing_xy(joint_positions)
    if person_xy is None:
        return (0.0, 0.0)
    target_xy = np.array([front_distance, 0.0])
    offset = target_xy - person_xy
    return (float(offset[0]), float(offset[1]))


def translate_joint_positions(joint_positions, offset):
    """骨格の全関節位置 (``joint_positions``) を x/y 方向にだけ ``offset``
    平行移動したコピーを返す (z は身長方向なので変えない)。``offset`` が
    ``(0.0, 0.0)`` のときは ``joint_positions`` をそのまま返す。
    """
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
    """掌の位置姿勢 JSON の 1 手分 (``palm``) の ``position`` を x/y 方向に
    だけ ``offset`` 平行移動したコピーを返す。向き (``x_axis``/``y_axis``)
    は平行移動の影響を受けないのでそのまま。``offset`` が ``(0.0, 0.0)``
    のときは ``palm`` をそのまま返す。
    """
    dx, dy = offset
    if dx == 0.0 and dy == 0.0:
        return palm
    translated = dict(palm)
    position = list(palm['position'])
    position[0] += dx
    position[1] += dy
    translated['position'] = position
    return translated


def seed_arm_pose(robot, robot_arm):
    """バッチ IK の初期値 0 (attempt 0) に使う「種の姿勢」をロボットに作る.

    向かい合わず、人間と同じ方向を向いて手を繋ぐ構え (肩を横に開き、
    手首はひねらないニュートラルな姿勢)。台車は常にワールド原点に置く。

    ``robot.reset_pose()`` は関節角度だけを戻し、台車の位置姿勢は変えない
    ため、``robot``/``base_link`` の両方を明示的に単位姿勢へ戻す
    (Aero では ``base_link.worldpos()`` が両者の変換の積で決まるため)。

    使わない方の腕 (``robot_arm`` の反対側) は IK の最適化対象に含めない
    (``solve_person_ik``/``solve_post_process`` とも ``link_list`` に
    含まれない) ため、ここで作った姿勢がそのまま最終結果の
    ``joint_angle_vector`` にも残る。``reset_pose()`` は肘を目一杯曲げた
    (``elbow_joint`` を -135 度にする) 姿勢で、これは前腕を肩の高さまで
    持ち上げた「構え」のような姿勢になってしまう (``plan_handshake_
    motion.arms_down_angles`` 参照) ため、使わない方の腕だけ肘を伸ばして
    (0 度) 体の横に自然に下ろした姿勢に戻しておく。
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


def restrict_elbow_range(robot, min_angle_deg=ELBOW_MIN_ANGLE_DEG):
    """``r_elbow_joint``/``l_elbow_joint`` の下限 (最大屈曲角) を
    ``min_angle_deg`` まで緩める (``ELBOW_MIN_ANGLE_DEG`` 参照)。

    ``batch_inverse_kinematics`` は各関節の ``min_angle``/``max_angle`` を
    呼び出しのたびに読み直すので、ここで書き換えれば以降の全人物の IK に
    効く。``rarm_whole_body``/``larm_whole_body`` も ``robot`` と同じ
    ``Joint`` オブジェクトを共有しているので、``robot`` 側だけ書き換えれば
    両方に反映される。
    """
    min_angle = math.radians(min_angle_deg)
    for arm in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(arm)).min_angle = min_angle


def restrict_joint_range_margin(link_list, margin_ratio):
    """``link_list`` 中の各関節の可動域を、上下限それぞれ全域幅の
    ``margin_ratio`` だけ内側に一時的に制限する
    (``DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO`` 参照)。

    可動域が有限でない関節 (台車の仮想関節等) や、全域幅が非正の関節は
    対象外とする。

    Parameters
    ----------
    link_list : list[skrobot.model.Link]
        対象のリンク (``link.joint`` が対象の関節)。``batch_inverse_
        kinematics`` に渡す ``link_list`` と同じものを渡す想定。
    margin_ratio : float
        上下限それぞれ全域幅に掛けるマージンの比率 (0 以上 0.5 未満)。

    Returns
    -------
    callable
        呼び出すと各関節の可動域を書き換え前の値に戻す関数
        (``try``/``finally`` で呼び出し側が必ず呼ぶ想定)。
    """
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
    """``p0``-``p1`` を結ぶ線分を近似する ``Cylinder`` (骨格の 1 本の骨)
    を作る。円柱のローカル +Z が線分の向きになるよう回転させる。"""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    diff = p1 - p0
    height = float(np.linalg.norm(diff))
    if height < 1e-6:
        # 関節位置がほぼ同一 (推定誤差等) のときに退化しないよう、
        # ごく小さい円柱にする。
        height = 1e-6
        z_axis = np.array([0.0, 0.0, 1.0])
    else:
        z_axis = diff / height
    # z_axis とほぼ平行にならない適当な軸から正規直交基底を作る。
    seed = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return Cylinder(radius=radius, height=height,
                    pos=((p0 + p1) / 2.0).tolist(), rot=rot)


def _palm_obstacle(points):
    """掌を近似する平たい ``Cylinder`` を作る。``points`` は手首 + 4 指の
    付け根 (``HAND_PALM_LANDMARKS`` の順, MediaPipe 手ランドマーク) の
    座標。円柱の軸 (厚み方向) は掌面の法線 (手首->人差し指付け根,
    手首->小指付け根 の外積) にする。"""
    points = np.asarray(points, dtype=np.float64)
    center = points.mean(axis=0)
    wrist, index_mcp, pinky_mcp = points[0], points[1], points[-1]
    normal = np.cross(index_mcp - wrist, pinky_mcp - wrist)
    norm = np.linalg.norm(normal)
    z_axis = normal / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    # z_axis とほぼ平行にならない適当な軸から正規直交基底を作る
    # (円柱は軸まわり対称なので x/y の向きは何でもよい)。
    seed = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(seed, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])
    return Cylinder(radius=HAND_PALM_RADIUS, height=HAND_PALM_HEIGHT,
                    pos=center.tolist(), rot=rot)


def _dummy_cylinder(radius):
    """``DUMMY_OBSTACLE_DISTANCE`` だけ離れた位置に置くダミー ``Cylinder``
    (骨・掌・指が欠けている場合に個数を揃えるため)。"""
    return Cylinder(radius=radius, height=1e-3,
                    pos=[DUMMY_OBSTACLE_DISTANCE] * 3)


def human_body_obstacles(joint_positions):
    """骨格の関節位置 (``generate_random_human_poses.py`` の
    ``skeleton.joint_positions``) から、干渉回避の障害物として使う
    ``Cylinder`` のリストを作る (``HUMAN_COLLISION_SEGMENTS``/
    ``HAND_PALM_LANDMARKS``/``HAND_FINGER_LANDMARKS`` 参照)。差し出して
    いる側の腕・手も含め、全身を障害物にする。

    常に ``len(HUMAN_COLLISION_SEGMENTS) + 2 * (1 + len(HAND_FINGER_
    LANDMARKS))`` 個 (骨格検出が全身分揃っているときの最大数) を返す --
    関節位置が片方でも欠けている骨・掌・指は、``DUMMY_OBSTACLE_DISTANCE``
    だけ離れたダミーの ``Cylinder`` で埋める。
    """
    obstacles = []
    for name_a, name_b, radius in HUMAN_COLLISION_SEGMENTS:
        if name_a in joint_positions and name_b in joint_positions:
            obstacles.append(_cylinder_between(
                joint_positions[name_a], joint_positions[name_b], radius))
        else:
            obstacles.append(_dummy_cylinder(radius))
    for side in ('R', 'L'):
        palm_names = ['{}Hand{}'.format(side, idx)
                     for idx in HAND_PALM_LANDMARKS]
        if all(name in joint_positions for name in palm_names):
            obstacles.append(_palm_obstacle(
                [joint_positions[name] for name in palm_names]))
        else:
            obstacles.append(_dummy_cylinder(HAND_PALM_RADIUS))
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


def human_obstacle_names():
    """``human_body_obstacles`` が返すリストと同じ順序・同じ個数の名前の
    リストを返す (``joint_positions`` の中身に依存しない構造だけの情報)。

    ``--collision-pairs`` (JSON) で人体側のオブジェクトを指定するときの
    名前、および ``analyze_collision_pairs.py`` が書き出す
    ``collision_pair_analysis.json`` の ``human_collision_min_dist`` の
    キーの後半と対応する。``load_collision_pairs`` がこのリストを使い、
    JSON 中の名前がロボットのリンク名でなければ人体セグメント名とみなして
    ``human_body_obstacles`` の出力中の対応するインデックスに解決する。
    """
    names = ['{}-{}'.format(name_a, name_b)
            for name_a, name_b, _ in HUMAN_COLLISION_SEGMENTS]
    for side in ('R', 'L'):
        names.append('{}_palm'.format(side))
        for label in HAND_FINGER_LABELS:
            names.append('{}_{}'.format(side, label))
    return names


def segment_points_distance(p0, p1, points):
    """線分 ``p0``-``p1`` と、複数の点 ``points`` (``(N, 3)``) それぞれとの
    最短距離 (``(N,)``)。"""
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
    """``human_body_obstacles`` と同じ順序・同じ部位のカプセル (線分 2 端点
    + 半径) のリストと、対応する名前 (``human_obstacle_names`` と同じ) の
    リストを返す。骨格の関節が欠けている部位は ``DUMMY_OBSTACLE_DISTANCE``
    だけ離れた点に潰す。``analyze_collision_pairs.py`` と
    ``collision_pairs_min_distance`` が、``Cylinder`` の代わりに素の
    (線分, 半径) を使いたいときに使う。"""
    caps = []
    names = []
    dummy = np.array([DUMMY_OBSTACLE_DISTANCE] * 3)
    for name_a, name_b, radius in HUMAN_COLLISION_SEGMENTS:
        if name_a in joint_positions and name_b in joint_positions:
            p0 = np.asarray(joint_positions[name_a], dtype=np.float64)
            p1 = np.asarray(joint_positions[name_b], dtype=np.float64)
        else:
            p0 = p1 = dummy
        caps.append((p0, p1, radius))
        names.append('{}-{}'.format(name_a, name_b))
    for side in ('R', 'L'):
        palm_names = ['{}Hand{}'.format(side, idx)
                     for idx in HAND_PALM_LANDMARKS]
        if all(name in joint_positions for name in palm_names):
            pts = np.array([joint_positions[name] for name in palm_names],
                           dtype=np.float64)
            center = pts.mean(axis=0)
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


# collision_pairs_min_distance が「実際には貫通していた」と判定する際の
# 許容誤差 [m] (プリミティブ形状のポリゴン近似誤差を吸収する程度の値)。
DEFAULT_COLLISION_VERIFY_TOLERANCE = 0.001  # [m]


def collision_pairs_min_distance(robot, collision_pairs, joint_positions):
    """``robot`` の現在の姿勢 (``angle_vector``/``base_pose`` 適用済み) で、
    ``collision_pairs`` (``load_collision_pairs`` が返す ``(Link, Link)``/
    ``(Link, int)`` 混在のリスト) の中で最も干渉している (最小の) 距離
    [m] を返す。負の値は貫通していることを意味する。``collision_pairs`` が
    空/``None`` のときは ``float('inf')`` を返す (検証対象なし)。

    ``analyze_collision_pairs.py`` が干渉ペア候補を洗い出すのに使ったのと
    同じ厳密な形状 (``apply_collision_model`` が差し替えた
    ``collision_mesh`` の頂点そのもの) を使って距離を計算する。
    """
    if not collision_pairs:
        return float('inf')
    caps = human_capsules(joint_positions)[0] if joint_positions else None
    min_dist = float('inf')
    world_vertices_by_link = {}

    def _world_vertices(link):
        if link not in world_vertices_by_link:
            local = np.asarray(link.collision_mesh.vertices, dtype=np.float64)
            world_vertices_by_link[link] = (
                local @ link.worldrot().T + link.worldpos())
        return world_vertices_by_link[link]

    for link_a, other in collision_pairs:
        verts_a = _world_vertices(link_a)
        if isinstance(other, int):
            if caps is None:
                continue
            p0, p1, radius = caps[other]
            dist = float(segment_points_distance(p0, p1, verts_a).min()) \
                - radius
        else:
            verts_b = _world_vertices(other)
            dist = float(np.linalg.norm(
                verts_a[:, np.newaxis, :] - verts_b[np.newaxis, :, :],
                axis=-1).min())
        if dist < min_dist:
            min_dist = dist
    return min_dist


def apply_collision_model(robot, primitive_type=None, force_convert=False,
                          collision_urdf_path=None):
    """``robot`` (実メッシュの Aero) の各リンクの ``collision_mesh`` を、
    ``view_aero_collision_model.py`` と同じ方法 (``skrobot.urdf.
    convert_meshes_to_primitives``) で生成したプリミティブ近似形状に
    差し替える。

    ``build_collision_model_urdf`` がプリミティブ近似 URDF をファイルと
    してキャッシュする (既に生成済みならそれを再利用し、``force_convert``
    を指定したときだけ作り直す)。

    差し替えは ``robot`` の各リンクを直接書き換えて行う (``RobotModel.
    load_urdf_file`` でロボット全体を作り直すのではない) ため、Aero
    クラスが提供する ``rarm_end_coords``/``rarm_whole_body`` などの
    キネマティクス関連の属性やジョイント名はそのまま使える。

    Parameters
    ----------
    collision_urdf_path : str or None
        指定すると、``build_collision_model_urdf`` によるプリミティブ近似
        URDF の自動生成・キャッシュを使わず、このパスの URDF から干渉
        ジオメトリを読み込む (``primitive_type``/``force_convert`` とは
        併用できない)。リンク名で ``robot`` 側と対応づけるので、ロボット
        本体はそのままに、干渉回避 (``collision_link_list_for_arm``) が
        使うジオメトリ・対象リンクの集合だけを差し替えられる。既定の
        自動生成モードと異なり、このモードでは指定した URDF に存在しない
        リンクは干渉回避の対象から明示的に外す (``collision_mesh`` を
        ``None`` にする)。
    """
    if collision_urdf_path is not None:
        if primitive_type is not None or force_convert:
            raise ValueError(
                'collision_urdf_path (--collision-urdf) は '
                'primitive_type (--collision-primitive-type) / '
                'force_convert (--force-convert-collision-model) と '
                '併用できません。')
    else:
        collision_urdf_path = build_collision_model_urdf(
            robot.urdf_path, primitive_type=primitive_type,
            force=force_convert)

    collision_robot = RobotModel()
    collision_robot.load_urdf_file(
        str(collision_urdf_path), include_mimic_joints=False)
    collision_links_by_name = {
        link.name: link for link in collision_robot.link_list}

    # --collision-urdf 指定時は、対応するリンクが見つからない/メッシュが
    # 無い場合に実メッシュを残さず明示的に None にして除外する。
    explicit_exclude = collision_urdf_path is not None

    n_replaced = 0
    n_excluded = 0
    for link in robot.link_list:
        collision_link = collision_links_by_name.get(link.name)
        mesh = (getattr(collision_link, 'collision_mesh', None)
               if collision_link is not None else None)
        if mesh is not None:
            link.collision_mesh = mesh
            n_replaced += 1
        elif explicit_exclude:
            link.collision_mesh = None
            n_excluded += 1
    print('[collision-model] {} 個のリンクの干渉ジオメトリを ({}) から '
          '差し替えました{}。'.format(
              n_replaced, collision_urdf_path,
              ' ({} 個のリンクを干渉回避の対象から除外)'.format(n_excluded)
              if explicit_exclude else ''))


def collision_link_list_for_arm(robot, robot_arm):
    """干渉ジオメトリを持つロボットリンクの一覧を作る。解く腕・反対側の
    腕・台車を含め、ロボットの全身 (``robot.link_list``) を対象にする
    (``robot_arm`` 引数は現状未使用のプレースホルダ)。

    ``collision_mesh`` を持たないリンクは除外する。``batch_inverse_
    kinematics`` 自体はこの関数を使わない (最適化でチェックする組み合わせ
    は常に ``collision_pairs`` だけで決まる)。``build_collision_
    verification_pairs`` (事後検証用) と ``analyze_collision_pairs.py``
    が、この関数で全リンクを集めてから総当たりの組み合わせを作る。
    """
    return [link for link in robot.link_list
           if getattr(link, 'collision_mesh', None) is not None]


def load_collision_pairs(path, robot):
    """``--collision-pairs`` (JSON) を読み、``batch_inverse_kinematics``
    の ``collision_pairs`` にそのまま渡せる ``(Link, Link)`` /
    ``(Link, int)`` のタプルのリストに変換する。

    JSON の形式は 2 要素のリストのリスト (``[[名前A, 名前B], ...]``)。
    各ペアの 1 要素目は常にロボットのリンク名。2 要素目は:

    * ロボットのリンク名なら自己干渉ペア (``(Link, Link)``) として扱う。
    * ``human_obstacle_names`` が返す人体セグメント名なら、そのセグメント
      との干渉ペア (``(Link, int)``、int はそのセグメントの
      ``collision_obstacles`` 中のインデックス) として扱う。

    ``build_collision_pairs.py`` が ``analyze_collision_pairs.py`` の出力
    からこの形式で生成する。``robot`` (``apply_collision_model`` 適用済み
    を想定) のリンク名と突き合わせ、どちらの解釈にも当てはまらない名前が
    含まれていた場合は ``ValueError`` にする。
    """
    with open(path) as f:
        pair_names = json.load(f)
    links_by_name = {link.name: link for link in robot.link_list}
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


def build_collision_verification_pairs(robot, robot_arm):
    """事後検証 (``pick_verified_candidate``/``collision_pairs_min_
    distance``) で使う、干渉しうる組み合わせを総当たりで網羅したペアの
    リストを作る (``(Link, Link)``/``(Link, int)`` 混在、``load_collision_
    pairs`` が返す形式と同じ)。

    ``--collision-pairs`` (JSON) は最適化のコストを抑えるために事前に
    絞り込んだ組み合わせだが、事後検証は絞り込む必要がないので、ここで
    作る全組み合わせに対して行う。

    * 自己干渉: ``collision_link_list_for_arm`` の全組み合わせから、
      ``create_self_collision_pairs`` (``ignore_adjacent=True``) が除く
      「隣接する (共通の関節で直接つながっている) 組み合わせ」を除いた
      もの。
    * 人体との干渉: 同じリンク一覧と ``human_obstacle_names`` の全人体
      セグメントの組み合わせ (除外なし)。

    ロボットの構造だけで決まり、人物ごとの骨格や姿勢には依存しないので、
    ``main`` から人物ループの外で 1 回だけ計算すればよい。
    """
    collision_link_list = collision_link_list_for_arm(robot, robot_arm)
    self_pairs = create_self_collision_pairs(
        collision_link_list, ignore_adjacent=True)
    pairs = [(collision_link_list[link_i], collision_link_list[link_j])
            for link_i, link_j in self_pairs]
    for link in collision_link_list:
        for obstacle_index in range(len(human_obstacle_names())):
            pairs.append((link, obstacle_index))
    return pairs


def solve_post_process(robot, robot_arm, palm, target_rot,
                       offset=POST_PROCESS_TARGET_HOVER_OFFSET,
                       stop=DEFAULT_POST_PROCESS_IK_STOP,
                       thre=DEFAULT_POST_PROCESS_IK_THRE,
                       rthre=DEFAULT_POST_PROCESS_IK_RTHRE,
                       gaze_ik_stop=DEFAULT_POST_PROCESS_GAZE_IK_STOP,
                       gaze_ik_rthre=DEFAULT_POST_PROCESS_GAZE_IK_RTHRE):
    """``pick_verified_candidate`` が干渉検証まで通した候補について、
    人間にロボットが実際に掌を押し付けられることを確認する後処理判定を
    行う (``POST_PROCESS_TARGET_HOVER_OFFSET`` 参照)。

    台車を動かさない通常のヤコビアン法 IK (干渉回避なし) で、以下の 2 つを
    ``robot.inverse_kinematics`` 1 回の呼び出しで同時に解く。

    1. 腕: 掌の目標位置を ``offset`` (既定 ``POST_PROCESS_TARGET_HOVER_
       OFFSET``、掌へわずかにめり込む位置) にした位置・姿勢
       (``target_rot`` を厳密に使う)。
    2. 首 (``robot.head.link_list``, 3 関節): ロボットの頭部エンド
       エフェクタの +Z が人間の掌 (``palm['position']``) を向くように
       (``rotation_mask='xy'``, 位置は制約しない)。

    視線の目標は人間自身の掌の位置 (IK を解く前から分かっている固定点)
    なので、腕・視線を逐次に分けず 1 回の呼び出しで同時に解ける。両タスク
    とも収束して初めて後処理判定は成功とする。

    ``robot`` は呼び出し前の姿勢 (``pick_verified_candidate`` が反映した
    候補の姿勢) を初期値として直接書き換えて解く。``revert_if_fail=True``
    なので、判定に失敗した場合 ``robot`` は呼び出し前の姿勢に戻る。

    Returns
    -------
    dict or None
        腕 IK・視線 IK のどちらも収束したときは、後処理後の手先・台車・
        関節角を持つ結果 dict (``solved_result`` と同じキー構成の一部)。
        どちらか一方でも収束しなければ ``None``。
    """
    start_time = time.time()
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    target_pos = position + normal * offset
    target_coords = Coordinates(pos=target_pos.tolist(), rot=target_rot)

    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    head_move_target = robot.head_end_coords
    head_target = Coordinates(
        pos=position.tolist()).align_axis_to_direction(
            position - head_move_target.worldpos())

    try:
        # position_mask/rotation_mask はタスクごと (腕/首) に別のマスクを
        # 使うため、normalize_mask で 3 要素配列に解決済みのリストにして
        # から渡す (そうしないと 1 つのマスク指定として誤解釈される)。
        result = robot.inverse_kinematics(
            target_coords=[target_coords, head_target],
            move_target=[move_target, head_move_target],
            link_list=[whole_body.link_list, robot.head.link_list],
            position_mask=[normalize_mask(True), normalize_mask(False)],
            rotation_mask=[normalize_mask(True), normalize_mask('xy')],
            stop=max(stop, gaze_ik_stop),
            thre=[thre, thre], rthre=[rthre, gaze_ik_rthre],
            revert_if_fail=True)
    except Exception as e:
        print('  [post-process] 押し付け/視線 IK で例外が発生したため '
              '棄却します: {}'.format(e))
        return None
    if result is False:
        return None

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


def pick_verified_candidate(robot, success_flags, angle_vectors, base_poses,
                            verification_pairs, joint_positions,
                            collision_verify_tolerance,
                            robot_arm, palm, rots,
                            attempts_per_pose=DEFAULT_ATTEMPTS_PER_POSE,
                            post_process_ik_stop=DEFAULT_POST_PROCESS_IK_STOP,
                            post_process_thre=DEFAULT_POST_PROCESS_IK_THRE,
                            post_process_rthre=DEFAULT_POST_PROCESS_IK_RTHRE):
    """``batch_inverse_kinematics`` が返した候補群 (``success_flags``/
    ``angle_vectors``/``base_poses``。全て同じ添字で対応する) の中から、
    以下を全て満たす候補を、添字最小 (最優先) のものから探して返す。

    候補は「向き (``TURN_CANDIDATES_DEG``) × 初期値 (``attempts_per_
    pose``)」の全組み合わせで、添字は向き優先の並び (``向きの添字 = 添字
    // attempts_per_pose``) になっている。優先順位は「向きが早いもの」→
    「その向きの中で初期値が早いもの」。

    1. IK が収束している (``success_flags``)。
    2. ``verification_pairs`` を実際には貫通していない
       (``collision_pairs_min_distance`` による事後検証、
       ``collision_verify_tolerance`` [m] まで許容)。
    3. 後処理判定 (``solve_post_process``) にも成功している。

    ``verification_pairs`` には最適化で使った (絞り込み済みの)
    ``collision_pairs`` ではなく、``build_collision_verification_pairs``
    が作る総当たりの組み合わせを渡す想定。

    1・2 を満たすが 3 に失敗した候補は棄却し、次の添字を試す。1・2・3 を
    全て満たす候補が見つからなかった場合のみ、1・2 を満たした最初の候補
    (``fallback``) を後処理前のまま (``post_process_result`` を ``None``
    にして) 採用するフォールバックを行う。

    候補を検証するにはロボットにその候補の姿勢を反映する必要があるため、
    検証のたびに ``robot`` を書き換える -- 呼び出し後の ``robot`` は最後に
    調べた候補の姿勢のままになる点に注意 (``solved_result``/``unsolved_
    result`` が改めて反映し直すので、最終的な姿勢を保証するのはそちら)。

    Returns
    -------
    tuple or None
        ``(turn_index, angle_vector, base_pose, post_process_result)``。
        ``turn_index`` は採用した候補の**向き**の添字 (``TURN_CANDIDATES_
        DEG``/``rots`` の添字)。``post_process_result`` は
        ``solve_post_process`` が返した後処理後の結果 dict、後処理判定に
        失敗した候補をフォールバックで採用した場合は ``None``。1・2 を
        満たす候補が 1 つも無ければ ``None`` を返す。
    """
    fallback = None
    for candidate_index, ok in enumerate(success_flags):
        if not ok:
            continue
        turn_index = candidate_index // attempts_per_pose
        attempt_index = candidate_index % attempts_per_pose
        label = 'turn={:.0f}deg/初期値 {}'.format(
            TURN_CANDIDATES_DEG[turn_index], attempt_index)
        robot.angle_vector(angle_vectors[candidate_index])
        robot.newcoords(base_poses[candidate_index])
        min_dist = collision_pairs_min_distance(
            robot, verification_pairs, joint_positions)
        if min_dist < -collision_verify_tolerance:
            print('  [collision-verify] {} の候補は IK は収束'
                  'したが、事後検証で {:.4f} m 貫通していたため棄却'
                  'します。'.format(label, min_dist))
            continue
        post_result = solve_post_process(
            robot, robot_arm, palm, rots[turn_index],
            stop=post_process_ik_stop,
            thre=post_process_thre, rthre=post_process_rthre)
        if post_result is not None:
            return (turn_index, angle_vectors[candidate_index],
                    base_poses[candidate_index], post_result)
        print('  [post-process] {} の候補は干渉検証を通過した '
              'が、後処理判定 (押し付け/視線 IK) には失敗したため、次の '
              '候補を試します。'.format(label))
        if fallback is None:
            fallback = (turn_index, angle_vectors[candidate_index],
                       base_poses[candidate_index], None)
    if fallback is not None:
        print('  [post-process] 全ての候補で後処理判定に失敗した '
              'ため、turn={:.0f}deg の候補を後処理前の解として採用 '
              'します。'.format(TURN_CANDIDATES_DEG[fallback[0]]))
    return fallback


def solve_person_ik(robot, palm, robot_arm, collision_obstacles,
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
                    post_process_thre=DEFAULT_POST_PROCESS_IK_THRE,
                    post_process_rthre=DEFAULT_POST_PROCESS_IK_RTHRE):
    """1 人分について、``TURN_CANDIDATES_DEG`` の全ての向き × 全ての初期値
    (``attempts_per_pose`` 個) を、その人の身体 (``collision_
    obstacles``) を障害物とした干渉回避付きバッチ IK でまとめて解く。
    ``self_collision=True`` (既定) のときは、ロボット自身のリンク同士の
    干渉もソフトなペナルティとして回避する。

    チェックする組み合わせは ``collision_pairs`` (``load_collision_
    pairs`` が返す形式) で明示的に指定したものだけに限る -- 「対象リンク
    の集合」は持たず、``collision_pairs`` に現れるリンクだけが計算対象に
    なる。``main`` は ``collision_pairs`` が ``None`` のときに
    ``self_collision``/``collision_obstacles`` の両方を無効にしてこの
    関数を呼ぶので、ここで全組み合わせにフォールバックすることはない。

    ``collision_ik_stop`` (最大反復回数) は、``backend='jax'`` の勾配降下
    法が収束の有無によらず毎回同じ回数だけ反復するため、実質的な
    「タイムアウト」として働く。閾値ぎりぎりまで反復してようやく収束する
    ような解を「収束しなかった」として弾く効果がある一方、絞りすぎると
    自然な姿勢も弾かれるので ``--attempts-per-pose`` とのトレードオフに
    なる。

    ``collision_obstacles`` はバッチ呼び出し全体で 1 つの集合しか渡せない
    ため、人物ごとに 1 回呼ぶ。``robot`` の腕は ``seed_arm_pose`` で種の
    姿勢にしてから呼ぶ (台車は常にワールド原点から開始する。``palm`` は
    ``main`` 側で ``translate_palm`` により、人物が Aero の前方
    ``HUMAN_FRONT_DISTANCE`` に来るよう平行移動済みのものを渡す想定)。
    バッチ IK 自体はロボットを動かさないので、戻り値は「解を反映するため
    の材料」であり、``robot`` は呼び出し後も種の姿勢のまま。

    ``success_flags`` (収束判定) は位置・姿勢誤差だけを見ており、干渉
    ペナルティが実際に解消しているかは保証しない。そのため収束した候補
    ごとに、採用する前に ``collision_pairs_min_distance`` で厳密な形状を
    使った事後検証を行い、貫通している候補は棄却する。この事後検証は
    ``collision_pairs`` (絞り込み済み) ではなく ``verification_pairs``
    (``build_collision_verification_pairs`` が作る総当たりの組み合わせ)
    に対して行う。``joint_positions`` (``main`` 側で平行移動済みのものを
    渡す想定) は人体セグメントとの干渉検証に使う (``None`` なら人体との
    干渉は検証しない)。

    ``collision_joint_limit_margin_ratio`` (既定 ``DEFAULT_COLLISION_IK_
    JOINT_LIMIT_MARGIN_RATIO``) は、この関数内の干渉回避付きバッチ IK
    だけに適用する関節可動域の上下マージン比率 (``restrict_joint_range_
    margin`` 参照)。バッチ IK 呼び出しの前後だけで適用・復元するので、
    ``pick_verified_candidate`` 以降は本来の可動域のまま使われる。

    Returns
    -------
    tuple
        ``(picked, collision_ik_time, candidate_selection_time)``。
        ``picked`` は ``(turn_index, angle_vector, base_pose,
        post_process_result)`` または ``None``
        (``pick_verified_candidate`` 参照)。``collision_ik_time`` は干渉
        回避付きバッチ IK (``batch_inverse_kinematics`` の呼び出し) 自体
        の計算時間 [秒]。``candidate_selection_time`` は
        ``pick_verified_candidate`` の呼び出し全体の計算時間 [秒]
        (``run_pipeline_test.py`` の集計で使う)。
    """
    seed_arm_pose(robot, robot_arm)
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    target_pos = palm_target_position(palm)
    rots = palm_to_target_rots(palm, robot_arm)
    target_coords = [Coordinates(pos=target_pos.tolist(), rot=rot)
                     for rot in rots]
    # collision_pairs 中の人体セグメントへの参照 (int) は
    # human_obstacle_names() の固定された並びを指しており、
    # collision_obstacles が空 (骨格 JSON が無い/--no-human-collision) の
    # ときは対応する障害物が存在しないので、そのまま渡すと
    # batch_inverse_kinematics 側でインデックス範囲外のエラーになる。
    # その場合は自己干渉ペア (Link 同士) だけを残す。
    effective_collision_pairs = collision_pairs
    effective_self_collision = self_collision
    if collision_pairs is not None and not collision_obstacles:
        effective_collision_pairs = [
            (link_a, other) for link_a, other in collision_pairs
            if not isinstance(other, int)]
        if not effective_collision_pairs:
            # collision_pairs が人体セグメントとのペアしか含んでおらず、
            # 自己干渉ペア (Link 同士) が 1 つも残らなかった場合。
            # self_collision=True のまま空リストを batch_inverse_
            # kinematics に渡すと、collision_link_list を導出できず
            # ValueError になるため、この呼び出しでは無効化する。
            effective_self_collision = False
    restore_joint_range = restrict_joint_range_margin(
        whole_body.link_list, collision_joint_limit_margin_ratio)
    collision_ik_start = time.time()
    try:
        # return_all_attempts=True: 誤差最小の解が後処理判定にも通るとは
        # 限らないため、初期値ごとの解を集約させず全部候補にする
        # (pick_verified_candidate 参照)。返る解は目標優先の並び
        # (目標 0 の初期値 0..N-1, 目標 1 の初期値 0..N-1, ...)。
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
                collision_obstacles=collision_obstacles,
                collision_weight=collision_weight,
                collision_margin=collision_margin,
                self_collision=effective_self_collision,
                collision_pairs=effective_collision_pairs,
                self_collision_weight=self_collision_weight,
                self_collision_margin=self_collision_margin)
    finally:
        restore_joint_range()
    collision_ik_time = time.time() - collision_ik_start
    # verification_pairs 中の人体セグメントへの参照 (int) も、
    # effective_collision_pairs と同じ理由で除く。
    effective_verification_pairs = verification_pairs
    if verification_pairs is not None and not collision_obstacles:
        effective_verification_pairs = [
            (link_a, other) for link_a, other in verification_pairs
            if not isinstance(other, int)]
    candidate_selection_start = time.time()
    picked = pick_verified_candidate(
        robot, success_flags, angle_vectors, base_poses,
        effective_verification_pairs, joint_positions,
        collision_verify_tolerance, robot_arm, palm, rots,
        attempts_per_pose=attempts_per_pose,
        post_process_thre=post_process_thre,
        post_process_rthre=post_process_rthre)
    candidate_selection_time = time.time() - candidate_selection_start
    return picked, collision_ik_time, candidate_selection_time


def base_movable_region(base_limits):
    """バッチ IK に渡した ``base_limits`` (台車の IK 開始位置を原点とした
    [x, y, yaw] の (下限, 上限)) を、そのままワールド座標の範囲として
    dict にまとめる (台車は常にワールド原点から IK を開始するため)。
    ``view_handshake_poses.py`` がこの範囲を台車の可動域として可視化する。
    """
    x_range, y_range, yaw_range = base_limits
    return dict(
        x_range=[float(x_range[0]), float(x_range[1])],
        y_range=[float(y_range[0]), float(y_range[1])],
        yaw_range=[float(yaw_range[0]), float(yaw_range[1])],
    )


def solved_result(robot, robot_arm, target_pos, target_rot, turn_index,
                  angle_vector, base_pose, base_limits, post_process_result,
                  collision_ik_time, candidate_selection_time):
    """採用した解をロボットに反映し、結果 dict を組む.

    バッチ IK はロボットを動かさないので、``angle_vector`` と
    ``base_pose`` を実際に反映してから手先・台車の姿勢を読み直す。

    ``post_process_result`` (``solve_post_process`` が返した後処理後の
    結果 dict) は、この関数が組んだ「後処理前」の位置姿勢と区別できる
    よう ``post_process`` キーにそのまま格納する。``view_handshake_
    poses.py`` はこの 2 つを切り替えて表示する。

    ``collision_ik_time``/``candidate_selection_time`` は
    ``solve_person_ik`` が計測した計算時間 [秒] をそのまま格納する
    (``run_pipeline_test.py`` が集計に使う)。
    """
    robot.angle_vector(angle_vector)
    robot.newcoords(base_pose)
    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    result = dict(
        target=True,
        solved=True,
        turn_deg=TURN_CANDIDATES_DEG[turn_index],
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
                    collision_ik_time, candidate_selection_time):
    """どの候補も解けなかった人物のための結果 dict.

    種の姿勢 (台車は ``solve_person_ik`` と同じくワールド原点) を反映して
    から手先・台車の姿勢を読む。``turn_deg``/``target_rot`` は最後に試した
    候補のものにする。
    """
    seed_arm_pose(robot, robot_arm)
    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    return dict(
        target=True,
        solved=False,
        turn_deg=TURN_CANDIDATES_DEG[-1],
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
    """IK の対象外だった人物のための結果 dict.

    IK は解かないので関節角・台車位置は持たず、``solved`` は ``False``、
    ``target`` が ``False`` になる。``view_handshake_poses.py`` は
    ``target`` を見て「対象外」と表示する。

    Parameters
    ----------
    offered_hand : str or None
        掌 JSON の ``offered_hand`` (対象外なので通常は ``None``)。
    reason : str
        対象外にした理由 (``'no_offered_hand'`` / ``'no_palm'``)。
    """
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


def main():
    parser = argparse.ArgumentParser(
        description='掌の位置姿勢 JSON (estimate_palm_poses.py の出力) を '
                    '入力とし、ベース移動型ロボットが全身 IK をバッチで '
                    '解いて手を繋ぐ姿勢を求め、JSON として保存する。IK を '
                    '解くのは掌推定が「手を差し出している」と判定した '
                    '(offered_hand が L/R の) 人物だけで、null の人物には '
                    'IK の結果を持たない JSON (target: false) を書き出す。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_palm_poses'),
        help='掌の位置姿勢 JSON の入力ディレクトリ (既定は '
            'estimate_palm_poses.py の既定の出力先と同じ '
            'random_palm_poses/)。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='IK の結果 JSON の保存先ディレクトリ (既定は '
            'random_handshake_poses/。入力と同じファイル名で保存する)。')
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='使うロボットの腕。既定 (auto) は人間の手の反対側。')
    parser.add_argument(
        '--attempts-per-pose', type=int,
        default=DEFAULT_ATTEMPTS_PER_POSE,
        help='1 つの目標姿勢に対して振る初期値の数。増やすと解ける人物が '
            '増えるが遅くなる (既定 {})。'.format(DEFAULT_ATTEMPTS_PER_POSE))
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='人体の全身関節位置を持つ骨格 JSON (generate_random_human_'
            'poses.py の出力, --input-dir と同じファイル名で対応させる) '
            'のディレクトリ (既定 random_human_poses/)。干渉回避の障害物 '
            '(この人物の身体) を作るのに使う。')
    parser.add_argument(
        '--human-front-distance', type=float,
        default=HUMAN_FRONT_DISTANCE,
        help='Aero の前方どれだけの位置に人物を置くか [m] (既定 {:.1f})。'
            .format(HUMAN_FRONT_DISTANCE))
    parser.add_argument(
        '--collision-weight', type=float,
        default=DEFAULT_COLLISION_WEIGHT,
        help='人体との干渉回避ペナルティの重み (既定 {})。'.format(
            DEFAULT_COLLISION_WEIGHT))
    parser.add_argument(
        '--collision-margin', type=float,
        default=DEFAULT_COLLISION_MARGIN,
        help='人体との干渉回避ペナルティが働き始める距離 [m] '
            '(既定 {})。'.format(DEFAULT_COLLISION_MARGIN))
    parser.add_argument(
        '--no-self-collision', dest='self_collision', action='store_false',
        help='ロボット自身のリンク同士の干渉を回避するペナルティを無効に '
            'する (既定は有効。--collision-pairs が使えない場合はどのみち '
            '無効になる)。')
    parser.add_argument(
        '--collision-pairs', type=str,
        default=os.path.join(_THIS_DIR, 'collision_pairs.json'),
        help='干渉回避で実際にチェックする組み合わせ (自己干渉のロボット '
            'リンク同士、および人体との干渉のロボットリンク×人体セグメント) '
            'を指定する JSON (2 要素の名前のリストのリスト。既定 '
            'collision_pairs.json。build_collision_pairs.py が生成する)。 '
            '既定のパスにファイルが無ければ、自己干渉・人体との干渉の両方 '
            'を無効にして通常のヤコビアン法の IK を解く。')
    parser.add_argument(
        '--no-human-collision', action='store_true',
        help='人体を障害物とした干渉回避を無効にする (既定は --skeleton-dir '
            'に骨格 JSON があれば有効)。人物の立ち位置の平行移動は従来 '
            'どおり働く。')
    parser.add_argument(
        '--self-collision-weight', type=float, default=None,
        help='自己干渉回避ペナルティの重み (既定は --collision-weight と '
            '同じ値を使う)。')
    parser.add_argument(
        '--self-collision-margin', type=float,
        default=DEFAULT_SELF_COLLISION_MARGIN,
        help='自己干渉回避ペナルティが働き始めるリンク間距離 [m] '
            '(既定 {})。'.format(DEFAULT_SELF_COLLISION_MARGIN))
    parser.add_argument(
        '--collision-ik-stop', type=int,
        default=DEFAULT_COLLISION_IK_STOP,
        help='干渉回避付きバッチ IK (backend=jax の勾配降下法) の最大反復 '
            '回数 (既定 {})。'.format(DEFAULT_COLLISION_IK_STOP))
    parser.add_argument(
        '--collision-ik-thre', type=float,
        default=DEFAULT_COLLISION_IK_THRE,
        help='干渉回避付きバッチ IK の位置収束閾値 [m] (既定 {})。'.format(
            DEFAULT_COLLISION_IK_THRE))
    parser.add_argument(
        '--collision-ik-rthre', type=float,
        default=DEFAULT_COLLISION_IK_RTHRE,
        help='干渉回避付きバッチ IK の姿勢収束閾値 [rad] (既定 {:.4f})。'
            .format(DEFAULT_COLLISION_IK_RTHRE))
    parser.add_argument(
        '--collision-joint-limit-margin', type=float,
        default=DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO,
        help='干渉回避付きバッチ IK だけに適用する関節可動域の上下 '
            'マージン比率 (既定 {})。0 を指定すると制限しない。'.format(
                DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO))
    parser.add_argument(
        '--collision-verify-tolerance', type=float,
        default=DEFAULT_COLLISION_VERIFY_TOLERANCE,
        help='収束した候補の事後の干渉検証 (collision_pairs_min_distance) '
            'で、この距離 [m] を超えて貫通していれば棄却する (既定 {})。'
            .format(DEFAULT_COLLISION_VERIFY_TOLERANCE))
    parser.add_argument(
        '--post-process-thre', type=float,
        default=DEFAULT_POST_PROCESS_IK_THRE,
        help='後処理判定 (solve_post_process) の腕タスクの位置収束閾値 '
            '[m] (既定 {})。視線タスクの閾値には影響しない。'.format(
                DEFAULT_POST_PROCESS_IK_THRE))
    parser.add_argument(
        '--post-process-rthre', type=float,
        default=math.degrees(DEFAULT_POST_PROCESS_IK_RTHRE),
        help='後処理判定 (solve_post_process) の腕タスクの姿勢収束閾値 '
            '[deg] (既定 {:.1f})。視線タスクの閾値には影響しない。'
            .format(math.degrees(DEFAULT_POST_PROCESS_IK_RTHRE)))
    parser.add_argument(
        '--base-x-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_X_RANGE),
        help='台車の前後方向 (x) の移動範囲 [m]。IK 開始時の台車位置を '
            '原点とする (既定 {} {})。'.format(*DEFAULT_BASE_X_RANGE))
    parser.add_argument(
        '--base-y-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_Y_RANGE),
        help='台車の左右方向 (y) の移動範囲 [m] (既定 {} {})。'.format(
            *DEFAULT_BASE_Y_RANGE))
    parser.add_argument(
        '--base-yaw-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_YAW_RANGE),
        help='台車の向き (yaw) の範囲 [rad] (既定 {:.4f} {:.4f})。'.format(
            *DEFAULT_BASE_YAW_RANGE))
    parser.add_argument(
        '--seed', type=int, default=None,
        help='バッチ IK の乱数初期値に使う numpy の乱数シード。指定すると '
            '実行ごとに同じ解が得られる (既定は指定なし)。')
    parser.add_argument(
        '--collision-primitive-type', choices=['box', 'cylinder', 'sphere'],
        default=None,
        help='干渉回避に使うロボット自身のジオメトリを、指定した形状に '
            '全リンク強制変換する (既定 (未指定) はリンクごとに自動選択)。')
    parser.add_argument(
        '--force-convert-collision-model', action='store_true',
        help='ロボット自身の干渉モデル (プリミティブ近似 URDF) のキャッシュ '
            'を使わず毎回作り直す。')
    parser.add_argument(
        '--collision-urdf', type=str, default=None,
        help='干渉回避のジオメトリ・対象リンクの読み込み元 URDF を明示的に '
            '指定する (既定は自動生成・キャッシュされるプリミティブ近似 '
            'URDF)。--collision-primitive-type/--force-convert-collision-'
            'model とは併用できない。IK を解くロボット本体のキネマティクス '
            'は変えず、干渉回避に使うリンクの集合・ジオメトリだけを '
            '(リンク名で対応づけて) 差し替える。')
    args = parser.parse_args()

    files = iter_palm_files(args.input_dir)
    if not files:
        print('{} に掌の位置姿勢 JSON が見つかりません。先に '
              'estimate_palm_poses.py を実行してください。'.format(
                  args.input_dir))
        return

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    # r/l_eef_grasp_link (IK が使う手先フレーム) は手あり/なし両方の URDF に
    # あるので、指の関節が要らないこのスクリプトでは手なしモデルを使う。
    robot = Aero(use_hand=False)
    restrict_elbow_range(robot)
    apply_collision_model(
        robot,
        primitive_type=args.collision_primitive_type,
        force_convert=args.force_convert_collision_model,
        collision_urdf_path=args.collision_urdf)

    # 干渉回避で実際にチェックする組み合わせは常に collision_pairs
    # (--collision-pairs の JSON) だけで決める。JSON が無ければ干渉回避
    # そのものを無効にして、通常のヤコビアン法の IK にフォールバックする。
    collision_pairs = None
    verification_pairs = None
    if os.path.exists(args.collision_pairs):
        collision_pairs = load_collision_pairs(args.collision_pairs, robot)
        print('[collision-pairs] {} 組の干渉ペアを {} から読み込みました。'
              .format(len(collision_pairs), args.collision_pairs))
        # 事後検証は collision_pairs (絞り込み済み) ではなく、隣接リンクの
        # 組み合わせだけを除いた総当たりの組み合わせに対して行う。ロボット
        # の構造だけで決まるので人物ループの外で 1 回だけ計算する
        # (robot_arm 引数は結果に影響しないプレースホルダ)。
        verification_pairs = build_collision_verification_pairs(robot, 'r')
    else:
        print('[collision-pairs] {} が見つからないため、干渉回避 (自己干渉'
              '・人体との干渉の両方) を無効にして IK を解きます。'.format(
                  args.collision_pairs))

    base_limits = [tuple(args.base_x_range), tuple(args.base_y_range),
                   tuple(args.base_yaw_range)]

    n_solved = 0
    n_total = 0
    n_not_target = 0
    for i, path in enumerate(files):
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
        robot_arm = (DEFAULT_ROBOT_ARM[human_hand]
                     if args.robot_arm == 'auto' else args.robot_arm)

        # この人物の身体を干渉回避の障害物にする。骨格 JSON が
        # --skeleton-dir に無い、または collision_pairs が使えないときは、
        # 人体との干渉回避なしのバッチ IK にフォールバックする。
        skeleton_path = os.path.join(args.skeleton_dir,
                                     os.path.basename(path))
        # 骨格 JSON があれば、人物の立ち位置がちょうど Aero の前方
        # --human-front-distance になるよう、骨格の全関節位置と掌の目標
        # 位置を平行移動してから IK を解く (干渉回避の有無に関わらず常に
        # 行う)。骨格 JSON が無ければ平行移動できないので、掌の位置は
        # そのまま使う。
        if os.path.exists(skeleton_path):
            joint_positions = load_skeleton_json(skeleton_path)
            offset = human_translation_offset(
                joint_positions, front_distance=args.human_front_distance)
            joint_positions = translate_joint_positions(
                joint_positions, offset)
            collision_obstacles = (
                [] if (args.no_human_collision or collision_pairs is None)
                else human_body_obstacles(joint_positions))
            palm = translate_palm(palm, offset)
        else:
            print('  {} に骨格 JSON が無いため、この人物は人体との干渉回避 '
                  'なしで解きます。'.format(skeleton_path))
            collision_obstacles = []
            joint_positions = None

        target_pos = palm_target_position(palm)
        rots = palm_to_target_rots(palm, robot_arm)
        picked, collision_ik_time, candidate_selection_time = solve_person_ik(
            robot, palm, robot_arm, collision_obstacles,
            attempts_per_pose=args.attempts_per_pose,
            base_limits=base_limits,
            collision_weight=args.collision_weight,
            collision_margin=args.collision_margin,
            self_collision=(args.self_collision and collision_pairs is not None),
            collision_pairs=collision_pairs,
            self_collision_weight=args.self_collision_weight,
            self_collision_margin=args.self_collision_margin,
            collision_ik_stop=args.collision_ik_stop,
            collision_ik_thre=args.collision_ik_thre,
            collision_ik_rthre=args.collision_ik_rthre,
            collision_joint_limit_margin_ratio=(
                args.collision_joint_limit_margin),
            joint_positions=joint_positions,
            verification_pairs=verification_pairs,
            collision_verify_tolerance=args.collision_verify_tolerance,
            post_process_thre=args.post_process_thre,
            post_process_rthre=math.radians(args.post_process_rthre))
        if picked is None:
            result = unsolved_result(
                robot, robot_arm, target_pos, rots[-1], base_limits,
                collision_ik_time, candidate_selection_time)
        else:
            turn_index, angle_vector, base_pose, post_process_result \
                = picked
            result = solved_result(
                robot, robot_arm, target_pos, rots[turn_index],
                turn_index, angle_vector, base_pose,
                base_limits, post_process_result, collision_ik_time,
                candidate_selection_time)
        result['offered_hand'] = human_hand
        result['robot_arm'] = robot_arm
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
