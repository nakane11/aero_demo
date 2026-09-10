#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""合成骨格 (``generate_random_human_poses.py``) の代わりに、実カメラ +
``PeoplePoseEstimator`` (MediaPipe) で推定した骨格に対して掌推定・IK・
軌道計画までのパイプラインを試すための ROS ノード。

``run_pipeline_test.py`` が subprocess + JSON ファイル経由で「多数の人物を
まとめて処理」するのに対し、実カメラは 1 フレームにつき 0〜1 人しか
検出できず、人物もロボットに対して任意の位置に立つ。そのためこのノードは

* 検出した骨格 (base_link 座標系) と Aero のロボットモデルを scikit-robot
  の viser ビューアに重ねて常時プレビュー表示し、位置関係を確認できる
  ようにする (骨格は線を重ねて描くだけ、ARMED でない限り IK は解かない)
* viser 画面の ``ARM`` ボタンを押すと ``ARMED`` 状態になり、以後の
  フレームで毎回掌推定をやり直し続け、``offered_hand`` (差し出し手) が
  決まった瞬間の骨格でその 1 人分だけ IK を解く
* IK が解けたら続けて ``plan_handshake_motion.py`` と同じ要領で、
  ロボットの初期姿勢 (腕を下ろし、最終台車位置から人間の反対方向へ
  ``--approach-distance`` 下がった位置) から握手姿勢へ至る干渉回避付きの
  軌道 (waypoint 列) を計画する
* ``--armed-timeout`` 秒たっても決まらなければ諦めて ``IDLE`` に戻る

IK 自体は指なしロボット (``self.robot``) で解く (指を含めると自己干渉
ペアの組み合わせが無駄に増えるため)。そのため画面の状態表示では、
``view_handshake_poses.py`` と同様に、指ありモデル (``self.display_robot``
と同じ形状の overlay) で事後検証を別に行い、指先まで含めて実際に貫通して
いる組み合わせ (自己干渉・人体との干渉) をテキストパネルに出す
(``colliding_link_pairs``/``collision_pairs_text`` 参照)。IK の探索自体が
指先まで考慮するわけではないことに注意。

という「ボタンを押すと1人分やる」形の対話的なテストを行う。viser 画面は
``view_handshake_motion.py`` と同様に、計画した軌道を waypoint スライダー/
Play ボタンで初期姿勢から握手姿勢まで確認できる。viser はブラウザで表示
するビューアなので、実行するとブラウザが開く (WSLg 環境などでは自動で
開く)。ブラウザが自動で開かない場合は、標準出力に表示される URL を手動で
開くこと。

``estimate_palm_poses.py``/``solve_palm_ik.py``/``plan_handshake_motion.py``
の関数・クラスをそのまま import して使う (骨格の入力形式は
``PeoplePoseEstimator.estimate_3d`` が返す ``{limb_name: [x, y, z]}`` の
dict で、合成骨格の ``skeleton.joint_positions`` と同じ形なので、生成元
による処理の違いは無い)。``plan_handshake_motion.py`` は ``jaxls``
(``pip install "git+https://github.com/brentyi/jaxls.git"``) が別途必要。

実カメラ特有の 2 つの問題への対策も入れてある。

* 深度が単発で背景側に飛ぶ (``aero_demo.skeleton_filters.OneEuroFilter``):
  関節位置に One Euro Filter (Casiez et al. 2012) をかけて時間方向に
  平滑化してから使う。``scripts/record_skeleton_data.py`` で録った実データ
  を ``scripts/filter_skeleton_data.py`` で分析した結果、単純な移動中央値
  (旧 ``_JointSmoother``) よりも跳びを抑えつつ追従の遅れが小さかったため
  採用した (``--joint-smoothing-mincutoff``/``--joint-smoothing-beta``、
  既定はそのときに良かった設定)。
* ARM を押しても差し出し手が見つからない: ``OfferedHandSelector`` は
  合成骨格向けにスコア閾値 (``--offer-score-min``, 既定は ``estimate_
  palm_poses.OFFER_SCORE_MIN``) が調整されているため、実カメラの姿勢では
  届きにくいことがある。ARMED 中は viser 画面 (と標準出力) に左右の
  スコア/判定不可の理由 (``no_palm``: 手のランドマークが取れていない、
  等) を表示するので、それを見ながら閾値を調整する。

Usage
-----
    python3 scripts/ros/run_camera_pipeline_test.py
    python3 scripts/ros/run_camera_pipeline_test.py --save-dir /tmp/camera_handshake_poses
"""

import argparse
import copy
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

import rospy
import message_filters
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image

# このファイルは ROS 依存プログラムをまとめた scripts/ros/ の下にあるので、
# ROS 非依存の scripts/ (estimate_palm_poses.py/solve_palm_ik.py がある) は
# 1 つ上の階層になる。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
_PKG_SRC_DIR = os.path.join(_SCRIPTS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import json_io  # noqa: E402
from aero_demo import palm_plane_view  # noqa: E402
from aero_demo import viewer_nav  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.people_pose_types import Bone  # noqa: E402
from aero_demo import skeleton_filters  # noqa: E402

import estimate_palm_poses as epp  # noqa: E402
import solve_palm_ik as spik  # noqa: E402
import plan_handshake_motion as phm  # noqa: E402
from handshake_viewer_common import HUMAN_COLLISION_OBSTACLE_COLOR  # noqa: E402
from handshake_viewer_common import apply_robot_pose as apply_result_pose  # noqa: E402,E501
from handshake_viewer_common import apply_waypoint_pose  # noqa: E402
from handshake_viewer_common import build_display_waypoints  # noqa: E402
from handshake_viewer_common import build_robot_collision_overlay  # noqa: E402
from handshake_viewer_common import colliding_link_pairs  # noqa: E402
from handshake_viewer_common import collision_pairs_text as common_collision_pairs_text  # noqa: E402,E501
from handshake_viewer_common import set_link_visible as common_set_link_visible  # noqa: E402,E501
from handshake_viewer_common import sync_robot_collision_overlay  # noqa: E402
from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

# ロボットの初期位置 (台車がワールド原点にいる姿勢) を示す Axis の大きさ
# [m]。view_handshake_poses.py の TARGET_AXIS_LENGTH と同程度の、グリッド
# 上で目立つ大きさにしてある。
INITIAL_POSE_AXIS_LENGTH = 0.2
INITIAL_POSE_AXIS_RADIUS = 0.008

# waypoint 自動再生 (Play チェックボックス) の既定の速さ [waypoint/秒]
# (view_handshake_motion.DEFAULT_PLAYBACK_FPS と同じ)。
DEFAULT_PLAYBACK_FPS = 20.0

# ロボット自身/人体側の干渉回避ジオメトリを重ねて表示する色、経路の後処理
# 補間フレーム数は view_handshake_poses.py/view_handshake_motion.py と共通
# なので handshake_viewer_common.py に一本化してある
# (ROBOT_COLLISION_LINK_COLOR/HUMAN_COLLISION_OBSTACLE_COLOR/
# PRESS_IN_DISPLAY_WAYPOINTS)。

# 経路の先頭に表示専用で追加する、ロボットの初期位置 (台車=ワールド原点,
# 関節=reset_pose) から経路計算の始点 (motion['waypoints'][0]、腕を下ろし
# 接近開始位置まで台車が下がった姿勢) までの補間フレーム数。
# plan_handshake_motion.py が計画するのは経路計算の始点から先だけなので、
# それより手前 (ロボットが実際にどこからその場所まで来るか) は干渉を
# 考慮せず単純な線形補間で表示するだけにする。
INITIAL_APPROACH_DISPLAY_WAYPOINTS = 10

# 骨格が一瞬未検出になるたびに viewer 画面の骨格表示を消して描き直すと
# ちらついて見づらいため、検出が途切れてもこの秒数の間は直前に検出できた
# 骨格をそのまま表示し続け、この秒数を超えて未検出が続いたときだけ消す
# (spin 参照)。ARMED 中に固定表示される骨格 (_frozen_joint_positions) には
# 適用しない (そちらは RESET されるまで意図的に固定表示するため)。
SKELETON_HOLD_TIMEOUT = 1.0

# 骨格の再描画周期 [秒]。認識自体 (PeoplePoseEstimator.estimate_3d) は
# カメラの frame rate のまま行うが、認識中は関節位置が毎フレーム微妙に
# 変わり続けるため、viewer への delete/add をそのフレームレートのまま
# 行うとちらつきが残る。認識周期とは独立に、この秒数間隔でのみ実際の
# 再描画 (delete/add) を行うことでちらつきを抑える (spin 参照)。ただし
# 骨格が現れる/消える (None との切り替わり) や ARMED 固定表示への切り替え
# などの状態変化は間引かず即座に反映する。
SKELETON_REDRAW_INTERVAL = 0.5

# 骨格の関節同士のつながり (関節名のペア)。draw_random_human_poses.py の
# BONE_NAME_PAIRS と同じ (PeoplePoseEstimator.limb_sequence/index2limbname
# と同じ骨格のつながり)。scripts/ros/ 層はこのファイル単独で完結させたい
# ので複製してある (draw_random_human_poses.py の module docstring にある
# 複製方針と同じ)。
BODY_BONE_PAIRS = [
    ('Neck', 'Nose'), ('Nose', 'LEye'), ('Nose', 'REye'),
    ('LShoulder', 'LEar'), ('RShoulder', 'REar'),
    ('Neck', 'RShoulder'), ('Neck', 'LShoulder'),
    ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('Neck', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
    ('Neck', 'LHip'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('REye', 'REar'), ('LEye', 'LEar'),
]
# 手のランドマーク (MediaPipe の並び) 同士のつながり。
# PeoplePoseEstimator.hand_sequence と同じ。
HAND_SEQUENCE = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
# 腕の手首と手ランドマーク index 0 (手首) の接続を追加
HAND_WRIST_PAIRS = [('RWrist', 'RHand0'), ('LWrist', 'LHand0')]
BONE_NAME_PAIRS = BODY_BONE_PAIRS + HAND_WRIST_PAIRS + [
    ('{}Hand{}'.format(side, a), '{}Hand{}'.format(side, b))
    for side in ('R', 'L') for a, b in HAND_SEQUENCE]


def _fill_missing_wrist_from_hand(positions):
    """手首 (``RWrist``/``LWrist``) が未検出でも、Hand モデルの手首
    ランドマーク (``RHand0``/``LHand0``) が検出できていればその位置を
    手首として補って返す (辞書のコピー、``positions`` 自体は書き換えない)。

    Pose モデルの手首 (``RWrist``/``LWrist``) と Hand モデルの手首
    (``RHand0``/``LHand0``) は別々に検出されるランドマークなので
    (``PeoplePoseEstimator._prune_implausible_hand_wrist_offset`` 参照)、
    人にカメラから見て手が体の陰に隠れる等で Pose 側の手首だけ未検出に
    なっても Hand 側は検出できていることがある。これを補わずに描画すると
    ``RElbow``-``RWrist`` と ``RWrist``-``RHand0`` のどちらのボーンも
    (``RWrist`` が無いので) 引けず、手のランドマークだけが肘から浮いて見え
    (肘から先が骨格線として繋がって見えない) てしまう。
    """
    filled = dict(positions)
    for wrist_name, hand_wrist_name in (('RWrist', 'RHand0'),
                                        ('LWrist', 'LHand0')):
        if wrist_name not in filled and hand_wrist_name in filled:
            filled[wrist_name] = filled[hand_wrist_name]
    return filled


def build_skeleton_links(joint_positions):
    """骨格を部位ごとに色分けした線 (``skrobot.model.primitives.
    LineString``) のリストにする。

    ``draw_random_human_poses.build_skeleton_links`` と同じ
    ``palm_plane_view.bone_line``/``bone_color`` を使うので、見た目
    (部位ごとの色) も同じになる。欠損した関節の補間や SMPL メッシュの
    表示は行わない (``_fill_missing_wrist_from_hand`` による手首の補完を
    除く) 。実際に検出できた関節だけを線でつなぐ。

    Parameters
    ----------
    joint_positions : dict
        関節名 -> ``np.ndarray([x, y, z])`` (base_link 座標系)。
        ``PeoplePoseEstimator.estimate_3d`` が返す形式。
    """
    joint_positions = _fill_missing_wrist_from_hand(joint_positions)
    links = []
    for start_name, end_name in BONE_NAME_PAIRS:
        if start_name not in joint_positions or end_name not in joint_positions:
            continue
        bone = Bone(name='{}->{}'.format(start_name, end_name),
                   start_point=joint_positions[start_name],
                   end_point=joint_positions[end_name])
        color = palm_plane_view.bone_color(bone.name)
        links.append(palm_plane_view.bone_line(bone, color))
    return links


# apply_result_pose (handshake_viewer_common.apply_robot_pose)/
# apply_waypoint_pose/build_robot_collision_overlay/
# sync_robot_collision_overlay/colliding_link_pairs/build_display_waypoints
# は view_handshake_poses.py/view_handshake_motion.py と共通なので
# handshake_viewer_common.py に一本化してある (モジュール先頭で import
# 済み)。collision_pairs_text だけは、IK 自体は指なしで解いているのに
# 画面表示は指先まで含めた事後検証であることが分かるよう、見出しを
# 変えたラッパー (下の collision_pairs_text) をこのファイルに残す。


def collision_pairs_text(colliding):
    """``handshake_viewer_common.collision_pairs_text`` に、IK 自体は
    指なしロボットで解いているが画面表示は指先まで含めた事後検証で
    あることを示す見出しを付けて呼ぶ (モジュール docstring 参照)。"""
    return common_collision_pairs_text(colliding, label='指先まで含めた事後検証')


def build_initial_approach_waypoints(initial_base_position, initial_base_yaw,
                                     initial_joint_names,
                                     initial_joint_angle_vector,
                                     first_waypoint, motion_joint_names):
    """ロボットの初期位置 (``initial_base_position``/``initial_base_yaw``、
    関節角 ``initial_joint_angle_vector``) から、経路計画上の始点
    ``first_waypoint`` (``motion['waypoints'][0]``、腕を下ろし接近開始位置
    まで台車が下がった姿勢) までを線形補間した、表示専用の waypoint 列を
    返す。

    ``plan_handshake_motion.py`` が干渉回避付きで計画するのは
    ``first_waypoint`` から先 (接近開始位置 -> 握手姿勢) だけで、それより
    手前 (ロボットの初期位置から接近開始位置まで) は計画対象外 -- 実機では
    ナビゲーションが別途担当する区間 (モジュール docstring 参照)。ここでは
    干渉は考慮せず、台車位置姿勢・関節角をそれぞれ単純に線形補間するだけの
    表示アニメーションにする (``build_display_waypoints`` の後処理補間と
    同じ考え方)。

    戻り値の末尾は ``first_waypoint`` 自身を含まない (呼び出し側で
    ``motion['waypoints']`` をそのまま続ける前提)。
    """
    start_name_to_angle = dict(zip(initial_joint_names,
                                   initial_joint_angle_vector))
    start_vec = np.array(
        [start_name_to_angle.get(name, 0.0) for name in motion_joint_names],
        dtype=np.float64)
    end_vec = np.asarray(first_waypoint['joint_angle_vector'],
                         dtype=np.float64)
    base_start = np.array([initial_base_position[0], initial_base_position[1],
                           initial_base_yaw])
    base_end = np.array([first_waypoint['base_position'][0],
                         first_waypoint['base_position'][1],
                         first_waypoint['base_yaw']])

    waypoints = []
    for t in np.linspace(0.0, 1.0, INITIAL_APPROACH_DISPLAY_WAYPOINTS,
                         endpoint=False):
        angle_vec = start_vec + (end_vec - start_vec) * t
        base_vec = base_start + (base_end - base_start) * t
        waypoints.append(dict(
            base_position=[float(base_vec[0]), float(base_vec[1]), 0.0],
            base_yaw=float(base_vec[2]),
            joint_angle_vector=[float(v) for v in angle_vec],
        ))
    return waypoints


def _format_offer_scores(selection, score_min):
    """``OfferedHandSelector.select`` の戻り値を viser 画面に出す文字列にする.

    ``ARM`` ボタンを押しても差し出し手が決まらないとき、原因が「そもそも
    手のランドマークが取れていない (``veto``: ``no_palm``)」のか「取れて
    いるがスコアが閾値 ``score_min`` に届いていない」のかを見分けられる
    ようにする。
    """
    lines = ['**差し出し手判定 (閾値 {:.2f}):**'.format(score_min)]
    for side in ('R', 'L'):
        veto = selection['veto'][side]
        score = selection['scores'][side]
        if veto is not None:
            lines.append('- {}: 判定不可 ({})'.format(side, veto))
        else:
            lines.append('- {}: {:.2f}'.format(side, score))
    return '\n'.join(lines)


def _transform_to_matrix(transform):
    """``geometry_msgs/Transform`` を 4x4 の同次変換行列にする."""
    t = transform.translation
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


# sensor_msgs/Image -> numpy 変換用の dtype/チャンネル数テーブル。
# cv_bridge はシステム (apt) 由来のバイナリで、ビルド時の NumPy 1.x の
# C-API を静的に埋め込んでいるため NumPy 2.x 実行時に ImportError/
# AttributeError (_ARRAY_API not found) を起こす。ここで使うのは
# bgr8/rgb8/mono8/16UC1/32FC1 だけなので、cv_bridge に頼らず
# Image.data を直接 numpy 配列に変換する。
_IMGMSG_DTYPE_CHANNELS = {
    'bgr8': (np.uint8, 3),
    'rgb8': (np.uint8, 3),
    'mono8': (np.uint8, 1),
    '8UC1': (np.uint8, 1),
    '16UC1': (np.uint16, 1),
    '32FC1': (np.float32, 1),
}


def _imgmsg_to_ndarray(msg, desired_encoding=None):
    """``sensor_msgs/Image`` を numpy 配列へ変換する (cv_bridge の代替)."""
    if msg.encoding not in _IMGMSG_DTYPE_CHANNELS:
        raise ValueError('Unsupported image encoding: {}'.format(msg.encoding))
    dtype, channels = _IMGMSG_DTYPE_CHANNELS[msg.encoding]
    dtype = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<')
    arr = np.frombuffer(msg.data, dtype=dtype)
    shape = (msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width)
    arr = arr.reshape(shape)

    if desired_encoding is not None and desired_encoding != msg.encoding:
        if {desired_encoding, msg.encoding} == {'bgr8', 'rgb8'}:
            arr = arr[..., ::-1]
        else:
            raise ValueError(
                'Cannot convert image encoding {} -> {}'.format(
                    msg.encoding, desired_encoding))
    return np.ascontiguousarray(arr)


def _ndarray_to_imgmsg(arr, encoding, header):
    """numpy 配列を ``sensor_msgs/Image`` に変換する
    (``_imgmsg_to_ndarray`` の逆、cv_bridge の代替)."""
    dtype, channels = _IMGMSG_DTYPE_CHANNELS[encoding]
    arr = np.ascontiguousarray(arr, dtype=dtype)
    msg = Image()
    msg.header = header
    msg.height, msg.width = arr.shape[0], arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * channels * np.dtype(dtype).itemsize
    msg.data = arr.tobytes()
    return msg


def draw_skeleton_overlay(color_bgr, joints_2d):
    """カメラ画像 (BGR) に、検出できた 2D 関節位置を重ねて描いた画像を
    返す (デバッグ用の publish 専用、元の ``color_bgr`` は書き換えない)。

    ``joints_2d`` は ``PeoplePoseEstimator.estimate``/``estimate_3d`` が
    返す 1 人分の ``[{"limb": str, "x": float, "y": float, "score": float},
    ...]`` (画像座標、score < 0 は未検出)。viser の 3D 骨格表示
    (``build_skeleton_links``) と同じ ``BONE_NAME_PAIRS``/
    ``palm_plane_view.bone_color`` を使うので、部位ごとの色も揃う。
    """
    overlay = color_bgr.copy()
    positions = {j['limb']: (int(round(j['x'])), int(round(j['y'])))
                for j in joints_2d if j['score'] >= 0}
    positions = _fill_missing_wrist_from_hand(positions)
    for start_name, end_name in BONE_NAME_PAIRS:
        if start_name not in positions or end_name not in positions:
            continue
        color = palm_plane_view.bone_color(
            '{}->{}'.format(start_name, end_name))
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.line(overlay, positions[start_name], positions[end_name],
                 bgr, 2, cv2.LINE_AA)
    for point in positions.values():
        cv2.circle(overlay, point, 3, (255, 255, 255), -1, cv2.LINE_AA)
    return overlay


class HandshakePipelineNode(object):
    """カメラ入力 -> 骨格推定 -> (ARM ボタン押下時) 掌推定・IK を行うノード."""

    def __init__(self, args):
        self.args = args
        # 既定の 10 秒だと、カメラ側と base_link 側の TF を配信している
        # マシン間でシステムクロックが数秒〜数十秒ズレている場合に、
        # 両者の有効期間が一度も重ならず TF が引けなくなる。根本的には
        # マシン間の時刻同期 (NTP/chrony) が必要だが、テストを進められる
        # よう ``--tf-cache-time`` でバッファの保持時間を延ばせるようにする。
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(args.tf_cache_time))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.pose_estimator = PeoplePoseEstimator(
            use_hand=True,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            min_visibility=args.min_visibility,
            min_joints=args.min_joints,
            max_z_diff=args.max_z_diff,
            min_body_size=args.min_body_size,
            max_body_size=args.max_body_size,
            max_limb_length=args.max_limb_length,
            max_hand_segment_length=args.max_hand_segment_length,
            max_hand_reach=args.max_hand_reach,
            depth_patch_size=args.depth_patch_size)

        # IK 自体は指なしロボットで解く (solve_palm_ik.py と同じ、指関節が
        # あると自己干渉ペアの組み合わせが無駄に増える)。画面には別に
        # 指ありモデル (self.display_robot) を表示する
        # (view_handshake_poses.py と同じ見た目にするため)。
        self.robot = Aero(use_hand=False)
        spik.restrict_elbow_range(self.robot)
        spik.apply_collision_model(self.robot)
        # self.robot の関節角ベクトル (motion['joint_names'] と同じ並び) での
        # 「初期姿勢」(両腕を体の横に下ろした姿勢, plan_handshake_motion.
        # arms_down_angles と同じ -- plan_handshake_motion.py の始点
        # (motion['waypoints'][0]) と見た目を揃える。``Aero.reset_pose`` の
        # ままだと肘を曲げた「構え」のような姿勢になる、同関数の docstring
        # 参照)。以降 self.robot は robot_position の計算 (seed_arm_pose) や
        # IK で上書きされ続けるため、ここで確保しておく (初期位置 -> 経路
        # 開始点の表示用アニメーション waypoint に使う、
        # build_initial_approach_waypoints 参照)。
        self._initial_joint_names = [j.name for j in self.robot.joint_list]
        self._initial_joint_angle_vector = [
            float(v) for v in
            phm.arms_down_angles(self.robot, self.robot.joint_list)]
        self.display_robot = load_aero(use_hand=True)
        # ロボットの「初期位置」(ARM 前および RESET 後に表示する姿勢)。
        # 台車はワールド原点、関節は両腕を下ろした姿勢とする (上と同じ)。
        phm.arms_down_angles(self.display_robot, self.display_robot.joint_list)
        self._initial_base_coords = self.display_robot.base_link.copy_worldcoords()
        # 事後検証 (pick_verified_candidate/plan_person_motion の
        # verify_waypoints) の総当たりペアは、ロボットの構造だけで決まり
        # --collision-pairs (最適化用に絞り込んだ組み合わせ) の有無に
        # よらず必要 (plan_handshake_motion.main と同じ理由) なので、
        # 常に作る。
        self.verification_pairs = spik.build_collision_verification_pairs(
            self.robot, 'r')
        self.collision_pairs = None
        if os.path.exists(args.collision_pairs):
            self.collision_pairs = spik.load_collision_pairs(
                args.collision_pairs, self.robot)
            print('[collision-pairs] {} 組を読み込みました。'.format(
                len(self.collision_pairs)))
        else:
            print('[collision-pairs] {} が見つからないため、干渉回避なしで '
                  '解きます。'.format(args.collision_pairs))
        self.base_limits = [tuple(args.base_x_range),
                            tuple(args.base_y_range),
                            tuple(args.base_yaw_range)]

        self.robot_position = self._resolve_robot_position()
        print('[robot-hand-position] {} (base_link)'.format(
            self.robot_position.tolist()))

        # 掌推定・差し出し手判定器は毎フレーム作り直さず使い回す (以前は
        # ARMED の全フレームで新規に作っていたが、無駄な上に判定の内訳
        # (スコア/veto 理由) を毎フレーム覗けなかった)。
        self.offered_hand_selector = epp.OfferedHandSelector(
            robot_position=self.robot_position,
            score_min=args.offer_score_min)
        self.palm_estimator = epp.PalmPoseEstimator(self.offered_hand_selector)

        # 深度ノイズによる関節位置の単発の飛び (「デプスが後ろの方に一瞬
        # 飛ぶ」) を抑える時間方向の平滑化 (aero_demo.skeleton_filters.
        # OneEuroFilter 参照。record_skeleton_data.py で録った実データを
        # filter_skeleton_data.py で比較し、単純な移動中央値より跳びを
        # 抑えつつ追従の遅れが小さかったため採用)。
        self._joint_smoother = skeleton_filters.OneEuroFilter(
            mincutoff=args.joint_smoothing_mincutoff,
            beta=args.joint_smoothing_beta,
            dcutoff=args.joint_smoothing_dcutoff)

        # --- 表示・状態管理用 (コールバックスレッドと表示ループの両方から
        # 触るので lock で保護する) ---
        self._lock = threading.Lock()
        # viewer.add/delete/redraw は内部の _linkid_to_handle 辞書を書き換える
        # ため、spin() (骨格更新) と _play_loop (waypoint 再生) の 2 スレッド
        # から同時に呼ぶと "dictionary changed size during iteration" で落ちる。
        # そのため viewer への呼び出しはすべてこの lock で直列化する
        # (self._lock とは別にしているのは、_apply_current_waypoint が
        # self._lock を保持したまま呼ばれることがあり、再入不可な Lock の
        # 二重取得によるデッドロックを避けるため)。
        self._viewer_lock = threading.Lock()
        self._latest_joint_positions = None  # 最新フレームの joint_positions (dict) or None
        self._latest_is_base_frame = False   # 上記が base_link 座標系かどうか (TF 解決済みか)
        self._latest_offer_selection = None  # ARMED 中の直近の差し出し手判定の内訳 (offered_hand_selector.select の戻り値) or None
        # 'idle' (ARM 待ち) -> 'armed' (差し出し手待ち) -> 'solving'
        # (offered_hand が決まって IK 計算中) -> 'result' (IK 完了、結果
        # 表示中。RESET ボタンで 'idle' に戻る)。
        self.state = 'idle'
        self.armed_deadline = None
        self._busy = False                # IK 計算中は次フレームの処理を止める
        self._frozen_joint_positions = None  # offered_hand が決まった瞬間の骨格 (以後この骨格を固定表示する) or None
        self._current_result = None       # 直近の solve_palm_ik の結果 dict (ボタン用) or None
        self._current_motion = None       # 直近の plan_handshake_motion の結果 dict or None
        self._display_waypoints = None    # build_display_waypoints の表示用 waypoint リスト or None
        self._display_n_prepend = 0        # 上記の先頭のうち、初期位置->経路開始点の表示専用フレームの個数
        self._display_n_approach = 0      # 上記のうち経路計画済み (表示専用の先頭/末尾フレームでない) 個数
        self._collision_pairs_text = ''   # 指ありでの事後検証結果 (colliding_link_pairs/collision_pairs_text の戻り値)。_refresh_collision_pairs_text で更新する
        # 骨格表示のちらつき対策 (spin 参照)。いずれも spin() のスレッドから
        # のみ読み書きするため lock は不要。
        self._last_detected_joint_positions = None  # 直近に検出できた骨格 (未検出フレームの間もこれを表示し続ける)
        self._last_detected_time = None             # 上記を検出した時刻 (time.time())
        self._displayed_joint_positions = None      # 直近に viewer へ実際に描画した骨格 (同じなら再描画しない)
        self._last_skeleton_redraw_time = 0.0       # 直近に骨格を実際に再描画 (delete/add) した時刻
        # デバッグ用: offered_hand が決まって IK を解いた (_solve_handshake
        # を呼んだ) 回数。ARM を押し直すたびに増える連番なので、標準出力の
        # どの行がどの試行のものかを人手で追えるようにするため各ログ行に
        # 付ける (_solve_handshake 参照)。
        self._attempt_count = 0
        # デバッグログファイルパス (JSON Lines 形式)。--save-dir があれば
        # そこに debug_log.jsonl を作り、各試行の進捗を追記する。
        self._debug_log_path = None
        if args.save_dir:
            self._debug_log_path = os.path.join(args.save_dir, 'debug_log.jsonl')
            os.makedirs(args.save_dir, exist_ok=True)

        self._warmup_ik()

        # デバッグ用: カメラ画像に検出できた 2D 骨格を重ねた画像を publish
        # する (draw_skeleton_overlay 参照)。rqt_image_view 等で購読すれば、
        # viser の 3D プレビューとは別に「実際にどの関節がどの画素で検出
        # されているか」を画像上で確認できる。購読者がいないフレームでは
        # cv2 描画のコストをかけない (_on_frame 参照)。
        self.skeleton_image_pub = rospy.Publisher(
            '~skeleton_image', Image, queue_size=1)

        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        info_sub = message_filters.Subscriber(
            args.camera_info_topic, CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_frame)

        self._setup_viewer(args)

    _WARMUP_PALM = dict(
        position=[0.5, 0.0, 1.0],
        x_axis=[1.0, 0.0, 0.0],
        y_axis=[0.0, 1.0, 0.0],
    )

    def _warmup_ik(self):
        """左右それぞれの腕で ``solve_person_ik`` をダミーの目標に対して
        1 回ずつ解いておき、JAX の関数トレース (jax.jit がその形状の
        呼び出しを初めて見たときに Python レベルで計算グラフを組み立てる
        処理。ディスクの永続コンパイルキャッシュではカバーされない) を
        ノード起動時に前倒しで済ませる。これをやらないと、実際の1人目の
        差し出し手に対して IK を解くときに腕ごと数秒単位でこのトレース
        コストがかかってしまう。

        ``_solve_handshake`` の ``solve_person_ik`` 呼び出しと引数
        (``attempts_per_pose``/``base_limits``/``self_collision``/
        ``collision_pairs``/``verification_pairs``) を完全に一致させる
        必要がある -- 1 つでも違うと JAX には「別の関数」に見えて別途
        トレースされ直し、ウォームアップの意味がなくなる。``joint_
        positions={}`` でも ``human_body_obstacles`` は骨格検出が全身分
        揃っているときと同じ固定長のダミー障害物を返すので、実際の骨格
        なしで形状だけ実データと揃えられる。
        """
        args = self.args
        print('[warmup] 左右の腕の IK トレースを事前に実行しています '
              '(数秒かかります)...')
        collision_obstacles = (
            [] if (args.no_human_collision or self.collision_pairs is None)
            else spik.human_body_obstacles({}))
        for robot_arm, label in (('l', '左'), ('r', '右')):
            t0 = time.time()
            spik.solve_person_ik(
                self.robot, self._WARMUP_PALM, robot_arm, collision_obstacles,
                attempts_per_pose=args.attempts_per_pose,
                base_limits=self.base_limits,
                self_collision=(not args.no_self_collision
                                and self.collision_pairs is not None),
                collision_pairs=self.collision_pairs,
                joint_positions={},
                verification_pairs=self.verification_pairs)
            print('[warmup] {}腕: {:.1f} 秒'.format(label, time.time() - t0))

    def _setup_viewer(self, args):
        """viser ビューアと ``ARM``/``RESET`` ボタン・状態表示パネル・
        ロボットモデルを準備する.

        rqt_image_view の代わりにこの viser 画面で骨格 (base_link 座標系)
        をプレビューし、``--trigger-key`` によるキー入力の代わりにこの
        画面の ``ARM`` ボタンで ARMED 状態に入る (spin() が骨格の描画と
        状態表示の更新を毎フレーム行う)。画面には指ありのロボットモデル
        (``self.display_robot``、``view_handshake_poses.py`` と同じ見た目)
        を重ねて表示し、検出した骨格との位置関係を目で確認できるように
        する。IK 自体は指なしの ``self.robot`` で解くので、IK が完了する
        たびに ``apply_result_pose`` で ``self.display_robot`` へ結果を
        反映する (関節名で突き合わせるので、途中で ``self.robot`` を直接
        表示する必要はない)。

        offered_hand が決まって IK を解き始めると ``ARM`` ボタンは
        ``RESET`` ボタンに切り替わる (``_try_handshake`` 参照)。``RESET``
        を押すと、固定表示していた骨格 (``_frozen_joint_positions``) と
        直近の IK 結果・軌道 (``_current_result``/``_current_motion``) を
        クリアし、``self.display_robot`` を初期位置に戻して ``ARM``
        ボタンに戻る。

        IK が解けると続けて軌道計画 (``plan_handshake_motion.
        plan_person_motion``) を行い、``view_handshake_motion.py`` と同様の
        waypoint スライダー・``Play`` チェックボックスで、ロボットの初期
        姿勢 (軌道の始点) から握手姿勢までの経路をコマ送り/自動再生で
        確認できるようにする (``_apply_current_waypoint``/``_play_loop``
        参照)。
        """
        self.viewer = ViserViewer(draw_grid=True)
        # IK 自体が実際に干渉判定に使っているのは指なしの self.robot と
        # 同じ形状だが、この overlay は指あり (self.display_robot と同じ
        # URDF) で作る -- view_handshake_poses.py の既定 (--no-hand を
        # 付けない) と同様に、指先まで含めた事後検証・表示を行うため
        # (build_robot_collision_overlay/colliding_link_pairs のモジュール
        # docstring 参照)。GUI コールバック (_apply_current_waypoint 経由)
        # から参照されるため、それらのコールバックを登録するより前に
        # (viewer への add より前でよい、build_robot_collision_overlay 自体
        # は viewer に依存しない) 作っておく。
        self.robot_collision_overlay = build_robot_collision_overlay(
            self.display_robot)
        sync_robot_collision_overlay(
            self.robot_collision_overlay, self.display_robot)
        # solve_palm_ik.py の事後検証 (pick_verified_candidate) と同じ総
        # 当たりの組み合わせを、指ありの overlay から作る (指同士/指と他
        # リンクの自己干渉ペアも含む)。ロボットの構造だけで決まり人物ごとの
        # 姿勢には依存しないので、ここで 1 回だけ作る。IK 自体が使う
        # self.verification_pairs (指なし) とは別物。
        self.hand_verification_pairs = spik.build_collision_verification_pairs(
            self.robot_collision_overlay, 'r')
        self._current_obstacle_links = []  # 人体側の干渉回避ジオメトリ (Cylinder) の overlay。RESET/再 ARM のたびに作り直す
        self._refresh_collision_pairs_text()
        self.arm_button = self.viewer._server.gui.add_button(
            'ARM (差し出し手を待つ)')
        self.reset_button = self.viewer._server.gui.add_button(
            'RESET (最初からやり直す)')
        self.reset_button.visible = False

        @self.arm_button.on_click
        def _on_arm(_):  # noqa: ANN001  (viser の GuiEvent は型を問わない)
            self.state = 'armed'
            self.armed_deadline = time.time() + self.args.armed_timeout
            self._latest_offer_selection = None
            print('[ARM] ARMED になりました。{:.0f} 秒以内に手を差し出して'
                  'ください。'.format(self.args.armed_timeout))

        @self.reset_button.on_click
        def _on_reset(_):  # noqa: ANN001
            with self._lock:
                self._frozen_joint_positions = None
                self._current_result = None
                self._current_motion = None
                self._display_waypoints = None
                self._display_n_prepend = 0
                self._display_n_approach = 0
            self.play_checkbox.value = False
            self._set_waypoint_slider_range(0)  # 表示を初期位置に戻す (_apply_current_waypoint 経由で事後検証も更新される)
            self.reset_button.visible = False
            self.arm_button.visible = True
            self.state = 'idle'
            with self._viewer_lock:
                for obstacle_link in self._current_obstacle_links:
                    self.viewer.delete(obstacle_link)
                self._current_obstacle_links = []
                self.viewer.redraw()
            print('[RESET] 骨格表示とロボットの姿勢を初期状態に戻しました。')

        self._status_text = self.viewer._server.gui.add_markdown('')
        # 軌道計画完了後、waypoint をコマ送り/自動再生で確認するための
        # スライダー・チェックボックス (view_handshake_motion.
        # PlaybackControls と同じ役割だが、この画面は常に「直近 1 件の
        # 軌道」だけを扱うので Back/Next (人物切り替え) は無い)。軌道が無い
        # (IDLE/IK 失敗) 間は waypoint が 1 つだけなので操作しても意味が無い。
        self.waypoint_slider = self.viewer._server.gui.add_slider(
            'waypoint', min=0, max=0, step=1, initial_value=0)
        self.play_checkbox = self.viewer._server.gui.add_checkbox(
            'Play', initial_value=False)

        @self.waypoint_slider.on_update
        def _on_waypoint(_):  # noqa: ANN001
            self._apply_current_waypoint()
            with self._viewer_lock:
                self.viewer.redraw()

        @self.play_checkbox.on_update
        def _on_play_toggle(_):  # noqa: ANN001
            # 最後まで再生し終わる (_play_loop) と Play は自動でオフになり、
            # スライダーは最終 waypoint (max) のままになる。その状態で
            # 再度 Play をオンにしても index >= max のままだと _play_loop が
            # 即座にオフに戻してしまい何度でも再生できないので、ここで
            # waypoint 0 まで巻き戻してから再生を始める。
            if (self.play_checkbox.value
                    and int(self.waypoint_slider.value)
                    >= self.waypoint_slider.max):
                self.waypoint_slider.value = 0

        threading.Thread(target=self._play_loop, daemon=True).start()

        # 干渉回避用の半透明モデル (ロボット自身の近似ジオメトリ overlay
        # ``robot_collision_overlay`` と、人体側の障害物 Cylinder
        # ``_current_obstacle_links``) の表示/非表示をまとめて切り替える
        # チェックボックス (view_handshake_poses.py の
        # show_collision_models_checkbox と同じ)。waypoint 29/34 のように
        # 「干渉余裕は正 (貫通なし) のはずなのに見た目は貫通しているように
        # 見える」場合に、事後検証 (colliding_link_pairs) が指先まで含めて
        # 実際に使っている (見た目のメッシュより粗い) プリミティブ形状を
        # 重ねて見比べられるようにするため。
        self.show_collision_models_checkbox = (
            self.viewer._server.gui.add_checkbox(
                '干渉回避用モデルの表示', initial_value=True))

        @self.show_collision_models_checkbox.on_update
        def _on_toggle_collision_models(_):  # noqa: ANN001
            visible = self.show_collision_models_checkbox.value
            with self._viewer_lock:
                for link in self.robot_collision_overlay.link_list:
                    self._set_link_visible(link, visible)
                for obstacle_link in self._current_obstacle_links:
                    self._set_link_visible(obstacle_link, visible)
                self.viewer.redraw()

        self._skeleton_links = []
        # ロボットの初期位置 (台車がワールド原点にいる姿勢) を示す Axis。
        # display_robot 自体は IK 結果 (台車が移動した姿勢) で上書きされて
        # しまうため、初期位置がどこだったか目で追えるよう別に固定表示
        # する (RESET しても消えない、常時表示の目印)。
        initial_pose_axis = Axis(axis_length=INITIAL_POSE_AXIS_LENGTH,
                                 axis_radius=INITIAL_POSE_AXIS_RADIUS)
        initial_pose_axis.newcoords(self._initial_base_coords.copy_worldcoords())
        self.viewer.add(initial_pose_axis)
        # ViserViewer は RobotModel を add() すると "Joint Angles" フォルダ
        # (関節ごとのスライダー) を自動で GUI パネルに追加してしまう
        # (view_handshake_poses.py と同じ注意点) ので、ボタン・チェック
        # ボックス・状態表示パネルを先に追加してからロボットを add() する。
        self.viewer.add(self.display_robot)
        # 指ありでの事後検証 (colliding_link_pairs) に使っているのと同じ
        # プリミティブ近似ジオメトリ (self.robot_collision_overlay、上で
        # 構築済み) を、表示用ロボット (self.display_robot、詳細なメッシュ
        # で見た目は不透明) に重ねて半透明で表示する (view_handshake_poses.py
        # と同じ)。
        self.viewer.add(self.robot_collision_overlay)
        self.viewer.show(open_browser=not args.no_open_browser)
        viewer_nav.wait_for_client(self.viewer, args.client_wait_timeout)

    def _set_link_visible(self, link, visible):
        common_set_link_visible(self.viewer, link, visible)

    def _resolve_robot_position(self):
        """既定のロボット手先位置 (掌推定の ``robot_position``)。

        ``--robot-hand-position`` が明示されていればそれを、なければ右腕の
        「種の姿勢」(``solve_person_ik`` が IK の初期値に使うのと同じ姿勢,
        台車はワールド原点) の手先位置を base_link 座標として使う。
        """
        if self.args.robot_hand_position is not None:
            return np.asarray(self.args.robot_hand_position, dtype=np.float64)
        spik.seed_arm_pose(self.robot, 'r')
        return np.asarray(self.robot.rarm_end_coords.worldpos(),
                          dtype=np.float64)

    def _lookup_camera_to_base(self, header):
        """``header`` (画像の frame_id/stamp) から base_link への TF を引く.

        まず画像の stamp ちょうどの TF を試み、それが (バッファに無い/
        extrapolation エラー等で) 引けなければ最新の TF (``rospy.Time(0)``)
        にフォールバックする。後者は画像とTFの時刻が厳密には一致しない
        (カメラ画像を出しているマシンと TF を配信しているマシンの間で
        システムクロックがズレていると、``ExtrapolationException`` が
        毎回発生してこの経路に入り続ける -- その場合は根本的には NTP 等で
        クロックを同期するべきだが、応急的にこのフォールバックでテストを
        続けられるようにしてある)。
        """
        try:
            return self.tf_buffer.lookup_transform(
                self.args.base_frame, header.frame_id, header.stamp,
                rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            # マシン間の時刻ズレで毎フレーム発生しうる想定内のフォール
            # バックなので、警告は出さず黙って最新の TF にフォールバック
            # する (それでも引けない場合だけ下の except で警告する)。
            pass
        try:
            # tf2 では Time(0) は「時刻 0」であり tf とは違って「最新」を
            # 意味しない。最新を取得するには現在時刻を渡す必要がある。
            return self.tf_buffer.lookup_transform(
                self.args.base_frame, header.frame_id, rospy.Time.now(),
                rospy.Duration(0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(
                5.0, 'TF lookup failed (%s -> %s): %s',
                header.frame_id, self.args.base_frame, e)
            return None

    # ------------------------------------------------------------------
    # camera callback
    # ------------------------------------------------------------------
    def _on_frame(self, color_msg, depth_msg, info_msg):
        if self._busy:
            return
        # TF が引けなくてもプレビューは止めない (master ブランチの
        # people_pose_estimator_ros.py と同じ考え方: 変換できなければ
        # カメラ座標系のまま推定を続ける)。ARMED での掌推定・IK だけは
        # base_link 座標系が要るので、変換できたフレームでのみ行う。
        transform = self._lookup_camera_to_base(color_msg.header)
        camera_to_base = (None if transform is None
                          else _transform_to_matrix(transform.transform))

        color = _imgmsg_to_ndarray(color_msg, desired_encoding='bgr8')
        depth_raw = _imgmsg_to_ndarray(depth_msg)
        depth_m = PeoplePoseEstimator.depth_to_meters(
            depth_raw, encoding=depth_msg.encoding)
        intrinsics = CameraIntrinsics.from_matrix(info_msg.K)

        people, joints_2d = self.pose_estimator.estimate_3d(
            color, depth_m, intrinsics, output_transform=camera_to_base)

        if self.skeleton_image_pub.get_num_connections() > 0:
            overlay = (draw_skeleton_overlay(color, joints_2d[0])
                      if joints_2d else color)
            self.skeleton_image_pub.publish(
                _ndarray_to_imgmsg(overlay, 'bgr8', color_msg.header))
        # TF が引けなくても viser のプレビューは止めない (camera_to_base が
        # None のフレームは people がカメラ座標系のままになるが、それでも
        # 骨格の形自体は見えるので、TF 未解決時に画面が真っ暗になるのを
        # 避けるためそのまま表示する)。ARMED での掌推定・IK だけは
        # base_link 座標系が要るので、変換できたフレームでのみ行う。
        is_base_frame = camera_to_base is not None
        raw_joint_positions = people[0] if people else None
        # 深度の単発の外れ値 (奥の壁に一瞬飛ぶ等) を One Euro Filter で時間
        # 方向に抑える (self._joint_smoother 参照)。座標系が変わったら (TF
        # 解決状況の変化) 履歴を自動でリセットする。One Euro Filter はフレーム
        # 数ではなく実時間に基づいて減衰するため、カメラ画像のタイムスタンプ
        # (color_msg.header.stamp) を渡す。
        preview_joint_positions = (
            None if raw_joint_positions is None
            else self._joint_smoother.update(
                raw_joint_positions, t=color_msg.header.stamp.to_sec(),
                frame_key=is_base_frame))
        armed_joint_positions = (
            preview_joint_positions if is_base_frame else None)

        with self._lock:
            self._latest_joint_positions = preview_joint_positions
            self._latest_is_base_frame = is_base_frame

        if self.state == 'armed' and armed_joint_positions is not None:
            self._try_handshake(armed_joint_positions)

        if (self.state == 'armed' and self.armed_deadline is not None
               and time.time() > self.armed_deadline):
            self.state = 'idle'
            self.armed_deadline = None
            print('[ARMED] タイムアウトしました。差し出し手が決まりません '
                  'でした。')

    def _try_handshake(self, joint_positions):
        palms = self.palm_estimator.estimate(joint_positions)
        # ARMED なのに offered_hand が決まらないとき、viser 画面 (と
        # スロットルした標準出力) にスコア/veto 理由の内訳を出す。「手の
        # ランドマークがそもそも取れていない (veto=no_palm)」のか
        # 「取れているがスコアが --offer-score-min に届いていない」のかを
        # 見分けられるようにするため (PalmPoseEstimator.estimate は
        # offered_hand しか返さないので、同じ入力で select() を呼び直す)。
        selection = self.offered_hand_selector.select(joint_positions, palms)
        with self._lock:
            self._latest_offer_selection = selection
        rospy.loginfo_throttle(
            1.0, '[ARMED] %s',
            _format_offer_scores(selection, self.offered_hand_selector.score_min)
            .replace('**', '').replace('\n', ' / '))

        offered_hand = palms['offered_hand']
        if offered_hand is None:
            return
        self.armed_deadline = None
        self._busy = True
        # 差し出し手が決まった瞬間の骨格を固定表示にする (以後 ARMED を
        # 抜けるので、この骨格はもうカメラの最新フレームで上書きされない)。
        # 同時に IK 計算に入るので ARM ボタンを RESET ボタンに切り替える。
        with self._lock:
            self._frozen_joint_positions = joint_positions
        self.state = 'solving'
        self.arm_button.visible = False
        self.reset_button.visible = True
        try:
            self._solve_handshake(joint_positions, palms, offered_hand)
        finally:
            self._busy = False
            self.state = 'result'

    def _log_debug(self, record):
        """デバッグ用ログを JSON 1 行として標準出力・ファイルに書く.

        テキストの整形ログだと ``grep``/後からの機械的な集計がしづらいため、
        ``_solve_handshake`` が offered_hand を検出してから結果が出るまでの
        各段階の情報を、すべて ``{"event": ...}`` の JSON 1 行にまとめて出す
        (``[debug]`` 接頭辞で ``grep '^\\[debug\\]'`` すれば debug ログだけ
        抜き出せる)。``--save-dir`` が指定されていれば、
        ``save_dir/debug_log.jsonl`` に追記される (JSONL 形式)。
        """
        log_line = '[debug] ' + json.dumps(record, ensure_ascii=False)
        print(log_line)
        if self._debug_log_path is not None:
            with open(self._debug_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def _solve_handshake(self, joint_positions, palms, offered_hand):
        args = self.args
        self._attempt_count += 1
        attempt = self._attempt_count
        self._log_debug(dict(event='armed', person=attempt,
                             offered_hand=offered_hand))
        robot_arm = (spik.DEFAULT_ROBOT_ARM[offered_hand]
                    if args.robot_arm == 'auto' else args.robot_arm)
        palm = palms[offered_hand]

        # 干渉計算 (human_body_obstacles) には実際に検出できた関節だけを
        # 使う (欠損部分の補間は行わない、human_body_obstacles 自身が
        # 欠けている部位をダミーの障害物で埋める)。
        offset = spik.human_translation_offset(
            joint_positions, front_distance=args.human_front_distance)
        translated_joints = spik.translate_joint_positions(
            joint_positions, offset)
        translated_palm = spik.translate_palm(palm, offset)
        collision_obstacles = (
            [] if (args.no_human_collision or self.collision_pairs is None)
            else spik.human_body_obstacles(translated_joints))

        target_pos = spik.palm_target_position(translated_palm)
        rots = spik.palm_to_target_rots(translated_palm, robot_arm)
        picked, collision_ik_time, candidate_selection_time = \
            spik.solve_person_ik(
                self.robot, translated_palm, robot_arm, collision_obstacles,
                attempts_per_pose=args.attempts_per_pose,
                base_limits=self.base_limits,
                self_collision=(not args.no_self_collision
                                and self.collision_pairs is not None),
                collision_pairs=self.collision_pairs,
                joint_positions=translated_joints,
                verification_pairs=self.verification_pairs)

        if picked is None:
            result = spik.unsolved_result(
                self.robot, robot_arm, target_pos, rots[-1],
                self.base_limits, collision_ik_time,
                candidate_selection_time)
        else:
            turn_index, angle_vector, base_pose, post_process_result = picked
            result = spik.solved_result(
                self.robot, robot_arm, target_pos, rots[turn_index],
                turn_index, angle_vector, base_pose, self.base_limits,
                post_process_result, collision_ik_time,
                candidate_selection_time)
        result['offered_hand'] = offered_hand
        result['robot_arm'] = robot_arm

        # IK が解けたら続けて軌道計画を行う (plan_handshake_motion.py の
        # main と同じ、target かつ solved の人物だけが対象)。IK は
        # translated_joints/translated_palm を使う仮想座標系で解いている
        # ため、軌道計画もこの座標系のまま (result を untranslate する前)
        # に行う -- plan_person_motion 自身が呼ぶ human_body_cylinder_
        # obstacles/approach_base_start が、この座標系の joint_positions/
        # result['base_position'] と対応している必要があるため。
        motion = None
        if result['solved']:
            human_xy = spik.human_standing_xy(translated_joints)
            if human_xy is None:
                human_xy = np.array([args.human_front_distance, 0.0])
            # plan_person_motion は args.collision_verify_tolerance を
            # waypoint の事後検証の許容誤差として読む (plan_handshake_
            # motion.py 自身の --collision-verify-tolerance と同じ意味) が、
            # このスクリプトでは同名のフラグを画面の状態表示用 (指先まで
            # 含めた事後検証, --collision-verify-tolerance) に使っている
            # ため、ここだけ --motion-collision-verify-tolerance の値に
            # 差し替えたコピーを渡す (self.args 自体は書き換えない)。
            motion_args = copy.copy(args)
            motion_args.collision_verify_tolerance = \
                args.motion_collision_verify_tolerance
            motion = phm.plan_person_motion(
                self.robot, robot_arm, result, translated_joints, human_xy,
                motion_args, self.verification_pairs)
            self._log_debug(dict(
                event='motion', person=attempt,
                verified=motion['verified'],
                min_dist=float(min(motion['waypoint_min_distances'])),
                waypoint_min_distances=[
                    float(d) for d in motion['waypoint_min_distances']],
                kind=phm.KIND_LABELS.get(motion['kind'], motion['kind']),
                compute_time=motion['compute_time']))

        self._untranslate_result(result, offset)
        if motion is not None:
            self._untranslate_motion(motion, offset)

        self._log_debug(dict(
            event='result', person=attempt, offered_hand=offered_hand,
            robot_arm=robot_arm, solved=result['solved'],
            base_position=[result['base_position'][0],
                          result['base_position'][1]],
            collision_ik_time=collision_ik_time,
            candidate_selection_time=candidate_selection_time))

        # solve_palm_ik.py が実際に干渉判定へ使ったのと同じ人体の近似
        # ジオメトリ (Cylinder) は、この joint_positions (frozen 表示中の
        # 骨格) に対して spin() -> _update_skeleton_view が継続的に描画・
        # 更新し、self._current_obstacle_links に保持している
        # (view_handshake_poses.py と同じ半透明表示)。指ありでの事後検証
        # (colliding_link_pairs) は、この self._current_obstacle_links を
        # そのまま使う (_refresh_collision_pairs_text 参照) ので、見た目の
        # メッシュと判定に使うメッシュが常に一致する。ここで改めて作り直す
        # 必要はない。

        # 画面の指ありロボットに軌道の waypoint 0 (初期姿勢) から表示する
        # (view_handshake_motion.py と同じ、waypoint スライダー/Play で
        # 握手姿勢まで確認できる)。軌道が無い (IK 失敗) 場合は種の姿勢
        # (apply_result_pose の後処理前) を 1 waypoint だけの表示にする。
        if motion is not None:
            display_waypoints, n_approach = build_display_waypoints(
                motion, result)
            # 経路計画は接近開始位置 (motion['waypoints'][0]) から始まる
            # ため、その手前にロボットの初期位置 (台車=ワールド原点,
            # 関節=reset_pose) からの表示専用アニメーションを継ぎ足す
            # (干渉は考慮しない、build_initial_approach_waypoints 参照)。
            initial_pos = self._initial_base_coords.worldpos()
            prepend_waypoints = build_initial_approach_waypoints(
                initial_pos, 0.0, self._initial_joint_names,
                self._initial_joint_angle_vector, motion['waypoints'][0],
                motion['joint_names'])
            n_prepend = len(prepend_waypoints)
            display_waypoints = prepend_waypoints + display_waypoints
        else:
            display_waypoints, n_prepend, n_approach = None, 0, 0
        with self._lock:
            self._current_result = result
            self._current_motion = motion
            self._display_waypoints = display_waypoints
            self._display_n_prepend = n_prepend
            self._display_n_approach = n_approach
        if display_waypoints is not None:
            self._set_waypoint_slider_range(len(display_waypoints) - 1)
        else:
            apply_result_pose(self.display_robot, result,
                              use_post_process=False)
            self._set_waypoint_slider_range(0)

        if args.save_dir:
            self._save_attempt(joint_positions, palms, result, motion)

    @staticmethod
    def _untranslate_result(result, offset):
        """IK は ``translate_joint_positions``/``translate_palm`` で人物を
        ``--human-front-distance`` の位置へ仮想的に平行移動した座標系で
        解いているため (``solve_palm_ik.py`` の ``HUMAN_FRONT_DISTANCE``
        参照)、``result`` の位置は全てこの仮想座標系のままになっている。
        カメラで実際に検出した人物の位置 (実際の ``base_link`` 座標系)
        へ戻すため、平行移動量 ``offset`` の逆を x/y に適用する
        (破壊的に書き換える)。これを行わずに ``base_position`` を実機の
        台車移動指令に使うと、実際の人物ではなく「前方 ``--human-front-
        distance`` m にいる仮想の人物」に向かって動いてしまう。
        """
        dx, dy = offset
        for key in ('target_position', 'hand_position', 'base_position'):
            if key in result and result[key] is not None:
                result[key][0] -= dx
                result[key][1] -= dy
        region = result.get('base_movable_region')
        if region:
            region['x_range'] = [v - dx for v in region['x_range']]
            region['y_range'] = [v - dy for v in region['y_range']]
        post_process = result.get('post_process')
        if post_process:
            for key in ('target_position', 'hand_position', 'base_position'):
                if key in post_process and post_process[key] is not None:
                    post_process[key][0] -= dx
                    post_process[key][1] -= dy

    @staticmethod
    def _untranslate_motion(motion, offset):
        """``_untranslate_result`` と同じ理由で、``motion['waypoints']``
        (仮想座標系の台車位置) を実際の ``base_link`` 座標系へ戻す
        (破壊的に書き換える)。waypoint の関節角は台車位置に依存しないので
        そのままでよい。"""
        dx, dy = offset
        for wp in motion['waypoints']:
            wp['base_position'][0] -= dx
            wp['base_position'][1] -= dy

    def _save_attempt(self, joint_positions, palms, result, motion=None):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        name = '{}.json'.format(stamp)
        skeleton_dir = os.path.join(self.args.save_dir, 'skeletons')
        palm_dir = os.path.join(self.args.save_dir, 'palms')
        handshake_dir = os.path.join(self.args.save_dir, 'handshakes')
        motion_dir = os.path.join(self.args.save_dir, 'motions')
        dirs = [skeleton_dir, palm_dir, handshake_dir]
        if motion is not None:
            dirs.append(motion_dir)
        for d in dirs:
            os.makedirs(d, exist_ok=True)
        json_io.save_json(
            os.path.join(skeleton_dir, name),
            dict(skeleton=dict(joint_positions={
                k: list(v) for k, v in joint_positions.items()}, height=0.0)))
        json_io.save_json(os.path.join(palm_dir, name), palms)
        json_io.save_json(os.path.join(handshake_dir, name), result)
        if motion is not None:
            json_io.save_json(os.path.join(motion_dir, name), motion)
            subdirs = '{skeletons,palms,handshakes,motions}'
        else:
            subdirs = '{skeletons,palms,handshakes}'
        print('[save] {} に保存しました (view_handshake_motion.py '
              '--skeleton-dir <dir>/skeletons --handshake-dir '
              '<dir>/handshakes --motion-dir <dir>/motions で後から '
              '見返せる)。'.format(
                  os.path.join(self.args.save_dir, subdirs, name)))

    # ------------------------------------------------------------------
    # waypoint スライダー/Play (view_handshake_motion.PlaybackControls の
    # 単純化版 -- この画面は常に「直近 1 件の軌道」だけを扱うので
    # Back/Next (人物切り替え) は無い)
    # ------------------------------------------------------------------
    def _set_waypoint_slider_range(self, max_index):
        """waypoint スライダーの範囲を ``[0, max_index]`` にし、waypoint 0
        (軌道が無ければ唯一の姿勢) を表示する。"""
        self.waypoint_slider.max = max_index
        # value を 0 に設定すると (既に 0 でない限り) on_update
        # (_on_waypoint) が同期的に発火し、_apply_current_waypoint が
        # 呼ばれる。既に 0 の場合は発火しないので、ここで明示的に呼ぶ
        # (view_handshake_motion.PlaybackControls.set_waypoint_count と
        # 同様の注意点)。
        self.waypoint_slider.value = 0
        self._apply_current_waypoint()

    def _apply_current_waypoint(self):
        """waypoint スライダーの現在値を ``self.display_robot`` に反映する.

        軌道計画済み (``_display_waypoints`` が設定されている) なら
        ``apply_waypoint_pose`` で該当 waypoint を反映し、そうでなければ
        (IDLE/IK 失敗) ``_current_result`` があれば後処理前の姿勢を、無け
        れば初期位置を表示する。
        """
        with self._lock:
            result = self._current_result
            motion = self._current_motion
            display_waypoints = self._display_waypoints
        if display_waypoints is not None:
            index = min(int(self.waypoint_slider.value),
                       len(display_waypoints) - 1)
            apply_waypoint_pose(
                self.display_robot, motion['joint_names'],
                display_waypoints, index)
        elif result is not None:
            apply_result_pose(self.display_robot, result,
                              use_post_process=False)
        else:
            phm.arms_down_angles(self.display_robot,
                                 self.display_robot.joint_list)
            self.display_robot.base_link.newcoords(
                self._initial_base_coords.copy_worldcoords())
        sync_robot_collision_overlay(
            self.robot_collision_overlay, self.display_robot)
        self._refresh_collision_pairs_text()

    def _refresh_collision_pairs_text(self):
        """指ありの ``self.robot_collision_overlay`` の現在の姿勢
        (``_apply_current_waypoint`` で ``self.display_robot`` に同期済み)
        で、``colliding_link_pairs`` による事後検証をやり直し、結果の文字列
        (``_update_status_text`` が表示する) を ``self._collision_pairs_text``
        に保存する。IK の探索自体は指なしで行っているため、この検証は表示
        用の別チェックであり ``motion['waypoint_min_distances']`` (指なしで
        の判定) を上書きするものではない。

        人体側は ``self._current_obstacle_links`` (``_update_skeleton_view``
        が画面に表示している、まさにその半透明 Cylinder) をそのまま渡すので、
        見た目のメッシュと判定に使うメッシュが常に一致する。これが空
        (``[]``, 骨格未検出/IDLE/RESET 直後) の間は人体との干渉は判定でき
        ないが、ロボットの自己干渉 (指同士/指と他リンク含む) はそれでも
        判定できる。
        """
        colliding = colliding_link_pairs(
            self.robot_collision_overlay, self.hand_verification_pairs,
            self._current_obstacle_links,
            tolerance=self.args.collision_verify_tolerance)
        self._collision_pairs_text = collision_pairs_text(colliding)

    def _play_loop(self):
        """``Play`` チェックボックスがオンの間、``--fps`` の周期で waypoint
        スライダーを進める (view_handshake_motion.PlaybackControls._play_
        loop と同じ、最後まで行ったら自動で止まる)。"""
        while not rospy.is_shutdown():
            time.sleep(1.0 / max(self.args.playback_fps, 1e-3))
            if not self.play_checkbox.value:
                continue
            index = int(self.waypoint_slider.value)
            if index >= self.waypoint_slider.max:
                self.play_checkbox.value = False
                continue
            # サーバー側で .value を代入すると on_update が同じスレッドで
            # 同期的に呼ばれる (view_handshake_motion.PlaybackControls と
            # 同じ実装で確認済み) ので、これだけで _on_waypoint 経由の
            # 描画が起きる。
            self.waypoint_slider.value = index + 1

    # ------------------------------------------------------------------
    # viser display
    # ------------------------------------------------------------------
    def _update_skeleton_view(self, joint_positions):
        """viser 画面の骨格の線と、人体側の干渉回避ジオメトリ (Cylinder)
        を最新フレームの内容に差し替える.

        毎フレーム古い線をすべて削除してから作り直す (人物ごとに検出
        できる関節の組み合わせが変わり、骨の本数自体が変わりうるため)。
        ``joint_positions`` が ``None`` (未検出/TF 未解決) なら何も描かず
        骨格を消す。

        干渉回避ジオメトリは ``solve_palm_ik.human_body_obstacles`` が
        実際の干渉計算に使うのと同じ ``Cylinder`` で、SMPL メッシュのような
        見た目の身体表示ではなく「実際に干渉判定へ使われている近似形状」
        そのものを見せる (``_solve_handshake`` が IK 計算時に使うのと同じ
        関数 -- 人物が仮想的に平行移動される前の実座標系の
        ``joint_positions`` を渡すので、骨格線と同じ位置に重なって見える)。
        """
        with self._viewer_lock:
            for link in self._skeleton_links:
                self.viewer.delete(link)
            self._skeleton_links = (
                [] if joint_positions is None
                else build_skeleton_links(joint_positions))
            for link in self._skeleton_links:
                self.viewer.add(link)

            for obstacle_link in self._current_obstacle_links:
                self.viewer.delete(obstacle_link)
            self._current_obstacle_links = (
                [] if joint_positions is None
                else spik.human_body_obstacles(joint_positions))
            for obstacle_link in self._current_obstacle_links:
                palm_plane_view.set_color(
                    obstacle_link, HUMAN_COLLISION_OBSTACLE_COLOR)
                self.viewer.add(obstacle_link)
                self._set_link_visible(
                    obstacle_link, self.show_collision_models_checkbox.value)

    def _update_status_text(self, joint_positions, is_base_frame, is_frozen):
        """viser 画面のテキストパネルに現在の状態を表示する.

        ``is_base_frame`` が ``False`` (TF 未解決) のときは、骨格は見えて
        いても ARMED での掌推定・IK には使われない (base_link 座標系が
        必要なため) ことが分かるよう注記する。``is_frozen`` は表示中の
        骨格が offered_hand 決定時のもので固定されているかどうか
        (``_frozen_joint_positions`` 参照)。
        """
        if self.state == 'armed' and self.armed_deadline is not None:
            remaining = max(0.0, self.armed_deadline - time.time())
            state_text = 'ARMED (残り {:.1f} 秒。手を差し出してください)'.format(
                remaining)
        elif self.state == 'solving':
            state_text = 'IK を計算中です...'
        elif self.state == 'result':
            state_text = ('結果を表示中です (waypoint スライダー/Play で '
                          '初期姿勢から握手姿勢までの軌道を確認できます。'
                          'RESET ボタンで最初からやり直せます)')
        else:
            state_text = 'IDLE (ARM ボタンを押すと手を差し出す人を待ちます)'
        if is_frozen:
            detected_text = '固定表示中 (差し出し手が決まった時点の骨格)'
        elif joint_positions is None:
            detected_text = '未検出'
        elif is_base_frame:
            detected_text = '検出中 (base_link 座標系。ARMED 動作可能)'
        else:
            detected_text = ('検出中 (TF 未解決のためカメラ座標系で表示中。'
                             'ARMED では使われません)')
        content = '**状態:** {}\n\n**骨格:** {}'.format(state_text, detected_text)
        # ARMED 中に判定できた差し出し手のスコア内訳を出す (ARM を押しても
        # 見つからないときの原因切り分け用、_try_handshake 参照)。
        if self.state == 'armed' and self._latest_offer_selection is not None:
            content += '\n\n' + _format_offer_scores(
                self._latest_offer_selection, self.offered_hand_selector.score_min)
        with self._lock:
            result = self._current_result
            motion = self._current_motion
            n_prepend = self._display_n_prepend
            n_approach = self._display_n_approach
        if result is not None:
            if motion is not None:
                kind = phm.KIND_LABELS.get(motion['kind'], motion['kind'])
                verified_text = ('OK (経路全体で干渉なし)' if motion['verified']
                                 else 'NG (経路上に干渉が残る waypoint あり)')
                waypoint_index = int(self.waypoint_slider.value)
                content += ('\n\n**軌道:** {} / 検証: {}\n\n'
                           'waypoint {}/{}'.format(
                               kind, verified_text, waypoint_index,
                               self.waypoint_slider.max))
                if waypoint_index < n_prepend:
                    content += (' (初期位置から経路開始点への移動、干渉は'
                               '考慮していない表示のみ)')
                elif waypoint_index < n_prepend + n_approach:
                    dist = motion['waypoint_min_distances'][
                        waypoint_index - n_prepend]
                    content += (' (この waypoint の干渉余裕: {:+.4f} m, {})'
                               .format(dist, '貫通' if dist < 0 else '干渉なし'))
                else:
                    content += (' (掌への押し込み: solve_palm_ik.py の後処理'
                               '判定、表示のみ)')
            else:
                content += ('\n\n**軌道:** 計画なし ({})'.format(
                    'IK 失敗' if not result['solved'] else '計算中'))
        # IK 自体は指なしで解いているが (self.verification_pairs/motion の
        # waypoint_min_distances)、指先まで含めた実際の貫通有無は別に事後
        # 検証している (view_handshake_poses.py と同じ、_refresh_collision_
        # pairs_text 参照)。表示中の waypoint が切り替わるたびに更新済み。
        content += '\n\n' + self._collision_pairs_text
        self._status_text.content = content

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def spin(self):
        """状態遷移・ARMED タイムアウト管理と、viser 画面の骨格・状態表示
        の更新を行うメインループ.

        rqt_image_view の代わりに viser (ブラウザ) で骨格をプレビューし、
        ARMED への切り替えも viser 画面の ARM ボタン (``_setup_viewer``
        参照) で行うので、このループは ``_latest_joint_positions`` を
        読んで骨格を描き直すだけでよい。
        """
        print('viser のブラウザ画面で骨格の確認と ARM ボタンの操作を '
              '行ってください (URL は起動時に表示されます)。')

        rate = rospy.Rate(10)  # ロープ的には遅くてOK、状態管理だけが目的
        while not rospy.is_shutdown():
            with self._lock:
                joint_positions = self._latest_joint_positions
                is_base_frame = self._latest_is_base_frame
                frozen_joint_positions = self._frozen_joint_positions

            if (self.state == 'armed' and self.armed_deadline is not None
                   and time.time() > self.armed_deadline):
                self.state = 'idle'
                self.armed_deadline = None
                print('[ARMED] タイムアウトしました。差し出し手が決まりませんでした。')

            # 差し出し手が決まった後 (frozen_joint_positions が設定されて
            # 以降、RESET されるまで) は、その時点の骨格を固定表示する
            # (カメラの最新フレームでは上書きしない)。
            now = time.time()
            is_frozen = frozen_joint_positions is not None
            if is_frozen:
                display_joint_positions = frozen_joint_positions
            elif joint_positions is not None:
                self._last_detected_joint_positions = joint_positions
                self._last_detected_time = now
                display_joint_positions = joint_positions
            elif (self._last_detected_time is not None
                  and now - self._last_detected_time < SKELETON_HOLD_TIMEOUT):
                # 検出が一瞬途切れただけなので、直前の骨格をそのまま
                # 表示し続ける (ここで即座に消すと毎フレームちらつく)。
                display_joint_positions = self._last_detected_joint_positions
            else:
                display_joint_positions = None

            # 表示すべき骨格が前回描画したものと変わっていないなら、
            # viewer への delete/add をせず (redraw だけ行い) ちらつきを
            # 防ぐ (_update_skeleton_view は毎回全リンクを消して作り直す
            # ため、変化していないのに毎フレーム呼ぶとちらつく)。
            # さらに認識中 (未固定表示) は関節位置が毎フレーム微妙に変わり
            # 続けるため、変化があっても SKELETON_REDRAW_INTERVAL より
            # 短い間隔では再描画しない (認識周期とは別に、実際の delete/add
            # の頻度だけを間引く)。ただし骨格が現れる/消える切り替わりや
            # ARMED 固定表示への切り替えは、体感の遅れを避けるため間引かず
            # 即座に反映する。
            changed = display_joint_positions is not self._displayed_joint_positions
            immediate = (is_frozen or display_joint_positions is None
                        or self._displayed_joint_positions is None)
            if changed and (immediate or now - self._last_skeleton_redraw_time
                           >= SKELETON_REDRAW_INTERVAL):
                self._update_skeleton_view(display_joint_positions)
                self._displayed_joint_positions = display_joint_positions
                self._last_skeleton_redraw_time = now
            self._update_status_text(joint_positions, is_base_frame, is_frozen)
            with self._viewer_lock:
                self.viewer.redraw()
            rate.sleep()
        self.pose_estimator.close()
        self.viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--color-topic', type=str,
                        default='/camera/color/image_raw/decompressed')
    parser.add_argument('--depth-topic', type=str,
                        default='/camera/depth/image_raw/decompressed')
    parser.add_argument('--camera-info-topic', type=str,
                        default='/camera/color/camera_info')
    parser.add_argument('--base-frame', type=str, default='base_link')
    parser.add_argument(
        '--tf-cache-time', type=float, default=30.0,
        help='tf2 バッファの保持時間 [秒] (既定 30.0)。カメラ側と '
            'base_link 側の TF を配信しているマシン間でシステムクロック '
            'がズレていると、既定の 10 秒では両者の有効期間が重ならず '
            'TF が引けないことがある。根本的にはマシン間の時刻同期が '
            '必要 (NTP/chrony)。')
    parser.add_argument(
        '--armed-timeout', type=float, default=15.0,
        help='ARMED になってから offered_hand が決まらなければ諦めて '
            'IDLE に戻るまでの秒数 (既定 15.0)。')
    parser.add_argument(
        '--client-wait-timeout', type=float, default=30.0,
        help='viser のブラウザクライアント接続を待つ 1 回あたりの秒数 '
            '(繰り返し待つ、既定 30.0)。')
    parser.add_argument(
        '--no-open-browser', action='store_true',
        help='viser のブラウザの自動起動を無効にする (URL を自分で開く '
            '場合)。')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5)
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5)
    parser.add_argument('--min-visibility', type=float, default=0.5)
    parser.add_argument('--min-joints', type=int, default=6)
    parser.add_argument('--max-z-diff', type=float, default=1.0)
    parser.add_argument(
        '--min-body-size', type=float, default=0.3,
        help='検出できた関節のバウンディングボックス対角線長 [m] がこれ '
            '未満の骨格を人でないとみなして棄却する (既定 0.3m、'
            'PeoplePoseEstimator._is_valid_person 参照)。')
    parser.add_argument(
        '--max-body-size', type=float, default=2.5,
        help='検出できた関節のバウンディングボックス対角線長 [m] がこれを '
            '超える骨格を人でないとみなして棄却する (既定 2.5m、深度ノイズ '
            'で関節が実際より大きく散らばった明らかに人間でない骨格を '
            'フィルタする)。')
    parser.add_argument(
        '--max-limb-length', type=float, default=0.7,
        help='肩-肘/肘-手首/腰-膝/膝-足首の各区間の長さ [m] がこれを超えたら '
            '遠位側の関節 (肘/手首/膝/足首) を検出できなかった扱いにして '
            '捨てる (既定 0.7m)。深度が単発で背景側に飛んで腕や脚が不自然 '
            'に伸びて見える現象への対策 (PeoplePoseEstimator._prune_'
            'implausible_limbs 参照)。')
    parser.add_argument(
        '--max-hand-segment-length', type=float, default=0.12,
        help='手首-各指の関節間の区間の長さ [m] がこれを超えたら遠位側の '
            'ランドマークを検出できなかった扱いにして捨てる (既定 0.12m)。'
            '指は輪郭が細く深度パッチが背景を拾いやすいため、指のランド '
            'マークが一瞬だけ全く違う場所に飛ぶ現象への対策 '
            '(PeoplePoseEstimator._prune_implausible_hand_landmarks 参照)。')
    parser.add_argument(
        '--max-hand-reach', type=float, default=0.22,
        help='手首 ({side}Hand0) から各指ランドマークまでの直線距離 [m] '
            'がこれを超えたら遠位側のランドマークを検出できなかった扱い '
            'にして捨てる (既定 0.22m)。--max-hand-segment-length は隣接 '
            '関節同士の距離しか見ないため、各区間が閾値ギリギリで同じ '
            '方向に連鎖すると手首-指先の累積では大きく伸びうる (指全体が '
            '花束状に開いて見える現象) のを防ぐための追加チェック '
            '(PeoplePoseEstimator._prune_implausible_hand_landmarks 参照)。')
    parser.add_argument('--depth-patch-size', type=int, default=3)
    parser.add_argument(
        '--joint-smoothing-mincutoff', type=float, default=0.5,
        help='関節位置の時間方向の平滑化 (One Euro Filter, aero_demo.'
            'skeleton_filters.OneEuroFilter 参照) の最小カットオフ周波数 '
            '[Hz] (既定 0.5)。下げるほど静止時のジッタが減るが追従が '
            '遅れる。record_skeleton_data.py で録った実データを '
            'filter_skeleton_data.py で比較して決めた値。')
    parser.add_argument(
        '--joint-smoothing-beta', type=float, default=0.3,
        help='One Euro Filter の速度依存カットオフの係数 (既定 0.3)。'
            '上げるほど速い動きへの追従の遅れが減るが静止時のジッタが '
            '増える。')
    parser.add_argument(
        '--joint-smoothing-dcutoff', type=float, default=1.0,
        help='One Euro Filter の速度推定のカットオフ周波数 [Hz] (既定 1.0)。')
    parser.add_argument(
        '--offer-score-min', type=float, default=0.7,
        help='差し出し手と判定するスコアの閾値 (既定 0.7)。'
            'estimate_palm_poses.OFFER_SCORE_MIN ({:.2f}) は合成骨格向けに '
            '調整された値で実カメラでは届きにくいため、実カメラ用にここで '
            '下げてある。それでも ARM を押して差し出し手が見つからない '
            '場合は、viser 画面に表示されるスコアを見ながらさらに調整する '
            'とよい。'.format(epp.OFFER_SCORE_MIN))
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='使うロボットの腕。既定 (auto) は人間の手の反対側 '
            '(solve_palm_ik.py の DEFAULT_ROBOT_ARM と同じ)。')
    parser.add_argument(
        '--robot-hand-position', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'Z'),
        help='掌推定 (差し出し手判定) が基準にするロボット手先の base_link '
            '座標 [m]。既定は右腕の種の姿勢の手先位置から自動計算する。')
    parser.add_argument(
        '--human-front-distance', type=float,
        default=spik.HUMAN_FRONT_DISTANCE,
        help='IK を解く際に Aero の前方どれだけの位置に人物を置くか [m] '
            '(既定 {:.1f})。'.format(spik.HUMAN_FRONT_DISTANCE))
    parser.add_argument(
        '--attempts-per-pose', type=int,
        default=spik.DEFAULT_ATTEMPTS_PER_POSE)
    parser.add_argument(
        '--collision-pairs', type=str,
        default=os.path.join(_SCRIPTS_DIR, 'collision_pairs.json'))
    parser.add_argument('--no-human-collision', action='store_true')
    parser.add_argument('--no-self-collision', action='store_true')
    parser.add_argument(
        '--collision-verify-tolerance', type=float,
        default=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE,
        help='画面の状態表示に出す、指先まで含めた事後検証 (colliding_'
            'link_pairs) の距離の許容誤差 [m] (view_handshake_poses.py の '
            '--collision-verify-tolerance と同じ意味。既定 {})。IK 自体の '
            '干渉判定 (指なし) には使わない。'.format(
                spik.DEFAULT_COLLISION_VERIFY_TOLERANCE))
    parser.add_argument(
        '--base-x-range', type=float, nargs=2,
        default=list(spik.DEFAULT_BASE_X_RANGE))
    parser.add_argument(
        '--base-y-range', type=float, nargs=2,
        default=list(spik.DEFAULT_BASE_Y_RANGE))
    parser.add_argument(
        '--base-yaw-range', type=float, nargs=2,
        default=list(spik.DEFAULT_BASE_YAW_RANGE))
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='指定すると、IK まで解いた試行ごとに骨格/掌/IK結果/軌道の '
            'JSON を保存する (view_handshake_motion.py --skeleton-dir '
            '<dir>/skeletons --handshake-dir <dir>/handshakes --motion-dir '
            '<dir>/motions で後から見返せる)。')
    # --- 軌道計画 (plan_handshake_motion.plan_person_motion) ---
    # plan_handshake_motion.py と同じオプション・既定値。詳細はそちらの
    # モジュール docstring/argparse のヘルプを参照。
    parser.add_argument(
        '--approach-distance', type=float,
        default=phm.DEFAULT_APPROACH_DISTANCE,
        help='軌道の始点で、最終台車位置から人間の反対方向へ下がる距離 '
            '[m] (既定 {})。'.format(phm.DEFAULT_APPROACH_DISTANCE))
    parser.add_argument(
        '--pretouch-standoff', type=float,
        default=phm.DEFAULT_PRETOUCH_STANDOFF,
        help='pre-touch 姿勢を、目標手先位置から人間の掌の法線方向へ '
            '引き戻す距離 [m] (既定 {})。'.format(
                phm.DEFAULT_PRETOUCH_STANDOFF))
    parser.add_argument(
        '--pretouch-split', type=float, default=phm.DEFAULT_PRETOUCH_SPLIT,
        help='軌道全体のうち pre-touch 姿勢に到達するまでに使う割合 '
            '(既定 {})。'.format(phm.DEFAULT_PRETOUCH_SPLIT))
    parser.add_argument(
        '--n-waypoints', type=int, default=phm.DEFAULT_N_WAYPOINTS,
        help='軌道の waypoint 数 (始点・終点を含む。既定 {})。'.format(
            phm.DEFAULT_N_WAYPOINTS))
    parser.add_argument('--dt', type=float, default=phm.DEFAULT_DT,
                        help='waypoint 間の時間刻み [秒] (既定 {})。'.format(
                            phm.DEFAULT_DT))
    parser.add_argument(
        '--max-iterations', type=int, default=phm.DEFAULT_MAX_ITERATIONS,
        help='軌道最適化 (jaxls) の最大反復回数 (既定 {})。'.format(
            phm.DEFAULT_MAX_ITERATIONS))
    parser.add_argument(
        '--collision-activation-distance', type=float,
        default=phm.DEFAULT_COLLISION_ACTIVATION_DISTANCE)
    parser.add_argument(
        '--self-collision-activation-distance', type=float,
        default=phm.DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE)
    parser.add_argument('--collision-weight', type=float, default=100.0)
    parser.add_argument('--self-collision-weight', type=float, default=100.0)
    parser.add_argument(
        '--smoothness-weight', type=float,
        default=phm.DEFAULT_SMOOTHNESS_WEIGHT)
    parser.add_argument(
        '--acceleration-weight', type=float,
        default=phm.DEFAULT_ACCELERATION_WEIGHT)
    parser.add_argument(
        '--motion-attempts', type=int, default=3,
        help='線形補間・pre-touch 経由の軌道で干渉が残った場合に、warm '
            'start を変えて厳密検証に通るまで最適化を解き直す最大回数 '
            '(既定 3)。')
    parser.add_argument(
        '--motion-attempt-perturbation', type=float, default=0.3)
    parser.add_argument(
        '--robot-spheres-per-link', type=int,
        default=phm.DEFAULT_ROBOT_SPHERES_PER_LINK)
    parser.add_argument(
        '--motion-collision-verify-tolerance', type=float,
        default=phm.DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE,
        help='軌道上の waypoint の事後検証で許容する最大貫通量 [m] '
            '(既定 {})。画面の状態表示用の --collision-verify-tolerance '
            '(指先まで含めた事後検証) とは別物 -- plan_person_motion '
            '呼び出し時だけこちらの値に差し替える (_solve_handshake 参照)。'
            .format(phm.DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE))
    parser.add_argument(
        '--seed', type=int, default=None,
        help='軌道計画の warm start を揺らす際に使う numpy の乱数シード '
            '(既定は指定なし)。')
    parser.add_argument(
        '--playback-fps', type=float, default=DEFAULT_PLAYBACK_FPS,
        help='Play チェックボックスをオンにしたときの waypoint 自動再生の '
            '速さ [waypoint/秒] (既定 {})。'.format(DEFAULT_PLAYBACK_FPS))
    # argparse は roslaunch が付ける残りの引数 (__name/__log 等) を無視する
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    rospy.init_node('run_camera_pipeline_test')
    node = HandshakePipelineNode(args)
    node.spin()


if __name__ == '__main__':
    main()
