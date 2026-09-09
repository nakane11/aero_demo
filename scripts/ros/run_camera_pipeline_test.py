#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""合成骨格 (``generate_random_human_poses.py``) の代わりに、実カメラ +
``PeoplePoseEstimator`` (MediaPipe) で推定した骨格に対して掌推定・IK まで
のパイプラインを試すための ROS ノード。

``run_pipeline_test.py`` が subprocess + JSON ファイル経由で「多数の人物を
まとめて処理」するのに対し、実カメラは 1 フレームにつき 0〜1 人しか
検出できず、人物もロボットに対して任意の位置に立つ。そのためこのノードは

* 検出した骨格 (base_link 座標系) と Aero のロボットモデルを scikit-robot
  の viser ビューアに重ねて常時プレビュー表示し、位置関係を確認できる
  ようにする (骨格は線を重ねて描くだけ、ARMED でない限り IK は解かない)
* viser 画面の ``ARM`` ボタンを押すと ``ARMED`` 状態になり、以後の
  フレームで毎回掌推定をやり直し続け、``offered_hand`` (差し出し手) が
  決まった瞬間の骨格でその 1 人分だけ IK を解く
* ``--armed-timeout`` 秒たっても決まらなければ諦めて ``IDLE`` に戻る

という「ボタンを押すと1人分やる」形の対話的なテストを行う。viser は
ブラウザで表示するビューアなので、実行するとブラウザが開く (WSLg 環境
などでは自動で開く)。ブラウザが自動で開かない場合は、標準出力に表示
される URL を手動で開くこと。

``estimate_palm_poses.py``/``solve_palm_ik.py`` の関数・クラスをそのまま
import して使う (骨格の入力形式は ``PeoplePoseEstimator.estimate_3d`` が
返す ``{limb_name: [x, y, z]}`` の dict で、合成骨格の
``skeleton.joint_positions`` と同じ形なので、生成元による処理の違いは無い)。

実カメラ特有の 2 つの問題への対策も入れてある。

* 深度が単発で背景側に飛ぶ (``_JointSmoother``): 関節位置を直近数フレーム
  (``--joint-smoothing-window``, 既定 3) の成分ごとの中央値で時間方向に
  平滑化してから使う。
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
import math
import os
import sys
import threading
import time

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

import estimate_palm_poses as epp  # noqa: E402
import solve_palm_ik as spik  # noqa: E402
from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

# ロボットの初期位置 (台車がワールド原点にいる姿勢) を示す Axis の大きさ
# [m]。view_handshake_poses.py の TARGET_AXIS_LENGTH と同程度の、グリッド
# 上で目立つ大きさにしてある。
INITIAL_POSE_AXIS_LENGTH = 0.2
INITIAL_POSE_AXIS_RADIUS = 0.008

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
BONE_NAME_PAIRS = BODY_BONE_PAIRS + [
    ('{}Hand{}'.format(side, a), '{}Hand{}'.format(side, b))
    for side in ('R', 'L') for a, b in HAND_SEQUENCE]


def build_skeleton_links(joint_positions):
    """骨格を部位ごとに色分けした線 (``skrobot.model.primitives.
    LineString``) のリストにする。

    ``draw_random_human_poses.build_skeleton_links`` と同じ
    ``palm_plane_view.bone_line``/``bone_color`` を使うので、見た目
    (部位ごとの色) も同じになる。

    Parameters
    ----------
    joint_positions : dict
        関節名 -> ``np.ndarray([x, y, z])`` (base_link 座標系)。
        ``PeoplePoseEstimator.estimate_3d`` が返す形式。
    """
    links = []
    for start_name, end_name in BONE_NAME_PAIRS:
        if start_name not in joint_positions or end_name not in joint_positions:
            continue
        bone = Bone(name='{}->{}'.format(start_name, end_name),
                   start_point=joint_positions[start_name],
                   end_point=joint_positions[end_name])
        links.append(palm_plane_view.bone_line(
            bone, palm_plane_view.bone_color(bone.name)))
    return links


def apply_result_pose(robot, result, use_post_process=False):
    """``solve_palm_ik`` の結果 dict (関節角・台車位置姿勢) を、表示用の
    (指ありの) ロボットモデルに反映する.

    ``scripts/view_handshake_poses.py`` の ``apply_robot_pose`` と同じ
    パターン -- ``result['joint_names']``/``joint_angle_vector`` は IK を
    解いた指なしロボットの ``joint_list`` の角度なので、指ありの
    ``robot`` とは関節の要素数・並びが異なる。そのため名前で突き合わせて
    該当する関節だけ角度を反映する (指関節は既定姿勢のまま)。

    Parameters
    ----------
    use_post_process : bool, optional
        ``True`` のとき、``result['post_process']`` (掌に押し付ける位置
        まで詰めた後処理後の姿勢) があればそれを反映する。無ければ
        (後処理判定に失敗した/IK 自体が解けなかった) 後処理前の姿勢に
        フォールバックする。
    """
    source = result
    if use_post_process and result.get('post_process') is not None:
        source = result['post_process']
    robot.reset_pose()
    name_to_angle = dict(zip(
        source['joint_names'], source['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.base_link.newcoords(Coordinates(
        pos=source['base_position'],
        rot=rpy_matrix(source['base_yaw'], 0.0, 0.0)))


class _JointSmoother(object):
    """関節位置を、直近数フレームの成分ごとの中央値で平滑化する.

    実カメラの深度は関節の輪郭付近で単発の外れ値を返すことがある
    (2D landmark が輪郭からわずかに外れた拍子に ``PeoplePoseEstimator.
    _sample_depth`` のパッチが背景側の画素を拾ってしまう、等) -- これが
    「デプスが後ろの方に一瞬飛ぶ」現象の主な原因で、関節が一瞬だけ背景の
    depth を拾って画面奥に跳ぶように見える。この外れ値が 2〜3 フレーム
    連続することは稀なので、直近 ``window`` フレームの位置を関節名ごとに
    ためておき、成分ごとの中央値を返すだけで単発の外れ値はほぼ消える
    (実際に人物が素早く動いた場合は数フレームで新しい位置に中央値も追従
    する)。空間方向 (``PeoplePoseEstimator.depth_patch_size``) の平滑化と
    直交する、時間方向の平滑化にあたる。

    カメラ座標系と base_link 座標系 (TF 解決状況によって毎フレーム変わり
    うる, ``run_camera_pipeline_test.HandshakePipelineNode._on_frame``
    参照) を混ぜて中央値を取ると数フレームだけ無意味な値になるため、
    座標系が変わったら ``update`` の ``frame_key`` が変わったとみなして
    履歴を作り直す。
    """

    def __init__(self, window=3):
        self.window = max(1, int(window))
        self._history = {}  # name -> list of np.ndarray (古い順)
        self._frame_key = None

    def update(self, joint_positions, frame_key=None):
        """今フレームの生の関節位置を履歴に積み、平滑化した結果を返す.

        Parameters
        ----------
        joint_positions : dict
            関節名 -> [x, y, z] (今フレームで検出できた関節だけ)。
        frame_key : hashable, optional
            座標系を識別するキー (例: base_link 座標系かどうか)。前回と
            異なれば履歴をリセットする。
        """
        if frame_key != self._frame_key:
            self._history = {}
            self._frame_key = frame_key
        smoothed = {}
        for name, pos in joint_positions.items():
            history = self._history.setdefault(name, [])
            history.append(np.asarray(pos, dtype=np.float64))
            if len(history) > self.window:
                del history[0]
            smoothed[name] = np.median(np.stack(history, axis=0), axis=0)
        # 今フレームで検出できなかった関節の履歴は消す (再検出したときに
        # 古い位置との中央値を取ってしまわないようにするため)。
        for name in list(self._history):
            if name not in joint_positions:
                del self._history[name]
        return smoothed


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
            depth_patch_size=args.depth_patch_size)

        # IK 自体は指なしロボットで解く (solve_palm_ik.py と同じ、指関節が
        # あると自己干渉ペアの組み合わせが無駄に増える)。画面には別に
        # 指ありモデル (self.display_robot) を表示する
        # (view_handshake_poses.py と同じ見た目にするため)。
        self.robot = Aero(use_hand=False)
        spik.restrict_elbow_range(self.robot)
        spik.apply_collision_model(self.robot)
        self.display_robot = load_aero(use_hand=True)
        # ロボットの「初期位置」(ARM 前および RESET 後に表示する姿勢)。
        # 台車はワールド原点、関節は既定姿勢とする。
        self.display_robot.reset_pose()
        self._initial_base_coords = self.display_robot.base_link.copy_worldcoords()
        self.collision_pairs = None
        self.verification_pairs = None
        if os.path.exists(args.collision_pairs):
            self.collision_pairs = spik.load_collision_pairs(
                args.collision_pairs, self.robot)
            self.verification_pairs = spik.build_collision_verification_pairs(
                self.robot, 'r')
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
        # 飛ぶ」) を抑える時間方向の平滑化 (_JointSmoother 参照)。
        self._joint_smoother = _JointSmoother(window=args.joint_smoothing_window)

        # --- 表示・状態管理用 (コールバックスレッドと表示ループの両方から
        # 触るので lock で保護する) ---
        self._lock = threading.Lock()
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

        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        info_sub = message_filters.Subscriber(
            args.camera_info_topic, CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_frame)

        self._setup_viewer(args)

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
        直近の IK 結果 (``_current_result``) をクリアし、
        ``self.display_robot`` を初期位置に戻して ``ARM`` ボタンに戻る。
        """
        self.viewer = ViserViewer(draw_grid=True)
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
            self.display_robot.reset_pose()
            self.display_robot.base_link.newcoords(
                self._initial_base_coords.copy_worldcoords())
            self._show_post_process_checkbox.value = False
            self._show_post_process_checkbox.visible = False
            self.reset_button.visible = False
            self.arm_button.visible = True
            self.state = 'idle'
            self.viewer.redraw()
            print('[RESET] 骨格表示とロボットの姿勢を初期状態に戻しました。')

        self._status_text = self.viewer._server.gui.add_markdown('')
        # IK 完了後、後処理前/後処理後 (掌に押し付け/自分の手を注視) の
        # どちらの姿勢を display_robot に反映するかを切り替えるチェック
        # ボックス (view_handshake_poses.py の show_post_process_checkbox
        # と同じ)。IK 完了前は結果が無いので隠しておく。
        self._show_post_process_checkbox = self.viewer._server.gui.add_checkbox(
            '後処理後の姿勢を表示 (掌に押し付け/自分の手を注視)',
            initial_value=False)
        self._show_post_process_checkbox.visible = False

        @self._show_post_process_checkbox.on_update
        def _on_toggle_post_process(_):  # noqa: ANN001
            with self._lock:
                result = self._current_result
            if result is not None:
                apply_result_pose(
                    self.display_robot, result,
                    self._show_post_process_checkbox.value)
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
        self.viewer.show(open_browser=not args.no_open_browser)
        viewer_nav.wait_for_client(self.viewer, args.client_wait_timeout)

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

        people, _joints_2d = self.pose_estimator.estimate_3d(
            color, depth_m, intrinsics, output_transform=camera_to_base)
        # TF が引けなくても viser のプレビューは止めない (camera_to_base が
        # None のフレームは people がカメラ座標系のままになるが、それでも
        # 骨格の形自体は見えるので、TF 未解決時に画面が真っ暗になるのを
        # 避けるためそのまま表示する)。ARMED での掌推定・IK だけは
        # base_link 座標系が要るので、変換できたフレームでのみ行う。
        is_base_frame = camera_to_base is not None
        raw_joint_positions = people[0] if people else None
        # 深度の単発の外れ値 (奥の壁に一瞬飛ぶ等) を時間方向の中央値で抑える
        # (_JointSmoother 参照)。座標系が変わったら (TF 解決状況の変化)
        # 履歴を自動でリセットする。
        preview_joint_positions = (
            None if raw_joint_positions is None
            else self._joint_smoother.update(raw_joint_positions,
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

    def _solve_handshake(self, joint_positions, palms, offered_hand):
        args = self.args
        print('[armed] offered_hand={} が決まりました。IK を解きます...'
              .format(offered_hand))
        robot_arm = (spik.DEFAULT_ROBOT_ARM[offered_hand]
                    if args.robot_arm == 'auto' else args.robot_arm)
        palm = palms[offered_hand]

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
        self._untranslate_result(result, offset)

        print('[result] offered_hand={} robot_arm={} solved={} '
              'base=({:.2f}, {:.2f}) (base_link) '
              '(collision_ik={:.2f}s, candidate_selection={:.2f}s)'.format(
                  offered_hand, robot_arm, result['solved'],
                  result['base_position'][0], result['base_position'][1],
                  collision_ik_time, candidate_selection_time))

        # 画面の指ありロボットに IK の結果 (後処理前の姿勢) を反映し、
        # 後処理前/後を切り替えるチェックボックスを出す (IK が失敗した
        # 場合も種の姿勢を表示する。post_process が無い/失敗している
        # ときはチェックボックスを ON にしても後処理前のまま変わらない)。
        with self._lock:
            self._current_result = result
        apply_result_pose(self.display_robot, result, use_post_process=False)
        self._show_post_process_checkbox.value = False
        self._show_post_process_checkbox.visible = True

        if args.save_dir:
            self._save_attempt(joint_positions, palms, result)

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

    def _save_attempt(self, joint_positions, palms, result):
        stamp = time.strftime('%Y%m%d_%H%M%S')
        name = '{}.json'.format(stamp)
        skeleton_dir = os.path.join(self.args.save_dir, 'skeletons')
        palm_dir = os.path.join(self.args.save_dir, 'palms')
        handshake_dir = os.path.join(self.args.save_dir, 'handshakes')
        for d in (skeleton_dir, palm_dir, handshake_dir):
            os.makedirs(d, exist_ok=True)
        json_io.save_json(
            os.path.join(skeleton_dir, name),
            dict(skeleton=dict(joint_positions={
                k: list(v) for k, v in joint_positions.items()}, height=0.0)))
        json_io.save_json(os.path.join(palm_dir, name), palms)
        json_io.save_json(os.path.join(handshake_dir, name), result)
        print('[save] {} に保存しました。'.format(
            os.path.join(self.args.save_dir, '{skeletons,palms,handshakes}',
                         name)))

    # ------------------------------------------------------------------
    # viser display
    # ------------------------------------------------------------------
    def _update_skeleton_view(self, joint_positions):
        """viser 画面の骨格の線を最新フレームの内容に差し替える.

        毎フレーム古い線をすべて削除してから作り直す (人物ごとに検出
        できる関節の組み合わせが変わり、骨の本数自体が変わりうるため)。
        ``joint_positions`` が ``None`` (未検出/TF 未解決) なら何も描かず
        骨格を消す。
        """
        for link in self._skeleton_links:
            self.viewer.delete(link)
        self._skeleton_links = (
            [] if joint_positions is None
            else build_skeleton_links(joint_positions))
        for link in self._skeleton_links:
            self.viewer.add(link)

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
            state_text = ('結果を表示中です (チェックボックスで後処理前/後を'
                          '切り替えられます。RESET ボタンで最初からやり直せ'
                          'ます)')
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
            is_frozen = frozen_joint_positions is not None
            display_joint_positions = (
                frozen_joint_positions if is_frozen else joint_positions)
            self._update_skeleton_view(display_joint_positions)
            self._update_status_text(joint_positions, is_base_frame, is_frozen)
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
    parser.add_argument('--depth-patch-size', type=int, default=3)
    parser.add_argument(
        '--joint-smoothing-window', type=int, default=3,
        help='関節位置の時間方向の平滑化に使う直近フレーム数 (既定 3, '
            '_JointSmoother 参照)。深度が単発で背景に飛ぶ外れ値を、直近 '
            'この枚数の成分ごとの中央値を取ることで抑える。1 にすると '
            '平滑化を無効化する (従来の挙動)。大きくするほど滑らかになる '
            '代わりに追従が遅れる。')
    parser.add_argument(
        '--offer-score-min', type=float, default=epp.OFFER_SCORE_MIN,
        help='差し出し手と判定するスコアの閾値 (既定 {:.2f}, '
            'estimate_palm_poses.OFFER_SCORE_MIN と同じ)。合成骨格向けに '
            '調整された値なので、実カメラで ARM を押しても差し出し手が '
            '見つからない場合は、viser 画面に表示されるスコアを見ながら '
            'この値を下げて試すとよい。'.format(epp.OFFER_SCORE_MIN))
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
        help='指定すると、IK まで解いた試行ごとに骨格/掌/IK結果の JSON '
            'を保存する (view_handshake_poses.py --skeleton-dir <dir>/'
            'skeletons --handshake-dir <dir>/handshakes で後から見返せる)。')
    # argparse は roslaunch が付ける残りの引数 (__name/__log 等) を無視する
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    rospy.init_node('run_camera_pipeline_test')
    node = HandshakePipelineNode(args)
    node.spin()


if __name__ == '__main__':
    main()
