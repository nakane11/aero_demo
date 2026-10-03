#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""実カメラ + ``PeoplePoseEstimator`` (MediaPipe) の骨格に対して掌推定・
IK・軌道計画を行う、「ARM を押すと1人分やる」対話的な ROS ノード。

viser 画面で骨格とロボットを常時表示し、``ARM`` 後に差し出し手が決まった
瞬間の骨格で IK・軌道計画を行う。IK は指なしロボットで解き、画面の干渉
表示は指ありモデルでの事後検証。``--auto-execute`` なら計画成功時に実機を
動かす (干渉検証 NG なら動かさない)。実機に接続できていれば、ARM 時に
初期姿勢へ戻し、差し出し手が決まるとうなずく。

Usage
-----
    python3 scripts/ros/run_camera_pipeline_test.py
    python3 scripts/ros/run_camera_pipeline_test.py --save-dir /tmp/camera_handshake_poses
    python3 scripts/ros/run_camera_pipeline_test.py --auto-arm --auto-execute
"""

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

import rospy
import message_filters
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image, LaserScan

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
_PKG_SRC_DIR = os.path.join(_SCRIPTS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# GPU メモリの一括確保の失敗ログ (RESOURCE_EXHAUSTED) を出さないため。
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# jax の永続コンパイルキャッシュ。jax の import (skrobot 経由) より前に
# 設定する必要がある (docs/jax_compilation_cache.md)。
os.environ.setdefault(
    'JAX_COMPILATION_CACHE_DIR',
    os.path.expanduser('~/.cache/jax_compilation_cache'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')

from aero_demo import hand_offer_advice  # noqa: E402
from aero_demo import json_io  # noqa: E402
from aero_demo import palm_plane_view  # noqa: E402
from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo import viewer_nav  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo import scan_matching  # noqa: E402
from aero_demo import skeleton_filters  # noqa: E402
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, lookup_camera_to_base, lookup_frame_position,
    ndarray_to_imgmsg, transform_to_matrix)

import estimate_palm_poses as epp  # noqa: E402
import solve_palm_ik as spik  # noqa: E402
import plan_handshake_motion as phm  # noqa: E402
import side_by_side_transition as sbs  # noqa: E402
from handshake_viewer_common import HUMAN_COLLISION_OBSTACLE_COLOR  # noqa: E402
from handshake_viewer_common import apply_robot_pose as apply_result_pose  # noqa: E402,E501
from handshake_viewer_common import apply_waypoint_pose  # noqa: E402
from handshake_viewer_common import build_display_waypoints  # noqa: E402
from handshake_viewer_common import build_press_in_waypoints  # noqa: E402
from handshake_viewer_common import build_robot_collision_overlay  # noqa: E402
from handshake_viewer_common import colliding_link_pairs  # noqa: E402
from handshake_viewer_common import collision_pairs_text as common_collision_pairs_text  # noqa: E402,E501
from handshake_viewer_common import remove_joint_angle_gui  # noqa: E402
from handshake_viewer_common import remove_obstacles_gui  # noqa: E402
from handshake_viewer_common import set_link_visible as common_set_link_visible  # noqa: E402,E501
from handshake_viewer_common import sync_robot_collision_overlay  # noqa: E402
from handshake_viewer_common import transition_waypoints  # noqa: E402
from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import matrix2ypr  # noqa: E402
from skrobot.interfaces.ros import AeroROSRobotInterface  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.model import LinearJoint  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.planner.trajectory_optimization.solvers import (  # noqa: E402
    create_solver)
from skrobot.viewers import ViserViewer  # noqa: E402

# 初期位置を示す Axis の大きさ [m]。
INITIAL_POSE_AXIS_LENGTH = 0.2
INITIAL_POSE_AXIS_RADIUS = 0.008

DEFAULT_PLAYBACK_FPS = 40.0  # [waypoint/秒]

ARM_INITIAL_POSE_MOVE_TIME = 5  # [秒] ARM 時に初期姿勢へ戻す時間
# うなずきで首 (neck_p_joint、既定 25 度) を下げる角度 [deg]。
# 実機で向きが逆なら符号を反転させること。
HEAD_NOD_PITCH_DEG = 40.0
HEAD_NOD_MOVE_TIME = 0.5  # [秒] 下げる・戻すそれぞれの時間の下限

# 実機再生時の waypoint 間の所要時間の下限 [秒] (腕の制御周期 15Hz の 1 周期)。
MIN_SEGMENT_TIME = 1.0 / 15.0

# 干渉 NG の waypoint が全て接近区間のこの割合より後ろなら手の出し方が原因とみなす。
FAILURE_NEAR_HAND_FRACTION = 0.7

# ARMED へ戻った後、掌がこれだけ動く/傾く/見えなくなるまで同じ手の
# 差し出しを受け付けない (_offer_changed 参照)。
RETRY_MIN_DISPLACEMENT = 0.05
RETRY_MIN_ROTATION_DEG = 20.0
RETRY_LOST_TIME = 0.5
# 失敗後に次の差し出しを受け付けるまでの待ち時間 [秒] (カメラ時刻)。
# BASE + PER_CHAR × 発話の文字数。
RETRY_COOLDOWN_BASE = 3.0
RETRY_COOLDOWN_PER_CHAR = 0.2

# 台車の速度上限。ロボット実機側の aero_base_link.yaml の max_velocity と
# 必ず一致させること (このリポジトリのコピーとは一致するとは限らない)。
BASE_MAX_VEL = 0.3  # [m/s] 実機の base_link_x/y の max_velocity
BASE_MAX_ANGVEL = 1.0  # [rad/s] 実機の base_link_pan の max_velocity

# 台車・関節の速度上限のうち指令で使う割合。実機側は上限超過を頭打ちにして
# 遅れるため、3 次スプラインの瞬間速度の最大値がこの割合になるよう区間の
# 時間を決める (_limited_time_list)。
VEL_LIMIT_RATIO = 0.9
# 上限速度 (× VEL_LIMIT_RATIO) まで加速するのにかける時間 [秒]。
ACCEL_TIME = 0.4
TIME_LIMIT_MAX_ITERATIONS = 100

# 押し込み直前の台車の位置補正 (_correct_base_residual)。
BASE_CORRECTION_MAX_ATTEMPTS = 3
BASE_CORRECTION_POSITION_TOLERANCE = 0.025  # [m]
BASE_CORRECTION_ANGLE_TOLERANCE = math.radians(2.5)  # [rad]
# /scan 照合による移動量推定 (--base-correction scan)。/odom は指令の
# 積分なのでスリップが表れない。
SCAN_CAPTURE_TIMEOUT = 0.5  # [s] 呼び出し後に新しいスキャンが届くまで待つ上限
SCAN_MAX_RANGE = 10.0  # [m] これより遠い点は使わない
SCAN_HUMAN_EXCLUDE_RADIUS = 0.5  # [m] 人の関節 (xy) 近傍の点は使わない
# 照合結果を採用する品質の下限/上限 (満たさなければ odom で代用)。
SCAN_MIN_INLIER_RATIO = 0.3
SCAN_MIN_INLIERS = 100
SCAN_MAX_RMS = 0.03  # [m]
# odom の停止判定 (_wait_odom_settle)。
ODOM_SETTLE_WINDOW = 0.05  # [s]
ODOM_SETTLE_POSITION = 0.001  # [m]
ODOM_SETTLE_ANGLE = math.radians(0.1)  # [rad]
ODOM_SETTLE_TIMEOUT = 1.0  # [s]
GOAL_STATUS_NAMES = {
    0: 'PENDING', 1: 'ACTIVE', 2: 'PREEMPTED', 3: 'SUCCEEDED', 4: 'ABORTED',
    5: 'REJECTED', 6: 'PREEMPTING', 7: 'RECALLING', 8: 'RECALLED', 9: 'LOST'}
# 腕が指令に追いつくのを待つ (_wait_joint_settle)。実機の腕は約 0.2 秒
# 遅れて追従し、wait_interpolation はその前に返るため。
JOINT_SETTLE_HAND_TOLERANCE = 0.01  # [m]
JOINT_SETTLE_TIMEOUT = 1.5  # [s]
JOINT_SETTLE_POLL_PERIOD = 0.05  # [s]

# hover 後の押し込み姿勢の補正 (_refine_press_in)。体が画角に入らない
# ことが多いので MediaPipe Hands で手だけ検出し、複数フレームの中央値を使う。
PRESS_IN_REFINE_FRAMES = 3
PRESS_IN_REFINE_TIMEOUT = 1.5  # [s]
PRESS_IN_REFINE_MAX_HAND_DISTANCE = 0.15  # [m] 計画時の掌からの距離
# Hands は score が低いと位置がずれ、掌と甲を取り違えて法線が反転する
# ことがある (score が高くても起きる) ので両方で弾く。
PRESS_IN_REFINE_MIN_HAND_SCORE = 0.8
PRESS_IN_REFINE_MAX_NORMAL_ANGLE_DEG = 45.0

# 骨格が未検出になってもこの秒数は直前の骨格を表示し続ける (ちらつき防止)。
SKELETON_HOLD_TIMEOUT = 1.0
# 骨格の再描画周期 [秒] (状態変化は即座に反映)。
SKELETON_REDRAW_INTERVAL = 0.5


def collision_pairs_text(colliding):
    """指先まで含めた事後検証であることを示す見出しを付けた干渉ペア表示。"""
    return common_collision_pairs_text(
        colliding, label='表示中の waypoint の事後検証 (指先まで含む)')


def project_base_point(frame, point):
    """base_link 系の点をフレームの画像へ投影し ``((u, v), 奥行き[m])`` を
    返す (カメラの後ろなら ``(None, None)``)。"""
    intr = frame['intrinsics']
    p = np.linalg.inv(frame['camera_to_base']) @ np.append(point, 1.0)
    if p[2] <= 0:
        return None, None
    return ((int(intr.fx * p[0] / p[2] + intr.cx),
             int(intr.fy * p[1] / p[2] + intr.cy)), p[2])


# 重要なログは print で画面とファイルへ、詳細は log_debug でファイルだけへ。
# ファイルは起動時が LOG_STARTUP_NAME、以後は試行ごと。起動時に全削除する。
LOG_DIR = '/tmp/run_camera_pipeline_test_logs'
LOG_STARTUP_NAME = 'startup.log'
LOG_PERSON_TEMPLATE = 'person_{:02d}.log'


class _TeeStream(object):
    """標準出力とログファイルの両方に書く (ファイルは行頭に時刻付き)。"""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file
        self._lock = threading.Lock()
        self._at_line_start = True

    def set_log_file(self, log_file):
        """書き込み先を ``log_file`` に切り替え、前のファイルを閉じる。"""
        with self._lock:
            old = self._log_file
            if not self._at_line_start:
                old.write('\n')
            old.close()
            self._log_file = log_file
            self._at_line_start = True

    def write(self, text):
        with self._lock:
            self._stream.write(text)
            self._write_file(text)
        return len(text)

    def write_file_only(self, text):
        with self._lock:
            self._write_file(text)

    def _write_file(self, text):
        for line in text.splitlines(True):
            if self._at_line_start:
                now = time.time()
                self._log_file.write('{}.{:03d} '.format(
                    time.strftime('%H:%M:%S', time.localtime(now)),
                    int(now * 1000) % 1000))
            self._log_file.write(line)
            self._at_line_start = line.endswith('\n')
        self._log_file.flush()

    def flush(self):
        with self._lock:
            self._stream.flush()
            self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_tee_stream = None


def setup_log_dir():
    """``LOG_DIR`` を作り直し、標準出力をファイルへも書くようにする。
    ログファイルのパスを返す。"""
    global _tee_stream
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, LOG_STARTUP_NAME)
    _tee_stream = _TeeStream(sys.stdout, open(path, 'w', encoding='utf-8'))
    sys.stdout = _tee_stream
    return path


def switch_log_file(person):
    """ログの書き込み先を試行 ``person`` のファイルに切り替えてパスを返す。"""
    if _tee_stream is None:
        return None
    path = os.path.join(LOG_DIR, LOG_PERSON_TEMPLATE.format(person))
    _tee_stream.set_log_file(open(path, 'w', encoding='utf-8'))
    return path


def log_debug(text):
    """ログファイルにだけ 1 行書く。"""
    if _tee_stream is not None:
        _tee_stream.write_file_only(text + '\n')



class HandshakePipelineNode(object):
    """カメラ入力 -> 骨格推定 -> (ARM ボタン押下時) 掌推定・IK を行うノード."""

    def __init__(self, args):
        self.args = args
        # マシン間の時計ずれで TF が引けないとき用に保持時間を延ばせる。
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
        # MediaPipe Hands のモデル読み込みを前倒しする。
        self.pose_estimator.estimate_hands_3d(
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.zeros((240, 320), dtype=np.float32),
            CameraIntrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0))

        # IK は指なしで解き、画面には指ありモデル (display_robot) を表示する。
        self.robot = Aero(use_hand=False)
        spik.restrict_elbow_range(self.robot)
        spik.restrict_leg_range(self.robot)
        spik.restrict_waist_range(self.robot)
        spik.restrict_neck_range(self.robot)
        spik.lock_fixed_joints(self.robot)
        spik.apply_collision_model(self.robot)
        spik.apply_hand_box(self.robot)
        # 指の点群のキャッシュを先に作る (約 1 秒)。
        spik.other_hand_points('r')
        self._attach_camera_optical_coords()
        # 初期姿勢 (両腕を下ろした姿勢) の関節角。self.robot は以後 IK で
        # 上書きされるのでここで確保する。
        self._initial_joint_names = [j.name for j in self.robot.joint_list]
        self._initial_joint_angle_vector = [
            float(v) for v in
            phm.arms_down_angles(self.robot, self.robot.joint_list)]
        self.display_robot = load_aero(use_hand=True)
        # 初期位置 (台車はワールド原点、両腕を下ろした姿勢)。
        phm.arms_down_angles(self.display_robot, self.display_robot.joint_list)
        self._initial_base_coords = self.display_robot.base_link.copy_worldcoords()
        # 事後検証用の総当たりペア (--collision-pairs の有無によらず必要)。
        self.verification_pairs = spik.build_verification_pairs_for_model(
            self.robot, args.collision_verify_model)
        self.motion_verification_pairs = self.verification_pairs
        # JIT キャッシュを効かせるためノードの寿命で使い回す。
        self.solver = create_solver(
            'jaxls', max_iterations=args.max_iterations, verbose=False)
        self.collision_pairs = None
        if os.path.exists(args.collision_pairs):
            self.collision_pairs = spik.load_collision_pairs(
                args.collision_pairs, self.robot)
            log_debug('[collision-pairs] {} 組を読み込みました。'.format(
                len(self.collision_pairs)))
        else:
            print('[collision-pairs] {} が見つからないため、干渉回避なしで '
                  '解きます。'.format(args.collision_pairs))
        self.base_limits = [tuple(args.base_x_range),
                            tuple(args.base_y_range),
                            tuple(args.base_yaw_range)]

        # TF 未解決時のフォールバック値 (起動時に 1 回だけ計算)。
        self._robot_hand_position_fallback = \
            self._compute_robot_hand_position_fallback()
        self.robot_position = self._resolve_robot_position()
        log_debug('[robot-hand-position] {} (base_link)'.format(
            self.robot_position.tolist()))

        # robot_position は ARMED 中フレームごとに差し替える (_on_frame)。
        max_distance = (None if args.max_person_distance <= 0
                        else args.max_person_distance)
        self.offered_hand_selector = epp.OfferedHandSelector(
            robot_position=self.robot_position,
            score_min=args.offer_score_min,
            max_distance=max_distance)
        self.palm_estimator = epp.PalmPoseEstimator(self.offered_hand_selector)

        # 深度ノイズによる関節位置の単発の飛びを抑える時間方向の平滑化。
        self._joint_smoother = skeleton_filters.OneEuroFilter(
            mincutoff=args.joint_smoothing_mincutoff,
            beta=args.joint_smoothing_beta,
            dcutoff=args.joint_smoothing_dcutoff)

        # --- 表示・状態管理用 (コールバックと表示ループの両方から触る) ---
        self._lock = threading.Lock()
        # viewer の add/delete/redraw は別スレッドから同時に呼ぶと落ちるので
        # 直列化する。self._lock を保持したまま呼ばれることがあるので別 lock。
        self._viewer_lock = threading.Lock()
        self._latest_joint_positions = None  # 最新フレームの joint_positions (dict) or None
        self._latest_is_base_frame = False   # 上記が base_link 座標系か (TF 解決済みか)
        # _refine_press_in の要求中だけ手の検出結果を溜める。
        self._hand_frames_requested = False
        self._hand_frames = []
        self._current_palm = None         # 直近の IK に使った掌 (base_link、平行移動前) or None
        self._latest_offer_selection = None  # ARMED 中の直近の差し出し手判定の内訳 or None
        # 'idle' -> 'armed' -> 'solving' -> 'result' (RESET で 'idle')。
        # --auto-execute で失敗したら 'armed' に戻る。
        self.state = 'idle'
        # ARMED に戻った時点の左右の掌 (_offer_changed)。None ならすぐ受け付ける。
        self._retry_reference = None
        self._busy = False                # IK 計算中は次フレームの処理を止める
        self._last_failure_speech = None
        # うなずきが終わっていれば set (_execute_on_robot が待つ)。
        self._nod_done = threading.Event()
        self._nod_done.set()
        self._frozen_joint_positions = None  # 差し出し手が決まった瞬間の骨格 (固定表示) or None
        self._current_result = None       # 直近の IK 結果 dict or None
        self._current_motion = None       # 直近の軌道計画結果 dict or None
        self._handshake_total_time = None  # 掌推定開始〜軌道計画完了の時間 [秒] or None
        self._display_waypoints = None    # 表示用 waypoint リスト or None
        self._display_n_prepend = 0        # 先頭の初期位置->経路開始点の表示専用フレーム数
        self._display_n_approach = 0      # 経路計画済みのフレーム数
        self._display_n_transition = 0    # 末尾の押し込み後の横並び移動のフレーム数
        self._collision_pairs_text = ''   # 指ありでの事後検証結果
        # 骨格表示のちらつき対策 (spin のスレッドのみが触る)。
        self._last_detected_joint_positions = None
        self._last_detected_time = None
        self._displayed_joint_positions = None
        self._last_skeleton_redraw_time = 0.0
        self._attempt_count = 0  # 試行番号 (ログ用)

        self._warmup_ik()

        # 実機接続は --auto-execute によらず試みる (ARM 時の初期姿勢復帰・
        # うなずきに使う)。失敗しても viewer として動作を続ける。表示スレッド
        # との競合を避けるため実機操作専用の robot_model を持つ。
        self.real_robot = None
        self.ri = None
        if args.no_robot_interface:
            print('[execute] --no-robot-interface が指定されたため、実機 '
                  '(AeroROSRobotInterface) への接続を試みません。')
        else:
            try:
                self.real_robot = load_aero(use_hand=True)
                print('[execute] 実機 (AeroROSRobotInterface) に接続しています...')
                # skrobot 既定の /base_odometry/odom は本機に無く odom 待ちで固まる。
                self.ri = AeroROSRobotInterface(self.real_robot, odom_topic='/odom')
                print('[execute] 実機への接続が完了しました (--auto-execute={})。'
                      .format(args.auto_execute))
            except Exception as exc:  # noqa: BLE001  (実機/ROS 環境が無くても viewer 単体としては動作を継続したい)
                self.real_robot = None
                self.ri = None
                print('[execute] 実機 (AeroROSRobotInterface) への接続に失敗した '
                      'ため、ARM 時の初期姿勢への復帰・うなずき/--auto-execute '
                      'による実機操作は無効の '
                      'ままになります ({})。'.format(exc))

        # 発話 (失敗しても発話だけ諦めて続ける)。
        self.sound_client = None
        if args.auto_execute and self.ri is not None:
            try:
                from sound_play.libsoundplay import SoundClient
                self.sound_client = SoundClient(
                    sound_action='robotsound_jp', sound_topic='robotsound_jp')
            except Exception as exc:  # noqa: BLE001
                print('[speech] SoundClient の初期化に失敗したため発話し '
                      'ません ({})。'.format(exc))
        # IK 失敗時の手の出し方の助言用 reachability map。
        self.offer_advisor = None
        if args.speech_advice:
            try:
                self.offer_advisor = hand_offer_advice.OfferAdvisor(
                    args.hand_offer_table)
            except Exception as exc:  # noqa: BLE001
                print('[advice] 手の出し方の表 ({}) を読めないため、IK 失敗時'
                      'に出し方を助言しません ({})。'.format(
                          args.hand_offer_table, exc))

        # デバッグ用: 2D 骨格を重ねた画像 (購読者がいるときだけ描く)。
        self.skeleton_image_pub = rospy.Publisher(
            '~skeleton_image', Image, queue_size=1)

        # camera_info は同期に含めず最新の 1 つだけ持つ。画像の転送が遅れると
        # 同じ時刻の camera_info が履歴から押し出され組ができなくなるため。
        self._latest_camera_info = None
        self.info_sub = rospy.Subscriber(
            args.camera_info_topic, CameraInfo, self._on_camera_info,
            queue_size=1)
        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_frame)
        # (PC の受信時刻, msg) の最新 1 つ (--base-correction scan 用)。
        self._latest_scan = None
        if args.base_correction == 'scan':
            self.scan_sub = rospy.Subscriber(
                args.scan_topic, LaserScan, self._on_scan, queue_size=1)

        self._setup_viewer(args)

        if args.auto_arm:
            self._arm('--auto-arm', move_to_initial_pose=False)

    _WARMUP_PALM = dict(
        position=[0.5, 0.0, 1.0],
        x_axis=[1.0, 0.0, 0.0],
        y_axis=[0.0, 1.0, 0.0],
    )

    # ウォームアップ中だけ固定する numpy の乱数シード (後で元に戻す)。
    _WARMUP_SEED = 0

    def _warmup_ik(self):
        """左右の腕で IK と軌道最適化をダミー目標で 1 回ずつ解き、JAX の
        トレース・コンパイルを起動時に済ませる。

        ``_solve_handshake`` の呼び出しと引数を完全に一致させること
        (1 つでも違うと別関数としてトレースし直され、キャッシュミスする)。
        """
        random_state = np.random.get_state()
        np.random.seed(self._WARMUP_SEED)
        try:
            self._warmup_ik_body()
        finally:
            np.random.set_state(random_state)

    def _warmup_ik_body(self):
        """``_warmup_ik`` の本体 (乱数固定は呼び出し側が行う)。"""
        args = self.args
        print('[warmup] 左右の腕の IK・軌道最適化トレースを事前に実行して '
              'います (数秒~数十秒かかります)...')
        warmup_t0 = time.time()
        collision_obstacles = (
            [] if (args.no_human_collision or self.collision_pairs is None)
            else spik.human_body_obstacles({}))
        warmup_human_xy = np.array([args.human_front_distance, 0.0])
        motion_args = copy.copy(args)
        motion_args.collision_verify_tolerance = \
            args.motion_collision_verify_tolerance
        motion_args.force_optimize = True
        target_pos = spik.palm_target_position(self._WARMUP_PALM)
        # 差し出し手は実際の自動割り当てで各腕が担当する側に合わせる。
        robot_arm_to_hand = {arm: hand
                             for hand, arm in spik.DEFAULT_ROBOT_ARM.items()}
        for robot_arm, label in (('l', '左'), ('r', '右')):
            hand = robot_arm_to_hand[robot_arm]
            t0 = time.time()
            picked, _, _ = spik.solve_person_ik(
                self.robot, self._WARMUP_PALM, hand, robot_arm,
                collision_obstacles,
                attempts_per_pose=args.attempts_per_pose,
                base_limits=self.base_limits,
                self_collision=(not args.no_self_collision
                                and self.collision_pairs is not None),
                collision_pairs=self.collision_pairs,
                joint_positions={},
                verification_pairs=self.verification_pairs)
            log_debug('[warmup] {}腕: IK {:.1f} 秒'.format(
                label, time.time() - t0))
            if picked is None:
                print('[warmup] {}腕: ダミー目標の IK が解けなかったため '
                      '軌道最適化のトレースはスキップします。'.format(label))
                continue
            turn_index, angle_vector, base_pose, post_process_result = picked
            rots = spik.palm_to_target_rots(self._WARMUP_PALM, hand, robot_arm)
            handshake = spik.solved_result(
                self.robot, robot_arm, target_pos, rots[turn_index],
                turn_index, angle_vector, base_pose, self.base_limits,
                post_process_result, 0.0, 0.0, hand, self._WARMUP_PALM)
            t0 = time.time()
            phm.plan_person_motion(
                self.robot, robot_arm, handshake, {}, warmup_human_xy,
                motion_args, self.motion_verification_pairs, self.solver)
            log_debug('[warmup] {}腕: 軌道最適化 {:.1f} 秒'.format(
                label, time.time() - t0))
            if args.side_by_side_transition:
                # 横並び移動のバッチ IK (回転の拘束が違い別にコンパイルされる)。
                t0 = time.time()
                sbs.warmup_batch_ik(
                    self.robot, robot_arm, hand, angle_vector,
                    self.collision_pairs, self.base_limits)
                log_debug('[warmup] {}腕: 横並び移動のバッチ IK {:.1f} 秒'
                          .format(label, time.time() - t0))
        print('[warmup] 完了しました ({:.1f} 秒)。'.format(
            time.time() - warmup_t0))

    def _setup_viewer(self, args):
        """viser ビューアに ARM/RESET ボタン・状態表示・waypoint スライダー・
        ロボットモデルを準備する。"""
        self.viewer = ViserViewer(draw_grid=True)
        # 干渉 overlay は指ありで作る (指先まで含めた事後検証・表示用)。
        # GUI コールバックから参照されるので登録より前に作る。
        self.robot_collision_overlay = build_robot_collision_overlay(
            self.display_robot)
        sync_robot_collision_overlay(
            self.robot_collision_overlay, self.display_robot)
        # 画面表示用の指ありでの総当たりペア (IK 用の verification_pairs とは別)。
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
            self._arm('ARM ボタン')

        @self.reset_button.on_click
        def _on_reset(_):  # noqa: ANN001
            self._reset_view()
            self.state = 'idle'
            self._retry_reference = None
            self._latest_offer_selection = None
            print('[RESET] 骨格表示とロボットの姿勢を初期状態に戻しました。')

        self._status_text = self.viewer._server.gui.add_markdown('')
        # 直近の軌道の waypoint をコマ送り/自動再生する。
        self.waypoint_slider = self.viewer._server.gui.add_slider(
            'waypoint', min=0, max=0, step=1, initial_value=0)
        self.play_checkbox = self.viewer._server.gui.add_checkbox(
            'Play', initial_value=False)

        @self.waypoint_slider.on_update
        def _on_waypoint(_):  # noqa: ANN001
            # redraw は _apply_current_waypoint 内で lock 付きで行う。
            self._apply_current_waypoint()

        @self.play_checkbox.on_update
        def _on_play_toggle(_):  # noqa: ANN001
            # 最終 waypoint にいるときは先頭へ巻き戻してから再生する。
            if (self.play_checkbox.value
                    and int(self.waypoint_slider.value)
                    >= self.waypoint_slider.max):
                self.waypoint_slider.value = 0

        threading.Thread(target=self._play_loop, daemon=True).start()

        # 干渉回避用の半透明モデル (ロボット・人体) の表示切り替え。
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
        # ロボットの初期位置を示す常時表示の Axis。
        initial_pose_axis = Axis(axis_length=INITIAL_POSE_AXIS_LENGTH,
                                 axis_radius=INITIAL_POSE_AXIS_RADIUS)
        initial_pose_axis.newcoords(self._initial_base_coords.copy_worldcoords())
        self.viewer.add(initial_pose_axis)
        # ロボットの add はボタン等の GUI を追加した後に行う (add で関節
        # スライダーが自動追加されるため)。
        self.viewer.add(self.display_robot)
        self.viewer.add(self.robot_collision_overlay)
        # 自動で付く関節スライダー・障害物 GUI は使わないので消す。
        remove_joint_angle_gui(self.viewer)
        remove_obstacles_gui(self.viewer)
        self.viewer.show(open_browser=not args.no_open_browser)
        if not args.no_wait_for_client:
            viewer_nav.wait_for_client(self.viewer, args.client_wait_timeout)

    def _set_link_visible(self, link, visible):
        common_set_link_visible(self.viewer, link, visible)

    def _attach_camera_optical_coords(self, timeout=3.0):
        """視線 IK で使うカメラ光軸を TF (head_link -> optical frame) から
        ``self.robot`` に取り付ける。TF が引けなければ既定値を使う。"""
        head_frame = self.robot.head_link.name
        optical_frame = self.args.camera_optical_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                head_frame, optical_frame, rospy.Time(0),
                rospy.Duration(timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            spik.attach_camera_optical_coords(self.robot)
            print('[camera-optical][WARN] TF ({} -> {}) が {:.1f} 秒以内に '
                  '引けなかったため、視線 IK のカメラ光軸は既定値 '
                  '(launch/decompress.launch の head_to_camera_link) を '
                  '使います: {}'.format(head_frame, optical_frame, timeout, e))
            return
        # 既定値の光軸 (head_link 系) と比べてずれを表示する。
        default_coords = spik.attach_camera_optical_coords(self.robot)
        default_axis = self.robot.head_link.worldrot().T.dot(
            default_coords.worldrot()[:, 2])
        matrix = transform_to_matrix(transform.transform)
        spik.attach_camera_optical_coords(
            self.robot, pos=matrix[:3, 3], rot=matrix[:3, :3])
        diff_deg = math.degrees(math.acos(float(np.clip(
            np.dot(default_axis, matrix[:3, 2]), -1.0, 1.0))))
        log_debug('[camera-optical] 視線 IK のカメラ光軸を TF ({} -> {}) から '
              '設定しました (pos=[{:.4f}, {:.4f}, {:.4f}]、既定値との光軸の '
              'ずれ {:.2f} 度)。'.format(head_frame, optical_frame,
                                     *(list(matrix[:3, 3]) + [diff_deg])))

    def _resolve_robot_position(self):
        """差し出し手判定の基準にするロボット手先の base_link 座標を返す。

        ``--robot-hand-position`` > TF > 右腕の種の姿勢の手先位置 の順。
        """
        if self.args.robot_hand_position is not None:
            return np.asarray(self.args.robot_hand_position, dtype=np.float64)
        return lookup_frame_position(
            self.tf_buffer, self.args.base_frame, self.args.robot_hand_frame,
            self._robot_hand_position_fallback,
            warn_label='[run-camera-pipeline-test] ')

    def _compute_robot_hand_position_fallback(self):
        """TF 未解決時のフォールバック (右腕の種の姿勢の手先位置) を計算する。
        ``self.robot`` の姿勢を書き換えるが IK は毎回作り直すので影響しない。"""
        spik.seed_arm_pose(self.robot, 'r')
        return np.asarray(self.robot.rarm_end_coords.worldpos(),
                          dtype=np.float64)

    def _lookup_camera_to_base(self, header):
        """画像の ``header`` から base_link への TF を引く。"""
        return lookup_camera_to_base(
            self.tf_buffer, self.args.base_frame, header)

    # ------------------------------------------------------------------
    # camera callback
    # ------------------------------------------------------------------
    def _on_camera_info(self, msg):
        self._latest_camera_info = msg

    def _on_frame(self, color_msg, depth_msg):
        if self._busy:
            return
        info_msg = self._latest_camera_info
        if info_msg is None:
            rospy.logwarn_throttle(
                5.0, '{} をまだ受信していないため、画像を処理しません。'.format(
                    self.args.camera_info_topic))
            return
        # TF が引けなければカメラ座標系のままプレビューする (IK は base_link
        # に変換できたフレームでのみ)。
        transform = self._lookup_camera_to_base(color_msg.header)
        camera_to_base = (None if transform is None
                          else transform_to_matrix(transform.transform))

        color = imgmsg_to_ndarray(color_msg, desired_encoding='bgr8')
        depth_raw = imgmsg_to_ndarray(depth_msg)
        depth_m = PeoplePoseEstimator.depth_to_meters(
            depth_raw, encoding=depth_msg.encoding)
        intrinsics = CameraIntrinsics.from_matrix(info_msg.K)

        people, joints_2d = self.pose_estimator.estimate_3d(
            color, depth_m, intrinsics, output_transform=camera_to_base)

        if self.skeleton_image_pub.get_num_connections() > 0:
            overlay = (skeleton_drawing.draw_skeleton_overlay(color, joints_2d[0])
                      if joints_2d else color)
            self.skeleton_image_pub.publish(
                ndarray_to_imgmsg(overlay, 'bgr8', color_msg.header))
        is_base_frame = camera_to_base is not None
        raw_joint_positions = people[0] if people else None
        # 時刻はカメラ画像の stamp。座標系が変わると履歴はリセットされる。
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
            hand_frames_requested = self._hand_frames_requested
        if hand_frames_requested and is_base_frame:
            hands = self.pose_estimator.estimate_hands_3d(
                color, depth_m, intrinsics, output_transform=camera_to_base)
            with self._lock:
                if self._hand_frames_requested:
                    # 画像保存用に画像・カメラ姿勢・内部パラメータも持つ。
                    self._hand_frames.append(dict(
                        hands=hands, color=color, depth=depth_m,
                        camera_to_base=camera_to_base, intrinsics=intrinsics,
                        stamp=color_msg.header.stamp.to_sec()))

        if self.state == 'armed' and armed_joint_positions is not None:
            self.offered_hand_selector.robot_position = \
                self._resolve_robot_position()
            self._try_handshake(armed_joint_positions,
                                color_msg.header.stamp.to_sec())

    def _arm(self, reason, move_to_initial_pose=True, wait_offer_change=False):
        """ARMED にする (差し出し手を時間制限なく待つ)。

        ``move_to_initial_pose``: 実機があれば別スレッドで初期姿勢へ戻す。
        ``wait_offer_change``: 手を動かすまで次の差し出しを受け付けない。
        """
        self._latest_offer_selection = None
        self._retry_reference = (
            dict(palms={}, lost_since={}, until=None,
                 cooldown=RETRY_COOLDOWN_BASE + RETRY_COOLDOWN_PER_CHAR * len(
                     self._last_failure_speech or ''))
            if wait_offer_change else None)
        self._last_failure_speech = None
        with self._lock:
            self._handshake_total_time = None
        self.arm_button.visible = False
        self.reset_button.visible = True
        self.state = 'armed'
        if move_to_initial_pose and self.ri is not None:
            threading.Thread(
                target=self._move_to_initial_pose, daemon=True).start()
        print('[ARM] ARMED になりました ({})。手を差し出してください{}。'.format(
            reason, ' (今の手を動かすか向きを変えてから)'
            if wait_offer_change else ''))

    def _reset_view(self):
        """IK・軌道の結果表示を消して画面を初期位置に戻す (状態は変えない)。"""
        with self._lock:
            self._frozen_joint_positions = None
            self._current_result = None
            self._current_motion = None
            self._display_waypoints = None
            self._display_n_prepend = 0
            self._display_n_approach = 0
            self._display_n_transition = 0
            self._handshake_total_time = None
        self.play_checkbox.value = False
        self._set_waypoint_slider_range(0)
        self.reset_button.visible = False
        self.arm_button.visible = True
        with self._viewer_lock:
            for obstacle_link in self._current_obstacle_links:
                self.viewer.delete(obstacle_link)
            self._current_obstacle_links = []
            self.viewer.redraw()

    def _offer_changed(self, palms, stamp):
        """ARMED に戻ってから最初に見えた掌を基準に、どちらかの手が
        動いた/傾いた/一定時間見えなくなったら True。"""
        reference = self._retry_reference
        if reference['until'] is None:
            reference['until'] = stamp + reference['cooldown']
        if stamp < reference['until']:
            return False
        for side in ('R', 'L'):
            palm = palms.get(side)
            ref = reference['palms'].get(side)
            if palm is None:
                if ref is None:
                    continue
                since = reference['lost_since'].setdefault(side, stamp)
                if stamp - since >= RETRY_LOST_TIME:
                    print('[ARM] {}手が見えなくなったので、次の差し出しを'
                          '受け付けます。'.format(side))
                    return True
                continue
            reference['lost_since'].pop(side, None)
            if ref is None:
                reference['palms'][side] = palm
                continue
            displacement = float(np.linalg.norm(
                np.asarray(palm['position']) - np.asarray(ref['position'])))
            cos = float(np.dot(
                hand_offer_advice._unit(palm['y_axis']),
                hand_offer_advice._unit(ref['y_axis'])))
            rotation = math.degrees(math.acos(max(-1.0, min(1.0, cos))))
            if (displacement >= RETRY_MIN_DISPLACEMENT
                    or rotation >= RETRY_MIN_ROTATION_DEG):
                print('[ARM] {}手が動いたので (位置 {:.0f} mm、向き {:.0f} 度)、'
                      '次の差し出しを受け付けます。'.format(
                          side, displacement * 1e3, rotation))
                return True
        return False

    def _try_handshake(self, joint_positions, stamp):
        handshake_t0 = time.time()
        # stamp は静止判定に使う。
        palms = self.palm_estimator.estimate(joint_positions, t=stamp)
        # 画面表示用に判定の内訳 (スコア/veto 理由) を取り直す。
        selection = self.offered_hand_selector.select(
            joint_positions, palms, t=stamp)
        with self._lock:
            self._latest_offer_selection = selection

        if self._retry_reference is not None:
            if not self._offer_changed(palms, stamp):
                return
            self._retry_reference = None
        offered_hand = palms['offered_hand']
        if offered_hand is None:
            return
        self._busy = True
        if self.ri is not None:
            # うなずきは軌道計画と並行に行う。
            self._nod_done.clear()
            threading.Thread(target=self._nod_head, daemon=True).start()
        with self._lock:
            self._frozen_joint_positions = joint_positions
        self.state = 'solving'
        self.arm_button.visible = False
        self.reset_button.visible = True
        outcome = None
        try:
            outcome = self._solve_handshake(joint_positions, palms, offered_hand)
        finally:
            self._busy = False
            self.state = 'result'
            with self._lock:
                self._handshake_total_time = time.time() - handshake_t0
        if outcome == 'execute':
            # カメラのコールバックを止めないよう別スレッドで実行する。
            threading.Thread(
                target=self._execute_on_robot, daemon=True).start()
        elif outcome == 'failed':
            # 実機は動いていないので ARMED のまま次の差し出しを待つ。
            self._reset_view()
            self._arm('失敗', move_to_initial_pose=False,
                      wait_offer_change=True)

    def _log_debug(self, record):
        """デバッグ用 ``{"event": ...}`` を JSON 1 行でログファイルにだけ書く。"""
        log_debug('[debug] ' + json.dumps(record, ensure_ascii=False))

    def _solve_handshake(self, joint_positions, palms, offered_hand):
        args = self.args
        self._attempt_count += 1
        attempt = self._attempt_count
        log_path = switch_log_file(attempt)
        print('[ARMED] 試行{}: {}手を差し出しました。IK・軌道計画を始めます '
              '(ログ: {})。'.format(attempt, offered_hand, log_path))
        self._log_debug(dict(event='armed', person=attempt,
                             offered_hand=offered_hand))
        robot_arm = (spik.DEFAULT_ROBOT_ARM[offered_hand]
                    if args.robot_arm == 'auto' else args.robot_arm)
        palm = palms[offered_hand]

        offset = spik.human_translation_offset(
            joint_positions, front_distance=args.human_front_distance)
        translated_joints = spik.translate_joint_positions(
            joint_positions, offset)
        translated_palm = spik.translate_palm(palm, offset)
        # 関節は体の表面の点なので、干渉判定だけは体幹を奥へずらした骨格を
        # 使う (立ち位置・向きの判定は translated_joints)。
        collision_joints = spik.shift_torso_joints_from_surface(
            translated_joints, offset, args.torso_surface_offset)
        collision_obstacles = (
            [] if (args.no_human_collision or self.collision_pairs is None)
            else spik.human_body_obstacles(collision_joints))

        # 差し出し手の側・人の正面方向に合わせた台車の可動域。
        person_base_limits = self.base_limits
        side_sign = spik.offered_hand_side_sign(
            offered_hand, translated_joints, translated_palm)
        if side_sign is not None:
            person_base_limits = [
                person_base_limits[0],
                spik.restrict_base_y_range_to_hand_side(
                    person_base_limits[1], side_sign),
                person_base_limits[2]]
        human_yaw = spik.human_facing_yaw(translated_joints)
        if human_yaw is not None:
            person_base_limits = [
                person_base_limits[0],
                person_base_limits[1],
                spik.restrict_base_yaw_range_to_human_facing(
                    person_base_limits[2], human_yaw,
                    margin=math.radians(
                        spik.DEFAULT_BASE_YAW_FACING_MARGIN_DEG))]
        # 台車の前後位置を人の立ち位置付近に絞る (解けなければ窓を広げる)。
        standing_xy = spik.human_standing_xy(translated_joints)
        standing_x = None if standing_xy is None else float(standing_xy[0])

        target_pos = spik.palm_target_position(translated_palm)
        rots = spik.palm_to_target_rots(translated_palm, offered_hand, robot_arm)
        (picked, collision_ik_time, candidate_selection_time,
         person_base_limits, x_margin) = \
            spik.solve_person_ik_side_by_side(
                self.robot, translated_palm, offered_hand, robot_arm,
                collision_obstacles, person_base_limits, standing_x,
                x_margins=args.base_x_standing_margins,
                front_offset_weight=args.front_offset_weight,
                facing_yaw_weight=args.facing_yaw_weight,
                attempts_per_pose=args.attempts_per_pose,
                self_collision=(not args.no_self_collision
                                and self.collision_pairs is not None),
                collision_pairs=self.collision_pairs,
                joint_positions=collision_joints,
                placement_joint_positions=translated_joints,
                verification_pairs=self.verification_pairs)

        if picked is None:
            result = spik.unsolved_result(
                self.robot, robot_arm, target_pos, rots[-1],
                person_base_limits, collision_ik_time,
                candidate_selection_time, offered_hand, translated_palm)
        else:
            turn_index, angle_vector, base_pose, post_process_result = picked
            result = spik.solved_result(
                self.robot, robot_arm, target_pos, rots[turn_index],
                turn_index, angle_vector, base_pose, person_base_limits,
                post_process_result, collision_ik_time,
                candidate_selection_time, offered_hand, translated_palm)
        result['offered_hand'] = offered_hand
        result['robot_arm'] = robot_arm
        result['base_x_standing_margin'] = x_margin

        # 押し込み姿勢まで解けたら軌道計画する。IK と同じ平行移動後の座標系
        # のまま (untranslate する前に) 行うこと。
        motion = None
        if result['solved'] and result.get('post_process') is None:
            print('[motion] 試行{}: 押し込み姿勢 (押し付け/視線 IK) が解けな'
                  'かったため、軌道計画をしません。'.format(attempt))
        elif result['solved']:
            human_xy = spik.human_standing_xy(translated_joints)
            if human_xy is None:
                human_xy = np.array([args.human_front_distance, 0.0])
            # 同名フラグは画面表示用なので、軌道用の許容誤差に差し替えて渡す。
            motion_args = copy.copy(args)
            motion_args.collision_verify_tolerance = \
                args.motion_collision_verify_tolerance
            # 実機の現在地 (base_link 原点) を平行移動後の座標系で渡す。
            initial_base_pose = np.array([offset[0], offset[1], 0.0])
            motion = phm.plan_person_motion(
                self.robot, robot_arm, result, collision_joints, human_xy,
                motion_args, self.motion_verification_pairs, self.solver,
                initial_base_pose=initial_base_pose)
            self._log_debug(dict(
                event='motion', person=attempt,
                verified=motion['verified'],
                lead_in_verified=motion['lead_in_verified'],
                min_dist=float(min(motion['waypoint_min_distances'])),
                waypoint_min_distances=[
                    float(d) for d in motion['waypoint_min_distances']],
                kind=phm.KIND_LABELS.get(motion['kind'], motion['kind']),
                compute_time=motion['compute_time']))
            # 押し込み後に手を下ろしながら横並びになる区間 (実行可能なときだけ)。
            if (args.side_by_side_transition
                    and result.get('post_process') is not None
                    and self._motion_verified(motion)):
                transition = sbs.plan_transition(
                    self.robot, robot_arm, offered_hand,
                    result['post_process'], result['turn_deg'],
                    translated_joints, collision_joints,
                    self.verification_pairs,
                    self.collision_pairs, self.base_limits)
                motion['transition'] = transition
                print('[transition] 試行{}: 横並び移動 {}'.format(
                    attempt, sbs.transition_summary(transition)))
                self._log_debug(dict(
                    event='transition', person=attempt,
                    verified=transition['verified'],
                    fraction=transition['fraction'],
                    x_margin=transition['x_margin'],
                    placement_before=transition['placement_before'],
                    placement_after=transition['placement_after'],
                    reason=transition['reason'],
                    compute_time=transition['compute_time']))

        # 最終の台車位置と人の位置関係 (untranslate の前に求める)。
        placement = (spik.base_placement_metrics(
            translated_joints, result['base_position'], result['base_yaw'])
            if result['solved'] else None)

        self._untranslate_result(result, offset)
        if motion is not None:
            self._untranslate_motion(motion, offset)

        self._log_debug(dict(
            event='result', person=attempt, offered_hand=offered_hand,
            robot_arm=robot_arm, solved=result['solved'],
            base_position=[result['base_position'][0],
                          result['base_position'][1]],
            base_x_standing_margin=x_margin,
            placement=placement,
            collision_ik_time=collision_ik_time,
            candidate_selection_time=candidate_selection_time))
        if not result['solved']:
            print('[result] 試行{} ({}手/{}腕): IK 失敗 ({:.1f} 秒)。'.format(
                attempt, offered_hand, robot_arm,
                collision_ik_time + candidate_selection_time))
        elif motion is None:
            print('[result] 試行{} ({}手/{}腕): IK 成功、軌道計画なし{}。'.format(
                attempt, offered_hand, robot_arm,
                ' (押し込み姿勢が解けない)'
                if result.get('post_process') is None else ''))
        else:
            print('[result] 試行{} ({}手/{}腕): IK 成功、軌道計画{} '
                  '({}、IK {:.1f} 秒 + 軌道 {:.1f} 秒)。'.format(
                      attempt, offered_hand, robot_arm,
                      '成功' if self._motion_verified(motion)
                      else '失敗 (干渉検証 NG)',
                      phm.KIND_LABELS.get(motion['kind'], motion['kind']),
                      collision_ik_time + candidate_selection_time,
                      motion['compute_time']))

        # 軌道が無ければ後処理前の姿勢を 1 waypoint だけ表示する。
        if motion is not None:
            display_waypoints, n_approach = build_display_waypoints(
                motion, result)
            # 初期位置から接近開始位置までの直進 (lead-in) を先頭に継ぎ足す。
            initial_pos = self._initial_base_coords.worldpos()
            prepend_waypoints = phm.build_lead_in_waypoints(
                [initial_pos[0], initial_pos[1], 0.0],
                motion['waypoints'][0], motion['joint_names'],
                start_joint_angles=dict(zip(
                    self._initial_joint_names,
                    self._initial_joint_angle_vector)))
            n_prepend = len(prepend_waypoints)
            display_waypoints = prepend_waypoints + display_waypoints
            n_transition = len(transition_waypoints(motion))
        else:
            display_waypoints, n_prepend, n_approach = None, 0, 0
            n_transition = 0
        with self._lock:
            self._current_result = result
            self._current_motion = motion
            self._current_palm = palm
            self._display_waypoints = display_waypoints
            self._display_n_prepend = n_prepend
            self._display_n_approach = n_approach
            self._display_n_transition = n_transition
        # 実機を動かすのは、押し込み姿勢まで解けて軌道が干渉検証に通ったときだけ。
        has_post_process = result.get('post_process') is not None
        executable = (
            args.auto_execute and self.ri is not None and result['solved']
            and has_post_process
            and motion is not None and display_waypoints is not None
            and self._motion_verified(motion))
        if motion is not None and not self._motion_verified(motion):
            print('[execute] 軌道の干渉検証に通らなかったため (verified={}, '
                  'lead_in_verified={})、--auto-execute でも実機を動かし'
                  'ません。'.format(
                      motion['verified'], motion['lead_in_verified']))
        if result['solved'] and not has_post_process:
            print('[execute] 後処理 (押し付け/視線 IK) が解けず押し込み区間が'
                  'ないため、--auto-execute でも実機を動かしません。')
        if display_waypoints is not None:
            self._set_waypoint_slider_range(len(display_waypoints) - 1)
        else:
            # 姿勢の反映は _apply_current_waypoint が lock 付きで行う。
            self._set_waypoint_slider_range(0)

        if args.save_dir:
            self._save_attempt(joint_positions, palms, result, motion)

        # 戻り値: 'execute' (実機で実行) / 'failed' (--auto-execute で失敗、
        # ARMED に戻る) / 'solved' (それ以外、RESET 待ち)。
        if executable:
            print('[auto-execute] IK・軌道計画が成功したため実機を動かします '
                  '(--auto-execute)。')
            return 'execute'
        if not (result['solved'] and has_post_process
                and motion is not None and self._motion_verified(motion)):
            speech = self._failure_speech(
                attempt, result, motion, joint_positions, palm, offered_hand)
            if args.auto_execute:
                self._last_failure_speech = speech
                self._say(speech)
                return 'failed'
        return 'solved'

    def _failure_speech(self, attempt, result, motion, joint_positions, palm,
                        offered_hand):
        """実機を動かせなかったときに発話する文を返す。

        手の出し方が原因なら reachability map で近い解ける出し方との差を
        助言し、接近途中の干渉なら ``--speech-approach-fail-text`` を返す。
        """
        args = self.args
        cause = self._failure_cause(result, motion)
        if cause == 'approach':
            print('[advice] 試行{}: 接近の途中で干渉したため、手の出し方は'
                  '助言しません。'.format(attempt))
            self._log_debug(dict(event='advice', person=attempt, cause=cause))
            return args.speech_approach_fail_text
        measure = hand_offer_advice.measure_offer(
            joint_positions, palm, offered_hand)
        advice, target = [], None
        if self.offer_advisor is not None and measure is not None:
            advice, target = self.offer_advisor.advise(measure, offered_hand)
        text = hand_offer_advice.advice_speech(advice)
        if measure is not None:
            print('[advice] 試行{} ({}): 今の出し方 前{forward:.2f} 外{lateral:.2f} '
                  '高さ{height:.2f} m、指先 yaw{yaw:.0f} pitch{pitch:.0f}、'
                  'ひねり{roll:.0f} 度 -> {}'.format(
                      attempt, cause, text or '(助言なし)', **measure))
        self._log_debug(dict(
            event='advice', person=attempt, cause=cause, measure=measure,
            target=target, advice=[[k, d] for k, d, _ in advice], text=text))
        tail = text or args.speech_retry_text
        return '。'.join(t for t in (args.speech_fail_text, tail) if t)

    @staticmethod
    def _failure_cause(result, motion,
                       near_hand_fraction=FAILURE_NEAR_HAND_FRACTION):
        """実機を動かせなかった原因を ``ik``/``no_press``/``near_hand``
        (干渉 NG が押し込み付近だけ)/``approach`` に分類する。"""
        if not result['solved']:
            return 'ik'
        if result.get('post_process') is None:
            return 'no_press'
        if motion is None or not motion['lead_in_verified']:
            return 'approach'
        distances = motion['waypoint_min_distances']
        margins = motion.get('waypoint_human_clearance_margins') or \
            [None] * len(distances)
        failed = [i for i, (d, c) in enumerate(zip(distances, margins))
                  if not phm.motion_passes([d], [c])]
        start = near_hand_fraction * (len(distances) - 1)
        if failed and min(failed) >= start:
            return 'near_hand'
        return 'approach'

    @staticmethod
    def _motion_verified(motion):
        """経路計画の区間と lead-in の両方が干渉検証に通っているか。"""
        return bool(motion['verified']) and bool(motion['lead_in_verified'])

    @staticmethod
    def _untranslate_result(result, offset):
        """人物を平行移動した仮想座標系の ``result`` を実際の base_link 系へ
        戻す (破壊的)。忘れると仮想の人物に向かって台車が動く。"""
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
        """``motion`` の各 waypoint の台車位置を実際の base_link 系へ戻す (破壊的)。"""
        dx, dy = offset
        transition = motion.get('transition') or {}
        for wp in (motion['waypoints'] + motion.get('lead_in_waypoints', [])
                   + transition.get('waypoints', [])):
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
    # waypoint スライダー/Play (直近 1 件の軌道だけを扱う)
    # ------------------------------------------------------------------
    def _set_waypoint_slider_range(self, max_index):
        """スライダーの範囲を ``[0, max_index]`` にし、waypoint 0 を表示する。"""
        self.waypoint_slider.max = max_index
        # 既に 0 なら on_update が発火しないので明示的に呼ぶ。
        self.waypoint_slider.value = 0
        self._apply_current_waypoint()

    def _apply_current_waypoint(self):
        """スライダーの現在値の姿勢 (軌道が無ければ IK 結果か初期位置) を
        ``self.display_robot`` に反映する。"""
        with self._lock:
            result = self._current_result
            motion = self._current_motion
            display_waypoints = self._display_waypoints
        # 姿勢の更新は複数ステップにまたがるので、途中の不整合な姿勢を
        # spin() 側の redraw に描かれないよう redraw まで lock 内で行う。
        with self._viewer_lock:
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
            self.viewer.redraw()
        self._refresh_collision_pairs_text()

    def _refresh_collision_pairs_text(self):
        """表示中の姿勢で指ありの事後検証をやり直し、表示用の文字列を更新する
        (表示専用。人体は画面に出している Cylinder をそのまま使う)。"""
        colliding = colliding_link_pairs(
            self.robot_collision_overlay, self.hand_verification_pairs,
            self._current_obstacle_links,
            tolerance=self.args.collision_verify_tolerance)
        self._collision_pairs_text = collision_pairs_text(colliding)

    # ------------------------------------------------------------------
    # 実機動作 (初期姿勢への復帰、うなずき、--auto-execute)
    # ------------------------------------------------------------------
    def _move_to_initial_pose(self):
        """実機の全身を初期姿勢 (両腕を下ろした姿勢) へ動かす。

        初期姿勢に含まれない関節 (指など) は実機の現在値のままにする
        (``self.real_robot`` は未同期なので先に現在値を反映する)。
        """
        try:
            current_av = self.ri.angle_vector()
        except RuntimeError as exc:
            print('[ARM] 実機の関節角を取得できなかったため、姿勢を初期化 '
                  'する動作をスキップしました ({})。'.format(exc))
            return
        self.real_robot.angle_vector(current_av)
        name_to_initial_angle = dict(
            zip(self._initial_joint_names, self._initial_joint_angle_vector))
        for joint in self.real_robot.joint_list:
            if joint.name in name_to_initial_angle:
                joint.joint_angle(name_to_initial_angle[joint.name])
        target_av = self.real_robot.angle_vector()
        # 速度上限を超えない時間まで延ばす。
        move_time, = self._limited_time_list(
            [target_av], None, [ARM_INITIAL_POSE_MOVE_TIME])
        self.ri.angle_vector(target_av, move_time)
        print('[ARM] 腕・首を初期姿勢に戻しました。')

    def _nod_head(self):
        """首を ``HEAD_NOD_PITCH_DEG`` まで下げて初期角度に戻す (うなずく)。
        終わったら (失敗しても) ``self._nod_done`` を set する。
        ``head_controller`` だけに送るので腕のゴールは打ち切らない。"""
        try:
            current_av = np.asarray(self.ri.angle_vector(), dtype=np.float64)
            joint_list = self.real_robot.joint_list
            neck_index = joint_list.index(self.real_robot.neck_p_joint)
            initial_angle = dict(zip(
                self._initial_joint_names,
                self._initial_joint_angle_vector))[
                    self.real_robot.neck_p_joint.name]
            down_av = current_av.copy()
            down_av[neck_index] = np.deg2rad(HEAD_NOD_PITCH_DEG)
            up_av = current_av.copy()
            up_av[neck_index] = initial_angle
            time_list = self._limited_time_list(
                [down_av, up_av], None,
                [HEAD_NOD_MOVE_TIME, HEAD_NOD_MOVE_TIME])
            self.ri.angle_vector_sequence(
                [down_av, up_av], time_list, controller_type='head_controller')
            self.ri.wait_interpolation(controller_type='head_controller')
            log_debug('[nod] 首を {:.0f} 度まで下げて {:.0f} 度に戻しました '
                  '({:.2f} 秒)。'.format(HEAD_NOD_PITCH_DEG,
                                        np.rad2deg(initial_angle),
                                        sum(time_list)))
        except Exception as exc:  # noqa: BLE001  (うなずけなくても握手の実行は止めない)
            print('[nod] うなずきに失敗しました ({})。'.format(exc))
        finally:
            self._nod_done.set()

    def _execute_on_robot(self):
        """計画済みの waypoint 列を実機に送る。

        接近区間 (初期位置 -> hover 目標) を 1 ゴールで送り、台車の位置ずれを
        補正してから押し込み区間、必要なら横並び移動を実行する。台車は
        costmap を使わず ``move_trajectory_sequence`` で直接送る (干渉は
        計画側で検証済み)。
        """
        with self._lock:
            result = self._current_result
            motion = self._current_motion
            display_waypoints = self._display_waypoints
            n_prepend = self._display_n_prepend
            n_approach = self._display_n_approach
            n_transition = self._display_n_transition
        if not self.args.auto_execute:
            print('[execute] --auto-execute が指定されていないため実機を '
                  '動かせません。')
            return
        if self.ri is None:
            print('[execute] 実機 (AeroROSRobotInterface) への接続に失敗 '
                  'しているため実機を動かせません。')
            return
        if result is None or not result.get('solved') or motion is None \
               or display_waypoints is None:
            print('[execute] IK・軌道計画が完了していないため実機を動かせ '
                  'ません。')
            return
        if not self._motion_verified(motion):
            print('[execute] 軌道の干渉検証に通っていないため実機を動かせ '
                  'ません (verified={}, lead_in_verified={})。'.format(
                      motion['verified'], motion['lead_in_verified']))
            return

        # うなずきが終わるまで待つ。
        if not self._nod_done.wait(timeout=10.0):
            print('[execute] うなずきが 10 秒以内に終わらなかったため、'
                  '待たずに実行します。')

        joint_names = motion['joint_names']
        # [:reach_boundary] が接近区間 (末尾が hover 目標)、[reach_boundary-1:]
        # が押し込み区間。末尾の n_transition 個は横並び移動。
        transition_waypoints_ = display_waypoints[
            len(display_waypoints) - n_transition:]
        display_waypoints = display_waypoints[
            :len(display_waypoints) - n_transition]
        reach_boundary = min(max(n_prepend + n_approach, 1),
                             len(display_waypoints))
        print('[execute] 実機で waypoint を {} 個実行します '
              '(接近={}個+押し込み={}個+横並び移動={}個)。'.format(
                  len(display_waypoints) + n_transition, reach_boundary,
                  len(display_waypoints) - reach_boundary, n_transition))

        self._say(self.args.speech_start_text)

        # 台車位置補正の基準スキャン (静止中に取る) と照合から除く人の位置。
        ref_scan = None
        exclude_centers = None
        if self.args.base_correction == 'scan':
            ref_scan = self._capture_scan()
            with self._lock:
                frozen = self._frozen_joint_positions
            if frozen:
                exclude_centers = np.array(
                    [p[:2] for p in frozen.values()], dtype=np.float64)

        start_odom_coords, final_traj_point = self._execute_waypoint_segment(
            display_waypoints[:reach_boundary], joint_names)

        if self.args.base_correction != 'none':
            self._correct_base_residual(
                start_odom_coords, final_traj_point, ref_scan=ref_scan,
                exclude_centers=exclude_centers)

        # 腕の追従遅れが解消してから押し込む (hover を経由させるため)。
        robot_arm = result['robot_arm']
        self._wait_joint_settle(
            'hover', display_waypoints[reach_boundary - 1], joint_names,
            robot_arm)

        if reach_boundary < len(display_waypoints):
            press_waypoints = display_waypoints[reach_boundary - 1:]
            if self.args.press_in_refine:
                refined = self._refine_press_in(
                    press_waypoints[0], joint_names, result)
                if refined is not None:
                    press_waypoints = [press_waypoints[0]] + refined
            self._execute_waypoint_segment(press_waypoints, joint_names)
            self._wait_joint_settle(
                '押し込み終了', press_waypoints[-1], joint_names,
                robot_arm)

        self._say(self.args.speech_done_text)
        press_end_time = time.time()
        do_transition = (transition_waypoints_
                         and reach_boundary < len(display_waypoints))

        if (self.args.grasp_capture_duration > 0
                and reach_boundary < len(display_waypoints)):
            # 横並び移動をするときは動き出すまでの画像だけ集める。
            duration = self.args.grasp_capture_duration
            if do_transition:
                duration = min(duration, self.args.side_by_side_delay)
            self._capture_grasp_images(
                press_waypoints[-1], joint_names, result, duration)

        if do_transition:
            wait = press_end_time + self.args.side_by_side_delay - time.time()
            if wait > 0:
                rospy.sleep(wait)
            self._execute_transition(
                press_waypoints[-1], display_waypoints[-1],
                transition_waypoints_, joint_names, robot_arm)

        print('[execute] 実行を終了しました。')

    def _execute_transition(self, press_wp, planned_press_wp, waypoints,
                            joint_names, robot_arm):
        """押し込み後の横並び移動を実行する。

        押し込みを解き直したときの関節角の差 (``press_wp`` -
        ``planned_press_wp``) を各 waypoint に足す。人を連れて動くので
        各区間を ``--side-by-side-segment-time`` 秒以上かける。
        """
        delta = (np.asarray(press_wp['joint_angle_vector'])
                 - np.asarray(planned_press_wp['joint_angle_vector']))
        segment = [press_wp] + [
            dict(wp, joint_angle_vector=[
                float(v) for v in np.asarray(wp['joint_angle_vector'])
                + delta])
            for wp in waypoints]
        self._say(self.args.speech_transition_text)
        print('[execute] 手を下ろしながら横並びの位置へ移動します '
              '(waypoint {} 個)。'.format(len(waypoints)))
        self._execute_waypoint_segment(
            segment, joint_names,
            min_segment_time=self.args.side_by_side_segment_time)
        self._wait_joint_settle(
            '横並び移動終了', segment[-1], joint_names, robot_arm)

    def _collect_hand_frames(self, n_frames, timeout):
        """以降のフレームで手だけの検出を行い、``n_frames`` 枚分 (None なら
        ``timeout`` 秒の間の全て) のフレーム情報のリストを返す。"""
        with self._lock:
            self._hand_frames = []
            self._hand_frames_requested = True
        deadline = time.time() + timeout
        try:
            while not rospy.is_shutdown() and time.time() < deadline:
                with self._lock:
                    if (n_frames is not None
                            and len(self._hand_frames) >= n_frames):
                        break
                rospy.sleep(0.01)
        finally:
            with self._lock:
                self._hand_frames_requested = False
                frames = self._hand_frames
                self._hand_frames = []
        return frames

    def _select_offered_hand(self, hand_frames, side, expected_position,
                             expected_normal):
        """各フレームで掌が ``expected_position`` (今の base_link 系) に最も
        近い手を選び、ランドマークのフレーム間中央値を ``{side}Hand*`` の
        名前で返す。Hands の左右判定は誤りうるので位置で選ぶ。score の低い手・
        法線が ``expected_normal`` から大きく傾いた手は除く。

        Returns: ``(joint_positions or None, distances [m], candidates)``。
        ``candidates`` はフレームごとの全候補の内訳 (ログ・保存用)。
        """
        chosen = []
        distances = []
        candidates = []
        for frame in hand_frames:
            best = None
            frame_candidates = []
            for hand in frame['hands']:
                prefix = '{}Hand'.format(hand['side'])
                joints = {'{}Hand{}'.format(side, name[len(prefix):]): p
                          for name, p in hand['positions'].items()}
                palm = self.palm_estimator.estimate_palm(joints, side)
                dist = normal_angle = rejected = None
                if palm is not None:
                    dist = float(np.linalg.norm(
                        np.asarray(palm['position']) - expected_position))
                    normal_angle = math.degrees(math.acos(np.clip(np.dot(
                        palm['y_axis'], expected_normal), -1.0, 1.0)))
                    if hand['score'] < PRESS_IN_REFINE_MIN_HAND_SCORE:
                        rejected = 'low_score'
                    elif normal_angle > PRESS_IN_REFINE_MAX_NORMAL_ANGLE_DEG:
                        rejected = 'normal'
                frame_candidates.append(dict(
                    side=hand['side'], score=hand['score'],
                    n_points=len(hand['positions']),
                    palm_position=(None if palm is None
                                   else [float(v) for v in palm['position']]),
                    distance=dist, normal_angle_deg=normal_angle,
                    rejected=rejected))
                if (dist is not None and rejected is None
                        and (best is None or dist < best[0])):
                    best = (dist, joints)
            candidates.append(frame_candidates)
            if best is not None and best[0] <= PRESS_IN_REFINE_MAX_HAND_DISTANCE:
                distances.append(best[0])
                chosen.append(best[1])
        if not chosen:
            return None, distances, candidates
        names = [name for name in chosen[0]
                 if sum(name in joints for joints in chosen) * 2
                 >= len(chosen)]
        return ({name: np.median([joints[name] for joints in chosen
                                  if name in joints], axis=0).tolist()
                 for name in names}, distances, candidates)

    def _save_refine_failure(self, hand_frames, expected_position, candidates,
                             record):
        """押し込み直前の手の再検出に失敗したときの画像 (元画像と検出結果・
        計画時の掌を描いたもの) と ``record.json`` を保存する。"""
        root = (os.path.join(self.args.save_dir, 'refine_failures')
                if self.args.save_dir else '/tmp/aero_demo_refine_failures')
        out_dir = os.path.join(root, time.strftime('%Y%m%d_%H%M%S'))
        try:
            os.makedirs(out_dir, exist_ok=True)
            for i, (frame, frame_candidates) in enumerate(
                    zip(hand_frames, candidates)):
                color = frame['color']
                cv2.imwrite(os.path.join(out_dir, 'frame{}_raw.png'.format(i)),
                            color)
                image = color.copy()
                uv, depth = project_base_point(frame, expected_position)
                if uv is not None:
                    radius = int(frame['intrinsics'].fx
                                 * PRESS_IN_REFINE_MAX_HAND_DISTANCE / depth)
                    cv2.drawMarker(image, uv, (0, 0, 255), cv2.MARKER_CROSS,
                                   30, 3)
                    cv2.circle(image, uv, radius, (0, 0, 255), 2)
                    cv2.putText(image, 'expected palm', (uv[0] + 10, uv[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                for hand, cand in zip(frame['hands'], frame_candidates):
                    pixels = list(hand['pixels'].values())
                    for u, v in pixels:
                        cv2.circle(image, (int(u), int(v)), 3, (0, 255, 0), -1)
                    if cand['palm_position'] is not None:
                        palm_uv, _ = project_base_point(
                            frame, np.asarray(cand['palm_position']))
                        if palm_uv is not None:
                            cv2.drawMarker(image, palm_uv, (255, 0, 0),
                                           cv2.MARKER_SQUARE, 16, 2)
                    label = '{} {:.2f} {}'.format(
                        cand['side'], cand['score'],
                        'no palm' if cand['distance'] is None
                        else '{:.0f}mm {:.0f}deg'.format(
                            cand['distance'] * 1e3, cand['normal_angle_deg']))
                    if cand.get('rejected'):
                        label += ' x' + cand['rejected']
                    u0, v0 = pixels[0]
                    cv2.putText(image, label, (int(u0), int(v0) + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(out_dir, 'frame{}.png'.format(i)),
                            image)
            json_io.save_json(os.path.join(out_dir, 'record.json'), dict(
                record, stamps=[frame['stamp'] for frame in hand_frames]))
            log_debug('[execute][refine] 手の再検出に失敗したときの画像を {} に '
                  '保存しました。'.format(out_dir))
        except Exception as exc:  # noqa: BLE001  (保存失敗で実機動作を止めない)
            print('[execute][refine][WARN] 画像の保存に失敗しました ({})。'
                  .format(exc))

    def _capture_grasp_images(self, press_wp, joint_names, result, duration):
        """押し込み終了後 ``duration`` 秒のカメラフレームを、握ったかの判定を
        作る資料として別スレッドで保存する (一時的なデータ収集用)。

        掌・手先の位置は ``press_wp`` の台車位置から見た base_link 系
        (追従遅れ・スリップは含まない)。
        """
        with self._lock:
            planned_palm = self._current_palm
        robot_arm = result['robot_arm']
        self._place_robot_at_waypoint(joint_names, press_wp)
        base = self.robot.base_link
        base_rot = base.worldrot().copy()
        base_pos = base.worldpos().copy()
        end_coords = (self.robot.rarm_end_coords if robot_arm == 'r'
                      else self.robot.larm_end_coords)
        record = dict(
            event='grasp_capture', robot_arm=robot_arm,
            offered_hand=result['offered_hand'],
            robot_hand_position=[float(v) for v in base_rot.T @ (
                end_coords.worldpos() - base_pos)])
        if planned_palm is not None:
            record['expected_palm_position'] = [float(v) for v in base_rot.T @ (
                np.asarray(planned_palm['position']) - base_pos)]
            record['expected_palm_normal'] = [float(v) for v in base_rot.T @
                                              np.asarray(planned_palm['y_axis'])]
        t0 = time.time()
        hand_frames = self._collect_hand_frames(None, duration)
        log_debug('[execute][grasp] 押し込み後の画像を {} フレーム取得しました '
                  '({:.2f}s)。'.format(len(hand_frames), time.time() - t0))
        if hand_frames:
            threading.Thread(
                target=self._save_grasp_capture, args=(hand_frames, record),
                daemon=True).start()

    def _save_grasp_capture(self, hand_frames, record):
        """集めたフレームの元画像・深度 (16bit mm、欠損 0)・注釈画像と
        ``record.json`` を保存する。"""
        root = (os.path.join(self.args.save_dir, 'grasp_captures')
                if self.args.save_dir else '/tmp/aero_demo_grasp_captures')
        out_dir = os.path.join(root, time.strftime('%Y%m%d_%H%M%S'))
        markers = [('expected palm', record.get('expected_palm_position'),
                    (0, 0, 255)),
                   ('robot hand', record['robot_hand_position'],
                    (255, 255, 0))]
        frames = []
        try:
            os.makedirs(out_dir, exist_ok=True)
            for i, frame in enumerate(hand_frames):
                prefix = os.path.join(out_dir, 'frame{:03d}'.format(i))
                cv2.imwrite(prefix + '_raw.png', frame['color'])
                depth_mm = np.nan_to_num(frame['depth'] * 1e3, nan=0.0,
                                         posinf=0.0, neginf=0.0)
                cv2.imwrite(prefix + '_depth.png',
                            np.clip(depth_mm, 0, 65535).astype(np.uint16))
                image = frame['color'].copy()
                for label, point, color in markers:
                    if point is None:
                        continue
                    uv, _ = project_base_point(frame, np.asarray(point))
                    if uv is not None:
                        cv2.drawMarker(image, uv, color, cv2.MARKER_CROSS,
                                       30, 3)
                        cv2.putText(image, label, (uv[0] + 10, uv[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                for hand in frame['hands']:
                    pixels = list(hand['pixels'].values())
                    for u, v in pixels:
                        cv2.circle(image, (int(u), int(v)), 3, (0, 255, 0), -1)
                    if pixels:
                        u0, v0 = pixels[0]
                        cv2.putText(image, '{} {:.2f}'.format(
                            hand['side'], hand['score']),
                            (int(u0), int(v0) + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imwrite(prefix + '.png', image)
                intr = frame['intrinsics']
                frames.append(dict(
                    stamp=frame['stamp'],
                    camera_to_base=np.asarray(frame['camera_to_base']).tolist(),
                    intrinsics=dict(fx=intr.fx, fy=intr.fy, cx=intr.cx,
                                    cy=intr.cy),
                    hands=[dict(side=hand['side'], score=float(hand['score']),
                                positions={k: [float(x) for x in p] for k, p
                                           in hand['positions'].items()})
                           for hand in frame['hands']]))
            json_io.save_json(os.path.join(out_dir, 'record.json'),
                              dict(record, frames=frames))
            print('[execute][grasp] 押し込み後の画像 {} 枚を {} に保存しました。'
                  .format(len(hand_frames), out_dir))
        except Exception as exc:  # noqa: BLE001  (保存失敗で実機動作を止めない)
            print('[execute][grasp][WARN] 画像の保存に失敗しました ({})。'
                  .format(exc))

    def _place_robot_at_waypoint(self, joint_names, waypoint):
        """``self.robot`` を ``waypoint`` の姿勢にする。IK で本体が動かされた
        ままなので先に本体を原点に戻す。"""
        self.robot.newcoords(Coordinates())
        apply_waypoint_pose(self.robot, joint_names, [waypoint], 0)

    def _refine_press_in(self, hover_wp, joint_names, result):
        """hover 到達後に手を検出し直して押し込み姿勢を解き直し、押し込み
        区間の waypoint (``hover_wp`` の次以降) を返す。失敗したら None。

        検出した掌を「台車が hover 目標にいる」前提で計画時の座標系に移す
        ので、台車のスリップと手の動きの両方を腕で吸収する。
        """
        t0 = time.time()
        with self._lock:
            planned_palm = self._current_palm
        offered_hand = result['offered_hand']
        planned_post = result.get('post_process')
        if planned_post is None or planned_palm is None:
            return None
        # 計画時の掌を、hover 目標にいる前提で今の base_link 系に直す。
        self._place_robot_at_waypoint(joint_names, hover_wp)
        base = self.robot.base_link
        base_rot = base.worldrot().copy()
        base_pos = base.worldpos().copy()
        expected_position = base_rot.T @ (
            np.asarray(planned_palm['position']) - base_pos)
        expected_normal = base_rot.T @ np.asarray(planned_palm['y_axis'])
        hand_frames = self._collect_hand_frames(
            PRESS_IN_REFINE_FRAMES, PRESS_IN_REFINE_TIMEOUT)
        joint_positions, distances, candidates = self._select_offered_hand(
            hand_frames, offered_hand, expected_position, expected_normal)
        wait_time = time.time() - t0
        palm = None
        if joint_positions is not None:
            palm = self.palm_estimator.estimate_palm(
                joint_positions, offered_hand)
        if palm is None:
            nearest = [c['distance'] for frame in candidates for c in frame
                       if c['distance'] is not None and c['rejected'] is None]
            n_rejected = {reason: sum(
                1 for frame in candidates for c in frame
                if c['rejected'] == reason)
                for reason in ('low_score', 'normal')}
            print('[execute][refine] hover 到達後に {}手を検出できなかった '
                  'ため、計画どおりに押し込みます ({:.2f}s 待機、{} フレーム'
                  '中 手を検出 {} フレーム、計画時の掌から {:.0f}mm 以内 {} '
                  'フレーム、最も近い手 {}、除外した手 score<{:.2f}: {} / '
                  '法線>{:.0f}deg: {})。'.format(
                      offered_hand, wait_time, len(hand_frames),
                      sum(1 for frame in hand_frames if frame['hands']),
                      PRESS_IN_REFINE_MAX_HAND_DISTANCE * 1e3,
                      len(distances),
                      '{:.0f}mm'.format(min(nearest) * 1e3) if nearest
                      else 'なし', PRESS_IN_REFINE_MIN_HAND_SCORE,
                      n_rejected['low_score'],
                      PRESS_IN_REFINE_MAX_NORMAL_ANGLE_DEG,
                      n_rejected['normal']))
            record = dict(
                event='press_in_refine', reason='no_hand',
                wait_time=wait_time, n_frames=len(hand_frames),
                expected_position=[float(v) for v in expected_position],
                expected_normal=[float(v) for v in expected_normal],
                candidates=candidates)
            self._log_debug(record)
            self._save_refine_failure(hand_frames, expected_position,
                                      candidates, record)
            return None

        t1 = time.time()
        self._place_robot_at_waypoint(joint_names, hover_wp)
        observed = spik.transform_palm(palm, base_rot, base_pos)
        info = spik.refine_post_process(
            self.robot, result['robot_arm'], observed, result['turn_deg'],
            planned_post)
        ik_time = time.time() - t1
        palm_shift = np.asarray(observed['position']) - np.asarray(
            planned_palm['position'])
        self._log_debug(dict(
            event='press_in_refine', reason=info['reason'],
            wait_time=wait_time, ik_time=ik_time,
            n_frames=len(hand_frames), hand_distances=distances,
            expected_position=[float(v) for v in expected_position],
            candidates=candidates,
            palm_shift=[float(v) for v in palm_shift],
            position_change=info['position_change'],
            rotation_change_deg=math.degrees(info['rotation_change'])))
        if info['post_process'] is None:
            print('[execute][refine] 押し込み姿勢を解き直せなかったため '
                  '({}、目標の変化 {:.1f}mm/{:.1f}deg)、計画どおりに押し込み '
                  'ます。'.format(info['reason'],
                                 info['position_change'] * 1e3,
                                 math.degrees(info['rotation_change'])))
            return None
        print('[execute][refine] 検出し直した手に合わせて押し込み姿勢を '
              '補正しました ({}、目標の変化 {:.1f}mm/{:.1f}deg、待機 {:.2f}s '
              '+ IK {:.3f}s)。'.format(
                  info['reason'], info['position_change'] * 1e3,
                  math.degrees(info['rotation_change']), wait_time, ik_time))
        return build_press_in_waypoints(
            hover_wp, joint_names, info['post_process'])

    def _wait_joint_settle(self, label, waypoint, joint_names, robot_arm,
                           tolerance=JOINT_SETTLE_HAND_TOLERANCE,
                           timeout=JOINT_SETTLE_TIMEOUT):
        """実機の腕が ``waypoint`` に追いつく (関節角由来の手先のずれ <=
        ``tolerance``) まで最大 ``timeout`` 秒待つ。時間切れでも止めない
        (接触の負荷で届かないことがあるため)。"""
        start = time.time()
        hand_err, over = self._joint_tracking_error(
            waypoint, joint_names, robot_arm)
        self._print_joint_tracking(
            '{} 到達直後'.format(label), hand_err, over)
        if np.linalg.norm(hand_err) <= tolerance:
            return
        while not rospy.is_shutdown():
            if time.time() - start >= timeout:
                print('[execute][WARN] {}: {:.1f} 秒待っても腕が指令に'
                      '追いつきませんでした (手先のずれ {:.1f}mm > {:.1f}mm)。'
                      'このまま進みます。'.format(
                          label, timeout, np.linalg.norm(hand_err) * 1e3,
                          tolerance * 1e3))
                break
            rospy.sleep(JOINT_SETTLE_POLL_PERIOD)
            hand_err, over = self._joint_tracking_error(
                waypoint, joint_names, robot_arm)
            if np.linalg.norm(hand_err) <= tolerance:
                break
        self._print_joint_tracking(
            '{} 待機後 ({:.2f}s)'.format(label, time.time() - start),
            hand_err, over)

    @staticmethod
    def _print_joint_tracking(label, hand_err, over):
        """``_joint_tracking_error`` の結果を ``[debug][joint]`` ログに出す。"""
        log_debug('[debug][joint] {}: 手先のずれ(関節角のみ由来)={:.1f}mm '
              '(dx={:+.1f} dy={:+.1f} dz={:+.1f}mm, world系), '
              '1deg/5mm超の関節 {}個: {}'.format(
                  label, np.linalg.norm(hand_err) * 1e3,
                  hand_err[0] * 1e3, hand_err[1] * 1e3, hand_err[2] * 1e3,
                  len(over), ', '.join(over) if over else 'なし'))

    def _joint_tracking_error(self, waypoint, joint_names, robot_arm):
        """``waypoint`` の指令関節角と実機の関節角の差を求める。

        Returns: ``(hand_err, over)``。``hand_err`` は関節角だけから求めた
        手先のずれ (指令 - 実機、world 系 [m])、``over`` は 1deg (直動は
        5mm) を超えた関節の表示文字列。``self.real_robot`` の姿勢は戻す。
        """
        saved_av = self.real_robot.angle_vector().copy()
        try:
            actual_av = self.ri.angle_vector().copy()
            name_to_angle = dict(zip(joint_names,
                                     waypoint['joint_angle_vector']))
            for joint in self.real_robot.joint_list:
                if joint.name in name_to_angle:
                    joint.joint_angle(name_to_angle[joint.name])
            target_av = self.real_robot.angle_vector().copy()
            end_coords = getattr(self.real_robot,
                                 '{}arm_end_coords'.format(robot_arm))
            target_hand = end_coords.worldpos().copy()
            self.real_robot.angle_vector(actual_av)
            actual_hand = end_coords.worldpos().copy()
        finally:
            self.real_robot.angle_vector(saved_av)

        controller_joint_names = {
            name for param in self.ri.controller_param_table[
                self.ri.controller_type]
            for name in param['joint_names']}
        diffs = self.ri.sub_angle_vector(target_av, actual_av)
        entries = []  # (閾値で正規化した大きさ, 表示文字列)
        for joint, diff, target, actual in zip(
                self.real_robot.joint_list, diffs, target_av, actual_av):
            if joint.name not in controller_joint_names:
                continue
            if isinstance(joint, LinearJoint):
                entries.append((abs(diff) / 0.005,
                                '{}={:+.1f}mm (指令{:.1f}/実測{:.1f})'.format(
                                    joint.name, diff * 1e3, target * 1e3,
                                    actual * 1e3)))
            else:
                entries.append((abs(diff) / math.radians(1.0),
                                '{}={:+.1f}deg (指令{:.1f}/実測{:.1f})'.format(
                                    joint.name, math.degrees(diff),
                                    math.degrees(target),
                                    math.degrees(actual))))
        entries.sort(reverse=True)
        over = [text for score, text in entries if score > 1.0]
        return target_hand - actual_hand, over

    def _say(self, text):
        """``text`` を発話する (非ブロッキング、発話できなければ何もしない)。"""
        if self.sound_client is None or not text:
            return
        try:
            self.sound_client.say(text, voice=self.args.speech_voice)
        except Exception:  # noqa: BLE001  (発話失敗で実機動作を止めない)
            pass

    def _execute_waypoint_segment(self, waypoints, joint_names,
                                  min_segment_time=MIN_SEGMENT_TIME):
        """``waypoints`` (先頭が現在の姿勢) を 1 つの滑らかな軌道として
        台車・腕を並行に実行する。

        Returns: ``(start_odom_coords, final_traj_point)`` (送信時の odom と
        最終 waypoint の先頭基準 ``[dx, dy, dyaw]``、台車移動なしなら None)。
        """
        arm_angle_vectors = []
        # 先頭 waypoint からの累積移動量 (move_trajectory_sequence は各要素を
        # 開始時の odom から独立に適用するので差分ではない)。
        base_trajectory_points = []
        first_base = None  # (x, y, yaw) この区間の先頭 waypoint の台車位置姿勢 (world 系)
        for wp in waypoints:
            name_to_angle = dict(zip(joint_names, wp['joint_angle_vector']))
            for joint in self.real_robot.joint_list:
                if joint.name in name_to_angle:
                    joint.joint_angle(name_to_angle[joint.name])
            arm_angle_vectors.append(self.real_robot.angle_vector())

            bx, by, byaw = (wp['base_position'][0], wp['base_position'][1],
                           wp['base_yaw'])
            if first_base is None:
                # 実機は今この姿勢にいる前提。
                first_base = (bx, by, byaw)
            x0, y0, yaw0 = first_base
            dx_world = bx - x0
            dy_world = by - y0
            dyaw = byaw - yaw0
            # 実行開始時点の台車の向き基準に回転させる。
            cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)
            dx = cos_yaw * dx_world + sin_yaw * dy_world
            dy = -sin_yaw * dx_world + cos_yaw * dy_world
            base_trajectory_points.append([dx, dy, dyaw])

        # それぞれ 1 ゴールで送る (非ブロッキング)。台車と腕には同じ
        # time_list を渡してタイミングを揃える (ずれると検証済み経路から外れる)。
        time_list = self._limited_time_list(
            arm_angle_vectors, base_trajectory_points,
            [min_segment_time] * len(waypoints))
        start_odom_coords = None
        if arm_angle_vectors:
            self.ri.angle_vector_sequence(arm_angle_vectors, time_list)
        if base_trajectory_points:
            # 補正計算用に送信時の odom を控える。
            start_odom_coords = self.ri.odom
            log_debug('[debug][segment] waypoint数={} 計画上のdyaw={:.1f}deg '
                  '(=[{}]) 送信直前odom_yaw={:.1f}deg '
                  'time_list合計={:.3f}s (=[{}])'
                  .format(
                      len(base_trajectory_points),
                      math.degrees(base_trajectory_points[-1][2]),
                      ', '.join('{:.1f}'.format(math.degrees(p[2]))
                               for p in base_trajectory_points),
                      math.degrees(matrix2ypr(start_odom_coords.rotation)[0]),
                      sum(time_list),
                      ', '.join('{:.2f}'.format(t) for t in time_list)))
            self._send_base_trajectory(
                base_trajectory_points, time_list, wait=False)

        if arm_angle_vectors:
            self.ri.wait_interpolation()
        if base_trajectory_points:
            self.ri.move_base_trajectory_action.wait_for_result()
            # [debug] 完了直後の odom yaw と計画との差。
            odom_after = self.ri.odom
            expected_yaw = (matrix2ypr(start_odom_coords.rotation)[0]
                            + base_trajectory_points[-1][2])
            actual_yaw = matrix2ypr(odom_after.rotation)[0]
            state = self.ri.move_base_trajectory_action.get_state()
            log_debug('[debug][segment] wait_for_result 直後 (ゴール終了状態 {}) '
                  'odom_yaw={:.1f}deg (期待値={:.1f}deg, 差={:.1f}deg)'.format(
                      GOAL_STATUS_NAMES.get(state, state),
                      math.degrees(actual_yaw), math.degrees(expected_yaw),
                      math.degrees(
                          (expected_yaw - actual_yaw + math.pi)
                          % (2 * math.pi) - math.pi)))

        final_traj_point = (
            base_trajectory_points[-1] if base_trajectory_points else None)
        return start_odom_coords, final_traj_point

    def _send_base_trajectory(self, base_trajectory_points, time_list,
                              wait):
        """区間先頭基準の累積 ``[dx, dy, dyaw]`` 列を ``time_list`` で台車に送る。

        各点の速度は ``_point_velocities`` で付け直す (skrobot のままだと
        静止から区間の平均速度へいきなり跳ぶため)。
        """
        goal = self.ri.move_trajectory_sequence(
            base_trajectory_points, time_list, stop=True, send_action=False)
        points = goal.goal.trajectory.points
        deltas = np.diff(np.vstack([
            np.zeros(3),
            np.asarray(base_trajectory_points, dtype=np.float64)]), axis=0)
        point_vel = self._point_velocities(
            deltas, np.asarray(time_list, dtype=np.float64))
        # 区間先頭の台車の向き基準 -> odom 系へ回す。
        yaw0 = points[0].positions[2]
        cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)
        for point, (vx, vy, vyaw) in zip(points, point_vel):
            point.velocities = [cos_yaw * vx - sin_yaw * vy,
                                sin_yaw * vx + cos_yaw * vy,
                                vyaw]
        self.ri.move_base_trajectory_action.send_goal(goal.goal)
        if wait:
            self.ri.move_base_trajectory_action.wait_for_result()

    def _limited_time_list(self, arm_angle_vectors, base_trajectory_points,
                           time_list):
        """区間ごとの所要時間を、律速する軸の指令の速度・加速度の最大値が
        ちょうど上限 (× ``VEL_LIMIT_RATIO``、加速度は ``ACCEL_TIME``) に
        なるよう決めて返す。``time_list`` は区間ごとの下限。

        ``arm_angle_vectors``: 送る関節角列 (始点は実機の現在角)。
        ``base_trajectory_points``: 区間先頭からの累積 ``[dx, dy, dyaw]`` 列。
        None/空なら判定から外す。コントローラは 3 次エルミートで補間する
        ので、その最大値で判定する。
        """
        axis_names = []
        arm_deltas = None
        base_deltas = None
        if arm_angle_vectors:
            ri = self.ri
            controller_joint_names = {
                name for param in ri.controller_param_table[ri.controller_type]
                for name in param['joint_names']}
            # コントローラで動かさない関節は判定から外す (inf)。
            arm_max_vel = np.array([
                joint.max_joint_velocity * VEL_LIMIT_RATIO
                if (joint.name in controller_joint_names
                    and joint.max_joint_velocity > 0)
                else np.inf
                for joint in self.real_robot.joint_list])
            avs = [ri.angle_vector()] + list(arm_angle_vectors)
            arm_deltas = np.array([ri.sub_angle_vector(avs[i + 1], avs[i])
                                   for i in range(len(arm_angle_vectors))])
            axis_names += [joint.name for joint in self.real_robot.joint_list]
        if base_trajectory_points:
            points = np.vstack([np.zeros(3),
                                np.asarray(base_trajectory_points, dtype=np.float64)])
            base_deltas = np.diff(points, axis=0)
            axis_names += ['base_xy', 'base_yaw']
        if not axis_names:
            return [float(t) for t in time_list]

        def peak_ratios(times):
            # 区間・軸ごとの 速度/上限 と sqrt(加速度/上限)。
            vel_ratios = []
            acc_ratios = []
            if arm_deltas is not None:
                point_vel = self._point_velocities(arm_deltas, times)
                vel, acc = self._hermite_peaks(arm_deltas, times, point_vel)
                vel_ratios.append(vel / arm_max_vel)
                acc_ratios.append(acc / (arm_max_vel / ACCEL_TIME))
            if base_deltas is not None:
                point_vel = self._point_velocities(base_deltas, times)
                # 並進は向きによらず超えないようベクトルの大きさで判定する。
                vel_xy, acc_xy = self._hermite_peaks(
                    base_deltas[:, :2], times, point_vel[:, :2], norm=True)
                vel_yaw, acc_yaw = self._hermite_peaks(
                    base_deltas[:, 2:], times, point_vel[:, 2:])
                vel_limit = np.array([BASE_MAX_VEL, BASE_MAX_ANGVEL]) \
                    * VEL_LIMIT_RATIO
                vel_ratios.append(
                    np.stack([vel_xy, vel_yaw[:, 0]], axis=1) / vel_limit)
                acc_ratios.append(
                    np.stack([acc_xy, acc_yaw[:, 0]], axis=1)
                    / (vel_limit / ACCEL_TIME))
            return np.concatenate(
                vel_ratios + [np.sqrt(r) for r in acc_ratios], axis=1)

        min_times = np.array(time_list, dtype=np.float64)
        times = min_times.copy()
        for _ in range(TIME_LIMIT_MAX_ITERATIONS):
            # ratio 倍そのままだと振動するので平方根で半分だけ動かす。
            ratio = np.max(peak_ratios(times), axis=1)
            new_times = np.maximum(min_times, times * np.sqrt(ratio))
            if np.allclose(new_times, times, rtol=1e-4, atol=0.0):
                times = new_times
                break
            times = new_times
        # 残った超過は全区間を一律に延ばして必ず上限内にする。
        worst = float(np.max(peak_ratios(times)))
        if worst > 1.0:
            if worst > 1.01:
                log_debug('[debug][segment] 速度上限の反復で収まらなかったため '
                      '全区間を {:.3f} 倍に延ばします。'.format(worst))
            times *= worst
        return [float(t) for t in times]

    @staticmethod
    def _point_velocities(deltas, times):
        """各点の速度 ``(区間数 + 1, 軸数)``。skrobot の
        ``angle_vector_sequence`` と同じ規則 (途中は前後区間の平均速度の
        平均、符号が逆の軸と始点・終点は 0)。"""
        n_segments = len(times)
        seg_vel = deltas / times[:, None]
        point_vel = np.zeros((n_segments + 1, deltas.shape[1]))
        if n_segments > 1:
            same_sign = deltas[:-1] * deltas[1:] >= 0.0
            point_vel[1:n_segments] = np.where(
                same_sign, 0.5 * (seg_vel[:-1] + seg_vel[1:]), 0.0)
        return point_vel

    @staticmethod
    def _hermite_peaks(deltas, times, point_vel, norm=False, n_samples=101):
        """各区間を点の速度 ``point_vel`` の 3 次エルミート補間で動かした
        ときの速度と加速度の最大値 (絶対値) を返す。形状は ``deltas`` と
        同じ、``norm=True`` ならベクトルの大きさで ``(区間数,)``
        (このときの速度は ``n_samples`` 点で評価する)。"""
        seg_times = times[:, None]
        v0 = point_vel[:-1]
        v1 = point_vel[1:]
        a = (v0 + v1) / seg_times ** 2 - 2.0 * deltas / seg_times ** 3
        b = 3.0 * deltas / seg_times ** 2 - (2.0 * v0 + v1) / seg_times
        acc0 = 2.0 * b
        acc1 = 6.0 * a * seg_times + 2.0 * b
        if norm:
            # t: (区間数, n_samples, 1)、速度: (区間数, n_samples, 軸数)
            t = (np.linspace(0.0, 1.0, n_samples)[None, :]
                 * seg_times)[:, :, None]
            vel = (3.0 * a[:, None] * t ** 2 + 2.0 * b[:, None] * t
                   + v0[:, None])
            return (np.max(np.linalg.norm(vel, axis=2), axis=1),
                    np.maximum(np.linalg.norm(acc0, axis=1),
                               np.linalg.norm(acc1, axis=1)))
        peak = np.maximum(np.abs(v0), np.abs(v1))
        with np.errstate(divide='ignore', invalid='ignore'):
            t_star = -b / (3.0 * a)
            v_star = v0 - b ** 2 / (3.0 * a)
        inside = (a != 0.0) & (t_star > 0.0) & (t_star < seg_times)
        return (np.where(inside, np.maximum(peak, np.abs(v_star)), peak),
                np.maximum(np.abs(acc0), np.abs(acc1)))

    def _wait_odom_settle(self, timeout=ODOM_SETTLE_TIMEOUT):
        """odom が止まるまで最大 ``timeout`` 秒待つ。
        ``(待った時間 [s], (その間の移動距離 [m], 回転 [rad]))`` を返す。"""
        start = time.time()
        first = self._odom_pose(self.ri.odom)
        prev = first
        while not rospy.is_shutdown() and time.time() - start < timeout:
            rospy.sleep(ODOM_SETTLE_WINDOW)
            cur = self._odom_pose(self.ri.odom)
            step = scan_matching.relative(prev, cur)
            prev = cur
            if (math.hypot(step[0], step[1]) < ODOM_SETTLE_POSITION
                    and abs(step[2]) < ODOM_SETTLE_ANGLE):
                break
        total = scan_matching.relative(first, prev)
        return time.time() - start, (math.hypot(total[0], total[1]),
                                     abs(total[2]))

    @staticmethod
    def _odom_pose(coords):
        """odom の ``Coordinates`` を ``[x, y, yaw]`` にする。"""
        return np.array([coords.translation[0], coords.translation[1],
                         matrix2ypr(coords.rotation)[0]])

    def _on_scan(self, msg):
        # 受信時刻は PC の時計で持つ (ロボットとの時計ずれの影響を受けない)。
        with self._lock:
            self._latest_scan = (time.time(), msg)

    def _capture_scan(self, timeout=SCAN_CAPTURE_TIMEOUT):
        """呼び出し以降の /scan を 1 つ待ち、台車座標系の点群 ``(N, 2)`` で
        返す (取れなければ None)。"""
        after = time.time()
        deadline = after + timeout
        msg = None
        while not rospy.is_shutdown() and time.time() < deadline:
            with self._lock:
                latest = self._latest_scan
            if latest is not None and latest[0] > after:
                msg = latest[1]
                break
            rospy.sleep(0.005)
        if msg is None:
            print('[execute][scan][WARN] {} 秒以内に {} が届きませんでした。'
                  .format(timeout, self.args.scan_topic))
            return None
        transform = lookup_camera_to_base(
            self.tf_buffer, self.args.base_frame, msg.header)
        if transform is None:
            return None
        matrix = transform_to_matrix(transform.transform)
        sensor_pose = (matrix[0, 3], matrix[1, 3],
                       math.atan2(matrix[1, 0], matrix[0, 0]))
        return scan_matching.scan_to_points(
            msg.ranges, msg.angle_min, msg.angle_increment, msg.range_min,
            msg.range_max, sensor_pose=sensor_pose, max_range=SCAN_MAX_RANGE)

    def _match_scan(self, ref_scan, odom_delta, exclude_centers, attempt,
                    plan):
        """今のスキャンを ``ref_scan`` と照合し、区間先頭から実際に動いた量
        ``[x, y, yaw]`` を返す (照合の品質が悪ければ None)。"""
        cur_scan = self._capture_scan()
        if cur_scan is None:
            return None
        t0 = time.time()
        match = scan_matching.icp_2d(
            ref_scan, cur_scan, odom_delta, exclude_centers=exclude_centers,
            exclude_radius=SCAN_HUMAN_EXCLUDE_RADIUS)
        match_time = time.time() - t0
        scan_delta = np.asarray(match['pose'])
        ok = (match['inlier_ratio'] >= SCAN_MIN_INLIER_RATIO
              and match['n_inliers'] >= SCAN_MIN_INLIERS
              and match['rms'] <= SCAN_MAX_RMS)
        slip = scan_matching.relative(odom_delta, scan_delta)
        self._log_debug(dict(
            event='scan_correction', attempt=attempt, ok=bool(ok),
            plan=[float(v) for v in plan],
            odom_delta=[float(v) for v in odom_delta],
            scan_delta=[float(v) for v in scan_delta],
            slip=[float(v) for v in slip],
            rms=match['rms'], inlier_ratio=match['inlier_ratio'],
            n_inliers=match['n_inliers'], iterations=match['iterations'],
            n_points=[len(ref_scan), len(cur_scan)], match_time=match_time))
        log_debug('[execute][scan] attempt={} スキャン照合{}: 実際の移動 '
              '({:+.3f}m, {:+.3f}m, {:+.1f}deg) / odom ({:+.3f}m, {:+.3f}m, '
              '{:+.1f}deg) -> スリップ ({:+.3f}m, {:+.3f}m, {:+.1f}deg) '
              '[rms={:.1f}mm, 対応率={:.2f}, {:.3f}s]'.format(
                  attempt, '' if ok else ' (品質不足のため odom を使用)',
                  scan_delta[0], scan_delta[1], math.degrees(scan_delta[2]),
                  odom_delta[0], odom_delta[1], math.degrees(odom_delta[2]),
                  slip[0], slip[1], math.degrees(slip[2]),
                  match['rms'] * 1e3, match['inlier_ratio'], match_time))
        return scan_delta if ok else None

    def _correct_base_residual(
            self, start_odom_coords, final_traj_point, ref_scan=None,
            exclude_centers=None,
            max_attempts=BASE_CORRECTION_MAX_ATTEMPTS,
            position_tolerance=BASE_CORRECTION_POSITION_TOLERANCE,
            angle_tolerance=BASE_CORRECTION_ANGLE_TOLERANCE):
        """押し込み前に台車の hover 目標からの位置ずれを検出し、収束するまで
        相対移動で補正する。

        引数は ``_execute_waypoint_segment`` の戻り値。実際の移動量は
        ``ref_scan`` があればスキャン照合 (``exclude_centers`` 近傍の人の点は
        除く)、無ければ odom (スリップは検出できない) で求める。
        """
        if start_odom_coords is None or final_traj_point is None:
            return
        plan = np.asarray(final_traj_point, dtype=np.float64)
        start_odom = self._odom_pose(start_odom_coords)

        settle_time, drift = self._wait_odom_settle()
        log_debug('[execute][correct] 接近区間の終了後に odom が止まるまで {:.2f}s '
              '(その間の移動 {:.1f}mm/{:.2f}deg)'.format(
                  settle_time, drift[0] * 1e3, math.degrees(drift[1])))
        err_norm = 0.0
        err_yaw = 0.0
        for attempt in range(max_attempts):
            odom_delta = scan_matching.relative(
                start_odom, self._odom_pose(self.ri.odom))
            actual, source = odom_delta, 'odom'
            if ref_scan is not None:
                scan_delta = self._match_scan(
                    ref_scan, odom_delta, exclude_centers, attempt, plan)
                if scan_delta is not None:
                    actual, source = scan_delta, 'scan'
            # 今の台車から見た hover 目標 (= 補正で動かす量)。
            err_x, err_y, err_yaw = scan_matching.relative(actual, plan)
            err_norm = math.hypot(err_x, err_y)
            log_debug('[execute][correct] attempt={} ({}基準) 目標までの残差 '
                  '{:+.3f}m {:+.3f}m {:+.1f}deg'.format(
                      attempt, source, err_x, err_y, math.degrees(err_yaw)))

            if err_norm <= position_tolerance and abs(err_yaw) <= angle_tolerance:
                if attempt > 0:
                    print('[execute] 押し込み前に台車の位置ずれを補正し '
                          'ました ({} 回、残差 {:.3f}m / {:.1f}deg)。'.format(
                              attempt, err_norm, math.degrees(abs(err_yaw))))
                return

            print('[execute] 押し込み前の hover 目標に対して台車の位置ずれ '
                  'を検出 (残差 {:.3f}m / {:.1f}deg)、補正します '
                  '({}/{})。'.format(
                      err_norm, math.degrees(abs(err_yaw)), attempt + 1,
                      max_attempts))
            sec, = self._limited_time_list(
                None, [[err_x, err_y, err_yaw]], [1.0])
            before = self._odom_pose(self.ri.odom)
            self._send_base_trajectory(
                [[err_x, err_y, err_yaw]], [sec], wait=True)
            state = self.ri.move_base_trajectory_action.get_state()
            # ゴール終了後も台車が動き続けることがあるので止まるまで待つ。
            settle_time, drift = self._wait_odom_settle()
            moved = scan_matching.relative(
                before, self._odom_pose(self.ri.odom))
            log_debug('[execute][correct] 補正の移動: 指令 ({:+.3f}m, {:+.3f}m, '
                  '{:+.1f}deg) {:.2f}s -> odom 上の移動 ({:+.3f}m, {:+.3f}m, '
                  '{:+.1f}deg)、ゴール終了状態 {}、終了後に odom が止まるまで '
                  '{:.2f}s (その間の移動 {:.1f}mm/{:.2f}deg)'.format(
                      err_x, err_y, math.degrees(err_yaw), sec,
                      moved[0], moved[1], math.degrees(moved[2]),
                      GOAL_STATUS_NAMES.get(state, state), settle_time,
                      drift[0] * 1e3, math.degrees(drift[1])))

        print('[execute][WARN] 押し込み前の台車の位置ずれ補正が {} 回で '
              '収束しませんでした (残差 {:.3f}m / {:.1f}deg)。このまま '
              '押し込み動作へ進みます。'.format(
                  max_attempts, err_norm, math.degrees(abs(err_yaw))))

    def _play_loop(self):
        """Play がオンの間 waypoint スライダーを進める (最後で自動停止)。"""
        while not rospy.is_shutdown():
            time.sleep(1.0 / max(self.args.playback_fps, 1e-3))
            if not self.play_checkbox.value:
                continue
            index = int(self.waypoint_slider.value)
            if index >= self.waypoint_slider.max:
                self.play_checkbox.value = False
                continue
            # 代入で on_update が同期的に呼ばれ描画される。
            self.waypoint_slider.value = index + 1

    # ------------------------------------------------------------------
    # viser display
    # ------------------------------------------------------------------
    def _update_skeleton_view(self, joint_positions):
        """骨格の線と人体の干渉回避ジオメトリ (IK と同じ Cylinder) を
        作り直す (``joint_positions`` が None なら消す)。"""
        with self._viewer_lock:
            for link in self._skeleton_links:
                self.viewer.delete(link)
            self._skeleton_links = (
                [] if joint_positions is None
                else skeleton_drawing.build_skeleton_links(joint_positions))
            for link in self._skeleton_links:
                self.viewer.add(link)

            for obstacle_link in self._current_obstacle_links:
                self.viewer.delete(obstacle_link)
            self._current_obstacle_links = (
                [] if joint_positions is None
                else spik.human_body_obstacles(
                    spik.shift_torso_joints_from_surface(
                        joint_positions, (0.0, 0.0),
                        self.args.torso_surface_offset)))
            for obstacle_link in self._current_obstacle_links:
                palm_plane_view.set_color(
                    obstacle_link, HUMAN_COLLISION_OBSTACLE_COLOR)
                self.viewer.add(obstacle_link)
                self._set_link_visible(
                    obstacle_link, self.show_collision_models_checkbox.value)

    def _update_status_text(self, joint_positions, is_base_frame, is_frozen):
        """viser 画面のテキストパネルに現在の状態を表示する。"""
        if self.state == 'armed':
            state_text = ('ARMED (手を差し出してください)'
                          if self._retry_reference is None else
                          'ARMED (今の手を動かすか向きを変えてから、'
                          'もう一度差し出してください)')
        elif self.state == 'solving':
            state_text = 'IK を計算中です...'
        elif self.state == 'result':
            state_text = ('結果を表示中です')
        else:
            state_text = 'IDLE (ARM ボタンを押すと手を差し出す人を待ちます)'
        if is_frozen:
            detected_text = '固定表示中'
        elif joint_positions is None:
            detected_text = '未検出'
        elif is_base_frame:
            detected_text = '検出中 (base_link 座標系。ARMED 動作可能)'
        else:
            detected_text = ('検出中 (TF 未解決のためカメラ座標系で表示中。'
                             'ARMED では使われません)')
        content = '**状態:** {}\n\n**骨格:** {}'.format(state_text, detected_text)
        if self.state == 'armed' and self._latest_offer_selection is not None:
            content += '\n\n' + epp.format_offer_scores(
                self._latest_offer_selection, self.offered_hand_selector.score_min)
        with self._lock:
            result = self._current_result
            motion = self._current_motion
            n_prepend = self._display_n_prepend
            n_approach = self._display_n_approach
            handshake_total_time = self._handshake_total_time
        if handshake_total_time is not None:
            content += ('\n\n**計算時間 (掌推定 ~ 軌道計画):** {:.2f} 秒'
                       .format(handshake_total_time))
        if result is not None:
            if motion is not None:
                kind = phm.KIND_LABELS.get(motion['kind'], motion['kind'])
                verified_text = ('OK (経路全体で干渉なし)' if motion['verified']
                                 else 'NG (経路上に干渉が残る waypoint あり)')
                lead_in_text = (
                    'OK' if motion.get('lead_in_verified', True)
                    else 'NG (人間の近くで干渉あり)')
                waypoint_index = int(self.waypoint_slider.value)
                model_label = (
                    '指あり' if getattr(
                        self.motion_verification_pairs, 'model', None)
                    is not None
                    else '箱+人体距離は指あり' if getattr(
                        self.motion_verification_pairs, 'clearance_pairs',
                        None)
                    else '指なし+手の箱')
                content += ('\n\n**軌道:** {} / 計画時 ({}) の検証: {} / '
                           '初期位置からの直進の検証: {}\n\n'
                           'waypoint {}/{}'.format(
                               kind, model_label, verified_text, lead_in_text,
                               waypoint_index, self.waypoint_slider.max))
                if waypoint_index < n_prepend:
                    lead_in_dists = motion.get('lead_in_min_distances', [])
                    dist = (lead_in_dists[waypoint_index]
                            if waypoint_index < len(lead_in_dists) else None)
                    content += (
                        ' (初期位置から接近開始位置への直進、人間から離れて'
                        'いるため干渉検証の対象外)' if dist is None
                        else ' (初期位置から接近開始位置への直進、計画時 '
                             '({}) の干渉余裕: {:+.4f} m)'.format(
                                 model_label, dist))
                elif waypoint_index < n_prepend + n_approach:
                    dist = motion['waypoint_min_distances'][
                        waypoint_index - n_prepend]
                    content += (' (この waypoint の計画時 ({}) の干渉'
                               '余裕: {:+.4f} m)'.format(model_label, dist))
                else:
                    content += (' (掌への押し込み、経路計画の干渉検証の対象外)')
            else:
                content += ('\n\n**軌道:** 計画なし ({})'.format(
                    'IK 失敗' if not result['solved']
                    else '押し込み姿勢が解けない'
                    if result.get('post_process') is None else '計算中'))
        # 表示中の姿勢の干渉の結論は事後検証で出す (上は計画時の数値のみ)。
        content += '\n\n' + self._collision_pairs_text
        self._status_text.content = content

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def spin(self):
        """viser 画面の骨格・状態表示を更新するメインループ。"""
        print('viser のブラウザ画面で骨格の確認と ARM ボタンの操作を '
              '行ってください (URL は起動時に表示されます)。')

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self._lock:
                joint_positions = self._latest_joint_positions
                is_base_frame = self._latest_is_base_frame
                frozen_joint_positions = self._frozen_joint_positions

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
                display_joint_positions = self._last_detected_joint_positions
            else:
                display_joint_positions = None

            # ちらつき防止のため、骨格の作り直しは変化があり、かつ間隔が
            # 空いたときだけ (出現・消失・固定表示への切り替えは即座に)。
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
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                # rosbag 再生でシミュレーション時刻が巻き戻ったときは無視する。
                pass
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
        '--camera-optical-frame', type=str,
        default='camera_color_optical_frame',
        help='視線 IK のカメラ光軸のフレーム (TF が引けなければ既定値)。')
    parser.add_argument(
        '--base-correction', choices=['scan', 'odom', 'none'],
        default='scan',
        help='押し込み前の台車位置補正で移動量を求める方法 (scan: /scan '
            '照合、odom: スリップは検出不可、none: 補正しない)。')
    parser.add_argument('--scan-topic', type=str, default='/scan')
    parser.add_argument(
        '--tf-cache-time', type=float, default=30.0,
        help='tf2 バッファの保持時間 [秒] (マシン間の時計ずれ対策)。')
    parser.add_argument(
        '--client-wait-timeout', type=float, default=30.0,
        help='viser のクライアント接続を待つ 1 回あたりの秒数。')
    parser.add_argument(
        '--no-open-browser', action='store_true',
        help='ブラウザを自動で開かない。')
    parser.add_argument(
        '--no-wait-for-client', action='store_true',
        help='viser のクライアント接続を待たない (無人の bag テスト用)。')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5)
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5)
    parser.add_argument('--min-visibility', type=float, default=0.5)
    parser.add_argument('--min-joints', type=int, default=6)
    parser.add_argument('--max-z-diff', type=float, default=1.0)
    parser.add_argument(
        '--min-body-size', type=float, default=0.3,
        help='関節の bbox 対角線長 [m] がこれ未満の骨格を棄却する。')
    parser.add_argument(
        '--max-body-size', type=float, default=2.5,
        help='関節の bbox 対角線長 [m] がこれを超える骨格を棄却する。')
    parser.add_argument(
        '--max-limb-length', type=float, default=0.7,
        help='四肢の区間長 [m] がこれを超えたら遠位側の関節を捨てる。')
    parser.add_argument(
        '--max-hand-segment-length', type=float, default=0.12,
        help='手の区間長 [m] がこれを超えたら遠位側のランドマークを捨てる。')
    parser.add_argument(
        '--max-hand-reach', type=float, default=0.22,
        help='手首からの距離 [m] がこれを超えた指のランドマークを捨てる。')
    parser.add_argument('--depth-patch-size', type=int, default=3)
    parser.add_argument(
        '--joint-smoothing-mincutoff', type=float, default=0.5,
        help='関節平滑化 (One Euro Filter) の最小カットオフ周波数 [Hz]。')
    parser.add_argument(
        '--joint-smoothing-beta', type=float, default=0.3,
        help='One Euro Filter の速度依存カットオフの係数。')
    parser.add_argument(
        '--joint-smoothing-dcutoff', type=float, default=1.0,
        help='One Euro Filter の速度推定のカットオフ周波数 [Hz]。')
    parser.add_argument(
        '--offer-score-min', type=float, default=0.65,
        help='差し出し手と判定するスコアの閾値 (実カメラ用に合成骨格の '
            '{:.2f} より低い)。'.format(epp.OFFER_SCORE_MIN))
    parser.add_argument(
        '--max-person-distance', type=float, default=3.0,
        help='人物からロボット手先までの距離 [m] がこれを超えたら差し出し '
            '候補から外す (0 以下で無効)。')
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='使うロボットの腕 (auto: 人の手の反対側)。')
    parser.add_argument(
        '--robot-hand-position', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'Z'),
        help='差し出し手判定の基準にするロボット手先の base_link 座標 [m] '
            '(指定すると TF より優先)。')
    parser.add_argument(
        '--robot-hand-frame', type=str, default='r_eef_grasp_link',
        help='差し出し手判定の基準にするロボット手先の TF フレーム。')
    parser.add_argument(
        '--human-front-distance', type=float,
        default=spik.HUMAN_FRONT_DISTANCE,
        help='IK で人物を置く Aero の前方距離 [m]。')
    parser.add_argument(
        '--attempts-per-pose', type=int,
        default=spik.DEFAULT_ATTEMPTS_PER_POSE)
    parser.add_argument(
        '--collision-pairs', type=str,
        default=os.path.join(_SCRIPTS_DIR, 'collision_pairs.json'))
    parser.add_argument('--no-human-collision', action='store_true')
    parser.add_argument(
        '--torso-surface-offset', type=float,
        default=spik.DEFAULT_TORSO_SURFACE_OFFSET,
        help='干渉判定用に体幹の関節をカメラから離れる向きへずらす距離 [m] '
            '(関節が体の表面にあるため)。')
    parser.add_argument('--no-self-collision', action='store_true')
    parser.add_argument(
        '--collision-verify-model', choices=spik.COLLISION_VERIFY_MODELS,
        default=spik.DEFAULT_COLLISION_VERIFY_MODEL,
        help='IK と軌道の事後検証に使うモデル。')
    parser.add_argument(
        '--collision-verify-tolerance', type=float,
        default=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE,
        help='画面表示用の指ありの事後検証の許容誤差 [m] (IK には使わない)。')
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
        '--base-x-standing-margins', type=float, nargs='+',
        default=list(spik.DEFAULT_BASE_X_STANDING_MARGINS),
        help='台車の x 可動範囲を人の立ち位置 ±この幅 [m] に絞る (順に試す、'
            '負は絞らない)。')
    parser.add_argument(
        '--front-offset-weight', type=float,
        default=spik.DEFAULT_FRONT_OFFSET_WEIGHT,
        help='IK 候補のコストに足す、台車の人の正面方向へのずれ [m] の重み。')
    parser.add_argument(
        '--facing-yaw-weight', type=float,
        default=spik.DEFAULT_FACING_YAW_WEIGHT,
        help='IK 候補のコストに足す、台車の向きのずれ [rad] の重み。')
    parser.add_argument(
        '--save-dir', type=str, default=None,
        help='試行ごとに骨格/掌/IK 結果/軌道の JSON を保存するディレクトリ。')
    # --- 軌道計画 (plan_handshake_motion.py と同じオプション・既定値) ---
    parser.add_argument(
        '--approach-distance', type=float,
        default=phm.DEFAULT_APPROACH_DISTANCE,
        help='接近開始位置の円の半径への上乗せ分 [m]。')
    parser.add_argument(
        '--pretouch-standoff', type=float,
        default=phm.DEFAULT_PRETOUCH_STANDOFF,
        help='pre-touch 姿勢を掌の法線方向へ引き戻す距離 [m]。')
    parser.add_argument(
        '--pretouch-split', type=float, default=phm.DEFAULT_PRETOUCH_SPLIT,
        help='軌道のうち pre-touch 姿勢までに使う割合。')
    parser.add_argument(
        '--n-waypoints', type=int, default=phm.DEFAULT_N_WAYPOINTS,
        help='軌道の waypoint 数 (始点・終点を含む)。')
    parser.add_argument(
        '--max-iterations', type=int, default=phm.DEFAULT_MAX_ITERATIONS,
        help='軌道最適化 (jaxls) の最大反復回数。')
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
        help='軌道の干渉が残ったとき warm start を変えて解き直す最大回数。')
    parser.add_argument(
        '--motion-attempt-perturbation', type=float, default=0.3)
    parser.add_argument(
        '--motion-collision-verify-tolerance', type=float,
        default=phm.DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE,
        help='軌道の事後検証で許容する最大貫通量 [m]。')
    parser.add_argument(
        '--force-optimize', action='store_true',
        help='候補が検証に通っても必ず jaxls の軌道最適化まで実行する。')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='軌道計画の warm start を揺らす乱数シード。')
    parser.add_argument(
        '--playback-fps', type=float, default=DEFAULT_PLAYBACK_FPS,
        help='waypoint 自動再生の速さ [waypoint/秒]。')
    # --- 実機動作 ---
    parser.add_argument(
        '--auto-execute', action='store_true',
        help='IK・軌道計画が成功したら実機 (台車・関節) を自動で動かす。')
    parser.add_argument(
        '--speech-start-text', type=str, default='今から行きますね',
        help='実機が動き出すときの発話 (空で発話しない)。')
    parser.add_argument(
        '--no-press-in-refine', dest='press_in_refine', action='store_false',
        help='hover 到達後に手を検出し直して押し込み姿勢を補正しない。')
    parser.add_argument(
        '--speech-done-text', type=str, default='どうぞ、手を握ってください',
        help='掌を差し出し終えたときの発話 (空で発話しない)。')
    parser.add_argument(
        '--grasp-capture-duration', type=float, default=3.0,
        help='押し込み後にカメラ画像を保存する秒数 (0 で保存しない)。')
    parser.add_argument(
        '--speech-fail-text', type=str,
        default='ごめんなさい、うまく手を出せませんでした',
        help='実機を動かせなかったときの発話 (空で発話しない)。')
    parser.add_argument(
        '--no-speech-advice', dest='speech_advice', action='store_false',
        help='失敗時に手の出し方の助言をしない。')
    parser.add_argument(
        '--hand-offer-table', type=str,
        default=hand_offer_advice.DEFAULT_TABLE_PATH,
        help='手の出し方の助言に使う reachability map の表。')
    parser.add_argument(
        '--speech-retry-text', type=str,
        default='手の位置を少し変えて、もう一度差し出してください',
        help='助言が見つからないときに --speech-fail-text に続ける発話。')
    parser.add_argument(
        '--speech-approach-fail-text', type=str,
        default='ごめんなさい、近づく道が見つかりませんでした。'
                'もう一度手を出してください',
        help='接近途中の干渉で実機を動かせなかったときの発話。')
    parser.add_argument(
        '--no-side-by-side-transition', dest='side_by_side_transition',
        action='store_false',
        help='押し込み後の横並び移動を行わない。')
    parser.add_argument(
        '--side-by-side-segment-time', type=float, default=0.5,
        help='横並び移動の 1 区間にかける最短時間 [秒]。')
    parser.add_argument(
        '--side-by-side-delay', type=float, default=2.0,
        help='押し込み終了から横並び移動を始めるまでの時間 [秒]。')
    parser.add_argument(
        '--speech-transition-text', type=str,
        default='一緒に横に並びますね',
        help='横並び移動を始めるときの発話 (空で発話しない)。')
    parser.add_argument(
        '--speech-voice', type=str, default='四国めたん-ノーマル',
        help='発話に使う声 (sound_play の voice)。')
    # --- rosbag での実カメラ無しテスト ---
    parser.add_argument(
        '--bag', type=str, default=None,
        help='実カメラの代わりに再生する rosbag (内部で rosbag play を起動)。')
    parser.add_argument(
        '--bag-rate', type=float, default=1.0,
        help='--bag 再生時の速度倍率。')
    parser.add_argument(
        '--bag-loop', action='store_true',
        help='--bag をループ再生する。')
    parser.add_argument(
        '--auto-arm', action='store_true',
        help='起動直後から ARMED にする (実機の姿勢は動かさない)。')
    parser.add_argument(
        '--no-robot-interface', action='store_true',
        help='実機 (AeroROSRobotInterface) に接続しない。')
    # roslaunch が付ける引数 (__name 等) は無視する。
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    log_path = setup_log_dir()
    print('[log] 詳細なログ ([debug] など) は画面に出さず {} 以下に試行 (人) '
          'ごとのファイルで保存します (起動時: {})。'.format(LOG_DIR, log_path))

    bag_process = None
    if args.bag:
        if shutil.which('rosbag') is None:
            sys.exit(
                'rosbag が見つかりません。source /opt/ros/noetic/setup.bash '
                '等で ROS の setup.bash を読み込んでから実行してください。')
        if not os.path.exists(args.bag):
            sys.exit('--bag で指定したファイルが見つかりません: {}'.format(
                args.bag))
        # rosbag play --clock の /clock に同期させる。
        rospy.set_param('/use_sim_time', True)

    rospy.init_node('run_camera_pipeline_test')
    node = HandshakePipelineNode(args)

    if args.bag:
        cmd = ['rosbag', 'play', args.bag, '--clock',
              '-r', str(args.bag_rate)]
        if args.bag_loop:
            cmd.append('--loop')
        print('[bag] 再生します: {}'.format(' '.join(cmd)))
        bag_process = subprocess.Popen(cmd)

    try:
        node.spin()
    finally:
        if bag_process is not None:
            bag_process.terminate()
            bag_process.wait()


if __name__ == '__main__':
    main()
