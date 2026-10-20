#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""実カメラ + ``PeoplePoseEstimator`` の骨格に対し、ARM 後に差し出し手が決まった
瞬間の掌推定・IK・軌道計画を行う対話的な ROS ノード。

IK は指なしロボットで解き、画面の干渉表示は指ありでの事後検証。
``--auto-execute`` なら計画成功時 (干渉検証 OK) に実機を動かす。

Usage
-----
    python3 scripts/ros/run_camera_pipeline_test.py
    python3 scripts/ros/run_camera_pipeline_test.py --save-dir /tmp/camera_handshake_poses
    python3 scripts/ros/run_camera_pipeline_test.py --auto-arm --auto-execute
"""

import argparse
import collections
import copy
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

import control_msgs.msg
import rospy
import message_filters
import tf2_ros
import trajectory_msgs.msg
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_srvs.srv import SetBool

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
_PKG_SRC_DIR = os.path.join(_SCRIPTS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# GPU メモリの一括確保の失敗ログ (RESOURCE_EXHAUSTED) を出さないため。
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# jax の永続コンパイルキャッシュ (jax の import より前に設定)。
os.environ.setdefault(
    'JAX_COMPILATION_CACHE_DIR',
    os.path.expanduser('~/.cache/jax_compilation_cache'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')

from aero_demo import hand_offer_advice  # noqa: E402
from aero_demo import in_motion_correction  # noqa: E402
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

INITIAL_POSE_AXIS_LENGTH = 0.2
INITIAL_POSE_AXIS_RADIUS = 0.008

DEFAULT_PLAYBACK_FPS = 40.0  # [waypoint/秒]

ARM_INITIAL_POSE_MOVE_TIME = 10  # [秒] ARM 時に初期姿勢へ戻す時間
# うなずきで下げる首ピッチ角 [deg] (neck_p_joint 既定 25 度から)。
HEAD_NOD_PITCH_DEG = 40.0
HEAD_NOD_MOVE_TIME = 0.5  # [秒] 下げる・戻すそれぞれの時間の下限

# ARMED 中の首の動き (_head_tracking_loop)。人が見えれば光軸を体幹へ向け、
# 見えなければ首ヨーを 0 -> ±HEAD_SCAN_YAW_DEG と往復して探す (端・中央で止まり
# 首ピッチを変えて下も見る)。
HEAD_TRACK_PERIOD = 0.2  # [秒] 首の指令を見直す周期
HEAD_TRACK_TARGET_JOINTS = ('Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip')
# 光軸と体幹のずれがこれ未満なら動かさない (カメラ揺れ防止)。
HEAD_TRACK_DEADBAND_DEG = 8.0
HEAD_TRACK_IK_STOP = 40
HEAD_TRACK_IK_RTHRE = math.radians(2.0)
HEAD_TRACK_SPEED_DEG = 45.0  # [deg/s] 人の方を向くときの首の速さの上限
HEAD_TRACK_MIN_MOVE_TIME = 0.3  # [秒]
HEAD_SCAN_START_DELAY = 2.0  # [秒]
HEAD_SCAN_YAW_DEG = 60.0
# 止まった位置で見る首ピッチの初期角度からの差 [deg] (負が上向き)。
HEAD_SCAN_PITCH_OFFSETS_DEG = (0.0, 15.0)
HEAD_SCAN_DWELL = 1.0  # [秒] 止まった各向きで骨格を検出する時間
HEAD_SCAN_SPEED_DEG = 30.0  # [deg/s] 見渡すときの首の速さ
# 実行中に顔 (鼻、なければ両耳の中点、体幹) を追う首の動き。見えた位置を odom 系で
# 覚え、見失っても向き続ける。
HEAD_FACE_TRACK_DEADBAND_DEG = 4.0
HEAD_FACE_TRACK_SPEED_DEG = 80.0  # [deg/s] (実機の首ヨーの上限は約 92 deg/s)
# 指令の変化がこれ未満なら送り直さない (毎周期の再送で head_controller のゴールが
# 終わらなくなるため)。
HEAD_TRACK_RESEND_DEG = 0.5
ODOM_FRAME = 'odom'

# 実機再生時の waypoint 間の所要時間の下限 [秒] (腕の制御周期 15Hz の 1 周期)。
MIN_SEGMENT_TIME = 1.0 / 15.0

# 干渉 NG の waypoint が全て接近区間のこの割合より後ろなら手の出し方が原因とみなす。
FAILURE_NEAR_HAND_FRACTION = 0.7

# ARMED へ戻った後、掌がこれだけ動く/傾く/見えなくなるまで同じ手の差し出しを受け付けない。
RETRY_MIN_DISPLACEMENT = 0.05
RETRY_MIN_ROTATION_DEG = 20.0
RETRY_LOST_TIME = 0.5
# 失敗後に次の差し出しを受け付けるまでの待ち [秒] = BASE + PER_CHAR × 発話文字数。
RETRY_COOLDOWN_BASE = 3.0
RETRY_COOLDOWN_PER_CHAR = 0.2

# 台車の速度上限。実機の aero_base_link.yaml の max_velocity と一致させること。
BASE_MAX_VEL = 0.3  # [m/s] 実機の base_link_x/y の max_velocity
BASE_MAX_ANGVEL = 1.0  # [rad/s] 実機の base_link_pan の max_velocity

# 指令で使う速度上限の割合。3 次スプライン瞬間速度の最大がこの割合になるよう
# 区間の時間を決める (_limited_time_list)。
VEL_LIMIT_RATIO = 0.9
# 上限速度 (× VEL_LIMIT_RATIO) まで加速するのにかける時間 [秒]。
ACCEL_TIME = 0.4
TIME_LIMIT_MAX_ITERATIONS = 100

# 接近終盤の台車の最終位置のずらし (_correct_base_tail)。到着 IN_MOTION_LEAD〜
# IN_MOTION_LEAD_MIN 秒前に、台車停止 (odom 速度が MAX_YAW_RATE・MAX_SPEED 以下)
# から SETTLE 秒後の掌検出が WINDOW 秒に MIN_DETECTIONS 件たまったら、中央値に
# 合わせ残りの軌道をずらす。回転中は odom が先行するので使わない。ずらしは
# IN_MOTION_MAX_SHIFT 以下で人体距離の検証に通る量だけ。
IN_MOTION_LEAD = 3.0  # [s]
IN_MOTION_LEAD_MIN = 0.8  # [s]
IN_MOTION_POLL_PERIOD = 0.1  # [s]
IN_MOTION_MAX_YAW_RATE = 0.1  # [rad/s]
IN_MOTION_MAX_SPEED = 0.05  # [m/s]
IN_MOTION_SETTLE = 0.5  # [s]
IN_MOTION_WINDOW = 1.0  # [s]
IN_MOTION_MIN_DETECTIONS = 5
IN_MOTION_MAX_SHIFT = in_motion_correction.MAX_SHIFT  # [m]
# ずらした軌道の人体距離の検証で試すずらしの縮め方。
IN_MOTION_CLEARANCE_SCALES = (1.0, 0.5, 0.25)

# 押し込み直前の台車の位置補正 (_correct_base_residual)。yaw は補正で収束しない
# ので許容 5° 以内は補正しない (指令 0)。
BASE_CORRECTION_MAX_ATTEMPTS = 2
BASE_CORRECTION_POSITION_TOLERANCE = 0.025  # [m]
BASE_CORRECTION_ANGLE_TOLERANCE = math.radians(5.0)  # [rad]
# /scan 照合による移動量推定 (--base-correction scan)。
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
# 押し込み前の補正 1 回の時間 (odom は微動が続くので待ちは短く切る)。
BASE_CORRECTION_MIN_TIME = 0.4  # [s] 補正の移動の下限 (上限速度で延びる)
BASE_CORRECTION_SETTLE_TIMEOUT = 0.25  # [s]
GOAL_STATUS_NAMES = {
    0: 'PENDING', 1: 'ACTIVE', 2: 'PREEMPTED', 3: 'SUCCEEDED', 4: 'ABORTED',
    5: 'REJECTED', 6: 'PREEMPTING', 7: 'RECALLING', 8: 'RECALLED', 9: 'LOST'}
# 腕が指令に追いつくのを待つ (_wait_joint_settle)。
JOINT_SETTLE_HAND_TOLERANCE = 0.01  # [m]
JOINT_SETTLE_TIMEOUT = 1.5  # [s]
JOINT_SETTLE_POLL_PERIOD = 0.05  # [s]

# hover 後の押し込み姿勢の補正 (_refine_press_in)。MediaPipe Hands で手だけ検出し、
# 複数フレームの中央値を使う。差し出し判定時、直近 OFFER_HANDS_WINDOW 秒の Hands の
# 左右判定が過半数食い違えば Holistic も掌甲を取り違えているとみなし掌を裏返す。
OFFER_HANDS_WINDOW = 1.0  # [s]
OFFER_HANDS_MIN_FRAMES = 5
OFFER_HANDS_MAX_DISTANCE = 0.15  # [m] Holistic の掌からの距離

PRESS_IN_REFINE_FRAMES = 3
PRESS_IN_REFINE_TIMEOUT = 1.5  # [s]
PRESS_IN_REFINE_MAX_HAND_DISTANCE = 0.25  # [m] 計画時の掌からの距離
# Hands は掌甲を取り違えて法線が反転することがあるので score と法線角の両方で弾く。
PRESS_IN_REFINE_MIN_HAND_SCORE = 0.8
PRESS_IN_REFINE_MAX_NORMAL_ANGLE_DEG = 70.0
# 接近の螺旋〜押し込み直前、首は差し出し手の掌を向く。候補は直前位置からこの距離
# 以内・score 以上のみ。
HAND_FOLLOW_MAX_JUMP = 0.25  # [m]
HAND_FOLLOW_MIN_SCORE = PRESS_IN_REFINE_MIN_HAND_SCORE

# 骨格が未検出になってもこの秒数は直前の骨格を表示し続ける (ちらつき防止)。
SKELETON_HOLD_TIMEOUT = 1.0
# 骨格の再描画周期 [秒] (状態変化は即座に反映)。
SKELETON_REDRAW_INTERVAL = 0.5


def collision_pairs_text(colliding):
    """指先まで含めた事後検証であることを示す見出しを付けた干渉ペア表示。"""
    return common_collision_pairs_text(
        colliding, label='表示中の waypoint の事後検証 (指先まで含む)')


def face_point(joint_positions):
    """骨格の顔の位置 (鼻、なければ両耳の中点、どちらも無ければ None)。"""
    nose = joint_positions.get('Nose')
    if nose is not None:
        return np.asarray(nose, dtype=np.float64)[:3]
    ears = [joint_positions.get(name) for name in ('LEar', 'REar')]
    if all(ear is not None for ear in ears):
        return (np.asarray(ears[0], dtype=np.float64)[:3]
                + np.asarray(ears[1], dtype=np.float64)[:3]) / 2.0
    return None


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
# 起動ごとに LOG_ROOT/<起動時刻> を作り、LOG_ROOT/latest をそこへ向ける。
LOG_ROOT = '/tmp/run_camera_pipeline_test_logs'
LOG_DIR = LOG_ROOT  # setup_log_dir() で起動ごとのディレクトリに置き換わる
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


class _StderrTee(object):
    """標準エラー (rospy.logwarn・警告・トレースバック) を画面に出しつつ、
    標準出力と同じログファイルへも書く。"""

    def __init__(self, stream, tee_stream):
        self._stream = stream
        self._tee_stream = tee_stream

    def write(self, text):
        self._stream.write(text)
        self._tee_stream.write_file_only(text)
        return len(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


_tee_stream = None


def setup_log_dir():
    """起動ごとのログディレクトリ (``LOG_ROOT/<起動時刻>``、既存のものは消さない)
    を作り、標準出力をファイルへも書くようにする。ログファイルのパスを返す。"""
    global _tee_stream, LOG_DIR
    LOG_DIR = os.path.join(LOG_ROOT, time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(LOG_DIR, exist_ok=True)
    latest = os.path.join(LOG_ROOT, 'latest')
    try:
        if os.path.lexists(latest):
            os.remove(latest)
        os.symlink(LOG_DIR, latest)
    except OSError:
        pass
    path = os.path.join(LOG_DIR, LOG_STARTUP_NAME)
    _tee_stream = _TeeStream(sys.stdout, open(path, 'w', encoding='utf-8'))
    sys.stdout = _tee_stream
    # init_node より前に差し替える (rospy は init_node 時の sys.stderr を掴む)。
    sys.stderr = _StderrTee(sys.stderr, _tee_stream)
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



def log_debug_event(record):
    """デバッグ用 ``{"event": ...}`` を JSON 1 行でログファイルにだけ書く。"""
    log_debug('[debug] ' + json.dumps(record, ensure_ascii=False))


def motion_verified(motion):
    """経路計画の区間と lead-in の両方が干渉検証に通っているか。"""
    return bool(motion['verified']) and bool(motion['lead_in_verified'])


def untranslate_result(result, offset):
    """人物を平行移動した仮想座標系の ``result`` を実際の base_link 系へ戻す (破壊的)。"""
    dx, dy = offset
    for key in ('target_position', 'hand_position', 'base_position',
                'palm_position'):
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
        # 押し込み中の台車 [x, y, yaw]。
        for base in post_process.get('press_in_base_path') or []:
            base[0] -= dx
            base[1] -= dy


def untranslate_motion(motion, offset):
    """``motion`` の各 waypoint の台車位置 (横並び移動は人の腕も) を実際の
    base_link 系へ戻す (破壊的)。"""
    dx, dy = offset
    transition = motion.get('transition') or {}
    for wp in (motion['waypoints'] + motion.get('lead_in_waypoints', [])
               + transition.get('waypoints', [])):
        wp['base_position'][0] -= dx
        wp['base_position'][1] -= dy
    arms = [wp.get('human_arm') for wp in transition.get('waypoints', [])]
    for arm in [transition.get('press_arm')] + arms:
        if not arm:
            continue
        for key in ('elbow', 'wrist', 'palm_position'):
            if arm.get(key) is not None:
                arm[key][0] -= dx
                arm[key][1] -= dy


class _DeferredSolver(object):
    """軌道最適化のソルバーを包み、裏の準備が終わるまで ``solve`` だけを待たせる
    (準備していなければ待たない)。"""

    def __init__(self, solver):
        self._solver = solver
        self._ready = threading.Event()
        self._ready.set()

    def set_pending(self):
        self._ready.clear()

    def set_ready(self):
        self._ready.set()

    def wait_ready(self, timeout=None):
        return self._ready.wait(timeout)

    def solve(self, *args, **kwargs):
        if not self._ready.is_set():
            t0 = time.time()
            print('[motion] 軌道最適化の準備が終わるのを待っています...')
            self._ready.wait()
            print('[motion] 軌道最適化の準備を {:.1f} 秒待ちました。'.format(
                time.time() - t0))
        return self._solver.solve(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._solver, name)


def _optimizer_warmup_child(args):
    """軌道最適化の準備を行う別プロセスの本体 (spawn)。JAX の永続キャッシュへ
    書くだけでよい。GIL 回避のため別プロセス。"""
    args = copy.copy(args)
    args.side_by_side_approach = 'sync'  # 寄せ用の別モデルは作らない
    planner = HandshakePlanner(args)
    phm.arms_down_angles(planner.robot, planner.robot.joint_list)
    planner.warmup(batch_ik=False)


class HandshakePlanner(object):
    """差し出された掌に対する IK -> 軌道計画 -> 横並び移動の計画。

    カメラ・画面・実機に依存せず、実機ノードと ``resolve_saved_trials.py`` が
    共有する。IK で ``self.robot`` の姿勢が書き換わる。
    """

    _WARMUP_PALM = dict(
        position=[0.5, 0.0, 1.0],
        x_axis=[1.0, 0.0, 0.0],
        y_axis=[0.0, 1.0, 0.0],
    )

    # ウォームアップ中だけ固定する numpy の乱数シード。
    _WARMUP_SEED = 0

    def __init__(self, args):
        self.args = args
        self.robot, self.verification_pairs = self._build_robot_and_pairs()
        self.motion_verification_pairs = self.verification_pairs
        # JIT キャッシュのためノードの寿命で使い回す。
        self._real_solver = create_solver(
            'jaxls', max_iterations=args.max_iterations, verbose=False)
        self.solver = _DeferredSolver(self._real_solver)
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
        # 寄せ用の別モデル (IK で姿勢を書き換えるので共有できない)。
        self.refine_robot = None
        if self.side_by_side_approach == 'background':
            self.refine_robot, self.refine_verification_pairs = \
                self._build_robot_and_pairs()
            self.refine_collision_pairs = (
                None if self.collision_pairs is None
                else spik.load_collision_pairs(args.collision_pairs,
                                               self.refine_robot))

    @property
    def side_by_side_approach(self):
        return getattr(self.args, 'side_by_side_approach', 'sync')

    def _build_robot_and_pairs(self):
        """IK・軌道計画用の指なしロボットと、事後検証用の総当たりペアを作る。"""
        robot = Aero(use_hand=False)
        spik.restrict_elbow_range(robot)
        spik.restrict_leg_range(robot)
        spik.restrict_waist_range(robot)
        spik.restrict_neck_range(robot)
        spik.lock_fixed_joints(robot)
        spik.apply_collision_model(robot)
        spik.apply_hand_box(robot)
        spik.other_hand_points('r')
        pairs = spik.build_verification_pairs_for_model(
            robot, self.args.collision_verify_model)
        return robot, pairs

    def warmup(self, background_optimizer=False, batch_ik=True):
        """左右の腕で IK と軌道最適化をダミー目標で解き、JAX のトレース・
        コンパイルを済ませる。

        ``plan`` と引数を完全に一致させること (違うとキャッシュミス)。
        ``background_optimizer`` なら軌道最適化は別プロセス
        (``_optimizer_warmup_child``) で行い、最適化が要る計画だけが完了を待つ。
        ``batch_ik=False`` は横並び移動のバッチ IK の準備を省く。
        """
        random_state = np.random.get_state()
        np.random.seed(self._WARMUP_SEED)
        try:
            self._warmup_body(background_optimizer, batch_ik)
        finally:
            np.random.set_state(random_state)

    def wait_optimizer_ready(self, timeout=None):
        """軌道最適化の準備が終わるまで待つ (終わっていれば True)。"""
        return self.solver.wait_ready(timeout)

    def _pin_warmup_fixed_joints(self, robot_arm, handshake):
        """軌道で動かさない関節 (差し出さない腕・頭など) の角度を IK の結果に
        依らない固定値にする (FK 定数として焼き込まれ、変わるとキャッシュが外れる)。
        """
        chain = {link.joint.name for link in getattr(
            self.robot, '{}arm_whole_body'.format(robot_arm)).link_list}
        other = 'l' if robot_arm == 'r' else 'r'
        shoulder_p, elbow = spik.OTHER_ARM_POSTURES_DEG[0]
        posture = {'{}_shoulder_p_joint'.format(other): np.radians(shoulder_p),
                   '{}_elbow_joint'.format(other): np.radians(elbow)}
        joints = {j.name: j for j in self.robot.joint_list}
        names = handshake['joint_names']
        angles = list(handshake['joint_angle_vector'])
        for i, name in enumerate(names):
            if name in chain:
                continue
            joint = joints[name]
            angle = posture.get(name, 0.0)
            angles[i] = float(np.clip(angle, joint.min_angle, joint.max_angle))
        handshake['joint_angle_vector'] = angles

    def _optimizer_warmup_job(self, robot, pairs, robot_arm, label, handshake,
                              human_xy, motion_args, solver):
        """軌道最適化の準備 1 件 (``plan`` と同じ引数で ``plan_person_motion`` を解く)。"""
        t0 = time.time()
        phm.plan_person_motion(
            robot, robot_arm, handshake, {}, human_xy, motion_args, pairs,
            solver)
        log_debug('[warmup] {}腕: 軌道最適化 {:.1f} 秒'.format(
            label, time.time() - t0))

    def _start_optimizer_warmup_process(self):
        """軌道最適化の準備を別プロセスで始め、終わったら ``self.solver`` の待ちを
        解く監視スレッドを立てる (失敗でも解く)。"""
        process = multiprocessing.get_context('spawn').Process(
            target=_optimizer_warmup_child, args=(self.args,),
            name='optimizer-warmup', daemon=True)
        self.solver.set_pending()
        t0 = time.time()
        process.start()

        def watch():
            process.join()
            if process.exitcode == 0:
                print('[warmup] 軌道最適化の準備が別プロセスで完了しました '
                      '({:.1f} 秒)。'.format(time.time() - t0))
            else:
                print('[warmup] 軌道最適化の別プロセスの準備が失敗しました '
                      '(終了コード {})。最適化が要る人の最初の計画でコンパイル'
                      'します。'.format(process.exitcode))
            self.solver.set_ready()

        threading.Thread(target=watch, name='optimizer-warmup-watch',
                         daemon=True).start()
        self._optimizer_process = process

    def _warmup_body(self, background_optimizer=False, batch_ik=True):
        """``warmup`` の本体 (乱数固定は呼び出し側が行う)。"""
        args = self.args
        print('[warmup] 左右の腕の IK・軌道最適化トレースを事前に実行して '
              'います (数秒~数十秒かかります)...')
        warmup_t0 = time.time()
        self._optimizer_process = None
        if background_optimizer:
            self._start_optimizer_warmup_process()
        collision_obstacles = (
            [] if self.collision_pairs is None
            else spik.human_body_obstacles({}))
        warmup_human_xy = np.array([args.human_front_distance, 0.0])
        motion_args = copy.copy(args)
        motion_args.collision_verify_tolerance = \
            args.motion_collision_verify_tolerance
        motion_args.force_optimize = True
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
                self_collision=self.collision_pairs is not None,
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
                self.robot, robot_arm, rots[turn_index],
                turn_index, angle_vector, base_pose, self.base_limits,
                post_process_result, 0.0, 0.0, hand, self._WARMUP_PALM)
            self._pin_warmup_fixed_joints(robot_arm, handshake)
            if not background_optimizer:
                self._optimizer_warmup_job(
                    self.robot, self.motion_verification_pairs, robot_arm,
                    label, handshake, warmup_human_xy, motion_args,
                    self.solver)
            if batch_ik:
                # 横並び移動のバッチ IK (回転の拘束が違い別にコンパイルされる)。
                t0 = time.time()
                sbs.warmup_batch_ik(
                    self.robot, robot_arm, hand, angle_vector,
                    self.collision_pairs, self.base_limits)
                if self.refine_robot is not None:
                    sbs.warmup_batch_ik(
                        self.refine_robot, robot_arm, hand, angle_vector,
                        self.refine_collision_pairs, self.base_limits)
                log_debug('[warmup] {}腕: 横並び移動のバッチ IK {:.1f} 秒'
                          .format(label, time.time() - t0))
        print('[warmup] 完了しました ({:.1f} 秒)。{}'.format(
            time.time() - warmup_t0,
            '軌道最適化の準備は別プロセスで続けています。'
            if self._optimizer_process is not None else ''))

    def plan(self, joint_positions, palms, offered_hand, attempt):
        """差し出された掌 ``palms[offered_hand]`` に対して IK -> 軌道計画 ->
        横並び移動を計画する (``attempt`` はログ用)。

        戻り値は dict(``result`` (base_link 系の IK 結果)、``motion`` (軌道。
        押し込み姿勢が解けなければ None)、``robot_arm``)。
        """
        args = self.args
        robot_arm = spik.DEFAULT_ROBOT_ARM[offered_hand]
        palm = palms[offered_hand]

        offset = spik.human_translation_offset(
            joint_positions, front_distance=args.human_front_distance)
        translated_joints = spik.translate_joint_positions(
            joint_positions, offset)
        translated_palm = spik.translate_palm(palm, offset)
        # 干渉判定だけは体幹を奥へずらした骨格を使う (関節は体表面の点)。
        collision_joints = spik.shift_torso_joints_from_surface(
            translated_joints, offset, args.torso_surface_offset)
        collision_obstacles = (
            [] if self.collision_pairs is None
            else spik.human_body_obstacles(collision_joints))
        refine_job = None

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
                self_collision=self.collision_pairs is not None,
                collision_pairs=self.collision_pairs,
                joint_positions=collision_joints,
                placement_joint_positions=translated_joints,
                verification_pairs=self.verification_pairs)

        if picked is None:
            result = spik.unsolved_result(
                self.robot, robot_arm, rots[-1],
                person_base_limits, collision_ik_time,
                candidate_selection_time, offered_hand, translated_palm)
        else:
            turn_index, angle_vector, base_pose, post_process_result = picked
            result = spik.solved_result(
                self.robot, robot_arm, rots[turn_index],
                turn_index, angle_vector, base_pose, person_base_limits,
                post_process_result, collision_ik_time,
                candidate_selection_time, offered_hand, translated_palm)
        result['offered_hand'] = offered_hand
        result['robot_arm'] = robot_arm
        result['base_x_standing_margin'] = x_margin

        # 押し込み姿勢まで解けたら軌道計画する (untranslate の前に行う)。
        motion = None
        if result['solved'] and result.get('post_process') is None:
            print('[motion] 試行{}: 押し込み姿勢 (押し付け/視線 IK) が解けな'
                  'かったため、軌道計画をしません。'.format(attempt))
        elif result['solved']:
            human_xy = spik.human_standing_xy(translated_joints)
            if human_xy is None:
                human_xy = np.array([args.human_front_distance, 0.0])
            motion_args = copy.copy(args)
            motion_args.collision_verify_tolerance = \
                args.motion_collision_verify_tolerance
            initial_base_pose = np.array([offset[0], offset[1], 0.0])
            motion = phm.plan_person_motion(
                self.robot, robot_arm, result, collision_joints, human_xy,
                motion_args, self.motion_verification_pairs, self.solver,
                initial_base_pose=initial_base_pose)
            log_debug_event(dict(
                event='motion', person=attempt,
                verified=motion['verified'],
                lead_in_verified=motion['lead_in_verified'],
                min_dist=float(min(motion['waypoint_min_distances'])),
                waypoint_min_distances=[
                    float(d) for d in motion['waypoint_min_distances']],
                kind=phm.KIND_LABELS.get(motion['kind'], motion['kind']),
                compute_time=motion['compute_time']))
            # 押し込み後に手を下ろしながら横並びになる区間 (実行可能なときだけ)。
            if (result.get('post_process') is not None
                    and motion_verified(motion)):
                # background/off は寄せを解かず最初の解で止める (background は
                # 移動中に refine_transition で解く)。
                transition = sbs.plan_transition(
                    self.robot, robot_arm, offered_hand,
                    result['post_process'], result['turn_deg'],
                    translated_joints, collision_joints,
                    self.verification_pairs,
                    self.collision_pairs, self.base_limits,
                    approach=self.side_by_side_approach == 'sync')
                motion['transition'] = transition
                if (self.refine_robot is not None
                        and transition['verified']):
                    # 平行移動後の座標系のまま渡す (untranslate は破壊的)。
                    refine_job = dict(
                        attempt=attempt, robot_arm=robot_arm,
                        hand=offered_hand,
                        post=copy.deepcopy(result['post_process']),
                        turn_deg=result['turn_deg'],
                        translated_joints=copy.deepcopy(translated_joints),
                        collision_joints=copy.deepcopy(collision_joints),
                        secured=copy.deepcopy(transition), offset=offset)
                print('[transition] 試行{}: 横並び移動 {}'.format(
                    attempt, sbs.transition_summary(transition)))
                log_debug_event(dict(
                    event='transition', person=attempt,
                    verified=transition['verified'],
                    fraction=transition['fraction'],
                    x_margin=transition['x_margin'],
                    placement_before=transition['placement_before'],
                    placement_after=transition['placement_after'],
                    reason=transition['reason'],
                    compute_time=transition['compute_time']))

        placement = (spik.base_placement_metrics(
            translated_joints, result['base_position'], result['base_yaw'])
            if result['solved'] else None)

        untranslate_result(result, offset)
        if motion is not None:
            untranslate_motion(motion, offset)

        log_debug_event(dict(
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
                      '成功' if motion_verified(motion)
                      else '失敗 (干渉検証 NG)',
                      phm.KIND_LABELS.get(motion['kind'], motion['kind']),
                      collision_ik_time + candidate_selection_time,
                      motion['compute_time']))
        return dict(result=result, motion=motion, robot_arm=robot_arm,
                    transition_refine=refine_job)

    def refine_transition(self, job):
        """``plan`` が返した ``transition_refine`` の横並び移動を、寄せ (段階 2) と
        掌の向きで良くして解き、base_link 系に戻す (裏スレッド用。
        ``self.refine_robot`` のみ使う)。"""
        transition = sbs.refine_transition(
            self.refine_robot, job['robot_arm'], job['hand'], job['post'],
            job['turn_deg'], job['translated_joints'],
            job['collision_joints'], self.refine_verification_pairs,
            self.refine_collision_pairs, self.base_limits, job['secured'])
        untranslate_motion(dict(waypoints=[], transition=transition),
                           job['offset'])
        return transition


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
        self.pose_estimator.estimate_hands_3d(
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.zeros((240, 320), dtype=np.float32),
            CameraIntrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0))

        # IK・軌道計画 (IK は指なし、画面は指ありの display_robot)。
        self.planner = HandshakePlanner(args)
        self.robot = self.planner.robot
        self.verification_pairs = self.planner.verification_pairs
        self.motion_verification_pairs = \
            self.planner.motion_verification_pairs
        self.solver = self.planner.solver
        self.collision_pairs = self.planner.collision_pairs
        self.base_limits = self.planner.base_limits
        self._attach_camera_optical_coords()
        # 初期姿勢の関節角 (self.robot は IK で上書きされる)。
        self._initial_joint_names = [j.name for j in self.robot.joint_list]
        self._initial_joint_angle_vector = [
            float(v) for v in
            phm.arms_down_angles(self.robot, self.robot.joint_list)]
        self.display_robot = load_aero(use_hand=True)
        phm.arms_down_angles(self.display_robot, self.display_robot.joint_list)
        self._initial_base_coords = self.display_robot.base_link.copy_worldcoords()
        self._robot_hand_position_fallback = \
            self._compute_robot_hand_position_fallback()
        self.robot_position = self._resolve_robot_position()
        log_debug('[robot-hand-position] {} (base_link)'.format(
            self.robot_position.tolist()))

        max_distance = (None if args.max_person_distance <= 0
                        else args.max_person_distance)
        self.offered_hand_selector = epp.OfferedHandSelector(
            robot_position=self.robot_position,
            score_min=args.offer_score_min,
            max_distance=max_distance)
        self.palm_estimator = epp.PalmPoseEstimator(self.offered_hand_selector)

        self._joint_smoother = skeleton_filters.OneEuroFilter(
            mincutoff=args.joint_smoothing_mincutoff,
            beta=args.joint_smoothing_beta,
            dcutoff=args.joint_smoothing_dcutoff)

        self._lock = threading.Lock()
        # viewer の add/delete/redraw は直列化する (別スレッド同時呼びで落ちる。
        # self._lock 保持中にも呼ぶので別 lock)。
        self._viewer_lock = threading.Lock()
        self._latest_joint_positions = None  # 最新フレームの joint_positions (dict) or None
        self._latest_is_base_frame = False   # 上記が base_link 座標系か (TF 解決済みか)
        self._hand_frames_requested = False
        self._hand_frames = []
        self._hand_tracking = False
        # ARMED 中の Hands の結果 [(stamp, hands, カメラ位置)] (直近だけ)。
        self._offer_hands = collections.deque()
        self._current_palm = None         # 直近の IK に使った掌 (base_link、平行移動前) or None
        self._latest_offer_selection = None  # ARMED 中の直近の差し出し手判定の内訳 or None
        # 'idle' -> 'armed' -> 'solving' -> 'result' (RESET で 'idle')。
        # --auto-execute で失敗したら 'armed' に戻る。
        self.state = 'idle'
        # ARMED に戻った時点の左右の掌 (_offer_changed)。None ならすぐ受け付ける。
        self._retry_reference = None
        self._busy = False                # IK 計算中は次フレームの処理を止める
        self._last_failure_speech = None
        self._nod_done = threading.Event()
        self._nod_done.set()
        self._initial_pose_done = threading.Event()
        self._initial_pose_done.set()
        # 首の追従の指令とうなずきの開始を排他にする (_head_tracking_loop)。
        self._head_lock = threading.Lock()
        # 直近に人が見えた (time.time(), 体幹の base_link 座標, 顔の座標 or None,
        #  odom <- base_link の 4x4 or None) or None
        self._head_person_seen = None
        # set の間は状態によらず首で人の顔を追う (実機の実行中)。
        self._head_face_tracking = threading.Event()
        # set の間は首で差し出し手の掌を追う (顔の追従より優先)。_head_hand_seen は
        # (time.time(), 掌の base_link 座標, odom <- base_link)、検出まで計画時の掌。
        # _hand_follow は dict(side, normal_odom) or None。
        self._head_hand_following = threading.Event()
        self._head_hand_seen = None
        self._hand_follow = None
        # 手の追従中の掌検出履歴 (time.time(), odom 系の位置, 台車停止から
        # IN_MOTION_SETTLE 秒たったか)。_correct_base_tail 用。
        self._hand_detections = collections.deque(maxlen=60)
        # odom の速度が最後に IN_MOTION_MAX_* を超えていた時刻 (time.time())。
        self._base_moving_time = time.time()
        # 直近に送った台車ゴールの計画 (_execute_waypoint_segment): 区間先頭基準の
        # 累積 [dx, dy, dyaw]・time_list・送信時の odom・送信時刻。
        self._segment_base_plan = None
        self._tail_correction = None
        self._segment_time_list = None
        # 直近に人へ向けた首ヨー指令 [rad] (見渡し中・停止中は None)。
        self._head_tracking_yaw = None
        self._frozen_joint_positions = None  # 差し出し手が決まった瞬間の骨格 (固定表示) or None
        self._current_result = None       # 直近の IK 結果 dict or None
        self._current_motion = None       # 直近の軌道計画結果 dict or None
        self._handshake_total_time = None  # 掌推定開始〜軌道計画完了の時間 [秒] or None
        self._display_waypoints = None    # 表示用 waypoint リスト or None
        self._display_n_prepend = 0        # 先頭の初期位置->経路開始点の表示専用フレーム数
        self._display_n_approach = 0      # 経路計画済みのフレーム数
        self._display_n_transition = 0    # 末尾の押し込み後の横並び移動のフレーム数
        self._refine_lock = threading.Lock()  # 寄せの裏の計画を 1 つずつ
        self._collision_pairs_text = ''   # 指ありでの事後検証結果
        self._last_detected_joint_positions = None
        self._last_detected_time = None
        self._displayed_joint_positions = None
        self._last_skeleton_redraw_time = 0.0
        self._attempt_count = 0  # 試行番号 (ログ用)

        self.planner.warmup(
            background_optimizer=args.background_optimizer_warmup)

        # 実機接続は --auto-execute によらず試みる (失敗しても viewer として続行)。
        # 表示スレッドとの競合を避け、実機専用の robot_model を持つ。
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

        # ARMED 中の首の動き用に、視線 IK で首だけ動かすモデルを別に持つ。
        self._gaze_robot = None
        if self.ri is not None and args.head_tracking:
            self._gaze_robot = Aero(use_hand=False)
            spik.restrict_neck_range(self._gaze_robot)
            name_to_initial_angle = dict(zip(
                self._initial_joint_names, self._initial_joint_angle_vector))
            for joint in self._gaze_robot.joint_list:
                if joint.name in name_to_initial_angle:
                    joint.joint_angle(name_to_initial_angle[joint.name])
            spik.attach_camera_optical_coords(
                self._gaze_robot, *self._camera_optical_pos_rot)
            threading.Thread(
                target=self._head_tracking_loop, daemon=True).start()

        self.sound_client = None
        if args.auto_execute and self.ri is not None:
            try:
                from sound_play.libsoundplay import SoundClient
                self.sound_client = SoundClient(
                    sound_action='robotsound_jp', sound_topic='robotsound_jp')
            except Exception as exc:  # noqa: BLE001
                print('[speech] SoundClient の初期化に失敗したため発話し '
                      'ません ({})。'.format(exc))
        self.offer_advisor = None
        if args.speech_advice:
            try:
                self.offer_advisor = hand_offer_advice.OfferAdvisor(
                    args.hand_offer_table)
            except Exception as exc:  # noqa: BLE001
                print('[advice] 手の出し方の表 ({}) を読めないため、IK 失敗時'
                      'に出し方を助言しません ({})。'.format(
                          args.hand_offer_table, exc))

        self.skeleton_image_pub = rospy.Publisher(
            '~skeleton_image', Image, queue_size=1)

        # camera_info は同期に含めず最新 1 つだけ持つ (画像が遅れると組ができない)。
        self._latest_camera_info = None
        self.info_sub = rospy.Subscriber(
            args.camera_info_topic, CameraInfo, self._on_camera_info,
            queue_size=1)
        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_frame)
        self._latest_scan = None
        if args.base_correction == 'scan':
            self.scan_sub = rospy.Subscriber(
                args.scan_topic, LaserScan, self._on_scan, queue_size=1)

        self._setup_viewer(args)

        if args.auto_arm:
            self._arm('--auto-arm', move_to_initial_pose=False)

    def _setup_viewer(self, args):
        """viser ビューアに ARM/RESET ボタン・状態表示・waypoint スライダー・
        ロボットモデルを準備する。"""
        self.viewer = ViserViewer(draw_grid=True)
        # 干渉 overlay は指ありで作る (GUI コールバックから参照されるので登録前に)。
        self.robot_collision_overlay = build_robot_collision_overlay(
            self.display_robot)
        sync_robot_collision_overlay(
            self.robot_collision_overlay, self.display_robot)
        self.hand_verification_pairs = spik.build_collision_verification_pairs(
            self.robot_collision_overlay, 'r')
        self._current_obstacle_links = []  # 人体側の干渉回避ジオメトリの overlay
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
        self.waypoint_slider = self.viewer._server.gui.add_slider(
            'waypoint', min=0, max=0, step=1, initial_value=0)
        self.play_checkbox = self.viewer._server.gui.add_checkbox(
            'Play', initial_value=False)

        @self.waypoint_slider.on_update
        def _on_waypoint(_):  # noqa: ANN001
            self._apply_current_waypoint()

        @self.play_checkbox.on_update
        def _on_play_toggle(_):  # noqa: ANN001
            if (self.play_checkbox.value
                    and int(self.waypoint_slider.value)
                    >= self.waypoint_slider.max):
                self.waypoint_slider.value = 0

        threading.Thread(target=self._play_loop, daemon=True).start()

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
        initial_pose_axis = Axis(axis_length=INITIAL_POSE_AXIS_LENGTH,
                                 axis_radius=INITIAL_POSE_AXIS_RADIUS)
        initial_pose_axis.newcoords(self._initial_base_coords.copy_worldcoords())
        self.viewer.add(initial_pose_axis)
        # add で関節スライダーが自動追加されるので GUI の後に行う。
        self.viewer.add(self.display_robot)
        self.viewer.add(self.robot_collision_overlay)
        remove_joint_angle_gui(self.viewer)
        remove_obstacles_gui(self.viewer)
        self.viewer.show(open_browser=not args.no_open_browser)
        if not args.no_wait_for_client:
            viewer_nav.wait_for_client(self.viewer, args.client_wait_timeout)

    def _set_link_visible(self, link, visible):
        common_set_link_visible(self.viewer, link, visible)

    def _attach_camera_optical_coords(self, timeout=3.0):
        """視線 IK のカメラ光軸を TF (head_link -> optical frame) から
        ``self.robot`` に取り付ける (引けなければ既定値)。head_link 系の位置・
        回転は ``self._camera_optical_pos_rot`` (既定値なら ``(None, None)``) に控える。"""
        head_frame = self.robot.head_link.name
        optical_frame = self.args.camera_optical_frame
        self._camera_optical_pos_rot = (None, None)
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
        default_coords = spik.attach_camera_optical_coords(self.robot)
        default_axis = self.robot.head_link.worldrot().T.dot(
            default_coords.worldrot()[:, 2])
        matrix = transform_to_matrix(transform.transform)
        self._camera_optical_pos_rot = (matrix[:3, 3], matrix[:3, :3])
        spik.attach_camera_optical_coords(
            self.robot, *self._camera_optical_pos_rot)
        diff_deg = math.degrees(math.acos(float(np.clip(
            np.dot(default_axis, matrix[:3, 2]), -1.0, 1.0))))
        log_debug('[camera-optical] 視線 IK のカメラ光軸を TF ({} -> {}) から '
              '設定しました (pos=[{:.4f}, {:.4f}, {:.4f}]、既定値との光軸の '
              'ずれ {:.2f} 度)。'.format(head_frame, optical_frame,
                                     *(list(matrix[:3, 3]) + [diff_deg])))

    def _resolve_robot_position(self):
        """差し出し手判定の基準にするロボット手先の base_link 座標を返す
        (TF が引けなければ右腕の種の姿勢の手先位置)。"""
        return lookup_frame_position(
            self.tf_buffer, self.args.base_frame, self.args.robot_hand_frame,
            self._robot_hand_position_fallback,
            warn_label='[run-camera-pipeline-test] ')

    def _compute_robot_hand_position_fallback(self):
        """TF 未解決時のフォールバック (右腕の種の姿勢の手先位置) を計算する。
        ``self.robot`` の姿勢を書き換える。"""
        spik.seed_arm_pose(self.robot, 'r')
        return np.asarray(self.robot.rarm_end_coords.worldpos(),
                          dtype=np.float64)

    def _lookup_camera_to_base(self, header):
        """画像の ``header`` から base_link への TF を引く。"""
        return lookup_camera_to_base(
            self.tf_buffer, self.args.base_frame, header)

    def _lookup_odom_from_base(self, stamp=None):
        """``stamp`` (None なら最新) の odom <- base_link の 4x4 を返す
        (引けなければ None)。stamp で引けなければ最新で引き直す。"""
        for when in ([stamp] if stamp is not None else []) + [rospy.Time(0)]:
            try:
                transform = self.tf_buffer.lookup_transform(
                    ODOM_FRAME, self.args.base_frame, when)
                return transform_to_matrix(transform.transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        return None

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
        # TF が引けなければカメラ座標系のままプレビューする (IK は base_link のみ)。
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
        preview_joint_positions = (
            None if raw_joint_positions is None
            else self._joint_smoother.update(
                raw_joint_positions, t=color_msg.header.stamp.to_sec(),
                frame_key=is_base_frame))
        armed_joint_positions = (
            preview_joint_positions if is_base_frame else None)
        torso_points = [
            armed_joint_positions[name] for name in HEAD_TRACK_TARGET_JOINTS
            if armed_joint_positions and name in armed_joint_positions]
        person_seen = None
        if torso_points:
            person_seen = (
                time.time(), np.mean(np.asarray(torso_points), axis=0),
                face_point(armed_joint_positions),
                self._lookup_odom_from_base(color_msg.header.stamp))

        with self._lock:
            self._latest_joint_positions = preview_joint_positions
            self._latest_is_base_frame = is_base_frame
            if person_seen is not None:
                self._head_person_seen = person_seen
            hand_frames_requested = self._hand_frames_requested
            hand_tracking = self._hand_tracking
            hand_follow = self._hand_follow
        armed = self.state == 'armed'
        if ((hand_frames_requested or hand_tracking or armed
             or hand_follow is not None) and is_base_frame):
            hands = self.pose_estimator.estimate_hands_3d(
                color, depth_m, intrinsics, output_transform=camera_to_base)
            stamp = color_msg.header.stamp.to_sec()
            if hand_follow is not None:
                self._update_hand_follow(
                    hands, hand_follow, camera_to_base,
                    self._lookup_odom_from_base(color_msg.header.stamp),
                    color_msg.header.stamp)
            with self._lock:
                if armed:
                    self._offer_hands.append(
                        (stamp, hands, np.asarray(camera_to_base)[:3, 3]))
                    while (self._offer_hands and self._offer_hands[0][0]
                           < stamp - OFFER_HANDS_WINDOW):
                        self._offer_hands.popleft()
                if self._hand_frames_requested:
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
        if move_to_initial_pose and self.ri is not None:
            # 首の追従は初期姿勢に戻ってから始める (全身のゴールを打ち切らない)。
            self._initial_pose_done.clear()
            self._head_tracking_yaw = None
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
        palms, offer_check = self._correct_mirrored_offer(
            palms, offered_hand, stamp)
        # 首の追従の指令送信中でないときに止める (うなずきが打ち切られないよう)。
        with self._head_lock:
            self._busy = True
            if self.ri is not None:
                self._nod_done.clear()
        if self.ri is not None:
            threading.Thread(target=self._nod_head, daemon=True).start()
        with self._lock:
            self._frozen_joint_positions = joint_positions
        self.state = 'solving'
        self.arm_button.visible = False
        self.reset_button.visible = True
        outcome = None
        try:
            outcome = self._solve_handshake(joint_positions, palms,
                                            offered_hand, offer_check)
        finally:
            self._busy = False
            self.state = 'result'
            with self._lock:
                self._handshake_total_time = time.time() - handshake_t0
        if outcome == 'execute':
            threading.Thread(
                target=self._execute_on_robot, daemon=True).start()
        elif outcome == 'failed':
            self._reset_view()
            self._arm('失敗', move_to_initial_pose=False,
                      wait_offer_change=True)

    def _correct_mirrored_offer(self, palms, side, stamp):
        """差し出し手 ``side`` の掌を Hands の左右判定と照合し、取り違え (鏡像) なら
        裏返した ``palms`` を返す (``(palms, ログ用の内訳)``)。

        右手の甲側は左手の掌側と同じ形なので、Hands が逆の手と判定し続けるときは
        Holistic の掌も裏返っている。Hands が手を捉えていなければ何もしない。
        """
        palm = palms[side]
        with self._lock:
            recent = [(hands, camera) for t, hands, camera in self._offer_hands
                      if stamp - OFFER_HANDS_WINDOW <= t <= stamp]
        labels = []
        for hands, camera in recent:
            best = None
            for hand in hands:
                if hand['score'] < PRESS_IN_REFINE_MIN_HAND_SCORE:
                    continue
                hand_palm = self.palm_estimator.estimate_palm(
                    epp.hand_joint_positions(hand, side), side, camera,
                    palm['y_axis'])
                if hand_palm is None:
                    continue
                dist = float(np.linalg.norm(np.asarray(hand_palm['position'])
                                            - np.asarray(palm['position'])))
                if dist <= OFFER_HANDS_MAX_DISTANCE and (
                        best is None or dist < best[0]):
                    best = (dist, hand['side'])
            if best is not None:
                labels.append(best[1])
        mismatched = sum(label != side for label in labels)
        record = dict(
            event='offer_hands_check', side=side, n_frames=len(recent),
            n_hands=len(labels), n_mismatched=mismatched,
            mirrored=(len(labels) >= OFFER_HANDS_MIN_FRAMES
                      and mismatched * 2 > len(labels)))
        if record['mirrored']:
            palms = dict(palms)
            palms[side] = dict(epp.flip_palm(palm), flipped_by_hands=True)
        return palms, record

    def _log_debug(self, record):
        """デバッグ用 ``{"event": ...}`` を JSON 1 行でログファイルにだけ書く。"""
        log_debug_event(record)

    def _solve_handshake(self, joint_positions, palms, offered_hand,
                         offer_check=None):
        args = self.args
        self._attempt_count += 1
        attempt = self._attempt_count
        log_path = switch_log_file(attempt)
        print('[ARMED] 試行{}: {}手を差し出しました。IK・軌道計画を始めます '
              '(ログ: {})。'.format(attempt, offered_hand, log_path))
        self._log_debug(dict(event='armed', person=attempt,
                             offered_hand=offered_hand))
        if offer_check is not None:
            self._log_debug(offer_check)
            if offer_check['mirrored']:
                print('[ARMED][WARN] Hands が {} フレーム中 {} フレームで逆の手'
                      'と判定したため、掌と甲を取り違えているとみなして{}手の'
                      '掌を裏返します。'.format(
                          offer_check['n_hands'],
                          offer_check['n_mismatched'], offered_hand))
        planned = self.planner.plan(
            joint_positions, palms, offered_hand, attempt)
        result, motion = planned['result'], planned['motion']
        palm = palms[offered_hand]
        if motion is not None and planned.get('transition_refine'):
            self._start_transition_refine(planned['transition_refine'],
                                          motion)

        if motion is not None:
            display_waypoints, n_approach = build_display_waypoints(
                motion, result)
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
        has_post_process = result.get('post_process') is not None
        executable = (
            args.auto_execute and self.ri is not None and result['solved']
            and has_post_process
            and motion is not None and display_waypoints is not None
            and self._execution_verified(motion))
        if motion is not None and not self._motion_verified(motion):
            print('[execute] 軌道の干渉検証に通らなかったため (verified={}, '
                  'lead_in_verified={})、--auto-execute でも実機を動かし'
                  'ません。'.format(
                      motion['verified'], motion['lead_in_verified']))
        elif motion is not None and not self._execution_verified(motion):
            print('[execute] 横並び移動が解けなかったため ({})、'
                  '--auto-execute でも実機を動かしません。'.format(
                      (motion.get('transition') or {}).get('reason')))
        if result['solved'] and not has_post_process:
            print('[execute] 後処理 (押し付け/視線 IK) が解けず押し込み区間が'
                  'ないため、--auto-execute でも実機を動かしません。')
        if display_waypoints is not None:
            self._set_waypoint_slider_range(len(display_waypoints) - 1)
        else:
            self._set_waypoint_slider_range(0)

        self._save_attempt(joint_positions, palms, result, motion)

        # 戻り値: 'execute' (実機で実行) / 'failed' (--auto-execute で失敗、
        # ARMED に戻る) / 'solved' (それ以外、RESET 待ち)。
        if executable:
            print('[auto-execute] IK・軌道計画が成功したため実機を動かします '
                  '(--auto-execute)。')
            return 'execute'
        if not (result['solved'] and has_post_process
                and motion is not None
                and self._execution_verified(motion)):
            speech = self._failure_speech(
                attempt, result, motion, joint_positions, palm, offered_hand)
            if args.auto_execute:
                self._last_failure_speech = speech
                self._say(speech)
                return 'failed'
        return 'solved'

    def _failure_speech(self, attempt, result, motion, joint_positions, palm,
                        offered_hand):
        """実機を動かせなかったときの発話文を返す。

        手の出し方が原因なら reachability map で近い解ける出し方との差を助言し、
        接近途中の干渉・横並び移動の失敗なら対応する ``--speech-*-fail-text`` を返す。
        """
        args = self.args
        cause = self._failure_cause(result, motion)
        if cause == 'approach':
            print('[advice] 試行{}: 接近の途中で干渉したため、手の出し方は'
                  '助言しません。'.format(attempt))
            self._log_debug(dict(event='advice', person=attempt, cause=cause))
            return args.speech_approach_fail_text
        if cause == 'transition':
            print('[advice] 試行{}: 横並び移動が解けなかったため、手の出し方は'
                  '助言しません。'.format(attempt))
            self._log_debug(dict(event='advice', person=attempt, cause=cause))
            return args.speech_transition_fail_text
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
        if motion_verified(motion):
            return 'transition'
        distances = motion['waypoint_min_distances']
        margins = motion.get('waypoint_human_clearance_margins') or \
            [None] * len(distances)
        failed = [i for i, (d, c) in enumerate(zip(distances, margins))
                  if not phm.motion_passes([d], [c])]
        start = near_hand_fraction * (len(distances) - 1)
        if failed and min(failed) >= start:
            return 'near_hand'
        return 'approach'

    _motion_verified = staticmethod(motion_verified)

    def _start_transition_refine(self, job, motion):
        """寄せ (横並び移動の段階 2) と掌の向きを裏スレッドで解き、終わったら
        ``motion['transition_refined']`` に入れる (``_refined_transition_waypoints``
        が使う)。裏の計画は 1 つずつ行う。"""
        def run():
            with self._refine_lock:
                t0 = time.time()
                try:
                    transition = self.planner.refine_transition(job)
                except Exception as e:  # noqa: BLE001 (確保済みの解で動ける)
                    print('[transition] 試行{}: 裏の計画 (寄せ) が失敗しました'
                          ' ({}: {})。確保済みの横並び移動を使います。'.format(
                              job['attempt'], type(e).__name__, e))
                    return
            with self._lock:
                motion['transition_refined'] = transition
            approach = transition.get('approach') or {}
            print('[transition] 試行{}: 裏の計画 ({:.1f} 秒) 横並び移動 {}'
                  .format(job['attempt'], time.time() - t0,
                          sbs.transition_summary(transition)))
            log_debug_event(dict(
                event='transition_refine', person=job['attempt'],
                verified=transition['verified'],
                fraction=transition['fraction'],
                palm_facing=transition.get('palm_facing'),
                approach_lateral=approach.get('lateral_after'),
                approach_waypoints=approach.get('n_waypoints'),
                compute_time=time.time() - t0))

        threading.Thread(target=run, name='transition-refine',
                         daemon=True).start()

    def _refined_transition_waypoints(self, motion, secured_waypoints):
        """押し込みの後に実行する横並び移動の waypoint。裏の計画
        (``_start_transition_refine``) が終わり確保済みの解以上に進めていればそれ、
        でなければ ``secured_waypoints``。"""
        with self._lock:
            refined = motion.get('transition_refined')
        secured = motion.get('transition') or {}
        if refined is None:
            if self.planner.refine_robot is not None:
                print('[execute] 裏の計画 (寄せ) が間に合わなかったため、確保'
                      '済みの横並び移動で進めます。')
            return secured_waypoints
        if (not refined.get('verified')
                or refined['fraction'] < secured.get('fraction', 0.0)):
            print('[execute] 裏の計画が確保済みの横並び移動より進めなかった'
                  'ため、確保済みの方で進めます。')
            return secured_waypoints
        approach = refined.get('approach') or {}
        print('[execute] 裏の計画の横並び移動を使います (掌は {}、寄せ '
              '{})。'.format(
                  refined.get('palm_facing'),
                  '横 {:.2f} -> {:.2f} m'.format(
                      approach['lateral_before'], approach['lateral_after'])
                  if approach.get('n_waypoints') else 'なし'))
        return transition_waypoints(dict(
            transition=refined, joint_names=motion['joint_names']))

    def _execution_verified(self, motion):
        """実機で実行してよい軌道か。接近・押し込みが検証に通り、横並び
        移動も解けていること。"""
        if not motion_verified(motion):
            return False
        transition = motion.get('transition')
        return bool(transition and transition.get('verified'))

    def _save_attempt(self, joint_positions, palms, result, motion):
        """試行の骨格/掌/IK 結果/軌道を JSON で保存する (``--save-dir``、無ければ
        ``LOG_DIR``)。失敗しても試行は続ける。"""
        try:
            self._save_attempt_files(joint_positions, palms, result, motion)
        except Exception as e:  # noqa: BLE001 (保存の失敗で実機を止めない)
            print('[save] 試行の JSON を保存できませんでした: {!r}'.format(e))

    def _save_attempt_files(self, joint_positions, palms, result, motion):
        root = self.args.save_dir or LOG_DIR
        stamp = time.strftime('%Y%m%d_%H%M%S')
        name = '{}.json'.format(stamp)
        skeleton_dir = os.path.join(root, 'skeletons')
        palm_dir = os.path.join(root, 'palms')
        handshake_dir = os.path.join(root, 'handshakes')
        motion_dir = os.path.join(root, 'motions')
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
        log_debug('[save] {} に保存しました (view_handshake_motion.py '
                  '--skeleton-dir <dir>/skeletons --handshake-dir '
                  '<dir>/handshakes --motion-dir <dir>/motions で後から '
                  '見返せる)。'.format(
                      os.path.join(root, subdirs, name)))

    def _set_waypoint_slider_range(self, max_index):
        """スライダーの範囲を ``[0, max_index]`` にし、waypoint 0 を表示する。"""
        self.waypoint_slider.max = max_index
        self.waypoint_slider.value = 0
        self._apply_current_waypoint()

    def _apply_current_waypoint(self):
        """スライダーの現在値の姿勢 (軌道が無ければ IK 結果か初期位置) を
        ``self.display_robot`` に反映する。"""
        with self._lock:
            result = self._current_result
            motion = self._current_motion
            display_waypoints = self._display_waypoints
        # 途中の不整合な姿勢が spin() の redraw に描かれないよう redraw まで lock 内で行う。
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
        """表示中の姿勢で指ありの事後検証をやり直し、表示用文字列を更新する
        (人体は画面の Cylinder を使う)。"""
        colliding = colliding_link_pairs(
            self.robot_collision_overlay, self.hand_verification_pairs,
            self._current_obstacle_links,
            tolerance=self.args.collision_verify_tolerance)
        self._collision_pairs_text = collision_pairs_text(colliding)

    def _move_to_initial_pose(self):
        """実機の全身を初期姿勢へ動かす。

        初期姿勢に含まれない関節 (指など) は実機の現在値のまま (``self.real_robot``
        は未同期なので先に反映)。終わったら (失敗しても) ``self._initial_pose_done``
        を set する。
        """
        try:
            self._move_to_initial_pose_body()
        finally:
            self._initial_pose_done.set()

    def _move_to_initial_pose_body(self):
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
        move_time, = self._limited_time_list(
            [target_av], None, [ARM_INITIAL_POSE_MOVE_TIME])
        self.ri.angle_vector(target_av, move_time)
        self.ri.wait_interpolation()
        print('[ARM] 腕・首を初期姿勢に戻しました。')

    def _nod_head(self):
        """首を ``HEAD_NOD_PITCH_DEG`` まで下げて初期角度に戻す。終わったら (失敗しても)
        ``self._nod_done`` を set する。``head_controller`` のみに送り腕のゴールは
        打ち切らない。"""
        try:
            current_av = np.asarray(self.ri.angle_vector(), dtype=np.float64)
            joint_list = self.real_robot.joint_list
            neck_index = joint_list.index(self.real_robot.neck_p_joint)
            initial_angle = dict(zip(
                self._initial_joint_names,
                self._initial_joint_angle_vector))[
                    self.real_robot.neck_p_joint.name]
            tracking_yaw = self._head_tracking_yaw
            if tracking_yaw is not None:
                current_av[joint_list.index(
                    self.real_robot.neck_y_joint)] = tracking_yaw
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

    def _head_tracking_allowed(self):
        """首の追従・見渡しの指令を送ってよいか (ARMED で、復帰・うなずきが終わっている)。"""
        return (self.state == 'armed' and not self._busy
                and self._nod_done.is_set()
                and self._initial_pose_done.is_set())

    def _face_target(self, seen):
        """``self._head_person_seen`` の顔 (無ければ体幹) を、今の base_link 系に
        直した点 (見えていなければ None)。"""
        if seen is None:
            return None
        _, torso, face, odom_from_base_then = seen
        point = torso if face is None else face
        if odom_from_base_then is None:
            return point
        odom_from_base_now = self._lookup_odom_from_base()
        if odom_from_base_now is None:
            return point
        point_odom = odom_from_base_then @ np.append(point, 1.0)
        return np.linalg.solve(odom_from_base_now, point_odom)[:3]

    def _head_tracking_loop(self):
        """ARMED 中、人が見えていればカメラ光軸を体幹へ向け、見えなくなって
        ``HEAD_SCAN_START_DELAY`` 秒たったら首ヨーを往復させて探す。

        ``head_controller`` のみに送る。差し出し手が決まると ``_try_handshake`` が
        ``self._head_lock`` の中で止める (うなずきのゴールを打ち切らないため)。
        ``self._head_face_tracking`` が set の間は状態によらず顔を追う (見失っても
        見渡さない)。
        """
        joint_list = self.real_robot.joint_list
        neck_y_index = joint_list.index(self.real_robot.neck_y_joint)
        neck_p_index = joint_list.index(self.real_robot.neck_p_joint)
        neck_r_index = joint_list.index(self.real_robot.neck_r_joint)
        gaze_robot = self._gaze_robot
        gaze_coords = spik.camera_optical_coords(gaze_robot)
        # 首以外 (腰・脚など) は毎回実機の角度に合わせる (実行中は動くため)。
        real_index = {joint.name: i for i, joint in enumerate(joint_list)}
        head_joint_names = {self.real_robot.neck_y_joint.name,
                            self.real_robot.neck_p_joint.name,
                            self.real_robot.neck_r_joint.name}
        body_joints = [
            (joint, real_index[joint.name]) for joint in gaze_robot.joint_list
            if joint.name in real_index and joint.name not in head_joint_names]
        initial_neck_p = gaze_robot.neck_p_joint.joint_angle()
        scan_yaw = np.deg2rad(HEAD_SCAN_YAW_DEG)
        scan_pitches = [initial_neck_p + np.deg2rad(offset)
                        for offset in HEAD_SCAN_PITCH_OFFSETS_DEG]
        mode = None  # 'face' / 'track' / 'scan' / None (止めている)
        # 見渡しの (首ヨー, 首ピッチ) の止まる向きの列と次の添字。
        scan_steps = []
        scan_index = 0
        scan_step_end_time = 0.0  # 今の向きで止まり終える時刻
        last_command = None  # 直近に送った追従の (首ヨー, 首ピッチ) [rad]
        while not rospy.is_shutdown():
            time.sleep(HEAD_TRACK_PERIOD)
            try:
                with self._head_lock:
                    prev_mode = mode
                    hand = self._head_hand_following.is_set()
                    face = self._head_face_tracking.is_set() or hand
                    if not face and not self._head_tracking_allowed():
                        if mode is not None:
                            log_debug('[head] 首の追従・見渡しを止めました。')
                        # _head_tracking_yaw はうなずきが読むので残す。
                        mode = None
                        last_command = None
                        continue
                    with self._lock:
                        seen = self._head_person_seen
                        hand_seen = self._head_hand_seen
                    now = time.time()
                    current_av = np.asarray(
                        self.ri.angle_vector(), dtype=np.float64)
                    for joint, index in body_joints:
                        joint.joint_angle(current_av[index])
                    target_av = current_av.copy()
                    target_av[neck_r_index] = spik.NECK_ROLL_FIXED_ANGLE
                    if face:
                        # 実行中は人の顔 (螺旋以降は差し出し手の掌) を追う。見えなければ待つ。
                        track_mode = 'hand' if hand else 'face'
                        target = self._face_target(
                            hand_seen if hand else seen)
                        if target is None:
                            continue
                        if mode != track_mode:
                            gaze_robot.neck_y_joint.joint_angle(
                                current_av[neck_y_index])
                            gaze_robot.neck_p_joint.joint_angle(
                                current_av[neck_p_index])
                        elif (spik._gaze_error(gaze_coords, target)
                              < np.deg2rad(HEAD_FACE_TRACK_DEADBAND_DEG)):
                            continue
                        spik._reaim_gaze(
                            gaze_robot, gaze_coords, target,
                            HEAD_TRACK_IK_STOP, HEAD_TRACK_IK_RTHRE)
                        target_av[neck_y_index] = \
                            gaze_robot.neck_y_joint.joint_angle()
                        target_av[neck_p_index] = \
                            gaze_robot.neck_p_joint.joint_angle()
                        speed = np.deg2rad(HEAD_FACE_TRACK_SPEED_DEG)
                        if mode != track_mode:
                            log_debug('[head] {}を追います。'.format(
                                '差し出し手の掌' if hand else '人の顔'))
                        mode = track_mode
                    elif (seen is not None
                            and now - seen[0] < HEAD_SCAN_START_DELAY):
                        # 人の方を向く。追従中は前回の指令から解く。
                        if mode != 'track':
                            gaze_robot.neck_y_joint.joint_angle(
                                current_av[neck_y_index])
                            gaze_robot.neck_p_joint.joint_angle(
                                current_av[neck_p_index])
                        elif (spik._gaze_error(gaze_coords, seen[1])
                              < np.deg2rad(HEAD_TRACK_DEADBAND_DEG)):
                            continue
                        spik._reaim_gaze(
                            gaze_robot, gaze_coords, seen[1],
                            HEAD_TRACK_IK_STOP, HEAD_TRACK_IK_RTHRE)
                        target_av[neck_y_index] = \
                            gaze_robot.neck_y_joint.joint_angle()
                        target_av[neck_p_index] = \
                            gaze_robot.neck_p_joint.joint_angle()
                        speed = np.deg2rad(HEAD_TRACK_SPEED_DEG)
                        if mode != 'track':
                            log_debug('[head] 人の方を向きます。')
                        mode = 'track'
                        self._head_tracking_yaw = target_av[neck_y_index]
                    else:
                        # 見渡す。端・中央の各向きで止まって下も見る (最初は今向いている側の端へ)。
                        if mode == 'scan' and now < scan_step_end_time:
                            continue
                        if mode != 'scan':
                            sign = (1.0 if current_av[neck_y_index] >= 0
                                    else -1.0)
                            scan_steps = [
                                (yaw, pitch)
                                for yaw in (sign * scan_yaw, 0.0,
                                            -sign * scan_yaw, 0.0)
                                for pitch in scan_pitches]
                            scan_index = 0
                            log_debug('[head] 人が見えないので見渡します。')
                        yaw, pitch = scan_steps[scan_index]
                        scan_index = (scan_index + 1) % len(scan_steps)
                        target_av[neck_y_index] = yaw
                        target_av[neck_p_index] = pitch
                        speed = np.deg2rad(HEAD_SCAN_SPEED_DEG)
                        mode = 'scan'
                        self._head_tracking_yaw = None
                    command = target_av[[neck_y_index, neck_p_index]]
                    if (mode in ('face', 'hand', 'track')
                            and prev_mode == mode and last_command is not None
                            and float(np.max(np.abs(command - last_command)))
                            < np.deg2rad(HEAD_TRACK_RESEND_DEG)):
                        continue
                    last_command = command.copy()
                    delta = float(np.max(np.abs(
                        target_av[[neck_y_index, neck_p_index]]
                        - current_av[[neck_y_index, neck_p_index]])))
                    time_list = self._limited_time_list(
                        [target_av], None,
                        [max(HEAD_TRACK_MIN_MOVE_TIME, delta / speed)])
                    self.ri.angle_vector_sequence(
                        [target_av], time_list,
                        controller_type='head_controller')
                    scan_step_end_time = now + sum(time_list) + HEAD_SCAN_DWELL
            except Exception as exc:  # noqa: BLE001  (首が動かせなくても差し出しの検出は続ける)
                rospy.logwarn_throttle(
                    5.0, '[head] 首の追従・見渡しに失敗しました ({})。'.format(
                        exc))

    def _set_face_tracking(self, enabled):
        """実行中の首の顔追従を始める/止める (追従が無効なら何もしない)。
        止めるときは ``self._head_lock`` の中で行う。"""
        if self._gaze_robot is None:
            return
        with self._head_lock:
            if enabled:
                self._head_face_tracking.set()
            else:
                self._head_face_tracking.clear()

    def _begin_hand_follow(self, side, palm_base, normal_base):
        """首で差し出し手 ``side`` の掌を追い始める。``palm_base`` / ``normal_base``
        は今の base_link 系での計画時の掌の位置と法線 (検出までこの点を向く)。
        odom が引けなければ False。"""
        odom_from_base = self._lookup_odom_from_base()
        if odom_from_base is None:
            return False
        normal_odom = odom_from_base[:3, :3] @ np.asarray(normal_base)
        with self._lock:
            self._head_hand_seen = (
                time.time(), np.asarray(palm_base, dtype=np.float64), None,
                odom_from_base)
            self._hand_follow = dict(side=side, normal_odom=normal_odom)
            self._hand_detections.clear()
            self._base_moving_time = time.time()
        with self._head_lock:
            self._head_hand_following.set()
        return True

    def _end_hand_follow(self):
        """差し出し手の掌の追従を止める (``_head_lock`` の中で止める)。"""
        with self._head_lock:
            self._head_hand_following.clear()
        with self._lock:
            self._hand_follow = None
            self._head_hand_seen = None

    def _base_settled_time(self):
        """odom の速度が IN_MOTION_MAX_* 以下になってからの秒数 (超えていれば 0)。
        呼ぶたびに更新する。"""
        msg = self.ri.odom_msg
        now = time.time()
        if msg is None:
            self._base_moving_time = now
        else:
            twist = msg.twist.twist
            if (abs(twist.angular.z) > IN_MOTION_MAX_YAW_RATE
                    or math.hypot(twist.linear.x, twist.linear.y)
                    > IN_MOTION_MAX_SPEED):
                self._base_moving_time = now
        return now - self._base_moving_time

    def _update_hand_follow(self, hands, follow, camera_to_base,
                            odom_from_base_then, stamp):
        """このフレームの ``hands`` から、直前に追った掌に最も近い差し出し手の掌を
        ``self._head_hand_seen`` に反映する (無ければ変えない)。``stamp`` は
        撮影時刻 (rospy.Time)。"""
        # 撮影時に台車が止まって IN_MOTION_SETTLE 秒たっていたか (処理遅れ分だけ長く待つ)。
        age = min(max((rospy.Time.now() - stamp).to_sec(), 0.0), 0.5)
        settled = self._base_settled_time() >= IN_MOTION_SETTLE + age
        if odom_from_base_then is None:
            return
        with self._lock:
            seen = self._head_hand_seen
        if seen is None:
            return
        point_odom = seen[3] @ np.append(seen[1], 1.0)
        expected = np.linalg.solve(odom_from_base_then, point_odom)[:3]
        expected_normal = odom_from_base_then[:3, :3].T @ follow['normal_odom']
        camera_position = np.asarray(camera_to_base)[:3, 3]
        best = None
        for hand in hands:
            if hand['score'] < HAND_FOLLOW_MIN_SCORE:
                continue
            joints = epp.hand_joint_positions(hand, follow['side'])
            palm = self.palm_estimator.estimate_palm(
                joints, follow['side'], camera_position, expected_normal)
            if palm is None:
                continue
            dist = float(np.linalg.norm(
                np.asarray(palm['position']) - expected))
            if dist <= HAND_FOLLOW_MAX_JUMP and (
                    best is None or dist < best[0]):
                best = (dist, np.asarray(palm['position'], dtype=np.float64))
        if best is not None:
            with self._lock:
                # 止められた後に書き戻さない。
                if self._hand_follow is follow:
                    self._head_hand_seen = (
                        time.time(), best[1], None, odom_from_base_then)
                    self._hand_detections.append((
                        time.time(),
                        (odom_from_base_then @ np.append(best[1], 1.0))[:3],
                        settled))

    def _set_hand_tracking(self, enabled):
        """手の追跡 (毎フレームの Hands 検出) を始める/止める。"""
        with self._lock:
            self._hand_tracking = bool(enabled)

    def _send_head_to_waypoint(self, waypoint, joint_names):
        """首だけを ``waypoint`` の角度へ向かわせる (待たない)。"""
        name_to_angle = dict(zip(joint_names, waypoint['joint_angle_vector']))
        joint_list = self.real_robot.joint_list
        current_av = np.asarray(self.ri.angle_vector(), dtype=np.float64)
        target_av = current_av.copy()
        for joint in (self.real_robot.neck_y_joint,
                      self.real_robot.neck_p_joint,
                      self.real_robot.neck_r_joint):
            if joint.name in name_to_angle:
                target_av[joint_list.index(joint)] = name_to_angle[joint.name]
        delta = float(np.max(np.abs(target_av - current_av)))
        time_list = self._limited_time_list(
            [target_av], None,
            [max(HEAD_TRACK_MIN_MOVE_TIME,
                 delta / np.deg2rad(HEAD_FACE_TRACK_SPEED_DEG))])
        self.ri.angle_vector_sequence(
            [target_av], time_list, controller_type='head_controller')

    def _execute_on_robot(self):
        """計画済みの waypoint 列を実機に送る (``_execute_on_robot_body``)。
        途中で例外になっても顔追従・手の追跡は止める。"""
        self._hand_grasped = False
        try:
            self._execute_on_robot_body()
        finally:
            # 握った場合は試行終了時 (失敗時も) に開く。
            if self._hand_grasped:
                self._set_hand_grasp(False)
            self._set_face_tracking(False)
            self._end_hand_follow()
            self._set_hand_tracking(False)

    def _hand_follow_target(self, waypoint, joint_names):
        """``waypoint`` に台車がいる前提での、計画時の掌の位置と法線
        (base_link 系)。計画の掌が無ければ None。"""
        with self._lock:
            planned_palm = self._current_palm
        if planned_palm is None:
            return None
        self._place_robot_at_waypoint(joint_names, waypoint)
        base = self.robot.base_link
        base_rot = base.worldrot().copy()
        base_pos = base.worldpos().copy()
        return (base_rot.T @ (np.asarray(planned_palm['position']) - base_pos),
                base_rot.T @ np.asarray(planned_palm['y_axis']))

    def _neck_indices(self, joint_names):
        """``joint_names`` の中の首 (ヨー・ピッチ・ロール) の添字。"""
        neck_names = {self.real_robot.neck_y_joint.name,
                      self.real_robot.neck_p_joint.name,
                      self.real_robot.neck_r_joint.name}
        return [i for i, name in enumerate(joint_names) if name in neck_names]

    def _hold_neck(self, waypoints, joint_names):
        """``waypoints`` の首の角度を今の実測値で固定したコピーを返す (手の追従で
        向いた首を押し込みの最初に戻さないため)。"""
        indices = self._neck_indices(joint_names)
        current = dict(zip(
            (joint.name for joint in self.real_robot.joint_list),
            np.asarray(self.ri.angle_vector(), dtype=np.float64)))
        held = []
        for wp in waypoints:
            vector = [float(v) for v in wp['joint_angle_vector']]
            for i in indices:
                vector[i] = float(current[joint_names[i]])
            held.append(dict(wp, joint_angle_vector=vector))
        return held

    def _execute_on_robot_body(self):
        """計画済みの waypoint 列を実機に送る。

        接近区間を 1 ゴールで送り、台車の位置ずれを補正してから押し込み区間、
        必要なら横並び移動を実行する。台車は costmap を使わず
        ``move_trajectory_sequence`` で直接送る。

        首は接近の直線 (lead-in) と横並び移動中は顔を追い (``_head_tracking_loop``)、
        螺旋〜押し込み直前は検出した差し出し手の掌を追う (見つかるまでは計画の掌)。
        押し込み区間は最後に向いていた首の角度のまま実行する。
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
        if not self._execution_verified(motion):
            print('[execute] 横並び移動が解けていないため実機を動かせ '
                  'ません。')
            return

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

        ref_scan = None
        exclude_centers = None
        if self.args.base_correction == 'scan':
            ref_scan = self._capture_scan()
            with self._lock:
                frozen = self._frozen_joint_positions
            if frozen:
                exclude_centers = np.array(
                    [p[:2] for p in frozen.values()], dtype=np.float64)

        # 接近の直線は首で顔を追い、螺旋に入ったら差し出し手の掌を追う (全身ゴール
        # 送信後に開始。螺旋の開始は time_list の累積時間)。押し込み直前の補正まで
        # 手の追跡も続ける。
        self._set_hand_tracking(self.args.press_in_refine)
        offered_hand = result['offered_hand']
        follow_target = None
        if self._gaze_robot is not None:
            follow_target = self._hand_follow_target(
                display_waypoints[min(n_prepend, reach_boundary - 1)],
                joint_names)
        follow_timers = []

        def on_approach_sent():
            self._set_face_tracking(True)
            if follow_target is None:
                return
            delay = float(sum(self._segment_time_list[:n_prepend + 1]))
            timer = threading.Timer(
                delay, self._begin_hand_follow,
                args=(offered_hand,) + tuple(follow_target))
            timer.daemon = True
            follow_timers.append(timer)
            timer.start()
            if self.args.in_motion_correction:
                # 到着少し前から、台車が止まり掌が検出できたら最終位置を掌に合わせてずらす。
                total = float(sum(self._segment_time_list))
                delay = max(0.0, total - self.args.in_motion_lead)
                deadline = time.time() + max(0.0, total - IN_MOTION_LEAD_MIN)
                timer = threading.Timer(
                    delay, self._correct_base_tail_when_settled,
                    args=(tail_stop, deadline, display_waypoints[0],
                          display_waypoints[reach_boundary - 1], joint_names,
                          result, ref_scan, exclude_centers))
                timer.daemon = True
                follow_timers.append(timer)
                timer.start()

        self._tail_correction = None
        tail_stop = threading.Event()
        start_odom_coords, final_traj_point = self._execute_waypoint_segment(
            display_waypoints[:reach_boundary], joint_names,
            on_sent=on_approach_sent)
        tail_stop.set()
        for timer in follow_timers:
            timer.cancel()
            timer.join()
        self._set_face_tracking(False)
        tail = self._tail_correction or {}
        tail_applied = bool(tail.get('applied'))
        # 止まってからの補正で合わせる掌: 終盤に検出した掌があればそれ、無ければ計画の掌。
        if tail.get('palm_world') is not None:
            palm_anchor = (np.asarray(tail['palm_world']),
                           np.asarray(tail['palm_in_base']))
        else:
            palm_anchor = self._planned_palm_anchor(
                display_waypoints[0], display_waypoints[reach_boundary - 1])
        hand_followed = follow_target is not None
        if hand_followed and not self._head_hand_following.is_set():
            self._begin_hand_follow(offered_hand, *follow_target)
        # 手を追わないときは hover 到着で首を計画の角度へ戻す (台車補正と並行、
        # 再検出前に待つ)。
        head_to_hover = self._gaze_robot is not None and not hand_followed
        if head_to_hover:
            self._send_head_to_waypoint(
                display_waypoints[reach_boundary - 1], joint_names)

        if self.args.base_correction != 'none':
            # 終盤のずらし後に残ったスリップを直す (目標は掌を計画どおり見る位置
            # なので終盤のずらしは戻さない)。
            self._correct_base_residual(
                start_odom_coords, final_traj_point, ref_scan=ref_scan,
                exclude_centers=exclude_centers, palm_anchor=palm_anchor)
        elif tail_applied:
            settle_time, drift = self._wait_odom_settle()
            log_debug('[execute][correct] 終盤で台車の最終位置をずらしました '
                      '(odom が止まるまで {:.2f}s、その間の移動 '
                      '{:.1f}mm/{:.2f}deg)'.format(
                          settle_time, drift[0] * 1e3,
                          math.degrees(drift[1])))

        robot_arm = result['robot_arm']
        self._wait_joint_settle(
            'hover', display_waypoints[reach_boundary - 1], joint_names,
            robot_arm)
        self._end_hand_follow()
        if head_to_hover or hand_followed:
            self.ri.wait_interpolation(controller_type='head_controller')

        if reach_boundary < len(display_waypoints):
            press_waypoints = display_waypoints[reach_boundary - 1:]
            if self.args.press_in_refine:
                refined = self._refine_press_in(
                    press_waypoints[0], joint_names, result)
                if refined is not None:
                    press_waypoints = [press_waypoints[0]] + refined
            if hand_followed:
                press_waypoints = self._hold_neck(
                    press_waypoints, joint_names)
            self._set_hand_tracking(False)
            self._execute_waypoint_segment(press_waypoints, joint_names)
            self._wait_joint_settle(
                '押し込み終了', press_waypoints[-1], joint_names,
                robot_arm)

        if reach_boundary < len(display_waypoints):
            self._set_hand_grasp(True)
        self._say(self.args.speech_done_text)
        press_end_time = time.time()
        do_transition = (transition_waypoints_
                         and reach_boundary < len(display_waypoints))

        if (self.args.grasp_capture_duration > 0
                and reach_boundary < len(display_waypoints)):
            duration = self.args.grasp_capture_duration
            if do_transition:
                duration = min(duration, self.args.side_by_side_delay)
            self._capture_grasp_images(
                press_waypoints[-1], joint_names, result, duration)

        if do_transition:
            wait = press_end_time + self.args.side_by_side_delay - time.time()
            if wait > 0:
                rospy.sleep(wait)
            transition_waypoints_ = self._refined_transition_waypoints(
                motion, transition_waypoints_)
            self._execute_transition(
                press_waypoints[-1], display_waypoints[-1],
                transition_waypoints_, joint_names, robot_arm,
                hold_neck=hand_followed)

        print('[execute] 実行を終了しました。')

    def _execute_transition(self, press_wp, planned_press_wp, waypoints,
                            joint_names, robot_arm, hold_neck=False):
        """押し込み後の横並び移動を実行する。

        押し込みを解き直した関節角の差 (``press_wp`` - ``planned_press_wp``) を
        各 waypoint に足し、各区間は ``--side-by-side-segment-time`` 秒以上かける。
        """
        delta = (np.asarray(press_wp['joint_angle_vector'])
                 - np.asarray(planned_press_wp['joint_angle_vector']))
        if hold_neck:
            # 押し込みの首の差 (手の追従による) は横並びの首に足さない。
            delta[self._neck_indices(joint_names)] = 0.0
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
            min_segment_time=self.args.side_by_side_segment_time,
            on_sent=lambda: self._set_face_tracking(True))
        self._wait_joint_settle(
            '横並び移動終了', segment[-1], joint_names, robot_arm)
        self._set_face_tracking(False)

    def _collect_hand_frames(self, n_frames, timeout):
        """以降のフレームで手だけ検出し、``n_frames`` 枚 (None なら ``timeout`` 秒
        の間の全て) のフレーム情報のリストを返す。"""
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
        """各フレームで掌が ``expected_position`` (今の base_link 系) に最も近い手を
        選び、ランドマークのフレーム間中央値を ``{side}Hand*`` の名前で返す。
        Hands の左右判定は誤りうるので位置で選び、score の低い手・法線が
        ``expected_normal`` から大きく傾いた手は除く。

        Returns: ``(joint_positions or None, distances [m], candidates,
        camera_position)``。``candidates`` はフレームごとの全候補の内訳 (ログ・
        保存用)、``camera_position`` は選んだフレームのカメラ位置の中央値。
        """
        chosen = []
        cameras = []
        distances = []
        candidates = []
        for frame in hand_frames:
            best = None
            frame_candidates = []
            camera_position = np.asarray(frame['camera_to_base'])[:3, 3]
            for hand in frame['hands']:
                joints = epp.hand_joint_positions(hand, side)
                palm = self.palm_estimator.estimate_palm(
                    joints, side, camera_position, expected_normal)
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
                    edge_on=palm is not None and palm['edge_on'],
                    rejected=rejected))
                if (dist is not None and rejected is None
                        and (best is None or dist < best[0])):
                    best = (dist, joints)
            candidates.append(frame_candidates)
            if best is not None and best[0] <= PRESS_IN_REFINE_MAX_HAND_DISTANCE:
                distances.append(best[0])
                chosen.append(best[1])
                cameras.append(camera_position)
        if not chosen:
            return None, distances, candidates, None
        names = [name for name in chosen[0]
                 if sum(name in joints for joints in chosen) * 2
                 >= len(chosen)]
        return ({name: np.median([joints[name] for joints in chosen
                                  if name in joints], axis=0).tolist()
                 for name in names}, distances, candidates,
                np.median(cameras, axis=0))

    def _save_refine_failure(self, hand_frames, expected_position, candidates,
                             record):
        """押し込み直前の手の再検出に失敗したときの画像と ``record.json`` を保存する。"""
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
        """押し込み終了後 ``duration`` 秒のカメラフレームを、握ったかの判定の資料
        として別スレッドで保存する。

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
                                           in hand['positions'].items()},
                                model_positions={
                                    k: [float(x) for x in p] for k, p
                                    in hand.get('model_positions', {}).items()})
                           for hand in frame['hands']]))
            json_io.save_json(os.path.join(out_dir, 'record.json'),
                              dict(record, frames=frames))
            print('[execute][grasp] 押し込み後の画像 {} 枚を {} に保存しました。'
                  .format(len(hand_frames), out_dir))
        except Exception as exc:  # noqa: BLE001  (保存失敗で実機動作を止めない)
            print('[execute][grasp][WARN] 画像の保存に失敗しました ({})。'
                  .format(exc))

    def _place_robot_at_waypoint(self, joint_names, waypoint):
        """``self.robot`` を ``waypoint`` の姿勢にする (IK で動いた本体を先に原点へ戻す)。"""
        self.robot.newcoords(Coordinates())
        apply_waypoint_pose(self.robot, joint_names, [waypoint], 0)

    def _refine_press_in(self, hover_wp, joint_names, result):
        """hover 到達後に手を検出し直して押し込み姿勢を解き直し、押し込み区間の
        waypoint (``hover_wp`` の次以降) を返す (失敗なら None)。

        検出した掌を「台車が hover 目標にいる」前提で計画時の座標系に移すので、
        台車のスリップと手の動きの両方を腕で吸収する。
        """
        t0 = time.time()
        with self._lock:
            planned_palm = self._current_palm
        offered_hand = result['offered_hand']
        planned_post = result.get('post_process')
        if planned_post is None or planned_palm is None:
            return None
        self._place_robot_at_waypoint(joint_names, hover_wp)
        base = self.robot.base_link
        base_rot = base.worldrot().copy()
        base_pos = base.worldpos().copy()
        expected_position = base_rot.T @ (
            np.asarray(planned_palm['position']) - base_pos)
        expected_normal = base_rot.T @ np.asarray(planned_palm['y_axis'])
        hand_frames = self._collect_hand_frames(
            PRESS_IN_REFINE_FRAMES, PRESS_IN_REFINE_TIMEOUT)
        joint_positions, distances, candidates, camera_position = \
            self._select_offered_hand(hand_frames, offered_hand,
                                      expected_position, expected_normal)
        wait_time = time.time() - t0
        palm = None
        if joint_positions is not None:
            palm = self.palm_estimator.estimate_palm(
                joint_positions, offered_hand, camera_position,
                expected_normal)
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
            planned_post,
            planned_palm=(planned_palm if 'x_axis' in planned_palm else None))
        ik_time = time.time() - t1
        palm_shift = np.asarray(observed['position']) - np.asarray(
            planned_palm['position'])
        self._log_debug(dict(
            event='press_in_refine', reason=info['reason'],
            legs=info.get('legs'), blend=info.get('blend'),
            wait_time=wait_time, ik_time=ik_time,
            n_frames=len(hand_frames), hand_distances=distances,
            expected_position=[float(v) for v in expected_position],
            candidates=candidates, edge_on=palm['edge_on'],
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
        blend = info.get('blend')
        print('[execute][refine] 検出し直した手{}に合わせて押し込み姿勢を '
              '補正しました ({}、脚 {}{}、目標の変化 {:.1f}mm/{:.1f}deg、'
              '待機 {:.2f}s + IK {:.3f}s)。'.format(
                  ' (真横から見た掌)' if palm['edge_on'] else '',
                  info['reason'], info.get('legs'),
                  '、掌の向きを計画の向きへ {:.0f}% 寄せた'.format(blend * 100)
                  if blend else '',
                  info['position_change'] * 1e3,
                  math.degrees(info['rotation_change']), wait_time, ik_time))
        return build_press_in_waypoints(
            hover_wp, joint_names, info['post_process'])

    @staticmethod
    def _planar_matrix(x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])

    def _planned_base_in_odom(self, plan):
        """送った台車のゴール (``_segment_base_plan``) の各点を odom 系で返す:
        ``(時刻 [s], 位置 (n, 2), 速度 (n, 2), yaw (n,), yaw の速度 (n,))``
        (先頭が送信時の位置)。"""
        points = np.vstack([np.zeros(3), plan['points']])
        times = np.asarray(plan['times'], dtype=np.float64)
        velocities = self._point_velocities(np.diff(points, axis=0), times)
        x0, y0, yaw0 = self._odom_pose(plan['start_odom'])
        rot = self._planar_matrix(0.0, 0.0, yaw0)[:2, :2]
        return (np.concatenate([[0.0], np.cumsum(times)]),
                np.array([x0, y0]) + points[:, :2] @ rot.T,
                velocities[:, :2] @ rot.T, yaw0 + points[:, 2],
                velocities[:, 2])

    def _hover_clearance_margin(self, waypoint, joint_names, offered_hand):
        """``waypoint`` (hover) の、差し出し時の骨格に対する人体との距離の余裕 [m]
        (0 以上なら通過、骨格が無ければ inf)。"""
        with self._lock:
            frozen = self._frozen_joint_positions
        if not frozen:
            return float('inf')
        collision_joints = spik.shift_torso_joints_from_surface(
            frozen, (0.0, 0.0), self.args.torso_surface_offset)
        return float(phm.verify_human_clearance(
            self.robot, joint_names, [waypoint],
            self.motion_verification_pairs,
            phm.build_obstacle_cache(collision_joints),
            hand=offered_hand)[0])

    def _send_tail_goal(self, points):
        """``in_motion_correction.build_tail_points`` の点列を台車の新しいゴールとして
        送る (動作中のゴールは取り消さず置き換える。時刻 0 は受信時点)。"""
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = list(
            self.ri.move_base_trajectory_joint_names)
        for t, position, velocity, yaw, yaw_rate in points:
            goal.trajectory.points.append(
                trajectory_msgs.msg.JointTrajectoryPoint(
                    positions=[float(position[0]), float(position[1]),
                               float(yaw)],
                    velocities=[float(velocity[0]), float(velocity[1]),
                                float(yaw_rate)],
                    time_from_start=rospy.Duration(t)))
        self.ri.move_base_trajectory_action.send_goal(goal)

    def _planned_palm_anchor(self, origin_wp, hover_wp):
        """計画の掌の、区間先頭 (``origin_wp`` の台車) を原点とする系の xy と、
        hover の台車から見た xy の組 (計画の掌が無ければ None)。"""
        with self._lock:
            planned_palm = self._current_palm
        if planned_palm is None:
            return None
        palm = np.asarray(planned_palm['position'], dtype=np.float64)[:2]
        origin = [origin_wp['base_position'][0], origin_wp['base_position'][1],
                  origin_wp['base_yaw']]
        hover = [hover_wp['base_position'][0], hover_wp['base_position'][1],
                 hover_wp['base_yaw']]
        return (in_motion_correction.inverse_transform_point(origin, palm),
                in_motion_correction.inverse_transform_point(hover, palm))

    def _settled_detections(self, now):
        """直近 IN_MOTION_WINDOW 秒の、台車停止後の掌検出 (odom 系の位置) のリスト。"""
        with self._lock:
            return [position for t, position, settled in self._hand_detections
                    if settled and now - t <= IN_MOTION_WINDOW]

    def _correct_base_tail_when_settled(self, stop, deadline, *args):
        """台車が止まってからの掌検出がたまるまで ``stop`` が立つか ``deadline``
        (time.time()) まで待ち、たまったら ``_correct_base_tail`` を行う
        (間に合わなければ行わない)。"""
        n = 0
        while not stop.is_set() and not rospy.is_shutdown():
            self._base_settled_time()
            n = len(self._settled_detections(time.time()))
            if n >= IN_MOTION_MIN_DETECTIONS:
                self._correct_base_tail(*args)
                return
            if time.time() >= deadline:
                break
            stop.wait(IN_MOTION_POLL_PERIOD)
        record = dict(event='in_motion_correction', applied=False,
                      reason='not_settled', n_detections=n)
        self._tail_correction = record
        self._log_debug(record)
        log_debug('[execute] 接近の終盤に台車が止まった状態で掌を検出できず、'
                  '台車の最終位置のずらしは行いませんでした (検出 {} 件)。'
                  .format(n))

    def _correct_base_tail(self, origin_wp, hover_wp, joint_names, result,
                           ref_scan=None, exclude_centers=None):
        """接近終盤に台車の最終位置を検出した掌に合わせてずらす。結果は
        ``self._tail_correction`` に残す (``applied`` が True ならずらした。
        ``palm_world`` があれば止まってからの補正もその掌に合わせる)。

        動いている間に残りの軌道を滑らかにずらして差し替える (止まってから戻すと
        後退・回転が入るため)。腕の軌道は変えない。``ref_scan`` があればスキャン
        照合でスリップも含めてずらす。ずらしは後退しない向き・上限以下・人体距離の
        検証に通る量に抑える。
        """
        record = dict(event='in_motion_correction', applied=False,
                      reason=None)
        try:
            self._plan_base_tail(record, origin_wp, hover_wp, joint_names,
                                 result, ref_scan, exclude_centers)
        except Exception as exc:  # noqa: BLE001  (失敗しても止まってからの補正に戻る)
            record['reason'] = 'exception: {!r}'.format(exc)
        self._tail_correction = record
        self._log_debug(record)
        if record['applied']:
            print('[execute] 接近の終盤で台車の最終位置を掌に合わせてずらし'
                  'ました ({:+.0f}mm, {:+.0f}mm、掌のずれ {:.0f}mm)。'.format(
                      record['shift'][0] * 1e3, record['shift'][1] * 1e3,
                      record['palm_error'] * 1e3))
        else:
            log_debug('[execute] 接近の終盤の台車の最終位置のずらしは行いま'
                      'せんでした ({})。'.format(record['reason']))

    def _plan_base_tail(self, record, origin_wp, hover_wp, joint_names,
                        result, ref_scan, exclude_centers):
        plan = self._segment_base_plan
        if plan is None or plan['sent_time'] is None:
            record['reason'] = 'no_plan'
            return
        recent = self._settled_detections(time.time())
        anchor = self._planned_palm_anchor(origin_wp, hover_wp)
        record['n_detections'] = len(recent)
        if anchor is None or len(recent) < IN_MOTION_MIN_DETECTIONS:
            record['reason'] = 'few_detections'
            return
        planned_palm_start, palm_in_base = anchor

        origin = self._odom_pose(plan['start_odom'])
        odom_delta = scan_matching.relative(
            origin, self._odom_pose(self.ri.odom))
        final_pose = np.asarray(plan['points'][-1], dtype=np.float64)
        odom_to_actual = np.zeros(3)
        record['scan'] = False
        if ref_scan is not None:
            scan_delta = self._match_scan(ref_scan, odom_delta,
                                          exclude_centers, 'tail', final_pose)
            if scan_delta is not None:
                odom_to_actual = scan_matching.compose(
                    scan_delta, scan_matching.inverse(odom_delta))
                record['scan'] = True
        record['odom_to_actual'] = [float(v) for v in odom_to_actual]
        # 掌 (odom 系) -> start 系 -> 実際の系 (検出の間はスリップ一定とみなす)。
        palm_odom = np.median(np.asarray(recent), axis=0)[:2]
        palm_world = in_motion_correction.transform_point(
            odom_to_actual, in_motion_correction.inverse_transform_point(
                origin, palm_odom))
        record['palm_error'] = float(np.linalg.norm(
            palm_world - planned_palm_start))
        # 実際の台車が動く量 (実際の系)。スリップは含めず、上限・縮めるのはこの量だけ。
        displacement, actual_yaw = in_motion_correction.anchored_displacement(
            palm_world, palm_in_base, final_pose, odom_to_actual)
        actual_final = in_motion_correction.actual_final_position(
            final_pose, odom_to_actual)
        slip_offset = actual_final - np.asarray(final_pose[:2])
        record['wanted'] = [float(v) for v in displacement]
        norm = float(np.linalg.norm(displacement))
        if norm > in_motion_correction.MAX_PALM_ERROR:
            record['reason'] = 'too_large'
            return
        max_shift = self.args.in_motion_max_shift
        if norm > max_shift:
            displacement = displacement * (max_shift / norm)
            record['clamped'] = True

        # 人体距離の検証に通る量まで縮める (実際の系は計画の系を区間先頭の向きだけ回したもの)。
        offered_hand = result['offered_hand']
        start_to_plan = self._planar_matrix(
            0.0, 0.0, origin_wp['base_yaw'])[:2, :2]
        margin = None
        for scale in IN_MOTION_CLEARANCE_SCALES:
            # 人体との距離は最終的な実際の位置 (スリップ + 動かす量) で見る。
            shift_plan = start_to_plan @ (slip_offset + displacement * scale)
            moved = dict(hover_wp, base_position=[
                hover_wp['base_position'][0] + float(shift_plan[0]),
                hover_wp['base_position'][1] + float(shift_plan[1]), 0.0])
            margin = self._hover_clearance_margin(
                moved, joint_names, offered_hand)
            if margin >= 0.0:
                displacement = displacement * scale
                break
        else:
            record.update(reason='clearance', clearance_margin=margin)
            return
        target = actual_final + displacement
        # 止まってからの補正も同じ位置に合わせる (向き actual_yaw で target になる掌の位置)。
        anchor_palm = target + self._planar_matrix(
            0.0, 0.0, actual_yaw)[:2, :2] @ palm_in_base
        record.update(
            displacement=[float(v) for v in displacement],
            clearance_margin=margin,
            palm_world=[float(v) for v in anchor_palm],
            palm_in_base=[float(v) for v in palm_in_base])
        wanted_shift = self._planar_matrix(0.0, 0.0, origin[2])[:2, :2] \
            @ in_motion_correction.odom_shift(target, final_pose,
                                              odom_to_actual)

        cum_times, positions, velocities, yaws, yaw_rates = \
            self._planned_base_in_odom(plan)
        elapsed = time.time() - plan['sent_time']
        record['elapsed'] = elapsed

        x, y, yaw = self._odom_pose(self.ri.odom)
        twist = self.ri.odom_msg.twist.twist
        c, s = math.cos(yaw), math.sin(yaw)
        current_velocity = np.array(
            [c * twist.linear.x - s * twist.linear.y,
             s * twist.linear.x + c * twist.linear.y])
        current_xy = np.array([x, y])
        final_xy = positions[-1]
        direction = final_xy - current_xy
        if np.linalg.norm(direction) < 0.02:
            direction = current_velocity
        # 後退が要る成分などで縮めた分は、止まってからの補正が埋める。
        shift, info = in_motion_correction.tail_shift(
            wanted_shift, final_xy, current_xy, direction,
            max_shift=max_shift)
        record.update(clamped=record.get('clamped', False) or info['clamped'],
                      projected=info['projected'])
        if shift is None:
            record['reason'] = info['reason']
            return

        def build(candidate):
            return in_motion_correction.build_tail_points(
                cum_times, positions, velocities, yaws, yaw_rates, elapsed,
                current_xy, current_velocity, yaw, twist.angular.z,
                candidate)

        points, scale = in_motion_correction.monotone_tail(
            build, shift, direction,
            max_speed=BASE_MAX_VEL * VEL_LIMIT_RATIO)
        if points is None:
            record['reason'] = 'late_or_reverse'
            return
        shift = shift * scale
        self._send_tail_goal(points)
        record.update(
            applied=True, shift=[float(v) for v in shift],
            shift_scale=float(scale), tail_time=float(points[-1][0]))

    def _wait_joint_settle(self, label, waypoint, joint_names, robot_arm,
                           tolerance=JOINT_SETTLE_HAND_TOLERANCE,
                           timeout=JOINT_SETTLE_TIMEOUT):
        """実機の腕が ``waypoint`` に追いつく (関節角由来の手先のずれ <=
        ``tolerance``) まで最大 ``timeout`` 秒待つ (時間切れでも止めない)。"""
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

    def _set_hand_grasp(self, grasp):
        """ハンドの握る (True) / 開く (False) を ``/aero_hand_grasp_service/grasp``
        (SetBool) で切り替える (失敗しても実機動作は止めない)。"""
        name = '/aero_hand_grasp_service/grasp'
        try:
            rospy.wait_for_service(name, timeout=2.0)
            response = rospy.ServiceProxy(name, SetBool)(bool(grasp))
            if grasp:
                self._hand_grasped = True
            print('[execute] ハンドを{} (success={}, {})'.format(
                '握ります' if grasp else '開きます',
                response.success, response.message))
        except Exception as e:  # noqa: BLE001
            print('[execute] ハンドの切り替えに失敗しました ({}): {}'.format(
                grasp, e))
            if grasp:
                self._hand_grasped = True

    def _say(self, text):
        """``text`` を発話する (非ブロッキング、発話できなければ何もしない)。"""
        if self.sound_client is None or not text:
            return
        try:
            self.sound_client.say(text, voice=self.args.speech_voice)
        except Exception:  # noqa: BLE001  (発話失敗で実機動作を止めない)
            pass

    def _execute_waypoint_segment(self, waypoints, joint_names,
                                  min_segment_time=MIN_SEGMENT_TIME,
                                  on_sent=None):
        """``waypoints`` (先頭が現在の姿勢) を 1 つの滑らかな軌道として台車・腕を
        並行に実行する。``on_sent`` はゴール送信直後 (完了待ちの前) に呼ぶ。

        Returns: ``(start_odom_coords, final_traj_point)`` (送信時の odom と
        最終 waypoint の先頭基準 ``[dx, dy, dyaw]``、台車移動なしなら None)。
        """
        arm_angle_vectors = []
        # 先頭 waypoint からの累積移動量 (move_trajectory_sequence は各要素を
        # 開始時の odom から独立に適用する)。
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
                first_base = (bx, by, byaw)
            x0, y0, yaw0 = first_base
            dx_world = bx - x0
            dy_world = by - y0
            dyaw = byaw - yaw0
            cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)
            dx = cos_yaw * dx_world + sin_yaw * dy_world
            dy = -sin_yaw * dx_world + cos_yaw * dy_world
            base_trajectory_points.append([dx, dy, dyaw])

        # それぞれ 1 ゴールで送る (非ブロッキング)。台車と腕に同じ time_list を渡して揃える。
        time_list = self._limited_time_list(
            arm_angle_vectors, base_trajectory_points,
            [min_segment_time] * len(waypoints))
        start_odom_coords = None
        self._segment_time_list = list(time_list)
        if arm_angle_vectors:
            self.ri.angle_vector_sequence(arm_angle_vectors, time_list)
        if base_trajectory_points:
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
            plan = dict(
                points=np.asarray(base_trajectory_points, dtype=np.float64),
                times=[float(t) for t in time_list],
                start_odom=start_odom_coords, sent_time=None)
            self._segment_base_plan = plan
            self._send_base_trajectory(
                base_trajectory_points, time_list, wait=False)
            plan['sent_time'] = time.time()
        if on_sent is not None:
            on_sent()

        if arm_angle_vectors:
            self._wait_body_interpolation()
            self._log_controller_states()
        if base_trajectory_points:
            self.ri.move_base_trajectory_action.wait_for_result()
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

    @staticmethod
    def _action_name(action):
        return getattr(getattr(action, 'action_client', None), 'ns',
                       None) or getattr(action, 'name', '?')

    def _wait_body_interpolation(self):
        """腕・脚・腰などの各コントローラのゴールが終わるまで待つ
        (``ri.wait_interpolation`` 相当)。首は顔・掌の追従が指令を送り直し続けて
        ゴールが終わらないので追従中は待たない。待ち時間をログに出す。"""
        skip_head = (self._head_face_tracking.is_set()
                     or self._head_hand_following.is_set())
        waited = []
        for action in self.ri.controller_table[self.ri.controller_type]:
            name = self._action_name(action)
            if skip_head and name.startswith('head_controller'):
                waited.append('{}=待たない'.format(name))
                continue
            t0 = time.time()
            action.wait_for_result(timeout=rospy.Duration(0))
            waited.append('{}={:.1f}s'.format(name, time.time() - t0))
        log_debug('[debug][segment] ゴール完了待ち: ' + ', '.join(waited))

    def _log_controller_states(self):
        """腕・脚・腰などの各コントローラのゴール終了状態をログに出す
        (ABORTED の関節を見つけるため)。"""
        parts = []
        for action in self.ri.controller_table[self.ri.controller_type]:
            state = action.get_state()
            name = self._action_name(action)
            parts.append('{}={}'.format(
                name, GOAL_STATUS_NAMES.get(state, state)))
        log_debug('[debug][segment] 腕・脚コントローラのゴール終了状態: '
                  + ', '.join(parts))

    def _send_base_trajectory(self, base_trajectory_points, time_list,
                              wait):
        """区間先頭基準の累積 ``[dx, dy, dyaw]`` 列を ``time_list`` で台車に送る。

        各点の速度は ``_point_velocities`` で付け直す (skrobot のままだと静止から
        区間の平均速度へ跳ぶため)。
        """
        goal = self.ri.move_trajectory_sequence(
            base_trajectory_points, time_list, stop=True, send_action=False)
        points = goal.goal.trajectory.points
        deltas = np.diff(np.vstack([
            np.zeros(3),
            np.asarray(base_trajectory_points, dtype=np.float64)]), axis=0)
        point_vel = self._point_velocities(
            deltas, np.asarray(time_list, dtype=np.float64))
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
        """区間ごとの所要時間を、律速する軸の指令の速度・加速度の最大値がちょうど
        上限 (× ``VEL_LIMIT_RATIO``、加速度は ``ACCEL_TIME``) になるよう決めて返す。
        ``time_list`` は区間ごとの下限。

        ``arm_angle_vectors``: 送る関節角列 (始点は実機の現在角)。
        ``base_trajectory_points``: 区間先頭からの累積 ``[dx, dy, dyaw]`` 列。
        None/空なら判定から外す。コントローラが 3 次エルミートで補間するので、
        その最大値で判定する。
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
            exclude_centers=None, palm_anchor=None,
            max_attempts=BASE_CORRECTION_MAX_ATTEMPTS,
            position_tolerance=BASE_CORRECTION_POSITION_TOLERANCE,
            angle_tolerance=BASE_CORRECTION_ANGLE_TOLERANCE):
        """押し込み前に台車の hover 目標からの位置ずれを検出し、収束するまで
        相対移動で補正する。

        引数は ``_execute_waypoint_segment`` の戻り値。実際の移動量は ``ref_scan``
        があればスキャン照合 (``exclude_centers`` 近傍の人の点は除く)、無ければ
        odom (スリップは検出できない) で求める。

        ``palm_anchor`` (区間先頭を原点とする系の掌の xy, hover の台車から見た
        掌の xy) があれば、位置の目標は掌を計画どおりに見る位置にする
        (終盤に掌に合わせてずらした位置も戻さない)。
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
            # 向きは許容を超えたときだけ直す (yaw の補正はキックで打ち消される)。
            yaw_error = scan_matching.wrap_angle(plan[2] - actual[2])
            target = plan.copy()
            if abs(yaw_error) <= angle_tolerance:
                target[2] = actual[2]
            if palm_anchor is not None:
                target[:2] = in_motion_correction.palm_anchored_position(
                    palm_anchor[0], palm_anchor[1], target[2])
            err_x, err_y, err_yaw = scan_matching.relative(actual, target)
            err_norm = math.hypot(err_x, err_y)
            log_debug('[execute][correct] attempt={} ({}基準{}) 目標までの残差 '
                  '{:+.3f}m {:+.3f}m {:+.1f}deg (計画の向きとの差 {:+.1f}deg)'
                  .format(attempt, source,
                          '、掌に合わせる' if palm_anchor is not None else '',
                          err_x, err_y, math.degrees(err_yaw),
                          math.degrees(yaw_error)))

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
                None, [[err_x, err_y, err_yaw]], [BASE_CORRECTION_MIN_TIME])
            before = self._odom_pose(self.ri.odom)
            self._send_base_trajectory(
                [[err_x, err_y, err_yaw]], [sec], wait=True)
            state = self.ri.move_base_trajectory_action.get_state()
            settle_time, drift = self._wait_odom_settle(
                BASE_CORRECTION_SETTLE_TIMEOUT)
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
            self.waypoint_slider.value = index + 1

    def _update_skeleton_view(self, joint_positions):
        """骨格の線と人体の干渉回避ジオメトリ (IK と同じ Cylinder) を作り直す
        (``joint_positions`` が None なら消す)。"""
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
        content += '\n\n' + self._collision_pairs_text
        self._status_text.content = content

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

            # ちらつき防止のため、変化があり間隔が空いたときだけ作り直す
            # (出現・消失・固定表示への切り替えは即座に)。
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
                pass
        self.pose_estimator.close()
        self.viewer.close()


def build_arg_parser():
    """ノードのオプション (保存した試行の解き直しツールも同じ既定値を使う)。"""
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
        help='押し込み前の台車位置補正の方法 (scan: /scan 照合、odom: '
            'スリップ検出不可、none: 補正しない)。')
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
        help='差し出し手と判定するスコアの閾値 (合成骨格は {:.2f})。'
            .format(epp.OFFER_SCORE_MIN))
    parser.add_argument(
        '--max-person-distance', type=float, default=3.0,
        help='人物からロボット手先までの距離 [m] の上限 (超えたら候補から外す、0 以下で無効)。')
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
    parser.add_argument(
        '--torso-surface-offset', type=float,
        default=spik.DEFAULT_TORSO_SURFACE_OFFSET,
        help='干渉判定用に体幹の関節をカメラから離れる向きへずらす距離 [m]。')
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
        help='試行ごとの骨格/掌/IK 結果/軌道の JSON の保存先 (既定: {}/<起動時刻>)。'
             '押し込み解き直しの失敗・把持画像もここ。'.format(LOG_ROOT))
    # --- 軌道計画 (plan_handshake_motion.py と同じオプション・既定値) ---
    parser.add_argument(
        '--approach-distance', type=float,
        default=phm.DEFAULT_APPROACH_DISTANCE,
        help='接近開始位置の円の半径への上乗せ分 [m]。')
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
        '--motion-collision-verify-tolerance', type=float,
        default=phm.DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE,
        help='軌道の事後検証で許容する最大貫通量 [m]。')
    parser.add_argument(
        '--background-optimizer-warmup', dest='background_optimizer_warmup',
        action='store_true',
        help='軌道最適化 (jaxls) の準備を別プロセスで行い、最適化が要る人の計画だけ'
             '完了を待つ。')
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
        '--in-motion-correction', dest='in_motion_correction',
        action='store_true',
        help='接近終盤に台車の最終位置を検出した掌に合わせてずらす '
             '(既定は無効)。')
    parser.add_argument(
        '--in-motion-lead', type=float, default=IN_MOTION_LEAD,
        help='接近終盤の台車のずらしを、到着の何秒前から行えるか。')
    parser.add_argument(
        '--in-motion-max-shift', type=float, default=IN_MOTION_MAX_SHIFT,
        help='接近の終盤の台車の最終位置のずらしの上限 [m]。')
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
        '--speech-transition-fail-text', type=str,
        default='ごめんなさい、横に並ぶ道が見つかりませんでした。'
                'もう一度手を出してください',
        help='横並び移動が解けず実機を動かせなかったときの発話。')
    parser.add_argument(
        '--side-by-side-segment-time', type=float, default=0.2,
        help='横並び移動の 1 区間にかける最短時間 [秒]。')
    parser.add_argument(
        '--side-by-side-delay', type=float, default=4.0,
        help='押し込み終了から横並び移動を始めるまでの時間 [秒]。')
    parser.add_argument(
        '--side-by-side-approach', choices=('background', 'sync', 'off'),
        default='background',
        help='横並び後に台車を人の横へ寄せる区間の計画: background は動かす前に'
             '横並びまでの解を確保し寄せは動作中に裏で解く、sync は全部動かす前に'
             '解く、off は寄せない。')
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
    parser.add_argument(
        '--no-head-tracking', dest='head_tracking', action='store_false',
        help='ARMED 中に首を人の方へ向けたり、見渡して人を探したりしない。')
    return parser


def main():
    parser = build_arg_parser()
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
