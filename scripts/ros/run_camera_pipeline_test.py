#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""合成骨格 (``generate_random_human_poses.py``) の代わりに、実カメラ +
``PeoplePoseEstimator`` (MediaPipe) で推定した骨格に対して掌推定・IK・
軌道計画までのパイプラインを試す、「ボタンを押すと1人分やる」形の
対話的な ROS ノード。

* 検出した骨格 (base_link 座標系) と Aero のロボットモデルを scikit-robot
  の viser ビューアに重ねて常時プレビュー表示する (ARMED でない限り
  IK は解かない)
* viser 画面の ``ARM`` ボタンを押すと ``ARMED`` 状態になり、
  ``offered_hand`` (差し出し手) が決まった瞬間の骨格でその 1 人分だけ
  IK を解く
* IK が解けたら続けて ``plan_handshake_motion.py`` と同じ要領で、
  ロボットの初期姿勢から握手姿勢へ至る干渉回避付きの軌道 (waypoint 列)
  を計画する
* ``--armed-timeout`` 秒たっても決まらなければ諦めて ``IDLE`` に戻る

IK 自体は指なしロボット (``self.robot``) で解く (自己干渉ペアの組み合わせ
を抑えるため)。画面の状態表示では指ありモデルで事後検証を別に行い、
指先まで含めて実際に貫通している組み合わせをテキストパネルに出す
(``colliding_link_pairs``/``collision_pairs_text`` 参照)。

viser 画面は ``view_handshake_motion.py`` と同様に、計画した軌道を
waypoint スライダー/Play ボタンで確認できる。``estimate_palm_poses.py``/
``solve_palm_ik.py``/``plan_handshake_motion.py`` の関数・クラスをそのまま
import して使う (骨格の入力形式 ``{limb_name: [x, y, z]}`` は合成骨格と
同じ)。``plan_handshake_motion.py`` は ``jaxls`` (``pip install
"git+https://github.com/brentyi/jaxls.git"``) が別途必要。

実カメラ特有の 2 つの問題への対策も入れてある。

* 深度が単発で背景側に飛ぶ (``aero_demo.skeleton_filters.OneEuroFilter``):
  関節位置に One Euro Filter (Casiez et al. 2012) をかけて時間方向に
  平滑化してから使う (``--joint-smoothing-mincutoff``/
  ``--joint-smoothing-beta`` で調整できる)。
* ARM を押しても差し出し手が見つからない: ``OfferedHandSelector`` は
  合成骨格向けにスコア閾値 (``--offer-score-min``, 既定は ``estimate_
  palm_poses.OFFER_SCORE_MIN``) が調整されているため、実カメラの姿勢では
  届きにくいことがある。ARMED 中は viser 画面に左右の
  スコア/判定不可の理由 (``no_palm``: 手のランドマークが取れていない、
  等) を表示するので、それを見ながら閾値を調整する。

Usage
-----
    python3 scripts/ros/run_camera_pipeline_test.py
    python3 scripts/ros/run_camera_pipeline_test.py --save-dir /tmp/camera_handshake_poses

IK・軌道計画 (state 'result') まで進むと、``--auto-execute`` を付けて
いれば計画済みの waypoint 列を 1 つの軌道としてまとめて自動的に実機へ
送る (``_execute_on_robot`` 参照)。``--auto-execute`` を付けていなければ
実機は一切動かさず、viser 画面の waypoint スライダー/Play チェックボックス
での確認 (viewer 上の再生) のみができる。ただし軌道の干渉検証 (経路計画の
区間・初期位置からの直進の両方) に通らなかった場合は、``--auto-execute``
を付けていても実機を動かさない (画面の「軌道」欄に NG と表示される)。
実機 (``AeroROSRobotInterface``) への接続自体は ``--auto-execute`` の
指定に関わらず起動時に常に試みるため、接続に成功していれば
(``--auto-execute`` を指定していなくても) ``ARM`` ボタンを押した瞬間に
実機の首を少し下げると同時に腕を初期姿勢 (体の横に下ろした姿勢) まで
戻し、人間が手を差し出しやすい姿勢にする (``_nod_head_for_arm`` 参照):

    python3 scripts/ros/run_camera_pipeline_test.py

``--auto-execute`` を付けると、IK・軌道計画が成功した時点で自動的に
台車・関節の両方を動かして実機を実行する。``--auto-arm`` (起動直後から
ARMED) と組み合わせると、ブラウザを一切触らずに「手を差し出す ->
握手しに行く」一連の動作を実行できる:

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

# plan_handshake_motion (jaxls 経由で JAX/XLA を使う) を import する際、
# JAX がデフォルトで GPU メモリを一括プリアロケートしようとして失敗し、
# "Failed to allocate device memory ... RESOURCE_EXHAUSTED" というエラー
# ログが標準エラーに出る (実際には失敗後にサイズを縮小して再試行するため
# 動作上は問題ないが、紛らわしいので抑制する)。import 前に一括確保をやめ
# 必要な分だけ確保する設定にしておくことで、このログ自体を出さなくする。
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

# jax の永続コンパイルキャッシュ (solve_palm_ik.py/plan_handshake_motion.py
# と同じ設定)。jax を import する前 (下の `from aero_demo import json_io`
# 経由で skrobot が無条件に `import jax` するより前) に設定する必要がある
# (詳細: docs/jax_compilation_cache.md)。
os.environ.setdefault(
    'JAX_COMPILATION_CACHE_DIR',
    os.path.expanduser('~/.cache/jax_compilation_cache'))
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES', '0')

from aero_demo import json_io  # noqa: E402
from aero_demo import palm_plane_view  # noqa: E402
from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo import viewer_nav  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo import skeleton_filters  # noqa: E402
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, lookup_camera_to_base, lookup_frame_position,
    ndarray_to_imgmsg, transform_to_matrix)

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
from handshake_viewer_common import remove_joint_angle_gui  # noqa: E402
from handshake_viewer_common import remove_obstacles_gui  # noqa: E402
from handshake_viewer_common import set_link_visible as common_set_link_visible  # noqa: E402,E501
from handshake_viewer_common import sync_robot_collision_overlay  # noqa: E402
from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from skrobot.coordinates.math import matrix2ypr  # noqa: E402
from skrobot.interfaces.ros import AeroROSRobotInterface  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.model import LinearJoint  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.planner.trajectory_optimization.solvers import (  # noqa: E402
    create_solver)
from skrobot.viewers import ViserViewer  # noqa: E402

# ロボットの初期位置 (台車がワールド原点にいる姿勢) を示す Axis の大きさ
# [m]。view_handshake_poses.py の TARGET_AXIS_LENGTH と同程度の、グリッド
# 上で目立つ大きさにしてある。
INITIAL_POSE_AXIS_LENGTH = 0.2
INITIAL_POSE_AXIS_RADIUS = 0.008

# waypoint 自動再生 (Play チェックボックス) の既定の速さ [waypoint/秒]
# (view_handshake_motion.DEFAULT_PLAYBACK_FPS と同じ)。
DEFAULT_PLAYBACK_FPS = 40.0

# ARM ボタンを押した瞬間 (実機 self.ri への接続に成功している状態のとき
# に、--auto-execute の指定有無に関わらず、_nod_head_for_arm 参照) に
# 実機の首を下げる目標角度 [deg]。``Aero.reset_pose`` の既定
# (neck_p_joint = 25 度、以後このパイプライン全体の「見ている」基準姿勢)
# からさらに下げ、まっすぐ人の顔の高さを見続けるより控えめにうつむかせる
# ことで、人間が手 (ロボットの手先の高さ) を差し出しやすい・近づきやすい
# 印象にする。実機の首の可動方向 (どちらが「下」か) は個体差の可能性が
# あるため、実機で確認して向きが逆なら符号を反転させること。
ARM_HEAD_NOD_PITCH_DEG = 25.0
# 上記の首下げ・腕を初期姿勢まで下ろす動作 (_nod_head_for_arm) にかける
# 時間 [秒]。あまり速いと会釈というより首を振っただけに見えるため、
# ゆっくりめにしてある。
ARM_HEAD_NOD_MOVE_TIME = 5

# 実機で軌道を再生するときの waypoint 間の所要時間の下限 [秒]。所要時間
# は固定の dt (motion['dt']、軌道最適化のコスト正規化用) ではなく、区間
# ごとに律速する軸が上限 × VEL_LIMIT_RATIO になるよう _limited_time_list
# で決める (2026-09-26 ユーザー要望、それまでは dt / EXECUTION_SPEED_SCALE
# が下限だった)。この下限はほとんど動かない区間の時間が 0 に潰れない
# ためだけのもので、腕のコントローラの制御周期 (15Hz) 1 周期分にしてある。
MIN_SEGMENT_TIME = 1.0 / 15.0

# 台車の速度上限。実機の base_controller (pr2_base_trajectory_action) が
# 使っている aero_base_link.yaml の base_link_x/y/pan の max_velocity と
# 必ず一致させること。この yaml は台車自身 (ロボット本体) の中にあり、
# jsk_aero_startup/config/aero_base_link.yaml (このリポジトリのコピー、
# 単なる参考用でロボット実機には自動反映されない) とは値が一致すると
# は限らない。2026-09-25 時点でロボット実機側は base_link_x/y=0.3,
# base_link_pan=1.0 (ssh 先で rosparam get /base_controller/
# joint_trajectory_action/base_link_{x,y,pan}/max_velocity で確認済み)。
# pr2_base_trajectory_action はこれを超える指令速度を odom 系の軸ごとに
# 頭打ちにする。当初 skrobot (ROSRobotMoveBaseInterface.go_pos_unsafe_
# wait) の値 (0.295/0.495、Aero の実際の設定とは無関係の値) をそのまま
# 流用していたところ、sec (残差解消にかける時間) を実際の 2.5 倍速く
# 見積もってしまい、時間切れで毎回残差の 3-5 割程度しか補正できなかった
# (2026-09-24 実機検証)。
BASE_MAX_VEL = 0.3  # [m/s] 実機の base_link_x/y の max_velocity
BASE_MAX_ANGVEL = 1.0  # [rad/s] 実機の base_link_pan の max_velocity

# 台車 (BASE_MAX_VEL/ANGVEL) と腕・首・腰・リフター (URDF の velocity) の
# 速度上限のうち、指令で使ってよい割合 (2026-09-26 ユーザー要望で両方
# 0.6 に統一、同日 0.8 -> 0.9 に変更)。上限を超えた指令を実機側で頭打ち
# にされて遅れ、後から補正するのではなく、初めから超えない指令を送る
# ため、_limited_time_list で実機側が補間に使う 3 次スプラインの瞬間
# 速度の最大値がちょうどこの割合になるよう区間の所要時間を決める (区間の
# 平均速度だけを見ると、速度 0 の境界を持つ区間では瞬間速度が平均の最大
# 1.5 倍になる)。実機の
# aero_ros_controller も PositionJointSaturationInterface で URDF の
# velocity を超える指令を制御周期ごとに頭打ちにする (実機の
# robot_description と skrobot の URDF の velocity は一致、2026-09-25
# 確認、2026-09-25 実機ログでは頭打ちによる押し込み区間先頭の約 6 度の
# 追従ずれを確認)。上限との差は、pr2_base_trajectory_action の位置誤差の
# P 補正が指令速度に上乗せされる分の余裕も兼ねる。
VEL_LIMIT_RATIO = 0.9
# 加速度の上限を、上記の速度の上限 (× VEL_LIMIT_RATIO) まで何秒かけて
# 加速するかで表したもの [秒]。台車・関節とも同じ値を使う (台車・関節の
# 加速度の仕様値が無いため、2026-09-26 ユーザー要望「ゆるやかに加減速
# してほしい」に対し仮に 0.5 秒とし、同日ユーザー指示で 0.3 秒に変更したが、
# 最後の手先位置がずれるようになった気がするとの報告 (odom・関節角の
# ログ上はずれ無し、台車のスリップの疑い、未確認) で 0.4 秒に変更)。
# 大きくするほどゆるやかになるが、動き出し・止まり際の区間が延びて全体の
# 所要時間も延びる。
ACCEL_TIME = 0.4
# 上記を満たす区間の所要時間を求める反復の上限回数 (これで収まらなければ
# 全区間を一律に延ばして必ず上限内にする)。
TIME_LIMIT_MAX_ITERATIONS = 100

# 押し込み (post_process) 動作の直前で行う台車の位置補正
# (_correct_base_residual 参照) のパラメータ。
BASE_CORRECTION_MAX_ATTEMPTS = 3
BASE_CORRECTION_POSITION_TOLERANCE = 0.025  # [m]
BASE_CORRECTION_ANGLE_TOLERANCE = math.radians(2.5)  # [rad]
# hover 目標・押し込み終了時に、腕が指令に追いつくのを待つときの
# パラメータ (_wait_joint_settle 参照)。実機の腕は制御周期 15Hz + サーボ
# への到達時間 (overlap 約 133ms) の分、指令から約 0.2 秒遅れて追従し、
# wait_interpolation はその遅れが解消する前に返る (2026-09-25 実機ログ:
# 上限速度で動いていた l_shoulder_r が約 7 度遅れ、hover 時点で手先が
# 関節由来だけで約 67mm (実際の手が約 5cm 高い)、押し込み終了直後 64mm
# -> 1 秒後 14.5mm)。関節角だけから FK で求めた手先のずれ (台車のずれは
# 含まない) が TOLERANCE 以下になるまで、最大 TIMEOUT 秒待つ (押し込み
# 終了時は接触の負荷で届かないこともあるため、時間切れでも止めない)。
JOINT_SETTLE_HAND_TOLERANCE = 0.01  # [m]
JOINT_SETTLE_TIMEOUT = 1.5  # [s]
JOINT_SETTLE_POLL_PERIOD = 0.05  # [s]

# ロボット自身/人体側の干渉回避ジオメトリを重ねて表示する色、経路の後処理
# 補間フレーム数は view_handshake_poses.py/view_handshake_motion.py と共通
# なので handshake_viewer_common.py に一本化してある
# (ROBOT_COLLISION_LINK_COLOR/HUMAN_COLLISION_OBSTACLE_COLOR/
# PRESS_IN_DISPLAY_WAYPOINTS)。

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
    return common_collision_pairs_text(
        colliding, label='表示中の waypoint の事後検証 (指先まで含む)')



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
        spik.lock_fixed_joints(self.robot)
        spik.apply_collision_model(self.robot)
        # self.robot の関節角ベクトル (motion['joint_names'] と同じ並び) での
        # 「初期姿勢」(両腕を体の横に下ろした姿勢, plan_handshake_motion.
        # arms_down_angles と同じ -- plan_handshake_motion.py の始点
        # (motion['waypoints'][0]) と見た目を揃える。``Aero.reset_pose`` の
        # ままだと肘を曲げた「構え」のような姿勢になる、同関数の docstring
        # 参照)。以降 self.robot は robot_position の計算 (seed_arm_pose) や
        # IK で上書きされ続けるため、ここで確保しておく (初期位置 -> 経路
        # 開始点の直進 (lead-in) の関節角の補間に使う、
        # plan_handshake_motion.build_lead_in_waypoints 参照)。
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
        # jaxls ソルバーも同じ理由 (plan_handshake_motion.main 参照) で
        # ノードの寿命で 1 個だけ作って使い回す (人物/試行ごとに作り直すと
        # JIT キャッシュが効かない)。
        self.solver = create_solver(
            'jaxls', max_iterations=args.max_iterations, verbose=False)
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

        # TF 未解決時のフォールバック値は起動時に 1 回だけ計算しキャッシュ
        # する (_resolve_robot_position 参照、毎フレーム呼ぶのは実機 TF の
        # 解決のみ)。
        self._robot_hand_position_fallback = \
            self._compute_robot_hand_position_fallback()
        self.robot_position = self._resolve_robot_position()
        print('[robot-hand-position] {} (base_link)'.format(
            self.robot_position.tolist()))

        # 掌推定・差し出し手判定器は毎フレーム作り直さず使い回す (以前は
        # ARMED の全フレームで新規に作っていたが、無駄な上に判定の内訳
        # (スコア/veto 理由) を毎フレーム覗けなかった)。robot_position は
        # ARMED 中フレームごとに _resolve_robot_position で TF から引き
        # 直して差し込み直す (_on_frame 参照、record_palm_offer_clips.py
        # と同じ)。
        max_distance = (None if args.max_person_distance <= 0
                        else args.max_person_distance)
        self.offered_hand_selector = epp.OfferedHandSelector(
            robot_position=self.robot_position,
            score_min=args.offer_score_min,
            max_distance=max_distance)
        self.palm_estimator = epp.PalmPoseEstimator(self.offered_hand_selector)

        # 深度ノイズによる関節位置の単発の飛び (「デプスが後ろの方に一瞬
        # 飛ぶ」) を抑える時間方向の平滑化 (aero_demo.skeleton_filters.
        # OneEuroFilter 参照)。
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
        self._handshake_total_time = None  # 掌推定開始 ~ 軌道計画完了までの合計時間 [秒] or None (_try_handshake 参照)
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

        # 実機接続 (--auto-execute の指定に関わらず常に AeroROSRobotInterface
        # への接続を試みる)。ARM ボタン押下時の首下げ (_nod_head_for_arm) は
        # 台車・腕を実際に動かすフラグとは独立に、self.ri さえ使えれば行い
        # たいため。台車・腕を実際に動かす実行は引き続き --auto-execute で
        # 制御する (_execute_on_robot 参照)。skrobot の
        # ROSRobotInterfaceBase はアクションサーバ待ちに controller_timeout
        # (既定 3 秒) の上限があり無限ブロックはしないため、実機/実機用
        # ROS ノードが立っていない環境でこのスクリプトを viewer 確認だけに
        # 使う用途を大きくは妨げない。接続失敗時にこの viewer 自体が
        # 使えなくなることは避けたいので、例外は握りつぶして self.ri を
        # None のままにする。台車・関節を別々の robot_model インスタンス
        # (self.robot/self.display_robot) と混ぜて操作すると angle_vector
        # 送信中に表示スレッドが同じインスタンスを書き換えてしまう恐れが
        # あるため、実機操作専用の robot_model を別に持つ (_execute_on_robot
        # 参照)。
        self.real_robot = None
        self.ri = None
        if args.no_robot_interface:
            # rosbag での実機なし動作確認用 (--no-robot-interface)。接続を
            # 試みること自体をやめる (self.ri は None のままになり、ARM 時の
            # 首下げ/--auto-execute による実機操作は無効のままになる)。
            print('[execute] --no-robot-interface が指定されたため、実機 '
                  '(AeroROSRobotInterface) への接続を試みません。')
        else:
            try:
                self.real_robot = load_aero(use_hand=True)
                print('[execute] 実機 (AeroROSRobotInterface) に接続しています...')
                # skrobot 側の既定値 (odom_topic='/base_odometry/odom') は本機
                # では配信されておらず、move_trajectory_sequence が odom 待ちで
                # 無限に固まる。実機の odom は /odom (/aero_ros_controller) な
                # のでそちらを明示する。
                self.ri = AeroROSRobotInterface(self.real_robot, odom_topic='/odom')
                print('[execute] 実機への接続が完了しました (--auto-execute={})。'
                      .format(args.auto_execute))
            except Exception as exc:  # noqa: BLE001  (実機/ROS 環境が無くても viewer 単体としては動作を継続したい)
                self.real_robot = None
                self.ri = None
                print('[execute] 実機 (AeroROSRobotInterface) への接続に失敗した '
                      'ため、ARM 時の首下げ/--auto-execute による実機操作は無効の '
                      'ままになります ({})。'.format(exc))

        # --auto-execute で実機を動かすときの発話 (動き出す直前と掌を
        # 差し出し終えた直後、_execute_on_robot 参照)。sound_play が無い
        # 環境でも viewer 単体としては動作を継続したいので、失敗時は
        # self.sound_client を None のままにして発話だけ諦める。
        self.sound_client = None
        if args.auto_execute and self.ri is not None:
            try:
                from sound_play.libsoundplay import SoundClient
                self.sound_client = SoundClient(
                    sound_action='robotsound_jp', sound_topic='robotsound_jp')
            except Exception as exc:  # noqa: BLE001
                print('[speech] SoundClient の初期化に失敗したため発話し '
                      'ません ({})。'.format(exc))

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

        if args.auto_arm:
            # ARM ボタンクリックの代わりに起動直後から ARMED にする
            # (_on_arm ボタンハンドラと全く同じ処理、--bag での無人テスト用)。
            self.state = 'armed'
            self.armed_deadline = time.time() + args.armed_timeout
            self._latest_offer_selection = None
            with self._lock:
                self._handshake_total_time = None
            print('[auto-arm] 起動直後に ARMED 状態にしました '
                  '(--auto-arm)。{:.0f} 秒以内に手を差し出してください。'
                  .format(args.armed_timeout))

    _WARMUP_PALM = dict(
        position=[0.5, 0.0, 1.0],
        x_axis=[1.0, 0.0, 0.0],
        y_axis=[0.0, 1.0, 0.0],
    )

    # ウォームアップ中だけ numpy のグローバル乱数を固定するシード
    # (``_warmup_ik`` 参照。値そのものに意味は無く、毎回同じであれば何でも
    # よい)。ウォームアップ後は元の乱数状態に戻すので、実運用の IK の
    # ランダム初期値には影響しない。
    _WARMUP_SEED = 0

    def _warmup_ik(self):
        """左右それぞれの腕で ``solve_person_ik`` と ``plan_person_motion``
        (jaxls 軌道最適化) をダミーの目標に対して 1 回ずつ解いておき、
        JAX の関数トレース (JIT の初回コンパイルより手前の、Python
        レベルで計算グラフを組み立てる処理。永続コンパイルキャッシュでは
        カバーされない) をノード起動時に前倒しで済ませる。

        ``_solve_handshake`` の ``solve_person_ik`` 呼び出しと引数を
        完全に一致させる必要がある -- 1 つでも違うと JAX には「別の
        関数」に見えて別途トレースされ直し、ウォームアップの意味が
        なくなる。同様の理由で、乱数シード (``_WARMUP_SEED``) を固定し、
        FK 由来の定数を量子化する (``JaxlsSolver._quantize_fk_constants``)
        必要がある -- ウォームアップと実運用で jaxls のトレース結果に
        焼き込まれる定数が 1 ビットでも変われば、コンパイル前 HLO の
        フィンガープリントが変わってキャッシュミスする。

        軌道最適化側は ``self.solver`` (``JaxlsSolver``、ノード寿命で
        使い回す) が l/r 腕それぞれ独立にキャッシュを持つため、両方を
        1 回ずつ ``force_optimize=True`` で通しておけば、以後 ARMED の
        たびに差し出し手の左右が入れ替わっても両方ともキャッシュヒット
        する。
        """
        # 乱数状態の退避 (finally で必ず復元する。上記 docstring 参照)。
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
        collision_obstacles = (
            [] if (args.no_human_collision or self.collision_pairs is None)
            else spik.human_body_obstacles({}))
        warmup_human_xy = np.array([args.human_front_distance, 0.0])
        motion_args = copy.copy(args)
        motion_args.collision_verify_tolerance = \
            args.motion_collision_verify_tolerance
        motion_args.force_optimize = True
        target_pos = spik.palm_target_position(self._WARMUP_PALM)
        # solve_person_ik/palm_to_target_rots に渡す差し出し手は、実際の
        # 自動割り当て (--robot-arm auto, spik.DEFAULT_ROBOT_ARM) で各腕が
        # 担当する側に合わせる (l 腕 <-> 人間の右手, r 腕 <-> 人間の左手)。
        # 向きの候補順序 (turn_candidates_deg) は差し出し手ごとに異なる
        # ため、実際の呼び出しと形を揃えておく。
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
            print('[warmup] {}腕: IK {:.1f} 秒'.format(
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
                motion_args, self.verification_pairs, self.solver)
            print('[warmup] {}腕: 軌道最適化 {:.1f} 秒'.format(
                label, time.time() - t0))

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
            with self._lock:
                self._handshake_total_time = None
            if self.ri is not None:
                # 首下げ・腕を初期姿勢まで下ろす動作 (_nod_head_for_arm)。
                # self.ri は --auto-execute の指定に関わらず接続を試みて
                # いるため (__init__ 参照)、接続さえ成功していれば
                # --auto-execute を指定していない場合でも実行する。
                # 実機通信 (joint_states 待ち/action 送信) をこの GUI
                # コールバックのスレッドで直接行うとブロックするため、
                # _execute_on_robot と同様に別スレッドに逃がす。
                threading.Thread(
                    target=self._nod_head_for_arm, daemon=True).start()
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
                self._handshake_total_time = None
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
            # redraw() は _apply_current_waypoint 内で self._viewer_lock を
            # 保持したまま (ロボットの姿勢更新と合わせて) 行うので、ここで
            # 別途呼ぶ必要はない (呼ぶと spin() 側の redraw() が姿勢更新の
            # 途中に割り込める隙が生まれてしまう、_apply_current_waypoint
            # のコメント参照)。
            self._apply_current_waypoint()

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
        # ロボットを add し終えたので、自動で付いてくる関節スライダーを
        # 消す (触ると display_robot と overlay の一方だけが動いて姿勢が
        # 食い違ったまま残るため、remove_joint_angle_gui 参照)。
        remove_joint_angle_gui(self.viewer)
        # 同様に、任意の障害物を画面から手動で追加・編集する GUI (Obstacles
        # フォルダ) も、人体の障害物は骨格から自動生成するこのビューアでは
        # 使わないので消す (remove_obstacles_gui 参照)。
        remove_obstacles_gui(self.viewer)
        self.viewer.show(open_browser=not args.no_open_browser)
        if not args.no_wait_for_client:
            viewer_nav.wait_for_client(self.viewer, args.client_wait_timeout)

    def _set_link_visible(self, link, visible):
        common_set_link_visible(self.viewer, link, visible)

    def _resolve_robot_position(self):
        """差し出し手判定の基準にするロボット手先の base_link 座標を返す.

        ``--robot-hand-position`` が明示されていればそれを固定で使う。
        そうでなければ実機の TF (``--robot-hand-frame`` -> ``--base-
        frame``、既定 ``r_eef_grasp_link`` -> ``base_link``) を毎回引き
        (``record_palm_offer_clips.py`` と共通の ``ros_camera_utils.
        lookup_frame_position``)、まだ引けなければ (ロボット未接続・
        /aero_state_publisher 未起動など) 右腕の「種の姿勢」
        (``solve_person_ik`` が IK の初期値に使うのと同じ姿勢、台車は
        ワールド原点) の手先位置 (``self._robot_hand_position_fallback``、
        起動時に 1 回だけ計算・キャッシュ済み) にフォールバックする。
        """
        if self.args.robot_hand_position is not None:
            return np.asarray(self.args.robot_hand_position, dtype=np.float64)
        return lookup_frame_position(
            self.tf_buffer, self.args.base_frame, self.args.robot_hand_frame,
            self._robot_hand_position_fallback,
            warn_label='[run-camera-pipeline-test] ')

    def _compute_robot_hand_position_fallback(self):
        """``_resolve_robot_position`` が TF 未解決時に使うフォールバック値
        (右腕の「種の姿勢」の手先位置、台車はワールド原点) を計算する.

        ``spik.seed_arm_pose`` は ``self.robot`` の関節角・台車位置姿勢を
        書き換えるが、``solve_person_ik`` (``_solve_handshake`` 経由) は
        呼ばれるたびに内部で ``seed_arm_pose`` を呼び直して姿勢を作り直す
        ため、ここで 1 回呼んでおいても実際の IK 計算には影響しない。
        """
        spik.seed_arm_pose(self.robot, 'r')
        return np.asarray(self.robot.rarm_end_coords.worldpos(),
                          dtype=np.float64)

    def _lookup_camera_to_base(self, header):
        """``header`` (画像の frame_id/stamp) から base_link への TF を引く
        (``aero_demo.ros_camera_utils.lookup_camera_to_base`` 参照)."""
        return lookup_camera_to_base(
            self.tf_buffer, self.args.base_frame, header)

    # ------------------------------------------------------------------
    # camera callback
    # ------------------------------------------------------------------
    def _on_frame(self, color_msg, depth_msg, info_msg):
        if self._busy:
            return
        # TF が引けなくてもプレビューは止めない (変換できなければカメラ
        # 座標系のまま推定を続ける)。ARMED での掌推定・IK だけは base_link
        # 座標系が要るので、変換できたフレームでのみ行う。
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
            # robot_position (差し出し手判定の基準にするロボット手先位置)
            # は ARMED 中フレームごとに実機 TF から引き直す
            # (record_palm_offer_clips.py の _on_frame と同じ、
            # _resolve_robot_position 参照)。
            self.offered_hand_selector.robot_position = \
                self._resolve_robot_position()
            self._try_handshake(armed_joint_positions)

        if (self.state == 'armed' and self.armed_deadline is not None
               and time.time() > self.armed_deadline):
            self.state = 'idle'
            self.armed_deadline = None
            print('[ARMED] タイムアウトしました。差し出し手が決まりません '
                  'でした。')

    def _try_handshake(self, joint_positions):
        handshake_t0 = time.time()
        palms = self.palm_estimator.estimate(joint_positions)
        # ARMED なのに offered_hand が決まらないとき、viser 画面にスコア/
        # veto 理由の内訳を出す。「手のランドマークがそもそも取れていない
        # (veto=no_palm)」のか「取れているがスコアが --offer-score-min に
        # 届いていない」のかを見分けられるようにするため (PalmPoseEstimator.
        # estimate は offered_hand しか返さないので、同じ入力で select() を
        # 呼び直す)。
        selection = self.offered_hand_selector.select(joint_positions, palms)
        with self._lock:
            self._latest_offer_selection = selection

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
            with self._lock:
                self._handshake_total_time = time.time() - handshake_t0

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

        # solve_palm_ik.py の main() と同じく、差し出している手の側・人間の
        # 正面方向に合わせてこの人物専用の base_limits (台車の可動域) を
        # 作る (docs/jax_compilation_cache.md 9節参照)。以前は
        # run_camera_pipeline_test.py だけこの制約を行っておらず
        # solve_palm_ik.py と挙動が食い違っていたが、台車の可動域を人物
        # ごとに変えても jax の再コンパイルが起きないことが確認できた
        # (scikit-robot フォークの案B、同節) ため、ここでも揃える。
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

        target_pos = spik.palm_target_position(translated_palm)
        rots = spik.palm_to_target_rots(translated_palm, offered_hand, robot_arm)
        picked, collision_ik_time, candidate_selection_time = \
            spik.solve_person_ik(
                self.robot, translated_palm, offered_hand, robot_arm,
                collision_obstacles,
                attempts_per_pose=args.attempts_per_pose,
                base_limits=person_base_limits,
                self_collision=(not args.no_self_collision
                                and self.collision_pairs is not None),
                collision_pairs=self.collision_pairs,
                joint_positions=translated_joints,
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

        # IK が解けたら続けて軌道計画を行う (plan_handshake_motion.py の
        # main と同じ、target かつ solved の人物だけが対象)。IK は
        # translated_joints/translated_palm を使う仮想座標系で解いている
        # ため、軌道計画もこの座標系のまま (result を untranslate する前)
        # に行う -- plan_person_motion 自身が呼ぶ human_body_cylinder_
        # obstacles/orbit_base_start が、この座標系の joint_positions/
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
            # 軌道の始点はまず最終台車位置と実機の現在地 (base_link 原点、
            # 向き +x) を結ぶ線分上に取る (plan_person_motion 参照)。IK と
            # 同じ仮想座標系 (実座標 + offset) に直した値を渡す。
            initial_base_pose = np.array([offset[0], offset[1], 0.0])
            motion = phm.plan_person_motion(
                self.robot, robot_arm, result, translated_joints, human_xy,
                motion_args, self.verification_pairs, self.solver,
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
            # ため、その手前にロボットの初期位置 (台車=ワールド原点, 関節=
            # 腕を下ろした初期姿勢) からの直進 (lead-in) を継ぎ足す。台車の
            # 動きは plan_person_motion が人間の近くだけ干渉を検証した
            # motion['lead_in_waypoints'] と同じで (phm.build_lead_in_
            # waypoints)、関節角だけ実機の初期姿勢から補間する。始点が初期
            # 位置と一致していれば空。
            initial_pos = self._initial_base_coords.worldpos()
            prepend_waypoints = phm.build_lead_in_waypoints(
                [initial_pos[0], initial_pos[1], 0.0],
                motion['waypoints'][0], motion['joint_names'],
                start_joint_angles=dict(zip(
                    self._initial_joint_names,
                    self._initial_joint_angle_vector)))
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
        # --auto-execute が指定されていて、実際に self.ri への接続も成功
        # していて、IK・軌道計画が成功し、かつ軌道の干渉検証
        # (verified/lead_in_verified) に通っている場合だけ実機を動かせる
        # (_motion_verified 参照、人との干渉が残ったままの軌道では動かさ
        # ない)。
        executable = (
            args.auto_execute and self.ri is not None and result['solved']
            and motion is not None and display_waypoints is not None
            and self._motion_verified(motion))
        if motion is not None and not self._motion_verified(motion):
            print('[execute] 軌道の干渉検証に通らなかったため (verified={}, '
                  'lead_in_verified={})、--auto-execute でも実機を動かし'
                  'ません。'.format(
                      motion['verified'], motion['lead_in_verified']))
        if display_waypoints is not None:
            self._set_waypoint_slider_range(len(display_waypoints) - 1)
        else:
            # _set_waypoint_slider_range(0) -> _apply_current_waypoint が
            # self._current_result (上で設定済み) から同じ姿勢を
            # self._viewer_lock 付きで反映してくれるので、ここで別途
            # apply_result_pose を呼ぶ必要はない (呼ぶと _viewer_lock なしの
            # 重複更新になり、spin() 側の redraw() と競合しうる)。
            self._set_waypoint_slider_range(0)

        if args.save_dir:
            self._save_attempt(joint_positions, palms, result, motion)

        # --auto-execute: 実機を動かせる状態になった時点で自動的に実行する
        # (--bag/--auto-arm での無人テストや、ブラウザを開けない状況で
        # 使う)。別スレッドへ逃がす -- この _solve_handshake はカメラ
        # フレームのコールバックスレッドで動いており、ここで実機の動作
        # 完了まで待つと以後のフレームを取りこぼすため。
        if executable:
            print('[auto-execute] IK・軌道計画が成功したため実機を動かします '
                  '(--auto-execute)。')
            threading.Thread(
                target=self._execute_on_robot, daemon=True).start()
        elif args.auto_execute:
            # IK が解けなかった、または軌道が干渉検証に通らなかったため
            # 実機を動かせなかったことを人に伝える。
            self._say(args.speech_fail_text)

    @staticmethod
    def _motion_verified(motion):
        """軌道 ``motion`` (``plan_person_motion`` の戻り値) が、経路計画の
        区間 (``verified``) と初期位置からの直進 (lead-in、
        ``lead_in_verified``) の両方で干渉検証に通っているか。どちらかが
        False なら実機を動かさない (``--auto-execute``・
        ``_execute_on_robot`` の両方でこれを見る)。
        """
        return bool(motion['verified']) and bool(motion['lead_in_verified'])

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
        """``_untranslate_result`` と同じ理由で、``motion['waypoints']``/
        ``motion['lead_in_waypoints']`` (仮想座標系の台車位置) を実際の
        ``base_link`` 座標系へ戻す (破壊的に書き換える)。waypoint の関節角は
        台車位置に依存しないのでそのままでよい。"""
        dx, dy = offset
        for wp in motion['waypoints'] + motion.get('lead_in_waypoints', []):
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
        # self.display_robot の関節角・台車位置姿勢の更新 (apply_waypoint_
        # pose 等) は reset_pose() -> 関節ごとの joint_angle() -> base_link.
        # newcoords() と複数ステップにまたがり、その間ロボットは一時的に
        # 「新しい関節角のままだが台車位置は古い」ような不整合な状態になる。
        # spin() が別スレッドで self._viewer_lock を取って独立に viewer.
        # redraw() を呼び続けている (10Hz) ため、この lock を取らずに更新
        # すると、更新の途中の不整合な姿勢がそのまま spin() 側の redraw()
        # に読まれて画面に出てしまう (実メッシュだけがおかしな位置で表示
        # される不具合の原因。sync_robot_collision_overlay は更新が完了した
        # 後に 1 回だけ呼ばれるため影響を受けにくく、干渉モデル側だけ
        # 正しく追従しているように見えていた)。そのため更新から redraw()
        # までを 1 つの self._viewer_lock 区間にして、spin() 側の redraw()
        # が更新の合間に割り込めないようにする。
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

    # ------------------------------------------------------------------
    # 実機動作 (ARM ボタン押下時の首下げ・初期姿勢への復帰、--auto-execute)
    # ------------------------------------------------------------------
    def _nod_head_for_arm(self):
        """ARM ボタン押下時 (``--auto-execute`` の指定有無に関わらず、
        実機 ``self.ri`` への接続に成功している状態のときのみ
        ``_on_arm`` から別スレッドで呼ばれる) に、実機の首を
        ``ARM_HEAD_NOD_PITCH_DEG`` まで下げると同時に、腕を含む全身を
        ``self._initial_joint_names``/``self._initial_joint_angle_vector``
        (``__init__`` 参照、``phm.arms_down_angles`` による「両腕を体の横に
        下ろした」初期姿勢 -- ARM 前/RESET 後に画面へ表示しているのと同じ
        姿勢) まで動かす。まっすぐ顔を見続け腕を構えたままより威圧感の
        少ない、人間が手を差し出しやすい姿勢にする。

        ``controller_type`` を指定せずに送ることで既定の
        ``'default_controller'`` (``larm_controller``/``rarm_controller``/
        ``head_controller``/``waist_controller``/``lifter_controller`` を
        まとめて 1 回のゴールとして送る、``AeroROSRobotInterface.
        default_controller`` 参照) が使われ、首と腕を同時に動かせる。腰・
        腰上げ機構は初期姿勢の値をそのまま使うため、既にその姿勢にいれば
        実質何も動かない。

        送るベクトルのうち初期姿勢に含まれない関節 (指など) は実機の現在値
        をそのまま使う (``self.ri.angle_vector()`` を引数なしで呼ぶと
        joint_states から読んだ実機の現在角を返す) -- ``self.real_robot`` は
        まだ一度も実機の姿勢を反映していない (角度 0 のまま) ため、先に
        現在値を反映してから首・腕だけを上書きする。
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
        self.real_robot.neck_p_joint.joint_angle(
            np.deg2rad(ARM_HEAD_NOD_PITCH_DEG))
        target_av = self.real_robot.angle_vector()
        # 静止 -> 静止の 1 区間でも、瞬間速度のピークは平均の 1.5 倍になる
        # ので、関節速度上限 (VEL_LIMIT_RATIO) を超えない時間まで延ばす。
        move_time, = self._limited_time_list(
            [target_av], None, [ARM_HEAD_NOD_MOVE_TIME])
        self.ri.angle_vector(target_av, move_time)
        print('[ARM] 腕を初期姿勢まで下ろし、首を {:.0f} 度まで下げました。'
              .format(ARM_HEAD_NOD_PITCH_DEG))

    def _execute_on_robot(self):
        """``--auto-execute`` が指定されていて実行条件を満たしたとき
        (``_solve_handshake`` 参照)、計画済みの waypoint 列
        (``self._display_waypoints``、waypoint スライダー/Play で画面
        確認しているのと同じもの) を実機に送る。

        台車移動には ``AeroROSRobotInterface.move_to`` (``move_base``
        経由、costmap を使う) ではなく ``move_trajectory_sequence``
        (costmap を見ず ``base_controller`` の
        ``FollowJointTrajectoryAction`` へ直接軌道を送るだけの相対移動)
        を使う。干渉回避は ``plan_handshake_motion.py`` 側で waypoint
        単位に検証済みのため、costmap 上の障害物回避や大域的な経路計画は
        不要。

        ``display_waypoints`` は ``build_display_waypoints`` により
        2 区間から成る (``self._display_n_prepend``/``_display_n_approach``
        参照): 「接近区間」(初期位置 -> 経路計画の始点 -> 人間の手から
        オフセット分離した hover 目標 (掌の少し手前) まで) と、その続きの
        「押し込み区間」(``result['post_process']`` -- hover 目標から掌へ
        わずかにめり込む位置までの表示専用の補間、``plan_handshake_motion.py``
        は干渉検証していない) である。接近区間の中では waypoint の境界で
        止まらない滑らかな軌道にするため、まとめて 1 つのゴールとして送り
        台車・腕を並行に動かす (従来通り)。ただし押し込みは台車が hover
        目標姿勢にいる前提の動きなので、接近区間の完了後・押し込み区間の
        開始前に、台車のスリップ等による位置ずれを odom 基準で検出・補正し
        (``_correct_base_residual``)、補正が収束してから押し込みへ進む。
        腕はこの補正の間、待たされない (接近区間の腕動作は既に完了して
        いる)。

        ``--auto-execute`` が指定されているとき、台車・関節は常に両方とも
        実際に動かす (``self.ri`` はこのフラグとは別に ARM 時の首下げの
        ため常に接続を試みているので、ここでは改めて ``--auto-execute``
        の指定有無を見る)。
        """
        with self._lock:
            result = self._current_result
            motion = self._current_motion
            display_waypoints = self._display_waypoints
            n_prepend = self._display_n_prepend
            n_approach = self._display_n_approach
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
        # _solve_handshake が検証 NG なら呼ばないが、呼び出し元に依らず
        # 干渉が残る軌道では動かさないようここでも改めて確認する。
        if not self._motion_verified(motion):
            print('[execute] 軌道の干渉検証に通っていないため実機を動かせ '
                  'ません (verified={}, lead_in_verified={})。'.format(
                      motion['verified'], motion['lead_in_verified']))
            return

        joint_names = motion['joint_names']
        # 接近区間 (display_waypoints[:reach_boundary]) は hover 目標
        # (waypoint index reach_boundary - 1) で終わり、押し込み区間
        # (display_waypoints[reach_boundary - 1:]、先頭に hover 目標を含む)
        # がそれに続く。
        reach_boundary = min(max(n_prepend + n_approach, 1),
                             len(display_waypoints))
        print('[execute] 実機で waypoint を {} 個実行します '
              '(接近={}個+押し込み={}個)。'.format(
                  len(display_waypoints), reach_boundary,
                  len(display_waypoints) - reach_boundary))

        self._say(self.args.speech_start_text)

        start_odom_coords, final_traj_point = self._execute_waypoint_segment(
            display_waypoints[:reach_boundary], joint_names)

        self._correct_base_residual(start_odom_coords, final_traj_point)

        # 腕は実機側の遅延で指令から約 0.2 秒遅れて追従し、
        # wait_interpolation はその遅れが解消する前に返る (JOINT_SETTLE_
        # HAND_TOLERANCE 参照)。遅れたまま押し込みを始めると hover を経由
        # しない (手が高い位置から押し付ける) 動きになるため、hover 目標に
        # 腕が追いつくのを待ってから押し込む。
        robot_arm = result['robot_arm']
        self._wait_joint_settle(
            'hover', display_waypoints[reach_boundary - 1], joint_names,
            robot_arm)

        if reach_boundary < len(display_waypoints):
            self._execute_waypoint_segment(
                display_waypoints[reach_boundary - 1:], joint_names)
            # 押し込み自体も遅れて完了するので、押し付け終わってから
            # 「どうぞ」と発話する。
            self._wait_joint_settle(
                '押し込み終了', display_waypoints[-1], joint_names,
                robot_arm)

        self._say(self.args.speech_done_text)

        print('[execute] 実行を終了しました。')

    def _wait_joint_settle(self, label, waypoint, joint_names, robot_arm,
                           tolerance=JOINT_SETTLE_HAND_TOLERANCE,
                           timeout=JOINT_SETTLE_TIMEOUT):
        """実機の腕が ``waypoint`` の関節角に追いつく (関節角だけから求めた
        手先のずれが ``tolerance`` 以下になる) まで、最大 ``timeout`` 秒
        待つ。到達直後と、待った場合は待機後の状態を ``[debug][joint]``
        ログに出す。時間切れでも (接触の負荷等で届かない場合があるため)
        止めずにそのまま戻る。
        """
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
        print('[debug][joint] {}: 手先のずれ(関節角のみ由来)={:.1f}mm '
              '(dx={:+.1f} dy={:+.1f} dz={:+.1f}mm, world系), '
              '1deg/5mm超の関節 {}個: {}'.format(
                  label, np.linalg.norm(hand_err) * 1e3,
                  hand_err[0] * 1e3, hand_err[1] * 1e3, hand_err[2] * 1e3,
                  len(over), ', '.join(over) if over else 'なし'))

    def _joint_tracking_error(self, waypoint, joint_names, robot_arm):
        """``waypoint`` の指令関節角と実機の現在の関節角の差を求める。

        Returns
        -------
        (hand_err, over)
            ``hand_err`` はその差による手先位置のずれ (指令 - 実機、world
            系 [m]、台車の位置ずれは含まない、関節角だけから FK で求めた
            ``{robot_arm}arm_end_coords`` の差)。``over`` は差が 1deg
            (直動関節は 5mm) を超えた関節の表示文字列 (大きい順)。

        ``self.ri.angle_vector()`` は ``self.real_robot`` を実機の関節角で
        上書きするので、呼び出し前の ``self.real_robot`` の姿勢は最後に
        戻す (以後の区間の waypoint 組み立てに影響させないため)。
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
        for joint, diff in zip(self.real_robot.joint_list, diffs):
            if joint.name not in controller_joint_names:
                continue
            if isinstance(joint, LinearJoint):
                entries.append((abs(diff) / 0.005,
                                '{}={:+.1f}mm'.format(joint.name, diff * 1e3)))
            else:
                entries.append((abs(diff) / math.radians(1.0),
                                '{}={:+.1f}deg'.format(
                                    joint.name, math.degrees(diff))))
        entries.sort(reverse=True)
        over = [text for score, text in entries if score > 1.0]
        return target_hand - actual_hand, over

    def _say(self, text):
        """``text`` をロボットに発話させる (非ブロッキング)。
        ``self.sound_client`` が無い・``text`` が空のときは何もしない。
        """
        if self.sound_client is None or not text:
            return
        try:
            self.sound_client.say(text, voice=self.args.speech_voice)
            print('[speech] 「{}」'.format(text))
        except Exception as exc:  # noqa: BLE001  (発話失敗で実機動作を止めない)
            print('[speech] 発話に失敗しました ({})。'.format(exc))

    def _execute_waypoint_segment(self, waypoints, joint_names):
        """``waypoints`` (先頭要素を基準にした 1 区間分) を、waypoint の
        境界で止まらない滑らかな軌道として実機で実行する (台車・腕は並行
        して動く、``_execute_on_robot`` が分割前に行っていたのと同じ処理)。

        Returns
        -------
        (start_odom_coords, final_traj_point)
            ``start_odom_coords`` は台車の軌道を送信した瞬間の odom
            (``self.ri.odom``、台車移動なしの場合は None)。
            ``final_traj_point`` はこの区間の最終 waypoint を
            ``waypoints[0]`` 基準に変換した ``[dx, dy, dyaw]``
            (同じく該当なしの場合は None)。いずれも
            ``_correct_base_residual`` にそのまま渡すためのもの。
        """
        arm_angle_vectors = []  # [av0, av1, ...] (angle_vector_sequence にそのまま渡す)
        # [[dx, dy, dyaw], ...] (この区間の先頭 waypoint からの累積移動量)。
        # move_trajectory_sequence は各要素をその都度 odom から独立に
        # 適用する (直前要素からの相対移動として積み上げない) ため、
        # 差分ではなく先頭 waypoint からの累積量を渡す必要がある。
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
                # この区間の最初の waypoint: 実機は今まさにこの姿勢に
                # いる前提 (区間の先頭が接近区間なら "台車がワールド
                # 原点にいる" という起動時の前提、押し込み区間なら
                # 直前の _correct_base_residual で合わせ込んだ hover
                # 目標姿勢) なので、絶対座標をそのまま原点からの移動量
                # として使える。以後の waypoint もこの姿勢を基準に
                # 累積量を計算する。
                first_base = (bx, by, byaw)
            x0, y0, yaw0 = first_base
            dx_world = bx - x0
            dy_world = by - y0
            dyaw = byaw - yaw0
            # world 系の移動量を、この区間の先頭 waypoint (= 実行開始
            # 時点の台車の向き) 基準 (move_trajectory_sequence が要求
            # する「実行開始時点の台車姿勢を基準にした前後左右」) に
            # 回転させる。
            cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)
            dx = cos_yaw * dx_world + sin_yaw * dy_world
            dy = -sin_yaw * dx_world + cos_yaw * dy_world
            base_trajectory_points.append([dx, dy, dyaw])

        # ここまでで区間分の関節角・移動量を集め終えたので、それぞれ 1 回の
        # ゴールとしてまとめて送る (どちらも非ブロッキング)。
        #
        # waypoint 間の所要時間は固定の dt (motion['dt']) ではなく、区間
        # ごとに台車と腕・首・腰・リフターのうち律速する軸の指令の瞬間
        # 速度がちょうど上限 × VEL_LIMIT_RATIO になるよう決める
        # (_limited_time_list 参照、下限は MIN_SEGMENT_TIME)。台車と腕には
        # 同じ time_list を渡し、律速しない側はその区間だけ上限より遅く
        # 動かす (skrobot の angle_vector_sequence に腕側の時間だけを
        # 延ばさせると、台車とのタイミングがずれて干渉検証済みの経路から
        # 外れるため)。
        time_list = self._limited_time_list(
            arm_angle_vectors, base_trajectory_points,
            [MIN_SEGMENT_TIME] * len(waypoints))
        start_odom_coords = None
        if arm_angle_vectors:
            self.ri.angle_vector_sequence(arm_angle_vectors, time_list)
        if base_trajectory_points:
            # move_trajectory_sequence 自身がこの直後に読む odom (基準
            # 座標) と同じものを、補正計算用に控えておく。
            start_odom_coords = self.ri.odom
            # [debug] 79 度規模の大きな残差が発生する原因調査用。この区間で
            # move_trajectory_sequence に渡す計画上の総回頭量・区間内の
            # waypoint 数・送信直前の odom yaw を記録しておき、
            # _correct_base_residual 側のログと突き合わせて、どの区間の
            # 送信が追従できていないかを切り分ける。
            print('[debug][segment] waypoint数={} 計画上のdyaw={:.1f}deg '
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

        # 送信は上でまとめて 1 回だけ行っているので、完了待ちも最後に
        # まとめて 1 回だけ行う (台車・腕は並行して動く)。
        if arm_angle_vectors:
            self.ri.wait_interpolation()
        if base_trajectory_points:
            self.ri.move_base_trajectory_action.wait_for_result()
            # [debug] wait_for_result() が返った直後 (=このトラジェクトリを
            # 「完了」とみなした瞬間) の実際の odom yaw。計画上の
            # target_yaw (start_odom_yaw + dyaw) とここが既に大きく違って
            # いれば、base_controller が今回の送信区間内で回頭を追従
            # しきれていないことになる。
            odom_after = self.ri.odom
            expected_yaw = (matrix2ypr(start_odom_coords.rotation)[0]
                            + base_trajectory_points[-1][2])
            actual_yaw = matrix2ypr(odom_after.rotation)[0]
            print('[debug][segment] wait_for_result 直後 odom_yaw={:.1f}deg '
                  '(期待値={:.1f}deg, 差={:.1f}deg)'.format(
                      math.degrees(actual_yaw), math.degrees(expected_yaw),
                      math.degrees(
                          (expected_yaw - actual_yaw + math.pi)
                          % (2 * math.pi) - math.pi)))

        final_traj_point = (
            base_trajectory_points[-1] if base_trajectory_points else None)
        return start_odom_coords, final_traj_point

    def _send_base_trajectory(self, base_trajectory_points, time_list,
                              wait):
        """``base_trajectory_points`` (区間先頭の台車姿勢を基準にした累積
        ``[dx, dy, dyaw]`` 列) を ``time_list`` で台車に送る。

        ゴールの組み立て (odom 系への変換) は skrobot の
        ``move_trajectory_sequence`` に任せるが、各点の速度は
        ``_point_velocities`` (始点・終点 0、途中は前後区間の平均速度の
        平均) で付け直してから送る。skrobot のままだと各点の速度が
        「その点から始まる区間の平均速度」になり、静止状態から最初の
        区間の速度へいきなり跳ぶため (2026-09-26 ユーザー要望: ゆるやかに
        加減速させる)。``_limited_time_list`` もこの速度で判定している。
        """
        goal = self.ri.move_trajectory_sequence(
            base_trajectory_points, time_list, stop=True, send_action=False)
        points = goal.goal.trajectory.points
        deltas = np.diff(np.vstack([
            np.zeros(3),
            np.asarray(base_trajectory_points, dtype=np.float64)]), axis=0)
        point_vel = self._point_velocities(
            deltas, np.asarray(time_list, dtype=np.float64))
        # point_vel の x/y は区間先頭の台車の向き基準なので、ゴールの位置
        # (odom 系) に合わせて、先頭の点の yaw (= 送信時の odom の yaw) で
        # 回す。
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
        """区間ごとの所要時間を、実機に送る指令の速度・加速度の最大値が、
        その区間で律速する軸 (台車の並進・回頭、腕・首・腰・リフターの
        各関節) でちょうど上限になるよう決めて返す (2026-09-26 ユーザー
        要望、固定の dt は使わない)。速度の上限は台車が ``BASE_MAX_VEL``/
        ``BASE_MAX_ANGVEL``、関節が URDF の velocity に ``VEL_LIMIT_RATIO``
        を掛けたもの、加速度の上限はその速度まで ``ACCEL_TIME`` 秒かけて
        加速する値。``time_list`` は区間ごとの所要時間の下限 (動きが小さい
        区間でもこれより短くはしない)。

        ``arm_angle_vectors`` は ``angle_vector_sequence`` に渡す関節角列
        (区間 i は ``avs[i] -> avs[i + 1]``、``avs[0]`` は送信直前の実機の
        関節角)、``base_trajectory_points`` は ``_send_base_trajectory``
        に渡す区間先頭からの累積 ``[dx, dy, dyaw]`` 列。どちらも None/空
        なら判定から外す。

        どちらのコントローラ (腕は JointTrajectoryController、台車は
        pr2_base_trajectory_action) も、各点の位置・速度から 3 次エルミート
        で補間するので、その速度・加速度の最大値で判定する (点の速度は
        どちらも ``_point_velocities``)。区間の時間を変えると前後の点の
        速度も変わり隣の区間の最大値も変わるため反復し、わずかでも上限を
        超えて終わった場合は全区間を一律に延ばす (全区間を k 倍すると速度
        は 1/k 倍、加速度は 1/k^2 倍になるので必ず収まる)。
        """
        axis_names = []
        arm_deltas = None
        base_deltas = None
        if arm_angle_vectors:
            ri = self.ri
            controller_joint_names = {
                name for param in ri.controller_param_table[ri.controller_type]
                for name in param['joint_names']}
            # コントローラで動かさない関節 (指など) は上限なし (inf) として
            # 判定から外す。
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
        axis_names += ['{}の加速度'.format(name) for name in axis_names]

        def peak_ratios(times):
            # 区間ごと・軸ごとの (速度の最大値 / 上限) と
            # sqrt(加速度の最大値 / 上限) (時間に対してどちらも反比例する
            # 形にそろえる)。列は axis_names の順。
            vel_ratios = []
            acc_ratios = []
            if arm_deltas is not None:
                point_vel = self._point_velocities(arm_deltas, times)
                vel, acc = self._hermite_peaks(arm_deltas, times, point_vel)
                vel_ratios.append(vel / arm_max_vel)
                acc_ratios.append(acc / (arm_max_vel / ACCEL_TIME))
            if base_deltas is not None:
                point_vel = self._point_velocities(base_deltas, times)
                # pr2_base_trajectory_action は odom 系の x/y 軸ごとに頭打ちに
                # するが、ここでの dx/dy は区間先頭の台車の向き基準なので、
                # どの向きの軸成分でも超えないよう並進は x/y のベクトルの
                # 大きさの最大値で判定する (回頭中のロボット座標系の成分も
                # これ以下になる)。
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
            # 区間ごとに、律速する軸がちょうど上限になる時間へ伸縮する
            # (速度の最大値は時間にほぼ反比例、加速度は 2 乗に反比例する
            # ので平方根をとってある)。加速度は隣の区間の時間にも強く
            # 依存し、ratio 倍そのままだと振動して収束しない (台車の滑らかな
            # 20 区間の例で ±8% の振動が続いた) ため、平方根で半分だけ
            # 動かす (同じ例で 14 回で収束)。
            ratio = np.max(peak_ratios(times), axis=1)
            new_times = np.maximum(min_times, times * np.sqrt(ratio))
            if np.allclose(new_times, times, rtol=1e-4, atol=0.0):
                times = new_times
                break
            times = new_times
        # 反復がわずかな超過を残して終わっても必ず上限内にする。
        worst = float(np.max(peak_ratios(times)))
        if worst > 1.0:
            if worst > 1.01:
                print('[debug][segment] 速度上限の反復で収まらなかったため '
                      '全区間を {:.3f} 倍に延ばします。'.format(worst))
            times *= worst

        # 区間ごとの所要時間と律速した軸 (下限で決まった区間は "下限"、
        # 実機ログで何が律速したかを切り分けるため)。
        final_ratio = peak_ratios(times)
        entries = [
            '{}:{:.2f}s({})'.format(
                i, times[i],
                '下限' if times[i] <= min_times[i] * (1.0 + 1e-3)
                else axis_names[int(np.argmax(final_ratio[i]))])
            for i in range(len(times))]
        print('[debug][segment] 速度上限 (×{}, 加速 {}s) で決めた所要時間 '
              '(合計 {:.2f}s): {}'.format(
                  VEL_LIMIT_RATIO, ACCEL_TIME, float(np.sum(times)),
                  ', '.join(entries)))
        return [float(t) for t in times]

    @staticmethod
    def _point_velocities(deltas, times):
        """各点に付ける速度 (形状は ``(区間数 + 1, 軸数)``)。skrobot の
        ``angle_vector_sequence`` と同じ規則で、途中の点は前後区間の平均
        速度の平均、前後で符号が逆の軸と最後の点は 0。始点 (送信直前の
        実機) も静止しているものとして 0。腕は skrobot がこの規則で送り、
        台車は ``_send_base_trajectory`` がこの規則で付け直して送る。
        """
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
        """各区間 (``deltas[i]`` = 区間 i の変化量、``times[i]`` = 所要時間)
        を、点の速度 ``point_vel`` (区間 i の両端は ``point_vel[i]``/
        ``point_vel[i + 1]``) の 3 次エルミート補間で動かしたときの、速度と
        加速度の最大値 (絶対値) の組を返す。形状は軸ごと (``deltas`` と
        同じ)、``norm=True`` なら全軸をまとめたベクトルの大きさで
        ``(区間数,)``。

        p(t) = a t^3 + b t^2 + v0 t (p(0) = 0, p(T) = delta, p'(0) = v0,
        p'(T) = v1)。加速度 p''(t) は 1 次式なので最大値は両端 (ベクトルの
        大きさでも同じ)。速度 p'(t) は 2 次式で、軸ごとなら最大値は両端か
        頂点。ベクトルの大きさは軸ごとの最大値が別々の時刻に出ることが
        あり合成すると過大になるため、区間内を ``n_samples`` 点で評価する
        (速度は 2 次式で滑らかなので誤差は上限の 0.1% 未満)。
        """
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

    def _correct_base_residual(
            self, start_odom_coords, final_traj_point,
            max_attempts=BASE_CORRECTION_MAX_ATTEMPTS,
            position_tolerance=BASE_CORRECTION_POSITION_TOLERANCE,
            angle_tolerance=BASE_CORRECTION_ANGLE_TOLERANCE):
        """接近区間の完了直後、押し込み (post_process) に進む前に台車の
        位置ずれ (スリップ等による ``move_trajectory_sequence`` のオープン
        ループ指令と実際の到達姿勢との差) を odom 基準で検出し、収束する
        まで相対移動で補正する。

        ``start_odom_coords``/``final_traj_point`` は接近区間を実行した
        ``_execute_waypoint_segment`` の戻り値そのもの --
        ``start_odom_coords`` は軌道送信時に基準として使われた odom、
        ``final_traj_point`` はその区間の最終 waypoint (= hover 目標) を
        区間先頭からの相対量 ``[dx, dy, dyaw]`` で表したもの。両者から
        ``move_trajectory_sequence`` が内部で行うのと同じ変換で目標の
        絶対姿勢 (odom 系) を求め、実行後の実際の odom との残差を
        ロボット正面基準に回転させて相対移動として送り返す。収束閾値は
        skrobot の ``go_pos_unsafe_wait`` と同じ (位置2.5cm/角度2.5度)。
        sec (所要時間) は、静止 -> 静止の 1 区間として指令の速度・加速度
        が上限 (``BASE_MAX_VEL``/``BASE_MAX_ANGVEL`` × ``VEL_LIMIT_RATIO``、
        加速度は ``ACCEL_TIME``) を超えない時間 (``_limited_time_list``
        参照)。

        接近区間に台車移動が無かった場合は何もしない。
        """
        if start_odom_coords is None or final_traj_point is None:
            return
        dx, dy, dyaw = final_traj_point
        start_yaw = matrix2ypr(start_odom_coords.rotation)[0]
        start_x, start_y = start_odom_coords.translation[:2]
        target_x = start_x + math.cos(start_yaw) * dx - math.sin(start_yaw) * dy
        target_y = start_y + math.sin(start_yaw) * dx + math.cos(start_yaw) * dy
        target_yaw = start_yaw + dyaw
        # [debug] 79 度規模の大きな残差の原因調査用。ここで求めた
        # target_yaw が「意図した hover 目標の向き」と一致しているかを
        # 見るためのログ (_execute_waypoint_segment の [debug][segment]
        # ログの start_odom_yaw/dyaw と同じ値になっているはず)。
        print('[debug][correct] start_odom(x={:.3f} y={:.3f} yaw={:.1f}deg) '
              'final_traj_point(dx={:.3f} dy={:.3f} dyaw={:.1f}deg) '
              '-> target(x={:.3f} y={:.3f} yaw={:.1f}deg)'.format(
                  start_x, start_y, math.degrees(start_yaw),
                  dx, dy, math.degrees(dyaw),
                  target_x, target_y, math.degrees(target_yaw)))

        err_norm = 0.0
        err_yaw = 0.0
        for attempt in range(max_attempts):
            odom = self.ri.odom
            cur_x, cur_y = odom.translation[:2]
            cur_yaw = matrix2ypr(odom.rotation)[0]
            err_x_world = target_x - cur_x
            err_y_world = target_y - cur_y
            err_yaw = (target_yaw - cur_yaw + math.pi) % (2 * math.pi) - math.pi
            # world 系の残差を、ロボットの現在の向き基準 (前後左右) に
            # 回転させる (move_trajectory に渡す相対移動量はこの基準)。
            err_x = math.cos(cur_yaw) * err_x_world + math.sin(cur_yaw) * err_y_world
            err_y = -math.sin(cur_yaw) * err_x_world + math.cos(cur_yaw) * err_y_world
            err_norm = math.hypot(err_x, err_y)
            # [debug] 生の odom 値そのもの (回転方向の符号が想定通りかも
            # ここで確認できる)。
            print('[debug][correct] attempt={} odom(x={:.3f} y={:.3f} '
                  'yaw={:.1f}deg) err_yaw={:.1f}deg'.format(
                      attempt, cur_x, cur_y, math.degrees(cur_yaw),
                      math.degrees(err_yaw)))

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
            # 静止 -> 静止の 1 区間として、速度・加速度が上限を超えない
            # 時間を _limited_time_list で求める (最低 1 秒は skrobot の
            # go_pos_unsafe_wait と同じ)。
            sec, = self._limited_time_list(
                None, [[err_x, err_y, err_yaw]], [1.0])
            self._send_base_trajectory(
                [[err_x, err_y, err_yaw]], [sec], wait=True)

        print('[execute][WARN] 押し込み前の台車の位置ずれ補正が {} 回で '
              '収束しませんでした (残差 {:.3f}m / {:.1f}deg)。このまま '
              '押し込み動作へ進みます。'.format(
                  max_attempts, err_norm, math.degrees(abs(err_yaw))))

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
                else skeleton_drawing.build_skeleton_links(joint_positions))
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
        # ARMED 中に判定できた差し出し手のスコア内訳を出す (ARM を押しても
        # 見つからないときの原因切り分け用、_try_handshake 参照)。
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
                content += ('\n\n**軌道:** {} / 計画時 (指なし) の検証: {} / '
                           '初期位置からの直進の検証: {}\n\n'
                           'waypoint {}/{}'.format(
                               kind, verified_text, lead_in_text,
                               waypoint_index, self.waypoint_slider.max))
                if waypoint_index < n_prepend:
                    lead_in_dists = motion.get('lead_in_min_distances', [])
                    dist = (lead_in_dists[waypoint_index]
                            if waypoint_index < len(lead_in_dists) else None)
                    content += (
                        ' (初期位置から接近開始位置への直進、人間から離れて'
                        'いるため干渉検証の対象外)' if dist is None
                        else ' (初期位置から接近開始位置への直進、計画時 '
                             '(指なし) の干渉余裕: {:+.4f} m)'.format(dist))
                elif waypoint_index < n_prepend + n_approach:
                    dist = motion['waypoint_min_distances'][
                        waypoint_index - n_prepend]
                    content += (' (この waypoint の計画時 (指なし) の干渉'
                               '余裕: {:+.4f} m)'.format(dist))
                else:
                    content += (' (掌への押し込み、経路計画の干渉検証の対象外)')
            else:
                content += ('\n\n**軌道:** 計画なし ({})'.format(
                    'IK 失敗' if not result['solved'] else '計算中'))
        # 干渉しているかどうかの結論は、上の計画時 (指なし) の干渉余裕では
        # なく下の事後検証で出す -- 経路計画・IK は指なしロボットで解いて
        # いる (self.verification_pairs/motion の waypoint_min_distances)
        # ため、指先や表示専用フレーム (初期位置からの移動/掌への押し込み)
        # を含む「いま画面に出ている姿勢が実際に貫通しているか」は、表示
        # 中の waypoint の姿勢に対して指ありで解き直した _refresh_collision
        # _pairs_text の結果だけが答えられる。両方に貫通の有無を書くと、
        # 同じ waypoint について食い違う判定が並んで紛らわしいため、
        # 上には計画時の数値だけを出す。
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
            try:
                rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                # rosbag 再生時などにシミュレーション時刻が巻き戻ると
                # rate.sleep() が例外を投げてループごと落ちる (viewer が
                # 閉じてしまう) ため、無視してループを継続する。
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
        '--tf-cache-time', type=float, default=30.0,
        help='tf2 バッファの保持時間 [秒] (既定 30.0)。カメラ側と '
            'base_link 側の TF を配信しているマシン間でシステムクロック '
            'がズレていると、既定の 10 秒では両者の有効期間が重ならず '
            'TF が引けないことがある。根本的にはマシン間の時刻同期が '
            '必要 (NTP/chrony)。')
    parser.add_argument(
        '--armed-timeout', type=float, default=30.0,
        help='ARMED になってから offered_hand が決まらなければ諦めて '
            'IDLE に戻るまでの秒数 (既定 30.0)。')
    parser.add_argument(
        '--client-wait-timeout', type=float, default=30.0,
        help='viser のブラウザクライアント接続を待つ 1 回あたりの秒数 '
            '(繰り返し待つ、既定 30.0)。')
    parser.add_argument(
        '--no-open-browser', action='store_true',
        help='viser のブラウザの自動起動を無効にする (URL を自分で開く '
            '場合)。')
    parser.add_argument(
        '--no-wait-for-client', action='store_true',
        help='viser のブラウザクライアント接続を待たずに起動を続ける。'
            '既定では ``viewer_nav.wait_for_client`` がクライアント接続 '
            'まで無期限に待ち続けるため、``--bag``/``--auto-arm`` を使った '
            '無人でのバッグ再生テストでは併せてこれを指定する。')
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
            '遅れる。')
    parser.add_argument(
        '--joint-smoothing-beta', type=float, default=0.3,
        help='One Euro Filter の速度依存カットオフの係数 (既定 0.3)。'
            '上げるほど速い動きへの追従の遅れが減るが静止時のジッタが '
            '増える。')
    parser.add_argument(
        '--joint-smoothing-dcutoff', type=float, default=1.0,
        help='One Euro Filter の速度推定のカットオフ周波数 [Hz] (既定 1.0)。')
    parser.add_argument(
        '--offer-score-min', type=float, default=0.65,
        help='差し出し手と判定するスコアの閾値 (既定 0.65)。'
            'estimate_palm_poses.OFFER_SCORE_MIN ({:.2f}) は合成骨格向けに '
            '調整された値で実カメラでは届きにくいため、実カメラ用にここで '
            '下げてある。それでも ARM を押して差し出し手が見つからない '
            '場合は、viser 画面に表示されるスコアを見ながらさらに調整する '
            'とよい。'.format(epp.OFFER_SCORE_MIN))
    parser.add_argument(
        '--max-person-distance', type=float, default=4.2,
        help='人物 (腰の中点) からロボット手先までの距離 [m] がこれを '
            '超えたら、スコアを見るまでもなく両手とも差し出し候補から '
            '外す (既定 4.2、record_palm_offer_clips.py の既定値と揃えて '
            'ある)。奥や画面の端に映り込んだだけの、手を差し出す気の無い '
            '通行人を拾わないための足切り (estimate_palm_poses.'
            'OfferedHandSelector の max_distance 引数、veto 理由は '
            '"too_far"、viser 画面のスコア表示にも出る)。0 以下を指定する '
            'と足切りを無効にする。')
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='使うロボットの腕。既定 (auto) は人間の手の反対側 '
            '(solve_palm_ik.py の DEFAULT_ROBOT_ARM と同じ)。')
    parser.add_argument(
        '--robot-hand-position', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'Z'),
        help='掌推定 (差し出し手判定) が基準にするロボット手先の base_link '
            '座標 [m] を固定値で指定する (既定 None)。指定すると '
            '--robot-hand-frame での TF 解決より優先される。')
    parser.add_argument(
        '--robot-hand-frame', type=str, default='r_eef_grasp_link',
        help='--robot-hand-position が未指定のとき、差し出し手判定の基準に '
            '毎フレーム TF (--base-frame からのこのフレーム) を引いて使う '
            '(既定 r_eef_grasp_link -- skrobot Aero モデルの rarm_end_'
            'coords に対応する実リンクで、実機では /aero_state_publisher '
            'が配信する、record_palm_offer_clips.py と同じ既定値)。ロボット '
            '未接続などでまだ TF が引けない間だけ、右腕の種の姿勢の手先 '
            '位置にフォールバックする。')
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
        help='接近開始位置 (人間の手を中心に公転を始める円) の半径への '
            '上乗せ分 [m] (既定 {})。'.format(phm.DEFAULT_APPROACH_DISTANCE))
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
        '--motion-collision-verify-tolerance', type=float,
        default=phm.DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE,
        help='軌道上の waypoint の事後検証で許容する最大貫通量 [m] '
            '(既定 {})。画面の状態表示用の --collision-verify-tolerance '
            '(指先まで含めた事後検証) とは別物 -- plan_person_motion '
            '呼び出し時だけこちらの値に差し替える (_solve_handshake 参照)。'
            .format(phm.DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE))
    parser.add_argument(
        '--force-optimize', action='store_true',
        help='pre-touch/線形補間の候補が事後検証に通っていても早期 '
            'return せず、必ず jaxls の軌道最適化まで実行する '
            '(plan_handshake_motion.plan_person_motion 参照、既定は '
            'オフ)。軌道最適化そのものの計算時間を単独で計測したい '
            'ときに使う。')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='軌道計画の warm start を揺らす際に使う numpy の乱数シード '
            '(既定は指定なし)。')
    parser.add_argument(
        '--playback-fps', type=float, default=DEFAULT_PLAYBACK_FPS,
        help='Play チェックボックスをオンにしたときの waypoint 自動再生の '
            '速さ [waypoint/秒] (既定 {})。'.format(DEFAULT_PLAYBACK_FPS))
    # --- 実機動作 (_execute_on_robot 参照) ---
    parser.add_argument(
        '--auto-execute', action='store_true',
        help='IK・軌道計画が成功した時点で、計画済みの軌道に沿って台車・'
            '関節の両方を実機で自動的に動かす (AeroROSRobotInterface。'
            '台車は go_pos_unsafe 相当の相対移動、move_to/move_base の '
            'costmap は使わない。--auto-arm と組み合わせるとブラウザ操作 '
            'なしで一連の動作を実行できる)。指定しなければ実機は動かさず、'
            'viser 画面での waypoint スライダー/Play による確認のみになる '
            '(既定オフ)。')
    parser.add_argument(
        '--speech-start-text', type=str, default='今から行きますね',
        help='--auto-execute で実機が動き出すときに発話する文 '
            '(空文字列で発話しない)。')
    parser.add_argument(
        '--speech-done-text', type=str, default='どうぞ、手を握ってください',
        help='--auto-execute で掌を差し出し終えたときに発話する文 '
            '(空文字列で発話しない)。')
    parser.add_argument(
        '--speech-fail-text', type=str,
        default='ごめんなさい、うまく手を出せませんでした',
        help='--auto-execute で IK・軌道計画に失敗して実機を動かせなかった '
            'ときに発話する文 (空文字列で発話しない)。')
    parser.add_argument(
        '--speech-voice', type=str, default='四国めたん-ノーマル',
        help='発話に使う声 (sound_play の voice、既定 "四国めたん-ノーマル")。')
    # --- 実カメラ無しでのテスト (rosbag 再生、record_palm_offer_clips.py
    # が保存したクリップを入力にする) ---
    parser.add_argument(
        '--bag', type=str, default=None,
        help='実カメラの代わりに再生する rosbag ファイル '
            '(record_palm_offer_clips.py が保存したクリップなど)。'
            'color/depth/camera_info/tf/tf_static を同じデフォルトの '
            'トピック名で記録済みなら、このノードのライブトピック '
            'subscribe をそのまま流用できる。指定すると内部で '
            '"rosbag play" をサブプロセスとして起動し、ノード終了時に '
            '終了させる。')
    parser.add_argument(
        '--bag-rate', type=float, default=1.0,
        help='--bag 再生時の速度倍率 ("rosbag play -r"、既定 1.0)。')
    parser.add_argument(
        '--bag-loop', action='store_true',
        help='--bag をループ再生する ("rosbag play --loop")。')
    parser.add_argument(
        '--auto-arm', action='store_true',
        help='起動直後に viser の ARM ボタンを押した状態 (ARMED) から '
            '始める。--bag での無人テスト時に、ブラウザで ARM ボタンを '
            'クリックする代わりに使う。')
    parser.add_argument(
        '--no-robot-interface', action='store_true',
        help='実機 (AeroROSRobotInterface) への接続を試みない (既定は '
            '--auto-execute の指定に関わらず常に接続を試みる)。実機なし '
            'で rosbag のみを使って動作確認する際に指定する。')
    # argparse は roslaunch が付ける残りの引数 (__name/__log 等) を無視する
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    bag_process = None
    if args.bag:
        if shutil.which('rosbag') is None:
            sys.exit(
                'rosbag が見つかりません。source /opt/ros/noetic/setup.bash '
                '等で ROS の setup.bash を読み込んでから実行してください。')
        if not os.path.exists(args.bag):
            sys.exit('--bag で指定したファイルが見つかりません: {}'.format(
                args.bag))
        # --bag 再生時は "rosbag play --clock" が配信する /clock に同期させ、
        # クリップ記録時のタイムスタンプのまま TF/画像の時刻整合性を保つ。
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
