#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""実カメラ (color/depth/camera_info) と TF (``/tf``/``/tf_static``) を
常時ローリング録画しておき、掌の差し出しを検出したら検出時刻の
``--pre-seconds`` 秒前から ``--post-seconds`` 秒後までを 1 本の rosbag
クリップとして ``--save-dir`` に切り出して保存するノード。差し出しを
検出した瞬間のカラー画像に骨格を重ねた PNG も、クリップ 1 本につき
1 枚あわせて保存する (人手での確認・データセットのサムネイル用)。

``run_camera_pipeline_test.py`` が ARM ボタンを押した後だけ掌推定を行う
のに対し、このノードは無人でバックグラウンドに常駐させることを想定し、
ARM 操作なしで常時検出を行う (掌推定の判定基準 -- ``PeoplePoseEstimator``
の各閾値・``--offer-score-min``・関節位置の時間方向平滑化
(``aero_demo.skeleton_filters.OneEuroFilter``)・差し出し手判定の基準にする
ロボット手先位置の TF 解決 (``aero_demo.ros_camera_utils.
lookup_frame_position``) -- はすべて ``run_camera_pipeline_test.py`` と
揃えてあるので、そちらの ARMED 中の判定と同じ基準でトリガーする)。IK・
軌道計画・viser 表示は行わない (scikit-robot/jax に依存しない、録画専用の
軽量なノード)。差し出し手判定の基準にするロボット手先の位置は、既定では
実機の TF (``--robot-hand-frame``, 既定 ``r_eef_grasp_link``) を毎フレーム
引いて使う (TF がまだ引けない間だけ概算値にフォールバックする、
``_FALLBACK_ROBOT_HAND_POSITION`` 参照)。

保存したクリップは ``/tf``・``/tf_static`` も含めて自己完結しているため、
``run_camera_pipeline_test.py --bag <クリップ>.bag`` に渡せば、実カメラ・
実ロボットの TF 配信なしにそのままパイプラインをテストできる (デフォルト
のトピック名がこのノードの録画対象と一致しているため、``rosbag play`` が
再生したトピックをそのまま subscribe できる)。

Usage
-----
    python3 scripts/ros/record_palm_offer_clips.py
    python3 scripts/ros/record_palm_offer_clips.py --save-dir /tmp/palm_offer_clips
"""

import argparse
import collections
import os
import sys
import threading
import time

import cv2
import numpy as np

import rosbag
import rospy
import message_filters
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
_PKG_SRC_DIR = os.path.join(_SCRIPTS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import json_io  # noqa: E402
from aero_demo import palm_plane_view  # noqa: E402
from aero_demo import skeleton_filters  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, lookup_camera_to_base, lookup_frame_position,
    ndarray_to_imgmsg, transform_to_matrix)

import estimate_palm_poses as epp  # noqa: E402

# バッグに書き込む各トピック名 (self._buffers/self._writer_seq のキーにも使う)。
_TF_TOPIC = '/tf'
_TF_STATIC_TOPIC = '/tf_static'

# 差し出し手判定の基準にするロボット手先の base_link 座標 [m] は、既定では
# 実機の TF (--robot-hand-frame, 既定 r_eef_grasp_link) を毎フレーム引いて
# 使う (aero-ros-pkg 側で /aero_state_publisher が r_hand_link の子として
# 配信している、skrobot Aero モデルの rarm_end_coords に対応する実リンク)。
# estimate_palm_poses.OfferedHandSelector 自身の既定動作 (robot_position=
# None) は合成骨格向けの「人物より world +x 側にロボットがいる」という
# 世界座標の仮定で、base_link 座標系の実カメラでは前提が食い違う (ロボット
# 自身はおよそ原点付近 = 人物より -x 側にいることが多い) ため使わない。
# TF がまだ引けない (ロボット未接続、/aero_state_publisher 未起動など) 間
# だけ使うフォールバック値 (Aero の右腕初期姿勢の手先位置に近い概算値)。
_FALLBACK_ROBOT_HAND_POSITION = (0.32, -0.55, 0.93)

# 骨格の関節同士のつながり (関節名のペア) と、それを画像に描画する関数
# (run_camera_pipeline_test.py の BONE_NAME_PAIRS/_fill_missing_wrist_from_
# hand/draw_skeleton_overlay と同じもの。scripts/ros/ 層は各ファイル単独で
# 完結させる方針 (run_camera_pipeline_test.py のモジュール docstring 付近
# のコメント参照) のためここにも複製してある)。
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
HAND_SEQUENCE = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
HAND_WRIST_PAIRS = [('RWrist', 'RHand0'), ('LWrist', 'LHand0')]
BONE_NAME_PAIRS = BODY_BONE_PAIRS + HAND_WRIST_PAIRS + [
    ('{}Hand{}'.format(side, a), '{}Hand{}'.format(side, b))
    for side in ('R', 'L') for a, b in HAND_SEQUENCE]


def _fill_missing_wrist_from_hand(positions):
    """手首 (``RWrist``/``LWrist``) が未検出でも、Hand モデルの手首
    ランドマーク (``RHand0``/``LHand0``) が検出できていればその位置を
    手首として補って返す (辞書のコピー、``positions`` 自体は書き換えない)。"""
    filled = dict(positions)
    for wrist_name, hand_wrist_name in (('RWrist', 'RHand0'),
                                        ('LWrist', 'LHand0')):
        if wrist_name not in filled and hand_wrist_name in filled:
            filled[wrist_name] = filled[hand_wrist_name]
    return filled


_OFFERED_HAND_BGR = (0, 0, 255)  # 差し出し手と判定された側を描く赤 (BGR)


def _is_offered_hand_joint(name, offered_side):
    return offered_side is not None and name.startswith(offered_side + 'Hand')


def _is_offered_hand_bone(start_name, end_name, offered_side):
    return (_is_offered_hand_joint(start_name, offered_side)
           or _is_offered_hand_joint(end_name, offered_side))


def draw_skeleton_overlay(color_bgr, joints_2d, offered_side=None):
    """カメラ画像 (BGR) に、検出できた 2D 関節位置を重ねて描いた画像を
    返す (元の ``color_bgr`` は書き換えない)。``joints_2d`` は
    ``PeoplePoseEstimator.estimate_3d`` が返す 1 人分の
    ``[{"limb": str, "x": float, "y": float, "score": float}, ...]``
    (画像座標、score < 0 は未検出)。``offered_side`` (``'R'``/``'L'``/
    ``None``) を渡すと、差し出し手と判定された側の手だけ赤で描く。"""
    overlay = color_bgr.copy()
    positions = {j['limb']: (int(round(j['x'])), int(round(j['y'])))
                for j in joints_2d if j['score'] >= 0}
    positions = _fill_missing_wrist_from_hand(positions)
    for start_name, end_name in BONE_NAME_PAIRS:
        if start_name not in positions or end_name not in positions:
            continue
        if _is_offered_hand_bone(start_name, end_name, offered_side):
            bgr = _OFFERED_HAND_BGR
        else:
            color = palm_plane_view.bone_color(
                '{}->{}'.format(start_name, end_name))
            bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.line(overlay, positions[start_name], positions[end_name],
                 bgr, 2, cv2.LINE_AA)
    for name, point in positions.items():
        dot_bgr = (_OFFERED_HAND_BGR
                  if _is_offered_hand_joint(name, offered_side)
                  else (255, 255, 255))
        cv2.circle(overlay, point, 3, dot_bgr, -1, cv2.LINE_AA)
    return overlay


class ClipWindowTracker(object):
    """``offered_hand`` の ``None -> 'R'/'L'`` への遷移 (立ち上がり) を
    検出時刻 ``t0`` として、録画すべき時間帯
    ``[t0 - pre_seconds, t0 + post_seconds]`` を管理する状態機械。

    ROS (rospy/rosbag) に一切依存しないので、実カメラ・roscore 無しで
    素の Python の float 時刻列だけを使って単体テストできる。
    """

    def __init__(self, pre_seconds, post_seconds, cooldown_seconds):
        self.pre_seconds = float(pre_seconds)
        self.post_seconds = float(post_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.trigger_time = None
        self.trigger_side = None
        self.window_end = None
        self._prev_side = None
        self._last_close_t = -float('inf')

    def observe_detection(self, t, offered_hand):
        """検出フレーム 1 つ分の結果を渡す。

        このフレームで新しくウィンドウを開始した (= 直前まで None だった
        ``offered_hand`` が非 None になり、クールダウンも空けている) なら
        True を返す。呼び出し側はこの戻り値が True のときだけ、ローリング
        バッファのうち ``[t - pre_seconds, t]`` をバッグへ書き出す。
        """
        prev_side = self._prev_side
        self._prev_side = offered_hand
        triggered = (
            self.window_end is None
            and prev_side is None and offered_hand is not None
            and t - self._last_close_t >= self.cooldown_seconds)
        if triggered:
            self.trigger_time = t
            self.trigger_side = offered_hand
            self.window_end = t + self.post_seconds
        return triggered

    def is_recording(self, t):
        """時刻 ``t`` のメッセージを現在アクティブなクリップへ書き込むべきか."""
        return self.window_end is not None and t <= self.window_end

    def in_cooldown(self, t):
        """時刻 ``t`` が、直前のクリップを閉じた後のクールダウン中か
        (録画中はクールダウンではない、``is_recording`` と排他)。"""
        return (self.window_end is None
               and t - self._last_close_t < self.cooldown_seconds)

    def maybe_close(self, t):
        """``t`` が ``window_end`` を過ぎていればウィンドウを閉じ、
        ``(trigger_time, trigger_side)`` を返す。閉じなければ ``None``。"""
        if self.window_end is None or t <= self.window_end:
            return None
        result = (self.trigger_time, self.trigger_side)
        self._last_close_t = t
        self.trigger_time = None
        self.trigger_side = None
        self.window_end = None
        return result


class PalmOfferClipRecorder(object):
    """カメラ入力 -> 骨格推定 -> 掌の差し出し検出 -> rosbag クリップ保存."""

    def __init__(self, args):
        self.args = args
        os.makedirs(args.save_dir, exist_ok=True)
        self._seq = 0
        self._lock = threading.Lock()

        # ローリングバッファ: トピック名 -> [(t, msg), ...] (t 昇順)。
        # トリガー時に [t0 - pre_seconds, t0] の分をバッグへまとめて書く。
        self._buffer_seconds = args.pre_seconds + 0.5
        self._buffers = collections.defaultdict(collections.deque)

        # /tf_static (URDF 固定オフセット、r_eef_grasp_link 等) はロボット
        # 起動時に 1 度しか配信されない (latched) ため、他のトピックと同じ
        # スライディングウィンドウ (_buffer_append) に乗せると、記録開始
        # までに buffer_seconds 秒以上経ってしまい単に消えてしまう (この
        # クラスがそれで r_eef_grasp_link 等を一切バッグに書けていなかった
        # 不具合があった)。子フレーム名をキーに最新の変換を保持し続け、
        # クリップ開始のたびに別途書き込む (_write_latest_tf_static)。
        self._tf_static_transforms = {}

        self._active_bag = None
        self._active_bag_path = None

        self.tracker = ClipWindowTracker(
            args.pre_seconds, args.post_seconds, args.cooldown_seconds)

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

        # 深度ノイズによる関節位置の単発の飛びを抑える時間方向の平滑化
        # (aero_demo.skeleton_filters.OneEuroFilter、run_camera_pipeline_
        # test.py の self._joint_smoother と同じクラス・同じ既定値)。
        self._joint_smoother = skeleton_filters.OneEuroFilter(
            mincutoff=args.joint_smoothing_mincutoff,
            beta=args.joint_smoothing_beta,
            dcutoff=args.joint_smoothing_dcutoff)

        # robot_position はここでは確定させない (毎フレーム _resolve_robot_
        # position で TF から引き直して選定器に差し込む、_on_frame 参照)。
        max_distance = (None if args.max_person_distance <= 0
                        else args.max_person_distance)
        offered_hand_selector = epp.OfferedHandSelector(
            robot_position=None, score_min=args.offer_score_min,
            max_distance=max_distance)
        self.palm_estimator = epp.PalmPoseEstimator(offered_hand_selector)

        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        info_sub = message_filters.Subscriber(
            args.camera_info_topic, CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_frame)

        rospy.Subscriber(_TF_TOPIC, TFMessage, self._on_tf, queue_size=50)
        rospy.Subscriber(
            _TF_STATIC_TOPIC, TFMessage, self._on_tf_static, queue_size=50)

        # デバッグ用: 骨格・掌の有無や録画中かどうかに関わらず、購読者が
        # いれば毎フレーム publish する (クールダウン中だけ止める、
        # _on_frame 末尾参照)。録画中に保存する PNG と同じ描画。
        self.skeleton_image_pub = rospy.Publisher(
            '~skeleton_image', Image, queue_size=1)

        print('[record-palm-offer-clips] 録画を開始しました -> {}'.format(
            os.path.abspath(args.save_dir)))

    # ------------------------------------------------------------------
    # buffering
    # ------------------------------------------------------------------
    def _buffer_append(self, topic, t, msg):
        buf = self._buffers[topic]
        buf.append((t, msg))
        threshold = t - self._buffer_seconds
        while buf and buf[0][0] < threshold:
            buf.popleft()

    def _flush_pre_buffer(self, bag, t0):
        """ローリングバッファのうち ``[t0 - pre_seconds, t0]`` を時刻順に
        まとめてバッグへ書き出す (トリガー直後、クリップ開始時に 1 回だけ
        呼ぶ)。"""
        window_start = t0 - self.args.pre_seconds
        entries = []
        for topic, buf in self._buffers.items():
            for t, msg in buf:
                if window_start <= t <= t0:
                    entries.append((t, topic, msg))
        entries.sort(key=lambda e: e[0])
        for t, topic, msg in entries:
            bag.write(topic, msg, rospy.Time.from_sec(t))
        self._write_latest_tf_static(bag, window_start)

    def _write_latest_tf_static(self, bag, stamp):
        """保持している最新の ``/tf_static`` (子フレーム名 -> 変換) を、
        クリップ先頭の時刻でまとめて書き込む。

        ``/tf_static`` はロボット起動時に 1 度だけ配信される (latched)
        ため、他のトピックと同じスライディングウィンドウの
        ``_buffers``/``_buffer_append`` には乗せていない (乗せると
        記録開始までに ``_buffer_seconds`` 秒以上経ってバッファから
        追い出され、``r_eef_grasp_link`` のような URDF 固定オフセットが
        クリップに一切書き込まれなくなる)。ここで別途、その時点で
        受信済みの全 ``/tf_static`` 変換をまとめて 1 メッセージとして
        書き込む。"""
        if not self._tf_static_transforms:
            return
        msg = TFMessage(transforms=list(self._tf_static_transforms.values()))
        bag.write(_TF_STATIC_TOPIC, msg, rospy.Time.from_sec(stamp))

    # ------------------------------------------------------------------
    # clip lifecycle
    # ------------------------------------------------------------------
    def _start_clip(self, t0, side, color, joints_2d):
        stamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(t0))
        name = '{}_{}_{:03d}'.format(stamp, side, self._seq)
        self._seq += 1
        path = os.path.join(self.args.save_dir, name + '.bag')
        bag = rosbag.Bag(path, 'w')
        self._flush_pre_buffer(bag, t0)
        self._active_bag = bag
        self._active_bag_path = path

        snapshot_path = path[:-len('.bag')] + '.png'
        overlay = (draw_skeleton_overlay(color, joints_2d, offered_side=side)
                  if joints_2d else color)
        cv2.imwrite(snapshot_path, overlay)

        print('[record-palm-offer-clips] 差し出し ({}) を検出、クリップ '
              '開始: {} (スナップショット: {})'.format(
                  side, path, snapshot_path))

    def _close_clip(self, t0, side):
        bag, path = self._active_bag, self._active_bag_path
        self._active_bag = None
        self._active_bag_path = None
        bag.close()
        snapshot_path = path[:-len('.bag')] + '.png'
        meta = {
            'trigger_stamp': t0,
            'offered_hand': side,
            'pre_seconds': self.args.pre_seconds,
            'post_seconds': self.args.post_seconds,
            'topics': [
                self.args.color_topic, self.args.depth_topic,
                self.args.camera_info_topic, _TF_TOPIC, _TF_STATIC_TOPIC],
            'bag_path': os.path.abspath(path),
            'snapshot_path': os.path.abspath(snapshot_path),
        }
        json_io.save_json(path[:-len('.bag')] + '.json', meta)
        print('[record-palm-offer-clips] クリップ保存完了: {}'.format(path))

    def _write_if_recording(self, topic, t, msg):
        if self._active_bag is not None and self.tracker.is_recording(t):
            self._active_bag.write(topic, msg, rospy.Time.from_sec(t))

    def _maybe_close_clip(self, t):
        closed = self.tracker.maybe_close(t)
        if closed is not None:
            self._close_clip(*closed)

    def _resolve_robot_position(self):
        """差し出し手判定の基準にするロボット手先の base_link 座標を返す.

        ``--robot-hand-position`` が明示されていればそれを固定で使う。
        そうでなければ実機の TF (``--robot-hand-frame`` -> ``--base-
        frame``、既定 ``r_eef_grasp_link`` -> ``base_link``) を毎回引き、
        まだ引けなければ (ロボット未接続・/aero_state_publisher 未起動
        など) ``_FALLBACK_ROBOT_HAND_POSITION`` に概算値でフォールバック
        する (TF 解決自体は ``run_camera_pipeline_test.py`` と共通の
        ``ros_camera_utils.lookup_frame_position`` を使う。フォールバック
        値だけは、こちらは概算の固定値、``run_camera_pipeline_test.py`` は
        右腕の「種の姿勢」から計算した値と別々に決めている)。
        """
        if self.args.robot_hand_position is not None:
            return np.asarray(self.args.robot_hand_position, dtype=np.float64)
        return lookup_frame_position(
            self.tf_buffer, self.args.base_frame, self.args.robot_hand_frame,
            _FALLBACK_ROBOT_HAND_POSITION,
            warn_label='[record-palm-offer-clips] ')

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _on_frame(self, color_msg, depth_msg, info_msg):
        t = rospy.Time.now().to_sec()
        with self._lock:
            self._buffer_append(self.args.color_topic, t, color_msg)
            self._buffer_append(self.args.depth_topic, t, depth_msg)
            self._buffer_append(self.args.camera_info_topic, t, info_msg)

        transform = lookup_camera_to_base(
            self.tf_buffer, self.args.base_frame, color_msg.header)
        camera_to_base = (None if transform is None
                          else transform_to_matrix(transform.transform))

        offered_hand = None
        color = None
        person_joints_2d = None
        if camera_to_base is not None:
            color = imgmsg_to_ndarray(color_msg, desired_encoding='bgr8')
            depth_raw = imgmsg_to_ndarray(depth_msg)
            depth_m = PeoplePoseEstimator.depth_to_meters(
                depth_raw, encoding=depth_msg.encoding)
            intrinsics = CameraIntrinsics.from_matrix(info_msg.K)
            people, joints_2d = self.pose_estimator.estimate_3d(
                color, depth_m, intrinsics, output_transform=camera_to_base)
            joint_positions = people[0] if people else None
            person_joints_2d = joints_2d[0] if joints_2d else None
            if joint_positions is not None:
                # 深度の単発の外れ値を時間方向に抑えてから掌推定に渡す
                # (run_camera_pipeline_test.py の self._joint_smoother と
                # 同じ、One Euro Filter はフレーム数ではなく実時間に基づく
                # ためカメラ画像のタイムスタンプを渡す)。
                joint_positions = self._joint_smoother.update(
                    joint_positions, t=color_msg.header.stamp.to_sec())
                self.palm_estimator.offered_hand_selector.robot_position = \
                    self._resolve_robot_position()
                palms = self.palm_estimator.estimate(joint_positions)
                offered_hand = palms['offered_hand']
                # run_camera_pipeline_test.py の ARMED 中と同じスコア内訳を
                # スロットルして標準出力に出す (このノードには viser 画面が
                # 無く、常時検出なので ARMED という区切りも無いため、閾値の
                # 調整にはこれが唯一の手がかりになる。PalmPoseEstimator.
                # estimate は offered_hand しか返さないので、同じ入力で
                # select() を呼び直す)。
                selection = self.palm_estimator.offered_hand_selector.select(
                    joint_positions, palms)
                rospy.loginfo_throttle(
                    1.0, '[record-palm-offer-clips] %s',
                    epp.format_offer_scores(
                        selection,
                        self.palm_estimator.offered_hand_selector.score_min))

        with self._lock:
            triggered = self.tracker.observe_detection(t, offered_hand)
            if triggered:
                # トリガーしたフレーム自身の画像・2D 関節位置を、差し出しを
                # 検出した瞬間のスナップショット (PNG) として使う。
                self._start_clip(self.tracker.trigger_time,
                                 self.tracker.trigger_side,
                                 color, person_joints_2d)
            self._write_if_recording(self.args.color_topic, t, color_msg)
            self._write_if_recording(self.args.depth_topic, t, depth_msg)
            self._write_if_recording(self.args.camera_info_topic, t, info_msg)
            self._maybe_close_clip(t)
            in_cooldown_now = self.tracker.in_cooldown(t)

        # 購読者がいれば、骨格・掌の有無やクリップを録画中かどうかに関わら
        # ず毎フレーム publish する (クールダウン中だけ止める)。
        if (not in_cooldown_now and color is not None
               and self.skeleton_image_pub.get_num_connections() > 0):
            overlay = (draw_skeleton_overlay(color, person_joints_2d,
                                             offered_side=offered_hand)
                      if person_joints_2d else color)
            self.skeleton_image_pub.publish(
                ndarray_to_imgmsg(overlay, 'bgr8', color_msg.header))

    def _on_tf(self, msg):
        self._on_tf_message(_TF_TOPIC, msg)

    def _on_tf_static(self, msg):
        # スライディングウィンドウ (_buffer_append) には乗せず、子フレーム
        # 名をキーに最新の変換を保持し続ける (_write_latest_tf_static
        # 参照)。録画中ならそのまま生でも書き込んでおく (実害はない)。
        t = rospy.Time.now().to_sec()
        with self._lock:
            for tr in msg.transforms:
                self._tf_static_transforms[tr.child_frame_id] = tr
            self._write_if_recording(_TF_STATIC_TOPIC, t, msg)
            self._maybe_close_clip(t)

    def _on_tf_message(self, topic, msg):
        t = rospy.Time.now().to_sec()
        with self._lock:
            self._buffer_append(topic, t, msg)
            self._write_if_recording(topic, t, msg)
            self._maybe_close_clip(t)

    def spin(self):
        rospy.spin()
        with self._lock:
            if self._active_bag is not None:
                self._active_bag.close()
        self.pose_estimator.close()


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
        help='tf2 バッファの保持時間 [秒] (既定 30.0、run_camera_pipeline_'
            'test.py の --tf-cache-time と同じ理由)。')
    # --- PeoplePoseEstimator (run_camera_pipeline_test.py と同じ既定値) ---
    parser.add_argument('--min-detection-confidence', type=float, default=0.5)
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5)
    parser.add_argument('--min-visibility', type=float, default=0.5)
    parser.add_argument('--min-joints', type=int, default=6)
    parser.add_argument('--max-z-diff', type=float, default=1.0)
    parser.add_argument('--min-body-size', type=float, default=0.3)
    parser.add_argument('--max-body-size', type=float, default=2.5)
    parser.add_argument('--max-limb-length', type=float, default=0.7)
    parser.add_argument('--max-hand-segment-length', type=float, default=0.12)
    parser.add_argument('--max-hand-reach', type=float, default=0.22)
    parser.add_argument('--depth-patch-size', type=int, default=3)
    # --- 関節位置の時間方向平滑化 (aero_demo.skeleton_filters.
    # OneEuroFilter、run_camera_pipeline_test.py と同じクラス・同じ既定値) ---
    parser.add_argument(
        '--joint-smoothing-mincutoff', type=float, default=0.5,
        help='One Euro Filter の最小カットオフ周波数 [Hz] (既定 0.5、'
            'run_camera_pipeline_test.py の既定値と揃えてある)。下げるほど '
            '静止時のジッタが減るが追従が遅れる。')
    parser.add_argument(
        '--joint-smoothing-beta', type=float, default=0.3,
        help='One Euro Filter の速度依存カットオフの係数 (既定 0.3、'
            'run_camera_pipeline_test.py の既定値と揃えてある)。上げるほど '
            '速い動きへの追従の遅れが減るが静止時のジッタが増える。')
    parser.add_argument(
        '--joint-smoothing-dcutoff', type=float, default=1.0,
        help='One Euro Filter の速度推定のカットオフ周波数 [Hz] (既定 1.0、'
            'run_camera_pipeline_test.py の既定値と揃えてある)。')
    # --- 差し出し手判定 (estimate_palm_poses.OfferedHandSelector) ---
    parser.add_argument(
        '--offer-score-min', type=float, default=0.65,
        help='差し出し手と判定するスコアの閾値 (既定 0.65、'
            'run_camera_pipeline_test.py の既定値と揃えてある)。')
    parser.add_argument(
        '--robot-hand-position', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'Z'),
        help='差し出し手判定が基準にするロボット手先の base_link 座標 '
            '[m] を固定値で指定する (既定 None)。指定すると --robot-hand-'
            'frame での TF 解決より優先される。OfferedHandSelector 自体の '
            '既定動作 (robot_position=None のときに人物の位置から world '
            '+x に 3m・高さ 1.2m の点を使う) は合成骨格向けで、人物が常に '
            'ロボットより +x 側にいることを前提にしている。実カメラは '
            'base_link 座標系で推定するためロボット自身がおよそ原点付近 '
            '(=人物より -x 側) にいることが多く前提と食い違い、差し出して '
            'いない手が高スコアになる/差し出した手が高スコアにならない '
            '原因になるため、この既定 (None) のときはその代わりに '
            '--robot-hand-frame の TF を使う (下記参照)。')
    parser.add_argument(
        '--robot-hand-frame', type=str, default='r_eef_grasp_link',
        help='--robot-hand-position が未指定のとき、差し出し手判定の基準に '
            '毎フレーム TF (--base-frame からのこのフレーム) を引いて使う '
            '(既定 r_eef_grasp_link -- skrobot Aero モデルの rarm_end_'
            'coords に対応する実リンクで、実機では /aero_state_publisher '
            'が配信する)。ロボット未接続などでまだ TF が引けない間だけ '
            '概算値 ({}) にフォールバックする。'.format(
                _FALLBACK_ROBOT_HAND_POSITION))
    parser.add_argument(
        '--max-person-distance', type=float, default=4.2,
        help='人物 (腰の中点) からロボット手先までの距離 [m] がこれを '
            '超えたら、スコアを見るまでもなく両手とも差し出し候補から '
            '外す (既定 4.2、run_camera_pipeline_test.py の既定値と揃えて '
            'ある)。奥や画面の端に映り込んだだけの、手を差し出す気の無い '
            '通行人を拾わないための足切り (estimate_palm_poses.'
            'OfferedHandSelector の max_distance 引数、veto 理由は '
            '"too_far")。0 以下を指定すると足切りを無効にする。')
    # --- クリップ切り出し ---
    parser.add_argument(
        '--pre-seconds', type=float, default=2.0,
        help='検出時刻の何秒前からクリップに含めるか (既定 2.0)。')
    parser.add_argument(
        '--post-seconds', type=float, default=2.0,
        help='検出時刻の何秒後までクリップに含めるか (既定 2.0)。')
    parser.add_argument(
        '--cooldown-seconds', type=float, default=3.0,
        help='1 つのクリップを保存し終えてから、次のトリガーを受け付ける '
            'までの最短間隔 [秒] (既定 3.0)。同じ差し出し動作を複数回に '
            '分けて録らないようにする。')
    parser.add_argument(
        '--save-dir', type=str, default='palm_offer_clips',
        help='クリップ (.bag) とメタデータ (.json) の保存先ディレクトリ '
            '(既定 palm_offer_clips/)。')
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    rospy.init_node('record_palm_offer_clips')
    node = PalmOfferClipRecorder(args)
    node.spin()


if __name__ == '__main__':
    main()
