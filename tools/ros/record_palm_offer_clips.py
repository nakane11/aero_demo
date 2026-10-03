#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""カメラと TF を常時ローリング録画し、掌の差し出しを検出したら前後を rosbag に切り出す。

検出時刻の ``--pre-seconds`` 前から ``--post-seconds`` 後までを ``--save-dir``
に保存し、検出瞬間の骨格重畳 PNG とメタデータ JSON も残す。判定基準は
``run_camera_pipeline_test.py`` と同じ (ARM 操作なしで常時検出)。クリップは
/tf・/tf_static を含み、``run_camera_pipeline_test.py --bag`` にそのまま渡せる。
認識されなかった差し出しは集まらないので、見逃しの調査は
``extract_skeletons_from_bag.py`` を使う。

Usage
-----
    python3 tools/ros/record_palm_offer_clips.py
    python3 tools/ros/record_palm_offer_clips.py --save-dir /tmp/palm_offer_clips
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
_TOOLS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, 'scripts')
_PKG_SRC_DIR = os.path.join(_REPO_ROOT, 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import json_io  # noqa: E402
from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo import skeleton_filters  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, lookup_camera_to_base, lookup_frame_position,
    ndarray_to_imgmsg, transform_to_matrix)

import estimate_palm_poses as epp  # noqa: E402

_TF_TOPIC = '/tf'
_TF_STATIC_TOPIC = '/tf_static'

# 手先の TF が引けない間に使う base_link 座標 [m] (右腕初期姿勢の概算)。
# OfferedHandSelector の robot_position=None は合成骨格向けの仮定なので使わない。
_FALLBACK_ROBOT_HAND_POSITION = (0.32, -0.55, 0.93)

class ClipWindowTracker(object):
    """``offered_hand`` の None -> 'R'/'L' を検出時刻 t0 とし、
    ``[t0 - pre_seconds, t0 + post_seconds]`` の録画区間を管理する (ROS 非依存)。
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
        """1 フレームの判定結果を渡し、新しく区間を開始したら True を返す。"""
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
        """クリップを閉じた後のクールダウン中か (録画中は False)。"""
        return (self.window_end is None
               and t - self._last_close_t < self.cooldown_seconds)

    def maybe_close(self, t):
        """区間を過ぎていれば閉じて ``(trigger_time, trigger_side)``、それ以外は None。"""
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

        # ローリングバッファ: トピック名 -> deque[(t, msg)] (t 昇順)。
        self._buffer_seconds = args.pre_seconds + 0.5
        self._buffers = collections.defaultdict(collections.deque)

        # /tf_static は latched で一度しか来ないのでバッファに乗せず、
        # 子フレーム名 -> 最新変換で保持してクリップ開始時に書く。
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

        self._joint_smoother = skeleton_filters.OneEuroFilter(
            mincutoff=args.joint_smoothing_mincutoff,
            beta=args.joint_smoothing_beta,
            dcutoff=args.joint_smoothing_dcutoff)

        # robot_position は毎フレーム _on_frame で設定する。
        max_distance = (None if args.max_person_distance <= 0
                        else args.max_person_distance)
        offered_hand_selector = epp.OfferedHandSelector(
            robot_position=None, score_min=args.offer_score_min,
            max_distance=max_distance)
        self.palm_estimator = epp.PalmPoseEstimator(offered_hand_selector)

        # camera_info は同期に含めず最新の 1 つを使う (画像遅延時に組が
        # できなくなるため)。
        self._latest_camera_info = None
        rospy.Subscriber(args.camera_info_topic, CameraInfo,
                         self._on_camera_info, queue_size=1)
        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_frame)

        rospy.Subscriber(_TF_TOPIC, TFMessage, self._on_tf, queue_size=50)
        rospy.Subscriber(
            _TF_STATIC_TOPIC, TFMessage, self._on_tf_static, queue_size=50)

        # デバッグ用の骨格重畳画像 (クールダウン中以外、購読者がいれば毎フレーム)。
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
        """バッファの ``[t0 - pre_seconds, t0]`` を時刻順にバッグへ書く。"""
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
        """受信済みの全 ``/tf_static`` を 1 メッセージにまとめて書く。"""
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
        overlay = (skeleton_drawing.draw_skeleton_overlay(
            color, joints_2d, offered_side=side) if joints_2d else color)
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
        """判定基準のロボット手先位置 (base_link)。固定値 > TF > フォールバック。"""
        if self.args.robot_hand_position is not None:
            return np.asarray(self.args.robot_hand_position, dtype=np.float64)
        return lookup_frame_position(
            self.tf_buffer, self.args.base_frame, self.args.robot_hand_frame,
            _FALLBACK_ROBOT_HAND_POSITION,
            warn_label='[record-palm-offer-clips] ')

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _on_camera_info(self, msg):
        self._latest_camera_info = msg

    def _on_frame(self, color_msg, depth_msg):
        info_msg = self._latest_camera_info
        if info_msg is None:
            rospy.logwarn_throttle(
                5.0, '[record-palm-offer-clips] {} をまだ受信していないため、'
                '画像を処理しません。'.format(self.args.camera_info_topic))
            return
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
                joint_positions = self._joint_smoother.update(
                    joint_positions, t=color_msg.header.stamp.to_sec())
                self.palm_estimator.offered_hand_selector.robot_position = \
                    self._resolve_robot_position()
                palms = self.palm_estimator.estimate(
                    joint_positions, t=color_msg.header.stamp.to_sec())
                offered_hand = palms['offered_hand']
                # 閾値調整用にスコア内訳を出す (estimate は内訳を返さないので再計算)。
                selection = self.palm_estimator.offered_hand_selector.select(
                    joint_positions, palms, t=color_msg.header.stamp.to_sec())
                rospy.loginfo_throttle(
                    1.0, '[record-palm-offer-clips] %s',
                    epp.format_offer_scores(
                        selection,
                        self.palm_estimator.offered_hand_selector.score_min))

        with self._lock:
            triggered = self.tracker.observe_detection(t, offered_hand)
            if triggered:
                self._start_clip(self.tracker.trigger_time,
                                 self.tracker.trigger_side,
                                 color, person_joints_2d)
            self._write_if_recording(self.args.color_topic, t, color_msg)
            self._write_if_recording(self.args.depth_topic, t, depth_msg)
            self._write_if_recording(self.args.camera_info_topic, t, info_msg)
            self._maybe_close_clip(t)
            in_cooldown_now = self.tracker.in_cooldown(t)

        if (not in_cooldown_now and color is not None
               and self.skeleton_image_pub.get_num_connections() > 0):
            overlay = (skeleton_drawing.draw_skeleton_overlay(
                color, person_joints_2d, offered_side=offered_hand)
                      if person_joints_2d else color)
            self.skeleton_image_pub.publish(
                ndarray_to_imgmsg(overlay, 'bgr8', color_msg.header))

    def _on_tf(self, msg):
        self._on_tf_message(_TF_TOPIC, msg)

    def _on_tf_static(self, msg):
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
        help='tf2 バッファの保持時間 [s]。')
    # --- PeoplePoseEstimator ---
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
    # --- 関節位置の平滑化 (One Euro Filter) ---
    parser.add_argument('--joint-smoothing-mincutoff', type=float, default=0.5,
                        help='最小カットオフ周波数 [Hz]。')
    parser.add_argument('--joint-smoothing-beta', type=float, default=0.3,
                        help='速度依存カットオフの係数。')
    parser.add_argument('--joint-smoothing-dcutoff', type=float, default=1.0,
                        help='速度推定のカットオフ周波数 [Hz]。')
    # --- 差し出し手判定 ---
    parser.add_argument('--offer-score-min', type=float, default=0.65,
                        help='差し出し手と判定するスコアの閾値。')
    parser.add_argument(
        '--robot-hand-position', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'Z'),
        help='判定基準のロボット手先位置 [m] (base_link) を固定する (既定は TF)。')
    parser.add_argument(
        '--robot-hand-frame', type=str, default='r_eef_grasp_link',
        help='判定基準のロボット手先の TF フレーム。')
    parser.add_argument(
        '--max-person-distance', type=float, default=3.0,
        help='人物とロボット手先がこれ [m] より遠ければ候補外 (0 以下で無効)。')
    # --- クリップ切り出し ---
    parser.add_argument('--pre-seconds', type=float, default=2.0,
                        help='検出前に含める秒数。')
    parser.add_argument('--post-seconds', type=float, default=2.0,
                        help='検出後に含める秒数。')
    parser.add_argument('--cooldown-seconds', type=float, default=3.0,
                        help='保存後、次のトリガーを受け付けるまでの秒数。')
    parser.add_argument('--save-dir', type=str, default='palm_offer_clips',
                        help='クリップ (.bag/.json/.png) の保存先。')
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    rospy.init_node('record_palm_offer_clips')
    node = PalmOfferClipRecorder(args)
    node.spin()


if __name__ == '__main__':
    main()
