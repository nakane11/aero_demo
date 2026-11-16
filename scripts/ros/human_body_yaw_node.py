#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""頭のカメラの画像から人の体の向き (ヨー角) を求めて publish するノード.

体の正面の向きを出す (aero_demo.body_yaw)。``~method`` で選ぶ。

- ``depth`` (既定): 深度画像だけを使う。つないでいる手 (``~hand_frame``)
  の近くで手より高い点を人とみなし、その最上部 (頭のてっぺん) から
  ``~head_to_shoulder`` 下を肩の高さとして、その高さの帯の点の主軸を肩の
  線とし、手の側を人の左とする。背中側から近くで見て MediaPipe が人を
  見つけないとき・別の物を人と見誤るときも使える (11/16 の実機)。
- ``torso_band``: 同じ帯の主軸を、MediaPipe (aero_demo.people_pose_estimator)
  の人の領域と手前の肩の関節から取る。
- 上の 2 つは主軸の 長軸/短軸 が ``~min_band_ratio`` 未満のときは向きを
  出さない。人がロボットへ真横・真後ろを向いて左右が曖昧なときは前回の
  向きに近い方を選ぶ。
- ``shoulders``: 両肩の関節の線 (~weight_hip・~weight_face を正にすると
  腰・顔も混ぜる)。背中側からの左右の取り違えは ``~max_relative_yaw``
  で折り返す。

手つなぎ歩行で台車を
人と同じ向きへ回す制御 (aero-hand-open の aero_hand_magnetic_cmd_vel_node)
が使う。

画像の遅れの間に台車が回っても向きがずれないよう、向きは撮影時刻の
TF で ``~odom_frame`` 座標系に直して出す (odom が引けなければ base_link)。
受け取る側が最新の TF で base_link に戻す。

Publish
-------
~body_pose (geometry_msgs/PoseStamped)
    位置は depth・torso_band なら帯の点の重心、shoulders なら肩の中点 (腰を使う
    ときは腰の中点)。向きは体の正面のヨー角。信頼度が ``~min_confidence``
    以上のときだけ出す。stamp は画像の時刻。
~debug (std_msgs/String)
    向き [deg, base_link]・ならした向き・信頼度の JSON。depth・torso_band
    は帯の比 (ratio)・点の数・左右の曖昧さ (hand_cos)、depth は頭の
    てっぺん (top)、torso_band は手前の肩、shoulders は部位ごとの向きと
    180° 折り返し後の向き (folded_base_deg)。
~skeleton_image (sensor_msgs/Image)
    骨格を重ねた画像 (depth 以外で購読者がいるときだけ)。

    rosrun aero_demo human_body_yaw_node.py
"""

import json
import math
import os
import sys
import threading

# numpy の行列計算 (OpenBLAS) のスレッドは計算の合間も回り続け、15Hz で
# 処理すると 2 コアを使い切る (11/16: 213% -> 1 本で 20%)。numpy を
# import する前に決める。
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import message_filters  # noqa: E402
import rospy  # noqa: E402
import tf2_ros  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import body_yaw  # noqa: E402
from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, lookup_camera_to_base, ndarray_to_imgmsg,
    transform_to_matrix)


def _yaw_of_matrix(matrix):
    return math.atan2(matrix[1, 0], matrix[0, 0])


class HumanBodyYawNode(object):

    def __init__(self):
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.min_confidence = rospy.get_param('~min_confidence', 0.2)
        self.face_max_deviation = math.radians(
            rospy.get_param('~face_max_deviation', 45.0))
        # 既定は肩だけ (横に並ぶと腰が画角の外になるため)。
        # 0 にした部位は使わない。
        self.weights = {
            'hip': rospy.get_param('~weight_hip', 0.0),
            'shoulder': rospy.get_param('~weight_shoulder', 1.0),
            'face': rospy.get_param('~weight_face', 0.0),
        }
        self.smoother = body_yaw.YawSmoother(
            time_constant=rospy.get_param('~smoothing_time', 0.3),
            reset_after=rospy.get_param('~smoothing_reset', 1.0))
        self._smoother_frame = None
        # ロボット正面からこの角度 [deg] を超える向きは 180° 折り返す
        # (背中側から見たときの左右の肩の取り違え対策)。180 で無効。
        self.max_relative_yaw = math.radians(
            rospy.get_param('~max_relative_yaw', 90.0))
        self.method = rospy.get_param('~method', 'depth')
        if self.method not in ('depth', 'torso_band', 'shoulders'):
            raise ValueError('~method は depth・torso_band・shoulders: %s'
                             % self.method)
        # 人の左手とつないでいるロボットの手のリンク。
        self.hand_frame = rospy.get_param('~hand_frame', 'r_hand_link')
        self.min_band_ratio = rospy.get_param('~min_band_ratio', 1.8)
        self.band_stride = int(rospy.get_param('~band_stride', 4))
        self.band = (rospy.get_param('~band_low', -0.20),
                     rospy.get_param('~band_high', 0.03))
        self.band_max_from_shoulder = rospy.get_param(
            '~band_max_from_shoulder', 0.45)
        # 主軸と 重心 -> 手 の |cos| がこれ未満 (人がロボットへ真横・真後ろ
        # を向いた ±90° 付近) なら、左右を前回の向きで決める。
        self.band_ambiguous_cos = rospy.get_param('~band_ambiguous_cos', 0.35)
        # 前回の向きがあれば |cos| がこれ未満のうちは前回に近い側を保つ
        # (右に回って背を向けると手が体の後ろ側に回るため)。
        self.band_keep_cos = rospy.get_param(
            '~band_keep_cos', body_yaw.DEFAULT_TORSO_KEEP_COS)
        # 前回の向きが無いときはロボットの正面 (0) を前回の代わりにする
        # (並んで歩き出す人はロボットとほぼ同じ向き。11/16: 真横を向いて
        # 立ち止まった人に左右が決まらず 4 秒反応しなかった)。
        self.front_prior = rospy.get_param('~front_prior', True)
        # depth: 頭のてっぺん (base_link) が x [~blind_x_min, ~blind_x_max]
        # かつ |y| < ~blind_max_y (ロボットのすぐ横) にあるときは向きの差を
        # 0 として出す (回らない)。ここでは頭のカメラから肩が頭に隠れ、人が
        # どちらを向いても帯が同じ形に見える (11/16: 正面向き 0° も後ろ向き
        # -90° も線が +60〜69° と出た)。回りたい人は左回りなら一歩前、
        # 右回りなら一歩下がる (ユーザー)。~blind_max_y を 0 にすると無効。
        self.blind_x = (rospy.get_param('~blind_x_min', -0.10),
                        rospy.get_param('~blind_x_max', 0.10))
        self.blind_max_y = rospy.get_param('~blind_max_y', 0.45)
        # depth: 頭のてっぺんが ~blind_x_min より後ろなら右回り (負) だけ、
        # ~blind_x_max より前なら左回り (正) だけ通し、逆は 0 にする。
        self.step_direction_gate = rospy.get_param(
            '~step_direction_gate', True)
        # depth: 深度画像をこの画素おきに使う (1920x1080 -> 480x270)。
        self.depth_step = max(1, int(rospy.get_param('~depth_step', 4)))
        # depth: 頭のてっぺんから手前の肩までの高さ [m]。
        self.head_to_shoulder = rospy.get_param('~head_to_shoulder', 0.27)
        # depth: 手から水平にこの距離 [m] 以内の点を人とみなす。
        self.person_radius = rospy.get_param(
            '~person_radius', body_yaw.DEFAULT_PERSON_RADIUS)
        self.min_band_points = int(rospy.get_param('~min_band_points', 30))

        color_topic = rospy.get_param(
            '~color_topic', '/camera/color/image_raw/decompressed')
        depth_topic = rospy.get_param(
            '~depth_topic', '/camera/depth/image_raw/decompressed')
        info_topic = rospy.get_param(
            '~camera_info_topic', '/camera/color/camera_info')

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(rospy.get_param('~tf_cache_time', 10.0)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        # 画像は届くのにカメラの TF が引けない状態がこれ [s] 続いたら TF の
        # 購読をつなぎ直す (11/16: ロボット側の TF の publisher が起動し
        # 直されたあと、このノードにだけ TF が届かなくなり向きが出なかった)。
        self.tf_reconnect_after = rospy.get_param('~tf_reconnect_after', 3.0)
        self._tf_ok_wall = rospy.get_time()
        self._tf_reconnect_wall = None

        # 手のランドマークは使わないので軽い Pose だけにする。
        self.estimator = None
        if self.method != 'depth':
            self.estimator = PeoplePoseEstimator(
                use_hand=False,
                min_visibility=rospy.get_param('~min_visibility', 0.5),
                # 横に並ぶと頭と肩しか写らず、背を向けると両肩が重なるので、
                # 関節の範囲の対角線が 0.2〜0.3m になる (推定器の既定 0.3m
                # だと背を向けた場面を人として扱わない)。
                min_body_size=rospy.get_param('~min_body_size', 0.15),
                enable_segmentation=(self.method == 'torso_band'))

        self.pose_pub = rospy.Publisher('~body_pose', PoseStamped,
                                        queue_size=1)
        self.debug_pub = rospy.Publisher('~debug', String, queue_size=1)
        self.image_pub = rospy.Publisher('~skeleton_image', Image,
                                         queue_size=1)

        self._camera_info = None
        rospy.Subscriber(info_topic, CameraInfo, self._on_camera_info,
                         queue_size=1)
        # 終了時に処理中の画像があると閉じた推定器を使ってしまうので、
        # 推定と終了処理を同じロックで分ける。
        self._lock = threading.Lock()
        self._closed = False
        rospy.on_shutdown(self._close)
        if self.method == 'depth':
            # 深度は色の画像と位置合わせ済み (内部パラメータは色のもの)。
            rospy.Subscriber(depth_topic, Image, self._on_depth,
                             queue_size=1, buff_size=2 ** 24)
            rospy.loginfo('human_body_yaw: %s -> ~body_pose (%s)',
                          depth_topic, self.odom_frame)
            return
        # 推定が遅れても古い画像を溜めないよう queue_size=1 にする。
        color_sub = message_filters.Subscriber(
            color_topic, Image, queue_size=1, buff_size=2 ** 24)
        depth_sub = message_filters.Subscriber(
            depth_topic, Image, queue_size=1, buff_size=2 ** 24)
        # 実機では深度が色より平均 0.1 秒 (ばらつき 0.14 秒) 遅れて届く。
        # 同期の待ち行列が短いと組む相手の色画像が先に捨てられ、組がほとんど
        # できない (2 枚 = 30Hz で 66ms 分のとき約 2Hz しか出なかった)。
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub],
            queue_size=rospy.get_param('~sync_queue_size', 10),
            slop=rospy.get_param('~sync_slop', 0.1))
        self.sync.registerCallback(self._on_frame)
        rospy.loginfo('human_body_yaw: %s + %s -> ~body_pose (%s)',
                      color_topic, depth_topic, self.odom_frame)

    def _on_camera_info(self, msg):
        self._camera_info = msg

    def _close(self):
        with self._lock:
            self._closed = True
            if self.estimator is not None:
                self.estimator.close()

    def _on_frame(self, color_msg, depth_msg):
        with self._lock:
            if not self._closed:
                self._process_frame(color_msg, depth_msg)

    def _on_depth(self, depth_msg):
        with self._lock:
            if not self._closed:
                self._process_depth(depth_msg)

    def _lookup_camera(self, header):
        """カメラ -> base_frame の TF. 失敗が続けば TF の購読をつなぎ直す."""
        transform = lookup_camera_to_base(self.tf_buffer, self.base_frame,
                                          header)
        now = rospy.get_time()
        if transform is not None:
            self._tf_ok_wall = now
            return transform
        if (self.tf_reconnect_after > 0
                and now - self._tf_ok_wall > self.tf_reconnect_after
                and (self._tf_reconnect_wall is None
                     or now - self._tf_reconnect_wall > 10.0)):
            rospy.logerr('human_body_yaw: カメラの TF が %.0f 秒引けないので'
                         ' TF の購読をつなぎ直します',
                         now - self._tf_ok_wall)
            self.tf_listener.unregister()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
            self._tf_reconnect_wall = now
        return None

    def _side_prior(self, previous):
        """左右を決める基準の向き: 前回の向き、無ければ (front_prior なら) 0."""
        if previous is None and self.front_prior:
            return 0.0
        return previous

    def _previous_base_yaw(self, stamp, odom_from_base):
        """左右が曖昧なときに使う前回のならした向き (base_link, 無ければ None)."""
        if odom_from_base is None or self._smoother_frame != self.odom_frame:
            return None
        previous = self.smoother.current(stamp.to_sec())
        if previous is None:
            return None
        return body_yaw.wrap_angle(previous - _yaw_of_matrix(odom_from_base))

    def _process_depth(self, depth_msg):
        info = self._camera_info
        if info is None:
            rospy.logwarn_throttle(5.0, 'human_body_yaw: camera_info を待っています')
            return
        transform = self._lookup_camera(depth_msg.header)
        if transform is None:
            return
        camera_to_base = transform_to_matrix(transform.transform)
        stamp = depth_msg.header.stamp
        hand = self._lookup_hand(stamp)
        if hand is None:
            rospy.logwarn_throttle(5.0, 'human_body_yaw: %s の TF がありません',
                                   self.hand_frame)
            return
        depth = PeoplePoseEstimator.depth_to_meters(
            imgmsg_to_ndarray(depth_msg), encoding=depth_msg.encoding)
        step = self.depth_step
        K = np.asarray(info.K, dtype=np.float64).reshape(3, 3).copy()
        if (depth.shape[1], depth.shape[0]) != (info.width, info.height) \
                and info.width > 0 and info.height > 0:
            K[0] *= float(depth.shape[1]) / info.width
            K[1] *= float(depth.shape[0]) / info.height
        small = depth[::step, ::step]
        K[:2] /= step
        points = body_yaw.masked_depth_points(
            small, np.ones(small.shape[:2], dtype=bool), K, camera_to_base,
            stride=1)
        odom_from_base = self._lookup_odom_from_base(stamp)
        previous = self._previous_base_yaw(stamp, odom_from_base)
        debug = {'method': 'depth'}
        head = body_yaw.head_top_from_depth(points, hand,
                                            radius=self.person_radius)
        if head is None:
            debug.update(valid=False, reason='no person')
            self._publish_debug(debug)
            return
        debug['top'] = [round(float(v), 3) for v in head.top]
        if (self.blind_x[0] <= head.top[0] <= self.blind_x[1]
                and abs(head.top[1]) < self.blind_max_y):
            # 向きが読めない領域: ロボットと同じ向きとみなす (次に回る
            # ときの左右の基準にもなる)。
            debug.update(valid=True, confidence=1.0, yaw_base_deg=0.0,
                         blind=True)
            self._publish_result(stamp, odom_from_base,
                                 (0.0, 1.0, head.top, debug))
            return
        near = head.top - np.array([0.0, 0.0, self.head_to_shoulder])
        result = body_yaw.torso_band_yaw(
            head.points, near, hand, band=self.band,
            max_from_shoulder=self.band_max_from_shoulder,
            min_points=self.min_band_points,
            previous=self._side_prior(previous),
            ambiguous_cos=self.band_ambiguous_cos,
            keep_cos=self.band_keep_cos)
        found = self._band_result(result, previous, debug)
        if self.step_direction_gate and found[0] is not None:
            found = self._gate_by_step(found, head.top[0])
        self._publish_result(stamp, odom_from_base, found)

    def _gate_by_step(self, found, top_x):
        """ロボットより後ろの人は右回りだけ、前の人は左回りだけ通す.

        回りたい人は左回りなら一歩前、右回りなら一歩下がる (ユーザー)。
        向きの推定には立ち位置で変わる偏りがあり (11/16 の確認: 正面向き
        のまま半歩下がると +25°)、逆向きに回らないよう 0 にする。
        """
        yaw, confidence, center, debug = found
        gated = yaw
        if top_x < self.blind_x[0]:
            gated = min(yaw, 0.0)
        elif top_x > self.blind_x[1]:
            gated = max(yaw, 0.0)
        if gated != yaw:
            debug.update(step_gated=True, yaw_base_deg=0.0,
                         raw_yaw_base_deg=round(math.degrees(yaw), 1))
        return gated, confidence, center, debug

    def _lookup_odom_from_base(self, stamp):
        for when in (stamp, rospy.Time(0)):
            try:
                t = self.tf_buffer.lookup_transform(
                    self.odom_frame, self.base_frame, when,
                    rospy.Duration(0.05))
                return transform_to_matrix(t.transform)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        return None

    def _process_frame(self, color_msg, depth_msg):
        info = self._camera_info
        if info is None:
            rospy.logwarn_throttle(5.0, 'human_body_yaw: camera_info を待っています')
            return
        transform = self._lookup_camera(color_msg.header)
        if transform is None:
            return
        camera_to_base = transform_to_matrix(transform.transform)

        color = imgmsg_to_ndarray(color_msg, desired_encoding='bgr8')
        depth = PeoplePoseEstimator.depth_to_meters(
            imgmsg_to_ndarray(depth_msg), encoding=depth_msg.encoding)
        people, joints_2d = self.estimator.estimate_3d(
            color, depth, CameraIntrinsics.from_matrix(info.K),
            output_transform=camera_to_base)

        if self.image_pub.get_num_connections() > 0:
            overlay = (skeleton_drawing.draw_skeleton_overlay(
                color, joints_2d[0]) if joints_2d else color)
            self.image_pub.publish(
                ndarray_to_imgmsg(overlay, 'bgr8', color_msg.header))

        if not people:
            return
        stamp = color_msg.header.stamp
        odom_from_base = self._lookup_odom_from_base(stamp)
        if self.method == 'torso_band':
            previous = self._previous_base_yaw(stamp, odom_from_base)
            found = self._estimate_torso_band(
                people[0], depth, info, camera_to_base, stamp, previous)
        else:
            found = self._estimate_shoulders(people[0], joints_2d[0])
        self._publish_result(stamp, odom_from_base, found)

    def _publish_result(self, stamp, odom_from_base, found):
        """(向き, 信頼度, 位置, debug) をならして publish する."""
        if found is None:
            return
        yaw_base, confidence, center_base, debug = found
        if yaw_base is None:
            self._publish_debug(debug)
            return

        if odom_from_base is not None:
            frame = self.odom_frame
            base_yaw = _yaw_of_matrix(odom_from_base)
            center = odom_from_base.dot(np.append(center_base, 1.0))[:3]
        else:
            rospy.logwarn_throttle(
                5.0, 'human_body_yaw: %s -> %s の TF が無いので %s で出します',
                self.base_frame, self.odom_frame, self.base_frame)
            frame = self.base_frame
            base_yaw = 0.0
            center = center_base
        if frame != self._smoother_frame:
            self.smoother.reset()
            self._smoother_frame = frame
        if self.method == 'shoulders':
            # 背中側からの左右の肩の取り違え (180° 逆) を、人はロボットと
            # 逆向きにならない前提で直す。前回の向きは base_link に戻して比べる。
            previous = self.smoother.current(stamp.to_sec())
            if previous is not None:
                previous = body_yaw.wrap_angle(previous - base_yaw)
            yaw_base = body_yaw.fold_to_front(
                yaw_base, self.max_relative_yaw, previous)
            debug['folded_base_deg'] = round(math.degrees(yaw_base), 1)
        yaw = body_yaw.wrap_angle(yaw_base + base_yaw)
        smoothed = self.smoother.update(yaw, stamp.to_sec(), confidence)
        debug['smoothed_deg'] = round(math.degrees(smoothed), 1)
        debug['frame'] = frame
        self._publish_debug(debug)
        if confidence < self.min_confidence:
            return

        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.pose.position.x = float(center[0])
        msg.pose.position.y = float(center[1])
        msg.pose.position.z = float(center[2])
        msg.pose.orientation.z = math.sin(0.5 * smoothed)
        msg.pose.orientation.w = math.cos(0.5 * smoothed)
        self.pose_pub.publish(msg)

    def _lookup_hand(self, stamp):
        for when in (stamp, rospy.Time(0)):
            try:
                t = self.tf_buffer.lookup_transform(
                    self.base_frame, self.hand_frame, when,
                    rospy.Duration(0.05))
                p = t.transform.translation
                return np.array([p.x, p.y, p.z])
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException):
                continue
        return None

    def _estimate_torso_band(self, joints, depth, info, camera_to_base,
                             stamp, previous=None):
        """(向き, 信頼度, 位置, debug) を返す. 向きが出せなければ向きは None."""
        mask = self.estimator.segmentation_mask
        if mask is None:
            return None, 0.0, None, {'valid': False, 'reason': 'no mask'}
        hand = self._lookup_hand(stamp)
        if hand is None:
            rospy.logwarn_throttle(5.0, 'human_body_yaw: %s の TF がありません',
                                   self.hand_frame)
            return None, 0.0, None, {'valid': False, 'reason': 'no hand tf'}
        near_name = body_yaw.nearest_joint(
            joints, ('LShoulder', 'RShoulder'), hand)
        if near_name is None:
            return None, 0.0, None, {'valid': False,
                                     'reason': 'no shoulder'}
        if mask.shape[:2] != depth.shape[:2]:
            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        points = body_yaw.masked_depth_points(
            depth, mask > 0.5, np.asarray(info.K).reshape(3, 3),
            camera_to_base, stride=self.band_stride)
        result = body_yaw.torso_band_yaw(
            points, joints[near_name], hand, band=self.band,
            max_from_shoulder=self.band_max_from_shoulder,
            previous=self._side_prior(previous),
            ambiguous_cos=self.band_ambiguous_cos,
            keep_cos=self.band_keep_cos)
        return self._band_result(
            result, previous, {'method': 'torso_band', 'near': near_name})

    def _band_result(self, result, previous, debug):
        """torso_band_yaw の結果を (向き, 信頼度, 位置, debug) にする."""
        if result is None:
            debug.update(valid=False, reason='few band points')
            return None, 0.0, None, debug
        debug.update(yaw_base_deg=round(math.degrees(result.yaw), 1),
                     ratio=round(result.ratio, 2), count=result.count,
                     hand_cos=round(result.hand_cos, 2),
                     ambiguous=result.ambiguous)
        if result.ratio < self.min_band_ratio:
            debug.update(valid=False, reason='low ratio')
            return None, 0.0, None, debug
        if result.ambiguous and self._side_prior(previous) is None:
            debug.update(valid=False, reason='ambiguous side')
            return None, 0.0, None, debug
        debug.update(valid=True, confidence=1.0)
        return result.yaw, 1.0, result.center, debug

    def _estimate_shoulders(self, joints, joints_2d):
        scores = {j['limb']: j['score'] for j in joints_2d if j['score'] >= 0}
        result = body_yaw.estimate_body_yaw(
            joints, scores, weights=self.weights,
            face_max_deviation=self.face_max_deviation)
        if result is None:
            return None, 0.0, None, {'valid': False}
        debug = {
            'method': 'shoulders',
            'valid': True,
            'yaw_base_deg': round(math.degrees(result.yaw), 1),
            'confidence': round(result.confidence, 3),
            'used': result.used,
            'parts': {name: {'yaw_base_deg': round(math.degrees(y), 1),
                             'confidence': round(c, 3)}
                      for name, (y, c) in result.parts.items()},
        }
        return result.yaw, result.confidence, result.center, debug

    def _publish_debug(self, data):
        if self.debug_pub.get_num_connections() == 0:
            return
        self.debug_pub.publish(String(data=json.dumps(data)))


if __name__ == '__main__':
    rospy.init_node('human_body_yaw')
    HumanBodyYawNode()
    rospy.spin()
