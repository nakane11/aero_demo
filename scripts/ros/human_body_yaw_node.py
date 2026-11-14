#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""頭のカメラの画像から人の体の向き (ヨー角) を求めて publish するノード.

MediaPipe (aero_demo.people_pose_estimator) で骨格と人の領域を求め、
体の正面の向きを出す (aero_demo.body_yaw)。``~method`` で選ぶ。

- ``torso_band`` (既定): 手前の肩の高さの帯にある人の点群の主軸を肩の線
  とし、つないでいる手 (``~hand_frame``) の側を人の左とする。反対側の肩が
  隠れていても、背中側から見ても使える。主軸の 長軸/短軸 が
  ``~min_band_ratio`` 未満のときは向きを出さない。
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
    位置は torso_band なら帯の点の重心、shoulders なら肩の中点 (腰を使う
    ときは腰の中点)。向きは体の正面のヨー角。信頼度が ``~min_confidence``
    以上のときだけ出す。stamp は画像の時刻。
~debug (std_msgs/String)
    向き [deg, base_link]・ならした向き・信頼度の JSON。torso_band は
    帯の比 (ratio)・点の数・手前の肩、shoulders は部位ごとの向きと
    180° 折り返し後の向き (folded_base_deg)。
~skeleton_image (sensor_msgs/Image)
    骨格を重ねた画像 (購読者がいるときだけ)。

    rosrun aero_demo human_body_yaw_node.py
"""

import json
import math
import os
import sys
import threading

import cv2
import numpy as np

import message_filters
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

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
        self.method = rospy.get_param('~method', 'torso_band')
        if self.method not in ('torso_band', 'shoulders'):
            raise ValueError('~method は torso_band か shoulders: %s'
                             % self.method)
        # 人の左手とつないでいるロボットの手のリンク。
        self.hand_frame = rospy.get_param('~hand_frame', 'r_hand_link')
        self.min_band_ratio = rospy.get_param('~min_band_ratio', 1.8)
        self.band_stride = int(rospy.get_param('~band_stride', 4))
        self.band = (rospy.get_param('~band_low', -0.20),
                     rospy.get_param('~band_high', 0.03))
        self.band_max_from_shoulder = rospy.get_param(
            '~band_max_from_shoulder', 0.45)

        color_topic = rospy.get_param(
            '~color_topic', '/camera/color/image_raw/decompressed')
        depth_topic = rospy.get_param(
            '~depth_topic', '/camera/depth/image_raw/decompressed')
        info_topic = rospy.get_param(
            '~camera_info_topic', '/camera/color/camera_info')

        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(rospy.get_param('~tf_cache_time', 10.0)))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 手のランドマークは使わないので軽い Pose だけにする。
        self.estimator = PeoplePoseEstimator(
            use_hand=False,
            min_visibility=rospy.get_param('~min_visibility', 0.5),
            enable_segmentation=(self.method == 'torso_band'))

        self.pose_pub = rospy.Publisher('~body_pose', PoseStamped,
                                        queue_size=1)
        self.debug_pub = rospy.Publisher('~debug', String, queue_size=1)
        self.image_pub = rospy.Publisher('~skeleton_image', Image,
                                         queue_size=1)

        self._camera_info = None
        rospy.Subscriber(info_topic, CameraInfo, self._on_camera_info,
                         queue_size=1)
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
        # 終了時に処理中の画像があると閉じた推定器を使ってしまうので、
        # 推定と終了処理を同じロックで分ける。
        self._lock = threading.Lock()
        self._closed = False
        self.sync.registerCallback(self._on_frame)
        rospy.on_shutdown(self._close)
        rospy.loginfo('human_body_yaw: %s + %s -> ~body_pose (%s)',
                      color_topic, depth_topic, self.odom_frame)

    def _on_camera_info(self, msg):
        self._camera_info = msg

    def _close(self):
        with self._lock:
            self._closed = True
            self.estimator.close()

    def _on_frame(self, color_msg, depth_msg):
        with self._lock:
            if not self._closed:
                self._process_frame(color_msg, depth_msg)

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
        transform = lookup_camera_to_base(self.tf_buffer, self.base_frame,
                                          color_msg.header)
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
        if self.method == 'torso_band':
            found = self._estimate_torso_band(
                people[0], depth, info, camera_to_base, stamp)
        else:
            found = self._estimate_shoulders(people[0], joints_2d[0])
        if found is None:
            return
        yaw_base, confidence, center_base, debug = found
        if yaw_base is None:
            self._publish_debug(debug)
            return

        odom_from_base = self._lookup_odom_from_base(stamp)
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
                             stamp):
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
            max_from_shoulder=self.band_max_from_shoulder)
        debug = {'method': 'torso_band', 'near': near_name}
        if result is None:
            debug.update(valid=False, reason='few band points')
            return None, 0.0, None, debug
        debug.update(yaw_base_deg=round(math.degrees(result.yaw), 1),
                     ratio=round(result.ratio, 2), count=result.count)
        if result.ratio < self.min_band_ratio:
            debug.update(valid=False, reason='low ratio')
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
