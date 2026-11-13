#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""頭のカメラの画像から人の体の向き (ヨー角) を求めて publish するノード.

MediaPipe (aero_demo.people_pose_estimator) で骨格を求め、腰・肩・顔の
関節から体の正面の向きを出す (aero_demo.body_yaw)。手つなぎ歩行で台車を
人と同じ向きへ回す制御 (aero-hand-open の aero_hand_magnetic_cmd_vel_node)
が使う。

画像の遅れの間に台車が回っても向きがずれないよう、向きは撮影時刻の
TF で ``~odom_frame`` 座標系に直して出す (odom が引けなければ base_link)。
受け取る側が最新の TF で base_link に戻す。

Publish
-------
~body_pose (geometry_msgs/PoseStamped)
    位置は腰の中点 (無ければ肩の中点)、向きは体の正面のヨー角。
    信頼度が ``~min_confidence`` 以上のときだけ出す。stamp は画像の時刻。
~debug (std_msgs/String)
    部位ごとの向き [deg, base_link]・使った部位・信頼度の JSON。
~skeleton_image (sensor_msgs/Image)
    骨格を重ねた画像 (購読者がいるときだけ)。

    rosrun aero_demo human_body_yaw_node.py
"""

import json
import math
import os
import sys

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
        self.weights = {
            'hip': rospy.get_param('~weight_hip', 0.5),
            'shoulder': rospy.get_param('~weight_shoulder', 0.3),
            'face': rospy.get_param('~weight_face', 0.2),
        }
        self.smoother = body_yaw.YawSmoother(
            time_constant=rospy.get_param('~smoothing_time', 0.3),
            reset_after=rospy.get_param('~smoothing_reset', 1.0))
        self._smoother_frame = None

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
            min_visibility=rospy.get_param('~min_visibility', 0.5))

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
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=2, slop=0.1)
        self.sync.registerCallback(self._on_frame)
        rospy.on_shutdown(self.estimator.close)
        rospy.loginfo('human_body_yaw: %s + %s -> ~body_pose (%s)',
                      color_topic, depth_topic, self.odom_frame)

    def _on_camera_info(self, msg):
        self._camera_info = msg

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

    def _on_frame(self, color_msg, depth_msg):
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
        scores = {j['limb']: j['score'] for j in joints_2d[0]
                  if j['score'] >= 0}
        result = body_yaw.estimate_body_yaw(
            people[0], scores, weights=self.weights,
            face_max_deviation=self.face_max_deviation)
        if result is None:
            self._publish_debug(None, None)
            return

        stamp = color_msg.header.stamp
        odom_from_base = self._lookup_odom_from_base(stamp)
        if odom_from_base is not None:
            frame = self.odom_frame
            yaw = body_yaw.wrap_angle(result.yaw
                                      + _yaw_of_matrix(odom_from_base))
            center = odom_from_base.dot(np.append(result.center, 1.0))[:3]
        else:
            rospy.logwarn_throttle(
                5.0, 'human_body_yaw: %s -> %s の TF が無いので %s で出します',
                self.base_frame, self.odom_frame, self.base_frame)
            frame = self.base_frame
            yaw = result.yaw
            center = result.center
        if frame != self._smoother_frame:
            self.smoother.reset()
            self._smoother_frame = frame
        smoothed = self.smoother.update(yaw, stamp.to_sec(),
                                        result.confidence)
        self._publish_debug(result, smoothed)
        if result.confidence < self.min_confidence:
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

    def _publish_debug(self, result, smoothed):
        if self.debug_pub.get_num_connections() == 0:
            return
        if result is None:
            data = {'valid': False}
        else:
            data = {
                'valid': True,
                'yaw_base_deg': round(math.degrees(result.yaw), 1),
                'smoothed_deg': round(math.degrees(smoothed), 1),
                'frame': self._smoother_frame,
                'confidence': round(result.confidence, 3),
                'used': result.used,
                'parts': {name: {'yaw_base_deg': round(math.degrees(y), 1),
                                 'confidence': round(c, 3)}
                          for name, (y, c) in result.parts.items()},
            }
        self.debug_pub.publish(String(data=json.dumps(data)))


if __name__ == '__main__':
    rospy.init_node('human_body_yaw')
    HumanBodyYawNode()
    rospy.spin()
