#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""ROS の画像トピックを購読して PeoplePoseEstimator に渡すクラス.

カラー画像 + 深度画像 + CameraInfo が揃っていることを前提とし、
3 次元の関節点を返す。

publish は一切行わない。subscribe した画像を people_pose_estimator の
ROS 非依存クラスへ渡し、結果は

  * ``wait_for_result(timeout)`` … 新しい結果が来るまで待って受け取る
  * ``last_result`` … 最後の結果をその場で参照する

で受け取る。3 次元の関節点は既定で ``~output_frame`` (既定 base_link) 相対
に変換して返す。実際にどの座標系かは ``EstimationResult.frame_id`` を見ること。
"""

import threading

import numpy as np
import rospy
import cv_bridge
import message_filters
import tf
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped

from aero_demo.people_pose_estimator import PeoplePoseEstimator
# EstimationResult は偽推定 (fake_people_pose_estimator_ros.py) と共有する
from aero_demo.people_pose_types import CameraIntrinsics, EstimationResult


class RosPeoplePoseEstimator(object):
    """画像トピックを購読して人物の 3 次元姿勢を推定する (publish なし).

    購読するトピック
      ~input       … カラー画像 (sensor_msgs/Image)
      ~input/depth … 深度画像 (sensor_msgs/Image, 16UC1 か 32FC1)
      ~input/info  … カメラ内部パラメータ (sensor_msgs/CameraInfo)

    Examples
    --------
    >>> rospy.init_node('people_pose_estimator')
    >>> node = RosPeoplePoseEstimator()
    >>> while not rospy.is_shutdown():
    ...     result = node.wait_for_result(timeout=1.0)
    ...     if result is None:
    ...         continue
    ...     for person in result.people:
    ...         # person.positions は result.frame_id (既定 base_link) 相対
    ...         rospy.loginfo('%s', person.position_of('Neck'))
    """

    def __init__(self, estimator=None, subscribe=True):
        """
        Parameters
        ----------
        estimator : PeoplePoseEstimator or None
            使用する推定器。None ならパラメータサーバの値で生成する。
        subscribe : bool
            False なら購読を開始しない。後から ``subscribe()`` を呼ぶ。
        """
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        # 結果の関節点をこのフレームで返す。空文字ならカメラ座標系のまま。
        self.output_frame = rospy.get_param('~output_frame', self.base_frame)
        self.enable_neck_height_filter = rospy.get_param(
            '~enable_neck_height_filter', False)

        self.tf_listener = None
        if self.enable_neck_height_filter or self.output_frame:
            self.tf_listener = tf.TransformListener()

        self.estimator = estimator or PeoplePoseEstimator(
            use_hand=rospy.get_param('~hand/enable', False),
            model_complexity=rospy.get_param('~model_complexity', 0),
            min_detection_confidence=rospy.get_param(
                '~min_detection_confidence', 0.5),
            min_tracking_confidence=rospy.get_param(
                '~min_tracking_confidence', 0.5),
            min_visibility=rospy.get_param('~min_visibility', 0.5),
            min_joints=rospy.get_param('~min_joints', 6),
            max_z_diff=rospy.get_param('~max_z_diff', 1.0),
            depth_patch_size=rospy.get_param('~depth_patch_size', 3),
            min_neck_height=rospy.get_param('~min_neck_height', 0.8),
            max_neck_height=rospy.get_param('~max_neck_height', 2.0),
            enable_neck_height_filter=self.enable_neck_height_filter,
            camera_to_base_transform=self._transform_point_to_base,
        )

        self.bridge = cv_bridge.CvBridge()
        self.subs = []
        self.sub_info = None
        self.camera_info_msg = None
        self._current_frame_id = ''
        self._condition = threading.Condition()
        self._last_result = None

        rospy.on_shutdown(self.clean_up)

        if subscribe:
            self.subscribe()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def clean_up(self):
        self.unsubscribe()
        self.estimator.close()
        with self._condition:
            # shutdown 時に wait_for_result を待たせ続けない
            self._condition.notify_all()

    @property
    def last_result(self):
        """最後に推定した ``EstimationResult`` (未推定なら None)."""
        with self._condition:
            return self._last_result

    def wait_for_result(self, timeout=None):
        """新しい推定結果が来るまで待ち、それを返す.

        呼び出し時点で保持している結果は無視するので、同じフレームが 2 度
        返ることはない。timeout [s] 以内に来なければ None を返す。
        timeout=None なら来るまで待つ (shutdown 時も None が返る)。
        """
        with self._condition:
            current = self._last_result
            if self._condition.wait_for(
                    lambda: self._last_result is not current, timeout):
                return self._last_result
            return None

    # ------------------------------------------------------------------
    # subscription
    # ------------------------------------------------------------------
    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        sub_img = message_filters.Subscriber(
            '~input', Image, queue_size=1, buff_size=2**24)
        sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1, buff_size=2**24)
        self.subs = [sub_img, sub_depth]

        sync_cam_info = rospy.get_param("~sync_camera_info", False)
        if sync_cam_info:
            sub_info = message_filters.Subscriber(
                '~input/info', CameraInfo, queue_size=1, buff_size=2**24)
            self.subs.append(sub_info)
        else:
            self.sub_info = rospy.Subscriber(
                '~input/info', CameraInfo, self._cb_cam_info)

        if rospy.get_param('~approximate_sync', True):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)

        if sync_cam_info:
            sync.registerCallback(self._cb_with_depth_info)
        else:
            self.camera_info_msg = None
            sync.registerCallback(self._cb_with_depth)

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
        self.subs = []
        if self.sub_info is not None:
            self.sub_info.unregister()
            self.sub_info = None

    # ------------------------------------------------------------------
    # callbacks
    # ------------------------------------------------------------------
    def _cb_cam_info(self, msg):
        self.camera_info_msg = msg
        self.sub_info.unregister()
        self.sub_info = None
        rospy.loginfo("Received camera info")

    def _cb_with_depth(self, img_msg, depth_msg):
        if self.camera_info_msg is None:
            return
        self._cb_with_depth_info(img_msg, depth_msg, self.camera_info_msg)

    def _cb_with_depth_info(self, img_msg, depth_msg, camera_info_msg):
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        try:
            depth_img = self.estimator.depth_to_meters(
                depth_img, depth_msg.encoding)
        except ValueError:
            rospy.logerr('Unsupported depth encoding: %s' % depth_msg.encoding)
            return

        # neck height filter の TF 変換で使うフレーム
        self._current_frame_id = img_msg.header.frame_id

        # 出力座標系への変換はフレームごとに 1 回だけ引く
        camera_frame_id = img_msg.header.frame_id
        output_transform = self._lookup_output_transform(
            camera_frame_id, img_msg.header.stamp)
        if output_transform is None:
            output_frame_id = camera_frame_id  # 変換できないのでカメラ座標系のまま
        else:
            output_frame_id = self.output_frame

        intrinsics = CameraIntrinsics.from_matrix(camera_info_msg.K)
        people, people_joint_positions = self.estimator.estimate_3d(
            img, depth_img, intrinsics, output_transform=output_transform)

        # camera_frame_id -> output_frame_id の姿勢。viewer (palm_plane_view)
        # が画角の四角すいを描くのに使う。TF が引けず output_transform が
        # None のとき (people はカメラ座標系のまま) は単位行列でよい。
        camera_pose = output_transform if output_transform is not None \
            else np.eye(4)

        self._set_result(EstimationResult(
            stamp=img_msg.header.stamp,
            frame_id=output_frame_id,
            camera_frame_id=camera_frame_id,
            image=img,
            joint_positions=people_joint_positions,
            people=people,
            camera_intrinsics=intrinsics,
            camera_width=camera_info_msg.width,
            camera_height=camera_info_msg.height,
            camera_pose=camera_pose))

    def _set_result(self, result):
        """結果を保持し、``wait_for_result`` の待ち手を起こす."""
        with self._condition:
            self._last_result = result
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # tf
    # ------------------------------------------------------------------
    def _lookup_output_transform(self, camera_frame_id, stamp):
        """camera_frame_id -> output_frame の 4x4 変換行列を返す.

        画像の時刻の TF を優先し、取れなければ最新 (Time(0)) で代用する。
        どちらも取れなければ None を返し、呼び出し側はカメラ座標系のまま扱う。
        """
        if not self.output_frame or self.output_frame == camera_frame_id:
            return None
        for stamp_to_use in (stamp, rospy.Time(0)):
            try:
                self.tf_listener.waitForTransform(
                    self.output_frame, camera_frame_id, stamp_to_use,
                    rospy.Duration(0.05))
                trans, rot = self.tf_listener.lookupTransform(
                    self.output_frame, camera_frame_id, stamp_to_use)
            except Exception:
                continue
            matrix = tf.transformations.quaternion_matrix(rot)
            matrix[:3, 3] = trans
            return matrix
        rospy.logwarn_throttle(
            5.0, 'TF %s <- %s not available; returning points in the camera '
            'frame' % (self.output_frame, camera_frame_id))
        return None

    def _transform_point_to_base(self, point):
        """カメラ座標系の点を base_frame へ変換する (推定器から呼ばれる).

        例外は推定器側で捕捉され、変換できなかったフレームは棄却されない。
        """
        camera_point = PointStamped()
        camera_point.header.frame_id = self._current_frame_id
        # Use latest available transform instead of exact stamp to avoid blocking
        camera_point.header.stamp = rospy.Time(0)
        camera_point.point.x = float(point[0])
        camera_point.point.y = float(point[1])
        camera_point.point.z = float(point[2])
        self.tf_listener.waitForTransform(
            self.base_frame,
            camera_point.header.frame_id,
            rospy.Time(0),
            rospy.Duration(0.05))
        base_point = self.tf_listener.transformPoint(
            self.base_frame, camera_point)
        return [base_point.point.x, base_point.point.y, base_point.point.z]


if __name__ == '__main__':
    rospy.init_node('people_pose_estimator')

    node = RosPeoplePoseEstimator()
    while not rospy.is_shutdown():
        result = node.wait_for_result(timeout=1.0)
        if result is None:
            rospy.logwarn_throttle(5.0, 'no image received')
            continue
        for person in result.people:
            neck = person.position_of('Neck')
            if neck is None:
                continue
            rospy.loginfo_throttle(
                1.0, 'Neck at (%.2f, %.2f, %.2f) in %s'
                % (neck[0], neck[1], neck[2], result.frame_id))
