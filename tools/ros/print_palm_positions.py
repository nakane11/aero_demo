#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""カメラから掌推定だけを常時行い、掌位置 (base_link) を print する検証用スクリプト。

差し出し手については IK 目標 (``palm_target_position``) とロボット現在手先
(``--robot-hand-frame`` の TF) も出す。viser に骨格・ロボット・掌の Axis を
表示し、骨格を ``skeleton_markers`` (MarkerArray, frame_id=``--base-frame``)
に publish する (rviz は ``launch/view_skeleton.launch``)。

Usage
-----
    python3 tools/ros/print_palm_positions.py
"""

import argparse
import os
import sys
import threading
import time

import numpy as np

import rospy
import message_filters
import tf2_ros
from geometry_msgs.msg import Point
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, 'scripts')
_PKG_SRC_DIR = os.path.join(_REPO_ROOT, 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import palm_plane_view  # noqa: E402
from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo import skeleton_filters  # noqa: E402
from aero_demo import viewer_nav  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, lookup_camera_to_base, lookup_frame_position,
    transform_to_matrix)
from aero_demo.aero_urdf_setup import load_aero  # noqa: E402

import estimate_palm_poses as epp  # noqa: E402
import solve_palm_ik as spik  # noqa: E402
from handshake_viewer_common import set_link_visible  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

# 掌の Axis の大きさ [m]。
PALM_AXIS_LENGTH = 0.1
PALM_AXIS_RADIUS = 0.005

# 検出が途切れてもこの秒数は直前の骨格を表示し続ける (ちらつき防止)。
SKELETON_HOLD_TIMEOUT = 1.0


def _arms_down_pose(robot):
    """両腕を下ろした表示用の姿勢 (jaxls 依存を避けて自前で作る)。"""
    robot.reset_pose()
    for side in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(side)).joint_angle(0.0)


class PrintPalmPositionsNode(object):
    """骨格推定 -> 掌推定を繰り返し掌位置を print するノード (実機は動かさない)。"""

    def __init__(self, args):
        self.args = args
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

        # TF が引けないときの手先位置 (右腕の種の姿勢)。
        robot_for_fallback = spik.Aero(use_hand=False)
        spik.seed_arm_pose(robot_for_fallback, 'r')
        self._robot_hand_position_fallback = np.asarray(
            robot_for_fallback.rarm_end_coords.worldpos(), dtype=np.float64)

        self.offered_hand_selector = epp.OfferedHandSelector(
            robot_position=self._robot_hand_position_fallback,
            score_min=args.offer_score_min,
            max_distance=(None if args.max_person_distance <= 0
                         else args.max_person_distance))
        self.palm_estimator = epp.PalmPoseEstimator(self.offered_hand_selector)

        self.skeleton_marker_pub = rospy.Publisher(
            'skeleton_markers', MarkerArray, queue_size=1)

        self._viewer_lock = threading.Lock()
        self._skeleton_links = []
        self._palm_axes = {'R': None, 'L': None}
        self._last_detected_joint_positions = None
        self._last_detected_time = None
        self._last_print_time = 0.0

        self.display_robot = load_aero(use_hand=True)
        _arms_down_pose(self.display_robot)

        self.viewer = ViserViewer(draw_grid=True)
        self.viewer.add(self.display_robot)
        for side in ('R', 'L'):
            axis = Axis(axis_length=PALM_AXIS_LENGTH,
                       axis_radius=PALM_AXIS_RADIUS)
            self._palm_axes[side] = axis
            self.viewer.add(axis)
        self.viewer.show(open_browser=not args.no_open_browser)
        if not args.no_wait_for_client:
            viewer_nav.wait_for_client(self.viewer, args.client_wait_timeout)

        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        info_sub = message_filters.Subscriber(
            args.camera_info_topic, CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_frame)

    def _lookup_camera_to_base(self, header):
        return lookup_camera_to_base(
            self.tf_buffer, self.args.base_frame, header)

    def _resolve_robot_hand_position(self):
        return lookup_frame_position(
            self.tf_buffer, self.args.base_frame, self.args.robot_hand_frame,
            self._robot_hand_position_fallback,
            warn_label='[print-palm-positions] ')

    # ------------------------------------------------------------------
    # camera callback
    # ------------------------------------------------------------------
    def _on_frame(self, color_msg, depth_msg, info_msg):
        transform = self._lookup_camera_to_base(color_msg.header)
        if transform is None:
            return
        camera_to_base = transform_to_matrix(transform.transform)

        color = imgmsg_to_ndarray(color_msg, desired_encoding='bgr8')
        depth_raw = imgmsg_to_ndarray(depth_msg)
        depth_m = PeoplePoseEstimator.depth_to_meters(
            depth_raw, encoding=depth_msg.encoding)
        intrinsics = CameraIntrinsics.from_matrix(info_msg.K)

        people, _joints_2d = self.pose_estimator.estimate_3d(
            color, depth_m, intrinsics, output_transform=camera_to_base)
        raw_joint_positions = people[0] if people else None
        joint_positions = (
            None if raw_joint_positions is None
            else self._joint_smoother.update(
                raw_joint_positions, t=color_msg.header.stamp.to_sec(),
                frame_key=True))

        now = time.time()
        if joint_positions is not None:
            self._last_detected_joint_positions = joint_positions
            self._last_detected_time = now
            display_joint_positions = joint_positions
        elif (self._last_detected_time is not None
              and now - self._last_detected_time < SKELETON_HOLD_TIMEOUT):
            display_joint_positions = self._last_detected_joint_positions
        else:
            display_joint_positions = None

        self._update_skeleton_view(display_joint_positions)
        self._publish_skeleton_markers(display_joint_positions, color_msg.header)

        if joint_positions is None:
            return

        robot_hand_position = self._resolve_robot_hand_position()
        self.offered_hand_selector.robot_position = robot_hand_position
        palms = self.palm_estimator.estimate(
            joint_positions, t=color_msg.header.stamp.to_sec())
        self._update_palm_axes(palms)
        self._print_palms(palms, robot_hand_position)

    def _update_skeleton_view(self, joint_positions):
        with self._viewer_lock:
            for link in self._skeleton_links:
                self.viewer.delete(link)
            self._skeleton_links = (
                [] if joint_positions is None
                else skeleton_drawing.build_skeleton_links(joint_positions))
            for link in self._skeleton_links:
                self.viewer.add(link)
            self.viewer.redraw()

    def _publish_skeleton_markers(self, joint_positions, header):
        """骨格を MarkerArray で publish する。

        frame_id は ``--base-frame`` (骨格は変換済み)、stamp はカラー画像の時刻。
        未検出時は DELETEALL のみ。
        """
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.args.base_frame
        delete_marker.header.stamp = header.stamp
        delete_marker.ns = 'skeleton'
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        if joint_positions:
            positions = skeleton_drawing.fill_missing_wrist_from_hand(
                joint_positions)

            bone_marker = Marker()
            bone_marker.header.frame_id = self.args.base_frame
            bone_marker.header.stamp = header.stamp
            bone_marker.ns = 'skeleton'
            bone_marker.id = 1
            bone_marker.type = Marker.LINE_LIST
            bone_marker.action = Marker.ADD
            bone_marker.pose.orientation.w = 1.0
            bone_marker.scale.x = 0.01
            for start_name, end_name in skeleton_drawing.BONE_NAME_PAIRS:
                if start_name not in positions or end_name not in positions:
                    continue
                rgba_255 = palm_plane_view.bone_color(
                    '{}->{}'.format(start_name, end_name))
                rgba = ColorRGBA(r=rgba_255[0] / 255.0, g=rgba_255[1] / 255.0,
                                 b=rgba_255[2] / 255.0, a=rgba_255[3] / 255.0)
                for name in (start_name, end_name):
                    p = positions[name]
                    bone_marker.points.append(
                        Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
                    bone_marker.colors.append(rgba)
            marker_array.markers.append(bone_marker)

            joint_marker = Marker()
            joint_marker.header.frame_id = self.args.base_frame
            joint_marker.header.stamp = header.stamp
            joint_marker.ns = 'skeleton'
            joint_marker.id = 2
            joint_marker.type = Marker.SPHERE_LIST
            joint_marker.action = Marker.ADD
            joint_marker.pose.orientation.w = 1.0
            joint_marker.scale.x = 0.02
            joint_marker.scale.y = 0.02
            joint_marker.scale.z = 0.02
            joint_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            for p in positions.values():
                joint_marker.points.append(
                    Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            marker_array.markers.append(joint_marker)

        self.skeleton_marker_pub.publish(marker_array)

    def _update_palm_axes(self, palms):
        with self._viewer_lock:
            for side in ('R', 'L'):
                palm = palms[side]
                axis = self._palm_axes[side]
                if palm is None:
                    set_link_visible(self.viewer, axis, False)
                    continue
                coords = Coordinates(
                    pos=palm['position'], rot=np.asarray(palm['rot']))
                axis.newcoords(coords)
                set_link_visible(self.viewer, axis, True)
            self.viewer.redraw()

    def _print_palms(self, palms, robot_hand_position):
        """掌位置と差し出し手の IK 目標を ``--print-interval`` 秒ごとに print する。"""
        if palms['R'] is None and palms['L'] is None:
            return
        now = time.time()
        if now - self._last_print_time < self.args.print_interval:
            return
        self._last_print_time = now

        offered_hand = palms['offered_hand']
        lines = ['--- {} ---'.format(
            time.strftime('%H:%M:%S', time.localtime(now)))]
        for side in ('R', 'L'):
            palm = palms[side]
            if palm is None:
                continue
            marker = ' <- offered_hand' if side == offered_hand else ''
            lines.append('  {}: 掌位置 (base_link) = {}{}'.format(
                side, _fmt_xyz(palm['position']), marker))
        if offered_hand is not None:
            target = spik.palm_target_position(palms[offered_hand])
            lines.append('  IK目標 (掌 + ホバーオフセット) = {}'.format(
                _fmt_xyz(target)))
            diff_z = target[2] - robot_hand_position[2]
            lines.append(
                '  ロボット現在手先 (base_link, {}) = {} '
                '(IK目標との高さ差 z: {:+.3f} m)'.format(
                    self.args.robot_hand_frame,
                    _fmt_xyz(robot_hand_position), diff_z))
        else:
            lines.append('  ロボット現在手先 (base_link, {}) = {}'.format(
                self.args.robot_hand_frame, _fmt_xyz(robot_hand_position)))
        print('\n'.join(lines))

    def spin(self):
        print('viser のブラウザ画面で骨格・ロボット・掌の位置を確認しつつ、'
              'rviz では ~skeleton_markers (MarkerArray, frame_id={}) '
              'で骨格を確認できます。ターミナルには base_link 座標系での'
              '掌位置が print され続けます '
              '(--print-interval={:.1f} 秒間隔)。'.format(
                  self.args.base_frame, self.args.print_interval))
        rospy.spin()


def _fmt_xyz(xyz):
    return '[{:+.3f}, {:+.3f}, {:+.3f}]'.format(
        float(xyz[0]), float(xyz[1]), float(xyz[2]))


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
    parser.add_argument(
        '--client-wait-timeout', type=float, default=30.0,
        help='viser クライアント接続を待つ 1 回あたりの秒数。')
    parser.add_argument('--no-open-browser', action='store_true')
    parser.add_argument('--no-wait-for-client', action='store_true')
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
    parser.add_argument('--joint-smoothing-mincutoff', type=float, default=0.5)
    parser.add_argument('--joint-smoothing-beta', type=float, default=0.3)
    parser.add_argument('--joint-smoothing-dcutoff', type=float, default=1.0)
    parser.add_argument('--offer-score-min', type=float, default=0.65)
    parser.add_argument('--max-person-distance', type=float, default=3.0)
    parser.add_argument(
        '--robot-hand-frame', type=str, default='r_eef_grasp_link',
        help='ロボット現在手先の TF フレーム。')
    parser.add_argument(
        '--print-interval', type=float, default=0.5,
        help='print 間隔 [s]。')
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    rospy.init_node('print_palm_positions')
    node = PrintPalmPositionsNode(args)
    node.spin()


if __name__ == '__main__':
    main()
