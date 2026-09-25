#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``run_camera_pipeline_test.py`` から ARM/RESET/IK/軌道計画を取り除き、
ボタン操作なしでカメラ画像から掌推定だけを常時実行し続け、検出できた人物の
掌の位置 (base_link 座標系) を標準出力に print し続けるだけの、検証用の
最小限のスクリプト。

IK (``solve_palm_ik.solve_person_ik``) が実際に狙う目標位置は掌の位置その
ものではなく ``solve_palm_ik.palm_target_position`` (掌の法線方向に
``TARGET_HOVER_OFFSET`` だけ浮かせた位置) なので、掌の生の位置に加えて
差し出し手についてはこの IK 目標位置も print する。あわせて実機の現在の
手先位置 (``--robot-hand-frame`` の TF) も print するので、
「IK の目標がロボットの現在の手先位置や実際の人間の掌の位置と比べて
どれだけずれているか (特に高さ z)」をこのスクリプトの出力だけで確認できる。

viser 画面には ``run_camera_pipeline_test.py`` と同じ骨格線に加えて、
ロボットモデル (両腕を下ろした初期姿勢, IK では動かさない) と、検出できた
左右の掌の位置に矢印 (Axis) を重ねて表示する。

あわせて骨格 (関節点・ボーン線) を ``visualization_msgs/MarkerArray``
として ``skeleton_markers`` トピック (ノードを名前空間なしで動かす通常
の使い方では ``/skeleton_markers``) に publish するので、rviz からも
確認できる。骨格の座標は ``--base-frame`` (既定 ``base_link``) 座標系
なので、Marker の ``header.frame_id`` は必ず ``--base-frame`` にする
(カメラ座標系のままではない)。``header.stamp`` は骨格推定に使った
カラー画像の timestamp を使う -- これは camera->base の TF 変換に
実際に使った時刻と同じなので、rviz 側の TF 表示ともずれない。

rviz でロボットモデル・``launch/decompress.launch`` が作る点群と一緒に
見るための rviz 設定・launch ファイルは ``rviz/skeleton_demo.rviz``/
``launch/view_skeleton.launch`` を参照 (ロボット本体の bringup
(``robot_description``/``robot_state_publisher``/joint_states の TF) は
aero-ros-pkg 側の ``aero_startup/aero_bringup.launch`` が担っており、実機
操作を伴うためこの launch には含めていない -- 別途起動しておくこと)。

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

# 掌の位置に重ねて表示する矢印 (Axis) の大きさ [m]。掌自体は小さいので、
# ロボットの初期位置マーカー (run_camera_pipeline_test.INITIAL_POSE_AXIS_
# LENGTH = 0.2) より一回り小さくしてある。
PALM_AXIS_LENGTH = 0.1
PALM_AXIS_RADIUS = 0.005

# 骨格が一瞬未検出になるたびに表示を消して描き直すとちらつくので、検出が
# 途切れてもこの秒数の間は直前に検出できた骨格をそのまま表示し続ける
# (run_camera_pipeline_test.SKELETON_HOLD_TIMEOUT と同じ考え方)。
SKELETON_HOLD_TIMEOUT = 1.0


def _arms_down_pose(robot):
    """両腕を体の横に自然に下ろした姿勢にする
    (``plan_handshake_motion.arms_down_angles`` と同じ、viewer 表示専用の
    初期姿勢を作るだけなので jaxls 依存の ``plan_handshake_motion`` 自体は
    import しない)。"""
    robot.reset_pose()
    for side in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(side)).joint_angle(0.0)


class PrintPalmPositionsNode(object):
    """カメラ入力 -> 骨格推定 -> 掌推定を常時繰り返し、掌の base_link 座標を
    print し続けるノード (IK・軌道計画・実機操作は一切行わない)。"""

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

        # 差し出し手判定の基準にするロボット手先位置 (run_camera_pipeline_
        # test.py の _resolve_robot_position と同じ、TF が引けなければ
        # 右腕の「種の姿勢」の手先位置にフォールバックする)。
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

        # rviz で見られるように骨格を publish するトピック
        # (座標系は self.args.base_frame -- 下の _publish_skeleton_markers
        # 参照)。
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
            # base_link 座標系に変換できないフレームは掌推定に使えない
            # (run_camera_pipeline_test.py と同じ)。
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

        # ロボット手先位置は毎フレーム TF から引き直す
        # (run_camera_pipeline_test.py の _resolve_robot_position と同じ)。
        robot_hand_position = self._resolve_robot_hand_position()
        self.offered_hand_selector.robot_position = robot_hand_position
        palms = self.palm_estimator.estimate(joint_positions)
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
        """骨格 (関節点・ボーン線) を ``MarkerArray`` として rviz 向けに
        publish する。

        骨格の座標はカメラ座標系ではなく ``self.args.base_frame``
        (camera_to_base で変換済み) なので、Marker の ``header.frame_id``
        は必ず ``self.args.base_frame`` にする (カメラの frame_id をそ
        のまま使うと rviz 側で位置がずれる)。``header.stamp`` は
        ``camera_to_base`` の TF 変換に実際に使った時刻 (このフレームの
        カラー画像の timestamp) をそのまま使う。

        検出が途切れた (``joint_positions is None``) フレームでは
        ``DELETEALL`` だけを publish して、rviz 側に古い骨格を残さない。
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
        """検出できた左右の掌の base_link 座標と、差し出し手についての IK
        目標位置 (``solve_palm_ik.palm_target_position``) を print する。
        左右とも未推定 (``palm`` が ``None``) のフレームは何も print しない。

        ``--print-interval`` 秒間隔でのみ実際に print する
        (``_on_frame`` はカメラの frame rate のまま呼ばれるため、そのまま
        print すると流れて読めなくなる)。
        """
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
        help='tf2 バッファの保持時間 [秒] (既定 30.0)。')
    parser.add_argument(
        '--client-wait-timeout', type=float, default=30.0,
        help='viser のブラウザクライアント接続を待つ 1 回あたりの秒数 '
            '(繰り返し待つ、既定 30.0)。')
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
    parser.add_argument('--max-person-distance', type=float, default=4.2)
    parser.add_argument(
        '--robot-hand-frame', type=str, default='r_eef_grasp_link',
        help='ロボットの現在の手先位置として TF を引くフレーム (既定 '
            'r_eef_grasp_link、run_camera_pipeline_test.py と同じ)。')
    parser.add_argument(
        '--print-interval', type=float, default=0.5,
        help='掌位置を print する間隔 [秒] (既定 0.5)。')
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    rospy.init_node('print_palm_positions')
    node = PrintPalmPositionsNode(args)
    node.spin()


if __name__ == '__main__':
    main()
