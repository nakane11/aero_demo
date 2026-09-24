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

Usage
-----
    python3 scripts/ros/print_palm_positions.py
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
from sensor_msgs.msg import CameraInfo, Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
_PKG_SRC_DIR = os.path.join(_SCRIPTS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

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
              'ターミナルに base_link 座標系での掌位置が print され続けます '
              '(--print-interval={:.1f} 秒間隔)。'.format(
                  self.args.print_interval))
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
