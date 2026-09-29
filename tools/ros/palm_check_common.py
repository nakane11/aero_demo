#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""掌の認識のずれを調べる検証スクリプト (``compare_hand_detectors.py``/
``measure_palm_viewpoint.py``) の共通部分。

``run_camera_pipeline_test.py`` は、計画時の掌を Holistic (``estimate_3d``、
体を検出してから手を切り出す) + One Euro Filter で、押し込み直前の再認識の
掌を Hands (``estimate_hands_3d``、手を直接検出) で求めている。掌の平面
フィット (``PalmPoseEstimator.estimate_palm``) と深度の取り方
(``_sample_depth``) は同じだが、入力のランドマークの出どころが違う。ここ
ではその両方を同じフレームに対して計算する (``PalmDetector.detect``)。
"""

import math
import os
import sys
import time

import numpy as np

import rospy
import message_filters
import tf2_ros
from sensor_msgs.msg import CameraInfo, Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_THIS_DIR)
_REPO_ROOT = os.path.dirname(_TOOLS_DIR)
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, 'scripts')
_PKG_SRC_DIR = os.path.join(_REPO_ROOT, 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import skeleton_filters  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, lookup_camera_to_base, transform_to_matrix)

import estimate_palm_poses as epp  # noqa: E402

REPO_ROOT = _REPO_ROOT

# 検出器の名前 (記録・集計のキー)。holistic_smoothed は計画時と同じ
# (Holistic の骨格に One Euro Filter をかけてから掌を求める)。
SOURCES = ('holistic', 'holistic_smoothed', 'hands')


def add_camera_args(parser):
    """カメラ・骨格推定の引数 (``run_camera_pipeline_test.py`` と同じ既定値)。"""
    parser.add_argument('--color-topic', type=str,
                        default='/camera/color/image_raw/decompressed')
    parser.add_argument('--depth-topic', type=str,
                        default='/camera/depth/image_raw/decompressed')
    parser.add_argument('--camera-info-topic', type=str,
                        default='/camera/color/camera_info')
    parser.add_argument('--base-frame', type=str, default='base_link')
    parser.add_argument('--tf-cache-time', type=float, default=30.0)
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


class FrameSource(object):
    """カラー・深度・camera_info を同期して受け、要求されたときに最新の
    1 フレームを (base_link への変換つきで) 返す。重い推定はコールバック
    ではなく呼び出し側のスレッドで行う。"""

    def __init__(self, args):
        self.args = args
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=rospy.Duration(args.tf_cache_time))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._latest = None
        color_sub = message_filters.Subscriber(args.color_topic, Image)
        depth_sub = message_filters.Subscriber(args.depth_topic, Image)
        info_sub = message_filters.Subscriber(
            args.camera_info_topic, CameraInfo)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub, info_sub], queue_size=5, slop=0.1)
        self.sync.registerCallback(self._on_frame)

    def _on_frame(self, color_msg, depth_msg, info_msg):
        # 受信時刻は PC の時計で持つ (ロボットと PC の時計がずれていても
        # 「呼び出し以降に届いたフレーム」を判定できるように)。
        self._latest = (time.time(), color_msg, depth_msg, info_msg)

    def get_frame(self, after, timeout=2.0):
        """``after`` (PC の時刻) より後に届いたフレームを待って返す。届か
        ない・TF が引けなければ ``None``。"""
        deadline = time.time() + timeout
        latest = None
        while not rospy.is_shutdown() and time.time() < deadline:
            latest = self._latest
            if latest is not None and latest[0] > after:
                break
            rospy.sleep(0.005)
        else:
            return None
        received, color_msg, depth_msg, info_msg = latest
        transform = lookup_camera_to_base(
            self.tf_buffer, self.args.base_frame, color_msg.header)
        if transform is None:
            return None
        return dict(
            received=received,
            stamp=color_msg.header.stamp.to_sec(),
            color=imgmsg_to_ndarray(color_msg, desired_encoding='bgr8'),
            depth=PeoplePoseEstimator.depth_to_meters(
                imgmsg_to_ndarray(depth_msg), encoding=depth_msg.encoding),
            intrinsics=CameraIntrinsics.from_matrix(info_msg.K),
            camera_to_base=transform_to_matrix(transform.transform))


class PalmDetector(object):
    """1 フレームから、Holistic (生/平滑化) と Hands の両方で掌を求める。"""

    def __init__(self, args):
        self.estimator = PeoplePoseEstimator(
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
        self.smoother = skeleton_filters.OneEuroFilter(
            mincutoff=args.joint_smoothing_mincutoff,
            beta=args.joint_smoothing_beta,
            dcutoff=args.joint_smoothing_dcutoff)
        self.palm_estimator = epp.PalmPoseEstimator()
        # Hands のモデル読み込みを先に済ませておく
        # (run_camera_pipeline_test.py と同じ)。
        self.estimator.estimate_hands_3d(
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.zeros((240, 320), dtype=np.float32),
            CameraIntrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0))

    def detect(self, frame, side):
        """``side`` ('R'/'L') の掌を 3 通りで求める (座標系は base_link)。

        Returns
        -------
        dict
            ``holistic``/``holistic_smoothed`` (palm dict or None)、
            ``hands`` (検出した手ごとの ``label``/``score``/``n_points``/
            ``palm``/``pixels`` のリスト、掌は ``side`` として求める --
            Hands の左右判定は誤りうるので ``_select_offered_hand`` と
            同じくラベルは使わない)、``holistic_pixels`` (Holistic の
            ``side`` の手の 2D 点)、``camera_position`` (base_link 系)。
        """
        color, depth = frame['color'], frame['depth']
        intr, cam = frame['intrinsics'], frame['camera_to_base']
        people, joints_2d = self.estimator.estimate_3d(
            color, depth, intr, output_transform=cam)
        raw = people[0] if people else None
        smoothed = (None if raw is None else self.smoother.update(
            raw, t=frame['stamp'], frame_key=True))
        holistic = (None if raw is None
                    else self.palm_estimator.estimate_palm(raw, side))
        holistic_smoothed = (
            None if smoothed is None
            else self.palm_estimator.estimate_palm(smoothed, side))
        prefix = '{}Hand'.format(side)
        holistic_pixels = {}
        if joints_2d:
            holistic_pixels = {j['limb']: [j['x'], j['y']]
                               for j in joints_2d[0]
                               if j['limb'].startswith(prefix)
                               and j['score'] >= 0}

        hands = []
        for hand in self.estimator.estimate_hands_3d(
                color, depth, intr, output_transform=cam):
            label_prefix = '{}Hand'.format(hand['side'])
            joints = {'{}{}'.format(prefix, name[len(label_prefix):]): p
                      for name, p in hand['positions'].items()}
            hands.append(dict(
                label=hand['side'], score=hand['score'],
                n_points=len(hand['positions']),
                palm=self.palm_estimator.estimate_palm(joints, side),
                pixels=hand['pixels']))
        return dict(holistic=holistic, holistic_smoothed=holistic_smoothed,
                    hands=hands, holistic_pixels=holistic_pixels,
                    camera_position=cam[:3, 3].copy())


def pick_hand(hands, reference, max_distance):
    """``hands`` のうち掌が ``reference`` に最も近いもの (``max_distance``
    より遠ければ ``None``)。"""
    best, best_dist = None, None
    for hand in hands:
        if hand['palm'] is None:
            continue
        dist = float(np.linalg.norm(
            np.asarray(hand['palm']['position']) - reference))
        if dist <= max_distance and (best_dist is None or dist < best_dist):
            best, best_dist = hand, dist
    return best


def rotation_angle_deg(rot_a, rot_b):
    """2 つの回転行列のなす角 [deg]。"""
    rel = np.asarray(rot_a).T @ np.asarray(rot_b)
    return math.degrees(math.acos(
        np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))


def normal_angle_deg(palm_a, palm_b):
    """掌の法線 (``y_axis``) のなす角 [deg]。"""
    return math.degrees(math.acos(np.clip(
        np.dot(palm_a['y_axis'], palm_b['y_axis']), -1.0, 1.0)))


def decompose(diff, camera_position, point):
    """ずれ ``diff`` を、カメラ -> ``point`` の視線方向の成分 (``along_ray``、
    正ならカメラから遠い側)・視線に直交する成分の大きさ (``across_ray``)・
    水平 (``horizontal``)・鉛直 (``vertical``、z) に分ける。"""
    diff = np.asarray(diff, dtype=np.float64)
    ray = np.asarray(point, dtype=np.float64) - np.asarray(camera_position)
    ray = ray / np.linalg.norm(ray)
    along = float(np.dot(diff, ray))
    return dict(along_ray=along,
                across_ray=float(np.linalg.norm(diff - along * ray)),
                horizontal=float(np.hypot(diff[0], diff[1])),
                vertical=float(diff[2]))


def stats(values):
    """平均・標準偏差・中央値・最大絶対値 (空なら ``None``)。"""
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if values.size == 0:
        return None
    return dict(n=int(values.size), mean=float(values.mean()),
                std=float(values.std()), median=float(np.median(values)),
                max_abs=float(np.abs(values).max()))


def fmt_stats_mm(s):
    if s is None:
        return '-'
    return '{:+7.1f} ±{:5.1f} (中央 {:+7.1f}, 最大|{:5.1f}|)'.format(
        s['mean'] * 1e3, s['std'] * 1e3, s['median'] * 1e3,
        s['max_abs'] * 1e3)


def fmt_xyz_mm(xyz):
    return '({:+7.1f}, {:+7.1f}, {:+7.1f})mm'.format(
        *[float(v) * 1e3 for v in xyz])


def palm_record(palm):
    """palm dict を JSON に書ける形 (位置・3 軸) にする。"""
    if palm is None:
        return None
    return dict(position=[float(v) for v in palm['position']],
                x_axis=[float(v) for v in palm['x_axis']],
                y_axis=[float(v) for v in palm['y_axis']])


def draw_overlay(color, detection, chosen_hand, path):
    """Holistic の手 (赤) と、採用した Hands の手 (緑) の 2D 点を重ねた
    画像を保存する。"""
    import cv2
    img = color.copy()
    for u, v in detection['holistic_pixels'].values():
        cv2.circle(img, (int(u), int(v)), 4, (0, 0, 255), -1)
    if chosen_hand is not None:
        for u, v in chosen_hand['pixels'].values():
            cv2.circle(img, (int(u), int(v)), 3, (0, 255, 0), 1)
    cv2.putText(img, 'red: Holistic  green: Hands', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(path, img)
