#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""ROS non-dependent people pose estimation with MediaPipe.

BGR 画像 (と深度) から人物の 2D/3D 関節位置を求める。
"""

import logging
import math
import time

import cv2
import numpy as np
import mediapipe as mp

from aero_demo.people_pose_types import CameraIntrinsics
from aero_demo.people_pose_types import HAND_SEQUENCE, INDEX2LIMBNAME

__all__ = ['CameraIntrinsics', 'PeoplePoseEstimator']

logger = logging.getLogger(__name__)

# estimate_hands_3d で検出する手の最大数。
MAX_NUM_HANDS = 2


class PeoplePoseEstimator(object):
    """MediaPipe による人物姿勢推定 (ROS 非依存).

    既定値は run_camera_pipeline_test.py の既定と同じ。

    Examples
    --------
    >>> estimator = PeoplePoseEstimator()
    >>> joints = estimator.estimate(bgr_img)                  # 2D
    >>> people, joints = estimator.estimate_3d(bgr_img, depth_m, intr)  # 3D
    >>> people[0]  # {'Neck': [x, y, z], 'RShoulder': [x, y, z], ...}
    >>> estimator.close()
    """

    index2limbname = INDEX2LIMBNAME
    hand_sequence = HAND_SEQUENCE

    # 腕・脚の (近位, 遠位) 関節ペア。長すぎる区間は遠位側を落とす。
    _LIMB_CHAINS = [
        ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
        ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
        ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
        ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ]

    mp_indices = {
        "Nose": 0,
        "RShoulder": 12,
        "RElbow": 14,
        "RWrist": 16,
        "LShoulder": 11,
        "LElbow": 13,
        "LWrist": 15,
        "RHip": 24,
        "RKnee": 26,
        "RAnkle": 28,
        "LHip": 23,
        "LKnee": 25,
        "LAnkle": 27,
        "REye": 5,
        "LEye": 2,
        "REar": 8,
        "LEar": 7,
    }

    def __init__(self,
                 use_hand=True,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 min_visibility=0.5,
                 min_joints=6,
                 max_z_diff=1.0,
                 min_body_size=0.3,
                 max_body_size=2.5,
                 max_limb_length=0.7,
                 max_hand_segment_length=0.12,
                 max_hand_reach=0.22,
                 max_hand_wrist_offset=0.08,
                 depth_patch_size=3,
                 history_duration=1.0,
                 history_distance=1.0):
        """
        Parameters
        ----------
        use_hand : bool
            True なら Holistic を使い手のランドマークも推定する。
        min_visibility : float
            この値以下の visibility の関節は無効 (score=-1)。
        max_z_diff, min_body_size, max_body_size : float
            人物判定に使う奥行きのばらつき・関節 bbox 対角線長の範囲 [m]。
        max_limb_length, max_hand_segment_length, max_hand_reach : float
            これより長い四肢/指の区間・手首からの距離は遠位側を捨てる [m]。
        max_hand_wrist_offset : float
            Pose の手首と Hand の手首 (Hand0) がこれ以上ずれたら手全体を捨てる [m]。
        depth_patch_size : int
            深度を取る近傍の一辺 [px] (有効画素の中央値)。
        history_duration, history_distance : float
            直近の検出位置付近 [m] を一定時間 [s] 人物として信頼する。
        """
        self.use_hand = use_hand
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.min_visibility = min_visibility
        self.min_joints = min_joints
        self.max_z_diff = max_z_diff
        self.min_body_size = min_body_size
        self.max_body_size = max_body_size
        self.max_limb_length = max_limb_length
        self.max_hand_segment_length = max_hand_segment_length
        self.max_hand_reach = max_hand_reach
        self.max_hand_wrist_offset = max_hand_wrist_offset
        self.depth_patch_size = max(1, int(depth_patch_size))
        self.history_duration = history_duration
        self.history_distance = history_distance
        self.recent_human_positions = []

        # Initialize MediaPipe Solutions
        if self.use_hand:
            self.holistic = mp.solutions.holistic.Holistic(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            self.pose = None
        else:
            self.holistic = None
            self.pose = mp.solutions.pose.Pose(
                model_complexity=0,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        # 手だけの検出 (estimate_hands_3d) は必要になったときに作る。
        self.hands = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def close(self):
        if self.use_hand:
            if self.holistic is not None:
                self.holistic.close()
                self.holistic = None
        else:
            if self.pose is not None:
                self.pose.close()
                self.pose = None
        if self.hands is not None:
            self.hands.close()
            self.hands = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    # ------------------------------------------------------------------
    # 2D estimation
    # ------------------------------------------------------------------
    def estimate(self, bgr_img):
        """BGR 画像から画像座標の関節位置を推定する.

        Returns
        -------
        list of list of dict
            人物ごとに dict(limb=str, x=float, y=float, score=float) のリスト。
            score が負の関節は未検出。
        """
        h, w, _ = bgr_img.shape
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        if self.use_hand:
            results = self.holistic.process(rgb_img)
            pose_landmarks = results.pose_landmarks
            left_hand_landmarks = results.left_hand_landmarks
            right_hand_landmarks = results.right_hand_landmarks
        else:
            results = self.pose.process(rgb_img)
            pose_landmarks = results.pose_landmarks
            left_hand_landmarks = None
            right_hand_landmarks = None

        if not pose_landmarks:
            return []

        people_joint_positions = []
        person_joint_positions = []
        landmarks = pose_landmarks.landmark

        for limb_name in self.index2limbname:
            if limb_name == "Bkg":
                person_joint_positions.append(
                    dict(limb=limb_name, x=0, y=0, score=-1))
            elif limb_name == "Neck":
                l_sh = landmarks[11]
                r_sh = landmarks[12]
                score = min(l_sh.visibility, r_sh.visibility)
                if score > self.min_visibility:
                    person_joint_positions.append(dict(
                        limb=limb_name,
                        x=(l_sh.x + r_sh.x) / 2.0 * w,
                        y=(l_sh.y + r_sh.y) / 2.0 * h,
                        score=score
                    ))
                else:
                    person_joint_positions.append(
                        dict(limb=limb_name, x=0, y=0, score=-1))
            else:
                idx = self.mp_indices[limb_name]
                lm = landmarks[idx]
                if lm.visibility > self.min_visibility:
                    person_joint_positions.append(dict(
                        limb=limb_name,
                        x=lm.x * w,
                        y=lm.y * h,
                        score=lm.visibility
                    ))
                else:
                    person_joint_positions.append(
                        dict(limb=limb_name, x=0, y=0, score=-1))

        if self.use_hand:
            person_joint_positions.extend(
                self._hand_joint_positions("RHand", right_hand_landmarks, w, h))
            person_joint_positions.extend(
                self._hand_joint_positions("LHand", left_hand_landmarks, w, h))

        people_joint_positions.append(person_joint_positions)
        return people_joint_positions

    def _hand_joint_positions(self, prefix, hand_landmarks, w, h):
        joint_positions = []
        if hand_landmarks:
            for idx, lm in enumerate(hand_landmarks.landmark):
                joint_positions.append(dict(
                    limb="{}{}".format(prefix, idx),
                    x=lm.x * w,
                    y=lm.y * h,
                    score=1.0
                ))
        else:
            for idx in range(21):
                joint_positions.append(
                    dict(limb="{}{}".format(prefix, idx), x=0, y=0, score=-1))
        return joint_positions

    # ------------------------------------------------------------------
    # 3D estimation
    # ------------------------------------------------------------------
    @staticmethod
    def depth_to_meters(depth_img, encoding='32FC1'):
        """深度画像をメートル単位の float32 画像へ変換するユーティリティ."""
        if encoding == '16UC1':
            return np.asarray(depth_img, dtype=np.float32) / 1000.0
        elif encoding == '32FC1':
            return np.asarray(depth_img, dtype=np.float32)
        raise ValueError('Unsupported depth encoding: {}'.format(encoding))

    def estimate_3d(self, bgr_img, depth_img, intrinsics,
                    output_transform=None):
        """深度画像 [m] とカメラ内部パラメータから 3 次元姿勢を求める.

        output_transform (4x4 行列か callable) を渡すと返す点をその座標系へ
        変換する。フィルタはカメラ座標系のまま行う。

        Returns
        -------
        (list of dict, list of list of dict)
            人物ごとの {関節名: [x, y, z]} (検出できた関節のみ) と 2D 関節位置。
        """
        people_joint_positions = self.estimate(bgr_img)

        people = []
        current_time = time.time()
        self.recent_human_positions = [
            (t, p) for t, p in self.recent_human_positions
            if current_time - t < self.history_duration]

        for person_joint_positions in people_joint_positions:
            positions = self._to_joint_positions(
                person_joint_positions, depth_img, intrinsics)

            neck_pos = positions.get("Neck", positions.get("Nose"))
            if neck_pos is None:
                continue

            if not self._is_valid_person(positions, neck_pos, current_time):
                continue

            # 履歴はカメラ座標系のまま保持する (フィルタと同じ座標系)
            self.recent_human_positions.append((current_time, neck_pos))
            if output_transform is not None:
                positions = self._apply_transform(positions, output_transform)
            people.append({name: [float(v) for v in p]
                           for name, p in positions.items()})

        return people, people_joint_positions

    def estimate_hands_3d(self, bgr_img, depth_img, intrinsics,
                          output_transform=None):
        """体を使わず MediaPipe Hands で手だけを検出する (体が画角外でも使える).

        Returns
        -------
        list of dict
            手ごとに ``side`` ('R'/'L'、人物自身の左右)、``score``、
            ``positions`` ({side}Hand0..20 -> [x, y, z]、深度が取れた点のみ)、
            ``pixels`` (同じ名前 -> [u, v]、全 21 点)。
        """
        if self.hands is None:
            self.hands = mp.solutions.hands.Hands(
                max_num_hands=MAX_NUM_HANDS,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence)
        h, w, _ = bgr_img.shape
        results = self.hands.process(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return []
        hands = []
        for landmarks, handedness in zip(results.multi_hand_landmarks,
                                         results.multi_handedness):
            classification = handedness.classification[0]
            # handedness は自撮り (左右反転) 前提なので逆にする。
            side = 'R' if classification.label == 'Left' else 'L'
            joints_2d = self._hand_joint_positions(
                '{}Hand'.format(side), landmarks, w, h)
            positions = {}
            for joint_pos in joints_2d:
                if not (0 <= joint_pos['y'] < depth_img.shape[0]
                        and 0 <= joint_pos['x'] < depth_img.shape[1]):
                    continue
                z = self._sample_depth(
                    depth_img, int(joint_pos['x']), int(joint_pos['y']))
                if z is None:
                    continue
                x = (joint_pos['x'] - intrinsics.cx) * z / intrinsics.fx
                y = (joint_pos['y'] - intrinsics.cy) * z / intrinsics.fy
                positions[joint_pos['limb']] = np.array([x, y, z])
            positions = self._prune_implausible_hand_landmarks(positions)
            if output_transform is not None:
                positions = self._apply_transform(positions, output_transform)
            hands.append(dict(
                side=side, score=float(classification.score),
                positions={name: [float(v) for v in p]
                           for name, p in positions.items()},
                pixels={joint_pos['limb']: [float(joint_pos['x']),
                                            float(joint_pos['y'])]
                        for joint_pos in joints_2d}))
        return hands

    def _sample_depth(self, depth_img, u, v):
        """(u, v) 近傍の有効な深度の中央値を返す (無ければ None)."""
        half = self.depth_patch_size // 2
        top = max(0, v - half)
        bottom = min(depth_img.shape[0], v + half + 1)
        left = max(0, u - half)
        right = min(depth_img.shape[1], u + half + 1)
        patch = depth_img[top:bottom, left:right]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _to_joint_positions(self, person_joint_positions, depth_img, intrinsics):
        """検出できた関節だけを持つ ``{limb_name: (3,) ndarray}`` を作る."""
        positions = {}
        for joint_pos in person_joint_positions:
            if joint_pos['score'] < 0:
                continue
            if not (0 <= joint_pos['y'] < depth_img.shape[0]
                    and 0 <= joint_pos['x'] < depth_img.shape[1]):
                continue
            z = self._sample_depth(
                depth_img, int(joint_pos['x']), int(joint_pos['y']))
            if z is None:
                continue
            x = (joint_pos['x'] - intrinsics.cx) * z / intrinsics.fx
            y = (joint_pos['y'] - intrinsics.cy) * z / intrinsics.fy
            positions[joint_pos['limb']] = np.array([x, y, z], dtype=np.float64)
        positions = self._prune_implausible_limbs(positions)
        positions = self._prune_implausible_hand_landmarks(positions)
        positions = self._prune_implausible_hand_wrist_offset(positions)
        return positions

    def _prune_implausible_limbs(self, positions):
        """``max_limb_length`` を超える区間の遠位側の関節を落とす."""
        positions = dict(positions)
        for parent_name, child_name in self._LIMB_CHAINS:
            if parent_name not in positions or child_name not in positions:
                continue
            length = np.linalg.norm(
                positions[child_name] - positions[parent_name])
            if length > self.max_limb_length:
                logger.warning(
                    "Joint %s dropped: distance from %s is %.2fm "
                    "(limit: %sm)", child_name, parent_name, length,
                    self.max_limb_length)
                del positions[child_name]
        return positions

    def _prune_implausible_hand_landmarks(self, positions):
        """指の区間長・手首からの距離が閾値を超えたランドマークを落とす.

        指は根元ごと浮くことがあるので、落とした関節の先は距離を見ずに
        連鎖的に落とす。
        """
        if not self.use_hand:
            return positions
        positions = dict(positions)
        removed = set()
        for prefix in ("RHand", "LHand"):
            wrist_name = "{}0".format(prefix)
            for parent_idx, child_idx in self.hand_sequence:
                parent_name = "{}{}".format(prefix, parent_idx)
                child_name = "{}{}".format(prefix, child_idx)
                if child_name not in positions:
                    continue
                if parent_name in removed:
                    logger.warning(
                        "Joint %s dropped: parent %s was already dropped "
                        "(cascaded)", child_name, parent_name)
                    del positions[child_name]
                    removed.add(child_name)
                    continue
                if parent_name not in positions:
                    continue
                length = np.linalg.norm(
                    positions[child_name] - positions[parent_name])
                reach = (
                    np.linalg.norm(positions[child_name] - positions[wrist_name])
                    if wrist_name in positions else 0.0)
                if length > self.max_hand_segment_length:
                    logger.warning(
                        "Joint %s dropped: distance from %s is %.2fm "
                        "(limit: %sm)", child_name, parent_name, length,
                        self.max_hand_segment_length)
                    del positions[child_name]
                    removed.add(child_name)
                elif reach > self.max_hand_reach:
                    logger.warning(
                        "Joint %s dropped: reach from %s is %.2fm "
                        "(limit: %sm)", child_name, wrist_name, reach,
                        self.max_hand_reach)
                    del positions[child_name]
                    removed.add(child_name)
        return positions

    def _prune_implausible_hand_wrist_offset(self, positions):
        """Pose の手首と Hand0 が ``max_hand_wrist_offset`` 以上ずれた手を丸ごと捨てる."""
        if not self.use_hand:
            return positions
        positions = dict(positions)
        for pose_wrist, hand_prefix in (("RWrist", "RHand"), ("LWrist", "LHand")):
            wrist0 = "{}0".format(hand_prefix)
            if pose_wrist not in positions or wrist0 not in positions:
                continue
            offset = np.linalg.norm(positions[wrist0] - positions[pose_wrist])
            if offset > self.max_hand_wrist_offset:
                logger.warning(
                    "Hand %s dropped: wrist offset from %s is %.2fm "
                    "(limit: %sm)", hand_prefix, pose_wrist, offset,
                    self.max_hand_wrist_offset)
                for name in list(positions.keys()):
                    if name.startswith(hand_prefix):
                        del positions[name]
        return positions

    def _is_valid_person(self, positions, neck_pos, current_time):
        """椅子などの誤検出を弾く."""
        is_valid_by_history = False
        for _, p in self.recent_human_positions:
            dist = math.sqrt((neck_pos[0] - p[0]) ** 2
                             + (neck_pos[1] - p[1]) ** 2
                             + (neck_pos[2] - p[2]) ** 2)
            if dist < self.history_distance:
                is_valid_by_history = True
                break

        if not is_valid_by_history:
            if len(positions) < self.min_joints:
                return False
            z_values = [p[2] for p in positions.values()]
            if z_values and (max(z_values) - min(z_values)) > self.max_z_diff:
                return False
            body_size = self._compute_body_size(positions)
            if body_size is not None and not (
                    self.min_body_size <= body_size <= self.max_body_size):
                logger.warning(
                    "Pose rejected by body size filter: size=%.2fm "
                    "(limits: %sm - %sm)",
                    body_size, self.min_body_size, self.max_body_size)
                return False
        return True

    @staticmethod
    def _compute_body_size(positions):
        """関節のバウンディングボックスの対角線長 [m] (2 点未満なら None)."""
        if len(positions) < 2:
            return None
        pts = np.array(list(positions.values()), dtype=np.float64)
        extent = pts.max(axis=0) - pts.min(axis=0)
        return float(np.linalg.norm(extent))

    @staticmethod
    def _apply_transform(positions, transform):
        """全関節点を transform の座標系へ移した新しい dict を返す."""
        if not positions:
            return positions
        names = list(positions.keys())
        if callable(transform):
            return {name: np.asarray(transform(positions[name]),
                                     dtype=np.float64)[:3]
                   for name in names}
        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
        points = np.array([positions[name] for name in names],
                          dtype=np.float64)   # (N, 3)
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed = homogeneous.dot(matrix.T)[:, :3]
        return {name: transformed[i] for i, name in enumerate(names)}
