#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""ROS non-dependent people pose estimation with MediaPipe.

people_pose_estimation_mediapipe.py の推定処理を、ROS の publish/subscribe から
切り離してクラス化したもの。画像 (numpy BGR) を渡すと素の Python
オブジェクトで結果が返る。
"""

import logging
import math
import time

import cv2
import numpy as np
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')  # Prevent GUI issues
import matplotlib.cm

# 2D 関節の型は偽推定 (scripts/ros/fake_people_pose_estimator_ros.py) と共有する
from aero_demo.people_pose_types import CameraIntrinsics
from aero_demo.people_pose_types import HAND_SEQUENCE, INDEX2HANDNAME
from aero_demo.people_pose_types import INDEX2LIMBNAME, LIMB_SEQUENCE

__all__ = ['CameraIntrinsics', 'PeoplePoseEstimator']

logger = logging.getLogger(__name__)


class PeoplePoseEstimator(object):
    """MediaPipe による人物姿勢推定 (ROS 非依存).

    Examples
    --------
    >>> estimator = PeoplePoseEstimator(use_hand=False)
    >>> joints = estimator.estimate(bgr_img)                  # 2D
    >>> people, joints = estimator.estimate_3d(bgr_img, depth_m, intr)  # 3D
    >>> people[0]  # {'Neck': [x, y, z], 'RShoulder': [x, y, z], ...}
    >>> vis = estimator.draw_joints(bgr_img.copy(), joints)
    >>> estimator.close()
    """

    limb_sequence = LIMB_SEQUENCE
    index2limbname = INDEX2LIMBNAME
    index2handname = INDEX2HANDNAME
    hand_sequence = HAND_SEQUENCE

    # 腕・脚の「近位 -> 遠位」の関節ペア。深度が単発でおかしくなったとき
    # 症状が出やすいのは、体幹から離れた末端側の関節 (肘/手首、膝/足首) が
    # 輪郭付近の細い部位で背景の深度を拾ってしまう場合 (腕が不自然に伸びて
    # 見える現象、_prune_implausible_limbs 参照)。近位側 (肩/腰) は胴体に
    # 近く輪郭も太いため相対的に信頼できるという前提で、超過時は常に遠位
    # 側を落とす。同じ考え方を手のランドマークにも適用したものが
    # ``hand_sequence`` (``HAND_SEQUENCE``、既に手首 -> 各指先の近位 ->
    # 遠位順で並んでいるのでそのまま使える、``_prune_implausible_hand_
    # landmarks`` 参照)。
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
                 use_hand=False,
                 model_complexity=0,
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
                 min_neck_height=0.8,
                 max_neck_height=2.0,
                 enable_neck_height_filter=False,
                 camera_to_base_transform=None,
                 history_duration=1.0,
                 history_distance=1.0):
        """
        Parameters
        ----------
        use_hand : bool
            True なら Holistic を使い手のランドマークも推定する。
        model_complexity : int
            Pose モデルの複雑さ。0 が最速、1 が既定、2 が最も高精度。
        min_visibility : float
            この値以下の visibility の関節は score=-1 として無効化する。
        min_joints : int
            3 次元姿勢としてこれ未満の関節数しか取れなければ棄却する。
        max_z_diff : float
            関節の奥行きのばらつきがこれを超えたら人でないとみなす [m]。
        min_body_size : float
            検出できた関節のバウンディングボックス対角線長がこれ未満なら
            (物の一部を人物と誤検出した等) 人でないとみなす [m]。
        max_body_size : float
            検出できた関節のバウンディングボックス対角線長がこれを超えたら
            (深度ノイズで関節が実際より大きく散らばった等) 人でないとみなす
            [m]。
        max_limb_length : float
            肩-肘/肘-手首/腰-膝/膝-足首の各区間がこれを超えて長ければ、
            遠位側の関節 (肘/手首/膝/足首) を検出できなかった扱いにして
            捨てる [m] (``_prune_implausible_limbs`` 参照)。深度が単発で
            背景側に飛んで腕や脚が不自然に伸びて見える現象への対策。
        max_hand_segment_length : float
            手のランドマーク (手首 - 各指の関節間、``HAND_SEQUENCE`` の
            各区間) がこれを超えて長ければ、遠位側のランドマークを検出
            できなかった扱いにして捨てる [m] (``_prune_implausible_hand_
            landmarks`` 参照)。指は輪郭が細く深度パッチが背景を拾い
            やすいため、``max_limb_length`` と同じ考え方で手専用に別の
            閾値を用意してある (実際の指の区間はどれも数 cm 程度なので、
            既定の 0.12m は明らかにおかしい跳びだけを弾く緩めの値)。
            ただしこれは隣接する関節同士の距離しか見ないため、各区間が
            閾値ギリギリで同じ方向に連鎖すると手首から指先までの累積では
            大きく伸びうる (実際の指の骨は 1 区間あたり数 cm 程度)。
            ``max_hand_reach`` はこれを補うためのもの。
        max_hand_reach : float
            手首 (``{side}Hand0``) から各指のランドマークまでの直線距離が
            これを超えたら、``max_hand_segment_length`` と同様に遠位側の
            ランドマークを検出できなかった扱いにして捨てる [m]。成人の
            手首-指先の最大到達距離はおよそ 0.18-0.20m 程度なので、既定の
            0.22m は多少の余裕を見た値 (``_prune_implausible_hand_
            landmarks`` 参照)。指が実際より大きく開いたブーケ状/花火状に
            見える現象 (区間ごとの伸びが連鎖して蓄積したもの) への対策。
        max_hand_wrist_offset : float
            Pose モデルの手首 (RWrist/LWrist) と Hand モデルの手首
            (RHand0/LHand0) は別々に検出された 2D 点であり、素の 3D 化
            処理では両者を一致させる制約が無い。速い動きやオクルージョン
            などでこの 2 点の距離がこれを超えたら、掌側の推定 (RHand0
            など) が信頼できないとみなしてその手のランドマーク全体を
            検出できなかった扱いにして捨てる [m] (``_prune_implausible_
            hand_wrist_offset`` 参照)。これを入れないと、掌の当たり判定
            (掌ランドマークから平面フィット) と前腕の当たり判定 (RElbow-
            RWrist の円柱) が大きく離れて見える。
        depth_patch_size : int
            関節の深度を取るときに参照する近傍の一辺 [px]。奇数。3 なら
            3x3 の有効画素の中央値を使う。1 にすると 1 画素だけを見る
            (従来の挙動)。
        enable_neck_height_filter : bool
            True かつ camera_to_base_transform が与えられている場合のみ、
            首の高さによるフィルタを行う。
        camera_to_base_transform : numpy.ndarray or callable or None
            カメラ座標系の点を基準座標系へ変換するもの。4x4 の同次変換行列か、
            (x, y, z) を受け取って (x, y, z) を返す callable を渡す。
            TF の代わりに呼び出し側から与える。
        history_duration : float
            直近の検出位置を人物として信頼し続ける秒数。
        history_distance : float
            直近の検出位置とみなす距離 [m]。
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
        self.min_neck_height = min_neck_height
        self.max_neck_height = max_neck_height
        self.enable_neck_height_filter = enable_neck_height_filter
        self.camera_to_base_transform = camera_to_base_transform
        self.history_duration = history_duration
        self.history_distance = history_distance
        self.recent_human_positions = []

        if self.enable_neck_height_filter and self.camera_to_base_transform is None:
            logger.warning(
                "enable_neck_height_filter is True but camera_to_base_transform "
                "is not given; the neck height filter is disabled.")
            self.enable_neck_height_filter = False

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
                model_complexity=model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )

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
                    people_joint_positions=None, output_transform=None):
        """深度画像とカメラ内部パラメータから 3 次元姿勢を求める.

        Parameters
        ----------
        depth_img : numpy.ndarray
            メートル単位の深度画像。単位が異なる場合は ``depth_to_meters`` で
            変換してから渡す。
        intrinsics : CameraIntrinsics
            カメラ内部パラメータ。``CameraIntrinsics.from_matrix(K)`` でも作れる。
        people_joint_positions : list or None
            ``estimate`` の結果を再利用したい場合に渡す。None なら内部で推定する。
        output_transform : numpy.ndarray or callable or None
            None ならカメラ座標系のまま返す。4x4 の同次変換行列か
            (x, y, z) -> (x, y, z) の callable を渡すと、返す関節点を
            その座標系 (base_link など) へ変換してから返す。
            フィルタ自体はカメラ座標系のまま行う (``max_z_diff`` は奥行きの
            ばらつきを見る指標なので、高さ方向を含む座標系では意味が変わる)。

        Returns
        -------
        (list of dict, list of list of dict)
            フィルタを通過した人物ごとの 3 次元関節位置 (関節名 ->
            [x, y, z] の dict, 検出できた関節だけを持つ -- ``estimate_
            palm_poses.py``/``solve_palm_ik.py`` が読む骨格 JSON の
            ``skeleton.joint_positions`` と同じ形) のリストと、描画などに
            使う 2 次元関節位置。
        """
        if people_joint_positions is None:
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

    def _sample_depth(self, depth_img, u, v):
        """(u, v) の近傍から有効な深度の中央値を返す (無ければ None).

        深度画像は関節の輪郭付近で欠測 (0) や外れ値が出やすいので、
        1 画素ではなく ``depth_patch_size`` 四方の有効画素だけを見る。
        """
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
        """深度エラーで腕/脚が不自然に伸びて見える関節を取り除く.

        ``_LIMB_CHAINS`` の各区間 (近位 -> 遠位) の長さが ``max_limb_length``
        を超えたら、遠位側の関節を検出できなかった扱いにして落とす (肘が
        既に落ちていれば、それに連なる手首の区間はそもそも判定できず自然に
        素通りする)。
        """
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
        """深度エラーで指のランドマークが一瞬だけ全く違う場所に飛ぶ現象を
        取り除く (``_prune_implausible_limbs`` の手版).

        指は輪郭が細く、``_sample_depth`` のパッチが背景側の画素を拾い
        やすい。``hand_sequence`` (``HAND_SEQUENCE``、手首から各指先まで
        近位 -> 遠位の順に並んだ関節ペア) の各区間の長さが ``max_hand_
        segment_length`` を超えたら、遠位側のランドマークを検出できなかった
        扱いにして落とす。あわせて、手首 (``{prefix}0``) から各ランド
        マークまでの直線距離が ``max_hand_reach`` を超えた場合も同様に
        落とす。区間ごとの判定だけでは、各区間が ``max_hand_segment_
        length`` ギリギリで同じ方向に連鎖したときに、手首-指先の累積では
        大きく伸びてしまう (実際の指の骨は 1 区間あたり数 cm 程度しかない
        ため、指全体が花束状/花火状に開いて見える現象になる) のを防げない
        ための追加チェック。

        腕/脚 (``_prune_implausible_limbs``) はチェーンが 2 区間 (肩-肘-
        手首) しかないため「親が落ちれば子は自然に判定されず残る」で
        十分だったが、指は 4 関節 (MCP-PIP-DIP-tip) と深く、実際のデータ
        では指全体が背景の深度をまとめて拾って一塊で浮く現象がよく起きる
        (根元の MCP だけ手首から離れて浮き、それより先の PIP/PIP 間の
        距離自体は正常に見えてしまうため、素通り版では先が残ってしまう)。
        そのため、この関数の中でこの呼び出し中に自分で落とした関節は
        ``removed`` に記録しておき、その関節を親とする区間は距離を見ずに
        連鎖的に落とす。use_hand=False のときは RHand*/LHand* が存在しない
        ため何もしない。
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
        """Pose モデルの手首と Hand モデルの手首が食い違う手を丸ごと捨てる.

        ``RWrist``/``LWrist`` (body-pose モデル) と ``RHand0``/``LHand0``
        (hand モデル、クロップした手の領域に対して独立に推定される) は
        同じ物理的な手首を指すはずだが、別々のモデル出力の 2D 点をそれぞれ
        独立に深度サンプリングして 3D 化しているため (``_to_joint_
        positions``)、両者を一致させる制約が存在しない。速い動きや
        オクルージョンで乖離すると、掌のランドマーク (``RHand0`` 等から
        平面フィットする掌の当たり判定) が前腕 (``RElbow``-``RWrist`` の
        円柱) から大きく離れて見える。ここで両手首間の距離が
        ``max_hand_wrist_offset`` を超えたら、掌側のランドマーク
        (``RHand*``/``LHand*``) を検出できなかった扱いにして全て捨てる
        (手首だけずらして残りの指を維持しても、平面フィットの基準点が
        誤っているままなので意味がない)。
        """
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

        if self.enable_neck_height_filter:
            neck_height = self._transform_to_base(neck_pos)
            if neck_height is None:
                return True  # 変換できないときは棄却しない
            neck_height = neck_height[2]
            if neck_height < self.min_neck_height \
                    or neck_height > self.max_neck_height:
                logger.warning(
                    "Pose rejected by neck height filter: height=%.2fm "
                    "(limits: %sm - %sm)",
                    neck_height, self.min_neck_height, self.max_neck_height)
                return False
        return True

    @staticmethod
    def _compute_body_size(positions):
        """検出できた関節のバウンディングボックスの対角線長を返す [m].

        特定の骨 (肩幅など) だけを見ると体の向きによって短く見えることが
        あるため、検出できた全関節の広がりを見る方が「人物ひとり分の
        大きさ」として頑健。関節が 2 点未満なら判定できないので None。
        """
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

    def _transform_to_base(self, point):
        transform = self.camera_to_base_transform
        try:
            if callable(transform):
                return np.asarray(transform(point), dtype=np.float64)
            matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
            homogeneous = np.array([point[0], point[1], point[2], 1.0])
            return matrix.dot(homogeneous)[:3]
        except Exception as e:
            logger.warning("Transform to base frame failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------
    @staticmethod
    def _get_cmap(name='hsv'):
        try:
            return matplotlib.colormaps.get_cmap(name)
        except AttributeError:
            return matplotlib.cm.get_cmap(name)

    def draw_joints(self, img, people_joint_positions):
        """``estimate`` の結果を img へ描き込む (img は破壊的に変更される)."""
        all_peaks = [[] for _ in range(len(self.index2limbname) - 1)]
        for person_joint_positions in people_joint_positions:
            for i in range(len(self.index2limbname) - 1):
                jt = person_joint_positions[i]
                if jt['score'] >= 0:
                    all_peaks[i].append((jt['x'], jt['y']))

        cmap = self._get_cmap('hsv')

        if all_peaks:
            # keypoints
            n = len(self.index2limbname) - 1
            for i in range(len(self.index2limbname) - 1):
                rgba = np.array(cmap(1. * i / n))
                color = rgba[:3] * 255
                for j in range(len(all_peaks[i])):
                    cv2.circle(img, (int(all_peaks[i][j][0]), int(
                        all_peaks[i][j][1])), 4, color, thickness=-1)

        # connections
        stickwidth = 4
        for joint_positions in people_joint_positions:
            n = len(self.limb_sequence)
            for i, conn in enumerate(self.limb_sequence):
                rgba = np.array(cmap(1. * i / n))
                color = rgba[:3] * 255
                j1, j2 = joint_positions[conn[0] - 1], joint_positions[conn[1] - 1]
                if j1['score'] < 0 or j2['score'] < 0:
                    continue
                self._draw_stick(img, j1, j2, color, stickwidth)

        # for hand
        if self.use_hand:
            offset = len(self.limb_sequence)
            for joint_positions in people_joint_positions:
                n = len(joint_positions[offset:])
                for i, jt in enumerate(joint_positions[offset:]):
                    if jt['score'] < 0.0:
                        continue
                    rgba = np.array(cmap(1. * i / n))
                    color = rgba[:3] * 255
                    cv2.circle(img, (int(jt['x']), int(jt['y'])),
                               2, color, thickness=-1)

            for joint_positions in people_joint_positions:
                offset = len(self.limb_sequence)
                n = len(self.hand_sequence)
                for _ in range(2):
                    # for both hands
                    for i, conn in enumerate(self.hand_sequence):
                        rgba = np.array(cmap(1. * i / n))
                        color = rgba[:3] * 255
                        j1 = joint_positions[offset + conn[0]]
                        j2 = joint_positions[offset + conn[1]]
                        if j1['score'] < 0 or j2['score'] < 0:
                            continue
                        self._draw_stick(img, j1, j2, color, stickwidth)
                    #
                    offset += int(len(self.index2handname) / 2)

        return img

    @staticmethod
    def _draw_stick(img, j1, j2, color, stickwidth):
        cx, cy = int((j1['x'] + j2['x']) / 2.), int((j1['y'] + j2['y']) / 2.)
        dx, dy = j1['x'] - j2['x'], j1['y'] - j2['y']
        length = np.linalg.norm([dx, dy])
        angle = int(np.degrees(np.arctan2(dy, dx)))
        polygon = cv2.ellipse2Poly((cx, cy), (int(length / 2.), stickwidth),
                                   angle, 0, 360, 1)
        top = max(0, np.min(polygon[:, 1]))
        left = max(0, np.min(polygon[:, 0]))
        bottom = min(img.shape[0], np.max(polygon[:, 1]))
        right = min(img.shape[1], np.max(polygon[:, 0]))
        if top >= bottom or left >= right:
            return
        roi = img[top:bottom, left:right]
        roi2 = roi.copy()
        cv2.fillConvexPoly(roi2, polygon - np.array([left, top]), color)
        cv2.addWeighted(roi, 0.4, roi2, 0.6, 0.0, dst=roi)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, default=0, help='camera device id')
    parser.add_argument('--no-hand', dest='hand', action='store_false',
                        help='do not estimate hands (estimated by default)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cap = cv2.VideoCapture(args.device)
    with PeoplePoseEstimator(use_hand=args.hand) as estimator:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            people_joint_positions = estimator.estimate(frame)
            cv2.imshow('people_pose_estimator',
                       estimator.draw_joints(frame, people_joint_positions))
            if cv2.waitKey(1) == 27:  # ESC
                break
    cap.release()
    cv2.destroyAllWindows()
