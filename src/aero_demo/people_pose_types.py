#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""people pose 推定結果を保持する ROS 非依存のデータ型.

people_pose_estimator.PeoplePoseEstimator (MediaPipe による実推定) と
scripts/fake_people_pose_estimator_ros.py (カメラ無しの偽推定) が同じ形式で
結果を返せるように、両者から共有される。
MediaPipe に依存しないので、偽推定側だけを使うときは MediaPipe が無くてもよい。

  Person3D / Bone / CameraIntrinsics … 1 人分の姿勢とカメラ内部パラメータ
  EstimationResult                   … ROS ノードが返す 1 フレーム分の結果
                                       (scripts/people_pose_estimator_ros.py と
                                        scripts/fake_people_pose_estimator_ros.py
                                        が同じ型で返す)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, K):
        """3x3 行列 (もしくは長さ 9 のシーケンス) から生成する."""
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        return cls(fx=float(K[0, 0]), fy=float(K[1, 1]),
                   cx=float(K[0, 2]), cy=float(K[1, 2]))


@dataclass
class Bone:
    name: str
    start_point: np.ndarray
    end_point: np.ndarray


@dataclass
class Person3D:
    """カメラ座標系での 1 人分の 3 次元姿勢.

    ``hidden_limb_names`` / ``hidden_positions`` は、可視性が足りない・
    画角の外・深度が取れない等の理由で ``limb_names`` / ``positions`` から
    除かれた関節を並行して保持する (最終的に人物とすら認識されない場合は
    空)。fake_people_pose_estimator_ros.py だけが埋める -- 本物の推定は
    見えていない関節の 3 次元位置を知りようがないので常に空のまま。
    viewer (aero_demo.palm_plane_view) がこれを薄く描くのに使う。
    """
    limb_names: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    positions: list = field(default_factory=list)  # np.ndarray([x, y, z])
    bones: list = field(default_factory=list)      # Bone
    hidden_limb_names: list = field(default_factory=list)
    hidden_positions: list = field(default_factory=list)  # np.ndarray([x, y, z])
    hidden_bones: list = field(default_factory=list)      # Bone, >=1 endpoint hidden

    def position_of(self, limb_name):
        if limb_name not in self.limb_names:
            return None
        return self.positions[self.limb_names.index(limb_name)]


@dataclass
class EstimationResult:
    """1 フレーム分の推定結果.

    実推定 (people_pose_estimator_ros.RosPeoplePoseEstimator) と
    偽推定 (fake_people_pose_estimator_ros.FakeRosPeoplePoseEstimator) が
    同じ形式で返す。
    """
    stamp: object = None                        # rospy.Time
    frame_id: str = ''                          # people の座標系 (既定 base_link)
    camera_frame_id: str = ''                   # 入力画像の frame_id
    image: np.ndarray = None                    # BGR 画像
    joint_positions: list = field(default_factory=list)  # 2D 関節 (dict のリスト)
    people: list = field(default_factory=list)           # Person3D
    # カメラの内部パラメータ・画像サイズ・frame_id 相対のカメラ姿勢。
    # viewer (aero_demo.palm_plane_view) が画角の四角すいを描くのに使う。
    # 本物は CameraInfo + TF から、偽推定は既知の仮想カメラ設定から埋める。
    # TF が引けず people がカメラ座標系のまま返ってきた場合や、実推定で
    # まだ CameraInfo を受け取っていない場合は camera_pose / camera_intrinsics
    # が None のままのことがある。
    camera_intrinsics: object = None             # CameraIntrinsics or None
    camera_width: int = 0
    camera_height: int = 0
    camera_pose: np.ndarray = None               # camera_frame_id -> frame_id, 4x4 or None
