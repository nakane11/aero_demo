#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""people pose 推定結果を保持する ROS 非依存のデータ型.

people_pose_estimator.PeoplePoseEstimator (MediaPipe による実推定) と
scripts/fake_people_pose_estimator_ros.py (カメラ無しの偽推定) が同じ形式で
結果を返せるように、両者から共有される。
MediaPipe に依存しないので、偽推定側だけを使うときは MediaPipe が無くてもよい。

  Person3D / Bone / CameraIntrinsics … 1 人分の姿勢とカメラ内部パラメータ
  EstimationResult                   … 1 フレーム分の推定結果
  LIMB_SEQUENCE / INDEX2LIMBNAME /
  INDEX2HANDNAME / HAND_SEQUENCE /
  HAND_LOCAL_LANDMARKS                … MediaPipe の関節レイアウト定数
"""

from dataclasses import dataclass, field

import numpy as np

# MediaPipe の身体関節 (limb_sequence) の接続関係。両端の index は
# INDEX2LIMBNAME を参照。
LIMB_SEQUENCE = [[2, 1], [1, 16], [1, 15], [6, 18], [3, 17],
                 [2, 3], [2, 6], [3, 4], [4, 5], [6, 7],
                 [7, 8], [2, 9], [9, 10], [10, 11], [2, 12],
                 [12, 13], [13, 14], [15, 17], [16, 18]]

INDEX2LIMBNAME = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                  "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
                  "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                  "LEye", "REar", "LEar", "Bkg"]

INDEX2HANDNAME = ["RHand{}".format(i) for i in range(21)] + \
                 ["LHand{}".format(i) for i in range(21)]

# MediaPipe の手の関節 (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
# 13-16 ring, 17-20 pinky) の接続関係。
HAND_SEQUENCE = [[0, 1],   [1, 2],   [2, 3],   [3, 4],
                 [0, 5],   [5, 6],   [6, 7],   [7, 8],
                 [0, 9],   [9, 10],  [10, 11], [11, 12],
                 [0, 13],  [13, 14], [14, 15], [15, 16],
                 [0, 17],  [17, 18], [18, 19], [19, 20]]

# 手のランドマークの局所座標 (手の長さを単位とする), MediaPipe の並び
# (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky)。
# 軸: u=手首->指先, v=親指側, n=掌の向き。
HAND_LOCAL_LANDMARKS = np.array([
    [0.00,  0.00, 0.00],   # 0  wrist
    [0.11,  0.13, 0.02],   # 1  thumb CMC
    [0.25,  0.26, 0.05],   # 2  thumb MCP
    [0.36,  0.34, 0.08],   # 3  thumb IP
    [0.45,  0.40, 0.10],   # 4  thumb tip
    [0.47,  0.17, 0.00],   # 5  index MCP
    [0.66,  0.18, 0.04],   # 6  index PIP
    [0.78,  0.18, 0.10],   # 7  index DIP
    [0.88,  0.18, 0.15],   # 8  index tip
    [0.48,  0.05, 0.00],   # 9  middle MCP
    [0.69,  0.05, 0.04],   # 10 middle PIP
    [0.82,  0.05, 0.11],   # 11 middle DIP
    [0.93,  0.05, 0.17],   # 12 middle tip
    [0.46, -0.08, 0.00],   # 13 ring MCP
    [0.65, -0.09, 0.04],   # 14 ring PIP
    [0.77, -0.10, 0.11],   # 15 ring DIP
    [0.87, -0.10, 0.16],   # 16 ring tip
    [0.43, -0.21, 0.00],   # 17 pinky MCP
    [0.57, -0.23, 0.03],   # 18 pinky PIP
    [0.67, -0.24, 0.08],   # 19 pinky DIP
    [0.74, -0.25, 0.12],   # 20 pinky tip
])


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
    # SMPL の体型パラメータ (10,), aero_demo.smpl_body.retarget_and_pose に
    # そのまま渡す想定。fake_people_pose_estimator_ros.py だけが埋める --
    # 本物の推定は体型を推定しないので常に None (平均体型として描かれる)。
    betas: np.ndarray = None

    def position_of(self, limb_name):
        if limb_name not in self.limb_names:
            return None
        return self.positions[self.limb_names.index(limb_name)]


@dataclass
class EstimationResult:
    """1 フレーム分の推定結果."""
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
