#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""姿勢推定結果のデータ型と MediaPipe の関節レイアウト定数."""

from dataclasses import dataclass

import numpy as np

# 身体関節の接続 (INDEX2LIMBNAME の 1 始まり index)
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

# 手の関節の接続 (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring,
# 17-20 pinky)。近位 -> 遠位の順。
HAND_SEQUENCE = [[0, 1],   [1, 2],   [2, 3],   [3, 4],
                 [0, 5],   [5, 6],   [6, 7],   [7, 8],
                 [0, 9],   [9, 10],  [10, 11], [11, 12],
                 [0, 13],  [13, 14], [14, 15], [15, 16],
                 [0, 17],  [17, 18], [18, 19], [19, 20]]

# 手のランドマークの局所座標 (手の長さ単位)。u=手首->指先, v=親指側, n=掌の向き。
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
