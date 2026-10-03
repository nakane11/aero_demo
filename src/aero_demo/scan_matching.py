#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""2D LiDAR スキャン同士の位置合わせ (point-to-line ICP).

台車の移動前後のスキャンから実際の移動量を求める (aero の /odom は指令の
積分でスリップが出ないため)。姿勢は全て ``(x, y, yaw)``。
"""

import math

import numpy as np
from scipy.spatial import cKDTree

# 法線推定の近傍点数と、近傍の広がりの上限 [m] (超えたら法線を使わない)。
NORMAL_NEIGHBORS = 6
NORMAL_MAX_SPREAD = 0.3
# 対応点の最大距離 [m] (段階的に絞る)。
CORRESPONDENCE_SCHEDULE = (0.5, 0.3, 0.15, 0.08)
ITERATIONS_PER_STAGE = 20
CONVERGENCE_TRANSLATION = 1e-4  # [m]
CONVERGENCE_ROTATION = 1e-4  # [rad]
HUBER_DELTA = 0.02  # [m]


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compose(a, b):
    """a ∘ b."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1],
                     a[1] + s * b[0] + c * b[1],
                     wrap_angle(a[2] + b[2])])


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2]])


def relative(a, b):
    """``a`` から見た ``b`` (a^-1 ∘ b)。"""
    return compose(inverse(a), b)


def transform_points(points, pose):
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T + np.asarray(pose[:2])


def scan_to_points(ranges, angle_min, angle_increment, range_min, range_max,
                   sensor_pose=(0.0, 0.0, 0.0), max_range=None):
    """LaserScan の値を台車座標系の ``(N, 2)`` 点群にする (無効値は捨てる).

    ``sensor_pose`` は台車座標系でのセンサの姿勢。
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    angles = angle_min + angle_increment * np.arange(len(ranges))
    upper = range_max if max_range is None else min(range_max, max_range)
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= upper)
    points = np.column_stack([ranges[valid] * np.cos(angles[valid]),
                              ranges[valid] * np.sin(angles[valid])])
    return transform_points(points, sensor_pose)


def exclude_near(points, centers, radius):
    """``centers`` (M, 2) のいずれかから ``radius`` 以内の点を除く."""
    if centers is None or len(centers) == 0 or len(points) == 0:
        return points
    tree = cKDTree(np.asarray(centers, dtype=np.float64))
    dist, _ = tree.query(points)
    return points[dist > radius]


def _reference_normals(points, tree):
    k = min(NORMAL_NEIGHBORS, len(points))
    _, idx = tree.query(points, k=k)
    neighbors = points[idx]                      # (N, k, 2)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    eigval, eigvec = np.linalg.eigh(cov)         # 昇順
    normals = eigvec[:, :, 0]                    # 最小固有値 = 法線方向
    spread = np.linalg.norm(neighbors - points[:, None, :], axis=2).max(axis=1)
    valid = spread <= NORMAL_MAX_SPREAD
    return normals, valid


def icp_2d(ref_points, cur_points, initial_pose, exclude_centers=None,
           exclude_radius=0.0, schedule=CORRESPONDENCE_SCHEDULE):
    """基準から見た現在の台車の姿勢を point-to-line ICP で求める.

    ``exclude_centers`` (基準座標系) から ``exclude_radius`` 以内の点は
    基準側・現在側とも使わない。

    Returns
    -------
    dict
        ``pose`` [x, y, yaw]、``rms`` [m]、``inlier_ratio``、``n_inliers``、
        ``iterations``。
    """
    ref = exclude_near(np.asarray(ref_points, dtype=np.float64),
                       exclude_centers, exclude_radius)
    cur = np.asarray(cur_points, dtype=np.float64)
    pose = np.asarray(initial_pose, dtype=np.float64).copy()
    empty = dict(pose=pose.tolist(), rms=float('inf'), inlier_ratio=0.0,
                 n_inliers=0, iterations=0)
    if len(ref) < NORMAL_NEIGHBORS or len(cur) < NORMAL_NEIGHBORS:
        return empty
    tree = cKDTree(ref)
    normals, normal_valid = _reference_normals(ref, tree)
    # 除外は初期姿勢で固定する (反復中に変わると目的関数が不連続になる)。
    if exclude_centers is not None and len(exclude_centers):
        moved = transform_points(cur, pose)
        dist, _ = cKDTree(np.asarray(exclude_centers)).query(moved)
        cur = cur[dist > exclude_radius]
    n_cur = len(cur)
    if n_cur < NORMAL_NEIGHBORS:
        return empty

    iterations = 0
    residuals = np.zeros(0)
    for max_dist in schedule:
        for _ in range(ITERATIONS_PER_STAGE):
            iterations += 1
            moved = transform_points(cur, pose)
            dist, idx = tree.query(moved)
            use = (dist <= max_dist) & normal_valid[idx]
            if use.sum() < NORMAL_NEIGHBORS:
                return empty
            q = moved[use]
            n = normals[idx[use]]
            r = np.einsum('ij,ij->i', n, q - ref[idx[use]])
            # δ=(tx, ty, θ) のヤコビアン (左から掛ける)。
            jac = np.column_stack([n[:, 0], n[:, 1],
                                   n[:, 0] * -q[:, 1] + n[:, 1] * q[:, 0]])
            abs_r = np.abs(r)
            w = np.where(abs_r <= HUBER_DELTA, 1.0,
                         HUBER_DELTA / np.maximum(abs_r, 1e-12))
            jw = jac * w[:, None]
            delta = np.linalg.solve(jw.T @ jac + 1e-9 * np.eye(3), -jw.T @ r)
            pose = compose(delta, pose)
            residuals = r
            if (np.hypot(delta[0], delta[1]) < CONVERGENCE_TRANSLATION
                    and abs(delta[2]) < CONVERGENCE_ROTATION):
                break
    return dict(pose=[float(v) for v in pose],
                rms=float(np.sqrt(np.mean(residuals ** 2))),
                inlier_ratio=float(len(residuals) / n_cur),
                n_inliers=int(len(residuals)), iterations=iterations)
