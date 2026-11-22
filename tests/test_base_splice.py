"""台車の点の速度 (止まっている区間) と、走っている台車を接近へつなぐ処理。"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aero_demo import execution_timing as et  # noqa: E402


def _wp(x, y, yaw):
    return dict(base_position=[x, y, 0.0], base_yaw=yaw,
                joint_angle_vector=[0.0])


def test_rest_segment_has_zero_end_velocity():
    # 先頭が長さ 0 の区間 (今の姿勢の waypoint) でも、その終点は速度 0。
    deltas = np.array([[0.0, 0.0, 0.0], [0.1, 0.02, 0.0], [0.1, 0.02, 0.0]])
    times = np.array([0.5, 0.4, 0.4])
    old = et.point_velocities(deltas, times)
    new = et.base_point_velocities(deltas, times)
    assert np.linalg.norm(old[1]) > 0.0
    np.testing.assert_allclose(new[1], 0.0)
    np.testing.assert_allclose(new[2], old[2])


def test_rest_axis_only():
    # 向きを変えない区間の次に回る区間: yaw だけ 0、並進はそのまま。
    deltas = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.1]])
    times = np.array([0.4, 0.4])
    vel = et.base_point_velocities(deltas, times)
    assert vel[1][2] == 0.0
    assert vel[1][0] > 0.0


def test_first_segment_not_stretched():
    points = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    times = et.limited_time_list(None, None, points,
                                 [et.MIN_SEGMENT_TIME] * 3)
    assert times[0] < 0.1


def test_start_velocity_shortens_first_segment():
    # 0.3 m/s で走っている途中からなら、静止からより短い時間で着く。
    points = [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]
    rest = et.limited_time_list(None, None, points, [et.MIN_SEGMENT_TIME] * 2)
    moving = et.limited_time_list(None, None, points,
                                  [et.MIN_SEGMENT_TIME] * 2,
                                  base_start_velocity=[0.3, 0.0, 0.0])
    assert moving[0] < rest[0]


def test_spliced_start_moves_only_first():
    wps = [_wp(1.0, 2.0, math.pi / 2), _wp(1.0, 2.3, math.pi / 2),
           _wp(1.0, 2.5, math.pi / 2)]
    spliced = et.spliced_start(wps, [0.1, 0.0, 0.0])
    # 先頭の系で x に 0.1 = world の +y に 0.1。
    np.testing.assert_allclose(spliced[0]['base_position'][:2], [1.0, 2.1],
                               atol=1e-12)
    assert spliced[1] is wps[1] and spliced[2] is wps[2]


def test_splice_target_index():
    # 添字 1 が待機点、lead-in は 0.1 m 刻み。
    wps = [_wp(0.0, 0.0, 0.0)] + [_wp(0.3 + 0.1 * i, 0.0, 0.0)
                                  for i in range(6)]
    assert et.splice_target_index(wps, 6, lookahead=0.25) == 4
    # lead-in が短ければ接近の始点 (max_index) まで。
    assert et.splice_target_index(wps, 3, lookahead=0.25) == 3
    # 先読みなしなら待機点。
    assert et.splice_target_index(wps, 6, lookahead=0.0) == 1
