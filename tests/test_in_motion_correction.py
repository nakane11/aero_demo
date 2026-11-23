"""接近中の再照準 (in_motion_correction) の単体テスト。

    python -m pytest tests/test_in_motion_correction.py -s
"""
import functools
import os
import sys

import numpy as np

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'src')
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from aero_demo import in_motion_correction as imc  # noqa: E402


PALM_IN_BASE = [0.3, 0.6]
FINAL = [1.0, 0.5, np.pi / 2]


def _planned_palm():
    return imc.transform_point(FINAL, PALM_IN_BASE)


def test_palm_anchored_position_keeps_palm_in_base():
    position = imc.palm_anchored_position(_planned_palm(), PALM_IN_BASE,
                                          FINAL[2] + 0.1)
    pose = [*position, FINAL[2] + 0.1]
    assert np.allclose(imc.inverse_transform_point(pose, _planned_palm()),
                       PALM_IN_BASE)


def _plan():
    # x 方向に 0.3 m を 3 区間で進んで止まる。
    cum = [0.0, 0.5, 1.0, 1.5]
    positions = [[0.0, 0.0], [0.12, 0.0], [0.25, 0.0], [0.3, 0.0]]
    velocities = [[0.0, 0.0], [0.25, 0.0], [0.15, 0.0], [0.0, 0.0]]
    yaws = [0.0, 0.0, 0.0, 0.0]
    yaw_rates = [0.0, 0.0, 0.0, 0.0]
    return cum, positions, velocities, yaws, yaw_rates


def _build(shift, elapsed=0.6):
    cum, positions, velocities, yaws, yaw_rates = _plan()
    return imc.build_retarget_points(
        cum, positions, velocities, yaws, yaw_rates, elapsed,
        [0.14, 0.0], [0.2, 0.0], 0.0, 0.0, [(shift, 0.0)])


def test_retarget_points_end_at_shifted_goal_and_stop():
    points = _build([0.03, 0.05])
    t, position, velocity, yaw, yaw_rate = points[-1]
    assert abs(t - 0.9) < 1e-9
    assert np.allclose(position, [0.33, 0.05])
    assert np.allclose(velocity, 0.0)
    # 先頭は今の状態、時刻は単調増加。
    assert points[0][0] == 0.0 and np.allclose(points[0][1], [0.14, 0.0])
    times = [p[0] for p in points]
    assert all(b - a >= imc.MIN_POINT_GAP - 1e-9
               for a, b in zip(times[:-1], times[1:]))


def test_retarget_points_too_close_to_the_end():
    assert _build([0.0, 0.0], elapsed=1.45) is None


def test_monotone_tail_shrinks_a_backward_shift():
    direction = [1.0, 0.0]
    build = functools.partial(_build)
    points, scale = imc.monotone_tail(build, [0.04, 0.02], direction)
    assert scale == 1.0 and points is not None
    # 大きく手前へ (後退を要する) ずらすと縮めても通らない。
    points, scale = imc.monotone_tail(build, [-0.25, 0.0], direction)
    assert points is None or scale < 1.0
