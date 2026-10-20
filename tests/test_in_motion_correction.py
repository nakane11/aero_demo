"""接近終盤の台車の最終位置のずらし (in_motion_correction) の単体テスト。

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


def test_shift_follows_palm_error():
    shift, info = imc.tail_shift([0.04, 0.03], [1.0, 0.0], [0.9, 0.0],
                                 [1.0, 0.0])
    assert np.allclose(shift, [0.04, 0.03])
    assert not info['clamped'] and not info['projected']


def test_shift_is_clamped():
    shift, info = imc.tail_shift([0.2, 0.0], [1.0, 0.0], [0.9, 0.0],
                                 [1.0, 0.0], max_shift=0.1)
    assert np.allclose(shift, [0.1, 0.0]) and info['clamped']


def test_large_or_small_error_is_not_corrected():
    shift, info = imc.tail_shift([0.4, 0.0], [1.0, 0.0], [0.9, 0.0],
                                 [1.0, 0.0])
    assert shift is None and info['reason'] == 'too_large'
    shift, info = imc.tail_shift([0.005, 0.0], [1.0, 0.0], [0.9, 0.0],
                                 [1.0, 0.0])
    assert shift is None and info['reason'] == 'small'


def test_no_reverse_motion():
    # 残り 5cm なのに 8cm 手前へずらすと後退になる -> 手前への成分を 5cm までに。
    shift, info = imc.tail_shift([-0.08, 0.03], [1.0, 0.0], [0.95, 0.0],
                                 [1.0, 0.0])
    assert info['projected']
    ahead = np.dot(np.array([1.0, 0.0]) + shift - np.array([0.95, 0.0]),
                   [1.0, 0.0])
    assert ahead > -1e-9
    assert abs(shift[1] - 0.03) < 1e-9  # 進行方向に直交する成分は残す


PALM_IN_BASE = [0.3, 0.6]
FINAL = [1.0, 0.5, np.pi / 2]


def _planned_palm():
    return imc.transform_point(FINAL, PALM_IN_BASE)


def _anchored_shift(palm, slip):
    displacement, _ = imc.anchored_displacement(palm, PALM_IN_BASE, FINAL,
                                                slip)
    return imc.odom_shift(
        imc.actual_final_position(FINAL, slip) + displacement, FINAL, slip)


def test_anchored_shift_is_zero_without_error():
    displacement, yaw = imc.anchored_displacement(
        _planned_palm(), PALM_IN_BASE, FINAL, [0.0, 0.0, 0.0])
    assert np.allclose(displacement, 0.0) and np.isclose(yaw, FINAL[2])
    assert np.allclose(_anchored_shift(_planned_palm(), [0.0, 0.0, 0.0]), 0.0)


def test_anchored_shift_follows_palm_and_cancels_slip():
    shift = _anchored_shift(_planned_palm() + [0.04, -0.02], [0.0, 0.0, 0.0])
    assert np.allclose(shift, [0.04, -0.02])
    # 実際の台車が odom より +x に 5cm ずれている -> その分手前に止める。
    shift = _anchored_shift(_planned_palm(), [0.05, 0.0, 0.0])
    assert np.allclose(shift, [-0.05, 0.0])


def test_anchored_shift_ignores_rotation_about_the_palm():
    # 掌のまわりに 12° 回ったスリップは odom 上でずらさ
    # なくてよい。
    palm = _planned_palm()
    yaw = np.radians(12.0)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    slip = np.r_[palm - rot @ palm, yaw]
    shift = _anchored_shift(palm, slip)
    assert np.allclose(shift, 0.0)
    # 実際に動かす量も 0 (スリップの梃子の分を含めない)。
    displacement, _ = imc.anchored_displacement(palm, PALM_IN_BASE, FINAL, slip)
    assert np.allclose(displacement, 0.0)
    assert np.linalg.norm(slip[:2]) > 0.05
    # ずらした先で実際に掌を計画どおりに見る。
    actual = [*imc.transform_point(slip, np.asarray(FINAL[:2]) + shift),
              FINAL[2] + yaw]
    assert np.allclose(imc.inverse_transform_point(actual, palm), PALM_IN_BASE)


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
    return imc.build_tail_points(
        cum, positions, velocities, yaws, yaw_rates, elapsed,
        [0.14, 0.0], [0.2, 0.0], 0.0, 0.0, shift)


def test_tail_points_end_at_shifted_goal_and_stop():
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


def test_tail_points_too_close_to_the_end():
    assert _build([0.0, 0.0], elapsed=1.45) is None


def test_monotone_tail_shrinks_a_backward_shift():
    direction = [1.0, 0.0]
    build = functools.partial(_build)
    points, scale = imc.monotone_tail(build, [0.04, 0.02], direction)
    assert scale == 1.0 and points is not None
    # 大きく手前へ (後退を要する) ずらすと縮めても通らない。
    points, scale = imc.monotone_tail(build, [-0.25, 0.0], direction)
    assert points is None or scale < 1.0
