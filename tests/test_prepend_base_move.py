import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from aero_demo import execution_timing as et  # noqa: E402
from aero_demo import scan_matching  # noqa: E402


def _wp(x, y, yaw, joints=(0.1, 0.2)):
    return dict(base_position=[x, y, 0.0], base_yaw=yaw,
                joint_angle_vector=list(joints))


def test_no_move_returns_same_list():
    wps = [_wp(1.0, 2.0, 0.3), _wp(1.1, 2.0, 0.3)]
    assert et.prepend_base_move(wps, [0.0, 0.0, 0.0]) is wps


def test_first_point_is_the_move():
    # 向きが 1 周以上の値 (接近で回った後) でも、先頭からの移動は base_move。
    for yaw in (0.3, 3.5, -4.0, 7.0):
        hover = _wp(1.5, -0.6, yaw, joints=(0.4, 0.5))
        press = _wp(1.55, -0.6, yaw, joints=(0.6, 0.7))
        move = [0.04, -0.03, math.radians(-7.0)]
        folded = et.prepend_base_move([hover, press], move)
        assert len(folded) == 3
        # 腕は hover のまま。
        assert folded[0]['joint_angle_vector'] == hover['joint_angle_vector']
        points = et.base_trajectory_points(folded)
        np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(points[1], move, atol=1e-12)
        # 押し込みの点は、動いた後の台車から見て元の押し込みと同じ相対移動。
        rel = scan_matching.relative(
            [hover['base_position'][0], hover['base_position'][1], yaw],
            [press['base_position'][0], press['base_position'][1], yaw])
        expected = scan_matching.compose(move, rel)
        np.testing.assert_allclose(points[2][:2], expected[:2], atol=1e-12)
        assert abs(points[2][2] - move[2]) < 1e-12
