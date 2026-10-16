"""MediaPipe の推定深度による掌の点と、掌平面の切り替えの単体テスト。

    python -m pytest tests/test_palm_model_depth.py -s
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.abspath(os.path.join(_ROOT, 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(_ROOT, 'scripts')))

pytest.importorskip('mediapipe')

from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
import estimate_palm_poses as epp  # noqa: E402

INTRINSICS = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)
PALM = (0, 5, 9, 13, 17)


def test_model_palm_points_are_scaled_to_measured_depth():
    estimator = PeoplePoseEstimator.__new__(PeoplePoseEstimator)
    estimator.use_hand = True
    anchor = 1.5
    pixels, model_depths = {}, {}
    for k, i in enumerate(PALM):
        name = 'LHand{}'.format(i)
        pixels[name] = np.array([1000.0 + 5 * k, 600.0 + 10 * k])
        model_depths[name] = -20.0 * k  # 手首から手前へ (画素単位)
    # 深度が取れたのが手首だけでも、それに縮尺を合わせる。
    positions = {'LHand0': np.array([0.0, 0.0, anchor])}
    points = estimator._model_palm_points(
        positions, pixels, model_depths, INTRINSICS)
    assert set(points) == {'LPalmModel{}'.format(i) for i in PALM}
    for k, i in enumerate(PALM):
        z = points['LPalmModel{}'.format(i)][2]
        assert z == pytest.approx(anchor * (1.0 - 20.0 * k / 1000.0))
    # 画素の位置は保つ。
    p = points['LPalmModel17']
    assert p[0] * INTRINSICS.fx / p[2] + INTRINSICS.cx == pytest.approx(1020.0)


def test_model_palm_points_need_all_palm_landmarks_and_a_depth():
    estimator = PeoplePoseEstimator.__new__(PeoplePoseEstimator)
    estimator.use_hand = True
    pixels = {'LHand{}'.format(i): np.array([1000.0, 600.0]) for i in PALM}
    model_depths = {'LHand{}'.format(i): 0.0 for i in PALM}
    assert estimator._model_palm_points(
        {}, pixels, model_depths, INTRINSICS) == {}
    del model_depths['LHand9']
    positions = {'LHand0': np.array([0.0, 0.0, 1.5])}
    assert estimator._model_palm_points(
        positions, pixels, model_depths, INTRINSICS) == {}


def _palm_points(prefix, lateral_scale=1.0, normal_tilt=0.0):
    """左手を縦にして +x 側 (ロボット) の逆の -x へ指を向けた掌の 5 点。"""
    base = np.array([1.5, 0.0, 1.0])
    offsets = {0: (0.0, 0.0), 5: (-0.09, 0.03), 9: (-0.095, 0.01),
               13: (-0.09, -0.01), 17: (-0.08, -0.03)}
    points = {}
    for i, (forward, lateral) in offsets.items():
        lateral *= lateral_scale
        points['{}{}'.format(prefix, i)] = base + np.array(
            [forward, normal_tilt * lateral, lateral])
    return points


def _estimate(joints):
    estimator = epp.PalmPoseEstimator()
    return estimator.estimate_palm(joints, 'L')


def test_uses_model_plane_when_depth_points_are_collinear():
    joints = _palm_points('LHand', lateral_scale=0.02)  # ほぼ一直線
    assert _estimate(joints) is None
    joints.update(_palm_points('LPalmModel'))
    palm = _estimate(joints)
    assert palm['from_model_depth']
    expected = _estimate(_palm_points('LHand'))
    np.testing.assert_allclose(palm['y_axis'], expected['y_axis'], atol=1e-6)


def test_uses_model_plane_when_depth_normal_is_flipped():
    joints = _palm_points('LHand', lateral_scale=-1.0)  # 親指側と小指側が逆
    joints.update(_palm_points('LPalmModel'))
    palm = _estimate(joints)
    assert palm['from_model_depth']
    assert float(np.dot(palm['y_axis'],
                        _estimate(_palm_points('LHand'))['y_axis'])) > 0.99


def _edge_on_joints():
    """掌 (_palm_points の x-z 平面) を真上 (+z) から見たときの点。深度で
    見えるのは手前の縁だけで、付け根 4 点は同じ視線上に重なる。"""
    joints = {}
    for prefix in ('LHand', 'LPalmModel'):
        for name, p in _palm_points(prefix).items():
            if not name.endswith('Hand0') and not name.endswith('Model0'):
                p = np.array([p[0], p[1], 1.03])
            joints[name] = p
    return joints


EDGE_ON_CAMERA = np.array([1.45, 0.0, 2.0])


def test_edge_on_palm_contains_view_ray():
    joints = _edge_on_joints()
    assert _estimate(joints) is None
    estimator = epp.PalmPoseEstimator()
    for sign in (1.0, -1.0):
        palm = estimator.estimate_palm(joints, 'L', EDGE_ON_CAMERA,
                                       [0.0, sign * 0.8, 0.6])
        assert palm['edge_on']
        np.testing.assert_allclose(palm['y_axis'], [0.0, sign, 0.0],
                                   atol=0.05)
        # 手首は掌の中央、付け根は手前の縁なので指先方向は少し傾く。
        assert float(np.dot(palm['x_axis'], [-1.0, 0.0, 0.0])) > 0.9
    # 中心は見えている縁から視線の奥 (-z) へずらす。
    assert palm['position'][2] < 1.015


def test_edge_on_sign_from_thumb_ignores_reference():
    """親指 (人差し指側、+z) が手前なら、参照の法線が逆でも表裏は親指で
    決まり、掌を正面から見たときの平面と同じ向きになる。"""
    joints = _edge_on_joints()
    for i in (2, 3, 4):
        joints['LPalmModel{}'.format(i)] = np.array(
            [1.5 - 0.02 * (i - 1), 0.0, 1.05])
    estimator = epp.PalmPoseEstimator()
    expected = _estimate(_palm_points('LHand'))['y_axis']
    for reference in ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0]):
        palm = estimator.estimate_palm(joints, 'L', EDGE_ON_CAMERA, reference)
        assert palm['edge_on']
        assert float(np.dot(palm['y_axis'], expected)) > 0.95


def test_edge_on_only_when_no_plane():
    estimator = epp.PalmPoseEstimator()
    # 画像上で掌の点が広がっていれば、推定深度の平面が使える。
    joints = _edge_on_joints()
    joints.update(_palm_points('LPalmModel'))
    palm = estimator.estimate_palm(joints, 'L', EDGE_ON_CAMERA,
                                   [0.0, 1.0, 0.0])
    assert palm['from_model_depth'] and not palm['edge_on']
    # 視線が指先方向に近いと平面が決まらない。
    assert estimator.estimate_palm(_edge_on_joints(), 'L',
                                   [3.0, 0.0, 1.02], [0.0, 1.0, 0.0]) is None


def test_keeps_depth_plane_when_it_agrees_with_model():
    joints = _palm_points('LHand', normal_tilt=0.5)
    joints.update(_palm_points('LPalmModel'))
    palm = _estimate(joints)
    assert not palm['from_model_depth']
    np.testing.assert_allclose(
        palm['y_axis'],
        _estimate(_palm_points('LHand', normal_tilt=0.5))['y_axis'])
