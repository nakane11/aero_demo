"""手に隠れた体幹の関節 (肩・腰) の深度補正の単体テスト。

    python -m pytest tests/test_torso_occlusion_repair.py -s
"""
import os
import sys

import numpy as np
import pytest

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'src')
sys.path.insert(0, os.path.abspath(_SRC_DIR))

pytest.importorskip('mediapipe')

from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)

INTRINSICS = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=960.0, cy=540.0)


def _estimator(occlusion_radius=0.06):
    # MediaPipe のモデルは読み込まない (補正の処理だけを試す)。
    estimator = PeoplePoseEstimator.__new__(PeoplePoseEstimator)
    estimator.use_hand = True
    estimator.occlusion_radius = occlusion_radius
    estimator.occlusion_depth_tolerance = 0.10
    return estimator


def _point(u, v, z):
    return np.array([(u - INTRINSICS.cx) * z / INTRINSICS.fx,
                     (v - INTRINSICS.cy) * z / INTRINSICS.fy, z])


def _scene(left_hip_depth):
    """正面を向いて立つ人 (胴体の深度 2.0 m) が左手を腰の前 1.5 m に出した画素と点。"""
    pixels = {
        'RShoulder': (880.0, 300.0), 'LShoulder': (1040.0, 300.0),
        'RHip': (900.0, 560.0), 'LHip': (1020.0, 560.0),
        'LElbow': (1060.0, 420.0), 'LWrist': (1030.0, 540.0),
    }
    depths = {'RShoulder': 2.0, 'LShoulder': 2.0, 'RHip': 1.9,
              'LHip': left_hip_depth, 'LElbow': 1.7, 'LWrist': 1.5}
    for i, (du, dv) in enumerate([(0, 0), (-10, 30), (0, 60), (10, 90)]):
        name = 'LHand{}'.format(i)
        pixels[name] = (1030.0 + du, 540.0 + dv)
        depths[name] = 1.5
    pixels = {k: np.array(v) for k, v in pixels.items()}
    positions = {k: _point(pixels[k][0], pixels[k][1], depths[k])
                 for k in pixels}
    return positions, pixels


def test_hip_behind_hand_takes_depth_from_other_torso_joints():
    positions, pixels = _scene(left_hip_depth=1.5)  # 手の深度を拾った
    repaired = _estimator()._repair_occluded_torso_joints(
        positions, pixels, INTRINSICS)
    # 右の肩→腰の深度差 (-0.1 m) を左の肩に足す。
    assert repaired['LHip'][2] == pytest.approx(1.9)
    # 画素の位置は保つ。
    u = repaired['LHip'][0] * INTRINSICS.fx / repaired['LHip'][2] + INTRINSICS.cx
    assert u == pytest.approx(pixels['LHip'][0])
    for name in ('RHip', 'RShoulder', 'LShoulder', 'LWrist'):
        np.testing.assert_allclose(repaired[name], positions[name])


def test_hip_with_its_own_depth_is_kept():
    # 画像上で手と重なっていても、深度が手と違えば (体の深度なら) 触らない。
    positions, pixels = _scene(left_hip_depth=1.9)
    repaired = _estimator()._repair_occluded_torso_joints(
        positions, pixels, INTRINSICS)
    np.testing.assert_allclose(repaired['LHip'], positions['LHip'])


def test_falls_back_to_opposite_joint_then_drops():
    positions, pixels = _scene(left_hip_depth=1.5)
    del positions['RShoulder']
    repaired = _estimator()._repair_occluded_torso_joints(
        positions, pixels, INTRINSICS)
    assert repaired['LHip'][2] == pytest.approx(1.9)  # 右腰と同じ深度

    del positions['RHip']
    repaired = _estimator()._repair_occluded_torso_joints(
        positions, pixels, INTRINSICS)
    assert 'LHip' not in repaired


def test_missing_shoulder_depth_is_filled_from_other_torso_joints():
    positions, pixels = _scene(left_hip_depth=1.9)
    del positions['LShoulder']  # 周りに有効な深度が無かった
    filled = _estimator()._fill_missing_torso_depths(
        positions, pixels, INTRINSICS, ['LShoulder'])
    # 左腰 1.9 + (右肩 2.0 - 右腰 1.9) = 2.0
    assert filled['LShoulder'][2] == pytest.approx(2.0)
    u = filled['LShoulder'][0] * INTRINSICS.fx / filled['LShoulder'][2] \
        + INTRINSICS.cx
    assert u == pytest.approx(pixels['LShoulder'][0])
    assert 'LShoulder' not in positions  # 入力は変えない


def test_missing_depth_falls_back_then_stays_missing():
    positions, pixels = _scene(left_hip_depth=1.9)
    del positions['LShoulder']
    del positions['LHip']
    filled = _estimator()._fill_missing_torso_depths(
        positions, pixels, INTRINSICS, ['LShoulder', 'LHip'])
    # 同じ側の腰も抜けていれば反対側の肩の深度、腰は右腰の深度。
    assert filled['LShoulder'][2] == pytest.approx(2.0)
    assert filled['LHip'][2] == pytest.approx(1.9)

    del positions['RShoulder']
    filled = _estimator()._fill_missing_torso_depths(
        positions, pixels, INTRINSICS, ['LShoulder', 'RShoulder', 'LHip'])
    assert 'LShoulder' not in filled and 'RShoulder' not in filled


def test_missing_depth_of_non_torso_joint_is_not_filled():
    positions, pixels = _scene(left_hip_depth=1.9)
    del positions['LElbow']
    filled = _estimator()._fill_missing_torso_depths(
        positions, pixels, INTRINSICS, ['LElbow'])
    assert 'LElbow' not in filled


def test_disabled_with_non_positive_radius():
    positions, pixels = _scene(left_hip_depth=1.5)
    repaired = _estimator(occlusion_radius=0.0)._repair_occluded_torso_joints(
        positions, pixels, INTRINSICS)
    np.testing.assert_allclose(repaired['LHip'], positions['LHip'])
