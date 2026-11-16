"""腰・肩・顔から体の向きを求める処理 (aero_demo.body_yaw) の単体テスト。

    python -m pytest tests/test_body_yaw.py -s
"""
import math
import os
import sys

import numpy as np
import pytest

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'src')
sys.path.insert(0, os.path.abspath(_SRC_DIR))

from aero_demo import body_yaw  # noqa: E402

# 腰・肩・顔を混ぜるときの重み (既定は肩だけ)。
MIXED_WEIGHTS = {'hip': 0.5, 'shoulder': 0.3, 'face': 0.2}


def _mixed(joints, scores=None):
    return body_yaw.estimate_body_yaw(joints, scores, weights=MIXED_WEIGHTS)


def _rot(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _person(body_yaw_deg, face_yaw_deg=None, center=(1.5, 0.3, 0.0)):
    """+x を向いた人の関節を body_yaw だけ回す (顔は face_yaw だけ回す)。"""
    body = {
        'RHip': [0.0, -0.15, 0.9], 'LHip': [0.0, 0.15, 0.9],
        'RShoulder': [0.0, -0.20, 1.4], 'LShoulder': [0.0, 0.20, 1.4],
    }
    face = {
        'REar': [0.0, -0.08, 1.6], 'LEar': [0.0, 0.08, 1.6],
        'REye': [0.08, -0.03, 1.62], 'LEye': [0.08, 0.03, 1.62],
        'Nose': [0.10, 0.0, 1.58],
    }
    center = np.asarray(center)
    joints = {}
    rb = _rot(math.radians(body_yaw_deg))
    rf = _rot(math.radians(body_yaw_deg if face_yaw_deg is None
                           else face_yaw_deg))
    for name, p in body.items():
        joints[name] = rb.dot(p) + center
    for name, p in face.items():
        joints[name] = rf.dot(p) + center
    return joints


@pytest.mark.parametrize('yaw_deg', [0.0, 30.0, 90.0, -120.0, 180.0])
def test_all_parts_agree(yaw_deg):
    result = _mixed(_person(yaw_deg))
    assert result is not None
    assert abs(body_yaw.wrap_angle(result.yaw - math.radians(yaw_deg))) \
        < 1e-6
    assert set(result.used) == {'hip', 'shoulder', 'face'}
    assert result.confidence == pytest.approx(1.0)


def test_face_looking_at_robot_is_ignored():
    # 体は +x、顔は横 (ロボットのほう) を 80 度向いている。
    result = _mixed(_person(0.0, face_yaw_deg=80.0))
    assert 'face' not in result.used
    assert abs(result.yaw) < 1e-6
    assert result.confidence == pytest.approx(0.8)


def test_face_small_deviation_is_blended():
    result = _mixed(_person(0.0, face_yaw_deg=30.0))
    assert 'face' in result.used
    expected = math.atan2(0.2 * math.sin(math.radians(30.0)),
                          0.8 + 0.2 * math.cos(math.radians(30.0)))
    assert result.yaw == pytest.approx(expected)


def test_hip_preferred_over_shoulder_by_weight():
    joints = _person(0.0)
    # 肩だけ 40 度ねじる (手を引かれて肩が回った)。
    rs = _rot(math.radians(40.0))
    mid = 0.5 * (joints['RShoulder'] + joints['LShoulder'])
    for name in ('RShoulder', 'LShoulder'):
        joints[name] = rs.dot(joints[name] - mid) + mid
    for name in ('REar', 'LEar', 'REye', 'LEye', 'Nose'):
        del joints[name]
    result = _mixed(joints)
    expected = math.atan2(0.3 * math.sin(math.radians(40.0)),
                          0.5 + 0.3 * math.cos(math.radians(40.0)))
    assert result.yaw == pytest.approx(expected)


def test_scores_weight_parts():
    joints = _person(0.0)
    rs = _rot(math.radians(40.0))
    mid = 0.5 * (joints['RShoulder'] + joints['LShoulder'])
    for name in ('RShoulder', 'LShoulder'):
        joints[name] = rs.dot(joints[name] - mid) + mid
    for name in ('REar', 'LEar', 'REye', 'LEye', 'Nose'):
        del joints[name]
    scores = {'RHip': 0.1, 'LHip': 0.1}
    result = _mixed(joints, scores)
    # 腰の信頼度が低いので肩 (40 度) 寄りになる。
    expected = math.atan2(0.3 * math.sin(math.radians(40.0)),
                          0.05 + 0.3 * math.cos(math.radians(40.0)))
    assert result.yaw == pytest.approx(expected)
    assert result.confidence == pytest.approx(0.35)


def test_face_only_returns_none():
    joints = _person(0.0)
    for name in ('RHip', 'LHip', 'RShoulder', 'LShoulder'):
        del joints[name]
    assert _mixed(joints) is None


def test_degenerate_pair_is_skipped():
    joints = _person(0.0)
    # 腰の左右が奥行き方向に重なって水平距離がほぼ 0。
    joints['LHip'] = joints['RHip'] + np.array([0.0, 0.02, 0.0])
    result = _mixed(joints)
    assert 'hip' not in result.parts
    assert abs(result.yaw) < 1e-6


def test_face_single_ear_uses_ear_to_nose():
    joints = _person(0.0, face_yaw_deg=20.0)
    for name in ('LEar', 'REye', 'LEye'):
        del joints[name]
    result = _mixed(joints)
    assert 'face' in result.parts
    assert 'face' in result.used
    # 耳と鼻の横のずれを差し引いて 20 度に近くなる。
    assert abs(math.degrees(result.parts['face'][0]) - 20.0) < 5.0


def test_face_front_back_from_nose():
    joints = _person(0.0)
    for name in ('RHip', 'LHip', 'RShoulder', 'LShoulder'):
        del joints[name]
    # 耳の左右を入れ替えても鼻で正面を決める。
    joints['REar'], joints['LEar'] = joints['LEar'], joints['REar']
    face = body_yaw._face_yaw(joints, {})
    assert abs(face[0]) < 1e-6


def test_center_is_hip_midpoint():
    result = _mixed(_person(30.0))
    assert result.center == pytest.approx([1.5, 0.3, 0.9])


def test_default_uses_shoulders_only():
    joints = _person(0.0, face_yaw_deg=30.0)
    # 腰を 40 度ねじっても既定 (肩だけ) では効かない。
    rh = _rot(math.radians(40.0))
    mid = 0.5 * (joints['RHip'] + joints['LHip'])
    for name in ('RHip', 'LHip'):
        joints[name] = rh.dot(joints[name] - mid) + mid
    result = body_yaw.estimate_body_yaw(joints, {'RShoulder': 0.9})
    assert result.used == ['shoulder']
    assert abs(result.yaw) < 1e-6
    assert result.confidence == pytest.approx(0.9)
    # 位置は肩の中点。
    assert result.center == pytest.approx([1.5, 0.3, 1.4])


def test_default_without_shoulders_returns_none():
    joints = _person(0.0)
    for name in ('RShoulder', 'LShoulder'):
        del joints[name]
    # 腰と顔があっても既定 (肩だけ) では向きを返さない。
    assert body_yaw.estimate_body_yaw(joints) is None


def test_smoother():
    smoother = body_yaw.YawSmoother(time_constant=0.3, reset_after=1.0)
    assert smoother.update(0.0, 0.0) == pytest.approx(0.0)
    # 180 度をまたいでも平均が壊れない。
    smoother.reset()
    smoother.update(math.radians(170.0), 0.0)
    y = smoother.update(math.radians(-170.0), 0.1)
    assert abs(abs(y) - math.pi) < math.radians(15.0)
    # 長く途切れたらやり直す。
    y = smoother.update(math.radians(10.0), 5.0)
    assert y == pytest.approx(math.radians(10.0))
    # 時間がたつほど観測に近づく。
    smoother.reset()
    smoother.update(0.0, 0.0)
    y1 = smoother.update(math.radians(60.0), 0.1)
    y2 = smoother.update(math.radians(60.0), 0.2)
    assert 0.0 < y1 < y2 < math.radians(60.0)


def test_fold_to_front():
    d = math.radians
    fold = body_yaw.fold_to_front
    # 正面寄りはそのまま、後ろ寄りは 180 度折り返す。
    assert fold(d(30.0)) == pytest.approx(d(30.0))
    assert fold(d(128.0)) == pytest.approx(d(-52.0))
    assert fold(d(-150.0)) == pytest.approx(d(30.0))
    # 境目の近くは前回に近い方 (ばたつかない)。
    assert fold(d(95.0), previous=d(-80.0)) == pytest.approx(d(-85.0))
    assert fold(d(95.0), previous=d(80.0)) == pytest.approx(d(95.0))
    # 境目から離れていれば前回によらず折り返す。
    assert fold(d(150.0), previous=d(150.0)) == pytest.approx(d(-30.0))
    # 180 度以上なら無効。
    assert fold(d(150.0), max_relative=math.pi) == pytest.approx(d(150.0))


def test_smoother_current():
    smoother = body_yaw.YawSmoother(time_constant=0.3, reset_after=1.0)
    assert smoother.current(0.0) is None
    smoother.update(0.5, 0.0)
    assert smoother.current(0.5) == pytest.approx(0.5)
    assert smoother.current(2.0) is None


def _band_points(yaw, center=(0.0, -0.6, 1.2), length=0.45, depth=0.12,
                 n=600, seed=0):
    """正面 yaw の人の肩の帯: 左右に長く前後に薄い点群."""
    rng = np.random.RandomState(seed)
    left = np.array([-math.sin(yaw), math.cos(yaw)])
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    s = rng.uniform(-length / 2, length / 2, n)
    d = rng.uniform(-depth / 2, depth / 2, n)
    xy = np.outer(s, left) + np.outer(d, fwd) + np.asarray(center[:2])
    z = center[2] + rng.uniform(-0.15, 0.0, n)
    return np.column_stack([xy, z]), left


@pytest.mark.parametrize('yaw_deg', [0.0, 45.0, -60.0, 90.0, -100.0])
def test_torso_band_yaw_hand_side_is_left(yaw_deg):
    yaw = math.radians(yaw_deg)
    pts, left = _band_points(yaw)
    center = np.array([0.0, -0.6, 1.2])
    near = center.copy()
    near[:2] += left * 0.18
    hand = np.append(center[:2] + left * 0.5, 0.8)
    result = body_yaw.torso_band_yaw(pts, near, hand)
    assert result is not None
    assert abs(body_yaw.wrap_angle(result.yaw - yaw)) < math.radians(5.0)
    assert result.ratio > 2.0
    # 手が反対側なら 180° 逆
    hand_other = np.append(center[:2] - left * 0.5, 0.8)
    other = body_yaw.torso_band_yaw(pts, near, hand_other)
    assert abs(body_yaw.wrap_angle(other.yaw - yaw - math.pi)) < \
        math.radians(5.0)


@pytest.mark.parametrize('yaw_deg', [90.0, -90.0])
def test_torso_band_yaw_ambiguous_side_uses_previous(yaw_deg):
    # 手が肩の線のほぼ真横 (重心の前) にあると左右が決まらない
    yaw = math.radians(yaw_deg)
    pts, left = _band_points(yaw)
    center = np.array([0.0, -0.6, 1.2])
    fwd = np.array([math.cos(yaw), math.sin(yaw)])
    hand = np.append(center[:2] + fwd * 0.5 + left * 0.05, 0.8)
    result = body_yaw.torso_band_yaw(pts, center, hand)
    assert result.ambiguous
    assert result.hand_cos < 0.35
    for prev in (yaw + math.radians(30.0), yaw + math.pi - 0.3):
        r = body_yaw.torso_band_yaw(pts, center, hand, previous=prev)
        expect = yaw if abs(body_yaw.wrap_angle(prev - yaw)) < 1.0 \
            else yaw + math.pi
        assert abs(body_yaw.wrap_angle(r.yaw - expect)) < math.radians(5.0)
    # 曖昧でなければ前回の向きに引きずられない
    hand_side = np.append(center[:2] + left * 0.5, 0.8)
    r = body_yaw.torso_band_yaw(pts, center, hand_side,
                                previous=yaw + math.pi)
    assert not r.ambiguous
    assert abs(body_yaw.wrap_angle(r.yaw - yaw)) < math.radians(5.0)


def test_torso_band_yaw_round_blob_has_low_ratio():
    rng = np.random.RandomState(1)
    xy = rng.normal(0.0, 0.1, (500, 2)) + [0.0, -0.6]
    pts = np.column_stack([xy, np.full(500, 1.1)])
    result = body_yaw.torso_band_yaw(pts, [0.0, -0.6, 1.2], [0.0, -0.2, 0.8])
    assert result is not None
    assert result.ratio < 1.5


def test_torso_band_yaw_ignores_points_outside_band():
    pts, _ = _band_points(0.0, n=30)
    # 帯の外 (腰の高さ・遠く) の点だけ多い
    far = np.column_stack([np.zeros(200), np.linspace(-1, 1, 200),
                           np.full(200, 0.7)])
    result = body_yaw.torso_band_yaw(np.vstack([pts, far]),
                                     [0.0, -0.6, 1.2], [0.0, 0.0, 0.8])
    assert result is None


def test_nearest_joint():
    joints = {'LShoulder': [0.0, -0.5, 1.2], 'RShoulder': [0.1, -0.8, 1.2]}
    assert body_yaw.nearest_joint(
        joints, ('LShoulder', 'RShoulder'), [0.0, -0.2, 0.8]) == 'LShoulder'
    assert body_yaw.nearest_joint({}, ('LShoulder',), [0, 0, 0]) is None


def test_masked_depth_points():
    depth = np.full((4, 4), 2.0, dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 2] = True
    K = [[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]
    pts = body_yaw.masked_depth_points(depth, mask, K, stride=1)
    assert pts.shape == (1, 3)
    assert pts[0] == pytest.approx([0.0, 0.0, 2.0])
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]
    pts = body_yaw.masked_depth_points(depth, mask, K, T, stride=1)
    assert pts[0] == pytest.approx([1.0, 0.0, 2.0])
