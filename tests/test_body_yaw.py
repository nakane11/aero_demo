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
    result = body_yaw.estimate_body_yaw(_person(yaw_deg))
    assert result is not None
    assert abs(body_yaw.wrap_angle(result.yaw - math.radians(yaw_deg))) \
        < 1e-6
    assert set(result.used) == {'hip', 'shoulder', 'face'}
    assert result.confidence == pytest.approx(1.0)


def test_face_looking_at_robot_is_ignored():
    # 体は +x、顔は横 (ロボットのほう) を 80 度向いている。
    result = body_yaw.estimate_body_yaw(_person(0.0, face_yaw_deg=80.0))
    assert 'face' not in result.used
    assert abs(result.yaw) < 1e-6
    assert result.confidence == pytest.approx(0.8)


def test_face_small_deviation_is_blended():
    result = body_yaw.estimate_body_yaw(_person(0.0, face_yaw_deg=30.0))
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
    result = body_yaw.estimate_body_yaw(joints)
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
    result = body_yaw.estimate_body_yaw(joints, scores)
    # 腰の信頼度が低いので肩 (40 度) 寄りになる。
    expected = math.atan2(0.3 * math.sin(math.radians(40.0)),
                          0.05 + 0.3 * math.cos(math.radians(40.0)))
    assert result.yaw == pytest.approx(expected)
    assert result.confidence == pytest.approx(0.35)


def test_face_only_returns_none():
    joints = _person(0.0)
    for name in ('RHip', 'LHip', 'RShoulder', 'LShoulder'):
        del joints[name]
    assert body_yaw.estimate_body_yaw(joints) is None


def test_degenerate_pair_is_skipped():
    joints = _person(0.0)
    # 腰の左右が奥行き方向に重なって水平距離がほぼ 0。
    joints['LHip'] = joints['RHip'] + np.array([0.0, 0.02, 0.0])
    result = body_yaw.estimate_body_yaw(joints)
    assert 'hip' not in result.parts
    assert abs(result.yaw) < 1e-6


def test_face_single_ear_uses_ear_to_nose():
    joints = _person(0.0, face_yaw_deg=20.0)
    for name in ('LEar', 'REye', 'LEye'):
        del joints[name]
    result = body_yaw.estimate_body_yaw(joints)
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
    result = body_yaw.estimate_body_yaw(_person(30.0))
    assert result.center == pytest.approx([1.5, 0.3, 0.9])


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
