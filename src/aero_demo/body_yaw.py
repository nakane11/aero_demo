#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人の体の向き (水平面のヨー角) を腰・肩・顔の関節から求める (ROS 非依存).

関節の座標は z 軸が鉛直上向きの座標系 (base_link や odom) で渡す。
向きは人の正面方向のヨー角 [rad] (x 軸から反時計回り)。

- 腰: 右腰 -> 左腰の線に直交する水平方向
- 肩: 右肩 -> 左肩の線に直交する水平方向
- 顔: 右耳 -> 左耳 (無ければ右目 -> 左目) の線に直交する水平方向。
  片耳しか見えないときは耳 -> 鼻の向き。

腰と肩を ``重み x 関節の信頼度`` で円周上の加重平均にして体の向きとする。
顔は話しかけるときにロボットのほうを向くなど体とずれやすいので、体の向き
から ``face_max_deviation`` 以上ずれていたら捨てる。腰と肩の両方とも
取れなければ (顔だけでは) 向きを返さない。重みが 0 の部位は使わない。

既定は肩だけ (腰 0・肩 1・顔 0)。手つなぎで横に並ぶと頭のカメラには
人の顔と胸元しか写らず、腰は画角の外になるため (10/21 試行1 の bag)。
"""

import math
from collections import namedtuple

import numpy as np

# 左右の関節の水平距離がこれより短いと、左右の線の向きが深度の誤差で
# 決まってしまうので使わない [m] (腰幅・肩幅は 0.3m 前後)。
MIN_HIP_WIDTH = 0.10
MIN_SHOULDER_WIDTH = 0.12
MIN_EAR_WIDTH = 0.06
MIN_EYE_WIDTH = 0.03
# 片耳 -> 鼻の水平距離の下限 [m]。
MIN_EAR_NOSE_DISTANCE = 0.04
# 耳 -> 鼻の向きと顔の正面のずれ [rad]。鼻は耳より約 10cm 前・約 7.5cm
# 体の中心寄りにあるので、右耳 -> 鼻は正面より左 (反時計回り) へ、
# 左耳 -> 鼻は右へこれだけ回っている。
EAR_NOSE_YAW_OFFSET = math.atan2(0.075, 0.10)

DEFAULT_WEIGHTS = {'hip': 0.0, 'shoulder': 1.0, 'face': 0.0}
DEFAULT_FACE_MAX_DEVIATION = math.radians(45.0)

BodyYaw = namedtuple('BodyYaw', [
    'yaw',          # 体の正面のヨー角 [rad] (-pi, pi]
    'confidence',   # 0..1 (使えた部位の 重み x 信頼度 の和 / 重みの和)
    'parts',        # {部位名: (yaw, 信頼度)} 求められた部位 (使わなかった顔も含む)
    'used',         # 平均に使った部位名のリスト
    'center',       # (3,) 腰の中点 (腰を使っていなければ肩の中点)
])


def wrap_angle(angle):
    """角度を (-pi, pi] に丸める."""
    return math.atan2(math.sin(angle), math.cos(angle))


def circular_mean(angles, weights):
    """重み付きの円周上の平均. 重みの和が 0 なら None."""
    s = sum(w * math.sin(a) for a, w in zip(angles, weights))
    c = sum(w * math.cos(a) for a, w in zip(angles, weights))
    if math.hypot(s, c) < 1e-9:
        return None
    return math.atan2(s, c)


def _forward_from_left_vector(right, left, min_width):
    """右 -> 左の水平ベクトルに直交する正面のヨー角 (短すぎれば None).

    正面方向は z 上向きのとき左向きベクトルを時計回りに 90 度回したもの。
    """
    lx = left[0] - right[0]
    ly = left[1] - right[1]
    if math.hypot(lx, ly) < min_width:
        return None
    return math.atan2(-lx, ly)


def _pair_yaw(joints, scores, right_name, left_name, min_width):
    if right_name not in joints or left_name not in joints:
        return None
    yaw = _forward_from_left_vector(joints[right_name], joints[left_name],
                                    min_width)
    if yaw is None:
        return None
    conf = min(scores.get(right_name, 1.0), scores.get(left_name, 1.0))
    return yaw, max(0.0, min(1.0, conf))


def _face_yaw(joints, scores):
    """耳 (無ければ目) の線、片耳なら耳 -> 鼻 から顔の正面のヨー角."""
    for right_name, left_name, min_width in (('REar', 'LEar', MIN_EAR_WIDTH),
                                             ('REye', 'LEye', MIN_EYE_WIDTH)):
        found = _pair_yaw(joints, scores, right_name, left_name, min_width)
        if found is None:
            continue
        yaw, conf = found
        # 左右の線だけだと前後が裏返っても分からないので、鼻が前にあるかを見る
        nose = joints.get('Nose')
        if nose is not None:
            mid = 0.5 * (np.asarray(joints[right_name], dtype=np.float64)
                         + np.asarray(joints[left_name], dtype=np.float64))
            ahead = ((nose[0] - mid[0]) * math.cos(yaw)
                     + (nose[1] - mid[1]) * math.sin(yaw))
            if ahead < 0.0:
                yaw = wrap_angle(yaw + math.pi)
        return yaw, conf
    nose = joints.get('Nose')
    if nose is None:
        return None
    for ear_name in ('REar', 'LEar'):
        ear = joints.get(ear_name)
        if ear is None:
            continue
        dx = nose[0] - ear[0]
        dy = nose[1] - ear[1]
        if math.hypot(dx, dy) < MIN_EAR_NOSE_DISTANCE:
            continue
        conf = min(scores.get(ear_name, 1.0), scores.get('Nose', 1.0))
        offset = (-EAR_NOSE_YAW_OFFSET if ear_name == 'REar'
                  else EAR_NOSE_YAW_OFFSET)
        return (wrap_angle(math.atan2(dy, dx) + offset),
                max(0.0, min(1.0, conf)))
    return None


def estimate_body_yaw(joints, scores=None, weights=None,
                      face_max_deviation=DEFAULT_FACE_MAX_DEVIATION,
                      min_part_confidence=0.0):
    """腰・肩・顔から体の向きを求める.

    Parameters
    ----------
    joints : dict
        関節名 -> (3,) 座標 (z 上向きの座標系)。``PeoplePoseEstimator`` の
        関節名 (RHip, LHip, RShoulder, LShoulder, REar, LEar, REye, LEye, Nose)。
    scores : dict or None
        関節名 -> 信頼度 (0..1)。無い関節は 1 とみなす。
    weights : dict or None
        'hip', 'shoulder', 'face' の重み (省略した部位は ``DEFAULT_WEIGHTS``、
        既定は肩だけ)。0 の部位は使わない。
    face_max_deviation : float
        顔の向きが腰・肩の向きからこれ [rad] 以上ずれていたら使わない。
    min_part_confidence : float
        部位の信頼度がこれ未満なら使わない。

    Returns
    -------
    BodyYaw or None
        (重みが正の) 腰も肩も取れなければ None。
    """
    scores = scores or {}
    weights = dict(DEFAULT_WEIGHTS, **(weights or {}))

    parts = {}
    for name, found in (
            ('hip', _pair_yaw(joints, scores, 'RHip', 'LHip', MIN_HIP_WIDTH)),
            ('shoulder', _pair_yaw(joints, scores, 'RShoulder', 'LShoulder',
                                   MIN_SHOULDER_WIDTH)),
            ('face', _face_yaw(joints, scores))):
        if found is not None and found[1] >= min_part_confidence \
                and found[1] > 0.0:
            parts[name] = found

    body_parts = [n for n in ('hip', 'shoulder')
                  if n in parts and weights[n] > 0.0]
    if not body_parts:
        return None
    body_yaw = circular_mean([parts[n][0] for n in body_parts],
                             [weights[n] * parts[n][1] for n in body_parts])
    if body_yaw is None:
        return None

    used = list(body_parts)
    if 'face' in parts and weights['face'] > 0.0 and \
            abs(wrap_angle(parts['face'][0] - body_yaw)) < face_max_deviation:
        used.append('face')
    yaw = circular_mean([parts[n][0] for n in used],
                        [weights[n] * parts[n][1] for n in used])
    if yaw is None:
        return None

    total_weight = sum(weights.values())
    confidence = (sum(weights[n] * parts[n][1] for n in used) / total_weight
                  if total_weight > 0 else 0.0)

    # 位置は使った部位の中点 (腰を使っていれば腰、無ければ肩)。
    center = None
    pairs = [('RShoulder', 'LShoulder')]
    if 'hip' in used:
        pairs.insert(0, ('RHip', 'LHip'))
    for right_name, left_name in pairs:
        if right_name in joints and left_name in joints:
            center = 0.5 * (np.asarray(joints[right_name], dtype=np.float64)
                            + np.asarray(joints[left_name], dtype=np.float64))
            break
    return BodyYaw(yaw=wrap_angle(yaw), confidence=float(confidence),
                   parts=parts, used=used, center=center)


class YawSmoother(object):
    """向きを (cos, sin) の指数移動平均でならす (信頼度で重み付け).

    ``time_constant`` [s] の一次遅れ。``reset_after`` [s] 以上更新が無ければ
    次の値からやり直す。
    """

    def __init__(self, time_constant=0.3, reset_after=1.0):
        self.time_constant = time_constant
        self.reset_after = reset_after
        self._vec = None
        self._stamp = None

    def reset(self):
        self._vec = None
        self._stamp = None

    def update(self, yaw, stamp, confidence=1.0):
        """``stamp`` [s] の観測 ``yaw`` を入れ、ならした向き [rad] を返す."""
        obs = np.array([math.cos(yaw), math.sin(yaw)])
        if (self._vec is None or self._stamp is None
                or stamp - self._stamp > self.reset_after
                or stamp < self._stamp):
            self._vec = obs
        else:
            dt = stamp - self._stamp
            alpha = 1.0 if self.time_constant <= 0 else \
                1.0 - math.exp(-dt / self.time_constant)
            alpha *= max(0.0, min(1.0, confidence))
            self._vec = (1.0 - alpha) * self._vec + alpha * obs
        self._stamp = stamp
        return math.atan2(self._vec[1], self._vec[0])
