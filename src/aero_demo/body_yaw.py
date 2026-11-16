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

手をつないで横に並ぶと反対側の肩も隠れるので、``torso_band_yaw`` は
人の領域の点群から胴体上部の主軸を取り、つないでいる手の側を左として
向きを出す (関節の左右のラベルに頼らない)。
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

    def current(self, stamp):
        """``stamp`` [s] の時点で有効なならした向き [rad] (無ければ None)."""
        if (self._vec is None or self._stamp is None
                or stamp - self._stamp > self.reset_after
                or stamp < self._stamp):
            return None
        return math.atan2(self._vec[1], self._vec[0])


TorsoBandYaw = namedtuple('TorsoBandYaw', [
    'yaw',      # 体の正面のヨー角 [rad] (-pi, pi]
    'ratio',    # 帯の点の水平の広がりの 長軸 / 短軸 (標準偏差の比)
    'count',    # 帯の点の数
    'center',   # (3,) 帯の点の重心
    'hand_cos',  # |主軸と 重心 -> 手 の向きの cos| (小さいほど左右が曖昧)
    'ambiguous',  # hand_cos < ambiguous_cos (左右を前回の向きで決めた/決められない)
])

# 手前の肩の高さからの帯 [m] (肩と上腕・胸の上部が入る)。
DEFAULT_TORSO_BAND = (-0.20, 0.03)
DEFAULT_TORSO_MAX_FROM_SHOULDER = 0.45
DEFAULT_TORSO_MIN_POINTS = 50
# 主軸と 重心 -> 手 の向きの |cos| がこれ未満なら左右を前回の向きで決める。
DEFAULT_TORSO_AMBIGUOUS_COS = 0.35


def nearest_joint(joints, names, target):
    """``names`` のうち ``target`` に水平距離で最も近い関節名 (無ければ None)."""
    best, best_dist = None, None
    for name in names:
        p = joints.get(name)
        if p is None:
            continue
        dist = math.hypot(p[0] - target[0], p[1] - target[1])
        if best_dist is None or dist < best_dist:
            best, best_dist = name, dist
    return best


def masked_depth_points(depth, mask, K, transform=None, stride=4,
                        min_depth=0.15, max_depth=3.0):
    """``mask`` が真で深度が有効な画素を 3 次元点 (N, 3) にする.

    ``depth`` [m] と ``mask`` は同じ大きさ、``K`` は 3x3 の内部パラメータ。
    ``transform`` (4x4) を渡すとその座標系に直す。``stride`` 個に 1 個を使う。
    """
    depth = np.asarray(depth, dtype=np.float32)
    ok = np.asarray(mask, dtype=bool) & (depth > min_depth) & (depth < max_depth)
    v, u = np.nonzero(ok)
    if stride > 1:
        v, u = v[::stride], u[::stride]
    z = depth[v, u].astype(np.float64)
    pts = np.stack([(u - K[0][2]) / K[0][0] * z,
                    (v - K[1][2]) / K[1][1] * z, z], axis=1)
    if transform is not None:
        transform = np.asarray(transform, dtype=np.float64)
        pts = pts.dot(transform[:3, :3].T) + transform[:3, 3]
    return pts


def torso_band_yaw(points, near_shoulder, hand,
                   band=DEFAULT_TORSO_BAND,
                   max_from_shoulder=DEFAULT_TORSO_MAX_FROM_SHOULDER,
                   min_points=DEFAULT_TORSO_MIN_POINTS,
                   previous=None,
                   ambiguous_cos=DEFAULT_TORSO_AMBIGUOUS_COS):
    """胴体の上部の点の並び (主軸) を肩の線とし、手をつないだ側を左として向きを出す.

    手をつないで横に並ぶと頭のカメラから反対側の肩が隠れ、両肩の関節の線は
    深度の誤差で決まってしまう。また背中側から見ると MediaPipe が左右の肩を
    取り違える。そこで

    1. 人の点群 (z 上向きの座標系) のうち、手前の肩 ``near_shoulder`` の
       高さから ``band`` の範囲で、肩から水平に ``max_from_shoulder`` 以内の
       点を取る (肩・上腕・見えている胸か背中の面で横に長い形になる)。
    2. その点の水平の広がりの主軸を肩の線とする。
    3. 主軸の両端のうち、つないでいる手 ``hand`` の側を人の左とする
       (人の左手とロボットの手をつなぐ前提)。

    主軸が決まりにくい (``ratio`` が 1 に近い) ときの扱いは呼び出し側で決める。

    人がロボットの方を真横に向く・真後ろを向く (±90° 付近) と、肩の線が
    帯の重心 -> 手の向きとほぼ直交し、3. の左右が深度の誤差で入れ替わる
    (11/16 の実機で向きが一瞬 180° 反転した)。主軸と 重心 -> 手 の向きの
    cos の絶対値が ``ambiguous_cos`` 未満で ``previous`` (同じ座標系の
    前回の向き) があれば、前回に近い方を選ぶ (前回が無ければ ``ambiguous``
    を立てて返すので、呼び出し側で捨てる)。

    Returns
    -------
    TorsoBandYaw or None
        帯の点が ``min_points`` 未満なら None。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    near = np.asarray(near_shoulder, dtype=np.float64)
    if len(pts) == 0:
        return None
    dz = pts[:, 2] - near[2]
    sel = ((dz > band[0]) & (dz < band[1])
           & (np.hypot(pts[:, 0] - near[0], pts[:, 1] - near[1])
              < max_from_shoulder))
    band_pts = pts[sel]
    if len(band_pts) < max(min_points, 3):
        return None
    center = band_pts.mean(axis=0)
    xy = band_pts[:, :2] - center[:2]
    eigvals, eigvecs = np.linalg.eigh(xy.T.dot(xy) / len(xy))
    axis = eigvecs[:, 1]
    to_hand = np.asarray(hand[:2], dtype=np.float64) - center[:2]
    cos = axis.dot(to_hand) / max(np.linalg.norm(to_hand), 1e-9)
    if cos < 0.0:
        axis = -axis
    yaw = math.atan2(-axis[0], axis[1])
    ambiguous = abs(cos) < ambiguous_cos
    if ambiguous and previous is not None:
        flipped = wrap_angle(yaw + math.pi)
        if abs(wrap_angle(flipped - previous)) < abs(wrap_angle(yaw - previous)):
            yaw = flipped
    ratio = math.sqrt(max(eigvals[1], 0.0) / max(eigvals[0], 1e-12))
    return TorsoBandYaw(yaw=yaw, ratio=ratio, count=int(len(band_pts)),
                        center=center, hand_cos=float(abs(cos)),
                        ambiguous=bool(ambiguous))


HeadTop = namedtuple('HeadTop', [
    'top',      # (3,) 頭のてっぺん (xy は上端付近の点の中央値)
    'points',   # (N, 3) 人とみなした点 (手の近く・手より上・ロボットより外)
])

# 手から水平にこの距離 [m] 以内の点を人とみなす。
DEFAULT_PERSON_RADIUS = 0.7
# 手よりこれ [m] 以上高い点だけ使う (ロボットの手・前腕を除く)。
DEFAULT_PERSON_MIN_ABOVE_HAND = 0.15
DEFAULT_PERSON_MAX_Z = 2.1
# 頭のてっぺんの高さは上からこの個数目の点の高さ (飛び値を避ける)。
DEFAULT_HEAD_TOP_RANK = 30


def head_top_from_depth(points, hand, radius=DEFAULT_PERSON_RADIUS,
                        min_above_hand=DEFAULT_PERSON_MIN_ABOVE_HAND,
                        max_z=DEFAULT_PERSON_MAX_Z, behind_hand=0.05,
                        top_rank=DEFAULT_HEAD_TOP_RANK, min_points=200):
    """骨格を使わずに、手をつないでいる人の頭のてっぺんを深度の点群から探す.

    背中側から近くで見ると頭の上と後ろ (髪) しか写らず、MediaPipe が人を
    見つけない (11/16 の実機)。人の左手はロボットの手 ``hand`` とつながって
    いるので、``points`` (base_link など z 上向き・原点がロボットの座標系)
    のうち

    - ``hand`` から水平に ``radius`` 以内
    - ``hand`` より ``min_above_hand`` 以上高く ``max_z`` より低い
    - ロボットの中心から手の向きに測って、手より ``behind_hand`` 以上
      内側でない (ロボットの体・腕を除く)

    の点を人とみなし、その最上部を頭のてっぺんとする。

    Returns
    -------
    HeadTop or None
        人とみなした点が ``min_points`` 未満なら None。
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    hand = np.asarray(hand, dtype=np.float64)
    if len(pts) == 0:
        return None
    hand_dist = math.hypot(hand[0], hand[1])
    if hand_dist < 1e-6:
        return None
    out = hand[:2] / hand_dist
    sel = ((np.hypot(pts[:, 0] - hand[0], pts[:, 1] - hand[1]) < radius)
           & (pts[:, 2] > hand[2] + min_above_hand) & (pts[:, 2] < max_z)
           & (pts[:, :2].dot(out) > hand_dist - behind_hand))
    person = pts[sel]
    if len(person) < max(min_points, top_rank + 1):
        return None
    z_top = np.partition(person[:, 2], -top_rank - 1)[-top_rank - 1]
    head = person[person[:, 2] > z_top - 0.05]
    top = np.array([np.median(head[:, 0]), np.median(head[:, 1]), z_top])
    return HeadTop(top=top, points=person)


def fold_to_front(yaw, max_relative=math.pi / 2, previous=None,
                  margin=math.radians(20.0)):
    """向き ``yaw`` をロボット正面 (0) から ``max_relative`` 以内に折り返す.

    背中側から見ると MediaPipe が左右の肩を取り違え、肩の線の向きが 180°
    逆になる。手をつないで横に並んでいる人はロボットと逆向きにはなれない
    (腕が交差する) ので、``yaw`` と ``yaw + 180°`` のうちロボット正面に近い
    方を選ぶ。どちらも境目 (``max_relative``) から ``margin`` 以内のときは、
    ``previous`` (同じ座標系の前回の向き) に近い方を選んでばたつきを防ぐ。
    ``max_relative`` が 180° 以上なら何もしない。
    """
    yaw = wrap_angle(yaw)
    if max_relative >= math.pi:
        return yaw
    flipped = wrap_angle(yaw + math.pi)
    if (previous is not None
            and abs(abs(yaw) - max_relative) < margin):
        if abs(wrap_angle(flipped - previous)) < abs(
                wrap_angle(yaw - previous)):
            return flipped
        return yaw
    return yaw if abs(yaw) <= max_relative else flipped
