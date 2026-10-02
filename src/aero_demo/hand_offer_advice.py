# -*- coding:utf-8 -*-

"""握手の IK に失敗したとき、人にどう手を差し出し直してほしいかを決める.

差し出した手を ``tools/handshake_reachability_map.py`` と同じ定義の 6 つの
量 (体の座標系での掌の中心の前方・外側・高さと、指先の左右・上下・指先
まわりのひねり) で測り、map で IK が後処理まで解けた格子点 (``config/
hand_offer_reachability.json``、``tools/build_hand_offer_reachability.py``
が作る) のうち今の出し方に一番近いものを探して、その差を発話用の指示に
する。

- 近さは位置 [m] と角度 [度] を ``ANGLE_WEIGHT_M_PER_DEG`` で揃えた
  重み付きの距離で測る。
- 境目の点を選ぶと少しずれただけで失敗するので、周り
  (``NEIGHBOR_RADIUS`` 以内、同じ向き) の格子点の解けた割合が
  ``NEIGHBOR_OK_MIN`` 以上の点だけを候補にする。
- 指示は差の大きい順に ``MAX_ADVICE`` 個まで。位置は 5 cm 単位の量
  (「あと 10 センチ前に」)、向きは目標の向き (「手のひらを斜め下に」) で
  伝える。

骨格・掌は同じ座標系 (``base_link`` など、z が床からの高さ) なら平行移動
されていてもよい。map は身長 1.70 m の合成骨格 (関節は体の中心) で作って
いるので、実カメラの骨格 (体の表面の点、肩が数 cm 手前に出る) では前方
距離が小さめに出て、前に出す量はやや多めに伝わる。
"""

import json
import math
import os

import numpy as np

DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config',
    'hand_offer_reachability.json')

# 角度 1 度を位置何 m 分とみなすか (45 度 = 0.10 m)。
ANGLE_WEIGHT_M_PER_DEG = 0.10 / 45.0
NEIGHBOR_RADIUS = 0.08
NEIGHBOR_OK_MIN = 0.6
# これ未満の差は指示しない。
POSITION_ADVICE_MIN = 0.04
ANGLE_ADVICE_MIN_DEG = 25.0
MAX_ADVICE = 2

_KEYS = ('forward', 'lateral', 'height', 'yaw', 'pitch', 'roll')
_ANGLE_KEYS = ('yaw', 'pitch', 'roll')


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _get(joint_positions, name):
    v = joint_positions.get(name)
    return None if v is None else np.asarray(v, dtype=np.float64)


def _wrap_deg(a):
    return (a + 180.0) % 360.0 - 180.0


def body_frame(joint_positions):
    """人の体の座標系 (原点、正面方向、左方向、方向はいずれも水平) を返す.

    原点は両肩の中点 (無ければ両腰、``Neck``)、正面方向は右肩->左肩
    (無ければ右腰->左腰) と鉛直上向きの外積 (``solve_palm_ik.
    human_facing_direction`` と同じ)。求まらなければ ``None``。
    """
    forward = None
    origin = None
    for a, b in (('RShoulder', 'LShoulder'), ('RHip', 'LHip')):
        pa, pb = _get(joint_positions, a), _get(joint_positions, b)
        if pa is None or pb is None:
            continue
        right_to_left = pb - pa
        right_to_left[2] = 0.0
        if np.linalg.norm(right_to_left) < 1e-6:
            continue
        forward = _unit(np.cross(right_to_left, np.array([0.0, 0.0, 1.0])))
        origin = 0.5 * (pa + pb)
        break
    if forward is None:
        return None
    left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
    return origin, forward, left


def measure_offer(joint_positions, palm, hand):
    """差し出した手 ``hand`` ('R'/'L') の掌 ``palm`` を、``tools/handshake_
    reachability_map.py`` の格子と同じ定義で測る.

    ``forward``/``lateral``/``height`` [m]: 掌の中心の、両肩の中点からの
    前方・外側 (差し出す手の側が正) と床からの高さ。
    ``yaw``/``pitch`` [度]: 指先 (``palm['x_axis']``) の外側・上への角度。
    ``roll`` [度]: 指先まわりの回内 (0 = 親指が上・掌が内側、正で掌が
    下を向き 90 で真下、-90 で真上)。
    体の座標系や掌の軸が求まらなければ ``None``。
    """
    frame = body_frame(joint_positions)
    if frame is None or palm is None or palm.get('x_axis') is None \
            or palm.get('y_axis') is None:
        return None
    origin, forward, left = frame
    side = -1.0 if hand == 'R' else 1.0
    position = np.asarray(palm['position'], dtype=np.float64)
    rel = position - origin
    u = _unit(palm['x_axis'])
    n = _unit(palm['y_axis'])
    ux, uy, uz = float(np.dot(u, forward)), float(side * np.dot(u, left)), \
        float(u[2])
    # handshake_reachability_map.hand_frame の逆: ひねり 0 と 90 度のときの
    # 掌の法線を作り、実際の法線をその 2 軸に射影した角度をひねりとする。
    up = np.array([0.0, 0.0, 1.0])
    up_perp = up - np.dot(up, u) * u
    if np.linalg.norm(up_perp) < 1e-6:
        return None
    up_perp = _unit(up_perp)
    medial = np.cross(up_perp, u) if hand == 'R' else np.cross(u, up_perp)

    def normal_of(v):
        return np.cross(v, u) if hand == 'R' else np.cross(u, v)

    n0, n90 = normal_of(up_perp), normal_of(medial)
    return dict(
        forward=float(np.dot(rel, forward)),
        lateral=float(side * np.dot(rel, left)),
        height=float(position[2]),
        yaw=math.degrees(math.atan2(uy, ux)),
        pitch=math.degrees(math.asin(np.clip(uz, -1.0, 1.0))),
        roll=math.degrees(math.atan2(np.dot(n, n90), np.dot(n, n0))))


class OfferAdvisor(object):
    """reachability map の表から、今の出し方に一番近い解ける出し方を探す。"""

    def __init__(self, table_path=DEFAULT_TABLE_PATH):
        with open(table_path) as f:
            table = json.load(f)
        self.points = {}
        for hand in ('R', 'L'):
            rows = np.asarray(table['points'][hand], dtype=np.float64)
            ok = rows[:, len(_KEYS)] > 0.5
            self.points[hand] = (rows[:, :len(_KEYS)], ok,
                                 self._robust(rows[:, :len(_KEYS)], ok))

    @staticmethod
    def _robust(x, ok):
        """各格子点について、同じ向きで ``NEIGHBOR_RADIUS`` 以内の点の
        解けた割合が ``NEIGHBOR_OK_MIN`` 以上の解けた点か。"""
        robust = np.zeros(len(x), dtype=bool)
        for i in np.nonzero(ok)[0]:
            same = np.all(np.abs(x[:, 3:] - x[i, 3:]) < 1e-6, axis=1)
            near = same & (np.linalg.norm(x[:, :3] - x[i, :3], axis=1)
                           <= NEIGHBOR_RADIUS + 1e-9)
            robust[i] = ok[near].mean() >= NEIGHBOR_OK_MIN
        return robust

    def nearest(self, measure, hand):
        """``measure`` (``measure_offer``) に一番近い解ける格子点を
        (格子点の dict, 重み付き距離) で返す。"""
        x, _, robust = self.points[hand]
        cur = np.array([measure[k] for k in _KEYS])
        d = x - cur
        d[:, 3:] = _wrap_deg(d[:, 3:]) * ANGLE_WEIGHT_M_PER_DEG
        dist = np.linalg.norm(d, axis=1)
        dist[~robust] = np.inf
        i = int(np.argmin(dist))
        if not np.isfinite(dist[i]):
            return None, None
        return dict(zip(_KEYS, (float(v) for v in x[i]))), float(dist[i])

    def advise(self, measure, hand, max_items=MAX_ADVICE):
        """直してほしい項目を差の大きい順に返す。各要素は
        ``(key, 差 (m または度), 発話用の句 (て形))``。"""
        if measure is None:
            return [], None
        target, _ = self.nearest(measure, hand)
        if target is None:
            return [], None
        items = []
        for key in ('forward', 'lateral', 'height'):
            diff = target[key] - measure[key]
            if abs(diff) >= POSITION_ADVICE_MIN:
                items.append((abs(diff), key, diff,
                              _position_phrase(key, diff)))
        for key in _ANGLE_KEYS:
            diff = _wrap_deg(target[key] - measure[key])
            if abs(diff) >= ANGLE_ADVICE_MIN_DEG:
                items.append((abs(diff) * ANGLE_WEIGHT_M_PER_DEG, key, diff,
                              _angle_phrase(key, diff, target)))
        items.sort(key=lambda item: -item[0])
        return [(key, diff, phrase)
                for _, key, diff, phrase in items[:max_items]], target


def _centimeters(diff):
    """発話する量 [cm] (5 cm 単位、最低 5 cm)。"""
    return max(5, int(round(abs(diff) * 100.0 / 5.0)) * 5)


def _position_phrase(key, diff):
    cm = _centimeters(diff)
    if key == 'forward':
        return ('あと{}センチ前に出して' if diff > 0
                else '{}センチほど体の方に引いて').format(cm)
    if key == 'lateral':
        return ('{}センチほど体の外側にずらして' if diff > 0
                else '{}センチほど体の正面に寄せて').format(cm)
    return ('あと{}センチ上げて' if diff > 0
            else '{}センチほど下げて').format(cm)


def _angle_phrase(key, diff, target):
    if key == 'roll':
        roll = target['roll']
        if 60.0 <= roll <= 120.0:
            return '手のひらを下に向けて'
        if 20.0 <= roll < 60.0:
            return '手のひらを斜め下に向けて'
        if -20.0 < roll < 20.0:
            return '親指を上にして、手のひらを横に向けて'
        if -60.0 < roll <= -20.0:
            return '手のひらを斜め上に向けて'
        if -120.0 <= roll <= -60.0:
            return '手のひらを上に向けて'
        return '手首を{}ひねって'.format(
            '手のひらが下を向くように' if diff > 0 else '手のひらが上を向くように')
    if key == 'pitch':
        return '指先を少し{}げて'.format('上' if diff > 0 else '下')
    return '指先を少し{}に向けて'.format('外側' if diff > 0 else '内側')


def advice_speech(advice):
    """``OfferAdvisor.advise`` の結果を 1 つの発話文にする (無ければ
    空文字列)。"""
    if not advice:
        return ''
    return '、'.join(phrase for _, _, phrase in advice) + 'ください。'
