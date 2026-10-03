#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""骨格 (MediaPipe 形式の関節位置 dict) から左右の掌の位置姿勢と、
人が差し出している手 (``offered_hand``) を推定して JSON に保存する。

合成骨格と実カメラで同じ ``PalmPoseEstimator.estimate`` を使う。手の
ランドマークが 3 点未満の側は ``None`` (前腕からのフォールバックはしない)。
掌のローカル座標系: +x=指先方向, +y=手の甲->掌 (法線), +z=x cross y。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py \
        --num-samples 100 --output-dir /tmp/random_human_poses
    rosrun aero_demo estimate_palm_poses.py \
        --input-dir /tmp/random_human_poses \
        --output-dir /tmp/random_palm_poses
"""

import argparse
import json
import os
import sys

from collections import namedtuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import json_io  # noqa: E402  (パス追加後に import)
from aero_demo import palm_plane  # noqa: E402
from aero_demo import vector_utils  # noqa: E402

_unit = vector_utils.unit
load_skeleton_json = json_io.load_skeleton_json
iter_skeleton_files = json_io.iter_json_files


# --- 差し出している手の判定に使う定数 -------------------------------------
# 既定のロボット手先位置: 人物から +x に 3.0 m、高さ 1.2 m (ワールド座標)。
ROBOT_FORWARD_DISTANCE = 3.0
ROBOT_HAND_HEIGHT = 1.2

# 特徴量 (0..1) の重み。合計 1.0。合成骨格と実カメラで共通。
OFFER_FEATURE_WEIGHTS = {
    'approach': 0.25,
    'separation': 0.05,
    'finger_to_robot': 0.55,
    'thumb_roll': 0.15,
}

# face_to_robot は重み付き和でなく減点:
#     score -= FACE_AWAY_PENALTY * (1 - face_to_robot)
FACE_AWAY_PENALTY = 0.15

# 各特徴量のランプ (下限, 上限)。下限以下で 0、上限以上で 1。距離は [m]。
APPROACH_RAMP = (0.0, 0.45)
SEPARATION_RAMP = (0.35, 0.55)
FINGER_TO_ROBOT_RAMP = (0.0, 0.90)
# 親指が明らかに下向きの不自然な差し出しを弾く。
THUMB_ROLL_RAMP = (-0.20, 0.30)
FACE_TO_ROBOT_RAMP = (-0.20, 0.50)

# thumb_roll の基準 (人体軸ではなく世界の鉛直)。
WORLD_UP = np.array([0.0, 0.0, 1.0])

# これ未満のスコアしか無ければ offered_hand=None (誤って掴みに行かない側に倒す)。
OFFER_SCORE_MIN = 0.86
# 左右のスコア差がこれ未満なら ambiguous。
AMBIGUOUS_MARGIN = 0.08

# 掌の静止判定 (select に t を渡したときだけ効く)。
STILLNESS_WINDOW = 0.5  # [s]
STILLNESS_MAX_DISPLACEMENT = 0.05  # [m]
# 履歴のサンプル間隔がこれを超えたら履歴を捨てる [s]。
STILLNESS_MAX_GAP = 0.4

# 欠測した関節の補完に使う人体比。
_TORSO_PER_SHOULDER_WIDTH = 1.25
_ARM_PER_TORSO = 1.15
_DEFAULT_SHOULDER_WIDTH = 0.40


_BodyFrame = namedtuple('_BodyFrame', [
    'up',               # (3,) 腰 -> 肩 の体軸 (単位ベクトル)
    'shoulder_center',  # (3,) 両肩の中点
    'hip_center',       # (3,) 両腰の中点
    'torso',            # float, 腰 -> 肩 の距離 [m]
])


def _ramp(value, low, high):
    """``low`` 以下で 0.0、``high`` 以上で 1.0 になる線形ランプ."""
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def _distance_to_segment(point, end_a, end_b):
    """``point`` から線分 ``end_a``--``end_b`` までの距離 [m]."""
    along = end_b - end_a
    length_sq = float(np.dot(along, along))
    if length_sq < 1e-12:
        return float(np.linalg.norm(point - end_a))
    t = float(np.dot(point - end_a, along)) / length_sq
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(point - (end_a + t * along)))


def _body_frame(joints):
    """関節位置から :class:`_BodyFrame` を作る.

    片肩欠けは ``Neck`` で、両腰欠けは +z 軸と肩幅からの胴長で補う。
    両肩とも無い・縮退時は ``None``。
    """
    r_sho = joints.get('RShoulder')
    l_sho = joints.get('LShoulder')
    if r_sho is not None and l_sho is not None:
        shoulder_center = 0.5 * (r_sho + l_sho)
    elif r_sho is not None or l_sho is not None:
        neck = joints.get('Neck')
        shoulder_center = neck if neck is not None else (
            r_sho if r_sho is not None else l_sho)
    else:
        return None

    r_hip = joints.get('RHip')
    l_hip = joints.get('LHip')
    hip_center = None
    up = None
    if r_hip is not None and l_hip is not None:
        hip_center = 0.5 * (r_hip + l_hip)
        up = _unit(shoulder_center - hip_center)
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
        if r_sho is not None and l_sho is not None:
            shoulder_width = float(np.linalg.norm(l_sho - r_sho))
        else:
            shoulder_width = _DEFAULT_SHOULDER_WIDTH
        torso = _TORSO_PER_SHOULDER_WIDTH * shoulder_width
        hip_center = shoulder_center - torso * up
    else:
        torso = float(np.linalg.norm(shoulder_center - hip_center))
    if torso < 1e-6:
        return None
    return _BodyFrame(up=up, shoulder_center=shoulder_center,
                      hip_center=hip_center, torso=torso)


def _midpoint(joints, name_a, name_b):
    """2 関節の中点。どちらかが欠けていれば ``None``."""
    point_a = joints.get(name_a)
    point_b = joints.get(name_b)
    if point_a is None or point_b is None:
        return None
    return 0.5 * (point_a + point_b)


def _face_frame(joints, body):
    """顔の前方向 (単位ベクトル) と顔の位置の組を返す。作れなければ ``None``.

    「両耳の中点 -> 鼻」を優先し、無ければ左右軸 x 体軸 (pitch は取れない)。
    """
    nose = joints.get('Nose')
    ear_center = _midpoint(joints, 'REar', 'LEar')
    eye_center = _midpoint(joints, 'REye', 'LEye')

    # 顔の位置: 耳の中点 > 目の中点 > 鼻 > 首 > 肩の中点。
    position = ear_center
    for candidate in (eye_center, nose, joints.get('Neck'),
                      body.shoulder_center):
        if position is not None:
            break
        position = candidate

    forward = None
    base = ear_center if ear_center is not None else eye_center
    if nose is not None and base is not None:
        forward = _unit(nose - base)
    if forward is None:
        # 左右軸 (人物の左方向)。
        lateral = None
        if 'REar' in joints and 'LEar' in joints:
            lateral = joints['LEar'] - joints['REar']
        elif 'REye' in joints and 'LEye' in joints:
            lateral = joints['LEye'] - joints['REye']
        if lateral is not None:
            forward = _unit(np.cross(lateral, body.up))
    if forward is None or position is None:
        return None
    return forward, position


def _shoulder_position(joints, side, body):
    """``side`` の肩の位置。欠けていれば反対の肩を ``Neck`` で鏡映して推定."""
    own = joints.get('{}Shoulder'.format(side))
    if own is not None:
        return own
    other_side = 'L' if side == 'R' else 'R'
    other = joints.get('{}Shoulder'.format(other_side))
    neck = joints.get('Neck')
    if other is not None and neck is not None:
        return 2.0 * neck - other
    return body.shoulder_center


def _arm_length(joints, side, torso):
    """上腕 + 前腕の長さ [m]。肘か手首が欠ければ胴長から補う."""
    shoulder = joints.get('{}Shoulder'.format(side))
    elbow = joints.get('{}Elbow'.format(side))
    wrist = joints.get('{}Wrist'.format(side))
    if shoulder is not None and elbow is not None and wrist is not None:
        arm = float(np.linalg.norm(elbow - shoulder)
                    + np.linalg.norm(wrist - elbow))
        if arm > 1e-6:
            return arm
    return _ARM_PER_TORSO * torso


class OfferedHandSelector(object):
    """左右の掌のうち、人が手繋ぎのために差し出している方を選ぶ.

    左右それぞれ 4 特徴量の重み付き和から顔の向きの減点を引いたスコアを
    付け、``score_min`` 以上の側の argmax を採る (無ければ ``None``)。

    特徴量 (距離は身長で正規化しない [m]):
      ``approach``: 脱力して垂らした掌に比べてロボットへどれだけ近づいたか。
      ``separation``: 掌と胴体 (腰-肩の線分) の距離。
      ``finger_to_robot``: 指先方向と掌->ロボット方向の cos。
      ``thumb_roll``: 指先軸まわりのロール (親指が上=1, 水平=0, 下=-1)。
      ``face_to_robot``: 顔の前方向と顔->ロボット方向の cos (減点のみ。
      顔が取れなければ 1.0 扱い)。

    ``select`` に時刻 ``t`` を渡すと掌の静止判定も加わる。掌位置の履歴を
    持つので人物ごとに別インスタンスを使うこと。
    """

    def __init__(self, robot_position=None, side_prior=None, weights=None,
                 score_min=OFFER_SCORE_MIN,
                 ambiguous_margin=AMBIGUOUS_MARGIN,
                 face_away_penalty=FACE_AWAY_PENALTY,
                 max_distance=None,
                 finger_to_robot_axis_blend=0.0,
                 finger_to_robot_ramp=FINGER_TO_ROBOT_RAMP,
                 approach_ramp=APPROACH_RAMP,
                 approach_height_scale=1.0,
                 stillness_window=STILLNESS_WINDOW,
                 stillness_max_displacement=STILLNESS_MAX_DISPLACEMENT,
                 stillness_max_gap=STILLNESS_MAX_GAP):
        """
        Parameters
        ----------
        robot_position : (3,) array_like or None
            ロボット手先のワールド座標。``None`` なら人物ごとに
            :meth:`_robot_position` の既定位置。
        side_prior : dict or None
            ``{'R': float, 'L': float}``。スコアに足す事前分布。
        max_distance : float or None
            人物 (腰) からロボットまでがこれを超えたら両手 ``veto='too_far'``。
        finger_to_robot_axis_blend : float
            ``finger_to_robot`` の軸を指先 (0.0) から掌の法線 (1.0) へ寄せる率。
        approach_height_scale : float
            ``approach`` の距離計算で z 差分に掛ける倍率 (0 で水平のみ)。
        stillness_window : float
            0.0 以下なら静止判定をしない。
        """
        self.robot_position = (None if robot_position is None
                               else np.asarray(robot_position,
                                               dtype=np.float64))
        self.side_prior = dict(side_prior or {})
        self.weights = dict(weights or OFFER_FEATURE_WEIGHTS)
        self.score_min = float(score_min)
        self.ambiguous_margin = float(ambiguous_margin)
        self.face_away_penalty = float(face_away_penalty)
        self.max_distance = (None if max_distance is None
                             else float(max_distance))
        self.finger_to_robot_axis_blend = float(finger_to_robot_axis_blend)
        self.finger_to_robot_ramp = tuple(finger_to_robot_ramp)
        self.approach_ramp = tuple(approach_ramp)
        self.approach_height_scale = float(approach_height_scale)
        self.stillness_window = float(stillness_window)
        self.stillness_max_displacement = float(stillness_max_displacement)
        self.stillness_max_gap = float(stillness_max_gap)
        # 左右の掌位置の履歴 [(t, position), ...] (t 昇順)。
        self._palm_history = {'R': [], 'L': []}

    def reset_stillness(self):
        """掌の静止判定の履歴を捨てる (人物が入れ替わったとき等)."""
        self._palm_history = {'R': [], 'L': []}

    def select(self, joint_positions, palms, t=None):
        """どちらの手を繋ぐべきかを判定する.

        Parameters
        ----------
        joint_positions : dict
            関節名 -> [x, y, z]。
        palms : dict
            ``PalmPoseEstimator.estimate`` の ``{'R': palm, 'L': palm}``。
        t : float or None
            フレーム時刻 [s]。渡すと静止判定を行う (同じ ``t`` の再呼び出しは
            履歴を増やさない)。

        Returns
        -------
        dict
            ``side`` ('R'/'L'/None), ``scores``, ``features``, ``margin``,
            ``ambiguous``, ``veto`` (除外理由), ``distance`` (腰->ロボット
            [m]), ``still``, ``still_duration`` [s] (静止判定なしの側は None)。
        """
        joints = {name: np.asarray(p, dtype=np.float64)
                  for name, p in joint_positions.items()}
        body = _body_frame(joints)
        still, still_duration = self._update_stillness(palms, t)

        distance = None
        too_far = False
        if body is not None:
            distance = float(np.linalg.norm(
                body.hip_center - self._robot_position(body)))
            too_far = (self.max_distance is not None
                      and distance > self.max_distance)

        scores = {'R': None, 'L': None}
        features = {'R': None, 'L': None}
        veto = {'R': None, 'L': None}
        for side in ('R', 'L'):
            if too_far:
                veto[side] = 'too_far'
                continue
            palm = palms.get(side)
            if palm is None:
                veto[side] = 'no_palm'
                continue
            if body is None:
                veto[side] = 'no_body_frame'
                continue
            feats = self._features(joints, body, side, palm)
            features[side] = feats
            scores[side] = sum(w * feats[key]
                               for key, w in self.weights.items()) \
                - self.face_away_penalty * (1.0 - feats['face_to_robot']) \
                + float(self.side_prior.get(side, 0.0))

        candidates = {s: v for s, v in scores.items() if v is not None}
        margin = None
        if len(candidates) == 2:
            margin = abs(scores['R'] - scores['L'])
        side = None
        if candidates:
            best = max(candidates, key=lambda s: candidates[s])
            # 静止していなければ見送る (反対の手には切り替えない)。
            if (candidates[best] >= self.score_min
                    and still[best] is not False):
                side = best
        ambiguous = bool(side is not None and margin is not None
                         and margin < self.ambiguous_margin)
        return dict(side=side, scores=scores, features=features,
                    margin=margin, ambiguous=ambiguous, veto=veto,
                    distance=distance, still=still,
                    still_duration=still_duration)

    def _update_stillness(self, palms, t):
        """掌の位置を履歴に積み、``(still, still_duration)`` を返す."""
        still = {'R': None, 'L': None}
        still_duration = {'R': None, 'L': None}
        if t is None or self.stillness_window <= 0.0:
            return still, still_duration
        t = float(t)
        for side in ('R', 'L'):
            history = self._palm_history[side]
            palm = palms.get(side)
            if palm is None:
                # 一時的なロストでは履歴を残す (max_gap を超えたら捨てる)。
                if history and t - history[-1][0] > self.stillness_max_gap:
                    del history[:]
                continue
            position = np.asarray(palm['position'], dtype=np.float64)
            if history and t < history[-1][0]:
                # 時刻が巻き戻った (rosbag のループ再生等)。
                del history[:]
            if history and t - history[-1][0] > self.stillness_max_gap:
                del history[:]
            if history and t == history[-1][0]:
                history[-1] = (t, position)
            else:
                history.append((t, position))
            # 窓の始点以前のサンプルは 1 つだけ残す。
            window_start = t - self.stillness_window
            while len(history) >= 2 and history[1][0] <= window_start:
                del history[0]

            # 現在位置からのずれが閾値以内に収まり続けている時間。
            since = t
            for sample_t, sample_position in reversed(history):
                if (np.linalg.norm(sample_position - position)
                        > self.stillness_max_displacement):
                    break
                since = sample_t
            still_duration[side] = t - since
            # 1e-6 はタイムスタンプの浮動小数点誤差ぶん。
            still[side] = (still_duration[side]
                           >= self.stillness_window - 1e-6)
        return still, still_duration

    def _robot_position(self, body):
        """ロボット手先のワールド座標 (既定は腰の xy から +x、高さ固定)."""
        if self.robot_position is not None:
            return self.robot_position
        return np.array([body.hip_center[0] + ROBOT_FORWARD_DISTANCE,
                         body.hip_center[1],
                         ROBOT_HAND_HEIGHT])

    def _offer_direction(self, finger, palm_normal):
        """``finger_to_robot`` の軸 (指先と掌の法線のブレンド)."""
        if finger is None and palm_normal is None:
            return None
        blend = self.finger_to_robot_axis_blend
        if finger is None or blend >= 1.0:
            return palm_normal if palm_normal is not None else finger
        if palm_normal is None or blend <= 0.0:
            return finger
        return _unit((1.0 - blend) * finger + blend * palm_normal)

    def _robot_distance(self, robot, point):
        """``approach`` 用の距離 [m] (z 差分に ``approach_height_scale``)."""
        diff = robot - point
        diff = np.array([diff[0], diff[1],
                         diff[2] * self.approach_height_scale])
        return float(np.linalg.norm(diff))

    def _features(self, joints, body, side, palm):
        """片手ぶんの特徴量 (クラス docstring 参照) を計算する."""
        center = np.asarray(palm['position'], dtype=np.float64)
        finger = _unit(palm['x_axis'])    # 手首 -> 指先
        offer_dir = self._offer_direction(finger, _unit(palm['y_axis']))
        shoulder = _shoulder_position(joints, side, body)
        arm = _arm_length(joints, side, body.torso)
        robot = self._robot_position(body)

        # 脱力して真下に垂れた掌の位置。
        rest = shoulder - arm * body.up
        approach = _ramp(self._robot_distance(robot, rest)
                         - self._robot_distance(robot, center),
                         *self.approach_ramp)
        separation = _ramp(
            _distance_to_segment(center, body.hip_center,
                                 body.shoulder_center),
            *SEPARATION_RAMP)
        to_robot = _unit(robot - center)
        if offer_dir is None or to_robot is None:
            finger_to_robot = 0.0
        else:
            finger_to_robot = _ramp(float(np.dot(offer_dir, to_robot)),
                                    *self.finger_to_robot_ramp)
        thumb_roll = _ramp(self._thumb_roll(palm, side, finger),
                           *THUMB_ROLL_RAMP)
        face_to_robot = _ramp(self._face_to_robot(joints, body, robot),
                              *FACE_TO_ROBOT_RAMP)

        return dict(approach=approach, separation=separation,
                    finger_to_robot=finger_to_robot, thumb_roll=thumb_roll,
                    face_to_robot=face_to_robot)

    @staticmethod
    def _face_to_robot(joints, body, robot):
        """顔の前方向と顔->ロボット方向の cos。顔が取れなければ 1.0."""
        face = _face_frame(joints, body)
        if face is None:
            return 1.0
        forward, position = face
        to_robot = _unit(robot - position)
        if to_robot is None:
            return 1.0
        return float(np.dot(forward, to_robot))

    @staticmethod
    def _thumb_roll(palm, side, finger):
        """指先軸まわりのロール: 親指がどれだけ上を向いているか (-1..1).

        親指方向は右手で ``z_axis``、左手で ``-z_axis`` (fit_palm_plane の
        法線の決め方による)。指が鉛直で定義できなければ 1.0。
        """
        z_axis = np.asarray(palm['z_axis'], dtype=np.float64)
        thumb = z_axis if side == 'R' else -z_axis
        if finger is None:
            return 1.0
        up_perp = _unit(WORLD_UP - float(np.dot(WORLD_UP, finger)) * finger)
        if up_perp is None:
            return 1.0
        return float(np.dot(thumb, up_perp))


def format_offer_scores(selection, score_min):
    """``OfferedHandSelector.select`` の結果をログ用の 1 行にする."""
    parts = ['差し出し手判定 (閾値 {:.2f})'.format(score_min)]
    if selection['distance'] is not None:
        parts.append('距離={:.2f}m'.format(selection['distance']))
    still_duration = selection.get('still_duration') or {}
    for side in ('R', 'L'):
        veto = selection['veto'][side]
        score = selection['scores'][side]
        if veto is not None:
            parts.append('{}=判定不可({})'.format(side, veto))
        else:
            text = '{}={:.2f}'.format(side, score)
            if still_duration.get(side) is not None:
                text += '(静止{:.2f}s)'.format(still_duration[side])
            parts.append(text)
    return ', '.join(parts)


class PalmPoseEstimator(object):
    """骨格から左右の掌の位置姿勢と、手繋ぎに使う手を推定する."""

    def __init__(self, offered_hand_selector=None):
        self.offered_hand_selector = \
            offered_hand_selector or OfferedHandSelector()

    def estimate(self, joint_positions, t=None):
        """左右の掌の位置姿勢と、手繋ぎに使うべき手を推定する.

        Parameters
        ----------
        joint_positions : dict
            関節名 -> [x, y, z] (ロボット座標系)。
        t : float or None
            フレーム時刻 [s]。渡すと静止判定も行う。

        Returns
        -------
        dict
            ``{'R': palm, 'L': palm, 'offered_hand': 'R'/'L'/None}``。
            ``palm`` は ``position``, ``x_axis``/``y_axis``/``z_axis``,
            ``rot`` (3 軸を列に並べた 3x3) を持つ dict、推定不可なら ``None``。
        """
        joints = {name: np.asarray(p, dtype=np.float64)
                 for name, p in joint_positions.items()}
        palms = {side: self._estimate_one(joints, side) for side in ('R', 'L')}
        selection = self.offered_hand_selector.select(joints, palms, t)
        result = dict(palms)
        result['offered_hand'] = selection['side']
        return result

    def estimate_palm(self, joint_positions, side):
        """``side`` ('R'/'L') の掌 1 つだけを推定する (差し出し判定なし)."""
        joints = {name: np.asarray(p, dtype=np.float64)
                 for name, p in joint_positions.items()}
        return self._estimate_one(joints, side)

    def _estimate_one(self, joints, side):
        points = {}
        for i in palm_plane.PLANE_LANDMARKS:
            key = '{}Hand{}'.format(side, i)
            if key in joints:
                points[i] = joints[key]
        plane = palm_plane.fit_palm_plane(points, hand=side)
        if plane is None:
            return None

        # plane.rot はロボット手先向け (+Y = -normal) なので、人の掌フレーム
        # (+y = normal) はここで組み直す。
        x_axis = plane.finger_dir
        y_axis = plane.normal
        z_axis = np.cross(x_axis, y_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis])
        return dict(
            position=[float(v) for v in plane.center],
            x_axis=[float(v) for v in x_axis],
            y_axis=[float(v) for v in y_axis],
            z_axis=[float(v) for v in z_axis],
            rot=[[float(v) for v in row] for row in rot])


def save_json(palms, path, keep_keys=('human_label',)):
    """``estimate`` の結果を保存する。既存 JSON の ``keep_keys`` は引き継ぐ."""
    saved = dict(palms)
    if os.path.exists(path):
        with open(path) as f:
            previous = json.load(f)
        for key in keep_keys:
            if key in previous:
                saved[key] = previous[key]
    json_io.save_json(path, saved)


def main():
    parser = argparse.ArgumentParser(
        description='骨格 JSON から左右の掌の位置姿勢と差し出し手を推定し '
                    'JSON に保存する。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='骨格 JSON の入力ディレクトリ。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_palm_poses'),
        help='掌の位置姿勢 JSON の保存先ディレクトリ。')
    args = parser.parse_args()

    files = iter_skeleton_files(args.input_dir)
    if not files:
        print('{} に骨格 JSON が見つかりません。先に '
              'generate_random_human_poses.py を実行してください。'.format(
                  args.input_dir))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    estimator = PalmPoseEstimator()

    for i, path in enumerate(files):
        joint_positions = load_skeleton_json(path)
        palms = estimator.estimate(joint_positions)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        save_json(palms, out_path)
        offered = palms['offered_hand']
        print('[{}/{}] saved {} (offered_hand: {})'.format(
            i + 1, len(files), out_path,
            offered if offered is not None else 'none'))


if __name__ == '__main__':
    main()
