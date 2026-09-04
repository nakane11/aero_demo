#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人体骨格 (MediaPipe 形式の関節位置の dict) を入力とし、左右の掌の位置
姿勢と、そのうちどちらの手を手繋ぎに使うべきか (人が差し出している手) を
推定して JSON として保存する。

``RandomSkeletonGenerator`` (``generate_random_human_poses.py``) が出力
する合成骨格でも、``fake_people_pose_estimator_ros.
FakeRosPeoplePoseEstimator`` のような実カメラ (MediaPipe) ベースの
推定を骨格の生成元として使う場合でも、同じ
``PalmPoseEstimator.estimate(joint_positions)`` で掌の位置姿勢を求められる
ようにしてある。骨格の生成元が変わっても入力形式 (関節名 -> [x, y, z] の
dict) は変わらないので、この推定器は生成元を一切区別しない。

手のランドマーク (MediaPipe の ``RHand0``..``RHand20`` / ``LHand*``,
``RandomSkeletonGenerator(include_hand=True)`` や実際の MediaPipe 推定が
出力する) が 3 点以上揃っている側だけ、``aero_demo.palm_plane.
fit_palm_plane`` で手首 + 知節 (MCP) へ平面を SVD フィットする。
実際にロボットを動かすときと同じ関数・同じ判定 (点がほぼ一直線でないか
等) を使うので、実カメラの推定結果
(関節ごとに欠測がある、本物の手首の捻りが乗っている、等) に対しても同じ
ロジックで動く。手のランドマークが 3 点未満の側 (体の関節だけの入力、
または実推定でその手が丸ごとロストしたフレーム) は推定しない
(``None``) -- 指のランドマークを見ずに前腕の向きだけから掌の姿勢を仮定
するようなフォールバックは持たない。

ローカル座標系は
    +x : 指先方向 (手首 -> 指先)
    +y : 手の甲 -> 掌の方向 (掌の法線, 体の外側を向く)
    +z : +x と +y に直交する軸 (+x, +y に対して右手系になるように
         ``z = x cross y`` で決める)
とし、手は左右 (``R``/``L``) を区別して独立に推定する。

実際に手を繋ぐときは左右どちらか一方を選ぶ必要があるので、
``OfferedHandSelector`` が「人がどちらの手を差し出しているか」を判定し、
``PalmPoseEstimator.estimate`` の戻り値 (と保存する JSON) に
``offered_hand`` (``'R'`` / ``'L'`` / ``None``) として入れる。判定に使う
特徴量・重みの詳細は ``OfferedHandSelector`` の docstring を参照。

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
# ロボット (相手) の位置。人物のいるワールド座標から +x に
# ROBOT_FORWARD_DISTANCE だけ離れた場所に、握手のために手を出す高さ
# ROBOT_HAND_HEIGHT で立っていると仮定する (合成骨格では人物が常に原点に
# 立つので、ロボットの手は概ね [3.0, 0.0, 1.2])。人物の体格からではなく
# ワールド座標で決めるので、低身長の人がロボットの手の高さに合わせて腕を
# 上に伸ばした姿勢も「ロボットに近づいた」として拾える。実際のロボットの
# 位置が分かっているなら ``OfferedHandSelector(robot_position=...)`` で
# 上書きする。
ROBOT_FORWARD_DISTANCE = 3.0
ROBOT_HAND_HEIGHT = 1.2

# 各特徴量 (すべて 0..1 に正規化済み) の重み。合計 1.0 なのでスコアも
# 0..1 に収まる。``approach`` と ``finger_to_robot`` (ロボットにどれだけ
# 近づき、どれだけロボットを指しているか) が主な手がかりで、
# ``separation`` と ``thumb_roll`` は単独での判別力が低いため補助的な
# 重みにしてある。
OFFER_FEATURE_WEIGHTS = {
    'approach': 0.375,
    'separation': 0.10,
    'finger_to_robot': 0.375,
    'thumb_roll': 0.15,
}

# ``face_to_robot`` (顔がロボットの方を向いているか) だけは重み付き和に
# 足し込まず、「向いていない分」を引く減点として使う
#     score = sum(重み付き和) - FACE_AWAY_PENALTY * (1 - face_to_robot)
# 顔は 1 つしか無いので左右で同じ値になり、どちらの手かの順位付けには効かず
# 「差し出していない」(null) との区別だけに効く -- 重み付き和の一項にすると
# 他の 4 つの重みを薄める (= どちらの手かの判定を鈍らせる) だけなので、
# 減点にして「ロボットを見ている人物のスコアはこれまでと完全に同じ」に
# なるようにする。満点との差が最大でも 0.15 = OFFER_SCORE_MIN との距離ぶん
# なので、「顔を背けている」だけで必ず null になるわけではない。
FACE_AWAY_PENALTY = 0.15

# 各特徴量のランプ (下限, 上限)。下限以下で 0、上限以上で 1。距離は腕長・
# 胴長で正規化しない絶対値 [m] なので、ランプの値もそのまま [m]。
# approach: 脱力して真下に垂れた掌からロボットまでの距離が、実際の掌の
#        位置でどれだけ縮んだか [m]。腕を下ろしていれば ~0、ロボットの方へ
#        まっすぐ差し出せば ~0.5 m。手を後ろに引けば負になる。
APPROACH_RAMP = (0.0, 0.45)
# separation: 掌が人物自身の胴体 (腰の中点 -> 肩の中点の線分) からどれだけ
#        離れているか [m]。体側に下ろした手でも肩幅の半分 (~0.2 m) は離れて
#        いるので、下限はそれより大きく取る。
SEPARATION_RAMP = (0.35, 0.55)
# finger_to_robot: 指先方向と「掌からロボットへの方向」の cos。ロボットを
#        まっすぐ指していれば 1、真逆を向いていれば -1。
FINGER_TO_ROBOT_RAMP = (0.0, 0.90)
# thumb_roll: 掌の x 軸 (指先方向) まわりのロール。親指が真上なら 1、水平
#        なら 0、真下 (掌が体の外を向き小指が親指より上) なら -1。
#        「水平より下」全体を罰すると、人が実際に差し出している手まで巻き
#        込んでしまうため、下限を -1.0 側に寄せて「明らかに手の甲が上」の
#        極端な姿勢だけを罰するようにしてある。
THUMB_ROLL_RAMP = (-1.00, -0.60)
# face_to_robot: 顔の前方向と「顔からロボットへの方向」の cos。ロボットを
#        まっすぐ見ていれば 1、真横を向いていれば 0、背を向けていれば -1。
#        「握手のときに必ず相手の顔を見ている」とまでは言えない (差し出した
#        手元を見ている、少し横を向いている) ので基準は緩めに取り、真横から
#        60 deg ほど内側 (cos 0.5) を向いていれば満点、真横より更に 10 deg
#        ほど背けた (cos -0.2) ところで 0 にする。顔のランドマークが足りない
#        フレームでは罰しない (1.0 扱い, ``_face_to_robot``)。
FACE_TO_ROBOT_RAMP = (-0.20, 0.50)

# ロール角を測る基準の鉛直方向 (ワールド座標系)。「水平より下」は人物の
# 体軸ではなく世界の水平を基準にした言い方なので、体軸ではなくこれを使う。
WORLD_UP = np.array([0.0, 0.0, 1.0])

# これを下回るスコアしか無ければ「どちらの手も差し出していない」と判定
# する (``offered_hand`` は None)。腕を下ろしたまま立っている人物 --
# generate_random_human_poses.py の肩の仰角は「腕を下ろした状態」を最頻値
# とする三角分布なので最も多いケース -- を argmax で拾ってしまわないため。
# 実機では「差し出していない手を掴みに行く」方が「差し出した手を見送る」
# より危ないので、迷ったら高めに (取りこぼす側に) 振る。
OFFER_SCORE_MIN = 0.86
# 左右のスコア差がこれ未満なら曖昧 (``select`` の戻り値の ``ambiguous``)。
# それでも argmax は返すので、呼び出し側 (右手しか差し出せない Aero の
# 到達性など、この判定器が知らない事情を持つ側) が覆せる。
AMBIGUOUS_MARGIN = 0.08

# 関節が欠測した入力 (実カメラで下半身がフレーム外、など) で胴長・腕長を
# 補うための人体比。身長 1.7 m の成人で 肩幅 ~0.40 m, 腰->肩 ~0.50 m,
# 上腕+前腕 ~0.58 m 程度。
_TORSO_PER_SHOULDER_WIDTH = 1.25
_ARM_PER_TORSO = 1.15


# 人体基準の座標系。「脱力して真下に垂れた腕」の位置 (体軸 ``up`` と肩) と
# 「人物自身の胴体」(``hip_center`` -> ``shoulder_center`` の線分) を作るのに
# 使う。人物の正面方向は要らない -- 相手の方向はロボットのワールド座標
# (ROBOT_FORWARD_DISTANCE / ROBOT_HAND_HEIGHT) から直接決めるので、人物が
# ロボットに背を向けている姿勢もそのまま「ロボットを向いていない」として
# 扱える。
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
    """関節位置から人体基準の座標系 (:class:`_BodyFrame`) を作る.

    両肩が要る。両腰が無い場合はロボット座標系の +z を体軸とみなし、胴長
    を肩幅から補う (実カメラで下半身が映っていないフレーム向けの保険で、
    合成骨格では常に両腰が揃っているのでこの経路は通らない)。

    Returns
    -------
    _BodyFrame or None
        両肩が無い、または体格が縮退している場合は ``None``。
    """
    r_sho = joints.get('RShoulder')
    l_sho = joints.get('LShoulder')
    if r_sho is None or l_sho is None:
        return None
    shoulder_center = 0.5 * (r_sho + l_sho)

    r_hip = joints.get('RHip')
    l_hip = joints.get('LHip')
    hip_center = None
    up = None
    if r_hip is not None and l_hip is not None:
        hip_center = 0.5 * (r_hip + l_hip)
        up = _unit(shoulder_center - hip_center)
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
        torso = _TORSO_PER_SHOULDER_WIDTH * float(np.linalg.norm(l_sho - r_sho))
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
    """顔の前方向と、その基準点 (顔の位置) を返す.

    第一候補は「両耳の中点 -> 鼻」で、首をひねった向き (yaw) だけでなく
    うつむき・見上げ (pitch) も乗る。鼻か耳が欠けているフレームでは、
    左右軸 (耳、無ければ目) と体軸の外積 ``前方 = 左 x 上`` で向きだけを
    作る (この経路では pitch は取れない)。

    Returns
    -------
    (forward, position) or None
        ``forward`` は顔の前方向 (単位ベクトル)、``position`` は顔の位置。
        顔のランドマークが足りず向きが作れなければ ``None``。
    """
    nose = joints.get('Nose')
    ear_center = _midpoint(joints, 'REar', 'LEar')
    eye_center = _midpoint(joints, 'REye', 'LEye')

    # 顔の位置。耳の中点 (頭の中心に一番近い) > 目の中点 > 鼻 > 首 の順。
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
        # 左右軸 (左耳/左目 - 右耳/右目 = 人物の左方向)。
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

    掌の位置姿勢 (``PalmPoseEstimator.estimate`` が返すもの) と体の関節
    だけから、左右それぞれに 0..1 のスコア (4 つの特徴量の重み付き和から、
    顔をロボットに向けていない分の減点を引いたもの) を付け、``score_min``
    を超えた
    側の argmax を採る。どちらも超えなければ「差し出していない」
    (``side`` は ``None``) と判定する。

    特徴量 (重み付き和に入る 4 つ + 減点として効く ``face_to_robot``。
    距離は腕長・胴長で正規化しない絶対値 [m])
    ------------------------------------------------------------------
    ``approach``
        脱力して真下に垂れた掌 (肩から体軸方向に腕長ぶん下げた点) から
        ロボットまでの距離が、実際の掌の位置でどれだけ縮んだか [m]。
        「手を差し出す」を「垂らした手をロボットに近づける」と読み替えた
        もので、腕を下ろしていれば ~0、まっすぐ差し出せば ~0.5 m、手を
        後ろに引けば負になる。
    ``separation``
        掌が人物自身の胴体 (腰の中点 -> 肩の中点の線分) からどれだけ
        離れているか [m]。体に付けたままの手を落とすための補助。
    ``finger_to_robot``
        指先方向 (``x_axis``) と「掌からロボットへの方向」の cos。ロボット
        をまっすぐ指していれば 1、真逆を向いていれば -1。
    ``thumb_roll``
        掌の x 軸 (指先方向) まわりのロール姿勢。親指の向きと、鉛直方向を
        x 軸に直交する平面へ射影した向きとの cos で測る (x 軸まわりの回転
        だけを見るので、指がどこを向いているかには影響されない)。親指が
        真上なら 1、水平なら 0、真下 -- 掌が体の中心から外を向き、小指が
        親指より上に来る姿勢 -- なら -1。握手として無理な向きに捻れた手を
        落とすための補助。
    ``face_to_robot``
        顔の前方向 (両耳の中点 -> 鼻, :func:`_face_frame`) と「顔から
        ロボットへの方向」の cos。ロボットをまっすぐ見ていれば 1、真横を
        向いていれば 0、背を向けていれば -1。ロボットに顔を向けていない
        人物 (よそ見・後ろ向き) を落とすための補助なので基準は緩め
        (:data:`FACE_TO_ROBOT_RAMP`)。顔は 1 つしか無いので左右で同じ値に
        なり、どちらの手かの順位付けには効かず「差し出していない」との
        区別だけに効く -- なので重み付き和の一項ではなく、満点からの不足
        ぶんをスコアから引く減点 (:data:`FACE_AWAY_PENALTY`) として使う。
        顔のランドマークが足りないフレーム (実カメラで顔が映っていない等)
        では罰しない側に倒して 1.0 扱いにする。

    距離を身長で正規化しないのは意図的で、低身長の人がロボットの手の高さに
    合わせて腕を上に伸ばすような姿勢を、腕長・胴長での割り算で潰して
    しまわないため。ロボットの位置はワールド座標で与える (既定は
    ``ROBOT_FORWARD_DISTANCE`` / ``ROBOT_HAND_HEIGHT``) ので、人物が
    ロボットに背を向けている姿勢も「ロボットを向いていない」として自然に
    扱える。

    掌の推定に失敗した側 (``palm`` が ``None``) と、体の座標系が作れない
    入力 (両肩が無い等) だけは、重み付き和を計算するまでもなく候補から
    外す。それ以外の足切りは持たず、``score_min`` だけで「差し出して
    いない」を決める。

    Examples
    --------
    >>> selector = OfferedHandSelector()
    >>> selection = selector.select(joint_positions, palms)
    >>> selection['side']
    'L'
    >>> selection['scores']
    {'R': 0.34, 'L': 0.71}
    """

    def __init__(self, robot_position=None, side_prior=None, weights=None,
                 score_min=OFFER_SCORE_MIN,
                 ambiguous_margin=AMBIGUOUS_MARGIN,
                 face_away_penalty=FACE_AWAY_PENALTY):
        """
        Parameters
        ----------
        robot_position : (3,) array_like or None
            相手 (ロボット) の手先のワールド座標。``approach`` と
            ``finger_to_robot`` の基準になる。``None`` (既定) なら人物の
            いる位置から +x に :data:`ROBOT_FORWARD_DISTANCE`、高さ
            :data:`ROBOT_HAND_HEIGHT` の点を人物ごとに使う。
        side_prior : dict or None
            ``{'R': float, 'L': float}``。スコアに直接足し込む事前分布。
            既定 (``None``) は左右とも 0.0。例えば Aero は右手しか差し
            出せない (``aero_demo.right_hand_offer``) ので、対面では人の
            左手の方が正対しやすい、といった事情を入れたい場合に使う。
            データセットに偏りを入れないよう既定では効かせない。
        weights : dict or None
            特徴量の重み。既定は :data:`OFFER_FEATURE_WEIGHTS`
            (``face_to_robot`` は含まない -- 下記 ``face_away_penalty``)。
        score_min : float
            これを超えるスコアが無ければ ``side`` は ``None``。
        ambiguous_margin : float
            左右のスコア差がこれ未満なら ``ambiguous`` を立てる。
        face_away_penalty : float
            顔をロボットに向けていない分 (``1 - face_to_robot``) に掛けて
            スコアから引く減点の大きさ。既定は
            :data:`FACE_AWAY_PENALTY`。0.0 にすれば顔向きを一切見なくなる。
        """
        self.robot_position = (None if robot_position is None
                               else np.asarray(robot_position,
                                               dtype=np.float64))
        self.side_prior = dict(side_prior or {})
        self.weights = dict(weights or OFFER_FEATURE_WEIGHTS)
        self.score_min = float(score_min)
        self.ambiguous_margin = float(ambiguous_margin)
        self.face_away_penalty = float(face_away_penalty)

    def select(self, joint_positions, palms):
        """どちらの手を繋ぐべきかを判定する.

        Parameters
        ----------
        joint_positions : dict
            関節名 (MediaPipe 形式) -> [x, y, z]。``PalmPoseEstimator.
            estimate`` に渡すものと同じ。
        palms : dict
            ``PalmPoseEstimator.estimate`` の戻り値 (``{'R': palm, 'L':
            palm}``、推定できなかった側は ``None``)。

        Returns
        -------
        dict
            ``side``
                ``'R'`` / ``'L'`` / ``None`` (どちらも差し出していない)。
                JSON に出すのはこれだけ。
            ``scores``
                左右のスコア (候補から外れた側は ``None``)。
            ``features``
                左右の特徴量の内訳 (閾値の調整・デバッグ用)。
            ``margin``
                左右のスコア差 (片側しか候補が無ければ ``None``)。
            ``ambiguous``
                左右のスコアが拮抗しているか。
            ``veto``
                候補から外した理由 (外していなければ ``None``)。
        """
        joints = {name: np.asarray(p, dtype=np.float64)
                  for name, p in joint_positions.items()}
        body = _body_frame(joints)

        scores = {'R': None, 'L': None}
        features = {'R': None, 'L': None}
        veto = {'R': None, 'L': None}
        for side in ('R', 'L'):
            palm = palms.get(side)
            if palm is None:
                veto[side] = 'no_palm'
                continue
            if body is None:
                veto[side] = 'no_body_frame'
                continue
            feats = self._features(joints, body, side, palm)
            features[side] = feats
            # ``face_to_robot`` だけは重み付き和ではなく減点で効かせる
            # (:data:`FACE_AWAY_PENALTY` のコメント参照)。
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
            if candidates[best] >= self.score_min:
                side = best
        ambiguous = bool(side is not None and margin is not None
                         and margin < self.ambiguous_margin)
        return dict(side=side, scores=scores, features=features,
                    margin=margin, ambiguous=ambiguous, veto=veto)

    def _robot_position(self, body):
        """ロボットの手先のワールド座標.

        ``robot_position`` を渡されていればそれを、渡されていなければ人物の
        いる位置 (腰の中点の x, y) から +x に ``ROBOT_FORWARD_DISTANCE``、
        高さ ``ROBOT_HAND_HEIGHT`` の点を返す。高さを人物の体格から決めない
        のがポイントで、こうすると低身長の人にとってロボットの手は「上に
        腕を伸ばして届かせる相手」になる。
        """
        if self.robot_position is not None:
            return self.robot_position
        return np.array([body.hip_center[0] + ROBOT_FORWARD_DISTANCE,
                         body.hip_center[1],
                         ROBOT_HAND_HEIGHT])

    def _features(self, joints, body, side, palm):
        """片手ぶんの特徴量 (クラス docstring 参照) を計算する."""
        center = np.asarray(palm['position'], dtype=np.float64)
        finger = _unit(palm['x_axis'])    # 手首 -> 指先
        shoulder = joints.get('{}Shoulder'.format(side), body.shoulder_center)
        arm = _arm_length(joints, side, body.torso)
        robot = self._robot_position(body)

        # 脱力して真下に垂れた掌の位置 (肩から体軸方向へ腕長ぶん下げた点)。
        # そこからロボットまでの距離が、実際の掌でどれだけ縮んだか。
        rest = shoulder - arm * body.up
        approach = _ramp(float(np.linalg.norm(robot - rest))
                         - float(np.linalg.norm(robot - center)),
                         *APPROACH_RAMP)
        # 掌が人物自身の胴体 (腰 -> 肩の線分) からどれだけ離れているか。
        separation = _ramp(
            _distance_to_segment(center, body.hip_center,
                                 body.shoulder_center),
            *SEPARATION_RAMP)
        # 指先がどれだけロボットの方を向いているか。
        to_robot = _unit(robot - center)
        if finger is None or to_robot is None:
            finger_to_robot = 0.0
        else:
            finger_to_robot = _ramp(float(np.dot(finger, to_robot)),
                                    *FINGER_TO_ROBOT_RAMP)
        thumb_roll = _ramp(self._thumb_roll(palm, side, finger),
                           *THUMB_ROLL_RAMP)
        # 顔がどれだけロボットの方を向いているか (左右で同じ値になる)。
        face_to_robot = _ramp(self._face_to_robot(joints, body, robot),
                              *FACE_TO_ROBOT_RAMP)

        return dict(approach=approach, separation=separation,
                    finger_to_robot=finger_to_robot, thumb_roll=thumb_roll,
                    face_to_robot=face_to_robot)

    @staticmethod
    def _face_to_robot(joints, body, robot):
        """顔がどれだけロボットの方を向いているか (-1..1).

        顔の前方向 (:func:`_face_frame`) と「顔からロボットへの方向」の
        cos。顔のランドマークが足りず向きが作れないときは、罰しない側に
        倒して 1.0 を返す (実カメラで顔だけロストしたフレームで、差し出して
        いる手を取り落とさないため)。
        """
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
        """掌の x 軸まわりのロール: 親指がどれだけ上を向いているか (-1..1).

        ``palm_plane.fit_palm_plane`` は親指側の横軸 ``v`` を使って法線の
        向きを決めており (人差し指 MCP が親指側, 小指 MCP が小指側)、
        右手では ``normal = v x x``、左手では ``normal = x x v`` になる。
        これを ``v`` について解くと ``v = x x normal`` (右手) /
        ``v = normal x x`` (左手)、すなわち掌フレームの ``z_axis``
        (``= x cross y``, ``y = normal``) がそのまま右手の親指方向、その
        符号を反転したものが左手の親指方向になる。

        鉛直方向を x 軸に直交する平面へ射影してから測るので、指がどこを
        向いていても「x 軸まわりの回転」だけを見ることになる。指がちょうど
        鉛直でロールが定義できないときは、罰しない側に倒して 1.0 を返す。
        """
        z_axis = np.asarray(palm['z_axis'], dtype=np.float64)
        thumb = z_axis if side == 'R' else -z_axis
        if finger is None:
            return 1.0
        up_perp = _unit(WORLD_UP - float(np.dot(WORLD_UP, finger)) * finger)
        if up_perp is None:
            return 1.0
        return float(np.dot(thumb, up_perp))


class PalmPoseEstimator(object):
    """骨格 (関節位置の dict) から左右の掌の位置姿勢と、手繋ぎに使う手を
    推定する.

    Examples
    --------
    >>> pose = load_skeleton_json('human_000.json')
    >>> estimator = PalmPoseEstimator()
    >>> palms = estimator.estimate(pose)
    >>> palms['R']['position']
    [0.32, -0.18, 0.95]
    >>> palms['offered_hand']
    'L'
    """

    def __init__(self, offered_hand_selector=None):
        """
        Parameters
        ----------
        offered_hand_selector : OfferedHandSelector or None
            手繋ぎに使う手の判定器。``None`` (既定) なら既定設定の
            :class:`OfferedHandSelector` を作る。実際のロボットの手先位置を
            ``robot_position`` に入れたい場合などはここで差し替える。
        """
        self.offered_hand_selector = \
            offered_hand_selector or OfferedHandSelector()

    def estimate(self, joint_positions):
        """左右の掌の位置姿勢と、手繋ぎに使うべき手を推定する.

        Parameters
        ----------
        joint_positions : dict
            関節名 (MediaPipe 形式) -> [x, y, z] (ロボット座標系)。骨格の
            生成元 (``RandomSkeletonGenerator`` / 実カメラの推定など) は
            問わない。

        Returns
        -------
        dict
            ``{'R': palm, 'L': palm, 'offered_hand': side}``。手のランド
            マーク (``{side}Hand0``..``{side}Hand20``) が 3 点未満の側の
            ``palm`` は ``None``。``palm`` は次のキーを持つ dict:

            ``position``
                掌中心の位置 [x, y, z]。
            ``x_axis`` / ``y_axis`` / ``z_axis``
                掌のローカル座標系の各軸 (単位ベクトル, ワールド座標系)。
                +x = 指先方向, +y = 手の甲->掌の方向, +z = x cross y。
            ``rot``
                上記 3 軸を列に並べた 3x3 回転行列 (``skrobot.coordinates.
                Coordinates(pos=position, rot=rot)`` にそのまま渡せる)。

            ``offered_hand`` は手繋ぎに使うべき手 (``'R'`` / ``'L'``)、
            どちらの手も差し出していなければ ``None``
            (:class:`OfferedHandSelector`)。スコアや特徴量の内訳は JSON
            には出さないので、必要なら ``self.offered_hand_selector.
            select(joint_positions, palms)`` を直接呼ぶ。
        """
        joints = {name: np.asarray(p, dtype=np.float64)
                 for name, p in joint_positions.items()}
        palms = {side: self._estimate_one(joints, side) for side in ('R', 'L')}
        selection = self.offered_hand_selector.select(joints, palms)
        result = dict(palms)
        result['offered_hand'] = selection['side']
        return result

    def _estimate_one(self, joints, side):
        points = {}
        for i in palm_plane.PLANE_LANDMARKS:
            key = '{}Hand{}'.format(side, i)
            if key in joints:
                points[i] = joints[key]
        plane = palm_plane.fit_palm_plane(points, hand=side)
        if plane is None:
            return None

        # palm_plane.py の plane.rot はロボットの手先座標系向け (+Y =
        # -normal) なので、人物自身の掌フレーム (+y = 手の甲->掌方向 =
        # normal そのもの) はここで組み直す。
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
    """``PalmPoseEstimator.estimate`` の戻り値を JSON として保存する.

    保存先に既に JSON があり、そこに ``keep_keys`` のキー (既定は
    ``human_label``, ``draw_random_human_poses.py`` の判定ボタンが書き込む
    人手ラベル) があれば、その値を引き継ぐ。
    """
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
        description='骨格 JSON (関節位置の dict) から左右の掌の位置姿勢と '
                    '手繋ぎに使うべき手を推定し、JSON として保存する。')
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
