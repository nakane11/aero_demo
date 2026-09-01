#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人体骨格 (MediaPipe 形式の関節位置の dict) を入力とし、左右の掌の位置
姿勢と、そのうちどちらの手を手繋ぎに使うべきか (人が差し出している手) を
推定して JSON として保存する。

``RandomSkeletonGenerator`` (``generate_random_human_poses.py``) が出力
する合成骨格でも、``people_pose_estimator_ros.RosPeoplePoseEstimator`` /
``fake_people_pose_estimator_ros.FakeRosPeoplePoseEstimator`` のような
実カメラ (MediaPipe) ベースの推定を骨格の生成元として使う場合でも、同じ
``PalmPoseEstimator.estimate(joint_positions)`` で掌の位置姿勢を求められる
ようにしてある。骨格の生成元が変わっても入力形式 (関節名 -> [x, y, z] の
dict) は変わらないので、この推定器は生成元を一切区別しない。

手のランドマーク (MediaPipe の ``RHand0``..``RHand20`` / ``LHand*``,
``RandomSkeletonGenerator(include_hand=True)`` や実際の MediaPipe 推定が
出力する) が 3 点以上揃っている側だけ、``aero_demo.palm_plane.
fit_palm_plane`` で手首 + 知節 (MCP) へ平面を SVD フィットする。
``human_palm_contact_behavior.py`` が実際にロボットを動かすときと同じ
関数・同じ判定 (点がほぼ一直線でないか等) を使うので、実カメラの推定結果
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
特徴量は掌の位置姿勢と体の関節だけから作れるものに限り、腕長・胴長で
正規化してあるので、身長の違う人物にも実カメラ (MediaPipe) 入力にも
そのまま効く。詳しくは ``OfferedHandSelector`` の docstring を参照。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py \
        --num-samples 100 --output-dir /tmp/random_human_poses
    rosrun aero_demo estimate_palm_poses.py \
        --input-dir /tmp/random_human_poses \
        --output-dir /tmp/random_palm_poses
"""

import argparse
import glob
import json
import os
import sys

from collections import namedtuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import palm_plane  # noqa: E402  (パス追加後に import)


# --- 差し出している手の判定に使う定数 -------------------------------------
# 各特徴量 (すべて 0..1 に正規化済み) の重み。合計 1.0 なのでスコアも
# 0..1 に収まる。``reach`` (体の前にどれだけ手を出しているか) が最も強い
# 手がかりで、次が掌の向き。腕の挙上・肘の伸び・高さは「明らかに違う
# 姿勢」を落とすための補助。``reach`` と ``height`` はそれぞれ REACH_MIN
# / HEIGHT_MIN の足切りにも使う。
OFFER_FEATURE_WEIGHTS = {
    'reach': 0.30,
    'palm_facing': 0.25,
    'elevation': 0.15,
    'extension': 0.15,
    'height': 0.15,
}

# 各特徴量のランプ (下限, 上限)。下限以下で 0、上限以上で 1。
# reach: 肩の中心から掌までの前方距離 / 腕長。腕を下ろしていれば ~0、
#        まっすぐ前に差し出せば ~0.9。
REACH_RAMP = (0.10, 0.55)
# palm_facing: 掌の法線と「相手/上/正中」の最も良い一致度 (cos)。握手の
#        ように掌が正中を向く場合・受け皿のように上を向く場合も拾う。
PALM_FACING_RAMP = (0.0, 0.6)
# elevation: 脱力して真下に垂れた腕からの離れ具合 (1 - cos)。真下なら 0、
#        水平なら 1。横に上げた手も拾える (reach は前方成分しか見ない)。
ELEVATION_RAMP = (0.15, 0.90)
# extension: 肩->手首の距離 / 腕長 (肘の伸展)。
EXTENSION_RAMP = (0.45, 0.90)
# height: 腰から掌までの高さ / 胴長 (腰=0, 肩=1) が入っていてほしい帯と、
#        その外側で 0 に落ちるまでの幅。万歳や膝下の手を落とす。
HEIGHT_BAND = (0.25, 0.95)
HEIGHT_BAND_SOFT = 0.35

# これを下回るスコアしか無ければ「どちらの手も差し出していない」と判定
# する (``offered_hand`` は None)。腕を下ろしたまま立っている人物 --
# generate_random_human_poses.py の肩の仰角は「腕を下ろした状態」を最頻値
# とする三角分布なので最も多いケース -- を argmax で拾ってしまわないため。
OFFER_SCORE_MIN = 0.50
# 左右のスコア差がこれ未満なら曖昧 (``select`` の戻り値の ``ambiguous``)。
# それでも argmax は返すので、呼び出し側 (右手しか差し出せない Aero の
# 到達性など、この判定器が知らない事情を持つ側) が覆せる。
AMBIGUOUS_MARGIN = 0.08
# 指先が体の後ろを向いている手 (cos がこれ未満) は、掌がどこを向いて
# いようと差し出しているとは言えないので候補から外す。
FINGER_BACKWARD_MIN = -0.6
# ``reach`` (体の前に手を出しているか) がこれ未満の手も候補から外す。
# 手を差し出すというのは相手と自分の間に手を置くことなので、体の前に
# 出ていない手は他の特徴量がどれだけ良くても差し出しではない -- 例えば
# 万歳した手や真横に伸ばした手は、掌が正中を向き肘も伸びているので重み
# 付き和だけでは閾値を超えうるが、reach は 0 になる。REACH_RAMP と合わせ
# ると「掌が肩の中心より腕長の約 0.19 倍 (成人で ~10 cm) 以上前」という
# 条件になる。
REACH_MIN = 0.20
# ``height`` (手繋ぎとして無理のない高さか) がこれ未満の手も候補から
# 外す。reach と同じ理由で、頭より高く上げた手や膝より低い手は手繋ぎの
# 対象になり得ないのに、重み 0.15 の減点だけでは他の特徴量が良ければ
# 閾値を超えてしまう (実際に合成骨格で、床から 2.0 m の高さに前へ振り
# 上げた手が 0.81 を取る)。HEIGHT_BAND / HEIGHT_BAND_SOFT と合わせると
# 「掌が腰から胴長の約 -0.03 倍〜1.23 倍の高さ (成人で腰の高さ〜肩より
# 15 cm ほど上)」という条件になる。
HEIGHT_MIN = 0.20

# 関節が欠測した入力 (実カメラで下半身がフレーム外、など) で胴長・腕長を
# 補うための人体比。身長 1.7 m の成人で 肩幅 ~0.40 m, 腰->肩 ~0.50 m,
# 上腕+前腕 ~0.58 m 程度。
_TORSO_PER_SHOULDER_WIDTH = 1.25
_ARM_PER_TORSO = 1.15


# 人体基準の座標系。ロボット (原点) との位置関係ではなく人物自身の向きで
# 特徴量を作るために使う: 合成骨格 (generate_random_human_poses.py) は
# 人物を常に原点に立たせるので、人物とロボットの位置関係が退化していて
# 「ロボットの方を向いているか」が使えない (palm_plane.fit_palm_plane が
# viewpoint を最後の手段にしているのと同じ事情)。
_BodyFrame = namedtuple('_BodyFrame', [
    'up',               # (3,) 腰 -> 肩 の体軸 (単位ベクトル)
    'left',             # (3,) 右肩 -> 左肩 (体軸に直交化, 単位ベクトル)
    'forward',          # (3,) 体の正面 = left cross up
    'shoulder_center',  # (3,) 両肩の中点
    'hip_center',       # (3,) 両腰の中点
    'torso',            # float, 腰 -> 肩 の距離 [m]
])


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return v / n


def _ramp(value, low, high):
    """``low`` 以下で 0.0、``high`` 以上で 1.0 になる線形ランプ."""
    if high <= low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def _band(value, low, high, soft):
    """``[low, high]`` で 1.0、その外側 ``soft`` の幅で 0.0 に落ちる台形."""
    return float(min(_ramp(value, low - soft, low),
                     1.0 - _ramp(value, high, high + soft)))


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

    across = l_sho - r_sho
    left = _unit(across - float(np.dot(across, up)) * up)
    if left is None:
        return None
    # ロボット座標系 (x=前, y=左, z=上) なら y cross z = x なので、
    # 直立した人物では forward がちょうど +x になる。
    forward = np.cross(left, up)
    return _BodyFrame(up=up, left=left, forward=forward,
                      shoulder_center=shoulder_center, hip_center=hip_center,
                      torso=torso)


def _wrist_position(joints, side, default):
    """手首の位置。``{side}Wrist`` が無ければ手のランドマークの 0 番を使う."""
    wrist = joints.get('{}Wrist'.format(side))
    if wrist is None:
        wrist = joints.get('{}Hand{}'.format(side, palm_plane.WRIST_INDEX))
    if wrist is None:
        return default
    return wrist


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
    だけから、左右それぞれに 0..1 のスコアを付け、``score_min`` を超えた
    側の argmax を採る。どちらも超えなければ「差し出していない」
    (``side`` は ``None``) と判定する。

    特徴量 (すべて腕長・胴長で正規化してあるので身長差に依らない)
    ------------------------------------------------------------------
    ``reach``
        肩の中心から掌までの距離のうち、体の正面方向の成分 / 腕長。手を
        差し出すというのは要するに体の前に手を出すことなので、これが最も
        強い手がかり。
    ``palm_facing``
        掌の法線 (``y_axis``, 手の甲 -> 掌) と「相手の方向」「上」「体の
        正中側」の 3 つの参照方向のうち、最も一致するものの cos。手繋ぎ
        の掌は、相手に正対する (差し出し)・上を向く (受け皿)・正中を向く
        (握手のように親指が上) のいずれもあり得るので 3 つの max を採る。
        逆に背中側や真下を向いた手はどれとも一致せず 0 になる。
    ``elevation``
        脱力して真下に垂れた腕からの離れ具合 (1 - cos)。``reach`` が前方
        成分しか見ないのに対し、横に上げた手も拾う。
    ``extension``
        肩 -> 手首の距離 / 腕長 (肘の伸展)。差し出した手は伸びている。
    ``height``
        腰から掌までの高さ / 胴長 (腰=0, 肩=1)。手繋ぎとして無理のない
        高さの帯から外れる (万歳, 膝下) と落ちる。

    「相手の方向」は ``viewpoint`` を渡せばそこへの方向、渡さなければ
    人物の正面 (相手は人の正面に立っていると仮定) を使う。合成骨格
    (generate_random_human_poses.py) は人物を常に原点に立たせるので
    ロボットとの位置関係が退化しており、既定の人体基準でなければならない。

    指先が体の後ろを向いている手 (``FINGER_BACKWARD_MIN``)、体の前に出て
    いない手 (``REACH_MIN``)、手繋ぎに使えない高さの手 (``HEIGHT_MIN``)、
    掌の推定に失敗した手 (``None``) は、重み付き和を計算するまでもなく
    候補から外す。``reach`` と ``height`` は「そもそも手繋ぎの対象になる
    姿勢か」という条件であって順位付けの手がかりではないので、重みだけ
    でなく足切りにも使う (万歳した手のように、他の特徴量が全て良ければ
    減点だけでは閾値を超えてしまうため)。

    Examples
    --------
    >>> selector = OfferedHandSelector()
    >>> selection = selector.select(joint_positions, palms)
    >>> selection['side']
    'L'
    >>> selection['scores']
    {'R': 0.34, 'L': 0.71}
    """

    def __init__(self, viewpoint=None, side_prior=None, weights=None,
                 score_min=OFFER_SCORE_MIN,
                 ambiguous_margin=AMBIGUOUS_MARGIN):
        """
        Parameters
        ----------
        viewpoint : (3,) array_like or None
            相手 (ロボット) の位置。指定するとスコアの ``palm_facing`` が
            「掌がそこを向いているか」を見るようになる。``None`` (既定)
            なら人物の正面方向を相手の方向とみなす。
        side_prior : dict or None
            ``{'R': float, 'L': float}``。スコアに直接足し込む事前分布。
            既定 (``None``) は左右とも 0.0。例えば Aero は右手しか差し
            出せない (``aero_demo.right_hand_offer``) ので、対面では人の
            左手の方が正対しやすい、といった事情を入れたい場合に使う。
            データセットに偏りを入れないよう既定では効かせない。
        weights : dict or None
            特徴量の重み。既定は :data:`OFFER_FEATURE_WEIGHTS`。
        score_min : float
            これを超えるスコアが無ければ ``side`` は ``None``。
        ambiguous_margin : float
            左右のスコア差がこれ未満なら ``ambiguous`` を立てる。
        """
        self.viewpoint = (None if viewpoint is None
                          else np.asarray(viewpoint, dtype=np.float64))
        self.side_prior = dict(side_prior or {})
        self.weights = dict(weights or OFFER_FEATURE_WEIGHTS)
        self.score_min = float(score_min)
        self.ambiguous_margin = float(ambiguous_margin)

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
            if feats['finger_forward'] < FINGER_BACKWARD_MIN:
                veto[side] = 'fingers_point_backward'
                continue
            if feats['reach'] < REACH_MIN:
                veto[side] = 'not_reaching_forward'
                continue
            if feats['height'] < HEIGHT_MIN:
                veto[side] = 'out_of_hold_height'
                continue
            scores[side] = sum(w * feats[key]
                               for key, w in self.weights.items()) \
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

    def _features(self, joints, body, side, palm):
        """片手ぶんの特徴量 (クラス docstring 参照) を計算する."""
        center = np.asarray(palm['position'], dtype=np.float64)
        normal = _unit(palm['y_axis'])    # 手の甲 -> 掌
        finger = _unit(palm['x_axis'])    # 手首 -> 指先
        shoulder = joints.get('{}Shoulder'.format(side), body.shoulder_center)
        wrist = _wrist_position(joints, side, center)
        arm = _arm_length(joints, side, body.torso)

        partner = None
        if self.viewpoint is not None:
            partner = _unit(self.viewpoint - center)
        if partner is None:
            partner = body.forward
        # 体の正中側 (左手なら体の右向き、右手なら体の左向き)。握手のよう
        # に掌が正中を向く姿勢を拾うための参照方向。
        medial = -body.left if side == 'L' else body.left

        from_shoulder = center - body.shoulder_center
        from_shoulder_dir = _unit(from_shoulder)

        reach = _ramp(float(np.dot(from_shoulder, body.forward)) / arm,
                      *REACH_RAMP)
        if normal is None:
            palm_facing = 0.0
        else:
            palm_facing = _ramp(max(float(np.dot(normal, partner)),
                                    float(np.dot(normal, body.up)),
                                    float(np.dot(normal, medial))),
                                *PALM_FACING_RAMP)
        if from_shoulder_dir is None:
            elevation = 0.0
        else:
            # 脱力した腕は肩の真下 (-up) に垂れるので、そこからの離れ具合。
            elevation = _ramp(1.0 - float(np.dot(from_shoulder_dir, -body.up)),
                              *ELEVATION_RAMP)
        extension = _ramp(float(np.linalg.norm(wrist - shoulder)) / arm,
                          *EXTENSION_RAMP)
        height = _band(
            float(np.dot(center - body.hip_center, body.up)) / body.torso,
            HEIGHT_BAND[0], HEIGHT_BAND[1], HEIGHT_BAND_SOFT)

        return dict(
            reach=reach,
            palm_facing=palm_facing,
            elevation=elevation,
            extension=extension,
            height=height,
            # 重みは掛からない診断用の値: 指先が体の前を向いているか (cos)。
            # FINGER_BACKWARD_MIN を下回る手は候補から外す。
            finger_forward=(0.0 if finger is None
                            else float(np.dot(finger, body.forward))))


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
            :class:`OfferedHandSelector` を作る。ロボットの位置を
            ``viewpoint`` に入れたい場合などはここで差し替える。
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


def load_skeleton_json(path):
    """``generate_random_human_poses.save_json`` が保存した 1 人分の JSON を読む.

    保存される JSON は骨格 (``skeleton``) と SMPL の人モデル (``smpl``)
    の両方を持つが、この推定器が要るのは骨格の関節位置だけなので
    ``skeleton.joint_positions`` だけを取り出す。
    """
    with open(path) as f:
        data = json.load(f)
    return data['skeleton']['joint_positions']


def save_json(palms, path):
    """``PalmPoseEstimator.estimate`` の戻り値を JSON として保存する.

    左右の掌 (``R``/``L``) に加えて、手繋ぎに使うべき手
    (``offered_hand``, ``'R'`` / ``'L'`` / ``None``) を含む。
    """
    with open(path, 'w') as f:
        json.dump(palms, f, indent=2)


def iter_skeleton_files(input_dir, pattern='*.json'):
    """``input_dir`` 内の骨格 JSON をファイル名順に列挙する."""
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


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
