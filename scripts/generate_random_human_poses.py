#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""SMPL の人体モデルをランダムに生成し、その姿勢から MediaPipe と同じ
関節名の骨格を作って、両方の情報を JSON として保存する。

``RandomSmplHumanGenerator``
    SMPL (Skinned Multi-Person Linear model) の体型 (``betas``) と姿勢
    (``pose``, 24 関節の axis-angle) をランダムに決め、``aero_demo.
    smpl_body`` の順運動学 (``smpl_forward`` / ``forward_world``) で
    姿勢済みの頂点・関節位置を計算する「人モデル」を作る。肩・肘・股
    関節・膝の可動域は解剖学的に不自然にならないよう実測ベースの範囲に
    限る (``generate`` 参照)。腰 (Spine1) と首 (Neck) には僅かな傾き
    (sway) に加えて鉛直軸まわりのひねり (回旋) も与える。「両足が地面に
    ついている・重心が安定している・頭が上を向いている」という制約を
    守るため、足首から先と骨盤の傾き (前後・左右) はランダム化せず、
    股関節の外転と膝の曲げは左右の脚に同じ角度を鏡写しに適用する。

``RandomSkeletonGenerator``
    ``RandomSmplHumanGenerator`` が作った SMPL の人モデルを入力として
    受け取り、その姿勢済み関節位置から MediaPipe 形式の骨格 (``aero_
    demo.people_pose_types.Person3D`` / ``fake_people_pose_estimator_
    ros.py`` の ``index2limbname`` と同じ関節名, ``Neck``, ``RShoulder``,
    ``LShoulder``, ``RElbow``, ``LElbow``, ``RWrist``, ``LWrist``,
    ``RHip``, ``LHip``, ``RKnee``, ``LKnee``, ``RAnkle``, ``LAnkle``,
    ``Nose``, ``REye``, ``LEye``, ``REar``, ``LEar``) を作る。手首から
    先は SMPL に関節が無いので、SMPL の前腕 (肘->手首) の実際の姿勢
    (回転) から手のランドマーク (``include_hand`` 既定 True,
    ``RHand0``..``RHand20`` / ``LHand0``..``LHand20``, 21 点 x 2 手,
    ``fake_people_pose_estimator_ros.py`` の ``index2handname`` /
    ``HAND_LOCAL`` と同じ MediaPipe 形式) を組み立てるので、手首の位置
    もその向きも SMPL の前腕とちょうど一致する (``_hand_frame`` 参照)。
    座標はロボット座標系 (x=前, y=左, z=上)。

出力する JSON には、骨格 (``skeleton``, 上記の ``joint_positions`` と
``height``) と SMPL の人モデル (``smpl``, ``pose``/``betas``/``root_
pos``/``gender``, ``draw_random_human_poses.py`` が ``aero_demo.
smpl_body.forward_world`` でメッシュを再構成するのに必要な情報) の両方
を含める。

SMPL のモデルファイル自体はライセンス上リポジトリに同梱されていないので、
呼び出し側がローカルパスを渡す (既定値は ``aero_demo.smpl_body`` と同じ
``~/SMPL_python_v.1.0.0/smpl/models/`` 以下)。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py \
        --num-samples 100 --output-dir /tmp/random_human_poses
"""

import argparse
import json
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import json_io  # noqa: E402  (パス追加後に import)
from aero_demo import people_pose_types  # noqa: E402
from aero_demo import smpl_body  # noqa: E402
from aero_demo import vector_utils  # noqa: E402

# MediaPipe と同じ関節名 (fake_people_pose_estimator_ros.py の
# index2limbname と同じ並び, 'Bkg' を除く)。
BODY_JOINT_NAMES = [
    'Nose', 'Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow',
    'RWrist', 'LWrist', 'RHip', 'LHip', 'RKnee', 'LKnee',
    'RAnkle', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar',
]

# MediaPipe の手のランドマーク名 (fake_people_pose_estimator_ros.py の
# index2handname と同じ, 'RHand0'..'RHand20' / 'LHand0'..'LHand20')。
HAND_JOINT_NAMES = ['{}Hand{}'.format(side, i)
                    for side in ('R', 'L') for i in range(21)]

HAND_LOCAL = people_pose_types.HAND_LOCAL_LANDMARKS

# 手の長さ (SMPL には手のランドマークが無いので、身長比の実測値を流用
# する)。
_HAND_LENGTH_HEIGHT_RATIO = 0.108

# 頭部ランドマーク (Nose/REye/LEye/REar/LEar) を Neck からのオフセット
# として置く距離 [m]。
_HEAD_FORWARD_OFFSET = 0.06
_NOSE_UP_OFFSET = 0.14
_EYE_UP_OFFSET = 0.16
_EYE_FORWARD_OFFSET = 0.05
_EYE_LATERAL_OFFSET = 0.03
_EAR_UP_OFFSET = 0.15
_EAR_LATERAL_OFFSET = 0.08


def _unit(v, fallback=None):
    if fallback is None:
        fallback = np.array([1.0, 0.0, 0.0])
    return vector_utils.unit(v, fallback=fallback)


def _rotate(v, axis, angle):
    """Rodrigues の回転公式: ``v`` を単位ベクトル ``axis`` まわりに
    ``angle`` [rad] だけ回転する。"""
    axis = _unit(np.asarray(axis, dtype=np.float64))
    return vector_utils.rotate(v, axis, angle)


class RandomSmplHumanGenerator(object):
    """SMPL の人体モデル (体型 + 姿勢) をランダムに生成する.

    ``generate()`` を呼ぶたびに、体型 (``betas``) と姿勢 (``pose``, 24
    関節の axis-angle) を毎回引き直した 1 人分の SMPL モデルを返す。
    肩・肘・股関節・膝の可動域はヒンジ関節としての肘の曲げ、仰角・方位角
    で表した肩の可動域など、解剖学的な可動域を大きく超えないよう実測
    ベースの範囲に限る (``generate`` 参照)。

    「両足が地面についている・重心が安定している・頭が上を向いている」
    という制約を必ず満たすように、足首から先と骨盤の前後・左右の傾きは
    ランダム化せず直立のままにする (root の向きを単位行列に固定し、脚の
    pose も左右対称にする)。股関節の外転と膝の曲げは左右の脚に必ず同じ
    角度を鏡写しに適用するので、両足は常に同じ高さのまま (= 生成後の
    床接地補正で両足が同時に接地する) になる。

    腰 (Spine1) と首 (Neck) には、僅かな傾き (sway) に加えて鉛直軸まわり
    のひねり (twist, 回旋) も与える (``_SPINE_TWIST_MAX_DEG`` /
    ``_NECK_TWIST_MAX_DEG``, ``_sway_twist_rotation`` 参照)。どちらも
    root より先の関節なので、ひねっても両足の接地・向きは変わらない。

    Examples
    --------
    >>> models = [smpl_body.load_smpl_model(male_path)]
    >>> gen = RandomSmplHumanGenerator(models, seed=0)
    >>> person = gen.generate()
    >>> person['pose'].shape
    (24, 3)
    """

    _STANCE_ABDUCTION_MAX_DEG = 25.0    # 足の開き (股関節の外転)
    _KNEE_FLEX_MAX_DEG = 45.0           # 膝の曲げ方
    _ELBOW_FLEX_MAX_DEG = 130.0         # 肘の曲げ方 (0=伸展のみ, ヒンジ関節)
    _SPINE_SWAY_MAX_DEG = 6.0           # 脊柱のごく僅かな傾き
    _NECK_SWAY_MAX_DEG = 8.0            # 首のごく僅かな傾き

    # 鉛直軸まわりのひねり (回旋)。傾き (sway) は最小回転
    # (``smpl_body.rotation_between``) で作るのでヨー成分を持たない
    # ので、ひねりはこの角度で別に与えて合成する (``generate`` 参照)。
    # 正の値で左 (+y 側) を向く向きの回旋。
    #   - 腰 (体幹) の回旋は実測で片側 35-45 度ほどだが、ここでは
    #     Spine1 の 1 関節だけで表現する (Spine2/Spine3 は pose=0 のまま
    #     引き継ぐ) ので、1 関節に集中しても不自然に見えない範囲に抑える。
    #   - 首の回旋は実測で片側 60-70 度ほど。同様に少し余裕を持たせる。
    # 脚は root (pelvis) の直接の子なので、Spine1 をひねっても両足の
    # 接地・向きには影響しない。
    _SPINE_TWIST_MAX_DEG = 25.0         # 腰 (体幹) のひねり
    _NECK_TWIST_MAX_DEG = 45.0          # 首のひねり

    # 肩の可動域を「T-pose (腕を真横に伸ばした状態, 仰角0度・方位角0度)」
    # を基準にした仰角 (上げ下げ) と方位角 (前後の振り) で定義する。
    #   - 仰角 (elevation): -90 度で腕が真下 (体側に下ろした状態) を向き、
    #     +90 度で腕が真上 (万歳) を向く。左右とも同じ符号でよい。
    #   - 方位角 (azimuth): 0 度で T-pose のまま真横、正の値で腕が前に
    #     振り出され (手を前に出す姿勢)、負の値で後ろに振れる (伸展)。
    # 実際の肩関節は屈曲 (前方挙上) の方が伸展 (後方) より大きく動く
    # ("最大 180 度 vs 最大 60 度程度") ので、前後の範囲は非対称にする。
    #
    # 仰角は一様分布ではなく、「腕を体側に下ろした状態」(-90 度) を最頻値
    # とする三角分布 (rng.triangular) からサンプリングする。直立した人物は
    # 腕を下ろしているのが最も一般的な姿勢であり、そこから稀に前へ出す・
    # 上げるといった姿勢が起こる、という自然な分布に近づけるため。
    _SHOULDER_ELEVATION_DOWN_DEG = -90.0
    _SHOULDER_ELEVATION_UP_DEG = 90.0
    _SHOULDER_AZIMUTH_DEG_RANGE = (-40.0, 110.0)

    # SMPL の体型パラメータ (betas) のばらつき。正規分布からサンプルし、
    # 極端な体型 (メッシュが破綻して見える) にならないようクリップする。
    _BETAS_STD = 1.5
    _BETAS_CLIP = 3.0

    # 生成する人物の身長の上限 [m]。これを超えた場合は betas を引き直す
    # (``generate`` 参照)。
    _MAX_HEIGHT_M = 1.7
    _MAX_HEIGHT_RESAMPLE_ATTEMPTS = 100

    def __init__(self, models, seed=None):
        """
        Parameters
        ----------
        models : list of (gender, smpl_body.SmplModel)
            使用可能な SMPL モデル。``generate()`` のたびにこの中から
            1 つをランダムに選ぶ (性別ごとに体型のばらつきが違って
            見えるように、複数渡しておくとよい)。
        seed : int, optional
            乱数シード (指定すると再現可能になる)。
        """
        if not models:
            raise ValueError('models must be a non-empty list')
        self.models = list(models)
        self.rng = np.random.RandomState(seed)

    def _sample_betas(self):
        return np.clip(
            self.rng.normal(scale=self._BETAS_STD, size=10),
            -self._BETAS_CLIP, self._BETAS_CLIP)

    def _sway_twist_rotation(self, sway_max_deg, twist_max_deg, xb, zb):
        """傾き (sway) とひねり (twist) を合成した局所回転行列を作る.

        腰 (Spine1) と首 (Neck) で共通の作り方。親の座標系での「上」
        ``zb`` まわりのひねりを先に適用し、そのあとで ``zb`` をランダムな
        軸まわりに少しだけ倒す最小回転 (``smpl_body.rotation_between``)
        を掛ける。最小回転は定義上ひねり (回転軸まわりのヨー) 成分を
        持たないので、ひねりはこうして別に与える。

        Parameters
        ----------
        sway_max_deg : float
            傾きの最大角 [deg] (0 から この値まで一様、軸はランダム)。
        twist_max_deg : float
            ひねりの最大角 [deg] (``-この値`` から ``+この値`` まで
            一様、正で左 (+y 側) を向く回旋)。
        xb, zb : (3,) ndarray
            親の座標系での前方 (x) と上 (z)。

        Returns
        -------
        (3, 3) ndarray
            親の座標系での局所回転行列 (ロボット座標系)。
        """
        rng = self.rng
        twist_angle = math.radians(
            rng.uniform(-twist_max_deg, twist_max_deg))
        twist_rot = smpl_body.rotation_between(xb, _rotate(xb, zb, twist_angle))
        sway_axis = _unit(rng.normal(size=3), fallback=xb)
        sway_angle = math.radians(rng.uniform(0.0, sway_max_deg))
        sway_rot = smpl_body.rotation_between(zb, _rotate(zb, sway_axis, sway_angle))
        return sway_rot.dot(twist_rot)

    def generate(self):
        """1 人分のランダムな SMPL 人モデルを生成する.

        Returns
        -------
        dict
            ``gender`` (str), ``model`` (``smpl_body.SmplModel``),
            ``betas`` ((10,) ndarray), ``pose`` ((24, 3) ndarray),
            ``root_pos`` ((3,) ndarray, pelvis の位置, 床 z=0 に接地),
            ``vertices`` ((6890, 3) ndarray), ``joints`` ((24, 3)
            ndarray, SMPL 関節順序, いずれもロボット座標系), ``wrist_
            rots`` (``{'L': (3, 3) ndarray, 'R': (3, 3) ndarray}``, 前腕
            (肘->手首) の T-pose からの累積回転行列), ``head_rot``
            ((3, 3) ndarray, 首 (Neck) の T-pose からの累積回転行列 --
            首のひねりを頭部ランドマークに反映するために使う,
            ``RandomSkeletonGenerator.generate`` 参照)。
        """
        rng = self.rng
        gender, model = self.models[rng.randint(len(self.models))]
        betas = self._sample_betas()

        xb = np.array([1.0, 0.0, 0.0])  # front
        yb = np.array([0.0, 1.0, 0.0])  # left
        zb = np.array([0.0, 0.0, 1.0])  # up

        pose = np.zeros((24, 3))
        # root (pelvis) の向きは常に単位行列 (常に +x を向いて直立) の
        # まま固定するので、cumulative[0] は root_rot=eye(3) 相当。脚は
        # root の直接の子なので、この固定によって足首から先・骨盤の傾き
        # は常にランダム化されない。
        cumulative = {0: np.eye(3)}

        def swing(pose_idx, child_idx, obs_dir_world):
            parent_rot = cumulative[model.parent[pose_idx]]
            rest_dir_robot = _unit(
                smpl_body.PERM.dot(model.J[child_idx] - model.J[pose_idx]))
            obs_dir_local = parent_rot.T.dot(obs_dir_world)
            R_local = smpl_body.rotation_between(rest_dir_robot, obs_dir_local)
            pose[pose_idx] = smpl_body.mat_to_axis_angle(
                smpl_body.to_smpl_rotation(R_local))
            cumulative[pose_idx] = parent_rot.dot(R_local)
            return obs_dir_world

        # --- 脚: 股関節の開き (外転) と膝の曲げ。左右対称な角度を使う
        # ので、両足は常に同じ高さのまま (床接地補正で両足が同時に
        # 接地する)。---
        stance = math.radians(rng.uniform(0.0, self._STANCE_ABDUCTION_MAX_DEG))
        knee_flex = math.radians(rng.uniform(0.0, self._KNEE_FLEX_MAX_DEG))
        for hip_idx, knee_idx, ankle_idx, sign in (
                (smpl_body.L_HIP, smpl_body.L_KNEE, smpl_body.L_ANKLE, 1.0),
                (smpl_body.R_HIP, smpl_body.R_KNEE, smpl_body.R_ANKLE, -1.0)):
            thigh_dir = _rotate(-zb, xb, stance * sign)
            swing(hip_idx, knee_idx, thigh_dir)
            shank_dir = _rotate(thigh_dir, yb, knee_flex)
            swing(knee_idx, ankle_idx, shank_dir)

        # --- 胴体・首 (ごく僅かにランダムな軸で傾け、さらに鉛直軸まわりに
        # ひねる)。脚は root の直接の子なので、この傾き・ひねりが脚の直立
        # には影響しない。Spine1 (3) だけを動かし、Spine2/Spine3/両肩の
        # Collar は pose=0 のまま Spine1 の回転をそのまま引き継がせる
        # (肩・首はその先で組み立てる) ので、腰をひねると肩・腕・首も
        # 一緒に回る (解剖学的に正しい: 腕の向き自体はこの後で世界座標
        # 基準に指定し直すので、肩の位置だけがひねりに従って動く)。---
        _SPINE1 = 3
        spine_rot = self._sway_twist_rotation(
            self._SPINE_SWAY_MAX_DEG, self._SPINE_TWIST_MAX_DEG, xb, zb)
        pose[_SPINE1] = smpl_body.mat_to_axis_angle(smpl_body.to_smpl_rotation(spine_rot))
        cumulative[_SPINE1] = cumulative[0].dot(spine_rot)
        for idx in (6, 9, 13, 14):  # Spine2, Spine3, LCollar, RCollar
            cumulative[idx] = cumulative[model.parent[idx]]

        neck_rot = self._sway_twist_rotation(
            self._NECK_SWAY_MAX_DEG, self._NECK_TWIST_MAX_DEG, xb, zb)
        pose[smpl_body.NECK] = smpl_body.mat_to_axis_angle(smpl_body.to_smpl_rotation(neck_rot))
        cumulative[smpl_body.NECK] = cumulative[9].dot(neck_rot)

        # --- 腕: 肩の仰角・方位角 (T-pose = 真横基準) + 肘のヒンジ曲げ。
        # 左右は独立にランダムな角度を割り当てる (例: 片手だけ前に出す、
        # 片手だけ下ろす、といった姿勢も許容する)。手首・手先の pose は
        # 0 のまま (手のランドマークは前腕の向きだけから組み立てるので、
        # 手首の捻りは中立姿勢を仮定する, RandomSkeletonGenerator._hand_
        # frame 参照)。---
        # 前腕 (肘->手首) の T-pose からの累積回転行列 (``cumulative[elbow_
        # idx]``, swing() が肩の回転もすでに合成した状態で作る full 3x3
        # 回転行列)。手首・手先の pose は 0 のままなので、この行列が前腕・
        # 手のボーンの実際の向き (捻り込み) をそのまま表す。手のランド
        # マークをこの行列で組み立てる (``RandomSkeletonGenerator.
        # _hand_frame`` 参照) ことで、SMPL メッシュの前腕が肩・肘の回転で
        # 蓄積する前腕軸まわりの捻りを、手のランドマークにも反映できる。
        wrist_rots = {}
        for shoulder_idx, elbow_idx, wrist_idx, side, sign in (
                (smpl_body.L_SHOULDER, smpl_body.L_ELBOW, smpl_body.L_WRIST,
                 'L', 1.0),
                (smpl_body.R_SHOULDER, smpl_body.R_ELBOW, smpl_body.R_WRIST,
                 'R', -1.0)):
            rest_dir = yb * sign
            elevation = math.radians(rng.triangular(
                self._SHOULDER_ELEVATION_DOWN_DEG,
                self._SHOULDER_ELEVATION_DOWN_DEG,
                self._SHOULDER_ELEVATION_UP_DEG))
            azimuth = math.radians(rng.uniform(*self._SHOULDER_AZIMUTH_DEG_RANGE))
            raised = _rotate(rest_dir, xb, elevation * sign)
            upper_dir = _rotate(raised, zb, -azimuth * sign)
            swing(shoulder_idx, elbow_idx, upper_dir)

            flex = math.radians(rng.uniform(0.0, self._ELBOW_FLEX_MAX_DEG))
            forearm_dir = _rotate(upper_dir, xb, flex * sign)
            swing(elbow_idx, wrist_idx, forearm_dir)
            wrist_rots[side] = cumulative[elbow_idx].copy()

        vertices, joints = smpl_body.forward_world(
            model, pose, betas, root_pos=np.zeros(3))

        # 身長 (``_MAX_HEIGHT_M``) を超える場合は betas (体型) だけを
        # 引き直す (姿勢 ``pose`` は betas にほぼ依存しないのでそのまま
        # 使う)。滅多に外れ値が続くことは無いはずだが、念のため試行回数
        # に上限を設け、それでも収まらなければ最後に引いた体型を諦めて
        # 使う。
        for _ in range(self._MAX_HEIGHT_RESAMPLE_ATTEMPTS):
            height = float(vertices[:, 2].max() - vertices[:, 2].min())
            if height <= self._MAX_HEIGHT_M:
                break
            betas = self._sample_betas()
            vertices, joints = smpl_body.forward_world(
                model, pose, betas, root_pos=np.zeros(3))

        # 姿勢に関わらず、体の最下点が必ず床 (z=0) に接するように上下
        # 移動する。足の開き・膝の曲げは左右対称なので、両足は常に同じ
        # 高さのまま床に接地する。
        min_z = float(vertices[:, 2].min())
        root_pos = np.array([0.0, 0.0, -min_z])
        vertices = vertices + root_pos
        joints = joints + root_pos

        return dict(gender=gender, model=model, betas=betas, pose=pose,
                   root_pos=root_pos, vertices=vertices, joints=joints,
                   wrist_rots=wrist_rots, head_rot=cumulative[smpl_body.NECK])


class RandomSkeletonGenerator(object):
    """SMPL の人モデル (``RandomSmplHumanGenerator.generate()`` の戻り値)
    から、MediaPipe 形式の人体骨格 (関節の 3 次元位置) を作る.

    骨格の関節位置は SMPL の姿勢済み関節 (``smpl_body.forward_world``)
    をそのまま読むだけなので、SMPL メッシュと骨格の胴体・四肢の関節位置
    は常に一致する。手首から先だけは SMPL に関節が無いので、手のランド
    マークは SMPL の前腕 (肘->手首) の実際の姿勢 (``RandomSmplHuman
    Generator.generate`` が計算したのと同じ累積回転行列 ``wrist_rots``)
    から組み立てる -- SMPL の前腕ボーンをそのまま回転させたのと同じ行列
    を使うので、手首の位置・向き (指先方向の軸まわりの捻りを含む) は
    SMPL モデルの前腕とちょうど一致する。

    Examples
    --------
    >>> smpl_gen = RandomSmplHumanGenerator(models, seed=0)
    >>> gen = RandomSkeletonGenerator(seed=0)
    >>> pose = gen.generate(smpl_gen.generate())
    >>> pose['joint_positions']['Neck']
    [0.01, -0.03, 1.42]
    """

    # SMPL 関節順序 (24,) -> MediaPipe 関節名。
    _SMPL_TO_MEDIAPIPE = {
        smpl_body.NECK: 'Neck',
        smpl_body.L_SHOULDER: 'LShoulder', smpl_body.R_SHOULDER: 'RShoulder',
        smpl_body.L_ELBOW: 'LElbow', smpl_body.R_ELBOW: 'RElbow',
        smpl_body.L_WRIST: 'LWrist', smpl_body.R_WRIST: 'RWrist',
        smpl_body.L_HIP: 'LHip', smpl_body.R_HIP: 'RHip',
        smpl_body.L_KNEE: 'LKnee', smpl_body.R_KNEE: 'RKnee',
        smpl_body.L_ANKLE: 'LAnkle', smpl_body.R_ANKLE: 'RAnkle',
    }

    def __init__(self, include_hand=True):
        """
        Parameters
        ----------
        include_hand : bool, optional
            MediaPipe の手のランドマーク (``RHand0``..``RHand20`` /
            ``LHand0``..``LHand20``, 21 点 x 2 手) も生成するか。既定
            True (fake_people_pose_estimator_ros.py の ``~hand/enable``
            と同じ既定値)。
        """
        self.include_hand = include_hand

    @staticmethod
    def _hand_frame(wrist_rot, rest_dir_robot, side):
        """手首の局所座標系 (u=指方向, v=親指側, n=掌の向き) を作る.

        指のランドマークは持たないので手首の捻り (回内/回外) を直接は
        観測できないが、SMPL 側は手首・手先の pose を 0 (捻りなし) の
        まま生成しているので、前腕ボーン (肘->手首) の T-pose からの
        累積回転 ``wrist_rot`` (``RandomSmplHumanGenerator.generate`` が
        返す ``wrist_rots[side]``, 肩・肘の回転をすでに合成した full 3x3
        回転行列) を T-pose での基準フレームにそのまま適用すれば、SMPL
        メッシュの前腕とちょうど同じ向き (肩・肘の回転で前腕軸まわりに
        溜まる捻りも込み) になる。

        T-pose での基準フレームは、u0=前腕の T-pose 方向 (``rest_dir_
        robot``, ほぼ体の左右軸), n0=T-pose で掌が向く向き (実測により
        -Z (下), ``smpl_body._REST_PALM_NORMAL`` と同じ値を使う), v0=
        u0×n0 (または n0×u0) から作る親指方向 (前方 +X, 解剖学的に自然)
        で決める。

        Parameters
        ----------
        wrist_rot : (3, 3) array_like
            前腕ボーンの T-pose からの累積回転行列 (ロボット座標系)。
        rest_dir_robot : (3,) array_like
            前腕の T-pose での向き (肘->手首, 単位ベクトル, ロボット
            座標系)。
        side : str
            ``'R'`` or ``'L'``。
        """
        wrist_rot = np.asarray(wrist_rot, dtype=np.float64)
        u0 = np.asarray(rest_dir_robot, dtype=np.float64)
        n0 = smpl_body._REST_PALM_NORMAL
        v0 = np.cross(u0, n0) if side == 'R' else np.cross(n0, u0)
        u = wrist_rot.dot(u0)
        v = wrist_rot.dot(v0)
        n = wrist_rot.dot(n0)
        return u, v, n

    @staticmethod
    def _hand_landmarks(side, wrist, u, v, n, hand_length):
        """21 個の手のランドマーク位置を作る (fake_people_pose_estimator_
        ros.py の ``_hand_positions`` と同じ)."""
        basis = np.vstack([u, v, n])
        pts = wrist + hand_length * HAND_LOCAL.dot(basis)
        return {'{}Hand{}'.format(side, i): pts[i] for i in range(len(pts))}

    def generate(self, smpl_person):
        """SMPL の人モデルから 1 人分の骨格を作る.

        Parameters
        ----------
        smpl_person : dict
            ``RandomSmplHumanGenerator.generate()`` の戻り値。

        Returns
        -------
        dict
            ``joint_positions`` (関節名 -> [x, y, z], ロボット座標系
            (x=前, y=左, z=上), 床 z=0 に接地) と ``height`` (身長 [m])
            を持つ、JSON にそのままシリアライズできる dict。
        """
        yb = np.array([0.0, 1.0, 0.0])  # left
        zb = np.array([0.0, 0.0, 1.0])  # up

        smpl_joints = smpl_person['joints']
        vertices = smpl_person['vertices']
        joints = {name: smpl_joints[idx]
                 for idx, name in self._SMPL_TO_MEDIAPIPE.items()}

        neck = joints['Neck']
        head = smpl_joints[smpl_body.HEAD]
        head_dir = _unit(head - neck, fallback=zb)
        # 顔の正面方向は首の累積回転 (``RandomSmplHumanGenerator.generate``
        # の ``head_rot``) が回した前方 +x を、首->頭の軸に直交する成分だけ
        # 取り出して使う。
        head_rot = smpl_person.get('head_rot')
        if head_rot is None:
            head_fwd_raw = np.cross(yb, head_dir)
        else:
            fwd = np.asarray(head_rot, dtype=np.float64).dot(
                np.array([1.0, 0.0, 0.0]))
            head_fwd_raw = fwd - head_dir * head_dir.dot(fwd)
        head_fwd = _unit(head_fwd_raw, fallback=np.array([1.0, 0.0, 0.0]))
        head_left = _unit(np.cross(head_dir, head_fwd), fallback=yb)
        joints['Nose'] = neck + head_dir * _NOSE_UP_OFFSET \
            + head_fwd * _HEAD_FORWARD_OFFSET
        joints['LEye'] = (neck + head_dir * _EYE_UP_OFFSET
                          + head_fwd * _EYE_FORWARD_OFFSET
                          + head_left * _EYE_LATERAL_OFFSET)
        joints['REye'] = (neck + head_dir * _EYE_UP_OFFSET
                          + head_fwd * _EYE_FORWARD_OFFSET
                          - head_left * _EYE_LATERAL_OFFSET)
        joints['LEar'] = neck + head_dir * _EAR_UP_OFFSET \
            + head_left * _EAR_LATERAL_OFFSET
        joints['REar'] = neck + head_dir * _EAR_UP_OFFSET \
            - head_left * _EAR_LATERAL_OFFSET

        height = float(vertices[:, 2].max() - vertices[:, 2].min())

        if self.include_hand:
            hand_length = height * _HAND_LENGTH_HEIGHT_RATIO
            model = smpl_person['model']
            for side, elbow_idx, wrist_idx in (
                    ('L', smpl_body.L_ELBOW, smpl_body.L_WRIST),
                    ('R', smpl_body.R_ELBOW, smpl_body.R_WRIST)):
                wrist = joints['{}Wrist'.format(side)]
                wrist_rot = smpl_person['wrist_rots'][side]
                rest_dir_robot = _unit(smpl_body.PERM.dot(
                    model.J[wrist_idx] - model.J[elbow_idx]))
                u, v, n = self._hand_frame(wrist_rot, rest_dir_robot, side)
                joints.update(self._hand_landmarks(
                    side, wrist, u, v, n, hand_length))

        joint_positions = {
            name: [float(x) for x in p] for name, p in joints.items()}
        return dict(joint_positions=joint_positions, height=height)


def build_person_json(smpl_person, skeleton):
    """1 人分の SMPL モデルと骨格を、保存用の 1 つの dict にまとめる.

    Parameters
    ----------
    smpl_person : dict
        ``RandomSmplHumanGenerator.generate()`` の戻り値。
    skeleton : dict
        ``RandomSkeletonGenerator.generate()`` の戻り値。

    Returns
    -------
    dict
        ``skeleton`` (``joint_positions``/``height``) と ``smpl``
        (``gender``/``betas``/``pose``/``root_pos``) を持つ、JSON に
        そのままシリアライズできる dict。``draw_random_human_poses.py``
        は ``smpl`` を ``aero_demo.smpl_body.forward_world`` に渡して
        メッシュを再構成し、``skeleton`` を色付きの線で重ねて描く。
    """
    return dict(
        skeleton=skeleton,
        smpl=dict(
            gender=smpl_person['gender'],
            betas=[float(x) for x in smpl_person['betas']],
            pose=[[float(x) for x in row] for row in smpl_person['pose']],
            root_pos=[float(x) for x in smpl_person['root_pos']],
        ))


def save_json(person, path):
    """``build_person_json`` の戻り値を JSON として保存する."""
    json_io.save_json(path, person)


def load_smpl_models(male_path, female_path):
    """使用可能な SMPL モデル (男性/女性) をロードする.

    女性モデルが見つからない場合は男性モデルのみで続行する。

    Returns
    -------
    list of (str, smpl_body.SmplModel)
        ``(gender, model)`` のリスト。
    """
    models = [('male', smpl_body.load_smpl_model(male_path))]
    female_path = os.path.expanduser(female_path)
    if os.path.exists(female_path):
        models.append(('female', smpl_body.load_smpl_model(female_path)))
    else:
        print('female SMPL model not found at {}, using the male model '
              'only (--female-model-path で指定できます)'.format(
                  female_path))
    return models


def main():
    parser = argparse.ArgumentParser(
        description='SMPL の人体モデルをランダムに生成し、そこから '
                    'MediaPipe 形式の骨格を作って、両方を JSON として '
                    '保存する。')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='生成する人物 (JSON) の数。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='JSON の保存先ディレクトリ。')
    parser.add_argument(
        '--model-path', type=str,
        default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'),
        help='SMPL (男性) モデル .pkl のパス。')
    parser.add_argument(
        '--female-model-path', type=str,
        default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicModel_f_lbs_10_207_0_v1.0.0.pkl'),
        help='SMPL (女性) モデル .pkl のパス (無ければ男性モデルのみ使う)。')
    parser.add_argument('--seed', type=int, default=None,
                        help='乱数シード (指定すると再現可能になる)。')
    parser.add_argument(
        '--start-index', type=int, default=0,
        help='ファイル名の連番の開始値 (既定 0 -> human_000.json から)。'
             '既にラベル付けした JSON がある所へ人物を追加したいときに、'
             '既存のファイルを上書きしないよう続きの番号から書き出す。'
             'その場合は --seed も既存と違う値にしないと同じ人物が'
             '生成されるので注意。')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    models = load_smpl_models(args.model_path, args.female_model_path)
    smpl_generator = RandomSmplHumanGenerator(models, seed=args.seed)
    skeleton_generator = RandomSkeletonGenerator()

    for i in range(args.num_samples):
        index = args.start_index + i
        out_path = os.path.join(args.output_dir,
                                'human_{:03d}.json'.format(index))
        if os.path.exists(out_path):
            print('{} は既にあります。--start-index を大きくしてください。'
                  .format(out_path))
            return
        smpl_person = smpl_generator.generate()
        skeleton = skeleton_generator.generate(smpl_person)
        person = build_person_json(smpl_person, skeleton)
        save_json(person, out_path)
        print('[{}/{}] saved {}'.format(i + 1, args.num_samples, out_path))


if __name__ == '__main__':
    main()
