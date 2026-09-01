#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""多様な姿勢・体格の人間の骨格 (関節の 3 次元位置) をランダムに生成し、
MediaPipe と同じ関節名の形式で JSON として保存する。

SMPL には一切依存しない。体格は身長と、身長に対する各部位の比率 (肩幅・
腰幅・上腕/前腕の長さなど。``fake_people_pose_estimator_ros.
FakeRosPeoplePoseEstimator._sample_person`` が使っている値と同じ実測
ベースの比率) からランダムに決め、姿勢は肩・肘・股関節・膝・脊柱・首を
単純な順運動学 (親関節からの相対回転をボーン方向ベクトルに適用して次の
関節位置を求める) で組み立てる。

出力される関節名は ``aero_demo.people_pose_types.Person3D`` /
``fake_people_pose_estimator_ros.py`` の ``index2limbname`` と同じ
MediaPipe 形式 (``Neck``, ``RShoulder``, ``LShoulder``, ``RElbow``,
``LElbow``, ``RWrist``, ``LWrist``, ``RHip``, ``LHip``, ``RKnee``,
``LKnee``, ``RAnkle``, ``LAnkle``, ``Nose``, ``REye``, ``LEye``, ``REar``,
``LEar``) で、座標はロボット座標系 (x=前, y=左, z=上) 。

``include_hand`` (既定 True) を立てると、``fake_people_pose_estimator_
ros.py`` の ``index2handname`` / ``HAND_LOCAL`` と同じ MediaPipe 形式の
手のランドマーク (``RHand0``..``RHand20``, ``LHand0``..``LHand20``, 21 点
x 2 手) も併せて生成する。指先のランドマークは持たないので前腕の向きから
は分からない手首の捻り (回内/回外) は、握手を差し出す前の中立姿勢
(親指が上・掌が体の外側を向く) を仮定する (``RandomSkeletonGenerator.
_hand_frame`` 参照)。

肩・肘は解剖学的な可動域を大きく超えたランダム姿勢(腕を棒のように
真っ直ぐ伸ばした/後方に反り返った不自然な姿勢) にならないよう、
ヒンジ関節としての肘の曲げと、仰角・方位角で表した肩の可動域 (下ろす
/前に出す/横に開く/上げる、のすべてをカバーする) を実測ベースで
定義している (``RandomSkeletonGenerator.generate`` 参照)。

一方で「両足が地面についている・重心が安定している・頭が上を向いて
いる」という制約を必ず満たすように、足首から先と骨盤の前後・左右の
傾きはランダム化せず直立のままにする。股関節の開き (外転) と膝の曲げは
左右の脚に必ず同じ角度を鏡写しに適用するので、両足は常に同じ高さの
まま (= 生成後の床接地補正で両足が同時に接地する) になる。

このファイルは骨格 (JSON) を作るだけで、描画は一切行わない。生成した
JSON を読み込み SMPL の人体メッシュを当てはめて viser で表示する部分は
``draw_random_human_poses.py`` に分割してある。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py \
        --num-samples 100 --output-dir /tmp/random_human_poses
"""

import argparse
import json
import math
import os

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

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

# 手のランドマークの局所座標 (手の長さを単位とする), MediaPipe の並び
# (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 pinky)。
# fake_people_pose_estimator_ros.py の HAND_LOCAL と同じ値
# (軸: u=手首->指先, v=親指側, n=掌の向き)。
HAND_LOCAL = np.array([
    [0.00,  0.00, 0.00],   # 0  wrist
    [0.11,  0.13, 0.02],   # 1  thumb CMC
    [0.25,  0.26, 0.05],   # 2  thumb MCP
    [0.36,  0.34, 0.08],   # 3  thumb IP
    [0.45,  0.40, 0.10],   # 4  thumb tip
    [0.47,  0.17, 0.00],   # 5  index MCP
    [0.66,  0.18, 0.04],   # 6  index PIP
    [0.78,  0.18, 0.10],   # 7  index DIP
    [0.88,  0.18, 0.15],   # 8  index tip
    [0.48,  0.05, 0.00],   # 9  middle MCP
    [0.69,  0.05, 0.04],   # 10 middle PIP
    [0.82,  0.05, 0.11],   # 11 middle DIP
    [0.93,  0.05, 0.17],   # 12 middle tip
    [0.46, -0.08, 0.00],   # 13 ring MCP
    [0.65, -0.09, 0.04],   # 14 ring PIP
    [0.77, -0.10, 0.11],   # 15 ring DIP
    [0.87, -0.10, 0.16],   # 16 ring tip
    [0.43, -0.21, 0.00],   # 17 pinky MCP
    [0.57, -0.23, 0.03],   # 18 pinky PIP
    [0.67, -0.24, 0.08],   # 19 pinky DIP
    [0.74, -0.25, 0.12],   # 20 pinky tip
])

# 前腕 (指) 方向がほぼ真上/真下に近いと、世界の「上」を指方向に垂直投影
# したベクトルの向きが数値誤差で不安定になる。fake_people_pose_estimator_
# ros.py の UP_PERP_BLEND_NORM と同じしきい値・同じ理由で、体の左右軸から
# 作ったフォールバックへ滑らかにブレンドする。
_UP_PERP_BLEND_NORM = 0.2


def _unit(v, fallback=None):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return fallback if fallback is not None else np.array([1.0, 0.0, 0.0])
    return v / n


def _rotate(v, axis, angle):
    """Rodrigues の回転公式: ``v`` を単位ベクトル ``axis`` まわりに
    ``angle`` [rad] だけ回転する。"""
    axis = _unit(np.asarray(axis, dtype=np.float64))
    c, s = math.cos(angle), math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * axis.dot(v) * (1.0 - c)


class RandomSkeletonGenerator(object):
    """MediaPipe 形式の人体骨格 (関節の 3 次元位置) をランダムに生成する.

    SMPL / trimesh / skrobot のいずれにも依存しない (純粋な numpy の
    幾何計算のみ)。``generate()`` を呼ぶたびに、体格 (身長と各部位比率)
    ・姿勢 (肩/肘/股関節/膝/脊柱/首の角度) を毎回引き直した 1 人分の
    骨格を返す。

    Examples
    --------
    >>> gen = RandomSkeletonGenerator(seed=0)
    >>> pose = gen.generate()
    >>> pose['joint_positions']['Neck']
    [0.01, -0.03, 1.42]
    """

    _HEIGHT_RANGE_M = (1.4, 1.9)

    # 身長に対する各部位の比率。fake_people_pose_estimator_ros.py の
    # FakeRosPeoplePoseEstimator._sample_person が使っている実測ベースの
    # 値と同じ (h_* は身長に対する高さの比率、他は長さの比率)。
    _RATIOS = dict(
        h_shoulder=0.818, h_hip=0.530, h_knee=0.285, h_ankle=0.039,
        shoulder_width=0.245, hip_width=0.185,
        upper_arm=0.186, forearm=0.146, hand_length=0.108,
    )
    _RATIO_SPREAD = 0.06  # 個体差 (比率に対する相対的なばらつき)

    # 「両足接地・重心安定・頭が上を向く」という制約を守るため、足首から
    # 先と骨盤の傾き (前後・左右方向) はランダム化しない (常に直立の
    # まま)。股関節の開き (足の開き具合) と膝の曲げ方は、左右の脚に必ず
    # 同じ角度を鏡写しに適用することで、両足が常に同じ高さになる (=
    # generate() の最後で行う床接地補正で両足が同時に接地する) ように
    # している。変化させるのは、脚の接地や重心に影響しない上半身 (肩/
    # 肘) と、ごくわずかな脊柱・首の動きのみに限る。
    _STANCE_ABDUCTION_MAX_DEG = 25.0    # 足の開き (股関節の外転)
    _KNEE_FLEX_MAX_DEG = 45.0           # 膝の曲げ方
    _ELBOW_FLEX_MAX_DEG = 130.0         # 肘の曲げ方 (0=伸展のみ, ヒンジ関節)
    _SPINE_SWAY_MAX_DEG = 6.0           # 脊柱のごく僅かな傾き
    _NECK_SWAY_MAX_DEG = 8.0            # 首のごく僅かな傾き

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

    def __init__(self, height_range=None, seed=None, include_hand=True):
        """
        Parameters
        ----------
        height_range : (float, float), optional
            身長の生成範囲 [m]。既定 ``_HEIGHT_RANGE_M`` (1.4-1.9 m)。
        seed : int, optional
            乱数シード (指定すると再現可能になる)。
        include_hand : bool, optional
            MediaPipe の手のランドマーク (``RHand0``..``RHand20`` /
            ``LHand0``..``LHand20``, 21 点 x 2 手) も生成するか。既定
            True (fake_people_pose_estimator_ros.py の ``~hand/enable``
            と同じ既定値)。
        """
        self.height_range = tuple(height_range) if height_range \
            else self._HEIGHT_RANGE_M
        self.include_hand = include_hand
        self.rng = np.random.RandomState(seed)

    def _sample_body(self):
        rng = self.rng
        h = rng.uniform(*self.height_range)

        def r(ratio):
            return h * ratio * rng.uniform(
                1.0 - self._RATIO_SPREAD, 1.0 + self._RATIO_SPREAD)

        body = {name: r(ratio) for name, ratio in self._RATIOS.items()}
        body['height'] = h
        body['thigh'] = body['h_hip'] - body['h_knee']
        body['shank'] = body['h_knee'] - body['h_ankle']
        body['torso'] = body['h_shoulder'] - body['h_hip']
        return body

    @staticmethod
    def _hand_frame(finger_dir, side, sign, yb, zb):
        """手首の局所座標系 (u=指方向, v=親指側, n=掌の向き) を作る.

        指のランドマークは持たないので、手首の捻り (回内/回外) は前腕の
        向きだけからは分からない。fake_people_pose_estimator_ros.py の
        ``~present_hand`` の基準姿勢 (wrist_roll=0, 手首を捻らない中立
        姿勢で親指が上・掌が体の外側を向く) と同じ式で、解剖学的に自然な
        既定の向きを組み立てる (``FakeRosPeoplePoseEstimator.
        _body_positions`` の ``n0`` 参照)。

        Parameters
        ----------
        finger_dir : (3,) array_like
            指先方向の単位ベクトル (ここでは前腕 = 肘->手首の延長)。
        side : str
            ``'R'`` or ``'L'``。
        sign : float
            体の左右方向の符号 (体の左が +1, 右が -1; ``generate`` の
            腕ループで使っているものと同じ)。
        yb, zb : (3,) array_like
            体基準の左方向・上方向の単位ベクトル。
        """
        u = np.asarray(finger_dir, dtype=np.float64)
        up_perp_raw = zb - float(np.dot(zb, u)) * u
        up_perp_norm = np.linalg.norm(up_perp_raw)
        if up_perp_norm >= _UP_PERP_BLEND_NORM:
            up_perp = up_perp_raw / up_perp_norm
        else:
            hinge = yb * sign
            hinge_perp = _unit(hinge - float(np.dot(hinge, u)) * u,
                               fallback=np.zeros(3))
            weight = up_perp_norm / _UP_PERP_BLEND_NORM
            up_perp = _unit(weight * up_perp_raw
                           + (1.0 - weight) * hinge_perp,
                           fallback=hinge_perp)
        n = _unit(np.cross(up_perp, u) if side == 'R'
                 else np.cross(u, up_perp))
        v = np.cross(u, n) if side == 'R' else np.cross(n, u)
        return u, v, n

    @staticmethod
    def _hand_landmarks(side, wrist, u, v, n, hand_length):
        """21 個の手のランドマーク位置を作る (fake_people_pose_estimator_
        ros.py の ``_hand_positions`` と同じ)."""
        basis = np.vstack([u, v, n])
        pts = wrist + hand_length * HAND_LOCAL.dot(basis)
        return {'{}Hand{}'.format(side, i): pts[i] for i in range(len(pts))}

    def generate(self):
        """1 人分のランダムな骨格を生成する.

        Returns
        -------
        dict
            ``joint_positions`` (関節名 -> [x, y, z], ロボット座標系
            (x=前, y=左, z=上), 床 z=0 に接地) と ``height`` (身長 [m])
            を持つ、JSON にそのままシリアライズできる dict。
        """
        rng = self.rng
        b = self._sample_body()

        xb = np.array([1.0, 0.0, 0.0])  # front
        yb = np.array([0.0, 1.0, 0.0])  # left
        zb = np.array([0.0, 0.0, 1.0])  # up

        pelvis = zb * b['h_hip']
        joints = {}

        # --- 脚: 股関節の開き (外転) と膝の曲げ。左右対称な角度を使う
        # ので、両足は常に同じ高さのまま (generate() 末尾の床接地補正で
        # 両足が同時に接地する)。---
        stance = math.radians(rng.uniform(0.0, self._STANCE_ABDUCTION_MAX_DEG))
        knee_flex = math.radians(rng.uniform(0.0, self._KNEE_FLEX_MAX_DEG))
        for side, sign in (('L', 1.0), ('R', -1.0)):
            hip = pelvis + yb * (b['hip_width'] / 2.0) * sign
            thigh_dir = _rotate(-zb, xb, stance * sign)
            knee = hip + b['thigh'] * thigh_dir
            shank_dir = _rotate(thigh_dir, yb, knee_flex)
            ankle = knee + b['shank'] * shank_dir
            joints['{}Hip'.format(side)] = hip
            joints['{}Knee'.format(side)] = knee
            joints['{}Ankle'.format(side)] = ankle

        # --- 胴体・首・頭 (ごく僅かにランダムな軸で傾ける) ---
        spine_axis = _unit(rng.normal(size=3), fallback=xb)
        spine_angle = math.radians(rng.uniform(0.0, self._SPINE_SWAY_MAX_DEG))
        torso_dir = _rotate(zb, spine_axis, spine_angle)
        neck = pelvis + b['torso'] * torso_dir
        joints['Neck'] = neck

        neck_axis = _unit(rng.normal(size=3), fallback=xb)
        neck_angle = math.radians(rng.uniform(0.0, self._NECK_SWAY_MAX_DEG))
        head_dir = _rotate(torso_dir, neck_axis, neck_angle)
        head_fwd = _unit(np.cross(yb, head_dir), fallback=xb)
        head_left = _unit(np.cross(head_dir, head_fwd), fallback=yb)
        joints['Nose'] = neck + head_dir * 0.14 + head_fwd * 0.06
        joints['LEye'] = (neck + head_dir * 0.16 + head_fwd * 0.05
                          + head_left * 0.03)
        joints['REye'] = (neck + head_dir * 0.16 + head_fwd * 0.05
                          - head_left * 0.03)
        joints['LEar'] = neck + head_dir * 0.15 + head_left * 0.08
        joints['REar'] = neck + head_dir * 0.15 - head_left * 0.08

        # --- 腕: 肩の仰角・方位角 (T-pose = 真横基準) + 肘のヒンジ曲げ。
        # 左右は独立にランダムな角度を割り当てる (例: 片手だけ前に出す、
        # 片手だけ下ろす、といった姿勢も許容する)。---
        for side, sign in (('L', 1.0), ('R', -1.0)):
            shoulder = neck + yb * (b['shoulder_width'] / 2.0) * sign
            rest_dir = yb * sign
            elevation = math.radians(rng.triangular(
                self._SHOULDER_ELEVATION_DOWN_DEG,
                self._SHOULDER_ELEVATION_DOWN_DEG,
                self._SHOULDER_ELEVATION_UP_DEG))
            azimuth = math.radians(rng.uniform(*self._SHOULDER_AZIMUTH_DEG_RANGE))
            raised = _rotate(rest_dir, xb, elevation * sign)
            upper_dir = _rotate(raised, zb, -azimuth * sign)
            elbow = shoulder + b['upper_arm'] * upper_dir

            flex = math.radians(rng.uniform(0.0, self._ELBOW_FLEX_MAX_DEG))
            forearm_dir = _rotate(upper_dir, xb, flex * sign)
            wrist = elbow + b['forearm'] * forearm_dir

            joints['{}Shoulder'.format(side)] = shoulder
            joints['{}Elbow'.format(side)] = elbow
            joints['{}Wrist'.format(side)] = wrist

            if self.include_hand:
                u, v, n = self._hand_frame(forearm_dir, side, sign, yb, zb)
                joints.update(self._hand_landmarks(
                    side, wrist, u, v, n, b['hand_length']))

        # --- 人物は常に原点に位置し、正面が x 軸正方向を向くようにする
        # (骨盤のヨー回転や水平方向のランダムなずらしは行わない)。---
        positions = joints

        # 姿勢に関わらず、体の最下点 (=両足) が必ず床 (z=0) に接する
        # ように上下移動する。足の開き・膝の曲げは左右対称なので、両足
        # は常に同じ高さのまま床に接地する。
        min_z = min(p[2] for p in positions.values())
        shift = np.array([0.0, 0.0, -min_z])
        joint_positions = {
            name: [float(v) for v in (p + shift)]
            for name, p in positions.items()}

        return dict(joint_positions=joint_positions, height=float(b['height']))

    def generate_many(self, num_samples):
        """``generate()`` を ``num_samples`` 回呼んだ結果をリストで返す."""
        return [self.generate() for _ in range(num_samples)]


def save_json(pose, path):
    """``RandomSkeletonGenerator.generate()`` の戻り値を JSON として保存する."""
    with open(path, 'w') as f:
        json.dump(pose, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='MediaPipe 形式の人体骨格をランダムに生成し、JSON として '
                    '保存する (SMPL には依存しない)。')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='生成する人物 (JSON) の数。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='JSON の保存先ディレクトリ。')
    parser.add_argument('--height-range', type=float, nargs=2, default=None,
                        metavar=('MIN_M', 'MAX_M'),
                        help='身長の生成範囲 [m] (既定 1.4 1.9)。')
    parser.add_argument('--seed', type=int, default=None,
                        help='乱数シード (指定すると再現可能になる)。')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    generator = RandomSkeletonGenerator(
        height_range=args.height_range, seed=args.seed)

    for i in range(args.num_samples):
        pose = generator.generate()
        out_path = os.path.join(args.output_dir, 'human_{:03d}.json'.format(i))
        save_json(pose, out_path)
        print('[{}/{}] saved {}'.format(i + 1, args.num_samples, out_path))


if __name__ == '__main__':
    main()
