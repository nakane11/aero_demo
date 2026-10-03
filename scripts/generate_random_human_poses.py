#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""SMPL の人体モデルをランダムに生成し、MediaPipe と同じ関節名の骨格
(手のランドマーク含む) を作って、両方を JSON に保存する。

座標はロボット座標系 (x=前, y=左, z=上)、床 z=0 に接地。SMPL モデル
ファイルはライセンス上同梱しないのでローカルパスを渡す。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py \
        --num-samples 100 --output-dir /tmp/random_human_poses
"""

import argparse
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

# MediaPipe と同じ関節名 (people_pose_types.INDEX2LIMBNAME, 'Bkg' を除く)。
BODY_JOINT_NAMES = [
    'Nose', 'Neck', 'RShoulder', 'LShoulder', 'RElbow', 'LElbow',
    'RWrist', 'LWrist', 'RHip', 'LHip', 'RKnee', 'LKnee',
    'RAnkle', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar',
]

HAND_JOINT_NAMES = ['{}Hand{}'.format(side, i)
                    for side in ('R', 'L') for i in range(21)]

HAND_LOCAL = people_pose_types.HAND_LOCAL_LANDMARKS

# 手の長さ / 身長。
_HAND_LENGTH_HEIGHT_RATIO = 0.108

# 頭部ランドマークの Neck からのオフセット [m]。
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
    """``v`` を ``axis`` まわりに ``angle`` [rad] 回転する."""
    axis = _unit(np.asarray(axis, dtype=np.float64))
    return vector_utils.rotate(v, axis, angle)


class RandomSmplHumanGenerator(object):
    """SMPL の人体モデル (体型 + 姿勢) をランダムに生成する.

    関節の可動域は解剖学的な範囲に限る。両足接地のため root と足首は
    固定し、股関節の外転と膝の曲げは左右対称にする。
    """

    _STANCE_ABDUCTION_MAX_DEG = 25.0    # 足の開き (股関節の外転)
    _KNEE_FLEX_MAX_DEG = 45.0           # 膝の曲げ方
    _ELBOW_FLEX_MAX_DEG = 130.0         # 肘の曲げ方 (0=伸展のみ, ヒンジ関節)
    _SPINE_SWAY_MAX_DEG = 6.0           # 脊柱のごく僅かな傾き
    _NECK_SWAY_MAX_DEG = 8.0            # 首のごく僅かな傾き

    # 鉛直軸まわりのひねり (正で左を向く)。sway は最小回転でヨーを持たない
    # ので別に与える。
    _SPINE_TWIST_MAX_DEG = 25.0         # 腰 (体幹) のひねり
    _NECK_TWIST_MAX_DEG = 45.0          # 首のひねり

    # 肩: T-pose 基準の仰角 (-90=真下, +90=真上) と方位角 (正=前)。
    # 仰角は腕を下ろした状態を最頻値とする三角分布。
    _SHOULDER_ELEVATION_DOWN_DEG = -90.0
    _SHOULDER_ELEVATION_UP_DEG = 90.0
    _SHOULDER_AZIMUTH_DEG_RANGE = (-40.0, 110.0)

    # 前腕軸まわりの手首のひねり (回内/回外)。
    _WRIST_TWIST_MAX_DEG = 90.0

    # betas: 正規分布からサンプルしてクリップ。
    _BETAS_STD = 1.5
    _BETAS_CLIP = 3.0

    # 身長の上限 [m]。超えたら betas を引き直す。
    _MAX_HEIGHT_M = 1.7
    _MAX_HEIGHT_RESAMPLE_ATTEMPTS = 100

    def __init__(self, models, seed=None):
        """
        Parameters
        ----------
        models : list of (gender, smpl_body.SmplModel)
            ``generate()`` のたびにこの中から 1 つを選ぶ。
        seed : int, optional
            乱数シード。
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
        """ひねり (``zb`` まわり) の後に傾き (最小回転) を掛けた局所回転 (3, 3).

        ``xb``, ``zb`` は親の座標系での前方と上。
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
            ``gender``, ``model``, ``betas`` (10,), ``pose`` (24, 3),
            ``root_pos`` (3,), ``vertices`` (6890, 3), ``joints`` (24, 3),
            ``wrist_rots`` ({'L'/'R': 手首の T-pose からの累積回転 (3, 3)}),
            ``head_rot`` (首の累積回転 (3, 3))。いずれもロボット座標系。
        """
        rng = self.rng
        gender, model = self.models[rng.randint(len(self.models))]
        betas = self._sample_betas()

        xb = np.array([1.0, 0.0, 0.0])  # front
        yb = np.array([0.0, 1.0, 0.0])  # left
        zb = np.array([0.0, 0.0, 1.0])  # up

        pose = np.zeros((24, 3))
        # root (pelvis) の向きは単位行列に固定。
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

        # --- 脚 (左右対称) ---
        stance = math.radians(rng.uniform(0.0, self._STANCE_ABDUCTION_MAX_DEG))
        knee_flex = math.radians(rng.uniform(0.0, self._KNEE_FLEX_MAX_DEG))
        for hip_idx, knee_idx, ankle_idx, sign in (
                (smpl_body.L_HIP, smpl_body.L_KNEE, smpl_body.L_ANKLE, 1.0),
                (smpl_body.R_HIP, smpl_body.R_KNEE, smpl_body.R_ANKLE, -1.0)):
            thigh_dir = _rotate(-zb, xb, stance * sign)
            swing(hip_idx, knee_idx, thigh_dir)
            shank_dir = _rotate(thigh_dir, yb, knee_flex)
            swing(knee_idx, ankle_idx, shank_dir)

        # --- 胴体・首: Spine1 だけ動かし、Spine2/Spine3/Collar はそれを引き継ぐ ---
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

        # --- 腕 (左右独立): 肩の仰角・方位角 + 肘 + 手首のひねり ---
        # 手首のひねりは pose[wrist_idx] に書く (メッシュとランドマークの
        # 掌の向きを一致させるため)。
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

            # 前腕の T-pose 方向 (局所軸) まわりにひねる。
            forearm_rest_dir = _unit(
                smpl_body.PERM.dot(model.J[wrist_idx] - model.J[elbow_idx]))
            twist = math.radians(
                rng.uniform(-self._WRIST_TWIST_MAX_DEG, self._WRIST_TWIST_MAX_DEG))
            R_local = smpl_body.rodrigues(forearm_rest_dir * twist)
            pose[wrist_idx] = smpl_body.mat_to_axis_angle(
                smpl_body.to_smpl_rotation(R_local))
            cumulative[wrist_idx] = cumulative[elbow_idx].dot(R_local)
            wrist_rots[side] = cumulative[wrist_idx]

        vertices, joints = smpl_body.forward_world(
            model, pose, betas, root_pos=np.zeros(3))

        # 身長上限を超えたら betas だけ引き直す (pose はそのまま)。
        for _ in range(self._MAX_HEIGHT_RESAMPLE_ATTEMPTS):
            height = float(vertices[:, 2].max() - vertices[:, 2].min())
            if height <= self._MAX_HEIGHT_M:
                break
            betas = self._sample_betas()
            vertices, joints = smpl_body.forward_world(
                model, pose, betas, root_pos=np.zeros(3))

        # 最下点を床 (z=0) に接地させる。
        min_z = float(vertices[:, 2].min())
        root_pos = np.array([0.0, 0.0, -min_z])
        vertices = vertices + root_pos
        joints = joints + root_pos

        return dict(gender=gender, model=model, betas=betas, pose=pose,
                   root_pos=root_pos, vertices=vertices, joints=joints,
                   wrist_rots=wrist_rots, head_rot=cumulative[smpl_body.NECK])


class RandomSkeletonGenerator(object):
    """SMPL の人モデルから MediaPipe 形式の骨格を作る.

    胴体・四肢は SMPL の関節位置そのもの。手のランドマークは SMPL の
    手首の累積回転 (``wrist_rots``) から組み立てるのでメッシュと一致する。
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
        self.include_hand = include_hand

    @staticmethod
    def _hand_frame(wrist_rot, rest_dir_robot, side):
        """手首の局所座標系 (u=指方向, v=親指側, n=掌の向き) を返す.

        T-pose の基準フレーム (u0=前腕方向, n0=``smpl_body._REST_PALM_NORMAL``,
        v0=親指側 +X) に ``wrist_rot`` (ロボット座標系) を掛けて作る。
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
        """21 個の手のランドマーク位置を作る."""
        basis = np.vstack([u, v, n])
        pts = wrist + hand_length * HAND_LOCAL.dot(basis)
        return {'{}Hand{}'.format(side, i): pts[i] for i in range(len(pts))}

    def generate(self, smpl_person):
        """``RandomSmplHumanGenerator.generate()`` の結果から骨格を作る.

        Returns
        -------
        dict
            ``joint_positions`` (関節名 -> [x, y, z]) と ``height`` [m]。
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
        # 顔の正面: head_rot で回した +x の、首->頭の軸に直交する成分。
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
    """SMPL モデルと骨格を保存用の dict (``skeleton`` と ``smpl``) にまとめる."""
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
    """SMPL モデルを ``[(gender, model), ...]`` でロードする (女性は任意)."""
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
        description='SMPL の人体モデルと MediaPipe 形式の骨格をランダムに '
                    '生成し JSON に保存する。')
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
        help='SMPL (女性) モデル .pkl のパス (無ければ男性のみ)。')
    parser.add_argument('--seed', type=int, default=None,
                        help='乱数シード。')
    parser.add_argument(
        '--start-index', type=int, default=0,
        help='ファイル名の連番の開始値 (追記時は --seed も変えること)。')
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
