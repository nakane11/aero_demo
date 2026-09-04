#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""SMPL (Skinned Multi-Person Linear model) を chumpy 無しで動かす.

``palm_plane_view.PalmPlaneScene`` が人物の胴体・頭を実体メッシュで描く
のに使う。skrobot / rospy には依存しない (``palm_plane.py`` と同じ方針)。

公式配布の SMPL v1.0.0 ``.pkl`` (``smpl_webuser`` 付属) は付属コードが
``chumpy`` でシリアライズしており、``shapedirs`` フィールドが
``chumpy.ch.Ch`` として pickle されているため、素の ``pickle.load`` は
``ModuleNotFoundError: chumpy`` で失敗する。本物の chumpy は
(numpy>=1.24 で削除された ``np.bool``/``np.int`` 等の別名に依存して
いて) この環境の numpy と非互換なのでインストールしない。代わりに
``chumpy.ch.Ch`` の pickle プロトコル (``__setstate__`` が
``self.__dict__.update(d)`` するだけ) を模した最小限の shim を使う。

SMPL のモデルファイル自体はライセンス上リポジトリに同梱できないので、
呼び出し側がローカルパスを渡す (例:
``~/SMPL_python_v.1.0.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl``)。

軸規約 (実測して決定): SMPL のローカル座標系は
axis0=左(+)/右(-), axis1=上(+)/下(-), axis2=前(+)/後(-) の右手系。
ロボット座標系 (x=前, y=左, z=上) との対応は
``robot_x=smpl_z, robot_y=smpl_x, robot_z=smpl_y`` (``PERM`` 参照)。
"""

import os
import pickle
import sys
import types
from dataclasses import dataclass

import numpy as np

from aero_demo.vector_utils import unit as _unit


@dataclass
class SmplModel:
    v_template: np.ndarray   # (6890, 3)
    shapedirs: np.ndarray    # (6890, 3, 10)
    posedirs: np.ndarray     # (6890, 3, 207)
    J_regressor: np.ndarray  # (24, 6890), dense
    weights: np.ndarray      # (6890, 24)
    parent: np.ndarray       # (24,) int, parent[0] == -1 (root)
    f: np.ndarray            # (F, 3) triangle faces
    J: np.ndarray            # (24, 3) rest-pose joint locations


class _ChumpyChShim(object):
    """``chumpy.ch.Ch`` の unpickle だけを肩代わりする最小限のダミー.

    本物の ``Ch`` は演算グラフ (自動微分) のノードだが、ここで欲しいのは
    leaf に入っている定数配列だけなので、``__setstate__`` で状態を
    ``self.__dict__`` にそのまま溜め込む以上のことはしない。
    """

    def __setstate__(self, state):
        self.__dict__.update(state)


def _unpickle_with_chumpy_shim(path):
    """``chumpy`` をインストールせずに、pickle 内の ``chumpy.ch.Ch``
    参照だけ ``_ChumpyChShim`` に差し替えて読む."""
    saved = {name: sys.modules.get(name) for name in ('chumpy', 'chumpy.ch')}
    chumpy_pkg = types.ModuleType('chumpy')
    chumpy_ch = types.ModuleType('chumpy.ch')
    chumpy_ch.Ch = _ChumpyChShim
    chumpy_pkg.ch = chumpy_ch
    sys.modules['chumpy'] = chumpy_pkg
    sys.modules['chumpy.ch'] = chumpy_ch
    try:
        with open(path, 'rb') as f:
            return pickle.load(f, encoding='latin1')
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def _as_array(value):
    """chumpy shim / scipy 疎行列 / 生 ndarray のいずれからも ndarray を取り出す."""
    if isinstance(value, _ChumpyChShim):
        return np.asarray(value.__dict__['x'], dtype=np.float64)
    if hasattr(value, 'toarray'):
        return np.asarray(value.toarray(), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def load_smpl_model(pkl_path):
    """SMPL v1.0.0 の ``.pkl`` を読んで ``SmplModel`` を返す.

    ファイルが無い/読めないときは例外を送出する (呼び出し側で catch して
    フォールバックする想定 -- ``palm_plane_view.PalmPlaneScene``)。
    """
    path = os.path.expanduser(pkl_path)
    dd = _unpickle_with_chumpy_shim(path)

    v_template = _as_array(dd['v_template'])
    shapedirs = _as_array(dd['shapedirs'])
    posedirs = _as_array(dd['posedirs'])
    J_regressor = _as_array(dd['J_regressor'])
    weights = _as_array(dd['weights'])
    kintree_table = np.asarray(dd['kintree_table'])
    f = np.asarray(dd['f'], dtype=np.int64)

    parent = kintree_table[0].astype(np.int64)
    parent[0] = -1  # kintree_table[0, 0] は "親なし" を表す uint32 の -1

    J = J_regressor.dot(v_template)

    return SmplModel(v_template=v_template, shapedirs=shapedirs,
                     posedirs=posedirs, J_regressor=J_regressor,
                     weights=weights, parent=parent, f=f, J=J)


# ----------------------------------------------------------------------
# forward kinematics (LBS)
# ----------------------------------------------------------------------
def rodrigues(r):
    """axis-angle ``r`` (3,) -> 回転行列 (3, 3) (Rodrigues の回転公式)."""
    r = np.asarray(r, dtype=np.float64)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    K = np.array([[0.0, -k[2], k[1]],
                 [k[2], 0.0, -k[0]],
                 [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * K.dot(K)


def smpl_forward(model, pose, betas, trans):
    """SMPL の順運動学 (Linear Blend Skinning).

    Parameters
    ----------
    model : SmplModel
    pose : (24, 3) array_like
        各関節の axis-angle (親関節相対)。
    betas : (10,) array_like
    trans : (3,) array_like

    Returns
    -------
    vertices : (6890, 3) ndarray
    joints : (24, 3) ndarray
        姿勢を反映した後の関節位置。
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(24, 3)
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    trans = np.asarray(trans, dtype=np.float64).reshape(3)

    v_shaped = model.v_template + np.tensordot(
        model.shapedirs, betas, axes=([2], [0]))
    J = model.J_regressor.dot(v_shaped)

    R = np.stack([rodrigues(pose[i]) for i in range(24)])
    pose_feature = (R[1:] - np.eye(3)).reshape(-1)
    v_posed = v_shaped + model.posedirs.dot(pose_feature)

    G = np.zeros((24, 4, 4))
    G[0, :3, :3] = R[0]
    G[0, :3, 3] = J[0]
    G[0, 3, 3] = 1.0
    for i in range(1, 24):
        p = model.parent[i]
        local = np.eye(4)
        local[:3, :3] = R[i]
        local[:3, 3] = J[i] - J[p]
        G[i] = G[p].dot(local)

    # rest-pose の関節位置の寄与を抜く (標準の SMPL のトリック)
    G_rel = G.copy()
    for i in range(24):
        G_rel[i, :3, 3] -= G[i, :3, :3].dot(J[i])

    T = np.tensordot(model.weights, G_rel, axes=([1], [0]))   # (6890, 4, 4)
    v_posed_h = np.concatenate(
        [v_posed, np.ones((v_posed.shape[0], 1))], axis=1)
    vertices = np.einsum('nij,nj->ni', T, v_posed_h)[:, :3] + trans
    joints = G[:, :3, 3] + trans
    return vertices, joints


# ----------------------------------------------------------------------
# retargeting: Person3D の関節位置 -> SMPL の pose
# ----------------------------------------------------------------------

# ロボット座標系 (x=前, y=左, z=上) <- SMPL ローカル座標系
# (axis0=左, axis1=上, axis2=前) への変換 (実測して決定, モジュール
# docstring 参照)。``generate_random_human_poses.py`` の SMPL 姿勢生成
# クラスも同じ変換を使うので公開名にしてある。
PERM = np.array([[0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]])

# SMPL 標準の 24 関節順序 (kintree_table の親子関係と一致することを
# 確認済み)。``generate_random_human_poses.py`` からも参照するので公開名
# にしてある。
PELVIS = 0
L_HIP, R_HIP = 1, 2
L_KNEE, R_KNEE = 4, 5
L_ANKLE, R_ANKLE = 7, 8
NECK = 12
HEAD = 15
L_SHOULDER, R_SHOULDER = 16, 17
L_ELBOW, R_ELBOW = 18, 19
L_WRIST, R_WRIST = 20, 21
L_HAND, R_HAND = 22, 23

# (pose を設定する関節, その先のボーンの終点関節,
#  ロボット側の始点関節名, ロボット側の終点関節名)。
# 親 -> 子の順で並べる (swing を計算するとき、親の回転が先に確定して
# いる必要があるため)。
# 手首 (LWrist/RWrist) -> 中指の付け根 (LHand9/RHand9, MediaPipe hand
# landmark の middle MCP) は、手のランドマークが取れているときだけ手首の
# 曲げ (wrist_pitch 相当) を反映するために追加している -- これが無いと
# pose[20]/pose[21] が常に 0 (肘の回転をそのまま継承するだけ) になり、
# 手のランドマークのカプセル (実際の手の向き) と SMPL の手先が指す向きが
# ずれて見える。手首の捻り (wrist_roll, 前腕軸まわり) は、指の付け根
# (knuckle) のランドマークが 3 点以上あれば ``_hand_plane_normal`` で
# 掌の向きを推定してそちらに合わせる (下の pose_idx が L_WRIST/R_WRIST
# の場合の特別扱い、_twist_from_normal 参照)。それも無理なら以前どおり
# 無視する (曲げのみで捻りは 0)。
_LIMB_CHAINS = [
    (L_HIP, L_KNEE, 'LHip', 'LKnee'),
    (L_KNEE, L_ANKLE, 'LKnee', 'LAnkle'),
    (R_HIP, R_KNEE, 'RHip', 'RKnee'),
    (R_KNEE, R_ANKLE, 'RKnee', 'RAnkle'),
    (L_SHOULDER, L_ELBOW, 'LShoulder', 'LElbow'),
    (L_ELBOW, L_WRIST, 'LElbow', 'LWrist'),
    (R_SHOULDER, R_ELBOW, 'RShoulder', 'RElbow'),
    (R_ELBOW, R_WRIST, 'RElbow', 'RWrist'),
    (L_WRIST, L_HAND, 'LWrist', 'LHand9'),
    (R_WRIST, R_HAND, 'RWrist', 'RHand9'),
]


# 掌の向き (捻り) 推定に使う手のランドマーク: 手首 + 4 knuckle。符号解決
# の考え方 (index MCP は親指側、pinky MCP は小指側という MediaPipe の
# レイアウトから左右を判定する) は palm_plane.py の PLANE_LANDMARKS /
# MCP_LATERAL_RANK と同じだが、smpl_body は (chumpy 回避と同じ理由で)
# rospy/ROS メッセージに依存しない方針なので import はせず、必要な部分
# だけここに複製してある。
_HAND_PLANE_INDICES = (0, 5, 9, 13, 17)
_HAND_MCP_RANK = {5: 1.5, 9: 0.5, 13: -0.5, 17: -1.5}
_HAND_MIN_SPAN_RATIO = 0.15

# SMPL の T-pose (pose=0, 腕を横に伸ばした姿勢) は掌が下 (ロボット座標系
# の -Z) を向く -- 手首の捻りを常に 0 のままにすると、この rest の
# 向きがそのまま残ってしまい (捻りが要る腕の角度でも) 掌が下を向いて
# 見えるのはこのため。捻りを推定できたときの「0 捻り」の基準として使う。
_REST_PALM_NORMAL = np.array([0.0, 0.0, -1.0])


def _hand_plane_normal(joints, prefix, hand, finger_dir):
    """手のランドマーク (wrist + knuckle) から掌の向き (法線) を推定する.

    ``palm_plane.fit_palm_plane`` の簡易版 (RMS のような詳しいゲートは
    省く -- ここでの用途は SMPL メッシュの捻りをおおよそ合わせる目安で
    あって、接触計算のような厳密さは要らないため)。3 点未満、点がほぼ
    一直線、またはナックルの左右判定に使える点が 2 点未満なら ``None``
    (呼び出し側は捻りを諦めて曲げのみにフォールバックする)。
    """
    points = {}
    for i in _HAND_PLANE_INDICES:
        key = '{}{}'.format(prefix, i)
        if key in joints:
            points[i] = np.asarray(joints[key], dtype=np.float64)
    if len(points) < 3:
        return None
    idxs = sorted(points)
    pts = np.array([points[i] for i in idxs], dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    if len(s) < 3 or s[0] < 1e-9 or s[1] / s[0] < _HAND_MIN_SPAN_RATIO:
        return None
    normal = _unit(vt[2])
    if normal is None:
        return None

    lateral = np.zeros(3)
    n_ranked = 0
    for i, rank in _HAND_MCP_RANK.items():
        if i in points:
            lateral += rank * (points[i] - centroid)
            n_ranked += 1
    if n_ranked < 2:
        return None
    v_axis = _unit(lateral - float(np.dot(lateral, finger_dir)) * finger_dir)
    if v_axis is None:
        return None
    anatomical_normal = (np.cross(v_axis, finger_dir) if hand == 'R'
                         else np.cross(finger_dir, v_axis))
    if float(np.dot(normal, anatomical_normal)) < 0.0:
        normal = -normal
    return normal


def _frame_from_axes(primary, secondary):
    """``primary`` を第 1 軸, それに直交化した ``secondary`` を第 2 軸,
    残りを第 3 軸にした正規直交フレーム (3, 3) を返す (直交化できなければ
    ``None``)."""
    u1 = _unit(primary)
    if u1 is None:
        return None
    u2 = _unit(secondary - float(np.dot(secondary, u1)) * u1)
    if u2 is None:
        return None
    u3 = np.cross(u1, u2)
    return np.column_stack([u1, u2, u3])


def _twist_from_normal(rest_dir, rest_normal, obs_dir, obs_normal):
    """(rest_dir, rest_normal) の 2 軸フレームを (obs_dir, obs_normal) に
    重ねる回転 (3, 3) を返す -- 向き (rest_dir -> obs_dir) だけでなく
    掌の法線も合わせるので、前腕軸まわりの捻りまで込みになる。どちらかの
    軸が直交化できなければ ``None``。
    """
    source = _frame_from_axes(rest_dir, rest_normal)
    target = _frame_from_axes(obs_dir, obs_normal)
    if source is None or target is None:
        return None
    return target.dot(source.T)


def rotation_between(a, b):
    """単位ベクトル ``a`` を ``b`` に重ねる最小回転 (3, 3)."""
    axis = np.cross(a, b)
    n = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if n < 1e-8:
        if dot > 0.0:
            return np.eye(3)
        # 180 度反転: a に垂直な適当な軸を選ぶ
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 \
            else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, perp))
        return rodrigues(axis * np.pi)
    axis = axis / n
    angle = np.arccos(dot)
    return rodrigues(axis * angle)


def mat_to_axis_angle(R):
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # 180 度付近は反対称部分が消えるので対角成分から軸を復元する
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        if A[0, 1] < 0:
            axis[1] = -axis[1]
        if A[0, 2] < 0:
            axis[2] = -axis[2]
        return axis * theta
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * np.sin(theta))
    return axis * theta


def to_smpl_rotation(R_robot):
    """ロボット座標系の回転行列を SMPL ローカル座標系の回転行列に直す.

    ``PERM`` は座標成分の並べ替えでしかない (関節ごとの回転ではない)
    ので、ロボット座標系の成分で計算した回転 ``R_robot`` をそのまま
    ``smpl_forward`` (SMPL 自身の座標系で ``pose`` を解釈する) に渡すと
    座標系がずれる。``v_robot = PERM @ v_smpl`` なので、共役
    ``R_smpl = PERM.T @ R_robot @ PERM`` を取って渡す必要がある。
    """
    return PERM.T.dot(R_robot).dot(PERM)


def retarget_and_pose(model, joints, betas=None):
    """1 人分の関節位置 (ロボット座標系) から姿勢済みの SMPL 頂点を作る.

    Parameters
    ----------
    model : SmplModel
    joints : dict
        MediaPipe 関節名 (``Neck``, ``RShoulder``, ... ``Person3D`` が
        使うものと同じ) -> ``np.ndarray([x, y, z])`` (ロボット座標系)。
        ``palm_plane_view.update_skeleton`` が集める ``visible_joints``
        と (fake 推定だけが持つ) ``hidden_positions`` 由来の関節を
        まとめたもの (見えている値を優先) を渡す想定。
    betas : (10,) array_like, optional
        SMPL の体型パラメータ (痩型/肥満型などの体格差)。``None`` なら
        平均体型 (全ゼロ) を使う。``Person3D.betas`` (fake 推定が人物
        ごとに 1 回だけ引く, ``fake_people_pose_estimator_ros.py`` 参照)
        をそのまま渡す想定 -- 実推定は体型を推定しないので常に ``None``。

    Returns
    -------
    (vertices, faces) : (ndarray(6890, 3), ndarray(F, 3)) or None
        胴体を組み立てるための関節 (``Neck`` + 両肩) が足りなければ
        ``None``。手首の曲げ (pitch) は手のランドマーク (``RHand9``/
        ``LHand9``, 中指の付け根) が取れていればそちらに合わせるが、
        前腕軸まわりの捻り (roll) は関節位置からは復元できないので
        常に無視する (既存のカプセル描画と同じ制約であり、後退ではない)。
    """
    if not ('Neck' in joints and 'RShoulder' in joints
            and 'LShoulder' in joints):
        return None

    hip_center = None
    if 'RHip' in joints and 'LHip' in joints:
        hip_center = (joints['RHip'] + joints['LHip']) / 2.0

    def permuted_dist(ia, ib):
        return float(np.linalg.norm(PERM.dot(model.J[ib] - model.J[ia])))

    # --- スケール: Neck と 足首(優先)/腰 の観測距離を SMPL 静止時の
    #     同じ距離で割って求める。どちらも無理なら 1.0 のまま。 ---
    scale = 1.0
    if 'LAnkle' in joints or 'RAnkle' in joints:
        if 'LAnkle' in joints:
            ankle_pos, ankle_idx = joints['LAnkle'], L_ANKLE
        else:
            ankle_pos, ankle_idx = joints['RAnkle'], R_ANKLE
        template = permuted_dist(NECK, ankle_idx)
        if template > 1e-6:
            scale = float(np.linalg.norm(joints['Neck'] - ankle_pos)) \
                / template
    elif hip_center is not None:
        hip_template = float(np.linalg.norm(PERM.dot(
            model.J[NECK] - (model.J[L_HIP] + model.J[R_HIP]) / 2.0)))
        if hip_template > 1e-6:
            scale = float(np.linalg.norm(joints['Neck'] - hip_center)) \
                / hip_template

    # --- ルートの向き・位置 (肩線 = 左右, Neck->腰中点 = 上) ---
    if hip_center is not None:
        world_up = _unit(joints['Neck'] - hip_center)
    else:
        world_up = None
    if world_up is None:
        world_up = np.array([0.0, 0.0, 1.0])

    world_left = _unit(joints['LShoulder'] - joints['RShoulder'])
    if world_left is None:
        return None
    world_left = _unit(world_left - np.dot(world_left, world_up) * world_up)
    if world_left is None:
        return None
    world_fwd = np.cross(world_left, world_up)
    root_rot = np.column_stack([world_fwd, world_left, world_up])

    if hip_center is not None:
        root_pos = hip_center
    else:
        # 腰が見えない: _head_sphere が Nose 不明時に固定オフセットで
        # 妥協するのと同じ発想で、SMPL 自身の Neck-Pelvis 静止距離
        # (スケール後) だけ Neck から下に置く。
        root_pos = joints['Neck'] \
            - world_up * (scale * permuted_dist(NECK, PELVIS))

    # --- 四肢の swing (捻りは無視) ---
    pose = np.zeros((24, 3))
    cumulative = {0: root_rot}

    def get_cumulative(idx):
        if idx not in cumulative:
            cumulative[idx] = get_cumulative(model.parent[idx])
        return cumulative[idx]

    for pose_idx, child_idx, robot_parent, robot_child in _LIMB_CHAINS:
        parent_rot = get_cumulative(model.parent[pose_idx])
        rest_dir_robot = _unit(PERM.dot(model.J[child_idx] - model.J[pose_idx]))
        matched = False
        if (rest_dir_robot is not None and robot_parent in joints
                and robot_child in joints):
            obs_dir_world = _unit(joints[robot_child] - joints[robot_parent])
            if obs_dir_world is not None:
                obs_dir_local = parent_rot.T.dot(obs_dir_world)
                R_local = rotation_between(rest_dir_robot, obs_dir_local)
                if pose_idx in (L_WRIST, R_WRIST):
                    hand = 'L' if pose_idx == L_WRIST else 'R'
                    normal_world = _hand_plane_normal(
                        joints, hand + 'Hand', hand, obs_dir_world)
                    if normal_world is not None:
                        target_normal_local = parent_rot.T.dot(normal_world)
                        R_local_full = _twist_from_normal(
                            rest_dir_robot, _REST_PALM_NORMAL,
                            obs_dir_local, target_normal_local)
                        if R_local_full is not None:
                            R_local = R_local_full
                pose[pose_idx] = mat_to_axis_angle(to_smpl_rotation(R_local))
                cumulative[pose_idx] = parent_rot.dot(R_local)
                matched = True
        if not matched:
            cumulative[pose_idx] = parent_rot

    # --- 首/頭: Neck -> Nose 方向に合わせる (無ければ中立のまま) ---
    if 'Nose' in joints:
        parent_rot = get_cumulative(model.parent[NECK])
        rest_dir_robot = _unit(PERM.dot(model.J[HEAD] - model.J[NECK]))
        obs_dir_world = _unit(joints['Nose'] - joints['Neck'])
        if rest_dir_robot is not None and obs_dir_world is not None:
            obs_dir_local = parent_rot.T.dot(obs_dir_world)
            R_local = rotation_between(rest_dir_robot, obs_dir_local)
            pose[NECK] = mat_to_axis_angle(to_smpl_rotation(R_local))

    # --- 頂点の生成・配置 ---
    betas = np.zeros(10) if betas is None else np.asarray(betas, dtype=np.float64)
    v_world, _ = forward_world(model, pose, betas, root_pos, root_rot, scale=scale)
    return v_world, model.f


def forward_world(model, pose, betas, root_pos, root_rot=None, scale=1.0):
    """SMPL の ``pose``/``betas`` から、ワールド (ロボット座標系) に配置
    した頂点・関節位置を返す.

    ``retarget_and_pose`` (骨格 -> SMPL) と ``generate_random_human_
    poses.RandomSmplHumanGenerator`` (SMPL をネイティブに乱数生成) の
    どちらも「pose/betas から姿勢済みの頂点・関節を作り、pelvis を原点に
    寄せてから ``PERM`` でロボット座標系に変換し、``root_rot``/``root_
    pos`` で好きな場所へ置く」という同じ手順を踏むので、その共通部分を
    ここに集約してある。

    Parameters
    ----------
    model : SmplModel
    pose : (24, 3) array_like
    betas : (10,) array_like
    root_pos : (3,) array_like
        pelvis をロボット座標系のどこに置くか。
    root_rot : (3, 3) array_like, optional
        pelvis の向き (ロボット座標系)。``None`` (既定) なら単位行列
        (体は常に +x を向く, ``generate_random_human_poses.py`` と同じ
        既定の置き方)。
    scale : float, optional
        SMPL の頂点・関節をこの倍率で拡大縮小する (``retarget_and_pose``
        が観測した身長に合わせるのに使う)。既定 1.0 (SMPL 自身の betas
        が表す実寸のまま使う, ``RandomSmplHumanGenerator`` はこちら)。

    Returns
    -------
    (vertices_world, joints_world) : (ndarray(6890, 3), ndarray(24, 3))
    """
    if root_rot is None:
        root_rot = np.eye(3)
    v_local, joints_local = smpl_forward(model, pose, betas, np.zeros(3))
    v_robot_local = (v_local - model.J[PELVIS]).dot(PERM.T)
    joints_robot_local = (joints_local - model.J[PELVIS]).dot(PERM.T)
    vertices_world = root_pos + scale * v_robot_local.dot(root_rot.T)
    joints_world = root_pos + scale * joints_robot_local.dot(root_rot.T)
    return vertices_world, joints_world
