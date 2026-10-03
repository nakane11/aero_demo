#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""SMPL 人体モデルを chumpy 無しで読み込み・順運動学する.

公式 ``.pkl`` は ``chumpy.ch.Ch`` を含むが、chumpy は新しい numpy と非互換
なので unpickle 用の shim で読む。モデルファイルは同梱しないので
呼び出し側がパスを渡す。

軸規約: SMPL は axis0=左, axis1=上, axis2=前。ロボット座標系 (x=前, y=左,
z=上) へは ``PERM`` で並べ替える。
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
    """``chumpy.ch.Ch`` の unpickle だけを肩代わりするダミー."""

    def __setstate__(self, state):
        self.__dict__.update(state)


def _unpickle_with_chumpy_shim(path):
    """``chumpy.ch.Ch`` を ``_ChumpyChShim`` に差し替えて pickle を読む."""
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
    """SMPL v1.0.0 の ``.pkl`` を読んで ``SmplModel`` を返す."""
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


def smpl_forward(model, pose, betas, trans, bone_scale=None):
    """SMPL の順運動学 (Linear Blend Skinning).

    Parameters
    ----------
    pose : (24, 3) array_like
        各関節の axis-angle (親関節相対)。
    bone_scale : dict, optional
        関節 index -> 親とのボーンの伸縮率。

    Returns
    -------
    vertices : (6890, 3) ndarray
    joints : (24, 3) ndarray
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
        offset = J[i] - J[p]
        if bone_scale is not None and i in bone_scale:
            offset = offset * bone_scale[i]
        local[:3, 3] = offset
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
# 座標変換・関節定数・回転ユーティリティ
# ----------------------------------------------------------------------

# v_robot = PERM @ v_smpl
PERM = np.array([[0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]])

# SMPL 標準の 24 関節 index
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

# T-pose (pose=0) の掌の向き (ロボット座標系)
_REST_PALM_NORMAL = np.array([0.0, 0.0, -1.0])


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
    """ロボット座標系の回転行列を SMPL 座標系に直す (PERM.T @ R @ PERM)."""
    return PERM.T.dot(R_robot).dot(PERM)


def forward_world(model, pose, betas, root_pos, root_rot=None, scale=1.0,
                  bone_scale=None):
    """pelvis を root_pos/root_rot (ロボット座標系) に置いた頂点・関節位置を返す.

    root_rot が None なら体は +x を向く。

    Returns
    -------
    (vertices_world, joints_world) : (ndarray(6890, 3), ndarray(24, 3))
    """
    if root_rot is None:
        root_rot = np.eye(3)
    v_local, joints_local = smpl_forward(
        model, pose, betas, np.zeros(3), bone_scale=bone_scale)
    v_robot_local = (v_local - model.J[PELVIS]).dot(PERM.T)
    joints_robot_local = (joints_local - model.J[PELVIS]).dot(PERM.T)
    vertices_world = root_pos + scale * v_robot_local.dot(root_rot.T)
    joints_world = root_pos + scale * joints_robot_local.dot(root_rot.T)
    return vertices_world, joints_world
