#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人物骨格を skrobot の viewer 向けの線分 (``LineString``) にする部品."""

import numpy as np
import trimesh

from skrobot.model.primitives import LineString
from skrobot.model.primitives import Sphere

# 部位ごとの骨格の色
COLOR_BONES = {
    'torso': [220, 220, 220, 255],
    'head': [255, 220, 150, 255],
    'rarm': [255, 110, 110, 255],
    'larm': [110, 170, 255, 255],
    'rleg': [255, 170, 70, 255],
    'lleg': [70, 210, 255, 255],
    'rhand': [255, 60, 60, 255],
    'lhand': [60, 120, 255, 255],
}

# 関節 -> 部位 (先に一致したもの。無ければ torso)
_BONE_GROUPS = (
    ('head', ('Nose', 'REye', 'LEye', 'REar', 'LEar')),
    ('rarm', ('RElbow', 'RWrist')),
    ('larm', ('LElbow', 'LWrist')),
    ('rleg', ('RKnee', 'RAnkle')),
    ('lleg', ('LKnee', 'LAnkle')),
)


def set_color(link, rgba):
    """primitive の色を塗る.

    viewer は ``concatenated_visual_mesh`` (構築時の複製) を描くのでそちらを
    塗る。ただし Sphere は ViserViewer が ``visual_mesh`` を直接読むので
    そちらを塗る。半透明なら viser (glTF) でアルファが効くよう
    ``alphaMode='BLEND'`` の PBRMaterial にする。
    """
    if isinstance(link, Sphere):
        meshes = getattr(link, 'visual_mesh', None)
        if meshes is None:
            return link
        if not isinstance(meshes, (list, tuple)):
            meshes = [meshes]
        for mesh in meshes:
            try:
                mesh.visual.face_colors = rgba
            except Exception:  # 描画できないほどのことではない
                pass
        return link

    mesh = getattr(link, 'concatenated_visual_mesh', None)
    if mesh is None:
        mesh = getattr(link, 'visual_mesh', None)
    if mesh is None:
        return link
    meshes = mesh if isinstance(mesh, (list, tuple)) else [mesh]
    for m in meshes:
        # 点群は face_colors を黙って無視するので vertex_colors を使う。
        has_faces = getattr(m, 'faces', None) is not None
        attr = 'face_colors' if has_faces else 'vertex_colors'
        try:
            setattr(m.visual, attr, rgba)
        except Exception:  # 描画できないほどのことではない
            pass
        if has_faces and len(rgba) >= 4 and rgba[3] < 255:
            try:
                m.visual = trimesh.visual.TextureVisuals(
                    material=trimesh.visual.material.PBRMaterial(
                        baseColorFactor=[c / 255.0 for c in rgba],
                        alphaMode='BLEND'))
            except Exception:
                pass
    return link


def bone_group(name):
    """ボーン名 ("Neck->RShoulder" 形式) から部位 (``COLOR_BONES`` のキー) を返す."""
    if 'RHand' in name:
        return 'rhand'
    if 'LHand' in name:
        return 'lhand'
    joints = name.split('->')
    for group, members in _BONE_GROUPS:
        if any(joint in members for joint in joints):
            return group
    return 'torso'


def bone_color(name):
    """ボーン名から部位の色を返す."""
    return COLOR_BONES[bone_group(name)]


def bone_line(bone, rgba):
    """1 本のボーンを (太さを持たない) 細い線にして返す."""
    start = np.asarray(bone.start_point, dtype=np.float64)
    end = np.asarray(bone.end_point, dtype=np.float64)
    return LineString(np.stack([start, end]), color=rgba)
