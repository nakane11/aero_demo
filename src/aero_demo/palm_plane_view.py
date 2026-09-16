#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人物骨格を skrobot の viewer 向けの線分 (``LineString``) にする部品.

``scripts/draw_random_human_poses.py`` 等が使う。

rospy は import しない。描けなかったことは例外ではなく戻り値で返すので、
ログをどう出すかは呼び出し側が決める。
"""

import numpy as np
import trimesh

from skrobot.model.primitives import LineString
from skrobot.model.primitives import Sphere

# 骨格の線は部位ごとに色を変える。ボーン名 ("Neck->RShoulder" 形式) から
# 下の bone_color で引く。
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

# どの関節が出てきたらその部位、という判定 (先に一致したものを採る)。
# ここに無い関節 (Neck, RShoulder, LShoulder, RHip, LHip) は torso 扱い。
_BONE_GROUPS = (
    ('head', ('Nose', 'REye', 'LEye', 'REar', 'LEar')),
    ('rarm', ('RElbow', 'RWrist')),
    ('larm', ('LElbow', 'LWrist')),
    ('rleg', ('RKnee', 'RAnkle')),
    ('lleg', ('LKnee', 'LAnkle')),
)


def set_color(link, rgba):
    """primitive の色を塗る (skrobot のバージョン差を吸収する).

    ``Sphere`` 以外の面付き primitive (Box/Cylinder/Capsule/MeshLink) は
    ``link.visual_mesh`` ではなく ``link.concatenated_visual_mesh`` を塗る。
    ``skrobot.model.link.Link.__init__`` は ``visual_mesh`` を
    ``trimesh.util.concatenate`` に通した*複製*を ``_concatenated_visual_
    mesh`` としてキャッシュし、trimesh/viser/pyrender のどの viewer も
    描画時にはそのキャッシュだけを読む (``concatenated_visual_mesh``
    プロパティ経由)。元の ``visual_mesh`` は Link 構築後の複製の元に
    なるだけで、以後は描画に使われない -- なので Link を作ってから (この
    関数のように) 色を塗っても ``visual_mesh`` を塗るだけでは viewer に
    反映されない。

    半透明 (``rgba[3] < 255``) なときは ``face_colors`` に加えて
    ``alphaMode='BLEND'`` の ``PBRMaterial`` も付ける。ViserViewer は
    メッシュを glTF (.glb) に変換して送るが (``skrobot.viewers._viser
    .ViserViewer._add_link`` の ``add_mesh_trimesh``)、trimesh の頂点色
    エクスポートは既定で ``alphaMode: OPAQUE`` になりアルファ値を無視する
    -- ``face_colors`` だけでは viser 側で不透明に見えてしまう。

    ``Sphere`` だけは例外で、従来どおり ``link.visual_mesh`` を塗る。
    ViserViewer は Sphere を icosphere として特別扱いする際、concatenated
    ではなく ``link.visual_mesh.visual.face_colors`` を直接読むため
    (``_add_link``) -- こちらは Link 構築後に塗っても効く。
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
        # 面を持つ primitive は face_colors、点群は vertex_colors。点群に
        # face_colors を入れても例外にはならず、色が付かないまま無視される
        # (trimesh.PointCloud.colors が空のままになる) ので型で振り分ける。
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
    """ボーン名 ("Neck->RShoulder" 形式) から部位 (``COLOR_BONES`` のキー) を返す.

    手のランドマークのボーン ("RHand0->RHand1" など) は手の部位にまとめる。
    """
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
