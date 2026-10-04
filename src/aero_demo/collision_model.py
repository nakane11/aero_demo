#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Aero の干渉モデル (メッシュをプリミティブ近似した URDF) を作る.

自動変換で表せない部分は ``EXTRA_COLLISION_BOXES`` の box を別リンクとして
足す (scikit-robot の干渉判定は 1 リンク 1 形状が前提のため)。
"""
from pathlib import Path
import re

from skrobot.urdf import convert_meshes_to_primitives


# 足すリンク: {名前: (親リンク, 親座標系での box 中心 xyz, size)} [m]。
# visual と collision の両方をこの box にする。
# wheel_base_front_link: 台車前方の高い部分 (元の台車 box と前端・下端・
# 幅をそろえ、奥行き 0.25、上端は地面から 0.32。wheel_base_link は z=0.075)。
_WHEEL_BASE_FRONT_X = 0.018 + 0.72 / 2
_WHEEL_BASE_FRONT_DEPTH = 0.25
_WHEEL_BASE_BOTTOM_Z = 0.0375 - 0.149 / 2
_WHEEL_BASE_FRONT_TOP_Z = 0.32 - 0.075
EXTRA_COLLISION_BOXES = {
    'wheel_base_front_link': (
        'wheel_base_link',
        (_WHEEL_BASE_FRONT_X - _WHEEL_BASE_FRONT_DEPTH / 2, 0.0,
         (_WHEEL_BASE_BOTTOM_Z + _WHEEL_BASE_FRONT_TOP_Z) / 2),
        (_WHEEL_BASE_FRONT_DEPTH, 0.52,
         _WHEEL_BASE_FRONT_TOP_Z - _WHEEL_BASE_BOTTOM_Z)),
}


def _format_box_geometry(tag, link_name, xyz, size):
    return (
        '    <{tag}>\n'
        '      <origin xyz="{x:.6g} {y:.6g} {z:.6g}" rpy="0 0 0"/>\n'
        '      <material name="{link}_{tag}_material">'
        '<color rgba="0.39 0.39 0.37 1.0"/></material>\n'
        '      <geometry>\n'
        '        <box size="{sx:.6g} {sy:.6g} {sz:.6g}"/>\n'
        '      </geometry>\n'
        '    </{tag}>\n').format(
            tag=tag, link=link_name, x=xyz[0], y=xyz[1], z=xyz[2],
            sx=size[0], sy=size[1], sz=size[2])


def _format_extra_link(link_name, parent_name, xyz, size):
    return (
        '  <link name="{link}">\n'
        '{visual}{collision}'
        '  </link>\n'
        '  <joint name="{link}_joint" type="fixed">\n'
        '    <parent link="{parent}"/>\n'
        '    <child link="{link}"/>\n'
        '    <origin xyz="0 0 0" rpy="0 0 0"/>\n'
        '  </joint>\n').format(
            link=link_name, parent=parent_name,
            visual=_format_box_geometry('visual', link_name, xyz, size),
            collision=_format_box_geometry('collision', link_name, xyz,
                                           size))


def write_extra_collision_urdf(src_path, dst_path):
    """``src_path`` に ``EXTRA_COLLISION_BOXES`` のリンクを足して ``dst_path`` に書く.

    内容が同じなら書き直さない。
    """
    extra_boxes = EXTRA_COLLISION_BOXES
    text = Path(src_path).read_text()
    for link_name, (parent_name, _, _) in extra_boxes.items():
        if re.search(r'<link name="{}"'.format(re.escape(link_name)), text):
            raise ValueError('{} には既にリンク {!r} があります。'.format(
                src_path, link_name))
        if not re.search(r'<link name="{}"'.format(re.escape(parent_name)),
                         text):
            raise ValueError('{} に親リンク {!r} が見つかりません。'.format(
                src_path, parent_name))
    extra = ''.join(
        _format_extra_link(link_name, parent_name, xyz, size)
        for link_name, (parent_name, xyz, size) in extra_boxes.items())
    end = text.rindex('</robot>')
    new_text = text[:end] + extra + text[end:]
    dst_path = Path(dst_path)
    if dst_path.exists() and dst_path.read_text() == new_text:
        return
    dst_path.write_text(new_text)
    print('[extra] {} に干渉ジオメトリのリンクを足しました: {}'.format(
        dst_path, ', '.join(extra_boxes)))


def build_collision_model_urdf(urdf_path, primitive_type=None, force=False):
    """URDF のメッシュをプリミティブに変換し、追加リンク込みの URDF パスを返す.

    ``primitive_type`` ('box'/'cylinder'/'sphere') で形状を強制できる
    (None なら自動選択)。変換結果はキャッシュし、``force`` で作り直す。
    """
    urdf_path = Path(urdf_path)
    primitives_path = urdf_path.parent / f"{urdf_path.stem}_primitives.urdf"
    output_path = urdf_path.parent / f"{urdf_path.stem}_primitives_extra.urdf"

    if primitives_path.exists() and not force:
        print(f"[skip] 既存の干渉モデルURDFを再利用します: {primitives_path}")
    else:
        print(f"[convert] {urdf_path} -> {primitives_path}")
        modified = convert_meshes_to_primitives(
            str(urdf_path),
            str(primitives_path),
            convert_visual=True,
            convert_collision=True,
            primitive_type=primitive_type,
        )
        print(f"[convert] {modified} 個のジオメトリをプリミティブに変換しました")
    write_extra_collision_urdf(primitives_path, output_path)
    return output_path
