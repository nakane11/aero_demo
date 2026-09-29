#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""Aero の干渉 (コリジョン) モデル (box/cylinder/sphere のプリミティブ
近似 URDF) を作る処理。

scikit-robot に付属する `skr convert-urdf-to-primitives`
(``skrobot.urdf.convert_meshes_to_primitives``) を使い、Aero のメッシュ
形状をプリミティブ形状に近似変換した URDF を生成する。``solve_palm_ik.py``
(干渉回避付きバッチ IK の最適化・事後検証) と ``handshake_viewer_common.py``
(ビューアでの半透明オーバーレイ表示) の両方がこの近似形状を使うため、
本番の ``scripts/`` から import できるよう ``aero_demo`` パッケージ側に
置く (プリミティブ近似モデルをそのまま viser で見るだけの CLI ツールは
``tools/view_aero_collision_model.py`` を参照)。

自動変換だけでは実機の形状を表せない部分は、``EXTRA_COLLISION_BOXES`` の
box を持つリンクを固定関節で足して補う。scikit-robot の干渉判定 (軌道
最適化・事後検証・ビューア) は 1 リンク 1 形状が前提なので、既存のリンク
に形状を足すのではなく、別リンクにする。
"""
from pathlib import Path
import re

from skrobot.urdf import convert_meshes_to_primitives


# 干渉ジオメトリとして足すリンク。値は (親リンク名, 親リンク座標系での
# box の中心 xyz, box の size) [m]。リンク自体の座標系は親リンクと同じ
# (固定関節の origin は単位)。visual と collision の両方をこの box にする
# (ビューアの overlay は visual を表示するため、判定と表示を一致させる)。
#
# wheel_base_front_link: 台車前方の高い部分。wheel_base_link (台車) は
# base_link (地面) から z=0.075 にあり、元 URDF の box (中心 x=0.018,
# z=0.0375, size 0.72 x 0.52 x 0.149、地面から 0.038〜0.187) はそのまま
# 残す。この box はその前端 (x=0.378)・下端・左右幅をそろえ、前後を前端
# から 0.25、上端を地面から 0.32 にしたもの。
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


def write_extra_collision_urdf(src_path, dst_path, extra_boxes=None):
    """``src_path`` の URDF に ``extra_boxes`` (既定 ``EXTRA_COLLISION_
    BOXES``) のリンクを固定関節で足したものを ``dst_path`` に書く。

    内容が既存の ``dst_path`` と同じなら書き直さないので、毎回呼んでよい
    (``EXTRA_COLLISION_BOXES`` を変えれば ``force`` なしでも次回の読み込み
    から反映される)。
    """
    if extra_boxes is None:
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
    """Aero URDFのvisual/collisionメッシュをプリミティブ形状に変換する.

    変換結果 (``*_primitives.urdf``、キャッシュ) に ``EXTRA_COLLISION_
    BOXES`` のリンクを足したもの (``*_primitives_extra.urdf``) を返す。

    Parameters
    ----------
    urdf_path : str
        変換元のURDFファイルパス。
    primitive_type : str or None
        'box' / 'cylinder' / 'sphere' を指定すると全リンクをその形状に強制する。
        Noneの場合はリンクごとに最も近い形状を自動選択する。
    force : bool
        既に生成済みのURDFがあっても作り直すかどうか。

    Returns
    -------
    output_path : Path
        生成されたプリミティブ近似URDF (追加リンク込み) のパス。
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
