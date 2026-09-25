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
"""
from pathlib import Path

from skrobot.urdf import convert_meshes_to_primitives


def build_collision_model_urdf(urdf_path, primitive_type=None, force=False):
    """Aero URDFのvisual/collisionメッシュをプリミティブ形状に変換する.

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
        生成されたプリミティブ近似URDFのパス。
    """
    urdf_path = Path(urdf_path)
    output_path = urdf_path.parent / f"{urdf_path.stem}_primitives.urdf"

    if output_path.exists() and not force:
        print(f"[skip] 既存の干渉モデルURDFを再利用します: {output_path}")
        return output_path

    print(f"[convert] {urdf_path} -> {output_path}")
    modified = convert_meshes_to_primitives(
        str(urdf_path),
        str(output_path),
        convert_visual=True,
        convert_collision=True,
        primitive_type=primitive_type,
    )
    print(f"[convert] {modified} 個のジオメトリをプリミティブに変換しました")
    return output_path
