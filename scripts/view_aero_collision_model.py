#!/usr/bin/env python3
"""Aeroの干渉(コリジョン)モデルをviser viewerで可視化する.

scikit-robotに付属する `skr convert-urdf-to-primitives`
(skrobot.urdf.convert_meshes_to_primitives) を使い、Aeroのメッシュ形状を
box/cylinder/sphereなどのプリミティブ形状に近似変換したURDFを生成する。
生成したURDFをもう一体のロボットとして読み込み、元のAero(不透明)に重ねて
半透明で表示することで、干渉モデルの形状を確認できる。
"""
import argparse
from pathlib import Path

from skrobot.models import Aero
from skrobot.model import RobotModel
from skrobot.urdf import convert_meshes_to_primitives
from skrobot.viewers import ViserViewer


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


def main():
    parser = argparse.ArgumentParser(
        description='Aeroの干渉モデル(プリミティブ近似形状)をviserで表示する')
    parser.add_argument(
        '--no-hand', action='store_true',
        help='ハンドなしモデル(aero_nohand)を使用する')
    parser.add_argument(
        '--primitive-type', choices=['box', 'cylinder', 'sphere'],
        default=None,
        help='全リンクを指定した形状に強制変換する(未指定なら自動選択)')
    parser.add_argument(
        '--force-convert', action='store_true',
        help='干渉モデルURDFを毎回作り直す')
    parser.add_argument(
        '--alpha', type=float, default=0.35,
        help='干渉モデルの半透明度(0.0=透明, 1.0=不透明)')
    parser.add_argument(
        '--no-browser', action='store_true',
        help='ブラウザを自動で開かない')
    args = parser.parse_args()

    use_hand = not args.no_hand

    # 元のAero(見た目そのままの通常モデル)
    robot = Aero(use_hand=use_hand)
    robot.reset_pose()

    # 元のURDFファイルパスを取得し、干渉モデル(プリミティブ近似)URDFを生成
    urdf_path = robot.urdf_path
    collision_urdf_path = build_collision_model_urdf(
        urdf_path,
        primitive_type=args.primitive_type,
        force=args.force_convert,
    )

    # 干渉モデルを別ロボットとして読み込み、姿勢をAeroに同期させる
    collision_robot = RobotModel()
    collision_robot.load_urdf_file(
        str(collision_urdf_path), include_mimic_joints=False)
    collision_robot.angle_vector(robot.angle_vector())

    # 干渉モデルの各リンクを青色・半透明にする
    for link in collision_robot.link_list:
        link.set_color((80, 160, 255, 255))
        link.set_alpha(args.alpha)

    viewer = ViserViewer()
    viewer.add(robot)
    viewer.add(collision_robot)
    viewer.redraw()
    viewer.show(open_browser=not args.no_browser)

    print("Ctrl+C で終了します。")
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
