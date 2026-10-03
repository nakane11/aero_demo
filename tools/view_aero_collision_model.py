#!/usr/bin/env python3
"""Aero の干渉モデル (``build_collision_model_urdf`` のプリミティブ近似) を
元の Aero に半透明で重ねて viser で表示する (デバッグ用).
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo.aero_urdf_setup import load_aero  # noqa: E402  (パス追加後に import)
from aero_demo.collision_model import build_collision_model_urdf  # noqa: E402
from skrobot.model import RobotModel  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description='Aeroの干渉モデル(プリミティブ近似形状)をviserで表示する')
    parser.add_argument(
        '--no-hand', action='store_true',
        help='ハンドなしモデル(aero_nohand)を使用する')
    parser.add_argument(
        '--alpha', type=float, default=0.35,
        help='干渉モデルの半透明度(0.0=透明, 1.0=不透明)')
    parser.add_argument(
        '--no-browser', action='store_true',
        help='ブラウザを自動で開かない')
    args = parser.parse_args()

    use_hand = not args.no_hand

    robot = load_aero(use_hand=use_hand)
    robot.reset_pose()

    urdf_path = robot.urdf_path
    collision_urdf_path = build_collision_model_urdf(urdf_path)

    # 干渉モデルを別ロボットとして読み込み、姿勢を同期させる
    collision_robot = RobotModel()
    collision_robot.load_urdf_file(
        str(collision_urdf_path), include_mimic_joints=False)
    collision_robot.angle_vector(robot.angle_vector())

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
