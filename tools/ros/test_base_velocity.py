#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""台車 (``base_controller``) へ超低速の x 方向速度指令を一定時間送るだけの
動作確認用スクリプト。

``estop_node.py`` (STOP 送信で ``/base_controller/follow_joint_trajectory/
cancel`` を publish する非常停止ノード) の動作確認用に作成した -- この
スクリプトで台車をゆっくり前進させている最中に AtomS3 のボタン (または
``estop_node.py`` への UDP STOP パケット) で実際に台車が止まるかを確認する
想定。低速指令なので、途中で止め損なっても被害を最小限にできる。

内部的には ``skrobot`` の ``AeroROSRobotInterface.go_velocity`` を使う。
これは指定した速度 [m/s] で ``--duration`` 秒だけ動く軌道を 1 つの
``FollowJointTrajectoryAction`` ゴールとして ``base_controller`` に送る
(cmd_vel を継続 publish するのではなく、事前に軌道全体を計算して送る
点に注意 -- estop の cancel が効けば軌道の途中で止まる)。

Usage
-----
    python3 tools/ros/test_base_velocity.py
    python3 tools/ros/test_base_velocity.py --velocity 0.03 --duration 5.0
"""

import argparse
import os
import sys

import rospy

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.dirname(_THIS_DIR)
_PKG_SRC_DIR = os.path.join(_TOOLS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from skrobot.interfaces.ros import AeroROSRobotInterface  # noqa: E402

# 「超低速」の既定値 [m/s]。5 秒間で 10cm しか進まない速さ。
DEFAULT_VELOCITY = 0.02
DEFAULT_DURATION = 20.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--velocity', type=float, default=DEFAULT_VELOCITY,
        help='x軸方向の速度 [m/s] (既定 {:.3f}、正で前進)。'.format(
            DEFAULT_VELOCITY))
    parser.add_argument(
        '--duration', type=float, default=DEFAULT_DURATION,
        help='指令を送る時間 [sec] (既定 {:.1f})。'.format(DEFAULT_DURATION))
    parser.add_argument(
        '--odom-topic', type=str, default='/odom',
        help='odom トピック名 (既定 /odom、実機の aero_ros_controller に'
            '合わせた値。skrobot 既定の /base_odometry/odom ではない '
            '点に注意)。')
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    rospy.init_node('test_base_velocity')

    robot = load_aero(use_hand=False)
    print('[test_base_velocity] 実機 (AeroROSRobotInterface) に接続してい'
          'ます...')
    ri = AeroROSRobotInterface(robot, odom_topic=args.odom_topic)

    print('[test_base_velocity] x方向に {:.3f} m/s で {:.1f} 秒間、'
          '台車を前進させます (STOP で止まるか確認してください)。'.format(
              args.velocity, args.duration))
    ri.go_velocity(x=args.velocity, sec=args.duration, wait=True)
    print('[test_base_velocity] 完了しました。')


if __name__ == '__main__':
    main()
