#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""台車を超低速で x 方向に動かし、``estop_node.py`` で止まるかを確認するスクリプト。

``go_velocity`` は軌道全体を 1 つの FollowJointTrajectory ゴールとして送る
(cmd_vel ではない) ので、estop の cancel が効けば途中で止まる。

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

DEFAULT_VELOCITY = 0.02
DEFAULT_DURATION = 20.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--velocity', type=float, default=DEFAULT_VELOCITY,
        help='x 方向の速度 [m/s] (正で前進)。')
    parser.add_argument(
        '--duration', type=float, default=DEFAULT_DURATION,
        help='動かす時間 [s]。')
    parser.add_argument(
        '--odom-topic', type=str, default='/odom',
        help='odom トピック (skrobot 既定の /base_odometry/odom ではない)。')
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
