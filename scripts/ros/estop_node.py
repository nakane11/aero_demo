#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""AtomS3 のボタンを UDP で受け、全コントローラの follow_joint_trajectory と
move_base のゴールを cancel する非常停止ノード。

プロトコル: ``--listen-port`` へ改行区切りの ``STOP`` / ``RESUME``。
停止中は ``/estop`` (Bool, latch) が True。cancel だけでは直後に送られる
次のゴールを防げないので、デモ側は True の間は新しい動作を送らないこと。
UDP は到達保証が無いため、送信側は STOP を複数回送り、こちらも cancel を
複数回 publish する。

Usage
-----
    rosrun aero_demo estop_node.py --listen-port 5555
"""

import argparse
import socket
import threading
import time

import rospy
from actionlib_msgs.msg import GoalID
from std_msgs.msg import Bool

# cancel の取りこぼし対策で STOP ごとに複数回 publish する。
CANCEL_REPEAT_COUNT = 3
CANCEL_REPEAT_INTERVAL = 0.05  # [sec]

# rospy.is_shutdown() を確認するための受信タイムアウト。
SOCKET_POLL_TIMEOUT = 0.5  # [sec]

DEFAULT_CONTROLLERS = [
    'base_controller',
    'rarm_controller',
    'larm_controller',
    'rhand_controller',
    'lhand_controller',
    'head_controller',
    'waist_controller',
    'lifter_controller',
]


class EstopNode(object):

    def __init__(self, args):
        self.args = args

        self._cancel_pubs = {
            name: rospy.Publisher(
                '/{}/follow_joint_trajectory/cancel'.format(name),
                GoalID, queue_size=1)
            for name in args.controllers
        }
        self._move_base_cancel_pub = rospy.Publisher(
            '/move_base/cancel', GoalID, queue_size=1)
        self._estop_pub = rospy.Publisher(
            'estop', Bool, queue_size=1, latch=True)

        self._lock = threading.Lock()
        self._stopped = False

        # latch の初期値を False にしておく。
        self._estop_pub.publish(Bool(data=False))

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.bind((args.listen_host, args.listen_port))
        self._sock.settimeout(SOCKET_POLL_TIMEOUT)

    # ------------------------------------------------------------------
    # STOP/RESUME
    # ------------------------------------------------------------------
    def trigger_stop(self):
        with self._lock:
            already_stopped = self._stopped
            self._stopped = True
        if not already_stopped:
            rospy.logwarn('[estop_node] STOP を受信、全ゴールを cancel します')
        self._estop_pub.publish(Bool(data=True))
        # 受信ループを塞がないよう別スレッドで publish する。
        threading.Thread(target=self._publish_cancels, daemon=True).start()

    def _publish_cancels(self):
        empty_goal_id = GoalID()  # 空の GoalID = 全ゴールを cancel
        for _ in range(CANCEL_REPEAT_COUNT):
            for pub in self._cancel_pubs.values():
                pub.publish(empty_goal_id)
            self._move_base_cancel_pub.publish(empty_goal_id)
            time.sleep(CANCEL_REPEAT_INTERVAL)

    def trigger_resume(self):
        with self._lock:
            was_stopped = self._stopped
            self._stopped = False
        if was_stopped:
            rospy.logwarn('[estop_node] RESUME を受信、停止を解除します')
        self._estop_pub.publish(Bool(data=False))

    @property
    def stopped(self):
        with self._lock:
            return self._stopped

    # ------------------------------------------------------------------
    # UDP受信
    # ------------------------------------------------------------------
    def _handle_line(self, line, addr):
        line = line.strip()
        if not line:
            return
        # 同じコマンドが連続で届くので debug レベルで記録する。
        rospy.logdebug('[estop_node] %s から受信: %r', addr, line)
        if line == 'STOP':
            self.trigger_stop()
        elif line == 'RESUME':
            self.trigger_resume()
        else:
            rospy.logwarn(
                '[estop_node] %s からの未知の入力を無視: %r', addr, line)

    def spin(self):
        rospy.loginfo(
            '[estop_node] UDP %s:%d で待受を開始します。対象コント'
            'ローラ: %s', self.args.listen_host, self.args.listen_port,
            self.args.controllers)

        while not rospy.is_shutdown():
            try:
                data, addr = self._sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError as e:
                rospy.logerr('[estop_node] UDP受信エラー: %s', e)
                time.sleep(1.0)
                continue
            for raw_line in data.split(b'\n'):
                if raw_line:
                    self._handle_line(
                        raw_line.decode('ascii', errors='replace'), addr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen-host', type=str, default='0.0.0.0')
    parser.add_argument('--listen-port', type=int, default=5555)
    parser.add_argument(
        '--controllers', type=str, nargs='+',
        default=DEFAULT_CONTROLLERS,
        help='cancel を送る対象コントローラの名前空間一覧。')
    args, _ = parser.parse_known_args(rospy.myargv()[1:])

    rospy.init_node('estop_node')
    node = EstopNode(args)
    node.spin()


if __name__ == '__main__':
    main()
