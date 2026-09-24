#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""AtomS3 (マイコン) のボタン押下をWiFi経由 (UDP) で受け取り、ロボットの
全コントローラ (base/rarm/larm/rhand/lhand/head/waist/lifter) の
``follow_joint_trajectory`` ゴールを cancel し、あわせて ``move_base`` の
現在ゴールも cancel する、独立した非常停止ノード。

AtomS3 側はこの PC の ``--listen-port`` (既定 UDP 5555) へ、改行区切りの
テキストを 1 パケットとして送るだけの単純なプロトコルを想定する。

    STOP\n      ボタン押下 (停止要求)
    RESUME\n    解除要求 (長押し等、ファーム側の実装は別途)

停止するとこのノードは ``/estop`` (std_msgs/Bool, latch) を True にして
publish し続ける。デモ側のスクリプトはこのトピックを subscribe し、True の
間は新しい動作 (angle_vector_sequence/move_trajectory 等) を発行しない
ようにすること -- cancel だけでは、cancel 直後にデモ側が次のゴールを送る
競合を防げない (詳細は本パッケージの CLAUDE.md 等の運用メモを参照)。

UDP は到達保証がないので、AtomS3 側はボタンを押している間 (または押した
直後の一定時間) STOP を連続で複数回送ることを推奨する。このノード自身も
STOP を 1 回受信するごとに cancel を複数回 publish するので、途中の
パケットが多少落ちても問題ない。

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

# cancel の取りこぼし対策として、STOP 検知のたびに同じ内容を複数回 publish
# する (actionlib の cancel トピックは QoS が保証されないベストエフォート
# の通常トピックなので、1 回だけだと購読側の接続確立タイミング次第で
# 届かないことがある)。
CANCEL_REPEAT_COUNT = 3
CANCEL_REPEAT_INTERVAL = 0.05  # [sec]

# UDP ソケットの受信待ちタイムアウト。rospy.is_shutdown() を定期的に
# チェックするために短い値にしてポーリングする。
SOCKET_POLL_TIMEOUT = 0.5  # [sec]

# follow_joint_trajectory の cancel を送る対象コントローラ名前空間
# (rostopic list で確認できたもの一式)。
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

        # 起動直後は購読側の接続が間に合っていないことがあるので、まず
        # False を publish して latch の初期値を確定させておく。
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
        # _publish_cancels は sleep を挟みながら複数回 publish するため
        # (最大 CANCEL_REPEAT_COUNT * CANCEL_REPEAT_INTERVAL 秒ブロックする)、
        # ここで同期的に呼ぶと UDP 受信ループ (シングルスレッド) がその間
        # 次のパケットを受信できなくなる。AtomS3 は 1 回の押下で同じコマンド
        # を連続送信してくる (取りこぼし対策) ため、同期呼び出しのままだと
        # 1 回の押下の処理だけで最大数百ms〜1秒近くブロックし、直後に来る
        # 反対方向のコマンド (例: STOP 連投の直後の RESUME) の処理が遅れて
        # 「ボタンを押しても反応が変わらない/遅れて切り替わる」ように見える
        # 原因になっていた。バックグラウンドスレッドにして受信ループを
        # 塞がないようにする。
        threading.Thread(target=self._publish_cancels, daemon=True).start()

    def _publish_cancels(self):
        empty_goal_id = GoalID()  # id="", stamp=0 -> 対象アクションの
                                  # 全ゴールを cancel する意味になる
        for _ in range(CANCEL_REPEAT_COUNT):
            for pub in self._cancel_pubs.values():
                pub.publish(empty_goal_id)
            self._move_base_cancel_pub.publish(empty_goal_id)
            time.sleep(CANCEL_REPEAT_INTERVAL)

    def trigger_resume(self):
        with self._lock:
            was_stopped = self._stopped
            self._stopped = False
        # trigger_stop の already_stopped と対称に、実際に状態が変わった
        # ときだけ警告ログを出す。AtomS3 は 1 回の押下で RESUME を連続
        # 送信してくるため、ここにガードが無いと重複パケットのたびに
        # 警告ログが連続で出て「チャタリングしている」ように見えていた
        # (実際には /estop の値自体は毎回 False で一貫しており、振動は
        # していなかった)。
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
        # 診断用に受信した生コマンドをログに残すが、AtomS3 は 1 回の押下で
        # 同じコマンドを連続送信してくる (取りこぼし対策) ため、既定の
        # loginfo のままだと 1 回の押下で毎回 5 行ずつ表示されてしまう。
        # 通常はここは無表示にし、状態が実際に変わったときの WARN
        # (trigger_stop/trigger_resume 側) だけを見せる。必要なときは
        # `rosrun aero_demo estop_node.py _log_level:=debug` 等で
        # DEBUG ログを有効にすれば、全パケットの受信タイミングを確認できる。
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
