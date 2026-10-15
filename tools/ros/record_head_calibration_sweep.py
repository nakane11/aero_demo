#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""実機の首を格子状に動かして止め、calibrate_head_camera.py 用の bag を録る。

腕・腰・脚には指令を送らない (今の姿勢のまま)。首は head_controller に
neck_y/neck_p だけを送り (neck_r は 0)、角度は ``--max-yaw`` と
``--pitch-min``/``--pitch-max`` の範囲に、速さは ``--speed`` [deg/s] 以下に
抑える。各姿勢で ``--dwell`` 秒止まる。終わったら (途中で止めても) 元の首の
角度に戻す。台車が動いていたら始めない。

``--base-yaws`` を指定すると、台車をその場で開始時の向きから各角度 [deg]
(``--max-base-yaw`` 以内) へ回してから、それぞれで首を動かす。台車は
/teleop/cmd_vel に ``--base-speed`` [deg/s] 以下の回転だけを送り、odom の向きで
止める。台車中心から ``--obstacle-min``〜``--obstacle-radius`` [m] に /scan の点が
あれば回さない (回転中も見て止める)。終わったら台車も元の向きに戻す。

録るトピックは record_camera_pipeline.launch からカラー画像を除いたもの
(``--with-color`` で含める)。decompress.launch を先に起動しておくこと。

Usage
-----
    rossetmaster 192.168.96.215; rossetip
    python3 tools/ros/record_head_calibration_sweep.py \\
        --output data/bags/head_calib.bag
    python3 tools/ros/calibrate_head_camera.py --bag data/bags/head_calib.bag
"""

import argparse
import math
import os
import signal
import subprocess
import sys
import time

import actionlib
import numpy as np
import rospy
import tf2_ros
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from sensor_msgs.msg import LaserScan
from trajectory_msgs.msg import JointTrajectoryPoint

HEAD_JOINTS = ('neck_y_joint', 'neck_p_joint', 'neck_r_joint')
# solve_palm_ik.restrict_neck_range と同じ実機の首ピッチの上限 (差動の頭打ち対策)
NECK_P_HARD_MAX_DEG = 44.0
RECORD_TOPICS = ('/camera/depth/image_raw/decompressed',
                 '/camera/color/camera_info', '/tf', '/tf_static',
                 '/scan', '/joint_states', '/odom')


class HeadSweeper(object):

    def __init__(self, args):
        self.args = args
        self.joint_state = None
        self.odom = None
        self.scan = None
        rospy.Subscriber('/joint_states', JointState, self._js_cb,
                         queue_size=1)
        rospy.Subscriber('/odom', Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self._scan_cb, queue_size=1)
        self.cmd_pub = rospy.Publisher(args.cmd_vel_topic, Twist,
                                       queue_size=1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.laser_to_base = None
        self.client = actionlib.SimpleActionClient(
            '/head_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction)
        if not self.client.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError('head_controller の action に接続できません。')
        deadline = time.time() + 10.0
        while (self.joint_state is None or self.odom is None) \
                and time.time() < deadline:
            time.sleep(0.05)
        if self.joint_state is None or self.odom is None:
            raise RuntimeError('/joint_states か /odom を受信できません。')

    def _js_cb(self, msg):
        if all(n in msg.name for n in HEAD_JOINTS):
            self.joint_state = msg

    def _odom_cb(self, msg):
        self.odom = msg

    def _scan_cb(self, msg):
        self.scan = msg

    def base_yaw(self):
        q = self.odom.pose.pose.orientation
        return math.atan2(2 * (q.w * q.z + q.x * q.y),
                          1 - 2 * (q.y * q.y + q.z * q.z))

    def nearest_obstacle(self):
        """台車中心から ``--obstacle-min`` より外で最も近い /scan の点の距離 [m]。"""
        msg = self.scan
        if msg is None or time.time() - msg.header.stamp.to_sec() > 1.0 + 0.5:
            # 古い scan では安全を判断できない (時計ずれ 0.5 秒まで許容)
            raise RuntimeError('/scan が新しくないため台車を回せません。')
        if self.laser_to_base is None:
            tr = self.tf_buffer.lookup_transform(
                'base_link', msg.header.frame_id, rospy.Time(0),
                rospy.Duration(3.0)).transform
            q = tr.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                             1 - 2 * (q.y * q.y + q.z * q.z))
            self.laser_to_base = (tr.translation.x, tr.translation.y, yaw)
        lx, ly, lyaw = self.laser_to_base
        r = np.asarray(msg.ranges, dtype=np.float64)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment + lyaw
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        x = lx + r[ok] * np.cos(ang[ok])
        y = ly + r[ok] * np.sin(ang[ok])
        d = np.hypot(x, y)
        d = d[d > self.args.obstacle_min]
        return float(d.min()) if len(d) else float('inf')

    def check_clear(self):
        d = self.nearest_obstacle()
        if d < self.args.obstacle_radius:
            raise RuntimeError(
                '台車中心から {:.2f} m に障害物があるため台車を回しません。'
                .format(d))
        return d

    def stop_base(self):
        for _ in range(10):
            self.cmd_pub.publish(Twist())
            time.sleep(0.05)

    def rotate_base_to(self, target_yaw):
        """台車をその場で odom の向き ``target_yaw`` [rad] へ回して止める。"""
        a = self.args
        max_w = math.radians(a.base_speed)
        accel = math.radians(a.base_accel)
        rate = 20.0
        w = 0.0
        deadline = time.time() + 60.0
        try:
            while True:
                err = math.atan2(math.sin(target_yaw - self.base_yaw()),
                                 math.cos(target_yaw - self.base_yaw()))
                if abs(err) < math.radians(0.5):
                    break
                if time.time() > deadline:
                    raise RuntimeError('台車の回転が時間内に終わりませんでした。')
                self.check_clear()
                # 残り角で止まれる速さ (台形) と上限・加速度で決める
                w_stop = math.sqrt(2 * accel * abs(err))
                w_target = math.copysign(
                    max(min(max_w, w_stop), math.radians(2.0)), err)
                dw = max(-accel / rate, min(accel / rate, w_target - w))
                w += dw
                cmd = Twist()
                cmd.angular.z = w
                self.cmd_pub.publish(cmd)
                time.sleep(1.0 / rate)
        finally:
            self.stop_base()
        time.sleep(1.0)  # 止まるのを待つ

    def head_angles(self):
        msg = self.joint_state
        return [msg.position[msg.name.index(n)] for n in HEAD_JOINTS]

    def base_moving(self):
        t = self.odom.twist.twist
        return (math.hypot(t.linear.x, t.linear.y) > 1e-3
                or abs(t.angular.z) > 1e-3)

    def clamp(self, yaw_deg, pitch_deg):
        a = self.args
        yaw = max(-a.max_yaw, min(a.max_yaw, yaw_deg))
        pitch = max(a.pitch_min, min(min(a.pitch_max, NECK_P_HARD_MAX_DEG),
                                     pitch_deg))
        return math.radians(yaw), math.radians(pitch)

    def move_to(self, yaw, pitch, roll=0.0):
        """首を (yaw, pitch, roll) [rad] へ ``--speed`` 以下で動かし、終わるまで待つ。"""
        cur = self.head_angles()
        target = [yaw, pitch, roll]
        delta = max(abs(t - c) for t, c in zip(target, cur))
        duration = max(self.args.min_move_time,
                       math.degrees(delta) / self.args.speed)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = list(HEAD_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = target
        point.velocities = [0.0, 0.0, 0.0]
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points = [point]
        goal.trajectory.header.stamp = rospy.Time(0)
        self.client.send_goal(goal)
        if not self.client.wait_for_result(rospy.Duration(duration + 5.0)):
            self.client.cancel_goal()
            raise RuntimeError('首の動作が時間内に終わりませんでした。')
        state = self.client.get_state()
        if state != GoalStatus.SUCCEEDED:
            raise RuntimeError(
                '首のゴールが {} で終わりました (サーボオフ・非常停止の可能性)。'
                .format(GoalStatus.to_string(state)))
        # 指令だけ成功して実際は動いていない (/joint_states が止まっている等) を検出する。
        time.sleep(0.3)
        reached = self.head_angles()
        err = max(abs(r - t) for r, t in zip(reached, target))
        if math.degrees(err) > self.args.max_reach_error:
            raise RuntimeError(
                '首が目標に届いていません (ずれ {:.1f} deg)。'.format(
                    math.degrees(err)))
        return duration


def grid(args):
    """蛇行順の (yaw, pitch) [deg] の列。"""
    yaws = [float(v) for v in args.yaws.split(',')]
    pitches = [float(v) for v in args.pitches.split(',')]
    poses = []
    for i, pitch in enumerate(pitches):
        row = yaws if i % 2 == 0 else list(reversed(yaws))
        poses += [(yaw, pitch) for yaw in row]
    return poses


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True, help='保存する bag のパス。')
    parser.add_argument('--yaws', default='-60,-40,-20,0,20,40,60',
                        help='首ヨー [deg] (カンマ区切り)。')
    parser.add_argument('--pitches', default='15,20,25,30',
                        help='首ピッチ [deg] (カンマ区切り、+ が下向き)。')
    parser.add_argument('--max-yaw', type=float, default=60.0)
    parser.add_argument('--base-yaws', default='0',
                        help='台車の向き [deg] (開始時からの相対、カンマ区切り)。')
    parser.add_argument('--max-base-yaw', type=float, default=40.0)
    parser.add_argument('--base-speed', type=float, default=15.0,
                        help='台車の回転の速さの上限 [deg/s]。')
    parser.add_argument('--base-accel', type=float, default=20.0,
                        help='台車の回転の加速度の上限 [deg/s^2]。')
    parser.add_argument('--cmd-vel-topic', default='/teleop/cmd_vel')
    parser.add_argument('--obstacle-min', type=float, default=0.40,
                        help='これ [m] より近い /scan の点は台車自身とみなす。')
    parser.add_argument('--obstacle-radius', type=float, default=0.70,
                        help='台車中心からこれ [m] 以内に物があれば台車を回さない。')
    parser.add_argument('--pitch-min', type=float, default=-10.0)
    parser.add_argument('--pitch-max', type=float, default=40.0)
    parser.add_argument('--speed', type=float, default=20.0,
                        help='首の速さの上限 [deg/s]。')
    parser.add_argument('--min-move-time', type=float, default=1.0)
    parser.add_argument('--max-reach-error', type=float, default=2.0,
                        help='動いた後の首の角度と目標の差がこれ [deg] を超えたら中止。')
    parser.add_argument('--dwell', type=float, default=2.0,
                        help='各姿勢で止まる時間 [s]。')
    parser.add_argument('--with-color', action='store_true',
                        help='カラー画像も録る (較正には不要、容量が約 2 倍)。')
    parser.add_argument('--dry-run', action='store_true',
                        help='動かさず、姿勢の列と所要時間の見積もりだけ表示する。')
    args = parser.parse_args()

    rospy.init_node('record_head_calibration_sweep', anonymous=True,
                    disable_signals=True)
    sweeper = HeadSweeper(args)
    start = sweeper.head_angles()
    start_base_yaw = sweeper.base_yaw()
    poses = [sweeper.clamp(y, p) for y, p in grid(args)]
    base_yaws = [max(-args.max_base_yaw, min(args.max_base_yaw, float(v)))
                 for v in args.base_yaws.split(',')]
    print('[sweep] 今の首: yaw {:+.1f} pitch {:+.1f} roll {:+.1f} deg'.format(
        *[math.degrees(v) for v in start]))
    print('[sweep] 首 {} 姿勢: {}'.format(len(poses), ', '.join(
        '({:+.0f},{:+.0f})'.format(math.degrees(y), math.degrees(p))
        for y, p in poses)))
    print('[sweep] 台車の向き (開始時から): {} deg、最も近い障害物 {:.2f} m'
          .format(', '.join('{:+.0f}'.format(v) for v in base_yaws),
                  sweeper.nearest_obstacle()))
    if args.dry_run:
        return
    if sweeper.base_moving():
        print('[sweep] 台車が動いているため中止します。')
        return
    if any(abs(v) > 0.5 for v in base_yaws):
        sweeper.check_clear()

    out = os.path.abspath(os.path.expanduser(args.output))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rec = subprocess.Popen(
        ['rosbag', 'record', '-O', out, '--buffsize=2048',
         '__name:=record_head_calibration_sweep_bag'] + list(RECORD_TOPICS)
        + (['/camera/color/image_raw/decompressed'] if args.with_color else []),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid)
    rotated = False
    try:
        time.sleep(3.0)  # 録画の開始待ち + 今の姿勢も 1 枚分録る
        for base_deg in base_yaws:
            if abs(math.degrees(sweeper.base_yaw() - start_base_yaw)
                   - base_deg) > 1.0:
                # 台車を回す間は首を正面に戻しておく
                sweeper.move_to(0.0, start[1])
                print('[sweep] 台車を {:+.0f} deg へ回します。'.format(base_deg),
                      flush=True)
                rotated = True
                sweeper.rotate_base_to(start_base_yaw + math.radians(base_deg))
            for i, (yaw, pitch) in enumerate(poses):
                if sweeper.base_moving():
                    raise RuntimeError('台車が動いています。')
                sweeper.move_to(yaw, pitch)
                time.sleep(args.dwell)
                print('[sweep] 台車 {:+.0f} / 首 {}/{} yaw {:+.0f} pitch {:+.0f} '
                      'deg'.format(base_deg, i + 1, len(poses),
                                   math.degrees(yaw), math.degrees(pitch)),
                      flush=True)
    except RuntimeError as e:
        print('[sweep] 中止します: {}'.format(e))
    finally:
        sweeper.stop_base()
        try:
            sweeper.move_to(*start)
            print('[sweep] 首を元の角度に戻しました。')
            if rotated:
                sweeper.rotate_base_to(start_base_yaw)
                print('[sweep] 台車を元の向きに戻しました。')
        except RuntimeError as e:
            print('[sweep] 元の姿勢に戻せませんでした: {}'.format(e))
        finally:
            time.sleep(1.0)
            os.killpg(os.getpgid(rec.pid), signal.SIGINT)
            rec.wait(timeout=30)
            print('[sweep] {} に保存しました。'.format(out))


if __name__ == '__main__':
    sys.exit(main())
