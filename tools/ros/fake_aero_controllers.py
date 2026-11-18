#!/usr/bin/env python3
"""rosbag 再生で実機ノードの実行の流れ (台車・腕のゴールの送り方、差し替え) を
確かめるための、偽の Aero のコントローラ。

- 腕・首・腰・脚の follow_joint_trajectory (skrobot の AeroROSRobotInterface と
  同じ 5 系統) と、台車の /base_controller/follow_joint_trajectory
  (base_link_x/y/pan、odom の系の絶対位置) のアクションサーバ。
- ゴールの差し替えは joint_trajectory_controller 0.22 と同じ扱い: 新しいゴールの
  開始時刻 (header.stamp、0 なら受信時刻) までは古い軌道を続け、その時刻の古い
  軌道の位置・速度から新しい点へ 3 次エルミートでつなぐ。開始時刻が過去なら
  受信時刻からつなぐ (点の時刻は stamp からのまま)。それより前の点は捨てる。
  速度の無い点は直線でつなぐ。古いゴールは PREEMPTED。
- 指令どおりに動いたとして /joint_states (100Hz)、各コントローラの state、
  /odom を出す (実機の /odom も指令の積分)。TF は出さない (bag の /tf を使う)。
- bag の再生が終わると /clock が止まって実行が進まなくなるので、パラメータ
  ``~continue_clock`` が true になってからは、/clock が 0.5 秒途切れたら壁時計で
  続きを出す。
- 終了時に ``--log`` へ、時刻ごとの関節角・台車の位置 (npz) と受けたゴール
  (同名の .jsonl) を書く。

使い方 (専用の roscore・use_sim_time で、ノードより先に起動する):
    python3 tools/ros/fake_aero_controllers.py --initial-from-bag X.bag \
        --log out/fake_trace.npz
"""
import argparse
import json
import math
import threading
import time

import actionlib
import numpy as np
import rospy
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryResult
from control_msgs.msg import JointTrajectoryControllerState
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

ARM_CONTROLLERS = {
    'larm_controller': ['l_shoulder_p_joint', 'l_shoulder_r_joint',
                        'l_shoulder_y_joint', 'l_elbow_joint',
                        'l_wrist_y_joint', 'l_wrist_p_joint',
                        'l_wrist_r_joint', 'l_hand_y_joint'],
    'rarm_controller': ['r_shoulder_p_joint', 'r_shoulder_r_joint',
                        'r_shoulder_y_joint', 'r_elbow_joint',
                        'r_wrist_y_joint', 'r_wrist_p_joint',
                        'r_wrist_r_joint', 'r_hand_y_joint'],
    'head_controller': ['neck_y_joint', 'neck_p_joint', 'neck_r_joint'],
    'waist_controller': ['waist_y_joint', 'waist_p_joint', 'waist_r_joint'],
    'lifter_controller': ['knee_joint', 'ankle_joint'],
}
BASE_CONTROLLER = 'base_controller'
BASE_JOINTS = ['base_link_x', 'base_link_y', 'base_link_pan']
PUBLISH_PERIOD = 0.01  # [s] 壁時計
CLOCK_GAP = 0.5  # [s] これだけ /clock が途切れたら続きを出す


class Track(object):
    """節点 (絶対時刻, 位置, 速度 or None) の列を 3 次エルミートでつないだ軌道。"""

    def __init__(self, position):
        self.knots = [(-1e18, np.asarray(position, dtype=np.float64),
                       np.zeros(len(position)))]

    def state(self, t):
        knots = self.knots
        if t <= knots[0][0]:
            return knots[0][1].copy(), np.zeros_like(knots[0][1])
        for (t0, p0, v0), (t1, p1, v1) in zip(knots[:-1], knots[1:]):
            if t <= t1:
                h = t1 - t0
                if h <= 1e-9:
                    return p1.copy(), np.zeros_like(p1)
                if v0 is None or v1 is None:
                    s = (t - t0) / h
                    return p0 + s * (p1 - p0), (p1 - p0) / h
                s = (t - t0) / h
                h00 = 2 * s ** 3 - 3 * s ** 2 + 1
                h10 = s ** 3 - 2 * s ** 2 + s
                h01 = -2 * s ** 3 + 3 * s ** 2
                h11 = s ** 3 - s ** 2
                pos = h00 * p0 + h10 * h * v0 + h01 * p1 + h11 * h * v1
                d00 = (6 * s ** 2 - 6 * s) / h
                d10 = 3 * s ** 2 - 4 * s + 1
                d01 = (-6 * s ** 2 + 6 * s) / h
                d11 = 3 * s ** 2 - 2 * s
                vel = d00 * p0 + d10 * v0 + d01 * p1 + d11 * v1
                return pos, vel
        last = knots[-1]
        return last[1].copy(), np.zeros_like(last[1])

    def end_time(self):
        return self.knots[-1][0]

    def replace(self, t_start, points):
        """``t_start`` からの点列 [(絶対時刻, 位置, 速度 or None)] に差し替える。"""
        p_s, v_s = self.state(t_start)
        kept = [k for k in self.knots if k[0] < t_start]
        if not kept:
            kept = [(t_start - 1e-3, p_s, v_s)]
        new = [(t, p, v) for t, p, v in points if t > t_start + 1e-6]
        velocity_given = all(v is not None for _, _, v in new)
        self.knots = kept + [(t_start, p_s, v_s if velocity_given else None)] \
            + new
        if not new:
            # 点が全部過去: その場で止める (JTC は拒否するが、止まるのは同じ)。
            self.knots = kept + [(t_start, p_s, np.zeros_like(p_s))]

    def hold(self, t):
        p, _ = self.state(t)
        self.knots = [(t, p, np.zeros_like(p))]


class Controller(object):
    def __init__(self, fake, name, joint_names, initial):
        self.fake = fake
        self.name = name
        self.joint_names = list(joint_names)
        self.track = Track(initial)
        self.active = None
        self.server = actionlib.ActionServer(
            '/{}/follow_joint_trajectory'.format(name),
            FollowJointTrajectoryAction, self.on_goal, self.on_cancel,
            auto_start=False)
        self.state_pub = None
        if name != BASE_CONTROLLER:
            self.state_pub = rospy.Publisher(
                '/{}/state'.format(name), JointTrajectoryControllerState,
                queue_size=1)

    def start(self):
        self.server.start()

    def on_goal(self, handle):
        goal = handle.get_goal()
        traj = goal.trajectory
        now = rospy.Time.now().to_sec()
        stamp = traj.header.stamp.to_sec()
        # 点の時刻は stamp から。つなぐのは stamp と受信時刻の遅い方 (JTC 0.22
        # も開始時刻が過去なら今の状態からつなぐ)。
        t_ref = stamp if stamp > 0 else now
        t_start = max(t_ref, now)
        index = {n: i for i, n in enumerate(traj.joint_names)}
        if not set(traj.joint_names) <= set(self.joint_names):
            handle.set_rejected(text='unknown joints {}'.format(
                sorted(set(traj.joint_names) - set(self.joint_names))))
            return
        with self.fake.lock:
            current, _ = self.track.state(max(now, t_start))
            points = []
            for point in traj.points:
                pos = current.copy()
                vel = None if not point.velocities else np.zeros_like(current)
                for k, name in enumerate(self.joint_names):
                    if name in index:
                        pos[k] = point.positions[index[name]]
                        if vel is not None:
                            vel[k] = point.velocities[index[name]]
                if self.name == BASE_CONTROLLER:
                    pos[2] = _unwrap_near(pos[2], current[2]) \
                        if not points else _unwrap_near(pos[2], points[-1][1][2])
                points.append((t_ref + point.time_from_start.to_sec(),
                               pos, vel))
            self.track.replace(t_start, points)
            if self.active is not None:
                self.active.set_canceled(text='replaced by a new goal')
            handle.set_accepted()
            self.active = handle
            self.end_time = self.track.end_time()
        self.fake.log_goal(dict(
            controller=self.name, recv=now, stamp=stamp, start=t_start,
            joint_names=list(traj.joint_names),
            times=[p.time_from_start.to_sec() for p in traj.points],
            positions=[list(p.positions) for p in traj.points],
            velocities=[list(p.velocities) for p in traj.points]))

    def on_cancel(self, handle):
        with self.fake.lock:
            if self.active is not None and \
                    self.active.get_goal_id().id == handle.get_goal_id().id:
                self.track.hold(rospy.Time.now().to_sec())
                self.active = None
            handle.set_canceled()
        self.fake.log_goal(dict(controller=self.name, cancel=True,
                                recv=rospy.Time.now().to_sec()))

    def update(self, now):
        """``now`` の位置・速度。ゴールの終わりを過ぎていれば成功にする。"""
        pos, vel = self.track.state(now)
        if self.active is not None and now >= self.end_time:
            self.active.set_succeeded(FollowJointTrajectoryResult())
            self.active = None
        return pos, vel


def _unwrap_near(angle, reference):
    return reference + math.atan2(math.sin(angle - reference),
                                  math.cos(angle - reference))


class FakeAero(object):
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.goals = []
        self.trace_t, self.trace_q, self.trace_base = [], [], []
        initial, self.extra = self._initial_joint_states(args.initial_from_bag)
        self.controllers = []
        for name, joints in ARM_CONTROLLERS.items():
            self.controllers.append(Controller(
                self, name, joints, [initial.get(j, 0.0) for j in joints]))
        self.base = Controller(self, BASE_CONTROLLER, BASE_JOINTS,
                               [0.0, 0.0, 0.0])
        self.controllers.append(self.base)
        self.joint_pub = rospy.Publisher('/joint_states', JointState,
                                         queue_size=10)
        self.odom_pub = rospy.Publisher('/odom', Odometry, queue_size=10)
        self.clock_pub = rospy.Publisher('/clock', Clock, queue_size=10)
        self.last_clock = None  # (sim 秒, 受けた壁時計)
        self.own_clock = False
        rospy.Subscriber('/clock', Clock, self._on_clock, queue_size=10)
        for controller in self.controllers:
            controller.start()

    @staticmethod
    def _initial_joint_states(bag_path):
        """bag の最初の /joint_states (無ければ空)。コントローラ外の関節は
        その値のまま出し続ける。"""
        if not bag_path:
            return {}, {}
        import rosbag
        with rosbag.Bag(bag_path) as bag:
            for _, msg, _ in bag.read_messages(topics=['/joint_states']):
                states = dict(zip(msg.name, msg.position))
                ours = {j for joints in ARM_CONTROLLERS.values()
                        for j in joints}
                extra = {k: v for k, v in states.items() if k not in ours}
                return states, extra
        return {}, {}

    def _on_clock(self, msg):
        if self.own_clock:
            return
        self.last_clock = (msg.clock.to_sec(), time.time())

    def log_goal(self, record):
        with self.lock:
            self.goals.append(record)

    def _clock_tick(self):
        """bag が終わった後、/clock を壁時計で進める。"""
        if self.own_clock:
            sim0, wall0 = self.last_clock
            t = sim0 + (time.time() - wall0)
            self.clock_pub.publish(Clock(clock=rospy.Time.from_sec(t)))
            return
        if self.last_clock is None or not rospy.get_param(
                '~continue_clock', False):
            return
        if time.time() - self.last_clock[1] > CLOCK_GAP:
            sim0, wall0 = self.last_clock
            self.last_clock = (sim0 + (time.time() - wall0), time.time())
            self.own_clock = True
            rospy.loginfo('[fake] bag の /clock が止まったので続きを出します')

    def spin(self):
        last_now = None
        last_odom = 0.0
        while not rospy.is_shutdown():
            time.sleep(PUBLISH_PERIOD)
            self._clock_tick()
            now_msg = rospy.Time.now()
            now = now_msg.to_sec()
            if now <= 0:
                continue
            with self.lock:
                if last_now is not None and now < last_now - 1.0:
                    # bag の頭出し (時刻が戻った): その場で止める。
                    for controller in self.controllers:
                        p, _ = controller.track.state(last_now)
                        controller.track = Track(p)
                        if controller.active is not None:
                            controller.active.set_aborted(
                                text='clock moved backwards')
                            controller.active = None
                last_now = now
                names, positions, velocities = [], [], []
                states = {}
                for controller in self.controllers:
                    pos, vel = controller.update(now)
                    states[controller.name] = (pos, vel)
                    if controller is self.base:
                        continue
                    names += controller.joint_names
                    positions += list(pos)
                    velocities += list(vel)
            msg = JointState()
            msg.header.stamp = now_msg
            msg.name = names + list(self.extra)
            msg.position = positions + list(self.extra.values())
            msg.velocity = velocities + [0.0] * len(self.extra)
            self.joint_pub.publish(msg)
            for controller in self.controllers:
                if controller.state_pub is None:
                    continue
                pos, vel = states[controller.name]
                state = JointTrajectoryControllerState()
                state.header.stamp = now_msg
                state.joint_names = controller.joint_names
                state.desired = JointTrajectoryPoint(
                    positions=list(pos), velocities=list(vel))
                state.actual = state.desired
                state.error = JointTrajectoryPoint(
                    positions=[0.0] * len(pos), velocities=[0.0] * len(pos))
                controller.state_pub.publish(state)
            base_pos, base_vel = states[BASE_CONTROLLER]
            if now - last_odom >= 0.02 or now < last_odom:
                last_odom = now
                self.odom_pub.publish(_odom_msg(now_msg, base_pos, base_vel))
            self.trace_t.append(now)
            self.trace_q.append(positions)
            self.trace_base.append(list(base_pos))
        self.save(names)

    def save(self, names):
        if not self.args.log:
            return
        np.savez(self.args.log, t=np.asarray(self.trace_t),
                 q=np.asarray(self.trace_q), base=np.asarray(self.trace_base),
                 names=np.asarray(names))
        with open(self.args.log.rsplit('.', 1)[0] + '.jsonl', 'w') as f:
            for record in self.goals:
                f.write(json.dumps(record) + '\n')
        print('[fake] {} に {} 点・ゴール {} 個を書きました'.format(
            self.args.log, len(self.trace_t), len(self.goals)))


def _odom_msg(stamp, pos, vel):
    msg = Odometry()
    msg.header.stamp = stamp
    msg.header.frame_id = 'odom'
    msg.child_frame_id = 'base_link'
    msg.pose.pose.position.x = float(pos[0])
    msg.pose.pose.position.y = float(pos[1])
    msg.pose.pose.orientation.z = math.sin(pos[2] / 2.0)
    msg.pose.pose.orientation.w = math.cos(pos[2] / 2.0)
    c, s = math.cos(pos[2]), math.sin(pos[2])
    msg.twist.twist.linear.x = float(c * vel[0] + s * vel[1])
    msg.twist.twist.linear.y = float(-s * vel[0] + c * vel[1])
    msg.twist.twist.angular.z = float(vel[2])
    return msg


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--initial-from-bag', default=None,
                        help='最初の /joint_states を初期姿勢にする bag。')
    parser.add_argument('--log', default=None,
                        help='終了時に軌跡を書く npz (ゴールは同名の .jsonl)。')
    args = parser.parse_args(rospy.myargv()[1:])
    rospy.init_node('fake_aero_controllers', disable_signals=False)
    FakeAero(args).spin()


if __name__ == '__main__':
    main()
