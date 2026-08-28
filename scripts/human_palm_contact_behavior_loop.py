#!/usr/bin/env python3
"""Repeat ``human_palm_contact_behavior.py`` against a new fake person forever.

The single-shot node reaches for one person and then just sits in ``DONE``.
This wraps it (by subclassing, not copy-pasting the state machine) so that
once a contact sequence finishes, it throws away the estimator, samples a
brand new person (new body, new standing position and, if ``~present_hand``
is set, a new random arm/wrist angle -- see
``fake_people_pose_estimator_ros.py``'s ``~present_*_deg_range`` params) and
starts over from ``WAITING``. Only makes sense with ``~source:=fake`` --
running it against ``~source:=real`` would just keep re-attempting on
whichever person the camera currently sees.

Every round also prints the offered arm's shoulder/elbow/wrist angles
(``presented_arm_angles``, the "human joint angles" that characterize how
the hand was held out) alongside whether the approach/press actually
reached, so a run can be used to explore which offered poses the whole-body
IK can and cannot contact. Always logged via ``rospy.loginfo``; also shown
as a text panel in the ``viser`` viewer (``~viewer:=viser`` -- ``trimesh``
has no way to draw text, see ``PalmPlaneScene.show_info_text``).

Parameters
----------
All of ``human_palm_contact_behavior.py``'s parameters, plus

``~round_pause`` (float, default 2.0)   何もせず待つ時間 [s]。次の周回の
                                         推定を始める前に置く間。結果パネル
                                         を読む時間と、直前の押し込みの絵が
                                         一瞬で次の人に上書きされないための
                                         もの。
"""

import os
import sys
import threading

import rospy

from skrobot.coordinates import Coordinates

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from human_palm_contact_behavior import (HumanPalmContactBehavior,
                                         create_pose_source)


class HumanPalmContactLoopBehavior(HumanPalmContactBehavior):
    """``DONE`` に着いたら次の人を生成してやり直す版."""

    def __init__(self):
        self.round_count = 0
        self.contact_results = []
        super(HumanPalmContactLoopBehavior, self).__init__()
        self.round_pause = rospy.get_param('~round_pause', 2.0)
        if self.source_name != 'fake':
            rospy.logwarn(
                "~source is '%s', not 'fake': looping just keeps "
                "re-attempting on whatever the real estimator currently "
                "sees, it will not sample new people.", self.source_name)
        elif rospy.get_param('~seed', -1) >= 0:
            # fake_people_pose_estimator_ros.py reads ~seed fresh for every
            # person it creates, so a fixed ~seed pins the *same* person
            # (body, standing position, offered-arm angles) every round --
            # useful for repeatedly re-testing one specific pose, but it
            # means the loop won't explore anything else until ~seed is
            # unset (or removed from the command line).
            rospy.logwarn(
                "~seed is fixed (%d): every round will offer the exact "
                "same person/pose. Unset ~seed to get a different person "
                "each round.", rospy.get_param('~seed'))

    # ------------------------------------------------------------------
    # state machine: intercept the DONE -> (restart) transition
    # ------------------------------------------------------------------
    def control_loop(self, event):
        if self.state == "RESTARTING":
            self._start_next_round()
            return

        super(HumanPalmContactLoopBehavior, self).control_loop(event)

        if self.state == "DONE":
            self._finish_round()
            # own state, unrecognized by the base class's control_loop, so
            # it just falls through and does nothing until we act on it
            # here -- keeps _finish_round from re-running every 0.1s tick.
            self.state = "RESTARTING"

    def _finish_round(self):
        self.round_count += 1
        approach = self._last_report.get('approach')
        press = self._last_report.get('press')
        ok = bool(approach and approach[1] and press and press[1])

        angles = getattr(self.source, 'presented_arm_angles', None)
        self.contact_results.append(dict(
            round=self.round_count,
            ok=ok,
            approach_mm=approach[0] * 1000.0 if approach else None,
            press_mm=press[0] * 1000.0 if press else None,
            angles=dict(angles) if angles else None,
        ))

        lines = ['Round {}: {}'.format(
            self.round_count, 'contact OK' if ok else 'FAILED')]
        if angles:
            lines.append('offered-arm angles [deg]:')
            lines += ['  {} = {:.1f}'.format(k, v)
                      for k, v in sorted(angles.items())]
        if approach:
            lines.append('approach: {:.1f} mm{}'.format(
                approach[0] * 1000.0, '' if approach[1] else ' (missed)'))
        if press:
            lines.append('press: {:.1f} mm{}'.format(
                press[0] * 1000.0, '' if press[1] else ' (missed)'))
        n_ok = sum(1 for r in self.contact_results if r['ok'])
        lines.append('{}/{} rounds reached the palm so far'.format(
            n_ok, self.round_count))

        rospy.loginfo('\n'.join(lines))
        if self.scene is not None:
            self.scene.show_info_text(lines)

        self._last_report = {}

    def _start_next_round(self):
        rospy.loginfo('Waiting for the next person...')
        if self.round_pause > 0.0:
            rospy.sleep(self.round_pause)
        if rospy.is_shutdown():
            return

        # retract to a neutral pose before the next person is judged, so
        # the next round's IK doesn't start from wherever the last press
        # left the arm. reset_pose() only resets joint angles -- the
        # previous round's approach IK (use_base='planar') can have
        # translated/rotated base_link itself, which reset_pose() leaves
        # untouched, so recenter it explicitly here too.
        self.robot.reset_pose()
        self.robot.base_link.newcoords(Coordinates())
        self.ri.frozen = False
        self.ri.angle_vector(self.robot.angle_vector(), 2.0)
        self.ri.wait_interpolation()

        # 前ラウンドの IK ターゲット座標系は _solve_palm_ik が描いたまま
        # DONE でも消えずに残るので (redraw が止まるだけ)、次の人物を待つ
        # 前に隠しておく。次にロックされたときまた update_ik_target が出す。
        if self.scene is not None:
            self.scene.update_ik_target(None)

        self.target_palm_pos = None
        self.target_palm_rot = None
        self.target_hand_rot = None
        self.target_palm_center = None
        self.target_palm_normal = None
        self.last_neck_cmd_time = rospy.Time.now()

        # 新しい人物 (~seed が固定されていなければ体格・立ち位置・差し出す
        # 腕の角度が引き直される) を推定側に作らせる。pose_thread の while
        # 条件が state を見るので、スレッドを起動する前に WAITING へ戻して
        # おく (先に起動すると起動直後の条件チェックで即座に抜けてしまう)。
        self.state = "WAITING"
        self.state_start_time = rospy.Time.now()
        self.source = create_pose_source(self.source_name)
        self._source_stopped = False
        self.pose_thread = threading.Thread(target=self.pose_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()

        rospy.loginfo("Waiting for human (round %d)...", self.round_count + 1)


if __name__ == '__main__':
    try:
        HumanPalmContactLoopBehavior()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
