#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""理想の手繋ぎを決め打ちで再生する viser デモ (認識なし・理想モード).

人 (SMPL) とロボット (Aero) の動きを台本で決めて、1 本の連続した動作として
再生する。実機の速度上限はかけない (理想モード)。各シーンは

    1 気づく・向く → 2 手を出す → 3 握る → 4 一緒に動く

の段階からなり、画面に今の段階と時刻を出す。

シーン
------
A  人がロボットの斜め後ろに来る: 台車は動かさず上半身 (waist_y) だけ斜め後ろへ
   ひねって手を取り、ひねりを戻しながら人を横へ迎えて一緒に歩く。
A' ロボットが人の斜め後ろに来る: 台車で人の右後ろへ回り込み、人が後ろへ出した
   右手を取って、人に手を引かれて歩く。
B  横に並んで歩きながら: 前を向いたまま (腰をひねらず) 手を繋ぐ。
C  手を繋いで一緒に走る (運動会): 繋いだ手を一緒に振って走る。

ロボットは左手、人は右手で繋ぐ (人はロボットの左側)。人の手の位置・向きを
先に決め、ロボットの手はそれに合わせて腕の IK で解く。

Usage
-----
    FEETECH_HAND_DIR=~/ros/hand/src/feetech_hand \
        python3 scripts/scripted_handholding_demo.py
"""

import argparse
import dataclasses
import hashlib
import os
import sys
import threading
import time

# numpy の並列化は小さな行列ばかりのこの計算では空回りして CPU を食うだけなので
# 1 スレッドにする (numpy の import より前に設定する)。
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np  # noqa: E402
from scipy.interpolate import PchipInterpolator  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import scripted_human as shm  # noqa: E402
from aero_demo import smpl_body  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402

DEFAULT_SMPL_MODEL = os.path.expanduser(
    '~/SMPL_python_v.1.0.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

# 人の体型 (SMPL の betas)。betas[0]=+2 で身長 1.64 m・肩 1.32 m。手を下ろした
# 掌の高さが Aero (肩 1.37 m、腕を下ろした手先 0.72 m) とそろい、ロボットが
# 腕をほぼ伸ばしたまま掌を合わせられる (既定の 0 だと身長 1.79 m で、Aero は
# 肘を大きく曲げないと届かず掌が下を向く)。
HUMAN_BETAS = np.array([2.0] + [0.0] * 9)

FPS = 30.0
DT = 1.0 / FPS

# 配色: ロボットは URDF のメッシュの色のまま。人はスライド (キーポーズの図) と
# 同じ青系、地面は白いロボットと青い人の両方が映える中間のグレー。
HUMAN_COLOR = (105, 155, 220)
GROUND_COLOR = (168, 170, 172)
GRID_CELL_COLOR = (150, 152, 155)
GRID_SECTION_COLOR = (120, 122, 126)

# 人がつなぐ手とロボットがつなぐ腕。
HUMAN_HOLD_SIDE = 'R'
ROBOT_HOLD_ARM = 'l'
ROBOT_FREE_ARM = 'r'

# ロボットの掌 (手先座標) を人の掌の表面からどれだけ離すか [m]。
ROBOT_PALM_GAP = 0.006
# ロボットの手先を人の掌の中心から指先側へずらす距離 [m]。
ROBOT_GRASP_DROP = 0.05
# 横に並ぶときの人とロボットの左右の間隔 (骨盤と台車の中心) [m]。
SIDE_DISTANCE = 0.50
# 並んで歩くときにロボットの台車を人より前に置く距離 [m]。Aero が腕を伸ばした
# まま掌を後ろへ向けると手先は台車中心の約 7 cm 前に来るので、人の掌 (骨盤の
# 少し前) の前にロボットの掌が来るように台車を少し前へ出す。後ろに置くと肘を
# 大きく曲げて届かせることになり、掌が下を向く。
ROBOT_AHEAD = 0.04
# 手を取る直前に掌の法線方向へ浮かせる距離 [m]。
PREGRASP_OFFSET = 0.08

# ロボットが走るときの前傾 (waist_p) [rad]。
ROBOT_RUN_LEAN = 0.12

# 首振りの応答 (臨界減衰の固有振動数 [Hz])。人とロボットで少し変える。
HUMAN_GAZE_FREQ = 2.4
ROBOT_GAZE_FREQ = 2.0

# 指を開いたとき・握ったときの関節角 [rad]。
FINGER_OPEN = dict(index=0.15, middle=0.15, ring=0.15, little=0.15,
                   abduction=0.0, thumb0=0.3, thumb=0.1)
FINGER_CLOSED = dict(index=1.15, middle=1.2, ring=1.2, little=1.2,
                     abduction=0.35, thumb0=1.2, thumb=0.9)


# ----------------------------------------------------------------------
# 時間の補間
# ----------------------------------------------------------------------
def minjerk(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


def ramp(t, t0, t1):
    """``t0`` で 0、``t1`` で 1 になる最小躍度の補間。"""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return minjerk((t - t0) / (t1 - t0))


def lerp(a, b, s):
    return (1.0 - s) * np.asarray(a, dtype=np.float64) \
        + s * np.asarray(b, dtype=np.float64)


class KeyframeTrack(object):
    """``(t, x, y, yaw)`` のキーフレームを単調な 3 次補間でつなぐ。"""

    def __init__(self, keys):
        keys = np.asarray(keys, dtype=np.float64)
        self.t0, self.t1 = keys[0, 0], keys[-1, 0]
        self._f = PchipInterpolator(keys[:, 0], keys[:, 1:], axis=0)

    def __call__(self, t):
        return self._f(np.clip(t, self.t0, self.t1))


class SpeedTrack(object):
    """直線 (+x) を速さの台本 ``speed(t)`` で進む。"""

    def __init__(self, x0, y, speed, duration, yaw=0.0):
        self._t = np.arange(0.0, duration + 1.0, 0.001)
        v = np.array([speed(t) for t in self._t])
        self._x = x0 + np.concatenate(
            [[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * 0.001)])
        self.y = y
        self.yaw = yaw

    def shift(self, dx):
        self._x = self._x + dx

    def __call__(self, t):
        return np.array([np.interp(t, self._t, self._x), self.y, self.yaw])


class GazeSchedule(object):
    """時刻ごとの視線先の名前 (``[(t, 'face'), ...]``)。"""

    def __init__(self, keys):
        self.keys = sorted(keys)

    def __call__(self, t):
        name = self.keys[0][1]
        for key_t, key_name in self.keys:
            if t >= key_t:
                name = key_name
        return name


class CriticalDamper(object):
    """首振りをなめらかにする臨界減衰の 2 次系 (速度 0 から目標へ寄る)。"""

    def __init__(self, freq, dim=2):
        self.omega = 2.0 * np.pi * freq
        self.x = None
        self.v = np.zeros(dim)

    def __call__(self, target):
        target = np.asarray(target, dtype=np.float64)
        if self.x is None:
            self.x = target.copy()
            return tuple(self.x)
        # 細かく刻んで積分する (1 フレームを 4 分割)。
        h = DT / 4.0
        for _ in range(4):
            a = self.omega ** 2 * (target - self.x) - 2.0 * self.omega * self.v
            self.v = self.v + a * h
            self.x = self.x + self.v * h
        return tuple(self.x)


# ----------------------------------------------------------------------
# 姿勢の部品
# ----------------------------------------------------------------------
def body_axes(yaw):
    """向き ``yaw`` の (前, 左, 上)。"""
    c, s = np.cos(yaw), np.sin(yaw)
    return (np.array([c, s, 0.0]), np.array([-s, c, 0.0]),
            np.array([0.0, 0.0, 1.0]))


def human_swing_goal(human, shoulders, side, yaw, swing, run, outward=0.06,
                     reach=0.97, swing_angle=0.35):
    """歩く・走るときの腕振りの手首目標。``swing`` は前へ振れて正 (-1〜1)。"""
    fwd, left, up = body_axes(yaw)
    out = left * shm.SIDE_SIGN[side]
    S = shoulders[side]
    L = human.arm_length
    theta = swing_angle * swing
    walk_target = S + reach * L * (np.sin(theta) * fwd - np.cos(theta) * up) \
        + outward * out
    # 走るときは肘を曲げて振る。
    run_target = S + (0.10 + 0.22 * swing) * fwd - 0.04 * out \
        + (-0.30 + 0.06 * swing) * up
    wrist = lerp(walk_target, run_target, run)
    bend = fwd + 0.6 * run * up
    return shm.ArmGoal(wrist=wrist, bend=bend)


def human_hold_goal(human, shoulders, yaw, swing, run, grip,
                    palm_back=False):
    """ロボットと繋いだ右手の目標 (ロボットの側へ少し出す)。掌は前向き
    (ロボットの手が前から掌を後ろへ向けて重なる)、``palm_back`` なら後ろ向き。"""
    fwd, left, up = body_axes(yaw)
    S = shoulders[HUMAN_HOLD_SIDE]
    L = human.arm_length
    theta = 0.04 + (0.22 + 0.25 * run) * swing
    radius = (0.99 - 0.14 * run) * L
    wrist = S + radius * (np.sin(theta) * fwd - np.cos(theta) * up) \
        - (0.08 + 0.02 * run) * left
    return shm.ArmGoal(wrist=wrist, bend=fwd,
                       palm=-fwd if palm_back else fwd, grip=grip)


def blend_goal(a, b, s):
    """2 つの ``ArmGoal`` を ``s`` で混ぜる。"""
    if s <= 0.0:
        return a
    if s >= 1.0:
        return b

    def mix(u, v):
        if u is None:
            return v
        if v is None:
            return u
        return lerp(u, v, s)
    return shm.ArmGoal(wrist=mix(a.wrist, b.wrist), bend=mix(a.bend, b.bend),
                       palm=mix(a.palm, b.palm),
                       grip=(1.0 - s) * a.grip + s * b.grip)


def robot_swing_angles(arm, swing, run):
    """ロボットの腕振りの関節角。``swing`` は前へ振れて正。"""
    mirror = 1.0 if arm == 'l' else -1.0
    shoulder_p = -(0.25 * (1.0 - run) + 0.45 * run) * swing - 0.05 * run
    return {
        '{}_shoulder_p_joint'.format(arm): shoulder_p,
        '{}_shoulder_r_joint'.format(arm): mirror * 0.08,
        '{}_shoulder_y_joint'.format(arm): 0.0,
        '{}_elbow_joint'.format(arm): -0.2 - 0.9 * run,
        '{}_wrist_y_joint'.format(arm): 0.0,
        '{}_wrist_p_joint'.format(arm): 0.0,
        '{}_wrist_r_joint'.format(arm): 0.0,
    }


# ----------------------------------------------------------------------
# シーン
# ----------------------------------------------------------------------
class Scene(object):
    """台本。``script(t, ctx)`` が人とロボットの台本を返す。"""

    key = ''
    title = ''
    duration = 10.0
    phases = []
    follow_camera = False

    def human_pose(self, t):
        return self.human_track(t)

    def robot_pose(self, t):
        return self.robot_track(t) + np.array([self.drift(t), 0.0, 0.0])

    def drift(self, t):
        """並んで動くときのロボットの前後のずれ [m] (前が正)。"""
        return 0.0


def side_by_side_drift(t, t0, t1, amplitude):
    """``t0``〜``t1`` の間、ロボットが人の真横から後ろへゆっくりずれたり戻ったり
    する量 (0〜-``amplitude`` 程度、負が後ろ)。2 つの周期を重ねて単調さを消す。
    Aero の肩は後ろへ 15 度しか振れず、前に出すぎると繋いだ手に届かないので
    後ろ側にだけずらす。"""
    w = ramp(t, t0, t0 + 1.0) - ramp(t, t1 - 1.5, t1)
    u = t - t0
    return -w * amplitude * (0.7 * 0.5 * (1.0 - np.cos(2.0 * np.pi * u / 3.3))
                             + 0.3 * 0.5 * (1.0 - np.cos(2.0 * np.pi * u
                                                         / 1.9)))


class SceneA(Scene):
    key = 'A'
    title = 'A: 人がロボットの斜め後ろに来る (上半身をひねって手を取る)'
    duration = 11.6
    phases = [('1 気づく・振り向く', 1.4, 3.3), ('2 手を出す', 3.3, 4.4),
              ('3 握る', 4.4, 4.9), ('4 横に迎えて一緒に歩く', 4.9, 11.6)]
    WAIST_TWIST = np.deg2rad(110.0)

    def __init__(self):
        walk = [(7.6, 0.45), (8.6, 1.45), (9.6, 2.55), (10.6, 3.65),
                (11.6, 4.75)]
        self.human_track = KeyframeTrack(
            [(0.0, -3.6, 0.62, 0.0), (1.0, -2.6, 0.60, 0.0),
             (2.0, -1.6, 0.55, 0.0), (2.8, -0.98, 0.50, 0.0),
             (3.4, -0.68, 0.48, 0.0), (5.3, -0.68, 0.48, 0.0),
             (6.6, -0.05, SIDE_DISTANCE, 0.0)]
            + [(t, x, SIDE_DISTANCE, 0.0) for t, x in walk])
        self.robot_track = KeyframeTrack(
            [(0.0, 0.0, 0.0, 0.0), (6.6, 0.0, 0.0, 0.0)]
            + [(t, x + ROBOT_AHEAD - 0.08, 0.0, 0.0) for t, x in walk])
        self.robot_gaze = GazeSchedule([
            (0.0, 'forward'), (1.4, 'face'), (3.4, 'hands'), (4.5, 'face'),
            (5.8, 'forward'), (8.6, 'face'), (9.3, 'forward')])
        self.human_gaze = GazeSchedule([
            (0.0, 'forward'), (1.0, 'face'), (3.1, 'hands'), (4.1, 'face'),
            (5.6, 'forward'), (8.7, 'face'), (9.3, 'forward')])

    def script(self, t, ctx):
        human, shoulders, yaw, run = ctx.human, ctx.shoulders, ctx.yaw, ctx.run
        fwd, left, up = body_axes(yaw)
        swing = np.sin(ctx.phase) * ctx.walk
        hold_swing = human_swing_goal(human, shoulders, 'R', yaw, swing, run)
        # 差し出す手: ロボットの手の来る側 (右前) へ。掌はロボットの方へ。
        offer = shm.ArmGoal(
            wrist=shoulders['R'] + 0.30 * fwd - 0.10 * left - 0.46 * up,
            bend=fwd, palm=unit_xy(ctx.robot_base[:2] - shoulders['R'][:2]),
            grip=0.0)
        hold = human_hold_goal(human, shoulders, yaw, 0.6 * swing, run, 1.0,
                               palm_back=True)
        grip = ramp(t, 4.4, 4.9)
        goal = blend_goal(hold_swing, offer, ramp(t, 2.9, 3.8))
        goal.grip = grip
        if t > 5.3:
            offer.grip = 1.0
            goal = blend_goal(offer, hold, ramp(t, 5.3, 6.6))
        arms = {'R': goal,
                'L': human_swing_goal(human, shoulders, 'L', yaw, -swing, run)}
        twist = self.WAIST_TWIST * (ramp(t, 2.5, 3.6) - ramp(t, 5.3, 6.6))
        return dict(
            arms=arms, human_gaze=self.human_gaze(t),
            waist_y=twist, waist_p=0.0, reach=ramp(t, 3.3, 4.4), grip=grip,
            # ひねった上半身から見て前 (後ろの人の方) へ掌を向ける。並んで
            # 歩くときもそのまま (人の掌が後ろ向きで、ロボットの手は後ろから)。
            hold_wrist_y=1.5,
            robot_gaze=self.robot_gaze(t))


class SceneA2(Scene):
    key = "A'"
    title = "A': ロボットが人の斜め後ろに来る (後ろに出した手を取る)"
    duration = 10.2
    phases = [('0 近づく', 0.0, 1.6), ('1 気づく・振り向く', 1.6, 2.6),
              ('2 手を出す', 2.6, 3.9), ('3 握る', 3.9, 4.5),
              ('4 手を引かれて歩く', 4.5, 10.2)]
    # 人の後ろの手は掌の中心が約 0.85 m と高いので、ロボットは腕を伸ばしたまま
    # 前へ上げて届く距離 (人の 0.6 m 後ろ) に立つ。近すぎると肘を曲げるしかなく、
    # Aero の肘は前にしか曲がらないので上腕を後ろへ引いた不自然な姿勢になる。
    OFFSET = np.array([-0.60, -0.42])

    def __init__(self):
        walk = [(5.2, 0.0), (6.2, 0.5), (7.2, 1.5), (8.2, 2.6), (9.2, 3.7),
                (10.2, 4.8)]
        self.human_track = KeyframeTrack(
            [(0.0, 0.0, 0.0, 0.0)] + [(t, x, 0.0, 0.0) for t, x in walk])
        self.robot_track = KeyframeTrack(
            [(0.0, -3.4, -1.0, 0.0), (1.0, -2.5, -0.85, 0.0),
             (2.0, -1.6, -0.62, 0.0), (2.8, -0.92, -0.48, 0.0),
             (3.3, -0.60, -0.42, 0.0)]
            + [(t, x + self.OFFSET[0], self.OFFSET[1], 0.0)
               for t, x in walk])
        self.robot_gaze = GazeSchedule([
            (0.0, 'face'), (2.9, 'hands'), (4.0, 'face'), (5.4, 'forward')])
        self.human_gaze = GazeSchedule([
            (0.0, 'forward'), (1.7, 'face'), (2.8, 'hands'), (3.9, 'face'),
            (4.7, 'forward')])

    def spine_yaw(self, t):
        # 振り返る (右へひねる)。
        return -0.55 * (ramp(t, 1.6, 2.4) - ramp(t, 4.6, 5.3))

    def script(self, t, ctx):
        human, shoulders, yaw, run = ctx.human, ctx.shoulders, ctx.yaw, ctx.run
        fwd, left, up = body_axes(yaw)
        swing = np.sin(ctx.phase) * ctx.walk
        rest = human_swing_goal(human, shoulders, 'R', yaw, swing, run)
        # 後ろへ出す手: 掌は後ろ向き。
        back = shm.ArmGoal(
            wrist=shoulders['R'] - 0.24 * fwd - 0.06 * left - 0.50 * up,
            bend=fwd, palm=-fwd - 0.3 * left, grip=0.0)
        grip = ramp(t, 3.9, 4.5)
        goal = blend_goal(rest, back, ramp(t, 2.6, 3.4))
        goal.grip = grip
        arms = {'R': goal,
                'L': human_swing_goal(human, shoulders, 'L', yaw, -swing, run)}
        return dict(
            arms=arms, human_gaze=self.human_gaze(t),
            spine_yaw=self.spine_yaw(t),
            waist_y=0.0, waist_p=0.0, reach=ramp(t, 2.9, 3.9), grip=grip,
            # 前から来る人の掌 (後ろ向き) へ、掌を前に向けて取る。
            hold_wrist_y=1.5,
            robot_gaze=self.robot_gaze(t))


class SceneB(Scene):
    key = 'B'
    title = 'B: 横に並んで歩きながら手を繋ぐ (前を向いたまま)'
    duration = 10.0
    phases = [('1 気づく・見合う', 1.6, 2.6), ('2 手を出す', 2.6, 3.7),
              ('3 握る', 3.7, 4.2), ('4 手を繋いで歩く', 4.2, 10.0)]
    follow_camera = True
    SPEED = 1.1

    def __init__(self):
        speed = (lambda t: self.SPEED * ramp(t, 0.0, 1.4))
        self.human_track = SpeedTrack(-4.0, SIDE_DISTANCE, speed,
                                      self.duration)
        self.robot_track = SpeedTrack(-4.0 + ROBOT_AHEAD, 0.0, speed,
                                      self.duration)

        # 「前を向いたまま」: 見合うのは最初だけで、手は見ずに取る。
        self.robot_gaze = GazeSchedule([
            (0.0, 'forward'), (1.7, 'face'), (2.5, 'forward'),
            (4.3, 'face'), (4.9, 'forward')])
        self.human_gaze = GazeSchedule([
            (0.0, 'forward'), (1.6, 'face'), (2.6, 'hands'), (3.8, 'face'),
            (4.9, 'forward')])

    def drift(self, t):
        return side_by_side_drift(t, 4.6, self.duration + 1.5, 0.12)

    def script(self, t, ctx):
        human, shoulders, yaw, run = ctx.human, ctx.shoulders, ctx.yaw, ctx.run
        swing = np.sin(ctx.phase) * ctx.walk
        rest = human_swing_goal(human, shoulders, 'R', yaw, swing, run)
        grip = ramp(t, 3.7, 4.2)
        hold = human_hold_goal(human, shoulders, yaw, 0.5 * swing, run, grip)
        goal = blend_goal(rest, hold, ramp(t, 2.6, 3.4))
        goal.grip = grip
        arms = {'R': goal,
                'L': human_swing_goal(human, shoulders, 'L', yaw, -swing, run)}
        return dict(
            arms=arms, human_gaze=self.human_gaze(t),
            waist_y=0.0, waist_p=0.0, reach=ramp(t, 2.8, 3.7), grip=grip,
            robot_gaze=self.robot_gaze(t))


class SceneC(Scene):
    key = 'C'
    title = 'C: 後ろから走ってきて、手を繋いで一緒に走る (運動会)'
    duration = 14.0
    phases = [('0 後ろから走ってくる', 0.0, 3.2), ('1 見合う', 3.2, 4.2),
              ('2 手を出す', 4.2, 5.0), ('3 握る', 5.0, 5.5),
              ('4 一緒に走る', 5.5, 12.8), ('5 ゴール', 12.8, 14.0)]
    follow_camera = True
    SPEED = 3.2           # 一緒に走る速さ [m/s]
    APPROACH_SPEED = 3.0  # ロボットが走ってくる速さ [m/s]
    START_X = -8.0        # 人が待っている位置 (スタートライン)
    STOP_T = 3.2          # ロボットが人の横で止まる時刻

    def __init__(self):
        together = (lambda t: self.SPEED * (ramp(t, 5.8, 7.8)
                                            - ramp(t, 11.3, 12.8)))
        approach = (lambda t: self.APPROACH_SPEED
                    * (1.0 - ramp(t, 1.8, self.STOP_T)))
        self.human_track = SpeedTrack(self.START_X, SIDE_DISTANCE, together,
                                      self.duration)
        self.robot_track = SpeedTrack(
            0.0, 0.0, lambda t: approach(t) + together(t), self.duration)
        # 止まった位置が人の横 (ROBOT_AHEAD だけ前) になるよう平行移動する。
        self.robot_track.shift(self.START_X + ROBOT_AHEAD
                               - self.robot_track(self.STOP_T)[0])
        self.robot_gaze = GazeSchedule([
            (0.0, 'face'), (4.3, 'hands'), (5.0, 'face'),
            (5.7, 'forward'), (12.9, 'face')])
        self.human_gaze = GazeSchedule([
            (0.0, 'forward'), (1.0, 'face'), (4.2, 'hands'), (5.0, 'face'),
            (5.6, 'forward'), (12.8, 'face')])

    def drift(self, t):
        return side_by_side_drift(t, 6.6, 12.6, 0.25)

    def spine_yaw(self, t):
        # 足音に気づいて右後ろを振り返り、ロボットが横に来たら戻す。
        return -0.45 * (ramp(t, 1.0, 1.8) - ramp(t, 2.6, 3.4))

    def script(self, t, ctx):
        human, shoulders, yaw, run = ctx.human, ctx.shoulders, ctx.yaw, ctx.run
        swing = np.sin(ctx.phase) * ctx.walk
        rest = human_swing_goal(human, shoulders, 'R', yaw, swing, run)
        grip = ramp(t, 5.0, 5.5)
        hold = human_hold_goal(human, shoulders, yaw, swing, run, grip)
        goal = blend_goal(rest, hold, ramp(t, 4.2, 4.9))
        goal.grip = grip
        arms = {'R': goal,
                'L': human_swing_goal(human, shoulders, 'L', yaw, -swing, run)}
        return dict(
            arms=arms, human_gaze=self.human_gaze(t),
            spine_yaw=self.spine_yaw(t),
            waist_y=0.0, waist_p=0.0, reach=ramp(t, 4.3, 5.0),
            grip=grip, robot_gaze=self.robot_gaze(t))


SCENES = [SceneA, SceneA2, SceneB, SceneC]


def unit_xy(v):
    v = np.array([v[0], v[1], 0.0])
    return v / max(np.linalg.norm(v), 1e-9)


# ----------------------------------------------------------------------
# ロボット
# ----------------------------------------------------------------------
def setup_robot():
    from aero_demo.aero_urdf_setup import load_aero
    robot = load_aero(use_hand=True)
    for name in ('r_hand_y_joint', 'l_hand_y_joint'):
        joint = getattr(robot, name)
        joint.min_angle = joint.max_angle = 0.0
    # 肘を曲げきった不自然な姿勢を避ける (solve_palm_ik と同じ -120 度)。
    for arm in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(arm)).min_angle = \
            np.deg2rad(-120.0)
    for joint in robot.joint_list:
        joint.joint_angle(0.0)
    return robot


def set_fingers(robot, arm, grip):
    for name in ('index', 'middle', 'ring', 'little', 'abduction', 'thumb0',
                 'thumb'):
        if name in ('thumb0',):
            joint_name = '{}_feetech_thumb0_joint'.format(arm)
        else:
            joint_name = '{}_feetech_{}_pulley_joint'.format(arm, name)
        joint = getattr(robot, joint_name)
        joint.joint_angle(float(np.clip(
            lerp(FINGER_OPEN[name], FINGER_CLOSED[name], grip),
            joint.min_angle, joint.max_angle)))


def robot_gaze_angles(robot, target):
    """首 (yaw, pitch) で頭の正面を ``target`` に向ける角度 (下向きが正)。"""
    body = robot.neck_y_joint.parent_link
    d = body.worldrot().T.dot(np.asarray(target) - robot.head_link.worldpos())
    yaw = float(np.arctan2(d[1], d[0]))
    pitch = float(np.arctan2(-d[2], np.hypot(d[0], d[1])))
    return (float(np.clip(yaw, robot.neck_y_joint.min_angle,
                          robot.neck_y_joint.max_angle)),
            float(np.clip(pitch, robot.neck_p_joint.min_angle,
                          robot.neck_p_joint.max_angle)))


def robot_hold_position(palm_center, palm_normal, finger):
    """人の掌の表面に合わせたロボットの手先の位置 (手先座標は指の付け根寄り
    なので、人の掌の中心より ``ROBOT_GRASP_DROP`` だけ指先側に置く)。"""
    n = palm_normal / np.linalg.norm(palm_normal)
    return palm_center + n * (shm.PALM_HALF_THICKNESS + ROBOT_PALM_GAP) \
        + ROBOT_GRASP_DROP * finger


def robot_palm_normal(robot, arm):
    """ロボットの掌の法線 (左手は手先の -y、右手は +y)。"""
    end = getattr(robot, '{}arm_end_coords'.format(arm))
    y = end.worldrot()[:, 1]
    return -y if arm == 'l' else y


# つなぐ腕の位置を合わせる関節 (肩 2 軸 + 肘)。上腕のひねり (shoulder_y) と
# 手首は台本で決める。Aero の手首はピッチがほぼ動かず、掌の向きまで IK で
# 合わせると肘を大きく曲げた不自然な姿勢になるため、掌の向きは人が手首を
# 曲げて合わせる。3 関節で位置 3 次元を解くので冗長な姿勢に流れない。
HOLD_IK_JOINTS = ('shoulder_p', 'shoulder_r', 'elbow')

# IK が局所解に落ちたときに試す初期姿勢 (左腕の符号)。
HOLD_IK_SEEDS = [
    dict(shoulder_p=-0.2, shoulder_r=0.25, elbow=-0.4),
    dict(shoulder_p=-0.5, shoulder_r=0.3, elbow=-1.0),
    dict(shoulder_p=0.15, shoulder_r=0.35, elbow=-0.3),
    dict(shoulder_p=-0.9, shoulder_r=0.5, elbow=-1.4),
]
# 解き直しを始める位置誤差 [m]。
HOLD_IK_POS_TOL = 0.005


def hold_ik_links(robot, arm=ROBOT_HOLD_ARM):
    return [getattr(robot, '{}_{}_joint'.format(arm, name)).child_link
            for name in HOLD_IK_JOINTS]


def solve_hold_ik(robot, target_pos, warm_start):
    """つなぐ腕の位置の IK。前フレームの解と自然な初期姿勢 (肘を少し曲げて
    下ろした腕) から解き、誤差が残れば初期姿勢を増やす。誤差・曲げ量・前
    フレームからの飛びの小さい解を採る (可動域の端に張りついた局所解を避ける)。
    位置誤差 [m] を返す。"""
    arm = ROBOT_HOLD_ARM
    links = hold_ik_links(robot, arm)
    end = getattr(robot, '{}arm_end_coords'.format(arm))
    target = Coordinates(pos=target_pos)
    mirror = 1.0 if arm == 'l' else -1.0

    def seed_angles(seed):
        return [seed[name] * (mirror if name == 'shoulder_r' else 1.0)
                for name in HOLD_IK_JOINTS]

    def attempt(angles):
        for link, angle in zip(links, angles):
            link.joint.joint_angle(float(np.clip(
                angle, link.joint.min_angle, link.joint.max_angle)))
        robot.inverse_kinematics(
            target, move_target=end, link_list=links, rotation_mask=False,
            stop=80, thre=0.002, revert_if_fail=False)
        p_err = float(np.linalg.norm(end.worldpos() - target_pos))
        solved = [link.joint.joint_angle() for link in links]
        jump = float(np.abs(np.subtract(solved, warm_start)).max())
        effort = sum(abs(a) for a in solved)
        # 肘を伸ばしきった 0 以外で可動域の端に張りついた解は避ける。
        at_limit = sum(
            1 for link in links
            if min(link.joint.joint_angle() - link.joint.min_angle,
                   link.joint.max_angle - link.joint.joint_angle()) < 0.02
            and not (link.joint.name.endswith('elbow_joint')
                     and abs(link.joint.joint_angle()) < 0.02))
        score = 10.0 * p_err + 0.05 * effort + 0.15 * jump + 0.3 * at_limit
        return score, p_err, solved

    candidates = [attempt(warm_start), attempt(seed_angles(HOLD_IK_SEEDS[0]))]
    if min(c[1] for c in candidates) > HOLD_IK_POS_TOL:
        candidates += [attempt(seed_angles(seed))
                       for seed in HOLD_IK_SEEDS[1:]]
    best = min(candidates, key=lambda c: c[0])
    for link, angle in zip(links, best[2]):
        link.joint.joint_angle(angle)
    return best[1]


# ----------------------------------------------------------------------
# タイムラインの計算
# ----------------------------------------------------------------------
class Context(object):
    pass


# 計算したタイムラインの保存先 (台本・人のモデルのコードが変わると作り直す)。
CACHE_DIR = os.path.expanduser('~/.cache/aero_demo/scripted_handholding')
CACHE_FIELDS = ('times', 'vertices', 'robot_angles', 'robot_bases', 'centers',
                'ik_errors', 'ik_rot_errors')


def cache_path(scene_key, model_path):
    """台本・人のモデルのコードと SMPL のモデル・体型から決まる保存先。"""
    digest = hashlib.sha1()
    for path in (os.path.abspath(__file__), os.path.abspath(shm.__file__)):
        with open(path, 'rb') as f:
            digest.update(f.read())
    digest.update(os.path.abspath(model_path).encode())
    digest.update(np.asarray(HUMAN_BETAS).tobytes())
    name = 'scene_{}_{}.npz'.format(scene_key.replace("'", '2'),
                                    digest.hexdigest()[:12])
    return os.path.join(CACHE_DIR, name)


class Timeline(object):
    """シーンを ``FPS`` で刻んで、人の頂点とロボットの関節角を前もって求める。"""

    def __init__(self, scene, human, robot):
        self.scene = scene
        self.times = np.arange(0.0, scene.duration + 1e-9, DT)
        self.vertices = []
        self.robot_angles = []
        self.robot_bases = []
        self.centers = []
        self.ik_errors = []
        self.ik_rot_errors = []
        self._human = human
        self._robot = robot

    def build(self, progress=None, should_stop=None):
        """全フレームを計算する。``should_stop()`` が真になったら途中でやめて
        None を返す。"""
        human, robot, scene = self._human, self._robot, self.scene
        joints = robot.joint_list
        hold_end = getattr(robot, '{}arm_end_coords'.format(ROBOT_HOLD_ARM))
        hold_links = hold_ik_links(robot)
        hold_wrist_y = getattr(robot, '{}_wrist_y_joint'.format(ROBOT_HOLD_ARM))
        hold_shoulder_y = getattr(
            robot, '{}_shoulder_y_joint'.format(ROBOT_HOLD_ARM))
        human_gaze = CriticalDamper(HUMAN_GAZE_FREQ)
        robot_gaze = CriticalDamper(ROBOT_GAZE_FREQ)
        phase = 0.0
        prev_xy = None
        prev_base = None
        robot_phase = 0.0
        ik_seed = None
        robot_head = np.array([0.0, 0.0, 1.55])
        meet = None

        for i, t in enumerate(self.times):
            if should_stop is not None and should_stop():
                return None
            hx, hy, hyaw = scene.human_pose(t)
            xy = np.array([hx, hy])
            speed = 0.0 if prev_xy is None else \
                float(np.linalg.norm(xy - prev_xy)) / DT
            prev_xy = xy
            phase += 2.0 * np.pi * speed * DT / shm.stride_length(speed)
            walk, run = shm.gait_weights(speed)
            base = np.asarray(scene.robot_pose(t), dtype=np.float64)

            ctx = Context()
            ctx.human, ctx.yaw, ctx.phase = human, hyaw, phase
            ctx.walk, ctx.run, ctx.robot_base = walk, run, base
            spec = shm.HumanSpec(xy=xy, yaw=hyaw, speed=speed,
                                 gait_phase=phase)
            # 背骨のひねりは肩の位置を変えるので先に入れる。
            plan = scene.script(t, _ShoulderProbe(human, spec, ctx))
            spec.spine_yaw = plan.get('spine_yaw', 0.0)
            ctx.shoulders = human.shoulder_position(spec)
            plan = scene.script(t, ctx)
            spec.arms = dict(plan['arms'])

            # 人: まず台本どおりの手。
            palm = human.solve(spec).palm(HUMAN_HOLD_SIDE)

            # ロボット: 台車・腰・腕振り。
            for joint in joints:
                joint.joint_angle(0.0)
            robot.base_link.newcoords(Coordinates(
                pos=[base[0], base[1], 0.0],
                rot=rpy_matrix(base[2], 0.0, 0.0)))
            robot.waist_y_joint.joint_angle(plan['waist_y'])
            robot.waist_p_joint.joint_angle(plan['waist_p'])
            # 腕振り: 振り幅は台車の速さ (止まっていれば振らない)、リズムは人の
            # 歩調にそろえる (つなぐ腕は人の右腕と同じ向き、空いた腕は逆)。
            robot_speed = 0.0 if prev_base is None else \
                float(np.linalg.norm(base[:2] - prev_base[:2])) / DT
            prev_base = base
            robot_walk, robot_run = shm.gait_weights(robot_speed)
            # 人が動いていれば人の歩調にそろえ、人が立っている間はロボット
            # 自身の速さで刻む (後ろから走ってくるときなど)。
            robot_phase += 2.0 * np.pi * robot_speed * DT \
                / shm.stride_length(robot_speed)
            if walk > 0.05:
                robot_phase = phase
            swing = np.sin(robot_phase) * robot_walk
            # 走るときは少し前に傾く。
            robot.waist_p_joint.joint_angle(
                plan['waist_p'] + ROBOT_RUN_LEAN * robot_run)
            for arm, value in ((ROBOT_FREE_ARM, -swing),
                               (ROBOT_HOLD_ARM, swing)):
                for name, angle in robot_swing_angles(
                        arm, value, robot_run).items():
                    joint = getattr(robot, name)
                    joint.joint_angle(float(np.clip(
                        angle, joint.min_angle, joint.max_angle)))

            # つなぐ腕: 腕振りの姿勢 → 掌の手前 → 人の掌。手首は手を伸ばし
            # ながら台本の角度へ回す (掌を人の掌へ向ける)。
            reach = plan['reach']
            err = 0.0
            if reach > 0.0:
                hold_wrist_y.joint_angle(
                    plan.get('hold_wrist_y', -1.5) * ramp(reach, 0.0, 0.7))
                hold_shoulder_y.joint_angle(
                    plan.get('hold_shoulder_y', 0.0) * ramp(reach, 0.0, 0.7))
                neutral_pos = hold_end.worldpos().copy()
                pre = PREGRASP_OFFSET * (1.0 - ramp(reach, 0.6, 1.0))
                target_pos = lerp(
                    neutral_pos,
                    robot_hold_position(*palm) + palm[1] * pre,
                    minjerk(min(1.0, reach / 0.8)))
                if ik_seed is None:
                    ik_seed = [link.joint.joint_angle() for link in hold_links]
                err = solve_hold_ik(robot, target_pos, ik_seed)
                ik_seed = [link.joint.joint_angle() for link in hold_links]
            else:
                ik_seed = None
            set_fingers(robot, ROBOT_HOLD_ARM, plan['grip'])
            set_fingers(robot, ROBOT_FREE_ARM, 0.0)

            # 人: 触れる直前から、掌をロボットの掌に合わせる (手首を返し、
            # 掌の中心をロボットの掌の上へ寄せる)。
            adapt = ramp(reach, 0.75, 1.0) if reach > 0.0 else 0.0
            rot_err = 0.0
            if adapt > 0.0:
                n_robot = robot_palm_normal(robot, ROBOT_HOLD_ARM)
                goal = dataclasses.replace(spec.arms[HUMAN_HOLD_SIDE])
                desired = goal.palm if goal.palm is not None else palm[1]
                goal.palm = lerp(desired, -n_robot, adapt)
                goal.align = adapt
                spec.arms[HUMAN_HOLD_SIDE] = goal
                contact = hold_end.worldpos() + n_robot * (
                    shm.PALM_HALF_THICKNESS + ROBOT_PALM_GAP) \
                    - ROBOT_GRASP_DROP * palm[2]
                want = lerp(palm[0], contact, adapt)
                moved = human.solve(spec).palm(HUMAN_HOLD_SIDE)[0]
                goal.wrist = np.asarray(goal.wrist) + (want - moved)
            spec.gaze = self._gaze_point(
                plan['human_gaze'], 'human', ctx, robot_head, meet,
                np.array([hx, hy, 1.5]), hyaw)
            frame = human.solve(spec, gaze_filter=human_gaze)
            palm = frame.palm(HUMAN_HOLD_SIDE)
            meet = palm[0]
            if adapt > 0.0:
                rot_err = float(np.arccos(np.clip(np.dot(
                    palm[1], -robot_palm_normal(robot, ROBOT_HOLD_ARM)),
                    -1.0, 1.0)))
                err = float(np.linalg.norm(
                    robot_hold_position(*palm) - hold_end.worldpos()))

            # ロボットの視線。
            target = self._gaze_point(
                plan['robot_gaze'], 'robot', ctx, frame.joints[smpl_body.HEAD],
                meet, robot.head_link.worldpos(), base[2])
            yaw, pitch = robot_gaze(robot_gaze_angles(robot, target))
            robot.neck_y_joint.joint_angle(yaw)
            robot.neck_p_joint.joint_angle(pitch)
            robot_head = robot.head_link.worldpos().copy()

            self.vertices.append(frame.vertices)
            self.robot_angles.append(
                np.array([j.joint_angle() for j in joints]))
            self.robot_bases.append(base)
            self.centers.append(0.5 * (np.array([hx, hy]) + base[:2]))
            self.ik_errors.append(err)
            self.ik_rot_errors.append(rot_err)
            if progress is not None:
                progress((i + 1) / len(self.times))
        return self

    @staticmethod
    def _gaze_point(name, who, ctx, other_head, meet, own_head, yaw):
        fwd, _, _ = body_axes(yaw)
        if name == 'face' and other_head is not None:
            return np.asarray(other_head)
        if name == 'hands' and meet is not None:
            return np.asarray(meet)
        # 進行方向の 4 m 先、少し下。
        p = np.asarray(own_head) + 4.0 * fwd
        p[2] = own_head[2] - 0.5
        return p

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp.npz'
        np.savez(tmp, **{name: np.asarray(getattr(self, name))
                         for name in CACHE_FIELDS})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, scene):
        tl = cls(scene, None, None)
        with np.load(path) as data:
            for name in CACHE_FIELDS:
                setattr(tl, name, data[name])
        return tl

    def phase_at(self, t):
        for name, t0, t1 in self.scene.phases:
            if t0 <= t < t1:
                return name
        return '-'


class _ShoulderProbe(object):
    """``script`` を先に 1 回呼んで背骨のひねりだけを取り出すための ctx。"""

    def __init__(self, human, spec, ctx):
        self.__dict__.update(ctx.__dict__)
        self.shoulders = human.shoulder_position(spec)


# ----------------------------------------------------------------------
# viser ビューア
# ----------------------------------------------------------------------
def phase_table_markdown(scene):
    """段階ごとの開始・終了・所要時間の表。"""
    lines = ['| 段階 | 開始 | 終了 | 所要 |', '|---|---:|---:|---:|']
    for name, t0, t1 in scene.phases:
        lines.append('| {} | {:.1f} | {:.1f} | {:.1f} s |'.format(
            name, t0, t1, t1 - t0))
    return '\n'.join(lines)


def remove_robot_gui(viewer):
    for attr in ('_joint_angles_folder', '_export_folder',
                 '_obstacles_folder'):
        folder = getattr(viewer, attr, None)
        if folder is not None:
            folder.remove()
            setattr(viewer, attr, None)
    viewer._joint_sliders.clear()
    viewer._joint_folders.clear()


class DemoViewer(object):
    """シーンの選択・再生の GUI。タイムラインは裏のスレッドで、選ばれたシーンを
    優先して計算し、ディスクに保存する (2 回目からは読み込むだけ)。"""

    def __init__(self, args):
        from skrobot.viewers import ViserViewer
        self.args = args
        self.robot = setup_robot()
        # タイムラインの計算は関節角を書き換えるので、表示用とは別のモデルで行う。
        self.build_robot = setup_robot()
        model = smpl_body.load_smpl_model(args.model_path)
        self.human = shm.ScriptedHuman(model, HUMAN_BETAS)
        self.faces = np.asarray(model.f, dtype=np.uint32)
        self.scenes = {cls.key: cls() for cls in SCENES}
        self.timelines = {}
        self.lock = threading.RLock()
        self.build_cv = threading.Condition()
        self.wanted = SCENES[0].key
        self.building = None

        self.viewer = ViserViewer(draw_grid=False)
        server = self.viewer._server
        self.server = server
        server.scene.add_box(
            '/ground', color=GROUND_COLOR, dimensions=(80.0, 30.0, 0.02),
            position=(0.0, 0.0, -0.011))
        self.grid = server.scene.add_grid(
            '/ground_grid', width=80.0, height=30.0, cell_size=1.0,
            section_size=5.0, cell_color=GRID_CELL_COLOR,
            section_color=GRID_SECTION_COLOR, position=(0.0, 0.0, 0.001))
        self.viewer.add(self.robot)
        remove_robot_gui(self.viewer)
        self.mesh = None

        gui = server.gui
        labels = [s.title for s in self.scenes.values()]
        self.scene_dropdown = gui.add_dropdown(
            'シーン', labels, initial_value=labels[0])
        self.status = gui.add_markdown('')
        self.phase_table = gui.add_markdown('')
        self.time_slider = gui.add_slider(
            '時刻 [s]', min=0.0, max=1.0, step=DT, initial_value=0.0)
        self.play = gui.add_checkbox('再生', initial_value=False)
        self.speed = gui.add_dropdown(
            '再生速度', ('1x', '0.5x', '0.25x'), initial_value='1x')
        self.loop = gui.add_checkbox('くり返す', initial_value=True)
        # 追従はカメラを動かさず、2 人が原点付近に留まるよう世界の方をずらす
        # (カメラを動かすとブラウザでのズーム・回転を上書きしてしまうため)。
        self.follow = gui.add_checkbox('2 人を画面の中央に留める',
                                       initial_value=False)
        self.current = None

        @self.scene_dropdown.on_update
        def _(_):  # noqa: ANN001
            key = self._key_of(self.scene_dropdown.value)
            # 計算待ちで GUI を止めないよう、切り替えは別スレッドで。
            threading.Thread(target=self.select, args=(key,),
                             daemon=True).start()

        @self.time_slider.on_update
        def _(_):  # noqa: ANN001
            self.render(self.time_slider.value)

        @self.follow.on_update
        def _(_):  # noqa: ANN001
            self.render(self.time_slider.value)

        @server.on_client_connect
        def _(client):  # noqa: ANN001
            if self.current is not None:
                self._look_at_scene([client])

        threading.Thread(target=self._build_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()
        print('[demo] viser を開いてください (既定 http://localhost:8080)')
        self.select(SCENES[0].key)

    # ------------------------------------------------------------------
    def _build_loop(self):
        """選ばれているシーンを優先して、まだのシーンを順に用意する。"""
        while True:
            with self.build_cv:
                pending = [cls.key for cls in SCENES
                           if cls.key not in self.timelines]
                if not pending:
                    return
                key = self.wanted if self.wanted in pending else pending[0]
                self.building = key
            timeline = self._prepare(key)
            with self.build_cv:
                if timeline is None:  # 別のシーンが選ばれたので後回し
                    self.building = None
                    continue
                self.timelines[key] = timeline
                self.building = None
                self.build_cv.notify_all()

    def _prepare(self, key):
        scene = self.scenes[key]
        path = cache_path(key, self.args.model_path)
        if os.path.exists(path):
            try:
                return Timeline.load(path, scene)
            except Exception as e:  # 壊れた保存は作り直す
                print('[demo] {} を読めないので計算し直します ({})'.format(
                    path, e))

        def progress(ratio):
            if self.wanted == key:
                self.status.content = '**{} を計算中** {:.0f}%'.format(
                    key, 100 * ratio)
        def should_stop():
            # 選ばれているシーンが別にあり、まだ用意できていなければ譲る。
            return self.wanted != key and self.wanted not in self.timelines

        started = time.time()
        timeline = Timeline(scene, self.human, self.build_robot).build(
            progress, should_stop)
        if timeline is None:
            return None
        print('[demo] {} を計算しました ({:.1f} 秒、最大の手先誤差 {:.3f} m)'
              .format(scene.title, time.time() - started,
                      max(timeline.ik_errors)))
        try:
            timeline.save(path)
            # 同じシーンの古い版 (コードが変わる前の計算結果) は消す。
            prefix = os.path.basename(path).rsplit('_', 1)[0] + '_'
            for name in os.listdir(CACHE_DIR):
                if name.startswith(prefix) and name != os.path.basename(path):
                    os.remove(os.path.join(CACHE_DIR, name))
        except OSError as e:
            print('[demo] 計算結果を保存できませんでした ({})'.format(e))
        return timeline

    def _key_of(self, title):
        for key, scene in self.scenes.items():
            if scene.title == title:
                return key
        return SCENES[0].key

    def select(self, key):
        self.play.value = False
        scene = self.scenes[key]
        with self.build_cv:
            self.wanted = key
            if key not in self.timelines:
                self.status.content = '**{} を計算中...** (1 シーン 20 秒ほど。' \
                    '次からは保存した結果を読む)'.format(key)
            while key not in self.timelines:
                self.build_cv.wait()
            timeline = self.timelines[key]
        if self.wanted != key:  # 待っている間に別のシーンが選ばれた
            return
        with self.lock:
            self.current = timeline
            self.follow.value = scene.follow_camera
        self.phase_table.content = phase_table_markdown(scene)
        self.time_slider.max = float(timeline.times[-1])
        self.time_slider.value = 0.0
        self.render(0.0)
        self._look_at_scene()

    def _offset(self, tl, i):
        """表示のずらし量 (2 人を中央に留めるときは 2 人の中点を原点へ)。"""
        if not self.follow.value:
            return np.zeros(2)
        return np.asarray(tl.centers[i], dtype=np.float64)

    def _look_at_scene(self, clients=None):
        tl = self.current
        center = np.asarray(tl.centers[0]) - self._offset(tl, 0)
        if clients is None:
            clients = list(self.server.get_clients().values())
        for client in clients:
            client.camera.look_at = (center[0], center[1], 0.9)
            client.camera.position = (center[0] + 1.2, center[1] - 3.2, 1.9)

    def render(self, t):
        with self.lock:
            tl = self.current
            if tl is None:
                return
            i = int(np.clip(round(t / DT), 0, len(tl.times) - 1))
            offset = self._offset(tl, i)
            for joint, angle in zip(self.robot.joint_list,
                                    tl.robot_angles[i]):
                joint.joint_angle(angle)
            base = tl.robot_bases[i]
            self.robot.base_link.newcoords(Coordinates(
                pos=[base[0] - offset[0], base[1] - offset[1], 0.0],
                rot=rpy_matrix(base[2], 0.0, 0.0)))
            self.viewer.redraw()
            vertices = np.asarray(tl.vertices[i], dtype=np.float32).copy()
            vertices[:, :2] -= offset.astype(np.float32)
            if self.mesh is None:
                self.mesh = self.server.scene.add_mesh_simple(
                    '/human', vertices=vertices, faces=self.faces,
                    color=HUMAN_COLOR)
            else:
                self.mesh.vertices = vertices
            # 地面の格子は 5 m 周期なので、ずらし量の余りだけ動かす。
            self.grid.position = (-(offset[0] % 5.0), -(offset[1] % 5.0),
                                  0.001)
            self.status.content = '**{}**  \n{:.2f} / {:.1f} 秒'.format(
                tl.phase_at(tl.times[i]), tl.times[i], tl.times[-1])

    def _play_loop(self):
        last = time.time()
        while True:
            time.sleep(DT)
            now = time.time()
            elapsed, last = now - last, now
            if not self.play.value or self.current is None:
                continue
            rate = {'1x': 1.0, '0.5x': 0.5, '0.25x': 0.25}[self.speed.value]
            t = self.time_slider.value + elapsed * rate
            if t >= self.time_slider.max:
                if self.loop.value:
                    t = 0.0
                else:
                    t = self.time_slider.max
                    self.play.value = False
            self.time_slider.value = t


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--model-path', default=DEFAULT_SMPL_MODEL,
                        help='SMPL (男性) モデル .pkl のパス。')
    args = parser.parse_args()
    DemoViewer(args)
    while True:
        time.sleep(1.0)


if __name__ == '__main__':
    main()
