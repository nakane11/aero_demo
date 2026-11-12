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


# 手を振る: 前へ上げた腕 (手先は頭の高さ、掌は前向き) を左右に揺らす。
WAVE_FREQ = 1.6  # [Hz]


def robot_wave_angles(arm, t):
    """手を振る腕の関節角 (左腕の符号)。"""
    mirror = 1.0 if arm == 'l' else -1.0
    phase = 2.0 * np.pi * WAVE_FREQ * t
    return {
        '{}_shoulder_p_joint'.format(arm): -1.35,
        '{}_shoulder_r_joint'.format(arm): mirror * (0.6 + 0.35 * np.sin(phase)),
        '{}_shoulder_y_joint'.format(arm): 0.0,
        '{}_elbow_joint'.format(arm): -1.4 + 0.2 * np.sin(phase + 0.5),
        '{}_wrist_y_joint'.format(arm): 0.0,
        '{}_wrist_p_joint'.format(arm): 0.0,
        '{}_wrist_r_joint'.format(arm): 0.0,
    }


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
    background = 'plain'  # 'plain' (灰色の地面) か 'track' (運動会のトラック)

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
    title = 'C: 後ろから走ってきて、手を繋いで一緒に走る (運動会、途中で転ぶ)'
    phases = [('0 手を振りながら走ってくる', 0.0, 3.2), ('1 見合う', 3.2, 4.2),
              ('2 手を出す', 4.2, 5.0), ('3 握る', 5.0, 5.5),
              ('4 一緒に走る', 5.5, 9.0), ('5 転んで手が離れる', 9.0, 9.6),
              ('6 振り返って手を差し出す', 9.6, 10.9),
              ('7 手を取って起こす', 10.9, 12.8), ('8 見合う', 12.8, 13.4),
              ('9 また走ってゴールへ', 13.4, 18.2), ('10 ゴール', 18.2, 19.7)]
    follow_camera = True
    background = 'track'
    SPEED = 3.2           # 一緒に走る速さ [m/s]
    APPROACH_SPEED = 3.0  # ロボットが走ってくる速さ [m/s]
    START_X = -8.0        # 人が待っている位置 (スタートライン)
    STOP_T = 3.2          # ロボットが人の横で止まる時刻
    FALL_T = 9.0          # 人がつまずく時刻
    OFFER_T = 10.0        # ロボットが手を差し出し始める時刻
    TAKE_T = 10.6         # 人が差し出された手へ手を伸ばし始める時刻
    REGRIP_T = 11.2       # もう一度握る時刻
    STAND_T = 11.6        # 引かれて立ち上がり始める時刻
    RESTART_T = 13.4      # また走り出す時刻
    duration = 19.7
    FALL_LEAD = 0.9       # 転んだとき、ロボットが行き過ぎて止まる距離 [m]
    BACK_LEAD = 0.45      # 人が立った後、ロボットが下がってくる位置 [m]
    TURN = 1.6            # 振り返るときの上半身のひねり [rad]
    # 差し出す手の位置 (台車から見て後ろ・左、高さ) [m]。
    OFFER_FROM_BASE = (-0.45, 0.22, 0.95)

    def __init__(self):
        v = self.SPEED
        r = self.RESTART_T
        together = (lambda t: v * (ramp(t, 5.8, 7.8)
                                   - ramp(t, self.FALL_T, self.FALL_T + 0.5)
                                   + ramp(t, r, r + 2.0)
                                   - ramp(t, r + 3.3, r + 4.8)))
        approach = (lambda t: self.APPROACH_SPEED
                    * (1.0 - ramp(t, 1.8, self.STOP_T)))
        self.human_track = SpeedTrack(self.START_X, SIDE_DISTANCE, together,
                                      self.duration)
        self.robot_track = SpeedTrack(
            0.0, 0.0, lambda t: approach(t) + together(t), self.duration)
        # 止まった位置が人の横 (ROBOT_AHEAD だけ前) になるよう平行移動する。
        self.robot_track.shift(self.START_X + ROBOT_AHEAD
                               - self.robot_track(self.STOP_T)[0])
        # 走っている間、ロボットはときどき人の方を振り返り、人も見返す。
        glances = [(7.9, 8.5), (r + 2.2, r + 2.8)]
        f = self.FALL_T
        robot_keys = [(0.0, 'face'), (4.3, 'hands'), (5.0, 'face'),
                      (5.7, 'forward'), (f + 0.15, 'face'),
                      (self.TAKE_T + 0.2, 'hands'),
                      (self.REGRIP_T + 0.3, 'face'), (r + 0.1, 'forward'),
                      (r + 4.8, 'face')]
        human_keys = [(0.0, 'forward'), (1.0, 'face'), (4.2, 'hands'),
                      (5.0, 'face'), (5.6, 'forward'), (f, 'ground'),
                      (self.OFFER_T + 0.2, 'face'),
                      (self.TAKE_T + 0.1, 'hands'),
                      (self.REGRIP_T + 0.4, 'face'), (r + 0.1, 'forward'),
                      (r + 4.7, 'face')]
        for t0, t1 in glances:
            robot_keys += [(t0, 'face'), (t1, 'forward')]
            human_keys += [(t0 + 0.15, 'face'), (t1 + 0.1, 'forward')]
        self.robot_gaze = GazeSchedule(robot_keys)
        self.human_gaze = GazeSchedule(human_keys)

    def drift(self, t):
        # 走っているときはロボットが少し前に出て引っぱる (5〜15 cm 前で揺らぐ。
        # 前に出すぎると Aero の肩が後ろへ振れきり、繋いだ手が離れる)。
        # 人が転ぶとロボットは少し行き過ぎて止まり、走り出すと横へ戻る。
        r = self.RESTART_T
        w = (ramp(t, 6.0, 7.5) - ramp(t, r + 3.3, r + 4.8))
        u = t - 6.0
        lead = w * (0.10 + 0.035 * np.sin(2.0 * np.pi * u / 3.3)
                    + 0.015 * np.sin(2.0 * np.pi * u / 1.9))
        # 立ち上がったら人の方へ下がってきて、走り出すと横に並ぶ。
        fall = self.FALL_LEAD * ramp(t, self.FALL_T, self.FALL_T + 1.0) \
            - (self.FALL_LEAD - self.BACK_LEAD) * ramp(t, 12.6, r) \
            - self.BACK_LEAD * ramp(t, r, r + 1.3)
        return lead + fall

    def spine_yaw(self, t):
        # 足音に気づいて右後ろを振り返り、ロボットが横に来たら戻す。
        return -0.45 * (ramp(t, 1.0, 1.8) - ramp(t, 2.6, 3.4))

    def kneel(self, t):
        """膝をつく量: つまずいて 0.5 秒で膝と手をつき、手を引かれて立つ。"""
        return ramp(t, self.FALL_T, self.FALL_T + 0.5) \
            - ramp(t, self.STAND_T, self.STAND_T + 1.2)

    def script(self, t, ctx):
        human, shoulders, yaw, run = ctx.human, ctx.shoulders, ctx.yaw, ctx.run
        fwd, left, up = body_axes(yaw)
        f, r = self.FALL_T, self.RESTART_T
        swing = np.sin(ctx.phase) * ctx.walk
        rest = human_swing_goal(human, shoulders, 'R', yaw, swing, run)
        # 握り: 最初に握る → 転んだ拍子に離す → 差し出された手をもう一度握る。
        grip = ramp(t, 5.0, 5.5) - ramp(t, f, f + 0.25) \
            + ramp(t, self.REGRIP_T, self.REGRIP_T + 0.4)
        hold = human_hold_goal(human, shoulders, yaw, swing, run, grip)
        goal = blend_goal(rest, hold, ramp(t, 4.2, 4.9))
        goal.grip = grip
        kneel = self.kneel(t)

        def on_ground(side, out):
            s = shoulders[side]
            p = s + 0.32 * fwd + out * left
            return shm.ArmGoal(wrist=np.array([p[0], p[1], 0.06]), bend=fwd)

        # ロボットが振り返って差し出す手の位置 (台車の後ろ、人との間)。
        rb = ctx.robot_base
        offer_point = np.array([rb[0] + self.OFFER_FROM_BASE[0],
                                rb[1] + self.OFFER_FROM_BASE[1],
                                self.OFFER_FROM_BASE[2]])
        # 差し出された手へ伸ばす右手 (掌は握手のように横向き)。
        take = shm.ArmGoal(wrist=offer_point - 0.06 * fwd,
                           bend=left + 0.3 * up, palm=left, grip=grip)
        # 走り出すときは手を前に伸ばしたまま、ロボットに追いついたら横へ。
        forward = shm.ArmGoal(
            wrist=shoulders['R'] + 0.46 * fwd - 0.10 * left - 0.04 * up,
            bend=left + 0.3 * up, palm=left, grip=grip)
        # 右手: 繋いだ手 → (離して) 地面 → 差し出された手 → 前 → 横。
        goal = blend_goal(goal, on_ground('R', -0.04),
                          ramp(t, f, f + 0.45))
        if t >= self.TAKE_T:
            goal = blend_goal(on_ground('R', -0.04), take,
                              ramp(t, self.TAKE_T, self.REGRIP_T))
        if t >= 12.6:
            goal = blend_goal(take, forward, ramp(t, 12.6, r))
        if t >= r:
            goal = blend_goal(forward, hold, ramp(t, r + 0.2, r + 1.4))
        goal.grip = grip
        # 左手は地面につき、立つときに離す。
        left_goal = human_swing_goal(human, shoulders, 'L', yaw, -swing, run)
        left_goal = blend_goal(left_goal, on_ground('L', 0.06), kneel)
        twist = self.TURN * (ramp(t, f + 0.4, f + 1.1)
                             - ramp(t, r + 0.3, r + 1.4))
        # ロボットの手: 離したら下ろし、振り返って差し出し、人の手を取る。
        reach = ramp(t, 4.3, 5.0) - ramp(t, f, f + 0.6) \
            + ramp(t, self.TAKE_T + 0.2, self.REGRIP_T)
        offer = ramp(t, self.OFFER_T, self.OFFER_T + 0.7) \
            - ramp(t, self.REGRIP_T, self.REGRIP_T + 0.2)
        return dict(
            arms={'R': goal, 'L': left_goal}, human_gaze=self.human_gaze(t),
            spine_yaw=self.spine_yaw(t), kneel=kneel,
            waist_y=twist, waist_p=0.0, reach=reach,
            offer=offer, offer_point=offer_point,
            # 走ってきながら (人から遠い側の手を) 大きく振り、止まる前に下ろす。
            wave=ramp(t, 0.2, 0.8) - ramp(t, 2.6, 3.4),
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


# この見出しより後ろ (ビューア) を変えても計算結果は作り直さない。
VIEWER_SECTION_MARK = '# viser ' + 'ビューア'

# 計算したタイムラインの保存先 (台本・人のモデルのコードが変わると作り直す)。
CACHE_DIR = os.path.expanduser('~/.cache/aero_demo/scripted_handholding')
CACHE_FIELDS = ('times', 'vertices', 'robot_angles', 'robot_bases', 'centers',
                'ik_errors', 'ik_rot_errors')


def cache_path(scene_key, model_path):
    """台本・人のモデルのコードと SMPL のモデル・体型から決まる保存先。"""
    digest = hashlib.sha1()
    for path in (os.path.abspath(__file__), os.path.abspath(shm.__file__)):
        with open(path, 'rb') as f:
            text = f.read()
        # ビューア (背景・GUI) の部分は計算結果に効かないので見ない。
        digest.update(text.split(VIEWER_SECTION_MARK.encode())[0])
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
            spec.kneel = plan.get('kneel', 0.0)
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
                angles = robot_swing_angles(arm, value, robot_run)
                wave = plan.get('wave', 0.0)
                # 手は人から遠い側 (空いた腕) で振る。
                if arm == ROBOT_FREE_ARM and wave > 0.0:
                    waving = robot_wave_angles(arm, t)
                    angles = {name: lerp(angle, waving[name], wave)
                              for name, angle in angles.items()}
                for name, angle in angles.items():
                    joint = getattr(robot, name)
                    joint.joint_angle(float(np.clip(
                        angle, joint.min_angle, joint.max_angle)))

            # つなぐ腕: 腕振りの姿勢 → 掌の手前 → 人の掌。手首は手を伸ばし
            # ながら台本の角度へ回す (掌を人の掌へ向ける)。
            # 手を差し出す (``offer``): 腕を下ろした姿勢から ``offer_point`` へ
            # 手を伸ばして待つ。そこから人の掌へ合わせて握るのが ``reach``。
            reach = plan['reach']
            offer = plan.get('offer', 0.0)
            err = 0.0
            if reach > 0.0 or offer > 0.0:
                turn = max(ramp(reach, 0.0, 0.7), ramp(offer, 0.0, 0.7))
                hold_wrist_y.joint_angle(
                    plan.get('hold_wrist_y', -1.5) * turn)
                hold_shoulder_y.joint_angle(
                    plan.get('hold_shoulder_y', 0.0) * turn)
                target_pos = hold_end.worldpos().copy()
                if offer > 0.0:
                    target_pos = lerp(target_pos, plan['offer_point'],
                                      minjerk(offer))
                pre = PREGRASP_OFFSET * (1.0 - ramp(reach, 0.6, 1.0))
                target_pos = lerp(
                    target_pos,
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
        if name == 'ground':  # 足もと (1 m 先の地面)
            p = np.asarray(own_head) + 1.0 * fwd
            p[2] = 0.0
            return p
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


# 運動会のトラック (シーン C の背景)。
TRACK_COLOR = (178, 88, 62)
GRASS_COLOR = (104, 158, 82)
LINE_COLOR = (245, 245, 240)
LANE_WIDTH = 1.22
TRACK_X = (-30.0, 25.0)
TRACK_Y = (-3.4, 4.0)
FINISH_TAPE_Z = 1.1  # ゴールテープの高さ (胸のあたり) [m]
FINISH_X = 6.0  # 2 度目の走りの途中 (16 秒ごろ) で越える
FLAG_COLORS = [(220, 60, 60), (240, 200, 50), (60, 120, 220), (70, 170, 90),
               (250, 250, 250)]


def sky_image(top, bottom, height=256):
    """上から下へ ``top`` → ``bottom`` のグラデーションの背景画像。"""
    ratio = np.linspace(0.0, 1.0, height)[:, None, None]
    column = (1.0 - ratio) * np.asarray(top) + ratio * np.asarray(bottom)
    return np.repeat(column, 4, axis=1).astype(np.uint8)


def add_plain_ground(scene):
    """灰色の地面と 1 m の格子。格子の handle を返す。"""
    scene.add_box('/bg_plain/ground', color=GROUND_COLOR,
                  dimensions=(80.0, 30.0, 0.02), position=(0.0, 0.0, -0.011))
    return scene.add_grid(
        '/bg_plain/grid', width=80.0, height=30.0, cell_size=1.0,
        section_size=5.0, cell_color=GRID_CELL_COLOR,
        section_color=GRID_SECTION_COLOR, position=(0.0, 0.0, 0.001))


def add_track(scene, lane_center):
    """運動会のトラック (走路・レーン線・スタート/ゴール・万国旗) を
    ``/bg_track`` の下に作る。2 人は ``lane_center`` のレーンを走る。"""
    x0, x1 = TRACK_X
    y0, y1 = TRACK_Y
    scene.add_box('/bg_track/grass', color=GRASS_COLOR,
                  dimensions=(140.0, 70.0, 0.02), position=(0.0, 0.0, -0.03))
    scene.add_box('/bg_track/track', color=TRACK_COLOR,
                  dimensions=(x1 - x0, y1 - y0, 0.02),
                  position=(0.5 * (x0 + x1), 0.5 * (y0 + y1), -0.011))
    # レーン線 (2 人のレーンの両側と、その外側)。
    k = 0
    y = lane_center - 0.5 * LANE_WIDTH
    while y > y0:
        y -= LANE_WIDTH
    y += LANE_WIDTH
    while y < y1:
        scene.add_box('/bg_track/lane_{}'.format(k), color=LINE_COLOR,
                      dimensions=(x1 - x0, 0.05, 0.004),
                      position=(0.5 * (x0 + x1), y, 0.001))
        y += LANE_WIDTH
        k += 1
    # スタートとゴールの線。
    for name, x in (('start', SceneC.START_X), ('finish', FINISH_X)):
        scene.add_box('/bg_track/{}_line'.format(name), color=LINE_COLOR,
                      dimensions=(0.1, y1 - y0, 0.005),
                      position=(x, 0.5 * (y0 + y1), 0.0015))
    # ゴールの柱とテープ (胸の高さ)。テープは 2 人が越えると切れて、両側の柱
    # から垂れ下がる (handle を返して表示を切り替える)。
    poles_y = (y0 - 0.3, y1 + 0.3)
    for i, y in enumerate(poles_y):
        scene.add_box('/bg_track/finish_pole_{}'.format(i),
                      color=(235, 235, 235), dimensions=(0.06, 0.06, 1.5),
                      position=(FINISH_X, y, 0.75))
        scene.add_box('/bg_track/finish_flag_{}'.format(i),
                      color=(210, 50, 50), dimensions=(0.02, 0.4, 0.25),
                      position=(FINISH_X, y, 1.38))
    tape = scene.add_box(
        '/bg_track/finish_tape', color=LINE_COLOR,
        dimensions=(0.01, poles_y[1] - poles_y[0], 0.04),
        position=(FINISH_X, 0.5 * (y0 + y1), FINISH_TAPE_Z))
    broken = []
    for i, (y, sign) in enumerate(((poles_y[0], 1.0), (poles_y[1], -1.0))):
        # 柱から 0.9 m 垂れ下がる (鉛直から 20 度ほど内側へ)。x 軸まわりに
        # 回して、箱の長手 (z) を下・内向きにする。
        angle = np.deg2rad(20.0)
        wxyz = (np.cos(0.5 * sign * angle), np.sin(0.5 * sign * angle),
                0.0, 0.0)
        broken.append(scene.add_box(
            '/bg_track/finish_tape_broken_{}'.format(i), color=LINE_COLOR,
            dimensions=(0.01, 0.04, 0.9), wxyz=wxyz,
            position=(FINISH_X, y + sign * 0.45 * np.sin(angle),
                      FINISH_TAPE_Z - 0.45 * np.cos(angle)), visible=False))
    scene.add_label('/bg_track/finish_label', 'GOAL',
                    position=(FINISH_X, poles_y[1], 1.75),
                    font_size_mode='scene', font_scene_height=0.3,
                    anchor='center-center', depth_test=True)
    scene.add_label('/bg_track/start_label', 'START',
                    position=(SceneC.START_X, y0 - 0.6, 0.05),
                    font_size_mode='scene', font_scene_height=0.3,
                    anchor='center-center', depth_test=True)
    # 万国旗 (トラックの両側、高さ 2.4 m の綱から三角の旗を下げる)。
    for side, y in (('l', y1 + 2.0), ('r', y0 - 2.0)):
        for i, x in enumerate(np.arange(x0, x1, 4.0)):
            scene.add_box('/bg_track/pole_{}_{}'.format(side, i),
                          color=(240, 240, 240), dimensions=(0.05, 0.05, 2.5),
                          position=(x, y, 1.25))
        flags = {}
        for i, x in enumerate(np.arange(x0, x1, 0.5)):
            sag = 0.25 * np.sin(np.pi * ((x - x0) % 4.0) / 4.0)
            top = 2.4 - sag
            flags.setdefault(i % len(FLAG_COLORS), []).append(
                [[x - 0.18, y, top], [x + 0.18, y, top],
                 [x, y, top - 0.35]])
        for c, tris in flags.items():
            verts = np.asarray(tris, dtype=np.float32).reshape(-1, 3)
            faces = np.arange(len(verts), dtype=np.uint32).reshape(-1, 3)
            scene.add_mesh_simple('/bg_track/flags_{}_{}'.format(side, c),
                                  vertices=verts, faces=faces,
                                  color=FLAG_COLORS[c], side='double')
    return tape, broken


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
        self.bg_plain = server.scene.add_frame('/bg_plain', show_axes=False)
        self.grid = add_plain_ground(server.scene)
        self.bg_track = server.scene.add_frame('/bg_track', show_axes=False,
                                               visible=False)
        self.tape, self.tape_broken = add_track(
            server.scene, lane_center=0.5 * SIDE_DISTANCE)
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
        self._set_background(scene.background)
        with self.lock:
            self.current = timeline
            self.follow.value = scene.follow_camera
        self.phase_table.content = phase_table_markdown(scene)
        self.time_slider.max = float(timeline.times[-1])
        self.time_slider.value = 0.0
        self.render(0.0)
        self._look_at_scene()

    def _set_background(self, kind):
        track = kind == 'track'
        self.bg_track.visible = track
        self.bg_plain.visible = not track
        if track:  # 晴れた運動会の空
            sky = sky_image((70, 140, 225), (200, 225, 245))
        else:
            sky = sky_image((205, 218, 232), (238, 241, 245))
        self.server.scene.set_background_image(sky, format='png')

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
            # トラックは有限なのでそのままずらす。
            self.grid.position = (-(offset[0] % 5.0), -(offset[1] % 5.0),
                                  0.001)
            self.bg_track.position = (-offset[0], -offset[1], 0.0)
            # 2 人の胸 (前の方) がゴールを越えたらテープを切る。
            front = max(tl.robot_bases[i][0],
                        float(np.max(tl.vertices[i][:, 0])))
            crossed = front > FINISH_X
            self.tape.visible = not crossed
            for handle in self.tape_broken:
                handle.visible = crossed
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
