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

支える場面 (人が手を取ってロボットが握り返すだけでは成り立たない):
D  ベンチに座った高齢者の手を下から支えて、引き上げて立たせる。
E  段差を降りる高齢者の手を、下の床から向かい合って支える。
F  子どもの手を引き上げて水たまりを跳び越えさせる。
人の体型は対象者に合わせる (``PERSONS``)。

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
import subprocess
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

# 支える場面 (D〜F) の対象者。シーンの ``person`` で選ぶ。
#   model   : SMPL の男性 'm' / 女性 'f'
#   betas   : 体型 (女性は betas[0] が負で背が低い、betas[1] が負でふくよか)
#   scale / head : 全体と頭の倍率 (子どもは大人を縮めて頭の割合を大きくする)
#   stoop / crouch : いつもの姿勢 (背中の丸まり、股・膝の曲げ) [rad]
PERSONS = {
    'adult': dict(label='大人 (身長 1.64 m)', model='m', betas=HUMAN_BETAS,
                  scale=1.0, head=1.0, stoop=0.0, crouch=0.0,
                  color=(105, 155, 220)),
    # 小柄な高齢の女性 (身長 1.49 m): 背中が丸く、膝が少し曲がっている。
    'elderly': dict(label='高齢の女性 (身長 1.49 m、背中が丸い)', model='f',
                    betas=np.array([-2.2, -1.0] + [0.0] * 8), scale=1.0,
                    head=1.0, stoop=0.38, crouch=0.07,
                    color=(170, 120, 175)),
    # 5〜6 歳の子ども (身長 1.12 m): 頭が大きめ。
    'child': dict(label='5〜6 歳の子ども (身長 1.12 m)', model='m',
                  betas=np.zeros(10), scale=0.6, head=1.4, stoop=0.0,
                  crouch=0.0, color=(240, 160, 60)),
}


def person_model_path(person, model_path):
    """男性モデルのパス ``model_path`` から、``person`` の SMPL モデルのパス。"""
    if PERSONS[person]['model'] == 'm':
        return model_path
    return os.path.join(os.path.dirname(model_path),
                        'basicModel_f_lbs_10_207_0_v1.0.0.pkl')


def make_person(person, model_path, models=None):
    """``person`` の ``ScriptedHuman`` (``models`` は読んだ SMPL の使い回し)。"""
    spec = PERSONS[person]
    path = person_model_path(person, model_path)
    if models is not None and path in models:
        model = models[path]
    else:
        model = smpl_body.load_smpl_model(path)
        if models is not None:
            models[path] = model
    body = shm.make_body_model(model, spec['betas'], spec['scale'],
                               spec['head'])
    return shm.ScriptedHuman(body, np.zeros(10))

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
    person = 'adult'      # 人の体型 (``PERSONS``)

    def setup(self, human):
        """計算の前に人の体の寸法から台本の位置を決める (必要なシーンだけ)。"""

    def still(self, t):
        """脚を台本で置く (歩行の脚を使わない) 時刻なら真。"""
        return False

    def props(self, timeline):
        """シーンに置く物 (ベンチ・段差・水たまり) の一覧。"""
        return []

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


# ----------------------------------------------------------------------
# 支える場面 (ロボットが握り返すだけでは成り立たない手繋ぎ)
# ----------------------------------------------------------------------
# 向かい合って手を引くとき: ロボットは人の方 (-x) を向き、掌を上へ向けて
# 差し出す。人はその上に掌を下にして手を乗せ、ロボットが指で包んで握る
# (介助で「手を下から支える」持ち方)。人が体重をかけてもロボットの手で
# 受け止められる。
FACE_YAW = np.pi
# ロボットの肩から掌までの距離 (腕を少し曲げた楽な長さ) [m] と、人の手首
# から台車の中心までの左右のずれ [m]。
FACE_ARM_REACH = 0.52
FACE_BASE_DY = 0.22
PALM_UP = np.array([0.0, 0.0, 1.0])


def robot_shoulder_offset(lifter, bow):
    """昇降 ``lifter`` とお辞儀 ``bow`` での肩の (前, 高さ) [m] (Aero の実測の
    近似: 昇降は 1 - cos で下がり、腰の軸から肩まで 0.305 m)。"""
    return (-0.018 + 0.305 * np.sin(bow),
            1.37 - 0.50 * (1.0 - np.cos(lifter))
            - 0.305 * (1.0 - np.cos(bow)))


def face_base(wrist, lifter=0.0, bow=0.0, min_x=None):
    """人の手首 ``wrist`` の手を向かい合って下から支えるときの台車 (x, y, yaw)。
    肩から掌が ``FACE_ARM_REACH`` になる距離に立つ。"""
    fwd, height = robot_shoulder_offset(lifter, bow)
    palm_x = wrist[0] + 0.07
    dz = height - wrist[2]
    horiz = np.sqrt(max(FACE_ARM_REACH ** 2 - dz * dz, 0.2 ** 2))
    x = palm_x + horiz + fwd
    if min_x is not None:
        x = max(x, min_x)
    return np.array([x, wrist[1] + FACE_BASE_DY, FACE_YAW])


def feet_arc(start, end, s, lift=0.05, early=0.35):
    """足首を ``start`` から ``end`` へ運ぶ (前へ出してから下ろす弧)。"""
    s = float(np.clip(s, 0.0, 1.0))
    p = lerp(start, end, minjerk(min(1.0, s / (1.0 - early))))
    p[2] = lerp(start[2], end[2], minjerk((s - early) / (1.0 - early)))
    p[2] += lift * np.sin(np.pi * s)
    return p


class SceneD(Scene):
    key = 'D'
    title = 'D: ベンチに座った高齢者の手を下から支えて立ち上がらせる'
    person = 'elderly'
    duration = 14.0
    APPROACH_END = 2.2   # ロボットが前に来て止まる
    CROUCH = (1.6, 3.0)  # しゃがんで目線を合わせる
    OFFER = (3.0, 3.8)   # 掌を上にして手を差し出す
    TAKE = (3.5, 4.4)    # 人がその上に手を乗せる
    REACH = (4.0, 4.7)   # ロボットが下から手を合わせる
    ROBOT_GRIP = (4.5, 5.0)
    HUMAN_GRIP = (4.7, 5.2)
    LEAN = (5.2, 6.2)    # 前かがみ (お辞儀) で重心を足の上へ
    RISE = (6.0, 8.0)    # お尻が浮いて立ち上がる
    WALK_T = 8.8         # 手を引かれて歩き出す
    WALK_SPEED = 0.45
    phases = [('0 ロボットが前に来る', 0.0, 2.2),
              ('1 しゃがんで目線を合わせる', 2.2, 3.0),
              ('2 掌を上にして手を差し出す', 3.0, 3.8),
              ('3 手を乗せる・下から包んで握る', 3.8, 5.2),
              ('4 前かがみになる', 5.2, 6.0),
              ('5 手を引いて立ち上がらせる', 6.0, 8.0),
              ('6 立って落ち着く', 8.0, 8.8),
              ('7 手を引いて歩く (ロボットは後ずさり)', 8.8, 14.0)]

    def __init__(self):
        w = self.WALK_T
        self.walk_track = SpeedTrack(
            0.0, 0.0, lambda t: self.WALK_SPEED * ramp(t, w, w + 1.2),
            self.duration)
        self.robot_gaze = GazeSchedule([
            (0.0, 'face'), (3.2, 'hands'), (4.9, 'face'), (6.3, 'hands'),
            (7.6, 'face')])
        self.human_gaze = GazeSchedule([
            (0.0, 'ground'), (0.7, 'face'), (3.4, 'hands'), (5.0, 'face'),
            (5.6, 'feet'), (7.9, 'face'), (w + 0.6, 'ground'),
            (w + 2.6, 'face'), (w + 3.4, 'ground')])

    def setup(self, human):
        sp = PERSONS[self.person]
        j = human.rest_joints
        leg = human.leg['R']
        # 座る: すねは鉛直、腿はほぼ水平 (座面の高さは膝下の長さ)。
        self.pz_sit = human.ankle_height + 0.98 * leg['L2'] \
            - j[smpl_body.R_HIP][2]
        self.feet_x = j[smpl_body.R_HIP][0] + 0.9 * leg['L1']
        self.feet = {side: np.array([
            self.feet_x, j[human.leg[side]['joints'][2]][1],
            human.ankle_height]) for side in ('L', 'R')}
        self.pz_stand, ankles = human.stance(sp['stoop'], sp['crouch'])
        self.x_stand = self.feet_x - 0.5 * (ankles['L'][0] + ankles['R'][0])
        shoulder_y = j[smpl_body.R_SHOULDER][1]
        self.w_take = np.array([self.feet_x + 0.06, shoulder_y + 0.01,
                                self.pz_sit + 0.16])
        # 立ったときの右肩から手首 (前へ出して少し下)。
        sh = human.body_joints(shm.HumanSpec(
            xy=np.array([self.x_stand, 0.0]), yaw=0.0, stoop=sp['stoop'],
            crouch=sp['crouch']))[smpl_body.R_SHOULDER]
        self.w_stand = sh + np.array([0.27, 0.01, -0.30])
        # 手の通り道: 前かがみで前へ → お尻が浮くときにいちばん前 → 立つと
        # 胸の前へ上がる。
        l0, l1 = self.LEAN
        r0, r1 = self.RISE
        wt = self.w_take
        self.hold_track = KeyframeTrack([
            (l0,) + tuple(wt),
            (l1,) + tuple(wt + [0.18, 0.0, -0.03]),
            (r0 + 0.35 * (r1 - r0),) + tuple(wt + [0.32, 0.0, 0.0]),
            (r1,) + tuple(self.w_stand)])

    def _rise(self, t):
        """(前へ出る量, 上がる量) 0〜1。お尻は前へ出てから上がる。"""
        r0, r1 = self.RISE
        return ramp(t, r0, r0 + 0.7 * (r1 - r0)), \
            ramp(t, r0 + 0.25 * (r1 - r0), r1)

    def human_pose(self, t):
        fwd, _ = self._rise(t)
        p = self.walk_track(t)
        p[0] += self.x_stand * fwd
        return p

    def still(self, t):
        return t < self.WALK_T

    def hold_point(self, t):
        """手を繋いだ人の右手首の位置。"""
        w = self.hold_track(t)
        w[0] += self.walk_track(t)[0]
        return w

    def lifter(self, t):
        """しゃがんで目線を合わせ、立ち上がらせるときに伸びる。"""
        r0, r1 = self.RISE
        return 1.3 * ramp(t, *self.CROUCH) - 0.9 * ramp(t, r0 + 0.3, r1)

    def bow(self, t):
        return 0.3 * ramp(t, *self.CROUCH) - 0.2 * ramp(t, *self.RISE)

    def robot_pose(self, t):
        base = face_base(self.hold_point(t), self.lifter(t), self.bow(t))
        # 前から近づいて止まる。
        base[0] += 2.4 * (1.0 - ramp(t, 0.0, self.APPROACH_END))
        return base

    def script(self, t, ctx):
        human, shoulders, yaw = ctx.human, ctx.shoulders, ctx.yaw
        fwd, left, up = body_axes(yaw)
        body = ctx.body
        swing = np.sin(ctx.phase) * ctx.walk
        _, rise_up = self._rise(t)
        r0, r1 = self.RISE
        # 膝の上に置いた手 (座っている間)。
        def on_thigh(side, along):
            knee = body[smpl_body.L_KNEE if side == 'L'
                        else smpl_body.R_KNEE]
            return shm.ArmGoal(wrist=knee - along * fwd + 0.09 * up,
                               bend=up - 0.5 * left * shm.SIDE_SIGN[side],
                               palm=-up)
        human_grip = 0.55 * ramp(t, *self.HUMAN_GRIP)  # 握る力は弱い
        take = shm.ArmGoal(wrist=self.hold_point(t),
                           bend=up + 0.4 * left, palm=-up, grip=human_grip)
        right = blend_goal(on_thigh('R', 0.14), take, ramp(t, *self.TAKE))
        right.grip = human_grip
        # 左手は膝に置き、立つときは膝を押して、立ったら下ろす。
        left_goal = blend_goal(on_thigh('L', 0.14), on_thigh('L', 0.03),
                               ramp(t, *self.LEAN))
        left_goal = blend_goal(
            left_goal, human_swing_goal(human, shoulders, 'L', yaw, -swing,
                                        0.0),
            ramp(t, r0 + 0.5 * (r1 - r0), r1 + 0.3))
        plan = dict(
            arms={'R': right, 'L': left_goal},
            human_gaze=self.human_gaze(t),
            # 座っているときは少し前かがみ、立ち上がるときに大きく傾けて
            # から起こす。
            lean=0.12 + 0.55 * ramp(t, *self.LEAN)
            - 0.67 * ramp(t, r0 + 0.45 * (r1 - r0), r1),
            waist_y=0.0,
            # しゃがんで (昇降を縮めて) 目線を合わせ、お辞儀して手を出す。
            # 立ち上がらせるときに昇降を伸ばして手を持ち上げる。
            waist_p=self.bow(t), lifter=self.lifter(t),
            offer=ramp(t, *self.OFFER),
            offer_point=self.w_take + 0.10 * fwd - 0.06 * up,
            reach=ramp(t, *self.REACH), hold_palm=PALM_UP,
            grip=ramp(t, *self.ROBOT_GRIP),
            robot_gaze=self.robot_gaze(t))
        if t < self.WALK_T:
            plan['feet'] = self.feet
            plan['pelvis_z'] = lerp(self.pz_sit, self.pz_stand, rise_up)
        return plan

    def props(self, timeline):
        """ベンチ: 座面は最初の姿勢のお尻・腿の下面に合わせる。"""
        v = np.asarray(timeline.vertices[0])
        under = v[(v[:, 0] > -0.25) & (v[:, 0] < self.feet_x - 0.12)]
        top = float(under[:, 2].min())
        back = float(v[:, 0].min()) - 0.03
        front = self.feet_x - 0.12
        wood, leg = (176, 128, 84), (90, 90, 95)
        return [
            dict(name='seat', color=wood,
                 dimensions=(front - back, 1.3, 0.04),
                 position=(0.5 * (front + back), 0.0, top - 0.02)),
            dict(name='back', color=wood, dimensions=(0.04, 1.3, 0.45),
                 position=(back - 0.02, 0.0, top + 0.30),
                 wxyz=(np.cos(-0.06), 0.0, np.sin(-0.06), 0.0)),
            dict(name='leg_l', color=leg,
                 dimensions=(front - back - 0.06, 0.05, top - 0.04),
                 position=(0.5 * (front + back), 0.6, 0.5 * (top - 0.04))),
            dict(name='leg_r', color=leg,
                 dimensions=(front - back - 0.06, 0.05, top - 0.04),
                 position=(0.5 * (front + back), -0.6, 0.5 * (top - 0.04))),
        ]


class SceneE(Scene):
    key = 'E'
    title = 'E: 段差を降りる高齢者の手を下から支える'
    person = 'elderly'
    duration = 14.6
    STEP_H = 0.15  # 1 段の高さ [m]
    STEP_D = 0.32  # 踏み面の奥行き [m]
    EDGE_X = -0.45  # 段差の手前で止まる骨盤の位置
    STOP_T = 2.4
    APPROACH = (2.4, 3.4)  # ロボットが寄ってくる
    OFFER = (3.3, 4.0)
    TAKE = (3.7, 4.6)
    REACH = (4.1, 4.8)
    ROBOT_GRIP = (4.6, 5.0)
    HUMAN_GRIP = (4.8, 5.2)
    # 1 歩ごとの (始め, 終わり, 足, 段): 先に左足を下ろし、右足をそろえる。
    STEPS = [(5.5, 6.4, 'L', 1), (6.6, 7.4, 'R', 1),
             (7.8, 8.7, 'L', 2), (8.9, 9.7, 'R', 2)]
    WALK_T = 10.5
    WALK_SPEED = 0.45
    phases = [('0 段差の手前まで歩いて止まる', 0.0, 2.4),
              ('1 足もとを見る・ロボットが寄る', 2.4, 3.3),
              ('2 掌を上にして手を差し出す', 3.3, 4.0),
              ('3 手を乗せる・下から包んで握る', 4.0, 5.2),
              ('4 1 段目を降りる', 5.5, 7.4), ('5 2 段目を降りる', 7.8, 9.7),
              ('6 降りて一息', 9.7, 10.5),
              ('7 手を引いて歩く (ロボットは後ずさり)', 10.5, 14.6)]

    def __init__(self):
        e, s = self.EDGE_X, self.STOP_T
        walk = SpeedTrack(0.0, 0.0, lambda t: 0.5 * (
            ramp(t, 0.0, 0.6) - ramp(t, s - 0.8, s)), self.duration)
        walk.shift(e - walk(s)[0])
        self.approach_track = walk
        w = self.WALK_T
        self.walk_track = SpeedTrack(
            0.0, 0.0, lambda t: self.WALK_SPEED * ramp(t, w, w + 1.2),
            self.duration)
        self.robot_gaze = GazeSchedule([
            (0.0, 'face'), (3.5, 'hands'), (5.0, 'face'), (5.4, 'hands'),
            (9.8, 'face'), (w + 1.0, 'hands'), (w + 2.0, 'face')])
        self.human_gaze = GazeSchedule([
            (0.0, 'forward'), (1.6, 'feet'), (2.9, 'face'), (3.6, 'hands'),
            (5.0, 'feet'), (9.8, 'face'), (w + 0.6, 'ground'),
            (w + 2.4, 'face')])

    def ground_at(self, x):
        """床の高さ (段差は -x 側が高い)。"""
        if x < -self.STEP_D:
            return 2.0 * self.STEP_H
        if x < 0.0:
            return self.STEP_H
        return 0.0

    def setup(self, human):
        sp = PERSONS[self.person]
        self.pz_stand, ankles = human.stance(sp['stoop'], sp['crouch'])
        self.ankles = ankles
        # 段ごとの足首の位置 (骨盤から見た左右はそのまま)。
        ah = human.ankle_height
        toe = 0.5 * (ankles['L'][0] + ankles['R'][0])
        tread = [self.EDGE_X + toe, -0.5 * self.STEP_D - 0.03, 0.18]
        self.foot_at = {
            (side, k): np.array([tread[k], ankles[side][1],
                                 (2 - k) * self.STEP_H + ah])
            for side in ('L', 'R') for k in range(3)}
        self.pelvis_x = [x - toe for x in tread]
        shoulder_y = human.rest_joints[smpl_body.R_SHOULDER][1]
        self.hold_rel = np.array([0.32, shoulder_y + 0.01, 0.07])

    def _feet(self, t):
        """各足の段 (0〜2) と運んでいる途中の位置。"""
        feet = {}
        for side in ('L', 'R'):
            level = 0
            pos = self.foot_at[(side, 0)]
            for t0, t1, foot, k in self.STEPS:
                if foot != side or t < t0:
                    continue
                pos = feet_arc(self.foot_at[(side, k - 1)],
                               self.foot_at[(side, k)],
                               (t - t0) / (t1 - t0))
                level = k
            feet[side] = (level, pos)
        return feet

    def _pelvis(self, t):
        """骨盤の (x, 高さ)。前の足を下ろす間は後ろの膝を曲げて沈み、
        そろえると立つ。"""
        x = self.pelvis_x[0]
        z = 2.0 * self.STEP_H + self.pz_stand
        for t0, t1, foot, k in self.STEPS:
            s = ramp(t, t0, t1)
            if foot == 'L':  # 前の足を下ろす: 半分前へ、後ろの膝を曲げて沈む
                x += 0.5 * (self.pelvis_x[k] - self.pelvis_x[k - 1]) * s
                z -= 0.95 * self.STEP_H * s
            else:  # 後ろの足をそろえる: 残りを前へ
                x += 0.5 * (self.pelvis_x[k] - self.pelvis_x[k - 1]) * s
                z -= 0.05 * self.STEP_H * s
        return x, z

    def human_pose(self, t):
        if t < self.STEPS[0][0]:
            return self.approach_track(t)
        x, _ = self._pelvis(t)
        return np.array([x + self.walk_track(t)[0], 0.0, 0.0])

    def still(self, t):
        return self.STEPS[0][0] <= t < self.WALK_T

    def hold_point(self, t):
        if t < self.STEPS[0][0]:
            x, z = self.approach_track(t)[0], 2.0 * self.STEP_H + self.pz_stand
        else:
            x, z = self._pelvis(t)
            x += self.walk_track(t)[0]
        if t >= self.WALK_T:
            z = self.pz_stand
        return np.array([x, 0.0, z]) + self.hold_rel

    def lifter(self, t):
        """降りていく人の手に合わせて少し低くなる。"""
        return 0.45 * ramp(t, self.STEPS[0][0], self.STEPS[-1][1])

    def robot_pose(self, t):
        # 段差の蹴上げ (x=0) に台車の前がぶつからない位置まで。
        base = face_base(self.hold_point(t), self.lifter(t), 0.1,
                         min_x=0.38)
        if t < self.APPROACH[1]:
            s = ramp(t, *self.APPROACH)
            arrive = face_base(self.hold_point(self.APPROACH[1]), 0.0, 0.1,
                               min_x=0.38)
            return lerp(arrive + np.array([0.9, 0.0, 0.0]), arrive, s)
        return base

    def script(self, t, ctx):
        human, shoulders, yaw = ctx.human, ctx.shoulders, ctx.yaw
        fwd, left, up = body_axes(yaw)
        swing = np.sin(ctx.phase) * ctx.walk
        rest = human_swing_goal(human, shoulders, 'R', yaw, swing, 0.0)
        human_grip = 0.55 * ramp(t, *self.HUMAN_GRIP)
        take = shm.ArmGoal(wrist=self.hold_point(t), bend=up + 0.4 * left,
                           palm=-up, grip=human_grip)
        right = blend_goal(rest, take, ramp(t, *self.TAKE))
        right.grip = human_grip
        # 左手は少し横へ広げて釣り合いをとる。
        stepping = ramp(t, 5.2, 5.8) - ramp(t, 9.7, 10.5)
        left_goal = human_swing_goal(
            human, shoulders, 'L', yaw, -swing, 0.0,
            outward=0.06 + 0.14 * stepping)
        plan = dict(
            arms={'R': right, 'L': left_goal},
            human_gaze=self.human_gaze(t),
            lean=0.12 * stepping,
            waist_y=0.0, waist_p=0.1, lifter=self.lifter(t),
            offer=ramp(t, *self.OFFER),
            offer_point=self.hold_point(t) + 0.10 * fwd - 0.06 * up,
            reach=ramp(t, *self.REACH), hold_palm=PALM_UP,
            grip=ramp(t, *self.ROBOT_GRIP),
            robot_gaze=self.robot_gaze(t))
        if t < self.STEPS[0][0]:
            plan['ground'] = 2.0 * self.STEP_H
        elif t < self.WALK_T:
            _, z = self._pelvis(t)
            plan['feet'] = {side: pos for side, (_, pos)
                            in self._feet(t).items()}
            plan['pelvis_z'] = z
            plan['ground'] = self.ground_at(self.human_pose(t)[0] + 0.1)
        return plan

    def props(self, timeline):
        stone = (196, 190, 178)
        d, h = self.STEP_D, self.STEP_H
        return [
            dict(name='step1', color=stone, dimensions=(d, 1.6, h),
                 position=(-0.5 * d, 0.0, 0.5 * h)),
            dict(name='step2', color=(184, 178, 166),
                 dimensions=(3.0, 1.6, 2.0 * h),
                 position=(-d - 1.5, 0.0, h)),
            dict(name='edge1', color=(230, 200, 60),
                 dimensions=(0.05, 1.6, 0.005), position=(-0.025, 0.0, h)),
            dict(name='edge2', color=(230, 200, 60),
                 dimensions=(0.05, 1.6, 0.005),
                 position=(-d - 0.025, 0.0, 2.0 * h)),
        ]


class SceneF(Scene):
    key = 'F'
    title = 'F: 子どもの手を引き上げて水たまりを跳び越えさせる'
    person = 'child'
    duration = 10.5
    SIDE = 0.48          # 子どもとロボットの左右の間隔 [m]
    SPEED = 0.8
    STOP_T = 2.6
    PUDDLE = (0.0, 0.40, 0.42, 0.34)  # 中心 x, y と半径 x, y [m]
    TAKE_OFF_X = -0.62
    CROUCH = (3.5, 4.1)  # せーの (かがむ)
    FLIGHT = (4.1, 4.65)
    LAND = (4.65, 5.2)   # 着地して膝で受ける
    JUMP_HEIGHT = 0.18   # 骨盤が上がる高さ [m] (足は膝を曲げてもっと上がる)
    FEET_LIFT = 0.28
    RESTART_T = 6.4
    phases = [('0 手を繋いで歩く', 0.0, 2.6), ('1 水たまりの前で止まり見合う',
                                              2.6, 3.5),
              ('2 せーの (かがむ)', 3.5, 4.1),
              ('3 手を引き上げて跳ぶ', 4.1, 4.65),
              ('4 着地', 4.65, 5.2), ('5 見合って喜ぶ', 5.2, 6.4),
              ('6 また歩く', 6.4, 10.5)]

    def __init__(self):
        s, r = self.STOP_T, self.RESTART_T
        walk = SpeedTrack(0.0, self.SIDE, lambda t: self.SPEED * (
            1.0 - ramp(t, s - 1.0, s) + ramp(t, r, r + 1.0)), self.duration)
        walk.shift(self.TAKE_OFF_X - walk(s)[0])
        self.walk_track = walk
        self.jump = 2.0 * abs(self.TAKE_OFF_X)  # 跳ぶ距離
        self.robot_gaze = GazeSchedule([
            (0.0, 'forward'), (1.0, 'face'), (1.6, 'forward'),
            (2.4, 'puddle'), (3.0, 'face'), (4.0, 'forward'), (4.9, 'face'),
            (self.RESTART_T + 0.6, 'forward'), (8.3, 'face'),
            (8.9, 'forward')])
        self.human_gaze = GazeSchedule([
            (0.0, 'forward'), (1.1, 'face'), (1.7, 'forward'),
            (2.1, 'puddle'), (3.1, 'face'), (3.7, 'puddle'),
            (4.9, 'face'), (self.RESTART_T + 0.5, 'forward'), (8.4, 'face'),
            (9.0, 'forward')])

    def setup(self, human):
        sp = PERSONS[self.person]
        self.pz_stand, self.ankles = human.stance(sp['stoop'], sp['crouch'])
        self.ankle_height = human.ankle_height
        self.leg_length = human.leg['R']['L1'] + human.leg['R']['L2']

    def _flight(self, t):
        """跳んでいる割合 0〜1 (空中では水平に等速)。"""
        return float(np.clip(
            (t - self.FLIGHT[0]) / (self.FLIGHT[1] - self.FLIGHT[0]),
            0.0, 1.0))

    def human_pose(self, t):
        p = self.walk_track(t)
        p[0] += self.jump * self._flight(t)
        return p

    def robot_pose(self, t):
        p = self.human_pose(t)
        return np.array([p[0] + ROBOT_AHEAD, 0.0, 0.0])

    def still(self, t):
        return self.CROUCH[0] <= t < self.LAND[1]

    def script(self, t, ctx):
        human, shoulders, yaw = ctx.human, ctx.shoulders, ctx.yaw
        fwd, left, up = body_axes(yaw)
        swing = np.sin(ctx.phase) * ctx.walk
        s = self._flight(t)
        # かがむ量 (せーの と着地)。
        dip = ramp(t, *self.CROUCH) - ramp(t, self.FLIGHT[0] - 0.08,
                                           self.FLIGHT[0] + 0.1) \
            + ramp(t, self.LAND[0] - 0.05, self.LAND[0] + 0.15) \
            - ramp(t, self.LAND[0] + 0.2, self.LAND[1])
        air = np.sin(np.pi * s)
        # 繋いだ右手: ロボットの方へ斜め下に伸ばす。跳ぶときは引き上げられる。
        S = shoulders['R']
        hold = shm.ArmGoal(
            wrist=S + (0.10 - 0.04 * air) * fwd - 0.12 * left
            - (0.25 - 0.17 * air) * up,
            bend=fwd + 0.5 * up, palm=fwd, grip=1.0)
        # 空いた左手: かがむと後ろへ引き、跳ぶと前へ振り上げる。
        SL = shoulders['L']
        back = shm.ArmGoal(wrist=SL - 0.14 * fwd + 0.06 * left - 0.26 * up,
                           bend=fwd)
        throw = shm.ArmGoal(wrist=SL + 0.24 * fwd + 0.16 * left - 0.06 * up,
                            bend=up + left)
        left_goal = human_swing_goal(human, shoulders, 'L', yaw, -swing, 0.0)
        left_goal = blend_goal(left_goal, back, ramp(t, *self.CROUCH))
        left_goal = blend_goal(left_goal, throw,
                               ramp(t, self.FLIGHT[0] - 0.05,
                                    self.FLIGHT[0] + 0.2))
        left_goal = blend_goal(
            left_goal,
            human_swing_goal(human, shoulders, 'L', yaw, -swing, 0.0),
            ramp(t, self.LAND[0] + 0.2, self.RESTART_T))
        cx, cy, _, _ = self.PUDDLE
        plan = dict(
            arms={'R': hold, 'L': left_goal},
            human_gaze=(self.human_gaze(t) if self.human_gaze(t) != 'puddle'
                        else np.array([cx, cy, 0.0])),
            lean=0.35 * dip + 0.15 * air,
            waist_y=0.0, waist_p=0.0,
            # 子どもの背に合わせて低くし、子どもと一緒に沈んで、跳ぶときに
            # 伸ばして手を引き上げる。
            lifter=0.65 + 0.3 * dip - 0.55 * air,
            reach=1.0, grip=1.0,
            robot_gaze=(self.robot_gaze(t) if self.robot_gaze(t) != 'puddle'
                        else np.array([cx, cy, 0.0])))
        if self.still(t):
            x0 = self.TAKE_OFF_X
            x1 = x0 + self.jump
            ah = self.ankle_height
            feet = {}
            for side in ('L', 'R'):
                a = self.ankles[side]
                start = np.array([x0 + a[0], self.SIDE + a[1], ah])
                end = np.array([x1 + a[0], self.SIDE + a[1], ah])
                p = lerp(start, end, minjerk(s))
                # 跳んでいる間は膝を曲げて足を引きつける。
                p[2] += self.FEET_LIFT * air
                feet[side] = p
            plan['feet'] = feet
            plan['pelvis_z'] = self.pz_stand - 0.13 * dip \
                + self.JUMP_HEIGHT * air
        return plan

    def props(self, timeline):
        cx, cy, rx, ry = self.PUDDLE
        n = 48
        a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        wobble = 1.0 + 0.08 * np.sin(3.0 * a) + 0.05 * np.cos(5.0 * a)
        ring = np.stack([cx + rx * wobble * np.cos(a),
                         cy + ry * wobble * np.sin(a),
                         np.full(n, 0.004)], axis=1)
        verts = np.vstack([[cx, cy, 0.004], ring]).astype(np.float32)
        faces = np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)],
                         dtype=np.uint32)
        return [dict(name='puddle', mesh=(verts, faces),
                     color=(96, 128, 168))]


SCENES = [SceneA, SceneA2, SceneB, SceneC, SceneD, SceneE, SceneF]


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


def best_wrist_y(robot, arm, desired):
    """掌の法線を ``desired`` (ワールド) へいちばん近づける手首のひねり
    (wrist_y) の角度。手首のひねりは前腕の軸まわりの回転なので、法線は
    c + a cos(q) + b sin(q) と書け、3 点で係数を決めて最大を解析的に求める。"""
    joint = getattr(robot, '{}_wrist_y_joint'.format(arm))
    keep = joint.joint_angle()
    q = 1.2
    normals = []
    for angle in (0.0, q, -q):
        joint.joint_angle(angle)
        normals.append(robot_palm_normal(robot, arm))
    joint.joint_angle(keep)
    n0, n1, n2 = normals
    b = (n1 - n2) / (2.0 * np.sin(q))
    a = (n0 - 0.5 * (n1 + n2)) / (1.0 - np.cos(q))
    d = np.asarray(desired, dtype=np.float64)
    best = float(np.arctan2(np.dot(d, b), np.dot(d, a)))
    best = float(np.clip(best, joint.min_angle, joint.max_angle))
    score = float(np.dot(d, a * np.cos(best) + b * np.sin(best)
                         + (n0 - a)))
    return best, score


def solve_hold_ik_palm(robot, target_pos, desired, warm_start,
                       prev_shoulder_y, turn=1.0):
    """手先の位置を合わせつつ、掌をできるだけ ``desired`` (ワールド) へ向ける。
    上腕のひねり (shoulder_y) をいくつか試し、それぞれで位置の IK と手首の
    ひねり (wrist_y) を交互に解いて、位置の誤差・掌の向きのずれ・前フレーム
    からの飛びの小さい解を採る (Aero の手首は掌を真上まで回しきれないので、
    残ったずれは人が手首を曲げて合わせる)。(位置誤差, shoulder_y, wrist_y)
    を返す。"""
    arm = ROBOT_HOLD_ARM
    shoulder = getattr(robot, '{}_shoulder_y_joint'.format(arm))
    wrist = getattr(robot, '{}_wrist_y_joint'.format(arm))
    links = hold_ik_links(robot, arm)
    best = None
    for dq in (0.0, -0.3, 0.3, -0.6, 0.6):
        q = float(np.clip(prev_shoulder_y + dq, max(shoulder.min_angle, -1.2),
                          min(shoulder.max_angle, 0.8)))
        shoulder.joint_angle(q * turn)
        for link, angle in zip(links, warm_start):
            link.joint.joint_angle(angle)
        w = wrist.joint_angle()
        err = solve_hold_ik(robot, target_pos, warm_start)
        for _ in range(2):
            w, _ = best_wrist_y(robot, arm, desired)
            wrist.joint_angle(w * turn)
            err = solve_hold_ik(robot, target_pos,
                                [link.joint.joint_angle() for link in links])
        n = robot_palm_normal(robot, arm)
        angle = float(np.arccos(np.clip(np.dot(n, desired), -1.0, 1.0)))
        score = 10.0 * err + 0.3 * angle + 0.3 * abs(dq)
        if best is None or score < best[0]:
            best = (score, err, q, w,
                    [link.joint.joint_angle() for link in links])
        if dq == 0.0 and err < HOLD_IK_POS_TOL and angle < 0.5:
            break
    _, err, q, w, angles = best
    shoulder.joint_angle(q * turn)
    wrist.joint_angle(w * turn)
    for link, angle in zip(links, angles):
        link.joint.joint_angle(angle)
    return err, q, w


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


def cache_path(scene_key, model_path, person='adult'):
    """台本・人のモデルのコードと SMPL のモデル・体型から決まる保存先。"""
    digest = hashlib.sha1()
    for path in (os.path.abspath(__file__), os.path.abspath(shm.__file__)):
        with open(path, 'rb') as f:
            text = f.read()
        # ビューア (背景・GUI) の部分は計算結果に効かないので見ない。
        digest.update(text.split(VIEWER_SECTION_MARK.encode())[0])
    digest.update(os.path.abspath(person_model_path(person, model_path))
                  .encode())
    digest.update(repr(sorted(
        (k, np.asarray(v).tolist() if k == 'betas' else v)
        for k, v in PERSONS[person].items())).encode())
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
        scene.setup(human)
        person = PERSONS[scene.person]
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
        prev_wrist_y = 0.0
        prev_shoulder_y = 0.0
        eye_height = 0.93 * human.height

        for i, t in enumerate(self.times):
            if should_stop is not None and should_stop():
                return None
            hx, hy, hyaw = scene.human_pose(t)
            xy = np.array([hx, hy])
            speed = 0.0 if prev_xy is None or scene.still(t) else \
                float(np.linalg.norm(xy - prev_xy)) / DT
            prev_xy = xy
            phase += 2.0 * np.pi * speed * DT / human.stride_length(speed)
            walk, run = shm.gait_weights(speed)
            base = np.asarray(scene.robot_pose(t), dtype=np.float64)

            ctx = Context()
            ctx.human, ctx.yaw, ctx.phase = human, hyaw, phase
            ctx.walk, ctx.run, ctx.robot_base = walk, run, base
            spec = shm.HumanSpec(xy=xy, yaw=hyaw, speed=speed,
                                 gait_phase=phase, stoop=person['stoop'],
                                 crouch=person['crouch'])
            # 背骨のひねり・脚の置き方は肩の位置を変えるので先に入れる。
            plan = scene.script(t, _ShoulderProbe(human, spec, ctx))
            for name in ('spine_yaw', 'kneel', 'lean', 'ground', 'feet',
                         'pelvis_z'):
                if name in plan:
                    setattr(spec, name, plan[name])
            spec.stoop += plan.get('stoop', 0.0)
            spec.crouch += plan.get('crouch', 0.0)
            ctx.body = human.body_joints(spec)
            ctx.shoulders = {side: ctx.body[shm.ARM_JOINTS[side][1]]
                             for side in ('L', 'R')}
            ctx.ground = spec.ground
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
            # 昇降 (膝・足首) で上体を下げる量 [rad]。
            lifter = plan.get('lifter', 0.0)
            robot.knee_joint.joint_angle(-lifter)
            robot.ankle_joint.joint_angle(lifter)
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
            hold_palm = plan.get('hold_palm')
            err = 0.0
            if reach > 0.0 or offer > 0.0:
                turn = max(ramp(reach, 0.0, 0.7), ramp(offer, 0.0, 0.7))
                if hold_palm is None:
                    hold_wrist_y.joint_angle(
                        plan.get('hold_wrist_y', -1.5) * turn)
                    hold_shoulder_y.joint_angle(
                        plan.get('hold_shoulder_y', 0.0) * turn)
                else:
                    hold_wrist_y.joint_angle(prev_wrist_y * turn)
                    hold_shoulder_y.joint_angle(prev_shoulder_y * turn)
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
                if hold_palm is not None:
                    err, prev_shoulder_y, prev_wrist_y = solve_hold_ik_palm(
                        robot, target_pos, hold_palm,
                        [link.joint.joint_angle() for link in hold_links],
                        prev_shoulder_y, turn)
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
                np.array([hx, hy, spec.ground + eye_height]), hyaw)
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
        if not isinstance(name, str):  # 見る先の位置をじかに与えたとき
            return np.asarray(name, dtype=np.float64)
        if name == 'face' and other_head is not None:
            return np.asarray(other_head)
        if name == 'hands' and meet is not None:
            return np.asarray(meet)
        if name == 'ground':  # 足もと (1 m 先の地面)
            p = np.asarray(own_head) + 1.0 * fwd
            p[2] = 0.0
            return p
        if name == 'feet':  # すぐ足もと (段差を降りるとき)
            p = np.asarray(own_head) + 0.45 * fwd
            p[2] = ctx.ground
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
        self.body = human.body_joints(spec)
        self.shoulders = {side: self.body[shm.ARM_JOINTS[side][1]]
                          for side in ('L', 'R')}
        self.ground = spec.ground


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


# 動画の保存先。
VIDEO_DIR = os.path.expanduser('~/aero_demo_videos')


class DemoViewer(object):
    """シーンの選択・再生の GUI。タイムラインは裏のスレッドで、選ばれたシーンを
    優先して計算し、ディスクに保存する (2 回目からは読み込むだけ)。"""

    def __init__(self, args):
        from skrobot.viewers import ViserViewer
        self.args = args
        self.robot = setup_robot()
        # タイムラインの計算は関節角を書き換えるので、表示用とは別のモデルで行う。
        self.build_robot = setup_robot()
        self.models = {}
        self.humans = {}
        model = smpl_body.load_smpl_model(args.model_path)
        self.models[args.model_path] = model
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
        self.mesh_person = None
        self.props = {}  # scene key -> 物を置いた frame

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
        self.record_button = gui.add_button('動画を保存 (今の画角)')
        self.record_lock = threading.Lock()
        self.current = None

        @self.record_button.on_click
        def _(event):  # noqa: ANN001
            threading.Thread(target=self.record, args=(event.client,),
                             daemon=True).start()

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

    def _human(self, person):
        if person not in self.humans:
            self.humans[person] = make_person(person, self.args.model_path,
                                              self.models)
        return self.humans[person]

    def _prepare(self, key):
        scene = self.scenes[key]
        path = cache_path(key, self.args.model_path, scene.person)
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
        timeline = Timeline(scene, self._human(scene.person),
                            self.build_robot).build(progress, should_stop)
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
        self._show_props(key, timeline)
        with self.lock:
            if self.mesh is not None and self.mesh_person != scene.person:
                self.mesh.remove()
                self.mesh = None
            self.mesh_person = scene.person
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

    def _show_props(self, key, timeline):
        """シーンの物 (ベンチ・段差・水たまり) を出し、他のシーンの物は隠す。"""
        if key not in self.props:
            name = '/props_{}'.format(key.replace("'", '2'))
            frame = self.server.scene.add_frame(name, show_axes=False)
            for item in self.scenes[key].props(timeline):
                path = '{}/{}'.format(name, item['name'])
                if 'mesh' in item:
                    verts, faces = item['mesh']
                    self.server.scene.add_mesh_simple(
                        path, vertices=verts, faces=faces,
                        color=item['color'], side='double')
                else:
                    self.server.scene.add_box(
                        path, color=item['color'],
                        dimensions=item['dimensions'],
                        position=item['position'],
                        wxyz=item.get('wxyz', (1.0, 0.0, 0.0, 0.0)))
            self.props[key] = frame
        for k, frame in self.props.items():
            frame.visible = k == key

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
                    color=PERSONS[tl.scene.person]['color'])
            else:
                self.mesh.vertices = vertices
            # 地面の格子は 5 m 周期なので、ずらし量の余りだけ動かす。
            # トラックは有限なのでそのままずらす。
            self.grid.position = (-(offset[0] % 5.0), -(offset[1] % 5.0),
                                  0.001)
            self.bg_track.position = (-offset[0], -offset[1], 0.0)
            props = self.props.get(tl.scene.key)
            if props is not None:
                props.position = (-offset[0], -offset[1], 0.0)
            # 2 人の胸 (前の方) がゴールを越えたらテープを切る。
            front = max(tl.robot_bases[i][0],
                        float(np.max(tl.vertices[i][:, 0])))
            crossed = front > FINISH_X
            self.tape.visible = not crossed
            for handle in self.tape_broken:
                handle.visible = crossed
            self.status.content = '**{}**  \n{:.2f} / {:.1f} 秒'.format(
                tl.phase_at(tl.times[i]), tl.times[i], tl.times[-1])

    def record(self, client):
        """ボタンを押したブラウザの今のカメラで、シーンの最初から最後までを
        描画して MP4 に保存し、そのブラウザにダウンロードとして送る。
        画像は 3D の画面だけで、右上の GUI パネルは写らない。"""
        if client is None or self.current is None:
            return
        if not self.record_lock.acquire(blocking=False):
            return
        try:
            self.play.value = False
            tl = self.current
            width = int(client.camera.image_width) // 2 * 2 or 1280
            height = int(client.camera.image_height) // 2 * 2 or 720
            os.makedirs(VIDEO_DIR, exist_ok=True)
            path = os.path.join(VIDEO_DIR, 'scene_{}_{}.mp4'.format(
                tl.scene.key.replace("'", '2'),
                time.strftime('%Y%m%d_%H%M%S')))
            ffmpeg = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo',
                 '-pix_fmt', 'rgb24', '-s', '{}x{}'.format(width, height),
                 '-r', str(FPS), '-i', '-', '-c:v', 'libx264',
                 '-pix_fmt', 'yuv420p', '-crf', '18', path],
                stdin=subprocess.PIPE)
            n = len(tl.times)
            for i, t in enumerate(tl.times):
                self.render(t)
                image = client.camera.get_render(
                    height=height, width=width, transport_format='png')
                ffmpeg.stdin.write(np.ascontiguousarray(
                    image[:, :, :3], dtype=np.uint8).tobytes())
                if i % 10 == 0:
                    self.record_button.label = '録画中 {:.0f}%'.format(
                        100.0 * (i + 1) / n)
            ffmpeg.stdin.close()
            ffmpeg.wait()
            print('[demo] 動画を保存しました: {}'.format(path))
            with open(path, 'rb') as f:
                client.send_file_download(os.path.basename(path), f.read())
            message = '動画を保存しました: `{}`'.format(path)
        except Exception as e:  # 録画の失敗で表示を止めない
            print('[demo] 動画を保存できませんでした: {}'.format(e))
            message = '動画を保存できませんでした: {}'.format(e)
        finally:
            self.record_button.label = '動画を保存 (今の画角)'
            self.time_slider.value = 0.0
            self.record_lock.release()
        self.status.content = message

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
