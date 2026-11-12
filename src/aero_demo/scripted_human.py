#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""SMPL の人を決め打ちで動かす (歩行・走行の脚、腕の解析 IK、視線).

手繋ぎデモ (``scripts/scripted_handholding_demo.py``) 用。認識は使わず、
台本 (腰の位置・向き・速さ、手首の目標、掌の向き、視線の先) から 1 フレームずつ
SMPL の pose を作って頂点まで求める。

関節の回転はロボット座標系 (x=前, y=左, z=上) で考え、最後に SMPL の
axis-angle に直す (``smpl_body.PERM`` で並べ替えるだけ)。
"""

from dataclasses import dataclass
from dataclasses import field

import numpy as np

from aero_demo import smpl_body as sb


SPINE1, SPINE2, SPINE3 = 3, 6, 9
L_FOOT, R_FOOT = 10, 11
L_COLLAR, R_COLLAR = 13, 14

ARM_JOINTS = {
    # side: (collar, shoulder, elbow, wrist, hand)
    'L': (L_COLLAR, sb.L_SHOULDER, sb.L_ELBOW, sb.L_WRIST, sb.L_HAND),
    'R': (R_COLLAR, sb.R_SHOULDER, sb.R_ELBOW, sb.R_WRIST, sb.R_HAND),
}
# 体の外側 (左腕は +y、右腕は -y)。
SIDE_SIGN = {'L': 1.0, 'R': -1.0}

# T ポーズ (pose=0) の掌の法線 (ロボット座標系)。
REST_PALM_NORMAL = np.array([0.0, 0.0, -1.0])

# 掌の中心を手首から指の付け根 (SMPL の hand 関節) へ寄せる割合。
PALM_CENTER_RATIO = 0.55
# 手の厚みの半分 [m] (掌の中心から掌の表面まで)。
PALM_HALF_THICKNESS = 0.013

# 手首を曲げて掌を合わせるときの上限 [rad]。
WRIST_ALIGN_MAX = np.deg2rad(70.0)

# 視線: 首と頭への配分、可動の上限。
GAZE_NECK_RATIO = 0.4
GAZE_MAX_YAW = np.deg2rad(80.0)
GAZE_PITCH_RANGE = (np.deg2rad(-35.0), np.deg2rad(50.0))

# 歩行・走行の脚の振り [rad]。
WALK_HIP_AMP = 0.38
WALK_KNEE_STANCE = 0.08
WALK_KNEE_SWING = 0.65
RUN_HIP_AMP = 0.62
RUN_KNEE_STANCE = 0.45
RUN_KNEE_SWING = 1.45
RUN_LEAN = 0.16  # 走るときの前傾 [rad]

# 膝をつく (転んだ) ときの脚と上体 [rad]。腿はほぼ鉛直、すねは後ろへ水平。
KNEEL_HIP = 0.05
KNEEL_KNEE = 1.65
KNEEL_LEAN = 0.6
# 膝の関節から膝頭の表面まで [m] (膝をついたときの接地に使う)。
KNEE_RADIUS = 0.055

# 背中を丸める (``stoop``) ときの背骨 3 関節への配分。
STOOP_RATIO = (0.3, 0.35, 0.35)


def rot(axis, angle):
    return sb.rodrigues(np.asarray(axis, dtype=np.float64) * angle)


def rot_x(angle):
    return rot([1.0, 0.0, 0.0], angle)


def rot_y(angle):
    return rot([0.0, 1.0, 0.0], angle)


def rot_z(angle):
    return rot([0.0, 0.0, 1.0], angle)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def frame_from(x, y):
    """``x`` と (``x`` に直交化した) ``y`` を列に持つ回転行列."""
    x = unit(x)
    y = unit(y - np.dot(y, x) * x)
    return np.stack([x, y, np.cross(x, y)], axis=1)


def signed_angle_about(a, b, axis):
    """``axis`` まわりに ``a`` を ``b`` へ回す符号つき角度 (両方を軸に直交化)."""
    axis = unit(axis)
    a = unit(a - np.dot(a, axis) * axis)
    b = unit(b - np.dot(b, axis) * axis)
    return float(np.arctan2(np.dot(np.cross(a, b), axis), np.dot(a, b)))


@dataclass
class ArmGoal:
    """片腕の目標。``wrist`` は手首のワールド位置。"""
    wrist: np.ndarray
    # 前腕の曲がる向きの目安 (肘はこの逆に出る)。ワールド系。
    bend: np.ndarray = None
    # 掌を向けたい向き (ワールド系、None ならひねらない)。
    palm: np.ndarray = None
    # 指を握る量 0〜1。
    grip: float = 0.0
    # 前腕のひねりで合わせきれない掌の向きを手首を曲げて合わせる割合 0〜1
    # (相手の手に合わせるとき)。
    align: float = 0.0


@dataclass
class HumanSpec:
    """1 フレーム分の台本."""
    xy: np.ndarray            # 骨盤の (x, y)
    yaw: float                # 体の向き [rad]
    speed: float = 0.0        # 移動の速さ [m/s] (歩行/走行の混ぜ具合に使う)
    gait_phase: float = 0.0   # 歩行周期の位相 [rad]
    spine_yaw: float = 0.0    # 上半身のひねり (左が正) [rad]
    lean: float = 0.0         # 前傾の追加分 [rad]
    kneel: float = 0.0        # 膝をつく量 0〜1 (転んだとき)
    stoop: float = 0.0        # 背中の丸まり (前へ曲げて正) [rad]
    crouch: float = 0.0       # 立っているときの股・膝の曲げ (膝は 2 倍) [rad]
    ground: float = 0.0       # 足もとの地面の高さ [m] (歩行の脚のとき)
    # 足首の目標 (side -> ワールド位置)。与えると歩行の脚の代わりに脚の
    # IK で足を置き、骨盤の高さは ``pelvis_z`` を使う (座る・段差・跳ぶ)。
    feet: dict = None
    pelvis_z: float = None
    arms: dict = field(default_factory=dict)  # side -> ArmGoal
    gaze: np.ndarray = None   # 見る先 (ワールド位置、None なら正面)


@dataclass
class HumanFrame:
    vertices: np.ndarray      # (6890, 3) float32
    joints: np.ndarray        # (24, 3)
    world_rots: np.ndarray    # (24, 3, 3)

    def palm(self, side):
        """掌の (中心, 法線, 指の向き)。"""
        _, _, _, wrist, hand = ARM_JOINTS[side]
        rw = self.world_rots[wrist]
        normal = rw.dot(REST_PALM_NORMAL)
        finger = unit(self.joints[hand] - self.joints[wrist])
        center = self.joints[wrist] + PALM_CENTER_RATIO * (
            self.joints[hand] - self.joints[wrist])
        return center, normal, finger


def gait_weights(speed):
    """(歩行, 走行) の重み 0〜1。止まると 0。"""
    def smooth(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    walk = smooth(speed / 0.6)
    run = smooth((speed - 1.8) / 1.0)
    return walk, run


def stride_length(speed):
    """1 周期 (左右 1 歩ずつ) で進む距離 [m]。"""
    return 0.7 + 0.6 * max(speed, 0.0)


def make_body_model(model, betas, scale=1.0, head_scale=1.0):
    """体型 ``betas`` を焼き込み、全体を ``scale`` 倍、頭を ``head_scale`` 倍に
    した SMPL モデル (betas はすべて 0 で使う)。子どもは大人の体を縮めて頭の
    割合を大きくして作る。"""
    betas = np.asarray(betas, dtype=np.float64)
    v = model.v_template + np.tensordot(model.shapedirs, betas,
                                        axes=([2], [0]))
    if head_scale != 1.0:
        joints = model.J_regressor.dot(v)
        w = model.weights[:, sb.HEAD][:, None]
        v = v + (head_scale - 1.0) * w * (v - joints[sb.HEAD])
    v = v * scale
    return sb.SmplModel(
        v_template=v, shapedirs=np.zeros_like(model.shapedirs),
        posedirs=model.posedirs * scale, J_regressor=model.J_regressor,
        weights=model.weights, parent=model.parent, f=model.f,
        J=model.J_regressor.dot(v))


class ScriptedHuman(object):
    """1 人分の SMPL (体型固定) を台本から姿勢づけする."""

    def __init__(self, model, betas):
        self.model = model
        self.betas = np.asarray(betas, dtype=np.float64)
        v_shaped = model.v_template + np.tensordot(
            model.shapedirs, self.betas, axes=([2], [0]))
        j_smpl = model.J_regressor.dot(v_shaped)
        # 骨盤原点・ロボット座標系の静止姿勢の関節位置。
        self.rest_joints = (j_smpl - j_smpl[0]).dot(sb.PERM.T)
        self.parent = model.parent

        # 足の関節から足裏までの高さ。
        v_rest, joints_rest = sb.smpl_forward(
            model, np.zeros((24, 3)), self.betas, np.zeros(3))
        v_rest = (v_rest - joints_rest[0]).dot(sb.PERM.T)
        feet = [sb.L_ANKLE, sb.R_ANKLE, L_FOOT, R_FOOT]
        self.sole_offset = float(
            self.rest_joints[feet, 2].min() - v_rest[:, 2].min())
        # 身長、足首の関節から足裏までの高さ、脚の長さ。
        self.height = float(v_rest[:, 2].max() - v_rest[:, 2].min())
        self.ankle_height = float(
            self.rest_joints[sb.L_ANKLE, 2] - v_rest[:, 2].min())
        j = self.rest_joints
        self.leg = {}
        for side, (hip, knee, ankle) in (
                ('L', (sb.L_HIP, sb.L_KNEE, sb.L_ANKLE)),
                ('R', (sb.R_HIP, sb.R_KNEE, sb.R_ANKLE))):
            t0 = j[knee] - j[hip]
            s0 = j[ankle] - j[knee]
            self.leg[side] = dict(
                joints=(hip, knee, ankle),
                L1=float(np.hypot(t0[0], t0[2])),
                L2=float(np.hypot(s0[0], s0[2])),
                a1=float(np.arctan2(t0[0], -t0[2])),
                a2=float(np.arctan2(s0[0], -s0[2])))
        # 立って膝を伸ばしたときの骨盤の高さ。
        self.pelvis_height = float(-v_rest[:, 2].min())
        # 歩幅は脚の長さに比例させる (既定の大人 (betas[0]=2) の股から足首 0.736 m を 1 とする)。
        self.leg_scale = float(
            (j[sb.L_HIP, 2] - j[sb.L_ANKLE, 2]) / 0.736)

        self.arm = {}
        for side, (_, sh, el, wr, _) in ARM_JOINTS.items():
            j = self.rest_joints
            b0 = unit(j[el] - j[sh])
            b0f = unit(j[wr] - j[el])
            d0 = unit(np.array([1.0, 0.0, 0.0])
                      - np.dot([1.0, 0.0, 0.0], b0) * b0)
            self.arm[side] = dict(
                L1=float(np.linalg.norm(j[el] - j[sh])),
                L2=float(np.linalg.norm(j[wr] - j[el])),
                b0=b0, b0f=b0f, d0=d0, rest_frame=frame_from(b0, d0))

    def stride_length(self, speed):
        """1 周期で進む距離 [m] (脚の長さに比例)。"""
        return stride_length(speed) * self.leg_scale

    @property
    def arm_length(self):
        a = self.arm['R']
        return a['L1'] + a['L2']

    # ------------------------------------------------------------------
    def fk(self, local_rots, root_pos):
        """関節だけの順運動学 (ワールド位置, ワールド回転)。"""
        world_rots = np.zeros((24, 3, 3))
        joints = np.zeros((24, 3))
        for i in range(24):
            p = self.parent[i]
            if p < 0:
                world_rots[i] = local_rots[i]
                joints[i] = root_pos
            else:
                world_rots[i] = world_rots[p].dot(local_rots[i])
                joints[i] = joints[p] + world_rots[p].dot(
                    self.rest_joints[i] - self.rest_joints[p])
        return joints, world_rots

    def shoulder_position(self, spec):
        """台本 ``spec`` の腕以外を決めたときの肩の位置 (腕の目標を作るため)。"""
        joints = self.body_joints(spec)
        return {side: joints[ARM_JOINTS[side][1]] for side in ('L', 'R')}

    def body_joints(self, spec):
        """台本 ``spec`` の腕以外を決めたときの関節位置 (24, 3)。"""
        local, root = self._body(spec)
        joints, _ = self.fk(local, root)
        return joints

    def stance(self, stoop=0.0, crouch=0.0, lean=0.0):
        """止まって立つ姿勢の (骨盤の高さ, 骨盤から見た足首の位置 {side: (前,
        左, 上)})。脚の IK で立たせるときに、歩行の脚で立つ姿勢とそろえる。"""
        spec = HumanSpec(xy=np.zeros(2), yaw=0.0, stoop=stoop, crouch=crouch,
                         lean=lean)
        joints = self.body_joints(spec)
        return float(joints[0, 2]), {
            side: joints[self.leg[side]['joints'][2]] - joints[0]
            for side in ('L', 'R')}

    def _body(self, spec):
        """腰・脚・背骨の局所回転と骨盤の位置。``spec.feet`` があれば脚の IK で
        足を置き、なければ歩行の脚で足 (か膝) が地面に着く高さに置く。"""
        local = np.tile(np.eye(3), (24, 1, 1))
        walk, run = gait_weights(spec.speed)
        if spec.feet is not None:
            walk, run = 0.0, 0.0
        k = spec.kneel
        lean = spec.lean + RUN_LEAN * run + KNEEL_LEAN * k
        local[0] = rot_z(spec.yaw).dot(rot_y(lean))

        hip_amp = (1.0 - run) * WALK_HIP_AMP + run * RUN_HIP_AMP
        knee_st = (1.0 - run) * WALK_KNEE_STANCE + run * RUN_KNEE_STANCE
        knee_sw = (1.0 - run) * WALK_KNEE_SWING + run * RUN_KNEE_SWING
        for hip, knee, ankle, phase in (
                (sb.L_HIP, sb.L_KNEE, sb.L_ANKLE, spec.gait_phase),
                (sb.R_HIP, sb.R_KNEE, sb.R_ANKLE,
                 spec.gait_phase + np.pi)):
            # sin > 0 で腿が前、cos > 0 で腿が前へ振れている (遊脚)。
            flex = walk * (hip_amp * np.sin(phase) + 0.1 * run) + spec.crouch
            bend = walk * (knee_st + knee_sw * max(0.0, np.cos(phase)) ** 2) \
                + 2.0 * spec.crouch
            flex = (1.0 - k) * flex + k * KNEEL_HIP
            bend = (1.0 - k) * bend + k * KNEEL_KNEE
            local[hip] = rot_y(-flex - lean)
            local[knee] = rot_y(bend)
            # 足裏をおおよそ水平に保つ。
            local[ankle] = rot_y(0.7 * (flex - bend) + 0.3 * lean)

        # 歩くときの骨盤の小さな回旋は背骨で打ち消す。
        pelvis_twist = 0.08 * walk * np.sin(spec.gait_phase)
        local[0] = local[0].dot(rot_z(pelvis_twist))
        for i, ratio, bow in ((SPINE1, 0.3, STOOP_RATIO[0]),
                              (SPINE2, 0.35, STOOP_RATIO[1]),
                              (SPINE3, 0.35, STOOP_RATIO[2])):
            local[i] = rot_z(ratio * (spec.spine_yaw - pelvis_twist)).dot(
                rot_y(bow * spec.stoop))
        local[SPINE1] = local[SPINE1].dot(rot_y(-0.5 * lean * run))

        if spec.feet is not None:
            root = np.array([spec.xy[0], spec.xy[1], spec.pelvis_z])
            for side, target in spec.feet.items():
                self._leg_ik(local, root, spec.yaw, lean, side, target)
            return local, root

        joints, _ = self.fk(local, np.zeros(3))
        feet = [sb.L_ANKLE, sb.R_ANKLE, L_FOOT, R_FOOT]
        lowest = min(joints[feet, 2].min() - self.sole_offset,
                     joints[[sb.L_KNEE, sb.R_KNEE], 2].min() - KNEE_RADIUS)
        height = spec.ground - lowest
        root = np.array([spec.xy[0], spec.xy[1], height])
        return local, root

    def _leg_ik(self, local, root, yaw, lean, side, target):
        """足首を ``target`` へ置く股・膝の曲げ (体の前後の面内で解く 2 リンク)。
        足裏は水平に保つ。"""
        leg = self.leg[side]
        hip, knee, ankle = leg['joints']
        rest_hip = self.rest_joints[hip] - self.rest_joints[0]
        hip_world = root + rot_z(yaw).dot(rot_y(lean)).dot(rest_hip)
        d = rot_z(yaw).T.dot(np.asarray(target, dtype=np.float64) - hip_world)
        L1, L2 = leg['L1'], leg['L2']
        D = float(np.clip(np.hypot(d[0], d[2]), abs(L1 - L2) + 1e-3,
                          (L1 + L2) * 0.9995))
        gamma = np.arctan2(d[0], -d[2])
        th1 = gamma + np.arccos(np.clip(
            (L1 * L1 + D * D - L2 * L2) / (2.0 * L1 * D), -1.0, 1.0))
        th2 = gamma - np.arccos(np.clip(
            (L2 * L2 + D * D - L1 * L1) / (2.0 * L2 * D), -1.0, 1.0))
        flex = th1 - leg['a1']
        bend = flex + leg['a2'] - th2
        local[hip] = rot_y(-flex - lean)
        local[knee] = rot_y(bend)
        local[ankle] = rot_y(flex - bend)

    def _arm_ik(self, local, world_rots, joints, side, goal):
        """肩・肘を解析的に決める (2 リンク)。"""
        collar, sh, el, wr, hand = ARM_JOINTS[side]
        a = self.arm[side]
        L1, L2 = a['L1'], a['L2']
        S = joints[sh]
        v = np.asarray(goal.wrist, dtype=np.float64) - S
        D = float(np.clip(np.linalg.norm(v), abs(L1 - L2) + 1e-3,
                          (L1 + L2) * 0.999))
        vh = unit(v)
        bend = goal.bend if goal.bend is not None else \
            world_rots[0].dot([1.0, 0.0, 0.0])
        w = np.asarray(bend, dtype=np.float64) - np.dot(bend, vh) * vh
        if np.linalg.norm(w) < 1e-6:
            w = np.cross(vh, [0.0, 0.0, 1.0])
        w = unit(w)
        cos_a = (L1 * L1 + D * D - L2 * L2) / (2.0 * L1 * D)
        alpha = np.arccos(np.clip(cos_a, -1.0, 1.0))
        b = np.cos(alpha) * vh - np.sin(alpha) * w
        d = np.sin(alpha) * vh + np.cos(alpha) * w
        cos_e = (L1 * L1 + L2 * L2 - D * D) / (2.0 * L1 * L2)
        phi = np.pi - np.arccos(np.clip(cos_e, -1.0, 1.0))

        shoulder_world = frame_from(b, d).dot(a['rest_frame'].T)
        local[sh] = world_rots[collar].T.dot(shoulder_world)
        # 肘: 静止姿勢の前腕の向きを、手首へ向かう向きへ最小回転で重ねる
        # (静止姿勢の上腕と前腕はわずかに折れているので phi では合わない)。
        forearm = shoulder_world.T.dot(
            unit(np.cos(phi) * b + np.sin(phi) * d))
        local[el] = sb.rotation_between(a['b0f'], forearm)
        local[wr] = np.eye(3)
        local[hand] = np.eye(3)

    def _arm_twist(self, local, world_rots, side, goal):
        """掌を ``goal.palm`` へ向ける前腕のひねり (肘と手首に半分ずつ)。"""
        _, _, el, wr, hand = ARM_JOINTS[side]
        a = self.arm[side]
        if goal.palm is not None:
            forearm = world_rots[wr].dot(a['b0f'])
            normal = world_rots[wr].dot(REST_PALM_NORMAL)
            psi = signed_angle_about(normal, goal.palm, forearm)
            psi = float(np.clip(psi, -1.9, 1.9))
            local[el] = local[el].dot(rot(a['b0f'], 0.5 * psi))
            local[wr] = rot(a['b0f'], 0.5 * psi)
        if goal.grip > 0.0:
            # 指を掌の側へ曲げる。
            axis = np.cross(a['b0f'], REST_PALM_NORMAL)
            local[hand] = rot(axis, 0.9 * goal.grip)
            local[wr] = local[wr].dot(rot(axis, 0.15 * goal.grip))

    def _arm_align(self, local, world_rots, side, goal):
        """手首を曲げて掌の法線を ``goal.palm`` へ寄せる (``goal.align`` の割合)。"""
        _, _, el, wr, _ = ARM_JOINTS[side]
        normal = world_rots[wr].dot(REST_PALM_NORMAL)
        aa = sb.mat_to_axis_angle(sb.rotation_between(normal, unit(goal.palm)))
        angle = np.linalg.norm(aa)
        if angle < 1e-6:
            return
        scale = goal.align * min(1.0, WRIST_ALIGN_MAX / angle)
        align = sb.rodrigues(aa * scale)
        local[wr] = world_rots[el].T.dot(align).dot(world_rots[wr])

    @staticmethod
    def gaze_angles(joints, world_rots, target):
        """胸の正面から見た ``target`` の (yaw, pitch) (下向きが正)。"""
        d = world_rots[SPINE3].T.dot(np.asarray(target) - joints[sb.HEAD])
        yaw = float(np.clip(np.arctan2(d[1], d[0]),
                            -GAZE_MAX_YAW, GAZE_MAX_YAW))
        pitch = float(np.clip(np.arctan2(-d[2], np.hypot(d[0], d[1])),
                              *GAZE_PITCH_RANGE))
        return yaw, pitch

    def solve(self, spec, gaze_filter=None):
        """台本 ``spec`` の姿勢を作って頂点まで求める.

        ``gaze_filter`` は頭の目標 (yaw, pitch) を受けて実際に使う角度を返す
        関数 (首振りをなめらかにするため。None ならそのまま)。"""
        local, root = self._body(spec)
        joints, world_rots = self.fk(local, root)
        for side, goal in spec.arms.items():
            if goal is not None:
                self._arm_ik(local, world_rots, joints, side, goal)
        joints, world_rots = self.fk(local, root)
        for side, goal in spec.arms.items():
            if goal is not None:
                self._arm_twist(local, world_rots, side, goal)
        if any(g is not None and g.align > 0.0 and g.palm is not None
               for g in spec.arms.values()):
            joints, world_rots = self.fk(local, root)
            for side, goal in spec.arms.items():
                if goal is not None and goal.align > 0.0 \
                        and goal.palm is not None:
                    self._arm_align(local, world_rots, side, goal)

        yaw, pitch = 0.0, 0.0
        if spec.gaze is not None:
            yaw, pitch = self.gaze_angles(joints, world_rots, spec.gaze)
        if gaze_filter is not None:
            yaw, pitch = gaze_filter((yaw, pitch))
        local[sb.NECK] = rot_z(GAZE_NECK_RATIO * yaw).dot(
            rot_y(GAZE_NECK_RATIO * pitch))
        local[sb.HEAD] = rot_z((1.0 - GAZE_NECK_RATIO) * yaw).dot(
            rot_y((1.0 - GAZE_NECK_RATIO) * pitch))

        pose = np.stack([sb.mat_to_axis_angle(sb.to_smpl_rotation(r))
                         for r in local])
        v_local, j_local = sb.smpl_forward(
            self.model, pose, self.betas, np.zeros(3))
        vertices = (v_local - j_local[0]).dot(sb.PERM.T) + root
        joints, world_rots = self.fk(local, root)
        return HumanFrame(vertices=vertices.astype(np.float32),
                          joints=joints, world_rots=world_rots)
