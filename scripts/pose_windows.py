# -*- coding:utf-8 -*-
"""握手の各段階で、台車の置き方と全身姿勢を許す範囲 (窓)。

台車の置き方や「無理のない姿勢」は、候補を並べるコスト (重み) では誘導せず、
IK の台車の可動域と関節の可動域で与える。窓に入っていればどの解も同じ扱い
(候補の順番は IK の初期値の順)。

腕の関節の窓は右腕の向き (``r_*_joint`` の符号) で書き、左腕では肩ロール・
肩ヨー・手首ヨー・手首ロールの符号を反転して使う (左右の可動域が鏡像のため)。
"""

import contextlib
import math
import os

# --- 評価用: 制約にする項目 (環境変数 AERO_POSE_WINDOWS、カンマ区切り) -----
# hover_base hover_bend hover_crouch hover_bow hover_twist
# goal_place goal_upper goal_legsum goal_crouch goal_wrist goal_bend
ENABLED = frozenset(
    item for item in os.environ.get('AERO_POSE_WINDOWS', '').split(',')
    if item)


def enabled(item):
    return item in ENABLED


# --- hover (precontact_pose) ---------------------------------------------
# 台車の x を人の立ち位置 ± この幅 [m] に絞る (1 段、広げ直さない)。
HOVER_BASE_X_MARGIN = float(os.environ.get('AERO_HOVER_BASE_X_MARGIN', 1.0))
# 台車の向きを人の正面方向 ± この角度 [deg] に絞る。
HOVER_BASE_YAW_MARGIN_DEG = float(
    os.environ.get('AERO_HOVER_BASE_YAW_MARGIN_DEG', 30.0))
# 差し出す腕の関節の窓 [deg] (右腕の向き、(下限, 上限)、None は可動域のまま)。
HOVER_ARM_WINDOW_DEG = {
    'shoulder_p': (-70.0, None),
    'shoulder_r': (-70.0, None),
    'elbow': (-100.0, None),
}
# 肩ヨーの上限 [deg] (肘を外へ大きく回したねじれた枝を除く)。
HOVER_SHOULDER_Y_MAX_DEG = 60.0
# 腰ピッチの上限 [deg] (前傾)。
HOVER_WAIST_P_MAX_DEG = 25.0

# --- 横並び・寄せの移動先 (settled_pose / close_gap_pose) ------------------
# 差し出す腕の関節の窓 [deg] (右腕の向き)。
GOAL_ARM_WINDOW_DEG = {
    'shoulder_p': (-60.0, None),
    'shoulder_r': (-60.0, None),
    'elbow': (-100.0, None),
}
GOAL_WRIST_WINDOW_DEG = {
    'wrist_y': (-75.0, 75.0),
    'wrist_r': (-20.0, 75.0),
}
# 脚の前後の傾き |ankle + knee| の上限 [deg] (膝で上体を前後に出さない)。
GOAL_LEG_SUM_MAX_DEG = 45.0

# 右腕と左腕で符号が逆の関節。
MIRRORED_ARM_JOINTS = ('shoulder_r', 'shoulder_y', 'wrist_y', 'wrist_r')


def _arm_window(robot_arm, suffix, window):
    lo, hi = window
    if robot_arm == 'l' and suffix in MIRRORED_ARM_JOINTS:
        lo, hi = (None if hi is None else -hi), (None if lo is None else -lo)
    return lo, hi


def hover_window(robot_arm):
    """hover の IK に掛ける ``{関節名: (下限, 上限) [deg]}``。"""
    window = {}
    if enabled('hover_bend'):
        for suffix, w in HOVER_ARM_WINDOW_DEG.items():
            window['{}_{}_joint'.format(robot_arm, suffix)] = _arm_window(
                robot_arm, suffix, w)
    if enabled('hover_twist'):
        window['{}_shoulder_y_joint'.format(robot_arm)] = _arm_window(
            robot_arm, 'shoulder_y', (None, HOVER_SHOULDER_Y_MAX_DEG))
    if enabled('hover_bow'):
        window['waist_p_joint'] = (None, HOVER_WAIST_P_MAX_DEG)
    return window


def goal_window(robot_arm):
    """横並び・寄せの移動先の IK に掛ける ``{関節名: (下限, 上限) [deg]}``。"""
    window = {}
    if enabled('goal_bend'):
        for suffix, w in GOAL_ARM_WINDOW_DEG.items():
            window['{}_{}_joint'.format(robot_arm, suffix)] = _arm_window(
                robot_arm, suffix, w)
    if enabled('goal_wrist'):
        for suffix, w in GOAL_WRIST_WINDOW_DEG.items():
            window['{}_{}_joint'.format(robot_arm, suffix)] = _arm_window(
                robot_arm, suffix, w)
    return window


@contextlib.contextmanager
def joint_window(robot, window):
    """この間だけ、``window`` の関節の可動域をいまの可動域との共通部分に狭める
    (共通部分が空になる関節は狭めない)。"""
    saved = []
    try:
        for name, (lo, hi) in window.items():
            joint = getattr(robot, name)
            min0, max0 = joint.min_angle, joint.max_angle
            new_min = min0 if lo is None else max(min0, math.radians(lo))
            new_max = max0 if hi is None else min(max0, math.radians(hi))
            if new_min > new_max:
                continue
            saved.append((joint, min0, max0))
            joint.min_angle, joint.max_angle = new_min, new_max
        yield
    finally:
        for joint, min0, max0 in reversed(saved):
            joint.min_angle, joint.max_angle = min0, max0
