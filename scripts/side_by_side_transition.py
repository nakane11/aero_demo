#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""掌を押し込んだ後、つないだ手を人の体の横へ下ろしながら、ロボットも人と
横並びの位置・姿勢へ移る区間 (横並び移動) を計画する。

- 段階 1: つないだ手を人が肩から腕を下ろした位置 (``HumanArm.lowering``) へ
  動かしながら、台車を人と前後ずれ 0・同じ向きの位置へ動かし上体を起こす
  (移動先は台車も動かすバッチ IK、``solve_goal_candidates``)。腕は押し込み時と
  同じ解の枝に限る。waypoint は人の腕・握りの向き・台車を補間して解き直す。
- 上半身 (body_link) は人の正面から ``UPPER_BODY_LIMIT_DEG`` 以内、首は人と
  同じ前方を見る (``forward_gaze_point``)。
- 段階 2: 人の腕・掌を動かさず台車だけを人の方へ寄せる (``plan_approach``)。

人の腕は肩を動かさず、手首から先は押し込み時の手に対して剛体 (``HumanArm``)。
各 waypoint で人の腕が届くか (``reach``) などを記録し、届かなくなる手前で止める。

干渉の検証は人体との距離 (``TRANSITION_HUMAN_CLEARANCE``) と自己干渉。ロボット
の差し出す腕の肘から先と、人の差し出した手・前腕は手を合わせているので距離を
見ない。
"""

import contextlib
import copy
import math
import threading
import time
import zlib

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import normalize_mask
from skrobot.coordinates.math import rpy_matrix

import solve_palm_ik as spik


# 下ろした腕: 上腕を真下から前へ LOWERED_ARM_DEG_GROUPS の角度に上げ、
# 外へ LOWERED_ARM_ABDUCTION_DEG 開き、肘を LOWERED_ELBOW_FLEX_DEG 曲げる。
# LOWERED_ARM_GROUPS は移動先のバッチ IK の組 ((上腕の角度 3 つ), 脇の開き)
# で、腕を低く下ろせる順に試す。脇 25 度は手が胴体の干渉円柱の外に出る角度、
# 近すぎる人は 40 度も試す。
LOWERED_ARM_ABDUCTION_DEG = 25.0
LOWERED_ARM_WIDE_ABDUCTION_DEG = 40.0
LOWERED_ELBOW_FLEX_DEG = 15.0
LOWERED_ARM_DEG_GROUPS = ((0.0, 15.0, 30.0), (45.0, 60.0, 75.0))
LOWERED_ARM_GROUPS = tuple(
    (degs, abduction) for degs in LOWERED_ARM_DEG_GROUPS
    for abduction in (LOWERED_ARM_ABDUCTION_DEG,
                      LOWERED_ARM_WIDE_ABDUCTION_DEG))
LOWERED_PALM_FACINGS = ('inward', 'forward', 'backward')
LOWERED_TARGET_COUNT = 3

# 腕を下ろす途中で上腕を外へ膨らませる角度 [度] (両端の姿勢は変えない)。
# 前腕が人の腰・太ももに近づくのをよける。
LOWERED_BULGE_DEG = 10.0

# 掌の向き 'natural': 前腕をひねらず、腕を下ろす回転だけで動かした向き。
# 終点の掌の法線の外向き成分が LOWERED_NATURAL_MAX_OUTWARD、上下の成分が
# LOWERED_NATURAL_MAX_VERTICAL を超えるときは使わない。
LOWERED_NATURAL_MAX_OUTWARD = 0.3
LOWERED_NATURAL_MAX_VERTICAL = 0.5

# 掌の向きは LOWERED_FACING_ORDER の順に試す (``plan_transition`` の
# ``facing_variants``)。前腕のひねりがこの角度 [度] を超える向きは試さない
# (握手の姿勢から前腕をひねれる範囲は約 ±80〜90 度)。
LOWERED_TWIST_MAX_DEG = 80.0

# 腕を下ろしきったときに押し込み時の手首の曲げを戻す割合 (0: 戻さず
# 剛体、1: 指先が前腕の向き)。下ろす間に線形に戻す。
LOWERED_WRIST_STRAIGHTEN = 1.0

# 前腕のひねりを始める、腕を下ろす動きの割合 (先に下ろして指先を下へ
# 向け、下がった前腕まわりで掌を回す)。
LOWERED_TWIST_START = 0.5

# 腕を下ろしきった掌を、押し込み時の掌よりこの高さ [m] 以上下げる (下がらない
# 下ろし方は試さない)。
LOWERED_MIN_DROP = 0.05

# 横並びの計画を比べるとき、腕を下ろしきった掌が一番低い計画よりこの高さ [m]
# を超えて高い計画は採らない (寄せのために腕を上げさせない)。
LOWERED_HEIGHT_TOLERANCE = 0.02

# 掌の向きの候補を試す順 ('natural' は前腕をひねらない向き)。
LOWERED_FACING_ORDER = ('natural', 'backward', 'inward', 'forward')

# 押し込み時の掌の法線の上向き成分がこの値以上なら、下の順で試す
# (上向きの掌を後ろへ回すと前腕を大きくひねるため)。
PALM_UP_NORMAL_Z = 0.0
LOWERED_FACING_ORDER_PALM_UP = ('forward', 'backward', 'natural', 'inward')

# 移動先の台車の前後位置を、人の立ち位置からずらしてよい量 [m]。
GOAL_X_MARGIN = 0.05

# 移動先のバッチ IK の目標 1 つあたりの初期値の数。
GOAL_ATTEMPTS_PER_POSE = 128

# 移動先のコストの重み: 人の正面方向の前方ずれ [m]、向きのずれ [rad]、
# 上体の前後ずれ |ankle + knee| [rad]。
GOAL_FRONT_OFFSET_WEIGHT = 100.0
GOAL_FACING_YAW_WEIGHT = spik.DEFAULT_FACING_YAW_WEIGHT
GOAL_LEG_OFFSET_WEIGHT = 10.0

# 移動先のコストで、しゃがみ量 ankle + |knee| [rad] に掛ける重み。
GOAL_CROUCH_WEIGHT = 30.0
# 移動先のコストで、腰の前傾 (waist_p が正) [rad] の二乗に掛ける重み。
GOAL_BOW_WEIGHT = 0.0
# 腰 (脚) を押し込み終了時より上げないか: 'strict' は移動先の候補を
# 腰が押し込み終了時以下 (許容 GOAL_LEG_RISE_TOLERANCE) に絞り、経路の脚の IK
# も腰が高くなる側へ動かさない。'fallback' は候補が全滅するときだけ絞らない。
# 'off' は制限しない。
GOAL_LEG_RISE = 'strict'
GOAL_LEG_RISE_MODES = ('off', 'strict', 'fallback')
GOAL_LEG_RISE_TOLERANCE = spik.LEG_RISE_ALLOW  # [m]
# 移動先の候補の腰の超過がこの量 [m] 以下なら、捨てずに脚を押し込み
# 終了時の角度に直す。
GOAL_LEG_RISE_REPAIR_MAX = 0.10
# 経路の各 waypoint で、腰が直前より高くなってよい許容 [m]。
PATH_LEG_RISE_TOLERANCE = spik.LEG_RISE_ALLOW

# 移動先のコストで、差し出す腕の手首ヨー・ロールが可動域の端に近いほど
# 増やす項 (可動域に対する割合が GOAL_WRIST_MARGIN_START を超えた分を、
# GOAL_WRIST_RATIO_RANGE を 1 として 2 乗)。手首ピッチは見ない。
GOAL_WRIST_WEIGHT = 30.0
GOAL_WRIST_MARGIN_START = 0.7
GOAL_WRIST_RATIO_RANGE = 0.3
GOAL_WRIST_AXES = ('y', 'r')

# 移動先のコストで、ロボットの指先 (手先の +Z、``finger_direction``) が
# 下を向いていない分 (1 - 下向きの成分、0〜2) に掛ける重み。
GOAL_ROBOT_FINGER_DOWN_WEIGHT = 30.0

# 指先の鉛直成分 (-1 が真下) がこの値以下の候補があれば、そうでない
# 候補は捨てる。
GOAL_FINGER_DOWN_MAX_Z = -0.5

# 移動先のコストで、差し出す腕の手首の曲げ (ロール・ヨーの |角度| [rad]
# の和) に掛ける重み。
GOAL_WRIST_BEND_WEIGHT = 10.0

# 移動先で試す握りの向き [度] (solve_palm_ik.palm_target_rot の
# turn_deg)。前の組で移動先の IK が 1 つも解けなかったときだけ次の組を試す。
GOAL_TURN_DEG_GROUPS = ((0.0,), (90.0, -90.0))
GOAL_TURN_DEGS = tuple(t for group in GOAL_TURN_DEG_GROUPS for t in group)

# 移動先のバッチ IK の解のうち、差し出す腕の肩ヨー・手首ヨーが押し込み
# 姿勢からこの範囲 [度] を超えるものは捨てる (腕を前に出した枝と体の横に
# 垂らした枝は、手を合わせたまま移れない)。
GOAL_BRANCH_JOINT_RANGE_DEG = 60.0
GOAL_BRANCH_JOINTS = ('{}_shoulder_y_joint', '{}_wrist_y_joint')

# --- 上半身・台車の向き・台車と脚の距離・寄せ (段階 2) ----------------------
# 横並び (と寄せ) の間、上半身 (body_link の +x) を人の正面から
# ±UPPER_BODY_LIMIT_DEG に収める。移動先は範囲外の候補を捨て、ずれの二乗に
# UPPER_BODY_WEIGHT [1/rad^2] を掛けたコストを足す。経路の IK は、始点の
# 上半身が範囲内 (UPPER_BODY_START_SLACK_DEG まで超えていても範囲内とみなす)
# なら全体を、範囲外なら移動先だけを範囲に絞る (押し込み姿勢から出る途中の
# 腰ヨーを絞ると、腕が掌に届かず経路が序盤で止まる)。
UPPER_BODY_LIMIT_DEG = 20.0
UPPER_BODY_WEIGHT = 100.0
UPPER_BODY_START_SLACK_DEG = 3.0

# 横並び移動の計画 (``plan_transition``) の探索の時間の上限 [秒]。超えたら
# それまでの一番良い解 (無ければ横並びしない) で終える。
TRANSITION_TIME_LIMIT = 10.0

# 移動先の台車の向きを人の正面から ±GOAL_YAW_MARGIN_DEG に絞る。
# 寄せ後の横の距離が APPROACH_FALLBACK_LATERAL を超える人だけ
# GOAL_YAW_FALLBACK_DEG まで広げて解き直す (終点は GOAL_YAW_MARGIN_DEG
# に収まるものだけ採る)。None なら広げない。
GOAL_YAW_MARGIN_DEG = 20.0
GOAL_YAW_FALLBACK_DEG = 30.0
APPROACH_FALLBACK_LATERAL = 0.85

# 台車 (車輪の箱) と人の脚などの距離 [m] (ほかの組は
# TRANSITION_HUMAN_CLEARANCE)。人の脚は円柱で足先が入っていないので余裕を見る。
TRANSITION_BASE_CLEARANCE = 0.10

# 人の掌の向きの候補 (``HumanArm.lowering`` の facing): 腕を下ろしきった
# ときの掌の法線を後ろ向き (0 度) から人側 (90 度) へ回した角度 [度] の
# 'phi<角度>'、続けて 'natural' と 'forward'。
# 寄せまで解くとき (``plan_transition`` の approach=True) は向きごとに段階 2
# まで解き、寄せ後の横の距離が APPROACH_STOP_LATERAL [m] 以下で打ち切る。
# 寄せないとき (approach=False) は上腕の角度の組を外側・この順の向きを内側に
# 試し、横並びまで進んだ最初の解で止める。
PALM_FACING_ORDER = ('phi40', 'phi60', 'phi20', 'phi80', 'phi90', 'phi0',
                     'natural', 'forward')
APPROACH_STOP_LATERAL = 0.75

# 寄せ (段階 2): 横並びの終わりから、人の腕・掌を動かさず台車だけを人の
# 方へ寄せる。寄せ後の横の距離 (台車の中心から人の立ち位置までの人の左右方向)
# の目標 [m] の窓 (目標 - 0.1 〜 目標) を近い方から APPROACH_WINDOW_COUNT 個、
# 0.1 m ずつ遠くして試す。
APPROACH_TARGET_LATERAL = 0.6
APPROACH_WINDOW_COUNT = 5
APPROACH_STEP = 0.04  # 台車の 1 waypoint あたりの移動 [m] (直線寄せ)
APPROACH_MIN_GAIN = 0.03  # これ未満しか寄れないなら寄せない [m]
APPROACH_MAX_GOAL_TRIES = 6

# 移動先の候補のうち、経路まで試す数と、押し込み IK・検証まで試す数
# (どちらも目標 1 つあたり)。
MAX_GOAL_CANDIDATES = 2
MAX_GOAL_CHECKS = 8

# 1 人の横並び移動で経路 (plan_path) を計画する回数の上限 (None は無制限)。
MAX_PATH_TRIES = None

# 1 waypoint あたりの台車・掌の移動量の上限 (位置 [m]・向き [rad])。
MAX_STEP = 0.10
MAX_ANGLE_STEP = math.radians(10.0)

# 隣り合う waypoint で関節角 [rad] がこれより大きく変わったら IK が別の
# 解へ飛んだとみなして止める (首は見ない)。
MAX_JOINT_STEP = math.radians(20.0)

# waypoint の数は、押し込み姿勢と移動先の関節角の差がこの角度 [rad]
# ごとに 1 区間以上にもなるようにする。
PLANNED_JOINT_STEP = math.radians(20.0)

# 移動先までのうち進めた割合がこれに満たなければ、横並び移動はしない。
MIN_FRACTION = 0.3

# 人の腕の目安 (表示・ログ用、超えても止めない。reach <= 1 だけ止める)。
HUMAN_ELBOW_FLEX_MAX_DEG = 145.0
HUMAN_WRIST_BEND_MAX_DEG = 70.0
HUMAN_TWIST_CHANGE_MAX_DEG = 90.0

# --- 人の手首 (人の手をロボットの手に固定したモデル) -----------------------
# 人の手はロボットの手について回るとみなし、人の手の向きをロボットの手先の
# 向きから押し込み時の握りの向きで逆算する (``hand_on_robot``)。前腕
# (HumanArm の肘 -> 手首) に対する手の曲げを、掌屈 (+)・背屈 (-)、撓屈 (+)・
# 尺屈 (-)、押し込み時からの前腕のひねりに分ける (``human_wrist_angles``)。
# HUMAN_WRIST_CHECK: 'hand_fixed' はこの角度が目安を超える移動先を捨て、
# 経路はその手前で止める。'turn_cap' は握りの向きの押し込み時からの変化が
# HUMAN_TURN_CAP_DEG を超えるものを同じく扱う (簡易版)。'off' は見ない。
HUMAN_WRIST_CHECKS = ('hand_fixed', 'turn_cap', 'off')
HUMAN_WRIST_CHECK = 'hand_fixed'
HUMAN_WRIST_RADIAL_MAX_DEG = 20.0
HUMAN_WRIST_ULNAR_MAX_DEG = 30.0
HUMAN_WRIST_FLEX_MAX_DEG = 60.0
HUMAN_WRIST_TWIST_MAX_DEG = LOWERED_TWIST_MAX_DEG
HUMAN_TURN_CAP_DEG = 20.0
# 押し込み時にすでにこれ以上曲がっている成分は、押し込み時の値 + この余裕
# [度] までを許す (骨格・掌の推定の手首の曲げは誤差が大きく、押し込み直後で
# 止まらないように)。
HUMAN_WRIST_PRESS_SLACK_DEG = 5.0

# --- 人の前腕・手の甲を障害物にする ----------------------------------------
# 移動先のバッチ IK と検証で、人の前腕 (肘 -> 手首) と手の甲側 (手首 -> 掌 +
# 指先 HUMAN_FINGER_LENGTH を、掌の面から甲側へ HUMAN_HAND_BACK_RADIUS ずらした
# 円柱。掌の接触面は含まない) を、ロボットの差し出す腕の TRANSITION_FOREARM_LINKS
# から OFFERED_HAND_CLEARANCE 離す。手首の評価 (HUMAN_WRIST_CHECK) だけで前腕の
# 近さもほぼ無くなり、こちらは成功数が減り計画時間が延びるので既定では使わない
# (docs/ik_and_motion_constraints.md の 5 節)。
HUMAN_FOREARM_OBSTACLE = False
HUMAN_FINGER_LENGTH = 0.08
HUMAN_HAND_BACK_RADIUS = 0.02
TRANSITION_FOREARM_LINKS = ('{}_elbow_link', '{}_forearm_link',
                            '{}_hand_yaw_link')


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def _angle_between(a, b):
    return math.degrees(math.acos(float(np.clip(
        np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


# --- 人の掌・腕 -------------------------------------------------------

def palm_frame(palm):
    """掌 (``position``/``x_axis`` (指先)/``y_axis`` (法線)) の位置と回転
    行列 (列が指先・法線・指先 x 法線)。"""
    x = _unit(palm['x_axis'])
    n = _unit(np.asarray(palm['y_axis'], dtype=np.float64)
              - np.dot(palm['y_axis'], x) * x)
    return (np.asarray(palm['position'], dtype=np.float64),
            np.column_stack([x, n, np.cross(x, n)]))


def palm_from_frame(position, rot):
    return dict(position=[float(v) for v in position],
                x_axis=[float(v) for v in rot[:, 0]],
                y_axis=[float(v) for v in rot[:, 1]],
                z_axis=[float(v) for v in rot[:, 2]])


def human_palm_from_press(post, turn_deg, robot_arm):
    """押し込み目標 (``post`` の ``target_position``/``target_rot``) から、
    それを作った人の掌を逆算する (``solve_palm_ik.palm_target_rot`` と
    ``POST_PROCESS_TARGET_HOVER_OFFSET`` の逆)。"""
    rot = hand_on_robot(post['target_rot'], turn_deg, robot_arm)
    palm = palm_from_frame(
        np.asarray(post['target_position'], dtype=np.float64)
        - rot[:, 1] * spik.POST_PROCESS_TARGET_HOVER_OFFSET, rot)
    # 最終目標は指先方向へ PRESS_IN_ADVANCE 進めてあるので、その分を戻す。
    palm['position'] = [float(v) for v in (
        np.asarray(palm['position'])
        - spik.press_in_slide_direction(
            palm, np.asarray(post['target_rot'], dtype=np.float64))
        * spik.PRESS_IN_ADVANCE)]
    return palm


def hand_on_robot(robot_rot, turn_deg, robot_arm):
    """ロボットの手先 (``{l,r}arm_end_coords``) の回転行列 ``robot_rot`` に、
    握りの向き ``turn_deg`` で掌を合わせている人の手の回転行列 (列が指先・
    法線・指先 x 法線、``palm_frame`` と同じ)。``palm_target_rot`` の逆。"""
    rot = spik._correct_grasp_frame(
        np.asarray(robot_rot, dtype=np.float64), robot_arm)
    rot = spik._turn_about_y(rot, -turn_deg)
    fingers, normal = rot[:, 0], -rot[:, 1]
    return np.column_stack([fingers, normal, np.cross(fingers, normal)])


def _twist_angle(ref_forearm, ref_normal, forearm, normal):
    """前腕 ``ref_forearm`` で掌の法線が ``ref_normal`` だったときから、前腕
    ``forearm``・法線 ``normal`` までの前腕まわりのひねり [度]。"""
    a0, ref0 = HumanArm._twist_reference(ref_forearm, ref_normal)
    a1, ref1 = HumanArm._twist_reference(forearm, normal)
    moved = _rotation_between(a0, a1) @ ref0
    return math.degrees(math.atan2(float(np.dot(np.cross(moved, ref1), a1)),
                                   float(np.dot(moved, ref1))))


def human_wrist_angles(elbow, wrist, hand_rot, hand, press_forearm,
                       press_normal):
    """前腕 ``elbow`` -> ``wrist`` に対する手 ``hand_rot`` (``palm_frame`` の
    回転行列) の手首の角度 [度] の dict: ``flexion_deg`` (掌屈が正、背屈が
    負)、``deviation_deg`` (撓屈 = 親指側が正、尺屈が負)、``twist_deg``
    (押し込み時の前腕 ``press_forearm``・掌の法線 ``press_normal`` からの
    前腕まわりのひねり)。曲げは指先が前腕の向きのときを 0 とする。"""
    forearm = np.asarray(wrist, dtype=np.float64) - np.asarray(
        elbow, dtype=np.float64)
    a = _unit(forearm)
    hand_rot = np.asarray(hand_rot, dtype=np.float64)
    fingers, normal = hand_rot[:, 0], hand_rot[:, 1]
    # 指先 x 法線は右手では親指側、左手では小指側。
    radial = hand_rot[:, 2] if hand == 'R' else -hand_rot[:, 2]
    along = float(np.dot(a, fingers))
    # 掌屈 (指先が掌の側へ曲がる) では前腕が手から見て甲の側へ傾く。
    return dict(
        flexion_deg=math.degrees(math.atan2(-float(np.dot(a, normal)),
                                            along)),
        deviation_deg=math.degrees(math.atan2(-float(np.dot(a, radial)),
                                              along)),
        twist_deg=_twist_angle(press_forearm, press_normal, forearm, normal))


def wrist_limits(press_angles, s=1.0):
    """押し込み時の手首の角度 ``press_angles`` から、腕を下ろす動きの割合
    ``s`` で許す範囲の dict (``flexion``/``deviation``/``twist`` の (下限,
    上限) [度])。押し込み時にすでに目安を超えている側は、押し込み時の値 +
    ``HUMAN_WRIST_PRESS_SLACK_DEG`` までの超過を、``HumanArm`` が手首の曲げを
    戻すのと同じく ``s`` に比例して減らしながら許す (下ろしきれば目安どおり)。"""
    slack = HUMAN_WRIST_PRESS_SLACK_DEG
    keep = 1.0 - min(1.0, max(0.0, s))

    def bounds(value, lower, upper):
        return (-lower - keep * max(0.0, -lower - (value - slack)),
                upper + keep * max(0.0, value + slack - upper))
    return dict(
        flexion=bounds(press_angles['flexion_deg'], HUMAN_WRIST_FLEX_MAX_DEG,
                       HUMAN_WRIST_FLEX_MAX_DEG),
        deviation=bounds(press_angles['deviation_deg'],
                         HUMAN_WRIST_ULNAR_MAX_DEG,
                         HUMAN_WRIST_RADIAL_MAX_DEG),
        twist=bounds(press_angles['twist_deg'], HUMAN_WRIST_TWIST_MAX_DEG,
                     HUMAN_WRIST_TWIST_MAX_DEG))


def wrist_violation(angles, limits):
    """手首の角度 ``angles`` が範囲 ``limits`` を超えていればその説明、
    超えていなければ None。"""
    labels = dict(flexion=('掌屈', '背屈'), deviation=('撓屈', '尺屈'),
                  twist=('ひねり', 'ひねり'))
    for key, (lo, hi) in limits.items():
        value = angles['{}_deg'.format(key)]
        if value > hi or value < lo:
            return '人の手首の{} {:.0f} 度'.format(
                labels[key][0 if value > 0 else 1], abs(value))
    return None


def hand_back_segment(wrist, palm_pos, palm_rot):
    """人の手の甲側の円柱 (半径 ``HUMAN_HAND_BACK_RADIUS``) の軸の両端: 手首を
    掌の面に移した点から掌 + 指先 ``HUMAN_FINGER_LENGTH`` まで、掌の面から甲側
    へ半径分ずらす (掌の側の面が掌の面に接し、接触面を含まない)。"""
    palm_pos = np.asarray(palm_pos, dtype=np.float64)
    palm_rot = np.asarray(palm_rot, dtype=np.float64)
    fingers, normal = palm_rot[:, 0], palm_rot[:, 1]
    wrist = np.asarray(wrist, dtype=np.float64)
    base = wrist - normal * float(np.dot(wrist - palm_pos, normal))
    tip = palm_pos + fingers * HUMAN_FINGER_LENGTH
    shift = -normal * HUMAN_HAND_BACK_RADIUS
    return base + shift, tip + shift


def forearm_obstacles(elbow, wrist, palm_pos, palm_rot):
    """人の前腕と手の甲側の ``Cylinder`` 2 つ (``HUMAN_FOREARM_OBSTACLE``)。"""
    radius = dict(('{}-{}'.format(a, b), r) for a, b, r
                  in spik.HUMAN_COLLISION_SEGMENTS)['RElbow-RWrist']
    return [spik._cylinder_between(elbow, wrist, radius),
            spik._cylinder_between(
                *hand_back_segment(wrist, palm_pos, palm_rot),
                HUMAN_HAND_BACK_RADIUS)]


def press_target(palm, turn_deg, robot_arm):
    """人の掌 ``palm`` への押し込み目標 (``solve_post_process`` と同じ)。"""
    target_rot = spik.palm_target_rot(palm, turn_deg, robot_arm)
    return Coordinates(
        pos=spik.press_in_final_position(palm, target_rot).tolist(),
        rot=target_rot)


def _outward(joint_positions, hand):
    """人の差し出した手の側の外向き (水平の単位ベクトル)。"""
    other = 'L' if hand == 'R' else 'R'
    v = (np.asarray(joint_positions['{}Shoulder'.format(hand)])
         - np.asarray(joint_positions['{}Shoulder'.format(other)]))
    v[2] = 0.0
    return _unit(v)


class HumanArm(object):
    """人の差し出した腕 (肩は固定) を、掌の位置姿勢に追従させる。

    手 (手首から先) は押し込み時の掌に対して剛体。肘は上腕・前腕の長さを保つ
    2 リンクの IK で、下・少し外を向く側に置く。
    """

    def __init__(self, joint_positions, hand, palm, press_turn=0.0,
                 robot_arm='r'):
        self.hand = hand
        # 押し込み時の握りの向きとロボットの腕 (人の手をロボットの手に固定した
        # ときの手首の角度 ``robot_hand_wrist`` 用)。
        self.press_turn = float(press_turn)
        self.robot_arm = robot_arm
        self.joints = joint_positions
        self.shoulder = np.asarray(
            joint_positions['{}Shoulder'.format(hand)], dtype=np.float64)
        self.elbow0 = np.asarray(
            joint_positions['{}Elbow'.format(hand)], dtype=np.float64)
        self.wrist0 = np.asarray(
            joint_positions['{}Wrist'.format(hand)], dtype=np.float64)
        self.upper = float(np.linalg.norm(self.elbow0 - self.shoulder))
        self.fore = float(np.linalg.norm(self.wrist0 - self.elbow0))
        self.outward = _outward(joint_positions, hand)
        facing = spik.human_facing_direction(joint_positions)
        self.forward = _unit([facing[0], facing[1], 0.0])
        self.palm_pos0, self.palm_rot0 = palm_frame(palm)
        self.hand_prefix = '{}Hand'.format(hand)
        self.twist_ref0 = self._twist_reference(
            self.wrist0 - self.elbow0, self.palm_rot0[:, 1])
        press_elbow = self.elbow_for(self.wrist0)
        self.press_forearm = self.wrist0 - press_elbow
        self.press_wrist = human_wrist_angles(
            press_elbow, self.wrist0, self.palm_rot0, hand,
            self.press_forearm, self.palm_rot0[:, 1])

    def arm_fraction(self, s):
        """経路の割合 ``s`` のときの腕を下ろす動きの割合。"""
        return s

    def hand_wrist(self, arm, hand_rot):
        """腕 ``arm`` (``pose``) で手が ``hand_rot`` のときの手首の角度
        (``human_wrist_angles``)。"""
        return human_wrist_angles(arm['elbow'], arm['wrist'], hand_rot,
                                  self.hand, self.press_forearm,
                                  self.palm_rot0[:, 1])

    def robot_hand_wrist(self, robot, arm):
        """人の手をロボットの手先に固定したときの、腕 ``arm`` の手首の角度。"""
        end = getattr(robot, '{}arm_end_coords'.format(self.robot_arm))
        return self.hand_wrist(arm, hand_on_robot(
            end.worldrot(), self.press_turn, self.robot_arm))

    def _transform(self, palm_pos, palm_rot):
        rot = palm_rot @ self.palm_rot0.T
        return rot, palm_pos - rot @ self.palm_pos0

    def elbow_for(self, wrist):
        return two_link_elbow(self.shoulder, wrist, self.upper, self.fore,
                              self.outward)

    @staticmethod
    def _twist_reference(forearm, normal):
        a = _unit(forearm)
        return a, _unit(normal - np.dot(normal, a) * a)

    def pose(self, palm_pos, palm_rot, elbow=None):
        """掌が ``palm_pos``/``palm_rot`` のときの腕。``elbow``/``wrist``
        と目安 (``reach``: 肩-手首 / 腕の長さ、``arm_down_deg``: 上腕の
        真下からの角度、``elbow_flex_deg``、``wrist_bend_deg``: 前腕と手の
        なす角、``twist_change_deg``: 押し込み時からの前腕まわりのひねりの
        変化) の dict。肘 ``elbow`` を与えなければ ``elbow_for`` で決める。"""
        rot, trans = self._transform(palm_pos, palm_rot)
        wrist = rot @ self.wrist0 + trans
        if elbow is None:
            elbow = self.elbow_for(wrist)
        elbow = np.asarray(elbow, dtype=np.float64)
        a0, ref0 = self.twist_ref0
        a1, ref1 = self._twist_reference(wrist - elbow, palm_rot[:, 1])
        moved = _rotation_between(a0, a1) @ ref0
        twist = math.degrees(math.atan2(
            float(np.dot(np.cross(moved, ref1), a1)),
            float(np.dot(moved, ref1))))
        hand_vec = palm_pos - wrist
        return dict(
            elbow=[float(v) for v in elbow],
            wrist=[float(v) for v in wrist],
            palm_position=[float(v) for v in palm_pos],
            palm_rot=[[float(v) for v in row] for row in palm_rot],
            reach=float(np.linalg.norm(wrist - self.shoulder)
                        / (self.upper + self.fore)),
            arm_down_deg=_angle_between(elbow - self.shoulder, _DOWN),
            elbow_flex_deg=180.0 - _angle_between(
                self.shoulder - elbow, wrist - elbow),
            wrist_bend_deg=_angle_between(wrist - elbow, hand_vec),
            twist_change_deg=twist)

    def skeleton(self, arm):
        """骨格の差し出した腕 (肘・手首・手の landmark) を ``arm``
        (``pose`` の戻り値) に動かしたコピー。"""
        rot, trans = self._transform(
            np.asarray(arm['palm_position']), np.asarray(arm['palm_rot']))
        moved = dict(self.joints)
        moved['{}Elbow'.format(self.hand)] = list(arm['elbow'])
        moved['{}Wrist'.format(self.hand)] = list(arm['wrist'])
        for name, p in self.joints.items():
            if name.startswith(self.hand_prefix):
                moved[name] = (rot @ np.asarray(p) + trans).tolist()
        return moved

    # --- 肩から腕を下ろす (押し込み時の腕 -> 下ろした腕) ---------------
    # 腕は上腕の向き ua と肘の蝶番軸 n の組・肘の曲げ・前腕まわりのひねりで
    # 表す。前腕の向きは cos(曲げ) ua + sin(曲げ) (n x ua)。組は slerp、曲げと
    # ひねりは線形に補間する。

    def _press_arm_frame(self):
        """押し込み時の腕の (上腕の組, 肘の曲げ [rad])。"""
        elbow = self.elbow_for(self.wrist0)
        ua = _unit(elbow - self.shoulder)
        fa = _unit(self.wrist0 - elbow)
        flex = math.acos(float(np.clip(np.dot(ua, fa), -1.0, 1.0)))
        n = np.cross(ua, fa)
        if np.linalg.norm(n) < 1e-3:
            # 肘が伸びきっているときは、腕を前から下へ回す向きの蝶番
            n = self._sagittal_hinge(ua)
        return _arm_frame(ua, _unit(n)), flex

    def _sagittal_hinge(self, ua):
        """上腕 ``ua`` を前から下へ回す面 (前後と上下の面) に垂直な蝶番軸
        (``ua`` に垂直にしたもの)。前腕は上腕より前 (上) へ曲がる。"""
        ref = np.cross(_DOWN, self.forward)
        return _unit(ref - np.dot(ref, ua) * ua)

    def _arm(self, frame, flex, twist, start_frame, start_flex,
             straighten=0.0):
        """上腕の組 ``frame``・肘の曲げ ``flex``・前腕まわりのひねり
        ``twist`` [rad] の腕の (肘, 手首, 掌の位置, 掌の回転行列)。手首から
        先は押し込み時の手を、手首の曲げを割合 ``straighten`` (0: 押し込み
        時のまま、1: 指先が前腕の向き) だけ戻して前腕に付ける。"""
        def forearm(frame_, flex_):
            ua, n = frame_[:, 0], frame_[:, 1]
            fa = math.cos(flex_) * ua + math.sin(flex_) * np.cross(n, ua)
            return fa, _arm_frame(fa, n)

        _, fore0 = forearm(start_frame, start_flex)
        fa, fore = forearm(frame, flex)
        elbow = self.shoulder + self.upper * frame[:, 0]
        wrist = elbow + self.fore * fa
        # 押し込み時の前腕の座標系で、指先を前腕の向き (+x) へ回す回転。
        fingers = fore0.T @ self.palm_rot0[:, 0]
        bend = _rotation_between(fingers, np.array([1.0, 0.0, 0.0]))
        bend = Rotation.from_matrix(bend).as_rotvec() * straighten
        rot = (Rotation.from_rotvec(fa * twist).as_matrix() @ fore
               @ Rotation.from_rotvec(bend).as_matrix() @ fore0.T)
        return (elbow, wrist, wrist + rot @ (self.palm_pos0 - self.wrist0),
                rot @ self.palm_rot0)

    def lowering(self, arm_deg, abduction_deg=None, facing=None):
        """上腕を真下から前へ ``arm_deg`` 度上げ、外へ ``abduction_deg``
        (既定 ``LOWERED_ARM_ABDUCTION_DEG``) 開き、肘を
        ``LOWERED_ELBOW_FLEX_DEG`` 曲げた腕へ下ろす動き。掌は ``facing``
        (既定は前腕のひねりが一番少ない向き、'phi<角度>' なら後ろ向きから
        人側へその角度回した向き) にする。
        ``lowering_pose`` に渡す dict (``palm``: 下ろしきった掌の ``(位置,
        回転行列)`` を含む) を返す。使えない向きなら None。"""
        if abduction_deg is None:
            abduction_deg = LOWERED_ARM_ABDUCTION_DEG
        start_frame, start_flex = self._press_arm_frame()
        th = math.radians(arm_deg)
        ab = math.radians(abduction_deg)
        ua = (math.cos(ab) * (math.cos(th) * _DOWN
                              + math.sin(th) * self.forward)
              + math.sin(ab) * self.outward)
        frame = _arm_frame(ua, self._sagittal_hinge(ua))
        flex = math.radians(LOWERED_ELBOW_FLEX_DEG)
        _, wrist, _, rot = self._arm(frame, flex, 0.0, start_frame,
                                     start_flex, LOWERED_WRIST_STRAIGHTEN)
        fa = _unit(wrist - (self.shoulder + self.upper * ua))
        normal = rot[:, 1]
        facings = dict(inward=-self.outward, forward=self.forward,
                       backward=-self.forward)

        def twist_to(target):
            a = _unit(normal - np.dot(normal, fa) * fa)
            b = _unit(target - np.dot(target, fa) * fa)
            return math.atan2(float(np.dot(np.cross(a, b), fa)),
                              float(np.dot(a, b)))

        twists = {name: twist_to(facings[name])
                  for name in LOWERED_PALM_FACINGS}
        if self.natural_facing_ok(normal):
            twists['natural'] = 0.0
        if isinstance(facing, str) and facing.startswith('phi'):
            # 後ろ向きから人側 (外向きの反対) へ phi 度回した向き。
            phi = math.radians(float(facing[3:]))
            twists[facing] = twist_to(-math.cos(phi) * self.forward
                                      - math.sin(phi) * self.outward)
        if facing is None:
            facing = ('natural' if 'natural' in twists else
                      min(twists, key=lambda name: abs(twists[name])))
        if facing not in twists:
            return None
        goal = dict(arm_deg=float(arm_deg),
                    abduction_deg=float(abduction_deg), palm_facing=facing,
                    start_frame=start_frame, start_flex=start_flex,
                    frame=frame, flex=flex, twist=twists[facing])
        goal['palm'] = self.lowering_pose(goal, 1.0)[2:]
        return goal

    def natural_facing_ok(self, normal):
        """前腕をひねらずに下ろした掌 (法線 ``normal``) を使ってよいか
        (``LOWERED_NATURAL_MAX_OUTWARD``/``LOWERED_NATURAL_MAX_VERTICAL``)。"""
        return (float(np.dot(normal, self.outward))
                <= LOWERED_NATURAL_MAX_OUTWARD
                and abs(float(normal[2])) <= LOWERED_NATURAL_MAX_VERTICAL)

    def lowering_pose(self, goal, s):
        """``lowering`` の動きの割合 ``s`` (0: 押し込み時、1: 下ろしきった
        腕) の (肘, 手首, 掌の位置, 掌の回転行列)。"""
        frame = Slerp([0.0, 1.0], Rotation.from_matrix(
            [goal['start_frame'], goal['frame']]))([s]).as_matrix()[0]
        if LOWERED_BULGE_DEG > 0.0:
            # 途中で上腕を外へ膨らませる (両端は変えない)。
            ua = frame[:, 0] + (math.sin(math.pi * s)
                                * math.tan(math.radians(LOWERED_BULGE_DEG))
                                * self.outward)
            ua = _unit(ua)
            n = frame[:, 1] - np.dot(frame[:, 1], ua) * ua
            frame = _arm_frame(ua, _unit(n))
        flex = goal['start_flex'] + (goal['flex'] - goal['start_flex']) * s
        ramp = min(1.0, max(0.0, (s - LOWERED_TWIST_START)
                            / max(1.0 - LOWERED_TWIST_START, 1e-6)))
        return self._arm(frame, flex, goal['twist'] * ramp,
                         goal['start_frame'], goal['start_flex'],
                         LOWERED_WRIST_STRAIGHTEN * s)


_DOWN = np.array([0.0, 0.0, -1.0])


def _arm_frame(axis, hinge):
    """向き ``axis`` と蝶番軸 ``hinge`` (``axis`` に垂直) の回転行列。"""
    return np.column_stack([axis, hinge, np.cross(axis, hinge)])


def two_link_elbow(shoulder, wrist, upper, fore, outward, prefer=None):
    """肩 ``shoulder`` から手首 ``wrist`` へ、上腕 ``upper``・前腕 ``fore``
    の長さの腕を伸ばしたときの肘の位置。肘が取りうる円のうち、``prefer``
    (肩からの向き、既定は下・少し外 (``outward``)) を向く側を選ぶ。届かない
    ときは手首の方向へ伸ばしきる。"""
    shoulder = np.asarray(shoulder, dtype=np.float64)
    d = np.asarray(wrist, dtype=np.float64) - shoulder
    dist = float(np.linalg.norm(d))
    e1 = d / max(dist, 1e-9)
    reach = min(dist, upper + fore - 1e-6)
    a = (upper ** 2 - fore ** 2 + reach ** 2) / (2.0 * reach)
    r = math.sqrt(max(upper ** 2 - a ** 2, 0.0))
    pref = (np.asarray(prefer, dtype=np.float64) if prefer is not None
            else _DOWN + 0.3 * np.asarray(outward))
    p = pref - np.dot(pref, e1) * e1
    if np.linalg.norm(p) < 1e-6:
        p = np.cross(e1, outward)
    return shoulder + a * e1 + r * _unit(p)


def _rotation_between(a, b):
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-9:
        return np.eye(3)
    return Rotation.from_rotvec(axis / s * math.atan2(s, c)).as_matrix()


# --- ロボット ------------------------------------------------------------

class BasePath(object):
    """台車の ``start`` から ``goal`` (どちらも ``(x, y, yaw)``) への経路。
    ``center`` (人の立ち位置 xy) を与えれば、そのまわりに距離と方位を線形に
    補間して回り込む (与えなければ直線)。yaw は線形に補間する
    (``goal`` の yaw は ``start`` から近い回り方の値にしておく)。"""

    def __init__(self, start, goal, center=None):
        self.start = np.asarray(start, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.center = None
        if center is not None:
            self.center = np.asarray(center, dtype=np.float64)[:2]
            d0 = self.start[:2] - self.center
            d1 = self.goal[:2] - self.center
            self.r0, self.r1 = np.linalg.norm(d0), np.linalg.norm(d1)
            self.a0 = math.atan2(d0[1], d0[0])
            self.da = _wrap(math.atan2(d1[1], d1[0]) - self.a0)

    def at(self, s):
        yaw = self.start[2] + (self.goal[2] - self.start[2]) * s
        if self.center is None:
            xy = self.start[:2] + (self.goal[:2] - self.start[:2]) * s
        else:
            r = self.r0 + (self.r1 - self.r0) * s
            a = self.a0 + self.da * s
            xy = self.center + r * np.array([math.cos(a), math.sin(a)])
        return np.array([xy[0], xy[1], yaw])

    def length(self, samples=20):
        """xy の道のり [m]。"""
        points = [self.at(k / float(samples))[:2] for k in range(samples + 1)]
        return float(sum(np.linalg.norm(b - a)
                         for a, b in zip(points, points[1:])))

    def steps(self):
        """``MAX_STEP``/``MAX_ANGLE_STEP`` を超えない区間の数。"""
        return max(2, int(math.ceil(max(
            self.length() / MAX_STEP,
            abs(self.goal[2] - self.start[2]) / MAX_ANGLE_STEP))))


def _sync_leg_mimics(robot):
    """脚の平行リンクのミミック関節を、主関節 (足首・膝) に合わせ直す。"""
    spik.sync_leg_mimics(robot)


def _place(robot, angle_vector, base):
    robot.angle_vector(np.asarray(angle_vector, dtype=np.float64))
    _sync_leg_mimics(robot)
    robot.newcoords(Coordinates(pos=[float(base[0]), float(base[1]), 0.0],
                                rot=rpy_matrix(float(base[2]), 0.0, 0.0)))


def waist_height(robot):
    """台車 (床面) から腰までの高さ [m] (脚の角度だけで決まる)。"""
    return float(robot.waist_link.worldpos()[2]
                 - robot.base_link.worldpos()[2])


def _waist_height_for_legs(robot, ankle, knee):
    """足首・膝を ``ankle``/``knee`` [rad] にしたときの腰の高さ [m]
    (ロボットの脚の角度も変わる)。"""
    robot.ankle_joint.joint_angle(float(ankle))
    robot.knee_joint.joint_angle(float(knee))
    return waist_height(robot)


def cap_leg_rise(robot, ankle, knee, z_cap):
    """腰の高さが ``z_cap`` 以下になる足首・膝の角度 ``(ankle, knee)`` を、
    与えた角度に一番近いもの (まず膝をより曲げ、それでも足りなければ膝を
    曲げきって足首を上げる) で返す。もともと ``z_cap`` 以下ならそのまま。
    腰は膝を曲げる (knee が小さい)・足首を上げる (ankle が大きい) ほど低い。"""
    if _waist_height_for_legs(robot, ankle, knee) <= z_cap:
        return float(ankle), float(knee)
    knee_min = float(robot.knee_joint.min_angle)
    if _waist_height_for_legs(robot, ankle, knee_min) <= z_cap:
        lo, hi = knee_min, float(knee)  # lo は z_cap 以下、hi は超える
        for _ in range(20):
            mid = (lo + hi) / 2.0
            if _waist_height_for_legs(robot, ankle, mid) <= z_cap:
                lo = mid
            else:
                hi = mid
        return float(ankle), lo
    lo, hi = float(ankle), float(robot.ankle_joint.max_angle)
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if _waist_height_for_legs(robot, mid, knee_min) <= z_cap:
            hi = mid
        else:
            lo = mid
    return hi, knee_min


LEG_JOINT_NAMES = ('ankle_joint', 'ankle_joint_mimic', 'knee_joint',
                   'knee_joint_mimic')


# 握りの向きを自由にするときの手先の回転の拘束 (手先の +Y = 掌の法線
# まわりだけ自由)。
FREE_TURN_ROTATION_MASK = (1, 0, 1)

# 横並び移動の IK で握りの向きを自由にするか (False なら
# GOAL_TURN_DEG_GROUPS の角度に固定)。バッチ IK の回転の拘束が違うので別に
# コンパイルされる (warmup_batch_ik)。
GOAL_FREE_TURN = True


def _rotation_mask(free_turn):
    return normalize_mask(list(FREE_TURN_ROTATION_MASK) if free_turn
                          else True)


def grip_turn_deg(robot, robot_arm, palm):
    """ロボットの手先の今の向きが、人の掌 ``palm`` に対して握りの向き
    (``solve_palm_ik.palm_target_rot`` の ``turn_deg``) 何度にあたるか。"""
    rot = spik._correct_grasp_frame(np.asarray(
        getattr(robot, '{}arm_end_coords'.format(robot_arm)).worldrot()),
        robot_arm)
    base = spik.palm_target_rot(palm, 0.0, 'r')
    x = rot[:, 0]
    return math.degrees(math.atan2(float(np.dot(x, base[:, 2])),
                                   float(np.dot(x, base[:, 0]))))


def finger_direction(rot):
    """手先 (``{l,r}arm_end_coords``) の回転行列 ``rot`` から、ロボットの
    指先の向き (単位ベクトル)。指ありモデルの指の表面点の向きで確かめた
    もので、左右とも手先の +Z (``_correct_grasp_frame`` の +X ではない)。"""
    return np.asarray(rot)[:, 2]


def finger_z(robot, robot_arm):
    """ロボットの指先の鉛直成分 (-1 が真下、+1 が真上)。"""
    return float(finger_direction(
        getattr(robot, '{}arm_end_coords'.format(robot_arm)).worldrot())[2])


def down_turn_deg(palm, robot_arm):
    """人の掌 ``palm`` に掌を合わせたまま、ロボットの指先が一番下を向く握りの
    向き (``press_target`` の ``turn_deg`` [度]、1 度刻みで探す)。"""
    turns = np.arange(-180.0, 180.0)
    zs = [finger_direction(spik.palm_target_rot(palm, t, robot_arm))[2]
          for t in turns]
    return float(turns[int(np.argmin(zs))])


class _UpperBody(threading.local):
    """横並び移動の IK の上半身の向きの制約の状態 (スレッドごと)。
    ``human_yaw`` が None なら制約なし。``path`` は経路の IK 中だけ
    ``(始点の腰ヨー, 移動先の腰ヨー, 始点の上半身が範囲内か)``、``s`` は
    経路の割合。``_upper_body_limit`` で設定する。"""
    human_yaw = None
    path = None
    s = 0.0


_UPPER = _UpperBody()
_WAIST_SIGNS = {}


@contextlib.contextmanager
def _upper_body_limit(human_yaw):
    """この間の IK を、上半身 (body_link) の向きが人の正面 (``human_yaw``
    [rad]) から ``UPPER_BODY_LIMIT_DEG`` に収まるように制限する。"""
    before = (_UPPER.human_yaw, _UPPER.path)
    _UPPER.human_yaw, _UPPER.path = human_yaw, None
    try:
        yield
    finally:
        _UPPER.human_yaw, _UPPER.path = before


def upper_body_deviation(robot, human_yaw):
    """上半身 (body_link の +x) の向きの、人の正面 ``human_yaw`` [rad] からの
    ずれ [rad] (反時計回りが正)。"""
    x = robot.body_link.worldrot()[:, 0]
    return _wrap(math.atan2(x[1], x[0]) - human_yaw)


def _waist_sign(robot):
    """腰ヨーを増やすと上半身の向きが増える向きなら +1、減るなら -1。"""
    key = id(robot)
    if key not in _WAIST_SIGNS:
        before = robot.angle_vector().copy()
        d0 = upper_body_deviation(robot, 0.0)
        robot.waist_y_joint.joint_angle(
            float(robot.waist_y_joint.joint_angle()) + 0.1)
        d1 = upper_body_deviation(robot, 0.0)
        robot.angle_vector(before)
        _WAIST_SIGNS[key] = 1.0 if _wrap(d1 - d0) > 0.0 else -1.0
    return _WAIST_SIGNS[key]


def _restrict_waist_for_upper_body(robot):
    """上半身の向きを範囲内にするよう腰ヨーの可動域を絞り、元に戻す関数を
    返す (``_UPPER.human_yaw`` が None なら何もしない)。経路 (``_UPPER.path``)
    では、始点の上半身が範囲内なら全体を、範囲外なら移動先 (割合 1) だけを
    絞る。"""
    if _UPPER.human_yaw is None:
        return lambda: None
    joint = robot.waist_y_joint
    lo0, hi0 = joint.min_angle, joint.max_angle
    limit = math.radians(UPPER_BODY_LIMIT_DEG)
    w = float(joint.joint_angle())
    cur = upper_body_deviation(robot, _UPPER.human_yaw)
    sign = _waist_sign(robot)
    a = w + sign * (-limit - cur)
    b = w + sign * (limit - cur)
    torso_lo, torso_hi = min(a, b), max(a, b)
    if _UPPER.path is None:
        # 経路の外: 範囲内にするか、直前のずれより悪くしない
        lo, hi = min(torso_lo, w), max(torso_hi, w)
    else:
        start_ok = _UPPER.path[2]
        lo, hi = lo0, hi0
        if start_ok or _UPPER.s >= 1.0 - 1e-9:
            lo, hi = torso_lo, torso_hi
    joint.min_angle = max(lo0, lo - 1e-3)
    joint.max_angle = min(hi0, hi + 1e-3)
    if _UPPER.path is not None:
        # 初期値を範囲内に切り詰める (範囲は広げない)
        joint.joint_angle(min(max(w, joint.min_angle), joint.max_angle))

    def restore():
        joint.min_angle, joint.max_angle = lo0, hi0
    return restore


def _press_ik(robot, robot_arm, target, gaze_point, free_legs=False,
              free_turn=False, leg_floor=None):
    """``_press_ik_core`` を、上半身の向きの制約 (``_upper_body_limit`` の
    間) つきで解く。解けなければロボットは呼び出し前の姿勢に戻る。"""
    before = robot.angle_vector().copy()
    restore = _restrict_waist_for_upper_body(robot)
    try:
        solved = _press_ik_core(robot, robot_arm, target, gaze_point,
                                free_legs=free_legs, free_turn=free_turn,
                                leg_floor=leg_floor)
    finally:
        restore()
    if not solved and _UPPER.human_yaw is not None:
        robot.angle_vector(before)
        _sync_leg_mimics(robot)
    return solved


def _press_ik_core(robot, robot_arm, target, gaze_point, free_legs=False,
                   free_turn=False, leg_floor=None):
    """台車をいまの位置に固定し、腕・腰を押し込み目標 ``target`` に、カメラ
    の光軸を ``gaze_point`` に向ける (``solve_palm_ik.solve_post_process``
    と同じ IK)。脚 (``LEG_JOINT_NAMES``) は ``free_legs`` でなければ固定する。
    ``free_turn`` なら握りの向きは拘束しない。視線が解けなければ首はそのまま
    で腕だけを解く (``gaze_point`` が None なら視線は解かない)。腕が解けなけ
    れば ``False`` (呼び出し前の姿勢に戻る)。

    ``free_legs`` のときは IK が脚のミミック関節を食い違わせるので、ミミックを
    主関節に合わせてから、脚を固定して腕だけ解き直す。

    ``leg_floor`` (足首, 膝の角度 [rad]) を与えると、``free_legs`` の IK 中は
    足首の下限・膝の上限をその角度にして、腰が高くなる側へ脚を動かさない。"""
    if free_legs:
        before = robot.angle_vector().copy()
        limits = spik._leg_limits(robot)
        if leg_floor is not None:
            robot.ankle_joint.min_angle = max(
                limits[0][0], float(leg_floor[0]) - spik.LEG_RISE_PIN_TOLERANCE)
            robot.knee_joint.max_angle = min(
                limits[1][1], float(leg_floor[1]) + spik.LEG_RISE_PIN_TOLERANCE)
        try:
            solved = _press_ik_once(robot, robot_arm, target, gaze_point,
                                    free_legs=True, free_turn=free_turn)
        finally:
            spik._restore_leg_limits(robot, limits)
        if not solved:
            return False
        _sync_leg_mimics(robot)
        if _press_ik_once(robot, robot_arm, target, gaze_point,
                          free_legs=False, free_turn=free_turn):
            return True
        robot.angle_vector(before)
        _sync_leg_mimics(robot)
        return False
    return _press_ik_once(robot, robot_arm, target, gaze_point,
                          free_legs=False, free_turn=free_turn)


def _press_ik_once(robot, robot_arm, target, gaze_point, free_legs=False,
                   free_turn=False):
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    link_list = [link for link in whole_body.link_list
                 if free_legs or link.joint.name not in LEG_JOINT_NAMES]
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    stop = spik.DEFAULT_POST_PROCESS_IK_STOP
    thre = spik.DEFAULT_POST_PROCESS_IK_THRE
    rthre = spik.DEFAULT_POST_PROCESS_IK_RTHRE
    gaze_rthre = spik.DEFAULT_POST_PROCESS_GAZE_IK_RTHRE
    hand_mask = _rotation_mask(free_turn)
    gaze_links = spik.gaze_link_list(robot) if gaze_point is not None else None
    if gaze_point is None:
        result = robot.inverse_kinematics(
            target, move_target=move_target, link_list=link_list,
            rotation_mask=hand_mask,
            stop=stop, thre=thre, rthre=rthre, revert_if_fail=True)
        return result is not False
    gaze_coords = spik.camera_optical_coords(robot)
    result = robot.inverse_kinematics(
        target_coords=[target, spik._gaze_target(gaze_coords, gaze_point)],
        move_target=[move_target, gaze_coords],
        link_list=[link_list, gaze_links],
        position_mask=[normalize_mask(True), normalize_mask(False)],
        rotation_mask=[hand_mask, normalize_mask('xy')],
        stop=stop, thre=[thre, thre], rthre=[rthre, gaze_rthre],
        revert_if_fail=True)
    if result is not False:
        spik._reaim_gaze(robot, gaze_coords, gaze_point, stop, gaze_rthre)
        return True
    result = robot.inverse_kinematics(
        target, move_target=move_target, link_list=link_list,
        rotation_mask=hand_mask,
        stop=stop, thre=thre, rthre=rthre, revert_if_fail=True)
    return result is not False


# 首だけで視線を向け直す IK の最大繰り返し回数 (_aim_head)。
AIM_HEAD_ROUNDS = 10


def _aim_head(robot, gaze_point):
    """首だけで、カメラの光軸を ``gaze_point`` へ向ける。今の首の角度と首 0 度
    の両方から解き、視線のずれの小さい方にする。視線のずれ [rad] を返す。"""
    gaze_coords = spik.camera_optical_coords(robot)
    joints = [link.joint for link in spik.gaze_link_list(robot)]
    best = None
    for start in ([joint.joint_angle() for joint in joints],
                  [0.0] * len(joints)):
        for joint, angle in zip(joints, start):
            joint.joint_angle(angle)
        spik._reaim_gaze(robot, gaze_coords, gaze_point,
                         spik.DEFAULT_POST_PROCESS_IK_STOP,
                         spik.DEFAULT_POST_PROCESS_GAZE_IK_RTHRE,
                         rounds=AIM_HEAD_ROUNDS)
        error = spik._gaze_error(gaze_coords, gaze_point)
        if best is None or error < best[0]:
            best = (error, [joint.joint_angle() for joint in joints])
    for joint, angle in zip(joints, best[1]):
        joint.joint_angle(angle)
    return best[0]


def _below_elbow(link, robot_arm):
    """``link`` が ``robot_arm`` の腕の肘から先 (手・指を含む) か。"""
    elbow = '{}_elbow_joint'.format(robot_arm)
    while link is not None:
        joint = getattr(link, 'joint', None)
        if joint is not None and joint.name == elbow:
            return True
        link = getattr(link, 'parent_link', None)
    return False


# 横並び移動の検証で人体 (車輪と脚を含む) から空ける距離 [m]。
TRANSITION_HUMAN_CLEARANCE = 0.05


class TransitionChecker(object):
    """横並び移動の各姿勢の干渉検証 (自己干渉の貫通・人体からの距離)。
    ロボットの差し出す腕の肘から先と、人の差し出した手・前腕
    (``offered_hand_obstacle_indices``) の組は、手を合わせているので
    見ない。"""

    def __init__(self, robot_arm, hand, verification_pairs,
                 clearance=TRANSITION_HUMAN_CLEARANCE,
                 tolerance=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE,
                 base_clearance=TRANSITION_BASE_CLEARANCE):
        self.clearance = clearance
        self.base_clearance = max(base_clearance, clearance)
        self.tolerance = tolerance
        self.self_pairs = spik.self_collision_pairs(verification_pairs)
        offered = spik.offered_hand_obstacle_indices(hand)
        source = spik.human_clearance_pairs(verification_pairs)
        self.clearance_pairs = None
        if source is not None:
            pairs = [pair for pair in source
                     if not (isinstance(pair[1], int) and pair[1] in offered
                             and _below_elbow(pair[0], robot_arm))]
            self.clearance_pairs = (
                source.with_pairs(pairs)
                if isinstance(source, spik.VerificationPairs) else pairs)
        # 台車の車輪と人の脚などの組 (人の腕は動くので除く)。
        self.base_pairs = None
        if source is not None:
            arm = offered | set(_lowered_arm_slots(hand))
            pairs = [pair for pair in source
                     if isinstance(pair[1], int) and pair[1] not in arm
                     and pair[0].name.startswith('wheel')]
            self.base_pairs = (
                source.with_pairs(pairs)
                if isinstance(source, spik.VerificationPairs) else pairs)
        # ロボットの差し出す腕の前腕・肘・手首と、人の前腕 (0)・手の甲 (1)
        # (``forearm_obstacles``) の組。
        self.forearm_pairs = None
        if source is not None:
            names = {name.format(robot_arm)
                     for name in TRANSITION_FOREARM_LINKS}
            links = []
            for pair in source:
                if (isinstance(pair[1], int) and pair[0].name in names
                        and pair[0] not in links):
                    links.append(pair[0])
            pairs = [(link, i) for link in links for i in (0, 1)]
            self.forearm_pairs = (
                source.with_pairs(pairs)
                if isinstance(source, spik.VerificationPairs) else pairs)

    def forearm_clearances(self, robot, arm,
                           cull_distance=spik.OFFERED_HAND_CLEARANCE):
        """ロボットの差し出す腕の前腕・肘・手首から、人の腕 ``arm``
        (``HumanArm.pose``) の前腕・手の甲までの距離 [m] の (前腕, 手の甲)。
        ``cull_distance`` 以上の距離は下界 (``human_obstacle_clearances``)。"""
        if not self.forearm_pairs:
            return None
        clearances = spik.human_obstacle_clearances(
            robot, self.forearm_pairs, forearm_obstacles(
                arm['elbow'], arm['wrist'], arm['palm_position'],
                arm['palm_rot']),
            cull_distance=cull_distance)
        return (clearances.get(0, float('inf')),
                clearances.get(1, float('inf')))

    def check_base_path(self, robot, av, base_path, joint_positions):
        """台車を経路 ``base_path`` (``BasePath``、``plan_path`` と同じ) で
        動かすと、車輪が人体の脚などに ``clearance`` 未満まで近づけば説明、
        無ければ ``None`` を返す。``av`` は台車以外の姿勢 (車輪は台車の
        位置だけで決まる)。"""
        if not self.base_pairs:
            return None
        n = base_path.steps()
        obstacles = spik.human_body_obstacles(joint_positions)
        names = spik.human_obstacle_names()
        for k in range(1, n + 1):
            _place(robot, av, base_path.at(k / float(n)))
            clearances = spik.human_obstacle_clearances(
                robot, self.base_pairs, obstacles,
                cull_distance=self.base_clearance)
            if not clearances:
                continue
            index = min(clearances, key=clearances.get)
            if clearances[index] < self.base_clearance:
                return '経路の車輪が人体 ({}) まで {:.3f} m'.format(
                    names[index], clearances[index])
        return None

    def check(self, robot, joint_positions, arm=None):
        """``robot`` の今の姿勢を、人の骨格 ``joint_positions`` に対して
        検証し、問題があればその説明、無ければ ``None`` を返す。人の腕
        ``arm`` (``HumanArm.pose``) を与え ``HUMAN_FOREARM_OBSTACLE`` なら、
        前腕・手の甲からも ``OFFERED_HAND_CLEARANCE`` 離れているかを見る。"""
        dist, pair = spik.collision_pairs_min_distance(
            robot, self.self_pairs, None, return_pair=True)
        if dist < -self.tolerance:
            return '自己干渉 {:.3f} m ({} x {})'.format(
                -dist, *spik.collision_pair_name(pair))
        if arm is not None and HUMAN_FOREARM_OBSTACLE:
            clearances = self.forearm_clearances(robot, arm)
            if clearances is not None:
                for label, value in zip(('前腕', '手の甲'), clearances):
                    if value < spik.OFFERED_HAND_CLEARANCE:
                        return '人の{}まで {:.3f} m'.format(label, value)
        if not joint_positions or not self.clearance_pairs:
            return None
        clearances = spik.human_obstacle_clearances(
            robot, self.clearance_pairs,
            spik.human_body_obstacles(joint_positions),
            cull_distance=self.clearance)
        if clearances:
            index = min(clearances, key=clearances.get)
            if clearances[index] < self.clearance:
                return '人体 ({}) まで {:.3f} m'.format(
                    spik.human_obstacle_names()[index], clearances[index])
        if self.base_pairs and self.base_clearance > self.clearance:
            clearances = spik.human_obstacle_clearances(
                robot, self.base_pairs,
                spik.human_body_obstacles(joint_positions),
                cull_distance=self.base_clearance)
            if clearances:
                index = min(clearances, key=clearances.get)
                if clearances[index] < self.base_clearance:
                    return '台車が人体 ({}) まで {:.3f} m'.format(
                        spik.human_obstacle_names()[index],
                        clearances[index])
        return None


def wrist_problem(human_arm, robot, arm, s, turn_deg):
    """人の手をロボットの手に固定したときの手首の角度 (``robot_hand_wrist``)
    と、``HUMAN_WRIST_CHECK`` で目安を超えていればその説明 (無ければ None)
    の組。``s`` は腕を下ろす動きの割合、``turn_deg`` は今の握りの向き。"""
    angles = human_arm.robot_hand_wrist(robot, arm)
    if HUMAN_WRIST_CHECK == 'hand_fixed':
        return angles, wrist_violation(
            angles, wrist_limits(human_arm.press_wrist, s))
    if HUMAN_WRIST_CHECK == 'turn_cap':
        change = math.degrees(_wrap(math.radians(
            turn_deg - human_arm.press_turn)))
        if abs(change) > HUMAN_TURN_CAP_DEG:
            return angles, '握りの向きの変化 {:.0f} 度'.format(abs(change))
    return angles, None


# 移動先の握りの向きを人の手首の目安に収めて解き直すとき、目安の端からさらに
# 内側へ寄せる角度 [度] (IK の回転の許容 5 度で端を超えないように)。
WRIST_TURN_MARGIN_DEG = 5.0


def wrist_ok_turn(human_arm, arm, palm_pos, palm_rot, turn_deg, s=1.0):
    """掌 ``palm_pos``/``palm_rot`` に合わせたまま、人の手首が目安に収まる
    握りの向きのうち ``turn_deg`` に一番近いもの (収まる範囲の端から
    ``WRIST_TURN_MARGIN_DEG`` 内側、収まらなければ端) [度]。無ければ None。"""
    palm = palm_from_frame(palm_pos, palm_rot)
    limits = wrist_limits(human_arm.press_wrist, s)

    def ok(turn):
        if HUMAN_WRIST_CHECK == 'turn_cap':
            return abs(math.degrees(_wrap(math.radians(
                turn - human_arm.press_turn)))) <= HUMAN_TURN_CAP_DEG
        rot = hand_on_robot(spik.palm_target_rot(palm, turn,
                                                 human_arm.robot_arm),
                            human_arm.press_turn, human_arm.robot_arm)
        return wrist_violation(human_arm.hand_wrist(arm, rot), limits) is None

    for delta in sorted(range(-180, 181), key=abs):
        if ok(turn_deg + delta):
            inner = turn_deg + delta + math.copysign(WRIST_TURN_MARGIN_DEG,
                                                     delta)
            return inner if delta and ok(inner) else turn_deg + delta
    return None


def fix_goal_wrist(robot, robot_arm, human_arm, arm, palm_pos, palm_rot,
                   gaze_point, turn_deg):
    """移動先の姿勢 (``robot`` の今の姿勢、握りの向き ``turn_deg``) で人の
    手首が目安を超えていれば、収まる握りの向き (``wrist_ok_turn``) に固定して
    解き直す。``(握りの向き, 問題の説明 or None)`` を返す (直せなければ
    姿勢は元のまま)。"""
    if HUMAN_WRIST_CHECK == 'off':
        return turn_deg, None
    _, problem = wrist_problem(human_arm, robot, arm, 1.0, turn_deg)
    if problem is None:
        return turn_deg, None
    turn = wrist_ok_turn(human_arm, arm, palm_pos, palm_rot, turn_deg)
    if turn is None:
        return turn_deg, problem
    before = robot.angle_vector().copy()
    palm = palm_from_frame(palm_pos, palm_rot)
    if _press_ik(robot, robot_arm, press_target(palm, turn, robot_arm),
                 gaze_point):
        if gaze_point is not None:
            _aim_head(robot, gaze_point)
        turn = grip_turn_deg(robot, robot_arm, palm)
        if wrist_problem(human_arm, robot, arm, 1.0, turn)[1] is None:
            return turn, None
    robot.angle_vector(before)
    _sync_leg_mimics(robot)
    return turn_deg, problem


def _branch_joint_indices(robot, robot_arm):
    """差し出す腕の ``GOAL_BRANCH_JOINTS`` の ``joint_list`` での添字。"""
    names = [joint.name for joint in robot.joint_list]
    return [names.index(name.format(robot_arm)) for name in GOAL_BRANCH_JOINTS
            if name.format(robot_arm) in names]


# 移動先のバッチ IK に、押し込みの干渉ペアへ足す組 (_transition_pairs):
# ロボットの差し出す腕の前腕・肘と人の下ろした上腕・胴体、差し出さない腕の
# 前腕・手と腰の自己干渉。
TRANSITION_ARM_LINKS = ('{}_forearm_link', '{}_elbow_link')
TRANSITION_SELF_PAIRS = (
    ('{other}_forearm_link', 'hip_sphere_link'),
    ('{other}_forearm_link', 'waist_link'),
    ('{other}_hand_link', 'waist_link'),
    ('{arm}_forearm_link', 'waist_link'),
    ('{arm}_hand_link', 'waist_link'),
    ('{arm}_forearm_link', 'body_link'),
    ('{arm}_elbow_link', 'body_link'),
)


def _lowered_arm_slots(hand):
    """``_batch_ik`` で、下ろした人の上腕 (目標ごとに 1 本) を置く障害物の
    添字 (差し出した腕・手の枠を使い回す。個数は変えない)。"""
    names = spik.human_obstacle_names()
    return [names.index(name.format(hand)) for name in
            ('{0}Shoulder-{0}Elbow', '{0}Elbow-{0}Wrist', '{0}_palm')]


def _transition_pairs(robot, robot_arm, hand):
    """横並び移動の移動先のバッチ IK に、押し込みの干渉ペアへ足す組。"""
    other = 'l' if robot_arm == 'r' else 'r'
    links_by_name = {link.name: link for link in
                     list(robot.link_list)
                     + list(getattr(robot, 'extra_collision_links', []))}
    torso = spik.human_obstacle_names().index(
        '{0}Shoulder-{0}Hip'.format(hand))
    pairs = []
    for name in TRANSITION_ARM_LINKS:
        link = links_by_name.get(name.format(robot_arm))
        if link is not None:
            pairs += [(link, i) for i in _lowered_arm_slots(hand) + [torso]]
    for a, b in TRANSITION_SELF_PAIRS:
        a = links_by_name.get(a.format(arm=robot_arm, other=other))
        b = links_by_name.get(b.format(arm=robot_arm, other=other))
        if a is not None and b is not None:
            pairs.append((a, b))
    return pairs


def _forearm_obstacle_slots(obstacles, lowered_forearms):
    """``HUMAN_FOREARM_OBSTACLE`` なら、障害物 ``obstacles`` の後ろに人の
    前腕・手の甲 (目標ごとに 2 本、``forearm_obstacles``) を足し、足した添字の
    リストを返す。距離が ``OFFERED_HAND_CLEARANCE`` になるよう細らせ、無い
    目標の分はダミーで埋める (個数は常に ``2 * LOWERED_TARGET_COUNT``)。"""
    if not HUMAN_FOREARM_OBSTACLE:
        return []
    start = len(obstacles)
    for k in range(LOWERED_TARGET_COUNT):
        if k < len(lowered_forearms):
            obstacles += [
                spik.offered_hand_penalty_obstacle(
                    o, spik.DEFAULT_COLLISION_MARGIN)
                for o in forearm_obstacles(*lowered_forearms[k])]
        else:
            obstacles += [spik._dummy_cylinder(0.04),
                          spik._dummy_cylinder(HUMAN_HAND_BACK_RADIUS)]
    return list(range(start, len(obstacles)))


def _batch_ik(robot, robot_arm, hand, targets, seed_av, collision_joints,
              collision_pairs, base_limits, attempts_per_pose,
              free_turn=False, lowered_arms=(), lowered_forearms=()):
    """押し込み目標 ``targets`` (3 つ) に対して台車も動かす干渉回避付き
    バッチ IK (``solve_person_ik`` と同じ形: 目標 3 つ・初期値・干渉ペア)。
    人の差し出した手・前腕・上腕は手を合わせているので障害物から外す
    (遠くのダミーにする、個数は変えない)。代わりに、下ろした人の上腕
    ``lowered_arms`` (目標ごとの (肩, 肘)) をその枠に置き、ロボットの
    差し出す腕の前腕・肘に避けさせる (``_transition_pairs``)。差し出さない
    腕・首の初期値は ``seed_av``。収束した解 ``[(目標の添字,
    angle_vector, (x, y, yaw)), ...]`` を返す。"""
    obstacles = spik.human_body_obstacles(collision_joints)
    names = spik.human_obstacle_names()
    removed = spik.offered_hand_obstacle_indices(hand) | {
        names.index('{0}Shoulder-{0}Elbow'.format(hand))}
    obstacles = [spik._dummy_cylinder(o.radius) if i in removed else o
                 for i, o in enumerate(obstacles)]
    upper_radius = dict(
        ('{}-{}'.format(a, b), r) for a, b, r in spik.HUMAN_COLLISION_SEGMENTS
    )['{0}Shoulder-{0}Elbow'.format(hand)]
    for slot, (shoulder, elbow) in zip(_lowered_arm_slots(hand),
                                       lowered_arms):
        obstacles[slot] = spik._cylinder_between(shoulder, elbow,
                                                 upper_radius)
    # 下ろした人の前腕・手の甲 (``lowered_forearms``、目標ごとの (肘, 手首,
    # 掌の位置, 掌の回転行列)) を後ろに足し、ロボットの差し出す腕の前腕・肘・
    # 手首に避けさせる。
    forearm_slots = _forearm_obstacle_slots(obstacles, lowered_forearms)
    pairs = None
    if collision_pairs:
        links_by_name = {link.name: link for link in
                         list(robot.link_list)
                         + list(getattr(robot, 'extra_collision_links', []))}
        offered = sorted(spik.offered_hand_obstacle_indices(hand))
        pairs = list(collision_pairs) + [
            (links_by_name[name.format(robot_arm)], i)
            for name in spik.OFFERED_HAND_PENALTY_LINKS
            if name.format(robot_arm) in links_by_name for i in offered
            if i not in _lowered_arm_slots(hand)]
        pairs += _transition_pairs(robot, robot_arm, hand)
        pairs += [(links_by_name[name.format(robot_arm)], i)
                  for name in TRANSITION_FOREARM_LINKS
                  if name.format(robot_arm) in links_by_name
                  for i in forearm_slots]
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    # バッチ IK は台車をワールド原点から動かす (base_limits もその前提の
    # 絶対座標)。
    _place(robot, seed_av, (0.0, 0.0, 0.0))
    restore = spik.restrict_joint_range_margin(
        whole_body.link_list,
        spik.DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO)
    try:
        angle_vectors, base_poses, success_flags, _ = \
            robot.batch_inverse_kinematics(
                target_coords=list(targets),
                move_target=getattr(robot,
                                    '{}arm_end_coords'.format(robot_arm)),
                link_list=whole_body.link_list,
                position_mask=True,
                rotation_mask=(list(FREE_TURN_ROTATION_MASK) if free_turn
                               else True),
                stop=spik.DEFAULT_COLLISION_IK_STOP,
                thre=spik.DEFAULT_COLLISION_IK_THRE,
                rthre=spik.DEFAULT_COLLISION_IK_RTHRE,
                initial_angles='current',
                attempts_per_pose=attempts_per_pose,
                return_all_attempts=True,
                backend='jax',
                use_base='planar', base_limits=base_limits,
                collision_obstacles=obstacles,
                collision_weight=spik.DEFAULT_COLLISION_WEIGHT,
                collision_margin=spik.DEFAULT_COLLISION_MARGIN,
                self_collision=pairs is not None,
                collision_pairs=pairs,
                self_collision_margin=spik.DEFAULT_SELF_COLLISION_MARGIN,
                collision_geometry=spik.DEFAULT_IK_COLLISION_GEOMETRY)
    finally:
        restore()
    solutions = []
    for index, ok in enumerate(success_flags):
        if not ok:
            continue
        base_pose = base_poses[index]
        rot = np.asarray(base_pose.worldrot())
        solutions.append((
            index // attempts_per_pose,
            np.asarray(angle_vectors[index], dtype=np.float64),
            (float(base_pose.worldpos()[0]), float(base_pose.worldpos()[1]),
             math.atan2(rot[1, 0], rot[0, 0]))))
    return solutions


def warmup_batch_ik(robot, robot_arm, hand, seed_av, collision_pairs,
                    base_limits,
                    attempts_per_pose=GOAL_ATTEMPTS_PER_POSE):
    """移動先のバッチ IK (``_batch_ik``) を、実際の呼び出しと同じ形で 1 回
    解いて、JAX のトレース・コンパイルを前倒しで済ませる。``GOAL_FREE_TURN``
    が False なら何もしない。``seed_av`` は解ける目標を作るための姿勢。"""
    if not GOAL_FREE_TURN:
        return
    _place(robot, seed_av, (0.0, 0.0, 0.0))
    end = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    target = Coordinates(pos=end.worldpos().copy(), rot=end.worldrot().copy())
    _batch_ik(robot, robot_arm, hand, [target] * LOWERED_TARGET_COUNT,
              seed_av, {}, collision_pairs, base_limits, attempts_per_pose,
              free_turn=True)


class _rotation_about(object):
    """``center_xy`` を通る鉛直軸まわりに ``theta`` 回す変換。"""

    def __init__(self, theta, center_xy):
        c, s = math.cos(theta), math.sin(theta)
        self.rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.center = np.array([center_xy[0], center_xy[1], 0.0])

    def transform_point(self, point):
        point = np.asarray(point, dtype=np.float64)
        return (self.rot @ (point - self.center) + self.center).tolist()


def placement_cost(base, standing_xy, facing):
    """台車 ``base`` ``(x, y, yaw)`` の横並びからのずれのコスト (前方ずれ
    [m] と向きのずれ [rad] の絶対値の重み付き和)。"""
    front = float(np.dot(np.asarray(base[:2]) - standing_xy, facing))
    yaw = _wrap(base[2] - math.atan2(facing[1], facing[0]))
    return GOAL_FRONT_OFFSET_WEIGHT * abs(front) \
        + GOAL_FACING_YAW_WEIGHT * abs(yaw)


def _wrist_limit_table(robot, robot_arm, names):
    """差し出す腕の手首 (``GOAL_WRIST_AXES``) の ``[(添字, 正の側の可動域,
    負の側の可動域)]`` [rad]。"""
    table = []
    for axis in GOAL_WRIST_AXES:
        name = '{}_wrist_{}_joint'.format(robot_arm, axis)
        if name in names:
            joint = robot.joint_list[names.index(name)]
            table.append((names.index(name), max(float(joint.max_angle), 1e-3),
                          max(-float(joint.min_angle), 1e-3)))
    return table


def wrist_limit_penalty(av, table):
    """手首が可動域の端に近いほど大きい値 (``GOAL_WRIST_WEIGHT`` を掛ける
    前)。"""
    total = 0.0
    for index, upper, lower in table:
        ratio = av[index] / upper if av[index] >= 0.0 else -av[index] / lower
        total += (max(0.0, ratio - GOAL_WRIST_MARGIN_START)
                  / GOAL_WRIST_RATIO_RANGE) ** 2
    return total


def solve_goal_candidates(robot, robot_arm, hand, goal_palms, seed_av,
                          translated_joints, collision_joints,
                          collision_pairs, base_limits,
                          turn_degs=GOAL_TURN_DEGS,
                          attempts_per_pose=GOAL_ATTEMPTS_PER_POSE,
                          branch_range_deg=GOAL_BRANCH_JOINT_RANGE_DEG,
                          free_turn=False, lowered_arms=(),
                          max_waist_z=None, y_window=None,
                          yaw_margin_deg=None, lowered_forearms=()):
    """下ろした手の候補 ``goal_palms`` (腕の低い順の 3 つの ``(位置,
    回転行列)``) への押し込み目標を、握りの向き ``turn_degs`` ごとに 1 回の
    バッチ IK で解き、移動先の台車・姿勢の候補 (前後は人の立ち位置
    ±``GOAL_X_MARGIN``) を、腕の低い目標から、同じ目標ではコストの安い順に
    返す。

    ``max_waist_z`` (押し込み終了時の腰の高さ [m]) を与え、``GOAL_LEG_RISE``
    が 'off' でなければ、腰がそれより ``GOAL_LEG_RISE_TOLERANCE`` を超えて
    高い候補は外す ('fallback' は全部外れるときだけ外さない)。

    人の立ち位置まわりに回して人の正面方向が +x の座標系で解き、台車の位置
    姿勢は元の座標系に戻す。左右は差し出した手の側 (``y_window``
    ``(下限, 上限)`` [m] で手の側の横の距離を絞る)、向きは人の正面方向
    ±``yaw_margin_deg`` (既定 ``GOAL_YAW_MARGIN_DEG``) 度に絞る。上半身が人の
    正面から ``UPPER_BODY_LIMIT_DEG`` を超える候補は捨て、残りはずれの二乗の
    コスト (``UPPER_BODY_WEIGHT``) を足す。

    Returns
    -------
    (candidates, x_margin)
        ``candidates`` は ``(cost, angle_vector, base (x, y, yaw),
        turn_deg, goal_palms の添字)`` のリスト、``x_margin`` は前後の窓
        (候補が無ければ None)。
    """
    standing = spik.human_standing_xy(translated_joints)
    facing = spik.human_facing_direction(translated_joints)
    human_yaw = math.atan2(facing[1], facing[0])
    to_local = _rotation_about(-human_yaw, standing)
    to_world = _rotation_about(human_yaw, standing)
    local_joints = {name: to_local.transform_point(p)
                    for name, p in translated_joints.items()}
    local_collision_joints = {name: to_local.transform_point(p)
                              for name, p in collision_joints.items()}
    local_arms = [[to_local.transform_point(p) for p in arm]
                  for arm in lowered_arms]
    local_forearms = [
        (to_local.transform_point(elbow), to_local.transform_point(wrist),
         to_local.transform_point(palm_pos), to_local.rot @ palm_rot)
        for elbow, wrist, palm_pos, palm_rot in lowered_forearms]

    def local_targets(turn_deg):
        targets = []
        for goal_palm in goal_palms:
            target = press_target(palm_from_frame(*goal_palm), turn_deg,
                                  robot_arm)
            targets.append(Coordinates(
                pos=to_local.transform_point(target.worldpos()),
                rot=to_local.rot @ target.worldrot()))
        return targets

    limits = list(base_limits)
    # 下ろした手は人の体の左右どちらかに来るので、手の左右で台車の側を
    # 決める (人の正面が +x なので左は +y)。
    if y_window is None:
        limits[1] = spik.restrict_base_y_range_to_hand_side(
            limits[1], 1.0 if hand == 'L' else -1.0)
    else:
        lo, hi = y_window
        limits[1] = (lo, hi) if hand == 'L' else (-hi, -lo)
    limits[2] = spik.restrict_base_yaw_range_to_human_facing(
        limits[2], 0.0, margin=math.radians(
            GOAL_YAW_MARGIN_DEG if yaw_margin_deg is None
            else yaw_margin_deg))
    upper_limit = math.radians(UPPER_BODY_LIMIT_DEG)
    names = [joint.name for joint in robot.joint_list]
    ankle, knee = names.index('ankle_joint'), names.index('knee_joint')
    waist_p = names.index('waist_p_joint')
    bend_cost_indices = spik._joint_bend_cost_indices(robot, robot_arm)
    wrist = _wrist_limit_table(robot, robot_arm, names)
    end_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    branch = _branch_joint_indices(robot, robot_arm)
    seed = np.asarray(seed_av, dtype=np.float64)
    branch_range = math.radians(branch_range_deg)
    window = [spik.restrict_base_x_range_to_human_standing(
        limits[0], standing[0], GOAL_X_MARGIN), limits[1], limits[2]]
    candidates = []
    finger_zs = []
    waist_zs = []
    for turn_deg in turn_degs:
        for palm_index, av, base in _batch_ik(
                robot, robot_arm, hand, local_targets(turn_deg), seed_av,
                local_collision_joints, collision_pairs, window,
                attempts_per_pose, free_turn=free_turn,
                lowered_arms=local_arms, lowered_forearms=local_forearms):
            if np.any(np.abs(av[branch] - seed[branch]) > branch_range):
                continue
            # 指先の上下は台車の yaw に依らないので、解いた座標系のまま。
            _place(robot, av, base)
            if max_waist_z is not None and GOAL_LEG_RISE != 'off':
                # 腰が少しだけ高い候補は、脚を押し込み終了時の角度に直す。
                rise = waist_height(robot) - max_waist_z
                if (GOAL_LEG_RISE_TOLERANCE < rise
                        <= GOAL_LEG_RISE_REPAIR_MAX):
                    av = av.copy()
                    av[ankle], av[knee] = seed[ankle], seed[knee]
                    _place(robot, av, base)
            # 人の正面が +x の座標系なので、人の正面からのずれは yaw そのもの。
            upper_dev = upper_body_deviation(robot, 0.0)
            if abs(upper_dev) > upper_limit:
                continue
            finger = finger_direction(end_coords.worldrot())
            xy = to_world.transform_point([base[0], base[1], 0.0])[:2]
            base = (float(xy[0]), float(xy[1]), base[2] + human_yaw)
            cost = (spik._joint_bend_cost_from_vector(
                        av, bend_cost_indices)
                    + placement_cost(base, standing, facing)
                    + GOAL_LEG_OFFSET_WEIGHT * abs(av[ankle] + av[knee])
                    + GOAL_CROUCH_WEIGHT * (abs(av[ankle])
                                            + abs(av[knee]))
                    + GOAL_BOW_WEIGHT * max(0.0, float(av[waist_p])) ** 2
                    + UPPER_BODY_WEIGHT * upper_dev ** 2
                    + GOAL_WRIST_WEIGHT * wrist_limit_penalty(av, wrist)
                    + GOAL_ROBOT_FINGER_DOWN_WEIGHT
                    * (1.0 + float(finger[2]))
                    + GOAL_WRIST_BEND_WEIGHT
                    * sum(abs(av[index]) for index, _, _ in wrist))
            candidates.append((cost, av, base, turn_deg, palm_index))
            finger_zs.append(float(finger[2]))
            waist_zs.append(waist_height(robot))
    if max_waist_z is not None and GOAL_LEG_RISE != 'off':
        keep = [i for i, z in enumerate(waist_zs)
                if z <= max_waist_z + GOAL_LEG_RISE_TOLERANCE]
        if keep or GOAL_LEG_RISE == 'strict':
            candidates = [candidates[i] for i in keep]
            finger_zs = [finger_zs[i] for i in keep]
    down = [c for c, z in zip(candidates, finger_zs)
            if z <= GOAL_FINGER_DOWN_MAX_Z]
    if down:
        candidates = down
    if candidates:
        candidates.sort(key=lambda c: (c[4], c[0]))
        return candidates, GOAL_X_MARGIN
    return [], None


def _lowering_path_steps(human_arm, lowering, samples=20):
    """``lowering`` の動きの掌の道のり [m] と回転 [rad] (肩を中心に円弧で
    下りるので、両端の差より長い)。"""
    poses = [human_arm.lowering_pose(lowering, k / float(samples))
             for k in range(samples + 1)]
    length = sum(np.linalg.norm(b[2] - a[2]) for a, b in zip(poses, poses[1:]))
    angle = sum(Rotation.from_matrix(b[3] @ a[3].T).magnitude()
                for a, b in zip(poses, poses[1:]))
    return length, angle


def plan_path(robot, robot_arm, hand, start_turn, goal_turn, human_arm,
              lowering, start_av, base_path, goal_av,
              checker, free_turn=False):
    """``_plan_path_impl`` を、上半身の向きの制約の経路情報 (腰ヨーの始点・
    移動先、始点の上半身が範囲内か) を設定して解く。"""
    if _UPPER.human_yaw is None:
        return _plan_path_impl(
            robot, robot_arm, hand, start_turn, goal_turn, human_arm,
            lowering, start_av, base_path, goal_av, checker,
            free_turn=free_turn)
    index = [joint.name for joint in robot.joint_list].index('waist_y_joint')
    _place(robot, start_av, base_path.at(0.0))
    start_ok = abs(upper_body_deviation(robot, _UPPER.human_yaw)) <= \
        math.radians(UPPER_BODY_LIMIT_DEG + UPPER_BODY_START_SLACK_DEG)
    _UPPER.path = (float(start_av[index]), float(goal_av[index]), start_ok)
    try:
        return _plan_path_impl(
            robot, robot_arm, hand, start_turn, goal_turn, human_arm,
            lowering, start_av, base_path, goal_av, checker,
            free_turn=free_turn)
    finally:
        _UPPER.path = None


def _plan_path_impl(robot, robot_arm, hand, start_turn, goal_turn,
                    human_arm, lowering, start_av, base_path, goal_av,
                    checker, free_turn=False):
    """押し込み姿勢から移動先まで、台車 (``base_path``)・人の腕 (``lowering``)・
    握りの向き (``start_turn`` -> ``goal_turn`` [度]、``free_turn`` なら拘束せず
    IK に任せる)・首を補間しながら、ロボットの腕・脚を直前の解から解き進めた
    waypoint を返す。
    解けない waypoint は、直前の解を ``goal_av`` へ 1 区間寄せた初期値でも試す。
    人の腕が届かない・関節角が飛ぶ・干渉する waypoint があれば、その手前で
    止める。

    Returns
    -------
    (waypoints, fraction, reason)
        ``fraction`` は移動先までのうち進めた割合、``reason`` は止めた
        理由 (最後まで進めたら ``None``)。
    """
    palm_length, palm_angle = _lowering_path_steps(human_arm, lowering)
    turn_delta = math.degrees(_wrap(math.radians(goal_turn - start_turn)))
    start_av = np.asarray(start_av, dtype=np.float64)
    goal_av = np.asarray(goal_av, dtype=np.float64)
    n = max(2, int(math.ceil(max(
        base_path.steps(),
        palm_length / MAX_STEP,
        palm_angle / MAX_ANGLE_STEP,
        math.radians(abs(turn_delta)) / MAX_ANGLE_STEP,
        float(np.max(np.abs(goal_av - start_av))) / PLANNED_JOINT_STEP))))
    step_av = (goal_av - start_av) / n
    # 脚は IK で動かさず (_press_ik)、押し込み時から移動先へ補間する。
    names = [joint.name for joint in robot.joint_list]
    legs = [names.index(name) for name in ('ankle_joint', 'knee_joint')]
    head = {link.joint.name for link in robot.head.link_list}
    neck = [names.index(name) for name in sorted(head)]
    # 差し出さない腕 (肩ピッチ・肘) は台車の箱をよけるため差し替えるので、
    # 関節角の飛びの判定から外す。
    other = 'l' if robot_arm == 'r' else 'r'
    other_names = ['{}_shoulder_p_joint'.format(other),
                   '{}_elbow_joint'.format(other)]
    other_arm = [names.index(name) for name in other_names]
    jump_check = np.array([name not in head and name not in other_names
                           for name in names])
    prev = start_av
    z_start = z_prev = _waist_height_for_legs(
        robot, start_av[legs[0]], start_av[legs[1]])
    waypoints = []
    reason = None
    for k in range(1, n + 1):
        s = k / float(n)
        _UPPER.s = s
        base = base_path.at(s)
        elbow, _, palm_pos, palm_rot = human_arm.lowering_pose(lowering, s)
        arm = human_arm.pose(palm_pos, palm_rot, elbow=elbow)
        if arm['reach'] > 1.0:
            reason = '{}/{} で人の腕が届かない (到達率 {:.2f})'.format(
                k, n, arm['reach'])
            break
        turn_deg = start_turn + turn_delta * s
        target = press_target(palm_from_frame(palm_pos, palm_rot), turn_deg,
                              robot_arm)
        # 脚は直前から移動先へ残りの区間で均等に近づけた角度に固定して解く。
        # 解けなければ脚も IK で動かす。
        leg_angles = prev[legs] + (goal_av[legs] - prev[legs]) / (n - k + 1)
        if GOAL_LEG_RISE != 'off':
            # 補間の途中で腰が高くならないよう、高くなるときは膝をより曲げる。
            leg_angles = np.array(cap_leg_rise(
                robot, leg_angles[0], leg_angles[1],
                min(z_prev, z_start) + PATH_LEG_RISE_TOLERANCE))
        # 直前の解を移動先へ 1 区間寄せた初期値を先に試す (直前の解からだと
        # 手首などが直前の曲げに留まる)。
        for seed, free_legs in ((prev + step_av, False), (prev, False),
                                (prev, True)):
            seed = seed.copy()
            seed[legs] = leg_angles
            _place(robot, seed, base)
            if _press_ik(robot, robot_arm, target, None,
                         free_legs=free_legs,
                         leg_floor=(leg_angles if GOAL_LEG_RISE != 'off'
                                    else None)):
                break
        else:
            reason = '{}/{} で腕が掌に届かない'.format(k, n)
            break
        neck_av = robot.angle_vector().copy()
        neck_av[neck] = start_av[neck] + (goal_av[neck] - start_av[neck]) * s
        robot.angle_vector(neck_av)
        # 上体を傾けると差し出さない手が台車の前の箱に入るので、入らない姿勢に
        # 差し替える (直前より曲げの小さい候補には戻さない)。
        if spik.other_hand_base_clearance(robot, robot_arm) \
                < spik.OTHER_ARM_BASE_CLEARANCE:
            current = robot.angle_vector().copy()
            bend = float(np.sum(np.abs(prev[other_arm])))
            for index, posture in enumerate(spik.OTHER_ARM_POSTURES_DEG):
                if sum(abs(math.radians(v)) for v in posture) < bend - 1e-6:
                    continue
                spik.with_other_arm_posture(robot, current, robot_arm, index)
                if spik.other_hand_base_clearance(robot, robot_arm) \
                        >= spik.OTHER_ARM_BASE_CLEARANCE:
                    break
            else:
                robot.angle_vector(current)
        av = robot.angle_vector().copy()
        jump = int(np.argmax(np.abs(av - prev) * jump_check))
        if abs(av[jump] - prev[jump]) > MAX_JOINT_STEP:
            reason = '{}/{} で関節角が飛ぶ ({} {:.0f} 度)'.format(
                k, n, names[jump], math.degrees(av[jump] - prev[jump]))
            break
        problem = checker.check(robot, human_arm.skeleton(arm), arm)
        if problem is None:
            actual_turn = grip_turn_deg(
                robot, robot_arm, palm_from_frame(palm_pos, palm_rot))
            arm_s = human_arm.arm_fraction(s)
            wrist, problem = wrist_problem(human_arm, robot, arm, arm_s,
                                           actual_turn)
        if problem is not None:
            reason = '{}/{} で{}'.format(k, n, problem)
            break
        if free_turn:
            turn_deg = actual_turn
        arm = dict(arm, s=float(arm_s), robot_hand_wrist=wrist)
        waypoints.append(dict(
            base_position=[float(base[0]), float(base[1]), 0.0],
            base_yaw=float(base[2]),
            joint_angle_vector=[float(v) for v in av],
            turn_deg=float(turn_deg), human_arm=arm))
        prev = av
        z_prev = _waist_height_for_legs(robot, av[legs[0]], av[legs[1]])
    return waypoints, len(waypoints) / float(n), reason


# 横並びで前を見る点までの、人の正面方向への距離 [m] と、目の高さが
# 求まらないときの高さ [m]。
FORWARD_GAZE_DISTANCE = 10.0
FORWARD_GAZE_DEFAULT_HEIGHT = 1.4


def forward_gaze_point(joint_positions):
    """横並びで見る点: 人の立ち位置から人の正面方向へ
    ``FORWARD_GAZE_DISTANCE`` 先の、人の目 (鼻、なければ両耳の中点、なければ
    ``FORWARD_GAZE_DEFAULT_HEIGHT``) の高さの点。人の顔を見ると首・上半身が
    人の方へ向くので、人と同じ前方を見る。"""
    standing = spik.human_standing_xy(joint_positions)
    facing = spik.human_facing_direction(joint_positions)
    nose = joint_positions.get('Nose')
    ears = [joint_positions.get(name) for name in ('LEar', 'REar')]
    if nose is not None:
        height = float(nose[2])
    elif all(ear is not None for ear in ears):
        height = float(ears[0][2] + ears[1][2]) / 2.0
    else:
        height = FORWARD_GAZE_DEFAULT_HEIGHT
    xy = standing + facing * FORWARD_GAZE_DISTANCE
    return np.array([xy[0], xy[1], height])


def plan_transition(*args, **kwargs):
    """``_plan_transition`` を、乱数 (バッチ IK の初期値) を押し込み姿勢から
    決めたシードにして解く。呼び出し元の乱数の状態は戻す。"""
    post = args[3] if len(args) > 3 else kwargs['post']
    return _with_press_seed(post, _plan_transition, *args, **kwargs)


def _with_press_seed(post, function, *args, **kwargs):
    """``function`` を、乱数を押し込み姿勢 ``post`` から決めたシードにして
    呼び、呼び出し元の乱数の状態は戻す。"""
    state = np.random.get_state()
    np.random.seed(zlib.crc32(np.round(np.asarray(
        post['joint_angle_vector'], dtype=np.float64), 6).tobytes()))
    try:
        return function(*args, **kwargs)
    finally:
        np.random.set_state(state)


def refine_transition(robot, robot_arm, hand, post, turn_deg,
                      translated_joints, collision_joints, verification_pairs,
                      collision_pairs, base_limits, secured,
                      attempts_per_pose=GOAL_ATTEMPTS_PER_POSE):
    """動かす前に確保した横並び移動 ``secured`` (``plan_transition`` の
    approach=False の結果) を、寄せ (段階 2) と残りの掌の向きで良くした計画を
    返す (実機ノードは動いている間に裏のスレッドで解く)。

    まず ``secured`` の終わりから寄せを解き、寄せ後の横の距離が
    ``APPROACH_STOP_LATERAL`` 以下ならそれを返す。そうでなければ残りの向きも
    解き (``_plan_transition``)、``secured`` より腕を上げないもの
    (``_low_arm_results``) のうち寄せ後の一番近いものを返す。``robot`` は、
    ほかで使っていないロボットモデルにすること (IK で姿勢を書き換える)。"""
    return _with_press_seed(
        post, _refine_transition, robot, robot_arm, hand, post, turn_deg,
        translated_joints, collision_joints, verification_pairs,
        collision_pairs, base_limits, secured, attempts_per_pose)


def _refine_transition(robot, robot_arm, hand, post, turn_deg,
                       translated_joints, collision_joints,
                       verification_pairs, collision_pairs, base_limits,
                       secured, attempts_per_pose):
    t0 = time.time()
    facing = spik.human_facing_direction(translated_joints)
    common = (robot, robot_arm, hand, post, turn_deg, translated_joints,
              collision_joints, verification_pairs, collision_pairs,
              base_limits)
    results = []
    found = secured.get('palm_facing') if secured else None
    if (secured and secured.get('verified') and secured['fraction'] >= 1.0
            and facing is not None):
        first = copy.deepcopy(secured)
        with _upper_body_limit(math.atan2(facing[1], facing[0])):
            _add_approach(
                robot, robot_arm, hand, post, turn_deg, first,
                translated_joints, collision_joints, verification_pairs,
                collision_pairs, base_limits,
                first.get('yaw_margin', GOAL_YAW_MARGIN_DEG))
        results.append(first)
        approach = first['approach']
        if (approach['n_waypoints']
                and approach['lateral_after'] <= APPROACH_STOP_LATERAL):
            first['compute_time'] = time.time() - t0
            return first
    rest = [name for name in PALM_FACING_ORDER if name != found]
    if rest:
        results.append(_plan_transition(
            *common, attempts_per_pose=attempts_per_pose, facings=rest))
    if secured:
        # 確保済みより腕を上げる計画は採らない。
        results.append(secured)
    best = min(_low_arm_results(results),
               key=lambda r: _result_key(r, translated_joints, hand))
    if best is secured:
        best = copy.deepcopy(secured)
    best['compute_time'] = time.time() - t0
    return best


def _lowered_palm_z(result):
    """横並びまで進んだ計画 ``result`` の、腕を下ろしきった人の掌の高さ [m]
    (進んでいなければ None)。"""
    if not (result.get('verified') and result.get('fraction', 0.0) >= 1.0):
        return None
    for wp in reversed(result.get('waypoints') or []):
        arm = wp.get('human_arm')
        if arm is not None:
            return float(arm['palm_position'][2])
    return None


def _low_arm_results(results):
    """``results`` から、腕を下ろしきった掌が一番低い計画より
    ``LOWERED_HEIGHT_TOLERANCE`` を超えて高い計画を除いたもの。"""
    heights = [_lowered_palm_z(r) for r in results]
    known = [z for z in heights if z is not None]
    if not known:
        return results
    limit = min(known) + LOWERED_HEIGHT_TOLERANCE
    return [r for r, z in zip(results, heights) if z is None or z <= limit]


def _result_key(result, translated_joints, hand):
    """横並び移動の計画結果の良さの並べ替えの鍵 (小さいほど良い): 寄せまで
    進んだもの (寄せ後の横の距離、2 cm 刻み) > 横並びまで進んだもの (横の
    距離) > 途中まで (進めた割合) > 失敗。同じなら ``PALM_FACING_ORDER`` の
    先の向き。"""
    facing = result.get('palm_facing')
    phi = (PALM_FACING_ORDER.index(facing) if facing in PALM_FACING_ORDER
           else len(PALM_FACING_ORDER))
    approach = result.get('approach')
    if approach and approach['n_waypoints']:
        return (0, round(approach['lateral_after'] / 0.02), phi)
    if result['verified'] and result['fraction'] >= 1.0:
        end = result['waypoints'][-1]['base_position']
        return (1, human_lateral(translated_joints, end, hand), phi)
    if result['verified']:
        return (2, -result['fraction'], phi)
    return (3, 0.0, phi)


def _plan_stage1(robot, robot_arm, hand, post, turn_deg,
                 translated_joints, collision_joints, verification_pairs,
                 collision_pairs, base_limits,
                 attempts_per_pose=GOAL_ATTEMPTS_PER_POSE,
                 use_facing_variants=True, facings=None,
                 yaw_margin_deg=None, deadline=None):
    """押し込み姿勢 ``post`` (``solve_post_process`` の結果 dict、向き
    ``turn_deg``) から、つないだ手を人の体の横へ下ろしながら、ロボットも
    人と横並びになる位置・姿勢へ移る waypoint を計画する (段階 1)。

    ``translated_joints`` は立ち位置・向き・人の腕の判定用、
    ``collision_joints`` は干渉判定用 (体幹をずらした) の骨格 (どちらも
    IK と同じ座標系)。``base_limits`` は台車の可動域全体。``facings`` は
    試す掌の向き (``HumanArm.lowering`` の facing) の順 (None なら
    ``LOWERED_FACING_ORDER``)、``yaw_margin_deg`` は移動先の台車の向きの範囲。
    ``deadline`` (``time.time()`` の値) を過ぎたら探索をやめ、それまでの一番
    良い経路を返す (``timed_out`` が True)。

    Returns
    -------
    dict
        ``verified`` (横並び移動をするか)、``waypoints`` (押し込み姿勢の
        次から移動先まで、``plan_handshake_motion`` と同じ形に、人の腕
        ``human_arm`` (``HumanArm.pose``) を加えたもの)、``joint_names``、
        ``hand``、``press_arm`` (押し込み時の人の腕)、``fraction``、
        ``x_margin``、``reason`` (止めた・しなかった理由)、
        ``placement_before``/``placement_after``、``human_max`` (人の
        腕の目安の最大値)、``compute_time`` [秒]。
    """
    t0 = time.time()
    result = dict(verified=False, waypoints=[], hand=hand,
                  joint_names=list(post['joint_names']), fraction=0.0,
                  x_margin=None, reason=None,
                  placement_before=None, placement_after=None,
                  timed_out=False)
    needed = ['{}{}'.format(hand, name) for name in
              ('Shoulder', 'Elbow', 'Wrist')] + [
        '{}Shoulder'.format('L' if hand == 'R' else 'R')]
    if (spik.human_standing_xy(translated_joints) is None
            or spik.human_facing_direction(translated_joints) is None
            or any(name not in translated_joints for name in needed)):
        result.update(reason='人の立ち位置・向き・腕が骨格から求まらない',
                      compute_time=time.time() - t0)
        return result
    start_base = np.array([post['base_position'][0],
                           post['base_position'][1], post['base_yaw']])
    start_av = np.asarray(post['joint_angle_vector'], dtype=np.float64)
    result['placement_before'] = spik.base_placement_metrics(
        translated_joints, post['base_position'], post['base_yaw'])
    # 腰は押し込み終了時より上げない (GOAL_LEG_RISE)。
    _place(robot, start_av, start_base)
    start_waist_z = waist_height(robot)

    palm = human_palm_from_press(post, turn_deg, robot_arm)
    human_arm = HumanArm(translated_joints, hand, palm, turn_deg, robot_arm)
    result['press_arm'] = dict(human_arm.pose(*palm_frame(palm)),
                               s=0.0, robot_hand_wrist=human_arm.press_wrist)
    press_palm_z = result['press_arm']['palm_position'][2]
    gaze_forward = forward_gaze_point(translated_joints)
    standing = spik.human_standing_xy(translated_joints)
    checker = TransitionChecker(robot_arm, hand, verification_pairs)
    reasons = []
    best = [None]
    # plan_path を呼んだ回数と、100% に届いた (最初の) 回数。
    n_paths = [0]
    n_paths_full = [None]
    path_fractions = []
    free_turn = GOAL_FREE_TURN
    # 握りの向きを自由にするなら、目標の向きは 0 度の 1 組だけでよい。
    turn_groups = ((0.0,),) if free_turn else GOAL_TURN_DEG_GROUPS

    def out_of_time():
        if deadline is None or time.time() < deadline:
            return False
        if not result['timed_out']:
            result['timed_out'] = True
            reasons.append('探索の時間の上限に達した')
        return True

    def try_lowerings(lowerings):
        """腕の下ろし方 ``lowerings`` (3 つ) を移動先にして経路を作る。
        横並びまで (100%) 進めたか、時間の上限を過ぎたら True。"""
        goal_palms = [lowering['palm'] for lowering in lowerings]
        lowered_arms = [(human_arm.shoulder,
                         human_arm.lowering_pose(lowering, 1.0)[0])
                        for lowering in lowerings]
        lowered_forearms = [human_arm.lowering_pose(lowering, 1.0)
                            for lowering in lowerings]
        for turn_degs in turn_groups:
            if out_of_time():
                return True
            candidates, x_margin = solve_goal_candidates(
                robot, robot_arm, hand, goal_palms, start_av,
                translated_joints, collision_joints, collision_pairs,
                base_limits, turn_degs=turn_degs,
                attempts_per_pose=attempts_per_pose, free_turn=free_turn,
                lowered_arms=lowered_arms, max_waist_z=start_waist_z,
                yaw_margin_deg=yaw_margin_deg,
                lowered_forearms=lowered_forearms)
            if not candidates:
                reasons.append('移動先の IK が解けない (腕 {} 度、握りの向き '
                               '{})'.format(
                                   '/'.join('{:.0f}'.format(lw['arm_deg'])
                                            for lw in lowerings),
                                   '/'.join('{:.0f}'.format(t)
                                            for t in turn_degs)))
                continue
            tried = {}
            checked = {}
            for _, goal_av, goal_base, goal_turn, palm_index in candidates:
                if (tried.get(palm_index, 0) >= MAX_GOAL_CANDIDATES
                        or checked.get(palm_index, 0) >= MAX_GOAL_CHECKS):
                    continue
                if out_of_time():
                    return True
                lowering = lowerings[palm_index]
                goal_palm = lowering['palm']
                goal_arm = human_arm.pose(
                    *goal_palm,
                    elbow=human_arm.lowering_pose(lowering, 1.0)[0])
                goal_skeleton = human_arm.skeleton(goal_arm)
                # yaw を押し込み時から近い回り方の値にする。
                goal_base = np.array([goal_base[0], goal_base[1],
                                      start_base[2]
                                      + _wrap(goal_base[2] - start_base[2])])
                base_path = BasePath(start_base, goal_base, standing)
                # 台車の経路で車輪が人の脚に近づく候補は、経路を計画する前に
                # 外す (安く判定できるので ``checked`` には数えない)。
                problem = checker.check_base_path(
                    robot, goal_av, base_path, goal_skeleton)
                if problem is not None:
                    reasons.append(problem)
                    continue
                checked[palm_index] = checked.get(palm_index, 0) + 1
                # 移動先の姿勢そのものを、首も含めて押し込み姿勢に解き直して
                # 検証してから経路を解く。
                goal_target = press_target(palm_from_frame(*goal_palm),
                                           goal_turn, robot_arm)
                _place(robot, goal_av, goal_base)
                if not _press_ik(robot, robot_arm, goal_target, gaze_forward,
                                 free_turn=free_turn):
                    reasons.append('移動先で押し込み IK が解けない')
                    continue
                # 指先が上を向いた解は、握りの向きを指先が一番下を向く角度に固定して
                # 解き直す (解けなければそのまま)。
                if finger_z(robot, robot_arm) > GOAL_FINGER_DOWN_MAX_Z:
                    before = robot.angle_vector().copy()
                    before_z = finger_z(robot, robot_arm)
                    down_target = press_target(
                        palm_from_frame(*goal_palm),
                        down_turn_deg(palm_from_frame(*goal_palm), robot_arm),
                        robot_arm)
                    if (not _press_ik(robot, robot_arm, down_target,
                                      gaze_forward)
                            or finger_z(robot, robot_arm) >= before_z):
                        robot.angle_vector(before)
                # 腕と同時の視線 IK は首が大きく回せず解けないことがあるので、首だけで
                # 前へ向け直す (経路の首はここへ補間する)。
                _aim_head(robot, gaze_forward)
                if free_turn:
                    goal_turn = grip_turn_deg(robot, robot_arm,
                                              palm_from_frame(*goal_palm))
                # 人の手首が目安を超えるなら、収まる握りの向きで解き直す。
                goal_turn, problem = fix_goal_wrist(
                    robot, robot_arm, human_arm, goal_arm, *goal_palm,
                    gaze_forward, goal_turn)
                if problem is None:
                    problem = checker.check(robot, goal_skeleton, goal_arm)
                if problem is not None:
                    reasons.append('移動先で{}'.format(problem.split(' ')[0]))
                    continue
                tried[palm_index] = tried.get(palm_index, 0) + 1
                if (MAX_PATH_TRIES is not None
                        and n_paths[0] >= MAX_PATH_TRIES):
                    reasons.append('経路の試行回数が上限 ({}) に達した'.format(
                        MAX_PATH_TRIES))
                    return True
                n_paths[0] += 1
                waypoints, fraction, why = plan_path(
                    robot, robot_arm, hand, turn_deg, goal_turn, human_arm,
                    lowering, start_av, base_path,
                    robot.angle_vector().copy(), checker,
                    free_turn=free_turn)
                path_fractions.append(round(float(fraction), 3))
                if why is not None:
                    reasons.append('経路の{}'.format(why))
                if best[0] is None or fraction > best[0][1]:
                    best[0] = (waypoints, fraction)
                    result.update(arm_deg=lowering['arm_deg'],
                                  abduction_deg=lowering['abduction_deg'],
                                  palm_facing=lowering['palm_facing'],
                                  x_margin=x_margin)
                if fraction >= 1.0:
                    n_paths_full[0] = n_paths[0]
                    return True
            # 次の組 (握りの向き ±90 度) は、この組で移動先の IK が 1 つも解けなかった
            # ときだけ試す。
            return False
        return False

    def facing_variants(arm_degs, abduction_deg):
        """組 ``arm_degs`` の掌の向きごとの下ろし方 (3 つとも同じ向き) の
        ``[(向きの順位, 下ろし方)]``。向きは ``LOWERED_FACING_ORDER`` の順で、
        使えない向き・前腕のひねりが ``LOWERED_TWIST_MAX_DEG`` を超える向きは
        除く。``use_facing_variants`` が False なら前腕のひねりが一番少ない
        向き (上腕の角度ごとに違ってよい) の 1 つだけ。掌が押し込み時より
        ``LOWERED_MIN_DROP`` 下がらない下ろし方は、下がる下ろし方で置き換え
        (バッチ IK の目標の数を変えないため)、1 つも無ければその組を除く。"""
        def lowered_enough(lowerings):
            ok = [lw['palm'][0][2] <= press_palm_z - LOWERED_MIN_DROP
                  for lw in lowerings]
            if not any(ok):
                return None
            low = lowerings[ok.index(True)]
            return [lw if k else low for lw, k in zip(lowerings, ok)]

        if not use_facing_variants:
            lowerings = lowered_enough([human_arm.lowering(deg, abduction_deg)
                                        for deg in arm_degs])
            return [] if lowerings is None else [(0, lowerings)]
        variants = []
        palm_up = float(human_arm.palm_rot0[2, 1]) >= PALM_UP_NORMAL_Z
        facing_order = (facings if facings is not None
                        else LOWERED_FACING_ORDER_PALM_UP if palm_up
                        else LOWERED_FACING_ORDER)
        for rank, facing in enumerate(facing_order):
            lowerings = [human_arm.lowering(deg, abduction_deg, facing)
                         for deg in arm_degs]
            if any(lw is None for lw in lowerings):
                continue
            twist = max(abs(math.degrees(lw['twist'])) for lw in lowerings)
            if twist > LOWERED_TWIST_MAX_DEG:
                continue
            lowerings = lowered_enough(lowerings)
            if lowerings is None:
                continue
            variants.append((rank, lowerings))
        return variants

    # 人が腕を低く下ろせる組から試し、横並びまで進めた組で止める。同じ
    # 上腕の角度の組の中では、脇の開きより掌の向きを優先する。
    groups = []
    for arm_degs, abduction_deg in LOWERED_ARM_GROUPS:
        if not groups or groups[-1][0] != arm_degs:
            groups.append((arm_degs, []))
        groups[-1][1].append(abduction_deg)
    for arm_degs, abductions in groups:
        ordered = []
        for k, abduction_deg in enumerate(abductions):
            for rank, lowerings in facing_variants(arm_degs, abduction_deg):
                ordered.append(((rank, k), lowerings))
        if not ordered:
            reasons.append('押し込み時より掌が下がる下ろし方が無い (上腕 {} 度)'
                           .format('/'.join('{:.0f}'.format(d)
                                            for d in arm_degs)))
        ordered.sort(key=lambda item: item[0])
        if any(try_lowerings(lowerings) for _, lowerings in ordered):
            break
    best = best[0]
    result.update(n_paths=n_paths[0], n_paths_full=n_paths_full[0],
                  path_fractions=path_fractions)
    if best is None or best[1] < MIN_FRACTION:
        result.update(reason=', '.join(sorted(set(reasons))) or None,
                      compute_time=time.time() - t0)
        return result
    waypoints, fraction = best
    last = waypoints[-1]
    names = list(post['joint_names'])

    def legs(av):
        return [math.degrees(av[names.index('ankle_joint')]),
                math.degrees(av[names.index('knee_joint')])]

    result.update(
        legs_before=legs(start_av), legs_after=legs(last['joint_angle_vector']),
        turn_before=float(turn_deg), turn_after=last['turn_deg'])
    result.update(
        verified=True, waypoints=waypoints, fraction=fraction,
        reason=None if fraction >= 1.0 else ', '.join(sorted(set(reasons))),
        placement_after=spik.base_placement_metrics(
            translated_joints, last['base_position'], last['base_yaw']),
        human_max=human_arm_max(waypoints),
        compute_time=time.time() - t0)
    return result


def human_arm_max(waypoints):
    """waypoint (押し込み時は含まない) の人の腕の目安の最大値の dict。手首の
    角度 (``robot_hand_wrist``) は掌屈・背屈・撓屈・尺屈・ひねりに分ける。"""
    arms = [wp['human_arm'] for wp in waypoints if 'human_arm' in wp]
    out = dict(
        reach=max(a['reach'] for a in arms),
        elbow_flex_deg=max(a['elbow_flex_deg'] for a in arms),
        wrist_bend_deg=max(a['wrist_bend_deg'] for a in arms),
        twist_change_deg=max(abs(a['twist_change_deg']) for a in arms))
    wrists = [a['robot_hand_wrist'] for a in arms if 'robot_hand_wrist' in a]
    if wrists:
        out.update(
            flexion_deg=max(0.0, max(w['flexion_deg'] for w in wrists)),
            extension_deg=max(0.0, -min(w['flexion_deg'] for w in wrists)),
            radial_deg=max(0.0, max(w['deviation_deg'] for w in wrists)),
            ulnar_deg=max(0.0, -min(w['deviation_deg'] for w in wrists)),
            hand_twist_deg=max(abs(w['twist_deg']) for w in wrists))
    return out


class _StaticArm(object):
    """掌・人の腕を動かさない ``HumanArm`` の代わり (寄せの ``plan_path``
    用)。"""

    def __init__(self, real, arm_pose):
        self.real, self.arm_pose = real, arm_pose
        self.palm_pos = np.asarray(arm_pose['palm_position'],
                                   dtype=np.float64)
        self.palm_rot = np.asarray(arm_pose['palm_rot'], dtype=np.float64)
        self.elbow = np.asarray(arm_pose['elbow'], dtype=np.float64)

    def __getattr__(self, name):
        # 押し込み時の手首の角度など、ほかは本物の腕のもの。
        return getattr(self.real, name)

    def arm_fraction(self, s):
        return 1.0

    def lowering_pose(self, lowering, s):
        return self.elbow, None, self.palm_pos, self.palm_rot

    def pose(self, palm_pos, palm_rot, elbow=None):
        return copy.deepcopy(self.arm_pose)

    def skeleton(self, arm):
        return self.real.skeleton(arm)


def human_lateral(joint_positions, xy, hand):
    """台車の位置 ``xy`` の、人の立ち位置から差し出した手の側への、人の
    左右方向の距離 [m]。"""
    standing = spik.human_standing_xy(joint_positions)
    facing = spik.human_facing_direction(joint_positions)
    left = np.array([-facing[1], facing[0]])
    side = 1.0 if hand == 'L' else -1.0
    return float(np.dot(np.asarray(xy[:2]) - standing, left)) * side


def plan_approach(robot, robot_arm, hand, post, turn_deg, last,
                  translated_joints, collision_joints, verification_pairs,
                  collision_pairs, base_limits, yaw_margin_deg=None,
                  target_lateral=APPROACH_TARGET_LATERAL, windows=None,
                  attempts_per_pose=GOAL_ATTEMPTS_PER_POSE,
                  direct_below=None):
    """横並びの終わり ``last`` (``plan_path`` の最後の waypoint) の姿勢から、
    人の腕・掌を動かさず、台車を横へ人の方へ寄せる waypoint (寄せ、段階 2)
    を計画する。台車の向きは変えない。

    寄せ後の横の距離が目標 ``target_lateral`` の窓 (``APPROACH_WINDOW_COUNT``
    個、近い方から) の移動先をバッチ IK で解き、``plan_path`` で経路を作る。
    台車を窓の端へ直線で寄せる経路も作り、より近ければそちらを採る。

    Returns
    -------
    dict
        ``waypoints`` (寄せの waypoint、空なら寄せない)、``lateral_before``/
        ``lateral_after`` [m]、``window``、``mode`` ('batch'/'direct')、
        ``reason``。
    """
    start = np.array([last['base_position'][0], last['base_position'][1],
                      last['base_yaw']])
    lat0 = human_lateral(translated_joints, start, hand)
    out = dict(waypoints=[], lateral_before=lat0, lateral_after=lat0,
               window=None, mode=None, reason=None)
    facing = spik.human_facing_direction(translated_joints)
    left = np.array([-facing[1], facing[0]])
    side = 1.0 if hand == 'L' else -1.0
    palm0 = human_palm_from_press(post, turn_deg, robot_arm)
    real = HumanArm(translated_joints, hand, palm0, turn_deg, robot_arm)
    arm_pose = last['human_arm']
    static = _StaticArm(real, arm_pose)
    palm = palm_from_frame(static.palm_pos, static.palm_rot)
    start_av = np.asarray(last['joint_angle_vector'], dtype=np.float64)
    checker = TransitionChecker(robot_arm, hand, verification_pairs)
    skeleton = real.skeleton(arm_pose)
    _place(robot, start_av, start)
    start_waist_z = waist_height(robot)
    if windows is None:
        windows = [(round(target_lateral - 0.1 + 0.1 * i, 3),
                    round(target_lateral + 0.1 * i, 3))
                   for i in range(APPROACH_WINDOW_COUNT)]
    reasons = []
    best = None
    window = None
    for lo, hi in windows:
        if hi > lat0 - APPROACH_MIN_GAIN:
            break
        candidates, _ = solve_goal_candidates(
            robot, robot_arm, hand, [(static.palm_pos, static.palm_rot)] * 3,
            start_av, translated_joints, collision_joints, collision_pairs,
            base_limits, turn_degs=(0.0,), free_turn=GOAL_FREE_TURN,
            lowered_arms=[(real.shoulder, static.elbow)] * 3,
            max_waist_z=start_waist_z, branch_range_deg=100.0,
            y_window=(lo, hi), yaw_margin_deg=yaw_margin_deg,
            attempts_per_pose=attempts_per_pose,
            lowered_forearms=[(static.elbow, arm_pose['wrist'],
                               static.palm_pos, static.palm_rot)] * 3)
        if not candidates:
            reasons.append('窓 {}-{} m の移動先の IK が解けない'.format(lo, hi))
            continue
        # 横が近いものから (5 cm の幅の中ではコストの安い順)
        candidates = sorted(candidates, key=lambda c: round(
            human_lateral(translated_joints, c[2], hand) / 0.05))
        tried = 0
        for _, goal_av, goal_base, goal_turn, _ in candidates:
            if tried >= APPROACH_MAX_GOAL_TRIES:
                break
            # 台車の向きは変えない
            goal_base = np.array([goal_base[0], goal_base[1], start[2]])
            base_path = BasePath(start, goal_base)
            problem = checker.check_base_path(robot, goal_av, base_path,
                                              skeleton)
            if problem is not None:
                reasons.append(problem)
                continue
            target = press_target(palm, goal_turn, robot_arm)
            _place(robot, goal_av, goal_base)
            if not _press_ik(robot, robot_arm, target, None,
                             free_turn=GOAL_FREE_TURN):
                reasons.append('移動先で押し込み IK が解けない')
                continue
            if abs(upper_body_deviation(robot, _UPPER.human_yaw or 0.0)) \
                    > math.radians(UPPER_BODY_LIMIT_DEG) + 1e-3:
                reasons.append('移動先で上半身が範囲外')
                continue
            if GOAL_FREE_TURN:
                goal_turn = grip_turn_deg(robot, robot_arm, palm)
            goal_turn, problem = fix_goal_wrist(
                robot, robot_arm, static, arm_pose, static.palm_pos,
                static.palm_rot, None, goal_turn)
            goal_av = robot.angle_vector().copy()
            if problem is None:
                problem = checker.check(robot, skeleton, arm_pose)
            if problem is not None:
                reasons.append('移動先で' + problem.split(' ')[0])
                continue
            tried += 1
            waypoints, fraction, why = plan_path(
                robot, robot_arm, hand, last['turn_deg'], goal_turn, static,
                None, start_av, base_path, goal_av, checker,
                free_turn=GOAL_FREE_TURN)
            if why:
                reasons.append('経路の' + why)
            if fraction >= 1.0:
                best, window = waypoints, (lo, hi)
                break
        if best is not None:
            break
    mode = 'batch' if best is not None else None
    # 台車を目標の横の距離へ直線で寄せながら直前の解から腕・脚を解き進める
    # (移動先は決めない)。バッチ IK の結果より近ければ採る。
    goal = start.copy()
    goal[:2] = start[:2] - left * side * (lat0 - target_lateral)
    batch_lat = (human_lateral(translated_joints, best[-1]['base_position'],
                               hand) if best is not None else lat0)
    if (lat0 - target_lateral >= APPROACH_MIN_GAIN
            and (direct_below is None or batch_lat > direct_below)):
        waypoints, fraction, why = plan_path(
            robot, robot_arm, hand, last['turn_deg'], last['turn_deg'],
            static, None, start_av, BasePath(start, goal), start_av, checker,
            free_turn=GOAL_FREE_TURN)
        if waypoints:
            direct = human_lateral(translated_joints,
                                   waypoints[-1]['base_position'], hand)
            batch = (human_lateral(translated_joints,
                                   best[-1]['base_position'], hand)
                     if best is not None else lat0)
            if direct < batch - 0.02:
                best, window, mode = waypoints, None, 'direct'
    if best is None:
        out['reason'] = ', '.join(sorted(set(reasons)))[:300] or None
        return out
    out.update(waypoints=best, window=window, mode=mode,
               lateral_after=human_lateral(
                   translated_joints, best[-1]['base_position'], hand))
    return out


def _approach_ok(result):
    """寄せ (``result['approach']``) で ``APPROACH_FALLBACK_LATERAL`` 以内に
    寄れたか。"""
    approach = result.get('approach')
    return bool(approach and approach['n_waypoints']
                and approach['lateral_after'] <= APPROACH_FALLBACK_LATERAL)


def _add_approach(robot, robot_arm, hand, post, turn_deg, result,
                  translated_joints, collision_joints, verification_pairs,
                  collision_pairs, base_limits, yaw_margin_deg,
                  approach_kwargs=None):
    """段階 1 が横並びまで進んだ ``result`` に、寄せの waypoint を足す。
    寄せられなければ足さず、``result['approach']`` に理由だけ残す。"""
    approach = plan_approach(
        robot, robot_arm, hand, post, turn_deg, result['waypoints'][-1],
        translated_joints, collision_joints, verification_pairs,
        collision_pairs, base_limits, yaw_margin_deg=yaw_margin_deg,
        **(approach_kwargs or {}))
    waypoints = approach.pop('waypoints')
    approach['n_waypoints'] = len(waypoints)
    result['approach'] = approach
    if waypoints:
        result['waypoints'] = result['waypoints'] + waypoints
        result['human_max'] = human_arm_max(result['waypoints'])
        result['placement_after'] = spik.base_placement_metrics(
            translated_joints, waypoints[-1]['base_position'],
            waypoints[-1]['base_yaw'])
        names = list(result['joint_names'])
        av = waypoints[-1]['joint_angle_vector']
        result['legs_after'] = [math.degrees(av[names.index('ankle_joint')]),
                                math.degrees(av[names.index('knee_joint')])]
        result['turn_after'] = waypoints[-1]['turn_deg']


def _plan_transition(robot, robot_arm, hand, post, turn_deg,
                     translated_joints, collision_joints, verification_pairs,
                     collision_pairs, base_limits,
                     attempts_per_pose=GOAL_ATTEMPTS_PER_POSE,
                     use_facing_variants=True, approach=True, facings=None,
                     stop_lateral=APPROACH_STOP_LATERAL,
                     approach_kwargs=None,
                     time_limit=TRANSITION_TIME_LIMIT):
    """押し込み姿勢 ``post`` から、つないだ手を人の体の横へ下ろしながら
    ロボットも人と横並びになる位置・姿勢へ移り (段階 1、``_plan_stage1``)、
    さらに台車を人の方へ寄せる (段階 2、``plan_approach``) waypoint を計画する。

    人の掌の向き (``PALM_FACING_ORDER``) ごとに段階 1 を解き、横並びまで進んだ
    向きだけ段階 2 へ進める。寄せ後の横の距離が ``stop_lateral`` 以下の向きで
    打ち切り、打ち切れなければ全部解いて寄せ後の一番近い向き (差が 2 cm 以内
    なら回す量の小さい向き) を選ぶ。腕を下ろしきった掌が一番低い計画より
    ``LOWERED_HEIGHT_TOLERANCE`` を超えて高い計画は選ばない。台車の向きの範囲 ``GOAL_YAW_MARGIN_DEG``
    で ``APPROACH_FALLBACK_LATERAL`` 以内に寄れなければ、
    ``GOAL_YAW_FALLBACK_DEG`` まで広げて解き直す。上半身は人の正面から
    ``UPPER_BODY_LIMIT_DEG`` に収める。

    ``translated_joints``・``collision_joints``・``base_limits`` と戻り値は
    ``_plan_stage1`` と同じ。加えて ``approach`` (寄せ: ``lateral_before``/
    ``lateral_after`` [m]、``window``、``mode``、``n_waypoints``、``reason``)、
    ``yaw_margin`` を持ち、``waypoints`` は寄せまでを含む。``approach`` が
    False なら段階 2 は解かず、横並びまで進んだ最初の解で止める。
    ``time_limit`` [秒] (None なら無制限) を過ぎたら探索をやめ、それまでの
    一番良い解を返す (``timed_out`` が True)。
    """
    t0 = time.time()
    deadline = None if time_limit is None else t0 + time_limit
    standing = spik.human_standing_xy(translated_joints)
    facing = spik.human_facing_direction(translated_joints)
    if standing is None or facing is None or not use_facing_variants:
        # 人の向きが求まらなければ段階 1 の理由を返す。掌の向きを自分で決める
        # 呼び出し (use_facing_variants=False) は段階 1 だけ。
        return _plan_stage1(
            robot, robot_arm, hand, post, turn_deg, translated_joints,
            collision_joints, verification_pairs, collision_pairs,
            base_limits, attempts_per_pose=attempts_per_pose,
            use_facing_variants=use_facing_variants, deadline=deadline)
    human_yaw = math.atan2(facing[1], facing[0])
    facings = PALM_FACING_ORDER if facings is None else tuple(facings)
    # 寄せまで解くなら向きごとに段階 1・2 を解いて比べる。寄せないなら段階 1 を
    # 1 回で解き、最初に横並びまで進んだところで止める。
    groups = ([(name,) for name in facings] if approach
              else [tuple(facings)])
    margins = [GOAL_YAW_MARGIN_DEG]
    if GOAL_YAW_FALLBACK_DEG and GOAL_YAW_FALLBACK_DEG > GOAL_YAW_MARGIN_DEG:
        margins.append(GOAL_YAW_FALLBACK_DEG)
    results = []
    done = False

    def out_of_time():
        return deadline is not None and time.time() >= deadline
    with _upper_body_limit(human_yaw):
        for margin in margins:
            # 探索を広げるのは、寄せるなら近くへ寄れなかったとき、寄せないなら
            # 横並びまで進めた向きが無かったときだけ。
            if margin != margins[0] and (
                    any(_approach_ok(r) for r in results) if approach
                    else done):
                break
            for group in groups:
                if results and out_of_time():
                    done = True
                    break
                result = _plan_stage1(
                    robot, robot_arm, hand, post, turn_deg,
                    translated_joints, collision_joints, verification_pairs,
                    collision_pairs, base_limits,
                    attempts_per_pose=attempts_per_pose,
                    facings=group, yaw_margin_deg=margin, deadline=deadline)
                result.update(yaw_margin=margin, approach=None)
                if approach and result['verified'] \
                        and result['fraction'] >= 1.0:
                    _add_approach(
                        robot, robot_arm, hand, post, turn_deg, result,
                        translated_joints, collision_joints,
                        verification_pairs, collision_pairs, base_limits,
                        margin, approach_kwargs=approach_kwargs)
                if margin != margins[0] and result['verified']:
                    # 探索を広げた解も、終点の台車の向きは元の範囲内
                    limit = math.radians(GOAL_YAW_MARGIN_DEG) + 0.01
                    ends = [wp['base_yaw'] - human_yaw
                            for wp in result['waypoints'][-1:]]
                    if any(abs(_wrap(e)) > limit for e in ends):
                        continue
                results.append(result)
                ap = result['approach']
                if (ap and ap['n_waypoints']
                        and ap['lateral_after'] <= stop_lateral):
                    done = True
                    break
                if (not approach and result['verified']
                        and result['fraction'] >= 1.0):
                    # 寄せを解かないなら、横並びまで進んだ最初の向きで終える
                    done = True
                    break
            if done:
                break

    best = min(_low_arm_results(results),
               key=lambda r: _result_key(r, translated_joints, hand))
    best['tries'] = len(results)
    best['timed_out'] = out_of_time()
    best['compute_time'] = time.time() - t0
    return best


def add_human_arm_arguments(parser):
    """人の前腕の障害物・手首の評価を切り替える引数を ``parser`` に足す
    (``apply_human_arm_arguments`` で反映する)。"""
    parser.add_argument(
        '--transition-forearm-obstacle', choices=('on', 'off'),
        default='on' if HUMAN_FOREARM_OBSTACLE else 'off',
        help='横並び移動で人の前腕・手の甲をロボットの前腕の障害物にするか。')
    parser.add_argument(
        '--transition-human-wrist', choices=HUMAN_WRIST_CHECKS,
        default=HUMAN_WRIST_CHECK,
        help='横並び移動で人の手首をどう評価するか (hand_fixed: 人の手を'
             'ロボットの手に固定した手首の角度、turn_cap: 握りの向きの変化、'
             'off: 見ない。既定 {})。'.format(HUMAN_WRIST_CHECK))
    parser.add_argument(
        '--transition-wrist-limits', type=float, nargs=3, default=None,
        metavar=('RADIAL', 'ULNAR', 'FLEX'),
        help='hand_fixed の手首の目安 [度] (撓屈・尺屈・掌屈/背屈、既定 '
             '{:.0f} {:.0f} {:.0f})。'.format(
                 HUMAN_WRIST_RADIAL_MAX_DEG, HUMAN_WRIST_ULNAR_MAX_DEG,
                 HUMAN_WRIST_FLEX_MAX_DEG))


def apply_human_arm_arguments(args):
    global HUMAN_FOREARM_OBSTACLE, HUMAN_WRIST_CHECK
    global HUMAN_WRIST_RADIAL_MAX_DEG, HUMAN_WRIST_ULNAR_MAX_DEG
    global HUMAN_WRIST_FLEX_MAX_DEG
    HUMAN_FOREARM_OBSTACLE = args.transition_forearm_obstacle == 'on'
    HUMAN_WRIST_CHECK = args.transition_human_wrist
    if args.transition_wrist_limits is not None:
        (HUMAN_WRIST_RADIAL_MAX_DEG, HUMAN_WRIST_ULNAR_MAX_DEG,
         HUMAN_WRIST_FLEX_MAX_DEG) = args.transition_wrist_limits


def human_arm_text(arm):
    """人の腕の目安の 1 行の説明 (目安を超えたものに印を付ける)。"""
    def mark(value, limit):
        return '{:.0f}{}'.format(value, ' (!)' if value > limit else '')
    down = arm.get('arm_down_deg')
    return ('{}到達率 {:.2f}{}、肘の曲げ {} 度、手首の曲げ {} 度、前腕のひねり '
            '{} 度'.format(
                '' if down is None else '上腕 (真下から) {:.0f} 度、'.format(
                    down),
                arm['reach'], ' (!)' if arm['reach'] > 1.0 else '',
                mark(arm['elbow_flex_deg'], HUMAN_ELBOW_FLEX_MAX_DEG),
                mark(arm['wrist_bend_deg'], HUMAN_WRIST_BEND_MAX_DEG),
                mark(abs(arm['twist_change_deg']),
                     HUMAN_TWIST_CHANGE_MAX_DEG))
            + robot_hand_wrist_text(arm))


def robot_hand_wrist_text(arm):
    """``human_arm_max`` の手首の角度 (人の手をロボットの手に固定したとき)
    の説明 (無ければ空)。"""
    if 'flexion_deg' not in arm:
        return ''

    def mark(value, limit):
        return '{:.0f}{}'.format(value, ' (!)' if value > limit else '')
    return ('、手首 (ロボットの手に固定) 掌屈 {}・背屈 {}・撓屈 {}・尺屈 {}・'
            'ひねり {} 度'.format(
                mark(arm['flexion_deg'], HUMAN_WRIST_FLEX_MAX_DEG),
                mark(arm['extension_deg'], HUMAN_WRIST_FLEX_MAX_DEG),
                mark(arm['radial_deg'], HUMAN_WRIST_RADIAL_MAX_DEG),
                mark(arm['ulnar_deg'], HUMAN_WRIST_ULNAR_MAX_DEG),
                mark(arm['hand_twist_deg'], HUMAN_WRIST_TWIST_MAX_DEG)))


_PALM_FACING_NAMES = dict(inward='人側', forward='前向き', backward='後ろ向き',
                          natural='ひねらない向き')


def lowered_arm_text(transition):
    """移動先の人の腕 (上腕の真下からの角度と掌の向き) の説明。"""
    if 'arm_deg' not in transition:
        return '前腕 {:.0f} 度'.format(transition.get('forearm_deg',
                                                     float('nan')))
    return '上腕 {:.0f} 度・掌は{}'.format(
        transition['arm_deg'],
        _PALM_FACING_NAMES.get(transition['palm_facing'],
                               transition['palm_facing']))


def transition_summary(transition):
    """``plan_transition`` の結果の 1 行の説明。"""
    if not transition['verified']:
        return 'しない ({})'.format(transition['reason'])
    before = transition['placement_before'] or {}
    after = transition['placement_after'] or {}
    text = ('前方ずれ {:+.2f} -> {:+.2f} m、向きのずれ {:+.0f} -> {:+.0f} 度、'
            '方位 {:+.0f} -> {:+.0f} 度、脚 (ankle/knee) {:.0f}/{:.0f} -> '
            '{:.0f}/{:.0f} 度、握りの向き {:.0f} -> {:.0f} 度、人の腕 {} '
            '(移動先の {:.0f}%、窓 ±{} m、{:.2f} 秒)。人の腕の最大 (押し込み時を'
            '除く): {}'.format(
                before.get('front_offset', float('nan')),
                after.get('front_offset', float('nan')),
                before.get('yaw_offset_deg', float('nan')),
                after.get('yaw_offset_deg', float('nan')),
                before.get('bearing_deg', float('nan')),
                after.get('bearing_deg', float('nan')),
                *(transition['legs_before'] + transition['legs_after']),
                transition['turn_before'], transition['turn_after'],
                lowered_arm_text(transition), transition['fraction'] * 100.0,
                transition['x_margin'],
                transition['compute_time'],
                human_arm_text(transition['human_max'])))
    approach = transition.get('approach')
    if approach is not None:
        if approach['n_waypoints']:
            text += ' / 寄せ: 横 {:.2f} -> {:.2f} m ({})'.format(
                approach['lateral_before'], approach['lateral_after'],
                approach['mode'])
        else:
            text += ' / 寄せない ({})'.format(approach['reason'])
    if transition['reason']:
        text += ' / 途中で止めた: {}'.format(transition['reason'])
    return text
