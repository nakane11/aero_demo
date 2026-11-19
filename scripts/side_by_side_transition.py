#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""掌を押し込んだ後、つないだ手を人の体の横へ下ろしながら、ロボットも人と
横並びの位置・姿勢へ移る区間 (横並び移動) を計画する。

- 段階 1: つないだ手を人が肩から腕を下ろした位置 (``HumanArm.lowering``) へ
  動かしながら、台車を人と前後ずれ 0・同じ向きの位置へ動かし上体を起こす
  (移動先は台車も動かすバッチ IK、``solve_goal_candidates``)。腕は押し込み時と
  同じ解の枝に限る。waypoint は人の腕・握りの向き・台車を補間して解き直す。
- 上半身 (body_link) は ``UPPER_BODY_REFERENCE`` (既定は台車の前方向) から
  ``UPPER_BODY_LIMIT_DEG`` 以内、首は人と同じ前方を見る (``forward_gaze_point``)。
  移動先の台車の向きは人の正面から ``GOAL_YAW_MARGIN_DEG`` 以内で、前を向くのは
  前進前のその場回転に任せる。
- 段階 2: 人の腕・掌を動かさず台車だけを人の方へ寄せる (``plan_approach``)。

人の腕は肩を動かさず、手首から先は押し込み時の手に対して剛体 (``HumanArm``)。
各 waypoint で人の腕が届くか (``reach``) などを記録し、届かなくなる手前で止める。

干渉の検証は人体との距離 (``TRANSITION_HUMAN_CLEARANCE``) と自己干渉。ロボット
の差し出す腕の肘から先と、人の差し出した手・前腕は手を合わせているので距離を
見ない。
"""

import collections
import contextlib
import copy
import math
import os
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

# 移動先のバッチ IK の目標の作り方。'per_angle' は上腕の角度ごとに 3 つとも
# 同じ下ろし方 (初期値を 3 倍に使う)、'group' は組の 3 つの角度を 1 バッチで
# 解く。'group' は障害物が目標の間で共有されるので、低い下ろし方の目標も
# 高い下ろし方の上腕を避け、低い下ろし方が解けにくい (10/22 実機試行 2)。
LOWERED_ARM_BATCHES = ('per_angle', 'group')
LOWERED_ARM_BATCH = 'per_angle'

# 'per_angle' のとき、同じ上腕の角度・脇の開きで掌の向きだけが違う下ろし方を、
# PALM_FACING_ORDER の順に最大 LOWERED_TARGET_COUNT 個まとめて 1 回のバッチ IK
# で解き、人ごとのキャッシュ (_person_cache) に置く (試す順は変えない)。上腕は
# 同じなので障害物の共有で低い下ろし方が解けなくなることはない。バッチ IK の
# 時間は初期値の数よりも呼び出しの回数で決まる (384 -> 1152 レーンで 0.20 ->
# 0.33 秒)。目標 1 つあたりの初期値は GOAL_BUNDLE_ATTEMPTS_PER_POSE。まとめ
# ないときもこの数で解き、バッチ IK の形を 1 つにする (形が増えると warmup の
# コンパイルが実機ノードで約 20 秒延びる)。384 (まとめないときの同じ目標 3 つ
# と同じ数) は、確保する計画で横並びまで解けた人の時間が中央値で +0.5 秒延び、
# 成功数は 128 と同じだった (2026-10-23)。
GOAL_BUNDLE_FACINGS = True
GOAL_BUNDLE_ATTEMPTS_PER_POSE = spik.env_int('MFIX_GOAL_BUNDLE_ATTEMPTS_PER_POSE', 128)
# 段階 1 の下ろし方ごとの結果 (経路まで) を人ごとのキャッシュに置き、同じ人の
# 計画で再利用する (実機ノードは、動く前に確保した計画で試した下ろし方を、
# 動作中に裏で解く計画 refine_transition で解き直さない)。
GOAL_REUSE_OUTCOMES = True
# 横並びまで進んだ計画の中で掌が一番低いものより LOWERED_HEIGHT_TOLERANCE を
# 超えて高い下ろし方は、選ばれない (_low_arm_results) ので解かない。
PRUNE_HIGH_LOWERINGS = True
# 下ろした掌で人の手首が目安に収まる握りの向きが 1 つも無い下ろし方は、移動先
# がすべて fix_goal_wrist で捨てられるので、バッチ IK の前に打ち切る。
GOAL_PRECHECK_WRIST = True

# 腕を下ろす途中で上腕を外へ膨らませる角度 [度] (両端の姿勢は変えない)。
# 前腕が人の腰・太ももに近づくのをよける。
LOWERED_BULGE_DEG = 10.0

# 掌の向き 'natural': 前腕をひねらず、腕を下ろす回転だけで動かした向き。
# 終点の掌の法線の外向き成分が LOWERED_NATURAL_MAX_OUTWARD、上下の成分が
# LOWERED_NATURAL_MAX_VERTICAL を超えるときは使わない。
LOWERED_NATURAL_MAX_OUTWARD = 0.3
LOWERED_NATURAL_MAX_VERTICAL = 0.5

# 掌の向きは LOWERED_FACING_ORDER の順に試す (``plan_transition`` の
# ``facing_variants``)。押し込み時からの前腕のひねりがこの角度 [度] を超える
# 向きは試さない。None なら見ない (既定。前腕のひねりは HUMAN_ARM_LIMITS_DEG の
# 絶対的な回内・回外 (LUBA) で判定する。2026-10-27 まで 80 度)。
LOWERED_TWIST_MAX_DEG = None

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
GOAL_ATTEMPTS_PER_POSE = spik.env_int('MFIX_GOAL_ATTEMPTS_PER_POSE', 128)

# 横並び移動・寄せの移動先のバッチ IK (_batch_ik) の反復の上限。既定は 60 で、hover の
# spik.DEFAULT_COLLISION_IK_STOP (80) より少ない。寄せの時間の大半はここ。
# MFIX_SBS_COLLISION_IK_STOP=80 で前の形 (hover と同じ反復) に戻る。
SBS_COLLISION_IK_STOP = spik.env_int('MFIX_SBS_COLLISION_IK_STOP', 60)

# 移動先のコストの重み: 人の正面方向の前方ずれ [m]、向きのずれ [rad]、
# 上体の前後ずれ |ankle + knee| [rad]。
GOAL_FRONT_OFFSET_WEIGHT = 100.0
# 移動先のコストで、台車の人からの横の距離が GOAL_LATERAL_TARGET [m] を超えた
# 分 [m] に掛ける重み (遠くへ離れてから寄せで戻る動きを避ける)。0 で無効。
# 段階 1 の移動先は腕の枝の制限で人の近くの解がほとんど無く、100 でも離れて
# 戻る量が変わらなかったので使わない (MERGE_APPROACH で対処)。
GOAL_LATERAL_WEIGHT = 0.0
GOAL_LATERAL_TARGET = 0.6
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
# 横並び (と寄せ) の間、上半身 (body_link の +x) を UPPER_BODY_REFERENCE の
# 向きから ±UPPER_BODY_LIMIT_DEG に収める。'base' は台車の前方向 (前進前の
# その場回転で台車が前を向いたとき、上半身も前を向く)、'human' は人の正面
# (2026-11-19 まで)。移動先は範囲外の候補を捨て、ずれの二乗に
# UPPER_BODY_WEIGHT [1/rad^2] を掛けたコストを足す。経路の IK は、始点の
# 上半身が範囲内 (UPPER_BODY_START_SLACK_DEG まで超えていても範囲内とみなす)
# なら全体を、範囲外なら移動先だけを範囲に絞る (押し込み姿勢から出る途中の
# 腰ヨーを絞ると、腕が掌に届かず経路が序盤で止まる)。
UPPER_BODY_REFERENCES = ('base', 'human')
UPPER_BODY_REFERENCE = 'base'
UPPER_BODY_LIMIT_DEG = 20.0
UPPER_BODY_WEIGHT = 100.0
UPPER_BODY_START_SLACK_DEG = 3.0

# 動作中に裏で解く計画 (``refine_transition``) の、残りの掌の向きの探索
# (``_plan_transition``) の時間の上限 [秒]。超えたらそれまでの一番良い解で
# 終える。動く前に確保する計画 (``plan_transition``) は、呼び出し元が計画全体
# の締め切り (``solve_palm_ik.PLAN_TIME_LIMIT``) の残りを ``time_limit`` に渡す。
REFINE_TIME_LIMIT = 10.0

# 移動先の台車の向きを人の正面から ±GOAL_YAW_MARGIN_DEG に絞る。
# 寄せ後の横の距離が APPROACH_FALLBACK_LATERAL を超える人だけ
# GOAL_YAW_FALLBACK_DEG まで広げて解き直す (終点は GOAL_YAW_MARGIN_DEG
# に収まるものだけ採る)。None なら広げない。
# 2026-11-19 まで 20 / 30 度 (上半身は人の正面基準)。45 度・上半身を台車の
# 前方向基準にすると、実機 15 試行の計画で寄せ後の横の距離が平均 0.734 ->
# 0.699 m (最大 0.86 -> 0.77 m)、寄せまでの計算が中央 4.1 -> 1.6 秒。代わりに
# 前進前のその場回転が中央 16 -> 26 度 (最大 45 度) に増える。
GOAL_YAW_MARGIN_DEG = 45.0
GOAL_YAW_FALLBACK_DEG = None
APPROACH_FALLBACK_LATERAL = 0.85

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
# 寄せの移動先を、試す窓をすべて合わせた範囲の 1 回のバッチ IK で解き、
# 近い窓から順にその窓に入る候補を試す (窓ごとに解くと寄せ 1 回に 4〜5 回)。
# そのときの目標 1 つあたりの初期値の数 (目標は 3 つとも同じ掌)。
APPROACH_SINGLE_BATCH = True
APPROACH_SINGLE_BATCH_ATTEMPTS = spik.env_int('MFIX_APPROACH_SINGLE_BATCH_ATTEMPTS', 128)
# 寄せながら台車を人の正面へ回す (人の手は動かさず、回転の分は腕で受ける)。
# 移動先の台車の向きを人の正面から ±APPROACH_END_YAW_DEG に絞り、前進前の
# その場回転 (つないだ手が台車のまわりに弧を描いて前へ引かれる) を小さくする。
# 回して寄せられなければ、向きを変えない寄せに戻す。
APPROACH_TURN_TO_FRONT = True
APPROACH_END_YAW_DEG = 10.0

# 前進前のその場回転 (腕は止めたまま台車を人の正面へ回す) の後に、人の肩から
# つないだ手までの距離がこれ [m] を超える移動先は捨てる (人の腕が伸びきって
# 前へ引っ張られる)。None なら見ない。実機 15 試行の計画では、向き ±45° で
# 回転後の距離が最大 0.68 m (0.60 m 超は 2 試行)。
HUMAN_REACH_AFTER_TURN_MAX = 0.60

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
# 関節角が飛んだ区間を、この数に分けて直前の解から解き進め直す (順に試し、
# どれでも飛ぶなら止める)。空なら分けない。waypoint の刻みは
# PLANNED_JOINT_STEP と同じ 20 度なので、IK の解が少しずれるだけで飛びと
# 判定されていた (2026-10-30)。
PATH_JUMP_SUBDIVISIONS = (2, 3, 4)

# waypoint の数は、押し込み姿勢と移動先の関節角の差がこの角度 [rad]
# ごとに 1 区間以上にもなるようにする。
PLANNED_JOINT_STEP = math.radians(20.0)

# 移動先までのうち進めた割合がこれに満たなければ、横並び移動はしない (通らな
# かった扱い。裏の計画では確保した経路へ、どちらも無ければ横並び移動をしない)。
# ユーザー決定 (11/17): 移動先まで最後まで (1.0) 解けた経路だけ通ったとする
# (develop bfc768f の J23 の 2、UNIFIED_J23_FULL と同じ)。確保 (plan_transition)
# と裏の計画 ① (refine_transition) の両方に効く。``MFIX_SBS_FULL=0`` で前の形
# (0.3 進めば途中までで実行する)。
SBS_FULL = os.environ.get('MFIX_SBS_FULL', '1') == '1'
SBS_PARTIAL_MIN_FRACTION = 0.3  # 前の形の MIN_FRACTION
MIN_FRACTION = 1.0 if SBS_FULL else SBS_PARTIAL_MIN_FRACTION

# 人の腕の目安 (表示・ログ用、超えても止めない。reach <= 1 だけ止める)。
# 肘は LUBA の最後の段階の始まり。手首の曲げ (合成) と押し込み時からの
# ひねりは LUBA に対応する値が無いので印を付けない (None)。
HUMAN_ELBOW_FLEX_MAX_DEG = 120.0
HUMAN_WRIST_BEND_MAX_DEG = None
HUMAN_TWIST_CHANGE_MAX_DEG = None

# --- LUBA (Kee & Karwowski 2001) の立位の不快度スコア --------------------------
# 人の腕・手首の制約 (上限) はこの表だけから決める
# (docs/human_joint_angle_references.md)。成分ごとに (正の側, 負の側) の段階
# ((段階の終わりの角度 [度], スコア), ...)。正負は ``human_arm_joint_angles``・
# ``human_wrist_angles`` と同じ。None は段階が無い側 (正の側の表で数える)。
_INF = float('inf')
LUBA_STANDING_SCORES = collections.OrderedDict([
    ('shoulder_flexion', (((45, 1), (90, 3), (150, 6), (_INF, 11)),   # 屈曲
                          ((20, 1), (45, 3), (60, 6), (_INF, 10)))),  # 伸展
    ('shoulder_abduction', (((30, 1), (90, 3), (_INF, 7)),            # 外転
                            ((10, 1), (30, 2), (_INF, 8)))),          # 内転
    ('shoulder_rotation', (((10, 1), (30, 2), (_INF, 5)),             # 外旋
                           ((30, 1), (90, 2), (_INF, 5)))),           # 内旋
    ('elbow_flexion', (((45, 1), (120, 3), (_INF, 5)), None)),
    ('pronation', (((70, 2), (_INF, 7)),                              # 回内
                   ((90, 2), (_INF, 7)))),                            # 回外
    ('wrist_flexion', (((20, 1), (60, 2), (_INF, 5)),                 # 掌屈
                       ((20, 1), (45, 2), (_INF, 7)))),               # 背屈
    ('wrist_deviation', (((10, 1), (30, 3), (_INF, 7)),               # 撓屈
                         ((10, 1), (20, 3), (_INF, 6)))),             # 尺屈
])


def luba_limit(key, positive):
    """成分 ``key`` の ``positive`` (正/負) の側の上限 [度]: LUBA の最後の
    段階 (可動域の端を含む段階) の始まり。段階が無い側は None。"""
    stages = LUBA_STANDING_SCORES[key][0 if positive else 1]
    return None if stages is None else float(stages[-2][0])

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
# 手首の目安も LUBA の最後の段階の始まり (撓屈 30・尺屈 20・掌屈 60・背屈 45)。
HUMAN_WRIST_RADIAL_MAX_DEG = luba_limit('wrist_deviation', True)
HUMAN_WRIST_ULNAR_MAX_DEG = luba_limit('wrist_deviation', False)
# study/sbs-comfort: 尺屈の上限を環境変数で緩める (SBS_ULNAR_MAX [度])。
_LUBA_WRIST_ULNAR_MAX_DEG = HUMAN_WRIST_ULNAR_MAX_DEG
if os.environ.get('SBS_ULNAR_MAX'):
    HUMAN_WRIST_ULNAR_MAX_DEG = float(os.environ['SBS_ULNAR_MAX'])
HUMAN_WRIST_FLEX_MAX_DEG = luba_limit('wrist_flexion', True)
HUMAN_WRIST_TWIST_MAX_DEG = LOWERED_TWIST_MAX_DEG
HUMAN_TURN_CAP_DEG = 20.0
# 押し込み時にすでにこれ以上曲がっている成分は、押し込み時の値 + この余裕
# [度] までを許す (骨格・掌の推定の手首の曲げは誤差が大きく、押し込み直後で
# 止まらないように)。
HUMAN_WRIST_PRESS_SLACK_DEG = 5.0
# 背屈の目安 [度] (None なら掌屈と同じ HUMAN_WRIST_FLEX_MAX_DEG)。
HUMAN_WRIST_EXT_MAX_DEG = luba_limit('wrist_flexion', False)

# --- 人の肩・肘・前腕 (腕を下ろす動き HumanArm.lowering) --------------------
# 肩の屈曲 (前へ上げる)・伸展 (後ろへ引く)・外転・内転・外旋・内旋、肘の屈曲、
# 前腕の回内・回外 (``human_arm_joint_angles``)。人の肩・肘・掌の向きは下ろし方
# (``HumanArm.lowering_pose``) だけで決まり、ロボットの姿勢に依らない (ロボット
# の握りの向きは掌の法線まわりなので前腕のひねりも変えない) ので、
# HUMAN_ARM_CHECK が 'on' なら、腕を下ろす途中 (HUMAN_ARM_CHECK_SAMPLES 点) で
# 目安を超える下ろし方をバッチ IK の前に外す。押し込み時にすでに超えている
# 成分は、手首と同じく押し込み時の値 + HUMAN_ARM_PRESS_SLACK_DEG までを、腕を
# 下ろす割合 s に比例して減らしながら許す。目安は LUBA (Kee & Karwowski 2001)
# で可動域の端を含む最後の段階の始まり (可動域のおよそ 3/4、不快度の増え方が
# 一番大きい境目。docs/human_joint_angle_references.md)。
HUMAN_ARM_CHECKS = ('on', 'off')
HUMAN_ARM_CHECK = 'on'
HUMAN_ARM_CHECK_SAMPLES = 10
HUMAN_ARM_PRESS_SLACK_DEG = 5.0
# (成分, (負の側の目安, 正の側の目安))。None は見ない。肩 伸展 60 / 屈曲 150、
# 内転 30 / 外転 90、内旋 90 / 外旋 30、肘 120、回外 90 / 回内 70。
HUMAN_ARM_LIMITS_DEG = collections.OrderedDict(
    (key, (luba_limit(key, False), luba_limit(key, True)))
    for key in ('shoulder_flexion', 'shoulder_abduction', 'shoulder_rotation',
                'elbow_flexion', 'pronation'))
# study/sbs-comfort: 回内の上限を環境変数で緩める (SBS_PRONATION_MAX [度])。
_LUBA_ARM_LIMITS_DEG = collections.OrderedDict(HUMAN_ARM_LIMITS_DEG)
if os.environ.get('SBS_PRONATION_MAX'):
    HUMAN_ARM_LIMITS_DEG['pronation'] = (
        HUMAN_ARM_LIMITS_DEG['pronation'][0],
        float(os.environ['SBS_PRONATION_MAX']))

# --- 人の項 (develop の人の楽さの指標、study/sbs-comfort) --------------------
# develop の人の項 (sbs_human_term = develop の goal_ik/human_palm.py の写し) の
# 11 成分の 2 乗和を、人の掌 (位置・人の手の向き) から求めて使う。
# SBS_COMFORT_SELECT: 計画を選ぶ鍵 (_result_key) で、寄せ後の横の距離の刻み
#   ('tie' 2 cm、'bucket5' 5 cm) が同じなら人の項の小さい方 ('off' は今まで)。
# SBS_COMFORT_ORDER=1: 下ろし方・掌の向きを、下ろしきった掌の人の項の小さい順
#   に試す (今は上腕の低い組から、決まった向きの順)。
# SBS_COMFORT_GATE: 上限を緩めたとき (SBS_ULNAR_MAX・SBS_PRONATION_MAX)、LUBA の
#   上限を超える姿勢は人の項がこの値以下のときだけ許す (既定 inf = 見ない)。
# 11/20 ユーザー決定: oskel (ORDER=1・SELECT=tie・ARM=skeleton) を既定にする。
# 前の形は SBS_COMFORT_ORDER=0 SBS_COMFORT_SELECT=off。
COMFORT_SELECT = os.environ.get('SBS_COMFORT_SELECT', 'tie')
COMFORT_ORDER = os.environ.get('SBS_COMFORT_ORDER', '1') == '1'
COMFORT_GATE = float(os.environ.get('SBS_COMFORT_GATE', 'inf'))
# SBS_COMFORT_ARM: 人の項の関節角をどの腕で測るか。'develop' (既定) は develop と
#   同じく掌から求め直す (手首は掌から 4.1 cm、肘は 2 リンクの規則)。'skeleton'
#   (既定) は ctb の ``HumanArm`` の肘・手首 (下ろし方で決まる腕) をそのまま使う。
COMFORT_ARM = os.environ.get('SBS_COMFORT_ARM', 'skeleton')
_COMFORT_PERSON = {}


def comfort_total(joints, hand, palm_pos, hand_rot, elbow=None, wrist=None,
                  arm_mode=None):
    """人の掌 ``palm_pos``・人の手の向き ``hand_rot`` (列が指先・法線・指先 x
    法線) の人の項 (develop の 11 成分) の ``(2 乗和, 成分の配列)``。
    ``arm_mode`` (既定 ``COMFORT_ARM``) が 'skeleton' で ``elbow``・``wrist``
    (世界座標) があれば、その腕で関節角を測る。"""
    import sbs_human_term as ht
    key = (id(joints), hand)
    entry = _COMFORT_PERSON.get(key)
    if entry is None or entry[0] is not joints:
        p = {k: np.asarray(v, dtype=np.float64).reshape(-1)
             for k, v in ht.person_values(joints, hand).items()}
        p['palm_active'] = np.ones(1)
        if len(_COMFORT_PERSON) > 256:
            _COMFORT_PERSON.clear()
        entry = (joints, p, ht.constants())
        _COMFORT_PERSON[key] = entry
    _, p, consts = entry
    pos, rot = ht.canonical(np, np.asarray(palm_pos, dtype=np.float64),
                            np.asarray(hand_rot, dtype=np.float64), p)
    elbow_n = wrist_n = None
    if ((arm_mode or COMFORT_ARM) == 'skeleton' and elbow is not None
            and wrist is not None):
        basis = p['human_basis'].reshape(3, 3)
        length = p['human_length'][0]
        elbow_n = basis @ (np.asarray(elbow, dtype=np.float64)
                           - p['human_shoulder']) / length
        wrist_n = basis @ (np.asarray(wrist, dtype=np.float64)
                           - p['human_shoulder']) / length
    e = np.asarray(ht.errors(np, pos, rot, p['human_ratio'][0], consts,
                             elbow_n, wrist_n))
    return float(np.sum(e * e)), e


def comfort_hand_rot(palm_pos, palm_rot, turn_deg, press_turn, robot_arm):
    """掌 (``HumanArm`` の掌の位置・回転) に握りの向き ``turn_deg`` で合わせた
    ロボットの手に、押し込み時の握りの向き ``press_turn`` で付いた人の手の
    向き (``wrist_angles_for_turns`` と同じ)。"""
    palm = palm_from_frame(palm_pos, palm_rot)
    return hand_on_robot(spik.palm_target_rot(palm, turn_deg, robot_arm),
                         press_turn, robot_arm)


def waypoint_comfort(joints, hand, wp, press_turn, robot_arm, arm_mode=None):
    """横並び移動の waypoint ``wp`` の人の項 (2 乗和, 成分)。"""
    arm = wp['human_arm']
    return comfort_total(joints, hand, arm['palm_position'], comfort_hand_rot(
        arm['palm_position'], np.asarray(arm['palm_rot']), wp['turn_deg'],
        press_turn, robot_arm), arm.get('elbow'), arm.get('wrist'), arm_mode)


def result_comfort(result, joints):
    """横並びまで進んだ計画 ``result`` の終点の人の項 (2 乗和、無ければ None)。"""
    if not (result.get('verified') and result.get('waypoints')
            and result.get('robot_arm') and 'turn_before' in result):
        return None
    wp = result['waypoints'][-1]
    if 'human_arm' not in wp:
        return None
    return waypoint_comfort(joints, result['hand'], wp, result['turn_before'],
                            result['robot_arm'])[0]


def lowering_comfort(human_arm, lowering):
    """下ろし方 ``lowering`` の、下ろしきった掌 (握りの向きは押し込み時のまま、
    手首は下ろし方のとおり) の人の項の 2 乗和。"""
    elbow = wrist = None
    if COMFORT_ARM == 'skeleton':
        elbow, wrist = human_arm.lowering_pose(lowering, 1.0)[:2]
    return comfort_total(human_arm.joints, human_arm.hand,
                         lowering['palm'][0], lowering['palm'][1],
                         elbow, wrist)[0]


_HUMAN_ARM_LABELS = dict(
    shoulder_flexion=('肩の屈曲', '肩の伸展'),
    shoulder_abduction=('肩の外転', '肩の内転'),
    shoulder_rotation=('肩の外旋', '肩の内旋'),
    elbow_flexion=('肘の屈曲', '肘の伸展'),
    pronation=('前腕の回内', '前腕の回外'))
# 肘の曲げがこの角度 [度] 未満なら、上腕の回旋と前腕の回内・回外を見分け
# られない (肩の回旋は None、回内・回外は上腕の回旋を含めた値) とする。
HUMAN_ARM_ROTATION_MIN_ELBOW_DEG = 20.0

# --- 人の前腕・手の甲を障害物にする ----------------------------------------
# 移動先のバッチ IK と検証で、人の前腕と手の甲側 (``spik.forearm_obstacles``、
# 掌の接触面は含まない) を、ロボットの差し出す腕から OFFERED_HAND_CLEARANCE 離す。
# 検証は肘・前腕・手首 (``spik.FOREARM_CHECK_LINKS``)、バッチ IK は前腕・手首
# (``TRANSITION_FOREARM_LINKS``、precontact の最適化と同じく肘は入れない)。
# FOREARM_PRESS_ALLOWANCE なら、押し込み時にすでにそれより近い分は、腕を下ろす
# 割合 s に比例して減らしながら許す (s=1 で OFFERED_HAND_CLEARANCE)。
HUMAN_FOREARM_OBSTACLE = True
FOREARM_PRESS_ALLOWANCE = True
# 押し込み時の距離から許す余裕 [m] (IK の解の揺れで最初の waypoint が押し込み
# 時よりわずかに近づいても止めないように)。
FOREARM_PRESS_SLACK = 0.005
HUMAN_FINGER_LENGTH = spik.HUMAN_FINGER_LENGTH
TRANSITION_FOREARM_LINKS = ('{}_forearm_link', '{}_hand_yaw_link')
# バッチ IK で前腕・手の甲との距離を OFFERED_HAND_CLEARANCE にする方法。
# 'shrink': 障害物を共通の判定距離 (DEFAULT_COLLISION_MARGIN) との差だけ細らせる
# (手の甲は半径 2cm で削る量 4cm より細く点になるので、側面は約 4cm で効く)。
# 'pair': 細らせず、TRANSITION_FOREARM_LINKS とのペアだけ判定距離を
# FOREARM_PAIR_MARGIN にする (skrobot の collision_pair_margins)。
FOREARM_MARGIN_MODES = ('shrink', 'pair')
FOREARM_MARGIN_MODE = 'pair'
FOREARM_PAIR_MARGIN = spik.OFFERED_HAND_CLEARANCE


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


def wrist_limits(press_angles, s=1.0, luba=False):
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
    extension = (HUMAN_WRIST_FLEX_MAX_DEG if HUMAN_WRIST_EXT_MAX_DEG is None
                 else HUMAN_WRIST_EXT_MAX_DEG)
    limits = dict(
        flexion=bounds(press_angles['flexion_deg'], extension,
                       HUMAN_WRIST_FLEX_MAX_DEG),
        deviation=bounds(press_angles['deviation_deg'],
                         _LUBA_WRIST_ULNAR_MAX_DEG if luba
                         else HUMAN_WRIST_ULNAR_MAX_DEG,
                         HUMAN_WRIST_RADIAL_MAX_DEG))
    if HUMAN_WRIST_TWIST_MAX_DEG is not None:
        limits['twist'] = bounds(press_angles['twist_deg'],
                                 HUMAN_WRIST_TWIST_MAX_DEG,
                                 HUMAN_WRIST_TWIST_MAX_DEG)
    return limits


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


def _signed_angle(a, b, axis):
    """``axis`` まわりに ``a`` から ``b`` へ回る角度 [度] (どちらも ``axis``
    に垂直な成分で測る)。"""
    axis = _unit(axis)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - np.dot(a, axis) * axis
    b = b - np.dot(b, axis) * axis
    return math.degrees(math.atan2(float(np.dot(np.cross(a, b), axis)),
                                   float(np.dot(a, b))))


def human_arm_joint_angles(shoulder, elbow, wrist, normal, forward, outward):
    """人の腕 (肩 ``shoulder``・肘 ``elbow``・手首 ``wrist``、掌の法線
    ``normal``) の関節角度 [度] の dict。``forward``/``outward`` は人の正面と
    差し出した腕の側の外向き (水平)。

    - ``shoulder_flexion``: 上腕を前後の面に投影した真下からの角度 (前へ上げる
      屈曲が正、後ろへ引く伸展が負)。上腕が前後の面からほぼ外れる (外転 78 度
      超) ときは 0。
    - ``shoulder_abduction``: 上腕の前後の面からの角度 (外へ開く外転が正、体の
      前を内へ横切る内転が負)。
    - ``shoulder_rotation``: 上腕の回旋 (外旋が正)。基準は、腕を真下に下ろし
      肘を曲げた前腕が前を向く向きを、真下から上腕へ最短で回したもの。肘の
      曲げが ``HUMAN_ARM_ROTATION_MIN_ELBOW_DEG`` 未満なら None。
    - ``elbow_flexion``: 肘の曲げ (伸ばしきりが 0)。
    - ``pronation``: 前腕の回内 (掌を下へ向ける側が正、回外が負)。基準は肘を
      曲げて親指を上にした向き (掌が体の内側を向く)。肘の曲げが小さいときは、
      上腕の回旋を含めた (上の肩の回旋の基準からの) 値。
    """
    shoulder, elbow, wrist = (np.asarray(v, dtype=np.float64)
                              for v in (shoulder, elbow, wrist))
    up = np.array([0.0, 0.0, 1.0])
    f = _unit([forward[0], forward[1], 0.0])
    o = np.asarray(outward, dtype=np.float64)
    o = _unit(o - np.dot(o, f) * f - np.dot(o, up) * up)
    ua = _unit(elbow - shoulder)
    fa = _unit(wrist - elbow)
    sagittal = math.hypot(float(np.dot(ua, f)), float(np.dot(ua, up)))
    flexion = (math.degrees(math.atan2(float(np.dot(ua, f)),
                                       -float(np.dot(ua, up))))
               if sagittal > 0.2 else 0.0)
    abduction = math.degrees(math.asin(float(np.clip(np.dot(ua, o),
                                                     -1.0, 1.0))))
    elbow_flex = _angle_between(ua, fa)
    # 真下から上腕へ最短で回す回転で、腕を下ろしたときの基準を運ぶ。
    carry = _rotation_between(_DOWN, ua)
    neutral_fore = carry @ f
    # 外旋 (前腕が外へ回る) の向き: 腕を下ろした基準で f から o へ回る向き。
    ext_sign = 1.0 if float(np.dot(np.cross(f, o), _DOWN)) > 0.0 else -1.0
    # 親指を上にした掌の法線は体の内側 (-o) を向く: 肘を曲げた腕の面の法線
    # cross(前腕, 上腕) の内側向きの符号。
    medial = 1.0 if float(np.dot(np.cross(f, _DOWN), -o)) > 0.0 else -1.0
    normal = np.asarray(normal, dtype=np.float64)
    rotation = None
    if elbow_flex >= HUMAN_ARM_ROTATION_MIN_ELBOW_DEG:
        rotation = ext_sign * _signed_angle(neutral_fore, fa, ua)
        ref = medial * np.cross(fa, ua)
    else:
        # 上腕の回旋の基準のまま肘を伸ばした向きで、親指を上にした法線。
        ref = _rotation_between(ua, fa) @ (medial * np.cross(neutral_fore,
                                                             ua))
    # 回内は、前腕の向きまわりに法線が親指を上にした向きから掌を下へ向ける
    # 側へ回る角度 (右腕・前を向いた前腕では -x まわり)。
    pronation = -medial * _signed_angle(ref, normal, fa)
    return collections.OrderedDict([
        ('shoulder_flexion', flexion), ('shoulder_abduction', abduction),
        ('shoulder_rotation', rotation), ('elbow_flexion', elbow_flex),
        ('pronation', pronation)])


def arm_limits(press_angles, s=1.0):
    """押し込み時の腕の関節角度 ``press_angles`` から、腕を下ろす動きの割合
    ``s`` で許す範囲の dict (成分ごとの (下限, 上限) [度]、見ない側は
    ``None``)。押し込み時にすでに目安を超えている側は、押し込み時の値 +
    ``HUMAN_ARM_PRESS_SLACK_DEG`` までの超過を ``s`` に比例して減らしながら
    許す (``wrist_limits`` と同じ)。"""
    slack = HUMAN_ARM_PRESS_SLACK_DEG
    keep = 1.0 - min(1.0, max(0.0, s))
    limits = {}
    for key, (negative, positive) in HUMAN_ARM_LIMITS_DEG.items():
        value = press_angles.get(key)
        lo = hi = None
        if negative is not None:
            lo = -negative
            if value is not None:
                lo -= keep * max(0.0, -negative - (value - slack))
        if positive is not None:
            hi = positive
            if value is not None:
                hi += keep * max(0.0, value + slack - positive)
        limits[key] = (lo, hi)
    return limits


def arm_violation(angles, limits):
    """腕の関節角度 ``angles`` が範囲 ``limits`` を超えていればその説明、
    超えていなければ None。"""
    for key, (lo, hi) in limits.items():
        value = angles.get(key)
        if value is None:
            continue
        if (hi is not None and value > hi) or (lo is not None and value < lo):
            return '人の{} {:.0f} 度'.format(
                _HUMAN_ARM_LABELS[key][0 if value > 0 else 1], abs(value))
    return None


forearm_obstacles = spik.forearm_obstacles


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
        self.press_joints = self.joint_angles(press_elbow, self.wrist0,
                                              self.palm_rot0)
        # 腕を下ろす動き (``lowering``・``lowering_pose``・``lowering_problem``)
        # が使う値の鍵 (``LOWERING_REUSE``)。
        self._reuse_sig = (hand,) + tuple(
            _arm_value_key(v) for v in (
                self.shoulder, self.wrist0, self.upper, self.fore,
                self.outward, self.forward, self.palm_pos0, self.palm_rot0))

    def joint_angles(self, elbow, wrist, palm_rot):
        """肘 ``elbow``・手首 ``wrist``・掌 ``palm_rot`` の腕の関節角度
        (``human_arm_joint_angles``)。"""
        return human_arm_joint_angles(self.shoulder, elbow, wrist,
                                      np.asarray(palm_rot)[:, 1],
                                      self.forward, self.outward)

    def _goal_key(self, goal):
        return (self._reuse_sig,) + tuple(
            _arm_value_key(goal[name]) for name in (
                'start_frame', 'frame', 'start_flex', 'flex', 'twist'))

    def lowering_problem(self, lowering, samples=None):
        """下ろし方 ``lowering`` の途中 (``samples`` 点、既定
        ``HUMAN_ARM_CHECK_SAMPLES``) で腕の関節角度が目安 (``arm_limits``) を
        超えればその説明、超えなければ None。"""
        samples = samples or HUMAN_ARM_CHECK_SAMPLES
        if LOWERING_REUSE:
            return _lowering_reused(
                ('problem', self._goal_key(lowering), samples, LOWERING_BATCH),
                lambda: self._lowering_problem(lowering, samples),
                lambda v: v)
        return self._lowering_problem(lowering, samples)

    def _lowering_problem(self, lowering, samples):
        if LOWERING_BATCH:
            return self._batch_lowering_problem(lowering, samples)
        for k in range(1, samples + 1):
            s = k / float(samples)
            elbow, wrist, _, rot = self.lowering_pose(lowering, s)
            problem = arm_violation(self.joint_angles(elbow, wrist, rot),
                                    arm_limits(self.press_joints, s))
            if problem is not None:
                return problem
        return None

    def _batch_lowering_problem(self, lowering, samples):
        """``lowering_problem`` の、標本の点をまとめて計算する形
        (``LOWERING_BATCH``)。式は 1 点ずつの形と同じで、点ごとの積を配列の積に
        しただけ。全ての点を計算してから、前の点から順に目安を見る。"""
        s_values = [k / float(samples) for k in range(1, samples + 1)]
        elbow, wrist, rot = self._batch_lowering_rot(lowering, s_values)
        palm_rot = rot @ self.palm_rot0
        angles = _batch_joint_angles(self.shoulder, elbow, wrist,
                                     palm_rot[:, :, 1], self.forward,
                                     self.outward)
        for s, ang in zip(s_values, angles):
            problem = arm_violation(ang, arm_limits(self.press_joints, s))
            if problem is not None:
                return problem
        return None

    def _batch_lowering_frames(self, goal, s_values):
        """``lowering_pose`` の上腕の組 (膨らませた後) の上腕の向き ua・蝶番軸
        n ((S, 3)) と肘の曲げ (S 個) を点ごとにまとめたもの。"""
        s = np.asarray(s_values, dtype=np.float64)
        frames = _lowering_slerp_of(goal['start_frame'], goal['frame'])(
            s).as_matrix()
        ua = frames[:, :, 0]
        n = frames[:, :, 1]
        if LOWERED_BULGE_DEG > 0.0:
            bulge = math.tan(math.radians(LOWERED_BULGE_DEG))
            coef = np.array([math.sin(math.pi * v) * bulge for v in s_values])
            ua = _unit_rows(ua + coef[:, None] * self.outward)
            n = _unit_rows(n - _dot_rows(n, ua)[:, None] * ua)
        flex = [goal['start_flex'] + (goal['flex'] - goal['start_flex']) * v
                for v in s_values]
        return ua, n, flex

    def _batch_lowering_rot(self, goal, s_values):
        """``lowering_pose(goal, s)`` の肘・手首と手の回転 (``_arm`` の
        ``rot``、掌の回転行列は ``rot @ palm_rot0``) を ``s_values`` の点ごとに
        まとめたもの ((S, 3)・(S, 3)・(S, 3, 3))。"""
        ua, n, flex = self._batch_lowering_frames(goal, s_values)
        ramp = [min(1.0, max(0.0, (v - LOWERED_TWIST_START)
                             / max(1.0 - LOWERED_TWIST_START, 1e-6)))
                for v in s_values]
        twist = np.array([goal['twist'] * r for r in ramp])
        straighten = np.array([LOWERED_WRIST_STRAIGHTEN * v
                               for v in s_values])
        # ``_arm`` と同じ。出発の腕の部分は ``_arm`` の使い回しと同じ鍵・同じ値。
        start_frame = np.asarray(goal['start_frame'], dtype=np.float64)
        start_flex = goal['start_flex']

        def start_parts():
            ua0, n0 = start_frame[:, 0], start_frame[:, 1]
            fa0 = (math.cos(start_flex) * ua0
                   + math.sin(start_flex) * np.cross(n0, ua0))
            fore0_ = _arm_frame(fa0, n0)
            fingers = fore0_.T @ self.palm_rot0[:, 0]
            bend_ = _rotation_between(fingers, np.array([1.0, 0.0, 0.0]))
            return fore0_, Rotation.from_matrix(bend_).as_rotvec()

        if LOWERING_CACHE:
            fore0, bend0 = _lowering_cached(
                ('arm0', start_frame.tobytes(), float(start_flex),
                 self.palm_rot0.tobytes()), start_parts)
        else:
            fore0, bend0 = start_parts()
        cos_f = np.array([math.cos(v) for v in flex])
        sin_f = np.array([math.sin(v) for v in flex])
        fa = cos_f[:, None] * ua + sin_f[:, None] * np.cross(n, ua)
        fore = np.stack([fa, n, np.cross(fa, n)], axis=2)
        elbow = self.shoulder + self.upper * ua
        wrist = elbow + self.fore * fa
        bend = bend0[None, :] * straighten[:, None]
        rot = (Rotation.from_rotvec(fa * twist[:, None]).as_matrix() @ fore
               @ Rotation.from_rotvec(bend).as_matrix() @ fore0.T)
        return elbow, wrist, rot

    def batch_lowering_palms(self, goal, s_values):
        """``lowering_pose(goal, s)`` の掌の位置 (S, 3) と回転行列 (S, 3, 3) を
        ``s_values`` の点ごとにまとめて計算したもの。式は 1 点ずつの形と同じ。"""
        _, wrist, rot = self._batch_lowering_rot(goal, s_values)
        offset = self.palm_pos0 - self.wrist0
        palm = wrist + np.stack([r @ offset for r in rot])
        return palm, rot @ self.palm_rot0

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
            twist_change_deg=twist,
            joints=dict(self.joint_angles(elbow, wrist, palm_rot)))

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

        def start_parts():
            _, fore0_ = forearm(start_frame, start_flex)
            # 押し込み時の前腕の座標系で、指先を前腕の向き (+x) へ回す回転。
            fingers = fore0_.T @ self.palm_rot0[:, 0]
            bend_ = _rotation_between(fingers, np.array([1.0, 0.0, 0.0]))
            return fore0_, Rotation.from_matrix(bend_).as_rotvec()

        if LOWERING_CACHE:
            # 出発の腕 (押し込み時) だけで決まる部分は、同じ値の入力なら同じ
            # 結果なので使い回す。
            fore0, bend0 = _lowering_cached(
                ('arm0', np.asarray(start_frame, dtype=np.float64).tobytes(),
                 float(start_flex), self.palm_rot0.tobytes()), start_parts)
        else:
            fore0, bend0 = start_parts()
        fa, fore = forearm(frame, flex)
        elbow = self.shoulder + self.upper * frame[:, 0]
        wrist = elbow + self.fore * fa
        bend = bend0 * straighten
        rot =(Rotation.from_rotvec(fa * twist).as_matrix() @ fore
               @ Rotation.from_rotvec(bend).as_matrix() @ fore0.T)
        return (elbow, wrist, wrist + rot @ (self.palm_pos0 - self.wrist0),
                rot @ self.palm_rot0)

    def lowering(self, arm_deg, abduction_deg=None, facing=None,
                 elbow_deg=None):
        """上腕を真下から前へ ``arm_deg`` 度上げ、外へ ``abduction_deg``
        (既定 ``LOWERED_ARM_ABDUCTION_DEG``) 開き、肘を ``elbow_deg``
        (既定 ``LOWERED_ELBOW_FLEX_DEG``) 曲げた腕へ下ろす動き。掌は ``facing``
        (既定は前腕のひねりが一番少ない向き、'phi<角度>' なら後ろ向きから
        人側へその角度回した向き) にする。
        ``lowering_pose`` に渡す dict (``palm``: 下ろしきった掌の ``(位置,
        回転行列)`` を含む) を返す。使えない向きなら None。"""
        if abduction_deg is None:
            abduction_deg = LOWERED_ARM_ABDUCTION_DEG
        if LOWERING_REUSE:
            return _lowering_reused(
                ('lowering', self._reuse_sig) + tuple(
                    _arm_value_key(v) for v in (arm_deg, abduction_deg,
                                                facing, elbow_deg)),
                lambda: self._lowering(arm_deg, abduction_deg, facing,
                                       elbow_deg),
                _copy_lowering)
        return self._lowering(arm_deg, abduction_deg, facing, elbow_deg)

    def _lowering(self, arm_deg, abduction_deg, facing, elbow_deg):
        start_frame, start_flex = self._press_arm_frame()
        th = math.radians(arm_deg)
        ab = math.radians(abduction_deg)
        ua = (math.cos(ab) * (math.cos(th) * _DOWN
                              + math.sin(th) * self.forward)
              + math.sin(ab) * self.outward)
        frame = _arm_frame(ua, self._sagittal_hinge(ua))
        flex = math.radians(LOWERED_ELBOW_FLEX_DEG if elbow_deg is None
                            else elbow_deg)
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
                    elbow_deg=math.degrees(flex),
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
        if LOWERING_REUSE:
            return _lowering_reused(
                ('pose', self._goal_key(goal), _arm_value_key(s)),
                lambda: self._lowering_pose(goal, s),
                lambda v: tuple(a.copy() for a in v))
        return self._lowering_pose(goal, s)

    def _lowering_pose(self, goal, s):
        frame = _lowering_slerp(goal['start_frame'], goal['frame'], s)
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

# 腕を下ろす動き (``HumanArm.lowering_pose``・``_arm``) の、同じ両端の Slerp と
# 出発の腕だけで決まる部分を使い回す ('0' で毎回作る前の形)。値は毎回作るのと
# 同じ (同じ入力の同じ計算)。鍵は入力の値 (バイト列)。develop の Q2
# (UNIFIED_Q2_LOWERING_CACHE、04d89bf) と同じ。
LOWERING_CACHE = os.environ.get('MFIX_LOWERING_CACHE', '1') == '1'
_LOWERING_CACHE = {}
_LOWERING_CACHE_MAX = 4096


def _lowering_cached(key, make):
    value = _LOWERING_CACHE.get(key)
    if value is None:
        value = make()
        if len(_LOWERING_CACHE) >= _LOWERING_CACHE_MAX:
            _LOWERING_CACHE.clear()
        _LOWERING_CACHE[key] = value
    return value


# 腕を下ろす動きの下ろし方 (``HumanArm.lowering``)・途中の腕
# (``lowering_pose``)・人の腕の判定 (``lowering_problem``) を、同じ値の入力なら
# 前の結果を使い回す ('0' で毎回計算する前の形)。横並び移動・寄せは 1 人の
# 計画で ``HumanArm`` を何度も作り直し、同じ押し込み時の腕から同じ 64 通りほどの
# 下ろし方を毎回計算し直すので、人の腕 (肩・押し込み時の手首と掌・腕の長さ・
# 正面・外向き) と下ろし方の値を鍵にして、``HumanArm`` をまたいで使い回す。
# 値は計算し直すのと同じ (同じ入力の同じ計算)。配列は写しを返す。
LOWERING_REUSE = os.environ.get('MFIX_LOWERING_REUSE', '1') == '1'
_LOWERING_REUSE_CACHE = {}
_LOWERING_REUSE_MAX = 20000
_LOWERING_NONE = object()


def _arm_value_key(x):
    """値の鍵 (スカラーは float の 16 進表記 (-0.0 と 0.0 を分ける)、配列は
    型・形・バイト列)。"""
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, (float, int)):
        return float(x).hex()
    a = np.asarray(x)
    return a.dtype.str, a.shape, a.tobytes()


def _lowering_reused(key, make, copy_value):
    value = _LOWERING_REUSE_CACHE.get(key)
    if value is None:
        value = make()
        if value is None:
            value = _LOWERING_NONE
        if len(_LOWERING_REUSE_CACHE) >= _LOWERING_REUSE_MAX:
            _LOWERING_REUSE_CACHE.clear()
        _LOWERING_REUSE_CACHE[key] = value
    return None if value is _LOWERING_NONE else copy_value(value)


def _copy_lowering(goal):
    """``HumanArm.lowering`` の dict の写し (配列も写す)。"""
    out = dict(goal)
    for name in ('start_frame', 'frame'):
        out[name] = np.array(goal[name], copy=True)
    out['palm'] = tuple(np.array(v, copy=True) for v in goal['palm'])
    return out


def _lowering_slerp_of(a, b):
    """``a`` から ``b`` (回転行列) への ``Slerp`` (``LOWERING_CACHE`` なら
    使い回す)。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not LOWERING_CACHE:
        return Slerp([0.0, 1.0], Rotation.from_matrix([a, b]))
    return _lowering_cached(('slerp', a.tobytes(), b.tobytes()),
                            lambda: Slerp([0.0, 1.0],
                                          Rotation.from_matrix([a, b])))


def _lowering_slerp(a, b, s):
    """``Slerp([0, 1], Rotation.from_matrix([a, b]))([s]).as_matrix()[0]``。"""
    return _lowering_slerp_of(a, b)([s]).as_matrix()[0]


# 腕を下ろす動きの人の腕の判定 (``HumanArm.lowering_problem``) を、標本の点
# (10 点) をまとめた配列の計算にする ('0' で 1 点ずつ計算する前の形)。式は同じ。
# develop の Q3 (UNIFIED_Q3_LOWERING_BATCH、b320f2d) と同じ。
LOWERING_BATCH = os.environ.get('MFIX_LOWERING_BATCH', '1') == '1'


def _dot_rows(a, b):
    """行ごとの内積 (S,) (``b`` は (S, 3) か (3,))。"""
    b = np.broadcast_to(b, a.shape)
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def _unit_rows(v):
    """行ごとの ``_unit``。"""
    return v / np.maximum(np.sqrt(_dot_rows(v, v)), 1e-12)[:, None]


def _rotation_between_rows(a, b):
    """行ごとの ``_rotation_between`` (S, 3, 3)。"""
    a, b = _unit_rows(a), _unit_rows(b)
    axis = np.cross(a, b)
    s = np.sqrt(_dot_rows(axis, axis))
    c = _dot_rows(a, b)
    small = s < 1e-9
    angle = np.array([math.atan2(float(si), float(ci))
                      for si, ci in zip(s, c)])
    rotvec = axis / np.where(small, 1.0, s)[:, None] * angle[:, None]
    rotvec[small] = 0.0
    return Rotation.from_rotvec(rotvec).as_matrix()


def _signed_angle_rows(a, b, axis):
    """行ごとの ``_signed_angle`` [度] (S,)。"""
    axis = _unit_rows(axis)
    a = a - _dot_rows(a, axis)[:, None] * axis
    b = b - _dot_rows(b, axis)[:, None] * axis
    y = _dot_rows(np.cross(a, b), axis)
    x = _dot_rows(a, b)
    return np.array([math.degrees(math.atan2(float(yi), float(xi)))
                     for yi, xi in zip(y, x)])


def _batch_joint_angles(shoulder, elbow, wrist, normal, forward, outward):
    """``human_arm_joint_angles`` を点 (行) ごとにまとめて計算したもの (点ごと
    の OrderedDict のリスト)。"""
    shoulder = np.asarray(shoulder, dtype=np.float64)
    up = np.array([0.0, 0.0, 1.0])
    f = _unit([forward[0], forward[1], 0.0])
    o = np.asarray(outward, dtype=np.float64)
    o = _unit(o - np.dot(o, f) * f - np.dot(o, up) * up)
    elbow = np.asarray(elbow, dtype=np.float64)
    ua = _unit_rows(elbow - shoulder)
    fa = _unit_rows(np.asarray(wrist, dtype=np.float64) - elbow)
    uf = _dot_rows(ua, f)
    uu = _dot_rows(ua, up)
    uo = _dot_rows(ua, o)
    cos_elbow = _dot_rows(_unit_rows(ua), _unit_rows(fa))
    carry = _rotation_between_rows(np.broadcast_to(_DOWN, ua.shape), ua)
    neutral_fore = carry @ f
    ext_sign = 1.0 if float(np.dot(np.cross(f, o), _DOWN)) > 0.0 else -1.0
    medial = 1.0 if float(np.dot(np.cross(f, _DOWN), -o)) > 0.0 else -1.0
    normal = np.asarray(normal, dtype=np.float64)
    elbow_flex = [math.degrees(math.acos(float(np.clip(c, -1.0, 1.0))))
                  for c in cos_elbow]
    bent = np.array([e >= HUMAN_ARM_ROTATION_MIN_ELBOW_DEG
                     for e in elbow_flex])
    rotation = ext_sign * _signed_angle_rows(neutral_fore, fa, ua)
    ref = medial * np.cross(fa, ua)
    if not bent.all():
        straight = ~bent
        ref[straight] = (
            _rotation_between_rows(ua[straight], fa[straight])
            @ (medial * np.cross(neutral_fore[straight],
                                 ua[straight]))[:, :, None])[:, :, 0]
    pronation = -medial * _signed_angle_rows(ref, normal, fa)
    out = []
    for k in range(len(ua)):
        sagittal = math.hypot(float(uf[k]), float(uu[k]))
        flexion = (math.degrees(math.atan2(float(uf[k]), -float(uu[k])))
                   if sagittal > 0.2 else 0.0)
        abduction = math.degrees(math.asin(float(np.clip(uo[k], -1.0, 1.0))))
        out.append(collections.OrderedDict([
            ('shoulder_flexion', flexion), ('shoulder_abduction', abduction),
            ('shoulder_rotation', float(rotation[k]) if bent[k] else None),
            ('elbow_flexion', elbow_flex[k]),
            ('pronation', float(pronation[k]))]))
    return out


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
    """上半身 (body_link の +x) の向きのずれ [rad] (反時計回りが正)。
    ``UPPER_BODY_REFERENCE`` が 'base' なら台車 (base_link の +x) から、
    'human' なら人の正面 ``human_yaw`` [rad] から。"""
    x = robot.body_link.worldrot()[:, 0]
    if UPPER_BODY_REFERENCE == 'base':
        b = robot.base_link.worldrot()[:, 0]
        return _wrap(math.atan2(x[1], x[0]) - math.atan2(b[1], b[0]))
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
# _aim_head で、今の首の角度から解いた視線のずれがこれ以下 [rad] なら、首 0 度
# からは解かない。
AIM_HEAD_GOOD_ERROR = math.radians(0.5)


def _aim_head(robot, gaze_point):
    """首だけで、カメラの光軸を ``gaze_point`` へ向ける。今の首の角度と首 0 度
    の両方から解き、視線のずれの小さい方にする (今の首の角度からで
    ``AIM_HEAD_GOOD_ERROR`` 以下なら首 0 度からは解かない)。視線のずれ [rad]
    を返す。"""
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
        if best[0] <= AIM_HEAD_GOOD_ERROR:
            # 今の首の角度から十分向けられたら、首 0 度からは解かない。
            break
    for joint, angle in zip(joints, best[1]):
        joint.joint_angle(angle)
    return best[0]


# 横並び移動の検証で人体 (車輪と脚・足を含む) から空ける距離 [m]。
TRANSITION_HUMAN_CLEARANCE = 0.05

# 横並び移動・寄せの検証 (``TransitionChecker``) で、人体の障害物の円柱
# (``spik.human_body_obstacles``) と人の前腕・手の甲の円柱 (``forearm_obstacles``)
# を、同じ入力なら作り直さず使い回す ('0' で毎回作る前の形)。差し出した腕だけ
# 動かした骨格なので、胴・脚・頭・反対の腕は同じ円柱になる。値は同じ
# (``spik.human_body_obstacles_reused``)。
OBS_REUSE = os.environ.get('MFIX_OBS_REUSE', '1') == '1'

# ``TransitionChecker`` の組 (自己干渉・人体・車輪の経路・前腕) を、同じ
# ``verification_pairs``・手・腕なら前の検証器のものを使い回す ('0' で毎回
# 分け直す前の形)。組は同じ中身で、距離計算の段取り (``spik._pair_plan``・
# 表面の点の塊) も組に覚えられるので、作り直しが減る。
CHECKER_REUSE = os.environ.get('MFIX_CHECKER_REUSE', '1') == '1'
# (id(verification_pairs), hand, robot_arm) -> (verification_pairs, 組...)。
# verification_pairs を持っておくので id は使い回されない。
_CHECKER_PAIRS_CACHE = {}
_CHECKER_PAIRS_CACHE_MAX = 64


def _body_obstacles(joint_positions):
    """検証用の人体の障害物 (``OBS_REUSE`` なら円柱を使い回す)。"""
    if OBS_REUSE:
        return spik.human_body_obstacles_reused(joint_positions)
    return spik.human_body_obstacles(joint_positions)


class TransitionChecker(object):
    """横並び移動の各姿勢の干渉検証 (自己干渉の貫通・人体からの距離)。
    ロボットの差し出す腕の肘から先と、人の差し出した手・前腕
    (``offered_hand_obstacle_indices``) の組は、手を合わせているので
    見ない。"""

    def __init__(self, robot_arm, hand, verification_pairs,
                 clearance=TRANSITION_HUMAN_CLEARANCE,
                 tolerance=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE):
        self.clearance = clearance
        self.tolerance = tolerance
        # 押し込み時の (前腕, 手の甲) までの距離 [m] (``set_press``)。
        self.press_forearm = None
        key = (id(verification_pairs), hand, robot_arm)
        if CHECKER_REUSE:
            cached = _CHECKER_PAIRS_CACHE.get(key)
            if cached is not None and cached[0] is verification_pairs:
                (_, self.self_pairs, self.clearance_pairs, self.base_pairs,
                 self.forearm_pairs) = cached
                return
        self._make_pairs(robot_arm, hand, verification_pairs)
        if CHECKER_REUSE:
            if len(_CHECKER_PAIRS_CACHE) >= _CHECKER_PAIRS_CACHE_MAX:
                _CHECKER_PAIRS_CACHE.clear()
            _CHECKER_PAIRS_CACHE[key] = (
                verification_pairs, self.self_pairs, self.clearance_pairs,
                self.base_pairs, self.forearm_pairs)

    def _make_pairs(self, robot_arm, hand, verification_pairs):
        self.self_pairs = spik.self_collision_pairs(verification_pairs)
        offered = spik.offered_hand_obstacle_indices(hand)
        source = spik.human_clearance_pairs(verification_pairs)
        self.clearance_pairs = None
        if source is not None:
            self.clearance_pairs = spik.split_offered_hand_pairs(
                source, hand, robot_arm)[1]
        # 台車の経路 (``check_base_path``) で見る、車輪と人の脚・足などの組
        # (人の腕は経路の途中で動くので除く)。各姿勢の車輪は clearance_pairs
        # で見る。
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
        self.forearm_pairs = spik.forearm_clearance_pairs(source, robot_arm)

    def set_press(self, robot, arm):
        """``robot`` の今の姿勢 (押し込み姿勢) と押し込み時の人の腕 ``arm``
        で、前腕・手の甲までの距離を測って覚える (``FOREARM_PRESS_ALLOWANCE``)。"""
        self.press_forearm = self.forearm_clearances(
            robot, arm, cull_distance=spik.OFFERED_HAND_CLEARANCE * 2.0)

    def forearm_limits(self, s):
        """腕を下ろす割合 ``s`` で、前腕・手の甲から空ける距離 [m] の組。
        押し込み時にすでに ``OFFERED_HAND_CLEARANCE`` より近い分
        (``FOREARM_PRESS_SLACK`` の余裕を足す) は、``s`` に比例して減らしながら
        許す。"""
        need = spik.OFFERED_HAND_CLEARANCE
        if not FOREARM_PRESS_ALLOWANCE or self.press_forearm is None:
            return need, need
        keep = 1.0 - min(1.0, max(0.0, s))
        return tuple(need - keep * max(0.0, need - (press - FOREARM_PRESS_SLACK))
                     for press in self.press_forearm)

    def forearm_clearances(self, robot, arm,
                           cull_distance=spik.OFFERED_HAND_CLEARANCE):
        """ロボットの差し出す腕の前腕・肘・手首から、人の腕 ``arm``
        (``HumanArm.pose``) の前腕・手の甲までの距離 [m] の (前腕, 手の甲)。
        ``cull_distance`` 以上の距離は下界 (``human_obstacle_clearances``)。"""
        if not self.forearm_pairs:
            return None
        clearances = spik.human_obstacle_clearances(
            robot, self.forearm_pairs,
            (spik.forearm_obstacles_reused if OBS_REUSE
             else forearm_obstacles)(
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
        obstacles = _body_obstacles(joint_positions)
        names = spik.human_obstacle_names()
        for k in range(1, n + 1):
            _place(robot, av, base_path.at(k / float(n)))
            clearances = spik.human_obstacle_clearances(
                robot, self.base_pairs, obstacles,
                cull_distance=self.clearance)
            if not clearances:
                continue
            index = min(clearances, key=clearances.get)
            if clearances[index] < self.clearance:
                return '経路の車輪が人体 ({}) まで {:.3f} m'.format(
                    names[index], clearances[index])
        return None

    def check(self, robot, joint_positions, arm=None):
        """``robot`` の今の姿勢を、人の骨格 ``joint_positions`` に対して
        検証し、問題があればその説明、無ければ ``None`` を返す。人の腕
        ``arm`` (``HumanArm.pose``、腕を下ろす割合 ``s`` 付き、無ければ 1) を
        与え ``HUMAN_FOREARM_OBSTACLE`` なら、前腕・手の甲からも
        ``forearm_limits`` だけ離れているかを見る。"""
        dist, pair = spik.collision_pairs_min_distance(
            robot, self.self_pairs, None, return_pair=True)
        if dist < -self.tolerance:
            return '自己干渉 {:.3f} m ({} x {})'.format(
                -dist, *spik.collision_pair_name(pair))
        if arm is not None and HUMAN_FOREARM_OBSTACLE:
            clearances = self.forearm_clearances(robot, arm)
            if clearances is not None:
                limits = self.forearm_limits(arm.get('s', 1.0))
                for label, value, limit in zip(('前腕', '手の甲'), clearances,
                                               limits):
                    if value < limit:
                        return '人の{}まで {:.3f} m'.format(label, value)
        if not joint_positions or not self.clearance_pairs:
            return None
        clearances = spik.human_obstacle_clearances(
            robot, self.clearance_pairs,
            _body_obstacles(joint_positions),
            cull_distance=self.clearance)
        if clearances:
            index = min(clearances, key=clearances.get)
            if clearances[index] < self.clearance:
                return '人体 ({}) まで {:.3f} m'.format(
                    spik.human_obstacle_names()[index], clearances[index])
        return None


def wrist_problem(human_arm, robot, arm, s, turn_deg):
    """人の手をロボットの手に固定したときの手首の角度 (``robot_hand_wrist``)
    と、``HUMAN_WRIST_CHECK`` で目安を超えていればその説明 (無ければ None)
    の組。``s`` は腕を下ろす動きの割合、``turn_deg`` は今の握りの向き。"""
    angles = human_arm.robot_hand_wrist(robot, arm)
    if HUMAN_WRIST_CHECK == 'hand_fixed':
        problem = wrist_violation(
            angles, wrist_limits(human_arm.press_wrist, s))
        if (problem is None and COMFORT_GATE < float('inf')
                and wrist_violation(angles, wrist_limits(
                    human_arm.press_wrist, s, luba=True)) is not None):
            # 緩めた上限の中で LUBA の上限を超える: 人の項で許すかを決める。
            end = getattr(robot, '{}arm_end_coords'.format(
                human_arm.robot_arm))
            total, _ = comfort_total(
                human_arm.joints, human_arm.hand, arm['palm_position'],
                hand_on_robot(end.worldrot(), human_arm.press_turn,
                              human_arm.robot_arm))
            if total > COMFORT_GATE:
                problem = '人の項 {:.1f} (尺屈 {:.0f} 度)'.format(
                    total, abs(angles['deviation_deg']))
        return angles, problem
    if HUMAN_WRIST_CHECK == 'turn_cap':
        change = math.degrees(_wrap(math.radians(
            turn_deg - human_arm.press_turn)))
        if abs(change) > HUMAN_TURN_CAP_DEG:
            return angles, '握りの向きの変化 {:.0f} 度'.format(abs(change))
    return angles, None


# 移動先の握りの向きを人の手首の目安に収めて解き直すとき、目安の端からさらに
# 内側へ寄せる角度 [度] (IK の回転の許容 5 度で端を超えないように)。
WRIST_TURN_MARGIN_DEG = 5.0


def wrist_angles_for_turns(human_arm, arm, palm, turns):
    """掌 ``palm`` に合わせたまま握りの向きを ``turns`` (配列 [度]) に
    したときの、腕 ``arm`` の人の手首の角度の dict (``flexion``/
    ``deviation``/``twist`` の配列 [度])。"""
    turns = np.asarray(turns, dtype=np.float64)
    # 握りの向きは掌の法線まわりの回転なので、法線と前腕のひねりは
    # 変わらず、指先 (と指先 x 法線) が 2 つの向きの cos・sin の和になる。
    rot0, rot90 = [hand_on_robot(
        spik.palm_target_rot(palm, base, human_arm.robot_arm),
        human_arm.press_turn, human_arm.robot_arm)
        for base in (turns[0], turns[0] + 90.0)]
    angles = human_arm.hand_wrist(arm, rot0)
    d = np.radians(turns - turns[0])[:, None]
    fingers = np.cos(d) * rot0[:, 0] + np.sin(d) * rot90[:, 0]
    side = np.cos(d) * rot0[:, 2] + np.sin(d) * rot90[:, 2]
    radial = side if human_arm.hand == 'R' else -side
    a = _unit(np.asarray(arm['wrist'], dtype=np.float64)
              - np.asarray(arm['elbow'], dtype=np.float64))
    along = fingers @ a
    return dict(
        flexion=np.degrees(np.arctan2(-float(np.dot(a, rot0[:, 1])), along)),
        deviation=np.degrees(np.arctan2(-(radial @ a), along)),
        twist=np.full(len(turns), angles['twist_deg']))


def _wrist_within(values, limits):
    """``wrist_angles_for_turns`` の角度 ``values`` が範囲 ``limits`` に
    収まるか (配列)。"""
    good = np.ones(len(values['flexion']), dtype=bool)
    for key, (lo, hi) in limits.items():
        good &= (values[key] >= lo) & (values[key] <= hi)
    return good


def wrist_ok_turn(human_arm, arm, palm_pos, palm_rot, turn_deg, s=1.0):
    """掌 ``palm_pos``/``palm_rot`` に合わせたまま、人の手首が目安に収まる
    握りの向きのうち ``turn_deg`` に一番近いもの (収まる範囲の端から
    ``WRIST_TURN_MARGIN_DEG`` 内側、収まらなければ端) [度]。無ければ None。"""
    palm = palm_from_frame(palm_pos, palm_rot)
    limits = wrist_limits(human_arm.press_wrist, s)

    def ok(turns):
        """握りの向き ``turns`` (配列) ごとに、手首が目安に収まるか。"""
        turns = np.asarray(turns, dtype=np.float64)
        if HUMAN_WRIST_CHECK == 'turn_cap':
            change = (turns - human_arm.press_turn + 180.0) % 360.0 - 180.0
            return np.abs(change) <= HUMAN_TURN_CAP_DEG
        return _wrist_within(wrist_angles_for_turns(human_arm, arm, palm,
                                                    turns), limits)

    deltas = np.array(sorted(range(-180, 181), key=abs), dtype=np.float64)
    good = np.flatnonzero(ok(turn_deg + deltas))
    if len(good) == 0:
        return None
    delta = deltas[good[0]]
    inner = turn_deg + delta + math.copysign(WRIST_TURN_MARGIN_DEG, delta)
    return (float(inner) if delta and ok([inner])[0]
            else float(turn_deg + delta))


def fix_goal_wrist(robot, robot_arm, human_arm, arm, palm_pos, palm_rot,
                   gaze_point, turn_deg, aim_head=True):
    """移動先の姿勢 (``robot`` の今の姿勢、握りの向き ``turn_deg``) で人の
    手首が目安を超えていれば、収まる握りの向き (``wrist_ok_turn``) に固定して
    解き直す (``aim_head`` なら首だけで ``gaze_point`` へ向け直す)。
    ``(握りの向き, 問題の説明 or None)`` を返す (直せなければ姿勢は元の
    まま)。"""
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
        if gaze_point is not None and aim_head:
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
    リストを返す。``FOREARM_MARGIN_MODE`` が 'shrink' なら距離が
    ``OFFERED_HAND_CLEARANCE`` になるよう細らせる ('pair' ならそのまま)。無い
    目標の分はダミーで埋める (個数は常に ``2 * LOWERED_TARGET_COUNT``)。"""
    if not HUMAN_FOREARM_OBSTACLE:
        return []
    start = len(obstacles)
    for k in range(LOWERED_TARGET_COUNT):
        if k < len(lowered_forearms):
            obstacles += [
                spik.offered_hand_penalty_obstacle(
                    o, spik.DEFAULT_COLLISION_MARGIN)
                if FOREARM_MARGIN_MODE == 'shrink' else o
                for o in forearm_obstacles(*lowered_forearms[k])]
        else:
            obstacles += [spik._dummy_cylinder(spik.HUMAN_FOREARM_RADIUS),
                          spik._dummy_cylinder(spik.HUMAN_HAND_BACK_RADIUS)]
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
    pair_margins = None
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
        forearm_pairs = [(links_by_name[name.format(robot_arm)], i)
                         for name in TRANSITION_FOREARM_LINKS
                         if name.format(robot_arm) in links_by_name
                         for i in forearm_slots]
        if FOREARM_MARGIN_MODE == 'pair':
            # 前腕・手の甲とのペアだけ判定距離を変える (他は共通の
            # DEFAULT_COLLISION_MARGIN)。値は jit の引数なので再コンパイル
            # しない。
            pair_margins = ([None] * len(pairs)
                            + [FOREARM_PAIR_MARGIN] * len(forearm_pairs))
        pairs += forearm_pairs
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
                stop=SBS_COLLISION_IK_STOP,
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
                collision_geometry=spik.DEFAULT_IK_COLLISION_GEOMETRY,
                collision_pair_margins=pair_margins)
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
    が False なら何もしない。``seed_av`` は解ける目標を作るための姿勢。
    初期値の数が違うと別にコンパイルされるので、まとめて解くとき
    (``GOAL_BUNDLE_ATTEMPTS_PER_POSE``・``APPROACH_SINGLE_BATCH_ATTEMPTS``) の
    数でも解く。"""
    if not GOAL_FREE_TURN:
        return
    counts = [GOAL_BUNDLE_ATTEMPTS_PER_POSE if GOAL_BUNDLE_FACINGS
              else attempts_per_pose,
              APPROACH_SINGLE_BATCH_ATTEMPTS if APPROACH_SINGLE_BATCH
              else attempts_per_pose]
    for attempts in sorted(set(counts)):
        _place(robot, seed_av, (0.0, 0.0, 0.0))
        end = getattr(robot, '{}arm_end_coords'.format(robot_arm))
        target = Coordinates(pos=end.worldpos().copy(),
                             rot=end.worldrot().copy())
        _batch_ik(robot, robot_arm, hand, [target] * LOWERED_TARGET_COUNT,
                  seed_av, {}, collision_pairs, base_limits, attempts,
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


def reach_after_turn(robot, robot_arm, base, shoulder):
    """``robot`` (台車 ``base`` = (x, y, yaw)、人の正面が +x の座標系に置いた
    姿勢) を、腕を止めたまま台車の中心まわりに人の正面 (yaw 0) へ回したときの、
    人の肩 ``shoulder`` からつないだ手までの距離 [m] (肩が無ければ 0)。"""
    if shoulder is None:
        return 0.0
    hand = np.asarray(getattr(robot, '{}arm_end_coords'.format(
        robot_arm)).worldpos(), dtype=np.float64)
    c, s = math.cos(-base[2]), math.sin(-base[2])
    dx, dy = hand[0] - base[0], hand[1] - base[1]
    turned = np.array([base[0] + c * dx - s * dy, base[1] + s * dx + c * dy,
                       hand[2]])
    return float(np.linalg.norm(turned - np.asarray(shoulder,
                                                    dtype=np.float64)))


def solve_goal_candidates(robot, robot_arm, hand, goal_palms, seed_av,
                          translated_joints, collision_joints,
                          collision_pairs, base_limits,
                          turn_degs=GOAL_TURN_DEGS,
                          attempts_per_pose=GOAL_ATTEMPTS_PER_POSE,
                          branch_range_deg=GOAL_BRANCH_JOINT_RANGE_DEG,
                          free_turn=False, lowered_arms=(),
                          max_waist_z=None, y_window=None,
                          yaw_margin_deg=None, lowered_forearms=(),
                          per_target=False):
    """下ろした手の候補 ``goal_palms`` (腕の低い順の 3 つの ``(位置,
    回転行列)``) への押し込み目標を、握りの向き ``turn_degs`` ごとに 1 回の
    バッチ IK で解き、移動先の台車・姿勢の候補 (前後は人の立ち位置
    ±``GOAL_X_MARGIN``) を、腕の低い目標から、同じ目標ではコストの安い順に
    返す。``per_target`` なら腰の高さ・指先の向きによる絞り込みを目標ごとに
    行う (目標ごとに別の下ろし方をまとめて解くとき。目標の添字 -> 組の
    リストなら組ごと)。

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
    shoulder = local_joints.get('{}Shoulder'.format(hand))
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
            # 人の正面が +x の座標系なので、人の正面からのずれは yaw そのもの。
            upper_dev = upper_body_deviation(robot, 0.0)
            if abs(upper_dev) > upper_limit:
                continue
            if (HUMAN_REACH_AFTER_TURN_MAX is not None
                    and reach_after_turn(robot, robot_arm, base, shoulder)
                    > HUMAN_REACH_AFTER_TURN_MAX):
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
                    * sum(abs(av[index]) for index, _, _ in wrist)
                    + GOAL_LATERAL_WEIGHT * max(0.0, human_lateral(
                        translated_joints, base, hand) - GOAL_LATERAL_TARGET))
            candidates.append((cost, av, base, turn_deg, palm_index))
            finger_zs.append(float(finger[2]))
            waist_zs.append(waist_height(robot))
    # 腰の高さ・指先の向きの絞り込みは、per_target なら目標ごと (リストなら
    # 目標の添字 -> 組)、そうでなければ全体で行う。
    groups = {}
    for i, c in enumerate(candidates):
        group = (c[4] if per_target is True
                 else per_target[c[4]] if per_target else 0)
        groups.setdefault(group, []).append(i)
    kept = []
    for indices in groups.values():
        if max_waist_z is not None and GOAL_LEG_RISE != 'off':
            keep = [i for i in indices
                    if waist_zs[i] <= max_waist_z + GOAL_LEG_RISE_TOLERANCE]
            if keep or GOAL_LEG_RISE == 'strict':
                indices = keep
        down = [i for i in indices if finger_zs[i] <= GOAL_FINGER_DOWN_MAX_Z]
        kept += down or indices
    candidates = [candidates[i] for i in sorted(kept)]
    if candidates:
        candidates.sort(key=lambda c: (c[4], c[0]))
        return candidates, GOAL_X_MARGIN
    return [], None


def _lowering_path_steps(human_arm, lowering, samples=20):
    """``lowering`` の動きの掌の道のり [m] と回転 [rad] (肩を中心に円弧で
    下りるので、両端の差より長い)。"""
    if PATH_STEPS_BATCH:
        return _batch_lowering_path_steps(human_arm, lowering, samples)
    return _lowering_path_steps_loop(human_arm, lowering, samples)


# 経路の掌の道のり (``_lowering_path_steps``) の点をまとめて計算する ('0' で
# 1 点ずつの前の形)。develop の Q3b の P3 (UNIFIED_Q3_PATH_STEPS_BATCH、
# 450d4b1) と同じ。
PATH_STEPS_BATCH = os.environ.get('MFIX_PATH_STEPS_BATCH', '1') == '1'


def _batch_lowering_path_steps(human_arm, lowering, samples):
    """``_lowering_path_steps`` の点をまとめた形。足し方 (前から順) と 1 区間
    ごとの式は前の形と同じ。"""
    if isinstance(human_arm, _StaticArm):
        # 掌は動かないので、どの区間も同じ値 (前の形と同じく区間の数だけ前から
        # 足す)。
        rot = human_arm.palm_rot
        step = Rotation.from_matrix(rot @ rot.T).magnitude()
        zero = np.linalg.norm(human_arm.palm_pos - human_arm.palm_pos)
        return sum([zero] * samples), sum([step] * samples)
    if not hasattr(human_arm, 'batch_lowering_palms'):
        return _lowering_path_steps_loop(human_arm, lowering, samples)
    pos, rot = human_arm.batch_lowering_palms(
        lowering, [k / float(samples) for k in range(samples + 1)])
    length = sum(np.linalg.norm(d) for d in pos[1:] - pos[:-1])
    angle = sum(Rotation.from_matrix(
        rot[1:] @ np.transpose(rot[:-1], (0, 2, 1))).magnitude())
    return length, angle


def _lowering_path_steps_loop(human_arm, lowering, samples):
    """``_lowering_path_steps`` の 1 点ずつの形 (前の形)。"""
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
    z_start = _waist_height_for_legs(
        robot, start_av[legs[0]], start_av[legs[1]])

    def solve_at(s, s_prev, prev, z_prev, label):
        """割合 ``s`` の waypoint を、直前 (割合 ``s_prev``、関節角 ``prev``、
        腰の高さ ``z_prev``) から解く。``(waypoint, None, None)``、関節角が
        飛んだら ``(None, 理由, 'jump')``、それ以外で解けなければ
        ``(None, 理由, None)``。"""
        _check_stop()
        _UPPER.s = s
        base = base_path.at(s)
        elbow, _, palm_pos, palm_rot = human_arm.lowering_pose(lowering, s)
        arm = human_arm.pose(palm_pos, palm_rot, elbow=elbow)
        if arm['reach'] > 1.0:
            return None, '{} で人の腕が届かない (到達率 {:.2f})'.format(
                label, arm['reach']), None
        turn_deg = start_turn + turn_delta * s
        target = press_target(palm_from_frame(palm_pos, palm_rot), turn_deg,
                              robot_arm)
        # 脚は直前から移動先へ残りの割合で均等に近づけた角度に固定して解く。
        # 解けなければ脚も IK で動かす。
        ratio = (s - s_prev) / max(1.0 - s_prev, 1e-9)
        leg_angles = prev[legs] + (goal_av[legs] - prev[legs]) * ratio
        if GOAL_LEG_RISE != 'off':
            # 補間の途中で腰が高くならないよう、高くなるときは膝をより曲げる。
            leg_angles = np.array(cap_leg_rise(
                robot, leg_angles[0], leg_angles[1],
                min(z_prev, z_start) + PATH_LEG_RISE_TOLERANCE))
        # 直前の解を移動先へ割合の分だけ寄せた初期値を先に試す (直前の解
        # からだと手首などが直前の曲げに留まる)。
        step = (goal_av - start_av) * (s - s_prev)
        for seed, free_legs in ((prev + step, False), (prev, False),
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
            return None, '{} で腕が掌に届かない'.format(label), None
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
            return None, '{} で関節角が飛ぶ ({} {:.0f} 度)'.format(
                label, names[jump], math.degrees(av[jump] - prev[jump])), \
                'jump'
        arm_s = human_arm.arm_fraction(s)
        problem = checker.check(robot, human_arm.skeleton(arm),
                                dict(arm, s=arm_s))
        if problem is None:
            actual_turn = grip_turn_deg(
                robot, robot_arm, palm_from_frame(palm_pos, palm_rot))
            wrist, problem = wrist_problem(human_arm, robot, arm, arm_s,
                                           actual_turn)
        if problem is not None:
            return None, '{} で{}'.format(label, problem), None
        if free_turn:
            turn_deg = actual_turn
        arm = dict(arm, s=float(arm_s), robot_hand_wrist=wrist)
        return dict(
            base_position=[float(base[0]), float(base[1]), 0.0],
            base_yaw=float(base[2]),
            joint_angle_vector=[float(v) for v in av],
            turn_deg=float(turn_deg), human_arm=arm), None, None

    prev = start_av
    z_prev = z_start
    s_prev = 0.0
    waypoints = []
    reason = None
    for k in range(1, n + 1):
        s = k / float(n)
        label = '{}/{}'.format(k, n)
        waypoint, reason, kind = solve_at(s, s_prev, prev, z_prev, label)
        added = [waypoint] if waypoint is not None else []
        # 関節角が飛んだ区間は、細かく分けて直前の解から解き進め直す
        # (PATH_JUMP_SUBDIVISIONS)。どの分け方でも飛ぶなら止める。
        if kind == 'jump':
            for parts in PATH_JUMP_SUBDIVISIONS:
                sub, sub_prev, sub_z, sub_s = [], prev, z_prev, s_prev
                for j in range(1, parts + 1):
                    s_j = s_prev + (s - s_prev) * j / float(parts)
                    wp, _, _ = solve_at(s_j, sub_s, sub_prev, sub_z,
                                        '{} の {}/{}'.format(label, j, parts))
                    if wp is None:
                        break
                    sub.append(wp)
                    sub_prev = np.asarray(wp['joint_angle_vector'])
                    sub_z = _waist_height_for_legs(
                        robot, sub_prev[legs[0]], sub_prev[legs[1]])
                    sub_s = s_j
                if len(sub) == parts:
                    added, reason = sub, None
                    break
        if not added:
            break
        waypoints.extend(added)
        prev = np.asarray(added[-1]['joint_angle_vector'], dtype=np.float64)
        z_prev = _waist_height_for_legs(robot, prev[legs[0]], prev[legs[1]])
        s_prev = s
    # 進めた割合は、分けた区間も元の 1 区間として数える。
    n_done = n if reason is None else k - 1
    return waypoints, n_done / float(n), reason


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


# 人ごとのキャッシュ (_person_cache) を残す人数。
PERSON_CACHE_SIZE = 4
_PERSON_CACHES = collections.OrderedDict()
_PERSON_CACHES_LOCK = threading.Lock()


def _person_cache(post, turn_deg, collision_joints, base_limits):
    """押し込み姿勢 ``post``・握りの向き・骨格・台車の可動域が同じ計画で共有
    する dict (移動先のバッチ IK の候補と、下ろし方ごとの結果)。実機ノードの
    動く前の計画と裏の計画 (別スレッド) が同じものを使う。"""
    key = zlib.crc32(np.round(np.asarray(
        post['joint_angle_vector'], dtype=np.float64), 6).tobytes())
    key = zlib.crc32(np.round(np.asarray(
        list(post['base_position'][:2]) + [post['base_yaw'], turn_deg]
        + [v for limit in base_limits for v in limit],
        dtype=np.float64), 6).tobytes(), key)
    for name in sorted(collision_joints):
        key = zlib.crc32(name.encode(), key)
        key = zlib.crc32(np.round(np.asarray(
            collision_joints[name], dtype=np.float64), 6).tobytes(), key)
    with _PERSON_CACHES_LOCK:
        cache = _PERSON_CACHES.pop(key, None)
        if cache is None:
            cache = {}
        _PERSON_CACHES[key] = cache
        while len(_PERSON_CACHES) > PERSON_CACHE_SIZE:
            _PERSON_CACHES.popitem(last=False)
    return cache


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


class TransitionStopped(Exception):
    """裏の計画 (``refine_transition``) が止める合図で打ち切られた。"""


# スレッドごとの止める合図 (refine_transition の ``stop``)。
_STOP = threading.local()


def _check_stop():
    """このスレッドの止める合図が立っていれば ``TransitionStopped`` を投げる。"""
    stop = getattr(_STOP, 'event', None)
    if stop is not None and stop.is_set():
        raise TransitionStopped()


def refine_transition(robot, robot_arm, hand, post, turn_deg,
                      translated_joints, collision_joints, verification_pairs,
                      collision_pairs, base_limits, secured,
                      attempts_per_pose=GOAL_ATTEMPTS_PER_POSE, stop=None):
    """動かす前に確保した横並び移動 ``secured`` (``plan_transition`` の
    approach=False の結果) を、寄せ (段階 2) と残りの掌の向きで良くした計画を
    返す (実機ノードは動いている間に裏のスレッドで解く)。

    まず ``secured`` の終わりから寄せを解き、寄せ後の横の距離が
    ``APPROACH_STOP_LATERAL`` 以下ならそれを返す。そうでなければ残りの向きも
    解き (``_plan_transition``)、``secured`` より腕を上げないもの
    (``_low_arm_results``) のうち寄せ後の一番近いものを返す。``robot`` は、
    ほかで使っていないロボットモデルにすること (IK で姿勢を書き換える)。

    ``stop`` (``threading.Event``) が立つと、探索の区切り (下ろし方・移動先の
    候補・経路の waypoint ごと) で ``TransitionStopped`` を投げて打ち切る
    (実機ノードは、横並び移動を始める時点で終わっていなければ止める)。"""
    _STOP.event = stop
    try:
        return _with_press_seed(
            post, _refine_transition, robot, robot_arm, hand, post, turn_deg,
            translated_joints, collision_joints, verification_pairs,
            collision_pairs, base_limits, secured, attempts_per_pose)
    finally:
        _STOP.event = None


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
        secured_z = _lowered_palm_z(secured) if secured else None
        results.append(_plan_transition(
            *common, attempts_per_pose=attempts_per_pose, facings=rest,
            time_limit=REFINE_TIME_LIMIT, alt_arms=False,
            max_palm_z=(secured_z + LOWERED_HEIGHT_TOLERANCE
                        if PRUNE_HIGH_LOWERINGS and secured_z is not None
                        else None)))
    if secured:
        # 確保済みより腕を上げる計画は採らない。
        results.append(secured)
    best = min(_low_arm_results(results),
               key=lambda r: _result_key(r, translated_joints, hand))
    if best is secured:
        best = copy.deepcopy(secured)
    if APPROACH_ALT_ARMS and facing is not None:
        # 寄せで人の腕も変える案は、選んだ 1 つの計画にだけ試す。
        with _upper_body_limit(math.atan2(facing[1], facing[0])):
            _add_alt_arm_approach(
                robot, robot_arm, hand, post, turn_deg, best,
                translated_joints, collision_joints, verification_pairs,
                collision_pairs, base_limits)
    best['compute_time'] = time.time() - t0
    return best


def _lowered_palm_z(result):
    """横並びまで進んだ計画 ``result`` の、腕を下ろしきった人の掌の高さ [m]
    (進んでいなければ None)。"""
    if not (result.get('verified') and result.get('fraction', 0.0) >= 1.0):
        return None
    approach = result.get('approach') or {}
    if approach.get('stage1_palm_z') is not None:
        # 寄せで人の腕を変えた計画 (_approach_alt_arms) は段階 1 の掌で比べる
        return float(approach['stage1_palm_z'])
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
    if COMFORT_SELECT in ('tie', 'bucket5'):
        # study/sbs-comfort: 横の距離の刻みが同じなら人の項の小さい方。
        step = 0.02 if COMFORT_SELECT == 'tie' else 0.05
        comfort = result_comfort(result, translated_joints)
        comfort = float('inf') if comfort is None else comfort
        if approach and approach['n_waypoints']:
            return (0, round(approach['lateral_after'] / step), comfort, phi)
        if result['verified'] and result['fraction'] >= 1.0:
            end = result['waypoints'][-1]['base_position']
            return (1, round(human_lateral(translated_joints, end, hand)
                             / step), comfort, phi)
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
                 yaw_margin_deg=None, deadline=None, cache=None,
                 bundle_facings=None, max_palm_z=None, bundle_first=True,
                 hint=None):
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

    ``cache`` (``_person_cache``) を与えると、移動先のバッチ IK を
    ``bundle_facings`` (既定 ``PALM_FACING_ORDER``) の掌の向きでまとめて解き
    (``GOAL_BUNDLE_FACINGS``)、下ろし方ごとの結果を再利用する
    (``GOAL_REUSE_OUTCOMES``)。``bundle_first`` が False なら、最初に解く
    下ろし方はまとめずに解く (最初の下ろし方で横並びまで進むことが多い、
    寄せを解かない計画用)。``max_palm_z`` [m] を与えると、下ろしきった掌が
    それより高い下ろし方は試さない。

    ``hint`` (``arm_deg``・``abduction_deg``・``palm_facing`` の dict、
    予備の解の下ろし方) を与えると、その上腕の角度を含む組・その下ろし方から
    試す (押し込みを解き直した姿勢からの解き直し用。実機 8 試行で、解けた
    5 試行の時間 平均 2.1 -> 1.3 秒、寄せ後 平均 0.71 -> 0.68 m)。

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
    result = dict(verified=False, waypoints=[], hand=hand, robot_arm=robot_arm,
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
    _place(robot, start_av, start_base)
    checker.set_press(robot, result['press_arm'])
    result['press_forearm_clearance'] = checker.press_forearm
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
        _check_stop()
        if deadline is None or time.time() < deadline:
            return False
        if not result['timed_out']:
            result['timed_out'] = True
            reasons.append('探索の時間の上限に達した')
        return True

    yaw_key = (GOAL_YAW_MARGIN_DEG if yaw_margin_deg is None
               else yaw_margin_deg)
    # 移動先のバッチ IK をまとめて解く相手 (lowering_key -> 下ろし方のリスト、
    # 下の per_angle のループで作る)。
    bundles = {}
    # いま試している下ろし方の一番良い経路 (GOAL_REUSE_OUTCOMES 用)。
    current = [None]
    # この呼び出しで移動先のバッチ IK を解いた回数。
    n_solved = [0]

    def lowering_key(lowering):
        return (yaw_key, float(lowering['arm_deg']),
                float(lowering['abduction_deg']), lowering['palm_facing'])

    def solve_lowerings(lowerings, turn_degs, per_target=False,
                        attempts=None):
        return solve_goal_candidates(
            robot, robot_arm, hand, [lw['palm'] for lw in lowerings],
            start_av, translated_joints, collision_joints, collision_pairs,
            base_limits, turn_degs=turn_degs,
            attempts_per_pose=attempts or attempts_per_pose,
            free_turn=free_turn,
            lowered_arms=[(human_arm.shoulder,
                           human_arm.lowering_pose(lw, 1.0)[0])
                          for lw in lowerings],
            max_waist_z=start_waist_z, yaw_margin_deg=yaw_margin_deg,
            lowered_forearms=[human_arm.lowering_pose(lw, 1.0)
                              for lw in lowerings],
            per_target=per_target)

    arm_problems = {}
    # 手がかり (予備の解) の下ろし方 (上腕の角度, 脇の開き, 掌の向き)。
    hint = hint or {}
    hint_key = ((float(hint['arm_deg']), float(hint['abduction_deg']),
                 hint.get('palm_facing'))
                if hint.get('arm_deg') is not None else None)

    def arm_ok(lowering):
        """下ろし方 ``lowering`` の途中で人の肩・肘・前腕が目安に収まるか
        (``HUMAN_ARM_CHECK``)。収まらなければ理由を ``reasons`` に足す。"""
        if HUMAN_ARM_CHECK == 'off':
            return True
        key = ('arm',) + lowering_key(lowering)[1:]
        store = cache if cache is not None else arm_problems
        if key not in store:
            store[key] = human_arm.lowering_problem(lowering)
        problem = store[key]
        if problem is not None:
            reasons.append('下ろし方で{}'.format(problem.split(' ')[0]))
        return problem is None

    def wrist_feasible(lowering):
        """下ろした掌 ``lowering`` で、人の手首が目安に収まる握りの向きが
        あるか (``GOAL_PRECHECK_WRIST``)。無ければどの移動先も
        ``fix_goal_wrist`` で捨てられる (ロボットの姿勢に依らない)。"""
        if not GOAL_PRECHECK_WRIST or HUMAN_WRIST_CHECK == 'off':
            return True
        key = ('wrist',) + lowering_key(lowering)[1:]
        if cache is not None and key in cache:
            return cache[key]
        goal_arm = human_arm.pose(
            *lowering['palm'], elbow=human_arm.lowering_pose(lowering, 1.0)[0])
        ok = wrist_ok_turn(human_arm, goal_arm, *lowering['palm'],
                           0.0) is not None
        if cache is not None:
            cache[key] = ok
        return ok

    def cached_candidates(lowering):
        """下ろし方 ``lowering`` (目標 3 つとも同じ) の移動先の候補。まとめる
        相手 (``bundles``) と 1 回のバッチ IK で解き、``cache`` に置く。"""
        key = ('goal',) + lowering_key(lowering)
        if key not in cache:
            bundle = GOAL_BUNDLE_FACINGS and (bundle_first or n_solved[0] > 0)
            n_solved[0] += 1
            members = [lowering] + ([
                lw for lw in bundles.get(lowering_key(lowering)[1:], ())
                if ('goal',) + lowering_key(lw) not in cache
                and (max_palm_z is None or lw['palm'][0][2] <= max_palm_z)
                and wrist_feasible(lw)
            ][:LOWERED_TARGET_COUNT - 1] if bundle else [])
            # 足りない目標は最後の下ろし方で埋め、同じ下ろし方として扱う。
            slots = [min(k, len(members) - 1)
                     for k in range(LOWERED_TARGET_COUNT)]
            # まとめないときも初期値の数を揃える (バッチ IK の形を 1 つにして
            # warmup のコンパイルを増やさない)。
            candidates, _ = solve_lowerings(
                [members[k] for k in slots], (0.0,), per_target=slots,
                attempts=(GOAL_BUNDLE_ATTEMPTS_PER_POSE
                          if GOAL_BUNDLE_FACINGS else None))
            for k, member in enumerate(members):
                own = sorted([c[:4] + (0,) for c in candidates
                              if slots[c[4]] == k], key=lambda c: c[0])
                cache[('goal',) + lowering_key(member)] = (
                    own, GOAL_X_MARGIN if own else None)
        candidates, x_margin = cache[key]
        return list(candidates), x_margin

    def record_path(waypoints, fraction, lowering, x_margin):
        if current[0] is None or fraction > current[0][1]:
            current[0] = (waypoints, fraction, lowering['arm_deg'],
                          lowering['abduction_deg'], lowering['palm_facing'],
                          x_margin)
        if best[0] is None or fraction > best[0][1]:
            best[0] = (waypoints, fraction)
            result.setdefault('fraction_times', []).append(
                (time.time(), float(fraction)))
            result.update(arm_deg=lowering['arm_deg'],
                          abduction_deg=lowering['abduction_deg'],
                          palm_facing=lowering['palm_facing'],
                          x_margin=x_margin)

    def try_lowerings(lowerings):
        """腕の下ろし方 ``lowerings`` (3 つ) を移動先にして経路を作る。
        横並びまで (100%) 進めたか、時間の上限を過ぎたら True。
        ``GOAL_REUSE_OUTCOMES`` なら、同じ人で試し終えた下ろし方は結果を
        再利用する。"""
        single = all(lw is lowerings[0] for lw in lowerings)
        key = (('outcome',) + lowering_key(lowerings[0])
               if cache is not None and GOAL_REUSE_OUTCOMES and single
               else None)
        if key is not None and key in cache:
            path, why = cache[key]
            reasons.extend(why)
            if path is not None:
                waypoints, fraction, arm_deg, abduction_deg, facing, \
                    x_margin = copy.deepcopy(path)
                record_path(waypoints, fraction,
                            dict(arm_deg=arm_deg, abduction_deg=abduction_deg,
                                 palm_facing=facing), x_margin)
                if fraction >= 1.0:
                    return True
            return False
        mark = len(reasons)
        current[0] = None
        done = _try_lowerings(lowerings)
        if key is not None and not result['timed_out']:
            # 時間切れで途中までしか試していない結果は残さない。
            cache[key] = (copy.deepcopy(current[0]), list(reasons[mark:]))
        return done

    def _try_lowerings(lowerings):
        single = all(lw is lowerings[0] for lw in lowerings)
        if single and not wrist_feasible(lowerings[0]):
            reasons.append('移動先で人の手首が収まる握りの向きが無い')
            return False
        for turn_degs in turn_groups:
            if out_of_time():
                return True
            if cache is not None and single and free_turn:
                candidates, x_margin = cached_candidates(lowerings[0])
            else:
                candidates, x_margin = solve_lowerings(lowerings, turn_degs)
            if not candidates:
                reasons.append('移動先の IK が解けない (腕 {} 度、握りの向き '
                               '{})'.format(
                                   '/'.join('{:.0f}'.format(lw['arm_deg'])
                                            for lw in lowerings),
                                   '/'.join('{:.0f}'.format(t)
                                            for t in turn_degs)))
                continue
            if all(lw is lowerings[0] for lw in lowerings):
                # 3 つとも同じ下ろし方なら、1 つの目標としてコストの安い順。
                candidates = sorted(
                    [c[:4] + (0,) for c in candidates], key=lambda c: c[0])
            # 次の組 (握りの向き ±90 度) は、この組で移動先の IK が 1 つも解けなかった
            # ときだけ試す。
            return try_candidates(candidates, lowerings, x_margin)
        return False

    def try_candidates(candidates, lowerings, x_margin):
        """移動先の候補 ``candidates`` を順に試す。横並びまで進めたか、時間・
        経路の試行回数の上限なら True。"""
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
            if free_turn:
                goal_turn = grip_turn_deg(robot, robot_arm,
                                          palm_from_frame(*goal_palm))
            # 人の手首が目安を超えるなら、収まる握りの向きで解き直す。
            goal_turn, problem = fix_goal_wrist(
                robot, robot_arm, human_arm, goal_arm, *goal_palm,
                gaze_forward, goal_turn, aim_head=False)
            if problem is None:
                # 腕と同時の視線 IK は首が大きく回せず解けないことがあるので、
                # 首だけで前へ向け直す (経路の首はここへ補間する)。首は手首に
                # 関係しないので、手首の判定に通ったものだけ。
                _aim_head(robot, gaze_forward)
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
            record_path(waypoints, fraction, lowering, x_margin)
            if fraction >= 1.0:
                n_paths_full[0] = n_paths[0]
                return True
        return False

    def facing_variants(arm_degs, abduction_deg, order=None):
        """組 ``arm_degs`` の掌の向きごとの下ろし方 (3 つとも同じ向き) の
        ``[(向きの順位, 下ろし方)]``。向きは ``order`` (既定は ``facings``、
        それも None なら ``LOWERED_FACING_ORDER``) の順で、
        使えない向き・前腕のひねりが ``LOWERED_TWIST_MAX_DEG`` (None でなければ) を超える向きは
        除く。``use_facing_variants`` が False なら前腕のひねりが一番少ない
        向き (上腕の角度ごとに違ってよい) の 1 つだけ。掌が押し込み時より
        ``LOWERED_MIN_DROP`` 下がらない下ろし方は、下がる下ろし方で置き換え
        (バッチ IK の目標の数を変えないため)、1 つも無ければその組を除く。"""
        def lowered_enough(lowerings):
            ok = [lw['palm'][0][2] <= press_palm_z - LOWERED_MIN_DROP
                  and arm_ok(lw) for lw in lowerings]
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
        facing_order = (order if order is not None
                        else facings if facings is not None
                        else LOWERED_FACING_ORDER_PALM_UP if palm_up
                        else LOWERED_FACING_ORDER)
        for rank, facing in enumerate(facing_order):
            lowerings = [human_arm.lowering(deg, abduction_deg, facing)
                         for deg in arm_degs]
            if any(lw is None for lw in lowerings):
                continue
            twist = max(abs(math.degrees(lw['twist'])) for lw in lowerings)
            if (LOWERED_TWIST_MAX_DEG is not None
                    and twist > LOWERED_TWIST_MAX_DEG):
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
    if hint_key is not None:
        # 予備の解の上腕の角度を含む組から試す。
        groups.sort(key=lambda g: hint_key[0] not in g[0])
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
        if LOWERED_ARM_BATCH == 'per_angle':
            # 上腕の角度ごとに、3 つとも同じ下ろし方の目標で解く (低い角度
            # から、同じ角度の中では上の順)。
            batches = []
            seen = set()
            for j in range(len(arm_degs)):
                for _, lowerings in ordered:
                    if id(lowerings[j]) not in seen:
                        seen.add(id(lowerings[j]))
                        if (max_palm_z is not None
                                and lowerings[j]['palm'][0][2] > max_palm_z):
                            # 選ばれない高さ (PRUNE_HIGH_LOWERINGS)
                            continue
                        batches.append([lowerings[j]] * len(lowerings))
            if cache is not None and GOAL_BUNDLE_FACINGS and free_turn:
                # 同じ上腕の角度・脇の開きの下ろし方を、掌の向きの順
                # (bundle_facings) で次のものからまとめる相手にする。
                order = tuple(bundle_facings or facings or ())
                for abduction_deg in abductions:
                    bundle_key = ('bundles', tuple(arm_degs), abduction_deg,
                                  order)
                    if bundle_key not in cache:
                        variants = facing_variants(arm_degs, abduction_deg,
                                                   order=order or None)
                        table = {}
                        for j, deg in enumerate(arm_degs):
                            same = [lws[j] for _, lws in variants
                                    if lws[j]['arm_deg'] == deg]
                            for i, lw in enumerate(same):
                                table[lowering_key(lw)[1:]] = \
                                    same[i + 1:] + same[:i]
                        cache[bundle_key] = table
                    bundles.update(cache[bundle_key])
        else:
            batches = [lowerings for _, lowerings in ordered]
        if COMFORT_ORDER:
            # study/sbs-comfort: 下ろしきった掌の人の項の小さい順 (同じなら今の順)。
            batches.sort(key=lambda b: lowering_comfort(human_arm, b[0]))
        if hint_key is not None:
            # 予備の解の下ろし方を先頭に (残りは上の順のまま)。
            batches.sort(key=lambda b: lowering_key(b[0])[1:] != hint_key)
        if any(try_lowerings(lowerings) for lowerings in batches):
            break
    best = best[0]
    result.update(n_paths=n_paths[0], n_paths_full=n_paths_full[0],
                  path_fractions=path_fractions)
    if best is None or best[1] < MIN_FRACTION:
        reason = ', '.join(sorted(set(reasons))) or None
        if SBS_FULL and best is not None and best[1] > 0.0:
            # 途中まで進めた経路を捨てた (MIN_FRACTION に届かない)。何割まで
            # 進めたかを理由の頭とログに残す (止まった理由は続く reasons)。
            result['partial_fraction'] = float(best[1])
            head = '最後まで解けない (移動先の {:.0f}% まで)'.format(
                100.0 * best[1])
            reason = head if reason is None else head + ', ' + reason
            if best[1] >= SBS_PARTIAL_MIN_FRACTION:
                # 前の形なら途中までで実行していた経路だけログに出す。
                print('[transition] 横並び移動の経路が移動先の {:.0f}% までしか'
                      '解けなかったため、この探索では通らなかった扱いにします '
                      '(前の形 MFIX_SBS_FULL=0 なら途中までで実行していた)。'
                      .format(100.0 * best[1]))
        result.update(reason=reason, compute_time=time.time() - t0)
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
        lateral_before=human_lateral(translated_joints, start_base, hand),
        laterals=[human_lateral(translated_joints, wp['base_position'], hand)
                  for wp in waypoints],
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
    joints = [a['joints'] for a in arms if 'joints' in a]
    if joints:
        # 肩・肘・前腕の関節角度の (最小, 最大) [度]。
        ranges = {}
        for key in HUMAN_ARM_LIMITS_DEG:
            values = [j[key] for j in joints if j.get(key) is not None]
            if values:
                ranges[key] = [min(values), max(values)]
        out['joints'] = ranges
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
                  direct_below=None, turn_to_front=None):
    """``_plan_approach`` を、``turn_to_front`` (None なら
    ``APPROACH_TURN_TO_FRONT``) なら台車を人の正面へ回しながら解き、寄せ
    られなければ向きを変えずに解き直す。戻り値に ``turned`` (回したか) を
    足す。"""
    if turn_to_front is None:
        turn_to_front = APPROACH_TURN_TO_FRONT
    args = (robot, robot_arm, hand, post, turn_deg, last, translated_joints,
            collision_joints, verification_pairs, collision_pairs,
            base_limits)
    kwargs = dict(yaw_margin_deg=yaw_margin_deg, target_lateral=target_lateral,
                  windows=windows, attempts_per_pose=attempts_per_pose,
                  direct_below=direct_below)
    if turn_to_front:
        out = _plan_approach(*args, turn=True, **kwargs)
        if out['waypoints']:
            out['turned'] = True
            return out
        reason = out['reason']
        out = _plan_approach(*args, turn=False, **kwargs)
        if not out['waypoints'] and reason:
            out['reason'] = '回して: {} / 回さず: {}'.format(
                reason, out['reason'])[:300]
    else:
        out = _plan_approach(*args, turn=False, **kwargs)
    out['turned'] = False
    return out


def _plan_approach(robot, robot_arm, hand, post, turn_deg, last,
                   translated_joints, collision_joints, verification_pairs,
                   collision_pairs, base_limits, yaw_margin_deg=None,
                   target_lateral=APPROACH_TARGET_LATERAL, windows=None,
                   attempts_per_pose=GOAL_ATTEMPTS_PER_POSE,
                   direct_below=None, turn=False):
    """横並びの終わり ``last`` (``plan_path`` の最後の waypoint) の姿勢から、
    人の腕・掌を動かさず、台車を横へ人の方へ寄せる waypoint (寄せ、段階 2)
    を計画する。``turn`` なら台車を人の正面から ``APPROACH_END_YAW_DEG``
    以内へ回しながら寄せ (回転は腕で受ける)、そうでなければ向きは変えない。

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
    human_yaw = math.atan2(facing[1], facing[0])
    end_limit = math.radians(APPROACH_END_YAW_DEG)
    # 寄せの終わりの台車の向き (回すなら人の正面から end_limit 以内へ)。
    goal_yaw = (human_yaw + max(-end_limit, min(end_limit, _wrap(
        start[2] - human_yaw))) if turn else start[2])
    # BasePath は yaw を線形に補間するので、始点から近い回り方の値にする。
    goal_yaw = start[2] + _wrap(goal_yaw - start[2])
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
    # 試す窓 (近い方から、寄せが APPROACH_MIN_GAIN に満たない窓の手前まで)。
    usable = []
    for lo, hi in windows:
        if hi > lat0 - APPROACH_MIN_GAIN:
            break
        usable.append((lo, hi))

    def solve_window(y_window, attempts):
        return solve_goal_candidates(
            robot, robot_arm, hand, [(static.palm_pos, static.palm_rot)] * 3,
            start_av, translated_joints, collision_joints, collision_pairs,
            base_limits, turn_degs=(0.0,), free_turn=GOAL_FREE_TURN,
            lowered_arms=[(real.shoulder, static.elbow)] * 3,
            max_waist_z=start_waist_z, branch_range_deg=100.0,
            y_window=y_window,
            yaw_margin_deg=APPROACH_END_YAW_DEG if turn else yaw_margin_deg,
            attempts_per_pose=attempts,
            lowered_forearms=[(static.elbow, arm_pose['wrist'],
                               static.palm_pos, static.palm_rot)] * 3)[0]

    union = None
    if APPROACH_SINGLE_BATCH and usable:
        union = solve_window((min(lo for lo, _ in usable),
                              max(hi for _, hi in usable)),
                             APPROACH_SINGLE_BATCH_ATTEMPTS)
    for lo, hi in usable:
        if union is None:
            candidates = solve_window((lo, hi), attempts_per_pose)
        else:
            candidates = [c for c in union if lo <= human_lateral(
                translated_joints, c[2], hand) <= hi]
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
            _check_stop()
            # 回すなら移動先の向き (人の正面から end_limit 以内)、回さない
            # なら向きは変えない
            goal_base = np.array([
                goal_base[0], goal_base[1],
                start[2] + _wrap(goal_base[2] - start[2]) if turn
                else start[2]])
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
    goal[2] = goal_yaw
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


# 寄せられたら、押し込み姿勢から寄せの終点へ 1 本の経路で解き直し、横並びまで
# 進めればそちらを使うか (``_direct_to_approach_end``)。
MERGE_APPROACH = True

# 寄せで人の腕の姿勢も変えるか (``_approach_alt_arms``)。段階 1 の腕のままの
# 寄せで台車が人の横 APPROACH_STOP_LATERAL [m] まで寄れなかったとき、人の上腕を
# 前へ APPROACH_ALT_ARM_DEGS 度・外へ APPROACH_ALT_ABDUCTIONS 度、肘を
# APPROACH_ALT_ELBOWS 度曲げた下ろし方 (掌の向きは PALM_FACING_ORDER、LUBA の
# 目安内) を移動先にして、押し込み姿勢から 1 本の経路で解き直す。上半身を人の
# 正面に向けたまま近くで手をつなぐには、腕を真下に垂らした掌はロボットの手が
# 合わせにくく、少し前に出して肘を曲げた掌が合わせやすい (2026-10-30 の総当たり)。
APPROACH_ALT_ARMS = True
APPROACH_ALT_ARM_DEGS = (20.0, 40.0)
APPROACH_ALT_ABDUCTIONS = (15.0, 30.0)
APPROACH_ALT_ELBOWS = (30.0, 60.0)
# 経路 (plan_path) まで試す移動先の数の上限。
APPROACH_ALT_MAX_PATHS = spik.env_int('MFIX_APPROACH_ALT_MAX_PATHS', 6)
# 移動先のバッチ IK の初期値・腕の枝の基準: 'press' は押し込み姿勢、'stage1'
# は段階 1 (と寄せ) の終点。腕の枝の範囲 [度] (GOAL_BRANCH_JOINT_RANGE_DEG と
# 同じ意味)。腰を押し込み終了時より上げないか (GOAL_LEG_RISE と同じ)。
APPROACH_ALT_SEEDS = ('press', 'stage1')
APPROACH_ALT_SEED = 'stage1'
APPROACH_ALT_BRANCH_RANGE_DEG = GOAL_BRANCH_JOINT_RANGE_DEG
APPROACH_ALT_WAIST_LIMIT = True


def _direct_to_approach_end(robot, robot_arm, hand, post, turn_deg, result,
                            end, translated_joints, verification_pairs):
    """押し込み姿勢から、腕を段階 1 と同じく下ろしながら、台車を寄せの終点
    ``end`` (waypoint) へ直接動かす経路を ``plan_path`` で解く。移動先まで
    進めればその waypoint と None、進めなければ None と止めた理由。"""
    palm = human_palm_from_press(post, turn_deg, robot_arm)
    human_arm = HumanArm(translated_joints, hand, palm, turn_deg, robot_arm)
    lowering = human_arm.lowering(result['arm_deg'], result['abduction_deg'],
                                  result['palm_facing'])
    if lowering is None:
        return None, '下ろし方が作れない'
    start_base = np.array([post['base_position'][0],
                           post['base_position'][1], post['base_yaw']])
    goal_base = np.array([end['base_position'][0], end['base_position'][1],
                          start_base[2] + _wrap(end['base_yaw']
                                                - start_base[2])])
    base_path = BasePath(start_base, goal_base,
                         spik.human_standing_xy(translated_joints))
    checker = TransitionChecker(robot_arm, hand, verification_pairs)
    start_av = np.asarray(post['joint_angle_vector'], dtype=np.float64)
    _place(robot, start_av, start_base)
    checker.set_press(robot, human_arm.pose(human_arm.palm_pos0,
                                            human_arm.palm_rot0))
    waypoints, fraction, why = plan_path(
        robot, robot_arm, hand, turn_deg, end['turn_deg'], human_arm,
        lowering, np.asarray(post['joint_angle_vector'], dtype=np.float64),
        base_path, np.asarray(end['joint_angle_vector'], dtype=np.float64),
        checker, free_turn=GOAL_FREE_TURN)
    return (waypoints, None) if fraction >= 1.0 else (None, why)


def _approach_alt_arms(robot, robot_arm, hand, post, turn_deg,
                       translated_joints, collision_joints, verification_pairs,
                       collision_pairs, base_limits, yaw_margin_deg,
                       max_lateral, seed_av=None):
    """人の腕を ``APPROACH_ALT_*`` の下ろし方に変えて、台車を人の横
    ``max_lateral`` - ``APPROACH_MIN_GAIN`` [m] より近くへ寄せる移動先を解き、
    押し込み姿勢からそこへの経路を ``plan_path`` で作る。近い移動先から
    試し、横並びまで進めた最初の経路の ``(waypoints, 情報の dict)`` を返す。
    無ければ ``(None, 情報の dict)`` (``reason`` に理由)。``seed_av`` は
    移動先のバッチ IK の初期値・腕の枝の基準 (None なら押し込み姿勢)。"""
    t0 = time.time()
    info = dict(reason=None, n_lowerings=0, n_candidates=0, n_paths=0)
    hi = max_lateral - APPROACH_MIN_GAIN
    lo = APPROACH_TARGET_LATERAL - 0.1
    if hi <= lo:
        info['reason'] = '寄せる余地が無い'
        return None, info
    palm = human_palm_from_press(post, turn_deg, robot_arm)
    human_arm = HumanArm(translated_joints, hand, palm, turn_deg, robot_arm)
    press_arm = human_arm.pose(*palm_frame(palm))
    press_palm_z = press_arm['palm_position'][2]
    start_base = np.array([post['base_position'][0],
                           post['base_position'][1], post['base_yaw']])
    start_av = np.asarray(post['joint_angle_vector'], dtype=np.float64)
    _place(robot, start_av, start_base)
    start_waist_z = waist_height(robot)
    standing = spik.human_standing_xy(translated_joints)
    gaze_forward = forward_gaze_point(translated_joints)
    checker = TransitionChecker(robot_arm, hand, verification_pairs)
    checker.set_press(robot, press_arm)
    reasons = []

    lowerings = []
    for arm_deg in APPROACH_ALT_ARM_DEGS:
        for abduction_deg in APPROACH_ALT_ABDUCTIONS:
            for elbow_deg in APPROACH_ALT_ELBOWS:
                for facing in PALM_FACING_ORDER:
                    lw = human_arm.lowering(arm_deg, abduction_deg, facing,
                                            elbow_deg=elbow_deg)
                    if (lw is None or lw['palm'][0][2]
                            > press_palm_z - LOWERED_MIN_DROP):
                        continue
                    if (HUMAN_ARM_CHECK != 'off'
                            and human_arm.lowering_problem(lw) is not None):
                        continue
                    if GOAL_PRECHECK_WRIST and HUMAN_WRIST_CHECK != 'off':
                        goal_arm = human_arm.pose(
                            *lw['palm'],
                            elbow=human_arm.lowering_pose(lw, 1.0)[0])
                        if wrist_ok_turn(human_arm, goal_arm, *lw['palm'],
                                         0.0) is None:
                            continue
                    lowerings.append(lw)
    info['n_lowerings'] = len(lowerings)
    if not lowerings:
        info['reason'] = '人の目安内の下ろし方が無い'
        return None, info

    # 下ろし方 3 つずつを 1 回のバッチ IK で解く (段階 1 と同じ形)。
    candidates = []
    for i in range(0, len(lowerings), LOWERED_TARGET_COUNT):
        _check_stop()
        members = lowerings[i:i + LOWERED_TARGET_COUNT]
        slots = [min(k, len(members) - 1)
                 for k in range(LOWERED_TARGET_COUNT)]
        found, _ = solve_goal_candidates(
            robot, robot_arm, hand, [members[k]['palm'] for k in slots],
            start_av if seed_av is None else seed_av, translated_joints,
            collision_joints, collision_pairs,
            base_limits, turn_degs=(0.0,),
            attempts_per_pose=GOAL_BUNDLE_ATTEMPTS_PER_POSE,
            branch_range_deg=APPROACH_ALT_BRANCH_RANGE_DEG,
            free_turn=GOAL_FREE_TURN,
            lowered_arms=[(human_arm.shoulder,
                           human_arm.lowering_pose(members[k], 1.0)[0])
                          for k in slots],
            max_waist_z=start_waist_z if APPROACH_ALT_WAIST_LIMIT else None,
            y_window=(lo, hi),
            yaw_margin_deg=yaw_margin_deg,
            lowered_forearms=[human_arm.lowering_pose(members[k], 1.0)
                              for k in slots],
            per_target=slots)
        for cost, goal_av, goal_base, goal_turn, index in found:
            candidates.append((
                human_lateral(translated_joints, goal_base, hand), cost,
                goal_av, goal_base, goal_turn, members[slots[index]]))
    info['n_candidates'] = len(candidates)
    # 近いものから (5 cm の幅の中ではコストの安い順)。
    candidates.sort(key=lambda c: (round(c[0] / 0.05), c[1]))
    for lateral, _, goal_av, goal_base, goal_turn, lowering in candidates:
        if info['n_paths'] >= APPROACH_ALT_MAX_PATHS:
            reasons.append('経路の試行回数が上限')
            break
        _check_stop()
        goal_palm = lowering['palm']
        goal_arm = human_arm.pose(
            *goal_palm, elbow=human_arm.lowering_pose(lowering, 1.0)[0])
        goal_skeleton = human_arm.skeleton(goal_arm)
        goal_base = np.array([goal_base[0], goal_base[1],
                              start_base[2]
                              + _wrap(goal_base[2] - start_base[2])])
        base_path = BasePath(start_base, goal_base, standing)
        problem = checker.check_base_path(robot, goal_av, base_path,
                                          goal_skeleton)
        if problem is not None:
            reasons.append(problem)
            continue
        _place(robot, goal_av, goal_base)
        if not _press_ik(robot, robot_arm,
                         press_target(palm_from_frame(*goal_palm), goal_turn,
                                      robot_arm),
                         gaze_forward, free_turn=GOAL_FREE_TURN):
            reasons.append('移動先で押し込み IK が解けない')
            continue
        if GOAL_FREE_TURN:
            goal_turn = grip_turn_deg(robot, robot_arm,
                                      palm_from_frame(*goal_palm))
        goal_turn, problem = fix_goal_wrist(
            robot, robot_arm, human_arm, goal_arm, *goal_palm, gaze_forward,
            goal_turn, aim_head=False)
        if problem is None:
            _aim_head(robot, gaze_forward)
            problem = checker.check(robot, goal_skeleton, goal_arm)
        if problem is not None:
            reasons.append('移動先で{}'.format(problem.split(' ')[0]))
            continue
        info['n_paths'] += 1
        waypoints, fraction, why = plan_path(
            robot, robot_arm, hand, turn_deg, goal_turn, human_arm, lowering,
            start_av, base_path, robot.angle_vector().copy(), checker,
            free_turn=GOAL_FREE_TURN)
        if fraction >= 1.0:
            info.update(arm_deg=lowering['arm_deg'],
                        abduction_deg=lowering['abduction_deg'],
                        elbow_deg=lowering['elbow_deg'],
                        palm_facing=lowering['palm_facing'],
                        compute_time=time.time() - t0)
            return waypoints, info
        if why is not None:
            reasons.append('経路の{}'.format(why))
    info.update(reason=', '.join(sorted(set(reasons)))[:300] or
                '移動先の IK が解けない', compute_time=time.time() - t0)
    return None, info


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
    if waypoints and MERGE_APPROACH:
        merged, why = _direct_to_approach_end(
            robot, robot_arm, hand, post, turn_deg, result, waypoints[-1],
            translated_joints, verification_pairs)
        approach['merged'] = merged is not None
        approach['merge_reason'] = why
        if merged is not None:
            # 段階 1 と寄せを、押し込み姿勢から寄せの終点への 1 本の経路に
            # 置き換える (一度離れてから寄せで戻る動きをしない)。
            result['waypoints'] = []
            result['laterals'] = []
            waypoints = merged
    if waypoints:
        result['waypoints'] = result['waypoints'] + waypoints
        result['human_max'] = human_arm_max(result['waypoints'])
        result['laterals'] = list(result.get('laterals') or []) + [
            human_lateral(translated_joints, wp['base_position'], hand)
            for wp in waypoints]
        result['placement_after'] = spik.base_placement_metrics(
            translated_joints, waypoints[-1]['base_position'],
            waypoints[-1]['base_yaw'])
        names = list(result['joint_names'])
        av = waypoints[-1]['joint_angle_vector']
        result['legs_after'] = [math.degrees(av[names.index('ankle_joint')]),
                                math.degrees(av[names.index('knee_joint')])]
        result['turn_after'] = waypoints[-1]['turn_deg']


def _add_alt_arm_approach(robot, robot_arm, hand, post, turn_deg, result,
                          translated_joints, collision_joints,
                          verification_pairs, collision_pairs, base_limits):
    """横並びまで進んだ計画 ``result`` の終点が人の横 ``APPROACH_STOP_LATERAL``
    より遠ければ、人の腕を変えた寄せ (``_approach_alt_arms``) を解き、
    より近くへ寄れたら ``result`` の waypoint をそれに置き換える。"""
    if not (result.get('verified') and result.get('fraction', 0.0) >= 1.0
            and result['waypoints']):
        return
    lateral = human_lateral(translated_joints,
                            result['waypoints'][-1]['base_position'], hand)
    if lateral <= APPROACH_STOP_LATERAL:
        return
    if result.get('approach') is None:
        result['approach'] = dict(
            lateral_before=lateral, lateral_after=lateral, window=None,
            mode=None, reason=None, n_waypoints=0)
    approach = result['approach']
    stage1_palm_z = _lowered_palm_z(result)
    alt, info = _approach_alt_arms(
        robot, robot_arm, hand, post, turn_deg, translated_joints,
        collision_joints, verification_pairs, collision_pairs, base_limits,
        result.get('yaw_margin'), lateral,
        seed_av=(np.asarray(result['waypoints'][-1]['joint_angle_vector'],
                            dtype=np.float64)
                 if APPROACH_ALT_SEED == 'stage1' else None))
    approach['alt_arm'] = info
    if alt is None:
        return
    # 押し込み姿勢から、人の腕を変えた寄せの終点への 1 本の経路に置き換える。
    # 掌の高さの比べ方 (_lowered_palm_z) は段階 1 の腕のまま (肘を曲げた分
    # だけ高い計画が外れないように)。
    approach.update(
        mode='alt_arm', n_waypoints=len(alt), stage1_palm_z=stage1_palm_z,
        lateral_after=human_lateral(translated_joints,
                                    alt[-1]['base_position'], hand))
    _set_waypoints(result, alt, translated_joints, hand)


def _set_waypoints(result, waypoints, translated_joints, hand):
    """横並び移動の計画 ``result`` の waypoint を ``waypoints`` (押し込み姿勢の
    次から) に置き換え、終点から求める値を更新する。"""
    result['waypoints'] = waypoints
    result['human_max'] = human_arm_max(waypoints)
    result['laterals'] = [
        human_lateral(translated_joints, wp['base_position'], hand)
        for wp in waypoints]
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
                     time_limit=None, max_palm_z=None, alt_arms=True,
                     hint=None):
    """押し込み姿勢 ``post`` から、つないだ手を人の体の横へ下ろしながら
    ロボットも人と横並びになる位置・姿勢へ移り (段階 1、``_plan_stage1``)、
    さらに台車を人の方へ寄せる (段階 2、``plan_approach``) waypoint を計画する。

    人の掌の向き (``PALM_FACING_ORDER``) ごとに段階 1 を解き、横並びまで進んだ
    向きだけ段階 2 へ進める。寄せ後の横の距離が ``stop_lateral`` 以下の向きで
    打ち切り、打ち切れなければ全部解いて寄せ後の一番近い向き (差が 2 cm 以内
    なら回す量の小さい向き) を選ぶ。腕を下ろしきった掌が一番低い計画より
    ``LOWERED_HEIGHT_TOLERANCE`` を超えて高い計画は選ばない。台車の向きの範囲 ``GOAL_YAW_MARGIN_DEG``
    で ``APPROACH_FALLBACK_LATERAL`` 以内に寄れなければ、
    ``GOAL_YAW_FALLBACK_DEG`` まで広げて解き直す。上半身は ``UPPER_BODY_REFERENCE`` から
    ``UPPER_BODY_LIMIT_DEG`` に収める。

    ``translated_joints``・``collision_joints``・``base_limits`` と戻り値は
    ``_plan_stage1`` と同じ。加えて ``approach`` (寄せ: ``lateral_before``/
    ``lateral_after`` [m]、``window``、``mode``、``n_waypoints``、``reason``)、
    ``yaw_margin`` を持ち、``waypoints`` は寄せまでを含む。``approach`` が
    False なら段階 2 は解かず、横並びまで進んだ最初の解で止める。
    ``time_limit`` [秒] (None なら無制限) を過ぎたら探索をやめ、それまでの
    一番良い解を返す (``timed_out`` が True)。``max_palm_z`` [m] を与えると、
    下ろしきった掌がそれより高い下ろし方は試さない (呼び出し元に、それより
    LOWERED_HEIGHT_TOLERANCE 低い計画があるとき)。``alt_arms`` が False
    なら、寄せで人の腕も変える案 (``APPROACH_ALT_ARMS``) を試さない (呼び出し
    元で試すとき)。``hint`` (予備の解の ``palm_facing``・``arm_deg``・
    ``abduction_deg``) を与えると、その掌の向きから、その向きでは
    その下ろし方から試す (``_plan_stage1``)。
    """
    t0 = time.time()
    cache = _person_cache(post, turn_deg, collision_joints, base_limits)
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
    if COMFORT_ORDER and approach:
        facings = comfort_facing_order(post, turn_deg, robot_arm, hand,
                                       translated_joints, facings)
    hint_facing = (hint or {}).get('palm_facing')
    if hint_facing in facings:
        # 予備の解の掌の向きから試す (残りは上の順のまま)。
        facings = (hint_facing,) + tuple(f for f in facings
                                         if f != hint_facing)
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
        _check_stop()
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
                palm_limit = max_palm_z
                heights = [z for z in map(_lowered_palm_z, results)
                           if z is not None]
                if PRUNE_HIGH_LOWERINGS and heights:
                    # これより高い下ろし方は _low_arm_results で外れる。
                    limit = min(heights) + LOWERED_HEIGHT_TOLERANCE
                    palm_limit = (limit if palm_limit is None
                                  else min(palm_limit, limit))
                result = _plan_stage1(
                    robot, robot_arm, hand, post, turn_deg,
                    translated_joints, collision_joints, verification_pairs,
                    collision_pairs, base_limits,
                    attempts_per_pose=attempts_per_pose,
                    facings=group, yaw_margin_deg=margin, deadline=deadline,
                    cache=cache, bundle_facings=PALM_FACING_ORDER,
                    max_palm_z=palm_limit, bundle_first=approach,
                    hint=hint if hint_facing in group else None)
                result.update(yaw_margin=margin, approach=None)
                if approach and result['verified'] \
                        and result['fraction'] >= 1.0 and not out_of_time():
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
    if APPROACH_ALT_ARMS and alt_arms and approach and not out_of_time():
        with _upper_body_limit(human_yaw):
            _add_alt_arm_approach(
                robot, robot_arm, hand, post, turn_deg, best,
                translated_joints, collision_joints, verification_pairs,
                collision_pairs, base_limits)
    best['tries'] = len(results)
    for key, threshold in (('first_verified_time', MIN_FRACTION),
                           ('first_full_time', 1.0)):
        times = [ft - t0 for r in results
                 for ft, fraction in r.get('fraction_times', ())
                 if fraction >= threshold]
        best[key] = min(times) if times else None
    # 採った解が、最初に見つかった解 (MIN_FRACTION 以上進めた解) か (表示用)。
    best['chosen_try'] = next(
        i for i, r in enumerate(results) if r is best) + 1
    best['first_verified_try'] = next(
        (i + 1 for i, r in enumerate(results)
         if any(f >= MIN_FRACTION for _, f in r.get('fraction_times', ()))),
        None)
    chosen_times = [ft - t0 for ft, fraction in best.get('fraction_times', ())
                    if fraction >= MIN_FRACTION]
    best['chosen_time'] = min(chosen_times) if chosen_times else None
    best['timed_out'] = out_of_time()
    best['compute_time'] = time.time() - t0
    return best


def comfort_facing_order(post, turn_deg, robot_arm, hand, translated_joints,
                         facings):
    """掌の向き ``facings`` を、その向きの下ろし方 (最初の上腕の組・脇の開き)
    で下ろしきった掌の人の項の最小の小さい順に並べ替える (study/sbs-comfort)。
    使えない向きは後ろ (今の順)。"""
    palm = human_palm_from_press(post, turn_deg, robot_arm)
    human_arm = HumanArm(translated_joints, hand, palm, turn_deg, robot_arm)
    degs, _ = LOWERED_ARM_GROUPS[0]
    abductions = sorted({ab for d, ab in LOWERED_ARM_GROUPS if d == degs})
    score = {}
    for name in facings:
        values = []
        for ab in abductions:
            for deg in degs:
                lw = human_arm.lowering(deg, ab, name)
                if lw is not None:
                    values.append(lowering_comfort(human_arm, lw))
        score[name] = min(values) if values else float('inf')
    return tuple(sorted(facings, key=lambda name: score[name]))


def add_human_arm_arguments(parser):
    """人の前腕の障害物・手首の評価を切り替える引数を ``parser`` に足す
    (``apply_human_arm_arguments`` で反映する)。"""
    parser.add_argument(
        '--transition-forearm-obstacle', choices=('on', 'off'),
        default='on' if HUMAN_FOREARM_OBSTACLE else 'off',
        help='横並び移動で人の前腕・手の甲をロボットの前腕の障害物にするか。')
    parser.add_argument(
        '--transition-forearm-margin', choices=FOREARM_MARGIN_MODES,
        default=FOREARM_MARGIN_MODE,
        help='移動先のバッチ IK で前腕・手の甲との距離をどう効かせるか '
             '(shrink: 障害物を細らせる、pair: 細らせずペアの判定距離を '
             '--transition-forearm-pair-margin にする。既定 {})。'.format(
                 FOREARM_MARGIN_MODE))
    parser.add_argument(
        '--transition-forearm-pair-margin', type=float,
        default=FOREARM_PAIR_MARGIN,
        help='pair のときの判定距離 [m] (既定 {})。'.format(
            FOREARM_PAIR_MARGIN))
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
    parser.add_argument(
        '--transition-wrist-ext-limit', type=float, default=None,
        help='hand_fixed の背屈の目安 [度] (既定 {})。'.format(
            '掌屈と同じ' if HUMAN_WRIST_EXT_MAX_DEG is None
            else '{:.0f}'.format(HUMAN_WRIST_EXT_MAX_DEG)))
    parser.add_argument(
        '--transition-twist-limit', type=float, default=None,
        help='押し込み時からの前腕のひねりの目安 [度] (掌の向きの候補と '
             'hand_fixed の両方、既定は見ない = 絶対的な回内・回外だけで'
             '判定)。')
    parser.add_argument(
        '--transition-human-arm', choices=HUMAN_ARM_CHECKS,
        default=HUMAN_ARM_CHECK,
        help='横並び移動で、腕を下ろす途中の人の肩・肘・前腕が目安を超える'
             '下ろし方を外すか (既定 {})。'.format(HUMAN_ARM_CHECK))
    parser.add_argument(
        '--transition-lowering-batch', choices=LOWERED_ARM_BATCHES,
        default=LOWERED_ARM_BATCH,
        help='移動先のバッチ IK の目標の作り方 (既定 {})。'.format(
            LOWERED_ARM_BATCH))
    parser.add_argument(
        '--transition-forearm-press-allowance', choices=('on', 'off'),
        default='on' if FOREARM_PRESS_ALLOWANCE else 'off',
        help='押し込み時にすでに人の前腕・手の甲に近い分を、腕を下ろすにつれ'
             '減らしながら許すか。')
    parser.add_argument(
        '--transition-merge-approach', choices=('on', 'off'),
        default='on' if MERGE_APPROACH else 'off',
        help='寄せの終点へ押し込み姿勢から 1 本の経路で解き直すか。')
    parser.add_argument(
        '--transition-lateral-weight', type=float,
        default=GOAL_LATERAL_WEIGHT,
        help='移動先の台車の横の距離のコストの重み (既定 {})。'.format(
            GOAL_LATERAL_WEIGHT))
    parser.add_argument(
        '--transition-alt-arms', choices=('on', 'off'),
        default='on' if APPROACH_ALT_ARMS else 'off',
        help='寄せで人の腕の姿勢も変える案 (APPROACH_ALT_ARMS) を試すか。')
    parser.add_argument(
        '--transition-alt-seed', choices=APPROACH_ALT_SEEDS,
        default=APPROACH_ALT_SEED,
        help='人の腕を変える寄せの移動先のバッチ IK の初期値 (既定 {})。'
             .format(APPROACH_ALT_SEED))
    parser.add_argument(
        '--transition-jump-subdivisions', type=int, nargs='*',
        default=list(PATH_JUMP_SUBDIVISIONS),
        help='経路で関節角が飛んだ区間を分けて解き直す数 (順に試す、何も'
             '書かなければ分けない。既定 {})。'.format(
                 ' '.join(str(v) for v in PATH_JUMP_SUBDIVISIONS)))


def apply_human_arm_arguments(args):
    global HUMAN_FOREARM_OBSTACLE, HUMAN_WRIST_CHECK
    global HUMAN_WRIST_RADIAL_MAX_DEG, HUMAN_WRIST_ULNAR_MAX_DEG
    global HUMAN_WRIST_FLEX_MAX_DEG, LOWERED_ARM_BATCH, GOAL_LATERAL_WEIGHT
    global MERGE_APPROACH, FOREARM_PRESS_ALLOWANCE
    global HUMAN_WRIST_EXT_MAX_DEG, HUMAN_ARM_CHECK
    global LOWERED_TWIST_MAX_DEG, HUMAN_WRIST_TWIST_MAX_DEG
    global FOREARM_MARGIN_MODE, FOREARM_PAIR_MARGIN
    global APPROACH_ALT_ARMS, APPROACH_ALT_SEED, PATH_JUMP_SUBDIVISIONS
    APPROACH_ALT_ARMS = args.transition_alt_arms == 'on'
    APPROACH_ALT_SEED = args.transition_alt_seed
    PATH_JUMP_SUBDIVISIONS = tuple(args.transition_jump_subdivisions)
    FOREARM_MARGIN_MODE = args.transition_forearm_margin
    FOREARM_PAIR_MARGIN = args.transition_forearm_pair_margin
    if args.transition_twist_limit is not None:
        LOWERED_TWIST_MAX_DEG = HUMAN_WRIST_TWIST_MAX_DEG = (
            args.transition_twist_limit)
    if args.transition_wrist_ext_limit is not None:
        HUMAN_WRIST_EXT_MAX_DEG = args.transition_wrist_ext_limit
    HUMAN_ARM_CHECK = args.transition_human_arm
    MERGE_APPROACH = args.transition_merge_approach == 'on'
    FOREARM_PRESS_ALLOWANCE = args.transition_forearm_press_allowance == 'on'
    HUMAN_FOREARM_OBSTACLE = args.transition_forearm_obstacle == 'on'
    HUMAN_WRIST_CHECK = args.transition_human_wrist
    LOWERED_ARM_BATCH = args.transition_lowering_batch
    GOAL_LATERAL_WEIGHT = args.transition_lateral_weight
    if args.transition_wrist_limits is not None:
        (HUMAN_WRIST_RADIAL_MAX_DEG, HUMAN_WRIST_ULNAR_MAX_DEG,
         HUMAN_WRIST_FLEX_MAX_DEG) = args.transition_wrist_limits


def human_arm_text(arm):
    """人の腕の目安の 1 行の説明 (目安を超えたものに印を付ける)。"""
    def mark(value, limit):
        return '{:.0f}{}'.format(
            value, ' (!)' if limit is not None and value > limit else '')
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
        return '{:.0f}{}'.format(
            value, ' (!)' if limit is not None and value > limit else '')
    return ('、手首 (ロボットの手に固定) 掌屈 {}・背屈 {}・撓屈 {}・尺屈 {}・'
            'ひねり {} 度'.format(
                mark(arm['flexion_deg'], HUMAN_WRIST_FLEX_MAX_DEG),
                mark(arm['extension_deg'],
                     HUMAN_WRIST_FLEX_MAX_DEG if HUMAN_WRIST_EXT_MAX_DEG
                     is None else HUMAN_WRIST_EXT_MAX_DEG),
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
            text += ' / 寄せ: 横 {:.2f} -> {:.2f} m ({}{})'.format(
                approach['lateral_before'], approach['lateral_after'],
                approach['mode'],
                '、押し込みから 1 本の経路' if approach.get('merged') else '')
        else:
            text += ' / 寄せない ({})'.format(approach['reason'])
    if transition['reason']:
        text += ' / 途中で止めた: {}'.format(transition['reason'])
    return text
