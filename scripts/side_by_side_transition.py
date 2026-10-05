#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""掌を押し込んだ後、つないだ手を人の体の横へ下ろしながら、ロボットも人と
横並びの位置・姿勢へ移る区間 (横並び移動) を計画する。

押し込み (``solve_palm_ik`` の後処理姿勢) までは従来どおりで、ロボットは
膝を曲げて上体を前に出し (身を乗り出し)、人の手の位置に合わせて立つ。
押し込んだ後は、

- つないだ手 (人の掌) を、人が肩から腕を下ろした位置
  (``HumanArm.lowering``: 上腕を真下から前へ ``LOWERED_ARM_DEG_GROUPS``
  の角度・外へ少し開き、肘を少し曲げた腕。掌は人側・前・後ろのうち前腕の
  ひねりが一番少ない向き) まで動かす。人が一番低く下ろせる角度を優先する。
- 同時に、台車を人と前後ずれ 0・同じ向きの位置へ動かし、上体を起こす
  (移動先は台車も動かすバッチ IK で、前後ずれ・向きのずれ・上体の前後
  ずれの小さい解を選ぶ、``solve_goal_candidates``)。腕は押し込み時と
  同じ解の枝 (腕を前に出した形) に限る (``GOAL_BRANCH_JOINT_RANGE_DEG``)。
- 移動先では握りの向き (人とロボットの指先のなす角) を変えてよく
  (``GOAL_TURN_DEG_GROUPS``)、しゃがまない (脚を伸ばした) 解を優先する
  (``GOAL_CROUCH_WEIGHT``)。視線は掌から前 (``forward_gaze_point``) へ移す。
- 途中の waypoint は、人の腕 (肩まわりの上腕の向き・肘の曲げ・前腕の
  ひねり)・握りの向き・台車を補間し、ロボットの腕・脚・首を解き直す。

人の腕は肩を動かさず、手首から先は押し込み時の手に対して剛体 (``HumanArm``)。
各 waypoint で、人の腕が届くか (``reach``)・肘の曲げ・手首の曲げ・前腕の
ひねりの変化を記録し、腕が届かなくなる手前で止める。ビューアはこれを使って
人の腕をロボットの手に追従させる。

干渉の検証は人体との距離 (6 cm) と自己干渉。人の差し出した腕は waypoint
ごとに動かした骨格で見る。ロボットの差し出す腕の肘から先と、人の差し出した
手・前腕は手を合わせているので距離を見ない。
"""

import math
import time
import zlib

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import normalize_mask
from skrobot.coordinates.math import rpy_matrix

import solve_palm_ik as spik


# 下ろした腕 (人が肩から腕を下ろす): 上腕を真下から前へ
# LOWERED_ARM_DEG_GROUPS の各角度に上げ、外へ LOWERED_ARM_ABDUCTION_DEG
# 開き、肘は LOWERED_ELBOW_FLEX_DEG だけ曲げる (前腕は上腕より前へ)。
# 手首から先は押し込み時の手に対して剛体のまま、掌の法線を人側・前・
# 後ろのうち前腕まわりのひねりが一番少ない向きへ回す
# (LOWERED_PALM_FACINGS)。肘を曲げて前腕だけを上げる (肘を直角にして
# 前腕を水平にする) 形は人から見ると手を下ろしていないので使わない。
# 移動先のバッチ IK は組 ((上腕の角度 3 つ), 脇の開き) ごとに 3 つを目標に
# し、人が一番低く下ろせる角度の解を優先する。前の組で横並びまで進めなければ
# 次の組を試す (LOWERED_ARM_GROUPS)。
# 脇は、胴体の干渉円柱 (肩の関節から横へ約 13 cm) の外に手が出る 25 度開く
# (10 度だと低い目標が人体との距離 6 cm の検証でほぼ全部落ちる)。25 度で
# 移動先・経路が人の胴体・下ろした上腕に近すぎる人は 40 度開く組も試す。
# 腕を低く下ろす方を脇を狭める方より優先し、上腕 0/15/30 度は 25 -> 40 度、
# その後に 45/60/75 度を 25 -> 40 度の順。
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

# 掌の向きの最初の候補 'natural': 前腕をひねらず (ひねり 0)、腕を下ろす回転
# だけで手を動かした向き。人側・前・後ろのどれかに合わせようと前腕をひねると、
# 親指が上へ回り指先が下を向きにくい。ひねらなければ手は親指側の軸まわりに
# 回って指先が下を向く。終点の掌の法線が外向き (ロボット側) の成分
# LOWERED_NATURAL_MAX_OUTWARD を超える、上下の成分が
# LOWERED_NATURAL_MAX_VERTICAL を超えるとき (掌が下や上を向きすぎる) は
# 使わない (下向きは少しなら許す)。
LOWERED_NATURAL_MAX_OUTWARD = 0.3
LOWERED_NATURAL_MAX_VERTICAL = 0.5

# 掌の向きは LOWERED_FACING_ORDER の順に試し、その向きで移動先・経路が
# 通らなければ次の向きを試す (``plan_transition`` の ``facing_variants``)。
# 掌は外向き (ロボット側) にはしない。前腕のひねりがこの角度 [度] を超える
# 向きは試さない。押し込みで掌を上に向けて (回外して) 差し出した人が、腕を
# 下ろして掌を後ろに向ける (回内する) と 150 度前後ひねる。
LOWERED_TWIST_MAX_DEG = 180.0

# 腕を下ろしきったときに、押し込み時の手首の曲げをどれだけ戻すか (0: 戻さず
# 手を前腕に剛体で付ける、1: 指先が前腕の向き)。腕を下ろす間に線形に戻す。
# 戻さないと押し込み時の曲げ (約 20 度) が残り、前腕のひねりで曲げの向きが
# 上 (親指側) へ回って、前腕は下を向いても指先は前に残る。
LOWERED_WRIST_STRAIGHTEN = 1.0

# 前腕のひねりを始める、腕を下ろす動きの割合 (0 なら下ろしながら同時に
# ひねる)。同時にひねると、前に出した前腕のまわりで手が回って親指が上へ
# 向く。先に腕を下ろして指先を下へ向け、下がった前腕のまわりで掌を回す。
LOWERED_TWIST_START = 0.5

# 掌の向きの候補を試す順。腕を下ろしきって掌が後ろ向きなら、ロボットは
# 人の手の後ろから肘を伸ばして手を合わせられ、肘を曲げずに済み干渉しにくい。
# 'natural' は前腕をひねらない向き。
LOWERED_FACING_ORDER = ('backward', 'natural', 'inward', 'forward')

# 移動先の台車の前後位置を、人の立ち位置からどれだけずらしてよいか [m]。
# 狭い方から試し、解けなければ広げる。合成データでは横並びまで進めた人は
# 全員 ±0.05 m で解けており、広げて解けるのは移動先の検証・経路に通らない
# 解だけだった (広げると解けない人で 1 組あたりバッチ IK が 4 回になる) ので、
# ±0.05 m だけにしている。
DEFAULT_GOAL_X_MARGINS = (0.05,)

# 移動先のバッチ IK の目標 1 つあたりの初期値の数 (押し込みの IK の
# DEFAULT_ATTEMPTS_PER_POSE = 512 の半分)。合成データで 512 と成功数は
# 同じで、1 回あたり約 0.32 -> 0.23 秒。
GOAL_ATTEMPTS_PER_POSE = 256

# 移動先のコストの重み: 人の正面方向の前方ずれ [m]、向きのずれ [rad]、
# 上体の前後ずれ |ankle + knee| [rad] (脚は平行リンクで、胴体は台車の
# 中心から約 0.25 m x (sin(-knee) - sin(ankle)) 前に出る)。
GOAL_FRONT_OFFSET_WEIGHT = 100.0
GOAL_FACING_YAW_WEIGHT = spik.DEFAULT_FACING_YAW_WEIGHT
GOAL_LEG_OFFSET_WEIGHT = 10.0

# 移動先のコストで、しゃがみ量 ankle + |knee| [rad] (どちらも 0 で脚を
# 伸ばしきった一番高い姿勢) に掛ける重み。横並びではできるだけしゃがまない
# 解を優先する。
GOAL_CROUCH_WEIGHT = 30.0

# 移動先のコストで、差し出す腕の手首ヨー・ロールが可動域の端に近いほど
# 増やす項。手首が端にあると、経路で手首をそれ以上曲げられず掌を回せない
# (関節の可動域に対する角度の割合が ``GOAL_WRIST_MARGIN_START`` を超えた
# 分を、``GOAL_WRIST_RATIO_RANGE`` を 1 として 2 乗する)。手首ピッチは
# 可動域が狭く (約 4 度) 使わないので見ない。
GOAL_WRIST_WEIGHT = 30.0
GOAL_WRIST_MARGIN_START = 0.7
GOAL_WRIST_RATIO_RANGE = 0.3
GOAL_WRIST_AXES = ('y', 'r')

# 移動先のコストで、ロボットの指先 (手先の +X) が下を向いていない分
# (1 - 下向きの成分、0〜2) に掛ける重み。握りの向きを自由にすると、手首を
# 曲げたまま握りの向きだけが回り、指先が上を向いたまま横並びになる。
GOAL_ROBOT_FINGER_DOWN_WEIGHT = 30.0

# 移動先のコストで、差し出す腕の手首の曲げ (ロール・ヨーの |角度| [rad]
# の和) に掛ける重み。手首を曲げたままだと、横並びで手首が窮屈に見える。
GOAL_WRIST_BEND_WEIGHT = 10.0

# 経路で握りの向きを、押し込み時の値から移動先の解の値へ補間して固定するか
# (False なら経路でも握りの向きを拘束せず、直前の解から解き進める)。自由に
# すると直前の解の近くに留まり、移動先で選んだ指先の向きへ回っていかない。
PATH_INTERPOLATE_TURN = True

# 経路の首を、押し込み時の角度から移動先 (前を見る姿勢) の角度へ関節角で
# 線形に補間するか (False なら waypoint ごとに視線の IK で解く)。視線の IK
# で解くと、台車の向きの変化に合わせて首が細かく往復し、ロールが可動域の
# 端 (±5 度) で反対側へ跳ぶ。
PATH_INTERPOLATE_NECK = True

# 移動先で試す握りの向き (人の掌の法線まわりに、ロボットの指先を人の指先
# から回す角度 [度]、solve_palm_ik.palm_target_rot の turn_deg)。押し込み
# 時の向きから変わってよい (経路の途中で少しずつ回す)。向きごとに 1 回の
# バッチ IK (約 0.3 秒、窓を広げるとその回数分) なので、前の組で移動先の
# IK が 1 つも解けなかったときだけ次の組を試す (合成データでは ±90 度は
# 選ばれなかった)。
GOAL_TURN_DEG_GROUPS = ((0.0,), (90.0, -90.0))
GOAL_TURN_DEGS = tuple(t for group in GOAL_TURN_DEG_GROUPS for t in group)

# 移動先のバッチ IK の解のうち、差し出す腕の肩ヨー・手首ヨーが押し込み
# 姿勢の角度からこの範囲 [度] を超えるものは捨てる (バッチ IK はソルバーを
# 作ったときの関節の可動域をキャッシュするので、呼び出しごとに可動域を
# 絞っても効かない)。腕を前に出した押し込み姿勢と、腕を体の横に
# 垂らして肩・手首を大きくひねった姿勢は別の解の枝で、手を合わせたまま
# 一方から他方へは移れない (途中で手が大きく回り、人の手首がねじれる)。
# 垂らす姿勢を移動先に選ぶと、経路は前に出した腕のまま膝を曲げて手を
# 下げ、最後に肩・手首が 120 度以上飛んで止まるため、押し込みと同じ枝の
# 解だけを探す。
GOAL_BRANCH_JOINT_RANGE_DEG = 60.0
GOAL_BRANCH_JOINTS = ('{}_shoulder_y_joint', '{}_wrist_y_joint')

# 移動先の候補 (腕の低い目標から、同じ目標ではコストの安い順) のうち、
# 経路まで試す数と、押し込み IK・検証まで試す数 (どちらも目標 1 つ
# あたり)。手を低く下ろす目標は人の胴体に近く、検証に落ちる候補が数百
# 続くことがある (1 つ数十 ms)。
MAX_GOAL_CANDIDATES = 2
MAX_GOAL_CHECKS = 8

# 1 waypoint あたりの台車・掌の移動量の上限 (位置 [m]・向き [rad])。
MAX_STEP = 0.05
MAX_ANGLE_STEP = math.radians(5.0)

# 隣り合う waypoint の間で、関節角がこれより大きく変わったら IK が別の解へ
# 飛んだとみなして、そこで止める [rad]。首は視線の IK で別の解へ飛ぶことは
# なく、視線を掌から前へ移す最初の区間で大きく回るので見ない。
MAX_JOINT_STEP = math.radians(20.0)

# waypoint の数は、押し込み姿勢と移動先の姿勢の関節角の差がこの角度
# [rad] ごとに 1 区間以上になるようにも決める (台車・掌の移動量だけで
# 決めると、関節の動きの大きい人で 1 区間の変化が MAX_JOINT_STEP を
# 超える)。
PLANNED_JOINT_STEP = math.radians(15.0)

# 移動先までのうち進めた割合がこれに満たなければ、横並び移動はしない。
MIN_FRACTION = 0.3

# 人の腕の目安 (表示・ログ用、超えても止めない。届くか (reach <= 1) だけ
# は止める)。
HUMAN_ELBOW_FLEX_MAX_DEG = 145.0
HUMAN_WRIST_BEND_MAX_DEG = 70.0
HUMAN_TWIST_CHANGE_MAX_DEG = 90.0


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
    rot = np.asarray(post['target_rot'], dtype=np.float64)
    if robot_arm == 'l':
        rot = np.column_stack([rot[:, 0], -rot[:, 1], -rot[:, 2]])
    rot = spik._turn_about_y(rot, -turn_deg)
    fingers, normal = rot[:, 0], -rot[:, 1]
    position = (np.asarray(post['target_position'], dtype=np.float64)
                - normal * spik.POST_PROCESS_TARGET_HOVER_OFFSET)
    return palm_from_frame(position, np.column_stack(
        [fingers, normal, np.cross(fingers, normal)]))


def press_target(palm, turn_deg, robot_arm):
    """人の掌 ``palm`` への押し込み目標 (``solve_post_process`` と同じ)。"""
    position, rot = palm_frame(palm)
    return Coordinates(
        pos=(position + rot[:, 1]
             * spik.POST_PROCESS_TARGET_HOVER_OFFSET).tolist(),
        rot=spik.palm_target_rot(palm, turn_deg, robot_arm))


def _outward(joint_positions, hand):
    """人の差し出した手の側の外向き (水平の単位ベクトル)。"""
    other = 'L' if hand == 'R' else 'R'
    v = (np.asarray(joint_positions['{}Shoulder'.format(hand)])
         - np.asarray(joint_positions['{}Shoulder'.format(other)]))
    v[2] = 0.0
    return _unit(v)


class HumanArm(object):
    """人の差し出した腕 (肩は固定) を、掌の位置姿勢に追従させる。

    手 (手首から先) は押し込み時の掌に対して剛体のまま動かし、手首の位置
    を決める。肘は上腕・前腕の長さ (押し込み時の骨格) を保つ 2 リンクの
    IK で、下・少し外を向く側に置く。
    """

    def __init__(self, joint_positions, hand, palm):
        self.hand = hand
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
    #
    # 腕は「上腕の向き ua と肘の蝶番軸 n の組 (回転行列 [ua, n, ua x n])」・
    # 肘の曲げ・前腕まわりのひねりで表す。前腕の向きは
    # cos(曲げ) ua + sin(曲げ) (n x ua)。上腕の組は slerp、曲げとひねりは
    # 線形に補間するので、手は肩を中心に回って下りる。

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
        ``LOWERED_ELBOW_FLEX_DEG`` 曲げた腕へ下ろす動き。掌は
        ``LOWERED_PALM_FACINGS`` のうち ``facing`` (既定は前腕のひねりが
        一番少ない向き) にする。
        ``lowering_pose`` に渡す dict (``palm``: 下ろしきった掌の ``(位置,
        回転行列)``、``arm_deg``、``abduction_deg``、``palm_facing``) を
        返す。"""
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

# 台車の経路を、人の立ち位置を中心に距離と方位を補間する回り込みにするか
# (False なら押し込み位置から移動先へ直線)。人の正面から横へ移るとき、直線
# (弦) は人の足元に近いところを通り、車輪が人のすねに近づく (90 度回り込む
# と弦の中点は人から端点の約 0.71 倍の距離)。
BASE_PATH_ORBIT = True


class BasePath(object):
    """台車の ``start`` から ``goal`` (どちらも ``(x, y, yaw)``) への経路。
    ``center`` (人の立ち位置 xy) を与えれば、そのまわりに距離と方位を線形に
    補間して回り込む (``BASE_PATH_ORBIT``)。yaw は線形に補間する
    (``goal`` の yaw は ``start`` から近い回り方の値にしておく)。"""

    def __init__(self, start, goal, center=None):
        self.start = np.asarray(start, dtype=np.float64)
        self.goal = np.asarray(goal, dtype=np.float64)
        self.center = None
        if BASE_PATH_ORBIT and center is not None:
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


def _place(robot, angle_vector, base):
    robot.angle_vector(np.asarray(angle_vector, dtype=np.float64))
    robot.newcoords(Coordinates(pos=[float(base[0]), float(base[1]), 0.0],
                                rot=rpy_matrix(float(base[2]), 0.0, 0.0)))


LEG_JOINT_NAMES = ('ankle_joint', 'ankle_joint_mimic', 'knee_joint',
                   'knee_joint_mimic')


# 握りの向きを自由にするときの手先の回転の拘束 (手先座標系の +Y = 掌の
# 法線まわりだけ自由、掌は合わせたまま指先の向きが回る)。
FREE_TURN_ROTATION_MASK = (1, 0, 1)

# 横並び移動の IK で握りの向きを自由にするか (True なら移動先・経路とも
# 掌の法線まわりの回転を拘束しない。False なら GOAL_TURN_DEG_GROUPS の
# 角度に固定して解く)。押し込みまでは握りの向きを決めて合わせ、そこから
# 手を下ろしやすい向きへ少しずつ回す (合成データ 42 人で waypoint ごとに
# 最大 8 度、移動先で ±45 度以内)。固定と比べて成功数は同じ (36/42) で、
# 移動先の knee が中央値 -46 -> -19 度とほとんどしゃがまなくなり、時間も
# 平均 0.84 -> 0.53 秒。バッチ IK の回転の拘束が違うので、別にコンパイル
# される (warmup_batch_ik)。
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


def _press_ik(robot, robot_arm, target, gaze_point, free_legs=False,
              free_turn=False):
    """台車をいまの位置に固定し、腕・腰を押し込み目標 ``target`` に、カメラ
    の光軸を ``gaze_point`` に向ける (``solve_palm_ik.solve_post_process``
    と同じ IK)。脚 (``LEG_JOINT_NAMES``) は ``free_legs`` でなければいまの
    角度に固定する (呼び出し側が補間して与える)。``free_turn`` なら掌の
    法線まわりの回転 (握りの向き) は拘束しない。視線が解けなければ首は
    そのままで腕だけを解く (``gaze_point`` が None なら視線は解かない)。
    腕が解けなければ ``False`` (ロボットは呼び出し前の姿勢に戻る)。"""
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    link_list = [link for link in whole_body.link_list
                 if free_legs or link.joint.name not in LEG_JOINT_NAMES]
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    stop = spik.DEFAULT_POST_PROCESS_IK_STOP
    thre = spik.DEFAULT_POST_PROCESS_IK_THRE
    rthre = spik.DEFAULT_POST_PROCESS_IK_RTHRE
    gaze_rthre = spik.DEFAULT_POST_PROCESS_GAZE_IK_RTHRE
    hand_mask = _rotation_mask(free_turn)
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
        link_list=[link_list, robot.head.link_list],
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
    """首だけで、カメラの光軸を ``gaze_point`` へ向ける。今の首の角度と
    首 0 度の両方から解き、視線のずれの小さい方にする (押し込み時の首から
    前を向くまで大きく回すと、今の角度からでは解けないことがある)。視線の
    ずれ [rad] を返す。"""
    gaze_coords = spik.camera_optical_coords(robot)
    joints = [link.joint for link in robot.head.link_list]
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


class TransitionChecker(object):
    """横並び移動の各姿勢の干渉検証 (自己干渉の貫通・人体からの距離)。
    ロボットの差し出す腕の肘から先と、人の差し出した手・前腕
    (``offered_hand_obstacle_indices``) の組は、手を合わせているので
    見ない。"""

    def __init__(self, robot_arm, hand, verification_pairs,
                 clearance=spik.DEFAULT_HOVER_HUMAN_CLEARANCE,
                 tolerance=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE):
        self.clearance = clearance
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
        # 台車の車輪と人の脚などの組 (人の腕は動くので除く)。腕・腰・脚の
        # 姿勢によらず台車の位置だけで距離が決まる。
        self.base_pairs = None
        if source is not None:
            arm = offered | set(_lowered_arm_slots(hand))
            pairs = [pair for pair in source
                     if isinstance(pair[1], int) and pair[1] not in arm
                     and pair[0].name.startswith('wheel')]
            self.base_pairs = (
                source.with_pairs(pairs)
                if isinstance(source, spik.VerificationPairs) else pairs)

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
                cull_distance=self.clearance)
            if not clearances:
                continue
            index = min(clearances, key=clearances.get)
            if clearances[index] < self.clearance:
                return '経路の車輪が人体 ({}) まで {:.3f} m'.format(
                    names[index], clearances[index])
        return None

    def check(self, robot, joint_positions):
        """``robot`` の今の姿勢を、人の骨格 ``joint_positions`` に対して
        検証し、問題があればその説明、無ければ ``None`` を返す。"""
        dist, pair = spik.collision_pairs_min_distance(
            robot, self.self_pairs, None, return_pair=True)
        if dist < -self.tolerance:
            return '自己干渉 {:.3f} m ({} x {})'.format(
                -dist, *spik.collision_pair_name(pair))
        if not joint_positions or not self.clearance_pairs:
            return None
        clearances = spik.human_obstacle_clearances(
            robot, self.clearance_pairs,
            spik.human_body_obstacles(joint_positions),
            cull_distance=self.clearance)
        if not clearances:
            return None
        index = min(clearances, key=clearances.get)
        if clearances[index] < self.clearance:
            return '人体 ({}) まで {:.3f} m'.format(
                spik.human_obstacle_names()[index], clearances[index])
        return None


def _branch_joint_indices(robot, robot_arm):
    """差し出す腕の ``GOAL_BRANCH_JOINTS`` の ``joint_list`` での添字。"""
    names = [joint.name for joint in robot.joint_list]
    return [names.index(name.format(robot_arm)) for name in GOAL_BRANCH_JOINTS
            if name.format(robot_arm) in names]


# 移動先のバッチ IK に、押し込みの干渉ペア (collision_pairs.json) へ足す組
# (_transition_pairs)。移動先の検証で落ちる候補はほぼ、ロボットの差し出す
# 腕の前腕・肘が人の下ろした上腕・胴体の横に 6 cm 以内まで近づくもの
# (下ろした上腕は押し込みの IK には無い) と、差し出さない腕の前腕・手が腰に
# めり込む自己干渉 (押し込みの干渉ペアには自己干渉の組が無い) だった。
# IK に避けさせると、移動先の検証に落ちる候補が約半分になり、合成データで
# 移動先まで 100% の人が 33 -> 35/41 人に増えた。
TRANSITION_EXTRA_PAIRS = True
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
    """横並び移動の移動先のバッチ IK に、押し込みの干渉ペアへ足す組。
    移動先の検証で落ちるのはほぼ、ロボットの差し出す腕の前腕・肘と人の
    下ろした上腕・胴体の横、差し出さない腕の前腕・手と腰の自己干渉
    (押し込みの干渉ペアには無い組)。"""
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


def _batch_ik(robot, robot_arm, hand, targets, seed_av, collision_joints,
              collision_pairs, base_limits, attempts_per_pose,
              free_turn=False, lowered_arms=()):
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
        if TRANSITION_EXTRA_PAIRS:
            pairs += _transition_pairs(robot, robot_arm, hand)
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
    """横並び移動の移動先のバッチ IK (``_batch_ik``) を、実際の呼び出しと
    同じ形 (目標 3 つ・障害物・干渉ペア・回転の拘束) で 1 回解いて、JAX の
    トレース・コンパイルを前倒しで済ませる。握りの向きを自由にしない
    (``GOAL_FREE_TURN`` が False) ときは ``solve_person_ik`` と同じ形なので
    何もしない。``seed_av`` は押し込み姿勢など、解ける目標を作るための
    姿勢。"""
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
                          x_margins=None,
                          turn_degs=GOAL_TURN_DEGS,
                          attempts_per_pose=GOAL_ATTEMPTS_PER_POSE,
                          branch_range_deg=GOAL_BRANCH_JOINT_RANGE_DEG,
                          free_turn=False, lowered_arms=()):
    """下ろした手の候補 ``goal_palms`` (腕の低い順の 3 つの ``(位置,
    回転行列)``) への押し込み目標を、握りの向き ``turn_degs`` ごとに 1 回の
    バッチ IK で解き、移動先の台車・姿勢の候補を前後の窓の狭い方から
    (候補が出るまで) 集めて、腕の低い目標から、同じ目標ではコスト (曲げ量
    + 前方ずれ + 向きのずれ + 上体の前後ずれ + しゃがみ量) の安い順に返す。

    台車の可動域 (ワールドの x/y に沿った箱) で人の前後を絞れるよう、人の
    立ち位置まわりに回して人の正面方向が +x の座標系で解き、解いた台車の
    位置姿勢は元の座標系に戻す。左右は差し出した手の側、向きは人の正面
    方向 ±30 度に絞る。

    Returns
    -------
    (candidates, x_margin)
        ``candidates`` は ``(cost, angle_vector, base (x, y, yaw),
        turn_deg, goal_palms の添字)`` のリスト、``x_margin`` は候補が
        出た窓。
    """
    if x_margins is None:
        x_margins = DEFAULT_GOAL_X_MARGINS
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
    # 下ろした手は人の体の左右どちらか (人の正面が +x なので左は +y) に
    # 来るので、差し出した手首の位置 (体の中心線をまたぐことがある) では
    # なく、手の左右で台車の側を決める。
    limits[1] = spik.restrict_base_y_range_to_hand_side(
        limits[1], 1.0 if hand == 'L' else -1.0)
    limits[2] = spik.restrict_base_yaw_range_to_human_facing(
        limits[2], 0.0,
        margin=math.radians(spik.DEFAULT_BASE_YAW_FACING_MARGIN_DEG))
    names = [joint.name for joint in robot.joint_list]
    ankle, knee = names.index('ankle_joint'), names.index('knee_joint')
    bend_cost_indices = spik._joint_bend_cost_indices(robot, robot_arm)
    wrist = _wrist_limit_table(robot, robot_arm, names)
    end_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    branch = _branch_joint_indices(robot, robot_arm)
    seed = np.asarray(seed_av, dtype=np.float64)
    branch_range = math.radians(branch_range_deg)
    for x_margin in x_margins:
        window = [spik.restrict_base_x_range_to_human_standing(
            limits[0], standing[0], x_margin), limits[1], limits[2]]
        candidates = []
        for turn_deg in turn_degs:
            for palm_index, av, base in _batch_ik(
                    robot, robot_arm, hand, local_targets(turn_deg), seed_av,
                    local_collision_joints, collision_pairs, window,
                    attempts_per_pose, free_turn=free_turn,
                    lowered_arms=local_arms):
                if np.any(np.abs(av[branch] - seed[branch]) > branch_range):
                    continue
                # 指先の上下は台車の yaw に依らないので、解いた座標系のまま。
                _place(robot, av, base)
                finger = spik._correct_grasp_frame(
                    np.asarray(end_coords.worldrot()), robot_arm)[:, 0]
                xy = to_world.transform_point([base[0], base[1], 0.0])[:2]
                base = (float(xy[0]), float(xy[1]), base[2] + human_yaw)
                cost = (spik._joint_bend_cost_from_vector(
                            av, bend_cost_indices)
                        + placement_cost(base, standing, facing)
                        + GOAL_LEG_OFFSET_WEIGHT * abs(av[ankle] + av[knee])
                        + GOAL_CROUCH_WEIGHT * (abs(av[ankle])
                                                + abs(av[knee]))
                        + GOAL_WRIST_WEIGHT * wrist_limit_penalty(av, wrist)
                        + GOAL_ROBOT_FINGER_DOWN_WEIGHT
                        * (1.0 + float(finger[2]))
                        + GOAL_WRIST_BEND_WEIGHT
                        * sum(abs(av[index]) for index, _, _ in wrist))
                candidates.append((cost, av, base, turn_deg, palm_index))
        if candidates:
            candidates.sort(key=lambda c: (c[4], c[0]))
            return candidates, x_margin
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
              checker, gaze_forward, free_turn=False):
    """押し込み姿勢から移動先まで、台車 (``base_path``: ``BasePath``)・
    人の腕 (``lowering``:
    ``HumanArm.lowering`` の肩から腕を下ろす動き)・握りの向き
    (``start_turn`` -> ``goal_turn`` [度]、近い回り方。``free_turn`` なら
    拘束せず IK に任せる) を補間しながら、
    ロボットの腕・脚・首を直前の解から解き進めた waypoint を返す。視線は
    掌から、横並びで前を見る点 ``gaze_forward`` へ移していく。
    解けない waypoint は、直前の解に移動先の姿勢 ``goal_av`` への補間を
    1 区間分足した初期値でも試す。人の腕が届かない・関節角が飛ぶ・干渉
    する waypoint があれば、その手前までで止める。

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
    # 差し出さない腕 (肩ピッチ・肘) は台車の箱をよけるために姿勢の候補へ
    # 差し替えるので、関節角の飛びの判定から外す。
    other = 'l' if robot_arm == 'r' else 'r'
    other_names = ['{}_shoulder_p_joint'.format(other),
                   '{}_elbow_joint'.format(other)]
    other_arm = [names.index(name) for name in other_names]
    jump_check = np.array([name not in head and name not in other_names
                           for name in names])
    prev = start_av
    waypoints = []
    reason = None
    for k in range(1, n + 1):
        s = k / float(n)
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
        gaze_point = palm_pos + (gaze_forward - palm_pos) * s
        # 脚は直前の waypoint から移動先の脚へ、残りの区間で均等に近づけた
        # 角度に固定して解く。解けなければ脚も IK で動かす (以後はそこから
        # 移動先へ近づける)。
        leg_angles = prev[legs] + (goal_av[legs] - prev[legs]) / (n - k + 1)
        # 初期値は、直前の解を移動先の姿勢へ 1 区間分寄せたものを先に試す
        # (直前の解からだと手首などが直前の曲げに留まり、移動先へ向かわない)。
        for seed, free_legs in ((prev + step_av, False), (prev, False),
                                (prev, True)):
            seed = seed.copy()
            seed[legs] = leg_angles
            _place(robot, seed, base)
            if _press_ik(robot, robot_arm, target,
                         None if PATH_INTERPOLATE_NECK else gaze_point,
                         free_legs=free_legs,
                         free_turn=free_turn and not PATH_INTERPOLATE_TURN):
                break
        else:
            reason = '{}/{} で腕が掌に届かない'.format(k, n)
            break
        if PATH_INTERPOLATE_NECK:
            neck_av = robot.angle_vector().copy()
            neck_av[neck] = start_av[neck] + (goal_av[neck] - start_av[neck]) * s
            robot.angle_vector(neck_av)
        # 上体を傾けると差し出さない手が台車の前の箱に入るので、入らない
        # 姿勢に差し替える (押し込みの計画と同じ候補。直前の姿勢より曲げの
        # 小さい候補には戻さない)。
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
        problem = checker.check(robot, human_arm.skeleton(arm))
        if problem is not None:
            reason = '{}/{} で{}'.format(k, n, problem)
            break
        if free_turn:
            turn_deg = grip_turn_deg(
                robot, robot_arm, palm_from_frame(palm_pos, palm_rot))
        waypoints.append(dict(
            base_position=[float(base[0]), float(base[1]), 0.0],
            base_yaw=float(base[2]),
            joint_angle_vector=[float(v) for v in av],
            turn_deg=float(turn_deg), human_arm=arm))
        prev = av
    return waypoints, len(waypoints) / float(n), reason


# 横並びで前を見るときの、人の立ち位置から人の正面方向への距離 [m] と、
# 骨格から目の高さが求まらないときの高さ [m]。
FORWARD_GAZE_DISTANCE = 3.0
FORWARD_GAZE_DEFAULT_HEIGHT = 1.4


def forward_gaze_point(joint_positions):
    """横並びで前を見る点: 人の立ち位置から人の正面方向へ
    ``FORWARD_GAZE_DISTANCE`` 先の、人の目 (鼻) の高さの点。"""
    standing = spik.human_standing_xy(joint_positions)
    facing = spik.human_facing_direction(joint_positions)
    nose = joint_positions.get('Nose')
    height = (float(nose[2]) if nose is not None
              else FORWARD_GAZE_DEFAULT_HEIGHT)
    xy = standing + facing * FORWARD_GAZE_DISTANCE
    return np.array([xy[0], xy[1], height])


def plan_transition(*args, **kwargs):
    """``_plan_transition`` を、乱数 (バッチ IK の初期値) を押し込み姿勢から
    決めたシードで解く。結果が呼び出し前の乱数の状態 (ウォームアップや
    前の人の計算) に依らないようにし、呼び出し元の乱数の状態は戻す。"""
    post = args[3] if len(args) > 3 else kwargs['post']
    state = np.random.get_state()
    np.random.seed(zlib.crc32(np.round(np.asarray(
        post['joint_angle_vector'], dtype=np.float64), 6).tobytes()))
    try:
        return _plan_transition(*args, **kwargs)
    finally:
        np.random.set_state(state)


def _plan_transition(robot, robot_arm, hand, post, turn_deg,
                     translated_joints, collision_joints, verification_pairs,
                     collision_pairs, base_limits,
                     attempts_per_pose=GOAL_ATTEMPTS_PER_POSE,
                     use_facing_variants=True):
    """押し込み姿勢 ``post`` (``solve_post_process`` の結果 dict、向き
    ``turn_deg``) から、つないだ手を人の体の横へ下ろしながら、ロボットも
    人と横並びになる位置・姿勢へ移る waypoint を計画する.

    ``translated_joints`` は立ち位置・向き・人の腕の判定用、
    ``collision_joints`` は干渉判定用 (体幹をずらした) の骨格 (どちらも
    IK と同じ座標系)。``base_limits`` は台車の可動域全体。

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
                  placement_before=None, placement_after=None)
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

    palm = human_palm_from_press(post, turn_deg, robot_arm)
    human_arm = HumanArm(translated_joints, hand, palm)
    result['press_arm'] = human_arm.pose(*palm_frame(palm))
    gaze_forward = forward_gaze_point(translated_joints)
    standing = spik.human_standing_xy(translated_joints)
    checker = TransitionChecker(robot_arm, hand, verification_pairs)
    reasons = []
    best = [None]
    free_turn = GOAL_FREE_TURN
    # 握りの向きを自由にするなら、目標の向きは 0 度の 1 組だけでよい
    # (法線まわりは拘束しないので、どの角度から解いても同じ)。
    turn_groups = ((0.0,),) if free_turn else GOAL_TURN_DEG_GROUPS

    def try_lowerings(lowerings):
        """腕の下ろし方 ``lowerings`` (3 つ) を移動先にして経路を作る。
        横並びまで (100%) 進めたら True。"""
        goal_palms = [lowering['palm'] for lowering in lowerings]
        lowered_arms = [(human_arm.shoulder,
                         human_arm.lowering_pose(lowering, 1.0)[0])
                        for lowering in lowerings]
        for turn_degs in turn_groups:
            candidates, x_margin = solve_goal_candidates(
                robot, robot_arm, hand, goal_palms, start_av,
                translated_joints, collision_joints, collision_pairs,
                base_limits, turn_degs=turn_degs,
                attempts_per_pose=attempts_per_pose, free_turn=free_turn,
                lowered_arms=lowered_arms)
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
                lowering = lowerings[palm_index]
                goal_palm = lowering['palm']
                goal_skeleton = human_arm.skeleton(human_arm.pose(
                    *goal_palm,
                    elbow=human_arm.lowering_pose(lowering, 1.0)[0]))
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
                # 腕と同時の視線 IK は押し込み時の首から大きく回せず解けない
                # ことがあるので、首だけで前へ向け直す (経路の首はここへ補間
                # する、PATH_INTERPOLATE_NECK)。
                _aim_head(robot, gaze_forward)
                if free_turn:
                    goal_turn = grip_turn_deg(robot, robot_arm,
                                              palm_from_frame(*goal_palm))
                problem = checker.check(robot, goal_skeleton)
                if problem is not None:
                    reasons.append('移動先で{}'.format(problem.split(' ')[0]))
                    continue
                tried[palm_index] = tried.get(palm_index, 0) + 1
                waypoints, fraction, why = plan_path(
                    robot, robot_arm, hand, turn_deg, goal_turn, human_arm,
                    lowering, start_av, base_path,
                    robot.angle_vector().copy(), checker,
                    gaze_forward, free_turn=free_turn)
                if why is not None:
                    reasons.append('経路の{}'.format(why))
                if best[0] is None or fraction > best[0][1]:
                    best[0] = (waypoints, fraction)
                    result.update(arm_deg=lowering['arm_deg'],
                                  abduction_deg=lowering['abduction_deg'],
                                  palm_facing=lowering['palm_facing'],
                                  x_margin=x_margin)
                if fraction >= 1.0:
                    return True
            # 次の組 (握りの向き ±90 度) は、この組で移動先の IK が 1 つも
            # 解けなかったときだけ試す (経路が途中で止まっても、別の向きで
            # 進めたことは合成データでは無く、1 組あたり窓を広げて 1〜3 秒
            # かかる)。
            return False
        return False

    def facing_variants(arm_degs, abduction_deg):
        """組 ``arm_degs`` の掌の向きごとの下ろし方 (3 つとも同じ向き) の
        ``[(向きの順位, 下ろし方)]``。向きは ``LOWERED_FACING_ORDER`` の順で、
        使えない向き・前腕のひねりが ``LOWERED_TWIST_MAX_DEG`` を超える向きは
        除く。``use_facing_variants`` が False なら前腕のひねりが一番少ない
        向き (上腕の角度ごとに違ってよい) の 1 つだけ。"""
        if not use_facing_variants:
            return [(0, [human_arm.lowering(deg, abduction_deg)
                         for deg in arm_degs])]
        variants = []
        for rank, facing in enumerate(LOWERED_FACING_ORDER):
            lowerings = [human_arm.lowering(deg, abduction_deg, facing)
                         for deg in arm_degs]
            if any(lw is None for lw in lowerings):
                continue
            twist = max(abs(math.degrees(lw['twist'])) for lw in lowerings)
            if twist > LOWERED_TWIST_MAX_DEG:
                continue
            variants.append((rank, lowerings))
        return variants

    # 人が腕を低く下ろせる組から試し、横並びまで進めた組で止める。同じ上腕の
    # 角度の組の中では、脇の開きより掌の向きを優先する (掌の向きを
    # LOWERED_FACING_ORDER の順に、それぞれ脇の開きの狭い順に試す)。
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
        ordered.sort(key=lambda item: item[0])
        if any(try_lowerings(lowerings) for _, lowerings in ordered):
            break
    best = best[0]
    if best is None or best[1] < MIN_FRACTION:
        result.update(reason=', '.join(sorted(set(reasons))) or None,
                      compute_time=time.time() - t0)
        return result
    waypoints, fraction = best
    last = waypoints[-1]
    arms = [result['press_arm']] + [wp['human_arm'] for wp in waypoints]
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
        human_max=dict(
            reach=max(a['reach'] for a in arms),
            elbow_flex_deg=max(a['elbow_flex_deg'] for a in arms),
            wrist_bend_deg=max(a['wrist_bend_deg'] for a in arms),
            twist_change_deg=max(abs(a['twist_change_deg']) for a in arms)),
        compute_time=time.time() - t0)
    return result


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
                     HUMAN_TWIST_CHANGE_MAX_DEG)))


_PALM_FACING_NAMES = dict(inward='人側', forward='前向き', backward='後ろ向き',
                          natural='ひねらない向き')


def lowered_arm_text(transition):
    """移動先の人の腕 (上腕の真下からの角度と掌の向き) の説明。"""
    if 'arm_deg' not in transition:
        # 肩から下ろす形にする前の結果 (前腕の真下からの角度)
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
            '(移動先の {:.0f}%、窓 ±{} m、{:.2f} 秒)。人の腕の最大: {}'.format(
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
                human_arm_text(dict(transition['human_max'],
                                    twist_change_deg=transition['human_max']
                                    ['twist_change_deg']))))
    if transition['reason']:
        text += ' / 途中で止めた: {}'.format(transition['reason'])
    return text
