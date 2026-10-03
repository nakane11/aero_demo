#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""掌を押し込んだ後、つないだ手を人の体の横へ下ろしながら、ロボットも人と
横並びの位置・姿勢へ移る区間 (横並び移動) を計画する。

押し込み (``solve_palm_ik`` の後処理姿勢) までは従来どおりで、ロボットは
膝を曲げて上体を前に出し (身を乗り出し)、人の手の位置に合わせて立つ。
押し込んだ後は、

- つないだ手 (人の掌) を、人の腕を体の横へ下ろした位置
  (``lowered_palm``: 肩から真下より少し外へ開き、肘を少し曲げた位置。
  掌は後ろ向き、指先は下) まで動かす。
- 同時に、台車を人と前後ずれ 0・同じ向きの位置へ動かし、上体を起こす
  (移動先は台車も動かすバッチ IK で、前後ずれ・向きのずれ・上体の前後
  ずれの小さい解を選ぶ、``solve_goal_candidates``)。腕は押し込み時と
  同じ解の枝 (腕を前に出した形) に限る (``GOAL_BRANCH_JOINT_RANGE_DEG``)。
- 移動先では握りの向き (人とロボットの指先のなす角) を変えてよく
  (``GOAL_TURN_DEG_GROUPS``)、しゃがまない (脚を伸ばした) 解を優先する
  (``GOAL_CROUCH_WEIGHT``)。視線は掌から前 (``forward_gaze_point``) へ移す。
- 途中の waypoint は、掌の位置・向き・握りの向き・台車を補間し、腕・脚・首を
  解き直す。

人の腕は肩を動かさず、手首・肘を 2 リンクの IK で求める (``human_arm``)。
各 waypoint で、人の腕が届くか (``reach``)・肘の曲げ・手首の曲げ・前腕の
ひねりの変化を記録し、腕が届かなくなる手前で止める。ビューアはこれを使って
人の腕をロボットの手に追従させる。

干渉の検証は人体との距離 (6 cm) と自己干渉。人の差し出した腕は waypoint
ごとに動かした骨格で見る。ロボットの差し出す腕の肘から先と、人の差し出した
手・前腕は手を合わせているので距離を見ない。
"""

import math
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import normalize_mask
from skrobot.coordinates.math import rpy_matrix

import solve_palm_ik as spik


# 下ろした手の位置: 上腕は真下からこの角度だけ外へ開いて下ろし、前腕は
# 真下から前へ LOWERED_FOREARM_DEGS の各角度だけ上げる (肘を曲げて手の
# 高さを変える)。移動先のバッチ IK はこの 3 つを目標にし、しゃがまずに
# 済む高さを選ぶ (GOAL_CROUCH_WEIGHT)。指先は前腕の向き、掌は前腕を内へ
# 約 90 度ひねった向き (腕を下げると後ろ向き、前へ上げると下向き)。
LOWERED_ARM_ABDUCTION_DEG = 15.0
LOWERED_FOREARM_DEGS = (20.0, 50.0, 80.0)

# 移動先の台車の前後位置を、人の立ち位置からどれだけずらしてよいか [m]。
# 狭い方から試し、解けなければ広げる。
DEFAULT_GOAL_X_MARGINS = (0.05, 0.15, 0.3, 0.5)

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

# 移動先の候補 (コストの安い順) のうち、押し込み IK・検証・経路まで試す数。
MAX_GOAL_CANDIDATES = 5

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

    def pose(self, palm_pos, palm_rot):
        """掌が ``palm_pos``/``palm_rot`` のときの腕。``elbow``/``wrist``
        と目安 (``reach``: 肩-手首 / 腕の長さ、``elbow_flex_deg``、
        ``wrist_bend_deg``: 前腕と手のなす角、``twist_change_deg``: 押し
        込み時からの前腕まわりのひねりの変化) の dict。"""
        rot, trans = self._transform(palm_pos, palm_rot)
        wrist = rot @ self.wrist0 + trans
        elbow = self.elbow_for(wrist)
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

    def lowered_palm(self, forearm_deg):
        """上腕を体の横へ下ろし、前腕を真下から前へ ``forearm_deg`` 度
        上げたときの掌の位置と回転行列。"""
        down = np.array([0.0, 0.0, -1.0])
        forward = self.forward
        ab = math.radians(LOWERED_ARM_ABDUCTION_DEG)
        upper_dir = _unit(math.cos(ab) * down + math.sin(ab) * self.outward)
        elbow = self.shoulder + upper_dir * self.upper
        fa = math.radians(forearm_deg)
        fingers = _unit(math.cos(fa) * down + math.sin(fa) * forward)
        wrist = elbow + fingers * self.fore
        back_down = -forward + down
        normal = _unit(back_down - np.dot(back_down, fingers) * fingers)
        palm_rot = np.column_stack(
            [fingers, normal, np.cross(fingers, normal)])
        # 手首から掌の中心までは押し込み時と同じ (手は剛体)。
        offset = self.palm_rot0.T @ (self.palm_pos0 - self.wrist0)
        return wrist + palm_rot @ offset, palm_rot


def two_link_elbow(shoulder, wrist, upper, fore, outward):
    """肩 ``shoulder`` から手首 ``wrist`` へ、上腕 ``upper``・前腕 ``fore``
    の長さの腕を伸ばしたときの肘の位置。肘が取りうる円のうち、下・少し外
    (``outward``) を向く側を選ぶ。届かないときは手首の方向へ伸ばしきる。"""
    shoulder = np.asarray(shoulder, dtype=np.float64)
    d = np.asarray(wrist, dtype=np.float64) - shoulder
    dist = float(np.linalg.norm(d))
    e1 = d / max(dist, 1e-9)
    reach = min(dist, upper + fore - 1e-6)
    a = (upper ** 2 - fore ** 2 + reach ** 2) / (2.0 * reach)
    r = math.sqrt(max(upper ** 2 - a ** 2, 0.0))
    pref = np.array([0.0, 0.0, -1.0]) + 0.3 * np.asarray(outward)
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
    そのままで腕だけを解く。腕が解けなければ ``False`` (ロボットは呼び
    出し前の姿勢に戻る)。"""
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    link_list = [link for link in whole_body.link_list
                 if free_legs or link.joint.name not in LEG_JOINT_NAMES]
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    gaze_coords = spik.camera_optical_coords(robot)
    stop = spik.DEFAULT_POST_PROCESS_IK_STOP
    thre = spik.DEFAULT_POST_PROCESS_IK_THRE
    rthre = spik.DEFAULT_POST_PROCESS_IK_RTHRE
    gaze_rthre = spik.DEFAULT_POST_PROCESS_GAZE_IK_RTHRE
    hand_mask = _rotation_mask(free_turn)
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


def _batch_ik(robot, robot_arm, hand, targets, seed_av, collision_joints,
              collision_pairs, base_limits, attempts_per_pose,
              free_turn=False):
    """押し込み目標 ``targets`` (3 つ) に対して台車も動かす干渉回避付き
    バッチ IK (``solve_person_ik`` と同じ形: 目標 3 つ・初期値・干渉ペア)。
    人の差し出した手・前腕・上腕は手を合わせているので障害物から外す
    (遠くのダミーにする、個数は変えない)。差し出さない腕・首の初期値は
    ``seed_av``。収束した解 ``[(目標の添字, angle_vector, (x, y, yaw)),
    ...]`` を返す。"""
    obstacles = spik.human_body_obstacles(collision_joints)
    removed = spik.offered_hand_obstacle_indices(hand) | {
        spik.human_obstacle_names().index('{0}Shoulder-{0}Elbow'.format(hand))}
    obstacles = [spik._dummy_cylinder(o.radius) if i in removed else o
                 for i, o in enumerate(obstacles)]
    pairs = None
    if collision_pairs:
        links_by_name = {link.name: link for link in
                         list(robot.link_list)
                         + list(getattr(robot, 'extra_collision_links', []))}
        offered = sorted(spik.offered_hand_obstacle_indices(hand))
        pairs = list(collision_pairs) + [
            (links_by_name[name.format(robot_arm)], i)
            for name in spik.OFFERED_HAND_PENALTY_LINKS
            if name.format(robot_arm) in links_by_name for i in offered]
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
                    attempts_per_pose=spik.DEFAULT_ATTEMPTS_PER_POSE):
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
    _batch_ik(robot, robot_arm, hand, [target] * len(LOWERED_FOREARM_DEGS),
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


def solve_goal_candidates(robot, robot_arm, hand, goal_palms, seed_av,
                          translated_joints, collision_joints,
                          collision_pairs, base_limits,
                          x_margins=DEFAULT_GOAL_X_MARGINS,
                          turn_degs=GOAL_TURN_DEGS,
                          attempts_per_pose=spik.DEFAULT_ATTEMPTS_PER_POSE,
                          branch_range_deg=GOAL_BRANCH_JOINT_RANGE_DEG,
                          free_turn=False):
    """下ろした手の候補 ``goal_palms`` (3 つの ``(位置, 回転行列)``) への
    押し込み目標を、握りの向き ``turn_degs`` ごとに 1 回のバッチ IK で
    解き、移動先の台車・姿勢の候補を前後の窓の狭い方から (候補が出るまで)
    集めて、コスト (曲げ量 + 前方ずれ + 向きのずれ + 上体の前後ずれ +
    しゃがみ量) の安い順に返す。

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
    standing = spik.human_standing_xy(translated_joints)
    facing = spik.human_facing_direction(translated_joints)
    human_yaw = math.atan2(facing[1], facing[0])
    to_local = _rotation_about(-human_yaw, standing)
    to_world = _rotation_about(human_yaw, standing)
    local_joints = {name: to_local.transform_point(p)
                    for name, p in translated_joints.items()}
    local_collision_joints = {name: to_local.transform_point(p)
                              for name, p in collision_joints.items()}
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
    side_sign = spik.offered_hand_side_sign(hand, local_joints, None)
    if side_sign is not None:
        limits[1] = spik.restrict_base_y_range_to_hand_side(
            limits[1], side_sign)
    limits[2] = spik.restrict_base_yaw_range_to_human_facing(
        limits[2], 0.0,
        margin=math.radians(spik.DEFAULT_BASE_YAW_FACING_MARGIN_DEG))
    names = [joint.name for joint in robot.joint_list]
    ankle, knee = names.index('ankle_joint'), names.index('knee_joint')
    bend_cost_indices = spik._joint_bend_cost_indices(robot, robot_arm)
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
                    attempts_per_pose, free_turn=free_turn):
                if np.any(np.abs(av[branch] - seed[branch]) > branch_range):
                    continue
                xy = to_world.transform_point([base[0], base[1], 0.0])[:2]
                base = (float(xy[0]), float(xy[1]), base[2] + human_yaw)
                cost = (spik._joint_bend_cost_from_vector(
                            av, bend_cost_indices)
                        + placement_cost(base, standing, facing)
                        + GOAL_LEG_OFFSET_WEIGHT * abs(av[ankle] + av[knee])
                        + GOAL_CROUCH_WEIGHT * (abs(av[ankle])
                                                + abs(av[knee])))
                candidates.append((cost, av, base, turn_deg, palm_index))
        if candidates:
            candidates.sort(key=lambda c: c[0])
            return candidates, x_margin
    return [], None


def plan_path(robot, robot_arm, hand, start_turn, goal_turn, human_arm,
              start_av, start_base, start_palm, goal_av, goal_base, goal_palm,
              checker, gaze_forward, free_turn=False):
    """押し込み姿勢から移動先まで、台車・人の掌 (位置・向き)・握りの向き
    (``start_turn`` -> ``goal_turn`` [度]、近い回り方。``free_turn`` なら
    拘束せず IK に任せる) を補間しながら、
    ロボットの腕・脚・首を直前の解から解き進めた waypoint を返す。視線は
    掌から、横並びで前を見る点 ``gaze_forward`` へ移していく。
    解けない waypoint は、直前の解に移動先の姿勢 ``goal_av`` への補間を
    1 区間分足した初期値でも試す。人の腕が届かない・関節角が飛ぶ・干渉
    する waypoint があれば、その手前までで止める。

    ``start_palm``/``goal_palm`` は ``(位置, 回転行列)``。

    Returns
    -------
    (waypoints, fraction, reason)
        ``fraction`` は移動先までのうち進めた割合、``reason`` は止めた
        理由 (最後まで進めたら ``None``)。
    """
    start_base = np.asarray(start_base, dtype=np.float64)
    goal_base = np.asarray(goal_base, dtype=np.float64)
    delta = goal_base - start_base
    palm_delta = goal_palm[0] - start_palm[0]
    palm_angle = Rotation.from_matrix(
        goal_palm[1] @ start_palm[1].T).magnitude()
    turn_delta = math.degrees(_wrap(math.radians(goal_turn - start_turn)))
    start_av = np.asarray(start_av, dtype=np.float64)
    goal_av = np.asarray(goal_av, dtype=np.float64)
    n = max(2, int(math.ceil(max(
        np.linalg.norm(delta[:2]) / MAX_STEP,
        abs(delta[2]) / MAX_ANGLE_STEP,
        np.linalg.norm(palm_delta) / MAX_STEP,
        palm_angle / MAX_ANGLE_STEP,
        math.radians(abs(turn_delta)) / MAX_ANGLE_STEP,
        float(np.max(np.abs(goal_av - start_av))) / PLANNED_JOINT_STEP))))
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(
        [start_palm[1], goal_palm[1]]))
    step_av = (goal_av - start_av) / n
    # 脚は IK で動かさず (_press_ik)、押し込み時から移動先へ補間する。
    names = [joint.name for joint in robot.joint_list]
    legs = [names.index(name) for name in ('ankle_joint', 'knee_joint')]
    head = {link.joint.name for link in robot.head.link_list}
    jump_check = np.array([name not in head for name in names])
    prev = start_av
    waypoints = []
    reason = None
    for k in range(1, n + 1):
        s = k / float(n)
        base = start_base + delta * s
        palm_pos = start_palm[0] + palm_delta * s
        palm_rot = slerp([s]).as_matrix()[0]
        arm = human_arm.pose(palm_pos, palm_rot)
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
        for seed, free_legs in ((prev, False), (prev + step_av, False),
                                (prev, True)):
            seed = seed.copy()
            seed[legs] = leg_angles
            _place(robot, seed, base)
            if _press_ik(robot, robot_arm, target, gaze_point,
                         free_legs=free_legs, free_turn=free_turn):
                break
        else:
            reason = '{}/{} で腕が掌に届かない'.format(k, n)
            break
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


def plan_transition(robot, robot_arm, hand, post, turn_deg,
                    translated_joints, collision_joints, verification_pairs,
                    collision_pairs, base_limits,
                    attempts_per_pose=spik.DEFAULT_ATTEMPTS_PER_POSE):
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
    start_palm = palm_frame(palm)
    result['press_arm'] = human_arm.pose(*start_palm)
    goal_palms = [human_arm.lowered_palm(deg)
                  for deg in LOWERED_FOREARM_DEGS]
    gaze_forward = forward_gaze_point(translated_joints)
    checker = TransitionChecker(robot_arm, hand, verification_pairs)
    reasons = []
    best = None
    free_turn = GOAL_FREE_TURN
    # 握りの向きを自由にするなら、目標の向きは 0 度の 1 組だけでよい
    # (法線まわりは拘束しないので、どの角度から解いても同じ)。
    groups = ((0.0,),) if free_turn else GOAL_TURN_DEG_GROUPS
    for turn_degs in groups:
        candidates, x_margin = solve_goal_candidates(
            robot, robot_arm, hand, goal_palms, start_av, translated_joints,
            collision_joints, collision_pairs, base_limits,
            turn_degs=turn_degs, attempts_per_pose=attempts_per_pose,
            free_turn=free_turn)
        if not candidates:
            reasons.append('移動先の IK が解けない (握りの向き {})'.format(
                '/'.join('{:.0f}'.format(t) for t in turn_degs)))
            continue
        tried = 0
        for _, goal_av, goal_base, goal_turn, palm_index in candidates:
            if tried >= MAX_GOAL_CANDIDATES:
                break
            goal_palm = goal_palms[palm_index]
            goal_skeleton = human_arm.skeleton(human_arm.pose(*goal_palm))
            # yaw を押し込み時から近い回り方の値にする。
            goal_base = np.array([goal_base[0], goal_base[1], start_base[2]
                                  + _wrap(goal_base[2] - start_base[2])])
            # 移動先の姿勢そのものを、首も含めて押し込み姿勢に解き直して
            # 検証してから経路を解く。
            goal_target = press_target(palm_from_frame(*goal_palm),
                                       goal_turn, robot_arm)
            _place(robot, goal_av, goal_base)
            if not _press_ik(robot, robot_arm, goal_target, gaze_forward,
                             free_turn=free_turn):
                reasons.append('移動先で押し込み IK が解けない')
                continue
            if free_turn:
                goal_turn = grip_turn_deg(robot, robot_arm,
                                          palm_from_frame(*goal_palm))
            problem = checker.check(robot, goal_skeleton)
            if problem is not None:
                reasons.append('移動先で{}'.format(problem.split(' ')[0]))
                continue
            tried += 1
            waypoints, fraction, why = plan_path(
                robot, robot_arm, hand, turn_deg, goal_turn, human_arm,
                start_av, start_base, start_palm, robot.angle_vector().copy(),
                goal_base, goal_palm, checker, gaze_forward,
                free_turn=free_turn)
            if why is not None:
                reasons.append('経路の{}'.format(why))
            if best is None or fraction > best[1]:
                best = (waypoints, fraction)
                result['forearm_deg'] = LOWERED_FOREARM_DEGS[palm_index]
                result['x_margin'] = x_margin
            if fraction >= 1.0:
                break
        # 次の組 (握りの向き ±90 度) は、この組で移動先の IK が 1 つも
        # 解けなかったときだけ試す (経路が途中で止まっても、別の向きで
        # 進めたことは合成データでは無く、1 組あたり窓を広げて 1〜3 秒
        # かかる)。
        break
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
    return ('到達率 {:.2f}{}、肘の曲げ {} 度、手首の曲げ {} 度、前腕のひねり '
            '{} 度'.format(
                arm['reach'], ' (!)' if arm['reach'] > 1.0 else '',
                mark(arm['elbow_flex_deg'], HUMAN_ELBOW_FLEX_MAX_DEG),
                mark(arm['wrist_bend_deg'], HUMAN_WRIST_BEND_MAX_DEG),
                mark(abs(arm['twist_change_deg']),
                     HUMAN_TWIST_CHANGE_MAX_DEG)))


def transition_summary(transition):
    """``plan_transition`` の結果の 1 行の説明。"""
    if not transition['verified']:
        return 'しない ({})'.format(transition['reason'])
    before = transition['placement_before'] or {}
    after = transition['placement_after'] or {}
    text = ('前方ずれ {:+.2f} -> {:+.2f} m、向きのずれ {:+.0f} -> {:+.0f} 度、'
            '方位 {:+.0f} -> {:+.0f} 度、脚 (ankle/knee) {:.0f}/{:.0f} -> '
            '{:.0f}/{:.0f} 度、握りの向き {:.0f} -> {:.0f} 度、人の前腕 {:.0f} 度 '
            '(移動先の {:.0f}%、窓 ±{} m、{:.2f} 秒)。人の腕の最大: {}'.format(
                before.get('front_offset', float('nan')),
                after.get('front_offset', float('nan')),
                before.get('yaw_offset_deg', float('nan')),
                after.get('yaw_offset_deg', float('nan')),
                before.get('bearing_deg', float('nan')),
                after.get('bearing_deg', float('nan')),
                *(transition['legs_before'] + transition['legs_after']),
                transition['turn_before'], transition['turn_after'],
                transition['forearm_deg'], transition['fraction'] * 100.0, transition['x_margin'],
                transition['compute_time'],
                human_arm_text(dict(transition['human_max'],
                                    twist_change_deg=transition['human_max']
                                    ['twist_change_deg']))))
    if transition['reason']:
        text += ' / 途中で止めた: {}'.format(transition['reason'])
    return text
