#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人が手をどこに・どの向きで差し出せば、差し出しと判定され IK が解けるかを
格子状に調べる (reachability map)。

体格を固定した骨格で掌の位置 (前方・外側・高さ) と向きを振り、格子点ごとに
差し出し判定 (``OfferedHandSelector``、人はロボット正面 ``--person-distances``
[m]) と IK (事後検証・後処理まで、軌道計画なし) を評価する。人の腕が届かない
点は IK を解かない。結果は JSON Lines で ``--output`` に追記し、再実行すると
続きから解く。

Usage
-----
    # 位置の格子 (既定の向き = 握手の向き: 指先は前、親指が上)
    python3 tools/handshake_reachability_map.py --mode position \\
        --hands R L --output /tmp/reach_position.jsonl
    # 向きの格子 (位置は --orientation-position に固定)
    python3 tools/handshake_reachability_map.py --mode orientation \\
        --hands R L --output /tmp/reach_orientation.jsonl
"""

import argparse
import contextlib
import io
import itertools
import json
import math
import os
import re
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
for _path in (_SCRIPTS_DIR, _PKG_SRC_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aero_demo.people_pose_types import HAND_LOCAL_LANDMARKS  # noqa: E402
import estimate_palm_poses as epp  # noqa: E402
import solve_palm_ik as spik  # noqa: E402
from skrobot.models import Aero  # noqa: E402

# 人の体格 (身長約 1.70 m、人の座標系: x=前, y=左, z=上, 足元が原点)
NECK = (0.0, 0.0, 1.40)
SHOULDER_HALF_WIDTH = 0.18
SHOULDER_Z = 1.36
# MediaPipe の腰のランドマークは SMPL の股関節より外にあるので広めに取る。
HIP_HALF_WIDTH = 0.09
HIP_Z = 0.88
KNEE_Z = 0.48
ANKLE_Z = 0.08
UPPER_ARM_LENGTH = 0.29
FOREARM_LENGTH = 0.26
# generate_random_human_poses._HAND_LENGTH_HEIGHT_RATIO * 身長。
HAND_LENGTH = 0.108 * 1.70
# 肩-手首の距離が腕の長さのこの割合を超えたら人が届かない扱い。
MAX_REACH_RATIO = 0.98

# 失敗理由の分類に使う solve_palm_ik の標準出力のパターン。
_HOVER_CLEARANCE_RE = re.compile(r'\[hover-clearance\] .*人体 \((\S+)\)')
_SELF_COLLISION_RE = re.compile(
    r'\[collision-verify\] .*?(貫通|自己干渉) \((\S+) x (\S+)\)')
_POST_PROCESS_RE = re.compile(r'\[post-process\] .*後処理判定 .*失敗')


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _side_sign(hand):
    """人の座標系で ``hand`` 側の y の符号 (右手は -y)。"""
    return -1.0 if hand == 'R' else 1.0


def hand_frame(hand, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """差し出す手の局所座標系 (u=指先, v=親指側, n=掌の向き) を返す.

    全て 0 で握手の向き (指先が前、親指が上)。yaw は外側、pitch は上が正。
    roll は回内で 90 で掌が真下、-90 で真上。
    """
    sign = _side_sign(hand)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    u = np.array([math.cos(pitch) * math.cos(yaw),
                  sign * math.cos(pitch) * math.sin(yaw),
                  math.sin(pitch)])
    up = np.array([0.0, 0.0, 1.0])
    up_perp = _unit(up - np.dot(up, u) * u)
    medial = np.cross(up_perp, u) if hand == 'R' else np.cross(u, up_perp)
    v = math.cos(roll) * up_perp + math.sin(roll) * medial
    # RandomSkeletonGenerator._hand_frame と同じ約束。
    n = np.cross(v, u) if hand == 'R' else np.cross(u, v)
    return u, v, n


def hand_landmarks(hand, wrist, u, v, n):
    basis = np.vstack([u, v, n])
    pts = np.asarray(wrist) + HAND_LENGTH * HAND_LOCAL_LANDMARKS.dot(basis)
    return {'{}Hand{}'.format(hand, i): pts[i] for i in range(len(pts))}


def _elbow_position(shoulder, wrist, hand):
    """肩と手首から肘の位置を 2 リンク IK で決める (届かなければ ``None``)。"""
    d_vec = wrist - shoulder
    d = float(np.linalg.norm(d_vec))
    l1, l2 = UPPER_ARM_LENGTH, FOREARM_LENGTH
    if d > MAX_REACH_RATIO * (l1 + l2) or d < abs(l1 - l2) + 1e-3:
        return None
    axis = d_vec / d
    a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    r = math.sqrt(max(l1 * l1 - a * a, 0.0))
    # 肘はできるだけ真下に垂らす。
    prefer = np.array([-0.2, 0.0, -1.0])
    perp = prefer - np.dot(prefer, axis) * axis
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([0.0, _side_sign(hand), 0.0])
        perp = perp - np.dot(perp, axis) * axis
    return shoulder + a * axis + r * _unit(perp)


def build_skeleton(hand, palm_position, yaw_deg=0.0, pitch_deg=0.0,
                   roll_deg=0.0):
    """掌の中心が ``palm_position`` (人の座標系) に来る骨格を返す (届かなければ None)。"""
    joints = {}
    joints['Neck'] = np.array(NECK)
    for side in ('R', 'L'):
        s = _side_sign(side)
        joints['{}Shoulder'.format(side)] = np.array(
            [0.0, s * SHOULDER_HALF_WIDTH, SHOULDER_Z])
        joints['{}Hip'.format(side)] = np.array([0.0, s * HIP_HALF_WIDTH, HIP_Z])
        joints['{}Knee'.format(side)] = np.array([0.0, s * HIP_HALF_WIDTH, KNEE_Z])
        joints['{}Ankle'.format(side)] = np.array(
            [0.0, s * HIP_HALF_WIDTH, ANKLE_Z])
        joints['{}Eye'.format(side)] = joints['Neck'] + np.array(
            [0.05, s * 0.03, 0.16])
        joints['{}Ear'.format(side)] = joints['Neck'] + np.array(
            [0.0, s * 0.08, 0.15])
    joints['Nose'] = joints['Neck'] + np.array([0.06, 0.0, 0.14])

    # 差し出さない手は体の横に下ろす。
    other = 'L' if hand == 'R' else 'R'
    s = _side_sign(other)
    shoulder = joints['{}Shoulder'.format(other)]
    elbow = shoulder + np.array([0.0, s * 0.03, -UPPER_ARM_LENGTH])
    wrist = elbow + np.array([0.03, 0.0, -FOREARM_LENGTH])
    u = _unit(wrist - elbow)
    v = np.array([1.0, 0.0, 0.0])
    v = _unit(v - np.dot(v, u) * u)
    n = np.cross(v, u) if other == 'R' else np.cross(u, v)
    joints['{}Elbow'.format(other)] = elbow
    joints['{}Wrist'.format(other)] = wrist
    joints.update(hand_landmarks(other, wrist, u, v, n))

    # 掌の中心が palm_position に来るよう手首の位置を逆算する。
    u, v, n = hand_frame(hand, yaw_deg, pitch_deg, roll_deg)
    local = hand_landmarks(hand, np.zeros(3), u, v, n)
    center0 = np.asarray(epp.PalmPoseEstimator().estimate_palm(
        local, hand)['position'])
    wrist = np.asarray(palm_position, dtype=np.float64) - center0
    shoulder = joints['{}Shoulder'.format(hand)]
    elbow = _elbow_position(shoulder, wrist, hand)
    if elbow is None:
        return None
    joints['{}Elbow'.format(hand)] = elbow
    joints['{}Wrist'.format(hand)] = wrist
    joints.update(hand_landmarks(hand, wrist, u, v, n))
    return joints


def shoulder_angles(joints, hand):
    """差し出した上腕の肩の角度 [度]。

    abduction: 正面から見た真下からの角度 (外側が正)。flexion: 横から見た
    真下からの角度 (前が正)。elevation: 真下とのなす角。
    """
    upper = joints['{}Elbow'.format(hand)] - joints['{}Shoulder'.format(hand)]
    outward = _side_sign(hand) * upper[1]
    return dict(
        abduction=round(math.degrees(math.atan2(outward, -upper[2])), 1),
        flexion=round(math.degrees(math.atan2(upper[0], -upper[2])), 1),
        elevation=round(math.degrees(math.acos(
            -upper[2] / np.linalg.norm(upper))), 1))


def robot_hand_position_initial():
    """腕を下ろした初期姿勢での右手先の base_link 座標 (差し出し判定の基準)。"""
    robot = Aero(use_hand=False)
    robot.reset_pose()
    for side in ('r', 'l'):
        getattr(robot, '{}_elbow_joint'.format(side)).joint_angle(0.0)
    return np.asarray(robot.rarm_end_coords.worldpos(), dtype=np.float64)


def offer_scores(joints, robot_hand, distances, score_min):
    """人がロボット正面 ``distances`` [m] に向かい合って立つときの差し出し判定。

    base_link での手先 (hx, hy, hz) は人の座標系で (d - hx, -hy, hz)。
    """
    estimator = epp.PalmPoseEstimator(epp.OfferedHandSelector())
    palms = {side: estimator.estimate_palm(joints, side) for side in ('R', 'L')}
    out = {}
    for d in distances:
        selector = epp.OfferedHandSelector(
            robot_position=[d - robot_hand[0], -robot_hand[1], robot_hand[2]],
            score_min=score_min)
        sel = selector.select(joints, palms)
        out['{:.2f}'.format(d)] = dict(
            side=sel['side'],
            scores={k: (None if v is None else round(float(v), 4))
                    for k, v in sel['scores'].items()},
            features={k: (None if v is None else
                          {fk: round(float(fv), 4) for fk, fv in v.items()})
                      for k, v in sel['features'].items()})
    return palms, out


class IkEvaluator(object):
    """``run_camera_pipeline_test.py`` と同じ設定で 1 人分の IK を解く。"""

    def __init__(self, args):
        self.args = args
        robot = Aero(use_hand=False)
        spik.restrict_elbow_range(robot)
        spik.restrict_leg_range(robot)
        spik.restrict_waist_range(robot)
        spik.restrict_neck_range(robot)
        spik.lock_fixed_joints(robot)
        spik.apply_collision_model(robot)
        spik.apply_hand_box(robot)
        spik.other_hand_points('r')
        spik.attach_camera_optical_coords(robot)
        self.robot = robot
        self.verification_pairs = spik.build_verification_pairs_for_model(
            robot, spik.DEFAULT_COLLISION_VERIFY_MODEL)
        self.collision_pairs = spik.load_collision_pairs(
            args.collision_pairs, robot)
        self.base_limits = [tuple(spik.DEFAULT_BASE_X_RANGE),
                            tuple(spik.DEFAULT_BASE_Y_RANGE),
                            tuple(spik.DEFAULT_BASE_YAW_RANGE)]
        self._warmup()

    def _warmup(self):
        """全可動域のダミー目標で最初のバッチ IK を解いておく。

        最初の呼び出しを狭い可動域で行うと以後のバッチ IK が壊れるため必須。
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for robot_arm, hand in (('l', 'R'), ('r', 'L')):
                spik.solve_person_ik(
                    self.robot, spik._WARMUP_PALM, hand, robot_arm,
                    spik.human_body_obstacles({}),
                    attempts_per_pose=self.args.attempts_per_pose,
                    base_limits=self.base_limits, self_collision=True,
                    collision_pairs=self.collision_pairs, joint_positions={},
                    verification_pairs=self.verification_pairs)

    def solve(self, joints, palm, hand):
        joint_positions = {k: [float(x) for x in v] for k, v in joints.items()}
        robot_arm = spik.DEFAULT_ROBOT_ARM[hand]
        offset = spik.human_translation_offset(
            joint_positions, front_distance=spik.HUMAN_FRONT_DISTANCE)
        translated_joints = spik.translate_joint_positions(
            joint_positions, offset)
        translated_palm = spik.translate_palm(palm, offset)
        collision_obstacles = spik.human_body_obstacles(translated_joints)
        person_base_limits = list(self.base_limits)
        side_sign = spik.offered_hand_side_sign(
            hand, translated_joints, translated_palm)
        if side_sign is not None:
            person_base_limits[1] = spik.restrict_base_y_range_to_hand_side(
                person_base_limits[1], side_sign)
        human_yaw = spik.human_facing_yaw(translated_joints)
        if human_yaw is not None:
            person_base_limits[2] = spik.restrict_base_yaw_range_to_human_facing(
                person_base_limits[2], human_yaw,
                margin=math.radians(spik.DEFAULT_BASE_YAW_FACING_MARGIN_DEG))
        standing_xy = spik.human_standing_xy(translated_joints)
        standing_x = None if standing_xy is None else float(standing_xy[0])

        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            (picked, _, _, _, x_margin) = spik.solve_person_ik_side_by_side(
                self.robot, translated_palm, hand, robot_arm,
                collision_obstacles, person_base_limits, standing_x,
                x_margins=spik.DEFAULT_BASE_X_STANDING_MARGINS,
                front_offset_weight=spik.DEFAULT_FRONT_OFFSET_WEIGHT,
                facing_yaw_weight=spik.DEFAULT_FACING_YAW_WEIGHT,
                attempts_per_pose=self.args.attempts_per_pose,
                self_collision=True,
                collision_pairs=self.collision_pairs,
                joint_positions=translated_joints,
                verification_pairs=self.verification_pairs)
        elapsed = time.time() - t0
        log = buf.getvalue()
        # 呼び出し側向けに平行移動後の人物と解を残す。
        self.last_solution = dict(
            picked=picked, joints=translated_joints, palm=translated_palm,
            robot_arm=robot_arm, base_limits=person_base_limits)
        return classify(picked, log, elapsed, x_margin, robot_arm)


def classify(picked, log, elapsed, x_margin, robot_arm):
    """IK の結果と標準出力から結果の種類を決める.

    ok: 後処理まで成功。no_press: 後処理が全候補で失敗。human_clearance /
    self_collision: 人体距離 / 自己干渉で全て棄却。no_converge: 収束せず。
    """
    clearance = collections_counter(
        m.group(1) for m in _HOVER_CLEARANCE_RE.finditer(log))
    self_col = collections_counter(
        '{} x {}'.format(m.group(2), m.group(3))
        for m in _SELF_COLLISION_RE.finditer(log))
    n_post_fail = len(_POST_PROCESS_RE.findall(log))
    if picked is not None:
        status = 'ok' if picked[3] is not None else 'no_press'
    elif clearance and sum(clearance.values()) >= sum(self_col.values()):
        status = 'human_clearance'
    elif self_col:
        status = 'self_collision'
    else:
        status = 'no_converge'
    return dict(status=status, robot_arm=robot_arm, time=round(elapsed, 3),
                x_margin=x_margin, hover_clearance=clearance,
                self_collision=self_col, post_process_failures=n_post_fail)


def collections_counter(items):
    out = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
    return out


def position_grid(args):
    """(前方, 外側, 高さ) の格子。外側は差し出す手の側を正 (体の中心線から)。"""
    xs = np.arange(args.forward[0], args.forward[1] + 1e-9, args.forward[2])
    ls = np.arange(args.lateral[0], args.lateral[1] + 1e-9, args.lateral[2])
    zs = np.arange(args.height[0], args.height[1] + 1e-9, args.height[2])
    for z, x, lat in itertools.product(zs, xs, ls):
        yield dict(forward=round(float(x), 3), lateral=round(float(lat), 3),
                   height=round(float(z), 3), yaw=0.0, pitch=0.0,
                   roll=float(args.position_roll))


def orientation_grid(args):
    flat = args.orientation_position
    positions = [flat[i:i + 3] for i in range(0, len(flat) - 2, 3)]
    for (f, lat, z), pitch, yaw, roll in itertools.product(
            positions, args.pitch_values, args.yaw_values, args.roll_values):
        yield dict(forward=f, lateral=lat, height=z, yaw=float(yaw),
                   pitch=float(pitch), roll=float(roll))


def point_key(hand, p):
    return '{}|{forward}|{lateral}|{height}|{yaw}|{pitch}|{roll}'.format(
        hand, **p)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--mode', choices=('position', 'orientation'),
                        default='position')
    parser.add_argument('--hands', nargs='+', choices=('R', 'L'),
                        default=['R', 'L'])
    parser.add_argument('--output', required=True,
                        help='結果の JSON Lines (既にあれば続きから解く)。')
    parser.add_argument('--forward', type=float, nargs=3,
                        default=[0.10, 0.70, 0.05],
                        metavar=('MIN', 'MAX', 'STEP'),
                        help='掌の中心の前方距離 [m] (肩・腰を結ぶ面から)。')
    parser.add_argument('--lateral', type=float, nargs=3,
                        default=[-0.10, 0.50, 0.05],
                        metavar=('MIN', 'MAX', 'STEP'),
                        help='体の中心線からの外側距離 [m] (差し出す手の側が正)。')
    parser.add_argument('--height', type=float, nargs=3,
                        default=[0.70, 1.50, 0.10],
                        metavar=('MIN', 'MAX', 'STEP'),
                        help='掌の中心の床からの高さ [m]。')
    parser.add_argument('--position-roll', type=float, default=0.0,
                        help='--mode position での回内 [度] (--roll-values と同じ定義)。')
    parser.add_argument('--orientation-position', type=float, nargs='+',
                        default=[0.35, 0.20, 1.00],
                        help='--mode orientation で掌を置く (前方 外側 高さ) [m]、3 つ組を複数可。')
    parser.add_argument('--yaw-values', type=float, nargs='+',
                        default=[-30, 0, 30],
                        help='指先の向き (外側が正) [度]。')
    parser.add_argument('--pitch-values', type=float, nargs='+',
                        default=[-30, 0, 30],
                        help='指先の上下 (上が正) [度]。')
    parser.add_argument('--roll-values', type=float, nargs='+',
                        default=[-180, -135, -90, -45, 0, 45, 90, 135],
                        help='回内 [度] (0=親指が上、90=掌が下、-90=掌が上)。')
    parser.add_argument('--person-distances', type=float, nargs='+',
                        default=[1.0, 1.5, 2.0],
                        help='差し出し判定で人がロボットから立つ距離 [m]。')
    parser.add_argument('--offer-score-min', type=float, default=0.65,
                        help='差し出し判定の閾値。')
    parser.add_argument('--attempts-per-pose', type=int,
                        default=spik.DEFAULT_ATTEMPTS_PER_POSE)
    parser.add_argument('--collision-pairs', type=str,
                        default=os.path.join(_SCRIPTS_DIR,
                                             'collision_pairs.json'))
    parser.add_argument('--no-ik', action='store_true',
                        help='差し出し判定だけを計算する。')
    args = parser.parse_args()

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done.add(point_key(rec['hand'], rec['point']))
    grid = list(position_grid(args) if args.mode == 'position'
                else orientation_grid(args))
    todo = [(hand, p) for hand in args.hands for p in grid
            if point_key(hand, p) not in done]
    print('格子点 {} 点 x {} 手、残り {} 点'.format(
        len(grid), len(args.hands), len(todo)))
    if not todo:
        return

    robot_hand = robot_hand_position_initial()
    print('ロボットの初期姿勢の右手先 (base_link): {}'.format(
        robot_hand.round(3).tolist()))
    evaluator = None if args.no_ik else IkEvaluator(args)

    t_start = time.time()
    with open(args.output, 'a') as out:
        for i, (hand, p) in enumerate(todo):
            sign = _side_sign(hand)
            palm_pos = [p['forward'], sign * p['lateral'], p['height']]
            joints = build_skeleton(hand, palm_pos, p['yaw'], p['pitch'],
                                    p['roll'])
            rec = dict(hand=hand, point=p, robot_hand=robot_hand.tolist())
            if joints is None:
                rec['human_reachable'] = False
            else:
                rec['human_reachable'] = True
                rec['shoulder'] = shoulder_angles(joints, hand)
                palms, rec['offer'] = offer_scores(
                    joints, robot_hand, args.person_distances,
                    args.offer_score_min)
                if evaluator is not None:
                    np.random.seed(0)
                    rec['ik'] = evaluator.solve(joints, palms[hand], hand)
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
            out.flush()
            ik = rec.get('ik')
            elapsed = time.time() - t_start
            print('[{}/{}] {} f={forward:.2f} l={lateral:.2f} z={height:.2f} '
                  'yaw={yaw:.0f} pitch={pitch:.0f} roll={roll:.0f}: {} '
                  '(経過 {:.0f} 秒)'.format(
                      i + 1, len(todo), hand,
                      'human_unreachable' if joints is None
                      else (ik['status'] if ik else '-'), elapsed, **p))
            sys.stdout.flush()


if __name__ == '__main__':
    main()
