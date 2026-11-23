# -*- coding:utf-8 -*-
"""止まらない押し込み (台車は先に着き、腕は動き続ける間に手を検出し直して
押し込みの軌道を差し替える) の計算。ROS に依存しない。

実機ノード (``scripts/ros/run_camera_pipeline_test.py``) と確認用の表示ツール
(``tools/view_no_stop_press.py``) が同じ関数を使う。

流れ (時刻は接近の送信からの秒数)::

    台車  接近 ──────────────■ 着いて止まる (t_base) ──────── ■ 押し込み (t_s から)
    腕    接近 ──→ 手前の hover ～～ HOVER_DWELL 秒で hover へ ～～→ 押し込み
                              ↑ 検出・ずれの計測・IK        ↑ t_s = hover を通る時刻

- 接近の終わり (``retime_approach_arm``): 腕は計画の経路のまま、手先が hover
  に近づいてからの時間を、hover に着く時刻 (台車が着いてから ``HOVER_DWELL``
  秒後) まで一定の速さで進むよう割り振り直す。台車の着く点が「手前の hover」に
  なる (計画では腕が先に hover に着いているので、別の姿勢を置くと
  「伸ばす -> 引く -> 伸ばす」になる)。
- 腕は最初から [接近 (割り振り直した点を含む), hover, 計画の押し込み] を 1 本
  で送る。差し替えなければ計画どおり押し込む (止まらない)。
- 差し替え (``splice_in_dwell``): ``joint_trajectory_controller`` は、開始時刻
  t_s 付きの新しいゴールを受けると t_s までは古い軌道を続け、t_s の古い軌道の
  状態 (位置・速度) から新しい最初の点までを 1 本のスプラインでつなぐ
  (init_joint_trajectory.h)。t_s は判断できた直後 (手前の hover -> hover の
  ゆっくりした区間の途中) にし、新しいゴールは [hover (予定の時刻), 解き直した
  押し込み] にする。つなぎの区間の時間は、古い軌道の t_s の速度も入れて上限に
  収める (``fit_bridge``)。速度が速い押し込みの入口でつなぐと、向きを変える
  のに加速度の上限で 2〜3 秒延びるため、ゆっくりした区間でつなぐ。
"""

import math

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rpy_matrix

from aero_demo import execution_timing as et
from aero_demo import scan_matching

import solve_palm_ik as spik
from handshake_viewer_common import apply_waypoint_pose
from handshake_viewer_common import build_press_in_waypoints

# 台車が着いてから腕が hover に着くまで (手前の hover -> hover) の時間 [s]。
# 止まるのを待つ・検出 3 枚・ずれの計測・IK・送信の合計 (実機で 0.5〜1.2 秒)
# より長くする。
HOVER_DWELL = 1.5
# 接近の終わりの腕の時間を割り振り直す範囲 (手先が hover の手先からこの距離に
# 入った後) [m] (``retime_approach_arm``)。人体との距離の検証に通らなければ順に
# 狭める。
RETIME_RADII = (0.2, 0.15, 0.1)
# 差し替えで入る押し込みの点は、今の手先より押し込みの目標の向きにこれ以上前
# にある最初の点 [m] (``press_entry``)。
ENTRY_AHEAD = 0.005
# 接近の向き (``approach_slide_cos``) を見る、接近の末尾の点の数と、すべらせる
# 向きとの cos がこれ未満なら逆向きとみなしてほかの向き (turn) で計画し直す。
APPROACH_DIRECTION_POINTS = 6
REVERSE_APPROACH_COS = 0.0
# 差し替えの開始時刻を、計算を始める時刻のこれだけ後にする [s]。
SPLICE_LEAD = 0.15
# 計算の後、送る時点で開始時刻までこれだけ残っていなければ、かかった時間だけ
# 開始時刻を遅らせて計算し直す [s] (5 つのコントローラへの送信と通信の分)。
# 開始時刻の後にゴールが届くと、JTC は今の状態からつなぐので跳ばないが、つなぎの
# 区間が縮んで速くなる。
SPLICE_MIN_MARGIN = 0.05
SPLICE_ATTEMPTS = 3
# 台車のずれの許容 (これ以内なら動かさない) と、押し込みに織り込む上限。
BASE_POSITION_TOLERANCE = 0.025  # [m]
BASE_ANGLE_TOLERANCE = math.radians(5.0)  # [rad]
FOLD_MAX_POSITION = 0.12  # [m]
FOLD_MAX_ANGLE = math.radians(10.0)  # [rad]
# 検出した掌が計画から水平にこれ以上ずれていたら、その分 (人体との距離の検証に
# 通る量まで) 台車で寄せる。
PALM_SHIFT_MIN = 0.06  # [m]
CLEARANCE_SCALES = (1.0, 0.5, 0.25)


def place_at(robot, joint_names, waypoint):
    """``robot`` を ``waypoint`` の姿勢にする (IK で動いた本体を先に原点へ
    戻す。実機ノードの ``_place_robot_at_waypoint`` と同じ)。"""
    robot.newcoords(Coordinates())
    apply_waypoint_pose(robot, joint_names, [waypoint], 0)


def hand_position(robot, joint_names, waypoint, robot_arm):
    """``waypoint`` の手先の位置 (計画の系)。``robot`` の姿勢は書き換わる。"""
    place_at(robot, joint_names, waypoint)
    return getattr(robot, '{}arm_end_coords'.format(
        robot_arm)).worldpos().copy()


def press_direction(robot, joint_names, hover_wp, result):
    """``(hover の手先の位置, 押し込みの向きの単位ベクトル)`` (hover の手先から
    計画の押し込みの最終目標へ)。決まらなければ None。"""
    post = result.get('post_process')
    if post is None:
        return None
    hover_hand = hand_position(robot, joint_names, hover_wp,
                               result['robot_arm'])
    direction = np.asarray(post['target_position'],
                           dtype=np.float64) - hover_hand
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        return None
    return hover_hand, direction / norm


def approach_slide_cos(robot, joint_names, approach_wps, result, palm):
    """接近の終わりの手先の向き (末尾 ``APPROACH_DIRECTION_POINTS`` 点の始め
    から hover へ) と、
    押し込みですべらせる向き (``spik.press_in_slide_direction``) の cos。

    負なら手先は人の指先の側から来て、掌を越えた手首の側の hover まで行って
    から折り返してすべらせる (止まらないと「手首の方へ引いてから押し込む」に
    見える)。決まらなければ None。"""
    post = result.get('post_process')
    if post is None or len(approach_wps) < 2:
        return None
    robot_arm = result['robot_arm']
    first = hand_position(robot, joint_names,
                          approach_wps[-min(APPROACH_DIRECTION_POINTS,
                                            len(approach_wps))],
                          robot_arm)
    last = hand_position(robot, joint_names, approach_wps[-1], robot_arm)
    direction = last - first
    norm = float(np.linalg.norm(direction))
    if norm < 1e-3:
        return None
    slide = spik.press_in_slide_direction(
        palm, np.asarray(post['target_rot'], dtype=np.float64))
    return float(direction @ slide / norm)


def retime_approach_arm(robot, joint_names, approach_wps, result,
                        margin_fn=None):
    """接近の waypoint 列 (末尾が hover) の終わりの腕を、計画の経路のまま
    時間だけ割り振り直す (台車は同じ)。

    計画では、台車がゆっくり回り終える間に腕はもう hover の姿勢に着いて
    待っている。手先が hover の手先から ``radius`` に入った点 j から先は、
    腕を計画の経路 (点 j から hover までの関節角の列) に沿って、hover に着く
    時刻 (台車が着いてから ``HOVER_DWELL`` 秒後) まで一定の速さで進ませる
    (経路の長さは関節ごとの速度上限で割った時間の尺度)。台車が着く点は
    hover の少し手前 (「手前の hover」) になり、腕は後戻りしない。

    ``margin_fn(waypoint, None)`` があれば割り振り直した点の人体との距離の余裕
    を確かめ、通らなければ ``RETIME_RADII`` の次 (小さい範囲) で試す。
    Returns: ``(waypoints, info)``、どれも通らなければ ``(None, info)``。
    """
    robot_arm = result['robot_arm']
    hover_wp = approach_wps[-1]
    hover_hand = hand_position(robot, joint_names, hover_wp, robot_arm)
    hands = [hand_position(robot, joint_names, waypoint, robot_arm)
             for waypoint in approach_wps]
    joints = {joint.name: joint for joint in robot.joint_list}
    max_vel = max_velocities(joint_names, joints)
    times = limited_times(approach_wps, max_vel)
    arrivals = np.concatenate([[0.0], np.cumsum(times)])
    t_hover = float(arrivals[-1]) + HOVER_DWELL
    angles = np.array([wp['joint_angle_vector'] for wp in approach_wps],
                      dtype=np.float64)
    weights = np.where(np.isfinite(max_vel), 1.0 / max_vel, 0.0)
    n = len(approach_wps)
    info = dict(reason='no_radius')
    for radius in RETIME_RADII:
        far = [k for k, hand in enumerate(hands)
               if np.linalg.norm(hand - hover_hand) > radius]
        start = min(far[-1] + 1 if far else 0, n - 2)
        path = angles[start:]
        steps = np.abs(np.diff(path, axis=0)) * weights
        length = np.concatenate([[0.0], np.cumsum(np.max(steps, axis=1))])
        out = list(approach_wps)
        ok = True
        for k in range(start + 1, n):
            s = ((arrivals[k] - arrivals[start])
                 / (t_hover - arrivals[start]))
            if length[-1] > 1e-9:
                vector = np.array([np.interp(s * length[-1], length,
                                             path[:, i])
                                   for i in range(path.shape[1])])
            else:
                vector = path[-1]
            out[k] = dict(approach_wps[k], joint_angle_vector=[
                float(v) for v in vector])
            if margin_fn is not None:
                margin = margin_fn(out[k], None)
                if margin < 0.0:
                    info = dict(reason='clearance', radius=radius, index=k,
                                margin=margin)
                    ok = False
                    break
        if ok:
            out[-1] = dict(out[-1], pre_hover=True)
            pre_hand = hand_position(robot, joint_names, out[-1], robot_arm)
            return out, dict(
                reason='ok', radius=radius, start=start,
                retimed=n - 1 - start,
                arm_time=float(t_hover - arrivals[start]),
                pre_hover_distance=float(np.linalg.norm(
                    pre_hand - hover_hand)))
    return None, info


def no_stop_waypoints(retimed_approach, press_wps):
    """台車に送る接近 (``retime_approach_arm`` の結果、末尾が手前の hover) と、
    腕に送る 1 本 ([接近, hover, 押し込み (hover の次から)])。``press_wps``
    は先頭が hover。"""
    base_wps = list(retimed_approach)
    return base_wps, base_wps + list(press_wps)


# ---------------------------------------------------------------------------
# ずれ (表示ツールの仮想のずれ、実機の検出) から押し込みを解き直す
# ---------------------------------------------------------------------------

def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def wp_pose(wp):
    return np.array([wp['base_position'][0], wp['base_position'][1],
                     wp['base_yaw']], dtype=np.float64)


def offset_palm(palm, translation, rpy_deg=(0.0, 0.0, 0.0)):
    """掌 ``palm`` (world) を ``translation`` [m] (world) 動かし、掌の位置を
    中心に world の軸まわり (yaw, pitch, roll [deg]) に回したコピー。"""
    rot = rpy_matrix(*[math.radians(v) for v in rpy_deg])
    moved = spik.transform_palm(palm, rot, np.zeros(3))
    moved['position'] = (np.asarray(palm['position'], dtype=np.float64)
                         + np.asarray(translation, dtype=np.float64)).tolist()
    return moved


def offset_hand_joints(joint_positions, hand, palm, translation,
                       rpy_deg=(0.0, 0.0, 0.0)):
    """骨格の差し出した手 (手首と手の点) を ``offset_palm`` と同じに動かした
    コピー (表示と人体との距離の検証用)。"""
    rot = rpy_matrix(*[math.radians(v) for v in rpy_deg])
    center = np.asarray(palm['position'], dtype=np.float64)
    shift = np.asarray(translation, dtype=np.float64)
    moved = {}
    for name, p in joint_positions.items():
        if (name.startswith((hand + 'Hand', hand + 'PalmModel'))
                or name == hand + 'Wrist'):
            q = np.asarray(p, dtype=np.float64)
            moved[name] = (rot @ (q - center) + center + shift).tolist()
        else:
            moved[name] = p
    return moved


def palm_in_base(palm_world, base_pose):
    """world の掌を、台車の姿勢 ``base_pose`` ``[x, y, yaw]`` の系へ。"""
    rot = _rz(base_pose[2]).T
    return spik.transform_palm(
        palm_world, rot, -rot @ np.array([base_pose[0], base_pose[1], 0.0]))


def base_residual(target_pose, actual_pose):
    """実機ノードの ``_measure_base_residual`` と同じ目標までの残差 (今の台車の
    系の ``[dx, dy, dyaw]``)。向きは許容内なら直さない。"""
    target = np.asarray(target_pose, dtype=np.float64).copy()
    yaw_error = scan_matching.wrap_angle(target[2] - actual_pose[2])
    if abs(yaw_error) <= BASE_ANGLE_TOLERANCE:
        target[2] = actual_pose[2]
    return np.asarray(scan_matching.relative(actual_pose, target))


def fold_move(residual):
    """残差から、押し込みで動かす量 (許容内なら 0) と、織り込めるか。"""
    distance = math.hypot(residual[0], residual[1])
    within = (distance <= BASE_POSITION_TOLERANCE
              and abs(residual[2]) <= BASE_ANGLE_TOLERANCE)
    ok = distance <= FOLD_MAX_POSITION and abs(residual[2]) <= FOLD_MAX_ANGLE
    return (np.zeros(3) if within else np.asarray(residual, dtype=np.float64)
            ), bool(ok)


def clearance_scale(margin_at):
    """``margin_at(scale)`` (台車のずらしをその割合にしたときの人体との距離の
    余裕) で縮める割合を決める (実機ノードの ``_clearance_scale`` と同じ)。
    ``(割合 or None, 余裕)``。"""
    margins = []
    for scale in CLEARANCE_SCALES:
        margin = margin_at(scale)
        if margin >= 0.0:
            return scale, margin
        margins.append((margin, scale))
    best_margin, best_scale = max(margins)
    if best_margin > margin_at(0.0):
        return best_scale, best_margin
    return None, best_margin


def plan_palm_base_shift(palm, expected_position, hover_wp, base_error,
                         planned_palm, margin_fn):
    """検出した掌 ``palm`` (台車の系) が計画 (``expected_position``) から水平に
    ``PALM_SHIFT_MIN`` 以上ずれていれば、台車を寄せる量 (台車の系の xy、z=0)
    を人体との距離の検証で縮めて決める (動かさない)。

    ``margin_fn(waypoint, hand_shift)`` は hover の waypoint と、差し出した手を
    計画の系で ``hand_shift`` (xy) 動かした骨格での人体との距離の余裕。
    ``base_error`` は台車の hover 目標からのずれ (計画の系、None なら 0)。
    ``(shift or None, record)``。``record['reason']`` は 'small'/'clearance'/
    'scaled_small'/'shift'。
    """
    shift = np.asarray(palm['position'], dtype=np.float64) - expected_position
    shift[2] = 0.0
    distance = float(np.linalg.norm(shift))
    record = dict(event='press_in_base_shift', distance=distance,
                  shift=[float(v) for v in shift[:2]], applied=False)
    if distance < PALM_SHIFT_MIN:
        record['reason'] = 'small'
        return None, record
    wanted = shift.copy()
    actual_wp = hover_wp
    if base_error is not None and planned_palm is not None:
        actual = scan_matching.compose(wp_pose(hover_wp), base_error)
        actual_wp = dict(hover_wp, base_position=[actual[0], actual[1], 0.0],
                         base_yaw=float(actual[2]))
        to_plan = _rz(actual[2])[:2, :2]
        palm_plan = actual[:2] + to_plan @ np.asarray(
            palm['position'], dtype=np.float64)[:2]
        hand_shift = palm_plan - np.asarray(
            planned_palm['position'], dtype=np.float64)[:2]
        record.update(base_error=[float(v) for v in base_error],
                      hand_shift=[float(v) for v in hand_shift])
    else:
        to_plan = _rz(hover_wp['base_yaw'])[:2, :2]
        hand_shift = to_plan @ wanted[:2]

    def margin_at(scale):
        shift_plan = to_plan @ (wanted[:2] * scale)
        moved = dict(actual_wp, base_position=[
            actual_wp['base_position'][0] + float(shift_plan[0]),
            actual_wp['base_position'][1] + float(shift_plan[1]), 0.0])
        return margin_fn(moved, hand_shift)

    scale, margin = clearance_scale(margin_at)
    if scale is None:
        record.update(reason='clearance', clearance_margin=margin)
        return None, record
    shift = wanted * scale
    record.update(scale=scale, clearance_margin=margin)
    if float(np.linalg.norm(shift)) < PALM_SHIFT_MIN / 2:
        record['reason'] = 'scaled_small'
        return None, record
    record['reason'] = 'shift'
    return shift, record


def _without_retreat(move_xy, axis_xy):
    """台車の並進 ``move_xy`` から、``axis_xy`` (掌へ向かう単位ベクトル) と逆向き
    (掌から離れる向き) の成分を除く。"""
    move_xy = np.asarray(move_xy, dtype=np.float64)
    along = float(move_xy @ axis_xy)
    return move_xy - min(0.0, along) * axis_xy


def refine_folded(robot, joint_names, hover_wp, result, planned_palm,
                  palm_now, base_move, base_error=None, margin_fn=None):
    """台車がまだ ``base_move`` (今の台車の系) だけ動いていない前提で、検出した
    掌 ``palm_now`` (今の台車の系) から押し込みを解き直す。

    掌を動いた後の台車の系へ移し、掌の水平のずれを台車で寄せる分
    (``margin_fn`` があれば ``plan_palm_base_shift``) も ``base_move`` に足して
    から ``spik.refine_post_process`` で解く。

    台車の動き (ずれの補正・寄せる分) のうち押し込む向き
    (計画の hover の手先から押し込みの目標へ、水平) と逆向きの成分を除き、
    その分は腕で吸収する。台車が人の側へずれた・人が手を前に出したときに
    台車を下げると、押し込みの間に手先が引かれるため。それで解けなければ
    除かずに解き直す。

    Returns: ``dict(press_waypoints (hover の次から) or None, base_move,
    info (refine_post_process の結果), shift_record, observed (計画の系の掌),
    no_retreat (掌から離れる向きを除いて解けたか))``。解けなければ
    ``base_move`` は寄せる分を足さない元の値。
    """
    axis = np.zeros(2)
    found = press_direction(robot, joint_names, hover_wp, result)
    if found is not None:
        # 押し込む向きを hover の台車の系の水平へ。
        axis = _rz(hover_wp['base_yaw']).T[:2, :2] @ found[1][:2]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm > 1e-6:
        out = _refine_folded(robot, joint_names, hover_wp, result,
                             planned_palm, palm_now, base_move, base_error,
                             margin_fn, axis / axis_norm)
        if out['press_waypoints'] is not None:
            out['no_retreat'] = True
            return out
    out = _refine_folded(robot, joint_names, hover_wp, result, planned_palm,
                         palm_now, base_move, base_error, margin_fn, None)
    out['no_retreat'] = False
    return out


def _refine_folded(robot, joint_names, hover_wp, result, planned_palm,
                   palm_now, base_move, base_error, margin_fn, axis):
    place_at(robot, joint_names, hover_wp)
    base_rot = robot.base_link.worldrot().copy()
    base_pos = robot.base_link.worldpos().copy()
    expected_position = base_rot.T @ (
        np.asarray(planned_palm['position'], dtype=np.float64) - base_pos)
    move = np.asarray(base_move, dtype=np.float64).copy()
    if axis is not None:
        move[:2] = _without_retreat(move[:2], axis)
    original_move = move.copy()
    move_rot = _rz(move[2])
    move_pos = np.array([move[0], move[1], 0.0])
    palm = spik.transform_palm(palm_now, move_rot.T, -move_rot.T @ move_pos)
    shift_record = None
    pending = None
    if margin_fn is not None:
        shift, shift_record = plan_palm_base_shift(
            palm, expected_position, hover_wp, base_error, planned_palm,
            margin_fn)
        if shift is not None and axis is not None:
            # 寄せる分も掌から離れる向きは除く (動いた後の台車の系の向きで)。
            shift = shift.copy()
            shift[:2] = _without_retreat(shift[:2], move_rot[:2, :2].T @ axis)
            if float(np.linalg.norm(shift[:2])) < 1e-3:
                shift = None
        if shift is not None:
            palm = dict(palm, position=[
                float(v) for v in np.asarray(palm['position']) - shift])
            pending = scan_matching.compose(move, [shift[0], shift[1], 0.0])
    place_at(robot, joint_names, hover_wp)
    observed = spik.transform_palm(palm, base_rot, base_pos)
    info = spik.refine_post_process(
        robot, result['robot_arm'], observed, result['turn_deg'],
        result['post_process'],
        planned_palm=(planned_palm if 'x_axis' in planned_palm else None))
    press = None
    move = original_move
    if info['post_process'] is not None:
        press = build_press_in_waypoints(hover_wp, joint_names,
                                         info['post_process'])
        if pending is not None:
            move = np.asarray(pending, dtype=np.float64)
    return dict(press_waypoints=press, base_move=move, info=info,
                shift_record=shift_record, observed=observed)


def hover_clearance_margin(robot, joint_names, waypoint, joint_positions,
                           offered_hand, verification_pairs,
                           torso_surface_offset, hand_shift=None):
    """``waypoint`` の、骨格 ``joint_positions`` に対する人体との距離の余裕
    (実機ノードの ``_hover_clearance_margin`` と同じ)。"""
    import plan_handshake_motion as phm
    frozen = joint_positions
    if hand_shift is not None:
        moved = np.array([hand_shift[0], hand_shift[1], 0.0])
        frozen = {
            name: ((np.asarray(p, dtype=np.float64) + moved).tolist()
                   if name.startswith((offered_hand + 'Hand',
                                       offered_hand + 'PalmModel'))
                   or name == offered_hand + 'Wrist' else p)
            for name, p in frozen.items()}
    collision_joints = spik.shift_torso_joints_from_surface(
        frozen, (0.0, 0.0), torso_surface_offset)
    return float(phm.verify_human_clearance(
        robot, joint_names, [waypoint], verification_pairs,
        phm.build_obstacle_cache(collision_joints), hand=offered_hand,
        robot_arm=spik.DEFAULT_ROBOT_ARM[offered_hand])[0])


# ---------------------------------------------------------------------------
# 時間づけと差し替え
# ---------------------------------------------------------------------------

def max_velocities(joint_names, joints_by_name):
    """``joint_names`` の順の速度上限 (× VEL_LIMIT_RATIO)。"""
    return et.joint_max_velocities(
        [joints_by_name[name] for name in joint_names], None)


def limited_times(waypoints, max_vel, with_base=True):
    """``waypoints`` (先頭が動き始め) の区間ごとの時間 (len - 1 個)。"""
    angles = np.array([wp['joint_angle_vector'] for wp in waypoints],
                      dtype=np.float64)
    base = (np.asarray(et.base_trajectory_points(waypoints))[1:]
            if with_base else None)
    return et.limited_time_list(np.diff(angles, axis=0), max_vel, base,
                                [et.MIN_SEGMENT_TIME] * (len(waypoints) - 1))


def point_velocities(angles, times):
    """点 (len(times) + 1 個) の速度 (skrobot・execution_timing と同じ規則)。"""
    angles = np.asarray(angles, dtype=np.float64)
    return et.point_velocities(np.diff(angles, axis=0),
                               np.asarray(times, dtype=np.float64))


def fit_bridge(start_position, start_velocity, angles, times, max_vel):
    """差し替え: t_s の古い軌道の状態 (``start_position``・``start_velocity``)
    から ``angles`` (新しいゴールの点) へつなぐときの、区間の時間と点の速度。

    ``times`` は [start -> angles[0], angles[0] -> angles[1], ...] の時間の
    下限 (``limited_times`` などで決めたもの)。コントローラは最初の区間を
    (start_position, start_velocity) から (angles[0], 点の速度) への 3 次
    スプラインで動かすので、その区間の速度・加速度が上限に収まるまで延ばす。
    Returns: ``(times, velocities (angles と同じ形), 最初の区間の上限比)``。
    """
    times = np.asarray(times, dtype=np.float64).copy()
    path = np.vstack([np.asarray(start_position, dtype=np.float64),
                      np.asarray(angles, dtype=np.float64)])
    max_vel = np.asarray(max_vel, dtype=np.float64)
    ratio = 0.0
    for _ in range(50):
        vel = et.point_velocities(np.diff(path, axis=0), times)
        vel[0] = np.asarray(start_velocity, dtype=np.float64)
        delta = (path[1] - path[0])[None, :]
        peak_v, peak_a = et.hermite_peaks(
            delta, times[:1], np.vstack([vel[0], vel[1]]))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = float(np.nanmax(np.concatenate([
                peak_v[0] / max_vel,
                np.sqrt(peak_a[0] / (max_vel / et.ACCEL_TIME))])))
        if not np.isfinite(ratio) or ratio <= 1.0:
            break
        times[0] *= math.sqrt(ratio)
    vel = et.point_velocities(np.diff(path, axis=0), times)
    return times, vel[1:], ratio


def monotone_press(robot, joint_names, robot_arm, press_wps, current_hand,
                   target):
    """差し替えで通る押し込みの点 (``press_wps`` の添字の列、先頭が hover)。

    押し込む向き (押し込みの最初の点の手先から目標 ``target`` へ) に、今の手先
    ``current_hand`` より前へ単調に進む点だけを残す: 今の手先より後ろの点
    (人が手を前に出した・台車が人の側へずれたときの hover など) と、手前の点
    より後ろへ戻る点 (掌がずれて解き直した経路の始まりが計画の hover より後ろ
    のとき) は飛ばす。そこを通ると「引いてから押し込む」になる。最後の点
    (押し込みの最終姿勢) は必ず残す。
    """
    hands = [hand_position(robot, joint_names, waypoint, robot_arm)
             for waypoint in press_wps]
    target = np.asarray(target, dtype=np.float64)
    direction = target - hands[min(1, len(hands) - 1)]
    if float(np.linalg.norm(direction)) < 1e-3:
        direction = target - current_hand
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return [len(press_wps) - 1]
    direction /= norm
    last = float(current_hand @ direction) + ENTRY_AHEAD
    keep = []
    for k, hand in enumerate(hands):
        progress = float(hand @ direction)
        if progress > last:
            keep.append(k)
            last = progress
    if not keep or keep[-1] != len(press_wps) - 1:
        keep.append(len(press_wps) - 1)
    return keep


def splice_at_entry(track, t_s, angles, times, max_vel):
    """t_s の送った軌道 ``track`` の状態から、押し込みの入る点以降 ``angles``
    へ直接つなぐ (``press_entry`` が hover より先のとき)。``times`` は
    [今 -> angles[0], angles[0] -> angles[1], ...] の下限。
    Returns: ``(angles, velocities, times, 上限比)``。"""
    position, velocity = track.state(t_s)
    times, velocities, ratio = fit_bridge(position, velocity, angles, times,
                                          max_vel)
    return ([np.asarray(a, dtype=np.float64) for a in angles], velocities,
            times, ratio)


def splice_in_dwell(track, t_s, t_hover, hover_angles, new_angles,
                    new_times, max_vel):
    """手前の hover -> hover のゆっくりした区間の途中 ``t_s`` で差し替える
    ゴール: [hover (予定の ``t_hover`` に着く), ``new_angles`` (押し込み)]。

    ``track`` は送った腕の軌道 (``HermiteTrack``、時刻は同じ基準)。t_s の
    状態から hover までのつなぎを ``fit_bridge`` で上限に収める (延びたら
    hover に着くのも遅れる)。``new_times`` は hover -> new_angles の区間の
    時間。Returns: ``(angles, velocities, times, 上限比)`` (angles の先頭が
    hover、times の先頭が t_s から hover まで)。
    """
    position, velocity = track.state(t_s)
    angles = [np.asarray(hover_angles, dtype=np.float64)] + [
        np.asarray(a, dtype=np.float64) for a in new_angles]
    times = [max(t_hover - t_s, et.MIN_SEGMENT_TIME)] + list(new_times)
    times, velocities, ratio = fit_bridge(position, velocity, angles, times,
                                          max_vel)
    return angles, velocities, times, ratio


class HermiteTrack(object):
    """点・速度・区間の時間から、コントローラと同じ 3 次エルミートで時刻の
    姿勢を引く (表示ツールの再生用)。先頭の点が時刻 ``t0``。"""

    def __init__(self, points, velocities, times, t0=0.0):
        self.points = np.asarray(points, dtype=np.float64)
        self.velocities = np.asarray(velocities, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self.t0 = float(t0)
        self.starts = self.t0 + np.concatenate(
            [[0.0], np.cumsum(self.times)[:-1]])
        self.end = self.t0 + float(np.sum(self.times))

    def state(self, t):
        if len(self.times) == 0 or t <= self.t0:
            return self.points[0].copy(), self.velocities[0].copy()
        if t >= self.end:
            return self.points[-1].copy(), np.zeros_like(self.points[-1])
        k = int(np.searchsorted(self.starts, t, side='right') - 1)
        h = self.times[k]
        tau = t - self.starts[k]
        p0, p1 = self.points[k], self.points[k + 1]
        v0, v1 = self.velocities[k], self.velocities[k + 1]
        a = (v0 + v1) / h ** 2 - 2.0 * (p1 - p0) / h ** 3
        b = 3.0 * (p1 - p0) / h ** 2 - (2.0 * v0 + v1) / h
        return (p0 + v0 * tau + b * tau ** 2 + a * tau ** 3,
                v0 + 2.0 * b * tau + 3.0 * a * tau ** 2)

    def spliced(self, t_s, points, velocities, times):
        """``t_s`` で差し替えた軌道 (t_s までは自分、t_s の状態から ``points``
        へ)。"""
        p, v = self.state(t_s)
        keep = int(np.searchsorted(self.starts, t_s, side='right'))
        head_points = list(self.points[:keep]) + [p]
        head_vel = list(self.velocities[:keep]) + [v]
        head_times = list(self.times[:keep - 1]) + [
            t_s - self.starts[keep - 1]] if keep >= 1 else []
        return HermiteTrack(
            head_points + list(points), head_vel + list(velocities),
            head_times + list(times), self.t0)
