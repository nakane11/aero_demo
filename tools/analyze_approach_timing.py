#!/usr/bin/env python
"""実機試行の計画 (motions/*.json) の接近 (lead-in + 螺旋) を、実機ノードと同じ
``execution_timing.limited_time_list`` で時間づけし、区間ごとにどの軸の上限
(台車の並進・回転、関節、速度か加速度か) が時間を決めているかを集計する。

使い方 (FEETECH_HAND_DIR が要る):
    python tools/analyze_approach_timing.py data/real/2026-10-22/logs/*/person_*.log

``--base-vel``/``--base-angvel``/``--vel-ratio``/``--accel-time`` で上限を変えたときの
時間も出す (実機の上限を超える値は実機では使えない)。``--no-uniform-stretch`` は
反復で収まらなかった超過を全区間の一律の延長でなく、超えた区間だけ延ばす。
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, 'src'))

from aero_demo import aero_urdf_setup  # noqa: E402
from aero_demo import execution_timing as et  # noqa: E402

SAVE_RE = re.compile(r'(\d{8}_\d{6})\.json に保存')
SEGMENT_RE = re.compile(r'waypoint数=(\d+) .*time_list合計=([0-9.]+)s')


def find_trials(paths):
    for path in paths:
        name = None
        logged = None
        with open(path, encoding='utf-8', errors='replace') as f:
            for line in f:
                m = SAVE_RE.search(line)
                if m and name is None:
                    name = m.group(1) + '.json'
                m = SEGMENT_RE.search(line)
                if m and logged is None:
                    logged = (int(m.group(1)), float(m.group(2)))
        if name is None:
            continue
        motion_path = os.path.join(os.path.dirname(path), 'motions', name)
        if os.path.exists(motion_path):
            yield path, motion_path, logged


def axis_ratios(arm_deltas, arm_max_vel, base_deltas, times, accel_time,
                base_vel, base_angvel, vel_ratio):
    """区間 x 軸 の 速度/上限 と sqrt(加速度/上限)。列名も返す。"""
    cols, names = [], []
    pv = et.point_velocities(arm_deltas, times)
    vel, acc = et.hermite_peaks(arm_deltas, times, pv)
    cols += [vel / arm_max_vel, np.sqrt(acc / (arm_max_vel / accel_time))]
    names += ['vel:' + n for n in JOINT_NAMES] + ['acc:' + n for n in JOINT_NAMES]
    pv = et.point_velocities(base_deltas, times)
    vxy, axy = et.hermite_peaks(base_deltas[:, :2], times, pv[:, :2], norm=True)
    vyaw, ayaw = et.hermite_peaks(base_deltas[:, 2:], times, pv[:, 2:])
    lim = np.array([base_vel, base_angvel]) * vel_ratio
    cols += [np.stack([vxy, vyaw[:, 0]], 1) / lim,
             np.sqrt(np.stack([axy, ayaw[:, 0]], 1) / (lim / accel_time))]
    names += ['vel:base_xy', 'vel:base_yaw', 'acc:base_xy', 'acc:base_yaw']
    return np.concatenate(cols, axis=1), names


def timed(arm_deltas, joint_vel, base_points, n, args, uniform=True):
    CATMULL[0] = getattr(args, 'base_tangent', 'zero') == 'catmull'
    et.BASE_MAX_VEL = args.base_vel
    et.BASE_MAX_ANGVEL = args.base_angvel
    et.VEL_LIMIT_RATIO = args.vel_ratio
    et.ACCEL_TIME = args.accel_time
    arm_max_vel = joint_vel * args.vel_ratio
    times = np.asarray(et.limited_time_list(
        arm_deltas, arm_max_vel, base_points, [et.MIN_SEGMENT_TIME] * n))
    if not uniform:
        # 一律に延ばした分を戻し、超えた区間だけ延ばす。
        base_deltas = np.diff(np.vstack([np.zeros(3), base_points]), axis=0)
        for _ in range(200):
            r, _ = axis_ratios(arm_deltas, arm_max_vel, base_deltas, times,
                               args.accel_time, args.base_vel,
                               args.base_angvel, args.vel_ratio)
            worst = r.max(axis=1)
            if worst.max() <= 1.0 + 1e-3:
                break
            times = np.where(worst > 1.0, times * np.sqrt(worst), times)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--base-vel', type=float, default=et.BASE_MAX_VEL)
    parser.add_argument('--base-angvel', type=float, default=et.BASE_MAX_ANGVEL)
    parser.add_argument('--vel-ratio', type=float, default=et.VEL_LIMIT_RATIO)
    parser.add_argument('--accel-time', type=float, default=et.ACCEL_TIME)
    parser.add_argument('--base-tangent', choices=['zero', 'catmull'],
                        default='zero',
                        help='変更後の台車の点の速度。zero: 今の規則 (軸ごとに'
                             '符号が変わる点は 0)、catmull: 前後の点の差から'
                             '接線 (符号が変わっても 0 にしない)')
    parser.add_argument('--lead-in-concurrent', action='store_true',
                        help='変更後は lead-in の回転と直進を同時に行う '
                             '(回ってから進むのでなく、同じ点数で x・y・yaw '
                             'を一緒に補間する)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.base_tangent == 'catmull':
        original = et.point_velocities

        def point_velocities(deltas, times, _orig=original):
            # 台車 (3 軸、または並進の 2 軸) だけ接線を変える。
            if deltas.shape[1] > 3 or not CATMULL[0]:
                return _orig(deltas, times)
            n = len(times)
            pts = np.vstack([np.zeros(deltas.shape[1]),
                             np.cumsum(deltas, axis=0)])
            cum = np.concatenate([[0.0], np.cumsum(times)])
            vel = np.zeros((n + 1, deltas.shape[1]))
            if n > 1:
                vel[1:n] = ((pts[2:] - pts[:-2])
                            / (cum[2:] - cum[:-2])[:, None])
            return vel
        et.point_velocities = point_velocities
    paths = []
    for p in args.logs:
        paths += sorted(glob.glob(p))

    robot = aero_urdf_setup.load_aero(use_hand=True)
    joints = {j.name: j for j in robot.joint_list}
    global JOINT_NAMES
    default = argparse.Namespace(base_vel=et.BASE_MAX_VEL,
                                 base_angvel=et.BASE_MAX_ANGVEL,
                                 vel_ratio=et.VEL_LIMIT_RATIO,
                                 accel_time=et.ACCEL_TIME)
    share_total = {}
    rows = []
    for log, motion_path, logged in find_trials(paths):
        motion = json.load(open(motion_path))
        JOINT_NAMES = motion['joint_names']
        joint_vel = np.array([
            joints[n].max_joint_velocity if joints[n].max_joint_velocity > 0
            else np.inf for n in JOINT_NAMES])
        wps = list(motion.get('lead_in_waypoints') or []) + motion['waypoints']
        n_lead = len(motion.get('lead_in_waypoints') or [])
        angles = np.array([w['joint_angle_vector'] for w in wps])
        arm_deltas = np.diff(angles, axis=0)
        base_points = np.asarray(et.base_trajectory_points(wps))[1:]
        n = len(wps) - 1
        t_now = timed(arm_deltas, joint_vel, base_points, n, default)
        t_seg = timed(arm_deltas, joint_vel, base_points, n, default,
                      uniform=False)
        new_arm, new_base, new_n = arm_deltas, base_points, n
        if args.lead_in_concurrent and n_lead:
            lead = motion['lead_in_waypoints']
            first, last = lead[0], motion['waypoints'][0]
            start = np.array([first['base_position'][0],
                              first['base_position'][1], first['base_yaw']])
            end = np.array([last['base_position'][0],
                            last['base_position'][1], last['base_yaw']])
            k = max(int(np.ceil(np.linalg.norm(end[:2] - start[:2]) / 0.1)),
                    1)
            a0 = np.asarray(first['joint_angle_vector'])
            a1 = np.asarray(last['joint_angle_vector'])
            lead_new = [dict(base_position=[*(start + (end - start) * u)[:2],
                                            0.0],
                             base_yaw=float((start + (end - start) * u)[2]),
                             joint_angle_vector=list(a0 + (a1 - a0) * u))
                        for u in np.linspace(0.0, 1.0, k + 1)[:-1]]
            wps_new = lead_new + motion['waypoints']
            new_arm = np.diff(np.array(
                [w['joint_angle_vector'] for w in wps_new]), axis=0)
            new_base = np.asarray(et.base_trajectory_points(wps_new))[1:]
            new_n = len(wps_new) - 1
        t_new = timed(new_arm, joint_vel, new_base, new_n, args)
        t_new_seg = timed(new_arm, joint_vel, new_base, new_n, args,
                          uniform=False)
        base_deltas = np.diff(np.vstack([np.zeros(3), base_points]), axis=0)
        CATMULL[0] = False  # 律速の内訳は実機と同じ接線で見る
        r, names = axis_ratios(arm_deltas, joint_vel * default.vel_ratio,
                               base_deltas, t_seg, default.accel_time,
                               default.base_vel, default.base_angvel,
                               default.vel_ratio)
        binding = [names[i] for i in r.argmax(axis=1)]
        share = {}
        for b, t in zip(binding, t_seg):
            share[b] = share.get(b, 0.0) + t
            share_total[b] = share_total.get(b, 0.0) + t
        lead_t = float(np.sum(t_now[:max(n_lead - 1, 0)]))
        rows.append((os.path.relpath(log), logged, float(t_now.sum()), lead_t,
                     float(t_seg.sum()), float(t_new.sum()),
                     float(t_new_seg.sum())))
        if args.verbose:
            print(os.path.relpath(log))
            for k, (b, t) in enumerate(zip(binding, t_seg)):
                print('  seg {:2d} {:5.2f}s {}'.format(k, t, b))
    print('{:<55} {:>14} {:>7} {:>7} {:>9} {:>9} {:>11}'.format(
        'log', '実機ログ(点,s)', '今', 'lead-in', '区間だけ', '変更後', '変更後+区間'))
    for log, logged, now, lead, seg, new, new_seg in rows:
        print('{:<55} {:>14} {:7.2f} {:7.2f} {:9.2f} {:9.2f} {:11.2f}'.format(
            log[-55:], '-' if logged is None else '{} {:.2f}'.format(*logged),
            now, lead, seg, new, new_seg))
    total = sum(share_total.values())
    print('\n時間を決めている軸 (区間だけ延ばす場合の時間の割合):')
    for b, t in sorted(share_total.items(), key=lambda kv: -kv[1])[:12]:
        print('  {:<28} {:5.1f}%  ({:.1f}s)'.format(b, 100 * t / total, t))


JOINT_NAMES = []
CATMULL = [False]

if __name__ == '__main__':
    main()
