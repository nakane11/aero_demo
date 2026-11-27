#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人だかりのスキャン照合の比較 (人の脚をスキャンに書き込んで照合の誤差を測る).

実機の bag の /scan から、台車が動いている区間のスキャンの組を取り、人 (脚 2 本の
円) を光線の計算で両方のスキャンに書き込む。一部の人は 2 枚の間に動かす。正解は
人を足す前のスキャンを 0.1 秒ごとに照合してつないだ移動量。初期値には odom の
スリップ相当のずれを足し、次を比べる。

- ``base``: 今の照合 (``icp_2d``)。
- ``oracle``: 人の位置 (両方の時刻) を全部知っていて、その周りを除く
  (``exclude_centers``、カメラで全員が見えたときの上限の目安)。

座標はすべてレーザーの座標系 (台車との取り付けの差は比較に影響しない)。

例::

    source /opt/ros/noetic/setup.bash
    PYTHONPATH=src python3 tools/eval_scan_crowd.py \\
        --bag data/bags/camera_pipeline_2026-11-27-17-12-34.bag --start 26 --end 34
"""

import argparse
import json
import math

import numpy as np
import rosbag

from aero_demo import scan_matching as sm

LEG_RADIUS = 0.06
LEG_SEPARATION = 0.2
MIN_INLIER_RATIO = 0.3  # ノードの SCAN_MIN_INLIER_RATIO など
MIN_INLIERS = 100
MAX_RMS = 0.03
MAX_RANGE = 10.0


def read_scans(bag_path, start, end, topic='/scan'):
    scans = []
    with rosbag.Bag(bag_path) as bag:
        t0 = bag.get_start_time()
        for _, msg, t in bag.read_messages(topics=[topic]):
            rel = t.to_sec() - t0
            if rel < start:
                continue
            if rel > end:
                break
            scans.append(dict(
                t=rel, ranges=np.asarray(msg.ranges, dtype=np.float64),
                angle_min=msg.angle_min, angle_increment=msg.angle_increment,
                range_min=msg.range_min, range_max=msg.range_max))
    return scans


def to_points(scan, ranges=None):
    return sm.scan_to_points(
        scan['ranges'] if ranges is None else ranges, scan['angle_min'],
        scan['angle_increment'], scan['range_min'], scan['range_max'],
        max_range=MAX_RANGE)


def chain_poses(scans, step=0.1):
    """0.1 秒ごとに照合をつないで、先頭のスキャンから見た各スキャンの姿勢."""
    poses = [np.zeros(3)]
    last_i = 0
    out = {0: np.zeros(3)}
    prev_pose = np.zeros(3)
    for i in range(1, len(scans)):
        if scans[i]['t'] - scans[last_i]['t'] < step and i != len(scans) - 1:
            continue
        m = sm.icp_2d(to_points(scans[last_i]), to_points(scans[i]),
                      np.zeros(3))
        if m['rms'] > 0.02 or m['inlier_ratio'] < 0.6:
            return None
        prev_pose = sm.compose(prev_pose, np.asarray(m['pose']))
        out[i] = prev_pose
        last_i = i
    del poses
    return out


def raycast(scan, legs):
    """``legs`` (M, 2) [レーザー座標系] の円で ``scan`` の光線を遮る."""
    ranges = scan['ranges'].copy()
    if len(legs) == 0:
        return ranges
    angles = scan['angle_min'] + scan['angle_increment'] * np.arange(len(ranges))
    d = np.column_stack([np.cos(angles), np.sin(angles)])  # (N, 2)
    for c in legs:
        b = d @ c                       # 光線上で円の中心に最も近い距離
        h2 = c @ c - b ** 2
        hit = (b > 0) & (h2 <= LEG_RADIUS ** 2)
        r = b - np.sqrt(np.maximum(LEG_RADIUS ** 2 - h2, 0.0))
        closer = hit & (r > scan['range_min']) & ~(
            np.isfinite(ranges) & (ranges < r))
        ranges[closer] = r[closer]
    return ranges


def people_legs(centers, headings):
    legs = []
    for c, h in zip(centers, headings):
        off = 0.5 * LEG_SEPARATION * np.array([-math.sin(h), math.cos(h)])
        legs += [c + off, c - off]
    return np.array(legs).reshape(-1, 2)


def evaluate(ref_scan, cur_scan, gt, n_people, moving_ratio, rng, slip):
    # 人の位置 (基準のレーザー座標系、0.6〜4 m)。
    r = rng.uniform(0.6, 4.0, n_people)
    a = rng.uniform(-math.pi, math.pi, n_people)
    centers = np.column_stack([r * np.cos(a), r * np.sin(a)])
    headings = rng.uniform(-math.pi, math.pi, n_people)
    moved = centers.copy()
    n_moving = int(round(moving_ratio * n_people))
    if n_moving:
        step = rng.uniform(0.2, 0.6, n_moving)
        direction = rng.uniform(-math.pi, math.pi, n_moving)
        moved[:n_moving] += np.column_stack([step * np.cos(direction),
                                             step * np.sin(direction)])
    ref_ranges = raycast(ref_scan, people_legs(centers, headings))
    cur_legs_ref = people_legs(moved, headings)
    cur_legs = sm.transform_points(cur_legs_ref, sm.inverse(gt))
    cur_ranges = raycast(cur_scan, cur_legs)
    ref_pts = to_points(ref_scan, ref_ranges)
    cur_pts = to_points(cur_scan, cur_ranges)
    init = sm.compose(gt, slip)
    results = {}
    for name, kw in (
            ('base', {}),
            ('oracle', dict(exclude_centers=np.vstack([centers, moved]),
                            exclude_radius=0.5))):
        m = sm.icp_2d(ref_pts, cur_pts, init, **kw)
        err = sm.relative(gt, np.asarray(m['pose']))
        ok = (m['inlier_ratio'] >= MIN_INLIER_RATIO
              and m['n_inliers'] >= MIN_INLIERS and m['rms'] <= MAX_RMS)
        results[name] = dict(
            trans=float(math.hypot(err[0], err[1])),
            yaw=float(abs(math.degrees(err[2]))), ok=bool(ok),
            ratio=m['inlier_ratio'], rms=m['rms'])
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bag', action='append', required=True)
    p.add_argument('--start', type=float, action='append', required=True)
    p.add_argument('--end', type=float, action='append', required=True)
    p.add_argument('--people', type=int, nargs='+', default=[0, 5, 10, 20])
    p.add_argument('--moving-ratio', type=float, default=0.5)
    p.add_argument('--pairs', type=int, default=40)
    p.add_argument('--gap', type=float, nargs=2, default=[0.5, 3.0],
                   help='組にするスキャンの時刻の差 [s]')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--slip-scale', type=float, default=1.0,
                   help='初期値に足すずれ (標準偏差 0.07 m・7°) の倍率')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    pairs = []
    for bag, start, end in zip(args.bag, args.start, args.end):
        scans = read_scans(bag, start, end)
        chain = chain_poses(scans)
        if chain is None:
            print('[skip] 正解の照合が通らない', bag, start, end)
            continue
        idx = sorted(chain)
        for i in idx:
            for j in idx:
                gap = scans[j]['t'] - scans[i]['t']
                if args.gap[0] <= gap <= args.gap[1]:
                    gt = sm.relative(chain[i], chain[j])
                    if math.hypot(gt[0], gt[1]) > 0.05 or abs(gt[2]) > 0.05:
                        pairs.append((scans[i], scans[j], gt))
        print('[bag]', bag, start, end, 'scans', len(scans), 'pairs', len(pairs))
    if not pairs:
        return
    choose = rng.choice(len(pairs), size=min(args.pairs, len(pairs)),
                        replace=False)
    summary = {}
    for n in args.people:
        rows = []
        for k in choose:
            ref, cur, gt = pairs[k]
            slip = args.slip_scale * np.array([
                rng.normal(0, 0.07), rng.normal(0, 0.07),
                math.radians(rng.normal(0, 7.0))])
            rows.append(evaluate(ref, cur, gt, n, args.moving_ratio, rng,
                                 slip))
        summary[n] = {}
        for name in ('base', 'oracle'):
            tr = np.array([r[name]['trans'] for r in rows])
            yw = np.array([r[name]['yaw'] for r in rows])
            ok = np.array([r[name]['ok'] for r in rows])
            ratio = np.array([r[name]['ratio'] for r in rows])
            good = (tr < 0.03) & (yw < 1.0)
            summary[n][name] = dict(
                trans_median=float(np.median(tr)),
                trans_p90=float(np.percentile(tr, 90)),
                yaw_p90=float(np.percentile(yw, 90)),
                good=float(good.mean()), ok=float(ok.mean()),
                ok_but_bad=float((ok & ~good).mean()),
                ratio_median=float(np.median(ratio)))
            s = summary[n][name]
            print('人 {:2d} {:7s} 誤差 中央値 {:5.1f}mm p90 {:6.1f}mm 向き p90 {:5.2f}° '
                  '正確 (3cm・1° 以内) {:4.0%} 品質OK {:4.0%} OKなのに不正確 {:4.0%} '
                  '対応率 {:.2f}'.format(
                      n, name, 1e3 * s['trans_median'], 1e3 * s['trans_p90'],
                      s['yaw_p90'], s['good'], s['ok'], s['ok_but_bad'],
                      s['ratio_median']))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=1)


if __name__ == '__main__':
    main()
