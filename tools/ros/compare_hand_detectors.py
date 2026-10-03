#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""同じフレームに Holistic と Hands をかけ、掌の位置・向きの差 (検出器の差) を測る。

首を計画時と同じ下向きにした後は台車・首とも動かさない。人は肩〜手が
映る距離 (0.8〜1.5m) で手を差し出して静止する (Holistic は体が映らないと
手を返さない)。結果は ``--out-dir`` (既定 ``data/palm_detector_compare/<日時>/``)
に frames.jsonl / summary.json / overlay_*.png (Holistic=赤、Hands=緑)。

Usage
-----
    python3 tools/ros/compare_hand_detectors.py --side R
    python3 tools/ros/compare_hand_detectors.py --side L --frames 100
"""

import argparse
import datetime
import json
import os
import time

import numpy as np

import rospy

import palm_check_common as common

# 計画時の首の下げ角 [deg]。
PLANNING_HEAD_PITCH_DEG = 25.0


def look_down(args):
    """首を計画時と同じ下向きにする (他の関節は現在値のまま)。"""
    from aero_demo.aero_urdf_setup import load_aero
    from skrobot.interfaces.ros import AeroROSRobotInterface

    print('[compare] 首を下向き (neck_p={:.0f}deg) にします...'.format(
        args.head_pitch_deg))
    robot = load_aero(use_hand=True)
    ri = AeroROSRobotInterface(robot)
    robot.angle_vector(ri.angle_vector())
    robot.neck_y_joint.joint_angle(0.0)
    robot.neck_r_joint.joint_angle(0.0)
    robot.neck_p_joint.joint_angle(np.deg2rad(args.head_pitch_deg))
    ri.angle_vector(robot.angle_vector(), args.head_move_time)
    ri.wait_interpolation()
    rospy.sleep(0.5)
    robot.angle_vector(ri.angle_vector())
    print('[compare] 首の実測: neck_y={:+.1f} neck_p={:+.1f} neck_r={:+.1f}deg'
          .format(*[np.rad2deg(getattr(robot, name).joint_angle())
                    for name in ('neck_y_joint', 'neck_p_joint',
                                 'neck_r_joint')]))


def run(args):
    if not args.no_head_move:
        look_down(args)
    source = common.FrameSource(args)
    detector = common.PalmDetector(args)
    os.makedirs(args.out_dir, exist_ok=True)
    frames_path = os.path.join(args.out_dir, 'frames.jsonl')
    print('[compare] {}手を {} フレーム分比較します (結果: {})。'.format(
        args.side, args.frames, args.out_dir))
    print('[compare] 体 (肩〜手) が映る距離で手を差し出して静止してください。')

    records = []
    n_seen = n_holistic = n_hands = 0
    last = time.time()
    deadline = time.time() + args.timeout
    with open(frames_path, 'w') as f:
        while (not rospy.is_shutdown() and len(records) < args.frames
               and time.time() < deadline):
            frame = source.get_frame(after=last)
            if frame is None:
                continue
            last = frame['received']
            n_seen += 1
            det = detector.detect(frame, args.side)
            holistic = det['holistic']
            if holistic is not None:
                n_holistic += 1
            if any(h['palm'] is not None for h in det['hands']):
                n_hands += 1
            if holistic is None:
                continue
            ref = np.asarray(holistic['position'])
            hand = common.pick_hand(det['hands'], ref, args.max_pair_distance)
            if hand is None:
                continue
            hands_palm = hand['palm']
            cam = det['camera_position']
            rec = dict(
                stamp=frame['stamp'],
                camera_position=[float(v) for v in cam],
                camera_distance=float(np.linalg.norm(ref - cam)),
                hands_label=hand['label'], hands_score=hand['score'],
                hands_n_points=hand['n_points'])
            for name in common.SOURCES:
                palm = hands_palm if name == 'hands' else det[name]
                rec[name] = common.palm_record(palm)
            for base in ('holistic', 'holistic_smoothed'):
                if det[base] is None:
                    continue
                diff = (np.asarray(hands_palm['position'])
                        - np.asarray(det[base]['position']))
                rec['diff_vs_' + base] = dict(
                    xyz=[float(v) for v in diff],
                    norm=float(np.linalg.norm(diff)),
                    rotation_deg=common.rotation_angle_deg(
                        det[base]['rot'], hands_palm['rot']),
                    normal_deg=common.normal_angle_deg(
                        det[base], hands_palm),
                    **common.decompose(diff, cam, det[base]['position']))
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            f.flush()
            if (args.save_image_every > 0
                    and (len(records) - 1) % args.save_image_every == 0):
                common.draw_overlay(
                    frame['color'], det, hand, os.path.join(
                        args.out_dir,
                        'overlay_{:04d}.png'.format(len(records) - 1)))
            d = rec['diff_vs_holistic']
            print('[compare] {:3d}/{} 距離 {:.2f}m  Hands-Holistic = {} '
                  '(視線方向 {:+.1f}mm, 鉛直 {:+.1f}mm, 向き {:.1f}deg)'.format(
                      len(records), args.frames, rec['camera_distance'],
                      common.fmt_xyz_mm(d['xyz']), d['along_ray'] * 1e3,
                      d['vertical'] * 1e3, d['rotation_deg']))

    summary = summarize(records, n_seen, n_holistic, n_hands, args)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print_summary(summary)


def summarize(records, n_seen, n_holistic, n_hands, args):
    summary = dict(
        created=datetime.datetime.now().isoformat(),
        args=vars(args), n_frames_seen=n_seen,
        n_frames_holistic=n_holistic, n_frames_hands=n_hands,
        n_frames_paired=len(records))
    if not records:
        return summary
    summary['camera_distance'] = common.stats(
        [r['camera_distance'] for r in records])
    for base in ('holistic', 'holistic_smoothed'):
        key = 'diff_vs_' + base
        rows = [r[key] for r in records if key in r]
        summary[key] = dict(
            x=common.stats([d['xyz'][0] for d in rows]),
            y=common.stats([d['xyz'][1] for d in rows]),
            z=common.stats([d['xyz'][2] for d in rows]),
            norm=common.stats([d['norm'] for d in rows]),
            along_ray=common.stats([d['along_ray'] for d in rows]),
            across_ray=common.stats([d['across_ray'] for d in rows]),
            horizontal=common.stats([d['horizontal'] for d in rows]),
            rotation_deg=common.stats([d['rotation_deg'] for d in rows]),
            normal_deg=common.stats([d['normal_deg'] for d in rows]))
    jitter = {}
    for name in common.SOURCES:
        pos = np.array([r[name]['position'] for r in records
                        if r.get(name) is not None])
        if len(pos):
            jitter[name] = [float(v) for v in pos.std(axis=0)]
    summary['position_std_xyz'] = jitter
    return summary


def print_summary(summary):
    print('\n=== 検証 A: 検出器の差 (Hands - Holistic、同じフレーム) ===')
    print('フレーム: 受信 {} / Holistic で掌あり {} / Hands で掌あり {} / '
          '比較できた {}'.format(
              summary['n_frames_seen'], summary['n_frames_holistic'],
              summary['n_frames_hands'], summary['n_frames_paired']))
    if not summary['n_frames_paired']:
        print('比較できたフレームがありません (体が映っているか、'
              '--side が差し出した手と合っているかを確認してください)。')
        return
    dist = summary['camera_distance']
    print('カメラから掌までの距離: {:.2f}m (±{:.2f})'.format(
        dist['mean'], dist['std']))
    labels = (('x', 'x (base_link)'), ('y', 'y (base_link)'),
              ('z', 'z = 鉛直'), ('along_ray', '視線方向 (+=奥)'),
              ('across_ray', '視線と直交'), ('horizontal', '水平の大きさ'),
              ('norm', '距離'))
    for base, title in (('holistic', 'Holistic (生)'),
                        ('holistic_smoothed', 'Holistic + One Euro (計画時と同じ)')):
        s = summary['diff_vs_' + base]
        print('\n-- Hands - {} [mm] --'.format(title))
        for key, label in labels:
            print('  {:<16} {}'.format(label, common.fmt_stats_mm(s[key])))
        for key, label in (('rotation_deg', '姿勢の差'),
                           ('normal_deg', '法線の差')):
            r = s[key]
            print('  {:<16} {:6.1f} ±{:5.1f} deg (最大 {:.1f})'.format(
                label, r['mean'], r['std'], r['max_abs']))
    print('\n-- 各検出器の時間方向のばらつき (標準偏差 x, y, z) --')
    for name, std in summary['position_std_xyz'].items():
        print('  {:<18} {}'.format(name, common.fmt_xyz_mm(std)))
    print('\n見方: 平均が標準偏差より十分大きい成分は検出器の系統的な差。'
          '視線方向の成分が大きければ深度の拾い方 (手の見え方) の差、'
          '視線と直交する成分が大きければ 2D ランドマーク位置の差。')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_camera_args(parser)
    parser.add_argument('--side', choices=('R', 'L'), required=True,
                        help='比較する手 (人物自身の左右)。')
    parser.add_argument('--frames', type=int, default=60,
                        help='集める比較フレーム数。')
    parser.add_argument('--timeout', type=float, default=120.0,
                        help='集計までの最大待ち時間 [s]。')
    parser.add_argument('--max-pair-distance', type=float, default=0.15,
                        help='Holistic の掌とこの距離 [m] 以内の Hands の手を同じ手とみなす。')
    parser.add_argument('--save-image-every', type=int, default=10,
                        help='重ね描き画像を保存する間隔 (0 で保存しない)。')
    parser.add_argument('--head-pitch-deg', type=float,
                        default=PLANNING_HEAD_PITCH_DEG,
                        help='最初に首を下げる neck_p [deg]。')
    parser.add_argument('--head-move-time', type=float, default=5.0,
                        help='首を動かす時間 [s]。')
    parser.add_argument('--no-head-move', action='store_true',
                        help='首を動かさない (実機に接続しない)。')
    parser.add_argument('--out-dir', type=str, default=None)
    args, _ = parser.parse_known_args(rospy.myargv()[1:])
    if args.out_dir is None:
        args.out_dir = os.path.join(
            common.REPO_ROOT, 'data', 'palm_detector_compare',
            datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    rospy.init_node('compare_hand_detectors')
    run(args)


if __name__ == '__main__':
    main()
