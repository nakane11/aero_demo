#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人だかりの素材 (``make_crowd_bag.py``) を選ぶため、bag の各時刻の人の
差し出しスコアと見え方を出す.

スコアは見張りの ``offer_candidate_score`` と同じ式で、ロボットの向きは
最悪の場合 (手が肩から水平にどの向きへ出ていても「ロボットの方」とみなす)
にする。合成ではロボットとの向きが素材と変わるため。深度が抜けた関節は
体幹の深度に置いて数える (見張りより厳しめ)。

出力 (1 行 = 1 フレーム、JSON lines): t、score (最悪の向き)、reach・height
(左右)、体幹までの距離、足首が写っているか。

使い方::

    source /opt/ros/noetic/setup.bash
    python3 tools/screen_crowd_sources.py BAG OUT.jsonl
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))
sys.path.insert(0, _HERE)

import cv2  # noqa: E402

import multi_person_sources as mps  # noqa: E402
from make_crowd_bag import Segmenter, _FrameArgs  # noqa: E402
from aero_demo.multi_person_watcher import TORSO_JOINTS  # noqa: E402


def _ramp(x, a, b):
    return float(np.clip((x - a) / (b - a), 0.0, 1.0))


def worst_case_score(points):
    """``offer_candidate_score`` を、ロボットの向きを手の出ている向きにして計算."""
    hip = [points[n] for n in ('LHip', 'RHip') if n in points]
    shoulders = [points[n] for n in ('LShoulder', 'RShoulder') if n in points]
    if not hip and not shoulders:
        return 0.0, {}
    hip_c = (np.mean(hip, axis=0) if hip
             else np.mean(shoulders, axis=0) - np.array([0.0, 0.0, 0.5]))
    best = 0.0
    details = {}
    for side in ('R', 'L'):
        shoulder = points.get(side + 'Shoulder')
        hand_pts = [points[side + n] for n in ('Wrist', 'Index', 'Pinky')
                    if side + n in points]
        if shoulder is None or not hand_pts:
            continue
        hand = np.mean(hand_pts, axis=0)
        offset = hand - shoulder
        reach = float(np.hypot(offset[0], offset[1]))
        height = float(hand[2] - hip_c[2])
        score = 0.6 * _ramp(reach, 0.10, 0.30) + 0.4 * _ramp(height, -0.10,
                                                             0.10)
        details[side] = dict(reach=round(reach, 3), height=round(height, 3),
                             score=round(score, 3))
        best = max(best, score)
    return best, details


def joints_3d(pixels, depth, intrinsics, base_from_cam, min_visibility=0.5):
    """関節を base_link 系の 3 次元に. 深度が無い関節は体幹の深度に置く."""
    h, w = depth.shape

    def depth_at(u, v):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < h and 0 <= ui < w):
            return None
        patch = depth[max(0, vi - 2):vi + 3, max(0, ui - 2):ui + 3]
        valid = patch[np.isfinite(patch) & (patch > 0.2)]
        return float(np.median(valid)) if valid.size else None

    torso_z = [depth_at(*pixels[n][:2]) for n in TORSO_JOINTS
               if pixels[n][2] > min_visibility]
    torso_z = [z for z in torso_z if z is not None]
    if not torso_z:
        return None, None
    body_z = float(np.median(torso_z))
    out = {}
    for name, (u, v, vis) in pixels.items():
        if vis <= min_visibility:
            continue
        z = depth_at(u, v)
        if z is None or abs(z - body_z) > 0.5:
            z = body_z
        p = np.array([(u - intrinsics.cx) * z / intrinsics.fx,
                      (v - intrinsics.cy) * z / intrinsics.fy, z, 1.0])
        out[name] = base_from_cam.dot(p)[:3]
    return out, body_z


def main():
    bag_path, out_path = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    seg = Segmenter(os.path.expanduser(
        '~/.mediapipe/models/pose_landmarker_lite.task'))
    t0 = None
    with open(out_path, 'w') as f:
        for fr in mps.bag_frames(bag_path, 0.0, None, _FrameArgs(step)):
            if t0 is None:
                import rosbag
                with rosbag.Bag(bag_path) as b:
                    t0 = b.get_start_time()
            color, depth = fr['color'], fr['depth']
            h, w = depth.shape
            found = seg.detect(cv2.resize(color, (w // 2, h // 2)))
            row = dict(t=round(fr['t'] - t0, 2), person=False)
            if found is not None:
                pixels = {k: [2.0 * u, 2.0 * v, vis]
                          for k, (u, v, vis) in found[0].items()}
                pts, body_z = joints_3d(pixels, depth, fr['intrinsics'],
                                        fr['camera_to_base'])
                if pts is not None:
                    score, details = worst_case_score(pts)
                    feet = all(pixels[n][2] > 0.5 and pixels[n][1] < h - 5
                               for n in ('LAnkle', 'RAnkle'))
                    shoulder_h = max([pts[n][2] for n in
                                      ('LShoulder', 'RShoulder') if n in pts]
                                     or [0.0])
                    row.update(person=True, score=round(score, 3),
                               details=details, depth=round(body_z, 2),
                               feet=bool(feet),
                               shoulder_h=round(float(shoulder_h), 2))
            f.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
