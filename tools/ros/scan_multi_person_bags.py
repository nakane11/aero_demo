#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""rosbag のカラー画像を間引いて MediaPipe Tasks の PoseLandmarker (複数人) に通し、
2 人以上写っている区間を探す。

出力: ``<output-dir>/<bag名>.json`` (時刻ごとの人数) と、2 人以上のフレームの
縮小画像 ``<output-dir>/images/<bag名>_<秒>.jpg``。roscore 不要。

Usage
-----
    python3 tools/ros/scan_multi_person_bags.py \
        --bag data/bags/*.bag --output-dir ~/work/multiperson_data/scan \
        --model ~/work/multiperson_data/models/pose_landmarker_full.task
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import rosbag

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'src'))

from aero_demo.multi_person_watcher import PoseWatcher  # noqa: E402
from aero_demo.ros_camera_utils import imgmsg_to_ndarray  # noqa: E402


def scan_bag(bag_path, watcher, args, image_dir):
    name = os.path.splitext(os.path.basename(bag_path))[0]
    records = []
    last = None
    with rosbag.Bag(bag_path) as bag:
        for _topic, msg, _t in bag.read_messages(topics=[args.color_topic]):
            stamp = msg.header.stamp.to_sec()
            if last is not None and stamp - last < args.interval:
                continue
            last = stamp
            color = imgmsg_to_ndarray(msg, desired_encoding='bgr8')
            people = watcher.detect(color)
            records.append(dict(stamp=stamp, n=len(people),
                                boxes=[p['box'] for p in people]))
            if len(people) >= 2:
                small = color.copy()
                for p in people:
                    x0, y0, x1, y1 = [int(v) for v in p['box']]
                    cv2.rectangle(small, (x0, y0), (x1, y1), (0, 255, 0), 3)
                small = cv2.resize(small, None, fx=0.4, fy=0.4)
                cv2.imwrite(os.path.join(
                    image_dir, '{}_{:.1f}.jpg'.format(name, stamp)), small)
    with open(os.path.join(args.output_dir, name + '.json'), 'w') as f:
        json.dump(dict(bag=bag_path, records=records), f)
    counts = np.array([r['n'] for r in records]) if records else np.zeros(0)
    print('{}: {} 枚, 2 人以上 {} 枚, 3 人以上 {} 枚'.format(
        name, len(counts), int(np.sum(counts >= 2)), int(np.sum(counts >= 3))),
        flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--color-topic',
                        default='/camera/color/image_raw/decompressed')
    parser.add_argument('--interval', type=float, default=0.5,
                        help='間引きの間隔 [s]。')
    args = parser.parse_args()
    paths = []
    for pattern in args.bag:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    image_dir = os.path.join(args.output_dir, 'images')
    os.makedirs(image_dir, exist_ok=True)
    watcher = PoseWatcher(args.model)
    for path in paths:
        try:
            scan_bag(path, watcher, args, image_dir)
        except Exception as exc:  # noqa: BLE001  (壊れた bag は飛ばす)
            print('{}: 失敗 {!r}'.format(path, exc), flush=True)


if __name__ == '__main__':
    main()
