#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""閉ループの合成カメラ (tools/ros/crowd_sim_camera.py) の場面を作る.

場面の JSON の人ごとに、実機の bag の区間から人を切り抜き
(``make_crowd_bag.load_or_extract``、切り抜きはキャッシュを使う)、置く位置と
いっしょに pickle にまとめる。人は区間 (clip) を時刻で切り替えられる
(例: 手を下ろして立つ → 場面の 3 秒から手を差し出して止める)。

場面の JSON::

    {"people": [
      {"name": "B", "place": [0.7, -1.2], "yaw_deg": 0.0,
       "clips": [
         {"bag": "...bag", "start": 0.0, "end": 1.6, "from": 0.0,
          "mode": "pingpong", "max_source_score": 0.3},
         {"bag": "...bag", "start": 1.6, "end": 4.6, "from": 3.0,
          "mode": "once_then_pingpong", "hold": 1.0,
          "max_source_score": 0.0}]}]}

- ``place``: ロボットの最初の base_link 系 (= 合成の odom 系) [m]。
- 向きは、素材で人からカメラへ向かう向きが、置いた位置からロボットの頭へ
  向かう向きになるように回す (``yaw_deg`` を足す)。
- clip の ``from``: 場面の時刻 (ARMED からの秒) でこの区間に切り替える。
- ``mode``: ``pingpong`` は往復で繰り返す。``once_then_pingpong`` は一度
  流してから、最後の ``hold`` 秒を往復で繰り返す (差し出して止めた姿勢)。
- ``max_source_score``: 見張りの差し出しスコア (向きは最悪の場合) がこれ以上
  の素材のフレームを使わない (0 以下で使う)。

使い方 (ROS の python3。mediapipe が要る)::

    source /opt/ros/noetic/setup.bash
    python3 tools/crowd_sim_prepare.py scene.json scene.pkl
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))
sys.path.insert(0, _HERE)

import make_crowd_bag as mcb  # noqa: E402


class _ExtractArgs(object):
    """``make_crowd_bag`` の既定値 (キャッシュの鍵を合わせる)."""
    stride = 2
    source_frame_step = 2
    depth_margin = 0.35
    floor_cut = 0.05
    min_points = 3000
    body_radius = 0.7
    seg_threshold = 0.5
    max_source_score = 0.3
    cache_dir = os.path.expanduser('~/work/crowd_data/cache')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('scene')
    parser.add_argument('out')
    parser.add_argument('--model', default=os.path.expanduser(
        '~/.mediapipe/models/pose_landmarker_lite.task'))
    args = parser.parse_args()
    with open(args.scene) as f:
        scene = json.load(f)
    segmenter = None
    people = []
    for person in scene['people']:
        clips = []
        for clip in person['clips']:
            ex = _ExtractArgs()
            ex.max_source_score = float(clip.get('max_source_score', 0.3))
            spec = dict(name=person['name'], bag=clip['bag'],
                        start=clip['start'], end=clip['end'])
            try:
                frames = mcb.load_or_extract(spec, segmenter, ex)
            except AttributeError:
                # キャッシュが無く、切り抜くのに PoseLandmarker が要る。
                segmenter = mcb.Segmenter(args.model)
                frames = mcb.load_or_extract(spec, segmenter, ex)
            if not frames:
                sys.exit('{} の {} {:.1f}〜{:.1f} 秒から人が取れません'.format(
                    person['name'], clip['bag'], clip['start'], clip['end']))
            mcb.describe('{}[{:.1f}-{:.1f}]'.format(
                person['name'], clip['start'], clip['end']), frames)
            print('  スコア {:.2f}〜{:.2f}'.format(
                min(f['score'] for f in frames),
                max(f['score'] for f in frames)))
            clips.append(dict(
                frames=frames, start_t=float(clip.get('from', 0.0)),
                mode=clip.get('mode', 'pingpong'),
                hold=float(clip.get('hold', 1.0))))
        people.append(dict(name=person['name'],
                           place=np.asarray(person['place'], np.float64),
                           yaw_deg=float(person.get('yaw_deg', 0.0)),
                           clips=clips))
    with open(args.out, 'wb') as f:
        pickle.dump(dict(people=people, room=scene.get('room')), f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print('[done] {} ({} 人)'.format(args.out, len(people)))


if __name__ == '__main__':
    main()
