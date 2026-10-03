#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``tools/handshake_reachability_map.py`` の結果 (JSON Lines) から、手の出し方の
助言 (``aero_demo.hand_offer_advice``) が使う ``config/hand_offer_reachability.json`` を作る.

格子点ごとに ``[前方, 外側, 高さ, yaw, pitch, roll, ok]`` を手 (R/L) 別に並べる。
``ok`` は IK が後処理まで解け、かつ ``--offer-distance`` [m] で差し出しと判定されること。
人の腕が届かない格子点は含めない。

Usage
-----
    python3 tools/build_hand_offer_reachability.py \\
        /tmp/handshake_reachability/position*.jsonl \\
        /tmp/handshake_reachability/orientation.jsonl
"""

import argparse
import json
import os

_KEYS = ('forward', 'lateral', 'height', 'yaw', 'pitch', 'roll')
_DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'config',
    'hand_offer_reachability.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('inputs', nargs='+')
    parser.add_argument('--output', default=_DEFAULT_OUTPUT)
    parser.add_argument('--offer-distance', default='1.50',
                        help='差し出し判定を見る距離 (map の --person-distances のどれか)。')
    args = parser.parse_args()

    points = {'R': {}, 'L': {}}
    for path in args.inputs:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not rec.get('human_reachable') or 'ik' not in rec:
                    continue
                hand = rec['hand']
                offer = rec['offer'][args.offer_distance]
                ok = rec['ik']['status'] == 'ok' and offer['side'] == hand
                key = tuple(round(float(rec['point'][k]), 3) for k in _KEYS)
                points[hand][key] = int(ok)
    out = dict(
        description='tools/handshake_reachability_map.py の格子点 '
                    '[forward, lateral, height, yaw, pitch, roll, ok]',
        sources=[os.path.basename(p) for p in args.inputs],
        offer_distance=float(args.offer_distance),
        points={hand: [list(k) + [v] for k, v in sorted(rows.items())]
                for hand, rows in points.items()})
    with open(args.output, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
        f.write('\n')
    for hand, rows in points.items():
        print('{}: {} 点 (ok {})'.format(hand, len(rows), sum(rows.values())))
    print('-> {}'.format(args.output))


if __name__ == '__main__':
    main()
