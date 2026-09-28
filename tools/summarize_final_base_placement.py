#!/usr/bin/env python3
"""``run_pipeline_test.py`` の作業ディレクトリ (skeletons/ と handshakes/)
を読み、IK が決めた最終の台車位置と人の位置関係・IK の計算時間を集計する
(docs/handshake_base_placement.md の比較に使った開発用ツール)。

複数の作業ディレクトリを渡すと、まとめて 1 つの集計にする (例えば
``--seed 0`` と ``--seed 1`` の 2 回分)。

集計する指標:

- 方位: ロボットの台車 (base_link) の中心から人の立ち位置 (左右の腰の
  中点) を見た方向を、ロボットの正面を 0 度、左回りを正として測った角度。
  ±90 度が真横で、|方位| が 90 度より大きいほど人がロボットの斜め後ろに
  いる。
- 前方ずれ: 台車の位置が人の立ち位置から人の正面方向にどれだけ前 [m]
  にあるか (後ろなら負)。
- 向きのずれ: 台車の向きの人の正面方向からのずれ [度]。人のいる側へ
  回っている (人の方を向いている) 向きを正にする。

指標の計算は ``solve_palm_ik.base_placement_metrics`` (実ノードの
デバッグログと共通)。

usage::

    python3 tools/summarize_final_base_placement.py /tmp/aero_demo_pipeline_xxx [...]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
for _path in (_PKG_SRC_DIR, _SCRIPTS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import solve_palm_ik as spik  # noqa: E402  (パス追加後に import)


def load_rows(work_dir):
    """作業ディレクトリ 1 つ分の IK 対象者について、1 人 1 行の dict を返す。"""
    rows = []
    for path in sorted(glob.glob(os.path.join(work_dir, 'handshakes', '*.json'))):
        with open(path) as f:
            data = json.load(f)
        if not data.get('target'):
            continue
        name = os.path.splitext(os.path.basename(path))[0]
        row = dict(
            name=name, solved=bool(data.get('solved')),
            post=data.get('post_process') is not None,
            margin=data.get('base_x_standing_margin'),
            time=(data['collision_ik_time']
                  + data['candidate_selection_time']))
        if row['solved']:
            # solve_palm_ik.py と同じく人物を (HUMAN_FRONT_DISTANCE, 0) へ
            # 平行移動した座標系で比べる (結果 JSON はこの座標系)。
            joints = spik.load_skeleton_json(
                os.path.join(work_dir, 'skeletons', name + '.json'))
            joints = spik.translate_joint_positions(
                joints, spik.human_translation_offset(joints))
            metrics = spik.base_placement_metrics(
                joints, data['base_position'], data['base_yaw'])
            facing = spik.human_facing_direction(joints)
            row.update(
                human_yaw=math.degrees(math.atan2(facing[1], facing[0])),
                bearing=metrics['bearing_deg'],
                yaw_toward=metrics['yaw_offset_deg'],
                front=metrics['front_offset'],
                dist=float(np.linalg.norm(
                    spik.human_standing_xy(joints)
                    - np.asarray(data['base_position'][:2]))))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('work_dirs', nargs='+',
                        help='run_pipeline_test.py の作業ディレクトリ。')
    parser.add_argument('--verbose', action='store_true',
                        help='1 人ずつの値も表示する。')
    args = parser.parse_args()

    rows = []
    for work_dir in args.work_dirs:
        rows += load_rows(work_dir)
    if not rows:
        print('IK 対象者がいません。')
        return
    if args.verbose:
        for r in rows:
            if r['solved']:
                print('{name}: m={margin} post={post} human_yaw={human_yaw:+6.1f} '
                      'bearing={bearing:+7.1f} yaw_toward={yaw_toward:+6.1f} '
                      'front={front:+.3f} '
                      'dist={dist:.3f} time={time:.3f}'.format(**r))
            else:
                print('{name}: NOT solved time={time:.3f}'.format(**r))

    n = len(rows)
    solved = [r for r in rows if r['solved']]
    n_post = sum(r['post'] for r in rows)
    print('IK 対象 {} 人: solved {} ({:.1%})、後処理まで成功 {} ({:.1%})'
          .format(n, len(solved), len(solved) / n, n_post, n_post / n))
    margins = {}
    for r in solved:
        margins[r['margin']] = margins.get(r['margin'], 0) + 1
    print('解けた x の窓 (base_x_standing_margin、負は無制限): {}'.format(
        margins))
    if solved:
        bearing = np.abs([r['bearing'] for r in solved])
        front = np.array([r['front'] for r in solved])
        print('|方位| [度]: 中央値 {:.1f}、平均 {:.1f}、最大 {:.1f}、'
              '110 度以下 {}/{}'.format(
                  np.median(bearing), bearing.mean(), bearing.max(),
                  int((bearing <= 110.0).sum()), len(bearing)))
        print('前方ずれ [m]: |平均| {:.3f}、平均 {:+.3f}、最大 {:+.3f}'.format(
            np.abs(front).mean(), front.mean(), front.max()))
        toward = np.array([r['yaw_toward'] for r in solved])
        print('向きのずれ (人の側へ回った向きが正) [度]: 平均 {:+.1f}、'
              '中央値 {:+.1f}、最大 {:+.1f}、|ずれ|≤10 度 {}/{}'.format(
                  toward.mean(), np.median(toward), toward.max(),
                  int((np.abs(toward) <= 10.0).sum()), len(toward)))
    times = sorted(rows, key=lambda r: r['time'], reverse=True)
    print('IK 計算時間 (1段階目+2段階目) [秒/人]: 平均 {:.3f}、最大 {:.3f} ({})'
          .format(np.mean([r['time'] for r in rows]), times[0]['time'],
                  times[0]['name']))


if __name__ == '__main__':
    main()
