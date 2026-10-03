#!/usr/bin/env python3
"""``run_pipeline_test.py`` の作業ディレクトリ (複数可) を読み、IK が決めた
最終の台車位置と人の位置関係 (方位・前方ずれ・向きのずれ)・IK の計算時間を
集計する。指標の定義は docs/handshake_base_placement.md。

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
            # 結果 JSON と同じ、人物を平行移動した座標系で比べる。
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
