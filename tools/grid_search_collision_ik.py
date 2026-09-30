#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``solve_palm_ik.py`` の速度・成功率のトレードオフを調べるグリッドサーチ。

干渉ペア (``--collision-pairs``)・初期値の数 (``--attempts-per-pose``)・
バッチ IK の反復回数 (``--collision-ik-stop``)・干渉回避の重み/マージン・
台車の y の可動域・向きの候補数・後処理 IK の反復回数/収束閾値・候補数の
上限の全組み合わせについて、``solve_palm_ik.py`` をそのままサブプロセスで
実行し (本番と同じコードで測るため、1 人分のループをここに複製しない)、
結果 JSON と標準出力を集計する。

各組み合わせは同じデータで ``--repeat`` 回 (既定 2) 実行し、最後の回を
使う。干渉ペアの組数・内容や反復回数を変えると初回は jax の再コンパイルに
なるため (docs/jax_compilation_cache.md)。時間は warmup を含まない
``collision_ik_time`` (1 段目) / ``candidate_selection_time`` (2 段目)。

集計する値:

- IK 対象の人数、解けた人数 (``solved``)、後処理まで成功した人数。
- 1 段目・2 段目・IK 全体の 1 人あたりの平均と最大、最も遅い人。
- 事後検証で棄却された候補の数と、その原因の組 (``solve_palm_ik.
  pick_verified_candidate`` の ``[collision-verify]`` の行) の内訳。

データセットは ``--dataset`` に ``skeletons/``・``palms/`` を持つ
ディレクトリ (``run_pipeline_test.py`` の作業ディレクトリと同じ構成) を
1 つ以上渡す。表は全データセットの合計。

Usage
-----
    python3 tools/grid_search_collision_ik.py \\
        --dataset /tmp/data/tune \\
        --collision-pairs scripts/collision_pairs.json pairs16.json \\
        --attempts-per-pose 512 256 --output results.json
"""

import argparse
import collections
import glob
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')

_PERSON_RE = re.compile(r'^\[(\d+)/(\d+)\] (\S+) -> ')
_REJECT_RE = re.compile(r'^\s*\[collision-verify\] .*?(貫通|自己干渉) '
                        r'\((\S+) x (\S+)\)')
_WARMUP_RE = re.compile(r'^\[warmup\] .* ([0-9.]+) 秒')

# solve_palm_ik.py に渡すグリッドの軸: (引数名, 型, 既定値)。既定値 None は
# solve_palm_ik.py の既定のまま (引数を渡さない)。
AXES = (
    ('collision-pairs', str, None),
    ('attempts-per-pose', int, None),
    ('collision-ik-stop', int, None),
    ('collision-weight', float, None),
    ('collision-margin', float, None),
    ('self-collision-weight', float, None),
    ('self-collision-margin', float, None),
    ('base-y-half-range', float, None),
    ('turn-candidates', int, None),
    ('post-process-ik-stop', int, None),
    ('post-process-thre', float, None),
    ('post-process-rthre', float, None),
    ('post-process-max-candidates', int, None),
)


def pair_category(name_a, name_b):
    """棄却の原因の組を大まかな種類に分ける (表の見出し用)。人体側の名前は
    ``solve_palm_ik.human_obstacle_names`` の形 (骨は ``Neck-RShoulder``、
    手は ``R_palm``/``R_thumb`` など)。"""
    if name_b.startswith(('R_', 'L_')):
        return 'human_hand'
    if '-' in name_b:
        return 'human_body'
    return 'self'


def parse_stdout(stdout):
    """``solve_palm_ik.py`` の標準出力から、人ごとの棄却の原因の組と
    warmup の時間を取り出す。"""
    rejects_by_person = {}
    current = collections.Counter()
    warmup = 0.0
    for line in stdout.splitlines():
        match = _REJECT_RE.match(line)
        if match:
            current[(match.group(2), match.group(3))] += 1
            continue
        match = _PERSON_RE.match(line)
        if match:
            rejects_by_person[match.group(3)] = current
            current = collections.Counter()
            continue
        match = _WARMUP_RE.match(line)
        if match:
            warmup += float(match.group(1))
    return rejects_by_person, warmup


def summarize(handshake_dir, rejects_by_person):
    """結果 JSON と棄却の内訳を人ごとにまとめる。"""
    people = []
    for path in sorted(glob.glob(os.path.join(handshake_dir, '*.json'))):
        with open(path) as f:
            result = json.load(f)
        if not result.get('target', True):
            continue
        name = os.path.basename(path)
        stage1 = result.get('collision_ik_time', 0.0)
        stage2 = result.get('candidate_selection_time', 0.0)
        people.append(dict(
            name=name, robot_arm=result.get('robot_arm'),
            solved=bool(result.get('solved')),
            post=result.get('post_process') is not None,
            stage1=stage1, stage2=stage2, total=stage1 + stage2,
            rejects=dict(rejects_by_person.get(name, {}))))
    return people


def aggregate(people):
    """人ごとのまとめ (複数データセット分をつないだもの) を集計する。"""
    def stat(key, subset=None):
        values = [p[key] for p in (people if subset is None else subset)]
        if not values:
            return dict(mean=float('nan'), max=float('nan'))
        return dict(mean=float(np.mean(values)), max=float(np.max(values)))

    rejects = collections.Counter()
    for person in people:
        rejects.update(person['rejects'])
    by_category = collections.Counter()
    for (name_a, name_b), count in rejects.items():
        by_category[pair_category(name_a, name_b)] += count
    slowest = sorted(people, key=lambda p: -p['total'])[:5]
    return dict(
        n_target=len(people),
        n_solved=sum(p['solved'] for p in people),
        n_post=sum(p['post'] for p in people),
        stage1=stat('stage1'), stage2=stat('stage2'), total=stat('total'),
        stage2_by_arm={arm: stat('stage2', [p for p in people
                                            if p['robot_arm'] == arm])
                       for arm in ('l', 'r')},
        n_rejects=sum(rejects.values()),
        rejects_by_category=dict(by_category),
        rejects_top=[dict(pair=list(pair), count=count)
                     for pair, count in rejects.most_common(10)],
        slowest=[dict(name=p['name'], arm=p['robot_arm'], total=p['total'],
                      stage2=p['stage2'], post=p['post'],
                      n_rejects=sum(p['rejects'].values()))
                 for p in slowest])


def run_solve(python, dataset, output_dir, options):
    cmd = [python, os.path.join(_SCRIPTS_DIR, 'solve_palm_ik.py'),
           '--input-dir', os.path.join(dataset, 'palms'),
           '--skeleton-dir', os.path.join(dataset, 'skeletons'),
           '--output-dir', output_dir]
    for name, value in options:
        cmd += ['--' + name, str(value)]
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout = result.stdout.decode('utf-8', errors='replace')
    if result.returncode != 0:
        sys.stdout.write(stdout[-5000:])
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return stdout


def format_row(label, agg):
    cat = agg['rejects_by_category']
    return ('{:<48} {:>4}/{:<4} {:>4} | 1段目 {:.3f}/{:.2f} | 2段目 {:.3f}/'
            '{:.2f} (l {:.3f} r {:.3f}) | 計 {:.3f}/{:.2f} | 棄却 {} (自己 {} '
            '体 {} 手 {})'.format(
                label, agg['n_post'], agg['n_target'], agg['n_solved'],
                agg['stage1']['mean'], agg['stage1']['max'],
                agg['stage2']['mean'], agg['stage2']['max'],
                agg['stage2_by_arm']['l']['mean'],
                agg['stage2_by_arm']['r']['mean'],
                agg['total']['mean'], agg['total']['max'], agg['n_rejects'],
                cat.get('self', 0), cat.get('human_body', 0),
                cat.get('human_hand', 0)))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--dataset', type=str, nargs='+', required=True,
        help='skeletons/ と palms/ を持つディレクトリ (複数可、表は合計)。')
    for name, type_, default in AXES:
        parser.add_argument(
            '--' + name, type=type_, nargs='+', default=[default],
            help='solve_palm_ik.py の --{} に渡す値 (複数指定で振る。'
                 '省略時は solve_palm_ik.py の既定)。'.format(name))
    parser.add_argument(
        '--repeat', type=int, default=2,
        help='同じ組み合わせを実行する回数 (最後の回を使う。既定 2)。')
    parser.add_argument(
        '--work-dir', type=str, default=None,
        help='solve_palm_ik.py の出力先の親ディレクトリ (既定は /tmp 以下に '
             '作る)。組み合わせごとの結果 JSON と標準出力を残す。')
    parser.add_argument(
        '--output', type=str, default=None,
        help='集計結果を追記する JSON Lines ファイル (既定は保存しない)。')
    parser.add_argument('--python', type=str, default=sys.executable)
    args = parser.parse_args()

    work_dir = args.work_dir or tempfile.mkdtemp(
        prefix='aero_demo_grid_search_', dir='/tmp')
    os.makedirs(work_dir, exist_ok=True)
    print('作業ディレクトリ: {}'.format(work_dir))

    names = [name for name, _, _ in AXES]
    values = [getattr(args, name.replace('-', '_')) for name in names]
    combos = list(itertools.product(*values))
    print('[grid] {} 通り x {} データセット x {} 回'.format(
        len(combos), len(args.dataset), args.repeat))

    rows = []
    for index, combo in enumerate(combos):
        options = [(name, value) for name, value in zip(names, combo)
                   if value is not None]
        label = ' '.join(
            '{}={}'.format(name, os.path.basename(str(value)))
            for name, value in options) or 'default'
        people = []
        warmup = 0.0
        for dataset in args.dataset:
            out_dir = os.path.join(work_dir, '{:03d}'.format(index),
                                   os.path.basename(dataset.rstrip('/')))
            for _ in range(args.repeat):
                stdout = run_solve(args.python, dataset, out_dir, options)
            with open(out_dir + '.log', 'w') as f:
                f.write(stdout)
            rejects_by_person, dataset_warmup = parse_stdout(stdout)
            warmup += dataset_warmup
            people += summarize(out_dir, rejects_by_person)
        agg = aggregate(people)
        row = dict(index=index, label=label, options=dict(options),
                   warmup=warmup, **agg)
        rows.append(row)
        print(format_row(label, agg), flush=True)
        if args.output:
            with open(args.output, 'a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')

    print('\n=== 後処理まで成功した人数の多い順 → 全体の平均時間の短い順 ===')
    for row in sorted(rows, key=lambda r: (-r['n_post'],
                                           r['total']['mean'])):
        print(format_row(row['label'], row))


if __name__ == '__main__':
    main()
