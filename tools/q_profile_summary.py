#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""scripts/qprof.py の計測 (MFIX_QPROF のディレクトリの JSON) を表にする。

develop の ``tools/q_profile_summary.py`` (3352223・d4c9f88・7bdf134) の legacy 版。分類の export の鍵を
人の腕 (``human``) に替えた。

    python3 tools/q_profile_summary.py <prof のディレクトリ> [--depth 3] [--episodes] [--calls] [--fns] [--md]
        [--warmup all|only|exclude]

表 1 (段階ごと): 段階 (入れ子を ``--depth`` で切る) ごとに、solve 本体・コンパイル/トレース・FK・事後検証・人の腕・
Python の秒の合計と solve の回数。時間の長い順。
表 2 (1 回ごと、``--episodes``): 包んだ一番外の関数 (IK・traj・transition・refine_transition・warmup など) の
1 回の時間の平均・p90・最大と、平均の内訳。
表 3 (``--calls``): solve 1 回ごとの時間 (段階・関数・候補数ごとの回数・平均・最大・中のコンパイル)。
表 4 (``--fns``): 包んだ関数ごとの排他的な秒と回数。
"""

import argparse
import collections
import glob
import json
import os

import numpy as np

CATS = [('solve', 'solve 本体'), ('compile', 'コンパイル・トレース'), ('fk', 'FK'),
        ('verify', '事後検証'), ('human', '人の腕'), ('py', 'Python')]


def load(path):
    files = sorted(glob.glob(os.path.join(path, '*.json'))) if os.path.isdir(path) else [path]
    return [(os.path.basename(f), json.load(open(f))) for f in files]


def script_of(fname):
    return fname.rsplit('_', 1)[0]


def cut(stage, depth):
    parts = stage.split('/')
    return '/'.join(parts[:depth])


def stage_table(procs, depth, select=None):
    rows = collections.defaultdict(lambda: collections.defaultdict(float))
    counts = collections.Counter()
    for fname, data in procs:
        script = script_of(fname)
        for th in data['threads']:
            tag = script if th['thread'] == 'MainThread' else '{}[{}]'.format(script, th['thread'][:20])
            for stage, cat, sec in th['acc']:
                if select and not select(stage):
                    continue
                rows[(tag, cut(stage, depth))][cat] += sec
            for c in th['calls']:
                stage = c[0]
                if select and not select(stage):
                    continue
                counts[(tag, cut(stage, depth))] += 1
    if select is None:
        for fname, data in procs:
            # メインスレッドの段階の外 (人のループ・読み書き・import など) = 計測の開始からの壁時計 - 段階の合計。
            script = script_of(fname)
            main = [th for th in data['threads'] if th['thread'] == 'MainThread']
            if data.get('wall') and main:
                inside = sum(sec for _, _, sec in main[0]['acc'])
                rows[(script, '(段階の外)')]['py'] += max(0.0, data['wall'] - inside)
    out = []
    for key, cats in rows.items():
        total = sum(v for c, v in cats.items())
        out.append((total, key, cats, counts[key]))
    out.sort(key=lambda r: -r[0])
    return out


def pct(values, q):
    return float(np.percentile(values, q)) if values else float('nan')


def fmt_cats(cats, scale=1.0):
    return [cats.get(c, 0.0) * scale for c, _ in CATS]


def _print_row(line, md):
    print(('| ' + ' | '.join(line) + ' |') if md else '\t'.join(line))


def _print_head(head, md):
    if md:
        print('| ' + ' | '.join(head) + ' |')
        print('|' + '---|' * len(head))
    else:
        print('\t'.join(head))


def print_stage_table(rows, md, limit):
    _print_head(['プロセス', '段階', '合計 [秒]'] + [label for _, label in CATS] + ['solve 回数'], md)
    for total, (tag, stage), cats, n in rows[:limit]:
        vals = ['{:.1f}'.format(v) for v in fmt_cats(cats)]
        _print_row([tag, stage, '{:.1f}'.format(total)] + vals + [str(n)], md)


def episode_table(procs, md):
    eps = collections.defaultdict(list)
    for fname, data in procs:
        script = script_of(fname)
        for th in data['threads']:
            for e in th['episodes']:
                eps[(script, e['name'])].append(e)
    _print_head(['プロセス', '1 回の単位', '回数', '平均 [秒]', 'p90', '最大', '合計']
                + ['平均の内訳: ' + CATS[0][1]] + [label for _, label in CATS[1:]] + ['solve 回数/回'], md)
    rows = []
    for (script, name), lst in eps.items():
        walls = [e['wall'] for e in lst]
        cats = collections.defaultdict(float)
        nsol = 0
        for e in lst:
            for k, v in e['delta'].items():
                cats[k.rsplit('|', 1)[1]] += v
            nsol += len(e['solves'])
        n = len(lst)
        rows.append((sum(walls), script, name, n, walls, cats, nsol))
    rows.sort(key=lambda r: -r[0])
    for total, script, name, n, walls, cats, nsol in rows:
        vals = ['{:.3f}'.format(v) for v in fmt_cats(cats, 1.0 / n)]
        _print_row([script, name, str(n), '{:.3f}'.format(np.mean(walls)), '{:.3f}'.format(pct(walls, 90)),
                    '{:.3f}'.format(max(walls)), '{:.1f}'.format(total)] + vals + ['{:.1f}'.format(nsol / n)], md)


def calls_table(procs, md, depth):
    rows = collections.defaultdict(list)
    for fname, data in procs:
        script = script_of(fname)
        for th in data['threads']:
            for stage, name, n, sec, comp in th['calls']:
                rows[(script, cut(stage, depth), name, n)].append((sec, comp))
    _print_head(['プロセス', '段階', '関数', '候補数', '回数', '合計 [秒]', '平均', '最大', '中のコンパイル'], md)
    out = sorted(rows.items(), key=lambda kv: -sum(s for s, _ in kv[1]))
    for (script, stage, name, n), lst in out:
        secs = [s for s, _ in lst]
        _print_row([script, stage, name, str(n), str(len(lst)), '{:.1f}'.format(sum(secs)),
                    '{:.3f}'.format(np.mean(secs)), '{:.3f}'.format(max(secs)),
                    '{:.1f}'.format(sum(c for _, c in lst))], md)


def fn_table(procs, md, depth, limit, select=None):
    rows = collections.defaultdict(float)
    counts = collections.Counter()
    for fname, data in procs:
        script = script_of(fname)
        for th in data['threads']:
            for stage, cat, fn, sec in th.get('fn_acc', []):
                if select and not select(stage):
                    continue
                rows[(script, cut(stage, depth), cat, fn)] += sec
            for stage, fn, n in th['count']:
                if select and not select(stage):
                    continue
                counts[(script, cut(stage, depth), fn)] += n
    _print_head(['プロセス', '段階', '分類', '関数', '秒 (排他的)', '回数'], md)
    for (script, stage, cat, fn), sec in sorted(rows.items(), key=lambda kv: -kv[1])[:limit]:
        _print_row([script, stage, cat, fn, '{:.2f}'.format(sec), str(counts[(script, stage, fn)])], md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--episodes', action='store_true')
    p.add_argument('--calls', action='store_true')
    p.add_argument('--fns', action='store_true')
    p.add_argument('--md', action='store_true')
    p.add_argument('--limit', type=int, default=60)
    p.add_argument('--warmup', choices=('all', 'only', 'exclude'), default='all',
                   help='段階の名前に warmup を含むものだけ・除く')
    a = p.parse_args()
    procs = load(a.path)
    select = None
    if a.warmup == 'only':
        select = lambda s: 'warmup' in s  # noqa: E731
    elif a.warmup == 'exclude':
        select = lambda s: 'warmup' not in s  # noqa: E731
    print_stage_table(stage_table(procs, a.depth, select), a.md, a.limit)
    if a.episodes:
        print()
        episode_table(procs, a.md)
    if a.calls:
        print()
        calls_table(procs, a.md, a.depth)
    if a.fns:
        print()
        fn_table(procs, a.md, a.depth, a.limit, select)


if __name__ == '__main__':
    main()
