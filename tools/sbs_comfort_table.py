#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""study/sbs-comfort: sbs_comfort_eval.py run の結果 (案ごとの JSON) を表にする。

    python3 tools/sbs_comfort_table.py base=a.json tie=b.json ...
"""

import json
import sys

import numpy as np


def q(values, p):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, p)) if values else float('nan')


def load(spec):
    label, path = spec.split('=', 1)
    with open(path) as f:
        return label, json.load(f)['records']


def main():
    runs = [load(s) for s in sys.argv[1:]]
    base_label, base = runs[0]
    base_by = {(r['data_dir'], r['name']): r for r in base}
    for kind in ('early', 'resolve'):
        print('\n## {}'.format(kind))
        print('| 案 | n | 確保 通る | 実行 通る | 横 中央値 | 横≤0.75 | 横≤0.85 |'
              ' 人の項 中央値 | 人の項 90% | 経路の人の項最大 中央値 |'
              ' 確保 秒 中央/最大 | 裏 秒 中央/最大 |')
        print('|' + '---|' * 12)
        for label, recs in runs:
            recs = [r for r in recs if r['kind'] == kind]
            if not recs:
                continue
            ex = [r['executed'] for r in recs]
            ok = [e for e in ex if e['ok']]
            sec = [r.get('secured') for r in recs if r.get('secured')]
            ref = [r.get('refined') for r in recs if r.get('refined')]
            lat = [e['lateral'] for e in ok]
            print('| {} | {} | {} | {} | {:.2f} | {} | {} | {:.2f} | {:.2f} |'
                  ' {:.2f} | {} | {} |'.format(
                      label, len(recs),
                      sum(1 for s in sec if s['ok']) if sec else '-',
                      len(ok), q(lat, 50),
                      sum(1 for v in lat if v <= 0.75),
                      sum(1 for v in lat if v <= 0.85),
                      q([e['comfort_end'] for e in ok], 50),
                      q([e['comfort_end'] for e in ok], 90),
                      q([e['comfort_path_max'] for e in ok], 50),
                      '{:.1f}/{:.1f}'.format(q([s['wall'] for s in sec], 50),
                                             q([s['wall'] for s in sec], 100))
                      if sec else '-',
                      '{:.1f}/{:.1f}'.format(q([s['wall'] for s in ref], 50),
                                             q([s['wall'] for s in ref], 100))
                      if ref else (
                          '{:.1f}/{:.1f}'.format(
                              q([e['wall'] for e in ex], 50),
                              q([e['wall'] for e in ex], 100)))))
    # 試行ごとの差 (基準と違うものだけ)
    for label, recs in runs[1:]:
        print('\n## {} と {} で変わった試行'.format(label, base_label))
        print('| 試行 | 基準: 通る/横/人の項/向き | {}: 通る/横/人の項/向き |'.format(
            label))
        print('|---|---|---|')
        for r in recs:
            b = base_by.get((r['data_dir'], r['name']))
            if b is None:
                continue
            e, f = b['executed'], r['executed']

            def cell(x):
                if not x['ok']:
                    return '× ({})'.format(str(x.get('reason'))[:40])
                return '○ {:.2f} {:.2f} {}/{}'.format(
                    x['lateral'], x['comfort_end'], x.get('palm_facing'),
                    x.get('arm_deg'))
            if (e['ok'] != f['ok'] or (e['ok'] and (
                    abs(e['lateral'] - f['lateral']) > 0.005
                    or abs(e['comfort_end'] - f['comfort_end']) > 0.01))):
                print('| {} {} | {} | {} |'.format(
                    r['kind'][0], r['name'][:28], cell(e), cell(f)))


if __name__ == '__main__':
    main()
