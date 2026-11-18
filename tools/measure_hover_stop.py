#!/usr/bin/env python
"""実機ログ (run_camera_pipeline_test.py の person_*.log) から、接近の終わり (hover
到着) から押し込み区間を送るまでの止まっている時間と、その内訳を集計する。

区切りの行:
  到着     接近区間の ``[debug][segment] ゴール完了待ち`` (最初のもの)
  台車補正 到着後の最後の ``[execute][correct]``/``[execute][scan]`` の行
  refine 開始 ``[head] 首の追従・見渡しを止めました`` (腕と首の到着待ちの後)
  refine 終了 ``"event": "press_in_refine"`` の行
  押し込み 到着後に最初に出る ``[debug][segment] waypoint数=`` の行

使い方: python tools/measure_hover_stop.py data/real/2026-10-2*/ [--csv out.csv]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

TIME_RE = re.compile(r'^(\d\d):(\d\d):(\d\d\.\d+) (.*)$')


def seconds(match):
    h, m, s = match.group(1), match.group(2), match.group(3)
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_event(text):
    i = text.find('{')
    if i < 0:
        return None
    try:
        return json.loads(text[i:])
    except ValueError:
        return None


def measure(path):
    lines = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            m = TIME_RE.match(raw.rstrip('\n'))
            if m:
                lines.append((seconds(m), m.group(4)))
    arrival = None
    rec = dict(path=path)
    for i, (t, text) in enumerate(lines):
        if arrival is None:
            if '[debug][segment] ゴール完了待ち' in text:
                arrival = t
                rec['arrival_index'] = i
            continue
        if t < arrival:  # 日付をまたいだ
            t += 86400
        if 'press_send' in rec:
            break
        if ('[execute][correct]' in text or '[execute][scan]' in text
                or '"event": "scan_correction"' in text):
            rec['base_done'] = t - arrival
            if '"event": "scan_correction"' in text:
                rec['scan_attempts'] = rec.get('scan_attempts', 0) + 1
        elif '首の追従・見渡しを止めました' in text and 'refine_start' not in rec:
            rec['refine_start'] = t - arrival
        elif '"event": "press_in_base_shift"' in text:
            ev = parse_event(text) or {}
            rec['base_shift'] = bool(ev.get('applied'))
        elif '"event": "press_in_refine"' in text:
            ev = parse_event(text) or {}
            rec['refine_done'] = t - arrival
            rec['refine_reason'] = ev.get('reason')
            rec['refine_wait'] = ev.get('wait_time')
            rec['refine_ik'] = ev.get('ik_time')
            if ev.get('palm_shift') is not None:
                rec['palm_shift_mm'] = 1e3 * sum(
                    v * v for v in ev['palm_shift']) ** 0.5
        elif '[debug][segment] waypoint数=' in text:
            rec['press_send'] = t - arrival
    if arrival is None or 'press_send' not in rec:
        return None
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--csv')
    args = parser.parse_args()
    paths = []
    for root in args.roots:
        if os.path.isfile(root):
            paths.append(root)
        else:
            paths += sorted(glob.glob(os.path.join(root, '**', 'person_*.log'),
                                      recursive=True))
    # 同じ試行の写し (trial*_side_by_side と logs) は 1 つにする。
    seen = set()
    records = []
    for path in paths:
        rec = measure(path)
        if rec is None:
            continue
        key = (rec['press_send'], rec.get('refine_done'),
               os.path.basename(path))
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)

    cols = ['press_send', 'base_done', 'refine_start', 'refine_done',
            'refine_wait', 'refine_ik', 'scan_attempts', 'base_shift',
            'refine_reason', 'palm_shift_mm']
    print('{:<58} {}'.format('log', ' '.join('{:>13}'.format(c) for c in cols)))
    for rec in records:
        cells = []
        for c in cols:
            v = rec.get(c)
            cells.append('{:>13}'.format(
                '-' if v is None else
                '{:.2f}'.format(v) if isinstance(v, float) else str(v)))
        print('{:<58} {}'.format(os.path.relpath(rec['path'])[-58:],
                                 ' '.join(cells)))

    def stats(name, values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return
        print('{:<28} n={:>3} 平均 {:.2f}s  中央 {:.2f}s  最大 {:.2f}s'.format(
            name, len(values), sum(values) / len(values),
            values[len(values) // 2], values[-1]))

    print()
    stats('到着→押し込み送信', [r['press_send'] for r in records])
    stats('到着→台車補正の終わり', [r.get('base_done') for r in records])
    stats('台車補正→refine 開始',
          [r['refine_start'] - r.get('base_done', 0.0) for r in records
           if 'refine_start' in r])
    stats('refine (検出〜IK)',
          [r['refine_done'] - r['refine_start'] for r in records
           if 'refine_done' in r and 'refine_start' in r])
    stats('refine→押し込み送信',
          [r['press_send'] - r['refine_done'] for r in records
           if 'refine_done' in r])
    reasons = {}
    for r in records:
        reasons[r.get('refine_reason')] = reasons.get(
            r.get('refine_reason'), 0) + 1
    print('refine の結果:', reasons)
    print('台車を寄せた (press_in_base_shift):',
          sum(1 for r in records if r.get('base_shift')))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['path'] + cols)
            writer.writeheader()
            for rec in records:
                writer.writerow({k: rec.get(k) for k in ['path'] + cols})
    return 0


if __name__ == '__main__':
    sys.exit(main())
