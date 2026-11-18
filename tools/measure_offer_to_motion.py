#!/usr/bin/env python
"""実機ログ (run_camera_pipeline_test.py の person_*.log) から、差し出し手の確定
(``[ARMED] 試行N: ...を差し出しました``) から最初の動作指令までの時間と内訳を
試行ごとに集計する (ROS 不要)。

時刻はすべて ``[ARMED]`` の行 (t0) からの秒。区切りの行と意味:

  nod_send   うなずきの送信 (推定)。``[nod] ... (X 秒)`` の行の時刻 - X。
             うなずきは ``_try_handshake`` で計画より前に別スレッドで送るので、
             0 付近なら計画を待たずに出ている (負なら ``[ARMED]`` の行より前)。
  refine_bg  裏の計画 (寄せ) の計算時間 (最初の指令とは並行。参考)。
  ik         IK (``"event": "result"`` の collision_ik_time +
             candidate_selection_time)。
  motion     接近軌道の計画 (``"event": "motion"`` の compute_time)。
  transition 横並び移動の計画 (``"event": "transition"`` の compute_time)。
  plan_other 計画全体 - ik - motion - transition (前処理・干渉検証など)。
  plan_end   計画の終わり (``[result]`` の行)。
  save       ``[result]`` -> ``[auto-execute]`` (表示の更新・JSON 保存)。
  nod_wait   ``[auto-execute]`` -> ``[execute] 実機で waypoint を`` (実行
             スレッドの起動とうなずきの終わり待ち)。
  prep       ``[execute] 実機で waypoint を`` -> 最初の ``[debug][segment]
             waypoint数=`` (開始の発話 (非同期)・/scan 1 つ待ち・時間割り当て)。
  first_body 台車・腕の最初の指令 (``[debug][segment] waypoint数=`` の行。腕の
             ゴールを送った直後、台車のゴールの直前に出る)。

使い方: python tools/measure_offer_to_motion.py data/real/ [--since 2026-10-21]
       [--csv out.csv]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

TIME_RE = re.compile(r'^(\d\d):(\d\d):(\d\d\.\d+) (.*)$')
NOD_RE = re.compile(r'^\[nod\] .*\(([\d.]+) 秒\)')
PLAN_RE = re.compile(r'^\[plan\] 試行\d+: 計画全体 ([\d.]+) 秒')
OPT_WAIT_RE = re.compile(r'軌道最適化の準備を ([\d.]+) 秒待ちました')
ARMED_RE = re.compile(r'^\[ARMED\] 試行(\d+): (\w)手を差し出しました')


def parse_event(text):
    i = text.find('{')
    if i < 0:
        return None
    try:
        return json.loads(text[i:])
    except ValueError:
        return None


def read_lines(path):
    lines = []
    previous = None
    day = 0.0
    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            m = TIME_RE.match(raw.rstrip('\n'))
            if not m:
                continue
            t = (int(m.group(1)) * 3600 + int(m.group(2)) * 60
                 + float(m.group(3)) + day)
            if previous is not None and t < previous - 3600:
                day += 86400.0
                t += 86400.0
            previous = t
            lines.append((t, m.group(4)))
    return lines


def measure(path):
    lines = read_lines(path)
    t0 = None
    rec = dict(path=path)
    for t, text in lines:
        if t0 is None:
            m = ARMED_RE.match(text)
            if m:
                t0 = t
                rec['key'] = (round(t, 3), text)
                rec['attempt'] = int(m.group(1))
                rec['hand'] = m.group(2)
            continue
        rel = t - t0
        if text.startswith('[debug] {"event": "transition_refine"'):
            # 裏の計画 (寄せ) は最初の指令の後に終わることが多い。
            ev = parse_event(text) or {}
            rec.setdefault('refine_bg', ev.get('compute_time'))
            rec.setdefault('refine_bg_end', rel)
            continue
        if 'first_body' in rec:
            continue
        m = NOD_RE.match(text)
        if m and 'nod_end' not in rec:
            rec['nod_end'] = rel
            rec['nod_dur'] = float(m.group(1))
            # 負なら [ARMED] の行より前に送っている (行の出力が遅れた)。
            rec['nod_send'] = rel - float(m.group(1))
            continue
        m = OPT_WAIT_RE.search(text)
        if m:
            rec['opt_wait'] = float(m.group(1))
            continue
        if text.startswith('[debug] {"event": "motion"'):
            ev = parse_event(text) or {}
            rec['motion'] = ev.get('compute_time')
            rec['motion_end'] = rel
            rec['motion_kind'] = ev.get('kind')
        elif text.startswith('[debug] {"event": "transition"'):
            ev = parse_event(text) or {}
            rec['transition'] = ev.get('compute_time')
            rec['transition_end'] = rel
        elif text.startswith('[debug] {"event": "result"'):
            ev = parse_event(text) or {}
            rec['solved'] = ev.get('solved')
            ik = [ev.get('collision_ik_time'),
                  ev.get('candidate_selection_time')]
            if all(v is not None for v in ik):
                rec['ik'] = sum(ik)
        elif text.startswith('[result] 試行'):
            rec['plan_end'] = rel
        elif PLAN_RE.match(text):
            rec['plan_total_log'] = float(PLAN_RE.match(text).group(1))
        elif text.startswith('[auto-execute]'):
            rec['auto'] = rel
        elif text.startswith('[execute] 実機で waypoint を'):
            rec['exec_start'] = rel
        elif ('[debug][segment] waypoint数=' in text
              and 'exec_start' in rec):
            rec['first_body'] = rel
        elif text.startswith('[advice]') or text.startswith('[ARM] ARMED'):
            # 実行しない試行 (IK 失敗など) はここで終わり。
            if 'auto' not in rec:
                rec['no_exec_at'] = rel
    if t0 is None:
        return None
    if 'plan_end' in rec:
        parts = [rec.get(k) or 0.0 for k in ('ik', 'motion', 'transition')]
        rec['plan_other'] = rec['plan_end'] - sum(parts)
    if 'auto' in rec and 'plan_end' in rec:
        rec['save'] = rec['auto'] - rec['plan_end']
    if 'exec_start' in rec and 'auto' in rec:
        rec['nod_wait'] = rec['exec_start'] - rec['auto']
    if 'first_body' in rec:
        rec['prep'] = rec['first_body'] - rec['exec_start']
    return rec


COLS = ['ik', 'motion', 'transition', 'plan_other', 'plan_end', 'save',
        'nod_wait', 'prep', 'first_body', 'nod_send', 'nod_end', 'opt_wait',
        'refine_bg']


def label(path):
    parts = os.path.normpath(path).split(os.sep)
    # .../real/<日付>/<試行 or logs>/[<起動時刻>/]person_NN.log
    for i, p in enumerate(parts):
        if re.match(r'^\d{4}-\d\d-\d\d$', p):
            return '/'.join(parts[i:])
    return '/'.join(parts[-3:])


def stats(name, values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return
    print('{:<24} n={:>3} 平均 {:5.2f}s  中央 {:5.2f}s  最小 {:5.2f}s  '
          '最大 {:5.2f}s'.format(name, len(values), sum(values) / len(values),
                                values[len(values) // 2], values[0],
                                values[-1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--csv')
    parser.add_argument('--since', help='この日付 (YYYY-MM-DD) 以降のログだけ')
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
        if rec is None or rec['key'] in seen:
            continue
        if args.since and label(path).split('/')[0] < args.since:
            continue
        seen.add(rec['key'])
        records.append(rec)

    print('{:<46} {:>4} {:>5} {}'.format(
        'log', 'hand', 'exec', ' '.join('{:>10}'.format(c) for c in COLS)))
    for rec in records:
        cells = []
        for c in COLS:
            v = rec.get(c)
            cells.append('{:>10}'.format(
                '-' if v is None else '{:.2f}'.format(v)))
        print('{:<46} {:>4} {:>5} {}'.format(
            label(rec['path'])[-46:], rec.get('hand', '?'),
            'yes' if 'first_body' in rec else
            ('IK失敗' if rec.get('solved') is False else 'no'),
            ' '.join(cells)))

    executed = [r for r in records if 'first_body' in r]
    print()
    print('実行した試行 {} / 全 {} 試行'.format(len(executed), len(records)))
    for c in COLS:
        stats(c + ' (実行)', [r.get(c) for r in executed])
    print()
    stats('plan_end (全試行)', [r.get('plan_end') for r in records])
    stats('ik (全試行)', [r.get('ik') for r in records])
    kinds = {}
    for r in executed:
        kinds[r.get('motion_kind')] = kinds.get(r.get('motion_kind'), 0) + 1
    print('接近軌道の種類 (実行):', kinds)
    print()
    dates = sorted({label(r['path']).split('/')[0] for r in executed})
    for date in dates:
        group = [r for r in executed if label(r['path']).startswith(date)]
        stats('first_body ' + date, [r['first_body'] for r in group])
        stats('  transition ' + date, [r.get('transition') for r in group])
    # 律速: 最初の指令までの各区間の割合。
    if executed:
        total = sum(r['first_body'] for r in executed)
        print('first_body の内訳 (実行した試行の合計に対する割合):')
        for c in ('ik', 'motion', 'transition', 'plan_other', 'save',
                  'nod_wait', 'prep'):
            s = sum(r.get(c) or 0.0 for r in executed)
            print('  {:<11} {:6.2f}s ({:4.1f}%)'.format(c, s, 100 * s / total))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            fields = ['path', 'attempt', 'hand', 'solved', 'motion_kind'] + COLS
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for rec in records:
                writer.writerow({k: rec.get(k) for k in fields})
    return 0


if __name__ == '__main__':
    sys.exit(main())
