#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""scripts/qprof.py の計測を、計画の 5 段 (hover・後処理・軌道・横並び・寄せ) と warmup に分けて、
1 回 (1 人) あたりの秒を分類ごと・包んだ関数ごと・段階 (入れ子) ごとに出す (Markdown)。

関数ごとの表は秒/人・回/人・ms/回、段階の入れ子ごとの表は中の時間を含む。

    python3 tools/q_profile_report.py <prof のディレクトリ> [--n 67] [--min 0.005]

1 人あたりの分母は、まとまりの一番外の関数の回数 (hover・後処理は ``IK``、軌道は ``traj``、横並びは
``transition``、寄せは ``refine_transition``)。``--n`` を与えるとすべてそれで割る。
"""

import argparse
import collections
import glob
import json
import os

# (名前, 段階の規則, 分母にする一番外の段階)
GROUPS = [
    # plan_handshake_motion の main は横並びのバッチ IK の前に solve_person_ik を 1 回解く (一番外の hover_ik)。
    ('warmup', lambda s: 'warmup' in s or s.split('/')[0] == 'hover_ik', None),
    ('後処理 (pick_verified_candidate: 押し込み・後処理の IK と検証)',
     lambda s: s.startswith('IK') and '/pick' in s, 'IK'),
    ('hover (solve_person_ik のバッチ IK と準備)', lambda s: s.startswith('IK'), 'IK'),
    ('軌道 (plan_person_motion)', lambda s: s.startswith('traj'), 'traj'),
    ('横並び (plan_transition、確保)', lambda s: s.startswith('transition'), 'transition'),
    ('寄せ (refine_transition、裏の計画)', lambda s: s.startswith('refine_transition'), 'refine_transition'),
    ('その他', lambda s: True, None),
]
CATS = ['solve', 'compile', 'fk', 'verify', 'human', 'py']
CAT_LABELS = ['solve 本体', 'コンパイル・トレース', 'FK', '事後検証', '人の腕', 'Python']
# wrap_stage の段階の名前 -> 包んだ関数 (段階に入った回数を数えるため)
STAGE_ATTRS = {
    'IK': 'solve_person_ik_side_by_side', 'hover_ik': 'solve_person_ik', 'pick': 'pick_verified_candidate',
    'post_process': 'solve_post_process', 'press_in': 'solve_press_in', 'refine_post': 'refine_post_process',
    'transition': 'plan_transition', 'refine_transition': 'refine_transition', 'plan_tr': '_plan_transition',
    'stage1': '_plan_stage1', 'goal_candidates': 'solve_goal_candidates', 'path': '_plan_path_impl',
    'approach': 'plan_approach', 'add_approach': '_add_approach', 'direct_end': '_direct_to_approach_end',
    'add_alt_arm': '_add_alt_arm_approach', 'alt_arms': '_approach_alt_arms', 'fix_wrist': 'fix_goal_wrist',
    'aim_head': '_aim_head', 'traj': 'plan_person_motion',
    'warmup_batch_ik': '_warmup_batch_ik', 'warmup_sbs_batch_ik': 'warmup_batch_ik', 'warmup_traj': '_warmup_solver',
}


def group_of(stage):
    for name, rule, _ in GROUPS:
        if rule(stage):
            return name
    return 'その他'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--n', type=int, default=None, help='分母 (既定はまとまりの一番外の関数の回数)')
    p.add_argument('--min', type=float, default=0.005, help='関数・段階の表に出す下限 [秒/人]')
    a = p.parse_args()
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    fns = collections.defaultdict(lambda: collections.defaultdict(float))
    fn_n = collections.defaultdict(collections.Counter)
    stage_ex = collections.defaultdict(float)      # 段階 -> 排他的な秒
    stage_n = collections.Counter()                # 段階 -> 入った回数
    solves = collections.Counter()
    tops = collections.Counter()
    for f in sorted(glob.glob(os.path.join(a.path, '*.json'))):
        data = json.load(open(f))
        for th in data['threads']:
            for stage, cat, sec in th['acc']:
                if stage == '(none)':
                    continue
                acc[group_of(stage)][cat] += sec
                stage_ex[stage] += sec
            for stage, cat, fn, sec in th.get('fn_acc', []):
                fns[group_of(stage)][(cat, fn)] += sec
            for stage, fn, n in th['count']:
                fn_n[group_of(stage)][(stage, fn)] += n
                if STAGE_ATTRS.get(stage.split('/')[-1]) == fn:
                    stage_n[stage] += n
            for c in th['calls']:
                solves[group_of(c[0])] += 1
            for e in th['episodes']:
                tops[e['stage']] += 1

    def denom(name):
        if a.n:
            return float(a.n)
        top = [t for g, _, t in GROUPS if g == name][0]
        return float(tops.get(top) or 1)

    def fn_count(name, fn):
        return sum(n for (stage, f), n in fn_n[name].items() if f == fn)

    order = [g for g, _, _ in GROUPS if g in acc]
    print('## まとまりごと [秒/回]\n')
    print('| まとまり | 分母 (回) | 合計 | ' + ' | '.join(CAT_LABELS) + ' | solve 回数/回 |')
    print('|---|---|---|' + '---|' * len(CATS) + '---|')
    for name in order:
        cats = acc[name]
        n = denom(name)
        print('| {} | {:.0f} | {:.3f} | {} | {:.1f} |'.format(
            name, n, sum(cats.values()) / n,
            ' | '.join('{:.3f}'.format(cats.get(c, 0.0) / n) for c in CATS), solves[name] / n))
    for name in order:
        n = denom(name)
        print('\n## {} の関数ごと (排他的、分母 {:.0f})\n'.format(name, n))
        print('| 分類 | 関数 | 秒/回 | 呼ぶ回数/回 | ms/呼び出し | 合計 [秒] |')
        print('|---|---|---|---|---|---|')
        for (cat, fn), sec in sorted(fns[name].items(), key=lambda kv: -kv[1]):
            if sec / n < a.min:
                continue
            cnt = fn_count(name, fn)
            print('| {} | {} | {:.3f} | {:.1f} | {} | {:.1f} |'.format(
                cat, fn, sec / n, cnt / n, '{:.2f}'.format(1000.0 * sec / cnt) if cnt else '-', sec))
        fk_rest = acc[name].get('fk', 0.0) - sum(v for (c, _), v in fns[name].items() if c == 'fk')
        print('| py | (包んだ関数の外の Python) | {:.3f} | | | {:.1f} |'.format(acc[name].get('py', 0.0) / n,
                                                                       acc[name].get('py', 0.0)))
        if fk_rest > 0.0:
            print('| fk | (skrobot の FK・angle_vector、包んだ関数の外) | {:.3f} | | | {:.1f} |'.format(
                fk_rest / n, fk_rest))
        if acc[name].get('compile', 0.0) > 0.0:
            print('| compile | (jax のトレース・コンパイル) | {:.3f} | | | {:.1f} |'.format(
                acc[name]['compile'] / n, acc[name]['compile']))
        # 段階の入れ子ごと (中の時間を含む)
        stages = sorted(s for s in stage_ex if group_of(s) == name)
        if len(stages) > 1:
            print('\n段階の入れ子 (中の時間を含む、分母 {:.0f}):\n'.format(n))
            print('| 段階 | 秒/回 (中を含む) | うち自分 | 入った回数/回 |')
            print('|---|---|---|---|')
            for s in stages:
                incl = sum(v for t, v in stage_ex.items() if t == s or t.startswith(s + '/'))
                if incl / n < a.min:
                    continue
                print('| {} | {:.3f} | {:.3f} | {:.1f} |'.format(s, incl / n, stage_ex[s] / n, stage_n[s] / n))


if __name__ == '__main__':
    main()
