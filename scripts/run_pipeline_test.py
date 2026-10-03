#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""合成データのパイプライン (骨格生成 → 掌推定 → IK → [軌道計画] →
[ビューア]) を /tmp の一時ディレクトリで順に実行し、ステップごとの結果を
要約表示する回帰テスト。

Usage
-----
    python3 run_pipeline_test.py 20
    python3 run_pipeline_test.py 20 --viewer   # 最後に viser ビューアも開く
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import json_io  # noqa: E402  (パス追加後に import)


def run_step(label, script_name, extra_args):
    """``script_name`` を子プロセスで実行し ``(経過秒, 標準出力)`` を返す
    (失敗したら出力の末尾を表示して終了)。"""
    print('[{}] 実行中...'.format(label))
    script_path = os.path.join(_THIS_DIR, script_name)
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script_path] + extra_args, cwd=_THIS_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print('[{}] {} が exit code {} で失敗しました。'.format(
            label, script_name, result.returncode))
        sys.exit(1)
    return elapsed, result.stdout


def extract_warmup_lines(stdout):
    """標準出力の ``[warmup]`` 行と合計秒数 ``(lines, total)`` を返す。"""
    lines = [line for line in stdout.splitlines()
            if line.startswith('[warmup]')]
    total = 0.0
    for line in lines:
        match = re.search(r'([\d.]+)\s*秒\s*$', line)
        if match:
            total += float(match.group(1))
    return lines, total


def load_json_files(directory):
    for path in json_io.iter_json_files(directory):
        with open(path) as f:
            yield path, json.load(f)


def summarize_palms(palm_dir):
    counts = {'R': 0, 'L': 0, None: 0}
    n = 0
    for _, data in load_json_files(palm_dir):
        n += 1
        counts[data.get('offered_hand')] = \
            counts.get(data.get('offered_hand'), 0) + 1
    return n, counts


def summarize_handshakes(handshake_dir):
    """``solve_palm_ik.py`` の出力 JSON を集計する (IK 2 段階目の時間は
    ``candidate_selection_time``。``post_process.compute_time`` は棄却候補を
    含まないので使わない)。"""
    n_total = n_target = n_solved = n_post_process = 0
    collision_ik_times = []
    candidate_selection_times = []
    for _, data in load_json_files(handshake_dir):
        n_total += 1
        if not data.get('target'):
            continue
        n_target += 1
        if data.get('collision_ik_time') is not None:
            collision_ik_times.append(data['collision_ik_time'])
        if data.get('candidate_selection_time') is not None:
            candidate_selection_times.append(data['candidate_selection_time'])
        if data.get('solved'):
            n_solved += 1
            if data.get('post_process') is not None:
                n_post_process += 1

    def avg(values):
        return sum(values) / len(values) if values else None

    return dict(
        n_total=n_total, n_target=n_target, n_solved=n_solved,
        n_post_process=n_post_process,
        avg_collision_ik_time=avg(collision_ik_times),
        avg_candidate_selection_time=avg(candidate_selection_times),
        avg_total_ik_time=avg([
            c + s for c, s in zip(collision_ik_times,
                                  candidate_selection_times)]),
    )


def summarize_final_corrections(handshake_dir):
    """``final_correction`` (押し込み直前の最終補正の試行) を集計する
    (比較用に計画時の押し込み IK の平均時間も返す)。無ければ ``None``。"""
    times = []
    reasons = {}
    position_changes = []
    planned_times = []
    n_people = 0
    for _, data in load_json_files(handshake_dir):
        records = data.get('final_correction')
        if not records:
            continue
        n_people += 1
        planned_times.append(data['post_process']['compute_time'])
        for record in records:
            times.append(record['compute_time'])
            reasons[record['reason']] = reasons.get(record['reason'], 0) + 1
            position_changes.append(record['position_change'])
    if not times:
        return None
    times_sorted = sorted(times)
    return dict(
        n_people=n_people, n_trials=len(times), reasons=reasons,
        avg_time=sum(times) / len(times),
        median_time=times_sorted[len(times_sorted) // 2],
        max_time=times_sorted[-1],
        avg_planned_time=sum(planned_times) / len(planned_times),
        avg_position_change=sum(position_changes) / len(position_changes))


def summarize_motions(motion_dir):
    """``plan_handshake_motion.py`` の出力 JSON を集計する (``planned`` の人のみ)。"""
    n_planned = n_verified = n_lead_in_verified = n_both_verified = 0
    n_head_blended = 0
    kinds = {}
    approach_angles = {}
    compute_times = []
    for _, data in load_json_files(motion_dir):
        if not data.get('planned'):
            continue
        n_planned += 1
        verified = bool(data.get('verified'))
        lead_in_verified = bool(data.get('lead_in_verified', True))
        n_verified += int(verified)
        n_lead_in_verified += int(lead_in_verified)
        n_both_verified += int(verified and lead_in_verified)
        n_head_blended += int(bool(data.get('head_gaze_blended')))
        kind = data.get('kind')
        kinds[kind] = kinds.get(kind, 0) + 1
        if data.get('approach_angle') is not None:
            angle = int(round(math.degrees(data['approach_angle'])))
            approach_angles[angle] = approach_angles.get(angle, 0) + 1
        if data.get('compute_time') is not None:
            compute_times.append(data['compute_time'])

    # 最適化を要した人だけの平均は母集団が実行ごとに変わるので、全員の平均。
    avg_compute_time = (sum(compute_times) / len(compute_times)
                        if compute_times else None)
    return dict(n_planned=n_planned, n_verified=n_verified,
               n_lead_in_verified=n_lead_in_verified,
               n_both_verified=n_both_verified,
               n_head_blended=n_head_blended, kinds=kinds,
               approach_angles=approach_angles,
               avg_compute_time=avg_compute_time)


def main():
    parser = argparse.ArgumentParser(
        description='合成データのパイプラインを順に実行する回帰テスト。')
    parser.add_argument(
        'num_people', type=int, help='生成する人数。')
    parser.add_argument(
        '--viewer', action='store_true',
        help='最後に viser ビューアを起動する (ブラウザ接続を待つ)。')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='generate_random_human_poses.py の乱数シード。')
    parser.add_argument(
        '--plan-motion', action='store_true',
        help='plan_handshake_motion.py も実行する (jaxls が必要)。')
    parser.add_argument(
        '--force-optimize', action='store_true',
        help='plan_handshake_motion.py に --force-optimize を渡す。')
    parser.add_argument(
        '--initial-base-pose', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'YAW'),
        help='plan_handshake_motion.py に渡す初期台車姿勢 (例: 5 0 3.14)。')
    parser.add_argument(
        '--approach-distance', type=float, default=None,
        help='plan_handshake_motion.py に --approach-distance を渡す。')
    parser.add_argument(
        '--final-correction-trials', type=int, default=0,
        help='solve_palm_ik.py に渡す、1 人あたりの最終補正の試行回数 '
            '(既定 0 = 行わない)。')
    parser.add_argument(
        '--final-correction-slip', type=float, nargs=2, default=None,
        metavar=('XY', 'YAW_DEG'),
        help='solve_palm_ik.py に --final-correction-slip を渡す。')
    parser.add_argument(
        '--base-x-standing-margins', type=float, nargs='+', default=None,
        help='solve_palm_ik.py に --base-x-standing-margins を渡す。')
    parser.add_argument(
        '--front-offset-weight', type=float, default=None,
        help='solve_palm_ik.py に --front-offset-weight を渡す。')
    parser.add_argument(
        '--facing-yaw-weight', type=float, default=None,
        help='solve_palm_ik.py に --facing-yaw-weight を渡す。')
    parser.add_argument(
        '--collision-verify-model', choices=('mixed', 'nohand', 'hand'),
        default=None,
        help='solve_palm_ik.py と plan_handshake_motion.py に渡す。')
    parser.add_argument(
        '--side-by-side-transition', action='store_true',
        help='plan_handshake_motion.py に --side-by-side-transition を渡す。')
    args = parser.parse_args()

    base_dir = tempfile.mkdtemp(prefix='aero_demo_pipeline_', dir='/tmp')
    skeleton_dir = os.path.join(base_dir, 'skeletons')
    palm_dir = os.path.join(base_dir, 'palms')
    handshake_dir = os.path.join(base_dir, 'handshakes')
    motion_dir = os.path.join(base_dir, 'motions')
    print('作業ディレクトリ: {}'.format(base_dir))

    # 1. generate_random_human_poses.py
    gen_args = ['--num-samples', str(args.num_people),
               '--output-dir', skeleton_dir]
    if args.seed is not None:
        gen_args += ['--seed', str(args.seed)]
    run_step('1/5', 'generate_random_human_poses.py', gen_args)
    n_generated = len(json_io.iter_json_files(skeleton_dir))
    print('[1/5] generate_random_human_poses.py: 骨格 JSON を {} 件生成 '
          '({})'.format(n_generated, skeleton_dir))

    # 2. estimate_palm_poses.py
    palm_elapsed, _ = run_step(
        '2/5', 'estimate_palm_poses.py',
        ['--input-dir', skeleton_dir, '--output-dir', palm_dir])
    n_palms, offered = summarize_palms(palm_dir)
    print('[2/5] estimate_palm_poses.py: 掌の位置姿勢 JSON を {} 件推定 '
          '(offered_hand: R={}, L={}, null={})'.format(
              n_palms, offered.get('R', 0), offered.get('L', 0),
              offered.get(None, 0)))

    # 4. solve_palm_ik.py
    solve_args = ['--input-dir', palm_dir, '--output-dir', handshake_dir,
                  '--skeleton-dir', skeleton_dir]
    if args.final_correction_trials > 0:
        solve_args += ['--final-correction-trials',
                       str(args.final_correction_trials)]
        if args.final_correction_slip is not None:
            solve_args += ['--final-correction-slip'] + [
                str(v) for v in args.final_correction_slip]
    if args.base_x_standing_margins is not None:
        solve_args += ['--base-x-standing-margins'] + [
            str(v) for v in args.base_x_standing_margins]
    if args.front_offset_weight is not None:
        solve_args += ['--front-offset-weight', str(args.front_offset_weight)]
    if args.facing_yaw_weight is not None:
        solve_args += ['--facing-yaw-weight', str(args.facing_yaw_weight)]
    if args.collision_verify_model is not None:
        solve_args += ['--collision-verify-model',
                       args.collision_verify_model]
    solve_elapsed, solve_stdout = run_step('4/5', 'solve_palm_ik.py',
                                           solve_args)
    warmup_lines, warmup_total = extract_warmup_lines(solve_stdout)
    summary = summarize_handshakes(handshake_dir)
    print('[4/5] solve_palm_ik.py: IK 対象 {} 人中 {} 人 solved '
          '(対象外 {} 人)'.format(
              summary['n_target'], summary['n_solved'],
              summary['n_total'] - summary['n_target']))
    if warmup_lines:
        for line in warmup_lines:
            print('[4/5] {}'.format(line))
        print('[4/5] warmup 合計 {:.1f} 秒 (solve_palm_ik.py の壁時計時間 '
              '{:.1f} 秒中)'.format(warmup_total, solve_elapsed))
    else:
        print('[4/5] warmup 行が見つかりませんでした '
              '(--no-warmup 指定、または IK 対象が 0 人でスキップされた '
              '可能性があります)。壁時計時間 {:.1f} 秒'.format(solve_elapsed))

    # 4.5. plan_handshake_motion.py (--plan-motion 指定時のみ)
    motion_summary = None
    if args.plan_motion:
        motion_args = ['--input-dir', handshake_dir,
                      '--output-dir', motion_dir,
                      '--skeleton-dir', skeleton_dir]
        if args.force_optimize:
            motion_args.append('--force-optimize')
        if args.initial_base_pose is not None:
            motion_args += ['--initial-base-pose'] + [
                str(v) for v in args.initial_base_pose]
        if args.approach_distance is not None:
            motion_args += ['--approach-distance', str(args.approach_distance)]
        if args.collision_verify_model is not None:
            motion_args += ['--collision-verify-model',
                            args.collision_verify_model]
        if args.side_by_side_transition:
            motion_args.append('--side-by-side-transition')
        motion_elapsed, motion_stdout = run_step(
            '4.5/5', 'plan_handshake_motion.py', motion_args)
        motion_warmup_lines, motion_warmup_total = extract_warmup_lines(
            motion_stdout)
        motion_summary = summarize_motions(motion_dir)
        print('[4.5/5] plan_handshake_motion.py: 軌道計画対象 {} 人中 '
              '{} 人 verified (厳密形状で経路上の干渉なしを確認)。'
              '採用した軌道の作り方: {}'.format(
                  motion_summary['n_planned'], motion_summary['n_verified'],
                  ', '.join('{}={}'.format(k, v) for k, v
                            in sorted(motion_summary['kinds'].items(),
                                      key=lambda kv: str(kv[0])))))
        print('[4.5/5] 初期位置からの直進 (lead-in) が verified: {} 人、'
              '接近開始位置に選んだ候補の角度 [度]: {}'.format(
                  motion_summary['n_lead_in_verified'],
                  ', '.join('{}={}'.format(k, v) for k, v in sorted(
                      motion_summary['approach_angles'].items()))))
        print('[4.5/5] 接近区間の後半で首を押し込み姿勢 (掌を向く視線) へ '
              '補間した人数: {} / {}'.format(
                  motion_summary['n_head_blended'],
                  motion_summary['n_planned']))
        if args.side_by_side_transition:
            for line in motion_stdout.splitlines():
                if line.startswith('横並び移動:') or '[transition]' in line:
                    print('[4.5/5] {}'.format(line.strip()))
        if motion_warmup_lines:
            for line in motion_warmup_lines:
                print('[4.5/5] {}'.format(line))
            print('[4.5/5] warmup 合計 {:.1f} 秒 '
                  '(plan_handshake_motion.py の壁時計時間 {:.1f} 秒中)'
                  .format(motion_warmup_total, motion_elapsed))
        else:
            print('[4.5/5] warmup 行が見つかりませんでした '
                  '(--no-warmup 指定、または軌道計画対象が 0 人でスキップ '
                  'された可能性があります)。壁時計時間 {:.1f} 秒'
                  .format(motion_elapsed))

    print()
    print('=== 結果 ===')
    print('生成した骨格人数: {}'.format(n_generated))
    print('干渉回避まで解けた人数 (solved): {} / {}'.format(
        summary['n_solved'], n_generated))
    print('後処理まで含めて成功した人数: {} / {}'.format(
        summary['n_post_process'], n_generated))
    if motion_summary is not None:
        print('初期姿勢から握手姿勢までの軌道が経路上の干渉も含めて '
              '検証できた人数 (verified): {} / {}'.format(
                  motion_summary['n_verified'], n_generated))
        print('  うち初期位置からの直進 (lead-in) も含めて verified: '
              '{} / {}'.format(motion_summary['n_both_verified'],
                               n_generated))
        if motion_summary['avg_compute_time'] is not None:
            print('  軌道計画の 1人あたり平均計算時間 '
                  '(最適化を要さなかった人も含む全 {} 人の平均): '
                  '{:.3f} 秒/人'.format(
                      motion_summary['n_planned'],
                      motion_summary['avg_compute_time']))

    palm_time_per_person = palm_elapsed / n_generated if n_generated else None
    print()
    print('=== IK対象 ({} 人) での 1人あたり平均計算時間 ==='.format(
        summary['n_target']))
    print('  (1人を入力したときの所要時間の目安。掌推定は骨格 {} 人分 '
          'まとめて実行した壁時計時間を人数で割った値、IK 1/2段階目は '
          'IK対象 {} 人だけの平均。)'.format(n_generated, summary['n_target']))
    if palm_time_per_person is not None:
        print('  掌推定: {:.3f} 秒/人'.format(palm_time_per_person))
    if summary['avg_collision_ik_time'] is not None:
        print('  IK 1段階目 (干渉回避バッチIK): {:.3f} 秒/人'.format(
            summary['avg_collision_ik_time']))
    if summary['avg_candidate_selection_time'] is not None:
        print('  IK 2段階目 (事後の干渉検証+全後処理判定): {:.3f} 秒/人'
              .format(summary['avg_candidate_selection_time']))
    if (summary['avg_total_ik_time'] is not None
            and palm_time_per_person is not None):
        print('  掌推定込みの全体 (掌推定 + IK 1+2段階目): {:.3f} 秒/人'
              .format(palm_time_per_person + summary['avg_total_ik_time']))
    print('  (IK 1/2段階目は warmup を含まない。1段階目が長い場合は人物ごとの '
          '再コンパイルを疑うこと: docs/jax_compilation_cache.md)')

    if args.final_correction_trials > 0:
        fc = summarize_final_corrections(handshake_dir)
        print()
        if fc is None:
            print('=== 押し込み直前の最終補正 === 対象 (後処理まで解けた人物) '
                  'がいませんでした。')
        else:
            print('=== 押し込み直前の最終補正 ({} 人 x {} 回 = {} 試行) ==='
                  .format(fc['n_people'], args.final_correction_trials,
                          fc['n_trials']))
            print('  補正 IK の計算時間: 平均 {:.3f} 秒 / 中央値 {:.3f} 秒 / '
                  '最大 {:.3f} 秒 (計画時の押し込み IK は平均 {:.3f} 秒)'
                  .format(fc['avg_time'], fc['median_time'], fc['max_time'],
                          fc['avg_planned_time']))
            print('  結果の内訳: {} (押し込み目標の変化 平均 {:.1f}mm)'.format(
                ', '.join('{}={}'.format(k, v)
                          for k, v in sorted(fc['reasons'].items())),
                fc['avg_position_change'] * 1e3))

    # 5. view_handshake_poses.py / view_handshake_motion.py (--viewer 時のみ)
    if args.viewer:
        if args.plan_motion:
            print('[5/5] view_handshake_motion.py の viser ビューアを起動 '
                  'します。確認が終わったらビューアを閉じるか Ctrl-C して '
                  'ください。')
            subprocess.run([
                sys.executable,
                os.path.join(_THIS_DIR, 'view_handshake_motion.py'),
                '--skeleton-dir', skeleton_dir,
                '--handshake-dir', handshake_dir,
                '--motion-dir', motion_dir], cwd=_THIS_DIR)
        else:
            print('[5/5] view_handshake_poses.py の viser ビューアを起動 '
                  'します。確認が終わったらビューアを閉じるか Ctrl-C して '
                  'ください。')
            subprocess.run([
                sys.executable,
                os.path.join(_THIS_DIR, 'view_handshake_poses.py'),
                '--skeleton-dir', skeleton_dir,
                '--handshake-dir', handshake_dir], cwd=_THIS_DIR)
    else:
        print('[5/5] view_handshake_poses.py/view_handshake_motion.py: '
              '--viewer 未指定のためスキップしました。')


if __name__ == '__main__':
    main()
