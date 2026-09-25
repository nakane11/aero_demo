#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""README.md のパイプラインのうち 1 (generate_random_human_poses.py) ->
2 (estimate_palm_poses.py) -> 4 (solve_palm_ik.py) -> 5 (view_handshake_
poses.py または --plan-motion 指定時は view_handshake_motion.py、いずれも
--viewer 指定時のみ) を順に実行する。

各ステップの入出力 JSON は /tmp 以下に作る一時ディレクトリに保存・
読み出しし、各スクリプトの (人物ごとの) 生の画面出力はそのまま流さず、
ステップごとの結果だけを簡潔に表示する。実装を変更するたびにこのパイプ
ラインが最後まで通ることを確認する回帰テストを兼ねており、最後に
「後処理まで含めて成功した人数」を表示する。

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
    """``script_name`` を子プロセスで実行し、``(壁時計時間 [秒], 標準出力)``
    を返す。

    人物ごとの進捗行など生の標準出力はそのまま流さず捕捉するだけにし、
    失敗したとき (exit code != 0) だけ末尾を表示してから中断する。
    """
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
    """``solve_palm_ik.py`` の標準出力から ``_warmup_batch_ik`` が出す
    ``[warmup] ...`` 行を抜き出し、``(行のリスト, 合計秒数)`` を返す。
    ``--no-warmup`` 指定時など該当行が無ければ ``([], 0.0)``。
    """
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
    """``solve_palm_ik.py`` の出力 JSON を集計する。

    ``collision_ik_time``/``candidate_selection_time`` は IK 対象になった
    人物 (``target: true``) なら solved/unsolved を問わず必ず記録される
    (``solve_palm_ik.solved_result``/``unsolved_result`` 参照)。
    ``candidate_selection_time`` は事後の干渉検証と、棄却された候補も
    含めた全ての後処理判定 (``solve_post_process``) 呼び出しの合計であり、
    これが「実質的な IK 2 段階目」の時間になる -- ``post_process``
    キー配下の ``compute_time`` (採用された最後の 1 回分だけ) は棄却
    された候補の分を含まず過小評価になるため、ここでは集計しない。
    """
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


def summarize_motions(motion_dir):
    """``plan_handshake_motion.py`` の出力 JSON を集計する。

    ``planned`` が ``false`` (IK 対象外/IK 失敗) の人物は集計から除く。
    ``kind`` は採用した軌道の作り方 (``pretouch``/``linear``/
    ``optimized``、``plan_handshake_motion.KIND_LABELS`` 参照)。
    """
    n_planned = n_verified = n_lead_in_verified = n_both_verified = 0
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
        kind = data.get('kind')
        kinds[kind] = kinds.get(kind, 0) + 1
        if data.get('approach_angle') is not None:
            angle = int(round(math.degrees(data['approach_angle'])))
            approach_angles[angle] = approach_angles.get(angle, 0) + 1
        if data.get('compute_time') is not None:
            compute_times.append(data['compute_time'])

    # 軌道最適化 (jaxls) を要した人だけの平均は、その人数・顔ぶれが実行の
    # たびに変わる (pre-touch/線形補間の候補が厳密検証を通るかどうかは
    # GPU 上の jax 計算の非決定性の影響を受けうる境界ケースがあるため) 母
    # 集団に依存する指標になってしまい安定した比較に向かない。そのため
    # ここでは最適化が不要だった人も含めた全 planned 人物の compute_time
    # の平均だけを返す。
    avg_compute_time = (sum(compute_times) / len(compute_times)
                        if compute_times else None)
    return dict(n_planned=n_planned, n_verified=n_verified,
               n_lead_in_verified=n_lead_in_verified,
               n_both_verified=n_both_verified, kinds=kinds,
               approach_angles=approach_angles,
               avg_compute_time=avg_compute_time)


def main():
    parser = argparse.ArgumentParser(
        description='README.md のパイプライン 1/2/4/5 を順に実行する '
                    '回帰テスト。/tmp 以下に一時ディレクトリを作って '
                    'JSON を保存・読み出しし、各ステップの結果だけを '
                    '簡潔に表示する。')
    parser.add_argument(
        'num_people', type=int,
        help='generate_random_human_poses.py (ステップ 1) で生成する人数。')
    parser.add_argument(
        '--viewer', action='store_true',
        help='ステップ 5 (view_handshake_poses.py、--plan-motion 指定時は '
            'view_handshake_motion.py) の viser ビューアを実際に起動する。'
            '既定ではブラウザ接続を待ち続けて自動実行が止まってしまうため '
            '起動しない。')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='generate_random_human_poses.py (ステップ 1) に渡す乱数 '
            'シード (既定は指定なし)。')
    parser.add_argument(
        '--plan-motion', action='store_true',
        help='ステップ 4.5 (plan_handshake_motion.py) を実行し、握手の '
            '最終姿勢だけでなくロボットの初期姿勢からそこへ至る干渉回避 '
            '付きの軌道も生成する。jaxls (pip install "git+https://'
            'github.com/brentyi/jaxls.git") が別途必要で、1 人あたり '
            '数十秒かかるため既定では実行しない。')
    parser.add_argument(
        '--force-optimize', action='store_true',
        help='--plan-motion 指定時、plan_handshake_motion.py に '
            '--force-optimize を渡す (pre-touch/線形補間の候補が事後検証 '
            'に通っていても必ず jaxls の軌道最適化まで実行させ、全員分の '
            '軌道最適化の計算時間を計測できるようにする)。')
    parser.add_argument(
        '--initial-base-pose', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'YAW'),
        help='--plan-motion 指定時、plan_handshake_motion.py に '
            '--initial-base-pose として渡すロボットの初期台車姿勢 '
            '(既定は指定なし = 原点。人物は (3, 0) 付近に置かれるので、'
            '例えば 5 0 3.14 で人の向こう側から回り込む経路を試せる)。')
    parser.add_argument(
        '--approach-distance', type=float, default=None,
        help='--plan-motion 指定時、plan_handshake_motion.py に '
            '--approach-distance として渡す (既定は指定なし)。')
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
    solve_elapsed, solve_stdout = run_step('4/5', 'solve_palm_ik.py', [
        '--input-dir', palm_dir, '--output-dir', handshake_dir,
        '--skeleton-dir', skeleton_dir])
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
    print('  (上記の IK 1/2段階目は solve_palm_ik.py が出力する JSON の '
          'collision_ik_time/candidate_selection_time の平均であり、'
          'warmup (jax の JIT トレース/コンパイル、人物ループに入る前に '
          '1 回だけ払う) の時間は含まない。warmup 自体の実測は '
          '[4/5] 実行直後の行を参照。IK 1段階目が warmup 後も長い場合は '
          'warmup 漏れではなく、人物ごとに再コンパイルが起きている '
          '(docs/jax_compilation_cache.md 参照) 可能性を疑うこと。)')

    # 5. view_handshake_poses.py / view_handshake_motion.py
    #    (--viewer のときだけ実際に起動する。--plan-motion も指定されて
    #    いれば軌道再生付きの view_handshake_motion.py を、そうでなければ
    #    最終姿勢のみの view_handshake_poses.py を開く)
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
