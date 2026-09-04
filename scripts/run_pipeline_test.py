#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""README.md のパイプラインのうち 1 (generate_random_human_poses.py) ->
2 (estimate_palm_poses.py) -> 4 (solve_palm_ik.py) -> 5 (view_handshake_
poses.py, --viewer 指定時のみ) を順に実行する。

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
import os
import subprocess
import sys
import tempfile

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import json_io  # noqa: E402  (パス追加後に import)


def run_step(label, script_name, extra_args):
    """``script_name`` を子プロセスで実行する。

    人物ごとの進捗行など生の標準出力はそのまま流さず捕捉するだけにし、
    失敗したとき (exit code != 0) だけ末尾を表示してから中断する。
    """
    print('[{}] 実行中...'.format(label))
    script_path = os.path.join(_THIS_DIR, script_name)
    result = subprocess.run(
        [sys.executable, script_path] + extra_args, cwd=_THIS_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print('[{}] {} が exit code {} で失敗しました。'.format(
            label, script_name, result.returncode))
        sys.exit(1)


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
    n_total = n_target = n_solved = n_post_process = 0
    for _, data in load_json_files(handshake_dir):
        n_total += 1
        if not data.get('target'):
            continue
        n_target += 1
        if data.get('solved'):
            n_solved += 1
            if data.get('post_process') is not None:
                n_post_process += 1
    return dict(n_total=n_total, n_target=n_target, n_solved=n_solved,
               n_post_process=n_post_process)


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
        help='ステップ 5 (view_handshake_poses.py) の viser ビューアを '
            '実際に起動する。既定ではブラウザ接続を待ち続けて自動実行が '
            '止まってしまうため起動しない。')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='generate_random_human_poses.py (ステップ 1) に渡す乱数 '
            'シード (既定は指定なし)。')
    args = parser.parse_args()

    base_dir = tempfile.mkdtemp(prefix='aero_demo_pipeline_', dir='/tmp')
    skeleton_dir = os.path.join(base_dir, 'skeletons')
    palm_dir = os.path.join(base_dir, 'palms')
    handshake_dir = os.path.join(base_dir, 'handshakes')
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
    run_step('2/5', 'estimate_palm_poses.py',
             ['--input-dir', skeleton_dir, '--output-dir', palm_dir])
    n_palms, offered = summarize_palms(palm_dir)
    print('[2/5] estimate_palm_poses.py: 掌の位置姿勢 JSON を {} 件推定 '
          '(offered_hand: R={}, L={}, null={})'.format(
              n_palms, offered.get('R', 0), offered.get('L', 0),
              offered.get(None, 0)))

    # 4. solve_palm_ik.py
    run_step('4/5', 'solve_palm_ik.py', [
        '--input-dir', palm_dir, '--output-dir', handshake_dir,
        '--skeleton-dir', skeleton_dir])
    summary = summarize_handshakes(handshake_dir)
    print('[4/5] solve_palm_ik.py: IK 対象 {} 人中 {} 人 solved '
          '(対象外 {} 人)'.format(
              summary['n_target'], summary['n_solved'],
              summary['n_total'] - summary['n_target']))

    # 5. view_handshake_poses.py (--viewer のときだけ実際に起動する)
    if args.viewer:
        print('[5/5] view_handshake_poses.py の viser ビューアを起動 '
              'します。確認が終わったらビューアを閉じるか Ctrl-C して '
              'ください。')
        subprocess.run([
            sys.executable,
            os.path.join(_THIS_DIR, 'view_handshake_poses.py'),
            '--skeleton-dir', skeleton_dir,
            '--handshake-dir', handshake_dir], cwd=_THIS_DIR)
    else:
        print('[5/5] view_handshake_poses.py: --viewer 未指定のため '
              'スキップしました。')

    print()
    print('=== 結果 ===')
    print('生成した骨格人数: {}'.format(n_generated))
    print('後処理まで含めて成功した人数: {} / {}'.format(
        summary['n_post_process'], n_generated))


if __name__ == '__main__':
    main()
