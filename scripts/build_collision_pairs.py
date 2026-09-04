#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``collision_pairs.json`` (``solve_palm_ik.py --collision-pairs`` に
渡す、干渉回避で実際にチェックするリンクの組み合わせの JSON) を、事前に
別途作った分析結果 JSON を介さずに、以下の手順を自動で繰り返して作る。

1. ``generate_random_human_poses.py`` で人物 (既定 100 人) を生成する。
2. ``estimate_palm_poses.py`` で各人物の掌の位置姿勢を推定する。
3. 干渉回避無し (``solve_palm_ik.py --no-human-collision`` 相当。
   ``--collision-pairs`` に存在しないパスを渡すことで自己干渉・人体との
   干渉の両方を無効にする) で全員の IK を解く。
4. 3. (または直前の反復) の結果 (``analyze_collision_pairs.
   analyze_handshake_dir``) を集計し、実際に (指定した距離未満まで)
   近づいた -- 干渉した -- リンクの組み合わせの候補のうち、まだ
   ``collision_pairs.json`` に無く、かつ最も多くの人数で干渉した
   組み合わせを 1 つだけ選んで追加する (一度に全候補を追加するのではなく
   1 反復 1 組ずつ)。
5. 4. で更新した ``collision_pairs.json`` を使って全員の IK を解き直し、
   その所要時間を (掌が見つからず IK 対象外だった人物を除いた) 人数で
   割った「1 人あたりの IK 計算時間」を測る。この時間が ``--max-ik-
   seconds-per-person`` を超えた場合はここで終了する。超えていなければ
   5. の結果を 4. に戻して繰り返す。新たな干渉ペアの候補が見つからなく
   なった場合もそこで終了する (収束)。

``solve_palm_ik.py`` は「``--collision-pairs`` に指定した JSON が存在
しなければ、自己干渉・人体との干渉の両方を無効にする」という仕様
(``solve_palm_ik.load_collision_pairs`` 呼び出し部分参照) を利用して、
3. の「干渉回避無し」を実現している。

Usage
-----
    python3 build_collision_pairs.py

既存の (README 記載の) パイプラインと同じ既定ディレクトリ
(``random_human_poses/``/``random_palm_poses/``) を使い、``solve_palm_
ik.py`` の入出力には ``random_handshake_poses/`` を使う。
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from solve_palm_ik import HUMAN_FRONT_DISTANCE  # noqa: E402

from analyze_collision_pairs import analyze_handshake_dir  # noqa: E402


def load_pairs(path):
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        pair_names = json.load(f)
    return {tuple(pair) for pair in pair_names}


def save_pairs(pairs, path):
    with open(path, 'w') as f:
        json.dump([list(pair) for pair in sorted(pairs)], f,
                  indent=2, ensure_ascii=False)


def count_ik_targets(handshake_dir):
    """``handshake_dir`` の IK 結果のうち、掌が見つからず (``offered_hand``
    が null 等で) IK をスキップされた人物を除いた、実際に IK を解いた
    人数を返す (``solve_palm_ik.not_target_result`` が保存する ``target:
    false`` の人物を除外する)。"""
    n_targets = 0
    for path in glob.glob(os.path.join(handshake_dir, '*.json')):
        with open(path) as f:
            result = json.load(f)
        if result.get('target', True):
            n_targets += 1
    return n_targets


def find_collision_candidates(handshake_dir, skeleton_dir,
                              human_front_distance, dist_threshold):
    """``handshake_dir`` の IK 結果を集計し、``dist_threshold`` [m] 未満
    まで近づいた (干渉した) 組み合わせごとに、干渉したサンプル (人物) 数を
    ``{(名前A, 名前B): 人数, ...}`` の dict で返す。サンプルが 1 つも
    集計できなければ空の dict を返す。"""
    stats = analyze_handshake_dir(
        handshake_dir, skeleton_dir,
        human_front_distance=human_front_distance,
        dist_threshold=dist_threshold)
    if stats['n_samples'] == 0:
        return {}
    counts = dict(stats['self_collision_count'])
    for key, count in stats['human_collision_count'].items():
        counts[key] = counts.get(key, 0) + count
    return counts


def run(cmd):
    """``cmd`` を実行する。呼び出し先 (generate_random_human_poses.py/
    estimate_palm_poses.py/solve_palm_ik.py) が標準出力に print した文字列
    は、このスクリプト自身の進捗表示と混ざらないよう表示しない。呼び出し
    先がエラー終了した場合のみ、原因調査のためその出力を表示する。"""
    print('+ {}'.format(' '.join(cmd)))
    result = subprocess.run(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    if result.returncode != 0:
        sys.stdout.buffer.write(result.stdout)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def solve_ik(python, human_poses_dir, palm_poses_dir, handshake_dir,
            collision_pairs_path, robot_arm, seed, extra_args):
    cmd = [python, os.path.join(_THIS_DIR, 'solve_palm_ik.py'),
          '--input-dir', palm_poses_dir,
          '--output-dir', handshake_dir,
          '--skeleton-dir', human_poses_dir,
          '--collision-pairs', collision_pairs_path,
          '--robot-arm', robot_arm]
    if seed is not None:
        cmd += ['--seed', str(seed)]
    cmd += extra_args
    run(cmd)


def timed_solve_ik(python, human_poses_dir, palm_poses_dir, handshake_dir,
                   collision_pairs_path, robot_arm, seed, extra_args):
    """``solve_ik`` を実行し、所要時間 [秒] も返す。"""
    start = time.time()
    solve_ik(python, human_poses_dir, palm_poses_dir, handshake_dir,
             collision_pairs_path, robot_arm, seed, extra_args)
    return time.time() - start


def main():
    parser = argparse.ArgumentParser(
        description='人物生成 -> 掌推定 -> 干渉回避無しの IK -> 最も多くの '
                    '人数で干渉したリンクの組み合わせを 1 組ずつ追加 -> '
                    '干渉回避ありの IK による再抽出 (収束するか、1 人あたり '
                    'の IK 計算時間が上限を超えるまで反復) を自動で行い、'
                    'collision_pairs.json (solve_palm_ik.py --collision-'
                    'pairs 用) を作る。')
    parser.add_argument(
        '--num-samples', type=int, default=100,
        help='生成する人物の数 (既定 100。README のパイプライン手順 1 '
            'と同じ)。')
    parser.add_argument(
        '--human-poses-dir', type=str, default=None,
        help='人物の骨格 JSON のディレクトリ (既定は一時ディレクトリを '
            '自動作成し、プログラム終了時に削除する。既存のディレクトリを '
            '再利用して --skip-generate で手順 1・2 を省略したい場合は '
            'ここに明示的にパスを指定すること。その場合は終了時に削除 '
            'されない)。')
    parser.add_argument(
        '--palm-poses-dir', type=str, default=None,
        help='掌の位置姿勢 JSON のディレクトリ (既定は一時ディレクトリを '
            '自動作成し、プログラム終了時に削除する。明示的にパスを指定 '
            'した場合は終了時に削除されない)。')
    parser.add_argument(
        '--handshake-dir', type=str, default=None,
        help='solve_palm_ik.py の出力ディレクトリ (既定は一時ディレクトリ '
            'を自動作成し、プログラム終了時に削除する。反復のたびに上書き '
            'される。明示的にパスを指定した場合は終了時に削除されない)。')
    parser.add_argument(
        '--output', type=str,
        default=os.path.join(_THIS_DIR, 'collision_pairs.json'),
        help='書き出す干渉ペア JSON のパス (既定 collision_pairs.json。'
            'solve_palm_ik.py --collision-pairs の既定パスと同じ)。既に '
            'このファイルがあれば、そこに含まれる組み合わせから続きを '
            '積み上げる。')
    parser.add_argument(
        '--skip-generate', action='store_true',
        help='手順 1・2 (人物生成・掌推定) を省略し、既存の --human-poses-'
            'dir/--palm-poses-dir をそのまま使う。')
    parser.add_argument(
        '--collision-dist-threshold', type=float, default=0.0,
        help='この距離 [m] 未満まで近づいた組み合わせを「干渉した」と '
            'みなして collision_pairs.json に追加する (既定 0.0 = 実際に '
            '干渉用メッシュ同士がめり込んだ組み合わせのみ)。')
    parser.add_argument(
        '--max-ik-seconds-per-person', type=float, required=True,
        help='干渉ペアを 1 組追加するたびに干渉回避ありで IK を解き直し、'
            'その所要時間を (掌が見つからず IK 対象外だった人物を除いた) '
            '人数で割った「1 人あたりの IK 計算時間」を測る。干渉ペアが '
            '増えてこの時間 [秒] を超えたら、そこで反復を終了する (新たな '
            '干渉ペアの候補が見つからなくなった場合はそれより前に収束して '
            '終了する)。')
    parser.add_argument(
        '--human-front-distance', type=float, default=HUMAN_FRONT_DISTANCE,
        help='solve_palm_ik.py に渡すのと同じ --human-front-distance '
            '(既定 {:.1f})。'.format(HUMAN_FRONT_DISTANCE))
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='solve_palm_ik.py に渡す --robot-arm (既定 auto)。')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='solve_palm_ik.py に渡す --seed (既定は指定なし)。')
    parser.add_argument(
        '--python', type=str, default=sys.executable,
        help='generate_random_human_poses.py/estimate_palm_poses.py/'
            'solve_palm_ik.py を呼び出す Python インタプリタ (既定は '
            'このスクリプトと同じインタプリタ)。')
    parser.add_argument(
        '--solve-arg', dest='solve_args', action='append', default=[],
        help='solve_palm_ik.py にそのまま追加で渡す引数 (例: --solve-arg '
            '--attempts-per-pose --solve-arg 8)。複数回指定できる。')
    args = parser.parse_args()

    # collision_pairs.json (args.output) を除き、このプログラムが生成する
    # JSON (人物・掌・IK 結果、および「干渉回避無し」用のダミーパス) は
    # すべてここに作る一時ディレクトリの下に置き、終了時 (正常終了・
    # エラー終了のいずれでも) に丸ごと削除する。--human-poses-dir 等を
    # 明示的に指定した場合はそのディレクトリを削除しない (既存データの
    # 再利用・--skip-generate との併用を想定)。
    temp_dir = tempfile.mkdtemp(prefix='build_collision_pairs_')
    try:
        if args.human_poses_dir is None:
            args.human_poses_dir = os.path.join(
                temp_dir, 'random_human_poses')
        if args.palm_poses_dir is None:
            args.palm_poses_dir = os.path.join(
                temp_dir, 'random_palm_poses')
        if args.handshake_dir is None:
            args.handshake_dir = os.path.join(
                temp_dir, 'random_handshake_poses')
        nonexistent_collision_pairs = os.path.join(
            temp_dir, 'no_collision_pairs.json')

        if not args.skip_generate:
            run([args.python,
                os.path.join(_THIS_DIR, 'generate_random_human_poses.py'),
                '--num-samples', str(args.num_samples),
                '--output-dir', args.human_poses_dir])
            run([args.python,
                os.path.join(_THIS_DIR, 'estimate_palm_poses.py'),
                '--input-dir', args.human_poses_dir,
                '--output-dir', args.palm_poses_dir])

        pairs = load_pairs(args.output)
        if pairs:
            print('{} から既存の干渉ペア {} 組を読み込みました。'.format(
                args.output, len(pairs)))

        n_people = len(
            glob.glob(os.path.join(args.palm_poses_dir, '*.json')))
        if n_people == 0:
            print('{} に人物が見つかりません。'.format(args.palm_poses_dir))
            sys.exit(1)

        print('\n=== 手順 3: 干渉回避無しで IK を解く ===')
        solve_ik(args.python, args.human_poses_dir, args.palm_poses_dir,
                 args.handshake_dir, nonexistent_collision_pairs,
                 args.robot_arm, args.seed, args.solve_args)

        # 掌が見つからず (offered_hand が null 等で) IK をスキップされた
        # 人物は、以降の「1 人あたりの IK 計算時間」の母数から除外する。
        n_ik_people = count_ik_targets(args.handshake_dir)
        if n_ik_people == 0:
            print('{} に IK 対象の人物 (掌が見つかった人物) が見つかりません。'
                 .format(args.handshake_dir))
            sys.exit(1)
        if n_ik_people != n_people:
            print('{} 人中 {} 人は掌が見つからず IK 対象外だったため、1 人 '
                  'あたりの IK 計算時間の母数は {} 人とします。'.format(
                      n_people, n_people - n_ik_people, n_ik_people))

        iteration = 0
        while True:
            iteration_start = time.time()
            candidates = find_collision_candidates(
                args.handshake_dir, args.human_poses_dir,
                args.human_front_distance, args.collision_dist_threshold)
            candidates = {key: count for key, count in candidates.items()
                         if key not in pairs}
            if not candidates:
                print('\n新たな干渉ペアの候補が見つかりませんでした。収束'
                      'したので終了します。')
                break

            iteration += 1
            best_pair = max(candidates, key=lambda key: candidates[key])
            best_count = candidates[best_pair]
            pairs.add(best_pair)
            save_pairs(pairs, args.output)
            print('\n=== 反復 {}: 干渉ペアを 1 組追加 ==='.format(iteration))
            print('新規 1 組を追加: {} ({} / {} 人で干渉) (合計 {} 組) -> {}'
                 .format(best_pair, best_count, n_ik_people, len(pairs),
                         args.output))

            print('干渉回避ありで IK を解き直します (現在 {} 組)。'.format(
                len(pairs)))
            elapsed = timed_solve_ik(
                args.python, args.human_poses_dir, args.palm_poses_dir,
                args.handshake_dir, args.output, args.robot_arm, args.seed,
                args.solve_args)
            per_person = elapsed / n_ik_people
            print('IK 計算時間: {:.2f} 秒 ({:.3f} 秒/人 x {} 人)。'.format(
                elapsed, per_person, n_ik_people))

            iteration_elapsed = time.time() - iteration_start
            print('反復 {} の所要時間: {:.2f} 秒。'.format(
                iteration, iteration_elapsed))
            print('反復 {} 終了: 干渉ペア {} 個, 1 人あたり {:.3f} 秒。'
                 .format(iteration, len(pairs), per_person))

            if per_person > args.max_ik_seconds_per_person:
                print('1 人あたりの IK 計算時間が上限 ({:.2f} 秒) を超えた '
                      'ため終了します。'.format(
                          args.max_ik_seconds_per_person))
                break

        print('\n最終的な干渉ペア数: {} -> {}'.format(len(pairs), args.output))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
