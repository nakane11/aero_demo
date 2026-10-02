#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``scripts/collision_pairs.json`` (``solve_palm_ik.py --collision-pairs``
に渡す、干渉回避で実際にチェックするリンクの組み合わせの JSON。本番の
``scripts/`` 側が読む既定ファイルそのものを更新する、開発用ツール) を、
以下の手順で自動的に作る。

1. ``generate_random_human_poses.py`` で人物 (既定 100 人) を生成する。
2. ``estimate_palm_poses.py`` で各人物の掌の位置姿勢を推定する。
3. 干渉回避無し (``solve_palm_ik.py --collision-pairs`` に存在しない
   パスを渡すことで自己干渉・人体との干渉の両方と、事後の干渉検証を
   まとめて無効にする) で全員の IK を 1 回だけ解く。
4. 3. の結果を ``analyze_handshake_dir`` で集計し、事後検証
   (``solve_palm_ik.pick_verified_candidate``) と同じ組み合わせ・同じ
   判定で貫通していた -- 事後検証なら棄却される -- 組み合わせを、干渉した
   人数が多い順に並べたランキングを作る。自己干渉は ``self_collision_
   depth`` の貫通の深さで測り、``self_collision_ignored`` の除外ルールで
   絞った組だけを見る。
5. このランキングの上位 ``--num-pairs`` 組 (``--include-pairs`` の組を
   足したもの) を ``collision_pairs.json`` として書き出す。
   ``--max-ik-seconds-per-person`` を指定した場合は、代わりにランキングの
   先頭から 1・2・3... 組と増やしながら干渉回避ありで実際に IK を解き直し
   (``solve_palm_ik.py`` の通常のフルパイプライン、事後検証・後処理を
   含む)、1 人あたりの平均計算時間 (warmup を除く) が指定秒数を超えた
   直前の組数を採用する。

この手順は「干渉回避を全く行わない解に、実際にどのリンクの組み合わせが
どれだけの頻度で干渉するか」という一度きりの統計だけでランキングを作る
(1 組追加するたびに解き直して統計を取り直す反復はしない。干渉回避ありの
解は事後検証を通ったものだけが採用されるため、採用された解の統計はすぐ
「もう干渉していない」ように見えてしまう)。

``solve_palm_ik.py`` は「``--collision-pairs`` に指定した JSON が存在
しなければ、自己干渉・人体との干渉の両方と事後の干渉検証を無効にする」
という仕様 (``solve_palm_ik.load_collision_pairs`` 呼び出し部分参照) を
利用して、3. の「干渉回避無し」を実現している。

組数・選び方の比較は ``tools/grid_search_collision_ik.py`` で行う
(docs/ik_and_motion_constraints.md の 0 節)。

Usage
-----
    python3 tools/build_collision_pairs.py --num-samples 2000 --seed 10 \\
        --num-pairs 8 --ranking-output /tmp/ranking.json

人物・掌・IK 結果は、``--human-poses-dir`` 等を指定しなければ一時
ディレクトリに作り、終了時に削除する。
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from solve_palm_ik import (  # noqa: E402
    DEFAULT_COLLISION_VERIFY_TOLERANCE, HUMAN_FRONT_DISTANCE,
    apply_collision_model, build_collision_verification_pairs,
    collision_pair_distances, collision_pair_name, cylinder_surface_samples,
    human_body_obstacles, human_translation_offset, load_skeleton_json,
    lock_fixed_joints, restrict_elbow_range, restrict_leg_range,
    restrict_waist_range, translate_joint_positions)

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.models import Aero  # noqa: E402


def build_robot():
    """``solve_palm_ik.main`` と同じ手順でロボット (指なし) を作る。"""
    robot = Aero(use_hand=False)
    restrict_elbow_range(robot)
    restrict_leg_range(robot)
    restrict_waist_range(robot)
    lock_fixed_joints(robot)
    apply_collision_model(robot)
    return robot


def apply_result_pose(robot, result, angle_vector):
    """``solve_palm_ik.py`` の結果 ``result`` の台車位置と関節角
    ``angle_vector`` をロボットに反映する。"""
    robot.reset_pose()
    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    robot.angle_vector(np.asarray(angle_vector))
    robot.newcoords(Coordinates(
        pos=result['base_position']).rotate(result['base_yaw'], 'z'))


def analyze_handshake_dir(handshake_dir, skeleton_dir,
                          human_front_distance=HUMAN_FRONT_DISTANCE,
                          dist_threshold=-DEFAULT_COLLISION_VERIFY_TOLERANCE,
                          robot=None):
    """``handshake_dir`` (``solve_palm_ik.py`` の出力) と ``skeleton_dir``
    (骨格 JSON) を読み、組み合わせごとの「全サンプル中の最小距離」と
    「``dist_threshold`` [m] 未満だった (干渉した) サンプル数」を集計する。

    組み合わせと距離は事後検証 (``solve_palm_ik.pick_verified_candidate``)
    と同じ。組み合わせは ``build_collision_verification_pairs`` (自己干渉は
    ``self_collision_ignored`` の除外ルールで絞った組、人体との干渉は全
    リンク × 人体 26 本)、距離は ``collision_pair_distances`` (自己干渉は
    ``self_collision_depth`` による貫通の深さ、人体とは ``human_body_
    obstacles`` の円柱への入り込みの深さ。貫通していれば負) で測る。
    姿勢も事後検証に合わせ、hover 姿勢 (``joint_angle_vector``) は全組、
    押し込み姿勢 (``post_process``) は自己干渉の組だけを見て、小さい方を
    その人の距離とする。既定の ``dist_threshold`` は事後検証の棄却条件
    (``DEFAULT_COLLISION_VERIFY_TOLERANCE`` より深い貫通) と同じ。

    Returns
    -------
    dict
        ``min_dist``/``collision_count`` (キーは ``(名前A, 名前B)`` の
        タプル、値は最小距離 [m]/干渉した人数)、``n_samples`` を持つ dict。
    """
    if robot is None:
        robot = build_robot()
    pairs = build_collision_verification_pairs(robot, 'r')
    names = [collision_pair_name(pair) for pair in pairs]
    is_self = np.array([not isinstance(other, int) for _, other in pairs])

    min_dist = {}
    collision_count = {}
    n_samples = 0
    files = sorted(glob.glob(os.path.join(handshake_dir, '*.json')))
    for path in files:
        with open(path) as f:
            result = json.load(f)
        if not result.get('solved'):
            continue
        skeleton_path = os.path.join(skeleton_dir, os.path.basename(path))
        if not os.path.exists(skeleton_path):
            continue
        n_samples += 1

        joint_positions = load_skeleton_json(skeleton_path)
        offset = human_translation_offset(
            joint_positions, front_distance=human_front_distance)
        joint_positions = translate_joint_positions(joint_positions, offset)
        obstacles = human_body_obstacles(joint_positions)
        samples = [cylinder_surface_samples(o) for o in obstacles]

        apply_result_pose(robot, result, result['joint_angle_vector'])
        dists = np.asarray(collision_pair_distances(
            robot, pairs, joint_positions, obstacle_links=obstacles,
            obstacle_samples=samples))
        post = result.get('post_process')
        if post is not None:
            apply_result_pose(robot, result, post['joint_angle_vector'])
            press = np.full(len(pairs), np.inf)
            press[is_self] = collision_pair_distances(
                robot, [p for p, s in zip(pairs, is_self) if s],
                joint_positions, obstacle_links=obstacles,
                obstacle_samples=samples)
            dists = np.minimum(dists, press)

        for key, dist in zip(names, dists):
            if key not in min_dist or dist < min_dist[key]:
                min_dist[key] = float(dist)
            if dist < dist_threshold:
                collision_count[key] = collision_count.get(key, 0) + 1

    return dict(min_dist=min_dist, collision_count=collision_count,
                n_samples=n_samples)


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


def rank_collision_candidates(stats, exclude=()):
    """``analyze_handshake_dir`` の結果 ``stats`` のうち干渉した人数が 1 人
    以上の組み合わせを、人数の多い順 (同数なら最小距離の小さい順、さらに
    名前の辞書順) に並べた ``[(名前A, 名前B), ...]`` を返す (``exclude``
    に含まれる組み合わせは除く)。"""
    counts = stats['collision_count']
    return sorted((pair for pair in counts if pair not in exclude),
                  key=lambda pair: (-counts[pair], stats['min_dist'][pair],
                                    pair))


def select_pairs(ranking, num_pairs, include=()):
    """ランキングの上位 ``num_pairs`` 組に、``include`` の組 (ランキング外
    でもよい) を足した集合を返す (``include`` は ``num_pairs`` に数えない)。"""
    return set(ranking[:num_pairs]) | set(include)


def save_ranking(stats, ranking, path):
    """ランキングを ``[{"pair": [A, B], "count": 人数, "min_dist": 距離},
    ...]`` の JSON として保存する (``n_samples`` も持たせる)。"""
    with open(path, 'w') as f:
        json.dump(dict(
            n_samples=stats['n_samples'],
            ranking=[dict(pair=list(pair),
                          count=stats['collision_count'][pair],
                          min_dist=stats['min_dist'][pair])
                     for pair in ranking]), f, indent=2, ensure_ascii=False)


def mean_ik_time_per_person(handshake_dir):
    """``solve_palm_ik.py`` の結果 JSON の ``collision_ik_time`` +
    ``candidate_selection_time`` (warmup を含まない) の、IK 対象 1 人あたりの
    平均 [秒]。"""
    times = []
    for path in glob.glob(os.path.join(handshake_dir, '*.json')):
        with open(path) as f:
            result = json.load(f)
        if result.get('target', True):
            times.append(result.get('collision_ik_time', 0.0)
                         + result.get('candidate_selection_time', 0.0))
    return float(np.mean(times)) if times else float('nan')


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
            collision_pairs_path, robot_arm, extra_args):
    cmd = [python, os.path.join(_SCRIPTS_DIR, 'solve_palm_ik.py'),
          '--input-dir', palm_poses_dir,
          '--output-dir', handshake_dir,
          '--skeleton-dir', human_poses_dir,
          '--collision-pairs', collision_pairs_path,
          '--robot-arm', robot_arm]
    cmd += extra_args
    run(cmd)


def timed_solve_ik(python, human_poses_dir, palm_poses_dir, handshake_dir,
                   collision_pairs_path, robot_arm, extra_args):
    """``solve_ik`` を実行し、IK 対象 1 人あたりの平均計算時間 [秒]
    (``mean_ik_time_per_person``、warmup を含まない) を返す。"""
    solve_ik(python, human_poses_dir, palm_poses_dir, handshake_dir,
             collision_pairs_path, robot_arm, extra_args)
    return mean_ik_time_per_person(handshake_dir)


def main():
    parser = argparse.ArgumentParser(
        description='人物生成 -> 掌推定 -> 干渉回避無しの IK を 1 回解いて '
                    '干渉したリンクの組み合わせを頻度順にランキング -> '
                    '上位 --num-pairs 組 (または --max-ik-seconds-per-'
                    'person を満たす組数) を採用、を自動で行い、'
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
            'を自動作成し、プログラム終了時に削除する。既定動作 (手順3) '
            'の出力に使い、``--max-ik-seconds-per-person`` 指定時は組数を '
            '変えて解き直すたびに上書きされる。明示的にパスを指定した '
            '場合は終了時に削除されない)。')
    parser.add_argument(
        '--output', type=str,
        default=os.path.join(_SCRIPTS_DIR, 'collision_pairs.json'),
        help='書き出す干渉ペア JSON のパス (既定 scripts/collision_pairs.'
            'json。solve_palm_ik.py --collision-pairs の既定パスと同じ、'
            '本番が読む実体そのものを更新する)。')
    parser.add_argument(
        '--skip-generate', action='store_true',
        help='手順 1・2 (人物生成・掌推定) を省略し、既存の --human-poses-'
            'dir/--palm-poses-dir をそのまま使う。')
    parser.add_argument(
        '--collision-dist-threshold', type=float,
        default=-DEFAULT_COLLISION_VERIFY_TOLERANCE,
        help='距離 (貫通していれば負) がこの値 [m] 未満だった組み合わせを '
            '「干渉した」とみなしてランキングに使う (既定 {} = 事後検証の '
            '棄却条件と同じ)。'.format(-DEFAULT_COLLISION_VERIFY_TOLERANCE))
    parser.add_argument(
        '--skip-solve', action='store_true',
        help='手順 3 (干渉回避無しの IK) を省略し、既存の --handshake-dir の '
            '結果をそのまま集計する。')
    parser.add_argument(
        '--ranking-output', type=str, default=None,
        help='ランキング (組・干渉した人数・最小距離) を保存する JSON の '
            'パス (既定は保存しない)。')
    parser.add_argument(
        '--show-ranking', type=int, default=30,
        help='ランキングの上位何組を表示するか (既定 30)。')
    parser.add_argument(
        '--include-pairs', type=str, default=None,
        help='ランキングの順位によらず必ず採用する組の JSON (collision_'
            'pairs.json と同じ形式)。--num-pairs には数えない。')
    parser.add_argument(
        '--no-verify', action='store_true',
        help='--num-pairs で採用した組で IK を解き直して時間を測る確認を '
            '省略する。')
    count_group = parser.add_mutually_exclusive_group(required=True)
    count_group.add_argument(
        '--num-pairs', type=int,
        help='干渉頻度ランキングの上位何組を採用するか。')
    count_group.add_argument(
        '--max-ik-seconds-per-person', type=float,
        help='ランキングの上位から 1・2・3... 組と増やしながら干渉回避 '
            'ありで実際に IK を解き直し (フルパイプライン、事後検証・'
            '後処理を含む)、1 人あたりの平均計算時間がこの秒数を超えた '
            '直前の組数を採用する (ランキングを使い切っても超えなければ '
            '全組を採用する)。')
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='solve_palm_ik.py に渡す --robot-arm (既定 auto)。')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='generate_random_human_poses.py に渡す --seed (既定は指定 '
            'なし)。基準の合成データ (run_pipeline_test.py の seed 0/1) '
            'とは別の seed にして、基準に合わせ込まないようにする。')
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
        candidate_output = os.path.join(temp_dir, 'candidate_pairs.json')

        if not args.skip_generate:
            gen_cmd = [args.python,
                       os.path.join(_SCRIPTS_DIR,
                                    'generate_random_human_poses.py'),
                       '--num-samples', str(args.num_samples),
                       '--output-dir', args.human_poses_dir]
            if args.seed is not None:
                gen_cmd += ['--seed', str(args.seed)]
            run(gen_cmd)
            run([args.python,
                os.path.join(_SCRIPTS_DIR, 'estimate_palm_poses.py'),
                '--input-dir', args.human_poses_dir,
                '--output-dir', args.palm_poses_dir])

        n_people = len(
            glob.glob(os.path.join(args.palm_poses_dir, '*.json')))
        if n_people == 0:
            print('{} に人物が見つかりません。'.format(args.palm_poses_dir))
            sys.exit(1)

        if not args.skip_solve:
            print('\n=== 手順 3: 干渉回避無しで IK を解く ===')
            solve_ik(args.python, args.human_poses_dir, args.palm_poses_dir,
                     args.handshake_dir, nonexistent_collision_pairs,
                     args.robot_arm, args.solve_args)

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

        print('\n=== 手順 4: 干渉頻度をランキング ===')
        stats = analyze_handshake_dir(
            args.handshake_dir, args.human_poses_dir,
            human_front_distance=HUMAN_FRONT_DISTANCE,
            dist_threshold=args.collision_dist_threshold)
        ranking = rank_collision_candidates(stats)
        if args.ranking_output is not None:
            save_ranking(stats, ranking, args.ranking_output)
            print('ランキングを保存しました -> {}'.format(args.ranking_output))
        if not ranking:
            print('干渉した組み合わせが見つかりませんでした。')
            sys.exit(1)
        print('干渉頻度の高い順に {} 組の候補が見つかりました (解けた {} 人中、'
              '上位 {} 件):'.format(len(ranking), stats['n_samples'],
                                   args.show_ranking))
        for rank, pair in enumerate(ranking[:args.show_ranking], start=1):
            print('  {:3d}. {} x {}: {} 人 (最小 {:.3f} m)'.format(
                rank, pair[0], pair[1], stats['collision_count'][pair],
                stats['min_dist'][pair]))
        include = load_pairs(args.include_pairs) if args.include_pairs \
            else set()

        print('\n=== 手順 5: 採用する組数を決定 ===')
        if args.num_pairs is not None:
            if args.num_pairs > len(ranking):
                print('警告: --num-pairs ({}) がランキングの候補数 ({}) を '
                      '超えているため、候補数だけ採用します。'.format(
                          args.num_pairs, len(ranking)))
            pairs = select_pairs(ranking, args.num_pairs, include)
            save_pairs(pairs, args.output)
            print('上位 {} 組{}を採用しました -> {}'.format(
                min(args.num_pairs, len(ranking)),
                ' + 指定の {} 組'.format(len(include)) if include else '',
                args.output))
            if not args.no_verify:
                print('検証のため、この組で干渉回避ありの IK を解き直します。')
                per_person = timed_solve_ik(
                    args.python, args.human_poses_dir, args.palm_poses_dir,
                    args.handshake_dir, args.output, args.robot_arm,
                    args.solve_args)
                print('IK 計算時間: {:.3f} 秒/人 (IK 対象 {} 人の平均、'
                      'warmup を除く)。'.format(per_person, n_ik_people))
        else:
            n_pairs = 0
            for n_pairs in range(1, len(ranking) + 1):
                pairs = select_pairs(ranking, n_pairs, include)
                save_pairs(pairs, candidate_output)
                per_person = timed_solve_ik(
                    args.python, args.human_poses_dir, args.palm_poses_dir,
                    args.handshake_dir, candidate_output, args.robot_arm,
                    args.solve_args)
                print('{} 組: {:.3f} 秒/人 (IK 対象 {} 人の平均)。'.format(
                    len(pairs), per_person, n_ik_people))
                if per_person > args.max_ik_seconds_per_person:
                    print('1 人あたりの IK 計算時間が上限 ({:.2f} 秒) を '
                         '超えたため、直前の上位 {} 組を採用します。'.format(
                             args.max_ik_seconds_per_person, n_pairs - 1))
                    n_pairs -= 1
                    break
            else:
                print('ランキングを全て試しても上限を超えませんでした。'
                     '全 {} 組を採用します。'.format(n_pairs))
            pairs = select_pairs(ranking, n_pairs, include)
            save_pairs(pairs, args.output)
            print('{} 組を採用しました -> {}'.format(len(pairs), args.output))

        print('\n最終的な干渉ペア数: {} -> {}'.format(len(pairs), args.output))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
