#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``scripts/collision_pairs.json`` (``solve_palm_ik.py --collision-pairs``
の既定ファイル) を作り直す。

人物生成 -> 掌推定 -> 干渉回避無しで IK を 1 回解き、事後検証と同じ組・
判定で貫通した組を干渉人数の多い順にランキングして上位を採用する。
``--max-ik-seconds-per-person`` 指定時は上位から組数を増やして解き直し、
1 人あたりの平均 IK 時間が上限を超える直前の組数を採用する。
干渉回避無しは、存在しない ``--collision-pairs`` を渡して実現している。

Usage
-----
    python3 tools/build_collision_pairs.py --num-samples 2000 --seed 10 \\
        --num-pairs 8 --ranking-output /tmp/ranking.json
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
    restrict_neck_range, restrict_waist_range, translate_joint_positions)

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.models import Aero  # noqa: E402


def build_robot():
    """``solve_palm_ik.main`` と同じ手順でロボット (指なし) を作る。"""
    robot = Aero(use_hand=False)
    restrict_elbow_range(robot)
    restrict_leg_range(robot)
    restrict_waist_range(robot)
    restrict_neck_range(robot)
    lock_fixed_joints(robot)
    apply_collision_model(robot)
    return robot


def apply_result_pose(robot, result, angle_vector):
    """IK 結果の台車位置と関節角 ``angle_vector`` をロボットに反映する。"""
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
    """IK 結果と骨格から、組ごとの最小距離と干渉人数を集計する。

    組・距離は事後検証と同じ (貫通なら負)。hover 姿勢は全組、押し込み姿勢は
    自己干渉の組だけを見る。距離が ``dist_threshold`` [m] 未満で干渉とみなす。

    Returns
    -------
    dict
        ``min_dist``/``collision_count`` (キーは ``(名前A, 名前B)``) と
        ``n_samples``。
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
    """実際に IK を解いた人数 (``target: false`` を除く) を返す。"""
    n_targets = 0
    for path in glob.glob(os.path.join(handshake_dir, '*.json')):
        with open(path) as f:
            result = json.load(f)
        if result.get('target', True):
            n_targets += 1
    return n_targets


def rank_collision_candidates(stats, exclude=()):
    """干渉した組を人数の多い順 (同数なら最小距離・名前順) に並べる。"""
    counts = stats['collision_count']
    return sorted((pair for pair in counts if pair not in exclude),
                  key=lambda pair: (-counts[pair], stats['min_dist'][pair],
                                    pair))


def select_pairs(ranking, num_pairs, include=()):
    """上位 ``num_pairs`` 組 + ``include`` (組数に数えない) を返す。"""
    return set(ranking[:num_pairs]) | set(include)


def save_ranking(stats, ranking, path):
    with open(path, 'w') as f:
        json.dump(dict(
            n_samples=stats['n_samples'],
            ranking=[dict(pair=list(pair),
                          count=stats['collision_count'][pair],
                          min_dist=stats['min_dist'][pair])
                     for pair in ranking]), f, indent=2, ensure_ascii=False)


def mean_ik_time_per_person(handshake_dir):
    """IK 対象 1 人あたりの平均 IK 時間 [秒] (warmup を含まない)。"""
    times = []
    for path in glob.glob(os.path.join(handshake_dir, '*.json')):
        with open(path) as f:
            result = json.load(f)
        if result.get('target', True):
            times.append(result.get('collision_ik_time', 0.0)
                         + result.get('candidate_selection_time', 0.0))
    return float(np.mean(times)) if times else float('nan')


def run(cmd):
    """``cmd`` を実行する。出力はエラー終了時のみ表示する。"""
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
    """``solve_ik`` を実行し、1 人あたりの平均 IK 時間 [秒] を返す。"""
    solve_ik(python, human_poses_dir, palm_poses_dir, handshake_dir,
             collision_pairs_path, robot_arm, extra_args)
    return mean_ik_time_per_person(handshake_dir)


def main():
    parser = argparse.ArgumentParser(
        description='干渉回避無しの IK で干渉した組をランキングし、'
                    'collision_pairs.json を作る。')
    parser.add_argument(
        '--num-samples', type=int, default=100,
        help='生成する人物の数。')
    parser.add_argument(
        '--human-poses-dir', type=str, default=None,
        help='骨格 JSON のディレクトリ (既定は一時ディレクトリ)。')
    parser.add_argument(
        '--palm-poses-dir', type=str, default=None,
        help='掌 JSON のディレクトリ (既定は一時ディレクトリ)。')
    parser.add_argument(
        '--handshake-dir', type=str, default=None,
        help='solve_palm_ik.py の出力先 (既定は一時ディレクトリ)。')
    parser.add_argument(
        '--output', type=str,
        default=os.path.join(_SCRIPTS_DIR, 'collision_pairs.json'),
        help='書き出す干渉ペア JSON (既定は本番の scripts/collision_pairs.json)。')
    parser.add_argument(
        '--skip-generate', action='store_true',
        help='人物生成・掌推定を省略し既存ディレクトリを使う。')
    parser.add_argument(
        '--collision-dist-threshold', type=float,
        default=-DEFAULT_COLLISION_VERIFY_TOLERANCE,
        help='距離がこの値 [m] 未満なら干渉とみなす (既定は事後検証と同じ)。')
    parser.add_argument(
        '--skip-solve', action='store_true',
        help='干渉回避無しの IK を省略し既存の --handshake-dir を集計する。')
    parser.add_argument(
        '--ranking-output', type=str, default=None,
        help='ランキングを保存する JSON のパス。')
    parser.add_argument(
        '--show-ranking', type=int, default=30,
        help='表示するランキングの件数。')
    parser.add_argument(
        '--include-pairs', type=str, default=None,
        help='順位によらず必ず採用する組の JSON (--num-pairs に数えない)。')
    parser.add_argument(
        '--no-verify', action='store_true',
        help='採用後に IK を解き直して時間を測る確認を省略する。')
    count_group = parser.add_mutually_exclusive_group(required=True)
    count_group.add_argument(
        '--num-pairs', type=int,
        help='ランキングの上位何組を採用するか。')
    count_group.add_argument(
        '--max-ik-seconds-per-person', type=float,
        help='1 人あたりの平均 IK 時間がこの秒数を超える直前の組数を採用する。')
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='solve_palm_ik.py に渡す --robot-arm。')
    parser.add_argument(
        '--seed', type=int, default=None,
        help='generate_random_human_poses.py に渡す --seed。')
    parser.add_argument(
        '--python', type=str, default=sys.executable,
        help='子プロセスを実行する Python インタプリタ。')
    parser.add_argument(
        '--solve-arg', dest='solve_args', action='append', default=[],
        help='solve_palm_ik.py に追加で渡す引数 (複数回指定可)。')
    args = parser.parse_args()

    # 明示指定されなかった中間ファイルは一時ディレクトリに置き、終了時に削除する。
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
