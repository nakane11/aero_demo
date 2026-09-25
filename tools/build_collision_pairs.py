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
4. 3. の結果 (このファイル内の ``analyze_handshake_dir``、旧
   ``analyze_collision_pairs.py`` から移植したもの) を集計し、実際に
   (指定した距離未満まで) 近づいた -- 干渉した -- リンクの組み合わせを、
   干渉した人数が多い順に並べたランキングを作る。
5. このランキングの上位 ``--num-pairs`` 組をそのまま ``collision_pairs.
   json`` として書き出す。``--max-ik-seconds-per-person`` を指定した
   場合は、代わりにランキングの先頭から 1・2・3... 組と増やしながら
   干渉回避ありで実際に IK を解き直し (``solve_palm_ik.py`` の通常の
   フルパイプライン、事後検証・後処理を含む)、1 人あたりの平均計算時間が
   指定秒数を超えた直前の組数を採用する。

この手順は「干渉回避を全く行わない解に、実際にどのリンクの組み合わせが
どれだけの頻度で干渉するか」という一度きりの統計だけでランキングを作る
(以前のバージョンにあった「1 組追加するたびに干渉回避ありで解き直して
統計を取り直す」反復はしない)。かつて反復方式だったのは、干渉回避 IK を
一度でも有効にすると事後の干渉検証 (``pick_verified_candidate``) が総
当たりで効き、``attempts_per_pose`` を大きくした設定では 1 組追加した
だけで採用される解が軒並みクリーンになってしまい、統計がすぐ「もう干渉
していない」と誤認して 1 組で収束してしまう問題があったため。干渉回避
無しの解 (事後検証もしていない、生の頻度) を一度だけ集計してランキング
する今の方式は、この問題を避けつつ、より単純かつ (実測で) 実行時間・
成功率の両面でより良いペア集合を作れることを確認している。

``solve_palm_ik.py`` は「``--collision-pairs`` に指定した JSON が存在
しなければ、自己干渉・人体との干渉の両方と事後の干渉検証を無効にする」
という仕様 (``solve_palm_ik.load_collision_pairs`` 呼び出し部分参照) を
利用して、3. の「干渉回避無し」を実現している。

Usage
-----
    python3 tools/build_collision_pairs.py --num-pairs 8

既存の (README 記載の) パイプラインと同じ既定ディレクトリ
(``scripts/random_human_poses/``/``scripts/random_palm_poses/``) を使い、
``solve_palm_ik.py`` の入出力には ``scripts/random_handshake_poses/`` を
使う。
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

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from solve_palm_ik import (  # noqa: E402
    HUMAN_FRONT_DISTANCE, apply_collision_model, collision_link_list_for_arm,
    human_capsules, human_translation_offset, load_skeleton_json,
    segment_points_distance, translate_joint_positions)

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.planner.trajectory_optimization.collision import (  # noqa: E402
    create_self_collision_pairs)


def analyze_handshake_dir(handshake_dir, skeleton_dir,
                          human_front_distance=HUMAN_FRONT_DISTANCE,
                          dist_threshold=0.0):
    """``handshake_dir`` (``solve_palm_ik.py`` の出力) と ``skeleton_dir``
    (骨格 JSON) を読み、自己干渉・人体との干渉それぞれの組み合わせごとの
    「全サンプル中の最小距離」と「``dist_threshold`` [m] 未満まで近づいた
    (干渉した) サンプル数」を集計する (旧 ``analyze_collision_pairs.
    analyze_handshake_dir`` を移植したもの。干渉ペアの自動抽出以外の用途
    (単独でのレポート表示、``collision_pair_analysis.json`` への書き出し)
    は使われなくなったため削除し、この関数だけを残してある)。

    各リンクは ``apply_collision_model`` が差し替えた ``collision_mesh``
    (``trimesh.Trimesh``、box/cylinder/sphere のプリミティブ近似形状) の
    頂点をそのまま (半径 0 の点群として) 使う -- 干渉回避付きバッチ IK の
    勾配降下法が最適化のために使う球への近似 (``extract_collision_
    spheres``) は行わない。自己干渉の距離はリンク間の頂点対の最短距離、
    人体との干渉の距離は頂点と人体セグメント (線分 + 半径) との最短距離
    (``segment_points_distance``) からその半径を引いたもの。

    Returns
    -------
    dict
        ``self_min_dist``/``human_min_dist`` (キーは ``(名前A, 名前B)`` の
        タプル、値は距離 [m])、``self_collision_count``/``human_
        collision_count`` (同じキーで、``dist_threshold`` 未満まで
        近づいたサンプル数)、``self_pairs`` (組み合わせのリスト)、
        ``link_names``/``cap_names``、``n_samples`` を持つ dict。
        ``n_samples`` が 0 のときは他の値も空。
    """
    robot = Aero(use_hand=False)
    apply_collision_model(robot)
    collision_link_list = collision_link_list_for_arm(robot, 'r')
    link_names = [link.name for link in collision_link_list]
    n_links = len(collision_link_list)

    vertices_local_by_link = [
        np.asarray(link.collision_mesh.vertices, dtype=np.float64)
        for link in collision_link_list]

    self_pairs = create_self_collision_pairs(
        collision_link_list, ignore_adjacent=True)

    self_min_dist = {}
    self_collision_count = {}
    human_min_dist = {}
    human_collision_count = {}
    n_samples = 0
    cap_names = []

    files = sorted(glob.glob(os.path.join(handshake_dir, '*.json')))
    for path in files:
        with open(path) as f:
            result = json.load(f)
        if not result.get('solved'):
            continue
        base_name = os.path.basename(path)
        skeleton_path = os.path.join(skeleton_dir, base_name)
        if not os.path.exists(skeleton_path):
            continue
        n_samples += 1

        joint_positions = load_skeleton_json(skeleton_path)
        offset = human_translation_offset(
            joint_positions, front_distance=human_front_distance)
        joint_positions = translate_joint_positions(joint_positions, offset)
        caps, cap_names = human_capsules(joint_positions)

        robot.reset_pose()
        robot.newcoords(Coordinates())
        robot.base_link.newcoords(Coordinates())
        robot.angle_vector(np.asarray(result['joint_angle_vector']))
        base_coords = Coordinates(
            pos=result['base_position']).rotate(result['base_yaw'], 'z')
        robot.newcoords(base_coords)

        world_vertices_by_link = []
        for link, verts_local in zip(collision_link_list,
                                     vertices_local_by_link):
            world_vertices_by_link.append(
                verts_local @ link.worldrot().T + link.worldpos())

        for li, lj in self_pairs:
            verts_i = world_vertices_by_link[li]
            verts_j = world_vertices_by_link[lj]
            dists = np.linalg.norm(
                verts_i[:, np.newaxis, :] - verts_j[np.newaxis, :, :],
                axis=-1)
            best = float(dists.min())
            key = tuple(sorted((link_names[li], link_names[lj])))
            if key not in self_min_dist or best < self_min_dist[key]:
                self_min_dist[key] = best
            if best < dist_threshold:
                self_collision_count[key] = (
                    self_collision_count.get(key, 0) + 1)

        for li in range(n_links):
            verts_i = world_vertices_by_link[li]
            for ci, (p0, p1, cap_r) in enumerate(caps):
                best = float(
                    segment_points_distance(p0, p1, verts_i).min()) - cap_r
                key = (link_names[li], cap_names[ci])
                if key not in human_min_dist or best < human_min_dist[key]:
                    human_min_dist[key] = best
                if best < dist_threshold:
                    human_collision_count[key] = (
                        human_collision_count.get(key, 0) + 1)

    return dict(
        self_min_dist=self_min_dist, human_min_dist=human_min_dist,
        self_collision_count=self_collision_count,
        human_collision_count=human_collision_count,
        self_pairs=self_pairs, link_names=link_names, cap_names=cap_names,
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


def rank_collision_candidates(handshake_dir, skeleton_dir,
                              human_front_distance, dist_threshold,
                              exclude=()):
    """``find_collision_candidates`` の結果を、干渉した人数が多い順に
    並べた ``[(名前A, 名前B), ...]`` のリストにして返す (``exclude`` に
    含まれる組み合わせは除く)。同数の場合はタプルの辞書順で安定させる。"""
    candidates = find_collision_candidates(
        handshake_dir, skeleton_dir, human_front_distance, dist_threshold)
    candidates = {pair: count for pair, count in candidates.items()
                 if pair not in exclude}
    return sorted(candidates, key=lambda pair: (-candidates[pair], pair))


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
    cmd = [python, os.path.join(_SCRIPTS_DIR, 'solve_palm_ik.py'),
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
        '--collision-dist-threshold', type=float, default=0.0,
        help='この距離 [m] 未満まで近づいた組み合わせを「干渉した」と '
            'みなしてランキングに使う (既定 0.0 = 実際に干渉用メッシュ '
            '同士がめり込んだ組み合わせのみ)。')
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
        candidate_output = os.path.join(temp_dir, 'candidate_pairs.json')

        if not args.skip_generate:
            run([args.python,
                os.path.join(_SCRIPTS_DIR, 'generate_random_human_poses.py'),
                '--num-samples', str(args.num_samples),
                '--output-dir', args.human_poses_dir])
            run([args.python,
                os.path.join(_SCRIPTS_DIR, 'estimate_palm_poses.py'),
                '--input-dir', args.human_poses_dir,
                '--output-dir', args.palm_poses_dir])

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

        print('\n=== 手順 4: 干渉頻度をランキング ===')
        ranking = rank_collision_candidates(
            args.handshake_dir, args.human_poses_dir,
            HUMAN_FRONT_DISTANCE, args.collision_dist_threshold)
        if not ranking:
            print('干渉した組み合わせが見つかりませんでした。')
            sys.exit(1)
        print('干渉頻度の高い順に {} 組の候補が見つかりました (上位10件):'
             .format(len(ranking)))
        candidates_dict = find_collision_candidates(
            args.handshake_dir, args.human_poses_dir,
            HUMAN_FRONT_DISTANCE, args.collision_dist_threshold)
        for pair in ranking[:10]:
            print('  {} ({} / {} 人で干渉)'.format(
                pair, candidates_dict[pair], n_ik_people))

        print('\n=== 手順 5: 採用する組数を決定 ===')
        if args.num_pairs is not None:
            if args.num_pairs > len(ranking):
                print('警告: --num-pairs ({}) がランキングの候補数 ({}) を '
                      '超えているため、候補数だけ採用します。'.format(
                          args.num_pairs, len(ranking)))
            pairs = set(ranking[:args.num_pairs])
            save_pairs(pairs, args.output)
            print('上位 {} 組を採用しました -> {}'.format(
                len(pairs), args.output))
            print('検証のため、この組数で干渉回避ありの IK を解き直します。')
            elapsed = timed_solve_ik(
                args.python, args.human_poses_dir, args.palm_poses_dir,
                args.handshake_dir, args.output, args.robot_arm, args.seed,
                args.solve_args)
            per_person = elapsed / n_ik_people
            print('IK 計算時間: {:.2f} 秒 ({:.3f} 秒/人 x {} 人)。'.format(
                elapsed, per_person, n_ik_people))
        else:
            n_pairs = 0
            per_person = None
            for n_pairs in range(1, len(ranking) + 1):
                pairs = set(ranking[:n_pairs])
                save_pairs(pairs, candidate_output)
                elapsed = timed_solve_ik(
                    args.python, args.human_poses_dir, args.palm_poses_dir,
                    args.handshake_dir, candidate_output, args.robot_arm,
                    args.seed, args.solve_args)
                per_person = elapsed / n_ik_people
                print('{} 組: {:.2f} 秒 ({:.3f} 秒/人 x {} 人)。'.format(
                    n_pairs, elapsed, per_person, n_ik_people))
                if per_person > args.max_ik_seconds_per_person:
                    print('1 人あたりの IK 計算時間が上限 ({:.2f} 秒) を '
                         '超えたため、直前の {} 組を採用します。'.format(
                             args.max_ik_seconds_per_person, n_pairs - 1))
                    n_pairs -= 1
                    break
            else:
                print('ランキングを全て試しても上限を超えませんでした。'
                     '全 {} 組を採用します。'.format(n_pairs))
            pairs = set(ranking[:n_pairs])
            save_pairs(pairs, args.output)
            print('{} 組を採用しました -> {}'.format(len(pairs), args.output))

        print('\n最終的な干渉ペア数: {} -> {}'.format(len(pairs), args.output))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
