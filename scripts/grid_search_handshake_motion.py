#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``plan_handshake_motion.py`` の軌道最適化に関わるハイパーパラメータの
成功率 (``verify_waypoints`` を通る割合)・計算時間トレードオフを調べる
グリッドサーチ。``grid_search_collision_ik.py`` と同じ考え方 (2段階計測・
ランキング表示) を、掌IKではなく軌道最適化 (``plan_person_motion``) に
適用したもの。

対象にする3つのハイパーパラメータ (すべて ``plan_handshake_motion.py`` の
引数と同名):

``--motion-attempts``
    線形補間/pre-touchで干渉が残った場合に、warm start を変えて最適化を
    解き直す最大回数 (既定 3)。``perturb_initial_trajectory`` 参照。
``--n-waypoints``
    軌道の waypoint 数 = 最適化変数の次元 (既定
    ``plan_handshake_motion.DEFAULT_N_WAYPOINTS`` = 20)。
``--max-iterations``
    jaxls ソルバーの最大反復回数 (既定
    ``plan_handshake_motion.DEFAULT_MAX_ITERATIONS`` = 60)。

``plan_person_motion`` は、pre-touch/線形補間だけの幾何構成が事後の厳密な
干渉検証を通ればそもそも最適化を行わない。つまり上記3パラメータの効果は
「pre-touch/線形補間では干渉が残る人物」でしか観測できず、実際には
最適化が必要になる人物は少ない (``--num-samples`` を素朴に増やすだけ
では十分な数を集めにくい)。

そのため ``--force-optimize`` を指定すると、pre-touch/線形補間による
早期採用を行わず **全対象人物に対して必ず jaxls 最適化を実行する**
(``force_optimize_person_motion``、``plan_handshake_motion.py`` の
公開関数だけを使って組み立てている別ロジックで、``plan_person_motion``
自体は変更しない)。この場合「成功率」は「パイプライン全体の成功率」
ではなく「その最適化条件で jaxls がどれだけ収束するか」を表す点に注意
(pre-touch/線形補間で通っていたはずの簡単なケースも含めて全員を無理に
最適化にかけるため、自然な ``plan_person_motion`` より不利な数字になる)。
3パラメータの効果を確実に比較したい場合はこちらを使うことを推奨する。
``--force-optimize`` を付けない既定モードは、実際のパイプラインの
end-to-end 成功率・所要時間の参考値として使う。

本スクリプト側は1人の例外でも打ち切らずスキップして続行する。

``--n-waypoints`` は最適化問題の形状 (テンソル形状) を変えるため jaxls の
JIT 再コンパイルが発生する。``grid_search_collision_ik.py`` と同じ理由で、
各グリッド点は「ウォームアップ実行 (使い捨て) -> 本計測」の2回に分けて
計測する。

グリッドの組み方は既定で **OFAT (one-factor-at-a-time)**:
既定値をベースラインとして、3パラメータそれぞれについて他を固定したまま
値を振る (各パラメータ3値の既定なら 1 + 2x3 = 7 通り)。3パラメータ全部の
総当たり (既定のまま だと 3^3=27 通り) が必要なら、``--mode full`` を
指定したときだけ ``itertools.product`` で全数を作る (このとき本当に全数を
回したい軸だけ複数値を渡し、他は1値のまま固定する運用を想定)。

人物データセット (``generate_random_human_poses.py`` ->
``estimate_palm_poses.py`` -> ``solve_palm_ik.py``) はグリッドパラメータに
依存しないため、全グリッド点で使い回すために最初に1度だけ生成する
(``--human-poses-dir``/``--palm-poses-dir``/``--handshake-poses-dir`` を
明示的に指定すれば、既存のデータセットをそのまま使い回せる -- ただし
「既にファイルがあれば生成しない」判定なので、``--num-samples`` を変えて
使い回したい場合は別ディレクトリを指定すること)。

出力は各グリッド点のプリント表示に加え、``--output-csv`` (既定
``grid_search_handshake_motion_results.csv``) に集計結果を書き出す。

Usage
-----
    # 予行運転: 少人数・ベースライン1点だけを計測して1人あたりの所要時間を見る
    python3 grid_search_handshake_motion.py --num-samples 10 --force-optimize \\
        --motion-attempts-values 3 --n-waypoints-values 20 \\
        --max-iterations-values 60

    # 本番: 既定の OFAT グリッド (7通り) を、pre-touch/線形補間の早期採用を
    # 無効にして (--force-optimize) 全対象人物で必ず最適化させる
    python3 grid_search_handshake_motion.py --num-samples 100 --force-optimize

    # 参考: --force-optimize なしの自然なパイプライン成功率・所要時間
    # (最適化が必要なケースが少ないと分かっているので大きめのサンプル数で)
    python3 grid_search_handshake_motion.py --num-samples 300

    # 2軸だけ総当たり (残り1軸は固定) にしたい場合
    python3 grid_search_handshake_motion.py --mode full --force-optimize \\
        --n-waypoints-values 12 20 30 \\
        --motion-attempts-values 1 3 6 --max-iterations-values 60
"""

import argparse
import copy
import csv
import glob
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
_PKG_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

import numpy as np  # noqa: E402

# plan_handshake_motion は「jax を import する前に JAX_COMPILATION_CACHE_DIR
# を設定してディスクキャッシュを有効にする」処理をモジュール先頭で行っている
# (plan_handshake_motion.py の該当コメント参照)。skrobot 経由で jax が先に
# import されるとこの設定が手遅れになるため、必ず最初に import する
# (grid_search_collision_ik.py が solve_palm_ik を最初に import するのと
# 同じ理由)。
import plan_handshake_motion as phm  # noqa: E402
import solve_palm_ik as spik  # noqa: E402

from aero_demo import json_io  # noqa: E402
from skrobot.models import Aero  # noqa: E402


def generate_dataset(python, human_dir, palm_dir, handshake_dir,
                     num_samples, seed):
    """人物データセットを3段階のパイプライン (人物骨格 -> 掌姿勢 -> 握手IK)
    で作る。各ディレクトリに既に JSON があれば (グリッドパラメータに依存
    しないので) 作り直さず使い回す。"""
    if not glob.glob(os.path.join(human_dir, '*.json')):
        print('[grid] {} 人分の人物を生成します -> {}'.format(
            num_samples, human_dir))
        cmd = [python,
              os.path.join(_THIS_DIR, 'generate_random_human_poses.py'),
              '--num-samples', str(num_samples), '--output-dir', human_dir]
        if seed is not None:
            cmd += ['--seed', str(seed)]
        subprocess.run(cmd, check=True)
    if not glob.glob(os.path.join(palm_dir, '*.json')):
        print('[grid] 掌の位置姿勢を推定します -> {}'.format(palm_dir))
        subprocess.run([
            python, os.path.join(_THIS_DIR, 'estimate_palm_poses.py'),
            '--input-dir', human_dir, '--output-dir', palm_dir], check=True)
    if not glob.glob(os.path.join(handshake_dir, '*.json')):
        print('[grid] 握手姿勢 (掌IK) を解きます -> {}'.format(handshake_dir))
        cmd = [python, os.path.join(_THIS_DIR, 'solve_palm_ik.py'),
              '--input-dir', palm_dir, '--output-dir', handshake_dir,
              '--skeleton-dir', human_dir]
        if seed is not None:
            cmd += ['--seed', str(seed)]
        subprocess.run(cmd, check=True)


def load_targets(handshake_dir, human_dir):
    """``target and solved`` な握手姿勢 JSON を、対応する骨格 (障害物用の
    ``joint_positions``) ・人間の立ち位置とあわせてロードする。
    ``plan_handshake_motion.main`` が1人ずつ行っている前処理と同じ
    (``human_translation_offset``/``human_standing_xy``)。"""
    targets = []
    n_not_target = 0
    for path in json_io.iter_json_files(handshake_dir):
        handshake = json.load(open(path))
        if not handshake.get('target') or not handshake.get('solved'):
            n_not_target += 1
            continue
        skeleton_path = os.path.join(human_dir, os.path.basename(path))
        joint_positions = spik.load_skeleton_json(skeleton_path)
        offset = spik.human_translation_offset(joint_positions)
        joint_positions = spik.translate_joint_positions(
            joint_positions, offset)
        human_xy = spik.human_standing_xy(joint_positions)
        if human_xy is None:
            human_xy = np.array([spik.HUMAN_FRONT_DISTANCE, 0.0])
        targets.append(dict(
            name=os.path.basename(path), handshake=handshake,
            joint_positions=joint_positions, human_xy=human_xy))
    print('[grid] 対象人物 (target かつ IK 成功): {} 人 '
         '(対象外・IK失敗のため除外: {} 人)'.format(
             len(targets), n_not_target))
    return targets


def make_baseline_args(seed):
    """``plan_person_motion`` に渡す ``args`` のベースライン (グリッドで
    振らない項目は ``plan_handshake_motion.py`` の既定値のまま)。"""
    return argparse.Namespace(
        approach_distance=phm.DEFAULT_APPROACH_DISTANCE,
        pretouch_standoff=phm.DEFAULT_PRETOUCH_STANDOFF,
        pretouch_split=phm.DEFAULT_PRETOUCH_SPLIT,
        n_waypoints=phm.DEFAULT_N_WAYPOINTS,
        dt=phm.DEFAULT_DT,
        max_iterations=phm.DEFAULT_MAX_ITERATIONS,
        collision_activation_distance=(
            phm.DEFAULT_COLLISION_ACTIVATION_DISTANCE),
        self_collision_activation_distance=(
            phm.DEFAULT_SELF_COLLISION_ACTIVATION_DISTANCE),
        collision_weight=100.0,
        self_collision_weight=100.0,
        smoothness_weight=phm.DEFAULT_SMOOTHNESS_WEIGHT,
        acceleration_weight=phm.DEFAULT_ACCELERATION_WEIGHT,
        motion_attempts=3,
        motion_attempt_perturbation=0.3,
        collision_verify_tolerance=(
            phm.DEFAULT_MOTION_COLLISION_VERIFY_TOLERANCE),
        seed=seed,
    )


# グリッドで振る3パラメータ (argparse の属性名 = plan_person_motion の
# args 属性名)。
AXES = ['motion_attempts', 'n_waypoints', 'max_iterations']


def build_combos(baseline, axis_values, mode):
    """``mode='ofat'``: ベースライン + 各軸を1つずつ振った組み合わせ
    (他の軸はベースラインに固定)。``mode='full'``: 全軸の総当たり
    (``itertools.product``)。いずれも重複は除く。"""
    if mode == 'full':
        combos = []
        for values in itertools.product(*(axis_values[axis]
                                          for axis in AXES)):
            combos.append(dict(zip(AXES, values)))
        return combos

    combos = [dict(baseline)]
    seen = {tuple(sorted(baseline.items()))}
    for axis in AXES:
        for value in axis_values[axis]:
            combo = dict(baseline)
            combo[axis] = value
            key = tuple(sorted(combo.items()))
            if key in seen:
                continue
            seen.add(key)
            combos.append(combo)
    return combos


def force_optimize_person_motion(robot, robot_arm, handshake, joint_positions,
                                 human_xy, args, verification_pairs, solver):
    """``plan_person_motion`` と同じ入出力だが、pre-touch/線形補間の幾何
    構成による早期採用 (事後検証が通ればそれで確定させる分岐) を行わず、
    **必ず** jaxls 最適化 (``--motion-attempts`` 回まで) を実行する。

    ``plan_handshake_motion.py`` 自体は変更せず、同モジュールの公開関数
    (``build_start_and_goal``/``build_problem``/``build_initial_
    trajectory``/``build_pretouch_trajectory``/``perturb_initial_
    trajectory``/``trajectory_waypoints``/``verify_waypoints`` など) だけを
    組み合わせて再現している。pre-touch 姿勢が解ければそれを warm start の
    初期値として使う (最適化の出発点としては引き続き有用なため) が、
    それだけで採用を確定させることはしない。

    モジュール docstring の実測 (合成人物100人中、最適化まで必要になった
    のは0人) を踏まえた、4パラメータの効果を確実に比較するための評価
    モード。「成功率」の意味が ``plan_person_motion`` の自然な結果とは
    異なる (パイプライン全体ではなく、最適化条件そのものの成否) 点に
    注意。"""
    start_time = time.time()
    link_list, joint_list, q_start, base_start, q_goal, base_goal = \
        phm.build_start_and_goal(
            robot, robot_arm, handshake,
            # 最適化条件そのものの比較用なので、初期位置による接近開始位置
            # の縮小 (plan_person_motion の initial_base_pose) は行わない。
            phm.approach_base_start(
                phm.handshake_base_goal(handshake),
                phm.approach_direction(
                    human_xy, phm.handshake_base_goal(handshake)),
                args.approach_distance))
    n_joints = len(q_start)

    def make_candidate(trajectory, attempt, cost, solve_time):
        waypoints = phm.trajectory_waypoints(robot, joint_list, trajectory)
        joint_names = [j.name for j in robot.joint_list]
        distances = phm.verify_waypoints(
            robot, joint_names, waypoints, verification_pairs,
            joint_positions)
        return dict(
            planned=True, kind='optimized', optimized=True,
            verified=min(distances) >= -args.collision_verify_tolerance,
            attempt=attempt, cost=cost, n_waypoints=args.n_waypoints,
            dt=args.dt, robot_arm=robot_arm,
            approach_distance=args.approach_distance,
            joint_names=joint_names, waypoints=waypoints,
            waypoint_min_distances=[float(d) for d in distances],
            solve_time=solve_time,
        )

    initial_traj = phm.build_initial_trajectory(
        q_start, base_start, q_goal, base_goal, args.n_waypoints)
    warm_start_traj = initial_traj
    normal = phm.palm_normal_direction(handshake, joint_positions)
    if normal is not None:
        q_pre = phm.solve_pretouch_pose(
            robot, robot_arm, link_list, joint_list, handshake, q_goal,
            base_goal, normal, args.pretouch_standoff)
        if q_pre is not None:
            warm_start_traj = phm.build_pretouch_trajectory(
                q_start, base_start, q_pre, q_goal, base_goal,
                args.n_waypoints, args.pretouch_split)

    # solve_pretouch_pose が台車を最終位置へ動かしているので、最適化問題を
    # 組む前に基準の姿勢 (台車=ワールド原点, 腕=始点) に戻す
    # (plan_handshake_motion.build_start_and_goal の docstring 参照)。
    robot.newcoords(phm.Coordinates())
    robot.base_link.newcoords(phm.Coordinates())
    for joint, angle in zip(joint_list, q_start):
        joint.joint_angle(float(angle))

    world_obstacles = phm.human_body_cylinder_obstacles(joint_positions)
    collision_link_list = spik.collision_link_list_for_arm(robot, robot_arm)
    problem = phm.build_problem(
        robot, robot_arm, link_list, args.n_waypoints, args.dt,
        world_obstacles, collision_link_list,
        args.collision_activation_distance,
        args.self_collision_activation_distance,
        args.smoothness_weight, args.acceleration_weight,
        collision_weight=args.collision_weight,
        self_collision_weight=args.self_collision_weight)
    rng = np.random.RandomState(args.seed)

    best = None
    for attempt in range(args.motion_attempts):
        warm_start = warm_start_traj if attempt == 0 \
            else phm.perturb_initial_trajectory(
                warm_start_traj, n_joints, rng, args.motion_attempt_perturbation)
        solve_start = time.time()
        result = solver.solve(problem, warm_start)
        candidate = make_candidate(
            result.trajectory, attempt, float(result.cost),
            time.time() - solve_start)
        if candidate['verified']:
            best = candidate
            break
        if best is None or (min(candidate['waypoint_min_distances'])
                            > min(best['waypoint_min_distances'])):
            best = candidate

    best['compute_time'] = time.time() - start_time
    return best


def run_measurement(robot, targets, args, verification_pairs, solver,
                    force_optimize=False):
    """``targets`` 全員に対して1回ずつ計画を行い、結果 dict のリストを
    返す (ウォームアップ・本計測どちらでもこの関数をそのまま使う)。
    ``solver`` はグリッド点ごとに1個だけ作って使い回す
    (``plan_handshake_motion.main`` と同じ理由: 人物ごとに作り直すと
    jaxls の JIT キャッシュが効かず、初回コンパイルが人物ごとに走って
    しまう)。``force_optimize`` が真なら ``force_optimize_person_motion``
    (常に最適化) を、偽なら ``plan_person_motion`` (自然なパイプライン) を
    使う。

    特定の人物で計画関数自体が例外を投げることがある (例: jaxls 側の
    cost 項のバッチ軸不一致 ``broadcast_shapes`` エラー。100人規模の実測
    で実際に発生した -- ``plan_handshake_motion.py`` 本体にも起こりうる
    別件のバグで、本スクリプトの対症療法では直せない)。1人の失敗でグリッド
    全体を止めないよう、例外はその人物をスキップして記録するだけに
    とどめる。"""
    plan = (force_optimize_person_motion if force_optimize
           else phm.plan_person_motion)
    results = []
    for target in targets:
        handshake = target['handshake']
        try:
            result = plan(
                robot, handshake['robot_arm'], handshake,
                target['joint_positions'], target['human_xy'], args,
                verification_pairs, solver)
        except Exception as exc:  # noqa: BLE001  (1人の例外でグリッド全体を止めない)
            print('  [警告] {} で計画関数が例外を送出したため '
                 'スキップします: {}: {}'.format(
                     target['name'], type(exc).__name__, exc))
            continue
        results.append(result)
    return results


def aggregate(results, combo, wall_elapsed, n_failed=0):
    n = len(results)
    verified = [bool(r['verified']) for r in results]
    compute_times = np.array([float(r['compute_time']) for r in results])
    kind_counts = Counter(r['kind'] for r in results)
    optimized = [r for r in results if r['kind'] == 'optimized']
    optimized_verified_rate = (
        float(np.mean([bool(r['verified']) for r in optimized]))
        if optimized else float('nan'))
    return dict(
        label='attempts{motion_attempts}_wp{n_waypoints}'
             '_iter{max_iterations}'.format(**combo),
        **combo,
        n_target=n,
        n_failed=n_failed,
        success_rate=float(np.mean(verified)) if n else float('nan'),
        compute_time_mean=float(compute_times.mean()) if n else float('nan'),
        compute_time_median=(
            float(np.median(compute_times)) if n else float('nan')),
        compute_time_max=float(compute_times.max()) if n else float('nan'),
        n_pretouch=kind_counts.get('pretouch', 0),
        n_linear=kind_counts.get('linear', 0),
        n_optimized=kind_counts.get('optimized', 0),
        optimized_verified_rate=optimized_verified_rate,
        wall_elapsed=wall_elapsed,
    )


def print_row(row):
    print('  n_target={n_target} (例外でスキップ={n_failed}) '
         '成功率(verified)={success_rate:.1%} '
         '| 計算時間: 平均={compute_time_mean:.3f}s 中央値='
         '{compute_time_median:.3f}s 最大={compute_time_max:.3f}s '
         '(合計 {wall_elapsed:.1f}s)'.format(**row))
    print('    内訳(kind): pretouch={n_pretouch} linear={n_linear} '
         'optimized={n_optimized} '
         '(optimized のうち verified={optimized_verified_rate})'.format(
             n_pretouch=row['n_pretouch'], n_linear=row['n_linear'],
             n_optimized=row['n_optimized'],
             optimized_verified_rate=(
                 'n/a' if row['optimized_verified_rate'] != row[
                     'optimized_verified_rate']
                 else '{:.1%}'.format(row['optimized_verified_rate']))))


def print_ranking(rows):
    """成功率(降順) -> 平均計算時間(昇順) の順でグリッド点を並べ替えて表示
    する (``grid_search_collision_ik.py`` の ``print_ranking`` と同じ考え
    方)。"""
    def sort_key(row):
        rate = row['success_rate']
        t = row['compute_time_mean']
        rate = rate if rate == rate else -1.0
        t = t if t == t else float('inf')
        return (-rate, t)

    ranked = sorted(rows, key=sort_key)
    print('\n=== ランキング (成功率が高い順 -> 同率なら平均計算時間が短い順) ===')
    for rank, row in enumerate(ranked, start=1):
        print(
            '{:2d}位: attempts={:<2} n_waypoints={:<3} max_iter={:<4} '
            '| 成功率={:.1%} 平均時間={:.3f}s/人 '
            '中央値={:.3f}s 最大={:.3f}s'.format(
                rank, row['motion_attempts'], row['n_waypoints'],
                row['max_iterations'],
                row['success_rate'], row['compute_time_mean'],
                row['compute_time_median'], row['compute_time_max']))

    best = ranked[0]
    print('\n[推奨] motion_attempts={} n_waypoints={} max_iterations={}: '
         '成功率={:.1%}・平均計算時間={:.3f}秒/人 (全構成中で最良)'.format(
             best['motion_attempts'], best['n_waypoints'],
             best['max_iterations'],
             best['success_rate'], best['compute_time_mean']))


CSV_FIELDS = [
    'label', 'motion_attempts', 'n_waypoints',
    'max_iterations', 'n_target', 'n_failed', 'success_rate',
    'compute_time_mean', 'compute_time_median', 'compute_time_max',
    'n_pretouch', 'n_linear', 'n_optimized', 'optimized_verified_rate',
    'wall_elapsed',
]


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CSV_FIELDS})
    print('\n[grid] 集計結果を書き出しました -> {}'.format(path))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--human-poses-dir', type=str, default=None,
        help='人物の骨格 JSON のディレクトリ (既定は /tmp 以下に一時ディレ'
            'クトリを作ってその場で生成する。既存のデータセットを使い回す '
            '場合はここに明示的にパスを指定する)。')
    parser.add_argument('--palm-poses-dir', type=str, default=None)
    parser.add_argument('--handshake-poses-dir', type=str, default=None)
    parser.add_argument(
        '--num-samples', type=int, default=100,
        help='上記3つを指定しなかったときに生成する人数 (既定 100。'
            'target かつ IK 成功で残る人数はこれより少なくなる。'
            '--force-optimize を付けない自然なモードで最適化まで必要な '
            'ケースを集めたい場合は、モジュール docstring の実測を踏まえ '
            'さらに多く (数百人規模) 指定することを推奨する)。')
    parser.add_argument(
        '--force-optimize', action='store_true',
        help='pre-touch/線形補間による早期採用を行わず、全対象人物に '
            '対して必ず jaxls 最適化を実行する (モジュール docstring 参照。'
            '4パラメータの効果を確実に比較したい場合はこちらを推奨)。')
    parser.add_argument('--motion-attempts-values', type=int, nargs='+',
                        default=[1, 3, 6])
    parser.add_argument('--n-waypoints-values', type=int, nargs='+',
                        default=[12, phm.DEFAULT_N_WAYPOINTS, 30])
    parser.add_argument('--max-iterations-values', type=int, nargs='+',
                        default=[30, phm.DEFAULT_MAX_ITERATIONS, 120])
    parser.add_argument(
        '--mode', choices=['ofat', 'full'], default='ofat',
        help='"ofat" (既定): 既定値をベースラインに1軸ずつ振る。"full": '
            '3軸全部の総当たり (itertools.product) -- 値を複数指定した軸 '
            'だけ実質的に振りたい場合に使う (モジュール docstring 参照)。')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--python', type=str, default=sys.executable,
                        help='人物生成・掌推定・握手IKに使うPython '
                            '(既定はこのスクリプトと同じインタプリタ)。')
    parser.add_argument(
        '--output-csv', type=str,
        default=os.path.join(_THIS_DIR,
                             'grid_search_handshake_motion_results.csv'))
    args = parser.parse_args()

    if (args.human_poses_dir is None or args.palm_poses_dir is None
            or args.handshake_poses_dir is None):
        base_dir = tempfile.mkdtemp(
            prefix='aero_demo_grid_search_motion_', dir='/tmp')
        print('作業ディレクトリ: {}'.format(base_dir))
        if args.human_poses_dir is None:
            args.human_poses_dir = os.path.join(base_dir, 'skeletons')
        if args.palm_poses_dir is None:
            args.palm_poses_dir = os.path.join(base_dir, 'palms')
        if args.handshake_poses_dir is None:
            args.handshake_poses_dir = os.path.join(base_dir, 'handshakes')

    generate_dataset(args.python, args.human_poses_dir, args.palm_poses_dir,
                     args.handshake_poses_dir, args.num_samples, args.seed)
    targets = load_targets(args.handshake_poses_dir, args.human_poses_dir)
    if not targets:
        print('対象人物が0人のため終了します。--num-samples を増やすか、'
             '別のシードを試してください。')
        return
    if len(targets) < 5:
        print('[警告] 対象人物が {} 人しかいません。成功率の推定が粗くなる '
             'ため --num-samples を増やすことを推奨します。'.format(
                 len(targets)))

    baseline = dict(
        motion_attempts=3, n_waypoints=phm.DEFAULT_N_WAYPOINTS,
        max_iterations=phm.DEFAULT_MAX_ITERATIONS)
    axis_values = dict(
        motion_attempts=args.motion_attempts_values,
        n_waypoints=args.n_waypoints_values,
        max_iterations=args.max_iterations_values)
    combos = build_combos(baseline, axis_values, args.mode)
    print('[grid] {} 通りの組み合わせ ({} モード) を、各ウォームアップ+'
         '本計測の2回で計測します ({} 人 x 2 x {} 回 = {} 回の plan_person_'
         'motion 呼び出し)。'.format(
             len(combos), args.mode, len(targets), len(combos),
             len(targets) * 2 * len(combos)))

    robot = Aero(use_hand=False)
    spik.restrict_elbow_range(robot)
    spik.lock_fixed_joints(robot)
    spik.apply_collision_model(robot)
    # 事後検証 (verify_waypoints) の総当たりペアはロボット構造だけで決まり
    # グリッドパラメータに依存しないので1回だけ作る (plan_handshake_motion.
    # main と同じ。'r' はプレースホルダで結果に影響しない)。
    verification_pairs = spik.build_collision_verification_pairs(robot, 'r')

    rows = []
    for combo in combos:
        grid_args = make_baseline_args(args.seed)
        for axis in AXES:
            setattr(grid_args, axis, combo[axis])
        label = ('attempts{motion_attempts}_wp{n_waypoints}'
                '_iter{max_iterations}'.format(**combo))
        print('\n=== {} ==='.format(label))

        # jaxls ソルバーはグリッド点ごとに1個だけ作って使い回す
        # (plan_handshake_motion.main と同じ。max_iterations はソルバー
        # 構築時に固定されるため combo が変わるたびに作り直す必要がある)。
        solver = phm.create_solver(
            'jaxls', max_iterations=grid_args.max_iterations, verbose=False)

        # 1. ウォームアップ (使い捨て、JITコンパイルを消化するだけ)
        run_measurement(robot, targets, grid_args, verification_pairs, solver,
                        force_optimize=args.force_optimize)
        # 2. 本計測 (定常状態、ウォームアップと同じ solver を使い続けて
        # JIT キャッシュを引き継ぐ)
        t0 = time.time()
        results = run_measurement(robot, targets, grid_args,
                                  verification_pairs, solver,
                                  force_optimize=args.force_optimize)
        wall_elapsed = time.time() - t0

        n_failed = len(targets) - len(results)
        row = aggregate(results, combo, wall_elapsed, n_failed)
        rows.append(row)
        print_row(row)

    print_ranking(rows)
    write_csv(rows, args.output_csv)


if __name__ == '__main__':
    main()
