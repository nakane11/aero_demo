#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``plan_handshake_motion.py`` の軌道最適化のハイパーパラメータ
(``motion_attempts``/``n_waypoints``/``max_iterations``) の成功率・計算時間
トレードオフを調べるグリッドサーチ。

既定は OFAT (ベースラインから 1 軸ずつ振る)、``--mode full`` で総当たり。
各グリッド点はウォームアップ (JIT コンパイル) と本計測の 2 回実行する。
``--force-optimize`` は pre-touch/線形補間での早期採用をせず全員を最適化する
(成功率はパイプライン全体ではなく最適化そのものの収束率になる)。
データセットは既存ファイルがあれば再生成しない。

Usage
-----
    python3 tools/grid_search_handshake_motion.py --num-samples 100 \\
        --force-optimize
    python3 tools/grid_search_handshake_motion.py --mode full \\
        --force-optimize --n-waypoints-values 12 20 30 \\
        --motion-attempts-values 1 3 6 --max-iterations-values 60
"""

import argparse
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
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
_PKG_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

import numpy as np  # noqa: E402

# jax のキャッシュ設定を有効にするため、skrobot より先に import する。
import plan_handshake_motion as phm  # noqa: E402
import solve_palm_ik as spik  # noqa: E402

from aero_demo import json_io  # noqa: E402
from skrobot.models import Aero  # noqa: E402


def generate_dataset(human_dir, palm_dir, handshake_dir, num_samples, seed):
    """骨格 -> 掌 -> 握手 IK を作る (既に JSON があるディレクトリは使い回す)。"""
    python = sys.executable
    if not glob.glob(os.path.join(human_dir, '*.json')):
        print('[grid] {} 人分の人物を生成します -> {}'.format(
            num_samples, human_dir))
        cmd = [python,
              os.path.join(_SCRIPTS_DIR, 'generate_random_human_poses.py'),
              '--num-samples', str(num_samples), '--output-dir', human_dir]
        if seed is not None:
            cmd += ['--seed', str(seed)]
        subprocess.run(cmd, check=True)
    if not glob.glob(os.path.join(palm_dir, '*.json')):
        print('[grid] 掌の位置姿勢を推定します -> {}'.format(palm_dir))
        subprocess.run([
            python, os.path.join(_SCRIPTS_DIR, 'estimate_palm_poses.py'),
            '--input-dir', human_dir, '--output-dir', palm_dir], check=True)
    if not glob.glob(os.path.join(handshake_dir, '*.json')):
        print('[grid] 握手姿勢 (掌IK) を解きます -> {}'.format(handshake_dir))
        cmd = [python, os.path.join(_SCRIPTS_DIR, 'solve_palm_ik.py'),
              '--input-dir', palm_dir, '--output-dir', handshake_dir,
              '--skeleton-dir', human_dir]
        if seed is not None:
            cmd += ['--seed', str(seed)]
        subprocess.run(cmd, check=True)


def load_targets(handshake_dir, human_dir):
    """``target`` かつ ``solved`` の握手姿勢を、骨格・立ち位置とあわせて読む。"""
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
    """``plan_person_motion`` に渡す ``args`` のベースライン。"""
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


# グリッドで振るパラメータ (plan_person_motion の args 属性名)。
AXES = ['motion_attempts', 'n_waypoints', 'max_iterations']


def build_combos(baseline, axis_values, mode):
    """``'ofat'``: ベースライン + 1 軸ずつ振った組。``'full'``: 総当たり。"""
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
    """``plan_person_motion`` と同じ入出力で、早期採用せず必ず jaxls 最適化する。

    pre-touch 姿勢が解ければ warm start に使う。
    """
    start_time = time.time()
    base_goal = phm.handshake_base_goal(handshake)
    # 始点は角度 0 の orbit 候補 (初期位置による縮小はしない)。
    orbit_center = phm.orbit_center_xy(handshake)
    orbit_dir = phm.approach_direction(human_xy, base_goal)
    link_list, joint_list, q_start, base_start, q_goal, base_goal = \
        phm.build_start_and_goal(
            robot, robot_arm, handshake,
            phm.orbit_base_start(
                base_goal, orbit_center, orbit_dir, args.approach_distance),
            orbit=(orbit_center, human_xy))
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

    # solve_pretouch_pose が動かした台車・腕を始点に戻す。
    robot.newcoords(phm.Coordinates())
    robot.base_link.newcoords(phm.Coordinates())
    for joint, angle in zip(joint_list, q_start):
        joint.joint_angle(float(angle))

    world_obstacles = phm.human_body_cylinder_obstacles(joint_positions)
    collision_link_list = spik.collision_link_list_for_arm(robot)
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
    """``targets`` 全員を 1 回ずつ計画する (例外を出した人物はスキップ)。

    ``solver`` は JIT キャッシュを効かせるためグリッド点ごとに使い回す。
    """
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
        except Exception as exc:  # noqa: BLE001
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
    """成功率の降順 -> 平均計算時間の昇順で表示する。"""
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
        help='骨格 JSON のディレクトリ (既定は /tmp 以下に生成)。')
    parser.add_argument('--palm-poses-dir', type=str, default=None)
    parser.add_argument('--handshake-poses-dir', type=str, default=None)
    parser.add_argument(
        '--num-samples', type=int, default=100,
        help='生成する人数。')
    parser.add_argument(
        '--force-optimize', action='store_true',
        help='早期採用せず全員で jaxls 最適化を実行する。')
    parser.add_argument('--motion-attempts-values', type=int, nargs='+',
                        default=[1, 3, 6])
    parser.add_argument('--n-waypoints-values', type=int, nargs='+',
                        default=[12, phm.DEFAULT_N_WAYPOINTS, 30])
    parser.add_argument('--max-iterations-values', type=int, nargs='+',
                        default=[30, phm.DEFAULT_MAX_ITERATIONS, 120])
    parser.add_argument(
        '--mode', choices=['ofat', 'full'], default='ofat',
        help='ofat: 1 軸ずつ振る。full: 総当たり。')
    parser.add_argument('--seed', type=int, default=0)
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

    generate_dataset(args.human_poses_dir, args.palm_poses_dir,
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
    spik.restrict_leg_range(robot)
    spik.restrict_waist_range(robot)
    spik.restrict_neck_range(robot)
    spik.lock_fixed_joints(robot)
    spik.apply_collision_model(robot)
    spik.apply_hand_box(robot)
    # 検証ペアはロボット構造だけで決まる ('r' は結果に影響しない)。
    verification_pairs = spik.build_collision_verification_pairs(robot, 'r')

    rows = []
    for combo in combos:
        grid_args = make_baseline_args(args.seed)
        for axis in AXES:
            setattr(grid_args, axis, combo[axis])
        label = ('attempts{motion_attempts}_wp{n_waypoints}'
                '_iter{max_iterations}'.format(**combo))
        print('\n=== {} ==='.format(label))

        # max_iterations はソルバー構築時に固定されるので組ごとに作る。
        solver = phm.create_solver(
            'jaxls', max_iterations=grid_args.max_iterations, verbose=False)

        # ウォームアップ (JIT コンパイル) -> 本計測
        run_measurement(robot, targets, grid_args, verification_pairs, solver,
                        force_optimize=args.force_optimize)
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
