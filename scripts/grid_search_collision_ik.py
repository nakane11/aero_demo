#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``solve_palm_ik.py`` の速度・成功率トレードオフを調べるためのグリッド
サーチ。初期値の数 (``--attempts-per-pose``)・干渉回避ペア設定
(``--collision-pairs``)・干渉回避付きバッチIKの最大反復回数
(``--collision-ik-stop``) の全組み合わせについて、人物データセット
(``--human-poses-dir``/``--palm-poses-dir``で明示的に指定しなければ、
``run_pipeline_test.py`` と同様に ``/tmp`` 以下に一時ディレクトリを作って
生成する) で1人あたりの平均計算時間と成功率を **2段階** に分けて計測し、
段階Bの成功率・時間でランキングしてプリント文で表示する
(ファイル出力はしない)。

段階A (事後干渉検証まで)
    干渉回避付きバッチIKが収束し (``success_flags``)、かつ事後の厳密な
    干渉検証 (``collision_pairs_min_distance``、ロボット全リンク×人体
    全セグメントの総当たり) を通過する候補が見つかったかどうか。
    ``solve_palm_ik.pick_verified_candidate`` が最初に「採用してよい」と
    判定するまでの所要時間 (押し込み・視線のIKの成否には依存しない)。

段階B (押し込み・視線のIKまで)
    段階Aで見つけた候補 (押し込み・視線のIKが失敗すれば次の候補) に対して
    ``solve_palm_ik.solve_post_process`` (干渉回避なしの通常IKで、掌への
    押し込みと、差し出している手を見る視線合わせを同時に解く) まで成功
    したかどうか。段階Aの候補で失敗すれば次の候補を試すため、段階Bの所要
    時間は段階A以上になる。

``--collision-pairs`` に渡せる値:
    ``none``
        干渉回避の最適化 (collision_pairs/self_collision/
        collision_obstacles) を完全に無効にする (``'jacobian'`` 法)。
        事後の干渉検証 (``build_collision_verification_pairs`` による
        総当たり) は他の設定と同じく必ず行う。

        **既知の重大な注意点**: skrobot の ``'jacobian'`` 法は
        ``use_base`` (台車移動) 併用時、IK ソルバをオブジェクト単位で
        キャッシュしない実装になっている上に、目標の位置・姿勢の具体的な
        値がコンパイル結果に定数として焼き込まれてしまうため、
        **人物が変わるたびに毎回フル再コンパイル (数秒) が発生する**
        (``'gradient_descent'`` 法はキャッシュキーが目標値に依存しない
        ため、腕ごとに1回コンパイルすれば以降どんな人物でも高速)。
        「ウォームアップ実行→本計測」を同じ人物セットに対して行うと、
        本計測がウォームアップ実行で作られたキャッシュ (=その人物たち
        **専用** のコンパイル結果) を利用できてしまい、未知の人物に
        対する本番相当の速度より大幅に速く見える。この構成の数値は
        参考程度に留め、本番の速度見積もりには次の ``none-gd`` を使う
        こと。
    ``none-gd``
        最適化としては実質的に何もしない (実際の干渉ペアは0個) が、
        ``'gradient_descent'`` 法を使う。``self_collision=True`` かつ
        ``collision_link_list`` にリンクを1つだけ (``robot.body_link``)
        渡すことで実現する -- 自己干渉はリンクが2つ無いと組み合わせを
        作れないため、コスト0でも ``'gradient_descent'`` 法に切り替わる。
        このキャッシュは目標値に依存しないため、腕 (左右) ごとに1回だけ
        コンパイルすれば、以降はどんな新しい人物でも高速 (``none`` より
        even 高速なことが多い -- 干渉ペナルティの計算コスト自体が無い
        ため)。「未処理の干渉回避なしで最速な構成」を知りたいときは
        ``none`` ではなくこちらを使うこと。
    それ以外の文字列
        干渉回避ペアを列挙したJSONファイルへのパス (``solve_palm_ik.py
        --collision-pairs`` と同じ形式)。最適化・事後検証の両方をこの
        ファイルに基づいて行う (``solve_palm_ik.main()`` の既定動作と同じ)。
        このキャッシュも目標値に依存しないため ``none-gd`` と同様、腕
        ごとに1回のコンパイルで済む。

各グリッド点は「ウォームアップ実行 (使い捨て) -> 本計測」の2回に分けて
実行する。JAXはコンパイル済みの形状 (attempts_per_pose・ペア数など) が
変わるたびに再コンパイルが発生し、その時間が1人あたり時間に混入して
定常状態からかけ離れた値になる (2026年時点でのbuild_collision_pairs.py
の反復ループで実際に踏んだ落とし穴) ため、それを避けるための2回計測。

Usage
-----
    python3 grid_search_collision_ik.py \\
        --attempts-per-pose 16 64 \\
        --collision-ik-stop 100 500 \\
        --collision-pairs none-gd collision_pairs.json

(``--human-poses-dir``/``--palm-poses-dir`` を指定しなければ、
``run_pipeline_test.py`` と同様に ``/tmp`` 以下の一時ディレクトリに
``--num-samples`` 人分をその場で生成する。既存のデータセットを使い回したい
場合は ``--human-poses-dir``/``--palm-poses-dir`` を明示的に指定する。結果は
ファイルに保存せず、段階B(押し込み・視線のIKまで)の成功率・時間で
ランキングしてプリント文で表示するだけ。)
"""

import argparse
import glob
import itertools
import os
import subprocess
import sys
import tempfile
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np  # noqa: E402

# solve_palm_ik は「jax を import する前に JAX_COMPILATION_CACHE_DIR を
# 設定してディスクキャッシュを有効にする」処理をモジュール先頭で行っている
# (solve_palm_ik.py の該当コメント参照)。skrobot 経由で jax が先に
# import されるとこの設定が手遅れになりディスクキャッシュが効かなくなる
# (干渉回避なし・use_base 併用時のように skrobot 側のオブジェクトキャッシュ
# が効かない呼び出しパターンで、1人あたり数秒という異常な遅さになって
# 初めて気付いた -- 干渉回避ありは skrobot 側のオブジェクトキャッシュで
# 救われるため症状が出にくい)。そのため solve_palm_ik を必ず先に import
# する。
import solve_palm_ik as spi  # noqa: E402

from skrobot.models import Aero  # noqa: E402


def generate_dataset(python, human_dir, palm_dir, num_samples, seed):
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


def pick_verified_candidate_timed(
        robot, success_flags, angle_vectors, base_poses,
        verification_pairs, joint_positions, collision_verify_tolerance,
        robot_arm, palm, rots, attempts_per_pose, post_process_ik_stop):
    """``solve_palm_ik.pick_verified_candidate`` と同じ選定ロジックだが、
    段階A(事後干渉検証を通過した最初の候補が見つかった時点)と
    段階B(関数が最終的にどの候補を採用するか確定する時点、つまり
    押し込み・視線のIKまで含めた結果が決まる時点)それぞれの累積時間・
    成功可否も記録して返す。ロジック自体 (収束 -> 事後検証 -> 後処理判定
    -> フォールバック) は元の関数と同一 -- 計測のための計装を追加しただけ。

    Returns
    -------
    tuple
        ``(picked, timing)``。``picked`` は ``pick_verified_candidate`` と
        同じ形式 (``None`` または ``(turn_index, angle_vector, base_pose,
        post_process_result)``)。``timing`` は
        ``dict(stage_a_success, stage_a_time, stage_b_success,
        stage_b_time)`` -- ``stage_a_time``/``stage_b_time`` は候補選定
        ループに入ってからの経過秒数 (``None`` は段階Aが1候補も見つから
        なかったことを表す)。
    """
    t_loop_start = time.time()
    stage_a_success = False
    stage_a_time = None
    fallback = None
    picked = None
    for candidate_index, ok in enumerate(success_flags):
        if not ok:
            continue
        turn_index = candidate_index // attempts_per_pose
        robot.angle_vector(angle_vectors[candidate_index])
        robot.newcoords(base_poses[candidate_index])
        min_dist = spi.collision_pairs_min_distance(
            robot, verification_pairs, joint_positions)
        if min_dist < -collision_verify_tolerance:
            continue
        if not stage_a_success:
            stage_a_success = True
            stage_a_time = time.time() - t_loop_start
        post_result = spi.solve_post_process(
            robot, robot_arm, palm, rots[turn_index],
            stop=post_process_ik_stop)
        if post_result is not None:
            picked = (turn_index, angle_vectors[candidate_index],
                     base_poses[candidate_index], post_result)
            break
        if fallback is None:
            fallback = (turn_index, angle_vectors[candidate_index],
                       base_poses[candidate_index], None)
    if picked is None:
        picked = fallback
    stage_b_time = time.time() - t_loop_start
    stage_b_success = picked is not None and picked[3] is not None
    return picked, dict(
        stage_a_success=stage_a_success, stage_a_time=stage_a_time,
        stage_b_success=stage_b_success, stage_b_time=stage_b_time)


# ``--collision-pairs`` に指定できる特殊値。
MODE_NONE = 'none'
MODE_NONE_GD = 'none-gd'


def resolve_pairs_config(label, robot):
    """``label`` (``MODE_NONE``/``MODE_NONE_GD``/JSONファイルパス) から
    ``(collision_pairs, verification_pairs, n_pairs, self_collision,
    collision_link_list)`` を作る。事後検証 (``verification_pairs``) は
    ロボット構造だけで決まり最適化用の ``collision_pairs`` に依存しない
    ため、``label`` の値に関わらず常にフル (総当たり) で計算する。

    * ``MODE_NONE``: 最適化を完全に無効にする (``self_collision=False``、
      ``collision_link_list=None``)。``'jacobian'`` 法になるが、
      docstring の注意点の通り ``use_base`` 併用時は人物ごとに再コンパイル
      が発生する。
    * ``MODE_NONE_GD``: ``self_collision=True`` だが ``collision_link_
      list`` にリンクを1つだけ渡す (組み合わせが作れないので実際の
      ペアは0個)。``'gradient_descent'`` 法になり、目標値に依存しない
      キャッシュの恩恵を受けられる。
    * それ以外 (ファイルパス): ``collision_pairs`` をそのまま使う
      (``collision_link_list=None`` -- ``batch_inverse_kinematics`` が
      ``collision_pairs`` から自動導出する)。
    """
    verification_pairs = spi.build_collision_verification_pairs(robot, 'r')
    if label == MODE_NONE:
        return None, verification_pairs, 0, False, None
    if label == MODE_NONE_GD:
        return None, verification_pairs, 0, True, [robot.body_link]
    pairs = spi.load_collision_pairs(label, robot)
    return pairs, verification_pairs, len(pairs), True, None


def solve_one_grid_point(robot, palm_dir, human_dir, robot_arm_arg,
                         attempts_per_pose, collision_ik_stop,
                         pairs_label, collision_pairs, verification_pairs,
                         base_limits, self_collision, collision_link_list):
    """1つのグリッド点 (attempts_per_pose x collision_ik_stop x
    collision_pairs設定) について全人物を解き、集計結果を返す。
    ``solve_palm_ik.main()`` の1人ぶんのループ相当をここに直接書いている
    のは、``pick_verified_candidate_timed`` による段階A/B計測を差し込む
    ため (``solve_palm_ik.py`` をサブプロセスとして呼ぶ既存スクリプト群
    ``build_collision_pairs.py``/``analyze_collision_pairs.py`` とは異なる
    設計)。``self_collision``/``collision_link_list`` は
    ``resolve_pairs_config`` が決めたものをそのまま使う (``MODE_NONE_GD``
    のときだけ ``collision_link_list`` が非 ``None`` になる)。
    """
    files = spi.iter_palm_files(palm_dir)

    n_target = 0
    sum_collision_ik_time = 0.0
    n_stage_a = 0
    sum_stage_a_time = 0.0
    n_stage_b = 0
    sum_stage_b_time = 0.0

    t_wall_start = time.time()
    for path in files:
        palms = spi.load_palm_json(path)
        human_hand = palms.get('offered_hand')
        palm = palms.get(human_hand) if human_hand in ('L', 'R') else None
        if palm is None:
            continue
        robot_arm = (spi.DEFAULT_ROBOT_ARM[human_hand]
                    if robot_arm_arg == 'auto' else robot_arm_arg)

        skeleton_path = os.path.join(human_dir, os.path.basename(path))
        if os.path.exists(skeleton_path):
            joint_positions = spi.load_skeleton_json(skeleton_path)
            offset = spi.human_translation_offset(joint_positions)
            joint_positions = spi.translate_joint_positions(
                joint_positions, offset)
            collision_obstacles = (
                spi.human_body_obstacles(joint_positions)
                if collision_pairs is not None else [])
            palm = spi.translate_palm(palm, offset)
        else:
            joint_positions = None
            collision_obstacles = []

        n_target += 1
        spi.seed_arm_pose(robot, robot_arm)
        whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
        move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
        target_pos = spi.palm_target_position(palm)
        rots = spi.palm_to_target_rots(palm, robot_arm)
        target_coords = [spi.Coordinates(pos=target_pos.tolist(), rot=rot)
                         for rot in rots]

        effective_collision_pairs = collision_pairs
        if collision_pairs is not None and not collision_obstacles:
            effective_collision_pairs = [
                (link_a, other) for link_a, other in collision_pairs
                if not isinstance(other, int)]

        restore_joint_range = spi.restrict_joint_range_margin(
            whole_body.link_list,
            spi.DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO)
        t0 = time.time()
        try:
            angle_vectors, base_poses, success_flags, _ = \
                robot.batch_inverse_kinematics(
                    target_coords=target_coords,
                    move_target=move_target,
                    link_list=whole_body.link_list,
                    position_mask=True, rotation_mask=True,
                    stop=collision_ik_stop,
                    thre=spi.DEFAULT_COLLISION_IK_THRE,
                    rthre=spi.DEFAULT_COLLISION_IK_RTHRE,
                    initial_angles='current',
                    attempts_per_pose=attempts_per_pose,
                    return_all_attempts=True,
                    backend='jax',
                    use_base='planar', base_limits=base_limits,
                    collision_link_list=collision_link_list,
                    collision_obstacles=collision_obstacles,
                    collision_weight=spi.DEFAULT_COLLISION_WEIGHT,
                    collision_margin=spi.DEFAULT_COLLISION_MARGIN,
                    self_collision=self_collision,
                    collision_pairs=effective_collision_pairs,
                    self_collision_weight=None,
                    self_collision_margin=spi.DEFAULT_SELF_COLLISION_MARGIN)
        finally:
            restore_joint_range()
        collision_ik_time = time.time() - t0
        sum_collision_ik_time += collision_ik_time

        effective_verification_pairs = verification_pairs
        if verification_pairs is not None and not collision_obstacles \
                and collision_pairs is not None:
            # collision_obstacles が空でも、collision_pairs (最適化用ファ
            # イル) を使うモードで実際にそのファイルが指定されているときは
            # solve_palm_ik.solve_person_ik と同じ間引きを再現する
            # (--no-human-collision 相当・骨格が無い場合の挙動に合わせる)。
            # ``MODE_NONE`` (collision_pairs is None) のときはこの分岐に
            # 入らず、検証は常にフル (人体セグメント参照を含む) のまま。
            effective_verification_pairs = [
                (link_a, other) for link_a, other in verification_pairs
                if not isinstance(other, int)]

        _, timing = pick_verified_candidate_timed(
            robot, success_flags, angle_vectors, base_poses,
            effective_verification_pairs, joint_positions,
            spi.DEFAULT_COLLISION_VERIFY_TOLERANCE, robot_arm, palm, rots,
            attempts_per_pose,
            post_process_ik_stop=spi.DEFAULT_POST_PROCESS_IK_STOP)

        if timing['stage_a_success']:
            n_stage_a += 1
            sum_stage_a_time += collision_ik_time + timing['stage_a_time']
        if timing['stage_b_success']:
            n_stage_b += 1
            sum_stage_b_time += collision_ik_time + timing['stage_b_time']

    wall_elapsed = time.time() - t_wall_start
    return dict(
        n_target=n_target,
        wall_elapsed=wall_elapsed,
        wall_per_person=wall_elapsed / n_target if n_target else float('nan'),
        stage_a_success_rate=n_stage_a / n_target if n_target else float('nan'),
        stage_a_time_per_person=(
            sum_stage_a_time / n_stage_a if n_stage_a else float('nan')),
        stage_b_success_rate=n_stage_b / n_target if n_target else float('nan'),
        stage_b_time_per_person=(
            sum_stage_b_time / n_stage_b if n_stage_b else float('nan')),
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--human-poses-dir', type=str, default=None,
        help='人物の骨格 JSON のディレクトリ (既定は run_pipeline_test.py '
            'と同様に /tmp 以下に一時ディレクトリを作ってその場で生成する。'
            '既存のデータセットを使い回す場合はここに明示的にパスを指定 '
            'する)。')
    parser.add_argument(
        '--palm-poses-dir', type=str, default=None,
        help='掌の位置姿勢 JSON のディレクトリ (既定は --human-poses-dir と '
            '同様に /tmp 以下に一時ディレクトリを作ってその場で生成する)。')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='--human-poses-dir/--palm-poses-dir を指定 '
                            'しなかったときに生成する人数 (既定 500)。')
    parser.add_argument('--attempts-per-pose', type=int, nargs='+',
                        default=[64])
    parser.add_argument('--collision-ik-stop', type=int, nargs='+',
                        default=[500])
    parser.add_argument(
        '--collision-pairs', type=str, nargs='+', default=[MODE_NONE_GD],
        help='各エントリは "{none}" (最適化は無効・jacobian法。人物ごとに '
            '再コンパイルが発生し実運用の速度見積もりには不向き、参考用) / '
            '"{none_gd}" (実質ペア0だがgradient_descent法。腕ごとに1回の '
            'コンパイルで済み、干渉回避なしの実運用速度を知りたいときは '
            'こちらを使う) / collision_pairs.json 形式のファイルパス、の '
            'いずれか。事後検証はどれでも常にフルで行う。'
            .format(none=MODE_NONE, none_gd=MODE_NONE_GD))
    parser.add_argument('--robot-arm', choices=['auto', 'r', 'l'],
                        default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--python', type=str, default=sys.executable,
                        help='人物生成・掌推定に使うPython (既定はこの '
                            'スクリプトと同じインタプリタ)。')
    args = parser.parse_args()

    if args.human_poses_dir is None or args.palm_poses_dir is None:
        base_dir = tempfile.mkdtemp(prefix='aero_demo_grid_search_',
                                    dir='/tmp')
        print('作業ディレクトリ: {}'.format(base_dir))
        if args.human_poses_dir is None:
            args.human_poses_dir = os.path.join(base_dir, 'skeletons')
        if args.palm_poses_dir is None:
            args.palm_poses_dir = os.path.join(base_dir, 'palms')

    generate_dataset(args.python, args.human_poses_dir, args.palm_poses_dir,
                     args.num_samples, args.seed)

    robot = Aero(use_hand=False)
    spi.restrict_elbow_range(robot)
    spi.apply_collision_model(robot)
    base_limits = [tuple(spi.DEFAULT_BASE_X_RANGE),
                  tuple(spi.DEFAULT_BASE_Y_RANGE),
                  tuple(spi.DEFAULT_BASE_YAW_RANGE)]

    combos = list(itertools.product(
        args.attempts_per_pose, args.collision_ik_stop, args.collision_pairs))
    print('[grid] {} 通りの組み合わせを、各ウォームアップ+本計測の2回で '
         '計測します。'.format(len(combos)))

    rows = []
    for attempts, stop, pairs_label in combos:
        if args.seed is not None:
            np.random.seed(args.seed)
        (collision_pairs, verification_pairs, n_pairs, self_collision,
         collision_link_list) = resolve_pairs_config(pairs_label, robot)
        label = '{}pairs_attempts{}_stop{}'.format(
            pairs_label if pairs_label in (MODE_NONE, MODE_NONE_GD)
            else n_pairs, attempts, stop)
        print('\n=== {} ==='.format(label))

        # 1. ウォームアップ (使い捨て、JITコンパイルを消化するだけ)
        solve_one_grid_point(
            robot, args.palm_poses_dir, args.human_poses_dir,
            args.robot_arm, attempts, stop, pairs_label, collision_pairs,
            verification_pairs, base_limits, self_collision,
            collision_link_list)
        # 2. 本計測 (定常状態)
        if args.seed is not None:
            np.random.seed(args.seed)
        result = solve_one_grid_point(
            robot, args.palm_poses_dir, args.human_poses_dir,
            args.robot_arm, attempts, stop, pairs_label, collision_pairs,
            verification_pairs, base_limits, self_collision,
            collision_link_list)

        row = dict(
            label=label, attempts_per_pose=attempts,
            collision_ik_stop=stop, collision_pairs=pairs_label,
            n_pairs=n_pairs, **result)
        rows.append(row)
        print('  n_target={n_target} '
             'stageA(事後検証まで): 成功率={stage_a_success_rate:.1%} '
             '時間={stage_a_time_per_person:.4f}s/人 | '
             'stageB(押し込み+視線まで): 成功率={stage_b_success_rate:.1%} '
             '時間={stage_b_time_per_person:.4f}s/人'.format(**row))

    print_ranking(rows)


def print_ranking(rows):
    """段階Bの成功率(降順)→段階Bの時間(昇順)の順でグリッド点を並べ替え、
    ランキングと推奨構成をプリント文で表示する(CSV等のファイル出力は
    行わない)。段階Bが1件も成功しなかった行 (時間が NaN) は最後に回す。"""
    def sort_key(row):
        rate = row['stage_b_success_rate']
        time_ = row['stage_b_time_per_person']
        rate = rate if rate == rate else -1.0  # NaN -> 最下位扱い
        time_ = time_ if time_ == time_ else float('inf')  # NaN -> 最下位
        return (-rate, time_)

    ranked = sorted(rows, key=sort_key)

    print('\n=== ランキング (段階Bの成功率が高い順 → 同率なら時間が短い順) ===')
    print('    (事後の干渉検証はどの構成でも常にフル(総当たり)で行っている)')
    for rank, row in enumerate(ranked, start=1):
        print(
            '{:2d}位: attempts={:<3} stop={:<4} pairs={:<20} (n_pairs={:<2}) '
            '| stageA 成功率={:.1%} 時間={:.4f}s/人 '
            '| stageB 成功率={:.1%} 時間={:.4f}s/人'.format(
                rank, row['attempts_per_pose'], row['collision_ik_stop'],
                row['collision_pairs'], row['n_pairs'],
                row['stage_a_success_rate'], row['stage_a_time_per_person'],
                row['stage_b_success_rate'], row['stage_b_time_per_person']))

    best = ranked[0]
    print('\n[推奨] attempts={} stop={} pairs={} (n_pairs={}): '
         'stageB 成功率={:.1%}・時間={:.4f}秒/人 (全構成中で最良)'.format(
             best['attempts_per_pose'], best['collision_ik_stop'],
             best['collision_pairs'], best['n_pairs'],
             best['stage_b_success_rate'], best['stage_b_time_per_person']))


if __name__ == '__main__':
    main()
