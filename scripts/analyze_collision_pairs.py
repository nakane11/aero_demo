#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""干渉回避を無効にして解いた ``solve_palm_ik.py --no-human-collision
--no-self-collision`` の結果 (大量の「干渉を考慮しない」IK 解) を使って、
実際にどのリンクの組み合わせがどれくらい近づく (干渉しうる) かを集計する。

目的は、``collision_link_list_for_arm``/自己干渉ペアのうち「原理的・
実用上ほぼ干渉しない」組み合わせを見つけ、干渉回避付きバッチ IK の
コスト (JAX コンパイル・各反復の計算量) を減らすための削減候補を洗い出す
こと (``main``)。あわせて、``build_collision_pairs.py`` が「実際に干渉した
組み合わせ」を人数付きで拾い上げられるよう、集計処理だけを切り出した
``analyze_handshake_dir`` を提供する。

集計するのは 2 種類:

* 人体 (``human_body_obstacles`` と同じセグメント) と各ロボットリンク
  (``collision_link_list_for_arm``) の距離。
* ロボットリンク同士 (自己干渉、``create_self_collision_pairs`` と同じ
  隣接除外) の距離。

干渉回避付きバッチ IK (``batch_inverse_kinematics`` の ``backend='jax'``
実装) は勾配降下法での最適化のために各リンクを少数の球へ近似する
(``skrobot.kinematics.differentiable._build_collision_setup`` の
``extract_collision_spheres``) が、この分析スクリプトは最適化を行わず
実際に干渉したかどうかを調べるだけなので、その近似は使わない。各リンクは
``apply_collision_model`` が差し替えたプリミティブ近似形状の
``collision_mesh`` (``trimesh.Trimesh``) の頂点をそのまま (半径 0 の点群
として) 使う。

Usage
-----
    python3 solve_palm_ik.py --no-human-collision --no-self-collision \\
        --input-dir random_palm_poses --skeleton-dir random_human_poses \\
        --output-dir random_handshake_poses_nocol
    python3 analyze_collision_pairs.py \\
        --handshake-dir random_handshake_poses_nocol \\
        --skeleton-dir random_human_poses
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.planner.trajectory_optimization.collision import (  # noqa: E402
    create_self_collision_pairs)

from solve_palm_ik import (  # noqa: E402
    HAND_FINGER_LANDMARKS, HAND_FINGER_RADIUS, HAND_PALM_LANDMARKS,
    HAND_PALM_RADIUS, HUMAN_COLLISION_SEGMENTS, HUMAN_FRONT_DISTANCE,
    apply_collision_model, collision_link_list_for_arm,
    human_translation_offset, load_skeleton_json, translate_joint_positions)

# human_body_obstacles (solve_palm_ik.py) が骨格の関節が欠けているときに
# 埋めるダミー距離と同じ値。ここでも「欠けている部位は遠くにある」ものと
# して扱うのに使う (実際の干渉回避と同じ意味づけ)。
_DUMMY_DISTANCE = 100.0

_FINGER_LABELS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def human_capsules(joint_positions):
    """``solve_palm_ik.human_body_obstacles`` と同じ順序・同じ部位の
    カプセル (線分 2 端点 + 半径) のリストと、対応する名前のリストを返す。
    骨格の関節が欠けている部位は ``_DUMMY_DISTANCE`` だけ離れた点に潰す
    (半径だけの球として扱われるので十分遠ければ干渉回避のコストに
    影響しないのは human_body_obstacles と同じ)。
    """
    caps = []
    names = []
    dummy = np.array([_DUMMY_DISTANCE] * 3)
    for name_a, name_b, radius in HUMAN_COLLISION_SEGMENTS:
        if name_a in joint_positions and name_b in joint_positions:
            p0 = np.asarray(joint_positions[name_a], dtype=np.float64)
            p1 = np.asarray(joint_positions[name_b], dtype=np.float64)
        else:
            p0 = p1 = dummy
        caps.append((p0, p1, radius))
        names.append('{}-{}'.format(name_a, name_b))
    for side in ('R', 'L'):
        palm_names = ['{}Hand{}'.format(side, idx)
                     for idx in HAND_PALM_LANDMARKS]
        if all(name in joint_positions for name in palm_names):
            pts = np.array([joint_positions[name] for name in palm_names],
                           dtype=np.float64)
            center = pts.mean(axis=0)
        else:
            center = dummy
        caps.append((center, center, HAND_PALM_RADIUS))
        names.append('{}_palm'.format(side))
        for (base_idx, tip_idx), label in zip(
                HAND_FINGER_LANDMARKS, _FINGER_LABELS):
            base_name = '{}Hand{}'.format(side, base_idx)
            tip_name = '{}Hand{}'.format(side, tip_idx)
            if base_name in joint_positions and tip_name in joint_positions:
                p0 = np.asarray(joint_positions[base_name], dtype=np.float64)
                p1 = np.asarray(joint_positions[tip_name], dtype=np.float64)
            else:
                p0 = p1 = dummy
            caps.append((p0, p1, HAND_FINGER_RADIUS))
            names.append('{}_{}'.format(side, label))
    return caps, names


def segment_point_distance(p0, p1, c):
    """線分 ``p0``-``p1`` と点 ``c`` の最短距離。"""
    d = p1 - p0
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        t = 0.0
    else:
        t = np.clip(float(np.dot(c - p0, d)) / denom, 0.0, 1.0)
    closest = p0 + t * d
    return float(np.linalg.norm(c - closest))


def segment_points_distance(p0, p1, points):
    """線分 ``p0``-``p1`` と、複数の点 ``points`` (``(N, 3)``) それぞれとの
    最短距離 (``(N,)``)。``segment_point_distance`` のベクトル化版
    (メッシュ頂点をまとめて処理するのに使う)。"""
    d = p1 - p0
    denom = float(np.dot(d, d))
    if denom < 1e-12:
        t = np.zeros(len(points))
    else:
        t = np.clip((points - p0) @ d / denom, 0.0, 1.0)
    closest = p0 + t[:, np.newaxis] * d
    return np.linalg.norm(points - closest, axis=1)


def load_result(path):
    with open(path) as f:
        return json.load(f)


def analyze_handshake_dir(handshake_dir, skeleton_dir,
                          human_front_distance=HUMAN_FRONT_DISTANCE,
                          dist_threshold=0.0):
    """``handshake_dir`` (``solve_palm_ik.py`` の出力) と ``skeleton_dir``
    (骨格 JSON) を読み、自己干渉・人体との干渉それぞれの組み合わせごとの
    「全サンプル中の最小距離」と「``dist_threshold`` [m] 未満まで近づいた
    (干渉した) サンプル数」を集計する。``main`` (ファイル入出力・レポート
    表示) と ``build_collision_pairs.py`` (干渉ペアの自動抽出) の両方から
    呼べるよう、集計処理だけを切り出したもの。

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
        result = load_result(path)
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


def main():
    parser = argparse.ArgumentParser(
        description='干渉回避なしで解いた solve_palm_ik.py の結果を集計し、'
                    'ロボットの各リンクが人体・自分自身のどの部位と実際に '
                    '近づいているかを調べる (干渉回避の対象から外せる組み '
                    '合わせを見つけるための分析用)。')
    parser.add_argument(
        '--handshake-dir', type=str, required=True,
        help='solve_palm_ik.py --no-human-collision --no-self-collision '
            'の出力ディレクトリ。')
    parser.add_argument(
        '--skeleton-dir', type=str, required=True,
        help='骨格 JSON (generate_random_human_poses.py の出力) の '
            'ディレクトリ (--handshake-dir と同じファイル名で対応)。')
    parser.add_argument(
        '--human-front-distance', type=float, default=HUMAN_FRONT_DISTANCE,
        help='solve_palm_ik.py に渡したのと同じ --human-front-distance '
            '(既定 {:.1f})。骨格の平行移動をソルバと同じにするために必要。'
            .format(HUMAN_FRONT_DISTANCE))
    parser.add_argument(
        '--safe-threshold', type=float, default=0.30,
        help='この距離 [m] より近づいたことが 1 度も無い組み合わせを '
            '「安全 (除外候補)」として報告する (既定 0.30)。')
    parser.add_argument(
        '--top', type=int, default=25,
        help='「よく干渉する組み合わせ」として表示する上位件数 (既定 25)。')
    args = parser.parse_args()

    stats = analyze_handshake_dir(
        args.handshake_dir, args.skeleton_dir,
        human_front_distance=args.human_front_distance)
    self_min_dist = stats['self_min_dist']
    human_min_dist = stats['human_min_dist']
    self_pairs = stats['self_pairs']
    n_links = len(stats['link_names'])
    cap_names = stats['cap_names']
    n_samples = stats['n_samples']
    print('collision_link_list: {} 個のリンク'.format(n_links))
    print('self-collision pairs (隣接除外後): {} 組'.format(len(self_pairs)))

    print('\n集計に使った (干渉回避なしで解けた) サンプル数: {}'
          .format(n_samples))
    if n_samples == 0:
        print('サンプルが無いため集計できません。')
        return

    def report(title, min_dist_map, total_possible):
        items = sorted(min_dist_map.items(), key=lambda kv: kv[1])
        n_safe = sum(1 for _, d in items if d > args.safe_threshold)
        print('\n=== {} ===' .format(title))
        print('観測された組み合わせ数: {} / 全組み合わせ数: {}'.format(
            len(items), total_possible))
        print('常に {} m より遠かった (除外候補): {} 組'.format(
            args.safe_threshold, n_safe))
        print('-- 最も近づいた上位 {} 組 --'.format(args.top))
        for key, dist in items[:args.top]:
            flag = '!!' if dist < 0 else '  '
            print('{} {:8.4f} m  {}'.format(flag, dist, key))
        return items

    self_items = report(
        '自己干渉 (ロボットリンク同士)', self_min_dist, len(self_pairs))
    human_items = report(
        '人体との干渉 (ロボットリンク x 人体セグメント)',
        human_min_dist, n_links * len(cap_names))

    safe_self = [key for key, d in self_items if d > args.safe_threshold]
    safe_human = [key for key, d in human_items if d > args.safe_threshold]

    out = dict(
        n_samples=n_samples,
        n_links=n_links,
        safe_threshold=args.safe_threshold,
        self_collision_min_dist={
            '{}|{}'.format(*k): d for k, d in self_items},
        human_collision_min_dist={
            '{}|{}'.format(*k): d for k, d in human_items},
        safe_self_pairs=['{}|{}'.format(*k) for k in safe_self],
        safe_human_pairs=['{}|{}'.format(*k) for k in safe_human],
    )
    out_path = os.path.join(
        os.path.dirname(args.handshake_dir.rstrip('/')) or '.',
        'collision_pair_analysis.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print('\n詳細な結果を {} に保存しました。'.format(out_path))


if __name__ == '__main__':
    main()
