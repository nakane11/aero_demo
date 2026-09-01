#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``RandomSkeletonGenerator`` と ``PalmPoseEstimator`` のインスタンスを
それぞれ作り、``generate_random_human_poses.py`` / ``estimate_palm_poses.py``
の CLI と同じように JSON ファイルを介して骨格を受け渡しし、掌の位置姿勢を
推定できることを確かめる、簡単な結合テスト。

``draw_random_human_poses.py`` は骨格・掌のどちらも JSON ファイルから
読み込む前提になっているので、ここでも同じ ``save_json`` /
``load_skeleton_json`` を使ってファイル経由で受け渡し、そのまま
``draw_random_human_poses.py --input-dir ... --palm-dir ...`` で読める
出力になっていることも確認する。

Usage
-----
    rosrun aero_demo test_generate_and_estimate_palm_poses.py \
        --num-samples 20 --seed 0
"""

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from generate_random_human_poses import (  # noqa: E402
    RandomSkeletonGenerator, save_json as save_skeleton_json)
from estimate_palm_poses import (  # noqa: E402
    PalmPoseEstimator, load_skeleton_json, save_json as save_palm_json)


def check_orthonormal_right_handed(axes, atol=1e-6):
    """(x, y, z) 軸が単位直交・右手系 (z = x cross y) になっているか確かめる."""
    x_axis, y_axis, z_axis = (np.asarray(axes[k]) for k in
                              ('x_axis', 'y_axis', 'z_axis'))
    for name, v in (('x_axis', x_axis), ('y_axis', y_axis), ('z_axis', z_axis)):
        assert abs(np.linalg.norm(v) - 1.0) < atol, \
            '{} is not a unit vector: {}'.format(name, v)
    assert abs(np.dot(x_axis, y_axis)) < atol, 'x_axis, y_axis not orthogonal'
    assert abs(np.dot(y_axis, z_axis)) < atol, 'y_axis, z_axis not orthogonal'
    assert abs(np.dot(z_axis, x_axis)) < atol, 'z_axis, x_axis not orthogonal'
    assert np.allclose(np.cross(x_axis, y_axis), z_axis, atol=atol), \
        'z_axis != x_axis cross y_axis'


def run(num_samples, seed, work_dir):
    skeleton_dir = os.path.join(work_dir, 'skeletons')
    palm_dir = os.path.join(work_dir, 'palms')
    os.makedirs(skeleton_dir, exist_ok=True)
    os.makedirs(palm_dir, exist_ok=True)

    # ---- RandomSkeletonGenerator のインスタンス: 骨格 (関節位置) を作る側
    generator = RandomSkeletonGenerator(seed=seed)
    # ---- PalmPoseEstimator のインスタンス: 骨格から掌の位置姿勢を推定する側
    estimator = PalmPoseEstimator()

    ok = 0
    for i in range(num_samples):
        name = 'human_{:03d}.json'.format(i)
        skeleton_path = os.path.join(skeleton_dir, name)
        palm_path = os.path.join(palm_dir, name)

        # generator が作った骨格を、generate_random_human_poses.py と同じ
        # 形式で一旦 JSON ファイルに書き出す。
        pose = generator.generate()
        save_skeleton_json(pose, skeleton_path)

        # estimator 側は、その JSON ファイルを estimate_palm_poses.py と
        # 同じ手順 (load_skeleton_json) で読み込んで推定する。
        joint_positions = load_skeleton_json(skeleton_path)
        palms = estimator.estimate(joint_positions)
        save_palm_json(palms, palm_path)

        # draw_random_human_poses.py が読む形式と同じかどうかも確かめる。
        with open(palm_path) as f:
            reloaded = json.load(f)

        for side in ('R', 'L'):
            palm = reloaded[side]
            # RandomSkeletonGenerator は既定 (include_hand=True) で両手の
            # ランドマークを含む骨格を作るので、推定は必ず成功する。
            assert palm is not None, \
                '{} palm was not estimated for sample {}'.format(side, i)
            check_orthonormal_right_handed(palm)

            wrist = np.asarray(joint_positions['{}Wrist'.format(side)])
            elbow = np.asarray(joint_positions['{}Elbow'.format(side)])
            forearm_dir = (wrist - elbow) / np.linalg.norm(wrist - elbow)
            # +x (指先方向) は手のランドマーク (知節列) へのフィットで
            # 求めているので、前腕 (肘->手首) の延長方向とはおおむね一致
            # するが、知節の並びのぶんだけ厳密には一致しない。大きく
            # ずれていない (別人の手や誤ったフィットになっていない) こと
            # だけ緩く確認する。
            cos_angle = float(np.dot(palm['x_axis'], forearm_dir))
            assert cos_angle > 0.9, \
                '{} x_axis is far from the forearm direction ' \
                '(cos={:.3f})'.format(side, cos_angle)
        ok += 1

    print('{}/{} 体の骨格を {} 経由 (generate_random_human_poses.save_json '
         '-> estimate_palm_poses.load_skeleton_json/save_json) で受け渡し、'
         '左右の掌の位置姿勢を推定できました (x/y/z 軸は単位直交・右手系で '
         'あることも確認済み)。draw_random_human_poses.py --input-dir {} '
         '--palm-dir {} でそのまま描画できます。'.format(
             ok, num_samples, work_dir, skeleton_dir, palm_dir))


def main():
    parser = argparse.ArgumentParser(
        description='RandomSkeletonGenerator が作った骨格を JSON ファイル '
                    '経由で PalmPoseEstimator に渡し、掌の位置姿勢を推定 '
                    'できることを確認する。')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='テストに使う人物数。')
    parser.add_argument('--seed', type=int, default=0,
                        help='乱数シード (再現性のため既定で固定)。')
    parser.add_argument(
        '--work-dir', type=str,
        default=os.path.join(_THIS_DIR, 'test_palm_pose_pipeline'),
        help='骨格 JSON / 掌 JSON を書き出す作業ディレクトリ。')
    args = parser.parse_args()
    run(args.num_samples, args.seed, args.work_dir)


if __name__ == '__main__':
    main()
