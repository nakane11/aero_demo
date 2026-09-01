#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人体骨格 (MediaPipe 形式の関節位置の dict) を入力とし、左右の掌の位置
姿勢を推定して JSON として保存する。

``RandomSkeletonGenerator`` (``generate_random_human_poses.py``) が出力
する合成骨格でも、``people_pose_estimator_ros.RosPeoplePoseEstimator`` /
``fake_people_pose_estimator_ros.FakeRosPeoplePoseEstimator`` のような
実カメラ (MediaPipe) ベースの推定を骨格の生成元として使う場合でも、同じ
``PalmPoseEstimator.estimate(joint_positions)`` で掌の位置姿勢を求められる
ようにしてある。骨格の生成元が変わっても入力形式 (関節名 -> [x, y, z] の
dict) は変わらないので、この推定器は生成元を一切区別しない。

手のランドマーク (MediaPipe の ``RHand0``..``RHand20`` / ``LHand*``,
``RandomSkeletonGenerator(include_hand=True)`` や実際の MediaPipe 推定が
出力する) が 3 点以上揃っている側だけ、``aero_demo.palm_plane.
fit_palm_plane`` で手首 + 知節 (MCP) へ平面を SVD フィットする。
``human_palm_contact_behavior.py`` が実際にロボットを動かすときと同じ
関数・同じ判定 (点がほぼ一直線でないか等) を使うので、実カメラの推定結果
(関節ごとに欠測がある、本物の手首の捻りが乗っている、等) に対しても同じ
ロジックで動く。手のランドマークが 3 点未満の側 (体の関節だけの入力、
または実推定でその手が丸ごとロストしたフレーム) は推定しない
(``None``) -- 指のランドマークを見ずに前腕の向きだけから掌の姿勢を仮定
するようなフォールバックは持たない。

ローカル座標系は
    +x : 指先方向 (手首 -> 指先)
    +y : 手の甲 -> 掌の方向 (掌の法線, 体の外側を向く)
    +z : +x と +y に直交する軸 (+x, +y に対して右手系になるように
         ``z = x cross y`` で決める)
とし、手は左右 (``R``/``L``) を区別して独立に推定する。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py \
        --num-samples 100 --output-dir /tmp/random_human_poses
    rosrun aero_demo estimate_palm_poses.py \
        --input-dir /tmp/random_human_poses \
        --output-dir /tmp/random_palm_poses
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import palm_plane  # noqa: E402  (パス追加後に import)


class PalmPoseEstimator(object):
    """骨格 (関節位置の dict) から左右の掌の位置姿勢を推定する.

    Examples
    --------
    >>> from generate_random_human_poses import RandomSkeletonGenerator
    >>> generator = RandomSkeletonGenerator(seed=0)
    >>> pose = generator.generate()
    >>> estimator = PalmPoseEstimator()
    >>> palms = estimator.estimate(pose['joint_positions'])
    >>> palms['R']['position']
    [0.32, -0.18, 0.95]
    """

    def estimate(self, joint_positions):
        """左右の掌の位置姿勢を推定する.

        Parameters
        ----------
        joint_positions : dict
            関節名 (MediaPipe 形式) -> [x, y, z] (ロボット座標系)。骨格の
            生成元 (``RandomSkeletonGenerator`` / 実カメラの推定など) は
            問わない。

        Returns
        -------
        dict
            ``{'R': palm, 'L': palm}``。手のランドマーク
            (``{side}Hand0``..``{side}Hand20``) が 3 点未満の側は
            ``None``。``palm`` は次のキーを持つ dict:

            ``position``
                掌中心の位置 [x, y, z]。
            ``x_axis`` / ``y_axis`` / ``z_axis``
                掌のローカル座標系の各軸 (単位ベクトル, ワールド座標系)。
                +x = 指先方向, +y = 手の甲->掌の方向, +z = x cross y。
            ``rot``
                上記 3 軸を列に並べた 3x3 回転行列 (``skrobot.coordinates.
                Coordinates(pos=position, rot=rot)`` にそのまま渡せる)。
        """
        joints = {name: np.asarray(p, dtype=np.float64)
                 for name, p in joint_positions.items()}
        return {side: self._estimate_one(joints, side) for side in ('R', 'L')}

    def _estimate_one(self, joints, side):
        points = {}
        for i in palm_plane.PLANE_LANDMARKS:
            key = '{}Hand{}'.format(side, i)
            if key in joints:
                points[i] = joints[key]
        plane = palm_plane.fit_palm_plane(points, hand=side)
        if plane is None:
            return None

        # palm_plane.py の plane.rot はロボットの手先座標系向け (+Y =
        # -normal) なので、人物自身の掌フレーム (+y = 手の甲->掌方向 =
        # normal そのもの) はここで組み直す。
        x_axis = plane.finger_dir
        y_axis = plane.normal
        z_axis = np.cross(x_axis, y_axis)
        rot = np.column_stack([x_axis, y_axis, z_axis])
        return dict(
            position=[float(v) for v in plane.center],
            x_axis=[float(v) for v in x_axis],
            y_axis=[float(v) for v in y_axis],
            z_axis=[float(v) for v in z_axis],
            rot=[[float(v) for v in row] for row in rot])


def load_skeleton_json(path):
    """``generate_random_human_poses.save_json`` が保存した 1 人分の JSON を読む."""
    with open(path) as f:
        data = json.load(f)
    return data['joint_positions']


def save_json(palms, path):
    """``PalmPoseEstimator.estimate`` の戻り値を JSON として保存する."""
    with open(path, 'w') as f:
        json.dump(palms, f, indent=2)


def iter_skeleton_files(input_dir, pattern='*.json'):
    """``input_dir`` 内の骨格 JSON をファイル名順に列挙する."""
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def main():
    parser = argparse.ArgumentParser(
        description='骨格 JSON (関節位置の dict) から左右の掌の位置姿勢を '
                    '推定し、JSON として保存する。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='骨格 JSON の入力ディレクトリ。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_palm_poses'),
        help='掌の位置姿勢 JSON の保存先ディレクトリ。')
    args = parser.parse_args()

    files = iter_skeleton_files(args.input_dir)
    if not files:
        print('{} に骨格 JSON が見つかりません。先に '
              'generate_random_human_poses.py を実行してください。'.format(
                  args.input_dir))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    estimator = PalmPoseEstimator()

    for i, path in enumerate(files):
        joint_positions = load_skeleton_json(path)
        palms = estimator.estimate(joint_positions)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        save_json(palms, out_path)
        print('[{}/{}] saved {}'.format(i + 1, len(files), out_path))


if __name__ == '__main__':
    main()
