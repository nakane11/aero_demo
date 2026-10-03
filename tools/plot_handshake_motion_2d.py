#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``run_pipeline_test.py --plan-motion`` を実行し、軌道計画の結果を人物
ごとに真上から見た 2 次元の図 (PNG) にする。

台車の軌道 (lead-in は破線)・途中の向き、人の立ち位置・正面方向、差し出した
手を描く。座標は人物を Aero の前方へ平行移動した IK・軌道計画の座標系 [m]。

Usage
-----
    # 位置引数とオプションは run_pipeline_test.py にそのまま渡す
    python3 tools/plot_handshake_motion_2d.py 20 --seed 3 \\
        --initial-base-pose 5 0 3.14
    # 既存の作業ディレクトリを描き直す
    python3 tools/plot_handshake_motion_2d.py \\
        --work-dir /tmp/aero_demo_pipeline_xxx
"""

import argparse
import json
import math
import os
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
_PKG_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import json_io  # noqa: E402  (パス追加後に import)
import solve_palm_ik as spik  # noqa: E402

WORK_DIR_PREFIX = '作業ディレクトリ: '

# 向きを表す矢印の長さ [m]。
ARROW_LENGTH = 0.35

# 台車の向きの矢印を描く軌道上の点の数 (経路長で等間隔)。
N_HEADING_SAMPLES = 4


def sample_by_arc_length(points, n):
    """折れ線を経路長で等分した内側 ``n`` 点の ``(位置, 頂点 index, 区間内の割合)``。"""
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = cumulative[-1]
    samples = []
    if total < 1e-6:
        return samples
    for k in range(1, n + 1):
        target = total * k / (n + 1)
        i = int(np.searchsorted(cumulative, target, side='right')) - 1
        i = min(max(i, 0), len(seg) - 1)
        t = (target - cumulative[i]) / seg[i] if seg[i] > 1e-9 else 0.0
        samples.append((points[i] + (points[i + 1] - points[i]) * t, i, t))
    return samples


def run_pipeline(pipeline_args):
    """``run_pipeline_test.py --plan-motion`` を実行し作業ディレクトリを返す。"""
    command = [sys.executable,
              os.path.join(_SCRIPTS_DIR, 'run_pipeline_test.py')]
    command += pipeline_args
    if '--plan-motion' not in pipeline_args:
        command.append('--plan-motion')
    work_dir = None
    process = subprocess.Popen(
        command, cwd=_SCRIPTS_DIR, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
        if line.startswith(WORK_DIR_PREFIX):
            work_dir = line[len(WORK_DIR_PREFIX):].strip()
    if process.wait() != 0:
        sys.exit('run_pipeline_test.py が exit code {} で失敗しました。'
                 .format(process.returncode))
    if work_dir is None:
        sys.exit('run_pipeline_test.py の出力に作業ディレクトリが '
                 '見つかりませんでした。')
    return work_dir


def draw_arrow(ax, xy, yaw, color, label=None, length=ARROW_LENGTH,
               width=0.025):
    ax.arrow(xy[0], xy[1], length * math.cos(yaw), length * math.sin(yaw),
             width=width, head_width=width * 4, head_length=width * 4,
             length_includes_head=True, color=color, label=label, zorder=4)


def plot_person(motion, handshake, joint_positions, out_path, title):
    """1 人分の図を ``out_path`` に保存する。"""
    lead_in = motion.get('lead_in_waypoints') or []
    waypoints = motion['waypoints']
    path = lead_in + waypoints
    initial, goal = path[0], waypoints[-1]

    def xy(wp):
        return np.asarray(wp['base_position'][:2], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 7))

    # lead-in は waypoints[0] まで破線。
    if lead_in:
        line = np.array([xy(wp) for wp in lead_in + waypoints[:1]])
        ax.plot(line[:, 0], line[:, 1], '--', color='tab:blue', lw=1.5,
                label='lead-in (straight)')
    line = np.array([xy(wp) for wp in waypoints])
    ax.plot(line[:, 0], line[:, 1], '-', color='tab:blue', lw=2,
            label='planned approach')
    ax.plot(*xy(waypoints[0]), 'o', color='tab:blue', ms=4, zorder=3,
            label='approach start')

    # 途中の台車の向き (yaw は線形補間)。
    points = np.array([xy(wp) for wp in path])
    for k, (point, i, t) in enumerate(
            sample_by_arc_length(points, N_HEADING_SAMPLES)):
        yaw = path[i]['base_yaw'] + (
            path[i + 1]['base_yaw'] - path[i]['base_yaw']) * t
        ax.plot(point[0], point[1], 'o', color='tab:gray', ms=4, zorder=3)
        draw_arrow(ax, point, yaw, 'tab:gray', length=ARROW_LENGTH * 0.7,
                   width=0.015,
                   label='robot heading on path' if k == 0 else None)

    # 人間の立ち位置・正面方向・差し出した手。
    human_xy = spik.human_standing_xy(joint_positions)
    if human_xy is not None:
        ax.plot(human_xy[0], human_xy[1], 'o', color='tab:red', ms=10,
                zorder=5, label='human')
        human_yaw = spik.human_facing_yaw(joint_positions)
        if human_yaw is not None:
            draw_arrow(ax, human_xy, human_yaw, 'tab:red', width=0.015,
                       label='human facing')
    if handshake.get('target_position') is not None:
        hand = handshake['target_position']
        ax.plot(hand[0], hand[1], 'x', color='tab:red', ms=8, mew=2,
                zorder=5, label='offered hand')

    # ロボットの初期位置・目標位置とその向き。
    ax.plot(*xy(initial), 's', color='tab:green', ms=9, zorder=5,
            label='robot initial')
    draw_arrow(ax, xy(initial), initial['base_yaw'], 'tab:green')
    ax.plot(*xy(goal), '*', color='tab:purple', ms=14, zorder=5,
            label='robot goal')
    draw_arrow(ax, xy(goal), goal['base_yaw'], 'tab:purple')

    ax.set_aspect('equal', adjustable='datalim')
    ax.margins(0.15)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title(title, fontsize=10)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='軌道計画の結果を人物ごとに 2 次元の図にする。未知の引数は '
                    'run_pipeline_test.py に渡す。')
    parser.add_argument(
        '--work-dir', default=None,
        help='パイプラインを実行せず既存の作業ディレクトリを描く。')
    parser.add_argument(
        '--motion-dir', default=None,
        help='軌道計画の結果 JSON のディレクトリ (既定は <作業ディレクトリ>/motions)。')
    parser.add_argument(
        '--output-dir', default=None,
        help='PNG の保存先 (既定は <作業ディレクトリ>/plots)。')
    args, pipeline_args = parser.parse_known_args()

    if args.work_dir is None:
        if not pipeline_args:
            parser.error('人数 (run_pipeline_test.py の num_people) か '
                         '--work-dir を指定してください。')
        work_dir = run_pipeline(pipeline_args)
    else:
        work_dir = args.work_dir
    motion_dir = args.motion_dir or os.path.join(work_dir, 'motions')
    if not os.path.isdir(motion_dir):
        sys.exit('{} がありません (--plan-motion 付きで実行した作業 '
                 'ディレクトリを指定してください)。'.format(motion_dir))
    output_dir = args.output_dir or os.path.join(work_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    n_plotted = 0
    for motion_path in json_io.iter_json_files(motion_dir):
        name = os.path.basename(motion_path)
        with open(motion_path) as f:
            motion = json.load(f)
        if not motion.get('planned'):
            continue
        with open(os.path.join(work_dir, 'handshakes', name)) as f:
            handshake = json.load(f)
        # 軌道と同じ座標系 (人物を Aero の前方へ平行移動) に直す。
        joint_positions = spik.load_skeleton_json(
            os.path.join(work_dir, 'skeletons', name))
        joint_positions = spik.translate_joint_positions(
            joint_positions, spik.human_translation_offset(
                joint_positions, front_distance=spik.HUMAN_FRONT_DISTANCE))
        verified = bool(motion.get('verified')) and bool(
            motion.get('lead_in_verified', True))
        title = '{}  kind={}  verified={}'.format(
            os.path.splitext(name)[0], motion.get('kind'), verified)
        out_path = os.path.join(output_dir,
                                os.path.splitext(name)[0] + '.png')
        plot_person(motion, handshake, joint_positions, out_path, title)
        n_plotted += 1
    print('{} 人分の図を {} に保存しました。'.format(n_plotted, output_dir))


if __name__ == '__main__':
    main()
