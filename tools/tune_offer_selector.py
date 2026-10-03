#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``OfferedHandSelector`` のパラメータ (判定軸のブレンド・ランプ上限・高さ方向の
重み・分離度の重み) を人手ラベル (``human_label``) 付きサンプルから調整する。

格子からランダムに選んだ候補ごとに、最適な ``score_min`` を全探索して
3 クラス ('R'/'L'/None) の一致率を最大化する。

Usage
-----
    python3 tools/tune_offer_selector.py \\
        --skeleton-dir /path/to/skeletons --palm-dir /path/to/palms
"""

import argparse
import itertools
import json
import os
import random
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import json_io  # noqa: E402

import estimate_palm_poses as epp  # noqa: E402

load_skeleton_json = json_io.load_skeleton_json
iter_skeleton_files = json_io.iter_json_files


# 探索するパラメータの候補
AXIS_BLEND_CHOICES = [0.0, 0.25, 0.5, 0.75, 1.0]
FINGER_RAMP_UPPER_CHOICES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
APPROACH_HEIGHT_SCALE_CHOICES = [0.0, 0.25, 0.5, 0.75, 1.0]
SEPARATION_WEIGHT_CHOICES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

SIDES = ('R', 'L')


def load_samples(skeleton_dir, palm_dir, label_key='human_label'):
    """``label_key`` を持つサンプルの ``(name, joints, label, robot_position)`` を返す.

    ``robot_position`` は判定基準のロボット手先 (base_link)、無ければ ``None``。
    """
    samples = []
    n_unlabeled = 0
    n_no_robot = 0
    for skeleton_path in iter_skeleton_files(skeleton_dir):
        name = os.path.basename(skeleton_path)
        palm_path = os.path.join(palm_dir, name)
        if not os.path.exists(palm_path):
            continue
        with open(palm_path) as f:
            palm_data = json.load(f)
        if label_key not in palm_data:
            n_unlabeled += 1
            continue
        joint_positions = load_skeleton_json(skeleton_path)
        robot_position = palm_data.get('robot_position')
        if robot_position is None:
            n_no_robot += 1
        samples.append((name, joint_positions, palm_data[label_key],
                        robot_position))
    if n_unlabeled:
        print('{} 件は "{}" が無いため除外しました (label_offer_images.py '
              'で人手ラベルを付けてください)。'.format(n_unlabeled, label_key))
    if n_no_robot:
        print('{} 件は掌 JSON に robot_position が無いため、合成骨格向けの '
              '仮のロボット位置 (人物の +x {} m・高さ {} m) で評価します '
              '(実カメラのデータなら extract_skeletons_from_bag.py で抽出し '
              '直してください)。'.format(
                  n_no_robot, epp.ROBOT_FORWARD_DISTANCE,
                  epp.ROBOT_HAND_HEIGHT))
    return samples


def precompute_palms(samples):
    """パラメータに依存しない掌位置姿勢を事前計算する。"""
    plain_estimator = epp.PalmPoseEstimator.__new__(epp.PalmPoseEstimator)
    cache = []
    for name, joint_positions, label, robot_position in samples:
        joints = {n: np.asarray(p, dtype=np.float64)
                 for n, p in joint_positions.items()}
        palms = {side: plain_estimator._estimate_one(joints, side)
                for side in SIDES}
        if robot_position is not None:
            robot_position = np.asarray(robot_position, dtype=np.float64)
        cache.append((name, joints, palms, label, robot_position))
    return cache


def _selector_kwargs(params):
    weights = dict(epp.OFFER_FEATURE_WEIGHTS)
    remaining = 1.0 - weights.pop('separation')
    new_separation = params['separation_weight']
    scale = (1.0 - new_separation) / remaining if remaining > 1e-9 else 0.0
    weights = {k: v * scale for k, v in weights.items()}
    weights['separation'] = new_separation
    return dict(
        weights=weights,
        finger_to_robot_axis_blend=params['axis_blend'],
        finger_to_robot_ramp=(0.0, params['finger_ramp_upper']),
        approach_height_scale=params['approach_height_scale'],
        # score_min はここでは仮の値、best_score_min で決め直す。
        score_min=0.0,
    )


def scores_for_params(cache, params):
    """``(true_side, {'R': score, 'L': score})`` のリスト (閾値前の生スコア、veto は None)。"""
    kwargs = _selector_kwargs(params)
    selector = epp.OfferedHandSelector(**kwargs)
    out = []
    for name, joints, palms, label, robot_position in cache:
        # 抽出時の判定基準のロボット手先へ差し替える。
        selector.robot_position = robot_position
        body = epp._body_frame(joints)
        if body is None:
            out.append((label, {'R': None, 'L': None}))
            continue
        side_scores = {}
        for side in SIDES:
            palm = palms.get(side)
            if palm is None:
                side_scores[side] = None
                continue
            feats = selector._features(joints, body, side, palm)
            side_scores[side] = sum(
                w * feats[key] for key, w in selector.weights.items()) \
                - selector.face_away_penalty * (1.0 - feats['face_to_robot'])
        out.append((label, side_scores))
    return out


def best_score_min(scored_samples):
    """正解率を最大化する ``score_min`` を全探索する.

    Returns ``(閾値, 正解率, 混同行列 [正解][予測])``。
    """
    all_scores = sorted(set(
        s for _, side_scores in scored_samples for s in side_scores.values()
        if s is not None))
    # 各スコアのすぐ下と、何も通さない値。
    candidates = [s - 1e-6 for s in all_scores] + [
        (all_scores[-1] + 1.0) if all_scores else 1.0]

    best = (None, -1.0, None)
    for thresh in candidates:
        correct = 0
        confusion = {}
        for label, side_scores in scored_samples:
            candidates_above = {s: v for s, v in side_scores.items()
                                if v is not None and v >= thresh}
            pred = (max(candidates_above, key=lambda s: candidates_above[s])
                   if candidates_above else None)
            confusion.setdefault(label, {}).setdefault(pred, 0)
            confusion[label][pred] += 1
            if pred == label:
                correct += 1
        acc = correct / len(scored_samples) if scored_samples else 0.0
        if acc > best[1]:
            best = (thresh, acc, confusion)
    return best


def evaluate(cache, params):
    scored = scores_for_params(cache, params)
    thresh, acc, confusion = best_score_min(scored)
    return acc, thresh, confusion


def search(cache, n_random=200, seed=0):
    rng = random.Random(seed)
    param_grid = list(itertools.product(
        AXIS_BLEND_CHOICES, FINGER_RAMP_UPPER_CHOICES,
        APPROACH_HEIGHT_SCALE_CHOICES, SEPARATION_WEIGHT_CHOICES))
    if n_random and n_random < len(param_grid):
        param_grid = rng.sample(param_grid, n_random)

    best = None
    for axis_blend, finger_ramp_upper, height_scale, sep_w in param_grid:
        params = dict(axis_blend=axis_blend,
                     finger_ramp_upper=finger_ramp_upper,
                     approach_height_scale=height_scale,
                     separation_weight=sep_w)
        acc, thresh, confusion = evaluate(cache, params)
        if best is None or acc > best[0]:
            best = (acc, thresh, dict(params), confusion)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_SCRIPTS_DIR, 'random_human_poses'))
    parser.add_argument(
        '--palm-dir', type=str,
        default=os.path.join(_SCRIPTS_DIR, 'random_palm_poses'))
    parser.add_argument('--label-key', type=str, default='human_label')
    parser.add_argument(
        '--n-random', type=int, default=200,
        help='ランダムに選ぶ候補数 (0 で格子を全探索)。')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--output', type=str,
        default=os.path.join(_THIS_DIR, 'tuned_offer_selector_params.json'))
    args = parser.parse_args()

    samples = load_samples(args.skeleton_dir, args.palm_dir, args.label_key)
    if not samples:
        print('{} / {} にサンプルが見つかりません。'.format(
            args.skeleton_dir, args.palm_dir))
        return
    print('{} 件のサンプルを読み込みました。'.format(len(samples)))
    cache = precompute_palms(samples)

    # 参考: 現行デフォルトの精度。
    default_params = dict(
        axis_blend=0.0, finger_ramp_upper=epp.FINGER_TO_ROBOT_RAMP[1],
        approach_height_scale=1.0,
        separation_weight=epp.OFFER_FEATURE_WEIGHTS['separation'])
    default_acc, default_thresh, default_confusion = evaluate(
        cache, default_params)
    print('--- 既定パラメータ ---')
    print('accuracy={:.3f} best_score_min={:.3f}'.format(
        default_acc, default_thresh))
    print('confusion (行=正解, 列=予測):', default_confusion)

    acc, thresh, params, confusion = search(
        cache, n_random=args.n_random, seed=args.seed)
    print()
    print('--- 探索結果のベスト ---')
    print('accuracy={:.3f} (既定比 {:+.3f})'.format(
        acc, acc - default_acc))
    print('params:', params)
    print('score_min={:.3f}'.format(thresh))
    print('confusion (行=正解, 列=予測):', confusion)

    result = dict(
        accuracy=acc, score_min=thresh, params=params,
        confusion={str(k): {str(kk): vv for kk, vv in v.items()}
                  for k, v in confusion.items()},
        n_samples=len(samples),
        default_accuracy=default_acc)
    json_io.save_json(args.output, result)
    print('\n結果を {} に保存しました。'.format(args.output))
    print('OfferedHandSelector に渡すキーワード引数は次の通りです:')
    kwargs = _selector_kwargs(params)
    kwargs['score_min'] = thresh
    print(json.dumps(kwargs, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
