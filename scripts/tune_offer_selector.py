#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``OfferedHandSelector`` (``estimate_palm_poses.py``) のパラメータを、
人手ラベル付きサンプルから調整するツール。

対象にするのは、斜め前に軽く手を挙げて体から離す、より一般的な差し出し
方でも検出できるようにするための 4 つの改善案:

1. ``finger_to_robot_axis_blend``
   「指先がロボットを指しているか」ではなく「掌をロボットに見せている
   か」を評価するよう、判定軸を指先方向 (x_axis) から掌の法線方向
   (y_axis) へブレンドする。
2. ``finger_to_robot_ramp``
   1. の判定のランプ (下限, 上限)。上限を下げるほど、斜めに構えた姿勢
   でも満点になりやすくなる。
3. ``approach_height_scale``
   「脱力位置 -> ロボット」「実際の掌 -> ロボット」の距離差 (approach)
   を計算する際、高さ方向の寄与をどれだけ弱めるか (1.0=従来の 3 次元
   距離、0.0=水平面のみ)。
4. ``weights['separation']``
   「体から離しているか」の重み。

入力は ``estimate_palm_poses.py`` と同じ骨格 JSON (``--skeleton-dir``、
``skeleton.joint_positions`` を持つ) と、対応する掌 JSON
(``--palm-dir``、``draw_random_human_poses.py`` の判定ボタンが書き込む
``human_label`` を正解ラベルとして持つ。無ければ ``offered_hand`` に
フォールバックするが、これは推定値であって正解ではないので警告を出す)。

やることはシンプルなランダムサーチ + 最良点周辺のグリッド微調整で、
各パラメータ候補について:

  - 全サンプルの左右スコアを (score_min を除く全パラメータで) 計算
  - その候補でのスコア分布に対して最適な ``score_min`` を全探索
    (候補点はサンプルのスコア値そのものなので、境界を総当たりできる)
  - 正解ラベルとの一致率 (3 クラス: 'R'/'L'/None) を目的関数にする

を行い、目的関数を最大化する組み合わせを探す。

Usage
-----
    python3 tune_offer_selector.py \
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
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from aero_demo import json_io  # noqa: E402

import estimate_palm_poses as epp  # noqa: E402

load_skeleton_json = json_io.load_skeleton_json
iter_skeleton_files = json_io.iter_json_files


# --- 探索するパラメータの範囲 (1-4) ----------------------------------------
AXIS_BLEND_CHOICES = [0.0, 0.25, 0.5, 0.75, 1.0]
FINGER_RAMP_UPPER_CHOICES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
APPROACH_HEIGHT_SCALE_CHOICES = [0.0, 0.25, 0.5, 0.75, 1.0]
SEPARATION_WEIGHT_CHOICES = [0.10, 0.15, 0.20, 0.25, 0.30]

SIDES = ('R', 'L')


def load_samples(skeleton_dir, palm_dir, label_key='human_label'):
    """(joint_positions, ground_truth_side) のリストを返す.

    ``label_key`` (既定 ``human_label``、人手の正解ラベル) が掌 JSON に
    無いサンプルは、正解が無いので除外する (``offered_hand`` は自動推定値
    であって正解ではないため、フォールバックには使わない)。
    """
    samples = []
    n_unlabeled = 0
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
        samples.append((name, joint_positions, palm_data[label_key]))
    if n_unlabeled:
        print('{} 件は "{}" が無いため除外しました (label_offer_images.py '
              'で人手ラベルを付けてください)。'.format(n_unlabeled, label_key))
    return samples


def precompute_palms(samples):
    """各サンプルの掌位置姿勢 (パラメータに依存しないジオメトリ) を
    事前計算してキャッシュする (探索中に毎回re計算しなくて済むように)。"""
    plain_estimator = epp.PalmPoseEstimator.__new__(epp.PalmPoseEstimator)
    cache = []
    for name, joint_positions, label in samples:
        joints = {n: np.asarray(p, dtype=np.float64)
                 for n, p in joint_positions.items()}
        palms = {side: plain_estimator._estimate_one(joints, side)
                for side in SIDES}
        cache.append((name, joints, palms, label))
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
    """(true_side, side_scores) のリストを返す。

    ``side_scores`` は ``{'R': float or None, 'L': float or None}``
    (veto された側は ``None``)。``score_min`` を含まない生スコアなので、
    ここから任意の閾値を後付けで評価できる。
    """
    kwargs = _selector_kwargs(params)
    selector = epp.OfferedHandSelector(**kwargs)
    out = []
    for name, joints, palms, label in cache:
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
    """このパラメータ候補でのスコア分布に対し、正解率を最大化する
    ``score_min`` を全探索する (候補点はサンプルのスコア値そのもの)。

    Returns
    -------
    (best_threshold, best_accuracy, confusion) : 最良の閾値と、その精度、
        混同行列 (dict of dict, 行=正解, 列=予測)。
    """
    all_scores = sorted(set(
        s for _, side_scores in scored_samples for s in side_scores.values()
        if s is not None))
    # 全ての閾値候補 (各スコア値のすぐ下 = そのスコア以上を採用する境界、
    # および「何も通さない」既定より高い値も 1 つ試す)。
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
        default=os.path.join(_THIS_DIR, 'random_human_poses'))
    parser.add_argument(
        '--palm-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_palm_poses'))
    parser.add_argument('--label-key', type=str, default='human_label')
    parser.add_argument(
        '--n-random', type=int, default=200,
        help='グリッド全探索が多すぎる場合にランダムサンプルする候補数 '
            '(既定 200、0 でグリッド全探索)。')
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
    default_params = dict(axis_blend=0.0, finger_ramp_upper=0.90,
                          approach_height_scale=1.0, separation_weight=0.10)
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
