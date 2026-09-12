#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""rosbag (color/depth/camera_info/tf/tf_static を含む) を読み込み、
``offered_hand`` の自動判定結果に関わらず骨格・掌位置姿勢・骨格重畳画像を
保存するオフライン抽出ツール。

``record_palm_offer_clips.py`` は現行の ``OfferedHandSelector`` (判定器
そのもの) をトリガーにクリップを切り出すため、それで撮ったデータだけを
使うと「今の判定基準では見逃されている、より自然な差し出し方 (斜め前に
軽く、体から離す等)」のサンプルが原理的に集まらない。このツールは判定器
に一切依存しない生の bag (``rosbag record`` で判定なしに連続録画したもの
を想定) を入力にし、既定では ``--sample-interval`` 秒おきに全フレームを
機械的にサンプリングして保存する (判定結果はあくまで参考情報として掌
JSON に残すだけで、どのフレームを保存するかには使わない)。

``record_palm_offer_clips.py`` が切り出した 4 秒クリップ (判定器で既に
選別済みのデータ) を混ぜたい場合は ``--single-sample`` を付ける。この
場合は bag ごとに 1 フレームだけを代表として選ぶ -- 対応する
``<bag>.json`` があれば、その ``trigger_stamp`` (差し出しを検出した瞬間
の時刻) に最も近いフレーム、無ければ同期できたフレームの中央を使う (前後
の過渡的なフレームまで保存すると、同じ差し出し動作からほぼ重複したサン
プルが大量にできて後段の人手ラベル付けの手間が増えるだけなため)。

保存先には 3 つのサブディレクトリができる:

    <output-dir>/skeletons/<bag名>[_<連番>].json  (estimate_palm_poses.py
        の --input-dir にそのまま渡せる骨格 JSON)
    <output-dir>/palms/<bag名>[_<連番>].json      (掌位置姿勢 +
        offered_hand の自動判定。label_offer_images.py がここに
        human_label を書く)
    <output-dir>/images/<bag名>[_<連番>].png       (骨格重畳画像。自動
        判定で選ばれた手を赤で描く。label_offer_images.py の入力)

(``--single-sample`` のときだけファイル名に連番が付かない。)

roscore は不要 (``tf2_ros.BufferCore`` を使う、``record_palm_offer_
clips.py``/``run_camera_pipeline_test.py`` のようにライブ購読はしない)。

Usage
-----
    # 判定器なしで連続録画した bag から、2.5 秒おきにサンプリング (既定)
    python3 scripts/ros/extract_skeletons_from_bag.py \
        --bag session1.bag --output-dir /tmp/offer_dataset

    # record_palm_offer_clips.py が切り出した bag から 1 本 1 サンプル
    python3 scripts/ros/extract_skeletons_from_bag.py \
        --bag palm_offer_clips/*.bag --output-dir /tmp/offer_dataset \
        --single-sample
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

import genpy
import rosbag
import tf2_ros

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_DIR)
_PKG_SRC_DIR = os.path.join(_SCRIPTS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import json_io  # noqa: E402
from aero_demo import skeleton_filters  # noqa: E402
from aero_demo.people_pose_estimator import (  # noqa: E402
    CameraIntrinsics, PeoplePoseEstimator)
from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, transform_to_matrix)

import estimate_palm_poses as epp  # noqa: E402
# record_palm_offer_clips.py と同じ骨格重畳描画・フォールバック値を再利用
# する (見た目・判定基準を実カメラパイプラインと揃えるため、この抽出専用
# ファイルで再実装しない)。
from record_palm_offer_clips import (  # noqa: E402
    _FALLBACK_ROBOT_HAND_POSITION, draw_skeleton_overlay)


def load_tf_buffer(bag_path):
    """bag 内の全 ``/tf``/``/tf_static`` を読み込んだ ``BufferCore`` を作る.

    bag 全体をカバーできるよう、キャッシュ時間は bag の長さより十分長く
    取る (``genpy.Duration(3600)``, 1 時間)。roscore 不要。
    """
    buf = tf2_ros.BufferCore(genpy.Duration(3600))
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _t in bag.read_messages(topics=['/tf', '/tf_static']):
            for tr in msg.transforms:
                try:
                    if topic == '/tf_static':
                        buf.set_transform_static(tr, 'bag')
                    else:
                        buf.set_transform(tr, 'bag')
                except Exception:
                    pass
    return buf


def collect_synced_frames(bag_path, color_topic, depth_topic, info_topic,
                          slop):
    """color を基準に、depth/camera_info を最も近い時刻でマッチさせる
    (``message_filters.ApproximateTimeSynchronizer`` のオフライン再現)。

    Returns
    -------
    list of (color_msg, depth_msg, info_msg), 時刻昇順。
    """
    color_msgs, depth_msgs, info_msgs = [], [], []
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _t in bag.read_messages(
                topics=[color_topic, depth_topic, info_topic]):
            if topic == color_topic:
                color_msgs.append(msg)
            elif topic == depth_topic:
                depth_msgs.append(msg)
            elif topic == info_topic:
                info_msgs.append(msg)

    def _closest(msgs, stamp, used):
        best_idx, best_dt = None, slop
        for i, m in enumerate(msgs):
            if i in used:
                continue
            dt = abs(m.header.stamp.to_sec() - stamp)
            if dt <= best_dt:
                best_idx, best_dt = i, dt
        return best_idx

    used_depth, used_info = set(), set()
    frames = []
    for color_msg in color_msgs:
        stamp = color_msg.header.stamp.to_sec()
        di = _closest(depth_msgs, stamp, used_depth)
        ii = _closest(info_msgs, stamp, used_info)
        if di is None or ii is None:
            continue
        used_depth.add(di)
        used_info.add(ii)
        frames.append((color_msg, depth_msgs[di], info_msgs[ii]))
    frames.sort(key=lambda f: f[0].header.stamp.to_sec())
    return frames


def _lookup_robot_position(buf, base_frame, hand_frame, stamp,
                           fixed_position):
    """``record_palm_offer_clips.py`` の ``_resolve_robot_position`` と
    同じ優先順位 (固定値 > TF > フォールバック) をオフラインで再現する。"""
    if fixed_position is not None:
        return np.asarray(fixed_position, dtype=np.float64)
    try:
        transform = buf.lookup_transform_core(base_frame, hand_frame, stamp)
    except Exception:
        return np.asarray(_FALLBACK_ROBOT_HAND_POSITION, dtype=np.float64)
    t = transform.transform.translation
    return np.array([t.x, t.y, t.z], dtype=np.float64)


def _load_trigger_stamp(bag_path):
    """``record_palm_offer_clips.py`` が書き出したメタデータ
    (``<bag_stem>.json``) から ``trigger_stamp`` (差し出しを検出した
    瞬間の時刻) を読む。メタデータが無い/``trigger_stamp`` が無ければ
    ``None``。"""
    meta_path = os.path.splitext(bag_path)[0] + '.json'
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    return meta.get('trigger_stamp')


def _pick_representative_index(frames, trigger_stamp):
    """1 bag = 1 サンプルとして保存する代表フレームの index を選ぶ.

    ``trigger_stamp`` (差し出しを検出した瞬間の時刻) が分かれば、それに
    最も時刻が近い color フレームを選ぶ (その bag の中で最も代表的な
    瞬間なので)。メタデータが無ければ同期できたフレームの中央で代用する。
    """
    if not frames:
        return None
    if trigger_stamp is None:
        return len(frames) // 2
    return min(range(len(frames)),
              key=lambda i: abs(
                  frames[i][0].header.stamp.to_sec() - trigger_stamp))


def _save_frame(joint_positions, person_joints_2d, color, palm_estimator,
                buf, args, stamp, out_dirs, name):
    """1 フレーム分の掌推定・骨格重畳画像を計算して保存する共通処理."""
    palm_estimator.offered_hand_selector.robot_position = \
        _lookup_robot_position(
            buf, args.base_frame, args.robot_hand_frame, stamp,
            args.robot_hand_position)
    palms = palm_estimator.estimate(joint_positions)

    json_io.save_json(
        os.path.join(out_dirs['skeletons'], name + '.json'),
        dict(skeleton=dict(
            joint_positions={k: list(v)
                             for k, v in joint_positions.items()},
            height=0.0)))
    json_io.save_json(os.path.join(out_dirs['palms'], name + '.json'), palms)
    overlay = draw_skeleton_overlay(
        color, person_joints_2d, offered_side=palms['offered_hand'])
    cv2.imwrite(os.path.join(out_dirs['images'], name + '.png'), overlay)


def process_bag(bag_path, args, pose_estimator, joint_smoother,
                palm_estimator, out_dirs):
    """``bag_path`` の全フレームを走査し、保存対象のフレームを選んで保存
    する。

    ``args.single_sample`` が真なら、対応する ``<bag>.json`` の
    ``trigger_stamp`` (無ければ同期フレームの中央) に最も近いフレーム 1 つ
    だけを ``<bag名>.json``/``.png`` として保存する
    (``record_palm_offer_clips.py`` が切り出した判定済みクリップ向け)。

    既定 (偽) では、判定器に一切依存せず ``args.sample_interval`` 秒おきに
    機械的にサンプリングして ``<bag名>_<連番>.json``/``.png`` として複数
    保存する (``rosbag record`` で判定なしに連続録画した bag 向け)。

    Returns
    -------
    int
        保存できたフレーム数。
    """
    buf = load_tf_buffer(bag_path)
    frames = collect_synced_frames(
        bag_path, args.color_topic, args.depth_topic,
        args.camera_info_topic, args.sync_slop)
    print('{}: {} フレームを同期できました。'.format(bag_path, len(frames)))
    if not frames:
        print('  -> 同期できたフレームが無いためスキップします。')
        return 0

    bag_name = os.path.splitext(os.path.basename(bag_path))[0]
    if args.single_sample:
        trigger_stamp = _load_trigger_stamp(bag_path)
        target_indices = {_pick_representative_index(frames, trigger_stamp)}
    else:
        target_indices = None  # サンプリング間隔で都度判定する

    n_saved = 0
    n_no_tf = 0
    n_no_person = 0
    last_saved_stamp = None
    for idx, (color_msg, depth_msg, info_msg) in enumerate(frames):
        stamp_sec = color_msg.header.stamp.to_sec()
        try:
            transform = buf.lookup_transform_core(
                args.base_frame, color_msg.header.frame_id,
                color_msg.header.stamp)
        except Exception:
            camera_to_base = None
            n_no_tf += 1
        else:
            camera_to_base = transform_to_matrix(transform.transform)

        joint_positions = None
        person_joints_2d = None
        color = None
        if camera_to_base is not None:
            color = imgmsg_to_ndarray(color_msg, desired_encoding='bgr8')
            depth_raw = imgmsg_to_ndarray(depth_msg)
            depth_m = PeoplePoseEstimator.depth_to_meters(
                depth_raw, encoding=depth_msg.encoding)
            intrinsics = CameraIntrinsics.from_matrix(info_msg.K)
            people, joints_2d = pose_estimator.estimate_3d(
                color, depth_m, intrinsics, output_transform=camera_to_base)
            joint_positions = people[0] if people else None
            person_joints_2d = joints_2d[0] if joints_2d else None
            if joint_positions is not None:
                # OneEuroFilter は時系列順に通し続けないと平滑化の意味が
                # 無い (速度推定が飛ぶ) ため、保存対象でないフレームでも
                # 検出できていればここまでは必ず行う。
                joint_positions = joint_smoother.update(
                    joint_positions, t=stamp_sec)

        if target_indices is not None:
            should_save = idx in target_indices
        else:
            should_save = (last_saved_stamp is None
                          or stamp_sec - last_saved_stamp
                             >= args.sample_interval)
        if not should_save:
            continue
        if joint_positions is None:
            n_no_person += 1
            continue

        name = (bag_name if args.single_sample
                else '{}_{:05d}'.format(bag_name, idx))
        _save_frame(joint_positions, person_joints_2d, color, palm_estimator,
                   buf, args, color_msg.header.stamp, out_dirs, name)
        n_saved += 1
        last_saved_stamp = stamp_sec

    print('  -> 保存 {} 件 (TF 未解決で除外 {} 件, 人物未検出で除外 {} 件)'
         .format(n_saved, n_no_tf, n_no_person))
    return n_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--bag', type=str, nargs='+', required=True,
        help='入力 bag ファイル (複数可、シェルの glob もそのまま渡せる)。')
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='skeletons/palms/images の 3 つのサブディレクトリを作る '
            '保存先。')
    parser.add_argument('--color-topic', type=str,
                        default='/camera/color/image_raw/decompressed')
    parser.add_argument('--depth-topic', type=str,
                        default='/camera/depth/image_raw/decompressed')
    parser.add_argument('--camera-info-topic', type=str,
                        default='/camera/color/camera_info')
    parser.add_argument('--base-frame', type=str, default='base_link')
    parser.add_argument(
        '--sync-slop', type=float, default=0.1,
        help='color/depth/camera_info を同期させる際の最大時刻差 [秒] '
            '(既定 0.1、message_filters.ApproximateTimeSynchronizer の '
            'slop と同じ意味)。')
    parser.add_argument(
        '--sample-interval', type=float, default=2.5,
        help='(--single-sample を付けないとき) 何秒おきにサンプリングして '
            '保存するか (既定 2.5 秒)。判定器の結果には一切関係なく、'
            '機械的にこの間隔でサンプリングする。')
    parser.add_argument(
        '--single-sample', action='store_true',
        help='bag ごとに 1 フレームだけを代表として保存する '
            '(record_palm_offer_clips.py が切り出した判定済みクリップ '
            '向け、モジュール docstring 参照)。既定 (指定なし) は '
            '--sample-interval 秒おきに複数フレーム保存する '
            '(判定器なしで連続録画した bag 向け)。')
    # --- PeoplePoseEstimator (record_palm_offer_clips.py と同じ既定値) ---
    parser.add_argument('--min-detection-confidence', type=float, default=0.5)
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5)
    parser.add_argument('--min-visibility', type=float, default=0.5)
    parser.add_argument('--min-joints', type=int, default=6)
    parser.add_argument('--max-z-diff', type=float, default=1.0)
    parser.add_argument('--min-body-size', type=float, default=0.3)
    parser.add_argument('--max-body-size', type=float, default=2.5)
    parser.add_argument('--max-limb-length', type=float, default=0.7)
    parser.add_argument('--max-hand-segment-length', type=float, default=0.12)
    parser.add_argument('--max-hand-reach', type=float, default=0.22)
    parser.add_argument('--depth-patch-size', type=int, default=3)
    # --- 関節位置の時間方向平滑化 (record_palm_offer_clips.py と同じ) ---
    parser.add_argument('--joint-smoothing-mincutoff', type=float, default=0.5)
    parser.add_argument('--joint-smoothing-beta', type=float, default=0.3)
    parser.add_argument('--joint-smoothing-dcutoff', type=float, default=1.0)
    # --- 差し出し手判定 (estimate_palm_poses.OfferedHandSelector) ---
    parser.add_argument('--offer-score-min', type=float, default=0.65)
    parser.add_argument(
        '--robot-hand-position', type=float, nargs=3, default=None,
        metavar=('X', 'Y', 'Z'))
    parser.add_argument('--robot-hand-frame', type=str,
                        default='r_eef_grasp_link')
    parser.add_argument('--max-person-distance', type=float, default=4.2)
    args = parser.parse_args()

    bag_paths = []
    for pattern in args.bag:
        matched = sorted(glob.glob(pattern))
        bag_paths.extend(matched if matched else [pattern])
    if not bag_paths:
        print('--bag に一致する bag ファイルが見つかりません。')
        return

    out_dirs = {name: os.path.join(args.output_dir, name)
               for name in ('skeletons', 'palms', 'images')}
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    pose_estimator = PeoplePoseEstimator(
        use_hand=True,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        min_visibility=args.min_visibility,
        min_joints=args.min_joints,
        max_z_diff=args.max_z_diff,
        min_body_size=args.min_body_size,
        max_body_size=args.max_body_size,
        max_limb_length=args.max_limb_length,
        max_hand_segment_length=args.max_hand_segment_length,
        max_hand_reach=args.max_hand_reach,
        depth_patch_size=args.depth_patch_size)

    max_distance = (None if args.max_person_distance <= 0
                   else args.max_person_distance)
    offered_hand_selector = epp.OfferedHandSelector(
        robot_position=None, score_min=args.offer_score_min,
        max_distance=max_distance)
    palm_estimator = epp.PalmPoseEstimator(offered_hand_selector)

    total_saved = 0
    try:
        for bag_path in bag_paths:
            # bag ごとに OneEuroFilter は独立させる (別クリップの時系列を
            # 混ぜて平滑化しないため)。
            joint_smoother = skeleton_filters.OneEuroFilter(
                mincutoff=args.joint_smoothing_mincutoff,
                beta=args.joint_smoothing_beta,
                dcutoff=args.joint_smoothing_dcutoff)
            total_saved += process_bag(
                bag_path, args, pose_estimator, joint_smoother,
                palm_estimator, out_dirs)
    finally:
        pose_estimator.close()

    print('\n合計 {} フレームを {} に保存しました。'.format(
        total_saved, args.output_dir))
    print('次は label_offer_images.py で images/ を見ながら palms/ に '
         'human_label を付けてください。')


if __name__ == '__main__':
    main()
