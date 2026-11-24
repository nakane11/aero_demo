#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""複数人の見張りと切り出し (``multi_person_watcher``) を、今の方式 (Holistic を
画像全体に掛ける) と同じフレームで比べる。差し出しの確定 (``PalmPoseEstimator``
の ``offered_hand`` が決まる) までを見る。ROS には依存しない。

フレームの入力は ``multi_person_sources`` (rosbag・CMU Panoptic・合成) から。

出力 (``--output-dir``)
  ``frames.jsonl``: フレームごとの内訳 (両方式の確定・スコア・見ている人など)
  ``events.json``: 確定した時刻 (方式ごと) と処理時間の集計
  ``overlay.mp4``: 見張りの bbox・番号・候補スコア・切り出し範囲と骨格 (--video)

Usage
-----
    python3 tools/eval_multi_person_offer.py --source bag \
        --bag data/bags/xxx.bag --start 10 --end 40 \
        --model ~/work/multiperson_data/models/pose_landmarker_full.task \
        --output-dir /tmp/mp_eval --video
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (os.path.join(_REPO_ROOT, 'src'), os.path.join(_REPO_ROOT, 'scripts'),
           _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo import skeleton_filters  # noqa: E402
from aero_demo.multi_person_watcher import (  # noqa: E402
    MultiPersonWatcher, TargetChooser, joints_box, upper_body_height)
from aero_demo.people_pose_estimator import PeoplePoseEstimator  # noqa: E402

import estimate_palm_poses as epp  # noqa: E402
import multi_person_sources as sources  # noqa: E402

# run_camera_pipeline_test.py の既定と同じ。
OFFER_FAST_STILLNESS_WINDOW = 0.3
OFFER_FAST_STILLNESS_SCORE = 0.8


class OfferPipeline(object):
    """ノードの骨格→平滑化→掌→差し出し判定 (1 人分) を真似る."""

    def __init__(self, args):
        self.args = args
        self.estimator = PeoplePoseEstimator(use_hand=True)
        self._new_state()

    def _new_state(self):
        args = self.args
        self.selector = epp.OfferedHandSelector(
            robot_position=None, score_min=args.offer_score_min,
            max_distance=args.max_person_distance,
            fast_stillness_window=OFFER_FAST_STILLNESS_WINDOW,
            fast_stillness_score=OFFER_FAST_STILLNESS_SCORE)
        self.palm_estimator = epp.PalmPoseEstimator(self.selector)
        self.smoother = skeleton_filters.OneEuroFilter(
            mincutoff=0.5, beta=0.3, dcutoff=1.0)

    def reset(self):
        """見る人を変えたとき: Holistic の追跡と時間の状態を捨てる."""
        self.estimator.reset_tracking()
        self._new_state()

    def close(self):
        self.estimator.close()

    def step(self, frame, mask=None, person_filter=None):
        """``person_filter(points)`` が False の骨格は人が見えないのと同じに扱う."""
        t0 = time.perf_counter()
        people, joints_2d = self.estimator.estimate_3d(
            frame['color'], frame['depth'], frame['intrinsics'],
            output_transform=frame['camera_to_base'], mask=mask)
        holistic_time = time.perf_counter() - t0
        out = dict(holistic_time=holistic_time, joints_2d=None,
                   offered_hand=None, scores=None, neck_px=None, raw=None,
                   body_height=None)
        if not people:
            return out
        out.update(raw=people[0], joints_2d=joints_2d[0],
                   body_height=upper_body_height(people[0]))
        if person_filter is not None and not person_filter(people[0]):
            out.update(used=False)
            return out
        out.update(used=True)
        joints = self.smoother.update(people[0], t=frame['t'],
                                      frame_key=True)
        self.selector.robot_position = np.asarray(frame['robot_position'])
        palms = self.palm_estimator.estimate(joints, t=frame['t'])
        # ノード (_try_handshake) と同じく同じ時刻でもう一度 select する。
        selection = self.selector.select(joints, palms, t=frame['t'])
        out.update(joints_2d=joints_2d[0], offered_hand=palms['offered_hand'],
                   scores=selection.get('scores'),
                   neck_px=_neck_px(joints_2d[0]),
                   step_time=time.perf_counter() - t0)
        return out


def _neck_px(joints_2d):
    for joint in joints_2d:
        if joint['limb'] in ('Neck', 'Nose') and joint['score'] >= 0:
            return [joint['x'], joint['y']]
    return None


def _owner(px, tracks):
    """画像上の点 ``px`` を含む (無ければ中心が最も近い) 見張りの人の番号."""
    if px is None or not tracks:
        return None
    best = None
    for tr in tracks:
        x0, y0, x1, y1 = tr['box']
        inside = x0 <= px[0] <= x1 and y0 <= px[1] <= y1
        d = np.hypot((x0 + x1) / 2 - px[0], (y0 + y1) / 2 - px[1])
        key = (0 if inside else 1, d)
        if best is None or key < best[0]:
            best = (key, tr['track_id'])
    return best[1]


def _draw(frame, multi, base, tracks, view):
    img = PeoplePoseEstimator._masked(frame['color'], view)
    if img is frame['color']:
        img = img.copy()
    else:
        # 塗った所は半分だけ暗くして中が見えるようにする。
        img = cv2.addWeighted(frame['color'], 0.35, img, 0.65, 0)
    if multi['joints_2d'] is not None:
        img = skeleton_drawing.draw_skeleton_overlay(
            img, multi['joints_2d'], offered_side=multi['offered_hand'])
    for tr in tracks:
        x0, y0, x1, y1 = [int(v) for v in tr['box']]
        color = (0, 0, 255) if tr['target'] else (0, 200, 0)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.putText(img, 'id{} {:.2f}'.format(
            tr['track_id'], tr['score_ema'] or 0.0), (x0, max(20, y0 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if view is not None and view.get('keep') is not None:
        x0, y0, x1, y1 = [int(v) for v in view['keep']]
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 0, 255), 1)
    if base['neck_px'] is not None:
        cv2.circle(img, (int(base['neck_px'][0]), int(base['neck_px'][1])),
                   10, (255, 255, 0), 3)
    text = 't={:.2f} multi:{} base:{}'.format(
        frame['t'], multi['offered_hand'], base['offered_hand'])
    cv2.putText(img, text, (10, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(img, text, (10, img.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    return img


def run(args, frames):
    os.makedirs(args.output_dir, exist_ok=True)
    watcher = MultiPersonWatcher(
        args.model, num_poses=args.num_poses,
        chooser=TargetChooser(switch_score=args.switch_score,
                              switch_margin=args.switch_margin,
                              switch_hold=args.switch_hold))
    multi = OfferPipeline(args)
    multi.estimator.keep_spare()
    base = OfferPipeline(args)
    writer = None
    # 見張りは別スレッド 1 本で回す想定: フレーム時刻 t に始めた検出は、測った
    # 処理時間 dt の後 (t + dt 以降のフレーム) から使える。処理中に来たフレームは
    # 見張りには渡さない (最新のフレームだけ渡す)。
    pending = None  # (使える時刻, 画像の時刻, people)
    busy_until = -np.inf
    events = dict(multi=[], base=[])
    last = dict(multi=None, base=None)
    timings = dict(watch=[], multi_holistic=[], base_holistic=[])
    n = 0
    t_first = None
    with open(os.path.join(args.output_dir, 'frames.jsonl'), 'w') as log:
        for frame in frames:
            t = frame['t']
            if t_first is None:
                t_first = t
            h, w = frame['color'].shape[:2]
            roi, switched = watcher.view, False
            if pending is not None and pending[0] <= t:
                roi, switched = watcher.apply(pending[2], pending[1], w, h)
                pending = None
            if args.sync or busy_until <= t:
                t0 = time.perf_counter()
                people = watcher.detect(
                    frame['color'], frame['depth'], frame['intrinsics'],
                    frame['camera_to_base'], frame.get('base_to_world'),
                    frame['robot_position'])
                dt = time.perf_counter() - t0
                timings['watch'].append(dt)
                if args.sync:
                    roi, s2 = watcher.apply(people, t, w, h)
                    switched = switched or s2
                else:
                    pending = (t + dt, t, people)
                    busy_until = t + dt
            if switched:
                multi.reset()
            tracks = watcher.summary()
            m = multi.step(frame, mask=roi,
                           person_filter=watcher.looks_like_person)
            b = base.step(frame)
            # ノードと同じく、Holistic が人でないものを見続けたら作り直す。
            refocused = watcher.check_focus(
                m['raw'], joints_box(m['joints_2d']), t, w, h)
            if refocused:
                multi.reset()
            timings['multi_holistic'].append(m['holistic_time'])
            timings['base_holistic'].append(b['holistic_time'])
            record = dict(
                t=t - t_first, stamp=t, view=roi, switched=switched,
                refocused=refocused, target=watcher.target_id, tracks=tracks,
                multi=dict(offered=m['offered_hand'], scores=m['scores'],
                           owner=_owner(m['neck_px'], tracks),
                           body_height=m['body_height'],
                           box=joints_box(m['joints_2d']),
                           used=m.get('used')),
                base=dict(offered=b['offered_hand'], scores=b['scores'],
                          owner=_owner(b['neck_px'], tracks),
                          body_height=b['body_height'],
                          box=joints_box(b['joints_2d'])),
                gt=frame.get('gt'))
            for key, res in (('multi', record['multi']),
                             ('base', record['base'])):
                if res['offered'] is not None and last[key] is None:
                    events[key].append(dict(t=record['t'], side=res['offered'],
                                            owner=res['owner'],
                                            gt=frame.get('gt')))
                last[key] = res['offered']
            log.write(json.dumps(record, default=_json_default) + '\n')
            if args.video:
                img = _draw(frame, m, b, tracks, roi)
                img = cv2.resize(img, None, fx=args.video_scale,
                                 fy=args.video_scale)
                if writer is None:
                    writer = cv2.VideoWriter(
                        os.path.join(args.output_dir, 'overlay.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'), args.video_fps,
                        (img.shape[1], img.shape[0]))
                writer.write(img)
            n += 1
    if writer is not None:
        writer.release()
    summary = dict(
        n_frames=n, events=events,
        timing_ms={k: dict(mean=1e3 * float(np.mean(v)),
                           p95=1e3 * float(np.percentile(v, 95)))
                   for k, v in timings.items() if v})
    with open(os.path.join(args.output_dir, 'events.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=_json_default)
    print(json.dumps(summary, indent=1, default=_json_default))
    multi.close()
    base.close()
    watcher.close()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sources.add_arguments(parser)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--num-poses', type=int, default=2,
                        help='PoseLandmarker の 1 回あたりの人数 (塗って探し直す)。')
    parser.add_argument('--sync', action='store_true',
                        help='見張りの結果を同じフレームで使う (遅れなしの比較用)。')
    parser.add_argument('--switch-score', type=float, default=0.5)
    parser.add_argument('--switch-margin', type=float, default=0.25)
    parser.add_argument('--switch-hold', type=float, default=0.1)
    parser.add_argument('--offer-score-min', type=float, default=0.65)
    parser.add_argument('--max-person-distance', type=float, default=3.0)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video-scale', type=float, default=0.5)
    parser.add_argument('--video-fps', type=float, default=15.0)
    args = parser.parse_args()
    run(args, sources.open_source(args))


if __name__ == '__main__':
    main()
