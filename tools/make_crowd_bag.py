#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""実機の rosbag に、ほかの bag から切り抜いた人を重ねて人だかりを作る.

人ごとに素材の bag の区間から人を深度で切り抜き (PoseLandmarker の bbox +
深度)、3 次元の点 (色つき) にして、ベースの bag の部屋の指定の位置に立たせる。
ベースの各画像は、その時刻のカメラの位置 (TF の odom -> カメラ) から点を
投影して描き、深度は手前が勝つように重ねる。ロボットが動いても、人は部屋に
立ったまま見える。

出力:

- ``--out-bag``: ベースの区間の全トピックをそのまま写し、カラーと深度の画像
  だけを合成後に差し替えた bag (ノードの rosbag 再生に使える)。
- ``--out-video``: 確認用の mp4 (左: カラー、右: 深度)。

人の設定 (``--people``、JSON の配列) の各要素::

    {"name": "far_blue", "bag": ".../camera_pipeline_...bag",
     "start": 8.0, "end": 20.0,       # 素材の bag の先頭からの秒
     "place": [3.5, 0.3],             # --ref-time のベースの base_link 系 [m]
     "yaw_deg": 0.0,                  # 向きの追加の回転 (0: 素材でカメラを
                                      # 向いていた面がベースのカメラを向く)
     "time_offset": 0.0}              # 素材の区間のどこから始めるか [s]

素材の区間は往復で繰り返す (端で飛ばないように)。

制限: 人の見えている面 (素材のカメラから見た面) しか無いので、向きは素材の
ときのカメラへの向きに合わせる。ベースの深度が ``--backdrop-depth`` より遠い
画素 (壁など) は、重ねる人の方が奥でも上書きする (狭い部屋でも奥の人を
置くため。深度画像では壁に穴が開き、その奥に人が立つように見える)。

使い方 (ROS の python3。mediapipe が要る)::

    source /opt/ros/noetic/setup.bash
    python3 tools/make_crowd_bag.py --base BAG --start 0 --end 32 \\
        --people people.json --model ~/.mediapipe/models/pose_landmarker_lite.task \\
        --out-bag out.bag --out-video out.mp4
"""

import argparse
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'src'))
sys.path.insert(0, _HERE)

import multi_person_sources as mps  # noqa: E402
from aero_demo.multi_person_watcher import TORSO_JOINTS  # noqa: E402
from aero_demo.multi_person_watcher import lift_to_3d  # noqa: E402

COLOR_TOPIC = '/camera/color/image_raw/decompressed'
DEPTH_TOPIC = '/camera/depth/image_raw/decompressed'
INFO_TOPIC = '/camera/color/camera_info'


class _FrameArgs(object):
    color_topic = COLOR_TOPIC
    depth_topic = DEPTH_TOPIC
    camera_info_topic = INFO_TOPIC

    def __init__(self, frame_step):
        self.frame_step = frame_step


class Segmenter(object):
    """PoseLandmarker (人の領域つき) で、いちばん大きい人の関節と領域を取る."""

    def __init__(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        self.mp = mp
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE, num_poses=1,
            output_segmentation_masks=True)
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect(self, bgr_small):
        from aero_demo.multi_person_watcher import POSE_INDEX
        rgb = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(rgb))
        result = self.landmarker.detect(image)
        if not result.pose_landmarks or not result.segmentation_masks:
            return None
        h, w = bgr_small.shape[:2]
        lms = result.pose_landmarks[0]
        pixels = {name: [lms[i].x * w, lms[i].y * h, float(lms[i].visibility)]
                  for name, i in POSE_INDEX.items()}
        seg = result.segmentation_masks[0].numpy_view()
        return pixels, np.asarray(seg, dtype=np.float32).reshape(h, w)


def extract_person(spec, segmenter, args):
    """素材の区間から人の点群 (odom 系) を時刻ごとに取り出す.

    人の領域は PoseLandmarker の segmentation で取る。深度は黒い服・髪で
    抜けるので、領域の中で深度のある画素は 3 次元の点に、無い画素は体幹の
    深度に置いて ``valid=False`` (合成後の深度も 0 = 抜け) にする。
    """
    frames = mps.bag_frames(spec['bag'], spec['start'], spec['end'],
                            _FrameArgs(args.source_frame_step))
    out = []
    t0 = None
    n_miss = 0
    n_high = 0
    for fr in frames:
        if t0 is None:
            t0 = fr['t']
        color, depth = fr['color'], fr['depth']
        h, w = depth.shape
        found = segmenter.detect(cv2.resize(color, (w // 2, h // 2)))
        if found is None:
            n_miss += 1
            continue
        pixels_small, seg_small = found
        pixels = {k: [2.0 * u, 2.0 * v, vis]
                  for k, (u, v, vis) in pixels_small.items()}
        # 手を出していない人にしたいので、見張りの差し出しスコア (向きは
        # 最悪の場合) が高いフレームは使わない。
        from screen_crowd_sources import joints_3d, worst_case_score
        joints, _ = joints_3d(pixels, depth, fr['intrinsics'],
                              fr['camera_to_base'])
        score = 0.0 if joints is None else worst_case_score(joints)[0]
        if args.max_source_score > 0 and score >= args.max_source_score:
            n_high += 1
            continue
        seg = cv2.resize(seg_small, (w, h)) > args.seg_threshold
        torso_cam = lift_to_3d(pixels, depth, fr['intrinsics'])
        torso = [torso_cam[n] for n in TORSO_JOINTS if n in torso_cam]
        if not torso or not seg.any():
            n_miss += 1
            continue
        world_from_base = (np.eye(4) if fr['base_to_world'] is None
                           else fr['base_to_world'])
        base_from_cam = fr['camera_to_base']
        world_from_cam = world_from_base.dot(base_from_cam)
        torso_cam = np.mean(torso, axis=0)
        body_z = float(torso_cam[2])
        torso_base = base_from_cam.dot(np.append(torso_cam, 1.0))[:3]
        ys, xs = np.nonzero(seg[::args.stride, ::args.stride])
        ys = ys * args.stride
        xs = xs * args.stride
        z = depth[ys, xs].astype(np.float64)
        valid = (np.isfinite(z) & (z > 0.2)
                 & (np.abs(z - body_z) < args.depth_margin))
        z = np.where(valid, z, body_z)
        K = fr['intrinsics']
        pts_cam = np.stack([(xs - K.cx) * z / K.fx, (ys - K.cy) * z / K.fy,
                            z, np.ones_like(z)], axis=0)
        pts_base = base_from_cam.dot(pts_cam)[:3]
        # 深度のある点のうち、床 (足元) と体から横に離れた物を落とす
        # (領域の縁にかかった背景)。
        keep = ~valid | (
            (pts_base[2] > args.floor_cut)
            & (np.hypot(pts_base[0] - torso_base[0],
                        pts_base[1] - torso_base[1]) < args.body_radius))
        if keep.sum() < args.min_points:
            n_miss += 1
            continue
        pts_world = world_from_cam.dot(pts_cam[:, keep])[:3]
        torso_world = world_from_base.dot(np.append(torso_base, 1.0))[:3]
        out.append(dict(
            t=fr['t'] - t0,
            points=pts_world.T.astype(np.float32),
            colors=color[ys[keep], xs[keep]].astype(np.uint8),
            valid=valid[keep],
            torso=torso_world.astype(np.float64),
            camera=world_from_cam[:3, 3].astype(np.float64),
            depth=body_z, score=score))
    print('[extract] {}: {} フレーム (人なし {}、スコア {:.2f} 以上で除外 {})、'
          '区間 {:.1f} 秒、スコア最大 {:.2f}'.format(
              spec['name'], len(out), n_miss, args.max_source_score, n_high,
              out[-1]['t'] if out else 0.0,
              max([f['score'] for f in out] or [0.0])))
    return out


def describe(name, frames):
    """素材の人の動き (最初の体幹からの、カメラへ向かう向き・横の移動) を出す."""
    if not frames:
        return
    anchor = frames[0]['torso'][:2]
    to_cam = frames[0]['camera'][:2] - anchor
    to_cam = to_cam / max(1e-6, np.linalg.norm(to_cam))
    side = np.array([-to_cam[1], to_cam[0]])
    toward = [float((f['torso'][:2] - anchor).dot(to_cam)) for f in frames]
    lateral = [float((f['torso'][:2] - anchor).dot(side)) for f in frames]
    print('[source] {}: カメラまで {:.2f} m、カメラへ {:+.2f}〜{:+.2f} m、'
          '横 {:+.2f}〜{:+.2f} m、点 {}〜{} 個'.format(
              name, float(np.linalg.norm(frames[0]['camera'][:2] - anchor)),
              min(toward), max(toward), min(lateral), max(lateral),
              min(len(f['points']) for f in frames),
              max(len(f['points']) for f in frames)))


def load_or_extract(spec, segmenter, args):
    key = ('{}_{:.2f}_{:.2f}_s{}_m{:.2f}_f{:.2f}_r{:.2f}_g{:.2f}_o{:.2f}'
           '_v4.pkl').format(
        os.path.basename(spec['bag']).replace('.bag', ''), spec['start'],
        spec['end'], args.stride, args.depth_margin, args.floor_cut,
        args.body_radius, args.seg_threshold, args.max_source_score)
    path = os.path.join(args.cache_dir, key)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    frames = extract_person(spec, segmenter, args)
    os.makedirs(args.cache_dir, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
    return frames


def _yaw(v):
    return float(np.arctan2(v[1], v[0]))


class PlacedPerson(object):
    """素材の人を、ベースの部屋 (odom 系) の指定の位置に立たせる."""

    def __init__(self, spec, frames, place_world, ref_camera_world):
        self.name = spec['name']
        self.frames = frames
        self.times = np.array([f['t'] for f in frames])
        self.period = float(self.times[-1]) if len(frames) > 1 else 0.0
        self.time_offset = float(spec.get('time_offset', 0.0))
        anchor = frames[0]['torso'][:2]
        to_cam_src = frames[0]['camera'][:2] - anchor
        to_cam_dst = ref_camera_world[:2] - place_world[:2]
        theta = (_yaw(to_cam_dst) - _yaw(to_cam_src)
                 + np.deg2rad(spec.get('yaw_deg', 0.0)))
        c, s = np.cos(theta), np.sin(theta)
        self.rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.anchor = np.array([anchor[0], anchor[1], 0.0])
        self.place = np.array([place_world[0], place_world[1], 0.0])

    def at(self, t):
        """ベースの区間の先頭からの秒 ``t`` の (点 [N,3], 色 [N,3],
        深度があるか [N], 素材の深度)."""
        if self.period <= 0:
            frame = self.frames[0]
        else:
            u = (t + self.time_offset) % (2.0 * self.period)
            if u > self.period:
                u = 2.0 * self.period - u
            frame = self.frames[int(np.argmin(np.abs(self.times - u)))]
        pts = (frame['points'] - self.anchor).dot(self.rot.T) + self.place
        return pts, frame['colors'], frame['valid'], frame['depth']


def render(people, t, cam_from_world, intrinsics, shape, stride):
    """重ねる人を描いた (色, 深度 [m] (無い所は inf), 深度があるか)."""
    h, w = shape
    color = np.zeros((h, w, 3), np.uint8)
    depth = np.full((h, w), np.inf, np.float32)
    has_depth = np.zeros((h, w), bool)
    all_u, all_v, all_z, all_c, all_s, all_ok = [], [], [], [], [], []
    for person in people:
        pts, cols, valid, src_depth = person.at(t)
        if len(pts) == 0:
            continue
        pc = cam_from_world[:3, :3].dot(pts.T) + cam_from_world[:3, 3:4]
        z = pc[2]
        ok = z > 0.2
        if not ok.any():
            continue
        z = z[ok]
        u = intrinsics.fx * pc[0, ok] / z + intrinsics.cx
        v = intrinsics.fy * pc[1, ok] / z + intrinsics.cy
        # 素材より近くに置くと点の間が空くので、塗る大きさを広げる (画面の
        # 端では斜めから見て間隔がさらに広がるので 1.5 倍)。
        size = int(np.clip(np.ceil(1.5 * stride * src_depth / np.median(z)),
                           stride, 4 * stride))
        all_u.append(u)
        all_v.append(v)
        all_z.append(z)
        all_c.append(cols[ok])
        all_ok.append(valid[ok])
        all_s.append(np.full(len(z), size, np.int32))
    if not all_z:
        return color, depth, has_depth
    u = np.concatenate(all_u)
    v = np.concatenate(all_v)
    z = np.concatenate(all_z).astype(np.float32)
    c = np.concatenate(all_c)
    s = np.concatenate(all_s)
    d_ok = np.concatenate(all_ok)
    order = np.argsort(-z)  # 奥から描き、手前で上書きする
    u, v, z, c, s, d_ok = (u[order], v[order], z[order], c[order], s[order],
                           d_ok[order])
    for size in np.unique(s):
        sel = s == size
        half = size // 2
        ui = np.round(u[sel]).astype(int) - half
        vi = np.round(v[sel]).astype(int) - half
        zsel, csel, osel = z[sel], c[sel], d_ok[sel]
        for dy in range(size):
            for dx in range(size):
                x = ui + dx
                y = vi + dy
                ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                xs, ys = x[ok], y[ok]
                zs, cs, os_ = zsel[ok], csel[ok], osel[ok]
                # 同じ画素に複数の点: 最後 (手前) の書き込みが残る。大きさの
                # 違う組の間は深度で比べる。
                nearer = zs <= depth[ys, xs]
                depth[ys[nearer], xs[nearer]] = zs[nearer]
                color[ys[nearer], xs[nearer]] = cs[nearer]
                has_depth[ys[nearer], xs[nearer]] = os_[nearer]
    return color, depth, has_depth


def base_person_mask(segmenter, bgr, args):
    """ベースの (重ねる前の) カラーに写っている本人の領域 (少し太らせる)."""
    h, w = bgr.shape[:2]
    found = segmenter.detect(cv2.resize(bgr, (w // 2, h // 2)))
    if found is None:
        return None
    seg = (found[1] > args.seg_threshold).astype(np.uint8)
    seg = cv2.dilate(seg, np.ones((7, 7), np.uint8))
    return cv2.resize(seg, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def composite(base_depth, layer_depth, backdrop, protect=None):
    """重ねた人が見える画素 (手前が勝つ。``backdrop`` より遠いベースの画素は
    重ねた人の方が奥でも上書きする。``protect`` (ベースの本人の領域) には
    重ねない: 本人の黒い服・髪は深度が抜けていて前後を比べられないため)."""
    win = np.isfinite(layer_depth)
    if base_depth is not None:
        valid = np.isfinite(base_depth) & (base_depth > 0)
        win &= ~(valid & (base_depth < layer_depth) & (base_depth < backdrop))
    if protect is not None:
        win &= ~protect
    return win


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--base', required=True)
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--end', type=float, default=None)
    parser.add_argument('--ref-time', type=float, default=None,
                        help='人の位置 (place) を測る base_link の時刻 '
                             '(bag の先頭からの秒、既定は --start)。')
    parser.add_argument('--people', required=True)
    parser.add_argument('--model', default=os.path.expanduser(
        '~/.mediapipe/models/pose_landmarker_lite.task'))
    parser.add_argument('--out-bag')
    parser.add_argument('--out-video')
    parser.add_argument('--video-scale', type=float, default=0.5)
    parser.add_argument('--cache-dir', default=os.path.expanduser(
        '~/work/crowd_data/cache'))
    parser.add_argument('--stride', type=int, default=2,
                        help='素材の人の点を何画素おきに取るか。')
    parser.add_argument('--source-frame-step', type=int, default=2)
    parser.add_argument('--depth-margin', type=float, default=0.35)
    parser.add_argument('--floor-cut', type=float, default=0.05,
                        help='素材の人の点で、床からこの高さ [m] 以下は落とす。')
    parser.add_argument('--min-points', type=int, default=3000,
                        help='素材の人の点がこれより少ないフレームは使わない。')
    parser.add_argument('--body-radius', type=float, default=0.7)
    parser.add_argument('--seg-threshold', type=float, default=0.5)
    parser.add_argument('--max-source-score', type=float, default=0.3,
                        help='素材のフレームの差し出しスコア (見張りの式、向きは'
                             '最悪の場合) がこれ以上なら使わない (0 以下で無効)。')
    parser.add_argument('--backdrop-depth', type=float, default=2.2)
    parser.add_argument('--max-frames', type=int, default=0,
                        help='確認用: カラーをこの枚数だけ処理して止める。')
    args = parser.parse_args()

    import rosbag
    import rospy
    from aero_demo.people_pose_types import CameraIntrinsics

    with open(args.people) as f:
        specs = json.load(f)
    segmenter = Segmenter(args.model)
    sources = [load_or_extract(spec, segmenter, args) for spec in specs]
    for spec, frames in zip(specs, sources):
        describe(spec['name'], frames)
    if not args.out_bag and not args.out_video:
        return

    print('[base] TF を読み込みます: {}'.format(args.base))
    t_load = time.time()
    tf_buf = mps._load_tf(args.base)
    print('[base] TF {:.0f} 秒'.format(time.time() - t_load))

    bag = rosbag.Bag(args.base)
    t_begin = bag.get_start_time()
    start = rospy.Time.from_sec(t_begin + args.start)
    end = (None if args.end is None
           else rospy.Time.from_sec(t_begin + args.end))
    ref_stamp = rospy.Time.from_sec(
        t_begin + (args.start if args.ref_time is None else args.ref_time))

    def world_from_camera(frame_id, stamp):
        base_from_cam = mps._matrix(tf_buf, 'base_link', frame_id, stamp)
        if base_from_cam is None:
            return None
        world_from_base = mps._matrix(tf_buf, 'odom', 'base_link', stamp)
        if world_from_base is None:
            world_from_base = np.eye(4)
        return world_from_base.dot(base_from_cam)

    ref_world_from_base = mps._matrix(tf_buf, 'odom', 'base_link', ref_stamp)
    if ref_world_from_base is None:
        ref_world_from_base = np.eye(4)
    ref_cam = world_from_camera('camera_color_optical_frame', ref_stamp)
    people = []
    for spec, frames in zip(specs, sources):
        if not frames:
            print('[warn] {} は人が取れなかったので使いません'.format(spec['name']))
            continue
        place = ref_world_from_base.dot(
            np.array([spec['place'][0], spec['place'][1], 0.0, 1.0]))[:3]
        people.append(PlacedPerson(spec, frames, place, ref_cam[:3, 3]))

    out_bag = rosbag.Bag(args.out_bag, 'w') if args.out_bag else None
    video = None
    intrinsics = None
    base_depth_m = None  # 直近のベースの深度 (カラーの重ね合わせの前後判定に使う)
    protect = None  # 直近のベースのカラーでの本人の領域
    out_depth_vis = None
    n_color = 0
    t_start = time.time()
    try:
        for topic, msg, t in bag.read_messages(start_time=start,
                                               end_time=end):
            if topic == INFO_TOPIC and intrinsics is None:
                intrinsics = CameraIntrinsics.from_matrix(msg.K)
            if topic in (COLOR_TOPIC, DEPTH_TOPIC) and intrinsics is not None:
                stamp = msg.header.stamp
                world_cam = world_from_camera(msg.header.frame_id, stamp)
                rel = stamp.to_sec() - (t_begin + args.start)
                if topic == COLOR_TOPIC:
                    img = np.frombuffer(msg.data, np.uint8).reshape(
                        msg.height, msg.width, 3)
                    bgr = img[:, :, ::-1] if msg.encoding == 'rgb8' else img
                    protect = base_person_mask(segmenter, bgr, args)
                if world_cam is not None:
                    cam_world = np.linalg.inv(world_cam)
                    layer_c, layer_d, layer_ok = render(
                        people, rel, cam_world, intrinsics,
                        (msg.height, msg.width), args.stride)
                    if topic == DEPTH_TOPIC:
                        raw = np.frombuffer(msg.data, np.uint16).reshape(
                            msg.height, msg.width)
                        base_m = raw.astype(np.float32) / 1000.0
                        base_m[raw == 0] = np.nan
                        win = composite(base_m, layer_d, args.backdrop_depth,
                                        protect)
                        new = raw.copy()
                        new[win] = np.clip(layer_d[win] * 1000.0, 1,
                                           65535).astype(np.uint16)
                        # 黒い服・髪など、素材で深度が無かった所は抜けにする。
                        new[win & ~layer_ok] = 0
                        msg.data = new.tobytes()
                        base_depth_m = base_m
                        out_depth_vis = new
                    else:
                        win = composite(base_depth_m, layer_d,
                                        args.backdrop_depth, protect)
                        new = img.copy()
                        rgb = layer_c[:, :, ::-1] if msg.encoding == 'rgb8' \
                            else layer_c
                        new[win] = rgb[win]
                        msg.data = new.tobytes()
                        n_color += 1
                        if args.out_video:
                            video = _write_video(
                                video, args, new, msg.encoding,
                                out_depth_vis, rel)
                        if n_color % 100 == 0:
                            print('[render] {:.1f} 秒まで ({} 枚、{:.0f} 秒)'
                                  .format(rel, n_color,
                                          time.time() - t_start))
            if out_bag is not None:
                out_bag.write(topic, msg, t)
            if args.max_frames and n_color >= args.max_frames:
                break
    finally:
        bag.close()
        if out_bag is not None:
            out_bag.close()
        if video is not None:
            video.release()
    print('[done] カラー {} 枚、{:.0f} 秒'.format(n_color,
                                              time.time() - t_start))


def _write_video(video, args, img, encoding, depth_raw, rel):
    bgr = img[:, :, ::-1] if encoding == 'rgb8' else img
    h, w = bgr.shape[:2]
    sw, sh = int(w * args.video_scale), int(h * args.video_scale)
    left = cv2.resize(bgr, (sw, sh))
    if depth_raw is None:
        right = np.zeros_like(left)
    else:
        d = cv2.resize(depth_raw, (sw, sh),
                       interpolation=cv2.INTER_NEAREST).astype(np.float32)
        vis = np.clip(d / 6000.0 * 255.0, 0, 255).astype(np.uint8)
        right = cv2.applyColorMap(255 - vis, cv2.COLORMAP_JET)
        right[d == 0] = 0
    frame = np.hstack([left, right])
    cv2.putText(frame, '{:.1f}s'.format(rel), (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    if video is None:
        video = cv2.VideoWriter(args.out_video,
                                cv2.VideoWriter_fourcc(*'mp4v'), 30.0,
                                (frame.shape[1], frame.shape[0]))
    video.write(frame)
    return video


if __name__ == '__main__':
    main()
