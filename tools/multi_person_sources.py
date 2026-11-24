#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``eval_multi_person_offer.py`` に渡すフレームの入力元.

どれも ``dict(t, color (BGR), depth [m], intrinsics, camera_to_base (4x4),
base_to_world (4x4 or None), robot_position (base 系), gt)`` を順に返す。

- ``bag``: 実機の rosbag (TF でカメラ→base_link、odom、ロボット手先)。
- ``panoptic``: CMU Panoptic の Kinect (深度をカラーへ投影)。ロボットは
  カメラの位置にいるとみなす。
- ``composite``: 実機の rosbag 2 本 (A: 前からいる人、B: 後から手を出す人)
  を、深度で人を切り抜いて 1 枚に重ねる (``composite_frames``)。
"""

import json
import os

import cv2
import numpy as np

from aero_demo.people_pose_types import CameraIntrinsics

_FALLBACK_ROBOT_HAND = (0.35, -0.25, 0.85)


def add_arguments(parser):
    parser.add_argument('--source', choices=('bag', 'panoptic', 'composite'),
                        default='bag')
    parser.add_argument('--bag', help='bag (source=bag、composite の A)。')
    parser.add_argument('--start', type=float, default=0.0,
                        help='bag の先頭からの秒 (A)。')
    parser.add_argument('--end', type=float, default=None)
    parser.add_argument('--color-topic',
                        default='/camera/color/image_raw/decompressed')
    parser.add_argument('--depth-topic',
                        default='/camera/depth/image_raw/decompressed')
    parser.add_argument('--camera-info-topic',
                        default='/camera/color/camera_info')
    parser.add_argument('--frame-step', type=int, default=2,
                        help='このフレームおきに使う (既定 2: 約 15 fps、ノードの'
                             '処理の速さに近い)。')
    parser.add_argument('--panoptic-dir')
    parser.add_argument('--panoptic-node', type=int, default=1)
    parser.add_argument('--panoptic-scale', type=float, default=0.5,
                        help='カラーの縮小率 (1920x1080 → 960x540)。')
    parser.add_argument('--panoptic-flip', action='store_true',
                        help='深度を左右反転する (160422_haggling1 は不要)。')
    parser.add_argument('--panoptic-camera-height', type=float, default=1.3,
                        help='カメラの床からの高さ [m] (ロボットの頭くらい)。')
    # composite
    parser.add_argument('--bag-b', help='後から手を出す人の bag (composite)。')
    parser.add_argument('--start-b', type=float, default=0.0)
    parser.add_argument('--b-delay', type=float, default=0.0,
                        help='B を A の何秒後から重ねるか。')
    parser.add_argument('--b-shift', type=float, default=-0.45,
                        help='B を横にずらす量 (画像の幅に対する割合)。')
    parser.add_argument('--b-depth-margin', type=float, default=0.35,
                        help='B の人の切り抜き: 体幹の深度 + この値 [m] まで。')


def open_source(args):
    if args.source == 'bag':
        return bag_frames(args.bag, args.start, args.end, args)
    if args.source == 'panoptic':
        return panoptic_frames(args)
    return composite_frames(args)


# ----------------------------------------------------------------------
# rosbag
# ----------------------------------------------------------------------
def _load_tf(bag_path):
    import genpy
    import rosbag
    import tf2_ros
    buf = tf2_ros.BufferCore(genpy.Duration(3600))
    with rosbag.Bag(bag_path) as bag:
        for topic, msg, _t in bag.read_messages(topics=['/tf', '/tf_static']):
            for tr in msg.transforms:
                try:
                    if topic == '/tf_static':
                        buf.set_transform_static(tr, 'bag')
                    else:
                        buf.set_transform(tr, 'bag')
                except Exception:  # noqa: BLE001
                    pass
    return buf


def _matrix(buf, target, source, stamp):
    from aero_demo.ros_camera_utils import transform_to_matrix
    try:
        tr = buf.lookup_transform_core(target, source, stamp)
    except Exception:  # noqa: BLE001
        return None
    return transform_to_matrix(tr.transform)


def bag_frames(bag_path, start, end, args, slop=0.05):
    import rosbag
    from aero_demo.ros_camera_utils import imgmsg_to_ndarray
    buf = _load_tf(bag_path)
    with rosbag.Bag(bag_path) as bag:
        t_begin = bag.get_start_time()
        import rospy
        start_time = rospy.Time.from_sec(t_begin + start)
        end_time = (None if end is None
                    else rospy.Time.from_sec(t_begin + end))
        info = None
        depths = []  # 直近 2 枚 (深度はカラーより遅れて届く)
        waiting = []  # 深度を待っているカラー
        count = 0

        def pairs():
            for topic, msg, _t in bag.read_messages(
                    topics=[args.color_topic, args.depth_topic,
                            args.camera_info_topic],
                    start_time=start_time, end_time=end_time):
                if topic == args.camera_info_topic:
                    nonlocal_info[0] = msg
                elif topic == args.color_topic:
                    waiting.append(msg)
                else:
                    depths.append(msg)
                    del depths[:-2]
                    while waiting and waiting[0].header.stamp \
                            <= msg.header.stamp:
                        color = waiting.pop(0)
                        best = min(depths, key=lambda d: abs(
                            (d.header.stamp - color.header.stamp).to_sec()))
                        if abs((best.header.stamp - color.header.stamp)
                               .to_sec()) <= slop:
                            yield color, best

        nonlocal_info = [None]
        for msg, depth in pairs():
            info = nonlocal_info[0]
            if info is None:
                continue
            stamp = msg.header.stamp
            count += 1
            if (count - 1) % args.frame_step:
                continue
            camera_to_base = _matrix(buf, 'base_link', msg.header.frame_id,
                                     stamp)
            if camera_to_base is None:
                continue
            base_to_world = _matrix(buf, 'odom', 'base_link', stamp)
            hand = _matrix(buf, 'base_link', 'r_eef_grasp_link', stamp)
            robot = (np.asarray(_FALLBACK_ROBOT_HAND) if hand is None
                     else hand[:3, 3])
            depth_raw = imgmsg_to_ndarray(depth)
            depth_m = (depth_raw.astype(np.float32) / 1000.0
                       if depth.encoding == '16UC1'
                       else depth_raw.astype(np.float32))
            yield dict(
                t=stamp.to_sec(),
                color=imgmsg_to_ndarray(msg, desired_encoding='bgr8'),
                depth=depth_m,
                intrinsics=CameraIntrinsics.from_matrix(info.K),
                camera_to_base=camera_to_base, base_to_world=base_to_world,
                robot_position=robot, gt=None)


# ----------------------------------------------------------------------
# 合成 (実機 bag 2 本を重ねる)
# ----------------------------------------------------------------------
def _person_mask(depth, box, margin):
    """``box`` の中で、中央付近の深度 + ``margin`` より手前の画素."""
    h, w = depth.shape
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    mask = np.zeros((h, w), dtype=bool)
    if x1 <= x0 or y1 <= y0:
        return mask
    region = depth[y0:y1, x0:x1]
    cy0, cy1 = (y1 - y0) // 4, (y1 - y0) // 2
    cx0, cx1 = (x1 - x0) // 3, 2 * (x1 - x0) // 3
    core = region[cy0:cy1, cx0:cx1]
    valid = core[np.isfinite(core) & (core > 0)]
    if valid.size == 0:
        return mask
    body = float(np.median(valid))
    near = np.isfinite(region) & (region > 0.2) & (region < body + margin)
    near = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((5, 5), np.uint8)).astype(bool)
    mask[y0:y1, x0:x1] = near
    return mask


def composite_frames(args):
    """A の画面に B の人 (深度で切り抜き、横にずらす) を重ねる.

    B の人は見張りの PoseLandmarker の bbox で探す (毎フレーム、一番大きい人)。
    重ねた画素は深度も B のものにする (手前が勝つ)。``gt`` には B を重ね
    始めたかと、B の人の bbox (ずらした後) を入れる。
    """
    from aero_demo.multi_person_watcher import PoseWatcher
    watcher = PoseWatcher(args.model, num_poses=2)
    frames_a = bag_frames(args.bag, args.start, args.end, args)
    frames_b = None
    t_a0 = None
    t_b0 = None
    for fa in frames_a:
        if t_a0 is None:
            t_a0 = fa['t']
        gt = dict(b_visible=False, b_box=None, b_t=None)
        if fa['t'] - t_a0 >= args.b_delay:
            if frames_b is None:
                frames_b = bag_frames(args.bag_b, args.start_b, None, args)
            fb = next(frames_b, None)
            if fb is not None:
                if t_b0 is None:
                    t_b0 = fb['t']
                # B の bag の --start-b からの秒 (B 単独の評価と比べる物差し)。
                gt['b_t'] = fb['t'] - t_b0
                _paste(fa, fb, watcher, args, gt)
        fa['gt'] = gt
        yield fa


def _paste(fa, fb, watcher, args, gt):
    people = watcher.detect(fb['color'])
    if not people:
        return
    box = max(people, key=lambda p: (p['box'][2] - p['box'][0])
              * (p['box'][3] - p['box'][1]))['box']
    h, w = fb['depth'].shape
    pad_x = 0.25 * (box[2] - box[0])
    box = [box[0] - pad_x, box[1] - 0.1 * (box[3] - box[1]),
           box[2] + pad_x, box[3] + 0.05 * (box[3] - box[1])]
    mask = _person_mask(fb['depth'], box, args.b_depth_margin)
    shift = int(round(args.b_shift * w))
    m = np.zeros_like(mask)
    if shift >= 0:
        m[:, shift:] = mask[:, :w - shift]
    else:
        m[:, :w + shift] = mask[:, -shift:]
    color_b = np.roll(fb['color'], shift, axis=1)
    depth_b = np.roll(fb['depth'], shift, axis=1)
    depth_a = fa['depth']
    if depth_a.shape != depth_b.shape or fa['color'].shape != color_b.shape:
        return
    valid_a = np.isfinite(depth_a) & (depth_a > 0)
    front = m & (~valid_a | (depth_b < depth_a))
    fa['color'] = fa['color'].copy()
    fa['depth'] = depth_a.copy()
    fa['color'][front] = color_b[front]
    fa['depth'][front] = depth_b[front]
    gt.update(b_visible=True, b_box=[box[0] + shift, box[1], box[2] + shift,
                                     box[3]])


# ----------------------------------------------------------------------
# CMU Panoptic (Kinect v2)
# ----------------------------------------------------------------------
_PANOPTIC_DEPTH_SHAPE = (424, 512)
_PANOPTIC_DEPTH_BYTES = 424 * 512 * 2


class PanopticKinect(object):
    """CMU Panoptic の Kinect 1 台分 (カラー mp4 と深度 .dat)."""

    def __init__(self, directory, node=1, scale=0.5, flip_depth=False):
        self.dir = directory
        calib = json.load(open(os.path.join(directory, 'kcalibration.json')))
        sensor = calib['sensors'][node - 1]
        self.K_d = np.asarray(sensor['K_depth'], dtype=np.float64)
        self.dist_d = np.asarray(sensor['distCoeffs_depth'],
                                 dtype=np.float64).ravel()[:5]
        self.K_c = np.asarray(sensor['K_color'], dtype=np.float64)
        self.dist_c = np.asarray(sensor['distCoeffs_color'],
                                 dtype=np.float64).ravel()[:5]
        M_d = np.asarray(sensor['M_depth'], dtype=np.float64)
        M_c = np.asarray(sensor['M_color'], dtype=np.float64)
        self.depth_to_color = M_c.dot(np.linalg.inv(M_d))
        self.scale = scale
        self.flip_depth = flip_depth
        sync = json.load(open(os.path.join(directory, 'ksynctables.json')))
        key = 'KINECTNODE{}'.format(node)
        self.color_time = np.asarray(
            sync['kinect']['color'][key]['univ_time'], dtype=np.float64)
        self.depth_time = np.asarray(
            sync['kinect']['depth'][key]['univ_time'], dtype=np.float64)
        self.depth_path = os.path.join(directory, 'depth{}.dat'.format(node))
        self.n_depth = os.path.getsize(self.depth_path) // _PANOPTIC_DEPTH_BYTES
        self.video_path = os.path.join(directory, 'kinect{}.mp4'.format(node))
        h, w = _PANOPTIC_DEPTH_SHAPE
        u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                           np.arange(h, dtype=np.float64))
        pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
        self._rays = cv2.undistortPoints(pts, self.K_d, self.dist_d).reshape(
            -1, 2)
        self.intrinsics = CameraIntrinsics(
            fx=self.K_c[0, 0] * scale, fy=self.K_c[1, 1] * scale,
            cx=self.K_c[0, 2] * scale, cy=self.K_c[1, 2] * scale)

    def read_depth(self, index):
        with open(self.depth_path, 'rb') as f:
            f.seek(index * _PANOPTIC_DEPTH_BYTES)
            raw = np.fromfile(f, dtype=np.uint16,
                              count=_PANOPTIC_DEPTH_BYTES // 2)
        img = raw.reshape(_PANOPTIC_DEPTH_SHAPE).astype(np.float64) / 1000.0
        if self.flip_depth:
            img = img[:, ::-1]
        return img

    def depth_on_color(self, depth, color_shape):
        """深度をカラー (縮小後) の画素へ投影し、穴を手前優先で埋めた [m]."""
        z = depth.ravel()
        ok = z > 0.3
        pts = np.column_stack([self._rays[ok] * z[ok, None], z[ok]])
        pts = pts.dot(self.depth_to_color[:3, :3].T) + \
            self.depth_to_color[:3, 3]
        uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3),
                                  np.zeros(3), self.K_c, self.dist_c)
        uv = uv.reshape(-1, 2) * self.scale
        h, w = color_shape[:2]
        ui = np.round(uv[:, 0]).astype(int)
        vi = np.round(uv[:, 1]).astype(int)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        inv = np.zeros((h, w), dtype=np.float32)
        # 同じ画素に複数来たら手前 (逆数が大きい方)。
        np.maximum.at(inv, (vi[inside], ui[inside]),
                      (1.0 / pts[inside, 2]).astype(np.float32))
        filled = cv2.dilate(inv, np.ones((5, 5), np.uint8))
        inv = np.where(inv > 0, inv, filled)
        out = np.zeros_like(inv)
        out[inv > 0] = 1.0 / inv[inv > 0]
        return out

    def frames(self, start=0.0, end=None, step=2):
        """(t, color, depth) を ``step`` フレームおきに返す (t は動画の秒)."""
        valid_depth = np.where(self.depth_time[:self.n_depth] > 0)[0]
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        first = int(start * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        index = first
        while True:
            ok, color = cap.read()
            if not ok:
                break
            t = index / fps
            i = index
            index += 1
            if end is not None and t > end:
                break
            if (i - first) % step:
                continue
            tc = self.color_time[i] if i < len(self.color_time) else -1
            if tc <= 0 or valid_depth.size == 0:
                continue
            j = valid_depth[np.argmin(np.abs(self.depth_time[valid_depth]
                                             - tc))]
            if abs(self.depth_time[j] - tc) > 40:  # [ms]
                continue
            color = cv2.resize(color, None, fx=self.scale, fy=self.scale)
            yield t, color, self.depth_on_color(self.read_depth(j),
                                                color.shape)


def _estimate_up(kinect, model_path, start, n=20):
    """カメラ座標系の上向きを、見えている人の腰→肩の平均で決める."""
    from aero_demo.multi_person_watcher import PoseWatcher, lift_to_3d
    watcher = PoseWatcher(model_path)
    ups = []
    for k, (_t, color, depth) in enumerate(kinect.frames(start, step=15)):
        if k >= n:
            break
        for person in watcher.detect(color):
            pts = lift_to_3d(person['pixels'], depth, kinect.intrinsics)
            sh = [pts[s] for s in ('LShoulder', 'RShoulder') if s in pts]
            hp = [pts[s] for s in ('LHip', 'RHip') if s in pts]
            if sh and hp:
                d = np.mean(sh, axis=0) - np.mean(hp, axis=0)
                if 0.3 < np.linalg.norm(d) < 0.8:
                    ups.append(d / np.linalg.norm(d))
    watcher.close()
    if not ups:
        return np.array([0.0, -1.0, 0.0])
    up = np.mean(ups, axis=0)
    return up / np.linalg.norm(up)


def panoptic_frames(args):
    """Panoptic の Kinect をロボットのカメラとみなす.

    base 系: 原点はカメラの真下の床 (高さは ``--panoptic-camera-height``)、
    z は上、x はカメラの光軸を水平にした向き。ロボットの手先は base 系で
    ``_FALLBACK_ROBOT_HAND`` に置く。
    """
    kinect = PanopticKinect(args.panoptic_dir, args.panoptic_node,
                            args.panoptic_scale,
                            flip_depth=args.panoptic_flip)
    up = _estimate_up(kinect, args.model, args.start)
    forward = np.array([0.0, 0.0, 1.0]) - up[2] * up
    forward /= np.linalg.norm(forward)
    left = np.cross(up, forward)
    rot = np.vstack([forward, left, up])  # base <- camera
    camera_to_base = np.eye(4)
    camera_to_base[:3, :3] = rot
    camera_to_base[2, 3] = args.panoptic_camera_height
    print('[panoptic] up (camera) = {}'.format(np.round(up, 3).tolist()))
    for t, color, depth in kinect.frames(args.start, args.end,
                                         args.frame_step):
        yield dict(t=t, color=color, depth=depth.astype(np.float32),
                   intrinsics=kinect.intrinsics,
                   camera_to_base=camera_to_base, base_to_world=None,
                   robot_position=np.asarray(_FALLBACK_ROBOT_HAND), gt=None)
