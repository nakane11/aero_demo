#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""閉ループの合成カメラ: ロボットの今の首・台車の姿勢から、人だかりの部屋の
カラー・深度画像を描いて出す.

録画を流すだけでは首を振っても画像が変わらないので、偽のコントローラ
(``tools/ros/fake_aero_controllers.py``) が出す ``/joint_states``・``/odom``
から頭のカメラの位置を順運動学で求め、そこから見える画像を描く。

- 部屋: 床・壁・天井・箱を光線で解析的に描く (深度も同じ幾何から出す)。
  模様は木目の床・白い壁に扉や貼り紙などをほどほどに付ける。
- 人: ``tools/crowd_sim_prepare.py`` が作った場面 (実機の bag から切り抜いた
  人の点群) を、場面の時刻で動かして手前優先で重ねる。場面の時刻は、ノードが
  最初に ARMED になった時刻 (``/run_camera_pipeline_test/status``) から数える
  (それまでは 0 のまま)。
- 出すもの: ``/camera/color/image_raw/decompressed`` (rgb8)・
  ``/camera/depth/image_raw/decompressed`` (16UC1、mm、カラーに位置合わせ)・
  ``/camera/color/camera_info``、TF (odom -> base_link、base_link -> head_link・
  r_eef_grasp_link、head_link -> camera_color_optical_frame)。
- 合成の odom 系 = ロボットの最初の base_link 系 (場面の ``place`` の座標)。

時刻は壁時計 (``/use_sim_time`` は false)。確認用に描いた画像と首ヨーの推移を
mp4 (``--video``) に、時刻ごとの記録を JSONL (``--log``) に書く。

使い方 (ROS を source した aero-uv)::

    python3 tools/ros/crowd_sim_camera.py --scene scene.pkl --video out.mp4
"""

import argparse
import json
import math
import os
import pickle
import sys
import threading
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOOLS = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_TOOLS), 'src'))
sys.path.insert(0, _TOOLS)

# 実機の 17:12 の bag の camera_info と head_link -> camera_color_optical_frame。
FULL_K = (1128.78, 1128.59, 969.03, 531.82, 1920, 1080)
HEAD_FROM_OPTICAL = np.array([
    [0.0253, 0.0482, 0.9985, 0.0278],
    [-0.9996, -0.0093, 0.0258, -0.0253],
    [0.0105, -0.9988, 0.0479, 0.194],
    [0.0, 0.0, 0.0, 1.0]])
OPTICAL_FRAME = 'camera_color_optical_frame'
VIDEO_FPS = 10

# 部屋 (合成の odom 系) [m]: x 前・y 左。ロボットは原点で +x を向く。
ROOM = dict(x=(-2.0, 5.0), y=(-3.0, 3.0), z=(0.0, 2.7))
# 箱 (机・棚など): (min xyz, max xyz, 色 RGB)。
BOXES = [
    ((2.3, 1.7, 0.0), (3.5, 2.4, 0.74), (235, 235, 230)),   # 白い机
    ((4.5, -2.6, 0.0), (5.0, -1.4, 1.8), (120, 95, 70)),    # 棚
    ((-1.9, 1.8, 0.0), (-1.3, 3.0, 1.1), (200, 200, 205)),  # キャビネット
    ((1.0, 2.5, 0.0), (1.8, 3.0, 0.9), (90, 110, 140)),     # 青い箱
    ((-1.0, -3.0, 0.0), (0.2, -2.5, 0.75), (230, 225, 215)),  # 机
]
# 壁の貼り物: (壁, 横の範囲, 高さの範囲, 色 RGB)。壁は 'x+' (x = x_max の壁) など。
DECALS = [
    ('x+', (-1.2, -0.3), (0.0, 2.05), (120, 150, 140)),   # 扉
    ('x+', (0.6, 1.2), (1.2, 1.8), (210, 60, 50)),        # 貼り紙
    ('x+', (1.6, 2.6), (0.9, 1.6), (40, 40, 45)),         # モニタ
    ('y+', (0.5, 1.5), (1.3, 1.9), (230, 210, 90)),
    ('y-', (1.8, 3.0), (0.0, 2.05), (150, 130, 110)),     # 扉
    ('y-', (-0.5, 0.2), (1.2, 1.7), (60, 90, 170)),
    ('x-', (-1.5, 1.5), (0.0, 0.1), (90, 90, 90)),        # 幅木
]


def _noise(seed, size=256):
    rng = np.random.default_rng(seed)
    n = rng.normal(0.0, 1.0, (size, size)).astype(np.float32)
    return cv2.GaussianBlur(n, (0, 0), 2.0)


NOISE_A = _noise(1)
NOISE_B = _noise(2)


def _sample(noise, a, b, scale):
    size = noise.shape[0]
    i = (np.floor(a * scale).astype(np.int64)) % size
    j = (np.floor(b * scale).astype(np.int64)) % size
    return noise[i, j]


class Room(object):
    """部屋を光線で描く."""

    def __init__(self, width, height, fx, fy, cx, cy):
        u, v = np.meshgrid(np.arange(width, dtype=np.float32),
                           np.arange(height, dtype=np.float32))
        # カメラ系の光線 (z = 1)。t がそのまま深度になる。
        self.rays = np.stack([(u - cx) / fx, (v - cy) / fy,
                              np.ones_like(u)], axis=-1).reshape(-1, 3)
        self.shape = (height, width)

    def render(self, world_from_cam):
        R = world_from_cam[:3, :3].astype(np.float32)
        o = world_from_cam[:3, 3].astype(np.float32)
        d = self.rays.dot(R.T)
        n = len(d)
        eps = 1e-6
        # 部屋の内側から壁・床・天井へ。
        best_t = np.full(n, np.inf, np.float32)
        surf = np.zeros(n, np.int8)  # 0..5: x-,x+,y-,y+,z-,z+、6+: 箱
        for axis, key in enumerate(('x', 'y', 'z')):
            lo, hi = ROOM[key]
            da = d[:, axis]
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.where(da > eps, (hi - o[axis]) / da,
                             np.where(da < -eps, (lo - o[axis]) / da, np.inf))
            side = np.where(da > 0, 2 * axis + 1, 2 * axis).astype(np.int8)
            closer = t < best_t
            best_t = np.where(closer, t, best_t)
            surf = np.where(closer, side, surf)
        box_color = np.zeros((n, 3), np.float32)
        for b, (lo, hi, color) in enumerate(BOXES):
            lo = np.asarray(lo, np.float32)
            hi = np.asarray(hi, np.float32)
            with np.errstate(divide='ignore', invalid='ignore'):
                t1 = (lo - o) / d
                t2 = (hi - o) / d
            tmin = np.minimum(t1, t2)
            tmax = np.maximum(t1, t2)
            t_near = np.nanmax(tmin, axis=1)
            t_far = np.nanmin(tmax, axis=1)
            hit = (t_near < t_far) & (t_near > 0.05) & (t_near < best_t)
            if not hit.any():
                continue
            face = np.argmax(tmin[hit], axis=1)
            shade = np.array([0.85, 0.75, 1.0], np.float32)[face]
            best_t[hit] = t_near[hit]
            surf[hit] = 6 + b
            box_color[hit] = np.asarray(color, np.float32) * shade[:, None]
        p = o + d * best_t[:, None]
        color = np.zeros((n, 3), np.float32)
        # 床: 木目 (板は y 方向に 0.18 m 幅)。
        floor = surf == 4
        if floor.any():
            px, py = p[floor, 0], p[floor, 1]
            plank = np.floor(py / 0.18)
            tone = 0.85 + 0.15 * np.sin(plank * 12.9898) ** 2
            grain = 10.0 * np.sin(px * 25.0 + 3.0 * _sample(NOISE_A, px, py, 8))
            base = np.array([170, 115, 70], np.float32)
            color[floor] = (base[None] * tone[:, None]
                            + grain[:, None]
                            + 6.0 * _sample(NOISE_B, px, py, 30)[:, None])
            seam = np.abs(py / 0.18 - np.round(py / 0.18)) < 0.04
            color[np.flatnonzero(floor)[seam]] *= 0.7
        # 天井
        ceil = surf == 5
        color[ceil] = 215 + 4.0 * _sample(NOISE_A, p[ceil, 0], p[ceil, 1], 10)[:, None]
        # 壁
        for s, name, along in ((0, 'x-', 1), (1, 'x+', 1), (2, 'y-', 0),
                               (3, 'y+', 0)):
            m = surf == s
            if not m.any():
                continue
            a = p[m, along]
            z = p[m, 2]
            c = np.tile(np.array([226, 222, 210], np.float32), (len(a), 1))
            c += 5.0 * _sample(NOISE_B, a, z, 20)[:, None]
            for wall, (a0, a1), (z0, z1), dcol in DECALS:
                if wall != name:
                    continue
                sel = (a >= a0) & (a <= a1) & (z >= z0) & (z <= z1)
                c[sel] = np.asarray(dcol, np.float32)
            c[z < 0.08] *= 0.6  # 幅木
            color[m] = c
        boxes = surf >= 6
        color[boxes] = (box_color[boxes]
                        + 4.0 * _sample(NOISE_A, p[boxes, 0] + p[boxes, 2],
                                        p[boxes, 1], 20)[:, None])
        h, w = self.shape
        return (np.clip(color, 0, 255).astype(np.uint8).reshape(h, w, 3),
                best_t.reshape(h, w))


class ScenePerson(object):
    """場面の人 (区間を時刻で切り替える)."""

    def __init__(self, spec, head_xy):
        from make_crowd_bag import PlacedPerson
        self.name = spec['name']
        self.clips = []
        place = np.array([spec['place'][0], spec['place'][1], 0.0])
        for clip in spec['clips']:
            placed = PlacedPerson(
                dict(name=spec['name'], yaw_deg=spec['yaw_deg']),
                clip['frames'], place, np.array([head_xy[0], head_xy[1], 0.0]))
            self.clips.append((clip['start_t'], clip['mode'], clip['hold'],
                               placed))

    def at(self, t):
        start, mode, hold, placed = [c for c in self.clips if c[0] <= t][-1] \
            if any(c[0] <= t for c in self.clips) else self.clips[0]
        u = max(0.0, t - start)
        if mode == 'once_then_pingpong' and placed.period > 0:
            if u > placed.period:
                lo = max(0.0, placed.period - hold)
                span = placed.period - lo
                w = (u - placed.period) % (2.0 * span) if span > 0 else 0.0
                u = placed.period - (w if w <= span else 2.0 * span - w)
            frame = placed.frames[int(np.argmin(np.abs(placed.times - u)))]
            pts = (frame['points'] - placed.anchor).dot(placed.rot.T) \
                + placed.place
            return pts, frame['colors'], frame['valid'], frame['depth']
        return placed.at(u)


def render_people(people, t, cam_from_world, fx, fy, cx, cy, shape,
                  point_step=3, src_fx=FULL_K[0], stride=2):
    """人の点を手前優先で描く (色, 深度 (無ければ inf), 深度があるか)."""
    h, w = shape
    color = np.zeros((h, w, 3), np.uint8)
    depth = np.full((h, w), np.inf, np.float32)
    has_depth = np.zeros((h, w), bool)
    us, vs, zs, cs, oks, ss = [], [], [], [], [], []
    for person in people:
        pts, cols, valid, src_depth = person.at(t)
        pts, cols, valid = (pts[::point_step], cols[::point_step],
                            valid[::point_step])
        pc = cam_from_world[:3, :3].dot(pts.T) + cam_from_world[:3, 3:4]
        ok = pc[2] > 0.2
        if not ok.any():
            continue
        z = pc[2, ok]
        u = fx * pc[0, ok] / z + cx
        v = fy * pc[1, ok] / z + cy
        spacing = (stride * point_step ** 0.5 * src_depth / np.median(z)
                   * fx / src_fx)
        size = int(np.clip(np.ceil(1.6 * spacing), 1, 6))
        us.append(u)
        vs.append(v)
        zs.append(z)
        cs.append(cols[ok][:, ::-1])  # 素材の色は BGR
        oks.append(valid[ok])
        ss.append(np.full(len(z), size, np.int32))
    if not zs:
        return color, depth, has_depth
    u, v, z = np.concatenate(us), np.concatenate(vs), np.concatenate(zs)
    c, ok, s = np.concatenate(cs), np.concatenate(oks), np.concatenate(ss)
    order = np.argsort(-z)
    u, v, z, c, ok, s = u[order], v[order], z[order], c[order], ok[order], \
        s[order]
    for size in np.unique(s):
        sel = s == size
        half = size // 2
        ui = np.round(u[sel]).astype(int) - half
        vi = np.round(v[sel]).astype(int) - half
        zsel, csel, osel = z[sel], c[sel], ok[sel]
        for dy in range(size):
            for dx in range(size):
                x, y = ui + dx, vi + dy
                inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
                xs, ys = x[inside], y[inside]
                zz, cc, oo = zsel[inside], csel[inside], osel[inside]
                nearer = zz <= depth[ys, xs]
                depth[ys[nearer], xs[nearer]] = zz[nearer]
                color[ys[nearer], xs[nearer]] = cc[nearer]
                has_depth[ys[nearer], xs[nearer]] = oo[nearer]
    return color, depth, has_depth


def _matrix(pos, rot):
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = pos
    return m


def _transform_msg(parent, child, m, stamp):
    from geometry_msgs.msg import TransformStamped
    from scipy.spatial.transform import Rotation
    tr = TransformStamped()
    tr.header.stamp = stamp
    tr.header.frame_id = parent
    tr.child_frame_id = child
    tr.transform.translation.x, tr.transform.translation.y, \
        tr.transform.translation.z = [float(v) for v in m[:3, 3]]
    q = Rotation.from_matrix(m[:3, :3]).as_quat()
    tr.transform.rotation.x, tr.transform.rotation.y, \
        tr.transform.rotation.z, tr.transform.rotation.w = [float(v) for v in q]
    return tr


class SimCamera(object):
    def __init__(self, args):
        import rospy
        import tf2_ros
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import String
        from aero_demo.aero_urdf_setup import load_aero
        self.rospy = rospy
        self.Image = Image
        self.args = args
        scale = args.width / float(FULL_K[4])
        self.fx, self.fy = FULL_K[0] * scale, FULL_K[1] * scale
        self.cx, self.cy = FULL_K[2] * scale, FULL_K[3] * scale
        self.width = args.width
        self.height = int(round(FULL_K[5] * scale))
        # 部屋は args.room_scale 倍の解像度で描いて拡大する (速さのため。人は
        # 出す解像度で描く: 遠くの手が小さすぎると MediaPipe Hands が見つけない)。
        rs = args.room_scale
        self.room = Room(int(round(self.width * rs)),
                         int(round(self.height * rs)), self.fx * rs,
                         self.fy * rs, self.cx * rs, self.cy * rs)
        self.robot = load_aero(use_hand=True)
        self.joints = {j.name: j for j in self.robot.joint_list}
        self.head_link = self.robot.head_link
        self.hand_link = getattr(self.robot, 'r_eef_grasp_link', None)
        self.lock = threading.Lock()
        self.joint_positions = {}
        self.base = (0.0, 0.0, 0.0)
        self.state = None
        self.armed_wall = None
        self.events = []
        with open(args.scene, 'rb') as f:
            scene = pickle.load(f)
        # 人は最初の頭の位置 (原点の上) を向くように置く。
        self.people = [ScenePerson(p, (0.0, 0.0)) for p in scene['people']]
        info = CameraInfo()
        info.header.frame_id = OPTICAL_FRAME
        info.width, info.height = self.width, self.height
        info.distortion_model = 'plumb_bob'
        info.D = [0.0] * 5
        info.K = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        info.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.P = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.info = info
        self.color_pub = rospy.Publisher(
            '/camera/color/image_raw/decompressed', Image, queue_size=1)
        self.depth_pub = rospy.Publisher(
            '/camera/depth/image_raw/decompressed', Image, queue_size=1)
        self.info_pub = rospy.Publisher('/camera/color/camera_info',
                                        CameraInfo, queue_size=1)
        self.tf_pub = tf2_ros.TransformBroadcaster()
        rospy.Subscriber('/joint_states', JointState, self._on_joints,
                         queue_size=1)
        rospy.Subscriber('/odom', Odometry, self._on_odom, queue_size=1)
        rospy.Subscriber('/run_camera_pipeline_test/status', String,
                         self._on_status, queue_size=10)
        self.video = None
        self.closed = False
        self.watch_text = 'watch: -'
        self.log = open(args.log, 'w') if args.log else None
        # 見張りの内訳 (人ごとのスコア・見ている人) を記録する (購読すると
        # ノードが ~detections を出す)。
        self.det_log = (open(args.log.rsplit('.', 1)[0] + '_watch.jsonl', 'w')
                        if args.log else None)
        if self.det_log is not None:
            rospy.Subscriber('/run_camera_pipeline_test/detections', String,
                             self._on_detection, queue_size=50)

    def _on_detection(self, msg):
        try:
            det = json.loads(msg.data)
        except ValueError:
            return
        now = time.time()
        multi = det.get('multi_person') or []
        record = dict(
            wall=now, scene_t=self.scene_t(now), stamp=det.get('stamp'),
            state=det.get('state'), n_people=det.get('n_people'),
            watch=[dict(id=p.get('track_id'), target=p.get('target'),
                        score=round(p.get('score') or 0.0, 2),
                        ema=(None if p.get('score_ema') is None
                             else round(p['score_ema'], 2)),
                        cost=(None if p.get('idle_cost') is None
                              else round(p['idle_cost'], 2)),
                        box=[int(v) for v in p.get('box') or []])
                   for p in multi],
            hands=len(det.get('hands') or []))
        if self.det_log is not None:
            self.det_log.write(json.dumps(record) + '\n')
        target = [w for w in record['watch'] if w['target']]
        self.watch_text = 'watch: ' + '  '.join(
            '{}{}:{:.2f}'.format('*' if w['target'] else '', w['id'],
                                 w['ema'] or 0.0) for w in record['watch'])
        if not target:
            self.watch_text += '  (target not in view)'

    # --- 入力 ---
    def _on_joints(self, msg):
        with self.lock:
            self.joint_positions.update(zip(msg.name, msg.position))

    def _on_odom(self, msg):
        q = msg.pose.pose.orientation
        yaw = 2.0 * math.atan2(q.z, q.w)
        with self.lock:
            self.base = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                         yaw)

    def _on_status(self, msg):
        try:
            state = json.loads(msg.data).get('state')
        except ValueError:
            return
        if state != self.state:
            now = time.time()
            if state == 'armed' and self.armed_wall is None:
                self.armed_wall = now
            self.events.append(dict(wall=now, scene_t=self.scene_t(now),
                                    state=state))
            print('[sim] 状態 {} -> {} (場面 {:.2f} 秒)'.format(
                self.state, state, self.scene_t(now)), flush=True)
            self.state = state

    def scene_t(self, now):
        return 0.0 if self.armed_wall is None else now - self.armed_wall

    # --- 姿勢 ---
    def _poses(self):
        with self.lock:
            q = dict(self.joint_positions)
            bx, by, byaw = self.base
        for name, value in q.items():
            joint = self.joints.get(name)
            if joint is not None:
                joint.joint_angle(float(value))
        base_from_head = _matrix(self.head_link.worldpos(),
                                 self.head_link.worldrot())
        base_from_hand = (None if self.hand_link is None else _matrix(
            self.hand_link.worldpos(), self.hand_link.worldrot()))
        c, s = math.cos(byaw), math.sin(byaw)
        odom_from_base = _matrix([bx, by, 0.0],
                                 [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return (odom_from_base, base_from_head, base_from_hand,
                q.get('neck_y_joint', 0.0), q.get('neck_p_joint', 0.0))

    # --- 1 枚 ---
    def step(self):
        rospy = self.rospy
        now = time.time()
        stamp = rospy.Time.from_sec(now)
        odom_from_base, base_from_head, base_from_hand, neck_y, neck_p = \
            self._poses()
        transforms = [
            _transform_msg('odom', 'base_link', odom_from_base, stamp),
            _transform_msg('base_link', 'head_link', base_from_head, stamp),
            _transform_msg('head_link', OPTICAL_FRAME, HEAD_FROM_OPTICAL,
                           stamp)]
        if base_from_hand is not None:
            transforms.append(_transform_msg('base_link', 'r_eef_grasp_link',
                                             base_from_hand, stamp))
        self.tf_pub.sendTransform(transforms)
        world_from_cam = odom_from_base.dot(base_from_head).dot(
            HEAD_FROM_OPTICAL)
        t_scene = self.scene_t(now)
        bg_color, bg_depth = self.room.render(world_from_cam)
        if bg_color.shape[1] != self.width:
            size = (self.width, self.height)
            bg_color = cv2.resize(bg_color, size, interpolation=cv2.INTER_LINEAR)
            bg_depth = cv2.resize(bg_depth, size,
                                  interpolation=cv2.INTER_NEAREST)
        cam_from_world = np.linalg.inv(world_from_cam)
        p_color, p_depth, p_ok = render_people(
            self.people, t_scene, cam_from_world, self.fx, self.fy, self.cx,
            self.cy, (self.height, self.width))
        win = p_depth < bg_depth
        color = bg_color.copy()
        color[win] = p_color[win]
        depth = bg_depth.copy()
        depth[win] = p_depth[win]
        depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
        depth_mm[win & ~p_ok] = 0  # 素材で深度が無かった所 (黒い服・髪)
        depth_mm[~np.isfinite(depth)] = 0
        self._publish(color, depth_mm, stamp)
        if self.log is not None:
            self.log.write(json.dumps(dict(
                wall=now, scene_t=t_scene, neck_y=neck_y, neck_p=neck_p,
                base=list(self.base), state=self.state)) + '\n')
        # 終わる合図で close() した後に開き直すと、1 枚だけで上書きしてしまう。
        if self.args.video and not self.closed:
            self._write_video(color, depth_mm, t_scene, neck_y)

    def _publish(self, color_rgb, depth_mm, stamp):
        img = self.Image()
        img.header.stamp = stamp
        img.header.frame_id = OPTICAL_FRAME
        img.height, img.width = color_rgb.shape[:2]
        img.encoding = 'rgb8'
        img.step = img.width * 3
        img.data = color_rgb.tobytes()
        dep = self.Image()
        dep.header = img.header
        dep.height, dep.width = depth_mm.shape
        dep.encoding = '16UC1'
        dep.step = dep.width * 2
        dep.data = depth_mm.tobytes()
        self.info.header.stamp = stamp
        self.info_pub.publish(self.info)
        self.color_pub.publish(img)
        self.depth_pub.publish(dep)

    def _write_video(self, color_rgb, depth_mm, t_scene, neck_y):
        bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        vis = np.clip(depth_mm.astype(np.float32) / 6000.0 * 255, 0, 255)
        dep = cv2.applyColorMap(255 - vis.astype(np.uint8), cv2.COLORMAP_JET)
        dep[depth_mm == 0] = 0
        frame = np.hstack([bgr, dep])
        h = frame.shape[0]
        # 首ヨーの目盛 (下の帯): 中央 0°、左右 ±90°。
        band = np.full((40, frame.shape[1], 3), 30, np.uint8)
        x0 = frame.shape[1] // 2
        span = frame.shape[1] // 2 - 20
        for deg in (-60, 0, 60):
            x = int(x0 - deg / 90.0 * span)
            cv2.line(band, (x, 5), (x, 35), (120, 120, 120), 1)
        x = int(x0 - math.degrees(neck_y) / 90.0 * span)
        cv2.circle(band, (x, 20), 8, (0, 255, 255), -1)
        frame = np.vstack([frame, band])
        cv2.putText(frame, 'scene {:5.1f}s  neck_y {:+5.1f}deg  {}'.format(
            t_scene, math.degrees(neck_y), self.state or '-'), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, self.watch_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # ffmpeg (H.264) に生の画像を渡す。描く速さが揺れるので、壁時計で
        # VIDEO_FPS の枚数になるよう同じ画像を繰り返して書く。
        if self.video is None and getattr(self, 'video_t0', None) is not None:
            return  # 一度閉じた (終わる合図と競合した) 後は開き直さない
        if self.video is None:
            import subprocess
            h, w = frame.shape[:2]
            self.video = subprocess.Popen(
                ['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                 '-pix_fmt', 'bgr24', '-s', '{}x{}'.format(w, h),
                 '-r', str(VIDEO_FPS), '-i', '-', '-c:v', 'libx264',
                 '-pix_fmt', 'yuv420p', self.args.video],
                stdin=subprocess.PIPE)
            self.video_t0 = time.time()
            self.video_n = 0
        want = int((time.time() - self.video_t0) * VIDEO_FPS) + 1
        data = np.ascontiguousarray(frame).tobytes()
        while self.video_n < want:
            self.video.stdin.write(data)
            self.video_n += 1

    def spin(self):
        period = 1.0 / self.args.fps
        n = 0
        t_report = time.time()
        while not self.rospy.is_shutdown():
            t0 = time.time()
            self.step()
            n += 1
            if time.time() - t_report > 10.0:
                print('[sim] {:.1f} fps (1 枚 {:.0f} ms)'.format(
                    n / (time.time() - t_report),
                    1e3 * (time.time() - t0)), flush=True)
                n = 0
                t_report = time.time()
            time.sleep(max(0.0, period - (time.time() - t0)))
        self.close()

    def close(self):
        self.closed = True
        if self.video is not None:
            self.video.stdin.close()
            self.video.wait()
            self.video = None
        if self.log is not None:
            for e in self.events:
                self.log.write(json.dumps(dict(event=e)) + '\n')
            self.log.close()
            self.log = None
        if self.det_log is not None:
            self.det_log.close()
            self.det_log = None


def render_still(args):
    """確認用: 首ヨーを変えた静止画を描く (ROS なし)."""
    from aero_demo.aero_urdf_setup import load_aero
    with open(args.scene, 'rb') as f:
        scene = pickle.load(f)
    people = [ScenePerson(p, (0.0, 0.0)) for p in scene['people']]
    scale = args.width / float(FULL_K[4])
    fx, fy = FULL_K[0] * scale, FULL_K[1] * scale
    cx, cy = FULL_K[2] * scale, FULL_K[3] * scale
    width, height = args.width, int(round(FULL_K[5] * scale))
    room = Room(width, height, fx, fy, cx, cy)
    robot = load_aero(use_hand=True)
    robot.neck_p_joint.joint_angle(0.435)
    tiles = []
    for t_scene in (0.0, 8.0):
        for yaw_deg in (60.0, 0.0, -60.0):
            robot.neck_y_joint.joint_angle(math.radians(yaw_deg))
            head = _matrix(robot.head_link.worldpos(), robot.head_link.worldrot())
            world_from_cam = head.dot(HEAD_FROM_OPTICAL)
            t0 = time.time()
            bg_color, bg_depth = room.render(world_from_cam)
            p_color, p_depth, _ = render_people(
                people, t_scene, np.linalg.inv(world_from_cam), fx, fy, cx, cy,
                (height, width))
            win = p_depth < bg_depth
            color = bg_color.copy()
            color[win] = p_color[win]
            dt = time.time() - t0
            bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, 't={:.0f}s yaw={:+.0f} ({:.0f}ms)'.format(
                t_scene, yaw_deg, 1e3 * dt), (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(bgr)
    rows = [np.hstack(tiles[i:i + 3]) for i in (0, 3)]
    cv2.imwrite(args.still, np.vstack(rows))
    print('[still]', args.still)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scene', required=True)
    parser.add_argument('--width', type=int, default=960,
                        help='出す画像の幅 (高さは 16:9)。640 では 1.7 m 先の手が'
                             '約 20 px で、MediaPipe Hands が見つけなかった。')
    parser.add_argument('--room-scale', type=float, default=0.5,
                        help='部屋 (背景) を描く解像度の倍率 (拡大して使う)。')
    parser.add_argument('--fps', type=float, default=12.0)
    parser.add_argument('--video', default=None)
    parser.add_argument('--log', default=None)
    parser.add_argument('--still', default=None,
                        help='首ヨー ±60°・0° の静止画をこのファイルに描いて終わる。')
    args, _ = parser.parse_known_args()
    if args.still:
        render_still(args)
        return
    import rospy
    rospy.init_node('crowd_sim_camera')
    sim = SimCamera(args)
    rospy.on_shutdown(sim.close)
    sim.spin()


if __name__ == '__main__':
    main()
