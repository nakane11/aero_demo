#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""rosbag の深度画像を床と /scan に合わせて head_link -> camera_link を推定する。

台車と頭・腰・脚が止まっている時刻の深度点群を、今の head_link -> camera_link に
補正 (camera_link 系の回転・並進) を掛けて base_link へ変換し、

- 床の点: base_link 系で z = ``--floor-z`` (ピッチ・ロール・高さが決まる)
- /scan と同じ高さの断面の点: /scan の点列の線上 (ヨー・x・y が決まる)

になるよう、全時刻まとめて最小二乗で解く。首の向きが違う時刻を多く含むほど、
回転と並進を分けて決めやすい (ARMED 中の首の見渡しが使える)。並進は今の値から
``--translation-sigma`` [m] 程度しか動かない前提の事前分布を置く。

出力は推定した head_link -> camera_link (launch/decompress.launch の
head_to_camera_link と scripts/solve_palm_ik.py の HEAD_TO_CAMERA_LINK_* に
書く値) と、時刻ごとの補正前後の残差。``--no-optimize`` と ``--head-to-camera``
を組み合わせると、別の bag で値を検証できる。roscore 不要。

Usage
-----
    python3 tools/ros/calibrate_head_camera.py \\
        --bag data/bags/head_calib_a.bag data/bags/head_calib_b.bag

    # 推定した値を別の bag で検証する
    python3 tools/ros/calibrate_head_camera.py --bag other.bag --no-optimize \\
        --head-to-camera 0.067 0.0104 0.1852 0.0325 -0.0106 0.0225 0.9989
"""

import argparse
import collections
import math
import os
import sys

import numpy as np
import genpy
import rosbag
import rospy
import tf2_ros
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_PKG_SRC_DIR = os.path.join(_REPO_ROOT, 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo.ros_camera_utils import (  # noqa: E402
    imgmsg_to_ndarray, transform_to_matrix)

# 静止判定に使う関節 (base_link -> カメラの連鎖)。joint_states に無いものは無視。
CHAIN_JOINTS = ('neck_y_joint', 'neck_p_joint', 'neck_r_joint',
                'waist_y_joint', 'waist_p_joint', 'waist_r_joint',
                'knee_joint', 'ankle_joint')
# 外側ループごとの対応付けの許容 (粗く合わせてから絞る)。
FLOOR_TOLERANCES = (0.20, 0.10, 0.06, 0.04, 0.04)  # [m]
SCAN_MAX_DISTANCES = (0.20, 0.10, 0.06, 0.05, 0.05)  # [m]
SCAN_NEIGHBOR_GAP = 0.05  # [m] 法線を作る隣の点がこれより離れていたら使わない
INLIER_DISTANCE = 0.03  # [m] 残差表示の「合っている点」
ROBUST_SCALE = 0.01  # [m] これより大きい残差は soft_l1 で重みを下げる


def load_tf_buffer(bag, t0, t1):
    """``[t0, t1]`` の ``/tf`` と全 ``/tf_static`` を読んだ ``BufferCore``。"""
    buf = tf2_ros.BufferCore(genpy.Duration(3600))
    for _, msg, _ in bag.read_messages(topics=['/tf_static']):
        for tr in msg.transforms:
            buf.set_transform_static(tr, 'bag')
    for _, msg, _ in bag.read_messages(
            topics=['/tf'], start_time=rospy.Time.from_sec(t0 - 1.0),
            end_time=rospy.Time.from_sec(t1 + 1.0)):
        for tr in msg.transforms:
            try:
                buf.set_transform(tr, 'bag')
            except Exception:
                pass
    return buf


def lookup(buf, target, source, stamp):
    return transform_to_matrix(
        buf.lookup_transform_core(target, source, stamp).transform)


def read_static_series(bag, t0, t1):
    """静止判定用の (関節の時刻, 関節角 [n, k]) と (odom の時刻, [x, y, yaw])。"""
    js_t, js_q, names = [], [], None
    for _, msg, _ in bag.read_messages(
            topics=['/joint_states'], start_time=rospy.Time.from_sec(t0),
            end_time=rospy.Time.from_sec(t1)):
        if names is None:
            names = [n for n in CHAIN_JOINTS if n in msg.name]
        if not all(n in msg.name for n in names):
            continue
        js_t.append(msg.header.stamp.to_sec())
        js_q.append([msg.position[msg.name.index(n)] for n in names])
    od_t, od_p = [], []
    for _, msg, _ in bag.read_messages(
            topics=['/odom'], start_time=rospy.Time.from_sec(t0),
            end_time=rospy.Time.from_sec(t1)):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        od_t.append(msg.header.stamp.to_sec())
        od_p.append([p.x, p.y, yaw])
    return (np.array(js_t), np.array(js_q), names or [],
            np.array(od_t), np.array(od_p))


def is_static(t, series, window, max_joint_change, max_base_move,
              max_base_turn):
    """``t`` の前後 ``window`` 秒で連鎖の関節と台車が止まっているか。"""
    js_t, js_q, _, od_t, od_p = series

    def span(times, values, lo, hi):
        m = (times >= lo) & (times <= hi)
        if m.sum() < 2:
            return None
        v = values[m]
        return v.max(axis=0) - v.min(axis=0)

    lo, hi = t - window, t + window
    if len(js_t):
        dq = span(js_t, js_q, lo, hi)
        if dq is None or np.max(np.abs(dq)) > max_joint_change:
            return False
    if len(od_t):
        dp = span(od_t, od_p, lo, hi)
        if dp is None or math.hypot(dp[0], dp[1]) > max_base_move \
                or abs(dp[2]) > max_base_turn:
            return False
    return True


def scan_to_base(scan_msg, T_base_laser, self_radius, max_range):
    """/scan の点 (base_link の xy) と各点の線の法線 (無効なら nan)。"""
    ang = (scan_msg.angle_min
           + np.arange(len(scan_msg.ranges)) * scan_msg.angle_increment)
    r = np.asarray(scan_msg.ranges, dtype=np.float64)
    ok = np.isfinite(r) & (r > scan_msg.range_min) & (r < scan_msg.range_max)
    pts = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros_like(r),
                    np.ones_like(r)])
    xy = (T_base_laser @ pts)[:2].T
    # 法線は角度順の両隣から作る (隣が欠けているか離れていれば無効)。
    normals = np.full_like(xy, np.nan)
    prev_xy, next_xy = np.roll(xy, 1, axis=0), np.roll(xy, -1, axis=0)
    prev_ok, next_ok = np.roll(ok, 1), np.roll(ok, -1)
    tangent = next_xy - prev_xy
    good = (ok & prev_ok & next_ok
            & (np.linalg.norm(xy - prev_xy, axis=1) < SCAN_NEIGHBOR_GAP)
            & (np.linalg.norm(next_xy - xy, axis=1) < SCAN_NEIGHBOR_GAP))
    good[0] = good[-1] = False
    tn = np.linalg.norm(tangent, axis=1)
    good &= tn > 1e-6
    normals[good] = np.stack([-tangent[good, 1], tangent[good, 0]],
                             axis=1) / tn[good, None]
    dist = np.linalg.norm(xy, axis=1)
    keep = good & (dist > self_radius) & (dist < max_range)
    return xy[keep], normals[keep]


def depth_points(depth_msg, K, stride, max_points, max_depth, rng):
    """深度画像を ``depth_msg.header.frame_id`` 系の点 [n, 3] にする。"""
    depth = imgmsg_to_ndarray(depth_msg)
    if depth_msg.encoding == '16UC1':
        depth = depth.astype(np.float32) / 1000.0
    else:
        depth = depth.astype(np.float32)
    vs, us = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[vs, us]
    ok = np.isfinite(z) & (z > 0.2) & (z < max_depth)
    us, vs, z = us[ok].astype(np.float64), vs[ok].astype(np.float64), z[ok]
    pts = np.stack([(us - K[0, 2]) * z / K[0, 0],
                    (vs - K[1, 2]) * z / K[1, 1], z], axis=1)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    return pts


def correction_matrix(params):
    """``params`` = (回転ベクトル 3, 並進 3) の 4x4 (camera_link 系で右から掛ける)。"""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(params[:3]).as_matrix()
    T[:3, 3] = params[3:]
    return T


class Frame(object):
    """1 時刻ぶんの深度点・/scan・TF。"""

    def __init__(self, stamp, pts, T_base_head, T_camlink_frame, scan_xy,
                 scan_normals, laser_height, neck):
        self.stamp = stamp
        self.pts_h = np.hstack([pts, np.ones((len(pts), 1))]).T
        self.T_base_head = T_base_head
        self.T_camlink_frame = T_camlink_frame
        self.scan_xy = scan_xy
        self.scan_normals = scan_normals
        self.tree = cKDTree(scan_xy)
        self.laser_height = laser_height
        self.neck = neck
        # 外側ループで決める対応付け (floor_ok: 床が見えていたか、未判定は None)
        self.floor_ok = None
        self.floor_idx = np.zeros(0, dtype=int)
        self.band_idx = np.zeros(0, dtype=int)
        self.band_q = np.zeros((0, 2))
        self.band_n = np.zeros((0, 2))

    def to_base(self, T_head_cam, params, idx=None):
        T = (self.T_base_head @ T_head_cam @ correction_matrix(params)
             @ self.T_camlink_frame)
        pts = self.pts_h if idx is None else self.pts_h[:, idx]
        return (T @ pts)[:3].T


def select_correspondences(frame, T_head_cam, params, args, floor_tol,
                           scan_max_d, rng):
    """今のパラメータで床の点と、/scan 高さの断面の点 (と対応する /scan 点) を選ぶ。"""
    P = frame.to_base(T_head_cam, params)
    horiz = np.linalg.norm(P[:, :2], axis=1)
    usable = (horiz > args.self_radius) & (horiz < args.max_range)
    floor = np.nonzero(usable & (np.abs(P[:, 2] - args.floor_z) < floor_tol))[0]
    # 床が見えていない時刻 (低い家具などを拾う) は床の条件に使わない:
    # 当てはめた平面が傾きすぎているか、平面に乗る点が少なければ捨てる。
    frame.floor_ok = False
    if len(floor) >= args.min_floor_points:
        F = P[floor]
        A = np.c_[F[:, 0], F[:, 1], np.ones(len(F))]
        coef = np.linalg.lstsq(A, F[:, 2], rcond=None)[0]
        on_plane = np.abs(A @ coef - F[:, 2]) < max(INLIER_DISTANCE,
                                                    floor_tol / 2)
        tilt = math.degrees(math.atan(math.hypot(coef[0], coef[1])))
        if tilt < args.max_floor_tilt_deg \
                and on_plane.sum() >= args.min_floor_points:
            floor = floor[on_plane]
            frame.floor_ok = True
    if not frame.floor_ok:
        floor = floor[:0]
    if len(floor) > args.max_floor_points:
        floor = rng.choice(floor, args.max_floor_points, replace=False)
    band = np.nonzero(usable & (np.abs(P[:, 2] - frame.laser_height)
                                < args.band_height))[0]
    if len(band) > args.max_band_points:
        band = rng.choice(band, args.max_band_points, replace=False)
    d, nn = frame.tree.query(P[band, :2])
    keep = d < scan_max_d
    frame.floor_idx = floor
    frame.band_idx = band[keep]
    frame.band_q = frame.scan_xy[nn[keep]]
    frame.band_n = frame.scan_normals[nn[keep]]


def robust(r, scale=ROBUST_SCALE):
    """soft_l1 と同じ重みになるよう変換した残差 (二乗和が soft_l1 の損失)。

    least_squares の loss を全体に掛けると並進の事前分布まで頭打ちになって
    効かなくなるため、データの残差だけをここで変換する。
    """
    z = (r / scale) ** 2
    return np.sign(r) * scale * np.sqrt(2.0 * (np.sqrt(1.0 + z) - 1.0))


def full_params(x, args):
    """最適化変数 ``x`` を (回転ベクトル 3, 並進 3) にする (並進固定なら 0)。"""
    return np.r_[x, np.zeros(3)] if args.fix_translation else x


def residuals(x, frames, T_head_cam, args):
    params = full_params(x, args)
    res = []
    for frame in frames:
        if len(frame.floor_idx):
            z = frame.to_base(T_head_cam, params, frame.floor_idx)[:, 2]
            res.append(args.floor_weight * robust(z - args.floor_z)
                       / math.sqrt(len(z)))
        if len(frame.band_idx):
            xy = frame.to_base(T_head_cam, params, frame.band_idx)[:, :2]
            d = np.sum((xy - frame.band_q) * frame.band_n, axis=1)
            res.append(args.scan_weight * robust(d) / math.sqrt(len(d)))
    if args.translation_sigma > 0 and not args.fix_translation:
        # 並進が sigma だけずれると、1 時刻の床か /scan が 1cm ずれたのと同じ重み。
        res.append(params[3:] / args.translation_sigma
                   * 0.01 * math.sqrt(len(frames)))
    return np.concatenate(res) if res else np.zeros(1)


def frame_stats(frame, T_head_cam, params, args):
    """表示用: 床の平均高さ・傾き、/scan 断面の点の線までの距離の中央値と一致率。"""
    P = frame.to_base(T_head_cam, params)
    horiz = np.linalg.norm(P[:, :2], axis=1)
    usable = (horiz > args.self_radius) & (horiz < args.max_range)
    # 補正前は床が大きくずれて見えるので広めに取り、平面から外れる点を除いて当てはめる。
    floor = P[usable & (np.abs(P[:, 2] - args.floor_z) < FLOOR_TOLERANCES[0])]
    fz = slope_x = slope_y = float('nan')
    for _ in range(3):
        if len(floor) <= 100:
            break
        A = np.c_[floor[:, 0], floor[:, 1], np.ones(len(floor))]
        coef = np.linalg.lstsq(A, floor[:, 2], rcond=None)[0]
        fz = float(np.median(floor[:, 2]) - args.floor_z)
        slope_x = math.degrees(math.atan(coef[0]))
        slope_y = math.degrees(math.atan(coef[1]))
        floor = floor[np.abs(A @ coef - floor[:, 2]) < INLIER_DISTANCE]
    band = P[usable & (np.abs(P[:, 2] - frame.laser_height)
                       < args.band_height)][:, :2]
    med = inlier = float('nan')
    if len(band):
        d, nn = frame.tree.query(band)
        line = np.abs(np.sum((band - frame.scan_xy[nn])
                             * frame.scan_normals[nn], axis=1))
        line = np.where(d < 0.2, line, d)
        med = float(np.median(line))
        inlier = float(np.mean(line < INLIER_DISTANCE))
    return fz, slope_x, slope_y, med, inlier, len(floor), len(band)


def format_transform(T):
    pos = T[:3, 3]
    qx, qy, qz, qw = Rotation.from_matrix(T[:3, :3]).as_quat()
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler('xyz', degrees=True)
    return pos, (qx, qy, qz, qw), rpy


def collect_frames(bag, args):
    info_msg = next((m for _, m, _ in bag.read_messages(
        topics=[args.camera_info_topic])), None)
    if info_msg is None:
        raise RuntimeError('{} が bag にありません。'.format(
            args.camera_info_topic))
    K = np.array(info_msg.K).reshape(3, 3)
    t_begin, t_end = bag.get_start_time(), bag.get_end_time()
    t0 = t_begin + args.start if args.start is not None else t_begin
    t1 = t_begin + args.end if args.end is not None else t_end
    series = read_static_series(bag, t0, t1)
    print('[calib] 区間 {:.1f}〜{:.1f} 秒 (bag 先頭から)、静止判定の関節: {}'
          .format(t0 - t_begin, t1 - t_begin, ', '.join(series[2]) or 'なし'))
    buf = load_tf_buffer(bag, t0, t1)
    rng = np.random.RandomState(args.seed)

    frames = []
    candidates = np.arange(t0 + args.static_window, t1 - args.static_window,
                           args.step)
    skipped = collections.Counter()
    for t in candidates:
        if len(frames) >= args.max_frames:
            break
        if not is_static(t, series, args.static_window,
                         math.radians(args.max_joint_change_deg),
                         args.max_base_move,
                         math.radians(args.max_base_turn_deg)):
            skipped['動いている'] += 1
            continue
        # 深度は記録時刻がヘッダ時刻より遅れる (0.2 秒程度) ので、/scan は前後に
        # 広く読んでヘッダ時刻が最も近いものを選ぶ。
        depth_msg = scan_msg = None
        scans = []
        for topic, msg, rec_t in bag.read_messages(
                topics=[args.depth_topic, args.scan_topic],
                start_time=rospy.Time.from_sec(t - 0.5),
                end_time=rospy.Time.from_sec(t + 0.5)):
            if topic == args.scan_topic:
                scans.append(msg)
            elif depth_msg is None and rec_t.to_sec() >= t:
                depth_msg = msg
        if depth_msg is None or not scans:
            skipped['深度か /scan が無い'] += 1
            continue
        stamp = depth_msg.header.stamp
        scan_msg = min(scans, key=lambda m: abs(
            (m.header.stamp - stamp).to_sec()))
        if abs((scan_msg.header.stamp - stamp).to_sec()) > 0.1:
            skipped['深度と /scan の時刻差が 0.1 秒超'] += 1
            continue
        try:
            T_base_head = lookup(buf, 'base_link', args.head_frame, stamp)
            T_camlink_frame = lookup(buf, args.camera_frame,
                                     depth_msg.header.frame_id, stamp)
            T_base_laser = lookup(buf, 'base_link', scan_msg.header.frame_id,
                                  scan_msg.header.stamp)
            if args.T_head_cam is None:
                args.T_head_cam = lookup(buf, args.head_frame,
                                         args.camera_frame, stamp)
        except Exception as e:
            print('[calib] TF を引けなかったため {:.2f} 秒を飛ばします ({})'
                  .format(t - t_begin, e))
            skipped['TF が引けない'] += 1
            continue
        scan_xy, scan_normals = scan_to_base(
            scan_msg, T_base_laser, args.self_radius, args.max_range)
        if len(scan_xy) < 50:
            skipped['/scan の使える点が 50 未満'] += 1
            continue
        pts = depth_points(depth_msg, K, args.stride, args.max_points,
                           args.max_depth, rng)
        neck = []
        js_t, js_q, names = series[0], series[1], series[2]
        if len(js_t):
            i = int(np.argmin(np.abs(js_t - stamp.to_sec())))
            neck = [(n, math.degrees(js_q[i, j])) for j, n in enumerate(names)
                    if n in ('neck_y_joint', 'neck_p_joint', 'waist_y_joint')]
        frames.append(Frame(stamp.to_sec() - t_begin, pts, T_base_head,
                            T_camlink_frame, scan_xy, scan_normals,
                            T_base_laser[2, 3], neck))
    print('[calib] {} 時刻を使います (除外: {})。'.format(
        len(frames), '、'.join('{} {}'.format(k, v) for k, v in
                              skipped.items()) or 'なし'))
    return frames


def print_frame_table(frames, T_head_cam, params, args, title):
    print('\n[{}]'.format(title))
    print('  時刻[s]  首・腰 [deg]                        床の高さ  床の傾き x/y [deg]'
          '   /scan までの距離 中央値  3cm 以内  床を使用  補正の base_link 系 '
          'rpy [deg]')
    rows = []
    for f in frames:
        fz, sx, sy, med, inl, nf, nb = frame_stats(f, T_head_cam, params, args)
        rows.append((fz, med, inl))
        joints = ' '.join('{}={:+5.1f}'.format(
            n.replace('_joint', ''), v) for n, v in f.neck)
        # 補正が base_link 系で何度の回転に当たるか (カメラ位置まわり)。
        R_old = (f.T_base_head @ T_head_cam)[:3, :3]
        R_new = (f.T_base_head @ T_head_cam @ correction_matrix(params))[:3, :3]
        base_rpy = Rotation.from_matrix(R_new @ R_old.T).as_euler(
            'xyz', degrees=True)
        print('  {:7.1f}  {:34s}  {:+.3f} m  {:+5.2f} / {:+5.2f}         '
              '{:.3f} m              {:.2f}      {}      ({:+.2f}, {:+.2f}, '
              '{:+.2f})'.format(
                  f.stamp, joints, fz, sx, sy, med, inl,
                  '-' if f.floor_ok is None else ('o' if f.floor_ok else 'x'),
                  *base_rpy))
    rows = np.array(rows, dtype=float)
    print('  全体: 床の高さ |平均| {:.3f} m、/scan までの距離 中央値 {:.3f} m、'
          '3cm 以内 {:.2f}'.format(np.nanmean(np.abs(rows[:, 0])),
                                np.nanmedian(rows[:, 1]),
                                np.nanmean(rows[:, 2])))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bag', required=True, nargs='+',
                        help='bag (複数可、場所や台車の向きを変えて録ったものを'
                             'まとめて解くと環境による偏りが減る)。')
    parser.add_argument('--start', type=float, default=None,
                        help='使う区間の始まり [s] (bag 先頭から、全 bag 共通)。')
    parser.add_argument('--end', type=float, default=None,
                        help='使う区間の終わり [s] (bag 先頭から)。')
    parser.add_argument('--step', type=float, default=1.0,
                        help='候補時刻の間隔 [s]。')
    parser.add_argument('--max-frames', type=int, default=40)
    parser.add_argument('--depth-topic',
                        default='/camera/depth/image_raw/decompressed')
    parser.add_argument('--camera-info-topic',
                        default='/camera/color/camera_info',
                        help='深度画像と同じ解像度・内部パラメータの camera_info '
                             '(color に位置合わせ済みの深度なら color)。')
    parser.add_argument('--scan-topic', default='/scan')
    parser.add_argument('--head-frame', default='head_link')
    parser.add_argument('--camera-frame', default='camera_link')
    parser.add_argument('--head-to-camera', type=float, nargs=7, default=None,
                        metavar=('X', 'Y', 'Z', 'QX', 'QY', 'QZ', 'QW'),
                        help='初期値 (省略時は bag の TF)。--no-optimize と '
                             '組み合わせると検証になる。')
    parser.add_argument('--no-optimize', action='store_true',
                        help='推定せず、初期値での残差だけ表示する。')
    parser.add_argument('--floor-z', type=float, default=0.0,
                        help='base_link 系の床の高さ [m]。')
    parser.add_argument('--band-height', type=float, default=0.03,
                        help='/scan の高さ ± これ [m] の深度点を断面とする。')
    parser.add_argument('--self-radius', type=float, default=0.5,
                        help='base_link からこれ [m] 以内の点はロボット自身と'
                             'みなして使わない。')
    parser.add_argument('--max-range', type=float, default=4.0)
    parser.add_argument('--max-depth', type=float, default=4.5)
    parser.add_argument('--stride', type=int, default=3,
                        help='深度画像の間引き [px]。')
    parser.add_argument('--max-points', type=int, default=40000)
    parser.add_argument('--max-floor-points', type=int, default=1500)
    parser.add_argument('--max-band-points', type=int, default=1500)
    parser.add_argument('--max-floor-tilt-deg', type=float, default=5.0,
                        help='時刻ごとの床の当てはめがこれより傾いていたら、'
                             'その時刻は床を使わない。')
    parser.add_argument('--min-floor-points', type=int, default=300)
    parser.add_argument('--floor-weight', type=float, default=1.0)
    parser.add_argument('--scan-weight', type=float, default=1.0)
    parser.add_argument('--translation-sigma', type=float, default=0.03,
                        help='並進の補正の事前分布の標準偏差 [m] (0 で無制限)。')
    parser.add_argument('--fix-translation', action='store_true',
                        help='並進は今の値のまま、回転だけ推定する。')
    parser.add_argument('--static-window', type=float, default=0.5,
                        help='静止判定で前後に見る時間 [s]。')
    parser.add_argument('--max-joint-change-deg', type=float, default=0.3)
    parser.add_argument('--max-base-move', type=float, default=0.005)
    parser.add_argument('--max-base-turn-deg', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.T_head_cam = None
    if args.head_to_camera is not None:
        x, y, z, qx, qy, qz, qw = args.head_to_camera
        args.T_head_cam = np.eye(4)
        args.T_head_cam[:3, :3] = Rotation.from_quat(
            [qx, qy, qz, qw]).as_matrix()
        args.T_head_cam[:3, 3] = [x, y, z]

    frames = []
    for path in args.bag:
        print('[calib] {}'.format(path))
        with rosbag.Bag(os.path.expanduser(path)) as bag:
            frames += collect_frames(bag, args)
    if len(frames) < 3:
        print('[calib] 使える時刻が {} 個しかないため終了します '
              '(--start/--end/--step や静止判定の閾値を見直してください)。'
              .format(len(frames)))
        return
    T_head_cam = args.T_head_cam
    pos, quat, rpy = format_transform(T_head_cam)
    print('[calib] 初期値 head_link -> camera_link: pos ({:.5f}, {:.5f}, {:.5f}) '
          'rpy ({:+.2f}, {:+.2f}, {:+.2f}) deg'.format(*pos, *rpy))

    params = np.zeros(6)
    print_frame_table(frames, T_head_cam, params, args, '補正前')
    if args.no_optimize:
        return

    rng = np.random.RandomState(args.seed)
    for floor_tol, scan_max_d in zip(FLOOR_TOLERANCES, SCAN_MAX_DISTANCES):
        for frame in frames:
            select_correspondences(frame, T_head_cam, params, args, floor_tol,
                                   scan_max_d, rng)
        n_floor = sum(len(f.floor_idx) for f in frames)
        n_band = sum(len(f.band_idx) for f in frames)
        x0 = params[:3] if args.fix_translation else params
        result = least_squares(residuals, x0,
                               args=(frames, T_head_cam, args))
        params = full_params(result.x, args)
        rot = Rotation.from_rotvec(params[:3]).as_euler('xyz', degrees=True)
        print('[calib] 許容 床 {:.2f} m / scan {:.2f} m (床 {} 点, 断面 {} 点): '
              '補正 回転 ({:+.2f}, {:+.2f}, {:+.2f}) deg、並進 ({:+.1f}, {:+.1f}, '
              '{:+.1f}) mm'.format(floor_tol, scan_max_d, n_floor, n_band,
                                   *rot, *(params[3:] * 1000)))

    print_frame_table(frames, T_head_cam, params, args, '補正後')
    T_new = T_head_cam @ correction_matrix(params)
    pos, quat, rpy = format_transform(T_new)
    rot = Rotation.from_rotvec(params[:3]).as_euler('xyz', degrees=True)
    print('\n[calib] 補正 (camera_link 系、右から掛ける): 回転 rpy '
          '({:+.2f}, {:+.2f}, {:+.2f}) deg、並進 ({:+.1f}, {:+.1f}, {:+.1f}) mm'
          .format(*rot, *(params[3:] * 1000)))
    print('[calib] 推定した head_link -> camera_link: pos ({:.5f}, {:.5f}, '
          '{:.5f}) rpy ({:+.2f}, {:+.2f}, {:+.2f}) deg'.format(*pos, *rpy))
    print('\nlaunch/decompress.launch の head_to_camera_link:')
    print('      args="{:.5f} {:.5f} {:.5f} {:.6f} {:.6f} {:.6f} {:.6f} '
          'head_link camera_link 100" />'.format(*pos, *quat))
    print('scripts/solve_palm_ik.py:')
    print('HEAD_TO_CAMERA_LINK_POS = ({:.5f}, {:.5f}, {:.5f})  # [m]'
          .format(*pos))
    print('HEAD_TO_CAMERA_LINK_QUAT_XYZW = ({:.6f}, {:.6f}, {:.6f}, {:.6f})'
          .format(*quat))


if __name__ == '__main__':
    main()
