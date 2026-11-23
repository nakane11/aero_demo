# -*- coding:utf-8 -*-

"""接近中の再照準 (台車の残りの軌道を検出した掌に合わせてずらす) の純粋な計算。

残りの軌道を「元の軌道 + 滑らかに 0 -> 1 へ変わるずれ」で作り直したゴールに
差し替える。座標は odom 系 (台車の指令の積分) と、スキャン照合で求めた実際の系。
"""

import math

import numpy as np

# 掌が計画からこれ以上ずれていたら誤検出とみなして補正しない [m]。
MAX_PALM_ERROR = 0.25
# 作り直した軌道で、元の軌道より進行方向の速度がこれ以上は戻らないこと [m/s]。
REVERSE_TOLERANCE = 0.02
# 差し替えの先頭点と次の点の最小間隔 [s] (近すぎると加速度が跳ねる)。
MIN_POINT_GAP = 0.15
# エルミート補間の速度を調べる 1 区間あたりの点数。
HERMITE_SAMPLES = 21
# ``monotone_tail`` で試すずれの縮め方。
TAIL_SCALES = (1.0, 0.7, 0.5, 0.3)


def wrap_angle(angle):
    """``angle`` を (-pi, pi] にする。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def smoothstep(tau):
    """0 -> 1 で始点・終点の傾きが 0 の滑らかな補間。"""
    tau = float(np.clip(tau, 0.0, 1.0))
    return tau * tau * (3.0 - 2.0 * tau)


def smoothstep_rate(tau):
    """``smoothstep`` の ``tau`` での微分。"""
    tau = float(np.clip(tau, 0.0, 1.0))
    return 6.0 * tau * (1.0 - tau)


def _rotation(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def transform_point(pose, point):
    """2 次元の姿勢 ``pose`` ``[x, y, yaw]`` で点 ``point`` を変換する。"""
    return (_rotation(pose[2]) @ np.asarray(point, dtype=np.float64)[:2]
            + np.asarray(pose[:2], dtype=np.float64))


def inverse_transform_point(pose, point):
    """``transform_point`` の逆。"""
    return _rotation(pose[2]).T @ (np.asarray(point, dtype=np.float64)[:2]
                                   - np.asarray(pose[:2], dtype=np.float64))


def palm_anchored_position(palm_xy, palm_in_base_xy, yaw):
    """向き ``yaw`` の台車が掌 ``palm_xy`` を台車座標系の
    ``palm_in_base_xy`` に見る台車の位置。"""
    return (np.asarray(palm_xy, dtype=np.float64)[:2]
            - _rotation(yaw) @ np.asarray(palm_in_base_xy, dtype=np.float64)[:2])


def blended_offset(t, shifts, t_end):
    """``shifts`` の各 ``(shift, t_start)`` を、``t_start`` から ``t_end`` まで
    ``smoothstep`` で 0 -> 1 に効かせた合計のずれと、その速度を返す。"""
    offset = np.zeros(2)
    rate = np.zeros(2)
    for shift, t_start in shifts:
        duration = t_end - t_start
        if duration <= 0.0:
            continue
        tau = (t - t_start) / duration
        offset += np.asarray(shift, dtype=np.float64) * smoothstep(tau)
        rate += (np.asarray(shift, dtype=np.float64)
                 * smoothstep_rate(tau) / duration)
    return offset, rate


def build_retarget_points(cum_times, positions, velocities, yaws, yaw_rates,
                          elapsed, current_position, current_velocity,
                          current_yaw, current_yaw_rate, shifts):
    """元の台車軌道の ``elapsed`` 秒以降に ``shifts``
    (``[(ずれ (2), 効かせ始める時刻 [s, 元の軌道の先頭基準]), ...]``) を
    ``blended_offset`` で足して作り直した点列
    ``[(t, 位置(2), 速度(2), yaw, yaw の速度), ...]`` を返す (時刻は今が 0)。

    元の軌道は点 k の時刻 ``cum_times[k]`` (先頭 0) とその odom 系の
    ``positions[k]``/``velocities[k]``/``yaws[k]``/``yaw_rates[k]``。先頭の点は
    今の状態 (``current_*``)、以降は ``MIN_POINT_GAP`` 以上先の点を並べる。
    何度ずらしても、前のずらしの途中から続けられる。``yaws`` は折り返さない
    連続の値。残りが ``MIN_POINT_GAP`` 未満なら None。
    """
    cum_times = np.asarray(cum_times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64)
    t_end = float(cum_times[-1])
    if t_end - elapsed < MIN_POINT_GAP:
        return None
    # 今の時刻の計画の向きを今の向きの近くで表し、以降は計画の増分を足す。
    planned_now = float(np.interp(elapsed, cum_times, yaws))
    yaw_base = float(current_yaw) + wrap_angle(planned_now - current_yaw)
    points = [(0.0, np.asarray(current_position, dtype=np.float64),
               np.asarray(current_velocity, dtype=np.float64),
               float(current_yaw), float(current_yaw_rate))]
    for k in range(len(cum_times)):
        t = float(cum_times[k]) - elapsed
        if t < MIN_POINT_GAP:
            continue
        offset, rate = blended_offset(float(cum_times[k]), shifts, t_end)
        points.append((
            t, positions[k] + offset, velocities[k] + rate,
            yaw_base + float(yaws[k]) - planned_now, float(yaw_rates[k])))
    return points


def hermite_velocity_along(points, direction):
    """点列 (``build_retarget_points`` の形) を 3 次エルミート補間したときの、
    ``direction`` 方向の速度の最小値 [m/s]。"""
    direction = np.asarray(direction, dtype=np.float64)[:2]
    lowest = float('inf')
    for (t0, p0, v0, _, _), (t1, p1, v1, _, _) in zip(points[:-1], points[1:]):
        duration = t1 - t0
        if duration <= 0.0:
            continue
        a = (v0 + v1) / duration ** 2 - 2.0 * (p1 - p0) / duration ** 3
        b = 3.0 * (p1 - p0) / duration ** 2 - (2.0 * v0 + v1) / duration
        for t in np.linspace(0.0, duration, HERMITE_SAMPLES):
            velocity = 3.0 * a * t ** 2 + 2.0 * b * t + v0
            lowest = min(lowest, float(np.dot(velocity, direction)))
    return lowest


def hermite_peak_speed(points):
    """点列を 3 次エルミート補間したときの並進速度の大きさの最大値 [m/s]。"""
    peak = 0.0
    for (t0, p0, v0, _, _), (t1, p1, v1, _, _) in zip(points[:-1], points[1:]):
        duration = t1 - t0
        if duration <= 0.0:
            continue
        a = (v0 + v1) / duration ** 2 - 2.0 * (p1 - p0) / duration ** 3
        b = 3.0 * (p1 - p0) / duration ** 2 - (2.0 * v0 + v1) / duration
        for t in np.linspace(0.0, duration, HERMITE_SAMPLES):
            peak = max(peak, float(np.linalg.norm(
                3.0 * a * t ** 2 + 2.0 * b * t + v0)))
    return peak


def monotone_tail(build, shift, direction, max_speed=None):
    """``build(shift)`` の点列のうち、進行方向の速度が元の軌道 (``build(0)``)
    より ``REVERSE_TOLERANCE`` 以上戻らず、並進の速さが ``max_speed`` (None
    なら見ない) を超えないものを、``shift`` を ``TAIL_SCALES`` の順に縮めながら
    探して ``(points, scale)`` で返す (見つからなければ ``(None, None)``)。"""
    base = build(np.zeros(2))
    if base is None:
        return None, None
    floor = min(hermite_velocity_along(base, direction), 0.0) \
        - REVERSE_TOLERANCE
    # 元の軌道がすでに上限を超えていても、それ以上は増やさない。
    speed_cap = None if max_speed is None else max(
        max_speed, hermite_peak_speed(base))
    for scale in TAIL_SCALES:
        points = build(np.asarray(shift, dtype=np.float64) * scale)
        if points is None:
            continue
        if hermite_velocity_along(points, direction) < floor:
            continue
        if speed_cap is not None and hermite_peak_speed(points) > speed_cap:
            continue
        return points, scale
    return None, None
