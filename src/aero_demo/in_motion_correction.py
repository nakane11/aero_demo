# -*- coding:utf-8 -*-

"""接近の終盤 (台車が止まる前) に、台車の最終位置を検出した掌に合わせて
ずらすための純粋な計算。

接近の台車の軌道は 1 本のゴールで送っており、止まってから補正すると小さな
後退や回転を挟んで遅い (10/19 実機)。そこで終盤で、残りの軌道を「元の
軌道 + 滑らかに 0 -> 1 へ変わるずれ」で作り直したゴールに差し替える。ずれは
掌の水平方向の誤差で、後退しない・上限を超えない向きに抑える。座標は
odom 系 (台車の指令の積分) と、スキャン照合で求めた実際の系。

odom は指令の積分なので、台車が回っている間は実機より先行する (10/19 実機
の /scan 照合で向きが最大 21° 先行)。その間の掌の検出を odom 系にすると
見かけのずれになるので、回転が収まってからの検出だけを使う。
"""

import math

import numpy as np

# 台車の最終位置をずらす量の上限 [m]。
MAX_SHIFT = 0.12
# 掌が計画からこれ以上ずれていたら誤検出とみなして補正しない [m]。
MAX_PALM_ERROR = 0.25
# ずらす量がこれ未満なら補正しない [m]。
MIN_SHIFT = 0.01
# 作り直した軌道で、元の軌道より進行方向の速度がこれ以上は戻らないこと [m/s]。
REVERSE_TOLERANCE = 0.02
# 差し替えの先頭点と次の点の最小間隔 [s] (近すぎると加速度が跳ねる)。
MIN_POINT_GAP = 0.15


def _unit(v):
    norm = float(np.linalg.norm(v))
    return None if norm < 1e-9 else np.asarray(v, dtype=np.float64) / norm


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


def anchored_displacement(palm_xy, palm_in_base_xy, final_pose,
                          odom_to_actual):
    """実際の台車を、odom の最終位置 ``final_pose`` ``[x, y, yaw]`` に止めた
    ときの実際の位置 (``actual_final_position``) から、どれだけ動かせば掌
    ``palm_xy`` を台車座標系の ``palm_in_base_xy`` に見るか (実際の系の xy、
    台車が実際に動く量) と、そのときの実際の台車の向きを返す。

    実際の台車は ``odom_to_actual`` ∘ (odom の姿勢) にいる (スキャン照合で
    求めたスリップ、この後も変わらないとみなす) ので、odom の最終位置
    ``final_pose`` に止まると向きは ``final_pose[2] + odom_to_actual[2]``。
    スリップは含めない (計画の位置からの差ではなく、実際に動かす量)。
    """
    yaw = float(final_pose[2]) + float(odom_to_actual[2])
    target = palm_anchored_position(palm_xy, palm_in_base_xy, yaw)
    return target - actual_final_position(final_pose, odom_to_actual), yaw


def actual_final_position(final_pose, odom_to_actual):
    """odom の最終位置 ``final_pose`` に止めたときの実際の台車の位置
    (実際の系の xy)。"""
    return transform_point(odom_to_actual, final_pose[:2])


def odom_shift(target_xy, final_pose, odom_to_actual):
    """実際の台車を ``target_xy`` (実際の系) に止めるための、odom の最終位置
    ``final_pose`` のずらし (同じ系の xy)。"""
    return (inverse_transform_point(odom_to_actual, target_xy)
            - np.asarray(final_pose[:2], dtype=np.float64))


def tail_shift(palm_error_xy, final_xy, current_xy, direction_xy,
               max_shift=MAX_SHIFT, max_error=MAX_PALM_ERROR,
               min_shift=MIN_SHIFT):
    """台車の最終位置 ``final_xy`` をずらす量を返す。

    ``palm_error_xy``: 掌に合わせるのに要るずらし (``odom_shift``、odom
    系の水平成分)。``direction_xy`` は今の進行
    方向 (元の軌道の残りの向き)。新しい最終位置が今の位置より進行方向の手前
    になる (= 後退が要る) ずれは、その成分を除く。

    Returns: ``(shift_xy or None, info)``。補正しないときの ``shift_xy`` は
    None で ``info['reason']`` に ``'too_large'``/``'small'``。
    """
    error = np.asarray(palm_error_xy, dtype=np.float64)[:2]
    info = dict(error=float(np.linalg.norm(error)), clamped=False,
                projected=False, reason=None)
    if info['error'] > max_error:
        info['reason'] = 'too_large'
        return None, info
    shift = error.copy()
    norm = float(np.linalg.norm(shift))
    if norm > max_shift:
        shift *= max_shift / norm
        info['clamped'] = True
    direction = _unit(direction_xy)
    if direction is not None:
        ahead = float(np.dot(np.asarray(final_xy, dtype=np.float64)[:2]
                             + shift - np.asarray(current_xy)[:2], direction))
        if ahead < 0.0:
            shift -= ahead * direction
            info['projected'] = True
            norm = float(np.linalg.norm(shift))
            if norm > max_shift:
                # 縮めても (final - current) . d >= 0 なので後退しない。
                shift *= max_shift / norm
    if float(np.linalg.norm(shift)) < min_shift:
        info['reason'] = 'small'
        return None, info
    return shift, info


def build_tail_points(cum_times, positions, velocities, yaws, yaw_rates,
                      elapsed, current_position, current_velocity,
                      current_yaw, current_yaw_rate, shift,
                      min_gap=MIN_POINT_GAP):
    """元の台車軌道の ``elapsed`` 秒以降を、``shift`` だけずらして作り直した
    軌道の点列 ``[(t, 位置(2), 速度(2), yaw, yaw の速度), ...]`` を返す。

    元の軌道は点 k の時刻 ``cum_times[k]`` (先頭 0) とその odom 系の
    ``positions[k]``/``velocities[k]``/``yaws[k]``/``yaw_rates[k]`` (最後の
    速度は 0)。先頭の点は今の状態 (``current_*``)、以降は ``min_gap`` 以上
    先の点を ``shift * smoothstep`` だけずらして並べる (ずれの速度も足すので
    終点の速度は 0 のまま)。yaw はずらさない。残りが ``min_gap`` 未満なら
    None。
    """
    cum_times = np.asarray(cum_times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    t_end = float(cum_times[-1]) - elapsed
    if t_end < min_gap:
        return None
    points = [(0.0, np.asarray(current_position, dtype=np.float64),
               np.asarray(current_velocity, dtype=np.float64),
               float(current_yaw), float(current_yaw_rate))]
    for k in range(len(cum_times)):
        t = float(cum_times[k]) - elapsed
        if t < min_gap:
            continue
        tau = t / t_end
        points.append((
            t,
            positions[k] + shift * smoothstep(tau),
            velocities[k] + shift * smoothstep_rate(tau) / t_end,
            current_yaw + wrap_angle(float(yaws[k]) - current_yaw),
            float(yaw_rates[k])))
    return points


def hermite_velocity_along(points, direction, n_samples=21):
    """点列 (``build_tail_points`` の形) を 3 次エルミート補間したときの、
    ``direction`` 方向の速度の最小値 [m/s]。"""
    direction = np.asarray(direction, dtype=np.float64)[:2]
    lowest = float('inf')
    for (t0, p0, v0, _, _), (t1, p1, v1, _, _) in zip(points[:-1], points[1:]):
        duration = t1 - t0
        if duration <= 0.0:
            continue
        a = (v0 + v1) / duration ** 2 - 2.0 * (p1 - p0) / duration ** 3
        b = 3.0 * (p1 - p0) / duration ** 2 - (2.0 * v0 + v1) / duration
        for t in np.linspace(0.0, duration, n_samples):
            velocity = 3.0 * a * t ** 2 + 2.0 * b * t + v0
            lowest = min(lowest, float(np.dot(velocity, direction)))
    return lowest


def hermite_peak_speed(points, n_samples=21):
    """点列を 3 次エルミート補間したときの並進速度の大きさの最大値 [m/s]。"""
    peak = 0.0
    for (t0, p0, v0, _, _), (t1, p1, v1, _, _) in zip(points[:-1], points[1:]):
        duration = t1 - t0
        if duration <= 0.0:
            continue
        a = (v0 + v1) / duration ** 2 - 2.0 * (p1 - p0) / duration ** 3
        b = 3.0 * (p1 - p0) / duration ** 2 - (2.0 * v0 + v1) / duration
        for t in np.linspace(0.0, duration, n_samples):
            peak = max(peak, float(np.linalg.norm(
                3.0 * a * t ** 2 + 2.0 * b * t + v0)))
    return peak


def monotone_tail(build, shift, direction, scales=(1.0, 0.7, 0.5, 0.3),
                  max_speed=None):
    """``build(shift)`` (``build_tail_points`` の部分適用) の点列のうち、
    進行方向の速度が元の軌道 (``build(0)``) より ``REVERSE_TOLERANCE`` 以上
    戻らず、並進の速さが ``max_speed`` (None なら見ない) を超えないものを、
    ``shift`` を縮めながら探して ``(points, scale)`` で返す (見つからなければ
    ``(None, None)``)。"""
    base = build(np.zeros(2))
    if base is None:
        return None, None
    floor = min(hermite_velocity_along(base, direction), 0.0) \
        - REVERSE_TOLERANCE
    # 元の軌道がすでに上限を超えていても、それ以上は増やさない。
    speed_cap = None if max_speed is None else max(
        max_speed, hermite_peak_speed(base))
    for scale in scales:
        points = build(np.asarray(shift, dtype=np.float64) * scale)
        if points is None:
            continue
        if hermite_velocity_along(points, direction) < floor:
            continue
        if speed_cap is not None and hermite_peak_speed(points) > speed_cap:
            continue
        return points, scale
    return None, None
