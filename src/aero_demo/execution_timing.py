"""waypoint 列を実機で実行するときの区間の時間 (速度上限) と、その補間。

実機ノード (``scripts/ros/run_camera_pipeline_test.py``) が実機へ送る time_list を決め、
表示ツールのシミュレーション表示が同じ時間・同じ補間で計画を動かす。ROS に依存しない。

実機のコントローラは waypoint 間を 3 次エルミートで補間し、各点の速度は
``point_velocities`` の規則 (skrobot の ``angle_vector_sequence`` と同じ) で付く。
区間の時間は、その補間の速度・加速度の最大値が上限 (× ``VEL_LIMIT_RATIO``) に
収まる最短の時間にする (``limited_time_list``)。
"""

import math

import numpy as np


# 実機再生時の waypoint 間の所要時間の下限 [秒] (腕の制御周期 15Hz の 1 周期)。
MIN_SEGMENT_TIME = 1.0 / 15.0

# 台車の速度上限。実機の aero_base_link.yaml の max_velocity と一致させること。
BASE_MAX_VEL = 0.3  # [m/s] 実機の base_link_x/y の max_velocity
BASE_MAX_ANGVEL = 1.0  # [rad/s] 実機の base_link_pan の max_velocity

# 指令で使う速度上限の割合。3 次スプライン瞬間速度の最大がこの割合になるよう
# 区間の時間を決める (limited_time_list)。
VEL_LIMIT_RATIO = 0.9
# 上限速度 (× VEL_LIMIT_RATIO) まで加速するのにかける時間 [秒]。
ACCEL_TIME = 0.4
TIME_LIMIT_MAX_ITERATIONS = 100


def point_velocities(deltas, times):
    """各点の速度 ``(区間数 + 1, 軸数)``。skrobot の ``angle_vector_sequence`` と
    同じ規則 (途中は前後区間の平均速度の平均、符号が逆の軸と始点・終点は 0)。"""
    n_segments = len(times)
    seg_vel = deltas / times[:, None]
    point_vel = np.zeros((n_segments + 1, deltas.shape[1]))
    if n_segments > 1:
        same_sign = deltas[:-1] * deltas[1:] >= 0.0
        point_vel[1:n_segments] = np.where(
            same_sign, 0.5 * (seg_vel[:-1] + seg_vel[1:]), 0.0)
    return point_vel


def _hermite_coefficients(deltas, times, point_vel):
    """各区間の 3 次エルミート ``p(t) = a t^3 + b t^2 + v0 t`` の ``(a, b, v0)``。"""
    seg_times = times[:, None]
    v0 = point_vel[:-1]
    v1 = point_vel[1:]
    a = (v0 + v1) / seg_times ** 2 - 2.0 * deltas / seg_times ** 3
    b = 3.0 * deltas / seg_times ** 2 - (2.0 * v0 + v1) / seg_times
    return a, b, v0


def hermite_peaks(deltas, times, point_vel, norm=False, n_samples=101):
    """各区間を点の速度 ``point_vel`` の 3 次エルミート補間で動かしたときの速度と
    加速度の最大値 (絶対値) を返す。形状は ``deltas`` と同じ、``norm=True`` なら
    ベクトルの大きさで ``(区間数,)`` (このときの速度は ``n_samples`` 点で評価する)。"""
    seg_times = times[:, None]
    a, b, v0 = _hermite_coefficients(deltas, times, point_vel)
    v1 = point_vel[1:]
    acc0 = 2.0 * b
    acc1 = 6.0 * a * seg_times + 2.0 * b
    if norm:
        # t: (区間数, n_samples, 1)、速度: (区間数, n_samples, 軸数)
        t = (np.linspace(0.0, 1.0, n_samples)[None, :]
             * seg_times)[:, :, None]
        vel = (3.0 * a[:, None] * t ** 2 + 2.0 * b[:, None] * t
               + v0[:, None])
        return (np.max(np.linalg.norm(vel, axis=2), axis=1),
                np.maximum(np.linalg.norm(acc0, axis=1),
                           np.linalg.norm(acc1, axis=1)))
    peak = np.maximum(np.abs(v0), np.abs(v1))
    with np.errstate(divide='ignore', invalid='ignore'):
        t_star = -b / (3.0 * a)
        v_star = v0 - b ** 2 / (3.0 * a)
    inside = (a != 0.0) & (t_star > 0.0) & (t_star < seg_times)
    return (np.where(inside, np.maximum(peak, np.abs(v_star)), peak),
            np.maximum(np.abs(acc0), np.abs(acc1)))


def joint_max_velocities(joints, controlled_joint_names=None):
    """関節ごとの速度上限 × ``VEL_LIMIT_RATIO``。上限の無い関節と
    ``controlled_joint_names`` (None なら全関節) に無い関節は判定から外す (inf)。"""
    return np.array([
        joint.max_joint_velocity * VEL_LIMIT_RATIO
        if ((controlled_joint_names is None
             or joint.name in controlled_joint_names)
            and joint.max_joint_velocity > 0)
        else np.inf
        for joint in joints])


def base_trajectory_points(waypoints):
    """waypoint 列の台車の、先頭 waypoint 基準の累積 ``[dx, dy, dyaw]`` 列
    (先頭の向きの系。``move_trajectory_sequence`` に送る形)。"""
    points = []
    first = None
    for wp in waypoints:
        bx, by, byaw = (wp['base_position'][0], wp['base_position'][1],
                        wp['base_yaw'])
        if first is None:
            first = (bx, by, byaw)
        x0, y0, yaw0 = first
        dx_world = bx - x0
        dy_world = by - y0
        cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)
        points.append([cos_yaw * dx_world + sin_yaw * dy_world,
                       -sin_yaw * dx_world + cos_yaw * dy_world,
                       byaw - yaw0])
    return points


def limited_time_list(arm_deltas, arm_max_vel, base_points, time_list,
                      log=None):
    """区間ごとの所要時間を、律速する軸の指令の速度・加速度の最大値がちょうど
    上限 (× ``VEL_LIMIT_RATIO``、加速度は ``ACCEL_TIME``) になるよう決めて返す。
    ``time_list`` は区間ごとの下限。

    ``arm_deltas``: 区間ごとの関節角の差 ``(区間数, 関節数)``、``arm_max_vel``:
    関節ごとの上限 (``joint_max_velocities``)。``base_points``: 区間先頭からの
    累積 ``[dx, dy, dyaw]`` 列 (``base_trajectory_points``)。どちらも None/空なら
    判定から外す。``log``: 上限内に収まらず一律に延ばしたときに文を渡す関数。
    """
    base_deltas = None
    if base_points is not None and len(base_points):
        points = np.vstack([np.zeros(3),
                            np.asarray(base_points, dtype=np.float64)])
        base_deltas = np.diff(points, axis=0)
    if arm_deltas is not None and not len(arm_deltas):
        arm_deltas = None
    if arm_deltas is None and base_deltas is None:
        return [float(t) for t in time_list]
    if arm_deltas is not None:
        # 型は呼び出し側のまま (実機ノードの従来の計算と同じ結果にするため)。
        arm_deltas = np.asarray(arm_deltas)
        arm_max_vel = np.asarray(arm_max_vel)

    def peak_ratios(times):
        # 区間・軸ごとの 速度/上限 と sqrt(加速度/上限)。
        vel_ratios = []
        acc_ratios = []
        if arm_deltas is not None:
            point_vel = point_velocities(arm_deltas, times)
            vel, acc = hermite_peaks(arm_deltas, times, point_vel)
            vel_ratios.append(vel / arm_max_vel)
            acc_ratios.append(acc / (arm_max_vel / ACCEL_TIME))
        if base_deltas is not None:
            point_vel = point_velocities(base_deltas, times)
            # 並進は向きによらず超えないようベクトルの大きさで判定する。
            vel_xy, acc_xy = hermite_peaks(
                base_deltas[:, :2], times, point_vel[:, :2], norm=True)
            vel_yaw, acc_yaw = hermite_peaks(
                base_deltas[:, 2:], times, point_vel[:, 2:])
            vel_limit = np.array([BASE_MAX_VEL, BASE_MAX_ANGVEL]) \
                * VEL_LIMIT_RATIO
            vel_ratios.append(
                np.stack([vel_xy, vel_yaw[:, 0]], axis=1) / vel_limit)
            acc_ratios.append(
                np.stack([acc_xy, acc_yaw[:, 0]], axis=1)
                / (vel_limit / ACCEL_TIME))
        return np.concatenate(
            vel_ratios + [np.sqrt(r) for r in acc_ratios], axis=1)

    min_times = np.array(time_list, dtype=np.float64)
    times = min_times.copy()
    for _ in range(TIME_LIMIT_MAX_ITERATIONS):
        # ratio 倍そのままだと振動するので平方根で半分だけ動かす。
        ratio = np.max(peak_ratios(times), axis=1)
        new_times = np.maximum(min_times, times * np.sqrt(ratio))
        if np.allclose(new_times, times, rtol=1e-4, atol=0.0):
            times = new_times
            break
        times = new_times
    # 残った超過は全区間を一律に延ばして必ず上限内にする。
    worst = float(np.max(peak_ratios(times)))
    if worst > 1.0:
        if worst > 1.01 and log is not None:
            log('[debug][segment] 速度上限の反復で収まらなかったため '
                '全区間を {:.3f} 倍に延ばします。'.format(worst))
        times *= worst
    return [float(t) for t in times]


class SegmentMotion(object):
    """1 ゴールで送る waypoint 列 (先頭が動き始めの姿勢) を、実機と同じ時間・
    同じ 3 次エルミート補間で動かしたときの姿勢を時刻で引けるようにしたもの。

    ``joint_names`` の関節角と台車 (world 系の x, y, yaw) を補間する。台車は
    実機と同じく先頭 waypoint の向きの系の累積移動量で補間してから world 系に戻す。
    """

    def __init__(self, waypoints, joint_names, joints_by_name,
                 min_segment_time=MIN_SEGMENT_TIME):
        """``joints_by_name``: 関節名 -> skrobot の関節 (速度上限に使う)。"""
        self.joint_names = list(joint_names)
        angles = np.array([wp['joint_angle_vector'] for wp in waypoints],
                          dtype=np.float64)
        self._angles = angles
        first = waypoints[0]
        self._base0 = (float(first['base_position'][0]),
                       float(first['base_position'][1]),
                       float(first['base_yaw']))
        # 先頭 waypoint を原点にした累積移動量 (先頭は 0)。
        self._base_points = np.asarray(
            base_trajectory_points(waypoints), dtype=np.float64)
        n_segments = len(waypoints) - 1
        if n_segments <= 0:
            self.times = np.zeros(0)
            self.duration = 0.0
            return
        arm_max_vel = joint_max_velocities(
            [joints_by_name[name] for name in self.joint_names])
        self.times = np.asarray(limited_time_list(
            np.diff(angles, axis=0), arm_max_vel, self._base_points[1:],
            [min_segment_time] * n_segments), dtype=np.float64)
        self.duration = float(np.sum(self.times))
        self._starts = np.concatenate([[0.0], np.cumsum(self.times)[:-1]])
        self._arm = _hermite_coefficients(
            np.diff(angles, axis=0), self.times,
            point_velocities(np.diff(angles, axis=0), self.times))
        base_deltas = np.diff(self._base_points, axis=0)
        self._base = _hermite_coefficients(
            base_deltas, self.times,
            point_velocities(base_deltas, self.times))

    def waypoint_time(self, index):
        """``index`` 番目の waypoint に着く時刻 [秒] (先頭が 0)。"""
        return float(np.sum(self.times[:index]))

    def sample(self, t):
        """時刻 ``t`` [秒] の ``(関節角 (joint_names の順), (x, y, yaw), 区間の
        番号 + 区間内の割合)`` を返す (範囲外は端の姿勢)。"""
        if len(self.times) == 0 or t <= 0.0:
            return self._angles[0].copy(), self._base_world(
                self._base_points[0]), 0.0
        if t >= self.duration:
            return self._angles[-1].copy(), self._base_world(
                self._base_points[-1]), float(len(self.times))
        k = int(np.searchsorted(self._starts, t, side='right') - 1)
        tau = t - self._starts[k]
        a, b, v0 = self._arm
        angles = (self._angles[k] + a[k] * tau ** 3 + b[k] * tau ** 2
                  + v0[k] * tau)
        a, b, v0 = self._base
        base = (self._base_points[k] + a[k] * tau ** 3 + b[k] * tau ** 2
                + v0[k] * tau)
        return angles, self._base_world(base), k + tau / self.times[k]

    def _base_world(self, point):
        x0, y0, yaw0 = self._base0
        dx, dy, dyaw = point
        cos_yaw, sin_yaw = math.cos(yaw0), math.sin(yaw0)
        return (x0 + cos_yaw * dx - sin_yaw * dy,
                y0 + sin_yaw * dx + cos_yaw * dy,
                yaw0 + dyaw)
