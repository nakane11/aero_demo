#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""押し込み後の姿勢 (``solve_post_process`` の手先の位置姿勢) を jaxls の最小二乗で解く。

``--press-solver jaxls`` 用。skrobot のヤコビアン IK (収束するまで、閾値 1 cm・5°) の代わりに、
hover の jaxls (``hover_goal_ik``) と同じ手先の誤差項を最小化する:

- 手先の目標 (押し込み後の掌の位置と ``target_rot``) の誤差は ``hover_goal_ik.hand_error_residual``
  (法線方向の位置 / ``HOVER_GOAL_NORMAL_EPS``、それ以外の位置 / 位置の閾値、姿勢 / 姿勢の閾値、
  重み ``HOVER_GOAL_POSE_WEIGHT``)。成功判定は |e| <= eps (事後に skrobot の FK で測る)。
- 変数は腕・腰・脚 (ミミック関節は親に従う)。台車は動かさない (呼び出し時の位置に固定)。
- 冗長な自由度は、初期姿勢からの関節角の変化を小さい重みの誤差項 (e / eps、eps はロボットの
  可動域の上限) にして決める。ヤコビアン IK が初期姿勢に近い解になる性質の代わり。
- 制約は関節可動域だけ (ロボットの現在の可動域。腰を上げない側・脚の幅の制限もこれで表す)。
  人体距離・自己干渉は入れない (事後検証で見る)。

初期値は 1 つ (呼び出し時の姿勢)。人物ごとの値は全て ParamVar なので再コンパイルしない。
"""

import math
import os
import time

import numpy as np

import hover_goal_ik as hgi
import solve_palm_ik as spik

# 初期姿勢からの関節角の変化 (関節ごとに e / eps、eps は可動域の上限) の重み。手先の誤差の
# 重み (HOVER_GOAL_POSE_WEIGHT) より十分小さくして、手先の誤差を邪魔しないようにする。
PRESS_GOAL_REDUNDANCY_WEIGHT = 0.1
# jaxls の反復回数の上限。
DEFAULT_PRESS_GOAL_MAX_ITERATIONS = 100
# jaxls の終了判定 (コストの相対変化・勾配・パラメータの変化) の閾値。既定 (1e-5 等) だと
# 手先が目標から数 mm〜14 mm 手前で止まることがある (冗長性の重みで非単調)。
TERMINATION_TOLERANCE = 1e-9
# 成功の判定: 関節可動域の超過 [rad]。
LIMIT_TOLERANCE = 1e-3
# 脚を固定するとき、可動域をいまの角度の周りにこの幅 [rad] まで狭める。
FIXED_LEG_HALF_RANGE = 1e-5

# PressVar の並び: 台車 (x, y, yaw: 構築時の root からの相対), eps_pos, eps_rot, eps_normal,
# sqrt(w_pose), sqrt(w_redundancy), 下限 nj, 上限 nj, 初期姿勢 nj
_PRESS_FIXED_DIM = 8


class PressGoalSolver(object):
    """1 本の腕の押し込み後の姿勢を jaxls で解くソルバー (腕ごとに 1 つ)。"""

    def __init__(self, robot, robot_arm,
                 max_iterations=DEFAULT_PRESS_GOAL_MAX_ITERATIONS):
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        import jaxls
        from skrobot.planner.trajectory_optimization.fk_utils import \
            pose_error_log
        from skrobot.planner.trajectory_optimization.problem import \
            TrajectoryProblem
        from skrobot.planner.trajectory_optimization.solvers import \
            jaxls_solver as js

        self.robot = robot
        self.robot_arm = robot_arm
        self.max_iterations = max_iterations
        whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
        self.link_list = whole_body.link_list
        self.move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
        # 構築時の姿勢は ee_fk の定数 (連鎖の親変換) に焼き込まれるので、元に戻す。
        saved = robot.angle_vector().copy()
        saved_coords = robot.copy_worldcoords()
        try:
            problem = TrajectoryProblem(
                robot, self.link_list, n_waypoints=1,
                move_target=self.move_target, n_base_dof=3)
        finally:
            robot.newcoords(saved_coords)
            robot.angle_vector(saved)
        nj = problem.n_joints
        self.nj = nj
        root_pos_np, root_rot_np = js._root_link_world_pose(problem)
        self.root_xy = np.asarray(root_pos_np, dtype=np.float64)[:2]
        self.root_yaw = float(math.atan2(root_rot_np[1][0], root_rot_np[0][0]))
        root_pos = jnp.asarray(root_pos_np)
        root_rot = jnp.asarray(root_rot_np)
        ee_fk = js._build_root_relative_chain_ee_fks(problem, jnp)[0]

        fk_params = problem.fk_params
        mimic_parent = np.asarray(fk_params.get(
            'mimic_parent_indices', -np.ones(nj, dtype=int)))
        self.mimic_rows = [i for i in range(nj) if mimic_parent[i] >= 0]
        self.mimic_parent = [int(mimic_parent[i]) for i in self.mimic_rows]
        self.mimic_mult = [float(fk_params['mimic_multipliers'][i])
                           for i in self.mimic_rows]
        self.mimic_offset = [float(fk_params['mimic_offsets'][i])
                             for i in self.mimic_rows]
        fixed_rows = [i for i in range(nj) if i not in self.mimic_rows
                      and problem.joint_limits_lower[i]
                      >= problem.joint_limits_upper[i]]
        free = [i for i in range(nj)
                if i not in self.mimic_rows and i not in fixed_rows]
        expand = np.zeros((nj, len(free)))
        for column, i in enumerate(free):
            expand[i, column] = 1.0
        for row, parent, mult in zip(self.mimic_rows, self.mimic_parent,
                                     self.mimic_mult):
            expand[row, free.index(parent)] = mult
        expand = jnp.asarray(expand)
        # 初期姿勢からの変化の誤差項の eps (ロボットの可動域の上限) と対象の関節 (自由な関節だけ)。
        full_lower = problem.joint_limits_lower
        full_upper = problem.joint_limits_upper
        self.redundancy_eps = np.maximum(
            np.maximum(np.abs(full_lower), np.abs(full_upper)), 1e-3)
        redundancy_mask = np.zeros(nj)
        redundancy_mask[free] = 1.0
        redundancy_eps = jnp.asarray(self.redundancy_eps)
        redundancy_mask = jnp.asarray(redundancy_mask)

        class PoseVar(jaxls.Var[jnp.ndarray],
                      default_factory=lambda: jnp.zeros(nj),
                      retract_fn=lambda x, delta: x + expand @ delta,
                      tangent_dim=len(free)):
            pass

        TargetVar = hgi._param_var(jnp.zeros(12))
        PressVar = hgi._param_var(jnp.zeros(_PRESS_FIXED_DIM + 3 * nj))
        zero = jnp.array([0])

        def base_pose(p):
            return js._floating_base_pose(p[0:3], 3, root_pos, root_rot, jnp)

        def ee_position(q, p):
            base_pos, base_rot = base_pose(p)
            return ee_fk(q, base_pos, base_rot)

        costs = []

        @jaxls.Cost.factory(name='press_pose')
        def pose_cost(vals, var, target, press):
            p = vals[press]
            tgt = vals[target]
            ee_pos, ee_rot = ee_position(vals[var], p)
            rot_error = pose_error_log(ee_pos, ee_rot, tgt[:3],
                                       tgt[3:12].reshape(3, 3))[3:]
            return hgi.hand_error_residual(
                jnp, ee_pos, rot_error, tgt[:3], tgt[3:12].reshape(3, 3),
                p[6], p[3], p[4], p[5])
        costs.append(pose_cost(PoseVar(zero), TargetVar(zero),
                               PressVar(zero)))

        @jaxls.Cost.factory(name='press_redundancy')
        def redundancy_cost(vals, var, press):
            p = vals[press]
            q_ref = p[_PRESS_FIXED_DIM + 2 * nj:]
            return (p[7] * redundancy_mask * (vals[var] - q_ref)
                    / redundancy_eps)
        costs.append(redundancy_cost(PoseVar(zero), PressVar(zero)))

        @jaxls.Cost.factory(kind='constraint_geq_zero', name='press_limits')
        def limit_cost(vals, var, press):
            p = vals[press]
            lower = p[_PRESS_FIXED_DIM:_PRESS_FIXED_DIM + nj]
            upper = p[_PRESS_FIXED_DIM + nj:_PRESS_FIXED_DIM + 2 * nj]
            q = vals[var]
            return jnp.concatenate([q - lower, upper - q])
        costs.append(limit_cost(PoseVar(zero), PressVar(zero)))

        js._pin_cost_group_order(costs)
        ls_problem = jaxls.LeastSquaresProblem(
            costs=costs,
            variables=[PoseVar(zero), TargetVar(zero), PressVar(zero)]
        ).analyze()

        def solve_one(q0, target, press):
            init = jaxls.VarValues.make([
                PoseVar(zero).with_value(q0[None]),
                TargetVar(zero).with_value(target[None]),
                PressVar(zero).with_value(press[None])])
            solution = ls_problem.solve(
                initial_vals=init, verbose=bool(os.environ.get('PRESS_VERBOSE')),
                termination=jaxls.TerminationConfig(
                    max_iterations=max_iterations,
                    cost_tolerance=TERMINATION_TOLERANCE,
                    gradient_tolerance=TERMINATION_TOLERANCE,
                    parameter_tolerance=TERMINATION_TOLERANCE))
            return solution[PoseVar(zero)][0]

        self._solve = jax.jit(solve_one)
        self._ee_position = jax.jit(ee_position)

    def compile_count(self):
        return self._solve._cache_size()

    def _chain_angles(self):
        """ロボットの現在の連鎖の関節角。ミミックは親から合わせる。"""
        q = np.array([float(link.joint.joint_angle())
                      for link in self.link_list])
        for row, parent, mult, offset in zip(
                self.mimic_rows, self.mimic_parent, self.mimic_mult,
                self.mimic_offset):
            q[row] = mult * q[parent] + offset
        return q

    def _limits(self, fix_legs):
        """ロボットの現在の関節可動域 (腰を上げない側・脚の幅の制限を含む)。ミミックは見ない。"""
        lower = np.array([float(link.joint.min_angle)
                          for link in self.link_list])
        upper = np.array([float(link.joint.max_angle)
                          for link in self.link_list])
        lower[self.mimic_rows] = -1e6
        upper[self.mimic_rows] = 1e6
        if fix_legs:
            q = self._chain_angles()
            for column, link in enumerate(self.link_list):
                if link.joint.name in spik.LEG_JOINT_NAMES \
                        and column not in self.mimic_rows:
                    lower[column] = q[column] - FIXED_LEG_HALF_RANGE
                    upper[column] = q[column] + FIXED_LEG_HALF_RANGE
        return lower, upper

    def base_param(self):
        """台車の現在の姿勢 (構築時の root からの相対の x, y, yaw)。"""
        base = self.robot.base_link
        yaw, _, _ = spik.matrix2ypr(base.worldrot())
        xy = np.asarray(base.worldpos(), dtype=np.float64)[:2] - self.root_xy
        return np.array([xy[0], xy[1], float(yaw) - self.root_yaw])

    def solve(self, target_pos, target_rot, pose_weight, eps, redundancy_weight,
              fix_legs=False):
        """ロボットの現在の姿勢・台車から ``target_pos``/``target_rot`` へ手先を解き、
        関節角を ``robot`` に反映する (``sync_leg_mimics`` 済み)。

        ``eps`` = (位置の接線, 姿勢 [rad], 法線方向の位置)。解の誤差
        ``dict(normal, tangent, rotation, limit_excess)`` を返す (skrobot の FK で測る)。
        """
        import jax.numpy as jnp
        q_ref = self._chain_angles()
        lower, upper = self._limits(fix_legs)
        press = np.concatenate([
            self.base_param(),
            [eps[0], eps[1], eps[2], math.sqrt(pose_weight),
             math.sqrt(redundancy_weight)],
            lower, upper, q_ref])
        target = np.concatenate([
            np.asarray(target_pos, dtype=np.float64),
            np.asarray(target_rot, dtype=np.float64).ravel()])
        q = np.asarray(self._solve(
            jnp.asarray(q_ref), jnp.asarray(target), jnp.asarray(press)))
        for column, link in enumerate(self.link_list):
            if column in self.mimic_rows:
                continue
            link.joint.joint_angle(float(q[column]))
        spik.sync_leg_mimics(self.robot)
        if fix_legs:
            # 狭めた幅の分の変化も無くす (手先の誤差は下で測り直す)。
            for column, link in enumerate(self.link_list):
                if link.joint.name in spik.LEG_JOINT_NAMES \
                        and column not in self.mimic_rows:
                    link.joint.joint_angle(float(q_ref[column]))
            spik.sync_leg_mimics(self.robot)
        error = hand_errors(self.move_target, target_pos, target_rot)
        angles = np.array([float(link.joint.joint_angle())
                           for link in self.link_list])
        rows = [i for i in range(self.nj) if i not in self.mimic_rows]
        error['limit_excess'] = float(np.max(
            np.maximum(lower - angles, angles - upper)[rows]))
        return error

    def check_fk(self, tolerance=1e-4):
        """jax の FK が skrobot の FK と手先で一致するか確かめる (違えば ValueError)。"""
        import jax.numpy as jnp
        q = self._chain_angles()
        pos, _ = self._ee_position(jnp.asarray(q),
                                   jnp.asarray(self.base_param()))
        diff = float(np.linalg.norm(
            np.asarray(pos) - np.asarray(self.move_target.worldpos())))
        if diff > tolerance:
            raise ValueError('押し込みの jaxls の FK が skrobot と合いません '
                             '(手先 {:.6f} m)。'.format(diff))
        return diff


def hand_errors(move_target, target_pos, target_rot):
    """手先の目標からの誤差 dict(normal [m]、tangent [m]、rotation [rad]) (どれも大きさ)。"""
    target_rot = np.asarray(target_rot, dtype=np.float64)
    d = (np.asarray(move_target.worldpos(), dtype=np.float64)
         - np.asarray(target_pos, dtype=np.float64))
    normal = target_rot[:, 1]
    d_normal = float(np.dot(d, normal))
    tangent = float(np.linalg.norm(d - d_normal * normal))
    relative = np.asarray(move_target.worldrot()).T @ target_rot
    rotation = float(math.acos(np.clip((np.trace(relative) - 1.0) / 2.0,
                                       -1.0, 1.0)))
    return dict(normal=abs(d_normal), tangent=tangent, rotation=rotation)


_SOLVERS = {}


def get_solver(robot, robot_arm,
               max_iterations=DEFAULT_PRESS_GOAL_MAX_ITERATIONS):
    key = (id(robot), robot_arm, max_iterations)
    solver = _SOLVERS.get(key)
    if solver is None:
        solver = PressGoalSolver(robot, robot_arm,
                                 max_iterations=max_iterations)
        _SOLVERS[key] = solver
    return solver


def solve_press_goal(robot, robot_arm, target_pos, target_rot, fix_legs=False,
                     pose_weight=hgi.HOVER_GOAL_POSE_WEIGHT,
                     eps=(spik.DEFAULT_COLLISION_IK_THRE,
                          spik.DEFAULT_COLLISION_IK_RTHRE,
                          hgi.HOVER_GOAL_NORMAL_EPS),
                     redundancy_weight=PRESS_GOAL_REDUNDANCY_WEIGHT):
    """``robot`` の現在の姿勢から押し込み後の手先目標を jaxls で解く。手先の誤差が全て ``eps``
    以内で関節可動域に収まれば誤差 dict、そうでなければ ``robot`` を元に戻して None。"""
    initial = robot.angle_vector().copy()
    solver = get_solver(robot, robot_arm)
    error = solver.solve(target_pos, target_rot, pose_weight, eps,
                         redundancy_weight, fix_legs=fix_legs)
    ok = (error['tangent'] <= eps[0] and error['rotation'] <= eps[1]
          and error['normal'] <= eps[2]
          and error['limit_excess'] <= LIMIT_TOLERANCE)
    if not ok:
        robot.angle_vector(initial)
        spik.sync_leg_mimics(robot)
        return None
    return error


def warmup_press_goal_ik(robot):
    """左右の腕のソルバーを作ってコンパイルを済ませる (ダミー目標、結果は捨てる)。"""
    for robot_arm in ('l', 'r'):
        t0 = time.time()
        initial = robot.angle_vector().copy()
        coords = robot.copy_worldcoords()
        move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
        pos = np.asarray(move_target.worldpos()) + np.array([0.0, 0.0, 0.01])
        rot = np.asarray(move_target.worldrot())
        solve_press_goal(robot, robot_arm, pos, rot)
        robot.newcoords(coords)
        robot.angle_vector(initial)
        print('[warmup] {}腕: jaxls の押し込みのトレース/コンパイル {:.1f} 秒'
              .format(robot_arm, time.time() - t0))
