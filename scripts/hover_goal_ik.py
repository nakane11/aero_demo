#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""hover 姿勢 (``solve_person_ik`` と同じ目標) を jaxls の最小二乗で解く。

``solve_person_ik`` (skrobot のバッチ IK) は台車の置き方を x の窓
(``solve_person_ik_side_by_side``)・yaw の窓 (``restrict_base_yaw_range_to_human_facing``)
と候補の並べ替えのコスト (``pick_verified_candidate`` の front/yaw の重み) で決めていた。
ここではそれを最適化の中の目標の誤差項にする:

- 制約 (jaxls の拡張ラグランジュ): 関節可動域 (余裕なし、``HOVER_GOAL_JOINT_LIMIT_MARGIN_RATIO``)、
  人体との距離 (バッチ IK と同じ組・差し出した前腕の細らせ・人の上腕、
  ``DEFAULT_COLLISION_MARGIN`` 以上)。台車 (x, y, yaw) の範囲は制約にしない
  (初期値の x, y の範囲にだけ使う。置き方は下の誤差項で決める)。
- 手先の目標 (hover 目標) は重み ``POSE_WEIGHT`` の二乗誤差 (等式制約より収束が良かった)。
- 目標の誤差項 (重み付き二乗和を最小化): 台車の前方ずれ (押し込み後の台車、
  ``pick_verified_candidate`` と同じ定義) と向きのずれ。どちらも許容幅 eps で正規化した
  ``e / eps``。IK の手先の収束閾値と同じく、``|e| <= eps`` を満たさない解は収束していない
  (成功にしない) 扱い。

初期値ごとに独立な 1 姿勢の問題を ``jax.vmap`` でまとめて解く。人物ごとの値
(目標・立ち位置・正面方向・台車の可動域・障害物・eps・重み) は凍結した ParamVar
なので、人物が変わっても再コンパイルしない。

関節の曲げ・腰の沈み込み・前傾は、今の段階では最適化に入れず
``pick_verified_candidate`` の並べ替えのまま (front/yaw の重みだけ 0 にする)。
"""

import math
import time

import numpy as np

import solve_palm_ik as spik

# 台車の前方ずれ (押し込み後の台車、人の正面方向) の許容幅 [m] と重み。許容幅は誤差項の
# 正規化と成否の判定 (|e| <= eps) の両方に使う。
HOVER_GOAL_FRONT_EPS = 0.30
HOVER_GOAL_FRONT_WEIGHT = 1.0
# 台車の向きのずれ (人の正面方向から) の許容幅 [rad] と重み。
HOVER_GOAL_YAW_EPS = math.radians(15.0)
HOVER_GOAL_YAW_WEIGHT = 1.0

# 手先の hover 目標の二乗誤差の重み ([m]・[rad] の誤差に掛かる)。
POSE_WEIGHT = 1.0e6

# hover の jaxls の関節可動域の余裕 (全域幅に対する比率)。0 = 可動域そのものが制約。
HOVER_GOAL_JOINT_LIMIT_MARGIN_RATIO = 0.0

# 1 つの向きあたりの初期値の数 (向き 3 つで 3 倍の問題を 1 回で解く)。
DEFAULT_HOVER_GOAL_ATTEMPTS_PER_TURN = 128
# jaxls の反復回数の上限。
DEFAULT_HOVER_GOAL_MAX_ITERATIONS = 100

# 成功の判定: 手先 (バッチ IK と同じ)、関節・台車の可動域の超過 [rad/m]、
# 人体との距離の不足 (DEFAULT_COLLISION_MARGIN からの) [m]。
LIMIT_TOLERANCE = 1e-3
COLLISION_TOLERANCE = 0.005

# 台車の初期値の分布。押し込み後の台車の前方ずれを INIT_FRONT_RANGE から
# (INIT_WIDE_FRACTION の割合は INIT_FRONT_WIDE_RANGE から)、向きを人の正面方向
# ±INIT_YAW_SPREAD から、手先の目標を台車座標の横 INIT_TARGET_LATERAL_RANGE
# (差し出す腕の側) に置くように選ぶ。横の範囲は現行のバッチ IK の解
# (合成 67 人の 10〜90%: 0.47〜0.68 m) を含むように決めた。
INIT_FRONT_RANGE = (-0.25, 0.25)
INIT_FRONT_WIDE_RANGE = (-0.4, 0.8)
INIT_WIDE_FRACTION = 0.25
INIT_YAW_SPREAD = math.radians(25.0)
INIT_TARGET_LATERAL_RANGE = (0.4, 0.75)
# 人の立ち位置・向きが無いときの、台車座標での手先の目標の前後の範囲 [m]。
INIT_TARGET_FORWARD_RANGE = (0.0, 0.5)

# 人の上腕 (肩-肘) との距離の制約を足すロボットの差し出す腕のリンク ({} は腕)。
# バッチ IK の組 (collision_pairs.json) には上腕が無く、hover の事後検証で差し出した側の
# 上腕にめり込む解が多かった (肘・前腕・手首・手。反対側の上腕・肩より上は 5 cm 以内に
# 来なかった、2026-11-02 合成 6 人)。
UPPER_ARM_PAIR_LINKS = ('{}_elbow_link', '{}_forearm_link', '{}_hand_yaw_link',
                        '{}_hand_link', '{}_thumb_box_link')
# 上腕の制約を掛ける人の側: 'none'、'offered' (差し出した側)、'both'。
UPPER_ARM_PAIR_SIDES = ('none', 'offered', 'both')
DEFAULT_UPPER_ARM_PAIRS = 'offered'

# PersonVar の並び: 立ち位置 xy, 正面方向 xy, 台車の下限 xyz, 上限 xyz,
# eps_front, eps_yaw, sqrt(w_front), sqrt(w_yaw)
_PERSON_DIM = 14
# TargetVar の並び: 目標位置 3, 目標回転 9, 押し込みでの台車のずれ 3
_TARGET_DIM = 15


def _param_var(default):
    """値を最適化しない (tangent_dim=0) jaxls の変数型を作る。"""
    import jax.numpy as jnp
    import jaxls

    class ParamVar(jaxls.Var[jnp.ndarray], default_factory=lambda: default,
                   retract_fn=lambda x, delta: x, tangent_dim=0):
        pass
    return ParamVar


def _obstacle_dicts(obstacles):
    """``Cylinder`` のリストを skrobot の軌道最適化の障害物 dict にする。"""
    out = []
    for cylinder in obstacles:
        out.append(dict(
            type='cylinder',
            center=[float(v) for v in cylinder.worldpos()],
            rotation=np.asarray(cylinder.worldrot()).tolist(),
            radius=float(cylinder.radius),
            half_height=float(cylinder.height) / 2.0))
    return out


class HoverGoalSolver(object):
    """1 本の腕の hover 姿勢を jaxls で解くソルバー (腕・初期値の数ごとに 1 つ)。

    構造 (腕・初期値の数・干渉の組・関節可動域) だけで決まり、人物ごとの値は
    ``person_inputs`` が作る配列で渡す (同じ形なので再コンパイルしない)。
    """

    def __init__(self, robot, robot_arm, collision_pairs, n_instances,
                 collision_margin=spik.DEFAULT_COLLISION_MARGIN,
                 joint_limit_margin_ratio=(
                     spik.DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO),
                 max_iterations=DEFAULT_HOVER_GOAL_MAX_ITERATIONS,
                 pose_weight=POSE_WEIGHT,
                 upper_arm_pairs=DEFAULT_UPPER_ARM_PAIRS):
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        import jaxls
        from skrobot.planner.trajectory_optimization.fk_utils import \
            pose_error_log
        from skrobot.planner.trajectory_optimization.fk_utils import \
            prepare_fk_data
        from skrobot.planner.trajectory_optimization.problem import \
            TrajectoryProblem
        from skrobot.planner.trajectory_optimization.solvers import \
            jaxls_solver as js

        self.robot = robot
        self.robot_arm = robot_arm
        self.n_instances = n_instances
        self.max_iterations = max_iterations
        hand = {arm: h for h, arm in spik.DEFAULT_ROBOT_ARM.items()}[robot_arm]
        # 連鎖外のリンク (差し出さない腕・頭) と台車の原点はバッチ IK と同じ種の姿勢。
        spik.seed_arm_pose(robot, robot_arm)
        whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
        self.link_list = whole_body.link_list
        self.seed_chain_angles = np.array(
            [link.joint.joint_angle() for link in self.link_list])
        self.seed_angle_vector = robot.angle_vector().copy()
        move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
        restore = spik.restrict_joint_range_margin(
            self.link_list, joint_limit_margin_ratio)
        try:
            problem = TrajectoryProblem(robot, self.link_list, n_waypoints=1,
                                        move_target=move_target, n_base_dof=3)
        finally:
            restore()
        self.problem = problem
        nj = problem.n_joints
        self.nj = nj
        ndof = nj + 3
        self.ndof = ndof

        # 人体との組はバッチ IK と同じ (collision_pairs + 差し出した前腕 x 手先側のリンク)。
        self.forearm = spik.human_obstacle_names().index(
            '{0}Elbow-{0}Wrist'.format(hand))
        links_by_name = {link.name: link for link in
                         list(robot.link_list)
                         + list(getattr(robot, 'extra_collision_links', []))}
        pairs = [(link, other) for link, other in collision_pairs
                 if isinstance(other, int)]
        if len(pairs) != len(collision_pairs):
            print('[hover-goal] 自己干渉の組 {} 組は jaxls の hover では見ません '
                  '(事後検証で見ます)。'.format(len(collision_pairs) - len(pairs)))
        pairs += [(links_by_name[name.format(robot_arm)], self.forearm)
                  for name in spik.OFFERED_HAND_PENALTY_LINKS
                  if name.format(robot_arm) in links_by_name]
        if upper_arm_pairs not in UPPER_ARM_PAIR_SIDES:
            raise ValueError('upper_arm_pairs は {} のどれかです: {}'.format(
                UPPER_ARM_PAIR_SIDES, upper_arm_pairs))
        upper_arm_hands = {'none': (), 'offered': (hand,),
                           'both': ('R', 'L')}[upper_arm_pairs]
        for side in upper_arm_hands:
            upper_arm = spik.human_obstacle_names().index(
                '{0}Shoulder-{0}Elbow'.format(side))
            pairs += [(links_by_name[name.format(robot_arm)], upper_arm)
                      for name in UPPER_ARM_PAIR_LINKS
                      if name.format(robot_arm) in links_by_name]
        collision_links = []
        for link, _ in pairs:
            if link not in collision_links:
                collision_links.append(link)
        obstacle_pairs = [(collision_links.index(link), other)
                          for link, other in pairs]
        dummy_obstacles = spik.human_body_obstacles({})
        self.n_obstacles = len(dummy_obstacles)
        self.collision_margin = collision_margin
        problem.add_collision_cost(
            collision_links, _obstacle_dicts(dummy_obstacles), weight=1.0,
            activation_distance=collision_margin, as_constraint=True,
            obstacle_pairs=obstacle_pairs)
        collision_spec = problem.residuals[-1]

        fk_data = js._quantize_fk_constants(prepare_fk_data(problem, jnp), jnp)
        root_pos_np, root_rot_np = js._root_link_world_pose(problem)
        root_pos = jnp.asarray(root_pos_np)
        root_rot = jnp.asarray(root_rot_np)
        ee_fk = js._build_root_relative_chain_ee_fks(problem, jnp)[0]

        # ミミック関節 (脚) は親の関節に従わせる: 接空間は親だけで、親を動かすと
        # ミミック関節も倍率をかけて動く (retract)。可動域は親だけで見る (バッチ IK と同じ)。
        fk_params = problem.fk_params
        mimic_parent = np.asarray(fk_params.get(
            'mimic_parent_indices', -np.ones(nj, dtype=int)))
        self.mimic_rows = [i for i in range(nj) if mimic_parent[i] >= 0]
        self.mimic_parent = [int(mimic_parent[i]) for i in self.mimic_rows]
        self.mimic_mult = [float(fk_params['mimic_multipliers'][i])
                           for i in self.mimic_rows]
        self.mimic_offset = [float(fk_params['mimic_offsets'][i])
                             for i in self.mimic_rows]
        lower = problem.joint_limits_lower.copy()
        upper = problem.joint_limits_upper.copy()
        lower[self.mimic_rows] = -1e6
        upper[self.mimic_rows] = 1e6
        self.joint_lower = lower
        self.joint_upper = upper
        joint_lower = jnp.asarray(lower)
        joint_upper = jnp.asarray(upper)

        # 可動域の幅が 0 の関節 (lock_fixed_joints の手首ヨー) も接空間から外し、値を動かさない。
        fixed_rows = [i for i in range(nj) if i not in self.mimic_rows
                      and problem.joint_limits_lower[i]
                      >= problem.joint_limits_upper[i]]
        free = [i for i in range(ndof)
                if i not in self.mimic_rows and i not in fixed_rows]
        expand = np.zeros((ndof, len(free)))
        for column, i in enumerate(free):
            expand[i, column] = 1.0
        for row, parent, mult in zip(self.mimic_rows, self.mimic_parent,
                                     self.mimic_mult):
            expand[row, free.index(parent)] = mult
        expand = jnp.asarray(expand)

        class PoseVar(jaxls.Var[jnp.ndarray],
                      default_factory=lambda: jnp.zeros(ndof),
                      retract_fn=lambda x, delta: x + expand @ delta,
                      tangent_dim=len(free)):
            pass

        TargetVar = _param_var(jnp.zeros(_TARGET_DIM))
        PersonVar = _param_var(jnp.zeros(_PERSON_DIM))
        CylGeom = _param_var(jnp.zeros(self.n_obstacles * 5))
        CylRot = _param_var(jnp.zeros(self.n_obstacles * 9))
        SphereObs = _param_var(jnp.zeros(0))
        prim_buckets = problem.collision_primitives
        self.prim_flat = js._flatten_prim_poses(prim_buckets)
        PrimPose = _param_var(jnp.zeros(int(self.prim_flat.shape[0])))
        prim_state = {'used': False,
                      'layout': js._prim_pose_layout(prim_buckets),
                      'var': PrimPose}
        self._vars = (PoseVar, TargetVar, PersonVar, CylGeom, CylRot,
                      PrimPose)

        def base_pose(q):
            return js._floating_base_pose(q[nj:], 3, root_pos, root_rot, jnp)

        def ee_pose_error(q, target):
            base_pos, base_rot = base_pose(q)
            ee_pos, ee_rot = ee_fk(q[:nj], base_pos, base_rot)
            return pose_error_log(ee_pos, ee_rot, target[:3],
                                  target[3:12].reshape(3, 3))

        sqrt_pose_weight = math.sqrt(pose_weight)
        zero = jnp.array([0])
        costs = []

        @jaxls.Cost.factory(name='hover_pose')
        def pose_cost(vals, var, target):
            return sqrt_pose_weight * ee_pose_error(vals[var], vals[target])
        costs.append(pose_cost(PoseVar(zero), TargetVar(zero)))

        @jaxls.Cost.factory(kind='constraint_geq_zero', name='limits')
        def limit_cost(vals, var, person):
            # 関節の可動域だけが制約。台車 (x, y, yaw) の範囲は制約にせず、置き方は
            # placement_goal の誤差項で決める (person の下限・上限は初期値の範囲にだけ使う)。
            q = vals[var][:nj]
            return jnp.concatenate([q - joint_lower, joint_upper - q])
        costs.append(limit_cost(PoseVar(zero), PersonVar(zero)))

        # 人体との距離: skrobot の厳密形状の干渉コスト max(0, margin - d) を
        # 符号反転して ">= 0" の制約 (d >= margin) にする。
        soft_collision = js.JaxlsSolver._make_world_collision_cost(
            None, problem, PoseVar, SphereObs, CylGeom, CylRot, fk_data,
            collision_spec, prim_state)
        collision_residual = soft_collision.compute_residual

        def negated_collision(*args, _residual=collision_residual):
            return -_residual(*args)
        costs.append(jaxls.Cost(
            compute_residual=negated_collision, args=soft_collision.args,
            kind='constraint_geq_zero', name='human_collision'))

        def placement_errors(q, target, person):
            # 押し込み後の台車の前方ずれと、台車の +x の正面方向からのずれ。
            base_xy = q[nj:nj + 2] + target[12:14]
            front = jnp.dot(base_xy - person[0:2], person[2:4])
            yaw = q[nj + 2]
            c, s = jnp.cos(yaw), jnp.sin(yaw)
            dyaw = jnp.arctan2(person[2] * s - person[3] * c,
                               person[2] * c + person[3] * s)
            return front, dyaw

        @jaxls.Cost.factory(name='placement_goal')
        def placement_cost(vals, var, target, person):
            # 許容幅で正規化した誤差 e / eps (不感帯なし)。許容幅は成否の判定にも使う。
            p = vals[person]
            front, dyaw = placement_errors(vals[var], vals[target], p)
            return jnp.stack([p[12] * front / p[10], p[13] * dyaw / p[11]])
        costs.append(placement_cost(PoseVar(zero), TargetVar(zero),
                                    PersonVar(zero)))

        js._pin_cost_group_order(costs)
        variables = [PoseVar(zero), TargetVar(zero), PersonVar(zero),
                     CylGeom(zero), CylRot(zero), PrimPose(zero)]
        self.ls_problem = jaxls.LeastSquaresProblem(
            costs=costs, variables=variables).analyze()

        def solve_one(q0, target, person, cyl_geom, cyl_rot, prim):
            init = jaxls.VarValues.make([
                PoseVar(zero).with_value(q0[None]),
                TargetVar(zero).with_value(target[None]),
                PersonVar(zero).with_value(person[None]),
                CylGeom(zero).with_value(cyl_geom[None]),
                CylRot(zero).with_value(cyl_rot[None]),
                PrimPose(zero).with_value(prim[None])])
            solution = self.ls_problem.solve(
                initial_vals=init, verbose=False,
                termination=jaxls.TerminationConfig(
                    max_iterations=max_iterations))
            return solution[PoseVar(zero)][0]

        def solve_all(q0, targets, person, cyl_geom, cyl_rot, prim):
            # 初期値ごとに独立 (LM の減衰・拡張ラグランジュの係数・終了判定も別々)。
            return jax.vmap(lambda q, t: solve_one(
                q, t, person, cyl_geom, cyl_rot, prim))(q0, targets)

        def evaluate_all(q_all, targets, person, cyl_geom, cyl_rot, prim):
            # 手先の位置・姿勢の誤差、人体との距離の不足の最大、目標の誤差 (前方・向き)。
            def one(q, target):
                base_pos, base_rot = base_pose(q)
                ee_pos, ee_rot = ee_fk(q[:nj], base_pos, base_rot)
                position_error = jnp.linalg.norm(ee_pos - target[:3])
                relative = ee_rot.T @ target[3:12].reshape(3, 3)
                rotation_error = jnp.arccos(jnp.clip(
                    (jnp.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
                vals = jaxls.VarValues.make([
                    PoseVar(0).with_value(q), CylGeom(0).with_value(cyl_geom),
                    CylRot(0).with_value(cyl_rot),
                    PrimPose(0).with_value(prim)])
                shortfall = collision_residual(
                    vals, (PoseVar(0), CylGeom(0), CylRot(0), PrimPose(0)),
                    {})
                front, dyaw = placement_errors(q, target, person)
                return (position_error, rotation_error, jnp.max(shortfall),
                        front, dyaw)
            return jax.vmap(one)(q_all, targets)

        self._solve = jax.jit(solve_all)
        self._evaluate = jax.jit(evaluate_all)

        # angle_vector に戻すための関節の添字 (連鎖の関節 -> robot.joint_list)。
        index_by_name = {joint.name: i
                         for i, joint in enumerate(robot.joint_list)}
        self._chain_columns = []
        self._av_indices = []
        for column, link in enumerate(self.link_list):
            if link.joint.name in index_by_name:
                self._chain_columns.append(column)
                self._av_indices.append(index_by_name[link.joint.name])
        self.shoulder_yaw_column = [
            link.joint.name for link in self.link_list].index(
                '{}_shoulder_y_joint'.format(robot_arm))

    def compile_count(self):
        """``_solve`` のコンパイル済みの数 (人物で増えないことの確認用)。"""
        return self._solve._cache_size()

    def person_inputs(self, palm, hand, rots, base_limits, standing_xy=None,
                      facing=None, collision_obstacles=None,
                      front_eps=HOVER_GOAL_FRONT_EPS,
                      front_weight=HOVER_GOAL_FRONT_WEIGHT,
                      yaw_eps=HOVER_GOAL_YAW_EPS,
                      yaw_weight=HOVER_GOAL_YAW_WEIGHT,
                      random_state=np.random):
        """1 人分の入力 (初期値・目標・人物の値・障害物) を作る。

        初期値は向き優先の並び (添字 = 向き * 1 つの向きあたりの数 + 初期値) で、
        各向きの先頭は関節が ``seed_arm_pose`` の姿勢。立ち位置・正面方向が無いときは
        目標の誤差項の重みを 0 にする。
        """
        import jax.numpy as jnp
        n_turns = len(rots)
        if self.n_instances % n_turns != 0:
            raise ValueError('初期値の数 {} が向きの数 {} で割り切れません。'
                             .format(self.n_instances, n_turns))
        per_turn = self.n_instances // n_turns
        targets = []
        for k in range(self.n_instances):
            rot = rots[k // per_turn]
            targets.append(np.concatenate([
                spik.palm_target_position(palm, rot),
                np.asarray(rot, dtype=np.float64).ravel(),
                spik.press_in_base_shift(palm, rot, 1.0)]))
        targets = np.asarray(targets)
        lower = np.array([float(b[0]) for b in base_limits])
        upper = np.array([float(b[1]) for b in base_limits])
        has_placement = standing_xy is not None and facing is not None
        if has_placement:
            standing = np.asarray(standing_xy, dtype=np.float64)[:2]
            facing = np.asarray(facing, dtype=np.float64)[:2]
            facing = facing / np.linalg.norm(facing)
        else:
            standing = np.zeros(2)
            facing = np.array([1.0, 0.0])
        person = np.concatenate([
            standing, facing, lower, upper,
            [front_eps, yaw_eps,
             math.sqrt(front_weight) if has_placement else 0.0,
             math.sqrt(yaw_weight) if has_placement else 0.0]])

        if collision_obstacles:
            obstacles = list(collision_obstacles)
        else:
            obstacles = spik.human_body_obstacles({})
        obstacles[self.forearm] = spik.offered_hand_penalty_obstacle(
            obstacles[self.forearm], self.collision_margin)
        dicts = _obstacle_dicts(obstacles)
        cyl_geom = np.concatenate([list(o['center'])
                                   + [o['radius'], o['half_height']]
                                   for o in dicts])
        cyl_rot = np.concatenate([np.asarray(o['rotation']).ravel()
                                  for o in dicts])

        q0 = self._initial_values(targets, per_turn, lower, upper,
                                  standing if has_placement else None,
                                  facing if has_placement else None,
                                  random_state)
        return (jnp.asarray(q0), jnp.asarray(targets), jnp.asarray(person),
                jnp.asarray(cyl_geom), jnp.asarray(cyl_rot),
                jnp.asarray(self.prim_flat))

    def _initial_values(self, targets, per_turn, lower, upper, standing,
                        facing, random_state):
        """初期値 (n_instances, ndof)。関節は可動域内の一様乱数 (各向きの先頭は種の姿勢)、
        台車は目標の誤差項が小さく手が届きそうな範囲から選ぶ (モジュール冒頭の INIT_*)。"""
        n = self.n_instances
        nj = self.nj
        q0 = np.empty((n, self.ndof))
        q0[:, :nj] = random_state.uniform(self.joint_lower, self.joint_upper,
                                          size=(n, nj))
        q0[::per_turn, :nj] = np.clip(self.seed_chain_angles,
                                      self.joint_lower, self.joint_upper)
        for row, parent, mult, offset in zip(
                self.mimic_rows, self.mimic_parent, self.mimic_mult,
                self.mimic_offset):
            q0[:, row] = mult * q0[:, parent] + offset

        side = 1.0 if self.robot_arm == 'l' else -1.0
        lateral = side * random_state.uniform(*INIT_TARGET_LATERAL_RANGE,
                                              size=n)
        # yaw の初期値は人の正面方向 (無ければ 0) の周り。台車の範囲は制約ではないので
        # yaw は切らない (x, y は初期値の範囲として下限・上限で切る)。
        facing_yaw = (math.atan2(facing[1], facing[0])
                      if facing is not None else 0.0)
        yaw = facing_yaw + random_state.uniform(-INIT_YAW_SPREAD,
                                                INIT_YAW_SPREAD, size=n)
        wide = random_state.uniform(size=n) < INIT_WIDE_FRACTION
        front = np.where(wide,
                         random_state.uniform(*INIT_FRONT_WIDE_RANGE, size=n),
                         random_state.uniform(*INIT_FRONT_RANGE, size=n))
        forward = random_state.uniform(*INIT_TARGET_FORWARD_RANGE, size=n)
        c, s = np.cos(yaw), np.sin(yaw)
        target_xy = targets[:, :2]
        base_xy = np.empty((n, 2))
        for k in range(n):
            if facing is not None:
                # 押し込み後の台車の前方ずれ = front、台車座標での目標の横 = lateral。
                a = np.array([facing, [-s[k], c[k]]])
                b = np.array([
                    front[k] + facing @ standing - facing @ targets[k, 12:14],
                    np.array([-s[k], c[k]]) @ target_xy[k] - lateral[k]])
                base_xy[k] = np.linalg.solve(a, b)
            else:
                base_xy[k] = target_xy[k] - np.array(
                    [c[k] * forward[k] - s[k] * lateral[k],
                     s[k] * forward[k] + c[k] * lateral[k]])
        q0[:, nj:nj + 2] = np.clip(base_xy, lower[:2], upper[:2])
        q0[:, nj + 2] = yaw
        return q0

    def solve(self, inputs):
        """``person_inputs`` の入力で解き、全初期値の解 (n_instances, ndof) を返す。"""
        q = self._solve(*inputs)
        q.block_until_ready()
        return q

    def check(self, q, inputs, thre=spik.DEFAULT_COLLISION_IK_THRE,
              rthre=spik.DEFAULT_COLLISION_IK_RTHRE):
        """制約を満たした解の bool 配列と、各解の誤差の dict を返す。"""
        _, targets, person, cyl_geom, cyl_rot, prim = inputs
        position_error, rotation_error, shortfall, front, dyaw = (
            np.asarray(v) for v in self._evaluate(
                q, targets, person, cyl_geom, cyl_rot, prim))
        q = np.asarray(q)
        p = np.asarray(person)
        q_joints = q[:, :self.nj]
        limit_excess = np.maximum(self.joint_lower - q_joints,
                                  q_joints - self.joint_upper).max(axis=1)
        feasible = ((position_error < thre) & (rotation_error < rthre)
                    & (limit_excess < LIMIT_TOLERANCE)
                    & (shortfall < COLLISION_TOLERANCE))
        # 目標の誤差 (重み 0 = 立ち位置が無いときは見ない) も許容幅内で成功。
        goal = np.zeros_like(front)
        within = np.ones(len(front), dtype=bool)
        if p[12] > 0.0:
            goal += (front / p[10]) ** 2 * p[12] ** 2
            within &= np.abs(front) <= p[10]
        if p[13] > 0.0:
            goal += (dyaw / p[11]) ** 2 * p[13] ** 2
            within &= np.abs(dyaw) <= p[11]
        ok = feasible & within
        return ok, dict(feasible=feasible, position_error=position_error,
                        rotation_error=rotation_error,
                        limit_excess=limit_excess, collision_shortfall=shortfall,
                        front=front, yaw=dyaw, goal=goal)

    def twisted_flags(self, q):
        """肩ヨーの大きさが ``spik.TWISTED_SHOULDER_YAW_DEG`` を超える解の bool 配列。"""
        q = np.asarray(q)
        return np.abs(np.degrees(q[:, self.shoulder_yaw_column])) \
            > spik.TWISTED_SHOULDER_YAW_DEG

    def to_candidates(self, q):
        """解をバッチ IK と同じ (angle_vectors, base_poses) にする。"""
        from skrobot.coordinates import Coordinates
        q = np.asarray(q)
        angle_vectors = np.tile(self.seed_angle_vector, (q.shape[0], 1))
        angle_vectors[:, self._av_indices] = q[:, self._chain_columns]
        base_poses = []
        for x, y, yaw in q[:, self.nj:]:
            base = Coordinates(pos=[float(x), float(y), 0.0])
            base.rotate(float(yaw), 'z')
            base_poses.append(base)
        return list(angle_vectors), base_poses


# 直前の ``solve_person_ik_goal`` の解の数と、制約を満たした解のうち目標の誤差が最小の解の
# 前方ずれ・向きのずれ (結果 JSON の hover_goal に入れる)。
LAST_STATS = {}

# (id(robot), 腕, 初期値の数, 干渉の組の数, マージン, 可動域の比率, 反復回数,
#  上腕の組) -> ソルバー
_SOLVERS = {}


def get_solver(robot, robot_arm, collision_pairs, n_instances,
               collision_margin=spik.DEFAULT_COLLISION_MARGIN,
               joint_limit_margin_ratio=(
                   spik.DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO),
               max_iterations=DEFAULT_HOVER_GOAL_MAX_ITERATIONS,
               upper_arm_pairs=DEFAULT_UPPER_ARM_PAIRS):
    """ソルバーを作るか、同じ構造のものを使い回す (jit のキャッシュのため)。"""
    key = (id(robot), robot_arm, n_instances, len(collision_pairs),
           collision_margin, joint_limit_margin_ratio, max_iterations,
           upper_arm_pairs)
    solver = _SOLVERS.get(key)
    if solver is None:
        solver = HoverGoalSolver(
            robot, robot_arm, collision_pairs, n_instances,
            collision_margin=collision_margin,
            joint_limit_margin_ratio=joint_limit_margin_ratio,
            max_iterations=max_iterations, upper_arm_pairs=upper_arm_pairs)
        _SOLVERS[key] = solver
    return solver


def solve_person_ik_goal(robot, palm, hand, robot_arm, collision_obstacles,
                         base_limits, collision_pairs,
                         attempts_per_turn=DEFAULT_HOVER_GOAL_ATTEMPTS_PER_TURN,
                         collision_margin=spik.DEFAULT_COLLISION_MARGIN,
                         collision_joint_limit_margin_ratio=(
                             spik.DEFAULT_COLLISION_IK_JOINT_LIMIT_MARGIN_RATIO),
                         max_iterations=DEFAULT_HOVER_GOAL_MAX_ITERATIONS,
                         collision_ik_thre=spik.DEFAULT_COLLISION_IK_THRE,
                         collision_ik_rthre=spik.DEFAULT_COLLISION_IK_RTHRE,
                         joint_positions=None,
                         verification_pairs=None,
                         collision_verify_tolerance=(
                             spik.DEFAULT_COLLISION_VERIFY_TOLERANCE),
                         post_process_ik_stop=spik.DEFAULT_POST_PROCESS_IK_STOP,
                         post_process_thre=spik.DEFAULT_POST_PROCESS_IK_THRE,
                         post_process_rthre=spik.DEFAULT_POST_PROCESS_IK_RTHRE,
                         post_process_max_candidates=(
                             spik.DEFAULT_POST_PROCESS_MAX_CANDIDATES),
                         n_turn_candidates=None,
                         hover_human_clearance=(
                             spik.DEFAULT_HOVER_HUMAN_CLEARANCE),
                         placement_joint_positions=None,
                         front_eps=HOVER_GOAL_FRONT_EPS,
                         front_weight=HOVER_GOAL_FRONT_WEIGHT,
                         yaw_eps=HOVER_GOAL_YAW_EPS,
                         yaw_weight=HOVER_GOAL_YAW_WEIGHT,
                         prefer_untwisted=True,
                         upper_arm_pairs=DEFAULT_UPPER_ARM_PAIRS,
                         verbose=False, **_ignored):
    """``solve_person_ik`` の jaxls 版。全ての向き x 初期値を 1 回で解き、
    ``pick_verified_candidate`` (台車の front/yaw の重みは 0、曲げ・沈み込み・前傾の
    並べ替えはそのまま) で 1 つ選ぶ。

    ``base_limits`` は台車の可動域の上下限 (x の窓・yaw の窓は掛けない)。
    台車の置き方は ``placement_joint_positions`` (無ければ ``joint_positions``) の
    立ち位置・正面方向に対する目標の誤差項で決め、許容幅 (``front_eps``・``yaw_eps``) を
    超えた解は収束していない扱いにする。肩ヨーがねじれた解 (``prefer_untwisted``) は、
    ねじれていない解で後処理まで通らないときだけ使う。
    ``_ignored`` はバッチ IK だけの引数 (attempts_per_pose・collision_weight 等) を受け流す。

    ``(picked, collision_ik_time, candidate_selection_time)`` を返す (時間は秒)。
    """
    if not collision_pairs:
        raise ValueError('jaxls の hover は干渉の組 (collision_pairs) が必要です。')
    collision_ik_start = time.time()
    rots = spik.palm_to_target_rots(palm, hand, robot_arm)[:n_turn_candidates]
    # 関節可動域の余裕は持たず、可動域そのものを制約にする (ユーザー指示 2026-11-02。
    # 姿勢の好みは余裕ではなく重みの誤差項で表す)。バッチ IK 用の引数は受け流す。
    solver = get_solver(
        robot, robot_arm, collision_pairs, attempts_per_turn * len(rots),
        collision_margin=collision_margin,
        joint_limit_margin_ratio=HOVER_GOAL_JOINT_LIMIT_MARGIN_RATIO,
        max_iterations=max_iterations, upper_arm_pairs=upper_arm_pairs)
    if placement_joint_positions is None:
        placement_joint_positions = joint_positions
    standing_xy = facing = None
    if placement_joint_positions:
        standing_xy = spik.human_standing_xy(placement_joint_positions)
        facing = spik.human_facing_direction(placement_joint_positions)
    inputs = solver.person_inputs(
        palm, hand, rots, base_limits, standing_xy=standing_xy, facing=facing,
        collision_obstacles=collision_obstacles, front_eps=front_eps,
        front_weight=front_weight, yaw_eps=yaw_eps, yaw_weight=yaw_weight)
    q = solver.solve(inputs)
    success_flags, errors = solver.check(q, inputs, thre=collision_ik_thre,
                                         rthre=collision_ik_rthre)
    angle_vectors, base_poses = solver.to_candidates(q)
    collision_ik_time = time.time() - collision_ik_start
    feasible = errors['feasible']
    # 制約を満たした解のうち目標の誤差が最小のもの (許容幅で絞る前の最適値)。
    best = (int(np.argmin(np.where(feasible, errors['goal'], np.inf)))
            if feasible.any() else None)
    LAST_STATS.clear()
    LAST_STATS.update(
        n_instances=int(len(feasible)), n_feasible=int(feasible.sum()),
        n_success=int(success_flags.sum()), front_eps=float(front_eps),
        yaw_eps_deg=math.degrees(yaw_eps),
        best_front=None if best is None else float(errors['front'][best]),
        best_yaw_deg=None if best is None
        else math.degrees(float(errors['yaw'][best])))
    if verbose:
        if best is not None:
            best_text = 'front={:+.3f} m yaw={:+.1f} deg'.format(
                LAST_STATS['best_front'], LAST_STATS['best_yaw_deg'])
        else:
            best_text = 'none'
        print('  [hover-goal] {} 初期値中 制約を満たした解 {}、うち目標の許容幅内 {} '
              '(目標の誤差が最小の解: {}、{:.3f} 秒)。'
              .format(len(feasible), int(feasible.sum()),
                      int(success_flags.sum()), best_text, collision_ik_time))
    # 解けなかった人を後で見るため、ロボットは種の姿勢に戻す (バッチ IK と同じ)。
    spik.seed_arm_pose(robot, robot_arm)

    candidate_selection_start = time.time()

    def pick(flags):
        return spik.pick_verified_candidate(
            robot, list(flags), angle_vectors, base_poses,
            verification_pairs, joint_positions, collision_verify_tolerance,
            robot_arm, palm, hand, rots,
            attempts_per_pose=attempts_per_turn,
            post_process_ik_stop=post_process_ik_stop,
            post_process_thre=post_process_thre,
            post_process_rthre=post_process_rthre,
            post_process_max_candidates=post_process_max_candidates,
            front_offset_weight=0.0, facing_yaw_weight=0.0,
            hover_human_clearance=hover_human_clearance,
            placement_joint_positions=placement_joint_positions)

    # 肩ヨーがねじれた枝 (solve_person_ik_side_by_side と同じ判定) は、ねじれていない解で
    # 後処理まで通らないときだけ使う (後処理まで通った解 > ねじれた解 > 後処理前の解)。
    twisted = solver.twisted_flags(q) if prefer_untwisted else \
        np.zeros(len(success_flags), dtype=bool)
    picked = pick(success_flags & ~twisted)
    if (picked is None or picked[3] is None) and (success_flags & twisted).any():
        picked_twisted = pick(success_flags & twisted)
        if picked_twisted is not None and (
                picked is None or picked_twisted[3] is not None):
            picked = picked_twisted
    candidate_selection_time = time.time() - candidate_selection_start
    return picked, collision_ik_time, candidate_selection_time


def warmup_hover_goal_ik(robot, base_limits, collision_pairs, **kwargs):
    """人物ループの前に左右の腕でダミー目標を解き、jaxls のコンパイルを済ませる。
    ``kwargs`` は人物ループの ``solve_person_ik_goal`` と同じものを渡すこと
    (初期値の数・反復回数などが違うと別のコンパイル)。"""
    for robot_arm, hand in (('l', 'R'), ('r', 'L')):
        t0 = time.time()
        solve_person_ik_goal(
            robot, spik._WARMUP_PALM, hand, robot_arm,
            spik.human_body_obstacles({}), base_limits, collision_pairs,
            joint_positions={}, **kwargs)
        print('[warmup] {}腕: jaxls の hover のトレース/コンパイル {:.1f} 秒'
              .format(robot_arm, time.time() - t0))
