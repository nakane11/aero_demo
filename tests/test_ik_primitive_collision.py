"""バッチ IK の干渉ペナルティ (collision_geometry='primitive') の単体テスト。

プリミティブと円柱障害物の符号付き距離を、リンク表面サンプルによる値と比べる。

    python -m pytest tests/test_ik_primitive_collision.py -s
"""
import math
import os
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'scripts')
sys.path.insert(0, os.path.abspath(_SCRIPTS_DIR))

import solve_palm_ik as spik  # noqa: E402

jax = pytest.importorskip('jax')

from skrobot.backend import get_backend  # noqa: E402
from skrobot.kinematics import differentiable as diff  # noqa: E402
from skrobot.model.primitives import Cylinder  # noqa: E402

# 許容誤差 [m]。真の距離はサンプル最小値と spacing/sqrt(3) を引いた値の間。
TOLERANCE = 0.003
# 比べる距離の上限 [m]。
MAX_COMPARED_DISTANCE = 0.3


@pytest.fixture(scope='module')
def robot():
    robot = spik.Aero(use_hand=False)
    spik.restrict_elbow_range(robot)
    spik.restrict_leg_range(robot)
    spik.restrict_waist_range(robot)
    spik.restrict_neck_range(robot)
    spik.lock_fixed_joints(robot)
    spik.apply_collision_model(robot)
    return robot


def _collision_links(robot):
    """表面サンプルが取れる (collision_mesh がある) 全リンク。"""
    return [link for link in robot.link_list
            if getattr(link, 'collision_mesh', None) is not None]


def _random_cylinders(rng, center, n=12, spread=0.35):
    """``center`` のまわりにランダムな向き・大きさの円柱を置く。"""
    obstacles = []
    for _ in range(n):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        rot = diff._rotation_with_z_axis(axis)
        pos = center + rng.uniform(-spread, spread, size=3)
        obstacles.append(Cylinder(radius=float(rng.uniform(0.02, 0.08)),
                                  height=float(rng.uniform(0.05, 0.4)),
                                  pos=pos.tolist(), rot=rot))
    return obstacles


def _set_random_pose(robot, rng, link_list):
    """関節を可動域の中央 60% のランダムな角度にする (無限可動域は除く)。

    mimic 関節を親で上書きさせるため逆順に設定する。"""
    for link in reversed(link_list):
        joint = link.joint
        lo, hi = joint.min_angle, joint.max_angle
        if not (np.isfinite(lo) and np.isfinite(hi)):
            continue
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.6
        joint.joint_angle(float(rng.uniform(mid - half, mid + half)))


def _sample_distance(link, obstacle):
    """表面サンプルから円柱までの距離の (下界, 上界)。"""
    points = (spik.link_surface_samples(link) @ link.worldrot().T
              + link.worldpos())
    upper = float(spik.points_cylinder_distance(points, obstacle).min())
    lower = upper - spik._link_surface_sample_spacing(link) / math.sqrt(3.0)
    return lower, upper


def _primitive_distances(setup, link_list, fk_params, obstacles):
    """(リンク添字, 障害物添字) -> プリミティブの符号付き距離 (現在の関節角)。"""
    backend = get_backend('jax')
    cost = diff._make_primitive_collision_cost(setup, backend)
    _, obstacle_values = diff._collision_obstacle_types_and_values(
        setup, obstacles, backend)
    angles = backend.array(
        [link.joint.joint_angle() for link in link_list])
    positions, rotations = diff.forward_kinematics(backend, angles, fk_params)
    obstacle_dists, _ = cost.pair_distances(
        positions, rotations, obstacle_values)
    # (type, row) -> collision_link_list の添字
    row_to_link = {entry: li for li, entry in enumerate(setup['link_entry'])
                   if entry is not None}
    rows_of_type = setup['obstacle_type_rows']
    result = {}
    for (type_a, type_b, rows_a, rows_b), dist in zip(
            setup['obstacle_pair_groups'], obstacle_dists):
        for ra, rb, d in zip(rows_a, rows_b, np.asarray(dist)):
            result[(row_to_link[(type_a, int(ra))],
                    int(rows_of_type[type_b][int(rb)]))] = float(d)
    return result


def _compare(collision_links, obstacles, prim_dists):
    n_compared = 0
    errors = []
    for (li, oi), d in prim_dists.items():
        lower, upper = _sample_distance(collision_links[li], obstacles[oi])
        if not (0.0 < upper < MAX_COMPARED_DISTANCE):
            continue
        n_compared += 1
        err = max(lower - d, d - upper, 0.0)
        errors.append(err)
        assert err < TOLERANCE, (
            '{} x obstacle {}: primitive {:.4f} m, samples [{:.4f}, {:.4f}]'
            .format(collision_links[li].name, oi, d, lower, upper))
    return n_compared, (max(errors) if errors else 0.0)


def _build(robot, link_list, move_target, collision_links, obstacles):
    fk_params = diff.extract_fk_parameters(robot, link_list, move_target)
    pairs = [(link, i) for link in collision_links
             for i in range(len(obstacles))]
    setup = diff._build_primitive_collision_setup(
        link_list, fk_params, collision_links, obstacles,
        self_collision=False, collision_pairs=pairs)
    return fk_params, setup


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_distance_matches_samples_with_moving_base(robot, seed):
    """台車 3 自由度と全身を動かした姿勢で距離が表面サンプルと一致する。"""
    rng = np.random.default_rng(seed)
    robot.reset_pose()
    whole_body = robot.rarm_whole_body
    state = robot._attach_batch_virtual_base_chain(
        'planar', whole_body.link_list)
    try:
        link_list = state['link_list']
        collision_links = _collision_links(robot)
        center = robot.rarm_end_coords.worldpos()
        obstacles = _random_cylinders(rng, center)
        fk_params, setup = _build(robot, link_list, robot.rarm_end_coords,
                                  collision_links, obstacles)

        _set_random_pose(robot, rng, whole_body.link_list)
        x, y, yaw = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), \
            rng.uniform(-math.pi, math.pi)
        for joint, value in zip(state['chain_joints'], (x, y, yaw)):
            joint.joint_angle(value)
        base = spik.Coordinates(pos=[x, y, 0.0]).rotate(yaw, 'z')
        # 障害物もロボットに合わせて動かす (近い組を作るため)。
        obstacles = _random_cylinders(
            rng, base.transform_vector(robot.rarm_end_coords.worldpos()))
        prim = _primitive_distances(setup, link_list, fk_params, obstacles)
    finally:
        for joint in state['chain_joints']:
            joint.joint_angle(0.0)
        robot._detach_batch_virtual_base_chain(state)
    # 仮想台車関節は FK にしか効かないので、サンプル側はロボットを直接動かす。
    robot.newcoords(base)
    try:
        n_compared, max_err = _compare(collision_links, obstacles, prim)
    finally:
        robot.newcoords(spik.Coordinates())
    print('seed {}: {} pairs compared, max error {:.2f} mm'.format(
        seed, n_compared, max_err * 1000))
    assert n_compared >= 10


def test_distance_matches_samples_for_links_fixed_to_base(robot):
    """鎖の外の台車固定リンク (``is_static``) もロボット移動後に正しく置かれる。"""
    rng = np.random.default_rng(10)
    robot.reset_pose()
    robot.newcoords(spik.Coordinates(pos=[0.7, -0.4, 0.0]).rotate(
        math.radians(50), 'z'))
    try:
        link_list = robot.rarm.link_list
        collision_links = _collision_links(robot)
        obstacles = _random_cylinders(
            rng, robot.rarm_end_coords.worldpos(), n=16, spread=0.5)
        fk_params, setup = _build(robot, link_list, robot.rarm_end_coords,
                                  collision_links, obstacles)
        n_static = sum(int(b['is_static'].sum())
                       for b in setup['robot_buckets'].values())
        assert n_static > 0
        _set_random_pose(robot, rng, link_list)
        prim = _primitive_distances(setup, link_list, fk_params, obstacles)
        n_compared, max_err = _compare(collision_links, obstacles, prim)
        print('fixed-to-base: {} static links, {} pairs compared, max '
              'error {:.2f} mm'.format(n_static, n_compared, max_err * 1000))
        assert n_compared >= 10
    finally:
        robot.newcoords(spik.Coordinates())


def test_human_obstacle_clearances_agree(robot):
    """プリミティブの最短距離が ``human_obstacle_clearances`` 以上、サンプル最小値以下。"""
    rng = np.random.default_rng(3)
    robot.reset_pose()
    link_list = robot.rarm_whole_body.link_list
    collision_links = _collision_links(robot)
    obstacles = _random_cylinders(rng, robot.rarm_end_coords.worldpos())
    fk_params, setup = _build(robot, link_list, robot.rarm_end_coords,
                              collision_links, obstacles)
    _set_random_pose(robot, rng, link_list)
    prim = _primitive_distances(setup, link_list, fk_params, obstacles)
    pairs = [(link, i) for link in collision_links
             for i in range(len(obstacles))]
    clearances = spik.human_obstacle_clearances(
        robot, pairs, obstacles, cull_distance=MAX_COMPARED_DISTANCE)
    n_compared = 0
    for oi, clearance in clearances.items():
        prim_min = min(d for (li, o), d in prim.items() if o == oi)
        if not (0.0 < prim_min < MAX_COMPARED_DISTANCE):
            continue
        n_compared += 1
        sample_min = min(_sample_distance(link, obstacles[oi])[1]
                         for link in collision_links)
        assert clearance - TOLERANCE < prim_min < sample_min + TOLERANCE, \
            (oi, clearance, prim_min, sample_min)
    assert n_compared >= 3


def test_human_obstacle_clearances_match_brute_force(robot):
    """``human_obstacle_clearances`` が ``cull_distance`` 未満では総当たりと一致する。"""
    rng = np.random.default_rng(5)
    link_list = robot.rarm_whole_body.link_list
    collision_links = _collision_links(robot)
    cull = MAX_COMPARED_DISTANCE
    n_below = 0
    for _ in range(5):
        robot.reset_pose()
        obstacles = _random_cylinders(rng, robot.rarm_end_coords.worldpos())
        _set_random_pose(robot, rng, link_list)
        pairs = [(link, i) for link in collision_links
                 for i in range(len(obstacles))]
        clearances = spik.human_obstacle_clearances(
            robot, pairs, obstacles, cull_distance=cull)
        for oi, clearance in clearances.items():
            expected = min(
                float(spik.points_cylinder_distance(
                    spik.link_surface_samples(link) @ link.worldrot().T
                    + link.worldpos(), obstacles[oi]).min())
                - spik._link_surface_sample_spacing(link) / np.sqrt(3.0)
                for link in collision_links)
            if expected < cull:
                n_below += 1
                assert abs(clearance - expected) < 1e-12, \
                    (oi, clearance, expected)
            else:
                assert clearance >= cull, (oi, clearance, expected)
    assert n_below >= 3


def test_gradient_is_finite(robot):
    """深く貫通した組・軸上の点を含む姿勢でも、コストの勾配が有限。"""
    robot.reset_pose()
    whole_body = robot.rarm_whole_body
    state = robot._attach_batch_virtual_base_chain(
        'planar', whole_body.link_list)
    try:
        link_list = state['link_list']
        collision_links = _collision_links(robot)
        hand = robot.rarm_end_coords.worldpos()
        obstacles = [
            # 手先を貫く円柱 (軸がちょうど手先を通る)
            Cylinder(radius=0.05, height=0.3, pos=hand.tolist()),
            Cylinder(radius=0.1, height=0.6,
                     pos=(robot.body_link.worldpos()).tolist()),
            Cylinder(radius=0.03, height=0.2,
                     pos=(hand + [0.2, 0.0, 0.0]).tolist()),
        ]
        fk_params = diff.extract_fk_parameters(
            robot, link_list, robot.rarm_end_coords)
        pairs = ([(link, i) for link in collision_links for i in range(3)]
                 + [(a, b) for a in collision_links[:8]
                    for b in collision_links[-8:] if a is not b])
        setup = diff._build_primitive_collision_setup(
            link_list, fk_params, collision_links, obstacles,
            self_collision=True, collision_pairs=pairs)
        backend = get_backend('jax')
        cost = diff._make_primitive_collision_cost(setup, backend)
        _, obstacle_values = diff._collision_obstacle_types_and_values(
            setup, obstacles, backend)

        def f(angles):
            positions, rotations = diff.forward_kinematics(
                backend, angles, fk_params)
            return cost(positions, rotations, obstacle_values,
                        0.05, 10.0, 0.02, 10.0)

        angles = jax.numpy.array(
            [link.joint.joint_angle() for link in link_list])
        value, grad = jax.value_and_grad(f)(angles)
        assert float(value) > 0.0
        assert np.all(np.isfinite(np.asarray(grad)))
        assert np.any(np.asarray(grad) != 0.0)
    finally:
        robot._detach_batch_virtual_base_chain(state)


def test_constants_are_reproducible(robot):
    """jit 定数が姿勢を戻した後もビット単位で同じで負のゼロを含まない (キャッシュ安定化)。"""
    def build():
        state = robot._attach_batch_virtual_base_chain(
            'planar', robot.rarm_whole_body.link_list)
        try:
            fk_params = diff.extract_fk_parameters(
                robot, state['link_list'], robot.rarm_end_coords)
            return diff._build_primitive_collision_setup(
                state['link_list'], fk_params, _collision_links(robot), [],
                self_collision=True)
        finally:
            robot._detach_batch_virtual_base_chain(state)

    robot.reset_pose()
    first = build()
    _set_random_pose(robot, np.random.default_rng(5),
                     robot.rarm_whole_body.link_list)
    robot.reset_pose()
    second = build()
    assert list(first['robot_buckets']) == list(second['robot_buckets'])
    for ptype, bucket in first['robot_buckets'].items():
        for key, value in bucket.items():
            other = second['robot_buckets'][ptype][key]
            assert value.tobytes() == other.tobytes(), (ptype, key)
            if value.dtype.kind == 'f':
                assert not np.any((value == 0.0) & np.signbit(value)), \
                    (ptype, key)
    assert [g[:2] for g in first['self_pair_groups']] == \
        [g[:2] for g in second['self_pair_groups']]
