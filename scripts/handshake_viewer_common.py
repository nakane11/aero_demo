#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""握手の viser ビューア (``view_handshake_poses.py``/``view_handshake_
motion.py``/``scripts/ros/run_camera_pipeline_test.py``) の共通処理。

干渉ジオメトリの overlay 表示、貫通しているリンクの組の列挙、IK 結果・
waypoint の反映、押し込み区間の表示用 waypoint の生成など。
``solve_palm_ik`` に依存するので ``scripts/`` を ``sys.path`` に含めて
import すること。
"""

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import solve_palm_ik as spik  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402
from skrobot.model import RobotModel  # noqa: E402

from aero_demo.collision_model import build_collision_model_urdf  # noqa: E402
from aero_demo.palm_plane_view import set_color as set_translucent_color  # noqa: E402,E501


# 干渉ジオメトリの表示色 (RGBA, 0-255): ロボットは橙、人体は青。
ROBOT_COLLISION_LINK_COLOR = [220, 140, 80, 90]
HUMAN_COLLISION_OBSTACLE_COLOR = [80, 140, 220, 90]

# hover から押し込み姿勢までの補間フレーム数。
PRESS_IN_DISPLAY_WAYPOINTS = 5


def set_link_visible(viewer, link, visible):
    """``link`` の表示/非表示を切り替える (GUI スレッドとの競合で既に
    削除されたリンクなら何もしない)。"""
    handle = viewer._linkid_to_handle.get(str(id(link)))
    if handle is not None:
        handle.visible = visible


def remove_joint_angle_gui(viewer):
    """``ViserViewer`` の関節スライダー GUI を取り除く。

    スライダーは関節名で管理され、ロボットと overlay の片方だけを動かして
    しまうため。全てのロボットを ``viewer.add`` した後に呼ぶこと。
    """
    # 親フォルダごと消す (_joint_sliders は 2 体目のものしか持たない)。
    for attr in ('_joint_angles_folder', '_export_folder'):
        folder = getattr(viewer, attr, None)
        if folder is not None:
            folder.remove()
            setattr(viewer, attr, None)
    viewer._joint_sliders.clear()


def remove_obstacles_gui(viewer):
    """``ViserViewer`` の "Obstacles" GUI を取り除く (全てのロボットを
    ``viewer.add`` した後に呼ぶこと)。"""
    folder = getattr(viewer, '_obstacles_folder', None)
    if folder is not None:
        folder.remove()
        viewer._obstacles_folder = None
    viewer._joint_folders.clear()


def build_robot_collision_overlay(robot, primitive_type=None,
                                  force_convert=False):
    """``robot`` のプリミティブ近似ジオメトリを半透明の別 ``RobotModel``
    (overlay) として読み込む。姿勢は ``sync_robot_collision_overlay`` で
    追従させる。"""
    collision_urdf_path = build_collision_model_urdf(
        robot.urdf_path, primitive_type=primitive_type, force=force_convert)
    collision_robot = RobotModel()
    collision_robot.load_urdf_file(
        str(collision_urdf_path), include_mimic_joints=False)
    for link in collision_robot.link_list:
        set_translucent_color(link, ROBOT_COLLISION_LINK_COLOR)
    return collision_robot


def sync_robot_collision_overlay(collision_robot, robot):
    """overlay を ``robot`` の関節角 (関節名で対応) と台車姿勢に合わせる。"""
    name_to_angle = {joint.name: joint.joint_angle()
                     for joint in robot.joint_list}
    for joint in collision_robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    collision_robot.newcoords(robot.base_link.copy_worldcoords())


def colliding_link_pairs(robot, pairs, obstacle_links,
                         tolerance=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE):
    """``pairs`` のうち ``tolerance`` を超えて貫通している組をすべて列挙する。

    人体は表示中の円柱 ``obstacle_links`` で、頂点の入り込みと円柱表面
    からの入り込みの両方向を見る (片方向だと面の途中の貫通を見逃す)。
    自己干渉は ``solve_palm_ik.self_collision_depth`` で判定する。
    ``obstacle_links`` が空なら人体との組は読み飛ばす。

    Returns
    -------
    list of (kind, link_a_name, name_b, dist)
        ``kind`` は ``'self'``/``'human'``、``dist`` [m] は負なほど深い。
        深い順。
    """
    if not pairs:
        return []
    obstacle_names = spik.human_obstacle_names()
    world_vertices_by_link = {}
    shape_by_link = {}
    samples_by_obstacle = {}

    def _world_vertices(link):
        if link not in world_vertices_by_link:
            local = np.asarray(link.collision_mesh.vertices, dtype=np.float64)
            world_vertices_by_link[link] = (
                local @ link.worldrot().T + link.worldpos())
        return world_vertices_by_link[link]

    def _shape(link):
        if link not in shape_by_link:
            shape_by_link[link] = spik.link_collision_shape(link)
        return shape_by_link[link]

    def _samples(index):
        if index not in samples_by_obstacle:
            samples_by_obstacle[index] = spik.cylinder_surface_samples(
                obstacle_links[index])
        return samples_by_obstacle[index]

    colliding = []
    for link_a, other in pairs:
        verts_a = _world_vertices(link_a)
        if isinstance(other, int):
            if not obstacle_links:
                continue
            obstacle = obstacle_links[other]
            # 円柱のローカル座標 (Z が軸、原点中心) で入り込んだ深さ (内側で正)。
            local_pts = ((verts_a - obstacle.worldpos())
                        @ obstacle.worldrot())
            radial = np.linalg.norm(local_pts[:, :2], axis=1)
            axial = np.abs(local_pts[:, 2])
            depth = float(np.minimum(
                obstacle.radius - radial,
                obstacle.height / 2.0 - axial).max())
            # 逆向き (円柱がリンクの面の途中を貫通) も見る。
            depth = max(depth, spik.obstacle_into_link_depth(
                _samples(other), link_a, _shape(link_a)))
            dist = -depth
            kind, name_b = 'human', obstacle_names[other]
        else:
            # 包含球が重ならない組は省略する。
            kind, name_b = 'self', other.name
            center_gap = (np.linalg.norm(other.worldpos() - link_a.worldpos())
                          - _shape(link_a)[3] - _shape(other)[3])
            if center_gap > 0.0:
                continue
            dist = -spik.self_collision_depth(link_a, other)
        if dist < -tolerance:
            colliding.append((kind, link_a.name, name_b, dist))
    colliding.sort(key=lambda item: item[3])
    return colliding


def collision_pairs_text(colliding, label='干渉'):
    """``colliding_link_pairs`` の結果をテキストパネル用の文字列にする
    (``label`` は見出し)。"""
    self_pairs = [c for c in colliding if c[0] == 'self']
    human_pairs = [c for c in colliding if c[0] == 'human']
    if not colliding:
        return '{}: なし (自己干渉・人体との干渉ともに検出されていません)'.format(
            label)
    lines = ['{}: {} 件 (自己干渉 {} 件, 人体との干渉 {} 件)'.format(
        label, len(colliding), len(self_pairs), len(human_pairs))]
    for _, name_a, name_b, dist in self_pairs:
        lines.append('- [自己干渉] `{}` - `{}` ({:.4f} m 貫通)'.format(
            name_a, name_b, -dist))
    for _, name_a, name_b, dist in human_pairs:
        lines.append('- [対人干渉] `{}` - `{}` ({:.4f} m 貫通)'.format(
            name_a, name_b, -dist))
    return '\n\n'.join(lines)


def apply_robot_pose(robot, result, use_post_process=False):
    """IK 結果の関節角 (関節名で対応、指は初期姿勢のまま) と台車姿勢を
    ``robot`` に反映する。

    ``use_post_process`` なら押し込み姿勢 (``post_process``、無ければ
    hover 姿勢) を反映する。
    """
    source = result
    if use_post_process and result.get('post_process') is not None:
        source = result['post_process']
    robot.reset_pose()
    name_to_angle = dict(zip(
        source['joint_names'], source['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.base_link.newcoords(Coordinates(
        pos=source['base_position'],
        rot=rpy_matrix(source['base_yaw'], 0.0, 0.0)))


def apply_waypoint_pose(robot, joint_names, waypoints, index):
    """``waypoints[index]`` を ``robot`` に反映する (関節名で対応)。"""
    wp = waypoints[index]
    robot.reset_pose()
    name_to_angle = dict(zip(joint_names, wp['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.base_link.newcoords(Coordinates(
        pos=wp['base_position'], rot=rpy_matrix(wp['base_yaw'], 0.0, 0.0)))


def build_display_waypoints(motion, result, n_press_in=PRESS_IN_DISPLAY_WAYPOINTS):
    """接近経路の後ろに押し込み区間と横並び移動の waypoint を足す。

    Returns
    -------
    (waypoints, n_approach)
        ``n_approach`` は ``motion['waypoints']`` の個数 (以降は未検証の
        押し込み・横並び区間)。
    """
    waypoints = list(motion['waypoints'])
    n_approach = len(waypoints)
    post = result.get('post_process')
    if post is None:
        return waypoints, n_approach
    waypoints += build_press_in_waypoints(
        waypoints[-1], motion['joint_names'], post, n_press_in)
    waypoints += transition_waypoints(motion)
    return waypoints, n_approach


def transition_waypoints(motion):
    """横並び移動 (``motion['transition']``) の waypoint を
    ``motion['joint_names']`` の並びに直して返す (``transition: True`` 付き。
    無い・未検証なら空)。"""
    transition = motion.get('transition')
    if not transition or not transition.get('verified'):
        return []
    joint_names = motion['joint_names']
    waypoints = []
    for wp in transition['waypoints']:
        name_to_angle = dict(zip(transition['joint_names'],
                                 wp['joint_angle_vector']))
        waypoints.append(dict(
            wp, transition=True,
            joint_angle_vector=[float(name_to_angle[name])
                                for name in joint_names]))
    return waypoints


def build_press_in_waypoints(last_wp, joint_names, post,
                             n_press_in=PRESS_IN_DISPLAY_WAYPOINTS):
    """hover の waypoint ``last_wp`` から押し込み姿勢 ``post`` までを
    ``n_press_in`` 等分した waypoint のリスト (``last_wp`` は含まない)。"""
    waypoints = []
    start_vec = np.asarray(last_wp['joint_angle_vector'], dtype=np.float64)
    post_name_to_angle = dict(zip(post['joint_names'],
                                  post['joint_angle_vector']))
    end_vec = np.array([post_name_to_angle.get(name, start_vec[i])
                        for i, name in enumerate(joint_names)])
    base_start = np.array([last_wp['base_position'][0],
                           last_wp['base_position'][1], last_wp['base_yaw']])
    base_end = np.array([post['base_position'][0], post['base_position'][1],
                         post['base_yaw']])

    for t in np.linspace(0.0, 1.0, n_press_in + 1)[1:]:
        angle_vec = start_vec + (end_vec - start_vec) * t
        base_vec = base_start + (base_end - base_start) * t
        waypoints.append(dict(
            base_position=[float(base_vec[0]), float(base_vec[1]), 0.0],
            base_yaw=float(base_vec[2]),
            joint_angle_vector=[float(v) for v in angle_vec],
        ))
    return waypoints
