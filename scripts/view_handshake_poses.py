#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``solve_palm_ik.py`` の IK 結果 JSON と対応する SMPL モデルを
ファイル名で突き合わせ、viser で並べて表示する。

人体・ロボットの干渉用近似ジオメトリと台車の可動範囲を半透明で重ね、
IK 失敗時は目標 (長い Axis) と手先 (短い Axis) を描く。人間の顔は
ロボットの手先を向くよう首/頭だけ回して描く。IK 対象外 (``target``
が false) の人物は読み飛ばす。Back/Next/Good/Bad ボタンで切り替え・
判定 (``human_label`` を handshake JSON に書き込む)。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo solve_palm_ik.py
    rosrun aero_demo view_handshake_poses.py
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import trimesh

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from aero_demo import smpl_body  # noqa: E402  (パス追加後に import)
from aero_demo import viewer_nav  # noqa: E402
from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from aero_demo.palm_plane_view import set_color as set_translucent_color  # noqa: E402,E501

from generate_random_human_poses import load_smpl_models  # noqa: E402
from handshake_viewer_common import HUMAN_COLLISION_OBSTACLE_COLOR as COLLISION_OBSTACLE_COLOR  # noqa: E402,E501
from handshake_viewer_common import apply_robot_pose  # noqa: E402
from handshake_viewer_common import build_robot_collision_overlay  # noqa: E402
from handshake_viewer_common import colliding_link_pairs  # noqa: E402
from handshake_viewer_common import collision_pairs_text  # noqa: E402
from handshake_viewer_common import remove_joint_angle_gui  # noqa: E402
from handshake_viewer_common import remove_obstacles_gui  # noqa: E402
from handshake_viewer_common import set_link_visible as common_set_link_visible  # noqa: E402,E501
from handshake_viewer_common import sync_robot_collision_overlay  # noqa: E402
from solve_palm_ik import DEFAULT_COLLISION_VERIFY_TOLERANCE  # noqa: E402
from solve_palm_ik import HUMAN_FRONT_DISTANCE  # noqa: E402
from solve_palm_ik import build_collision_verification_pairs  # noqa: E402
from solve_palm_ik import human_body_obstacles  # noqa: E402
from solve_palm_ik import human_translation_offset  # noqa: E402
from solve_palm_ik import load_skeleton_json as load_joint_positions  # noqa: E402
from solve_palm_ik import translate_joint_positions  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.model import Link  # noqa: E402
from skrobot.model.primitives import Box  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

SKIN_COLOR = [180, 130, 110, 255]

# 台車の可動範囲 (薄い赤の半透明平面)。z は z-fighting 回避のため少し浮かす。
BASE_MOVABLE_REGION_COLOR = [220, 40, 40, 60]
BASE_MOVABLE_REGION_HEIGHT = 0.005
BASE_MOVABLE_REGION_Z = 0.005

# IK 失敗時の Axis [m]。色では区別できないので長さで区別 (長い方が目標)。
TARGET_AXIS_LENGTH = 0.12
TARGET_AXIS_RADIUS = 0.005
HAND_AXIS_LENGTH = 0.06
HAND_AXIS_RADIUS = 0.005

# 視線の回転のうち首 (NECK) に持たせる割合 (残りは頭)。
GAZE_NECK_RATIO = 0.5

# 胸の正面から顔の正面を離せる最大角度 [deg]。
GAZE_MAX_ANGLE_DEG = 80.0

# 首を回すと頭も動くので解き直す回数。
GAZE_ITERATIONS = 2

# 静止姿勢での顔の正面 (ロボット座標系の +x)。
GAZE_FORWARD_AXIS = np.array([1.0, 0.0, 0.0])


def load_skeleton_json(path):
    """骨格 JSON から SMPL のパラメータだけを読む."""
    with open(path) as f:
        data = json.load(f)
    smpl = data['smpl']
    return dict(
        gender=smpl['gender'],
        betas=np.asarray(smpl['betas'], dtype=np.float64),
        pose=np.asarray(smpl['pose'], dtype=np.float64),
        root_pos=np.asarray(smpl['root_pos'], dtype=np.float64))


def load_handshake_json(path):
    """``solve_palm_ik.save_json`` が保存した IK 結果 JSON を読む."""
    with open(path) as f:
        return json.load(f)


def smpl_world_rots(model, pose):
    """SMPL の各関節のワールド (ロボット座標系) 回転行列 ``(24, 3, 3)``.

    root は回転しない (``forward_world`` と同じ)。
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(24, 3)
    world_rots = np.zeros((24, 3, 3))
    for i in range(24):
        local = smpl_body.PERM.dot(smpl_body.rodrigues(pose[i])).dot(
            smpl_body.PERM.T)
        parent = model.parent[i]
        world_rots[i] = local if parent < 0 \
            else world_rots[parent].dot(local)
    return world_rots


def limit_direction(base, direction, max_angle):
    """``direction`` を ``base`` から ``max_angle`` [rad] 以内に丸める.

    回転量でなく方向を丸めるので ``look_at_pose`` の反復が収束する。
    """
    angle = np.arccos(np.clip(float(np.dot(base, direction)), -1.0, 1.0))
    if angle <= max_angle:
        return direction
    axis = np.cross(base, direction)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        # 真後ろ: 回転軸が決まらないので base のまま。
        return base
    return smpl_body.rodrigues(axis / norm * max_angle).dot(base)


def look_at_pose(model, person, target_position):
    """人間が ``target_position`` を見るよう首/頭だけ回した pose (24, 3) を返す.

    ``person['pose']`` は変更しない。
    """
    pose = np.array(person['pose'], dtype=np.float64).reshape(24, 3)
    target_position = np.asarray(target_position, dtype=np.float64)
    max_angle = np.deg2rad(GAZE_MAX_ANGLE_DEG)

    for _ in range(GAZE_ITERATIONS):
        _vertices, joints = smpl_body.forward_world(
            model, pose, person['betas'], person['root_pos'])
        world_rots = smpl_world_rots(model, pose)
        forward = world_rots[smpl_body.HEAD].dot(GAZE_FORWARD_AXIS)
        # 首をひねれる限界は胸 (首の親) の正面から測る。
        chest_forward = world_rots[model.parent[smpl_body.NECK]].dot(
            GAZE_FORWARD_AXIS)

        direction = target_position - joints[smpl_body.HEAD]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            break
        direction = limit_direction(
            chest_forward, direction / norm, max_angle)
        axis_angle = smpl_body.mat_to_axis_angle(
            smpl_body.rotation_between(forward, direction))
        if np.linalg.norm(axis_angle) < 1e-9:
            break

        # 首 -> 頭の順に、ワールド回転を親の座標系に移して入れる
        # (頭の親は回した後の首)。
        parent_world = world_rots[model.parent[smpl_body.NECK]]
        accumulated = np.eye(3)
        for joint_index, ratio in ((smpl_body.NECK, GAZE_NECK_RATIO),
                                   (smpl_body.HEAD, 1.0 - GAZE_NECK_RATIO)):
            accumulated = smpl_body.rodrigues(
                axis_angle * ratio).dot(accumulated)
            new_world = accumulated.dot(world_rots[joint_index])
            pose[joint_index] = smpl_body.mat_to_axis_angle(
                smpl_body.to_smpl_rotation(parent_world.T.dot(new_world)))
            parent_world = new_world

    return pose


def build_smpl_mesh(model, person, pose=None):
    """SMPL のメッシュを作る (``pose`` を渡すと ``person['pose']`` の代わりに使う)."""
    if pose is None:
        pose = person['pose']
    vertices, _joints = smpl_body.forward_world(
        model, pose, person['betas'], person['root_pos'])
    mesh = trimesh.Trimesh(vertices=vertices, faces=model.f, process=False)
    mesh.visual.face_colors = SKIN_COLOR
    return mesh


def is_target(handshake):
    """IK の対象にした人物か (``target`` キーが無い古い JSON は対象扱い)."""
    return bool(handshake.get('target', True))


def hand_text(handshake):
    """狙った人間の手とロボットの腕を表す文字列."""
    offered = handshake.get('offered_hand')
    robot_arm = handshake.get('robot_arm')
    return '人間の {} 手 -> ロボットの {}arm'.format(
        {'L': '左', 'R': '右'}.get(offered, '不明 ({})'.format(offered)),
        robot_arm if robot_arm is not None else '不明')


def pose_coords(handshake, pos_key, rot_key):
    """JSON の位置と回転行列から ``Coordinates`` を作る (無ければ ``None``)."""
    position = handshake.get(pos_key)
    rot = handshake.get(rot_key)
    if position is None or rot is None:
        return None
    return Coordinates(pos=np.asarray(position, dtype=np.float64),
                       rot=np.asarray(rot, dtype=np.float64))


def build_base_movable_region_link(handshake):
    """``base_movable_region`` の x/y 範囲から半透明の平面を作る (無ければ ``None``)."""
    region = handshake.get('base_movable_region')
    if region is None:
        return None
    x_min, x_max = region['x_range']
    y_min, y_max = region['y_range']
    extents = [x_max - x_min, y_max - y_min, BASE_MOVABLE_REGION_HEIGHT]
    center = [(x_min + x_max) / 2.0, (y_min + y_max) / 2.0,
             BASE_MOVABLE_REGION_Z]
    link = Box(extents=extents, pos=center, name='base_movable_region')
    set_translucent_color(link, BASE_MOVABLE_REGION_COLOR)
    return link


def gaze_target_position(handshake):
    """人間の注視点。成功時は手先、失敗時は目標位置 (無ければ ``None``)."""
    key = 'hand_position' if handshake.get('solved') else 'target_position'
    position = handshake.get(key)
    if position is None:
        return None
    return np.asarray(position, dtype=np.float64)


def pose_error_text(target_coords, hand_coords):
    """目標姿勢と手先姿勢のずれ (位置 [m] と向き [deg]) の文字列."""
    diff = hand_coords.worldpos() - target_coords.worldpos()
    rel = np.dot(target_coords.worldrot().T, hand_coords.worldrot())
    angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return '位置ずれ {:.3f} m, 向きずれ {:.1f} deg'.format(
        float(np.linalg.norm(diff)), float(np.rad2deg(angle)))


def iter_common_names(skeleton_dir, handshake_dir):
    """両ディレクトリにある IK 対象の JSON のファイル名をソートして返す."""
    skeleton_names = {os.path.basename(p) for p in
                      glob.glob(os.path.join(skeleton_dir, '*.json'))}
    handshake_names = {os.path.basename(p) for p in
                       glob.glob(os.path.join(handshake_dir, '*.json'))}
    return sorted(
        name for name in skeleton_names & handshake_names
        if is_target(load_handshake_json(os.path.join(handshake_dir, name))))


def main():
    parser = argparse.ArgumentParser(
        description='solve_palm_ik.py の IK 結果と SMPL モデルを viser で表示する。')
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='SMPL パラメータを持つ骨格 JSON のディレクトリ。')
    parser.add_argument(
        '--handshake-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='solve_palm_ik.py の出力 JSON のディレクトリ。')
    parser.add_argument(
        '--model-path', type=str,
        default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'),
        help='SMPL (男性) モデル .pkl のパス。')
    parser.add_argument(
        '--female-model-path', type=str,
        default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicModel_f_lbs_10_207_0_v1.0.0.pkl'),
        help='SMPL (女性) モデル .pkl のパス (無ければ男性のみ)。')
    parser.add_argument('--client-wait-timeout', type=float, default=30.0,
                        help='ブラウザ接続を待つ 1 回あたりの秒数。')
    parser.add_argument('--no-open-browser', action='store_true',
                        help='ブラウザを自動で開かない。')
    args = parser.parse_args()

    names = iter_common_names(args.skeleton_dir, args.handshake_dir)
    if not names:
        print('{} と {} の両方に対応する IK 対象のファイルが見つかりません。'
              '先に solve_palm_ik.py を実行してください。'.format(
                  args.skeleton_dir, args.handshake_dir))
        return

    models_by_gender = dict(
        load_smpl_models(args.model_path, args.female_model_path))

    robot = load_aero(use_hand=True)

    viewer = ViserViewer(draw_grid=True)
    # ボタンはロボットを add する前に作る (関節スライダーの下に埋もれるため)。
    nav = viewer_nav.ManualNav(viewer)
    label_text = viewer._server.gui.add_markdown('')

    show_collision_models_checkbox = viewer._server.gui.add_checkbox(
        '干渉回避用モデルの表示', initial_value=True)

    show_base_region_checkbox = viewer._server.gui.add_checkbox(
        '台車の可動範囲の表示', initial_value=True)

    show_post_process_checkbox = viewer._server.gui.add_checkbox(
        '後処理後の姿勢を表示 (掌に押し付け/自分の手を注視)',
        initial_value=False)
    # 表示中の handshake (クロージャから更新するため 1 要素リスト)。
    current_handshake = [None]

    def set_link_visible(link, visible):
        common_set_link_visible(viewer, link, visible)

    def refresh_robot_pose():
        """表示中の handshake をチェックボックスに応じた姿勢で反映する."""
        handshake = current_handshake[0]
        if handshake is None:
            return
        apply_robot_pose(robot, handshake, show_post_process_checkbox.value)
        sync_robot_collision_overlay(robot_collision_overlay, robot)
        viewer.redraw()

    @show_post_process_checkbox.on_update
    def _on_toggle_post_process(_):  # noqa: ANN001
        refresh_robot_pose()

    @show_collision_models_checkbox.on_update
    def _on_toggle_collision_models(_):  # noqa: ANN001
        visible = show_collision_models_checkbox.value
        for link in robot_collision_overlay.link_list:
            set_link_visible(link, visible)
        for obstacle_link in current_obstacle_links:
            set_link_visible(obstacle_link, visible)

    @show_base_region_checkbox.on_update
    def _on_toggle_base_region(_):  # noqa: ANN001
        if current_base_region_link is not None:
            set_link_visible(current_base_region_link,
                             show_base_region_checkbox.value)

    viewer.add(robot)
    robot_collision_overlay = build_robot_collision_overlay(robot)
    viewer.add(robot_collision_overlay)
    # 事後検証と同じ干渉ペア (人物に依存しない。'r' はプレースホルダ)。
    verification_pairs = build_collision_verification_pairs(
        robot_collision_overlay, 'r')
    # 関節スライダーは robot と overlay の姿勢を食い違わせるので消す。
    remove_joint_angle_gui(viewer)
    remove_obstacles_gui(viewer)
    viewer.show(open_browser=not args.no_open_browser)
    viewer_nav.wait_for_client(viewer, args.client_wait_timeout)
    viewer_nav.set_front_view(viewer)

    # IK 失敗時だけ表示する Axis (1 組を使い回す)。
    target_axis = Axis(axis_length=TARGET_AXIS_LENGTH,
                       axis_radius=TARGET_AXIS_RADIUS)
    hand_axis = Axis(axis_length=HAND_AXIS_LENGTH,
                     axis_radius=HAND_AXIS_RADIUS)
    axes_added = False

    current_mesh_link = None
    current_obstacle_links = []
    current_base_region_link = None
    i = 0
    while 0 <= i < len(names):
        name = names[i]
        skeleton_path = os.path.join(args.skeleton_dir, name)
        handshake_path = os.path.join(args.handshake_dir, name)
        person = load_skeleton_json(skeleton_path)
        handshake = load_handshake_json(handshake_path)

        # solve_palm_ik.py と同じく人物をロボットの前方へ平行移動する。
        joint_positions = load_joint_positions(skeleton_path)
        offset = human_translation_offset(
            joint_positions, front_distance=HUMAN_FRONT_DISTANCE)
        joint_positions = translate_joint_positions(joint_positions, offset)
        if offset != (0.0, 0.0):
            person = dict(person)
            root_pos = np.array(person['root_pos'], dtype=np.float64)
            root_pos[0] += offset[0]
            root_pos[1] += offset[1]
            person['root_pos'] = root_pos

        model = models_by_gender.get(
            person['gender'], models_by_gender['male'])
        gaze_target = gaze_target_position(handshake)
        pose = None if gaze_target is None \
            else look_at_pose(model, person, gaze_target)
        mesh = build_smpl_mesh(model, person, pose)
        link = Link(visual_mesh=mesh, name='smpl_human')
        if current_mesh_link is not None:
            viewer.delete(current_mesh_link)
        viewer.add(link)
        current_mesh_link = link

        for obstacle_link in current_obstacle_links:
            viewer.delete(obstacle_link)
        current_obstacle_links = human_body_obstacles(joint_positions)
        for obstacle_link in current_obstacle_links:
            set_translucent_color(obstacle_link, COLLISION_OBSTACLE_COLOR)
            viewer.add(obstacle_link)
            set_link_visible(obstacle_link,
                             show_collision_models_checkbox.value)

        if current_base_region_link is not None:
            viewer.delete(current_base_region_link)
            current_base_region_link = None
        current_base_region_link = build_base_movable_region_link(handshake)
        if current_base_region_link is not None:
            viewer.add(current_base_region_link)
            set_link_visible(current_base_region_link,
                             show_base_region_checkbox.value)

        # 人物を切り替えたら後処理前の姿勢から表示する。
        current_handshake[0] = handshake
        show_post_process_checkbox.value = False
        refresh_robot_pose()

        # 表示中の姿勢で貫通している組 (事後検証と同じ形状・許容誤差)。
        colliding_pairs = colliding_link_pairs(
            robot_collision_overlay, verification_pairs,
            current_obstacle_links,
            tolerance=DEFAULT_COLLISION_VERIFY_TOLERANCE)

        target_coords = pose_coords(
            handshake, 'target_position', 'target_rot')
        hand_coords = pose_coords(handshake, 'hand_position', 'hand_rot')
        show_axes = (not handshake.get('solved')
                     and target_coords is not None
                     and hand_coords is not None)
        if show_axes:
            target_axis.newcoords(target_coords)
            hand_axis.newcoords(hand_coords)
            if not axes_added:
                viewer.add(target_axis)
                viewer.add(hand_axis)
                axes_added = True
        elif axes_added:
            viewer.delete(target_axis)
            viewer.delete(hand_axis)
            axes_added = False

        if handshake.get('solved'):
            pre_status = 'solved'
        else:
            pre_status = 'NOT solved ({})'.format(hand_text(handshake))
        if not handshake.get('solved'):
            post_status = '-'
        elif handshake.get('post_process') is not None:
            post_status = 'solved'
        else:
            post_status = 'NOT solved'
        status = '後処理前 {} / 後処理後 {}'.format(pre_status, post_status)
        detail = ''
        if show_axes:
            detail = '\n\n目標 Axis (長い方, 長さ {:.2f} m) と手先 Axis ' \
                '(短い方, 長さ {:.2f} m): {}'.format(
                    TARGET_AXIS_LENGTH, HAND_AXIS_LENGTH,
                    pose_error_text(target_coords, hand_coords))
        label_text.content = '**{}** ({}/{})  IK: {}{}\n\n{}\n\n{}'.format(
            name, i + 1, len(names), status, detail,
            viewer_nav.format_label_text(handshake.get('human_label')),
            collision_pairs_text(colliding_pairs))

        viewer.redraw()
        print('[{}/{}] displayed {} ({}, 干渉 {} 件)'.format(
            i + 1, len(names), name, status, len(colliding_pairs)))

        direction, label = nav.wait(viewer)
        if direction == 0:
            print('ブラウザクライアントが切断されました。中断します。')
            break
        if label is not viewer_nav.NOT_PRESSED:
            viewer_nav.save_label(handshake_path, label)
            print('  -> {} として {} に記録しました。'.format(
                'Good' if label else 'Bad', handshake_path))
        i = max(0, i + direction)

    viewer.close()


if __name__ == '__main__':
    main()
