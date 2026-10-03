#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``plan_handshake_motion.py`` の軌道 JSON を、対応する骨格・握手姿勢
JSON と合わせて viser で再生する (waypoint スライダー/Play)。

各 waypoint の干渉余裕と貫通しているリンクの組を表示する。``planned``
が false の人物は読み飛ばす。経路末尾の押し込み (``post_process``) と
横並び移動は表示専用で、経路の検証対象ではない。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo solve_palm_ik.py
    rosrun aero_demo plan_handshake_motion.py
    rosrun aero_demo view_handshake_motion.py
"""

import argparse
import glob
import json
import os
import sys
import threading
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from aero_demo import smpl_body  # noqa: E402
from aero_demo import viewer_nav  # noqa: E402

from generate_random_human_poses import load_smpl_models  # noqa: E402
from handshake_viewer_common import HUMAN_COLLISION_OBSTACLE_COLOR as COLLISION_OBSTACLE_COLOR  # noqa: E402,E501
from handshake_viewer_common import apply_waypoint_pose  # noqa: E402
from handshake_viewer_common import build_display_waypoints  # noqa: E402
import side_by_side_transition as sbs  # noqa: E402
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
from solve_palm_ik import load_skeleton_json as load_joint_positions  # noqa: E402,E501
from solve_palm_ik import translate_joint_positions  # noqa: E402

from skrobot.viewers import ViserViewer  # noqa: E402

from view_handshake_poses import SKIN_COLOR  # noqa: E402
from view_handshake_poses import build_smpl_mesh  # noqa: E402
from view_handshake_poses import look_at_pose  # noqa: E402
from view_handshake_poses import smpl_world_rots  # noqa: E402
from view_handshake_poses import load_skeleton_json as load_smpl_params  # noqa: E402,E501

from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from aero_demo.palm_plane_view import set_color as set_translucent_color  # noqa: E402,E501

# waypoint 自動再生の既定の速さ [waypoint/秒]。
DEFAULT_PLAYBACK_FPS = 40.0

# plan_handshake_motion.KIND_LABELS の複製 (向こうは import が重い)。
KIND_LABELS = {
    'pretouch': 'pre-touch 経由 (最適化なし)',
    'linear': '線形補間のみ (最適化なし)',
    'optimized': '軌道最適化',
}


def load_handshake_json(path):
    with open(path) as f:
        return json.load(f)


def load_motion_json(path):
    with open(path) as f:
        return json.load(f)


def iter_common_names(skeleton_dir, handshake_dir, motion_dir):
    """3 ディレクトリ全てにあり ``planned`` が true のファイル名 (ソート済み)."""
    skeleton_names = {os.path.basename(p) for p in
                      glob.glob(os.path.join(skeleton_dir, '*.json'))}
    handshake_names = {os.path.basename(p) for p in
                       glob.glob(os.path.join(handshake_dir, '*.json'))}
    motion_names = {os.path.basename(p) for p in
                    glob.glob(os.path.join(motion_dir, '*.json'))}
    common = skeleton_names & handshake_names & motion_names
    return sorted(
        name for name in common
        if load_motion_json(os.path.join(motion_dir, name)).get('planned'))


class PlaybackControls(object):
    """waypoint スライダー・Play・Back/Next の GUI.

    切り替えのたびに ``on_person_change``/``on_waypoint_change`` を呼ぶ。
    Play は ``fps`` 周期で進め、最後の waypoint で止まる。
    """

    def __init__(self, viewer, fps):
        self.fps = fps
        self._lock = threading.Lock()
        self.person_index = 0
        self.n_people = 1
        self.n_waypoints = 1
        self.waypoint_index = 0
        self.on_person_change = None
        self.on_waypoint_change = None
        self._closed = False
        # viser のコールバックは並行に走り、シーングラフ操作は非スレッド
        # セーフなので、描画は必ずこのロックを握って行う。
        self._render_lock = threading.RLock()
        # 古い描画リクエストを捨てるための世代カウンタ。
        self._generation = 0

        self.back_button = viewer._server.gui.add_button('Back (人物)')
        self.next_button = viewer._server.gui.add_button('Next (人物)')
        self.waypoint_slider = viewer._server.gui.add_slider(
            'waypoint', min=0, max=0, step=1, initial_value=0)
        self.play_checkbox = viewer._server.gui.add_checkbox(
            'Play', initial_value=False)

        @self.back_button.on_click
        def _on_back(_):  # noqa: ANN001
            self._change_person(-1)

        @self.next_button.on_click
        def _on_next(_):  # noqa: ANN001
            self._change_person(1)

        @self.waypoint_slider.on_update
        def _on_waypoint(_):  # noqa: ANN001
            with self._lock:
                self.waypoint_index = int(self.waypoint_slider.value)
            self._render(self.on_waypoint_change)

        threading.Thread(target=self._play_loop, daemon=True).start()

    def _render(self, callback):
        """``callback`` を排他制御して呼ぶ。新しいリクエストに追い越されたら
        スキップする (index は呼ぶ前に更新しておくこと)。"""
        if callback is None:
            return
        with self._lock:
            self._generation += 1
            my_generation = self._generation
        with self._render_lock:
            with self._lock:
                if my_generation != self._generation:
                    return
            callback()

    def _change_person(self, direction):
        with self._lock:
            if self.n_people <= 1:
                return
            self.person_index = (
                (self.person_index + direction) % self.n_people)
            self.play_checkbox.value = False
        self._render(self.on_person_change)

    def set_person_count(self, n_people):
        self.n_people = max(1, n_people)

    def set_waypoint_count(self, n_waypoints):
        with self._lock:
            self.n_waypoints = max(1, n_waypoints)
            self.waypoint_index = 0
        self.waypoint_slider.max = self.n_waypoints - 1
        self.waypoint_slider.value = 0

    def close(self):
        self._closed = True

    def _play_loop(self):
        while not self._closed:
            time.sleep(1.0 / max(self.fps, 1e-3))
            if not self.play_checkbox.value:
                continue
            with self._lock:
                if self.waypoint_index >= self.n_waypoints - 1:
                    self.play_checkbox.value = False
                    continue
                self.waypoint_index += 1
                idx = self.waypoint_index
            # 代入で on_update が同期的に呼ばれ描画される (_render を直接
            # 呼ぶと二重描画になる)。
            self.waypoint_slider.value = idx


def follow_arm_pose(model, person, pose, human_arm, arm):
    """SMPL の差し出した腕を計画した人の腕 ``arm`` に合わせた pose を返す.

    手首から先は剛体で掌に合わせ、肩・肘は 2 リンク IK で決める。
    """
    if human_arm.hand == 'R':
        joints_ix = (smpl_body.R_SHOULDER, smpl_body.R_ELBOW,
                     smpl_body.R_WRIST)
    else:
        joints_ix = (smpl_body.L_SHOULDER, smpl_body.L_ELBOW,
                     smpl_body.L_WRIST)
    s, e, w = joints_ix
    pose = np.array(pose, dtype=np.float64).reshape(24, 3)

    def forward(p):
        _, joints = smpl_body.forward_world(
            model, p, person['betas'], person['root_pos'])
        return joints, smpl_world_rots(model, p)

    joints0, rots0 = forward(pose)
    palm_rot = np.asarray(arm['palm_rot'])
    palm_pos = np.asarray(arm['palm_position'])
    hand_rot = palm_rot @ human_arm.palm_rot0.T
    wrist_target = palm_pos + hand_rot @ (joints0[w] - human_arm.palm_pos0)
    elbow_target = sbs.two_link_elbow(
        joints0[s], wrist_target, np.linalg.norm(joints0[e] - joints0[s]),
        np.linalg.norm(joints0[w] - joints0[e]), human_arm.outward)

    def set_world(index, world_rot, rots):
        parent_world = rots[model.parent[index]]
        pose[index] = smpl_body.mat_to_axis_angle(
            smpl_body.to_smpl_rotation(parent_world.T.dot(world_rot)))

    joints, rots = joints0, rots0
    for joint, child, target in ((s, e, elbow_target), (e, w, wrist_target)):
        turn = smpl_body.rotation_between(
            sbs._unit(joints[child] - joints[joint]),
            sbs._unit(target - joints[joint]))
        set_world(joint, turn.dot(rots[joint]), rots)
        joints, rots = forward(pose)
    set_world(w, hand_rot.dot(rots0[w]), rots)
    return pose


def status_text(name, person_i, n_people, motion, display_waypoints,
                n_approach, waypoint_index, collision_text, has_post_process):
    n_display_waypoints = len(display_waypoints)
    kind = KIND_LABELS.get(motion['kind'], motion['kind'])
    verified_text = 'OK (経路全体で干渉なし)' if motion['verified'] \
        else 'NG (経路上に干渉が残る waypoint あり)'
    header = '**{}** ({}/{})  waypoint {}/{}\n\n経路: {}  検証: {}\n\n'.format(
        name, person_i + 1, n_people, waypoint_index,
        n_display_waypoints - 1, kind, verified_text)
    # 検証 OK は接近経路の干渉だけ。押し込みの可否は別。
    if not has_post_process:
        header += ('**押し込み: なし** (solve_palm_ik.py の後処理判定が解けず、'
                   '経路は hover で終わる -- 掌は合わない。実機では動かさない)'
                   '\n\n')
    transition = motion.get('transition')
    if transition is None:
        header += '**横並び移動:** 計画していない\n\n'
    else:
        header += '**横並び移動: {}** {}\n\n'.format(
            'できる' if transition['verified'] else 'できない',
            sbs.transition_summary(transition))
    if waypoint_index < n_approach:
        dist = motion['waypoint_min_distances'][waypoint_index]
        body = 'この waypoint の干渉余裕: {:+.4f} m ({})\n\n'.format(
            dist, '貫通' if dist < 0 else '干渉なし')
    elif display_waypoints[waypoint_index].get('transition'):
        wp = display_waypoints[waypoint_index]
        body = ('横並び移動 (つないだ手を体の横へ下ろしながら横並びへ)、'
                '握りの向き {:.0f} 度\n\nこの waypoint の人の腕: {}\n\n'.format(
                    wp['turn_deg'], sbs.human_arm_text(wp['human_arm'])))
    else:
        body = ('掌への押し込み (solve_palm_ik.py の後処理判定, 表示のみ '
                '-- 経路の検証対象ではない)\n\n')
    return header + body + collision_text


def main():
    parser = argparse.ArgumentParser(
        description='plan_handshake_motion.py の軌道を viser で再生する。')
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='骨格 JSON のディレクトリ。')
    parser.add_argument(
        '--handshake-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='solve_palm_ik.py の握手姿勢 JSON のディレクトリ。')
    parser.add_argument(
        '--motion-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_motion_poses'),
        help='plan_handshake_motion.py の軌道 JSON のディレクトリ。')
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
    parser.add_argument(
        '--no-hand', dest='use_hand', action='store_false',
        help='指関節なしの URDF で表示する。')
    parser.set_defaults(use_hand=True)
    parser.add_argument(
        '--fps', type=float, default=DEFAULT_PLAYBACK_FPS,
        help='自動再生の速さ [waypoint/秒]。')
    parser.add_argument('--client-wait-timeout', type=float, default=30.0,
                        help='ブラウザ接続を待つ 1 回あたりの秒数。')
    parser.add_argument('--no-open-browser', action='store_true',
                        help='ブラウザを自動で開かない。')
    args = parser.parse_args()

    names = iter_common_names(
        args.skeleton_dir, args.handshake_dir, args.motion_dir)
    if not names:
        print('{}/{}/{} の全てに対応する、軌道が計画済み (planned: true) '
              'のファイルが見つかりません。先に generate_random_human_'
              'poses.py -> solve_palm_ik.py -> plan_handshake_motion.py '
              'を実行してください。'.format(
                  args.skeleton_dir, args.handshake_dir, args.motion_dir))
        return

    models_by_gender = dict(
        load_smpl_models(args.model_path, args.female_model_path))
    robot = load_aero(use_hand=args.use_hand)

    viewer = ViserViewer(draw_grid=True)
    controls = PlaybackControls(viewer, args.fps)
    controls.set_person_count(len(names))
    label_text = viewer._server.gui.add_markdown('')

    show_collision_models_checkbox = viewer._server.gui.add_checkbox(
        '干渉回避用モデルの表示', initial_value=True)

    viewer.add(robot)
    robot_collision_overlay = build_robot_collision_overlay(robot)
    viewer.add(robot_collision_overlay)
    verification_pairs = build_collision_verification_pairs(
        robot_collision_overlay, 'r')
    # 関節スライダーは robot と overlay の姿勢を食い違わせるので消す。
    remove_joint_angle_gui(viewer)
    remove_obstacles_gui(viewer)
    viewer.show(open_browser=not args.no_open_browser)
    viewer_nav.wait_for_client(viewer, args.client_wait_timeout)
    viewer_nav.set_front_view(viewer)

    def set_link_visible(link, visible):
        common_set_link_visible(viewer, link, visible)

    @show_collision_models_checkbox.on_update
    def _on_toggle_collision_models(_):  # noqa: ANN001
        visible = show_collision_models_checkbox.value
        for link in robot_collision_overlay.link_list:
            set_link_visible(link, visible)
        for obstacle_link in current_obstacle_links:
            set_link_visible(obstacle_link, visible)

    human_mesh_handle = [None]
    current_obstacle_links = []
    current = {'name': None, 'motion': None,
              'person': None, 'model': None}

    def set_obstacles(joint_positions):
        for obstacle_link in current_obstacle_links:
            viewer.delete(obstacle_link)
        current_obstacle_links[:] = human_body_obstacles(joint_positions)
        for obstacle_link in current_obstacle_links:
            set_translucent_color(obstacle_link, COLLISION_OBSTACLE_COLOR)
            viewer.add(obstacle_link)
            set_link_visible(obstacle_link,
                             show_collision_models_checkbox.value)

    def refresh_person():
        name = names[controls.person_index]
        skeleton_path = os.path.join(args.skeleton_dir, name)
        handshake_path = os.path.join(args.handshake_dir, name)
        motion_path = os.path.join(args.motion_dir, name)

        person = load_smpl_params(skeleton_path)
        handshake = load_handshake_json(handshake_path)
        motion = load_motion_json(motion_path)

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

        display_waypoints, n_approach = build_display_waypoints(
            motion, handshake)
        # 横並び移動では人の腕をロボットの手に追従させる (sbs.HumanArm)。
        transition = motion.get('transition') or {}
        human_arm = None
        if transition.get('verified'):
            press_arm = transition['press_arm']
            human_arm = sbs.HumanArm(
                joint_positions, transition['hand'], sbs.palm_from_frame(
                    np.asarray(press_arm['palm_position']),
                    np.asarray(press_arm['palm_rot'])))
        current.update(name=name, motion=motion,
                      person=person,
                      model=model, handshake=handshake,
                      display_waypoints=display_waypoints,
                      n_approach=n_approach,
                      joint_positions=joint_positions,
                      human_arm=human_arm, obstacle_arm=None)
        set_obstacles(joint_positions)

        # set_waypoint_count は同期的に refresh_waypoint を呼び得るので、
        # current/障害物を更新し終えてから呼ぶ。
        controls.set_waypoint_count(len(display_waypoints))
        refresh_waypoint()

    def refresh_waypoint():
        motion = current['motion']
        if motion is None:
            return
        idx = controls.waypoint_index
        apply_waypoint_pose(
            robot, motion['joint_names'], current['display_waypoints'], idx)
        sync_robot_collision_overlay(robot_collision_overlay, robot)

        # 人間にロボットの手先を見させる。
        hand_move_target = getattr(
            robot, '{}arm_end_coords'.format(current['handshake']
                                             ['robot_arm']))
        gaze_target = hand_move_target.worldpos()
        pose = look_at_pose(current['model'], current['person'], gaze_target)
        # 横並び移動では人の腕 (SMPL と干渉円柱) を計画した腕に合わせる。
        arm = current['display_waypoints'][idx].get('human_arm')
        if arm is not None and current['human_arm'] is not None:
            pose = follow_arm_pose(
                current['model'], current['person'], pose,
                current['human_arm'], arm)
        if arm is not current['obstacle_arm']:
            set_obstacles(current['joint_positions'] if arm is None
                          else current['human_arm'].skeleton(arm))
            current['obstacle_arm'] = arm
        mesh = build_smpl_mesh(current['model'], current['person'], pose)
        # 作り直すと Play 中に点滅するので、メッシュは 1 つだけ作り頂点を
        # 書き換える (SMPL は男女とも同じ面構成)。
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        handle = human_mesh_handle[0]
        if handle is None or handle.faces.shape != faces.shape:
            if handle is not None:
                handle.remove()
            human_mesh_handle[0] = viewer._server.scene.add_mesh_simple(
                'smpl_human', vertices=vertices, faces=faces,
                color=tuple(SKIN_COLOR[:3]))
        else:
            handle.vertices = vertices

        colliding = colliding_link_pairs(
            robot_collision_overlay, verification_pairs,
            current_obstacle_links,
            tolerance=DEFAULT_COLLISION_VERIFY_TOLERANCE)
        label_text.content = status_text(
            current['name'], controls.person_index, controls.n_people,
            motion, current['display_waypoints'], current['n_approach'],
            idx, collision_pairs_text(colliding),
            current['handshake'].get('post_process') is not None)
        viewer.redraw()

    controls.on_person_change = refresh_person
    controls.on_waypoint_change = refresh_waypoint
    refresh_person()

    print('準備完了。ブラウザで Back/Next (人物切り替え)・waypoint '
          'スライダー・Play (自動再生) を操作してください。Ctrl-C で '
          '終了します。')
    try:
        while viewer._server.get_clients():
            time.sleep(0.2)
        print('ブラウザクライアントが切断されました。終了します。')
    except KeyboardInterrupt:
        print()
    controls.close()
    viewer.close()


if __name__ == '__main__':
    main()
