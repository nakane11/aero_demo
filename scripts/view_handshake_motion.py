#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``plan_handshake_motion.py`` が出力した軌道 (waypoint 列) JSON と、
対応する骨格 JSON・``solve_palm_ik.py`` の握手姿勢 JSON を読み込み、
scikit-robot の viser ビューアでロボットの接近動作を再生する。

``view_handshake_poses.py`` (最終姿勢 1 点だけを表示) と違い、こちらは
waypoint スライダーと Play ボタンで経路全体をコマ送り/自動再生できる。
表示するのは SMPL メッシュ・ロボットモデルに加え、``view_handshake_
poses.py`` と同じ干渉回避用の半透明ジオメトリ (人体のカプセル近似・
ロボット自身のプリミティブ近似) で、waypoint を切り替えるたびに
``solve_palm_ik.collision_pairs_min_distance`` と全く同じ厳密検証
(``plan_handshake_motion.py`` が保存した ``waypoint_min_distances``、
および実際に貫通しているリンクの組み合わせ) をテキストパネルに出す。

``planned`` が ``false`` (IK 対象外/IK 失敗で軌道が無い) の人物は
読み飛ばす。

握手姿勢 JSON に ``post_process`` (``solve_palm_ik.py`` の後処理判定:
実際に掌へわずかにめり込む位置まで腕を詰め、首を人間の手へ向ける) が
あれば、経路の最後にそこまでの補間フレームを追加で表示する
(``build_display_waypoints``)。``plan_handshake_motion.py`` はこの区間を
経路として計画・検証しない (接触そのものが目的の動きで、経路上の干渉
検証にはなじまないため) ので、あくまで見た目のための表示専用フレームで
あり、``waypoint_min_distances`` による検証の対象ではない。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo solve_palm_ik.py
    rosrun aero_demo plan_handshake_motion.py
    rosrun aero_demo view_handshake_motion.py

viser はブラウザで表示するビューアなので、実行するとブラウザが開く。
画面下の Back/Next で人物の切り替え、waypoint スライダーで経路上の
姿勢を切り替え、Play で自動再生する。
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

from aero_demo import viewer_nav  # noqa: E402

from generate_random_human_poses import load_smpl_models  # noqa: E402
from solve_palm_ik import DEFAULT_COLLISION_VERIFY_TOLERANCE  # noqa: E402
from solve_palm_ik import HUMAN_FRONT_DISTANCE  # noqa: E402
from solve_palm_ik import build_collision_verification_pairs  # noqa: E402
from solve_palm_ik import human_body_obstacles  # noqa: E402
from solve_palm_ik import human_translation_offset  # noqa: E402
from solve_palm_ik import load_skeleton_json as load_joint_positions  # noqa: E402,E501
from solve_palm_ik import translate_joint_positions  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

from view_handshake_poses import build_robot_collision_overlay  # noqa: E402
from view_handshake_poses import build_smpl_mesh  # noqa: E402
from view_handshake_poses import colliding_link_pairs  # noqa: E402
from view_handshake_poses import collision_pairs_text  # noqa: E402
from view_handshake_poses import look_at_pose  # noqa: E402
from view_handshake_poses import load_skeleton_json as load_smpl_params  # noqa: E402,E501
from view_handshake_poses import COLLISION_OBSTACLE_COLOR  # noqa: E402

from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from aero_demo.palm_plane_view import set_color as set_translucent_color  # noqa: E402,E501

from skrobot.model import Link  # noqa: E402

# 干渉回避用モデル (人体障害物・ロボット自身のプリミティブ近似) の色は
# view_handshake_poses.py と揃える (COLLISION_OBSTACLE_COLOR/
# ROBOT_COLLISION_LINK_COLOR)。

# waypoint 自動再生の既定の速さ [waypoint/秒]。
DEFAULT_PLAYBACK_FPS = 20.0

# 経路の最後に表示専用で追加する、後処理判定 (post_process = 掌への
# 押し込み) までの補間フレーム数。plan_handshake_motion.py の経路には
# 含めない (モジュール docstring 参照)。
PRESS_IN_DISPLAY_WAYPOINTS = 5

# 採用した軌道の作り方 (plan_handshake_motion.KIND_LABELS と同じ内容を
# ここでも持つ -- plan_handshake_motion は jaxls 依存で import が重い
# ため、表示用の対応表だけこちらにも複製する)。
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
    """3 つのディレクトリ全てに存在し、かつ軌道が計画済み (``planned:
    true``) のファイル名 (basename) をファイル名順に列挙する。"""
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


def apply_waypoint_pose(robot, joint_names, waypoints, waypoint_index):
    """``waypoints[waypoint_index]`` (台車位置姿勢・全身の関節角) を
    ``robot`` に反映する。``joint_names`` は ``waypoints`` の
    ``joint_angle_vector`` に対応する関節名の並び
    (``motion['joint_names']``)。"""
    wp = waypoints[waypoint_index]
    name_to_angle = dict(zip(joint_names, wp['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.base_link.newcoords(Coordinates(
        pos=wp['base_position'],
        rot=rpy_matrix(wp['base_yaw'], 0.0, 0.0)))


def build_display_waypoints(motion, handshake):
    """``motion['waypoints']`` (``plan_handshake_motion.py`` が計画・検証
    した経路) に、``handshake['post_process']`` (``solve_palm_ik.py`` の
    後処理判定) までの補間フレームを表示用に追加する
    (``PRESS_IN_DISPLAY_WAYPOINTS`` 個、台車は動かさず腕+首だけ)。

    ``post_process`` が無い (後処理判定が全ての候補で失敗し、後処理前の
    まま採用された) 場合は ``motion['waypoints']`` をそのまま返す。

    Returns
    -------
    (waypoints, n_approach)
        ``waypoints`` は表示用の waypoint リスト。``n_approach`` は
        ``motion['waypoints']`` の個数 (この添字以降が表示専用の後処理
        フレーム、``waypoint_min_distances`` による検証の対象外)。
    """
    waypoints = list(motion['waypoints'])
    n_approach = len(waypoints)
    post = handshake.get('post_process')
    if post is None:
        return waypoints, n_approach

    joint_names = motion['joint_names']
    last_wp = waypoints[-1]
    start_vec = np.asarray(last_wp['joint_angle_vector'], dtype=np.float64)
    post_name_to_angle = dict(zip(post['joint_names'],
                                  post['joint_angle_vector']))
    end_vec = np.array([post_name_to_angle.get(name, start_vec[i])
                        for i, name in enumerate(joint_names)])
    base_start = np.array([last_wp['base_position'][0],
                           last_wp['base_position'][1], last_wp['base_yaw']])
    base_end = np.array([post['base_position'][0], post['base_position'][1],
                         post['base_yaw']])

    for t in np.linspace(0.0, 1.0, PRESS_IN_DISPLAY_WAYPOINTS + 1)[1:]:
        angle_vec = start_vec + (end_vec - start_vec) * t
        base_vec = base_start + (base_end - base_start) * t
        waypoints.append(dict(
            base_position=[float(base_vec[0]), float(base_vec[1]), 0.0],
            base_yaw=float(base_vec[2]),
            joint_angle_vector=[float(v) for v in angle_vec],
        ))
    return waypoints, n_approach


def sync_robot_collision_overlay(collision_robot, robot):
    """``view_handshake_poses.sync_robot_collision_overlay`` と同じ
    (``robot`` の現在の姿勢に ``collision_robot`` を追従させる)。"""
    collision_robot.angle_vector(robot.angle_vector())
    collision_robot.newcoords(robot.base_link.copy_worldcoords())


class PlaybackControls(object):
    """waypoint スライダー・Play チェックボックス・Back/Next ボタンを
    まとめて GUI に追加し、waypoint/人物切り替えのたびに ``on_change``
    (引数無しのコールバック) を呼ぶ。

    Play 中は別スレッドが ``--fps`` の周期で waypoint を進め、最後まで
    行ったら停止する (人物を跨いでループはしない -- 経路の最後の姿勢を
    確認するのが目的のため)。
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
        # viser の GUI コールバック (on_click/on_update) はスレッドプールで
        # 実行される (スライダーを素早く操作すると複数のコールバックが
        # 同時に走る)。シーングラフの追加/削除は非スレッドセーフなので、
        # 実際に描画するコード (on_person_change/on_waypoint_change) は
        # 必ず ``_render_lock`` を握った状態でだけ呼ぶ。
        self._render_lock = threading.RLock()
        # 描画中に次のリクエストが来たら、今のリクエストは古いとみなして
        # 描画をスキップする (溜まった分だけ描き直すと点滅・二重表示の
        # 原因になる。世代カウンタで「自分がまだ最新か」を判定する)。
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
        """``callback`` (``on_person_change``/``on_waypoint_change``) を
        排他制御しつつ呼ぶ。呼び出し側は先に ``self.person_index``/
        ``self.waypoint_index`` を更新してから渡すこと -- ロック取得後に
        世代が進んでいたら (別の新しいリクエストに追い越されていたら)
        自分の描画はもう不要なのでスキップする。"""
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
            # サーバー側で .value を代入すると on_update が同じスレッドで
            # 同期的に呼ばれる (実測で確認済み) ので、これだけで
            # _on_waypoint 経由の描画が起きる (直接 _render を呼ぶと
            # 二重に描画してしまう)。
            self.waypoint_slider.value = idx


def status_text(name, person_i, n_people, motion, n_display_waypoints,
                n_approach, waypoint_index, collision_text):
    kind = KIND_LABELS.get(motion['kind'], motion['kind'])
    verified_text = 'OK (経路全体で干渉なし)' if motion['verified'] \
        else 'NG (経路上に干渉が残る waypoint あり)'
    header = '**{}** ({}/{})  waypoint {}/{}\n\n経路: {}  検証: {}\n\n'.format(
        name, person_i + 1, n_people, waypoint_index,
        n_display_waypoints - 1, kind, verified_text)
    if waypoint_index < n_approach:
        dist = motion['waypoint_min_distances'][waypoint_index]
        body = 'この waypoint の干渉余裕: {:+.4f} m ({})\n\n'.format(
            dist, '貫通' if dist < 0 else '干渉なし')
    else:
        body = ('掌への押し込み (solve_palm_ik.py の後処理判定, 表示のみ '
                '-- 経路の検証対象ではない)\n\n')
    return header + body + collision_text


def main():
    parser = argparse.ArgumentParser(
        description='plan_handshake_motion.py が出力した軌道 (waypoint '
                    '列) を、対応する骨格・握手姿勢 JSON と合わせて '
                    'viser で再生する。')
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='SMPL pose/betas/root_pos と全身関節位置を持つ骨格 JSON の '
            'ディレクトリ (既定 random_human_poses/)。')
    parser.add_argument(
        '--handshake-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='solve_palm_ik.py が出力した握手姿勢 JSON のディレクトリ '
            '(既定 random_handshake_poses/)。')
    parser.add_argument(
        '--motion-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_motion_poses'),
        help='plan_handshake_motion.py が出力した軌道 JSON のディレクトリ '
            '(既定 random_motion_poses/)。')
    parser.add_argument(
        '--human-front-distance', type=float, default=HUMAN_FRONT_DISTANCE,
        help='solve_palm_ik.py/plan_handshake_motion.py の '
            '--human-front-distance と同じ値を渡す (既定 {:.1f})。'.format(
                HUMAN_FRONT_DISTANCE))
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
        help='SMPL (女性) モデル .pkl のパス (無ければ男性モデルのみ)。')
    parser.add_argument(
        '--no-hand', dest='use_hand', action='store_false',
        help='指関節なしの URDF を使う (既定は指関節ありの URDF)。')
    parser.set_defaults(use_hand=True)
    parser.add_argument(
        '--fps', type=float, default=DEFAULT_PLAYBACK_FPS,
        help='Play チェックボックスをオンにしたときの自動再生の速さ '
            '[waypoint/秒] (既定 {})。'.format(DEFAULT_PLAYBACK_FPS))
    parser.add_argument('--client-wait-timeout', type=float, default=30.0,
                        help='ブラウザクライアント接続を待つ 1 回あたりの '
                             '秒数 (繰り返し待つ)。')
    parser.add_argument('--no-open-browser', action='store_true',
                        help='ブラウザの自動起動を無効にする。')
    parser.add_argument(
        '--collision-primitive-type', choices=['box', 'cylinder', 'sphere'],
        default=None,
        help='ロボット自身の干渉モデルの元になるジオメトリを、指定した '
            '形状に全リンク強制変換する (solve_palm_ik.py / '
            'plan_handshake_motion.py と同じオプション)。')
    parser.add_argument(
        '--force-convert-collision-model', action='store_true',
        help='ロボット自身の干渉モデル (プリミティブ近似 URDF) のキャッシュ '
            'を使わず毎回作り直す。')
    parser.add_argument(
        '--collision-verify-tolerance', type=float,
        default=DEFAULT_COLLISION_VERIFY_TOLERANCE,
        help='表示中の干渉をテキストパネルに列挙する距離の許容誤差 [m] '
            '(既定 {})。'.format(DEFAULT_COLLISION_VERIFY_TOLERANCE))
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
    robot_collision_overlay = build_robot_collision_overlay(
        robot,
        primitive_type=args.collision_primitive_type,
        force_convert=args.force_convert_collision_model)
    viewer.add(robot_collision_overlay)
    verification_pairs = build_collision_verification_pairs(
        robot_collision_overlay, 'r')
    viewer.show(open_browser=not args.no_open_browser)
    viewer_nav.wait_for_client(viewer, args.client_wait_timeout)
    viewer_nav.set_front_view(viewer)

    def set_link_visible(link, visible):
        handle = viewer._linkid_to_handle.get(str(id(link)))
        if handle is not None:
            handle.visible = visible

    @show_collision_models_checkbox.on_update
    def _on_toggle_collision_models(_):  # noqa: ANN001
        visible = show_collision_models_checkbox.value
        for link in robot_collision_overlay.link_list:
            set_link_visible(link, visible)
        for obstacle_link in current_obstacle_links:
            set_link_visible(obstacle_link, visible)

    current_mesh_link = [None]
    current_obstacle_links = []
    current = {'name': None, 'motion': None, 'joint_positions': None,
              'person': None, 'model': None}

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
            joint_positions, front_distance=args.human_front_distance)
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
        current.update(name=name, motion=motion,
                      joint_positions=joint_positions, person=person,
                      model=model, handshake=handshake,
                      display_waypoints=display_waypoints,
                      n_approach=n_approach)

        for obstacle_link in current_obstacle_links:
            viewer.delete(obstacle_link)
        current_obstacle_links[:] = human_body_obstacles(joint_positions)
        for obstacle_link in current_obstacle_links:
            set_translucent_color(obstacle_link, COLLISION_OBSTACLE_COLOR)
            viewer.add(obstacle_link)
            set_link_visible(obstacle_link,
                             show_collision_models_checkbox.value)

        # set_waypoint_count は waypoint スライダーの value を 0 に戻す
        # ため、前の値が 0 でなければ on_update (_on_waypoint) がこの場で
        # 同期的に発火し、_render 経由で refresh_waypoint が呼ばれ得る
        # (RLock なので再入自体は安全)。current/obstacle をすべて更新し
        # 終えた後でここに置くのはそのため -- 先に呼ぶと、古い人物の
        # 障害物のまま新しい人物を描いてしまう瞬間ができる。
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

        # ロボットの手先 (この waypoint の実際の位置) を人間に見させる。
        hand_move_target = getattr(
            robot, '{}arm_end_coords'.format(current['handshake']
                                             ['robot_arm']))
        gaze_target = hand_move_target.worldpos()
        pose = look_at_pose(current['model'], current['person'], gaze_target)
        mesh = build_smpl_mesh(current['model'], current['person'], pose)
        link = Link(visual_mesh=mesh, name='smpl_human')
        if current_mesh_link[0] is not None:
            viewer.delete(current_mesh_link[0])
        viewer.add(link)
        current_mesh_link[0] = link

        colliding = colliding_link_pairs(
            robot_collision_overlay, verification_pairs,
            current['joint_positions'],
            tolerance=args.collision_verify_tolerance)
        label_text.content = status_text(
            current['name'], controls.person_index, controls.n_people,
            motion, len(current['display_waypoints']), current['n_approach'],
            idx, collision_pairs_text(colliding))
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
