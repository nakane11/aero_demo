#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""握手デモの表示ツール (viser)。

表示は 2 つ:

- **シミュレーション**: 計画した軌道 (lead-in・接近・押し込み・横並び移動) を、実機と
  同じ速度上限・同じ補間で時間どおりに動かす (``aero_demo.handshake_timeline``)。
  ROS は要らない。
- **実機**: ``/joint_states`` と odom の TF で実機の姿勢を、実機ノード
  (``run_camera_pipeline_test.py``) の ``~status`` で今の段階を、``~skeleton_image``
  でカメラ画像を出す。ROS が要る (実機か rosbag)。

Usage
-----
合成骨格・保存した試行 (ROS 不要)::

    python3 scripts/handshake_viewer.py --work-dir /tmp/aero_demo_pipeline_xxxx
    python3 scripts/handshake_viewer.py --work-dir data/real/2026-10-21/trial3_side_by_side

実機・rosbag (実機ノードと同じ ROS マスター)::

    python3 scripts/handshake_viewer.py --ros
"""

import argparse
import bisect
import glob
import html
import json
import os
import sys
import threading
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
for _path in (_PKG_SRC_DIR, _THIS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aero_demo import handshake_stages as stages  # noqa: E402
from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo import viewer_nav  # noqa: E402
from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from aero_demo.handshake_timeline import PlanTimeline  # noqa: E402
from aero_demo.palm_plane_view import set_color as set_translucent_color  # noqa: E402,E501

from handshake_viewer_common import HUMAN_COLLISION_OBSTACLE_COLOR  # noqa: E402
from handshake_viewer_common import apply_pose  # noqa: E402
from handshake_viewer_common import build_display_waypoints  # noqa: E402
from handshake_viewer_common import build_robot_collision_overlay  # noqa: E402
from handshake_viewer_common import remove_joint_angle_gui  # noqa: E402
from handshake_viewer_common import remove_obstacles_gui  # noqa: E402
from handshake_viewer_common import set_link_visible  # noqa: E402
from handshake_viewer_common import sync_robot_collision_overlay  # noqa: E402
import plan_handshake_motion as phm  # noqa: E402
import side_by_side_transition as sbs  # noqa: E402
import solve_palm_ik as spik  # noqa: E402
from view_handshake_motion import follow_arm_pose  # noqa: E402
from view_handshake_poses import SKIN_COLOR  # noqa: E402
from view_handshake_poses import build_smpl_mesh  # noqa: E402
from view_handshake_poses import look_at_pose  # noqa: E402
from view_handshake_poses import load_skeleton_json as load_smpl_params  # noqa: E402,E501

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

MODE_REAL = '実機'
MODE_SIMULATION = 'シミュレーション'
SPEEDS = {'0.5 倍': 0.5, '1 倍': 1.0, '2 倍': 2.0, '4 倍': 4.0}
# 画面の更新周期 [秒]。
UPDATE_PERIOD = 0.05
# 状態パネルのカメラ画像の幅 [px] (初期値・スライダーの範囲) と JPEG 品質・更新周期。
CAMERA_WIDTH = 400
CAMERA_WIDTH_RANGE = (240, 1280)
CAMERA_JPEG_QUALITY = 70
CAMERA_PERIOD = 0.2
# 状態パネルの幅のうち画像以外 (余白) [px]。
PANEL_MARGIN = 40
# 実機表示の記録に残すカメラ画像の幅 [px]。
RECORD_IMAGE_WIDTH = 480
# 状態パネルの一覧の文字 (今の状態・その補足・ほかの状態)。
STATE_CURRENT_STYLE = ('font-size:20px;font-weight:700;color:#111;'
                       'margin:6px 0 2px')
STATE_SUB_STYLE = 'font-size:13px;color:#333;margin:0 0 4px 1.6em'
STATE_OTHER_STYLE = 'font-size:11px;color:#b0b0b0;line-height:1.5'
# ~status がこれより古ければ実機ノードが止まっているとみなす [秒]。
STATUS_STALE_TIME = 3.0
# ライブの骨格 (計画前) を描き直す周期 [秒] (ちらつきと負荷を抑える)。
LIVE_SKELETON_PERIOD = 0.5

# 計画の経路の区間 (それぞれ表示用 waypoint の範囲を持つ)。
PATH_LEAD_IN = 'lead_in'
PHASE_APPROACH = 'approach'
PHASE_PRESS = 'press'
PHASE_TRANSITION = 'transition'
# 経路の色 (RGB)。今通っている区間だけ台車は緑・手先は赤で太く、ほかは灰で細く。
PATH_COLORS = {'base': (30, 170, 60), 'hand': (220, 30, 30)}
PATH_IDLE_COLOR = (170, 170, 170)
PATH_WIDTH_CURRENT = 6.0
PATH_WIDTH_IDLE = 1.5
# 台車の経路を描く高さ [m] (床の格子に埋もれないよう少し浮かせる)。
PATH_FLOOR_HEIGHT = 0.01
# 実機表示で描く、実機が実際に通った軌跡の線の太さ [px]、間引く間隔 [m]、
# 描き直す間隔 [フレーム]。
TRACE_WIDTH = 4.0
TRACE_MIN_STEP = 0.005
TRACE_REDRAW_FRAMES = 5
# waypoint の番号を通過したとみなす誤差。
PROGRESS_EPS = 1e-3
# 目印の球: (半径 [m], 色) を、次の目標・これから・通過したもので。
MARKER_STYLES = {
    'next': (0.03, (30, 30, 30)),
    'future': (0.018, (120, 120, 120)),
    'passed': (0.012, (215, 215, 215)),
}
# 次の目標のラベルの大きさ (画面上の文字の倍率)。
NEXT_LABEL_SCALE = 2.0
# 実機ノードの状態のうち、台車・腕が経路の上を動いているもの。
MOVING_STAGES = {stages.STAGE_LEAD_IN[0], stages.STAGE_SPIRAL[0],
                 stages.STAGE_PRESS_IN[0], stages.STAGE_TRANSITION[0]}


class Plan(object):
    """表示する 1 試行分の計画 (骨格・IK 結果・軌道・表示用 waypoint)。

    座標は計画の系 (実機の試行は差し出し手が決まった時点の base_link、合成骨格は
    人を前方へ平行移動した系)。ロボットは原点の腕を下ろした姿勢から動き出す。
    """

    def __init__(self, name, joint_positions, result, motion, robot,
                 person=None, model=None, display_waypoints=None,
                 n_prepend=None, n_approach=None, n_transition=None,
                 real_pose_ref=None, torso_surface_offset=None):
        self.name = name
        self.joint_positions = joint_positions
        self.result = result
        self.motion = motion
        self.person = person
        self.model = model
        self.real_pose_ref = real_pose_ref
        # 干渉円柱は計画と同じく体幹を奥へずらす (SMPL の合成骨格はずらさない)。
        self.torso_surface_offset = (
            0.0 if person is not None else
            spik.DEFAULT_TORSO_SURFACE_OFFSET if torso_surface_offset is None
            else torso_surface_offset)
        self.timeline = None
        self.human_arm = None
        self.paths = None    # compute_paths で作る
        self.markers = []
        self.display_waypoints = display_waypoints
        if motion is None:
            return
        if display_waypoints is None:
            display_waypoints, n_approach = build_display_waypoints(
                motion, result)
            n_transition = sum(1 for wp in display_waypoints
                               if wp.get('transition'))
            start_angles = dict(zip(
                [joint.name for joint in robot.joint_list],
                phm.arms_down_angles(robot, robot.joint_list)))
            prepend = phm.build_lead_in_waypoints(
                [0.0, 0.0, 0.0], display_waypoints[0], motion['joint_names'],
                start_joint_angles=start_angles)
            n_prepend = len(prepend)
            display_waypoints = prepend + display_waypoints
        self.display_waypoints = display_waypoints
        self.n_prepend = n_prepend
        self.n_approach = n_approach
        self.n_transition = n_transition
        joints = {joint.name: joint for joint in robot.joint_list}
        self.timeline = PlanTimeline(
            display_waypoints, n_prepend, n_approach, n_transition,
            motion['joint_names'], joints)
        transition = motion.get('transition') or {}
        if transition.get('verified'):
            press_arm = transition['press_arm']
            self.human_arm = sbs.HumanArm(
                joint_positions, transition['hand'], sbs.palm_from_frame(
                    np.asarray(press_arm['palm_position']),
                    np.asarray(press_arm['palm_rot'])))

    def summary(self):
        """計画の一行の要約。"""
        result, motion = self.result, self.motion
        if not result.get('solved'):
            return 'IK 失敗'
        if motion is None:
            return '軌道なし'
        transition = motion.get('transition')
        return '横並び移動まで計画{}'.format(
            'できた' if transition and transition.get('verified')
            else 'できていない')

    def computation_text(self):
        """計算を行った順の結果と計算時間 (Markdown の箇条書き)。"""
        result, motion = self.result, self.motion
        post = result.get('post_process')
        post_time = None if post is None else post.get('compute_time')
        ik_time = result.get('collision_ik_time')
        selection_time = result.get('candidate_selection_time')
        lines = ['1. **hover の IK** (第1段バッチ IK、台車も): {} — {}'.format(
            _seconds(ik_time), '解あり' if result.get('solved') else '解けない')]
        if not result.get('solved'):
            return '\n'.join(lines)
        # 候補選びの時間には、採った候補の押し込みの IK も入っている。
        verify_time = (selection_time - post_time
                       if selection_time is not None and post_time is not None
                       else selection_time)
        margin = result.get('base_x_standing_margin')
        lines.append(
            '2. **hover の事後検証・候補選び**: {} — 台車の前後の窓 {}'.format(
                _seconds(verify_time),
                '無制限' if margin is None else '±{:.2f} m'.format(margin)))
        if post is None:
            lines.append('3. **押し込み + 視線の IK**: 解けない')
            return '\n'.join(lines)
        lines.append('3. **押し込み + 視線の IK**: {} — {}'.format(
            _seconds(post_time), '腕だけで解いた (視線は近い向き)'
            if post.get('arm_only') else '解けた'))
        if motion is None:
            lines.append('4. **接近軌道の計画**: しない ({})'.format(
                '計画の時間の上限' if result.get('plan_timed_out')
                else '軌道なし'))
            return '\n'.join(lines)
        kind = phm.KIND_LABELS.get(motion['kind'], motion['kind'])
        lines.append(
            '4. **接近軌道の計画**: {} — {} (経路の候補 {} 個、接近開始位置の'
            '候補 {} 個)'.format(
                _seconds(motion.get('compute_time')), kind,
                len(motion.get('candidates_tried') or []),
                motion.get('approach_candidates_tried', 1)))
        margins = [m for m in motion.get('waypoint_human_clearance_margins')
                   or [] if m is not None]
        lines.append(
            '5. **接近の干渉検証** (4 に含む): 経路 {} / lead-in {} — 干渉余裕の'
            '最小 {:+.1f} mm{}'.format(
                'OK' if motion['verified'] else 'NG',
                'OK' if motion.get('lead_in_verified', True) else 'NG',
                1e3 * min(motion['waypoint_min_distances']),
                '、人体距離の余裕の最小 {:+.1f} cm'.format(100 * min(margins))
                if margins else ''))
        transition = motion.get('transition')
        if transition is None:
            lines.append('6. **横並び移動**: 計画していない')
        else:
            lines.append('6. **横並び移動**: {} (最初の解 {}) — {}{}'.format(
                _seconds(transition.get('compute_time')),
                _seconds(transition.get('first_verified_time')),
                '移動先の {:.0f}% まで'.format(100 * transition['fraction'])
                if transition.get('verified')
                else 'できない ({})'.format(transition.get('reason')),
                '' if not transition.get('verified')
                else '、' + _chosen_solution_text(transition)))
            refined = motion.get('transition_refined')
            if refined is not None:
                lines.append('   - 裏の計画 (寄せ): {} — {}'.format(
                    _seconds(refined.get('compute_time')),
                    '移動先の {:.0f}% まで、{}'.format(
                        100 * refined['fraction'],
                        _chosen_solution_text(refined))
                    if refined.get('verified') else 'できない'))
        total = sum(v for v in (
            ik_time, selection_time, motion.get('compute_time'),
            None if transition is None else transition.get('compute_time'))
            if v is not None)
        lines.append('\n合計 {}'.format(_seconds(total)))
        return '\n'.join(lines)

    def compute_paths(self, robot):
        """表示する経路 (台車・ロボットの手先) と目印を作る (``robot`` の姿勢を
        書き換える)。区間ごとに ``{区間: {'base': 点列, 'hand': 点列}}``。"""
        self.paths = None
        if self.timeline is None:
            return
        waypoints = self.display_waypoints
        joint_names = self.motion['joint_names']
        end = getattr(robot, '{}arm_end_coords'.format(
            self.result['robot_arm']))
        bases, hands = [], []
        for wp in waypoints:
            base = (wp['base_position'][0], wp['base_position'][1],
                    wp['base_yaw'])
            apply_pose(robot, joint_names, wp['joint_angle_vector'], base)
            bases.append([base[0], base[1], PATH_FLOOR_HEIGHT])
            hands.append(end.worldpos().copy())
        bases = np.asarray(bases)
        hands = np.asarray(hands)
        n_total = len(waypoints)
        n_body = n_total - self.n_transition
        reach = min(max(self.n_prepend + self.n_approach, 1), n_body)
        ranges = {
            PATH_LEAD_IN: (0, self.n_prepend),
            PHASE_APPROACH: (self.n_prepend, reach - 1),
            PHASE_PRESS: (reach - 1, n_body - 1),
            PHASE_TRANSITION: (n_body - 1, n_total - 1),
        }
        paths = {key: dict(base=bases[start:stop + 1],
                           hand=hands[start:stop + 1])
                 for key, (start, stop) in ranges.items() if stop > start}
        # 目印 (ラベル, 表示用 waypoint の番号)。
        markers = [('接近開始', self.n_prepend), ('hover', reach - 1)]
        if reach < n_body:
            markers.append(('押し込み', n_body - 1))
        if self.n_transition:
            markers.append(('横並び', n_total - 1))
        self.paths = paths
        self.path_ranges = ranges
        self.markers = [
            (label, index, (waypoints[index]['base_position'][0],
                            waypoints[index]['base_position'][1],
                            waypoints[index]['base_yaw']))
            for label, index in markers]

    def progress_for_stage(self, title):
        """実機ノードの段階 (見出し) から、表示用 waypoint の上でどこまで進んだか
        (番号、小数)。実行前の段階は 0。"""
        if self.paths is None:
            return 0.0
        ranges = self.path_ranges

        def middle(key):
            start, stop = ranges.get(key, (0, 0))
            return 0.5 * (start + stop)

        n_total = len(self.display_waypoints)
        hover = ranges[PHASE_APPROACH][1]
        press_end = ranges.get(PHASE_PRESS, (hover, hover))[1]
        progress = {
            stages.STAGE_LEAD_IN[0]: middle(PATH_LEAD_IN),
            stages.STAGE_SPIRAL[0]: middle(PHASE_APPROACH),
            stages.STAGE_BASE_CORRECTION[0]: hover,
            stages.STAGE_HOVER_SETTLE[0]: hover,
            stages.STAGE_PRESS_REFINE[0]: hover,
            stages.STAGE_PRESS_IN[0]: middle(PHASE_PRESS),
            stages.STAGE_GRASP[0]: press_end,
            stages.STAGE_TRANSITION_WAIT[0]: press_end,
            stages.STAGE_TRANSITION[0]: middle(PHASE_TRANSITION),
            stages.STAGE_DONE[0]: n_total - 1,
        }
        return progress.get(title, 0.0)


def _seconds(value):
    return '—' if value is None else '{:.2f} 秒'.format(value)


def _chosen_solution_text(transition):
    """横並び移動で採った解が、最初に見つかった解かどうか。"""
    chosen = transition.get('chosen_try')
    first = transition.get('first_verified_try')
    if chosen is None:
        return '採った解: 記録なし (記録を足す前の計画)'
    if chosen == first:
        return '**最初の解を採用** (探索 {} 回)'.format(transition.get('tries'))
    return ('**最初の解ではなく {} 回目の解を採用** ({} に見つけた解。最初の解は'
            ' {} 回目、探索 {} 回)'.format(
                chosen, _seconds(transition.get('chosen_time')), first,
                transition.get('tries')))


def load_smpl_models_if_any(args):
    """SMPL のモデル (無ければ空 dict。合成骨格の人のメッシュだけに使う)。"""
    try:
        from generate_random_human_poses import load_smpl_models
        return dict(load_smpl_models(args.model_path, args.female_model_path))
    except Exception as e:  # noqa: BLE001 (メッシュ無しでも表示はできる)
        print('[viewer] SMPL のモデルを読めないため、人は骨格と干渉円柱だけで'
              '表示します ({})。'.format(e))
        return {}


class FileSource(object):
    """``skeletons/``・``handshakes/``・``motions/`` の 3 ディレクトリ
    (run_pipeline_test.py の作業ディレクトリ、実機ノードの --save-dir) から読む。"""

    def __init__(self, skeleton_dir, handshake_dir, motion_dir, robot, args):
        self.dirs = (skeleton_dir, handshake_dir, motion_dir)
        self.robot = robot
        self.args = args
        names = [set(os.path.basename(p) for p in glob.glob(
            os.path.join(d, '*.json'))) for d in self.dirs]
        self.names = sorted(names[0] & names[1] & names[2])
        self._models = None

    def load(self, index):
        name = self.names[index]
        skeleton_path, handshake_path, motion_path = (
            os.path.join(d, name) for d in self.dirs)
        joint_positions = spik.load_skeleton_json(skeleton_path)
        try:
            person = load_smpl_params(skeleton_path)
        except KeyError:
            person = None  # 実カメラの骨格 (SMPL なし、ロボットの座標のまま)
        model = None
        if person is not None:
            # 合成骨格は計画と同じく人を前方へ平行移動する。
            offset = spik.human_translation_offset(
                joint_positions, front_distance=spik.HUMAN_FRONT_DISTANCE)
            joint_positions = spik.translate_joint_positions(
                joint_positions, offset)
            person = dict(person)
            root_pos = np.array(person['root_pos'], dtype=np.float64)
            root_pos[:2] += offset
            person['root_pos'] = root_pos
            if self._models is None:
                self._models = load_smpl_models_if_any(self.args)
            model = self._models.get(person['gender'],
                                     self._models.get('male'))
            if model is None:
                person = None
        result = _load_json(handshake_path)
        motion = _load_json(motion_path)
        if not motion.get('planned', True):
            motion = None
        return Plan(name, joint_positions, result, motion, self.robot,
                    person=person, model=model)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


class Scene(object):
    """ロボット・人 (骨格の線・干渉円柱・SMPL メッシュ)・押し込みの目標線。"""

    def __init__(self, viewer, robot, collision_checkbox):
        self.viewer = viewer
        self.robot = robot
        self.collision_checkbox = collision_checkbox
        self.overlay = build_robot_collision_overlay(robot)
        self.lock = threading.RLock()
        self._skeleton_links = []
        self._obstacle_links = []
        self._human_key = None
        self._mesh = None
        self._path_handles = []
        self._paths_key = None
        self._trace_handles = []
        self._trace_key = None

    def add_robot(self):
        self.viewer.add(self.robot)
        self.viewer.add(self.overlay)

    def set_collision_visible(self, visible):
        with self.lock:
            for link in self.overlay.link_list + self._obstacle_links:
                set_link_visible(self.viewer, link, visible)

    def set_human(self, joint_positions, torso_surface_offset=0.0,
                  draw_skeleton=True, key=None):
        """人の骨格の線と干渉円柱を作り直す (``key`` が前回と同じなら何もしない)。"""
        with self.lock:
            if key is not None and key == self._human_key:
                return
            self._human_key = key
            for link in self._skeleton_links + self._obstacle_links:
                self.viewer.delete(link)
            self._skeleton_links = []
            self._obstacle_links = []
            if joint_positions is None:
                return
            joint_positions = {k: np.asarray(v, dtype=np.float64)
                               for k, v in joint_positions.items()}
            if draw_skeleton:
                self._skeleton_links = skeleton_drawing.build_skeleton_links(
                    joint_positions)
                for link in self._skeleton_links:
                    self.viewer.add(link)
            obstacle_joints = joint_positions
            if torso_surface_offset:
                obstacle_joints = spik.shift_torso_joints_from_surface(
                    joint_positions, (0.0, 0.0), torso_surface_offset)
            self._obstacle_links = spik.human_body_obstacles(obstacle_joints)
            for link in self._obstacle_links:
                set_translucent_color(link, HUMAN_COLLISION_OBSTACLE_COLOR)
                self.viewer.add(link)
                set_link_visible(self.viewer, link,
                                 self.collision_checkbox.value)

    def set_smpl(self, model, person, pose):
        """SMPL の人のメッシュ (``model`` が None なら消す)。"""
        with self.lock:
            if model is None:
                if self._mesh is not None:
                    self._mesh.remove()
                    self._mesh = None
                return
            mesh = build_smpl_mesh(model, person, pose)
            vertices = np.asarray(mesh.vertices, dtype=np.float32)
            faces = np.asarray(mesh.faces, dtype=np.uint32)
            if self._mesh is None or self._mesh.faces.shape != faces.shape:
                if self._mesh is not None:
                    self._mesh.remove()
                self._mesh = self.viewer._server.scene.add_mesh_simple(
                    'smpl_human', vertices=vertices, faces=faces,
                    color=tuple(SKIN_COLOR[:3]))
            else:
                # 再生中の点滅を避けるため頂点だけ書き換える。
                self._mesh.vertices = vertices

    def set_plan_paths(self, plan, progress, moving, visible):
        """計画の経路 (台車・手先) と目印の球を描く。

        ``progress`` は表示用 waypoint の上でどこまで進んだか (番号、小数)、
        ``moving`` は経路の上を動いている途中か。今通っている区間だけ台車は緑・
        手先は赤で太く、ほかは灰で細く描く。目印は次の目標だけ大きく濃く
        ラベルを付け、通過したものは薄く小さく描く。前回と同じなら何もしない。
        """
        if plan is None or plan.paths is None or not visible:
            states = None
        else:
            current = (_current_path(plan.path_ranges, plan.paths, progress)
                       if moving else None)
            states = (current, _next_marker(plan.markers, progress),
                      tuple(index <= progress + PROGRESS_EPS
                            for _, index, _ in plan.markers))
        key = (None if plan is None else id(plan), states)
        with self.lock:
            if key == self._paths_key:
                return
            self._paths_key = key
            for handle in self._path_handles:
                handle.remove()
            self._path_handles = []
            if states is None:
                return
            current, next_marker, passed = states
            scene = self.viewer._server.scene
            for name, path in plan.paths.items():
                for part, points in path.items():
                    points = np.asarray(points, dtype=np.float32)
                    if len(points) < 2:
                        continue
                    active = name == current
                    self._path_handles.append(scene.add_line_segments(
                        '/plan_paths/{}_{}'.format(name, part),
                        points=np.stack([points[:-1], points[1:]], axis=1),
                        colors=PATH_COLORS[part] if active else PATH_IDLE_COLOR,
                        line_width=(PATH_WIDTH_CURRENT if active
                                    else PATH_WIDTH_IDLE)))
            for i, (label, _, (x, y, _)) in enumerate(plan.markers):
                style = ('next' if i == next_marker
                         else 'passed' if passed[i] else 'future')
                radius, color = MARKER_STYLES[style]
                self._path_handles.append(scene.add_icosphere(
                    '/plan_markers/{}'.format(i), radius=radius, color=color,
                    position=(x, y, radius)))
                if style == 'next':
                    self._path_handles.append(scene.add_label(
                        '/plan_labels/{}'.format(i), '次: {}'.format(label),
                        position=(x, y, 2.0 * radius + 0.05),
                        font_screen_scale=NEXT_LABEL_SCALE,
                        anchor='bottom-center'))

    def set_real_trace(self, base, hand, key):
        """実機が実際に通った台車 (緑) と手先 (赤) の軌跡を描く (``key`` が前回と
        同じなら何もしない。``base`` が None なら消す)。"""
        with self.lock:
            if key == self._trace_key:
                return
            self._trace_key = key
            for handle in self._trace_handles:
                handle.remove()
            self._trace_handles = []
            if base is None:
                return
            scene = self.viewer._server.scene
            for part, points in (('base', base), ('hand', hand)):
                points = _thin_trace(np.asarray(points, dtype=np.float32))
                if len(points) < 2:
                    continue
                self._trace_handles.append(scene.add_line_segments(
                    '/real_trace/{}'.format(part),
                    points=np.stack([points[:-1], points[1:]], axis=1),
                    colors=PATH_COLORS[part], line_width=TRACE_WIDTH))

    def set_robot_pose(self, joint_names, angles, base):
        with self.lock:
            apply_pose(self.robot, joint_names, angles, base)
            sync_robot_collision_overlay(self.overlay, self.robot)

    def set_robot_matrix(self, joint_angles, world_from_base):
        """関節角 ``{名前: 角度}`` と台車の 4x4 を反映する。"""
        with self.lock:
            for joint in self.robot.joint_list:
                if joint.name in joint_angles:
                    joint.joint_angle(joint_angles[joint.name])
            self.robot.base_link.newcoords(Coordinates(
                pos=world_from_base[:3, 3], rot=world_from_base[:3, :3]))
            sync_robot_collision_overlay(self.overlay, self.robot)

    def set_robot_initial(self):
        with self.lock:
            self.robot.reset_pose()
            phm.arms_down_angles(self.robot, self.robot.joint_list)
            self.robot.base_link.newcoords(Coordinates())
            sync_robot_collision_overlay(self.overlay, self.robot)

    def redraw(self):
        with self.lock:
            self.viewer.redraw()


def _thin_trace(points):
    """軌跡の点を、前に残した点から ``TRACE_MIN_STEP`` 以上動いたものだけに
    間引く (最後の点は残す)。"""
    if len(points) < 3:
        return points
    kept = [points[0]]
    for point in points[1:-1]:
        if np.linalg.norm(point - kept[-1]) >= TRACE_MIN_STEP:
            kept.append(point)
    kept.append(points[-1])
    return np.asarray(kept)


def _current_path(ranges, paths, progress):
    """``progress`` (表示用 waypoint の番号、小数) を含む経路の区間の名前
    (無ければ None)。"""
    for name in paths:
        start, stop = ranges[name]
        if start - PROGRESS_EPS <= progress < stop - PROGRESS_EPS:
            return name
    return None


def _next_marker(markers, progress):
    """まだ通過していない最初の目印の番号 (全部通過したら None)。"""
    for i, (_, index, _) in enumerate(markers):
        if index > progress + PROGRESS_EPS:
            return i
    return None


class StatusPanel(object):
    """3D の画角によらない左上の状態パネル (今の段階とカメラ画像)。"""

    def __init__(self, viewer, with_camera):
        gui = viewer._server.gui
        self.panel = gui.add_panel()
        self.width = CAMERA_WIDTH
        self.image = None
        self._content = None
        self._notes = None
        self._image_version = None
        self._image_time = 0.0
        with self.panel.add_tab('ロボットの状態'):
            # 全状態の一覧 (今の状態だけ大きく濃く) と、その下の補足。
            self.text = gui.add_html('')
            self.notes = gui.add_markdown('')
            if with_camera:
                # 骨格を重ねた画像 (計算中は骨格を推定しないので生画像)。
                self.image = gui.add_image(
                    np.zeros((CAMERA_WIDTH * 3 // 4, CAMERA_WIDTH, 3),
                             dtype=np.uint8),
                    format='jpeg', jpeg_quality=CAMERA_JPEG_QUALITY)
                slider = gui.add_slider(
                    '画像の幅 [px]', min=CAMERA_WIDTH_RANGE[0],
                    max=CAMERA_WIDTH_RANGE[1], step=40,
                    initial_value=CAMERA_WIDTH)

                @slider.on_update
                def _on_width(_):  # noqa: ANN001
                    self.width = int(slider.value)
                    self._image_version = None  # 新しい幅ですぐ送り直す
                    self.panel.set_width(self.width + PANEL_MARGIN)
        self.panel.float(x=15, y=15, width=CAMERA_WIDTH + PANEL_MARGIN)

    def set_state(self, stage, info_lines=(), notes=''):
        """今の状態 ``stage`` ((見出し, 補足)) を全状態の一覧の中で強調して出す。
        ``info_lines`` は今の状態の下に添える行、``notes`` は一覧の下の
        Markdown。変化したときだけ送る。"""
        content = state_list_html(stage, info_lines)
        if content != self._content:
            self.text.content = content
            self._content = content
        if notes != self._notes:
            self.notes.content = notes
            self._notes = notes

    def set_image(self, version, image):
        """BGR の ``image`` を送る (版が変わらなければ・間隔が短ければ送らない)。"""
        now = time.time()
        if (self.image is None or image is None
                or version == self._image_version
                or now - self._image_time < CAMERA_PERIOD):
            return
        import cv2
        if self.width < image.shape[1]:
            height = int(round(image.shape[0] * self.width
                               / float(image.shape[1])))
            image = cv2.resize(image, (self.width, height),
                               interpolation=cv2.INTER_AREA)
        self.image.image = image[:, :, ::-1]
        self._image_version = version
        self._image_time = now


def state_list_html(stage, info_lines=()):
    """ロボットの状態 (``handshake_stages.STATE_GROUPS``) を番号付きで縦に並べ、
    今の処理 ``stage`` が属する状態だけ大きく濃く、``info_lines`` を添えた HTML。
    どの状態にも属さない処理 (実機ノードを待っているなど) は一覧の上に補足付きで
    出す (処理の細かい内容は右のパネル)。"""
    title, detail = stage
    state = stages.state_of(title)
    current = None if state is None else state[0]
    lines = list(info_lines) if state is not None else [detail] + list(
        info_lines)
    sub = ''.join(
        '<div style="{}">{}</div>'.format(STATE_SUB_STYLE, html.escape(line))
        for line in lines if line)
    rows = []
    if current is None:
        rows.append('<div style="{}">{}</div>{}'.format(
            STATE_CURRENT_STYLE, html.escape(title), sub))
    for number, (name, _) in enumerate(stages.STATE_GROUPS, start=1):
        if number == current:
            rows.append('<div style="{}">{}. {}</div>{}'.format(
                STATE_CURRENT_STYLE, number, html.escape(name), sub))
        else:
            rows.append('<div style="{}">{}. {}</div>'.format(
                STATE_OTHER_STYLE, number, html.escape(name)))
    return '<div>{}</div>'.format(''.join(rows))


class Simulation(object):
    """シミュレーション表示の GUI (時刻スライダー・Play・速さ) と再生時刻。"""

    def __init__(self, gui):
        self.slider = gui.add_slider('時刻 [秒]', min=0.0, max=1.0, step=0.05,
                                     initial_value=0.0)
        self.play = gui.add_checkbox('Play', initial_value=False)
        self.speed = gui.add_dropdown('再生の速さ', tuple(SPEEDS),
                                      initial_value='1 倍')
        self.duration = 1.0

    def set_duration(self, duration):
        self.duration = max(duration, 0.05)
        self.slider.max = self.duration
        self.slider.value = 0.0

    def set_visible(self, visible):
        self.slider.visible = visible
        self.play.visible = visible

    def advance(self, dt):
        """Play 中なら ``dt`` 秒 (× 速さ) 進め、今の時刻を返す (最後で止まる)。"""
        t = float(self.slider.value)
        if self.play.value:
            t += dt * SPEEDS[self.speed.value]
            if t >= self.duration:
                t = self.duration
                self.play.value = False
            self.slider.value = t
        return t


class RealRecording(object):
    """実機表示で受け取った実機の姿勢・ノードの状態・カメラ画像の記録 (1 試行分。
    計画を受け取ってから次の試行の計画まで)。時刻は ROS の時刻 (bag の時刻)。"""

    def __init__(self, attempt):
        self.attempt = attempt
        self.frames = []
        self._images = {}   # 画像の版 -> JPEG
        self._image_version = None
        self._image_time = None

    @property
    def duration(self):
        if not self.frames:
            return 0.0
        return self.frames[-1]['t'] - self.frames[0]['t']

    def add(self, frame, image_version, image):
        """``frame`` (``t``・``angles``・``world_from_base``・``status``) を足す。
        時刻が進んでいなければ (bag の再生が止まっているなど) 足さない。画像は
        ``CAMERA_PERIOD`` ごとに縮小して JPEG で持つ。"""
        if self.frames and frame['t'] <= self.frames[-1]['t']:
            return
        if (image is not None and image_version != self._image_version
                and (self._image_time is None
                     or frame['t'] - self._image_time >= CAMERA_PERIOD)):
            import cv2
            height = int(round(image.shape[0] * RECORD_IMAGE_WIDTH
                               / float(image.shape[1])))
            small = cv2.resize(image, (RECORD_IMAGE_WIDTH, height),
                               interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode('.jpg', small, [
                cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY])
            if ok:
                self._images[image_version] = jpeg
                self._image_version = image_version
                self._image_time = frame['t']
        self.frames.append(dict(frame, image_version=self._image_version))

    def index_at(self, t):
        """記録の始まりから ``t`` 秒の時点のフレームの番号 (記録が空なら None)。"""
        if not self.frames:
            return None
        target = self.frames[0]['t'] + t
        times = [f['t'] for f in self.frames]
        return max(0, bisect.bisect_right(times, target) - 1)

    def at(self, t):
        """記録の始まりから ``t`` 秒の時点のフレーム (記録が空なら None)。"""
        index = self.index_at(t)
        return None if index is None else self.frames[index]

    def trace(self, index):
        """番号 ``index`` のフレームまでに実機が通った台車 (床の高さ) と手先の
        位置の列 ``(台車, 手先)``。"""
        frames = [f for f in self.frames[:index + 1]
                  if f.get('hand') is not None]
        base = np.array([[f['world_from_base'][0, 3],
                          f['world_from_base'][1, 3], PATH_FLOOR_HEIGHT]
                         for f in frames]).reshape(-1, 3)
        hand = np.array([f['hand'] for f in frames]).reshape(-1, 3)
        return base, hand

    def image(self, version):
        """記録した画像 (BGR、無ければ None)。"""
        jpeg = self._images.get(version)
        if jpeg is None:
            return None
        import cv2
        return cv2.imdecode(jpeg, cv2.IMREAD_COLOR)


class RealPlayback(object):
    """実機表示の GUI (ライブ表示・記録の時刻スライダー・Play) と再生時刻。
    スライダーを動かすか Play を押すと記録の再生になり、「ライブ表示」で戻る。"""

    def __init__(self, gui):
        self.live = gui.add_checkbox('ライブ表示', initial_value=True)
        self.slider = gui.add_slider('記録の時刻 [秒]', min=0.0, max=1.0,
                                     step=0.05, initial_value=0.0)
        self.play = gui.add_checkbox('Play (記録)', initial_value=False)
        self._programmatic = False

        @self.slider.on_update
        def _on_slider(_):  # noqa: ANN001
            if not self._programmatic:
                self.live.value = False

        @self.play.on_update
        def _on_play(_):  # noqa: ANN001
            if self.play.value:
                self.live.value = False
                if float(self.slider.value) >= self.slider.max - 1e-6:
                    self._set_slider(0.0)

        @self.live.on_update
        def _on_live(_):  # noqa: ANN001
            if self.live.value:
                self.play.value = False

    def _set_slider(self, value, maximum=None):
        self._programmatic = True
        try:
            if maximum is not None:
                self.slider.max = max(maximum, 0.05)
            self.slider.value = min(value, self.slider.max)
        finally:
            self._programmatic = False

    def set_visible(self, visible):
        for handle in (self.live, self.slider, self.play):
            handle.visible = visible

    def advance(self, dt, duration, speed):
        """記録の長さ ``duration`` に合わせてスライダーを更新し、ライブなら None、
        記録の再生なら表示する時刻を返す (Play 中は ``dt`` × 速さだけ進める)。
        Play が記録の終わりに追いついたらライブ表示に戻す (記録がまだ伸びて
        いるときも、そのまま今の様子を追う)。"""
        if self.play.value and not self.live.value:
            t = float(self.slider.value) + dt * speed
            if t >= duration:
                self.live.value = True   # Play も止まる (_on_live)
        if self.live.value:
            self._set_slider(duration, duration)
            return None
        t = float(self.slider.value)
        if self.play.value:
            t += dt * speed
        self._set_slider(t, duration)
        return t


class HandshakeViewer(object):

    def __init__(self, args):
        self.args = args
        self.robot = load_aero(use_hand=True)
        self.viewer = ViserViewer(draw_grid=True)
        gui = self.viewer._server.gui
        self.bridge = None
        if args.ros:
            # ROS は実機・rosbag のときだけ使う (合成骨格は source 無しで動かす)。
            sys.path.insert(0, os.path.join(_THIS_DIR, 'ros'))
            from handshake_viewer_ros import RosBridge
            self.bridge = RosBridge(args.node,
                                    joint_states_topic=args.joint_states_topic,
                                    image_topic=args.image_topic)
        self.source = None
        if not args.ros:
            skeleton_dir, handshake_dir, motion_dir = input_dirs(args)
            self.source = FileSource(skeleton_dir, handshake_dir, motion_dir,
                                     self.robot, args)
            if not self.source.names:
                raise SystemExit(
                    '{} / {} / {} に共通の JSON がありません。'.format(
                        skeleton_dir, handshake_dir, motion_dir))

        self.mode = None
        if self.bridge is not None:
            self.mode = gui.add_dropdown(
                '表示', (MODE_REAL, MODE_SIMULATION), initial_value=MODE_REAL)
            self.arm_button = gui.add_button('ARM (差し出し手を待つ)')
            self.reset_button = gui.add_button('RESET (最初からやり直す)')

            @self.arm_button.on_click
            def _on_arm(_):  # noqa: ANN001
                print('[viewer] ARM: {}'.format(self.bridge.call('arm')))

            @self.reset_button.on_click
            def _on_reset(_):  # noqa: ANN001
                print('[viewer] RESET: {}'.format(self.bridge.call('reset')))
        else:
            self.person_index = 0
            back = gui.add_button('Back (人物)')
            forward = gui.add_button('Next (人物)')

            @back.on_click
            def _on_back(_):  # noqa: ANN001
                self._change_person(-1)

            @forward.on_click
            def _on_next(_):  # noqa: ANN001
                self._change_person(1)

        self.simulation = Simulation(gui)
        # 実機表示の記録 (計画を受け取るたびに作り直す) と、その再生の GUI。
        self.recording = None
        self.log_stages = None   # --log のログから読んだ状態 (最初の時刻で作る)
        self.trace_robot = None  # 実機の手先の軌跡を求めるモデル (表示とは別)
        self.real_playback = None
        if self.bridge is not None:
            self.real_playback = RealPlayback(gui)

            @self.mode.on_update
            def _on_mode(_):  # noqa: ANN001
                self._show_mode_controls()

            self._show_mode_controls()
        self.info_text = gui.add_markdown('')
        self.collision_checkbox = gui.add_checkbox(
            '干渉回避用モデルの表示', initial_value=True)
        self.paths_checkbox = gui.add_checkbox(
            '計画の経路の表示', initial_value=True)
        self._info_content = None
        self.scene = Scene(self.viewer, self.robot, self.collision_checkbox)

        @self.collision_checkbox.on_update
        def _on_collision(_):  # noqa: ANN001
            self.scene.set_collision_visible(self.collision_checkbox.value)

        self.status_panel = StatusPanel(self.viewer,
                                        with_camera=self.bridge is not None)
        self.scene.add_robot()
        remove_joint_angle_gui(self.viewer)
        remove_obstacles_gui(self.viewer)
        self.plan = None
        self._plan_version = None
        self._plan_lock = threading.Lock()

    def _show_mode_controls(self):
        """「表示」に合わせて、実機表示とシミュレーション表示の操作を出し分ける。"""
        real = self.mode.value == MODE_REAL
        self.real_playback.set_visible(real)
        self.simulation.set_visible(not real)

    def _change_person(self, direction):
        with self._plan_lock:
            self.person_index = ((self.person_index + direction)
                                 % len(self.source.names))
            self._set_plan(self.source.load(self.person_index))

    def _set_plan(self, plan):
        """表示する計画を替える (人の表示を作り直し、再生を最初に戻す)。"""
        self.plan = plan
        self.simulation.play.value = False
        if plan is None:
            self.scene.set_plan_paths(None, 0.0, False, False)
            self.scene.set_smpl(None, None, None)
            return
        plan.compute_paths(self.robot)
        if plan.timeline is not None:
            self.simulation.set_duration(plan.timeline.duration)
        else:
            self.simulation.set_duration(0.05)
        self.scene.set_human(
            plan.joint_positions, plan.torso_surface_offset,
            draw_skeleton=plan.person is None, key=('plan', id(plan), None))
        if plan.person is None:
            self.scene.set_smpl(None, None, None)

    def show(self):
        self.viewer.show(open_browser=not self.args.no_open_browser)
        viewer_nav.wait_for_client(self.viewer, self.args.client_wait_timeout)
        if self.source is not None:
            with self._plan_lock:
                self._set_plan(self.source.load(0))

    def spin(self):
        last = time.time()
        try:
            while True:
                now = time.time()
                dt, last = now - last, now
                frame = None
                if self.bridge is not None:
                    self._update_ros_plan()
                    # 表示によらず記録する (シミュレーション表示の間も残す)。
                    frame = self._live_frame()
                    if (self.recording is not None
                            and frame['angles'] is not None):
                        self.recording.add(frame, *self.bridge.image())
                with self._plan_lock:
                    if self.mode is not None and self.mode.value == MODE_REAL:
                        self._update_real(dt, frame)
                    else:
                        self._update_simulation(dt)
                self.scene.redraw()
                time.sleep(UPDATE_PERIOD)
        except KeyboardInterrupt:
            print()
        finally:
            if self.bridge is not None:
                self.bridge.shutdown()
            self.viewer.close()

    def _update_ros_plan(self):
        version, payload = self.bridge.plan()
        if version == self._plan_version:
            return
        self._plan_version = version
        plan = None
        if payload is not None and payload.get('result') is not None:
            plan = Plan(
                '試行{}'.format(payload['attempt']), payload['skeleton'],
                payload['result'], payload.get('motion'), self.robot,
                display_waypoints=payload.get('display_waypoints'),
                n_prepend=payload.get('n_prepend'),
                n_approach=payload.get('n_approach'),
                n_transition=payload.get('n_transition'),
                real_pose_ref=payload.get('real_pose_ref'),
                torso_surface_offset=payload.get('torso_surface_offset'))
        with self._plan_lock:
            if plan is not None and (
                    self.recording is None
                    or self.recording.attempt != payload['attempt']):
                # 新しい試行の計画から記録し直す (同じ試行の再送は続ける)。
                self.recording = RealRecording(payload['attempt'])
            self._set_plan(plan)

    def _live_frame(self):
        """今受け取っている実機の姿勢 (台車は計画の系)・ノードの状態。"""
        bridge = self.bridge
        plan = self.plan
        angles = bridge.joint_angles()
        world_from_base = None if angles is None else bridge.world_from_base(
            bridge.odom_from_base(),
            None if plan is None else plan.real_pose_ref)
        status, age = bridge.status()
        if status is not None:
            status = dict(status, elapsed=status['elapsed'] + age, age=age)
        now = bridge.now()
        log_status = self._log_status(now)
        if log_status is not None:
            status = log_status
        hand = None
        if angles is not None and plan is not None:
            hand = self._hand_position(angles, world_from_base,
                                       plan.result['robot_arm'])
        return dict(t=now, angles=angles, world_from_base=world_from_base,
                    status=status, hand=hand)

    def _hand_position(self, angles, world_from_base, robot_arm):
        """実機の関節角・台車位置での手先の位置 (表示とは別のモデルで計算する)。"""
        if self.trace_robot is None:
            self.trace_robot = load_aero(use_hand=True)
        robot = self.trace_robot
        for joint in robot.joint_list:
            if joint.name in angles:
                joint.joint_angle(angles[joint.name])
        robot.base_link.newcoords(Coordinates(
            pos=world_from_base[:3, 3], rot=world_from_base[:3, :3]))
        return getattr(robot, '{}arm_end_coords'.format(
            robot_arm)).worldpos().copy()

    def _log_status(self, now):
        """``--log`` のログから復元した時刻 ``now`` の状態 (``~status`` と同じ形。
        ログが無い・まだ最初の出来事の前なら None)。"""
        if not self.args.log or now <= 0.0:
            return None
        if self.log_stages is None:
            # ログの時刻 (時:分:秒) の日付は、最初に受け取った ROS の時刻で決める。
            from aero_demo.node_log_stages import LogStages
            self.log_stages = LogStages(self.args.log, now,
                                        self.args.log_utc_offset)
            print('[viewer] {} から状態の切り替わりを {} 件読みました。'.format(
                self.args.log, len(self.log_stages.events)))
        stage, since = self.log_stages.stage_at(now)
        if stage is None:
            return None
        return dict(stage=stage[0], detail=stage[1], elapsed=now - since,
                    age=0.0, background=None, offer_scores=None,
                    state='log')

    def _update_real(self, dt, live_frame):
        """実機の表示: 実機の姿勢・ノードの状態・カメラ画像 (ライブ、または
        記録の再生)。"""
        plan = self.plan
        recording = self.recording
        replay_t = self.real_playback.advance(
            dt, 0.0 if recording is None else recording.duration,
            SPEEDS[self.simulation.speed.value])
        frame = live_frame
        if replay_t is not None and recording is not None:
            frame = recording.at(replay_t) or live_frame
        live = frame is live_frame
        if plan is None and live:
            self._show_live_skeleton()
        elif plan is not None:
            self.scene.set_human(
                plan.joint_positions, plan.torso_surface_offset,
                draw_skeleton=True, key=('plan', id(plan), None))
        angles = frame['angles']
        if angles is None:
            self.scene.set_robot_initial()
        else:
            self.scene.set_robot_matrix(angles, frame['world_from_base'])
        status = frame['status']
        notes = []
        if status is None:
            stage = ('実機ノードの状態を待っています',
                     '{}/status が届いていません'.format(self.args.node))
            info = []
        else:
            stage = (status['stage'], status['detail'])
            info = ['経過 {:.0f} 秒'.format(status['elapsed'])]
            if status.get('background'):
                info.append('({})'.format(status['background']))
            if live and status['age'] > STATUS_STALE_TIME:
                notes.append('**{:.0f} 秒前の状態です (実機ノードが止まって'
                             'いる可能性)**'.format(status['age']))
            if live and status.get('offer_scores'):
                notes.append(status['offer_scores'])
        if not live:
            info.append('記録の再生 {:.1f} / {:.1f} 秒'.format(
                replay_t, recording.duration))
        if angles is None:
            notes.append('({} が届いていないため、ロボットは初期姿勢で表示)'
                         .format(self.args.joint_states_topic))
        self.status_panel.set_state(stage, info, '\n\n'.join(notes))
        if live:
            self.status_panel.set_image(*self.bridge.image())
        else:
            version = frame.get('image_version')
            self.status_panel.set_image(
                ('record', version), recording.image(version))
        title = None if status is None else status['stage']
        # 計画の線は灰で細く (今の区間も強調しない)、実際に通った軌跡を台車は緑・
        # 手先は赤で重ねる。
        self.scene.set_plan_paths(
            plan, 0.0 if plan is None or title is None
            else plan.progress_for_stage(title),
            False, self.paths_checkbox.value)
        index = None
        if recording is not None and plan is not None:
            index = (len(recording.frames) - 1 if live
                     else recording.index_at(replay_t))
        if index is None or index < 0 or not self.paths_checkbox.value:
            self.scene.set_real_trace(None, None, None)
        else:
            # 描き直しは TRACE_REDRAW_FRAMES フレームごと (軌跡を作るのも省く)。
            key = (id(recording), index // TRACE_REDRAW_FRAMES)
            if key != self.scene._trace_key:
                self.scene.set_real_trace(*recording.trace(index), key=key)
        self._set_info(plan, stage)

    def _set_info(self, plan, stage=None):
        """右パネルに今の処理 ``stage`` ((見出し, 補足)) の内容と計画の内容
        (計算の順・時間) を出す。"""
        parts = []
        if stage is not None and stages.state_of(stage[0]) is not None:
            title, detail = stage
            parts.append('**今の処理:** {}{}'.format(
                stages.numbered_title(title),
                ' — {}'.format(detail) if detail else ''))
        if plan is not None:
            parts.append('**{}** — {}\n\n{}'.format(
                plan.name, plan.summary(), plan.computation_text()))
        content = '\n\n'.join(parts)
        if content != self._info_content:
            self.info_text.content = content
            self._info_content = content

    def _show_live_skeleton(self):
        """計画前のライブの骨格を出す (``LIVE_SKELETON_PERIOD`` ごと)。"""
        self.scene.set_human(
            self.bridge.live_skeleton(), spik.DEFAULT_TORSO_SURFACE_OFFSET,
            key=('live', int(time.time() / LIVE_SKELETON_PERIOD), None))

    def _update_simulation(self, dt):
        """シミュレーション表示: 計画を実機と同じ時間で動かす。"""
        plan = self.plan
        self.scene.set_real_trace(None, None, None)
        if self.bridge is not None:
            self.status_panel.set_image(*self.bridge.image())
        if plan is None or plan.timeline is None:
            self.scene.set_robot_initial()
            self.scene.set_plan_paths(None, 0.0, False, False)
            if plan is None and self.bridge is not None:
                self._show_live_skeleton()
            self.status_panel.set_state(
                ('計画がありません', '差し出し手を待つか、計画の済んだ試行を'
                 '選んでください') if plan is None else
                ('表示する軌道がありません', plan.summary()))
            self._set_info(plan)
            return
        t = self.simulation.advance(dt)
        sample = plan.timeline.sample(t)
        self.scene.set_robot_pose(plan.motion['joint_names'],
                                  sample['angles'], sample['base'])
        index = int(round(sample['index']))
        waypoint = plan.display_waypoints[
            min(index, len(plan.display_waypoints) - 1)]
        arm = waypoint.get('human_arm')
        if arm is not None and plan.human_arm is not None:
            joint_positions = plan.human_arm.skeleton(arm)
        else:
            joint_positions, arm = plan.joint_positions, None
        self.scene.set_human(
            joint_positions, plan.torso_surface_offset,
            draw_skeleton=plan.person is None,
            key=('plan', id(plan), None if arm is None else id(arm)))
        if plan.person is not None:
            hand = getattr(self.robot, '{}arm_end_coords'.format(
                plan.result['robot_arm']))
            pose = look_at_pose(plan.model, plan.person, hand.worldpos())
            if arm is not None:
                pose = follow_arm_pose(plan.model, plan.person, pose,
                                       plan.human_arm, arm)
            self.scene.set_smpl(plan.model, plan.person, pose)
        self.status_panel.set_state(sample['stage'], [
            '{:.1f} / {:.1f} 秒 (シミュレーション)、waypoint {} / {}'.format(
                t, plan.timeline.duration, index,
                len(plan.display_waypoints) - 1)])
        self.scene.set_plan_paths(plan, sample['index'], sample['moving'],
                                  self.paths_checkbox.value)
        self._set_info(plan, sample['stage'])


def input_dirs(args):
    """``--work-dir`` か 3 ディレクトリの指定から (骨格, IK 結果, 軌道) の
    ディレクトリを返す。"""
    if args.work_dir:
        return tuple(os.path.join(args.work_dir, sub)
                     for sub in ('skeletons', 'handshakes', 'motions'))
    if not (args.skeleton_dir and args.handshake_dir and args.motion_dir):
        raise SystemExit('--ros、--work-dir、または --skeleton-dir・'
                         '--handshake-dir・--motion-dir を指定してください。')
    return args.skeleton_dir, args.handshake_dir, args.motion_dir


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--ros', action='store_true',
        help='実機・rosbag: 実機ノードの状態と実機の姿勢を表示する (ROS が要る)。')
    parser.add_argument(
        '--node', type=str, default='/run_camera_pipeline_test',
        help='実機ノードの名前 (--ros)。')
    parser.add_argument('--joint-states-topic', type=str,
                        default='/joint_states')
    parser.add_argument(
        '--image-topic', type=str, default=None,
        help='カメラ画像のトピック (既定は実機ノードの骨格を重ねた画像)。')
    parser.add_argument(
        '--log', type=str, default=None,
        help='bag と一緒に保存した実機ノードのログ (person_NN.log)。指定すると '
             '実機表示の状態を ~status の代わりにこのログから出す (~status を'
             '記録していない bag 用)。')
    parser.add_argument(
        '--log-utc-offset', type=float, default=None,
        help='--log を書いた PC の UTC からの時差 [時間] (既定はこの PC の地方時)。')
    parser.add_argument(
        '--work-dir', type=str, default=None,
        help='skeletons/・handshakes/・motions/ を持つディレクトリ '
             '(run_pipeline_test.py の作業ディレクトリ、実機ノードの --save-dir)。')
    parser.add_argument('--skeleton-dir', type=str, default=None)
    parser.add_argument('--handshake-dir', type=str, default=None)
    parser.add_argument('--motion-dir', type=str, default=None)
    parser.add_argument(
        '--model-path', type=str,
        default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'),
        help='SMPL (男性) モデル .pkl のパス (合成骨格の人のメッシュ用)。')
    parser.add_argument(
        '--female-model-path', type=str,
        default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicModel_f_lbs_10_207_0_v1.0.0.pkl'))
    parser.add_argument('--client-wait-timeout', type=float, default=30.0)
    parser.add_argument('--no-open-browser', action='store_true')
    return parser


def main():
    args = build_arg_parser().parse_args()
    viewer = HandshakeViewer(args)
    viewer.show()
    print('[viewer] 準備できました。Ctrl-C で終了します。')
    viewer.spin()


if __name__ == '__main__':
    main()
