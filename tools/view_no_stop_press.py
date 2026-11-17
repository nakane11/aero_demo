#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""止まらない押し込み (``scripts/no_stop_press.py``) を、保存した試行に人の手先と
台車の位置のずれを仮想的に与えて viser で確かめる。

実機ノードと同じ関数で、手前の hover・腕と台車の時間づけ・ずれの補正
(押し込みの台車の動きに織り込む)・押し込みの解き直し・腕の軌道の差し替え
(コントローラと同じく、hover を通る時刻の古い軌道の状態から 3 次スプラインで
つなぐ) を計算し、時刻スライダー/Play で動かす。

- 手先のずれ: 差し出した掌 (と骨格の手) を world で dx・dy・dz、掌の位置まわりに
  yaw・pitch・roll だけ動かす。検出はこの掌を誤差なく見たとする。
- 台車のずれ: 台車が hover 目標から (hover の台車の系で) dx・dy・dyaw ずれて
  止まったとする (接近の間に少しずつずれる表示)。
- 間に合うか: 台車が止まってから、止まるのを待つ・検出 3 枚・ずれの計測の
  ``FIXED_LATENCY`` 秒 + この PC で測った IK の時間 x ``--ik-scale`` が、
  hover を通る時刻の ``SPLICE_LEAD`` 秒前までに収まるか。

表示: 灰の線 = 計画 (ずれなし) の手先、赤の線 = 補正後の手先、緑の線 = 台車
(補正後)。青い球 = 計画の掌、橙の球 = ずれた掌。

Usage (ROS の環境と FEETECH_HAND_DIR が要る、roscore は不要)::

    python3 tools/view_no_stop_press.py --work-dir data/real/2026-10-22/logs/20261022_160029
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_THIS_DIR)
_SCRIPTS = os.path.join(_REPO, 'scripts')
for _path in (os.path.join(_SCRIPTS, 'ros'), _SCRIPTS,
              os.path.join(_REPO, 'src')):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_camera_pipeline_test as rcpt  # noqa: E402
import no_stop_press as nsp  # noqa: E402
import handshake_viewer as hv  # noqa: E402
from aero_demo import execution_timing as et  # noqa: E402
from aero_demo import scan_matching  # noqa: E402
from aero_demo import viewer_nav  # noqa: E402
from aero_demo.aero_urdf_setup import load_aero  # noqa: E402
from handshake_viewer_common import build_press_in_waypoints  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

spik = rcpt.spik
phm = rcpt.phm

# 台車が止まってから IK を始めるまで (止まるのを待つ 0.1 + 検出 3 枚 0.2 +
# ずれの計測 0.03) [s]。
FIXED_LATENCY = 0.33
SAMPLE_DT = 0.05
UPDATE_PERIOD = 0.05


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class Trial(object):
    """保存した試行 1 つ分の計画と、ずれを与えたときの止まらない押し込み。"""

    def __init__(self, work_dir, name, planner, args):
        self.name = name
        self.args = args
        self.planner = planner
        robot = planner.robot

        def load(sub):
            with open(os.path.join(work_dir, sub, name)) as f:
                return json.load(f)

        self.joint_positions = spik.load_skeleton_json(
            os.path.join(work_dir, 'skeletons', name))
        self.result = load('handshakes')
        self.motion = load('motions')
        palms = load('palms')
        self.hand = self.result['offered_hand']
        self.planned_palm = palms[self.hand]
        self.robot_arm = self.result['robot_arm']
        names = self.motion['joint_names']
        self.joint_names = names
        start_angles = dict(zip(
            [joint.name for joint in robot.joint_list],
            phm.arms_down_angles(robot, robot.joint_list)))
        lead_in = phm.build_lead_in_waypoints(
            [0.0, 0.0, 0.0], self.motion['waypoints'][0], names,
            start_joint_angles=start_angles)
        self.approach = lead_in + list(self.motion['waypoints'])
        self.hover = self.approach[-1]
        self.press = [self.hover] + build_press_in_waypoints(
            self.hover, names, self.result['post_process'])
        joints = {joint.name: joint for joint in robot.joint_list}
        self.max_vel = nsp.max_velocities(names, joints)
        retimed, self.retime_info = nsp.retime_approach_arm(
            robot, names, self.approach, self.result, margin_fn=self.margin)
        if retimed is None:
            raise RuntimeError('接近の終わりの腕を割り振り直すと人に近づき'
                               'すぎます ({})'.format(self.retime_info))
        self.pre = retimed[-1]
        self.pre_margin = self.margin(self.pre, None)
        self.base_wps, self.arm_wps = nsp.no_stop_waypoints(
            retimed, self.press)
        self.n_base = len(self.base_wps)
        self.base_times = nsp.limited_times(self.base_wps, self.max_vel)
        self.dwell = max(nsp.HOVER_DWELL, nsp.limited_times(
            [self.pre, self.hover], self.max_vel, with_base=False)[0])
        self.press_times = nsp.limited_times(self.press, self.max_vel)
        self.arm_times = (list(self.base_times) + [self.dwell]
                          + list(self.press_times))
        angles = np.array([wp['joint_angle_vector'] for wp in self.arm_wps])
        self.arm_track = nsp.HermiteTrack(
            angles, nsp.point_velocities(angles, self.arm_times),
            self.arm_times)
        self.t_base = float(np.sum(self.base_times))
        self.t_hover = self.t_base + self.dwell
        base_rel = np.asarray(et.base_trajectory_points(self.base_wps))
        self.base_track = nsp.HermiteTrack(
            base_rel, nsp.point_velocities(base_rel, self.base_times),
            self.base_times)
        self.base0 = nsp.wp_pose(self.base_wps[0])

    def margin(self, waypoint, hand_shift):
        return nsp.hover_clearance_margin(
            self.planner.robot, self.joint_names, waypoint,
            self.joint_positions, self.hand,
            self.planner.motion_verification_pairs,
            self.args.torso_surface_offset, hand_shift=hand_shift)

    def solve(self, hand_t, hand_rpy, slip):
        """ずれを与えて止まらない押し込みを計算し、再生用の状態を作る。"""
        H = nsp.wp_pose(self.hover)
        actual = scan_matching.compose(H, slip)
        palm_world = nsp.offset_palm(self.planned_palm, hand_t, hand_rpy)
        self.palm_world = palm_world
        self.joints_shown = nsp.offset_hand_joints(
            self.joint_positions, self.hand, self.planned_palm, hand_t,
            hand_rpy)
        palm_now = nsp.palm_in_base(palm_world, actual)
        residual = nsp.base_residual(H, actual)
        target = scan_matching.compose(actual, residual)
        base_move, fold_ok = nsp.fold_move(residual)
        base_error = scan_matching.relative(H, target)
        t0 = time.time()
        out = nsp.refine_folded(
            self.planner.robot, self.joint_names, self.hover, self.result,
            self.planned_palm, palm_now, base_move, base_error=base_error,
            margin_fn=self.margin)
        ik_time = time.time() - t0
        decision = self.t_base + FIXED_LATENCY + ik_time * self.args.ik_scale
        in_time = decision <= self.t_hover - nsp.SPLICE_LEAD
        refined = out['press_waypoints'] is not None
        spliced = refined and in_time and fold_ok
        press = ([self.hover] + out['press_waypoints']) if spliced \
            else self.press
        move = out['base_move'] if spliced else base_move
        now_wp = et.prepend_base_move(press, move)[0]
        base_wps = [now_wp] + list(press[1:])
        self.t_splice = None
        self.t_press = self.t_hover
        self.entry = 0
        self.skipped = 0
        target_plan = np.asarray(
            out['info']['post_process']['target_position'] if spliced
            else self.result['post_process']['target_position'])
        # 計画の系 (台車が hover にいる前提) -> world (台車は actual から move
        # 動いた位置)。押し込みの点・目標はこの変換で world に置かれる。
        to_world = scan_matching.compose(scan_matching.compose(actual, move),
                                         scan_matching.inverse(H))
        c, s = math.cos(to_world[2]), math.sin(to_world[2])
        self.target = np.array([
            to_world[0] + c * target_plan[0] - s * target_plan[1],
            to_world[1] + s * target_plan[0] + c * target_plan[1],
            target_plan[2]])
        if spliced:
            t_s = decision + nsp.SPLICE_LEAD
            q_s = self.arm_track.state(t_s)[0]
            # 今の手先 (台車はまだ base_move だけ動いていない)。
            current_hand = nsp.hand_position(
                self.planner.robot, self.joint_names,
                dict(now_wp, joint_angle_vector=[float(v) for v in q_s]),
                self.robot_arm)
            target = out['info']['post_process']['target_position']
            keep = nsp.monotone_press(
                self.planner.robot, self.joint_names, self.robot_arm, press,
                current_hand, target)
            self.entry = keep[0]
            self.skipped = len(press) - len(keep)
            # 押し込む向きに単調に進む点へ、腕と台車 (ずれの補正を含む) を
            # 差し替えの時刻から一緒に動かす。
            entry_wps = [press[k] for k in keep]
            base_wps = [now_wp] + list(entry_wps)
            seg = nsp.limited_times(
                [dict(now_wp, joint_angle_vector=[float(v) for v in q_s])]
                + list(entry_wps), self.max_vel)
            if self.entry == 0:
                # hover には予定の時刻に着く (早めない)。
                seg[0] = max(seg[0], self.t_hover - t_s)
            angles, vels, all_times, ratio = nsp.splice_at_entry(
                self.arm_track, t_s,
                [wp['joint_angle_vector'] for wp in entry_wps], seg,
                self.max_vel)
            self.t_press = t_s
            times = list(all_times)
            self.track = self.arm_track.spliced(t_s, angles, vels, all_times)
            self.t_splice = t_s
        else:
            times, ratio = list(self.press_times), None
            self.track = self.arm_track
        rel = np.asarray(et.base_trajectory_points(base_wps))
        self.press_base_track = nsp.HermiteTrack(
            rel, nsp.point_velocities(rel, times), times, self.t_press)
        self.actual = actual
        self.slip = np.asarray(slip, dtype=np.float64)
        self.duration = self.t_press + float(np.sum(times)) + 0.5
        info = out['info']
        shift = out['shift_record']
        self.summary = [
            '**{}** ({}手/{}腕)'.format(self.name, self.hand, self.robot_arm),
            '- 接近 (台車) {:.2f}s、手前の hover → hover {:.2f}s、押し込み '
            '{:.2f}s'.format(self.t_base, self.dwell, float(np.sum(times))),
            '- 接近の終わりの腕: hover から {:.0f}cm に入ってからの {} 点を '
            '{:.2f}s に割り振り直し、台車が着く時点で hover の {:.0f}mm 手前 '
            '(人体距離の余裕 {:.0f}mm)'.format(
                self.retime_info['radius'] * 100, self.retime_info['retimed'],
                self.retime_info['arm_time'],
                self.retime_info['pre_hover_distance'] * 1e3,
                self.pre_margin * 1e3),
            '- 台車の残差 ({:+.3f}m, {:+.3f}m, {:+.1f}deg) → {}'.format(
                residual[0], residual[1], math.degrees(residual[2]),
                '押し込みに織り込む' if fold_ok and np.any(base_move)
                else '許容内 (動かさない)' if fold_ok
                else '**織り込めない (実機は hover で止めて従来どおり)**'),
            '- 掌のずれで台車を寄せる: {}'.format(
                '-' if shift is None else '{} ({:.0f}mm)'.format(
                    shift['reason'], shift['distance'] * 1e3)),
            '- 押し込みで台車を動かす量 ({:+.3f}m, {:+.3f}m, {:+.1f}deg){}'
            .format(move[0], move[1], math.degrees(move[2]),
                    '' if not refined else
                    '、掌から離れる向きは除いた (腕で吸収)'
                    if out.get('no_retreat') else
                    '、**掌から離れる向きも動かす (除くと解けない)**'),
            '- 解き直し: {} (脚 {}、向きの寄せ {})、目標の変化 {:.0f}mm/{:.0f}deg'
            .format(info['reason'], info.get('legs'), info.get('blend'),
                    info['position_change'] * 1e3,
                    math.degrees(info['rotation_change'])),
            '- IK {:.2f}s (この PC) → 判断 {:.2f}s / 締め切り {:.2f}s: {}'.format(
                ik_time, decision, self.t_hover - nsp.SPLICE_LEAD,
                '**間に合う**' if in_time else '**間に合わない**'),
            '- 差し替え: {}{}'.format(
                'しない (計画どおり押し込む)' if not spliced else
                '{:.2f}s に (hover の {:.2f}s 前)'.format(
                    self.t_splice, self.t_hover - self.t_splice),
                '' if ratio is None else
                '、台車の補正と一緒に{}、後ろに戻る点を {} 個飛ばす (上限比 '
                '{:.2f})'.format(
                    ' hover へ' if self.entry == 0 else
                    '、手先より後ろの hover を飛ばして押し込みの {} 点目へ'
                    .format(self.entry), self.skipped, ratio)),
        ]
        print('\n'.join(self.summary))
        self.paths = self._paths(times if spliced else None)

    def base_pose(self, t):
        """時刻 ``t`` の台車 (world、ずれ込み)。"""
        if t < self.t_press:
            rel, _ = self.base_track.state(min(t, self.t_base))
            planned = scan_matching.compose(self.base0, rel)
            s = smoothstep(min(t, self.t_base) / max(self.t_base, 1e-6))
            return scan_matching.compose(planned, self.slip * s)
        rel, _ = self.press_base_track.state(t)
        return scan_matching.compose(self.actual, rel)

    def angles(self, t):
        return self.track.state(t)[0]

    def _hand(self, angles, base):
        robot = self.planner.robot
        robot.newcoords(hv.Coordinates())
        hv.apply_pose(robot, self.joint_names, angles, base)
        return getattr(robot, '{}arm_end_coords'.format(
            self.robot_arm)).worldpos().copy()

    def _paths(self, _):
        ts = np.arange(0.0, self.duration, SAMPLE_DT)
        hand, base = [], []
        for t in ts:
            b = self.base_pose(t)
            base.append([b[0], b[1], 0.01])
            hand.append(self._hand(self.angles(t), b))
        planned = []
        for t in ts:
            if t > self.t_hover + float(np.sum(self.press_times)):
                break
            if t < self.t_hover:
                rel, _ = self.base_track.state(min(t, self.t_base))
                b = scan_matching.compose(self.base0, rel)
            else:
                rel = np.asarray(et.base_trajectory_points(self.press))
                track = nsp.HermiteTrack(
                    rel, nsp.point_velocities(rel, self.press_times),
                    self.press_times, self.t_hover)
                b = scan_matching.compose(nsp.wp_pose(self.hover),
                                          track.state(t)[0])
            planned.append(self._hand(self.arm_track.state(t)[0], b))
        return dict(hand=np.array(hand), base=np.array(base),
                    planned=np.array(planned))


class App(object):

    def __init__(self, args):
        self.args = args
        self.dir = args.work_dir
        names = [set(os.path.basename(p) for p in glob.glob(
            os.path.join(self.dir, sub, '*.json')))
            for sub in ('skeletons', 'handshakes', 'motions', 'palms')]
        self.names = sorted(set.intersection(*names))
        node_args = rcpt.build_arg_parser().parse_args([])
        self.planner = rcpt.HandshakePlanner(node_args)
        spik.attach_camera_optical_coords(self.planner.robot)
        args.torso_surface_offset = node_args.torso_surface_offset
        self.robot = load_aero(use_hand=True)
        self.viewer = ViserViewer(draw_grid=True)
        gui = self.viewer._server.gui
        self.trial_dropdown = gui.add_dropdown(
            '試行', tuple(self.names) or ('(なし)',),
            initial_value=self.names[0] if self.names else '(なし)')
        with gui.add_folder('手先のずれ'):
            self.hand_sliders = [gui.add_slider(label, -15.0, 15.0, 0.5, 0.0)
                                 for label in ('dx [cm]', 'dy [cm]',
                                               'dz [cm]')]
            self.rot_sliders = [gui.add_slider(label, -45.0, 45.0, 1.0, 0.0)
                                for label in ('yaw [deg]', 'pitch [deg]',
                                              'roll [deg]')]
        with gui.add_folder('台車のずれ (hover の台車の系)'):
            self.slip_sliders = [
                gui.add_slider('dx [cm]', -15.0, 15.0, 0.5, 0.0),
                gui.add_slider('dy [cm]', -15.0, 15.0, 0.5, 0.0),
                gui.add_slider('dyaw [deg]', -12.0, 12.0, 0.5, 0.0)]
        self.solve_button = gui.add_button('この設定で解く')
        self.simulation = hv.Simulation(gui)
        self.info = gui.add_markdown('')
        self.collision = gui.add_checkbox('干渉回避用モデルの表示',
                                          initial_value=False)
        self.scene = hv.Scene(self.viewer, self.robot, self.collision)
        self.scene.add_robot()
        hv.remove_joint_angle_gui(self.viewer)
        hv.remove_obstacles_gui(self.viewer)
        self.trial = None
        self.handles = []
        self.pending = True

        @self.solve_button.on_click
        def _(_event):  # noqa: ANN001
            self.pending = True

        @self.trial_dropdown.on_update
        def _(_event):  # noqa: ANN001
            self.trial = None
            self.pending = True

    def _solve(self):
        name = self.trial_dropdown.value
        if self.trial is None or self.trial.name != name:
            self.info.content = '{} を読み込んでいます...'.format(name)
            self.trial = Trial(self.dir, name, self.planner, self.args)
        hand_t = [s.value / 100.0 for s in self.hand_sliders]
        hand_rpy = [s.value for s in self.rot_sliders]
        slip = [self.slip_sliders[0].value / 100.0,
                self.slip_sliders[1].value / 100.0,
                math.radians(self.slip_sliders[2].value)]
        self.info.content = '解いています...'
        self.trial.solve(hand_t, hand_rpy, slip)
        self.info.content = '\n'.join(self.trial.summary)
        self.simulation.set_duration(self.trial.duration)
        self.scene.set_human(self.trial.joints_shown,
                             self.args.torso_surface_offset,
                             key=('human', time.time()))
        self._draw_static()

    def _draw_static(self):
        scene = self.viewer._server.scene
        for handle in self.handles:
            handle.remove()
        self.handles = []
        paths = self.trial.paths
        for key, color, width in (('planned', (150, 150, 150), 2.0),
                                  ('hand', (220, 30, 30), 4.0),
                                  ('base', (30, 170, 60), 4.0)):
            points = np.asarray(paths[key], dtype=np.float32)
            if len(points) < 2:
                continue
            self.handles.append(scene.add_line_segments(
                '/no_stop/{}'.format(key),
                points=np.stack([points[:-1], points[1:]], axis=1),
                colors=color, line_width=width))
        for key, palm, color in (
                ('planned_palm', self.trial.planned_palm, (40, 90, 220)),
                ('palm', self.trial.palm_world, (240, 140, 20))):
            self.handles.append(scene.add_icosphere(
                '/no_stop/{}'.format(key), radius=0.02, color=color,
                position=tuple(float(v) for v in palm['position'])))

    def run(self):
        self.viewer.show(open_browser=not self.args.no_open_browser)
        viewer_nav.wait_for_client(self.viewer, 30.0)
        last = time.time()
        try:
            while True:
                if self.pending and self.names:
                    self.pending = False
                    try:
                        self._solve()
                    except Exception as exc:  # noqa: BLE001
                        self.info.content = '失敗: {!r}'.format(exc)
                        print('[view] 失敗: {!r}'.format(exc))
                now = time.time()
                dt, last = now - last, now
                if self.trial is not None and hasattr(self.trial, 'track'):
                    t = self.simulation.advance(dt)
                    base = self.trial.base_pose(t)
                    self.scene.set_robot_pose(self.trial.joint_names,
                                              self.trial.angles(t), base)
                    self.scene.redraw()
                time.sleep(UPDATE_PERIOD)
        except KeyboardInterrupt:
            print()
        finally:
            self.viewer.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--work-dir', required=True,
                        help='skeletons/・handshakes/・motions/・palms/ を持つ'
                             'ディレクトリ (実機の logs/<起動時刻>/ など)')
    parser.add_argument('--ik-scale', type=float, default=2.0,
                        help='この PC の IK の時間を実機の時間に直す倍率 (実機は '
                             'bag の再生の 2〜4 倍かかっていた)')
    parser.add_argument('--no-open-browser', action='store_true')
    App(parser.parse_args()).run()


if __name__ == '__main__':
    main()
