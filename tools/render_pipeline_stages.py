#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""計画済みの 1 人分の結果から、横並びまでの各段階で考慮する干渉・台車の
可動域・経路を図にして PNG に保存する (``docs/handshake_pipeline_stages.md``
の図)。

各段階の図は、左が 3 次元の描画 (オフスクリーン)、右が真上から見た図、
下が説明。``--viser`` を付けると同じシーンを viser のビューアにも出す。
最初は全段階のロボットを並べた全体表示で、ロボットをクリックするとその
段階の詳細を開く。項目 (干渉ペア・可動域など) はチェックボックスで表示を
切り替える。

段階:
  1. 差し出した手の検出 (掌の位置・法線、hover/押し込みの目標)
  2. 第 1 段: 干渉回避付きバッチ IK (台車の可動域、最適化で使う干渉ペア)
  3. 事後検証: hover 姿勢 (人体から 6 cm、自己干渉の貫通)
  4. 第 2 段: 押し込み姿勢 + 視線 (人体との判定なし)
  5. 接近軌道 (lead-in、接近開始位置の候補、螺旋、経路の検証 5 cm)
  6. 横並び移動 (移動先の可動域、下ろした人の腕、追加の干渉ペア)

Usage
-----
    source ~/venv/aero-uv/bin/activate
    # WSL ではソフトウェア描画の EGL を使う
    export PYOPENGL_PLATFORM=egl LIBGL_ALWAYS_SOFTWARE=1 \\
        GALLIUM_DRIVER=llvmpipe EGL_PLATFORM=surfaceless JAX_PLATFORMS=cpu
    python3 tools/render_pipeline_stages.py \\
        --work-dir /tmp/aero_demo_pipeline_xxx --name human_004 \\
        --output-dir docs/images/pipeline_stages [--viser]

``--work-dir`` は ``run_pipeline_test.py --plan-motion
--side-by-side-transition`` の作業ディレクトリ (合成データ、SMPL 付き)。
"""

import argparse
import collections
import json
import math
import os
import sys
import threading
import time

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager  # noqa: E402
from matplotlib import patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
_PKG_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

import solve_palm_ik as spik  # noqa: E402  (パス追加後に import)
import plan_handshake_motion as phm  # noqa: E402
import side_by_side_transition as sbs  # noqa: E402
from generate_random_human_poses import load_smpl_models  # noqa: E402
from handshake_viewer_common import apply_waypoint_pose  # noqa: E402
from view_handshake_motion import follow_arm_pose  # noqa: E402
from view_handshake_poses import build_smpl_mesh  # noqa: E402
from view_handshake_poses import load_skeleton_json as load_smpl_params  # noqa: E402,E501
from view_handshake_poses import look_at_pose  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402
from skrobot.models import Aero  # noqa: E402

from aero_demo.aero_urdf_setup import load_aero  # noqa: E402

# 日本語のフォント (無ければ matplotlib の既定)。
JP_FONT_PATHS = (
    '/usr/share/fonts/opentype/ipaexfont-gothic/ipaexg.ttf',
    '/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf',
)

# 色 (RGBA, 0-255)。
ROBOT_COLOR = (205, 205, 210, 255)
ROBOT_GHOST_COLOR = (150, 170, 200, 70)
SKIN_COLOR = (205, 165, 140, 255)
SKIN_FADED_COLOR = (205, 165, 140, 110)
HUMAN_CYL_COLOR = (80, 140, 220, 70)
HUMAN_SHELL_COLOR = (80, 170, 240, 28)
HAND_SHELL_COLOR = (60, 200, 170, 45)     # 差し出した手まわりの膜 (2 cm)
PRESS_CURVE_COLOR = (230, 40, 40, 255)
ROBOT_COLL_COLOR = (230, 150, 80, 80)
PAIR_COLOR = (220, 40, 40, 170)          # 最適化の干渉ペア (人体・胴体)
OFFERED_COLOR = (40, 170, 60, 170)       # 差し出した手まわりのペア
EXTRA_PAIR_COLOR = (170, 50, 200, 170)   # 横並びで足す組
EXCLUDED_COLOR = (150, 150, 150, 90)     # 判定の対象外
CLOSEST_COLOR = (250, 200, 0, 220)       # 最短距離の部位
REGION_COLOR = (60, 180, 90, 70)         # 台車の可動域
YAW_FAN_COLOR = (250, 160, 30, 110)
PATH_COLOR = (40, 90, 220, 255)
LEAD_IN_COLOR = (110, 110, 110, 255)
TRANSITION_PATH_COLOR = (170, 50, 200, 255)
TARGET_HOVER_COLOR = (240, 140, 0, 255)
TARGET_PRESS_COLOR = (220, 30, 30, 255)
GAZE_COLOR = (250, 210, 0, 255)

# 全体表示のロボットの色 (段階ごと)。
OVERVIEW_ROBOT_COLORS = ((170, 185, 235, 255), (240, 190, 140, 255),
                         (235, 235, 240, 255), (180, 220, 170, 255),
                         (225, 175, 220, 255), (230, 225, 150, 255))

IMAGE_SIZE = (1100, 820)
# 表示範囲の外とみなす距離 (欠けた部位は 100 m 先のダミー)。
DUMMY_DISTANCE = 50.0
# 床に描くものを床から浮かせる高さ [m] (z-fighting 回避)。
FLOOR_Z = 0.004
# 表示用に台車の y 可動域 (±3 m) を切り詰める幅 [m]。
REGION_DISPLAY_HALF = 1.4


def rgba(color):
    return np.asarray(color, dtype=np.float64) / 255.0


def setup_font():
    for path in JP_FONT_PATHS:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(
                fname=path).get_name()
            return


# ---------------------------------------------------------------------------
# シーン (3D の描画と viser で共有する、色付きメッシュのリスト)
# ---------------------------------------------------------------------------

class Scene(object):
    """色付きメッシュを「レイヤー」(viser のチェックボックス 1 つ分) に分けて
    持つ。``layer`` で登録したレイヤーに以降の ``add`` が入る。``colors`` を
    渡したレイヤーには、その色のメッシュが (今のレイヤーに関わらず) 入る
    (人体の円柱のように 1 回の呼び出しで役割の違う色を足すとき用)。"""

    def __init__(self):
        self.items = []  # (trimesh.Trimesh, rgba 0-255, レイヤー名)
        self.layers = collections.OrderedDict()
        self._current = None
        self._by_color = {}

    def layer(self, key, label, hint='', default=True, colors=(), click=None):
        """``click`` は viser でこのレイヤーのメッシュをクリックしたときに
        開く段階の添字。"""
        if key not in self.layers:
            self.layers[key] = dict(label=label, hint=hint, default=default,
                                    color=None, click=click)
        for color in colors:
            self._by_color[tuple(color)] = key
        if not colors:
            self._current = key
        return key

    def add(self, mesh, color):
        if mesh is None or not len(mesh.vertices):
            return
        color = tuple(color)
        key = self._by_color.get(color, self._current)
        if key is None:
            key = self.layer('misc', 'その他')
        if self.layers[key]['color'] is None:
            self.layers[key]['color'] = color
        self.items.append((mesh, color, key))

    def visible_items(self):
        return [(m, c) for m, c, key in self.items
                if self.layers[key]['default']]


# チェックボックスの見出しに付ける色見本 (viser の GUI は文字に色を付けられ
# ないので、色相で一番近い四角の絵文字)。(色相の上限 [deg], 絵文字)。
HUE_SWATCHES = ((15, '🟥'), (45, '🟧'), (70, '🟨'), (170, '🟩'),
                (260, '🟦'), (340, '🟪'), (360, '🟥'))


def swatch(color):
    import colorsys
    if color is None:
        return '▫️'
    h, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in color[:3]))
    if s < 0.2:
        return '⬜' if v > 0.45 else '⬛'
    return next(e for limit, e in HUE_SWATCHES if h * 360.0 <= limit)


def transformed(mesh, matrix):
    mesh = mesh.copy()
    mesh.apply_transform(matrix)
    return mesh


def link_transform(link):
    return link.worldcoords().T()


def tube(points, radius, sections=10):
    """折れ線 ``points`` を太さ ``radius`` の管にする。"""
    meshes = []
    points = np.asarray(points, dtype=np.float64)
    for a, b in zip(points[:-1], points[1:]):
        if np.linalg.norm(b - a) < 1e-6:
            continue
        meshes.append(trimesh.creation.cylinder(
            radius=radius, segment=[a, b], sections=sections))
        meshes.append(trimesh.creation.icosphere(subdivisions=1,
                                                 radius=radius).apply_translation(b))
    return trimesh.util.concatenate(meshes) if meshes else None


def arrow(start, end, radius=0.012):
    start, end = np.asarray(start, float), np.asarray(end, float)
    v = end - start
    length = float(np.linalg.norm(v))
    if length < 1e-6:
        return None
    head = min(0.06, 0.4 * length)
    neck = start + v * (1.0 - head / length)
    shaft = trimesh.creation.cylinder(radius=radius, segment=[start, neck])
    cone = trimesh.creation.cone(radius=radius * 2.5, height=head)
    z = np.array([0.0, 0.0, 1.0])
    rot = trimesh.geometry.align_vectors(z, v / length)
    rot[:3, 3] = neck
    cone.apply_transform(rot)
    return trimesh.util.concatenate([shaft, cone])


def sphere(center, radius):
    return trimesh.creation.icosphere(subdivisions=2, radius=radius) \
        .apply_translation(center)


def floor_polygon(xy, z=FLOOR_Z):
    """凸多角形 ``xy`` の床の板 (扇形に三角形分割)。"""
    xy = np.asarray(xy, dtype=np.float64)
    vertices = np.concatenate([[xy.mean(axis=0)], xy])
    vertices = np.column_stack([vertices, np.full(len(vertices), z)])
    n = len(xy)
    faces = [[0, 1 + i, 1 + (i + 1) % n] for i in range(n)]
    faces += [[0, 1 + (i + 1) % n, 1 + i] for i in range(n)]
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def fan_xy(center, yaw_range, radius, steps=24):
    angles = np.linspace(yaw_range[0], yaw_range[1], steps)
    return np.array([center[:2]] + [
        center[:2] + radius * np.array([math.cos(a), math.sin(a)])
        for a in angles])


def region_xy(x_range, y_range):
    return np.array([[x_range[0], y_range[0]], [x_range[1], y_range[0]],
                     [x_range[1], y_range[1]], [x_range[0], y_range[1]]])


def clip_range(value_range, center, half):
    return (max(value_range[0], center - half),
            min(value_range[1], center + half))


def circle_xy(center, radius, steps=72):
    a = np.linspace(0.0, 2 * math.pi, steps, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(a),
                            center[1] + radius * np.sin(a)])


def ring(center, radius, z=FLOOR_Z * 2, width=0.01):
    points = circle_xy(center, radius)
    points = np.vstack([points, points[:1]])
    return tube(np.column_stack([points, np.full(len(points), z)]), width,
                sections=6)


# ---------------------------------------------------------------------------
# ロボット・人体
# ---------------------------------------------------------------------------

def set_pose(robot, joint_names, angle_vector, base_position, base_yaw):
    name_to_angle = dict(zip(joint_names, angle_vector))
    robot.reset_pose()
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.base_link.newcoords(Coordinates(
        pos=list(base_position), rot=rpy_matrix(base_yaw, 0.0, 0.0)))


def set_both(robots, wp, joint_names):
    for robot in robots:
        set_pose(robot, joint_names, wp['joint_angle_vector'],
                 wp['base_position'], wp['base_yaw'])


def robot_visual_mesh(robot):
    meshes = []
    for link in robot.link_list:
        mesh = link.concatenated_visual_mesh
        if mesh is None or not hasattr(mesh, 'faces') or not len(mesh.vertices):
            continue
        meshes.append(transformed(trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, process=False),
            link_transform(link)))
    return trimesh.util.concatenate(meshes)


def collision_links(ik_robot):
    return spik.collision_link_list_for_arm(ik_robot)


def collision_mesh(link):
    return transformed(link.collision_mesh, link_transform(link))


def add_robot_collision(scene, ik_robot, colors=None,
                        default=ROBOT_COLL_COLOR):
    """IK・自己干渉で使う指なし+手の箱の干渉形状。``colors`` は
    ``{リンク名: 色}``。"""
    colors = colors or {}
    for link in collision_links(ik_robot):
        scene.add(collision_mesh(link), colors.get(link.name, default))


def obstacle_cylinders(joint_positions):
    obstacles = spik.human_body_obstacles(joint_positions)
    result = []
    for index, link in enumerate(obstacles):
        if np.linalg.norm(link.worldpos()) > DUMMY_DISTANCE:
            continue
        result.append((index, link))
    return obstacles, result


def cylinder_mesh(link, inflate=0.0):
    if inflate > 0.0:
        mesh = trimesh.creation.capsule(height=link.height,
                                        radius=link.radius + inflate,
                                        count=[16, 16])
    else:
        mesh = trimesh.creation.cylinder(radius=link.radius,
                                         height=link.height, sections=20)
    return transformed(mesh, link_transform(link))


def shell_of(shell, index):
    """``shell`` (全身共通の距離、または ``{添字: 距離}`` と ``'default'``)
    の部位 ``index`` の距離 (膜を描かないなら 0)。"""
    if isinstance(shell, dict):
        value = shell.get(index, shell.get('default', 0.0))
    else:
        value = shell
    return value if value is not None and value > 0.0 else 0.0


def hand_shell(hand, clearance, hand_clearance):
    """全身 ``clearance``、差し出した手まわりだけ ``hand_clearance`` (None
    なら膜なし = 判定しない) の ``shell``。"""
    shell = {'default': clearance}
    for index in spik.offered_hand_obstacle_indices(hand):
        shell[index] = hand_clearance
    return shell


def add_human_cylinders(scene, joint_positions, colors=None,
                        default=HUMAN_CYL_COLOR, shell=0.0,
                        shell_color=HUMAN_SHELL_COLOR, skip=(),
                        hand_shell_color=None, hand=None):
    """人体の円柱と、検証の距離だけ膨らませた膜。``hand_shell_color`` を
    渡すと差し出した手まわりの膜はその色にする (別のレイヤーに分ける用)。"""
    colors = colors or {}
    hand_indices = spik.offered_hand_obstacle_indices(hand) if hand else ()
    _, cylinders = obstacle_cylinders(joint_positions)
    for index, link in cylinders:
        if index in skip:
            continue
        scene.add(cylinder_mesh(link), colors.get(index, default))
        inflate = shell_of(shell, index)
        if inflate > 0.0:
            color = hand_shell_color if (hand_shell_color is not None
                                         and index in hand_indices) \
                else shell_color
            scene.add(cylinder_mesh(link, inflate), color)


def cylinder_endpoints(link):
    axis = link.worldrot()[:, 2]
    center = link.worldpos()
    return center - axis * link.height / 2.0, center + axis * link.height / 2.0


# ---------------------------------------------------------------------------
# 3D の描画 (pyrender、オフスクリーン)
# ---------------------------------------------------------------------------

def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def render_scene(scene, eye, target, size=IMAGE_SIZE, yfov=0.75):
    import pyrender
    pr = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                        ambient_light=[0.45, 0.45, 0.45])
    # 床 (薄い灰色) とグリッド。
    floor = trimesh.creation.box([12.0, 12.0, 0.01])
    floor.apply_translation([2.5, 0.0, -0.006])
    items = [(floor, (240, 240, 238, 255))]
    for v in np.arange(-3.0, 8.01, 1.0):
        items.append((trimesh.creation.box([0.006, 12.0, 0.002])
                      .apply_translation([v, 0.0, 0.0005]), (200, 200, 200, 255)))
    for v in np.arange(-6.0, 6.01, 1.0):
        items.append((trimesh.creation.box([12.0, 0.006, 0.002])
                      .apply_translation([2.5, v, 0.0005]), (200, 200, 200, 255)))
    visible = scene.visible_items()
    opaque = [(m, c) for m, c in items + visible if c[3] >= 255]
    blended = [(m, c) for m, c in visible if c[3] < 255]
    # 半透明は奥から順に描く。
    blended.sort(key=lambda mc: -np.linalg.norm(
        mc[0].bounds.mean(axis=0) - np.asarray(eye)))
    for mesh, color in opaque + blended:
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=rgba(color), metallicFactor=0.0,
            roughnessFactor=0.75,
            alphaMode='BLEND' if color[3] < 255 else 'OPAQUE')
        plain = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                                process=False)
        pr.add(pyrender.Mesh.from_trimesh(plain, material=material,
                                          smooth=False))
    camera_pose = look_at(eye, target)
    pr.add(pyrender.PerspectiveCamera(yfov=yfov,
                                      aspectRatio=size[0] / float(size[1])),
           pose=camera_pose)
    pr.add(pyrender.DirectionalLight(intensity=2.5), pose=camera_pose)
    pr.add(pyrender.DirectionalLight(intensity=1.5),
           pose=look_at([0.0, 0.0, 10.0], [0.5, 0.3, 0.0]))
    renderer = pyrender.OffscreenRenderer(*size)
    try:
        color, _ = renderer.render(pr, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    # 背景の透明部分は白にする。
    rgb = color[..., :3].astype(np.float64)
    alpha = color[..., 3:4].astype(np.float64) / 255.0
    return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)


def zoom(stage, point, ratio):
    """注視点を ``point`` へ寄せ、距離を ``ratio`` 倍にする。"""
    offset = stage.eye - stage.target
    stage.target = (stage.target + np.asarray(point, dtype=np.float64)) / 2.0
    stage.eye = stage.target + offset * ratio


def view_eye(target, azimuth_deg, elevation_deg, distance):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return np.asarray(target, float) + distance * np.array([
        math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
        math.sin(el)])


# ---------------------------------------------------------------------------
# 真上から見た図 (matplotlib)
# ---------------------------------------------------------------------------

def top_capsule(ax, link, color, inflate=0.0, zorder=2, **kwargs):
    p0, p1 = cylinder_endpoints(link)
    r = link.radius + inflate
    a, b = p0[:2], p1[:2]
    d = b - a
    norm = float(np.linalg.norm(d))
    fc = rgba(color)
    if norm > 1e-6:
        n = np.array([-d[1], d[0]]) / norm * r
        ax.add_patch(patches.Polygon([a + n, b + n, b - n, a - n],
                                     closed=True, fc=fc, ec='none',
                                     zorder=zorder, **kwargs))
    for p in (a, b):
        ax.add_patch(patches.Circle(p, r, fc=fc, ec='none', zorder=zorder,
                                    **kwargs))


def top_human(ax, joint_positions, colors=None, default=HUMAN_CYL_COLOR,
              shell=0.0, skip=()):
    colors = colors or {}
    _, cylinders = obstacle_cylinders(joint_positions)
    for index, link in cylinders:
        inflate = shell_of(shell, index)
        if index not in skip and inflate > 0.0:
            top_capsule(ax, link, (80, 170, 240, 40), inflate=inflate,
                        zorder=1)
    for index, link in cylinders:
        if index in skip:
            continue
        top_capsule(ax, link, colors.get(index, default), zorder=3)


def top_robot(ax, ik_robot, color=(230, 150, 80, 110), colors=None,
              zorder=4, ec=None):
    from scipy.spatial import ConvexHull
    colors = colors or {}
    for link in collision_links(ik_robot):
        pts = np.asarray(collision_mesh(link).vertices)[:, :2]
        try:
            hull = ConvexHull(pts)
        except Exception:
            continue
        c = colors.get(link.name, color)
        ax.add_patch(patches.Polygon(
            pts[hull.vertices], closed=True, fc=rgba(c),
            ec=ec if ec is not None else 'none', lw=0.5, zorder=zorder))


def top_arrow(ax, xy, yaw, color, length=0.35, width=0.02, zorder=6,
              label=None):
    ax.arrow(xy[0], xy[1], length * math.cos(yaw), length * math.sin(yaw),
             width=width, head_width=width * 3.5, head_length=width * 3.5,
             length_includes_head=True, color=rgba(color)[:3],
             zorder=zorder, label=label)


def top_region(ax, xy, color, label=None, zorder=0, **kwargs):
    ax.add_patch(patches.Polygon(xy, closed=True, fc=rgba(color),
                                 ec=rgba(color)[:3], lw=1.0, zorder=zorder,
                                 label=label, **kwargs))


def legend_handle(color, label, kind='patch'):
    if kind == 'line':
        return plt.Line2D([], [], color=rgba(color)[:3], lw=2.5, label=label)
    if kind == 'dash':
        return plt.Line2D([], [], color=rgba(color)[:3], lw=2, ls='--',
                          label=label)
    if kind == 'marker':
        return plt.Line2D([], [], color=rgba(color)[:3], marker='o', ls='',
                          label=label)
    c = rgba(color)
    c[3] = max(c[3], 0.35)
    return patches.Patch(fc=c, ec='none', label=label)


# ---------------------------------------------------------------------------
# 1 人分のデータ
# ---------------------------------------------------------------------------

class Person(object):
    def __init__(self, work_dir, name, models_by_gender):
        if not name.endswith('.json'):
            name += '.json'
        self.name = os.path.splitext(name)[0]

        def load(sub):
            with open(os.path.join(work_dir, sub, name)) as f:
                return json.load(f)
        self.handshake = load('handshakes')
        self.motion = load('motions')
        self.palms = load('palms')
        skeleton_path = os.path.join(work_dir, 'skeletons', name)
        joints = spik.load_skeleton_json(skeleton_path)
        self.offset = spik.human_translation_offset(
            joints, front_distance=spik.HUMAN_FRONT_DISTANCE)
        self.joints = spik.translate_joint_positions(joints, self.offset)
        # 合成データは体幹をずらさない (plan_handshake_motion.py の既定)。
        self.collision_joints = spik.shift_torso_joints_from_surface(
            self.joints, self.offset, 0.0)
        self.standing = np.asarray(spik.human_standing_xy(self.joints))
        self.facing_yaw = spik.human_facing_yaw(self.joints)
        self.hand = self.handshake['offered_hand']
        self.robot_arm = self.handshake['robot_arm']
        if self.handshake.get('palm_position') is not None:
            # IK と同じ (平行移動済みの) 掌。
            self.palm = np.asarray(self.handshake['palm_position'])
            self.palm_normal = np.asarray(self.handshake['palm_normal'])
        else:
            palm = self.palms[self.hand]
            self.palm = np.asarray(palm['position']) + np.array(
                [self.offset[0], self.offset[1], 0.0])
            self.palm_normal = np.asarray(palm['y_axis'])
        self.hover_target = np.asarray(self.handshake['target_position'])
        post = self.handshake['post_process']
        self.press_target = np.asarray(post['target_position'])
        # 押し込みの手先目標の曲線 (hover 目標 -> 最終目標)。
        palm_dict = dict(position=self.palm, y_axis=self.palm_normal)
        self.press_curve = np.array([
            spik.press_in_target_position(
                palm_dict, np.asarray(post['target_rot']), u)
            for u in np.linspace(0.0, 1.0, 31)])

        person = load_smpl_params(skeleton_path)
        root = np.array(person['root_pos'], dtype=np.float64)
        root[0] += self.offset[0]
        root[1] += self.offset[1]
        self.smpl = dict(person, root_pos=root)
        self.model = models_by_gender.get(person['gender'],
                                          models_by_gender['male'])

    def smpl_mesh(self, gaze_target, arm=None, human_arm=None):
        pose = look_at_pose(self.model, self.smpl, gaze_target)
        if arm is not None:
            pose = follow_arm_pose(self.model, self.smpl, pose, human_arm,
                                   arm)
        mesh = build_smpl_mesh(self.model, self.smpl, pose)
        return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces,
                               process=False)


def hover_wp(person):
    h = person.handshake
    return dict(joint_angle_vector=h['joint_angle_vector'],
                base_position=h['base_position'], base_yaw=h['base_yaw'])


def press_wp(person):
    p = person.handshake['post_process']
    return dict(joint_angle_vector=p['joint_angle_vector'],
                base_position=p['base_position'], base_yaw=p['base_yaw'])


def obstacle_index(name):
    return spik.human_obstacle_names().index(name)


# ---------------------------------------------------------------------------
# 段階
# ---------------------------------------------------------------------------

class Stage(object):
    def __init__(self, key, title, notes):
        self.key = key
        self.title = title
        self.notes = notes
        self.scene = Scene()
        self.eye = None
        self.target = None
        self.yfov = 0.75
        self.draw_top = None  # f(ax)
        self.top_limits = None  # (xmin, xmax, ymin, ymax)
        self.legend = []
        # 全体表示でこの段階を表すロボットの姿勢 (waypoint, 関節名)。
        self.robot_wp = None
        self.robot_joint_names = None
        self.robot_label = None  # 全体表示のラベル (短い名前)

    @property
    def short_title(self):
        return self.title.split(' (')[0]

    def set_robot(self, wp, joint_names, label):
        self.robot_wp = dict(wp)
        self.robot_joint_names = joint_names
        self.robot_label = label


class StageBuilder(object):
    def __init__(self, person, visual_robot, ik_robot, verification_pairs,
                 collision_pairs):
        self.p = person
        self.visual = visual_robot
        self.ik = ik_robot
        self.verification_pairs = verification_pairs
        self.collision_pairs = collision_pairs
        self.joint_names = person.handshake['joint_names']
        self.robots = (visual_robot, ik_robot)

    # -- 共通 --------------------------------------------------------------
    def pose(self, wp, joint_names=None):
        set_both(self.robots, wp, joint_names or self.joint_names)

    def end_coords(self, robot=None):
        robot = robot or self.visual
        return getattr(robot, '{}arm_end_coords'.format(self.p.robot_arm))

    def default_view(self, stage, base_xy, azimuth=None, distance=None,
                     elevation=22.0, height=0.85, turn=0.0):
        """人の立ち位置と ``base_xy`` の中点を、人とロボットを結ぶ線の横
        (人の正面側) の斜め上から見る。``turn`` [deg] だけ回す。"""
        base_xy = np.asarray(base_xy[:2], dtype=np.float64)
        center = (self.p.standing + base_xy) / 2.0
        v = base_xy - self.p.standing
        if azimuth is None:
            line_yaw = math.atan2(v[1], v[0])
            front = np.array([math.cos(self.p.facing_yaw),
                              math.sin(self.p.facing_yaw)])
            options = [line_yaw + math.pi / 2.0, line_yaw - math.pi / 2.0]
            best = max(options, key=lambda a: math.cos(a) * front[0]
                       + math.sin(a) * front[1])
            azimuth = math.degrees(best) + turn
        if distance is None:
            distance = 2.3 + 1.0 * float(np.linalg.norm(v))
        stage.target = np.array([center[0], center[1], height])
        stage.eye = view_eye(stage.target, azimuth, elevation, distance)

    def top_limits_around(self, points, margin=0.6):
        points = np.array([np.asarray(q, dtype=np.float64)[:2]
                           for q in points])
        lo, hi = points.min(axis=0) - margin, points.max(axis=0) + margin
        span = max(hi - lo)
        c = (lo + hi) / 2.0
        return (c[0] - span / 2, c[0] + span / 2,
                c[1] - span / 2, c[1] + span / 2)

    def top_common(self, ax, robot_wps=(), shell=0.0, human_colors=None,
                   skip=(), robot_colors=None, joints=None):
        top_human(ax, joints or self.p.collision_joints, colors=human_colors,
                  shell=shell, skip=skip)
        ax.plot(*self.p.standing, 'o', color='tab:blue', ms=6, zorder=7)
        top_arrow(ax, self.p.standing, self.p.facing_yaw, (40, 90, 200, 255),
                  length=0.4, width=0.015)
        ax.annotate('人の正面', self.p.standing
                    + 0.45 * np.array([math.cos(self.p.facing_yaw),
                                       math.sin(self.p.facing_yaw)]),
                    color='tab:blue', fontsize=9, zorder=8)
        for wp, kw in robot_wps:
            kw = dict(kw)
            self.pose(wp, kw.pop('joint_names', None))
            top_robot(ax, self.ik, colors=robot_colors, **kw)
            top_arrow(ax, wp['base_position'], wp['base_yaw'],
                      (230, 110, 30, 255), length=0.3, width=0.012)

    def palm_markers(self, scene, hover=True, press=True):
        scene.layer('palm', '掌の位置と法線',
                    'estimate_palm_poses.py が求めた差し出した手の掌')
        scene.add(sphere(self.p.palm, 0.012), (40, 40, 40, 255))
        scene.add(arrow(self.p.palm, self.p.palm + 0.15 * self.p.palm_normal,
                        radius=0.006), (40, 40, 40, 255))
        if hover:
            scene.layer('hover_target', 'hover 目標',
                        '第 1 段の IK の目標。押し込みの最終目標を掌の法線方向に '
                        '{:.0f} cm 浮かせ、ロボットの指先と逆向きに {:.0f} cm '
                        'ずらした位置 (向きは最終目標と同じ)'.format(
                            100 * spik.PRESS_IN_LIFT,
                            100 * spik.PRESS_IN_SLIDE))
            scene.add(sphere(self.p.hover_target, 0.016), TARGET_HOVER_COLOR)
        if press:
            scene.layer('press_target', '押し込みの最終目標 (掌 - 法線 × 0.01 m)',
                        '第 2 段の IK の目標。掌に 1 cm めり込む位置')
            scene.add(sphere(self.p.press_target, 0.016), TARGET_PRESS_COLOR)
        if hover and press:
            scene.layer('press_curve', '押し込みの手先目標の曲線',
                        'hover 目標から、指先を {:.0f} cm 押し出しながら法線方向に '
                        '{:.0f} cm 詰める (詰めるのは終盤)'.format(
                            100 * spik.PRESS_IN_SLIDE,
                            100 * spik.PRESS_IN_LIFT))
            scene.add(tube(self.p.press_curve, 0.004), PRESS_CURVE_COLOR)

    def ik_pair_colors(self):
        """第 1 段の最適化で使う干渉ペアの色 (ロボットのリンク名, 人体の
        添字)。"""
        robot_colors, human_colors = {}, {}
        for link, other in self.collision_pairs:
            if isinstance(other, int):
                robot_colors[link.name] = PAIR_COLOR
                human_colors[other] = PAIR_COLOR
        # 差し出した前腕 (細らせた円柱) × ロボットの手先側の 4 リンク。
        # 掌・指は組に入れない (hover が掌から約 2 cm のため)。
        for name in spik.OFFERED_HAND_PENALTY_LINKS:
            robot_colors[name.format(self.p.robot_arm)] = OFFERED_COLOR
        human_colors[obstacle_index(
            '{0}Elbow-{0}Wrist'.format(self.p.hand))] = OFFERED_COLOR
        return robot_colors, human_colors

    def thresholds(self, clearance, hand_clearance=spik.OFFERED_HAND_CLEARANCE):
        """部位ごとに空ける距離 (検証と同じ ``human_clearance_thresholds``)。"""
        return spik.human_clearance_thresholds(self.p.hand, clearance,
                                               hand_clearance=hand_clearance)

    def shortfall(self, joints, thresholds):
        """検証と同じ判定で、余裕 (距離 - 空ける距離) が最小の部位
        ``(添字, 距離, 余裕)`` (無ければ None)。"""
        clearances = self.clearances(joints)
        index, margin = spik.clearance_shortfall(clearances, thresholds)
        if index is None:
            return None
        return index, clearances[index], margin

    def clearances(self, joints, clearance_pairs=None):
        pairs = clearance_pairs or spik.human_clearance_pairs(
            self.verification_pairs)
        obstacles = spik.human_body_obstacles(joints)
        return spik.human_obstacle_clearances(self.ik, pairs, obstacles,
                                              cull_distance=0.15)

    def self_min_distance(self, pairs=None):
        pairs = pairs or spik.self_collision_pairs(self.verification_pairs)
        dist, pair = spik.collision_pairs_min_distance(
            self.ik, pairs, None, return_pair=True)
        return dist, spik.collision_pair_name(pair)

    # -- 1. 差し出した手 -------------------------------------------------
    def stage_input(self):
        p = self.p
        stage = Stage(
            '01_offered_hand', '1. 差し出した手の検出と目標',
            ['骨格と手の landmark から差し出している手を選ぶ '
             '(OfferedHandSelector。スコアが閾値以上 (オフライン 0.86、'
             '実機ノード 0.65) で静止している手)。'
             '人の {} 手 → ロボットの {} 腕。'.format(
                 '左' if p.hand == 'L' else '右',
                 '右' if p.robot_arm == 'r' else '左'),
             '人を Aero の前方 3.0 m (HUMAN_FRONT_DISTANCE) へ平行移動した'
             '座標系で解く。ロボットは原点・+x 向きから出発する。',
             '押し込みの最終目標は掌 - 法線 × 0.01 m。hover 目標はそれを法線方向に '
             '{:.0f} cm 浮かせ、ロボットの指先と逆向きに {:.0f} cm ずらした位置で、'
             'そこから指先を押し出しながら詰める曲線で押し込む。'
             'この段階では干渉は見ない。'.format(100 * spik.PRESS_IN_LIFT,
                                          100 * spik.PRESS_IN_SLIDE)])
        start = (self.p.motion.get('lead_in_waypoints')
                 or self.p.motion['waypoints'])[0]
        stage.set_robot(start, self.joint_names, '初期位置')
        self.pose(start)
        s = stage.scene
        s.layer('robot', 'ロボット (初期位置)')
        s.add(robot_visual_mesh(self.visual), ROBOT_COLOR)
        s.layer('smpl', '人 (SMPL)')
        s.add(p.smpl_mesh(p.palm), SKIN_COLOR)
        s.layer('cyl', '人体の干渉円柱 (26 本)',
                '骨格から作る円柱。以降の段階の干渉判定はすべてこれで行う',
                default=False, colors=[HUMAN_CYL_COLOR])
        add_human_cylinders(s, p.collision_joints)
        self.palm_markers(s)
        stage.target = (p.palm + np.array([p.standing[0], p.standing[1],
                                           1.2])) / 2.0
        stage.eye = view_eye(stage.target,
                             math.degrees(p.facing_yaw) + 35.0, 15.0, 1.9)

        def draw(ax):
            self.top_common(ax, robot_wps=[(start, {})])
            ax.plot(*p.palm[:2], 'o', color='k', ms=5, zorder=8)
            ax.plot(*p.hover_target[:2], 'o', color=rgba(TARGET_HOVER_COLOR),
                    ms=6, zorder=8)
            ax.annotate('差し出した手', p.palm[:2] + [0.1, 0.1], fontsize=9,
                        zorder=8)
            ax.annotate('ロボット初期位置 (0, 0, 0°)', [0.0, -0.45],
                        fontsize=9, ha='center')
            ax.annotate('', xy=(p.standing[0] - 0.3, -0.9),
                        xytext=(0.3, -0.9),
                        arrowprops=dict(arrowstyle='<->', color='gray'))
            ax.text((p.standing[0]) / 2.0, -1.0, '3.0 m', ha='center',
                    va='top', color='gray', fontsize=9)
        stage.draw_top = draw
        stage.top_limits = self.top_limits_around(
            [[0.0, 0.0], p.standing, p.palm], margin=1.0)
        stage.legend = [
            legend_handle(HUMAN_CYL_COLOR, '人体の干渉円柱 (26 本)'),
            legend_handle((230, 150, 80, 110), 'ロボットの干渉形状'),
            legend_handle(TARGET_HOVER_COLOR, 'hover 目標', 'marker')]
        return stage

    # -- 2. 第 1 段 IK ----------------------------------------------------
    def stage_batch_ik(self):
        p = self.p
        region = p.handshake['base_movable_region']
        x_margin = p.handshake.get('base_x_standing_margin')
        stage = Stage(
            '02_batch_ik', '2. 第1段: 干渉回避付きバッチ IK (hover 姿勢と台車の位置を同時に解く)',
            ['目標: hover 目標 (押し込みの最終目標から法線方向に 3 cm・指先と逆に '
             '15 cm)、向き 3 通り × 初期値 512。'
             '台車 (x, y, yaw) も解く。関節可動域は全域の 10% ずつ内側に絞る。',
             '台車の可動域: x は人の立ち位置 ±{:.2f} m '
             '(0.15 → 0.3 → 無制限の順に広げる)、y は差し出した手の側、'
             'yaw は人の正面 ±30°。'.format(x_margin or 0.0),
             '干渉はコスト (効き始め 6 cm、重み 10) で、対象は collision_pairs.json '
             'の 8 組 (台車・胴体・肩・上腕 × 人の脛・体幹) と、'
             'ロボットの手・親指・手首・前腕 × 差し出した前腕の 4 組だけ。前腕は '
             '{:.0f} cm で効くよう細らせ、掌・指は組に入れない (hover が掌から約 2 cm '
             'のため)。コストなので干渉が残っても収束扱いになり、必ず事後検証を'
             '挟む。'.format(100 * spik.OFFERED_HAND_CLEARANCE)])
        wp = hover_wp(p)
        stage.set_robot(wp, self.joint_names, 'hover')
        self.pose(wp)
        s = stage.scene
        s.layer('visual', 'ロボット (見た目)', default=False)
        s.add(robot_visual_mesh(self.visual), ROBOT_COLOR)
        s.layer('pairs', '最適化の干渉ペア (8 組)',
                'collision_pairs.json: 台車・胴体・肩・上腕 × 人の脛・体幹 '
                '(左右の肩–腰)。事前調査で干渉の多かった組',
                colors=[PAIR_COLOR])
        s.layer('offered', '差し出した前腕まわりの 4 組',
                'ロボットの手・親指の箱・手首・前腕 × 人の差し出した前腕。'
                '前腕の円柱は 2 cm で効くよう細らせて使う (表示は元の太さ)。'
                '掌・指は組に入れない',
                colors=[OFFERED_COLOR])
        s.layer('coll', 'ロボットのその他の干渉形状 (コストなし)',
                '指なしモデルのプリミティブ + 手の箱 2 個',
                colors=[ROBOT_COLL_COLOR])
        s.layer('cyl', '人体のその他の円柱 (コストなし)',
                colors=[HUMAN_CYL_COLOR])
        robot_colors, human_colors = self.ik_pair_colors()
        add_robot_collision(s, self.ik, robot_colors)
        add_human_cylinders(s, p.collision_joints, human_colors)
        s.layer('smpl', '人 (SMPL)')
        s.add(p.smpl_mesh(self.end_coords().worldpos()), SKIN_FADED_COLOR)
        self.palm_markers(s, press=False)
        base = np.asarray(wp['base_position'])
        y_range = clip_range(region['y_range'], p.standing[1],
                             REGION_DISPLAY_HALF)
        s.layer('region', '台車の位置の可動域 (x, y)',
                'x: 人の立ち位置 ±{:.2f} m、y: 差し出した手の側 '
                '(表示は ±{:.1f} m で切っている)'.format(
                    x_margin or 0.0, REGION_DISPLAY_HALF))
        s.add(floor_polygon(region_xy(region['x_range'], y_range)),
              REGION_COLOR)
        s.layer('yaw', '台車の向きの範囲 (人の正面 ±30°)',
                '矢印は解いた台車の向き')
        s.add(floor_polygon(fan_xy(base, region['yaw_range'], 0.6),
                            z=FLOOR_Z * 2), YAW_FAN_COLOR)
        s.add(arrow(base + [0, 0, 0.02], base + [
            0.6 * math.cos(wp['base_yaw']), 0.6 * math.sin(wp['base_yaw']),
            0.02]), (230, 110, 30, 255))
        self.default_view(stage, base)

        def draw(ax):
            top_region(ax, region_xy(region['x_range'], y_range),
                       REGION_COLOR)
            top_region(ax, fan_xy(base, region['yaw_range'], 0.6),
                       YAW_FAN_COLOR, zorder=1)
            self.top_common(ax, robot_wps=[(wp, {})],
                            human_colors=human_colors,
                            robot_colors=robot_colors)
            ax.plot(*p.hover_target[:2], 'o', color=rgba(TARGET_HOVER_COLOR),
                    ms=6, zorder=8)
            ax.axvline(p.standing[0], color='tab:blue', lw=0.8, ls=':')
            ax.text(region['x_range'][1], y_range[1] - 0.05,
                    ' x: 立ち位置 ±{:.2f} m'.format(x_margin or 0.0),
                    color='green', fontsize=9, va='top')
            ax.text(base[0] + 0.62 * math.cos(region['yaw_range'][1]),
                    base[1] + 0.62 * math.sin(region['yaw_range'][1]),
                    'yaw: 人の正面 ±30°', color='darkorange', fontsize=9)
        stage.draw_top = draw
        stage.top_limits = self.top_limits_around(
            [p.standing, base, [region['x_range'][0], y_range[0]],
             [region['x_range'][1], y_range[1]]], margin=0.3)
        stage.legend = [
            legend_handle(REGION_COLOR, '台車の可動域 (x, y)'),
            legend_handle(YAW_FAN_COLOR, '台車の向きの範囲'),
            legend_handle(PAIR_COLOR, '最適化の干渉ペア (8 組)'),
            legend_handle(OFFERED_COLOR, '差し出した前腕まわりの 4 組'),
            legend_handle(HUMAN_CYL_COLOR, 'その他の人体 (コストなし)'),
            legend_handle(TARGET_HOVER_COLOR, 'hover 目標', 'marker')]
        return stage

    # -- 3. hover の事後検証 ---------------------------------------------
    def stage_hover_verify(self):
        p = self.p
        wp = hover_wp(p)
        # 差し出さない腕は押し込みと同じ姿勢に差し替え済み (他の腕の差し替え)。
        self.pose(wp)
        clearance = spik.DEFAULT_HOVER_HUMAN_CLEARANCE
        hand_clearance = spik.OFFERED_HAND_CLEARANCE
        names = spik.human_obstacle_names()
        result = self.shortfall(p.collision_joints,
                                self.thresholds(clearance, hand_clearance))
        closest = result[0] if result is not None else None
        closest_text = '{} まで {:.1f} cm (余裕 {:+.1f} cm)'.format(
            names[closest], 100.0 * result[1], 100.0 * result[2]) \
            if result is not None else '-'
        self_dist, self_pair = self.self_min_distance()
        stage = Stage(
            '03_hover_verify', '3. 事後検証: hover 姿勢 (人体から 6 cm・手まわり 2 cm、自己干渉 0 mm)',
            ['収束した候補をコスト (関節の曲げ量 + 30 × 前方ずれ + 30 × 向きのずれ) '
             'の安い順に検証し、通った最初の候補で第 2 段 (押し込み) を解く。',
             '人体との距離: 指ありモデル (表示しているロボット) の表面から全身 26 本'
             'の円柱まで {:.0f} cm 以上。ただし差し出した手まわり (掌・指・同じ側の'
             '前腕) は hover が掌から約 2 cm なので {:.0f} cm。この段階で初めて'
             '全身を見る。この人の余裕が最小の部位: {}。'.format(
                 100 * clearance, 100 * hand_clearance, closest_text),
             '自己干渉: 指なし + 手の箱 2 個のモデルで貫通が 0 mm を超えないこと '
             '(この人の最小距離 {:+.1f} mm: {} × {})。差し出さない腕は台車の箱から 2 cm 離す姿勢に'
             '差し替えて再検証する。'.format(1000.0 * self_dist, *self_pair)])
        stage.set_robot(wp, self.joint_names, '')
        s = stage.scene
        s.layer('visual', 'ロボット (指ありモデル)',
                '人体との距離はこのモデルの干渉形状で測る')
        s.add(robot_visual_mesh(self.visual), ROBOT_COLOR)
        s.layer('coll', 'ロボットの干渉形状 (指なし + 手の箱)',
                '自己干渉の判定に使うモデル', default=False,
                colors=[ROBOT_COLL_COLOR])
        add_robot_collision(s, self.ik)
        s.layer('closest', '余裕が最小の人体の部位', closest_text,
                colors=[CLOSEST_COLOR])
        s.layer('cyl', '人体の干渉円柱 (26 本)', colors=[HUMAN_CYL_COLOR])
        s.layer('shell', '人体 + 6 cm (近づけない範囲)',
                '円柱を 6 cm 膨らませた形。ロボットの表面がこの中に入ると棄却',
                colors=[HUMAN_SHELL_COLOR])
        s.layer('hand_shell', '差し出した手まわり + 2 cm',
                '掌・指・同じ側の前腕だけは 2 cm (hover が掌から約 2 cm のため)',
                colors=[HAND_SHELL_COLOR])
        colors = {closest: CLOSEST_COLOR} if closest is not None else {}
        shell = hand_shell(p.hand, clearance, hand_clearance)
        add_human_cylinders(s, p.collision_joints, colors, shell=shell,
                            hand_shell_color=HAND_SHELL_COLOR, hand=p.hand)
        s.layer('smpl', '人 (SMPL)')
        s.add(p.smpl_mesh(self.end_coords().worldpos()), SKIN_FADED_COLOR)
        self.palm_markers(s, press=False)
        base = np.asarray(wp['base_position'])
        self.default_view(stage, base, turn=-20.0)
        zoom(stage, p.palm, 0.9)

        def draw(ax):
            self.top_common(ax, robot_wps=[(wp, {})], shell=shell,
                            human_colors=colors)
            if result is not None:
                obstacle = spik.human_body_obstacles(
                    p.collision_joints)[closest]
                ax.annotate('{:.1f} cm (余裕 {:+.1f} cm)'.format(
                    100.0 * result[1], 100.0 * result[2]),
                    obstacle.worldpos()[:2] + [-0.1, 0.1], fontsize=9,
                    color='darkgoldenrod', zorder=9)
        stage.draw_top = draw
        stage.top_limits = self.top_limits_around([p.standing, base, p.palm],
                                                  margin=0.6)
        stage.legend = [
            legend_handle(HUMAN_SHELL_COLOR, '人体 + 6 cm (手まわりは 2 cm)'),
            legend_handle(CLOSEST_COLOR, '余裕が最小の部位'),
            legend_handle((230, 150, 80, 110), 'ロボットの干渉形状')]
        return stage

    # -- 4. 押し込み + 視線 ----------------------------------------------
    def stage_press(self):
        p = self.p
        wp = press_wp(p)
        self.pose(wp)
        post = p.handshake['post_process']
        camera = spik.camera_optical_coords(self.ik)
        cam_pos = camera.worldpos()
        axis = camera.worldrot()[:, 2]
        to_palm = p.palm - cam_pos
        gaze_err = math.degrees(math.acos(float(np.clip(
            np.dot(axis, to_palm / np.linalg.norm(to_palm)), -1, 1))))
        self_dist, self_pair = self.self_min_distance()
        offered = spik.offered_hand_obstacle_indices(p.hand)
        stage = Stage(
            '04_press_in', '4. 第2段: 押し込み姿勢 + 視線 (台車は動かさない)',
            ['腕: 掌 - 法線 × 0.01 m (掌に 1 cm めり込む位置) へ、第 1 段と同じ向きで '
             '6 自由度 (0.01 m / 5°)。首: カメラ光軸を掌へ (5°、首ロールは 0° 固定)。'
             'この人の視線のずれ {:.1f}°{}。'.format(
                 gaze_err, '、腕だけで解いた代替' if post.get('arm_only') else ''),
             '人体との判定はしない (人の掌に触れる姿勢のため)。'
             '自己干渉だけ検証 (最小 {:+.1f} mm: {} × {})。'.format(
                 1000.0 * self_dist, *self_pair),
             'hover → 押し込みの区間も経路としては計画・検証しない。'
             '実機では hover 到着後に掌を検出し直して押し込みを解き直す '
             '(refine_post_process)。'])
        stage.set_robot(wp, self.joint_names, '押し込み')
        s = stage.scene
        s.layer('visual', 'ロボット (押し込み姿勢)')
        s.add(robot_visual_mesh(self.visual), ROBOT_COLOR)
        s.layer('coll', 'ロボットの干渉形状 (自己干渉の判定)',
                default=False, colors=[ROBOT_COLL_COLOR])
        add_robot_collision(s, self.ik)
        s.layer('excluded', '判定しない: 差し出した手・前腕',
                '掌に触れる姿勢なので当たるのが正しい (人体との判定は全身しない)',
                colors=[EXCLUDED_COLOR])
        s.layer('cyl', '人体のその他の円柱 (この段階は判定なし)',
                colors=[(80, 140, 220, 40)])
        colors = {i: EXCLUDED_COLOR for i in offered}
        add_human_cylinders(s, p.collision_joints, colors,
                            default=(80, 140, 220, 40))
        s.layer('smpl', '人 (SMPL)')
        s.add(p.smpl_mesh(cam_pos), SKIN_FADED_COLOR)
        self.palm_markers(s, hover=True)
        s.layer('gaze', 'カメラ光軸 (視線)',
                '首のヨー・ピッチでカメラの光軸を掌へ向ける。'
                'ずれ {:.1f}°'.format(gaze_err))
        s.add(tube([cam_pos, cam_pos + axis * np.linalg.norm(
            to_palm) * 1.05], 0.006), GAZE_COLOR)
        # hover の手先からの押し込みの向き。
        self.pose(hover_wp(p))
        hover_hand = self.end_coords().worldpos()
        self.pose(wp)
        s.layer('press_dir', 'hover → 押し込みの手先の動き',
                '経路としては計画・検証しない区間')
        s.add(arrow(hover_hand, self.end_coords().worldpos(),
                    radius=0.006), TARGET_PRESS_COLOR)
        base = np.asarray(wp['base_position'])
        self.default_view(stage, base, turn=25.0, elevation=18.0)
        zoom(stage, p.palm, 0.9)

        def draw(ax):
            self.top_common(ax, robot_wps=[(wp, {})], human_colors=colors)
            ax.plot([cam_pos[0], p.palm[0]], [cam_pos[1], p.palm[1]],
                    color=rgba(GAZE_COLOR)[:3], lw=2, zorder=9)
            ax.annotate('視線', (cam_pos[:2] + p.palm[:2]) / 2.0,
                        fontsize=9, color='goldenrod', zorder=9)
        stage.draw_top = draw
        stage.top_limits = self.top_limits_around([p.standing, base, p.palm],
                                                  margin=0.6)
        stage.legend = [
            legend_handle(EXCLUDED_COLOR, '判定しない (差し出した手・前腕)'),
            legend_handle(GAZE_COLOR, 'カメラ光軸', 'line'),
            legend_handle(TARGET_PRESS_COLOR, '押し込み目標', 'marker')]
        return stage

    # -- 5. 接近軌道 ------------------------------------------------------
    def stage_approach(self):
        p = self.p
        motion = p.motion
        lead_in = motion.get('lead_in_waypoints') or []
        waypoints = motion['waypoints']
        base_goal = phm.handshake_base_goal(p.handshake)
        center = phm.orbit_center_xy(p.handshake)
        initial = np.array(phm.INITIAL_BASE_POSE)
        candidates = phm.approach_start_candidates(
            base_goal, p.standing, phm.DEFAULT_APPROACH_DISTANCE, initial,
            orbit_center=center)
        radius = (float(np.linalg.norm(base_goal[:2] - center))
                  + phm.DEFAULT_APPROACH_DISTANCE)
        # 各 waypoint の人体までの最短距離 (検証と同じ指ありモデル)。
        closest = (float('inf'), None, None)
        for k, wp in enumerate(waypoints):
            self.pose(wp)
            clearances = self.clearances(p.collision_joints)
            if clearances:
                index = min(clearances, key=clearances.get)
                if clearances[index] < closest[0]:
                    closest = (clearances[index], k, index)
        kinds = {'pretouch': 'pre-touch 経由', 'linear': '線形補間',
                 'optimized': 'jaxls 最適化'}
        stage = Stage(
            '05_approach', '5. 接近軌道 (初期位置 → lead-in → 螺旋で hover へ)',
            ['lead-in: その場回転 → 直進。接近開始位置は手を中心とする'
             '半径 (手〜最終台車位置 + 0.3 m) の円上で、lead-in が螺旋の接線に'
             'なる位置。通らなければ ±30° 刻みで ±120° まで回した候補を試す。',
             '接近: 20 waypoint、両腕を下ろした姿勢 → hover。'
             'pre-touch 経由 → 線形補間 → jaxls 最適化 の順に作り最初に通ったものを'
             '採る (この人: {}、角度 {:.0f}°)。台車の経路の範囲は制限しない。'.format(
                 kinds.get(motion['kind'], motion['kind']),
                 math.degrees(motion.get('approach_angle') or 0.0)),
             '検証: 全 waypoint で人体から 5 cm・自己干渉 0 mm。lead-in は'
             '人から 1.0 m 以内の waypoint だけ検証。'
             'この人の経路の最短: waypoint {} で {} まで {:.1f} cm。'.format(
                 closest[1], spik.human_obstacle_names()[closest[2]]
                 if closest[2] is not None else '-', 100.0 * closest[0])])
        stage.set_robot(waypoints[0], self.joint_names, '接近開始')
        s = stage.scene
        s.layer('ghosts', 'ロボット: 接近の途中 (waypoint 0, 1/3, 2/3)',
                '半透明。waypoint 0 が接近開始位置')
        for index in (0, len(waypoints) // 3, 2 * len(waypoints) // 3):
            self.pose(waypoints[index])
            s.add(robot_visual_mesh(self.visual), ROBOT_GHOST_COLOR)
        self.pose(waypoints[-1])
        s.layer('visual', 'ロボット: 終点 (hover)')
        s.add(robot_visual_mesh(self.visual), ROBOT_COLOR)
        s.layer('cyl', '人体の干渉円柱 (26 本)', colors=[HUMAN_CYL_COLOR])
        s.layer('shell', '人体 + 5 cm (経路が入ってはいけない範囲)',
                'hover の IK は 6 cm、経路の検証は 1 cm 緩い 5 cm',
                colors=[HUMAN_SHELL_COLOR])
        add_human_cylinders(s, p.collision_joints,
                            shell=phm.DEFAULT_HUMAN_CLEARANCE)
        s.layer('smpl', '人 (SMPL)')
        s.add(p.smpl_mesh(self.end_coords().worldpos()), SKIN_FADED_COLOR)
        z = FLOOR_Z * 3
        if lead_in:
            s.layer('lead_in', 'lead-in (初期位置から回転 → 直進)',
                    '最適化しない。人から 1 m 以内の waypoint だけ検証')
            pts = [wp['base_position'][:2] for wp in lead_in + waypoints[:1]]
            s.add(tube([[x, y, z] for x, y in pts], 0.015), LEAD_IN_COLOR)
        s.layer('path', '接近の台車の経路 (螺旋)',
                '手を中心に公転 + 自転。pre-touch 経由 → 線形補間 → jaxls '
                '最適化の順に作り、最初に検証を通ったもの')
        pts = [wp['base_position'][:2] for wp in waypoints]
        s.add(tube([[x, y, z] for x, y in pts], 0.018), PATH_COLOR)
        s.layer('orbit', '接近開始位置を置く円',
                '手を中心に、半径 = 手〜最終台車位置 + 0.3 m')
        s.add(ring(center, radius, width=0.006), (60, 60, 60, 255))
        s.layer('check', 'lead-in の検証範囲 (人から 1.0 m)')
        s.add(ring(p.standing, phm.LEAD_IN_CHECK_RADIUS, width=0.008),
              (220, 60, 60, 255))
        s.layer('start0', '接近開始位置 (角度 0)',
                'lead-in が螺旋の接線になる位置')
        s.layer('starts', '回り込みの候補 (±30°〜±120°)',
                '角度 0 が通らないときに経路の短い順に試す')
        for angle, start in candidates:
            s.layer('start0' if angle == 0.0 else 'starts', '')
            s.add(sphere([start[0], start[1], 0.03],
                         0.04 if angle == 0.0 else 0.025),
                  (40, 90, 220, 255) if angle == 0.0
                  else (150, 150, 150, 255))
        # ロボットの側の斜め上から、接近開始位置と人を入れて見る。
        start_xy = np.asarray(waypoints[0]['base_position'][:2])
        side = start_xy - p.standing
        stage.target = np.array([(start_xy[0] + p.standing[0]) / 2.0,
                                 (start_xy[1] + p.standing[1]) / 2.0, 0.4])
        stage.eye = view_eye(stage.target, math.degrees(
            math.atan2(side[1], side[0])) + 40.0, 50.0, 4.3)

        def draw(ax):
            ax.add_patch(patches.Circle(
                p.standing, phm.LEAD_IN_CHECK_RADIUS, fc=(1, 0.3, 0.3, 0.08),
                ec=(0.85, 0.2, 0.2), lw=1.0, zorder=0))
            ax.add_patch(patches.Circle(center, radius, fc='none', ec='0.4',
                                        lw=1.0, ls=':', zorder=1))
            self.top_common(ax, robot_wps=[(waypoints[-1], {})],
                            shell=phm.DEFAULT_HUMAN_CLEARANCE)
            for wp in (lead_in[:1] if lead_in else []):
                self.pose(wp)
                top_robot(ax, self.ik, color=(150, 150, 150, 70))
                top_arrow(ax, wp['base_position'], wp['base_yaw'],
                          (120, 120, 120, 255), length=0.3, width=0.012)
            if lead_in:
                pts = np.array([wp['base_position'][:2]
                                for wp in lead_in + waypoints[:1]])
                ax.plot(pts[:, 0], pts[:, 1], '--', color='0.4', lw=2,
                        zorder=5)
            pts = np.array([wp['base_position'][:2] for wp in waypoints])
            ax.plot(pts[:, 0], pts[:, 1], '-', color=rgba(PATH_COLOR)[:3],
                    lw=2.5, zorder=5)
            for angle, start in candidates:
                if angle == 0.0:
                    ax.plot(start[0], start[1], '*', color='tab:blue', ms=14,
                            zorder=8)
                else:
                    ax.plot(start[0], start[1], 'o', mfc='white', mec='0.4',
                            ms=6, zorder=8)
                    ax.annotate('{:+.0f}°'.format(math.degrees(angle)),
                                start[:2] + [0.06, 0.04], fontsize=7,
                                color='0.4', zorder=8)
            ax.plot(*center, 'x', color='k', ms=7, zorder=8)
            ax.annotate('螺旋の中心 (手)', center + [0.1, 0.12], fontsize=8,
                        zorder=9)
            ax.annotate('lead-in の検証範囲 1.0 m', p.standing
                        + [-0.9, -phm.LEAD_IN_CHECK_RADIUS - 0.18],
                        color=(0.8, 0.2, 0.2), fontsize=9)
            ax.annotate('初期位置', [0.05, 0.25], fontsize=9, color='0.3')
        stage.draw_top = draw
        all_pts = ([[0.0, 0.0], p.standing]
                   + [c[1][:2] for c in candidates]
                   + [wp['base_position'][:2] for wp in waypoints])
        stage.top_limits = self.top_limits_around(all_pts, margin=0.5)
        stage.legend = [
            legend_handle(LEAD_IN_COLOR, 'lead-in (回転 → 直進)', 'dash'),
            legend_handle(PATH_COLOR, '接近の台車の経路', 'line'),
            legend_handle(HUMAN_SHELL_COLOR, '人体 + 5 cm'),
            legend_handle((220, 60, 60, 60), 'lead-in の検証範囲'),
            plt.Line2D([], [], color='tab:blue', marker='*', ls='', ms=12,
                       label='接近開始位置 (角度 0)'),
            plt.Line2D([], [], color='0.4', marker='o', mfc='white', ls='',
                       label='回り込みの候補 (±30°〜±120°)')]
        return stage

    # -- 6. 横並び移動 ----------------------------------------------------
    def stage_side_by_side(self):
        p = self.p
        transition = p.motion['transition']
        names = transition['joint_names']
        wps = transition['waypoints']
        press_arm = transition['press_arm']
        human_arm = sbs.HumanArm(p.joints, transition['hand'],
                                 sbs.palm_from_frame(
                                     np.asarray(press_arm['palm_position']),
                                     np.asarray(press_arm['palm_rot'])))
        final_arm = wps[-1]['human_arm']
        final_joints = human_arm.skeleton(final_arm)
        x_margin = transition.get('x_margin') or sbs.DEFAULT_GOAL_X_MARGINS[0]
        facing = p.facing_yaw
        rot = np.array([[math.cos(facing), -math.sin(facing)],
                        [math.sin(facing), math.cos(facing)]])
        side = 1.0 if p.hand == 'L' else -1.0
        # 人の正面を +x とした座標系での窓 (x: ±x_margin、y: 手の側)。
        local = region_xy((-x_margin, x_margin),
                          sorted((0.0, side * REGION_DISPLAY_HALF)))
        window = (local @ rot.T) + p.standing
        yaw_range = (facing - math.radians(30.0), facing + math.radians(30.0))
        final_wp = dict(wps[-1])
        start_wp = dict(wps[0])

        # 移動先のバッチ IK で足す組と、検証で除く組。
        robot_extra = {name.format(p.robot_arm): EXTRA_PAIR_COLOR
                       for name in sbs.TRANSITION_ARM_LINKS}
        torso = obstacle_index('{0}Shoulder-{0}Hip'.format(p.hand))
        human_extra = {i: EXTRA_PAIR_COLOR
                       for i in sbs._lowered_arm_slots(p.hand) + [torso]}
        excluded = spik.offered_hand_obstacle_indices(p.hand)
        human_colors = dict(human_extra)
        for i in excluded:
            human_colors.setdefault(i, EXCLUDED_COLOR)
        after = transition.get('placement_after') or {}
        facing_name = sbs._PALM_FACING_NAMES.get(
            transition.get('palm_facing'), transition.get('palm_facing'))
        stage = Stage(
            '06_side_by_side', '6. 押し込み後の横並び移動 (手をつないだまま人の横へ)',
            ['人の腕のモデル: 肩は固定、上腕を真下から前へ {:.0f}° (0/15/30° → '
             '45/60/75° の順)、外へ {:.0f}° 開き肘を 15° 曲げた形へ肩まわりに下ろす'
             '(掌の向き: {})。'.format(
                 transition.get('arm_deg', 0.0),
                 transition.get('abduction_deg', 25.0), facing_name),
             '移動先: 台車も動かすバッチ IK (初期値 256)。台車の窓は人の立ち位置'
             'から前後 ±{:.2f} m・差し出した手の側、向きは人の正面 ±30°。'
             '肩ヨー・手首ヨーは押し込みから ±60° 以内 (同じ腕の枝)。'
             'この人: 前方ずれ {:.2f} m、向きのずれ {:.0f}°。'.format(
                 x_margin, after.get('front_offset', 0.0),
                 after.get('yaw_offset_deg', 0.0)),
             '干渉: 移動先の IK では押し込みの組に、ロボットの前腕・肘 × 下ろした上腕・'
             '同じ側の胴体 (と腰まわりの自己干渉) を足す。検証は人体から 5 cm・'
             '自己干渉 0 mm、ただしロボットの肘から先 × 人の差し出した手・前腕は手を'
             'つないでいるので除く。経路は 10 cm・10° 刻み ({} waypoint)。'.format(
                 len(wps))])
        stage.set_robot(final_wp, names, '横並び')
        s = stage.scene
        s.layer('start', 'ロボット: 押し込み時 (移動の始点)', '半透明')
        set_both(self.robots, start_wp, names)
        s.add(robot_visual_mesh(self.visual), ROBOT_GHOST_COLOR)
        set_both(self.robots, final_wp, names)
        s.layer('visual', 'ロボット: 横並び (移動先)')
        s.add(robot_visual_mesh(self.visual), ROBOT_COLOR)
        s.layer('extra', '移動先の IK で足す干渉ペア',
                'ロボットの差し出す腕の前腕・肘 × 下ろした人の上腕・同じ側の胴体。'
                '押し込みの IK には無い組で、移動先の検証で落ちる候補はほぼこれ',
                colors=[EXTRA_PAIR_COLOR])
        for link in collision_links(self.ik):
            if link.name in robot_extra:
                s.add(collision_mesh(link), EXTRA_PAIR_COLOR)
        s.layer('excluded', '検証で除く: つないだ手・前腕',
                'ロボットの肘から先 × 人の差し出した手・前腕は手をつないで'
                'いるので見ない', colors=[EXCLUDED_COLOR])
        s.layer('cyl', '人体のその他の円柱 (腕は下ろした後)',
                colors=[HUMAN_CYL_COLOR])
        s.layer('shell', '人体 + 5 cm (近づけない範囲)',
                colors=[HUMAN_SHELL_COLOR])
        add_human_cylinders(s, final_joints, human_colors,
                            shell=sbs.TRANSITION_HUMAN_CLEARANCE)
        gaze = sbs.forward_gaze_point(p.joints)
        s.layer('smpl', '人 (SMPL、腕を下ろした後)')
        s.add(p.smpl_mesh(gaze, final_arm, human_arm), SKIN_FADED_COLOR)
        s.layer('region', '移動先の台車の窓',
                '人の立ち位置から前後 ±{:.2f} m・差し出した手の側'.format(
                    x_margin))
        s.add(floor_polygon(window), REGION_COLOR)
        base = np.asarray(final_wp['base_position'])
        s.layer('yaw', '移動先の台車の向きの範囲 (人の正面 ±30°)')
        s.add(floor_polygon(fan_xy(base, yaw_range, 0.6), z=FLOOR_Z * 2),
              YAW_FAN_COLOR)
        z = FLOOR_Z * 3
        s.layer('path', '台車の経路', '人の立ち位置を中心に距離と方位を補間')
        s.add(tube([[w['base_position'][0], w['base_position'][1], z]
                    for w in wps], 0.018), TRANSITION_PATH_COLOR)
        palms = [w['human_arm']['palm_position'] for w in wps]
        s.layer('hand_path', 'つないだ手の経路',
                '人が肩を中心に腕を下ろす円弧')
        s.add(tube(palms, 0.008), (200, 120, 60, 255))
        self.default_view(stage, base, turn=-15.0)

        def draw(ax):
            top_region(ax, window, REGION_COLOR)
            top_region(ax, fan_xy(base, yaw_range, 0.6), YAW_FAN_COLOR,
                       zorder=1)
            set_both(self.robots, start_wp, names)
            top_robot(ax, self.ik, color=(150, 150, 150, 60))
            self.top_common(ax, robot_wps=[(final_wp,
                                            dict(joint_names=names))],
                            shell=sbs.TRANSITION_HUMAN_CLEARANCE,
                            human_colors=human_colors, joints=final_joints)
            pts = np.array([w['base_position'][:2] for w in wps])
            ax.plot(pts[:, 0], pts[:, 1], '-', color=rgba(
                TRANSITION_PATH_COLOR)[:3], lw=2.5, zorder=6)
            pal = np.array(palms)
            ax.plot(pal[:, 0], pal[:, 1], '-', color=(0.8, 0.5, 0.25), lw=1.5,
                    zorder=6)
            d = pts[-1] - pts[0]
            d = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-3 \
                else np.array([1.0, 0.0])
            ax.annotate('押し込み時', pts[0] - 0.3 * d, fontsize=9,
                        color='0.3', zorder=9, ha='center')
            ax.annotate('横並び', pts[-1] + 0.25 * d, fontsize=9,
                        zorder=9, ha='center')
            ax.text(window[:, 0].max(), window[:, 1].max(),
                    ' 前後 ±{:.2f} m'.format(x_margin), color='green',
                    fontsize=9, va='top')
        stage.draw_top = draw
        stage.top_limits = self.top_limits_around(
            [p.standing, base, wps[0]['base_position']], margin=0.9)
        stage.legend = [
            legend_handle(REGION_COLOR, '移動先の台車の窓'),
            legend_handle(YAW_FAN_COLOR, '移動先の向きの範囲'),
            legend_handle(TRANSITION_PATH_COLOR, '台車の経路', 'line'),
            legend_handle((200, 120, 60, 255), 'つないだ手の経路', 'line'),
            legend_handle(EXTRA_PAIR_COLOR, '移動先 IK で足す干渉ペア'),
            legend_handle(EXCLUDED_COLOR, '検証で除く (つないだ手・前腕)'),
            legend_handle(HUMAN_SHELL_COLOR, '人体 + 5 cm')]
        return stage

    def build(self):
        stages = [self.stage_input(), self.stage_batch_ik(),
                  self.stage_hover_verify(), self.stage_press(),
                  self.stage_approach()]
        transition = self.p.motion.get('transition') or {}
        if transition.get('verified'):
            stages.append(self.stage_side_by_side())
        else:
            print('横並び移動が未計画か検証を通っていないので段階 6 は省きます。')
        return stages

    # -- 全体 -------------------------------------------------------------
    def overview(self, stages):
        """全段階のロボットを 1 つのシーンに並べた全体表示 (ロボットは段階
        ごとのレイヤーで、viser ではクリックするとその段階を開く)。同じ
        姿勢の段階 (hover の 2・3) は 1 体にまとめる。"""
        p = self.p
        stage = Stage('00_all_stages', '全体: 各段階のロボット',
                      ['ロボットをクリックすると、その段階で考慮する干渉・'
                       '可動域を表示する。',
                       '実行の順は 1 (初期位置) → 5 (接近) → 2〜4 (hover → '
                       '押し込み) → 6 (横並び)。番号は計画する順。'])
        s = stage.scene
        groups = []  # [(waypoint, 関節名, [段階の添字])]
        for i, st in enumerate(stages):
            if st.robot_wp is None:
                continue
            for wp, names, members in groups:
                same = (names == st.robot_joint_names
                        and np.allclose(wp['joint_angle_vector'],
                                        st.robot_wp['joint_angle_vector'])
                        and np.allclose(wp['base_position'],
                                        st.robot_wp['base_position']))
                if same:
                    members.append(i)
                    break
            else:
                groups.append((st.robot_wp, st.robot_joint_names, [i]))
        stage.labels = []
        placed = []  # 置いたラベルの水平位置 (近いものは高さをずらす)
        for k, (wp, names, members) in enumerate(groups):
            numbers = '・'.join(stages[i].short_title.split('.')[0]
                               for i in members)
            text = '{} {}'.format(numbers, '・'.join(
                stages[i].robot_label for i in members
                if stages[i].robot_label))
            s.layer('robot{}'.format(k), 'ロボット {}'.format(text),
                    'クリックで段階 {} を開く'.format(numbers),
                    click=members[0])
            set_both(self.robots, wp, names)
            s.add(robot_visual_mesh(self.visual),
                  OVERVIEW_ROBOT_COLORS[k % len(OVERVIEW_ROBOT_COLORS)])
            xy = np.asarray(wp['base_position'][:2], dtype=np.float64)
            rank = sum(1 for q in placed if np.linalg.norm(q - xy) < 1.0)
            placed.append(xy)
            stage.labels.append((text, np.array(
                [xy[0], xy[1], 1.8 + 0.17 * rank])))
        motion = p.motion
        lead_in = motion.get('lead_in_waypoints') or []
        z = FLOOR_Z * 3
        if lead_in:
            s.layer('lead_in', 'lead-in (初期位置 → 接近開始位置)')
            s.add(tube([[w['base_position'][0], w['base_position'][1], z]
                        for w in lead_in + motion['waypoints'][:1]], 0.015),
                  LEAD_IN_COLOR)
        s.layer('path', '接近の台車の経路')
        s.add(tube([[w['base_position'][0], w['base_position'][1], z]
                    for w in motion['waypoints']], 0.018), PATH_COLOR)
        transition = motion.get('transition') or {}
        if transition.get('verified'):
            s.layer('transition', '横並び移動の台車の経路')
            s.add(tube([[w['base_position'][0], w['base_position'][1], z]
                        for w in transition['waypoints']], 0.018),
                  TRANSITION_PATH_COLOR)
        s.layer('smpl', '人 (SMPL)')
        self.pose(hover_wp(p))
        s.add(p.smpl_mesh(self.end_coords().worldpos()), SKIN_COLOR)
        s.layer('cyl', '人体の干渉円柱 (26 本)', default=False,
                colors=[HUMAN_CYL_COLOR])
        add_human_cylinders(s, p.collision_joints)
        # 人のまわりに集まるロボットを横から、初期位置も入る向きで見る。
        cluster = np.mean([np.asarray(wp['base_position'][:2])
                           for wp, _, _ in groups[1:]] or [p.standing],
                          axis=0)
        start = np.asarray(groups[0][0]['base_position'][:2])
        d = cluster - start
        stage.target = np.array([start[0] + 0.62 * d[0],
                                 start[1] + 0.62 * d[1], 0.8])
        stage.eye = view_eye(stage.target, math.degrees(
            math.atan2(d[1], d[0])) - 75.0, 28.0, 4.8)
        return stage


# ---------------------------------------------------------------------------
# 保存・表示
# ---------------------------------------------------------------------------

def save_stage_figure(stage, image, path, subtitle):
    fig = plt.figure(figsize=(16, 10), dpi=100)
    fig.text(0.012, 0.97, stage.title, fontsize=17,
             va='top')
    fig.text(0.988, 0.97, subtitle, fontsize=9, color='0.4', va='top',
             ha='right')
    ax3d = fig.add_axes([0.005, 0.25, 0.58, 0.68])
    ax3d.imshow(image)
    ax3d.axis('off')
    ax = fig.add_axes([0.62, 0.3, 0.36, 0.62])
    stage.draw_top(ax)
    if stage.top_limits is not None:
        xmin, xmax, ymin, ymax = stage.top_limits
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('真上から', fontsize=11)
    if stage.legend:
        ax.legend(handles=stage.legend, fontsize=8, loc='upper left',
                  bbox_to_anchor=(0.0, -0.09), ncol=2, frameon=False)
    notes = '\n'.join('・' + n for n in stage.notes)
    fig.text(0.015, 0.235, notes, fontsize=10.5, va='top',
             linespacing=1.55)
    fig.savefig(path)
    plt.close(fig)


def wrap_notes(stage, width=112):
    """半角 ``width`` 文字分の幅で改行する (matplotlib は日本語を折り返さ
    ない。全角は半角 2 文字分)。"""
    import unicodedata
    wrapped = []
    for note in stage.notes:
        lines, line, used = [], '', 0
        for ch in note:
            w = 2 if unicodedata.east_asian_width(ch) in 'WFA' else 1
            if used + w > width:
                # 英数字の途中なら直前の空白・句読点で切る。
                cut = max(line.rfind(c) for c in ' 、。(')
                if ch.isascii() and ch.isalnum() and cut > len(line) - 20:
                    lines.append(line[:cut + 1])
                    line = line[cut + 1:]
                else:
                    lines.append(line)
                    line = ''
                used = sum(2 if unicodedata.east_asian_width(c) in 'WFA'
                           else 1 for c in line)
            line += ch
            used += w
        lines.append(line)
        wrapped.append('\n  '.join(lines))
    stage.notes = wrapped


def project(point, eye, target, size=IMAGE_SIZE, yfov=0.75):
    """``render_scene`` と同じカメラで 3 次元の点を画素座標にする。"""
    pose = look_at(eye, target)
    p = np.linalg.inv(pose) @ np.append(np.asarray(point, float), 1.0)
    if p[2] >= 0.0:
        return None
    t = math.tan(yfov / 2.0)
    aspect = size[0] / float(size[1])
    x = (p[0] / -p[2]) / (t * aspect)
    y = (p[1] / -p[2]) / t
    return (x + 1.0) / 2.0 * size[0], (1.0 - y) / 2.0 * size[1]


def save_all_stages_figure(overview, path, subtitle):
    """全段階のロボットを並べた図 (ロボットの上に段階の名前)。"""
    image = render_scene(overview.scene, overview.eye, overview.target)
    fig = plt.figure(figsize=(14, 10.5), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.94])
    ax.imshow(image)
    ax.axis('off')
    for text, position in overview.labels:
        uv = project(position, overview.eye, overview.target)
        if uv is not None:
            ax.text(uv[0], uv[1], text, ha='center', va='bottom',
                    fontsize=11, bbox=dict(fc='white', ec='0.6', alpha=0.85,
                                           boxstyle='round,pad=0.3'))
    fig.text(0.012, 0.975, overview.title, fontsize=16, va='top')
    fig.text(0.988, 0.975, subtitle, fontsize=9, color='0.4', va='top',
             ha='right')
    fig.savefig(path)
    plt.close(fig)


def save_overview(stages, images, path, subtitle):
    n = len(stages)
    cols = 3
    rows = int(math.ceil(n / float(cols)))
    fig = plt.figure(figsize=(16, 4.4 * rows + 0.6), dpi=100)
    fig.suptitle('握手から横並びまでの段階 ({})'.format(subtitle),
                 fontsize=15)
    for i, (stage, image) in enumerate(zip(stages, images)):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(stage.title.split(' (')[0], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


class StageViewer(object):
    """全体表示 (全段階のロボット) と段階ごとの詳細を viser で切り替える。

    全体表示でロボットをクリックするとその段階を開く。表示する項目
    (レイヤー) はチェックボックスで切り替える (見出しの色見本がその項目の
    色、マウスを乗せると説明)。
    """

    OVERVIEW = -1

    def __init__(self, stages, overview):
        from skrobot.viewers import ViserViewer
        self.stages = stages
        self.overview = overview
        self.viewer = ViserViewer(draw_grid=True)
        self.server = self.viewer._server
        self.lock = threading.RLock()
        self.layer_frames = {}  # (段階, レイヤー) -> frame handle
        self.built = set()
        self.label_handles = []
        self.checkboxes = []
        self.current = None

        gui = self.server.gui
        self.selector = gui.add_dropdown(
            '段階', options=['全体'] + [s.short_title for s in stages],
            initial_value='全体')
        self.back_button = gui.add_button('全体に戻る')
        self.prev_button = gui.add_button('← 前の段階')
        self.next_button = gui.add_button('次の段階 →')
        self.description = gui.add_markdown('')
        self.layer_folder = gui.add_folder('表示する項目')

        @self.selector.on_update
        def _(_):  # noqa: ANN001
            value = self.selector.value
            self.show(self.OVERVIEW if value == '全体' else
                      [s.short_title for s in stages].index(value))

        @self.back_button.on_click
        def _(_):  # noqa: ANN001
            self.select(self.OVERVIEW)

        @self.prev_button.on_click
        def _(_):  # noqa: ANN001
            if self.current is not None and self.current > 0:
                self.select(self.current - 1)

        @self.next_button.on_click
        def _(_):  # noqa: ANN001
            if self.current is not None and self.current < len(stages) - 1:
                self.select(self.current + 1)

    def stage(self, index):
        return self.overview if index == self.OVERVIEW else self.stages[index]

    def select(self, index):
        # ドロップダウンの更新で show が呼ばれる。
        value = '全体' if index == self.OVERVIEW \
            else self.stages[index].short_title
        if self.selector.value == value:
            self.show(index)
        else:
            self.selector.value = value

    def _build(self, index):
        """段階 ``index`` のメッシュを (初回だけ) 送る。レイヤーごとに
        フレームを作り、同じ色のメッシュは 1 つにまとめる。"""
        if index in self.built:
            return
        stage = self.stage(index)
        scene = stage.scene
        root = '/overview' if index == self.OVERVIEW else '/stage{}'.format(
            index)
        for key, info in scene.layers.items():
            frame = self.server.scene.add_frame(
                '{}/{}'.format(root, key), show_axes=False, visible=False)
            self.layer_frames[(index, key)] = frame
            by_color = collections.OrderedDict()
            for mesh, color, layer in scene.items:
                if layer == key:
                    by_color.setdefault(color, []).append(mesh)
            for k, (color, meshes) in enumerate(by_color.items()):
                mesh = trimesh.util.concatenate(meshes)
                handle = self.server.scene.add_mesh_simple(
                    '{}/{}/m{}'.format(root, key, k),
                    vertices=np.asarray(mesh.vertices, dtype=np.float32),
                    faces=np.asarray(mesh.faces, dtype=np.uint32),
                    color=tuple(int(c) for c in color[:3]),
                    opacity=None if color[3] >= 255 else color[3] / 255.0,
                    side='double' if color[3] < 255 else 'front')
                if info.get('click') is not None:
                    target = info['click']
                    handle.on_click(
                        lambda _event, target=target: self.select(target))
        self.built.add(index)

    def show(self, index):
        with self.lock:
            self._build(index)
            for (i, key), frame in self.layer_frames.items():
                frame.visible = (
                    i == index and self.stage(i).scene.layers[key]['default'])
            for handle in self.label_handles:
                handle.remove()
            self.label_handles = []
            if index == self.OVERVIEW:
                for k, (text, position) in enumerate(self.overview.labels):
                    self.label_handles.append(self.server.scene.add_label(
                        '/overview_labels/{}'.format(k), text,
                        position=tuple(position)))
            self._rebuild_checkboxes(index)
            stage = self.stage(index)
            self.description.content = '### {}\n\n{}'.format(
                stage.title, '\n'.join('- ' + n.replace('\n  ', '')
                                       for n in stage.notes))
            self.back_button.visible = index != self.OVERVIEW
            self.prev_button.visible = index != self.OVERVIEW
            self.next_button.visible = index != self.OVERVIEW
            self.current = None if index == self.OVERVIEW else index
            for client in self.server.get_clients().values():
                self.set_camera(client, stage)

    @staticmethod
    def set_camera(client, stage):
        if stage.eye is None:
            return
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.position = tuple(stage.eye)
        client.camera.look_at = tuple(stage.target)

    def _rebuild_checkboxes(self, index):
        for checkbox in self.checkboxes:
            checkbox.remove()
        self.checkboxes = []
        scene = self.stage(index).scene
        with self.layer_folder:
            for key, info in scene.layers.items():
                checkbox = self.server.gui.add_checkbox(
                    '{} {}'.format(swatch(info['color']), info['label']),
                    initial_value=info['default'],
                    hint=info['hint'] or None)

                @checkbox.on_update
                def _(_, key=key, checkbox=checkbox):  # noqa: ANN001
                    frame = self.layer_frames.get((index, key))
                    if frame is not None:
                        frame.visible = checkbox.value
                self.checkboxes.append(checkbox)

    def run(self):
        self.show(self.OVERVIEW)

        @self.server.on_client_connect
        def _(client):  # noqa: ANN001
            self.set_camera(client, self.stage(
                self.OVERVIEW if self.current is None else self.current))

        self.viewer.show()
        print('viser で表示中。全体表示のロボットをクリックすると段階の詳細を'
              '表示します。Ctrl-C で終了します。')
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass


def show_viser(stages, overview):
    StageViewer(stages, overview).run()


def main():
    parser = argparse.ArgumentParser(
        description='横並びまでの各段階の干渉・可動域・経路を図にする。')
    parser.add_argument('--work-dir', required=True,
                        help='run_pipeline_test.py --plan-motion '
                             '--side-by-side-transition の作業ディレクトリ。')
    parser.add_argument('--name', required=True,
                        help='人物のファイル名 (例: human_004)。')
    parser.add_argument('--output-dir', default='pipeline_stages',
                        help='PNG の保存先。')
    parser.add_argument(
        '--model-path', default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'))
    parser.add_argument(
        '--female-model-path', default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicModel_f_lbs_10_207_0_v1.0.0.pkl'))
    parser.add_argument('--collision-pairs', default=os.path.join(
        _SCRIPTS_DIR, 'collision_pairs.json'))
    parser.add_argument('--viser', action='store_true',
                        help='保存後に viser でも表示する (全体表示のロボットを'
                             'クリックで段階の詳細、項目はチェックボックスで'
                             '切り替え)。')
    parser.add_argument('--no-images', action='store_true',
                        help='PNG を保存しない (--viser だけ使うとき)。')
    args = parser.parse_args()

    setup_font()
    models = dict(load_smpl_models(args.model_path, args.female_model_path))
    person = Person(args.work_dir, args.name, models)

    visual = load_aero(use_hand=True)
    ik_robot = Aero(use_hand=False)
    spik.restrict_elbow_range(ik_robot)
    spik.restrict_leg_range(ik_robot)
    spik.restrict_waist_range(ik_robot)
    spik.restrict_neck_range(ik_robot)
    spik.lock_fixed_joints(ik_robot)
    spik.apply_collision_model(ik_robot)
    spik.apply_hand_box(ik_robot)
    verification_pairs = spik.build_verification_pairs_for_model(ik_robot)
    collision_pairs = spik.load_collision_pairs(args.collision_pairs,
                                                ik_robot)

    builder = StageBuilder(person, visual, ik_robot, verification_pairs,
                           collision_pairs)
    stages = builder.build()
    overview = builder.overview(stages)
    if not args.no_images:
        os.makedirs(args.output_dir, exist_ok=True)
        subtitle = '合成データ {} (人の {} 手 / ロボットの {} 腕)'.format(
            person.name, person.hand, person.robot_arm)
        images = []
        for stage in stages:
            wrap_notes(stage)
            image = render_scene(stage.scene, stage.eye, stage.target)
            images.append(image)
            path = os.path.join(args.output_dir, stage.key + '.png')
            save_stage_figure(stage, image, path, subtitle)
            print('保存: {}'.format(path))
        path = os.path.join(args.output_dir, '00_overview.png')
        save_overview(stages, images, path, person.name)
        print('保存: {}'.format(path))
        path = os.path.join(args.output_dir, '00_all_stages.png')
        save_all_stages_figure(overview, path, subtitle)
        print('保存: {}'.format(path))
    if args.viser:
        show_viser(stages, overview)


if __name__ == '__main__':
    main()
