#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""手のひら平面・人物骨格・ロボットを skrobot の viewer に描く部品.

``scripts/palm_plane_visualizer.py`` (平面フィットの検証だけをする) と
``scripts/human_palm_contact_behavior.py`` (実際に手を伸ばす) が同じ絵を
描くための共通部分。

rospy は import しない。描けなかったことは例外ではなく戻り値で返すので、
ログをどう出すかは呼び出し側が決める。
"""

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.model.primitives import Cylinder
from skrobot.model.primitives import LineString
from skrobot.model.primitives import Sphere

from aero_demo import palm_plane

# 色 (RViz のマーカーと合わせる), RGBA 0-255
COLOR_PLATE = [255, 255, 0, 90]
COLOR_NORMAL = [0, 255, 0, 255]
COLOR_FINGER = [0, 200, 255, 255]
COLOR_LANDMARK = [255, 255, 255, 255]
COLOR_APPROACH = [50, 100, 255, 255]
COLOR_PRESS = [255, 50, 50, 255]

# 骨格の線は部位ごとに色を変える。Person3D.bones の名前 ("Neck->RShoulder"
# 形式) から下の bone_color で引く。
COLOR_BONES = {
    'torso': [220, 220, 220, 255],
    'head': [255, 220, 150, 255],
    'rarm': [255, 110, 110, 255],
    'larm': [110, 170, 255, 255],
    'rleg': [255, 170, 70, 255],
    'lleg': [70, 210, 255, 255],
    'rhand': [255, 60, 60, 255],
    'lhand': [60, 120, 255, 255],
}

# どの関節が出てきたらその部位、という判定 (先に一致したものを採る)。
# ここに無い関節 (Neck, RShoulder, LShoulder, RHip, LHip) は torso 扱い。
_BONE_GROUPS = (
    ('head', ('Nose', 'REye', 'LEye', 'REar', 'LEar')),
    ('rarm', ('RElbow', 'RWrist')),
    ('larm', ('LElbow', 'LWrist')),
    ('rleg', ('RKnee', 'RAnkle')),
    ('lleg', ('LKnee', 'LAnkle')),
)

PLATE_SIZE = 0.12
NORMAL_LENGTH = 0.15
FINGER_LENGTH = 0.08


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return np.asarray(v, dtype=np.float64) / n


def rotation_from_z(z_axis):
    """+Z が z_axis を向く回転行列 (円柱は +Z 方向に伸びる)."""
    z = unit(z_axis)
    if z is None:
        return np.eye(3)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(z[2])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    x = unit(np.cross(ref, z))
    if x is None:
        return np.eye(3)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def set_color(link, rgba):
    """primitive の色を塗る (skrobot のバージョン差を吸収する)."""
    meshes = getattr(link, 'visual_mesh', None)
    if meshes is None:
        return link
    if not isinstance(meshes, (list, tuple)):
        meshes = [meshes]
    for mesh in meshes:
        # 面を持つ primitive は face_colors、点群は vertex_colors。点群に
        # face_colors を入れても例外にはならず、色が付かないまま無視される
        # (trimesh.PointCloud.colors が空のままになる) ので型で振り分ける。
        attr = 'face_colors' if getattr(mesh, 'faces', None) is not None \
            else 'vertex_colors'
        try:
            setattr(mesh.visual, attr, rgba)
        except Exception:  # 描画できないほどのことではない
            pass
    return link


def bone_color(name):
    """ボーン名 ("Neck->RShoulder" 形式) から部位の色を返す.

    手のランドマークのボーン ("RHand0->RHand1" など) は手の色にまとめる。
    """
    if 'RHand' in name:
        return COLOR_BONES['rhand']
    if 'LHand' in name:
        return COLOR_BONES['lhand']
    joints = name.split('->')
    for group, members in _BONE_GROUPS:
        if any(joint in members for joint in joints):
            return COLOR_BONES[group]
    return COLOR_BONES['torso']


class PalmPlaneScene(object):
    """手のひら平面まわりを描く viewer.

    描くもの
      * 半透明の黄色い板 … フィットした手のひら平面
      * 緑の矢印 … 手のひら法線 (ロボット側を向くはず)
      * 水色の矢印 … 指方向 (手首 -> 指の付け根)
      * 白い球 … フィットに使ったランドマーク
      * 青い球 … 接近目標 (中心 + offset * 法線)
      * 赤い球 … 押し込み目標 (中心 - depth * 法線)
      * 色分けした線 … 人物の骨格 (``Person3D.bones`` をそのまま繋ぐ)
      * ロボットモデル … ``add_robot`` を呼んだとき
      * 原点の座標軸 … 関節点の座標系 (``EstimationResult.frame_id``)

    Parameters
    ----------
    viewer : str
        ``trimesh`` (X のウィンドウ) か ``viser`` (ブラウザ表示)。X が
        pyglet のウィンドウを開けない環境 (WSLg など) では ``viser``。
    resolution : (int, int)
        ``trimesh`` のときのウィンドウの大きさ。
    draw_skeleton : bool
        骨格の線を描くか。
    """

    def __init__(self, viewer='trimesh', resolution=(960, 720),
                 draw_skeleton=True):
        self.viewer_name = str(viewer).lower()
        if self.viewer_name == 'viser':
            # X のウィンドウを開かずブラウザに描くので、WSL や ssh 越しでも
            # 動く。skrobot が listen した URL を起動時に表示する。
            from skrobot.viewers import ViserViewer
            self.viewer = ViserViewer()
        elif self.viewer_name == 'trimesh':
            from skrobot.viewers import TrimeshSceneViewer
            self.viewer = TrimeshSceneViewer(resolution=resolution)
        else:
            raise ValueError(
                "viewer must be 'trimesh' or 'viser', got '{}'".format(
                    self.viewer_name))

        self.draw_skeleton = draw_skeleton
        self.robot = None

        # 関節点の座標系 (既定 base_link) の原点
        self.viewer.add(Axis(axis_radius=0.005, axis_length=0.2))

        # 平面まわりの表示物。フィットできたフレームだけ viewer に入れる。
        self.plate = set_color(
            Box(extents=[PLATE_SIZE, 0.002, PLATE_SIZE]), COLOR_PLATE)
        self.normal_arrow = set_color(
            Cylinder(radius=0.004, height=NORMAL_LENGTH), COLOR_NORMAL)
        self.finger_arrow = set_color(
            Cylinder(radius=0.003, height=FINGER_LENGTH), COLOR_FINGER)
        self.approach = set_color(Sphere(radius=0.0125), COLOR_APPROACH)
        self.press = set_color(Sphere(radius=0.010), COLOR_PRESS)
        self.plane_links = [self.plate, self.normal_arrow, self.finger_arrow,
                            self.approach, self.press]

        # ランドマークは有無が毎フレーム変わるので 1 個ずつ出し入れする
        self.landmarks = {
            i: set_color(Sphere(radius=0.0075), COLOR_LANDMARK)
            for i in palm_plane.PLANE_LANDMARKS}

        self.shown = set()
        self.bone_links = []

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def open(self, open_browser=False, timeout=10.0):
        """viewer を表示する。描画できる状態になったら True.

        ``trimesh`` の viewer は X のウィンドウを開けないと描画スレッドが
        ロックを握ったまま止まり、以降の add / delete がすべてそこで待つ。
        そのままでは「推定結果が来ない」ようにしか見えないので、ここで
        検知して False を返す。
        """
        if self.viewer_name == 'viser':
            self.viewer.show(open_browser=open_browser)
            return True

        self.viewer.show()
        lock = getattr(self.viewer, 'lock', None)
        if lock is None:
            return True
        if lock.acquire(timeout=timeout):
            lock.release()
            return True
        return False

    def add_robot(self, robot):
        """ロボットモデルを置く.

        base_link がモデルのルートなので、原点に置くだけで関節点の座標系
        (既定 base_link) と揃う。
        """
        self.robot = robot
        self.viewer.add(robot)

    def redraw(self):
        self.viewer.redraw()

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------
    def _show(self, link, visible):
        """link を viewer に入れる / 外す (状態が変わったときだけ)."""
        key = id(link)
        if visible and key not in self.shown:
            self.viewer.add(link)
            self.shown.add(key)
        elif not visible and key in self.shown:
            self.viewer.delete(link)
            self.shown.discard(key)

    def update_plane(self, plane, points):
        """フィット結果に合わせて表示物を動かす.

        Parameters
        ----------
        plane : palm_plane.PalmPlane
        points : dict
            ランドマーク番号 -> 位置 (``collect_palm_points`` の戻り値)。
        """
        center = np.asarray(plane.center, dtype=np.float64)

        self.plate.newcoords(Coordinates(pos=center, rot=plane.rot))
        # 円柱は中心が原点で +Z 方向に伸びるので、始点から半分ずらす
        self.normal_arrow.newcoords(Coordinates(
            pos=center + plane.normal * (NORMAL_LENGTH / 2.0),
            rot=rotation_from_z(plane.normal)))
        self.finger_arrow.newcoords(Coordinates(
            pos=center + plane.finger_dir * (FINGER_LENGTH / 2.0),
            rot=rotation_from_z(plane.finger_dir)))
        self.approach.newcoords(
            Coordinates(pos=palm_plane.contact_target(plane)))
        self.press.newcoords(
            Coordinates(pos=palm_plane.embed_target(plane)))
        for link in self.plane_links:
            self._show(link, True)

        used = set(plane.used)
        for i, sphere in self.landmarks.items():
            if i in used:
                sphere.newcoords(Coordinates(pos=points[i]))
            self._show(sphere, i in used)

    def hide_plane(self):
        for link in self.plane_links:
            self._show(link, False)
        for sphere in self.landmarks.values():
            self._show(sphere, False)

    def update_skeleton(self, people):
        """ボーンを部位ごとに色を変えた線で描き直す.

        ``Person3D.bones`` の本数は関節の欠落で毎フレーム変わるので、線は
        作り直す。どの関節同士を繋ぐかは推定側 (people_pose_estimator の
        ``limb_sequence`` / ``hand_sequence``) が決めた ``bones`` に従う。

        線を作れなかったときは骨格の描画をあきらめて False を返す (以降の
        呼び出しは何もしない)。
        """
        if not self.draw_skeleton:
            return True
        for link in self.bone_links:
            self._show(link, False)
        self.bone_links = []
        for person in people:
            for bone in person.bones:
                try:
                    link = LineString(
                        np.asarray([bone.start_point, bone.end_point],
                                   dtype=np.float64),
                        color=bone_color(bone.name))
                except Exception:
                    self.draw_skeleton = False
                    return False
                self.bone_links.append(link)
                self._show(link, True)
        return True
