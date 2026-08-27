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
COLOR_FRUSTUM = [160, 180, 255, 90]

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
IK_TARGET_AXIS_LENGTH = 0.08
HAND_AXIS_LENGTH = 0.06
# 台車 (base_link) の初期位置を示す原点座標軸。planar な IK で台車が動く
# ことがあるので、動いたかどうか一目でわかるよう他の座標軸よりだいぶ
# 大きくしてある。
BASE_ORIGIN_AXIS_RADIUS = 0.015
BASE_ORIGIN_AXIS_LENGTH = 0.4
# カメラの画角の四角すいをどこまで伸ばして描くか [m]。人が立つ距離
# (fake_people_pose_estimator_ros.py の ~distance_range, 既定 0.8-1.0 m)
# を包む程度の見た目にしてある -- 実際の可視距離とは無関係な表示用の値。
FRUSTUM_DEPTH = 1.2


def dim_color(rgba, alpha=70, gray=200, gray_mix=0.5):
    """検出できなかった関節・骨用に、元の色を保ちつつ薄くする.

    完全な灰色にはせず ``gray_mix`` だけ灰色に寄せるので、どの部位か
    (例えば右腕は赤系、左腕は青系) は薄いなりに見分けがつく。
    """
    mixed = [int(c * (1.0 - gray_mix) + gray * gray_mix) for c in rgba[:3]]
    return mixed + [alpha]


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
      * 色分けした線 … 人物の骨格 (``Person3D.bones`` をそのまま繋ぐ)。
        検出できなかった関節 (``Person3D.hidden_bones``) も、色を薄くして
        併せて描く -- fake_people_pose_estimator_ros.py だけが埋める
        真値で、本物の推定では常に空 (何も薄く描かれない)。
      * 薄い四角すい … カメラの画角 (``EstimationResult.camera_*``).
        optical_frame から ``FRUSTUM_DEPTH`` だけ伸ばして描く。
      * ロボットモデル … ``add_robot`` を呼んだとき
      * 原点の座標軸 … 関節点の座標系 (``EstimationResult.frame_id``)
      * 太めの座標軸 … IK を解くときの目標座標系 (``update_ik_target``)
      * 細めの座標軸 … ロボットの手先 (eef_grasp_link 相当) の座標系。
        ``track_hand`` で追わせた属性を毎 ``redraw`` ごとに読みに行く

    Parameters
    ----------
    viewer : str
        ``trimesh`` (X のウィンドウ) か ``viser`` (ブラウザ表示)。X が
        pyglet のウィンドウを開けない環境 (WSLg など) では ``viser``。
    resolution : (int, int)
        ``trimesh`` のときのウィンドウの大きさ。
    draw_skeleton : bool
        骨格の線を描くか。
    draw_camera : bool
        カメラの画角の四角すいを描くか。
    """

    def __init__(self, viewer='trimesh', resolution=(960, 720),
                 draw_skeleton=True, draw_camera=True):
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
        self.draw_camera = draw_camera
        self.robot = None
        self._hand_coords_attr = None

        # 関節点の座標系 (既定 base_link) の原点 = 台車の初期位置。planar な
        # IK で台車が動いても原点は動かないので、他の座標軸よりだいぶ大きく
        # 描いて、動いたかどうかここと見比べればわかるようにしてある。
        self.viewer.add(Axis(axis_radius=BASE_ORIGIN_AXIS_RADIUS,
                             axis_length=BASE_ORIGIN_AXIS_LENGTH))

        # IK の目標座標系と、ロボットの手先座標系。どちらもフィットした
        # 手のひら平面 (self.plate 等) とは別に、実際に IK へ渡した/解けた
        # 姿勢を見せるためのもの -- 平面の向きと手先の向きがずれていれば
        # ここで気付ける。
        self.ik_target_axis = Axis(
            axis_radius=0.005, axis_length=IK_TARGET_AXIS_LENGTH)
        self.hand_axis = Axis(
            axis_radius=0.004, axis_length=HAND_AXIS_LENGTH)

        # 平面まわりの表示物。フィットできたフレームだけ viewer に入れる。
        # plane.rot の局所 +Y が -normal (palm_plane.fit_palm_plane 参照)
        # なので、薄くする軸は y。
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
        self.camera_links = []
        self._info_text = None

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

    def track_hand(self, coords_attr):
        """毎 ``redraw`` ごとに ``robot`` のこの属性から手先座標系を追う.

        ``coords_attr`` は ``self.robot`` の属性名 (``'rarm_end_coords'``
        など)。``None`` にすると手先の座標軸を隠す。
        """
        self._hand_coords_attr = coords_attr
        self._show(self.hand_axis, coords_attr is not None)

    def update_ik_target(self, coords):
        """IK に渡した目標座標系を描き直す (``None`` なら隠す)."""
        if coords is None:
            self._show(self.ik_target_axis, False)
            return
        self.ik_target_axis.newcoords(coords.copy_worldcoords())
        self._show(self.ik_target_axis, True)

    def redraw(self):
        if self.robot is not None and self._hand_coords_attr is not None:
            coords = getattr(self.robot, self._hand_coords_attr, None)
            if coords is not None:
                self.hand_axis.newcoords(coords.copy_worldcoords())
        self.viewer.redraw()

    def show_info_text(self, lines):
        """3D シーンの外側に、複数行のテキストパネルを出す/更新する.

        ``viser`` のブラウザ UI にだけ GUI パネルがあるので、そこでのみ
        表示する (``trimesh`` の pyglet ウィンドウには文字を描く手段が無い
        ので、その場合は何もしない -- 呼び出し側で ``rospy.loginfo`` も
        するとどちらの viewer でも確認できる)。

        Parameters
        ----------
        lines : list of str
        """
        if self.viewer_name != 'viser':
            return
        text = '\n\n'.join(lines)
        if self._info_text is None:
            self._info_text = self.viewer._server.gui.add_markdown(text)
        else:
            self._info_text.content = text

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
        ``Person3D.hidden_bones`` (fake 推定だけが埋める、検出できなかった
        関節を含む骨) があれば色を薄くして併せて描く。

        線を作れなかったときは骨格の描画をあきらめて False を返す (以降の
        呼び出しは何もしない)。薄い骨のほうだけ描けなかった場合は、通常の
        骨格自体は描けているので描画をあきらめない。
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
            for bone in getattr(person, 'hidden_bones', []):
                try:
                    link = LineString(
                        np.asarray([bone.start_point, bone.end_point],
                                   dtype=np.float64),
                        color=dim_color(bone_color(bone.name)))
                except Exception:
                    continue
                self.bone_links.append(link)
                self._show(link, True)
        return True

    def update_camera(self, intrinsics, width, height, pose):
        """カメラの画角を表す四角すいを薄く描き直す (optical_frame から伸びる).

        Parameters
        ----------
        intrinsics : aero_demo.people_pose_types.CameraIntrinsics or None
        width, height : int
            画像サイズ [px]。
        pose : array_like or None
            camera_frame_id -> 描画座標系 (``EstimationResult.frame_id``,
            既定 base_link) の 4x4 同次変換行列
            (``EstimationResult.camera_pose``)。どれか欠けていたら隠す。
        """
        for link in self.camera_links:
            self._show(link, False)
        self.camera_links = []

        if not self.draw_camera or intrinsics is None or pose is None \
                or width <= 0 or height <= 0:
            return

        matrix = np.asarray(pose, dtype=np.float64).reshape(4, 4)
        apex = matrix[:3, 3]
        rot = matrix[:3, :3]

        # 画像四隅の光線を、光学系フレーム (x=右, y=下, z=前) で作って
        # world/base 姿勢へ回してから FRUSTUM_DEPTH だけ伸ばす。
        corners = []
        for u, v in ((0, 0), (width, 0), (width, height), (0, height)):
            direction = np.array([(u - intrinsics.cx) / intrinsics.fx,
                                  (v - intrinsics.cy) / intrinsics.fy,
                                  1.0])
            corners.append(apex + rot.dot(direction) * FRUSTUM_DEPTH)

        # 頂点 -> 各隅 -> 頂点... と一筆書きにして 4 本の稜線を 1 本の
        # LineString にまとめる (往復になるが見た目は変わらない)。
        rays = LineString(np.asarray(
            [apex, corners[0], apex, corners[1],
             apex, corners[2], apex, corners[3], apex],
            dtype=np.float64), color=COLOR_FRUSTUM)
        far_rect = LineString(np.asarray(
            corners + [corners[0]], dtype=np.float64), color=COLOR_FRUSTUM)

        self.camera_links = [rays, far_rect]
        for link in self.camera_links:
            self._show(link, True)
