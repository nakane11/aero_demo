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
import trimesh

from skrobot.coordinates import Coordinates
from skrobot.model.primitives import Axis
from skrobot.model.primitives import Box
from skrobot.model.primitives import Capsule
from skrobot.model.primitives import Cylinder
from skrobot.model.primitives import LineString
from skrobot.model.primitives import MeshLink
from skrobot.model.primitives import Sphere

from aero_demo import palm_plane
from aero_demo import smpl_body

# 色 (RViz のマーカーと合わせる), RGBA 0-255
COLOR_PLATE = [255, 255, 0, 90]
COLOR_NORMAL = [0, 255, 0, 255]
COLOR_FINGER = [0, 200, 255, 255]
COLOR_LANDMARK = [255, 255, 255, 255]
COLOR_APPROACH = [50, 100, 255, 255]
COLOR_PRESS = [255, 50, 50, 255]
COLOR_FRUSTUM = [160, 180, 255, 90]
# SMPL で胴体・頭を実体メッシュとして描くときの単色 (肌色寄り)。骨格を
# 下から透けて見せたいので半透明にしてある。
COLOR_BODY = [225, 190, 160, 110]

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

# 骨格を線ではなくカプセル (体幹) で描くときの太さ [m]。部位ごとの見た目の
# 太さの目安で、実寸の体格を表すものではない。
BONE_RADIUS = {
    'torso': 0.05,
    'rarm': 0.032,
    'larm': 0.032,
    'rleg': 0.045,
    'lleg': 0.045,
    'rhand': 0.008,
    'lhand': 0.008,
}
# カプセル (円柱 + 半球キャップ) は高さがマイナスになると trimesh が例外を
# 出すので、両端の関節がほぼ重なっている骨でもこれ以上には潰さない。
MIN_BONE_HEIGHT = 0.005

# 胴体 (Neck-RShoulder/LShoulder/RHip/LHip の 4 本) は細いカプセルを
# Neck に集めて描くと扇のようになって不自然なので、1 枚の板 (Box) に
# まとめて描く。TORSO_DEPTH はその前後方向の厚み、TORSO_MIN_WIDTH は
# 肩・腰の関節がほぼ重なって見えるとき (真横向きなど) の下限の幅。
TORSO_DEPTH = 0.12
TORSO_MIN_WIDTH = 0.15
# 胴体の板を組み立てるのに要る関節名 (すべて揃わなければ描かない)。
_TORSO_JOINTS = ('Neck', 'RShoulder', 'LShoulder', 'RHip', 'LHip')

# 頭も細いカプセルの束ではなく 1 個の球で描く。前後・左右は Neck の真上
# (胴体の中心線上) に固定し、高さだけ Nose との Z 差 (Nose が隠れていれば
# HEAD_OFFSET) で決める -- Neck->Nose には前後方向の成分も大きいが、それを
# そのまま中心に使うと頭が前に出て胴体の中心からずれてしまうので使わない。
HEAD_RADIUS = 0.11
# Nose が隠れているときの保険の高さ (Neck から真上に何 m か)。
# fake_people_pose_estimator_ros.py の身長比 (h_nose - h_shoulder ≒ 身長の
# 12 %) から見積もった値。
HEAD_OFFSET = 0.20
# Nose が見えていても、しゃがむ・見上げるなどで Neck とほぼ同じ高さまで
# 沈んでしまった場合に頭が胴体に埋まって見えないよう、これより低くはしない。
HEAD_MIN_HEIGHT = 0.08
# 鼻は頭の球の目印として、実際に推定された Nose の位置に小さい球を
# 追加で描く (頭の球自体は前後方向を胴体中心に寄せた近似なので、鼻だけは
# 推定値そのままの位置を見せる -- 頭の球の前面からはみ出て見えてよい)。
NOSE_RADIUS = 0.028

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


def bone_group(name):
    """ボーン名 ("Neck->RShoulder" 形式) から部位 (``COLOR_BONES`` のキー) を返す.

    手のランドマークのボーン ("RHand0->RHand1" など) は手の部位にまとめる。
    """
    if 'RHand' in name:
        return 'rhand'
    if 'LHand' in name:
        return 'lhand'
    joints = name.split('->')
    for group, members in _BONE_GROUPS:
        if any(joint in members for joint in joints):
            return group
    return 'torso'


def bone_color(name):
    """ボーン名から部位の色を返す."""
    return COLOR_BONES[bone_group(name)]


def bone_radius(name):
    """ボーン名から部位のカプセル半径を返す."""
    return BONE_RADIUS[bone_group(name)]


def collect_joints(bones, joints=None):
    """``bone.name`` ("A->B" 形式) から端点の関節名と座標を拾い集める.

    胴体を 1 枚の板にまとめるのに、Neck/RShoulder/LShoulder/RHip/LHip の
    座標がどのボーンの端点として出てきたか知りたいだけなので、ボーンの
    向き (start が A か B か) 自体は問わない。既に分かっている関節名は
    上書きしない (``joints`` を渡せば複数回に分けて呼び出せる)。
    """
    if joints is None:
        joints = {}
    for bone in bones:
        parts = bone.name.split('->')
        if len(parts) != 2:
            continue
        a, b = parts
        joints.setdefault(a, np.asarray(bone.start_point, dtype=np.float64))
        joints.setdefault(b, np.asarray(bone.end_point, dtype=np.float64))
    return joints


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
    smpl_model_path : str or None
        SMPL v1.0.0 の ``.pkl`` (例:
        ``~/SMPL_python_v.1.0.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl``)。
        ライセンス上リポジトリには同梱できないのでローカルパスで渡す。
        ``None`` か、読み込みに失敗した場合 (ファイルが無い等) は
        ``smpl_load_error`` にエラーメッセージを入れたうえで、胴体・頭を
        従来通り Box/Sphere のプリミティブで描く (壊れない)。
    """

    def __init__(self, viewer='trimesh', resolution=(960, 720),
                 draw_skeleton=True, draw_camera=True,
                 smpl_model_path=None):
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

        # SMPL が使えれば胴体・頭を実体メッシュで描く。ファイルが無い/
        # 読めない環境でも壊れないよう、失敗したら黙って従来の
        # Box/Sphere 描画にフォールバックする (エラーメッセージだけ
        # smpl_load_error に残し、呼び出し側が rospy.logwarn 等できる
        # ようにする)。
        self._smpl_model = None
        self.smpl_load_error = None
        if smpl_model_path:
            try:
                self._smpl_model = smpl_body.load_smpl_model(smpl_model_path)
            except Exception as e:
                self.smpl_load_error = str(e)

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

    def _bone_capsule(self, bone, rgba):
        """1 本のボーンを部位の太さのカプセルにして返す (骨表現の体幹)."""
        start = np.asarray(bone.start_point, dtype=np.float64)
        end = np.asarray(bone.end_point, dtype=np.float64)
        radius = bone_radius(bone.name)
        length = float(np.linalg.norm(end - start))
        # カプセルは円柱部分の高さ + 両端の半球なので、関節位置 (start/end)
        # がちょうど半球の中心に来るよう円柱部分だけ短くする。骨が短すぎて
        # マイナスになる場合は MIN_BONE_HEIGHT で下げ止まる (見た目上は
        # 半球同士が重なるだけで、エラーにはならない)。
        height = max(length - 2.0 * radius, MIN_BONE_HEIGHT)
        capsule = Capsule(
            radius=radius, height=height,
            pos=(start + end) / 2.0, rot=rotation_from_z(end - start))
        return set_color(capsule, rgba)

    def _bone_line(self, bone, rgba):
        """1 本のボーンを (太さを持たない) 細い線にして返す.

        SMPL メッシュが胴体の実体を描いてくれるので、その上に重ねる
        骨格は太さの要らない細線で十分 -- ``_bone_capsule`` の体型
        表現の代わりに使う (骨格を線で描いていた頃の見た目に戻す)。
        """
        start = np.asarray(bone.start_point, dtype=np.float64)
        end = np.asarray(bone.end_point, dtype=np.float64)
        return LineString(np.stack([start, end]), color=rgba)

    def _torso_box(self, joints, rgba):
        """肩・腰の 4 関節 (+ Neck) から胴体を 1 枚の板として作る.

        Neck-肩中点-腰中点を通る軸を長さ方向 (Z) に、肩と腰の関節を結ぶ
        向きを幅方向 (X) にした直方体。奥行き (Y) は関節位置からは分から
        ないので ``TORSO_DEPTH`` で固定値にしてある。
        """
        neck = joints['Neck']
        hip_center = (joints['RHip'] + joints['LHip']) / 2.0

        z = unit(hip_center - neck)
        if z is None:
            raise ValueError('neck and hip center coincide')
        x = unit(joints['LShoulder'] - joints['RShoulder'])
        if x is None:
            x = unit(joints['LHip'] - joints['RHip'])
        if x is None:
            raise ValueError('shoulder and hip joints coincide')
        y = unit(np.cross(z, x))
        if y is None:
            raise ValueError('shoulder/hip line is parallel to the spine')
        # x を z, y に直交させ直す (肩の向きは脊柱と厳密には直交しない)。
        x = np.cross(y, z)

        width = max((float(np.linalg.norm(
            joints['LShoulder'] - joints['RShoulder']))
            + float(np.linalg.norm(joints['LHip'] - joints['RHip']))) / 2.0,
            TORSO_MIN_WIDTH)
        height = max(float(np.linalg.norm(hip_center - neck)),
                    MIN_BONE_HEIGHT)

        box = Box(extents=[width, TORSO_DEPTH, height],
                 pos=(neck + hip_center) / 2.0,
                 rot=np.column_stack([x, y, z]))
        return set_color(box, rgba)

    def _head_sphere(self, joints, rgba):
        """頭を Neck の真上 (胴体の中心線上) に 1 個の球として作る.

        前後・左右は Neck と同じにする (Neck->Nose には前後方向の成分も
        大きいので、そのまま使うと頭が前に出て胴体の中心からずれる)。
        高さだけ Nose の Z 座標と Neck の Z 座標の差で決めるので、首を
        縦に振れば (これは真上に固定した分は動かないが) その人の首・頭の
        実際の高さにはちゃんと合う。Nose が見えないときは ``HEAD_OFFSET``
        で妥協する。
        """
        neck = joints['Neck']
        if 'Nose' in joints:
            height = float(joints['Nose'][2] - neck[2])
        else:
            height = HEAD_OFFSET
        height = max(height, HEAD_MIN_HEIGHT)
        center = neck + np.array([0.0, 0.0, height])
        sphere = Sphere(radius=HEAD_RADIUS, pos=center)
        return set_color(sphere, rgba)

    def _nose_sphere(self, joints, rgba):
        """鼻先の目印を、推定された Nose の位置そのままに小さい球で描く."""
        sphere = Sphere(radius=NOSE_RADIUS, pos=joints['Nose'])
        return set_color(sphere, rgba)

    def update_skeleton(self, people):
        """ボーンを部位ごとに色を変えたカプセル (簡易体型メッシュ) で描き直す.

        ``Person3D.bones`` の本数は関節の欠落で毎フレーム変わるので、
        カプセルは毎回作り直す。どの関節同士を繋ぐかは推定側
        (people_pose_estimator の ``limb_sequence`` / ``hand_sequence``) が
        決めた ``bones`` に従う。``Person3D.hidden_bones`` (fake 推定だけが
        埋める、検出できなかった関節を含む骨) があれば色を薄くして併せて
        描く。

        胴体・頭・鼻だけは特別扱い。胴体は Neck-RShoulder/LShoulder/RHip/
        LHip の 4 本を Neck に集めてカプセルで描くと扇状になって不自然
        なので、5 関節が揃っていれば ``_torso_box`` で 1 枚の板として描く。
        頭も Neck-Nose 等の細いカプセルの束ではなく ``_head_sphere`` で
        1 個の球として描く -- 前後・左右は Neck の真上 (胴体の中心線上)
        に固定し、高さだけ Nose との Z 座標の差で決める。鼻はその頭の
        球の目印として、``_nose_sphere`` で推定された Nose の位置その
        ままに小さい球を追加で描く (頭の球の前面からはみ出て見える)。
        いずれも必要な関節が揃わなければ単に描かない (他の部位のように
        欠けた分だけ線を減らす、では絵にならない)。

        SMPL モデルが読み込めていれば (``__init__`` の
        ``smpl_model_path``)、胴体・頭・四肢の実体として半透明の SMPL
        メッシュ (``smpl_body.retarget_and_pose``, ``COLOR_BODY`` の
        alpha を下げてある) を追加で描く。ただし部位ごとに色分けした
        カプセル骨格 (``_bone_capsule``/``_torso_box``/``_head_sphere``)
        は SMPL の有無にかかわらず常に描き、透けた SMPL メッシュ越しに
        見えるようにする -- 鼻の目印 (``_nose_sphere``) だけは SMPL の
        頭メッシュと二重に見えるため SMPL 描画時は省略する。必要な関節
        (Neck + 両肩) が無い、またはリターゲットに失敗したフレーム/
        人物は、その人物だけ SMPL メッシュを諦めてカプセル骨格のみに
        なる (SMPL 未読み込みのときは常にこちら)。

        カプセル/胴体の板/頭・鼻の球/SMPL メッシュを作れなかったときは
        骨格の描画をあきらめて False を返す (以降の呼び出しは何もしない)。
        薄い方だけ描けなかった場合は、通常の骨格自体は描けているので
        描画をあきらめない。
        """
        if not self.draw_skeleton:
            return True
        for link in self.bone_links:
            self._show(link, False)
        self.bone_links = []
        use_smpl = self._smpl_model is not None
        # SMPL メッシュがあるときは、太いカプセル (簡易体型) は SMPL の
        # 実体と喧嘩して見苦しいので、代わりに元の細い線 (_bone_line) で
        # 骨格を描く -- 胴体・頭も (box/sphere の代役に頼らず) 他の部位
        # と同じくボーンをそのまま線にする。
        bone_shape = self._bone_line if use_smpl else self._bone_capsule
        for person in people:
            visible_joints = {}
            all_joints = {}
            for bone in person.bones:
                collect_joints([bone], visible_joints)
                collect_joints([bone], all_joints)
                group = bone_group(bone.name)
                if not use_smpl and group in ('torso', 'head'):
                    continue
                try:
                    link = bone_shape(bone, bone_color(bone.name))
                except Exception:
                    self.draw_skeleton = False
                    return False
                self.bone_links.append(link)
                self._show(link, True)
            for bone in getattr(person, 'hidden_bones', []):
                collect_joints([bone], all_joints)
                group = bone_group(bone.name)
                if not use_smpl and group in ('torso', 'head'):
                    continue
                try:
                    link = bone_shape(
                        bone, dim_color(bone_color(bone.name)))
                except Exception:
                    continue
                self.bone_links.append(link)
                self._show(link, True)

            if use_smpl:
                joints = dict(all_joints)
                joints.update(visible_joints)
                try:
                    result = smpl_body.retarget_and_pose(
                        self._smpl_model, joints)
                except Exception:
                    result = None
                if result is not None:
                    verts, faces = result
                    try:
                        mesh = trimesh.Trimesh(
                            vertices=verts, faces=faces, process=False)
                        link = set_color(MeshLink(visual_mesh=mesh),
                                         COLOR_BODY)
                    except Exception:
                        link = None
                    if link is not None:
                        self.bone_links.append(link)
                        self._show(link, True)

            # use_smpl のときは torso/head も上の bone_shape ループで既に
            # 線として描いてあるので、box/sphere の代役は不要 (SMPL の
            # リターゲットに失敗した人物だけ、素の骨格線がフォール
            # バックとして残る)。
            builders = [] if use_smpl else [
                (self._torso_box,
                 lambda j: all(k in j for k in _TORSO_JOINTS), 'torso'),
                (self._head_sphere,
                 lambda j: 'Neck' in j, 'head'),
                (self._nose_sphere,
                 lambda j: 'Nose' in j, 'head'),
            ]
            for builder, ready, group in builders:
                if ready(visible_joints):
                    joints, rgba = visible_joints, COLOR_BONES[group]
                elif ready(all_joints):
                    joints, rgba = all_joints, dim_color(COLOR_BONES[group])
                else:
                    continue
                try:
                    link = builder(joints, rgba)
                except Exception:
                    if joints is visible_joints:
                        self.draw_skeleton = False
                        return False
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
