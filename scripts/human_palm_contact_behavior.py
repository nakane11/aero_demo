#!/usr/bin/env python3
"""Reach out and touch the palm a human offers, drawing the whole thing.

The estimation half is exactly what ``palm_plane_visualizer.py`` shows: the
same pose source, the same ``palm_plane.fit_palm_plane``, and the same scene
(``aero_demo.palm_plane_view.PalmPlaneScene``).  On top of that this node
locks a target and actually reaches for it, so the viewer shows the robot's
hand closing in on the palm it fitted.

Sequence
--------
``WAITING``
    Take poses from the estimator, draw the skeleton and the fitted palm
    plane, and turn the neck towards the person.  Once a plane is fitted with
    the neck settled, the approach/press targets are locked.
``NODDING`` -> ``REACHING``
    Nod, raise the lifter, look at the palm, then solve whole-body IK for the
    approach pose and again for the press pose.  Both motions are drawn.
``DONE``
    Nothing more is commanded and the drawing stops updating.

Pose estimation is stopped the moment the target is locked, so the skeleton
stays on screen exactly as it was when the robot committed to it.  Likewise
the drawing stops updating once the press motion finishes -- the viewer stays
open so the final pose can be inspected.

Pose input
----------
トピックは購読しない。``~source`` で選んだ推定クラスのインスタンスを自分で
持ち、``wait_for_result()`` で 1 フレームずつ受け取る。

``~source: real`` (既定)
    ``people_pose_estimator_ros.RosPeoplePoseEstimator``
    … カメラの画像トピックを購読する本物の推定。
``~source: fake``
    ``fake_people_pose_estimator_ros.FakeRosPeoplePoseEstimator``
    … カメラ無しで偽の姿勢を生成する。``~use_robot_interface`` が既定で
      false になるので、実機もカメラも無い環境で動作を絵だけで確かめられる。

関節点は推定側で ``~output_frame`` (既定 base_link) 相対に変換済みなので、
このノードは TF を引かない。座標系が違う結果は捨てる (TF が引けなかった
ときにカメラ座標系のまま返ってくるので、そのまま IK を解くと危険)。

Parameters
----------
``~source`` (str, default ``real``)      ``real`` か ``fake``
``~use_robot_interface`` (bool)          実機に指令を出すか。既定は
                                         ``~source: real`` のときだけ true
``~use_hand`` (bool, default true)       手付きの URDF を読むか
``~hand_side`` (str, default ``R``)      人間が差し出す手、``R`` か ``L``。
                                         (``~hand`` は推定クラスの
                                         ``~hand/enable`` と衝突するので使わない)
``~same_hand`` (bool, default true)      ロボットが人間と同じ側の手で触れる
                                         か。true (既定, これまでの挙動) は
                                         同じ側の手で触れ、人間と向かい合っ
                                         て握手のように鏡写しした向きで触れ
                                         る。false は反対側の手で触れ、人間
                                         と同じ方向を向いた状態で (鏡写しせ
                                         ず) 掌平面をそのまま使う。次のラウ
                                         ンドの ``WAITING`` に入る直前に読み
                                         直すので、``human_palm_contact_
                                         behavior_loop.py`` を使えばラウンド
                                         の合間に rosparam で切り替えられる。
``~min_score`` (float, default 0.1)
``~viewer`` (str, default ``trimesh``)   ``trimesh`` / ``viser`` / ``none``
``~open_browser`` (bool, default false)  ``~viewer: viser`` のときブラウザを
                                         自動で開くか
``~draw_skeleton`` (bool, default true)  骨格を線で描くか。検出できな
                                         かった関節も (``~source: fake``
                                         のときだけ) 色を薄くして併せて描く
``~draw_camera_frustum`` (bool, default true)  カメラの画角を表す薄い
                                         四角すいを optical_frame から
                                         伸ばして描くか。``~source: fake``
                                         かつ ``~filter_by_fov`` が false
                                         (既定) のときは、画角の外でも
                                         関節が落ちず画角を考慮していない
                                         ので、この設定によらず描かない
``~viewer_width`` / ``~viewer_height`` (int, default 960 / 720)
``~smpl_model_path`` (str, default ``~/SMPL_python_v.1.0.0/smpl/models/``
                     ``basicmodel_m_lbs_10_207_0_v1.0.0.pkl``)
                                         SMPL v1.0.0 の .pkl (ライセンス上
                                         リポジトリには同梱しないので
                                         ローカルパスで渡す)。読めなければ
                                         警告を出して従来のカプセル/箱の
                                         骨格描画にフォールバックする。
``~base_frame`` (str, default base_link)
``~output_frame`` (str, default ``~base_frame``)
``~plane_fit_timeout`` (float, default 4.0)  人物は検出できているのに手のひ
                                         らの平面が何秒フィットできなければ
                                         諦めて ``DONE`` に進む (ループ/デー
                                         タセット生成側はこれを失敗ラウンド
                                         として次へ進む) か。
"""

import math
import os
import sys
import threading
from collections import namedtuple

import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray

from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import matrix2ypr
from skrobot.coordinates.math import rotate_vector
from skrobot.coordinates.math import rotation_matrix_z_to_axis
from skrobot.coordinates.math import rpy_matrix
from skrobot.model.primitives import Cylinder
from skrobot.planner import sqp_plan_trajectory
from skrobot.planner import SweptSphereSdfCollisionChecker
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config
from skrobot.sdf import UnionSDF

from aero_demo import palm_plane
from aero_demo.palm_plane import EMBED_DEPTH
from aero_demo.palm_plane_view import PalmPlaneScene

# catkin_install_python が devel space に置くのは実体ではなく exec() で
# 中継する relay script なので、それを import してもクラスがモジュールの
# 名前空間に入らない (ImportError になる)。同じ scripts/ にある推定クラス
# は実ファイル側から import する。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Reference point for "where the human is", used only for gaze.  Nose is
# preferred but was published in just 26.5% of a 34-frame sample, while
# Neck reached 100%, so fall back to Neck rather than dropping the frame.
HUMAN_REF_LIMBS = ('Nose', 'Neck')

# 立ち位置 (足元) の基準にする limb の候補。Ankle -> Hip -> HUMAN_REF_LIMBS
# の順に、見つかった最初のものを使う (_human_foot_xy 参照)。Ankle が一番
# 立ち位置に近いが脚は隠れて見えないことが多いので、Hip、それも無ければ
# 上半身の基準点 (Nose/Neck、実際の足元よりは体の中心寄りだが無いよりまし)
# まで段階的にフォールバックする。
HUMAN_FOOT_REF_LIMBS = (('RAnkle', 'LAnkle'), ('RHip', 'LHip'))

# 首を動かしてから次の判断までの待ち [s]。関節が動いている途中の角度で
# 目標を決めないための間。
NECK_SETTLE_TIME = 2.5

# 人体を障害物として近似するときの円柱半径 [m] (骨の名前に含まれるキーワード
# でおおまかな太さを変える。マッチしなければ胴/脚として HUMAN_BONE_RADIUS_
# DEFAULT を使う)。
HUMAN_BONE_RADIUS = (
    (('Shoulder', 'Elbow', 'Wrist'), 0.05),
    (('Eye', 'Ear', 'Nose'), 0.10),
)
HUMAN_BONE_RADIUS_DEFAULT = 0.09

# 人体回避の軌道最適化 (sqp_plan_trajectory) のウェイポイント数と安全マージン。
HUMAN_AVOIDANCE_WAYPOINTS = 8
HUMAN_AVOIDANCE_SAFETY_MARGIN = 0.03  # [m]

# IK の use_base='planar' が台車 (wheel_base_link) を動かした結果、人間の
# 足元 (_human_foot_sdf 参照) へ近づきすぎていないかの安全マージン。
CART_AVOIDANCE_SAFETY_MARGIN = 0.05  # [m]

# 人間の足元を近似する円柱の半径 [m]。person.bones の脚 (Hip->Knee->Ankle)
# は検出できないことが多く (肩から先の腕・頭に比べてカメラに映りづらい/
# 隠れやすい)、そこに頼らず立ち位置まわりに一律に置く。
HUMAN_FOOT_RADIUS = 0.3

# ~same_hand=False (人間と反対の手で繋ぎ、同じ方向を向く) のとき、whole-
# body IK (use_base='planar') の種として base_link をどれだけ人間の真横へ
# ずらすか [m] (_human_facing_xy 参照)。0 のまま (人間の正面) だと、その
# 場から見て目標がほぼ真後ろになり、IK が届く姿勢を見つけられず暴れる
# (base_link が発散し、ロボットが視界の外へ消えたように見える) ので、少な
# くとも掌までの奥行きに近い量はずらしておく必要がある -- 経験的な初期値
# で、実測して外れているようなら調整すること。
SAME_DIRECTION_STANDOFF = 0.45  # [m]


def _bone_radius(bone_name):
    for keywords, radius in HUMAN_BONE_RADIUS:
        if any(k in bone_name for k in keywords):
            return radius
    return HUMAN_BONE_RADIUS_DEFAULT


def _human_obstacle_sdf(person, touching_arm_prefix):
    """``person.bones`` (骨格, base_link 座標系) から、ロボットが避けるべき
    人体を近似した SDF (円柱の和) を作る。フレーム内に骨が見つからなければ
    ``None``。

    触れに行く側の前腕 (``"{R,L}Elbow->{R,L}Wrist"``) だけは除外する --
    ロボットの手はまさにそこへ向かうので、障害物にすると届かなくなる。
    掌自体 (``RHand*``/``LHand*``) はそもそも ``person.bones`` に含まれない
    (people_pose_estimator.PeoplePoseEstimator._create_bones は
    index2limbname の骨格だけを繋ぐ) ので、ここで別途除く必要はない。
    """
    exclude = '{0}Elbow->{0}Wrist'.format(touching_arm_prefix)
    sdf_list = []
    for bone in person.bones:
        if bone.name == exclude:
            continue
        p0 = np.asarray(bone.start_point, dtype=np.float64)
        p1 = np.asarray(bone.end_point, dtype=np.float64)
        axis = p1 - p0
        length = float(np.linalg.norm(axis))
        if length < 1e-3:
            continue
        cyl = Cylinder(radius=_bone_radius(bone.name), height=length,
                       with_sdf=True)
        cyl.newcoords(Coordinates(
            pos=(p0 + p1) / 2.0, rot=rotation_matrix_z_to_axis(axis)))
        sdf_list.append(cyl.sdf)
    if not sdf_list:
        return None
    return UnionSDF(sdf_list)

# r/l_eef_grasp_link のローカル Y 軸が甲->掌方向 (dorsum -> palm) -- URDF
# を skrobot で読み、指の曲げ関節を動かして先端がどちらへ寄るかを直接
# 確かめて特定した (palm_plane.fit_palm_plane 参照)。
#
# fit_palm_plane が作る plane.rot はそのままだと robot 側の +Y (甲->掌) が
# 人間の掌の法線と正対する ("人間の掌に向き合う") 向きになっているが、
# +X (指方向) と +Z (親指<->小指幅方向) は人間自身のものをそのまま流用
# しているので、前腕軸まわりの傾き (fake_people_pose_estimator_ros.py の
# ~present_wrist_roll_deg_range と同じ意味の量) はロボット側も人間と
# "同じ" 向きになってしまう。握手はそこが逆で、向かい合う 2 人は鏡写しの
# 関係にあるので、人間が鉛直の基準からある向きへ傾けたら、ロボットは同じ
# 基準から逆向きに傾けるのが対称になる (_mirror_target_rotation 参照)。
# その傾きさえ決めてしまえば向きは一意に決まるので、IK は
# ``rotation_axis=True`` (3 軸とも厳密に合わせる) で解く。
def _unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return np.asarray(v, dtype=np.float64) / n


# _mirror_target_rotation の握手向きは元々「指方向を 180 度反転」で一意に
# 決めていたが、平面フィットの誤差次第では 180 度固定だと IK が解けないこと
# があるので、代わりに ±90 度の 2 通りを試し、IK が解けた方を採用する
# (control_loop 参照)。
MIRROR_TURN_CANDIDATES_DEG = (90.0, -90.0)


def _mirror_target_rotation(rot, turn_deg=180.0):
    """Handshake-mirror a palm-target rotation (e.g. ``PalmPlane.rot``).

    ``rot``'s +X is the finger direction and +Z is the thumb<->pinky width
    axis, both copied straight from the human's fitted palm (see
    ``palm_plane.fit_palm_plane``); +Y already faces the robot's palm
    toward the human's.  Two things need fixing before this is a plausible
    handshake instead of a plain copy of the human's own frame:

    * Roll around +Y (the approach direction): copying +Z as-is would give
      the robot the *same* roll as the human -- i.e. the same tilt away
      from a plain vertical, thumb-up handshake -- instead of a mirrored
      one.  Two people shaking hands face each other as mirror images: if
      the human tilts N degrees off vertical one way, the robot should
      tilt N degrees off vertical the *other* way (both measured from the
      same "vertical, thumb-up" reference *for the shared +Y*, i.e. a
      reflection about the vertical plane containing +Y), so e.g. a human
      tilted 45 deg one way and a robot mirrored 45 deg the other way end
      up 90 deg apart rather than lined up.  +Y itself is left completely
      untouched by this reflection (unlike mirroring around +X, which
      would recompute +Y from the reflected +Z and let it drift away from
      the human-derived direction that already correctly faces the robot
      -- and degenerates whenever the *finger* direction is close to
      vertical, e.g. a presented arm hanging low, even though +Y itself
      is nowhere near vertical then).
    * Finger direction: the human's +X points from their wrist toward
      their fingertips, i.e. roughly away from their own body.  Copying
      it as the robot's +X would point the robot's fingertips the same
      way -- past the human's hand, away from them -- whereas in an
      actual handshake each hand's fingers point back across the other's
      palm toward *their* wrist.  So the robot's +X is turned ``turn_deg``
      about the shared +Y (which leaves +Y -- "face the human" --
      untouched); at the default 180 deg this is a plain negation, but
      callers trying several candidate orientations (see
      ``MIRROR_TURN_CANDIDATES_DEG``) pass other angles too.
    """
    y_axis = rot[:, 1]
    x_axis_in = _unit(rot[:, 0])
    if x_axis_in is None:
        return rot

    up = np.array([0.0, 0.0, 1.0])
    # Invariant under y_axis -> -y_axis (it only depends on y_axis via its
    # projection matrix), so it doubles as the reference for the
    # finger-reversed axis used below.
    up_perp = _unit(up - float(np.dot(up, y_axis)) * y_axis)
    if up_perp is None:
        # +Y (the approach direction) ~vertical: no well-defined
        # "vertical, thumb-up" reference to mirror around, so keep the
        # human's own orientation rather than doing something arbitrary.
        return rot

    theta = math.atan2(
        float(np.dot(y_axis, np.cross(up_perp, x_axis_in))),
        float(np.dot(up_perp, x_axis_in)))

    x_axis = _unit(rotate_vector(up_perp, -theta, y_axis))
    if x_axis is None:
        return rot
    z_axis = np.cross(x_axis, y_axis)
    # Turn (x_axis, z_axis) by turn_deg about +Y; +Y itself is untouched.
    phi = math.radians(turn_deg)
    turned_x = math.cos(phi) * x_axis + math.sin(phi) * z_axis
    turned_z = -math.sin(phi) * x_axis + math.cos(phi) * z_axis
    return np.column_stack([turned_x, y_axis, turned_z])


# 人間の掌平面 (palm_plane.fit_palm_plane の結果) を IK ターゲットへ変換した
# もの。pos/rot は人間自身の掌の位置・姿勢 (contact_target/plane.rot その
# まま、_lock_target が描画やデータセット生成のために残しておく分)、
# center/normal は gaze/lifter が使う掌中心とその法線。candidates は
# MIRROR_TURN_CANDIDATES_DEG のそれぞれについての PalmIkCandidate --
# ロボットの手が向くべき向きが turn_deg ごとに変わるので、IK が解けた方を
# control_loop が選ぶ。
PalmIkTargets = namedtuple('PalmIkTargets', [
    'pos', 'rot', 'center', 'normal', 'candidates'])

# turn_deg: _mirror_target_rotation に渡した角度 (どの向きの候補か)。
# hand_rot: ロボットの手が向くべき向き (_mirror_target_rotation 参照)。
# approach_coords/press_coords: hand_rot を向きに使う IK ターゲット。
PalmIkCandidate = namedtuple('PalmIkCandidate', [
    'turn_deg', 'hand_rot', 'approach_coords', 'press_coords'])


def _correct_grasp_frame(hand_rot, arm):
    """``l_eef_grasp_link`` は ``r_eef_grasp_link`` に対して +X (指方向)
    まわりに 180 度ずれている (甲->掌方向 +Y / 親指<->小指幅 +Z がどちらも
    反転している) -- 右手・右腕 (十分実績のある基準) のターゲットと左手・
    左腕のターゲットが、体の左右対称性から数学的に一致するはずの関係に
    なっているかを実際に計算して確認した。
    ``shoulder_r``/``shoulder_y``/``wrist_r`` が左右の腕で符号反転するのと
    同じ、URDF がミラーで作られていることに起因するずれ。

    +X はそのまま、+Y と +Z だけを反転する。1 軸だけ反転すると回転行列で
    なく鏡映行列 (行列式 -1) になり IK の回転目標として不正になるので、
    必ず 2 軸同時に反転すること (行列式は +1 のまま保たれる)。
    """
    if arm != 'l':
        return hand_rot
    return np.column_stack(
        [hand_rot[:, 0], -hand_rot[:, 1], -hand_rot[:, 2]])


def palm_plane_to_ik_targets(plane, turn_degs=MIRROR_TURN_CANDIDATES_DEG,
                              mirror=True, arm=None):
    """人間の掌平面から、ロボットが IK で解く approach/press ターゲット
    (skrobot ``Coordinates``, base_link frame) の候補群を作る.

    ``arm`` (``'r'``/``'l'``, 実際に触れに行くロボットの腕) が ``'l'`` なら
    ``_correct_grasp_frame`` で左腕の URDF のずれを補正する。``None`` のま
    まなら (呼び出し側で腕がまだ決まっていない等) 補正しない。

    ``mirror`` が True (既定, ``~same_hand`` が true のとき -- 人間と同じ
    側の手で触れ、向かい合う) なら、``turn_degs`` それぞれについて 1 つの
    ``PalmIkCandidate`` を作る -- 向きはどちらも ``hand_rot`` (人間の掌姿勢
    を握手のように鏡写しした向き, ``_mirror_target_rotation``) で揃える。

    ``mirror`` が False (``~same_hand`` が false のとき -- 人間と反対側の
    手で触れ、人間と同じ方向を向く) なら鏡写しせず、``plane.rot`` をその
    まま使った候補を 1 つだけ作る -- 向かい合わないぶん指方向を反転させる
    理由が無く (+X/+Z とも人間自身のものをそのまま使ってよい)、左右の手が
    互いに鏡像であること自体が握手の鏡写しの役目を果たすので、``turn_deg``
    による調整は不要 (このとき ``turn_degs`` は無視される)。

    ``approach_coords`` はまず触れに行く位置、``press_coords`` はそこから
    ``EMBED_DEPTH`` だけ掌の中へ押し込んだ位置。
    """
    pos = palm_plane.contact_target(plane)
    embed_pos = plane.center - plane.normal * EMBED_DEPTH
    candidates = []
    for turn_deg in (turn_degs if mirror else (0.0,)):
        hand_rot = _mirror_target_rotation(plane.rot, turn_deg=turn_deg) \
            if mirror else plane.rot
        hand_rot = _correct_grasp_frame(hand_rot, arm)
        candidates.append(PalmIkCandidate(
            turn_deg=turn_deg,
            hand_rot=hand_rot,
            approach_coords=Coordinates(pos=pos.tolist(), rot=hand_rot),
            press_coords=Coordinates(pos=embed_pos.tolist(), rot=hand_rot)))
    return PalmIkTargets(
        pos=pos.tolist(),
        rot=plane.rot,
        center=plane.center.tolist(),
        normal=plane.normal.tolist(),
        candidates=candidates)


def create_pose_source(name, hand_side=None):
    """``~source`` で選んだ推定クラスのインスタンスを作る.

    ``~source:=fake`` かつ ``hand_side`` が指定されているときは、fake
    推定器が差し出す手 (``~present_hand``) をそれに揃えてから作る --
    揃っていないと、このノードが追跡する ``~hand_side`` と違う手が
    差し出されてしまう (see human_palm_contact_dataset_generator.py の
    同様の処理)。
    """
    name = str(name).lower()
    if name == 'fake':
        if hand_side:
            rospy.set_param('~present_hand', hand_side)
        from fake_people_pose_estimator_ros import FakeRosPeoplePoseEstimator
        return FakeRosPeoplePoseEstimator()
    if name == 'real':
        from people_pose_estimator_ros import RosPeoplePoseEstimator
        return RosPeoplePoseEstimator()
    raise ValueError(
        "~source must be 'real' or 'fake', got '{}'".format(name))


class _DrawingRobotInterface(object):
    """ロボットへの指令を流しつつ、その間 viewer を描き直すラッパー.

    ``AeroROSRobotInterface`` のうちこのノードが使う ``angle_vector`` と
    ``wait_interpolation`` だけを真似る。

    ``ri`` が None なら実機には繋がず、モデルの関節角を time [s] かけて
    線形補間するだけになるので、実機の無い環境でも動作を絵で確かめられる。
    その場合 ``angle_vector()`` が返す「現在角」はモデルとは別に持つ:
    IK でモデルを目標姿勢に動かしたあと、そこへ向かって補間するため。
    """

    def __init__(self, robot, scene=None, ri=None, dt=0.05):
        self.robot = robot
        self.scene = scene
        self.ri = ri
        self.dt = dt
        # True にすると描画の更新を止める (viewer は開いたまま)
        self.frozen = False
        self._actual = np.asarray(
            robot.angle_vector(), dtype=np.float64).copy()

    def redraw(self):
        if self.scene is not None and not self.frozen:
            self.scene.redraw()

    def _follow(self, duration):
        """実機が動いているあいだ、その角度を追いかけて描き直す."""
        steps = max(int(duration / self.dt), 1)
        for _ in range(steps):
            rospy.sleep(self.dt)
            if rospy.is_shutdown():
                return
            self.robot.angle_vector(self.ri.angle_vector())
            self.redraw()

    def _interpolate(self, goal, duration):
        """実機が無いので、モデルを goal まで線形に動かしながら描き直す."""
        start = self._actual.copy()
        steps = max(int(duration / self.dt), 1)
        for i in range(1, steps + 1):
            if rospy.is_shutdown():
                break
            self._actual = start + (goal - start) * (float(i) / steps)
            self.robot.angle_vector(self._actual)
            self.redraw()
            if duration > 0.0:
                rospy.sleep(self.dt)

    def angle_vector(self, av=None, time=None):
        if av is None:
            if self.ri is not None:
                return self.ri.angle_vector()
            return self._actual.copy()

        goal = np.asarray(av, dtype=np.float64).copy()
        duration = float(time) if time else 0.0
        if self.ri is not None:
            self.ri.angle_vector(goal, duration)
            self._follow(duration)
            return goal
        self._interpolate(goal, duration)
        return self._actual.copy()

    def wait_interpolation(self):
        if self.ri is not None:
            self.ri.wait_interpolation()
            self.robot.angle_vector(self.ri.angle_vector())
        self.redraw()
        return True


class HumanPalmContactBehavior:
    def __init__(self):
        rospy.init_node('human_palm_contact_behavior')

        self.marker_pub = rospy.Publisher(
            '~target_markers', MarkerArray, queue_size=1)

        # 手のひら平面には手のランドマークが要る。推定クラスは同じ private
        # 名前空間からパラメータを読むので、既定値をここで立てておく。
        rospy.set_param('~hand/enable',
                        bool(rospy.get_param('~hand/enable', True)))
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        # 推定クラスが返す座標系。base_frame 以外 (TF が引けずカメラ座標系の
        # まま返ってきた場合など) の結果でロボットを動かすと危険なので使わない。
        self.pose_frame = rospy.get_param('~output_frame', self.base_frame)
        # 名前は ~hand ではなく ~hand_side (~hand/enable と衝突するため)。
        # ここでいう「手」は差し出す人間側の手。
        self.hand = str(rospy.get_param('~hand_side', 'R')).upper()[:1]
        # ロボット側の腕・向きの決め方 (~same_hand) を決める。次ラウンドの
        # WAITING に入る直前にも呼び直すので、ラウンドの合間に rosparam で
        # 切り替えられる (see _update_hand_side).
        self._update_hand_side()
        self.min_score = rospy.get_param('~min_score', 0.1)

        self.source_name = str(rospy.get_param('~source', 'real')).lower()
        # 実機に指令を出すか。~source: fake は実機の無い環境で絵だけを
        # 確かめるためのものなので、既定では繋がない。
        self.use_ri = bool(rospy.get_param(
            '~use_robot_interface', self.source_name == 'real'))

        rospy.loginfo('Initializing robot model for full body control...')
        from skrobot.models import Aero
        self.robot = Aero(use_hand=rospy.get_param('~use_hand', True))

        ri = None
        if self.use_ri:
            from skrobot.interfaces.ros import AeroROSRobotInterface
            ri = AeroROSRobotInterface(self.robot)
            self.robot.angle_vector(ri.angle_vector())
        else:
            self.robot.reset_pose()
            # reset_pose() defaults neck_p to 25deg (down). This behavior
            # wants a slightly higher initial gaze (i.e. less downward
            # pitch, 19deg) while no target is locked -- 25deg made the
            # head start out lower than the NODDING sequence's "nod down"
            # target (22.9deg, see control_loop), so the nod's first phase
            # actually raised the head instead of lowering it.
            self.robot.neck_p_joint.joint_angle(math.radians(19.0))
            rospy.logwarn('~use_robot_interface is false: only the model '
                          'moves, the real robot is never commanded')

        self.scene = self._init_scene()
        if self.scene is not None:
            # Draw the robot's actual hand frame every redraw, so a
            # mismatch against the IK target frame (drawn in
            # _solve_palm_ik) is visible at a glance.
            self.scene.track_hand('{}arm_end_coords'.format(self.arm))
        self.ri = _DrawingRobotInterface(self.robot, self.scene, ri)

        self.last_neck_cmd_time = rospy.Time.now()

        # State machine variables
        self.state = "WAITING"
        self.target_palm_pos = None       # approach target, base_link frame
        self.target_palm_rot = None       # human's fitted orientation, 3x3
        self.target_hand_rot = None       # robot's mirrored orientation, 3x3
        self.target_palm_center = None    # for gaze/lifter, base_link frame
        self.target_palm_normal = None    # unit vector, base_link frame
        self._ik_targets = None           # PalmIkTargets, set by _lock_target
        self._locked_person = None        # Person3D, set by _lock_target;
                                           # used to build the human obstacle
                                           # model for _move_avoiding_human
        self._human_sdf = None            # UnionSDF (bone cylinders), built
                                           # once by _lock_target from
                                           # _locked_person; used by
                                           # _move_avoiding_human for arm
                                           # avoidance
        self._human_foot_sdf = None        # CylinderSDF at the person's
                                           # standing position, built once
                                           # by _lock_target; used by
                                           # _solve_palm_ik to keep the cart
                                           # (wheel_base_link) away from the
                                           # human's feet/legs, which the
                                           # bone cylinders above usually
                                           # don't cover (see
                                           # HUMAN_FOOT_REF_LIMBS)
        self.state_start_time = rospy.Time.now()
        # 平面フィットに失敗し続けている時間の計測用 (_track_with_neck 参照)。
        # None なら「まだ失敗が始まっていない」。
        self.plane_fit_timeout = rospy.get_param('~plane_fit_timeout', 4.0)
        self._plane_fit_fail_since = None
        # 直近の approach/press の結果 ("approach"/"press" -> (distance
        # [m], reached, rotation_axis, use_base))。rotation_axis/use_base は
        # _solve_palm_ik が最後に試みた IK がどの厳密さ・台車設定で解けたか
        # (解けていなければ None)。human_palm_contact_behavior_loop.py が
        # 周回ごとの成否をまとめるのに使う。
        self._last_report = {}
        # _solve_palm_ik が最後に解いた (または解けなかった) 結果
        # ({'solved', 'rotation_axis', 'use_base'})。_report_reach が
        # _last_report に転記するまでの橋渡し用。
        self._last_ik_solve = None

        # 姿勢はトピックではなく推定クラスのインスタンスから受け取る。
        # 関節点は estimator 側で base_link 相対に変換済みなので、ここでは
        # TF を引かない。
        self.source = create_pose_source(self.source_name, self.hand)
        self._source_stopped = False
        self.pose_thread = threading.Thread(target=self.pose_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()

        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        rospy.loginfo("Human Palm Contact Behavior initialized "
                      "(source=%s human_hand=%sHand robot_arm=%sarm "
                      "same_hand=%s robot_interface=%s). Waiting for human...",
                      self.source_name, self.hand, self.arm, self.same_hand,
                      self.use_ri)
        if self.source_name == 'real':
            rospy.loginfo("Note: requires the pose estimator's hand tracking "
                          "(~hand/enable:=true) and the depth/color/info "
                          "topics remapped onto ~input, ~input/depth and "
                          "~input/info.")

    def _update_hand_side(self):
        """``~same_hand`` を読み直し、ロボットが使う腕を決める.

        true (既定): 人間と同じ側の手で触れる (向かい合う, 従来の挙動)。
        false: 人間と反対側の手で触れる (人間と同じ方向を向く) -- 使う腕を
        入れ替えるだけで、鏡写しするかどうかの向きの計算は
        ``palm_plane_to_ik_targets`` (``_lock_target`` が ``mirror=self.
        same_hand`` で呼ぶ) 側の仕事。
        """
        self.same_hand = bool(rospy.get_param('~same_hand', True))
        robot_hand = self.hand if self.same_hand \
            else ('L' if self.hand == 'R' else 'R')
        self.arm = 'r' if robot_hand == 'R' else 'l'

    # ------------------------------------------------------------------
    # viewer
    # ------------------------------------------------------------------
    def _init_scene(self):
        """``~viewer`` で選んだ viewer を用意する (``none`` なら描かない)."""
        viewer_name = str(rospy.get_param('~viewer', 'trimesh')).lower()
        if viewer_name == 'none':
            return None
        draw_camera = rospy.get_param('~draw_camera_frustum', True)
        if self.source_name == 'fake' \
                and not rospy.get_param('~filter_by_fov', False):
            # fake の既定では画角の外でも関節を落とさない (~filter_by_fov
            # 参照) = 画角を考慮していないので、四角すいを描いても意味が
            # ない。
            draw_camera = False
        scene = PalmPlaneScene(
            viewer=viewer_name,
            resolution=(rospy.get_param('~viewer_width', 960),
                        rospy.get_param('~viewer_height', 720)),
            draw_skeleton=rospy.get_param('~draw_skeleton', True),
            draw_camera=draw_camera,
            smpl_model_path=rospy.get_param(
                '~smpl_model_path',
                os.path.expanduser(
                    '~/SMPL_python_v.1.0.0/smpl/models/'
                    'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')))
        if scene.smpl_load_error is not None:
            rospy.logwarn(
                'could not load the SMPL model (%s), falling back to the '
                'capsule/box skeleton drawing. Set ~smpl_model_path to a '
                'valid basicmodel_*.pkl to draw an SMPL body mesh instead.',
                scene.smpl_load_error)
        # base_link がモデルのルートなので、原点に置くだけで関節点の座標系と
        # 揃う。以降 IK で動かした姿勢がそのまま viewer に出る。
        scene.add_robot(self.robot)
        if not scene.open(
                open_browser=rospy.get_param('~open_browser', False)):
            rospy.logfatal(
                'the trimesh viewer window did not open within 10 s, so '
                'nothing can be drawn (this happens when X cannot map a '
                'pyglet window, e.g. on WSLg). Re-run with _viewer:=viser '
                'to draw in a browser, or _viewer:=none to skip drawing.')
        return scene

    # ------------------------------------------------------------------
    # pose input
    # ------------------------------------------------------------------
    def pose_loop(self):
        """推定結果を 1 フレームずつ受け取る (rospy.Subscriber の代わり).

        目標が決まって WAITING を抜けたらループを抜け、推定を止める。
        描かれている骨格と平面はその瞬間のまま残る。
        """
        while not rospy.is_shutdown() and self.state == "WAITING":
            result = self.source.wait_for_result(timeout=1.0)
            if result is None:
                rospy.logwarn_throttle(
                    5.0, "no pose result from the %s estimator",
                    self.source_name)
                continue
            self.pose_cb(result)
        self._stop_source()

    def _stop_source(self):
        """推定を止める (目標が決まったので入力は要らない)."""
        if self._source_stopped:
            return
        self._source_stopped = True
        try:
            self.source.clean_up()
        except Exception as e:
            rospy.logwarn('could not stop the %s estimator: %s',
                          self.source_name, e)
            return
        rospy.loginfo('pose estimation stopped; the skeleton on screen is '
                      'frozen as it was when the target was locked')

    @staticmethod
    def _human_ref(person):
        """「人がどこにいるか」の基準点 (gaze 用, 無ければ None)."""
        for limb in HUMAN_REF_LIMBS:
            if limb in person.limb_names:
                j = person.limb_names.index(limb)
                if person.scores[j] > 0.1:
                    return person.positions[j]
        return None

    def _human_foot_xy(self, person):
        """立ち位置 (x, y) の推定 (台車の押し出し判定用, 無ければ None).

        ``HUMAN_FOOT_REF_LIMBS`` の候補を順に試し、両側とも見えていれば
        その中点を、片側だけなら見えている方を使う。
        """
        for left, right in HUMAN_FOOT_REF_LIMBS:
            pts = []
            for limb in (left, right):
                if limb in person.limb_names:
                    j = person.limb_names.index(limb)
                    if person.scores[j] > 0.1:
                        pts.append(person.positions[j])
            if pts:
                return np.mean(pts, axis=0)[:2]
        ref = self._human_ref(person)
        if ref is not None:
            return np.asarray(ref, dtype=np.float64)[:2]
        return None

    @staticmethod
    def _human_facing_xy(person):
        """人間の体が向いている水平方向 (x, y の単位ベクトル) を肩の並びか
        ら推定する (両肩が見えていなければ None).

        ``LShoulder - RShoulder`` が人間自身の「左」方向 (
        ``fake_people_pose_estimator_ros.py`` の ``_body_positions`` と同じ
        ``yh`` )。正面 (``xh``) はそれを鉛直軸まわりに -90 度回した向き
        (``yh = R(90) xh`` の関係から ``xh = R(-90) yh``)。掌平面だけでは
        「どちらを向いているか」までは分からない (``palm_plane.fit_palm_
        plane`` の法線は手のひら自身の向きで、体の正面とは別) ので、
        ``~same_hand=False`` (人間と同じ方向を向く) の IK 種を作るのに使う。
        """
        r = person.position_of('RShoulder')
        l = person.position_of('LShoulder')
        if r is None or l is None:
            return None
        lateral = (np.asarray(l, dtype=np.float64)[:2]
                  - np.asarray(r, dtype=np.float64)[:2])
        norm = float(np.linalg.norm(lateral))
        if norm < 1e-6:
            return None
        lateral /= norm
        return np.array([lateral[1], -lateral[0]])

    def _best_palm(self, result):
        """最初の有効な人物とその手のひら平面を返す.

        Fit the palm plane to whichever palm landmarks arrived (see
        palm_plane.py for why this is a least-squares fit and not a single
        cross product over a fixed quadruple).
        """
        for person in result.people:
            if self._human_ref(person) is None:
                continue
            points = palm_plane.collect_palm_points(
                person, hand=self.hand, min_score=self.min_score)
            return palm_plane.fit_palm_plane(points, hand=self.hand), \
                points, person
        return None, {}, None

    def _draw_pose(self, result, plane, palm_points):
        if self.scene is None:
            return
        if not self.scene.update_skeleton(result.people):
            rospy.logwarn('cannot draw the skeleton; disabling it')
        self.scene.update_camera(
            result.camera_intrinsics, result.camera_width,
            result.camera_height, result.camera_pose)
        if plane is None:
            self.scene.hide_plane()
        else:
            self.scene.update_plane(plane, palm_points)
        self.scene.redraw()

    def pose_cb(self, result):
        if self.state != "WAITING":
            return

        if result.frame_id != self.pose_frame:
            # TF が引けないと推定側はカメラ座標系のまま返す。その点で IK を
            # 解くとまったく違う場所へ手を伸ばすので、フレームごと捨てる。
            rospy.logwarn_throttle(
                2.0, "pose result is in %s, not %s; skipping (TF missing?)",
                result.frame_id, self.pose_frame)
            return

        plane, palm_points, person = self._best_palm(result)
        # 描画は判断より先。首を動かしている最中もここは通るので、絵は
        # 止まらない。
        self._draw_pose(result, plane, palm_points)

        if person is None:
            return

        now = rospy.Time.now()
        if (now - self.last_neck_cmd_time).to_sec() < NECK_SETTLE_TIME:
            return
        if self._track_with_neck(person, plane, palm_points, now):
            return    # 首を動かしたのでこのフレームはここまで
        if plane is None:
            return
        self._lock_target(plane, palm_points, person, now)

    def _track_with_neck(self, person, plane, palm_points, now):
        """人の方を向く。指令を出したら True を返す."""
        hx, hy, _ = self._human_ref(person)
        neck_yaw = math.atan2(hy, hx)

        current_p = self.robot.neck_p_joint.joint_angle()
        current_y = self.robot.neck_y_joint.joint_angle()

        if plane is None:
            # self.state_start_time より前のタイムスタンプは前のラウンド
            # (あるいは前回別の理由で WAITING に入ったとき) の残骸なので、
            # 今回の WAITING に入ってから初めて失敗したものとして計測し直す。
            if self._plane_fit_fail_since is None \
                    or self._plane_fit_fail_since < self.state_start_time:
                self._plane_fit_fail_since = now
            elapsed = (now - self._plane_fit_fail_since).to_sec()
            if elapsed >= self.plane_fit_timeout:
                rospy.logwarn(
                    "Palm plane still not fittable after %.1fs (%d of %s "
                    "landmarks); giving up on this person and moving on.",
                    elapsed, len(palm_points),
                    list(palm_plane.PLANE_LANDMARKS))
                self._plane_fit_fail_since = None
                self.state = "DONE"
                return True
            rospy.loginfo_throttle(
                2.0, "Human detected, but the palm plane could not be "
                "fitted (%d of %s landmarks). Looking down...",
                len(palm_points), list(palm_plane.PLANE_LANDMARKS))
            target_p = 0.3
        else:
            target_p = current_p
            self._plane_fit_fail_since = None
        target_y = np.clip(neck_yaw, -1.5, 1.5)

        if abs(current_p - target_p) <= 0.1 \
                and abs(current_y - target_y) <= 0.1:
            return False

        self.robot.angle_vector(self.ri.angle_vector())
        self.robot.neck_y_joint.joint_angle(target_y)
        self.robot.neck_p_joint.joint_angle(target_p)
        self.ri.angle_vector(self.robot.angle_vector(), 1.5)
        self.last_neck_cmd_time = now
        return True

    def _build_human_foot_sdf(self, person):
        """立ち位置に台車の高さ分の衝突円柱を置く (無ければ None).

        ``_human_sdf`` の骨格円柱は肩から先 (腕・頭) が中心で、脚
        (Hip->Knee->Ankle) は隠れて見えないことが多く実質カバーされない。
        台車 (``wheel_base_link``) はまさに床面のその高さを動くので、
        脚の代わりに立ち位置まわりへ一律に置いた円柱で代用する
        (``_solve_palm_ik`` の台車の押し出し判定に使う)。
        """
        foot_xy = self._human_foot_xy(person)
        if foot_xy is None:
            return None
        cart_mesh = self.robot.wheel_base_link.collision_mesh
        height = float(cart_mesh.extents[2]) if cart_mesh is not None \
            else 0.15
        cylinder = Cylinder(
            radius=HUMAN_FOOT_RADIUS, height=height, with_sdf=True)
        cylinder.newcoords(Coordinates(
            pos=[float(foot_xy[0]), float(foot_xy[1]), height / 2.0]))
        return cylinder.sdf

    def _lock_target(self, plane, palm_points, person, now):
        # target_palm_pos 等は palm_plane_to_ik_targets が人間の掌平面から
        # 作った IK ターゲット。target_palm_rot だけは人間自身の掌姿勢その
        # もの (ロボット側の向きではない) を残す -- データセット生成側が
        # 人間の指方向の矢印を描くのに使う。target_hand_rot は候補
        # (±90度) のうちどれが解けるか control_loop が試すまで決まらない
        # ので、ここでは None のままにする。
        self._ik_targets = palm_plane_to_ik_targets(
            plane, mirror=self.same_hand, arm=self.arm)
        self.target_palm_pos = self._ik_targets.pos
        self.target_palm_rot = self._ik_targets.rot
        self.target_palm_center = self._ik_targets.center
        self.target_palm_normal = self._ik_targets.normal
        self.target_hand_rot = None
        # 骨格 (person.bones) は _move_avoiding_human が人体を障害物として
        # 近似するのに使う。推定は直後に止まるので、以降はこのフレームの
        # 骨格が固定で残る。
        self._locked_person = person
        arm_prefix = 'R' if self.hand == 'R' else 'L'
        try:
            self._human_sdf = _human_obstacle_sdf(person, arm_prefix)
        except Exception as e:
            rospy.logwarn('could not build human obstacle model: %s', e)
            self._human_sdf = None
        self._human_foot_sdf = self._build_human_foot_sdf(person)

        # Same geometry the viewer draws, so what you see is what the robot
        # is about to reach for.
        marker_array = palm_plane.palm_plane_markers(
            plane, self.pose_frame, stamp=now, ns="targets", label="target")
        marker_array.markers.extend(palm_plane.palm_landmark_markers(
            palm_points, self.pose_frame, stamp=now, ns="targets",
            used=plane.used).markers)
        self.marker_pub.publish(marker_array)

        rospy.loginfo(
            "Found palm! used=%s rms=%.1fmm. Target locked. "
            "Executing contact sequence...",
            plane.used, plane.rms * 1000.0)
        self.state = "NODDING"
        self.state_start_time = rospy.Time.now()

    # ------------------------------------------------------------------
    # motion
    # ------------------------------------------------------------------
    def _look_at(self, target_pos):
        """Point the neck at ``target_pos`` (base_link frame).

        Solved in two stages, one per joint's own parent link, instead of
        the old formula's single fixed "neck is at height 1.2" guess in
        base_link: neck_y_joint's parent (body_link) and neck_p_joint's
        parent (neck_link) sit at different heights, and whole-body IK is
        free to turn the torso (waist_y/waist_p) while reaching, which
        rotates both relative to base_link -- a gaze angle computed before
        the arm IK runs no longer points at the target once the torso has
        turned.  Yaw is solved first and applied immediately so the pitch
        stage reads neck_link's post-yaw transform, matching how the two
        joints actually compose.  Call this again after each whole-body IK
        solve so the head keeps looking at the human's hand through the
        actual approach/press motion, not just during the initial "look"
        step.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64)

        yaw_lx, yaw_ly, _ = self.robot.neck_y_joint.parent_link \
            .inverse_transform_vector(target_pos)
        self.robot.neck_y_joint.joint_angle(
            np.clip(math.atan2(yaw_ly, yaw_lx), -1.5, 1.5))

        pitch_lx, pitch_ly, pitch_lz = self.robot.neck_p_joint.parent_link \
            .inverse_transform_vector(target_pos)
        neck_pitch = math.atan2(pitch_lz, math.hypot(pitch_lx, pitch_ly))
        self.robot.neck_p_joint.joint_angle(np.clip(-neck_pitch, -0.3, 0.5))


    def _report_reach(self, label, target):
        """IK が解けた姿勢で手先が目標からどれだけ離れているかを出す.

        IK が収束しなかったときは fallback 姿勢のまま指令が出るので、届いた
        つもりで空振りしていないかはここを見て判断する。
        """
        end_coords = getattr(self.robot, '{}arm_end_coords'.format(self.arm))
        distance = float(np.linalg.norm(
            end_coords.worldpos()
            - np.asarray(target, dtype=np.float64)))
        reached = distance <= 0.02
        # rotation_axis/use_base: 直前の _solve_palm_ik がどの厳密さ・台車
        # 設定で解けたか (解けていなければ両方 None) -- _solve_palm_ik
        # 参照。dataset generator の log.jsonl はこれをそのまま書き出す。
        ik_solve = self._last_ik_solve or {}
        self._last_report[label] = (
            distance, reached,
            ik_solve.get('rotation_axis'), ik_solve.get('use_base'))
        if reached:
            rospy.loginfo('%s pose: %.1f mm from the target',
                          label, distance * 1000.0)
        else:
            rospy.logwarn('%s pose: %.1f mm from the target -- IK did not '
                          'reach it', label, distance * 1000.0)

    def _clear_cart_from_foot(self, foot_sdf,
                              margin=CART_AVOIDANCE_SAFETY_MARGIN):
        """台車 (``wheel_base_link``) の中心 (x, y) を ``foot_sdf``
        (``_build_human_foot_sdf`` の結果, 立ち位置に置いた円柱) の半径
        ``HUMAN_FOOT_RADIUS`` + ``margin`` の外側まで押し出した位置を返す。
        台車の中心が既にその外側にあれば None。

        台車の footprint 自体の大きさは見ず、中心点が円の中に入っていない
        かだけを見る (台車の外接円まで足すと過剰にマージンを取ってしまう
        ため、ここでは中心点のみのシンプルな判定にしている)。

        台車自体は実機では動かない -- ``use_base='planar'`` はモデルの中だ
        けの仮想関節で (module docstring および ``_move_avoiding_human`` 参
        照)、実機へは ``angle_vector`` (関節角のみ) しか送らない。つまり
        台車と人間の干渉は経路計画で「避ける」ものではなく、IK が仮想的に
        選んだ台車位置が近すぎる場合にその場で押し出し、その位置に台車を
        固定した上で腕だけを解き直す (呼び出し側の ``_solve_palm_ik`` 参
        照) ためにここで押し出し先を計算する。
        """
        if foot_sdf is None:
            return None
        cart_xy = np.asarray(self.robot.translation[:2], dtype=np.float64)
        foot_xy = np.asarray(foot_sdf.worldpos()[:2], dtype=np.float64)
        needed_clear = HUMAN_FOOT_RADIUS + margin

        offset = cart_xy - foot_xy
        dist = float(np.linalg.norm(offset))
        if dist >= needed_clear:
            return None
        direction = offset / dist if dist > 1e-6 else np.array([1.0, 0.0])
        return foot_xy + direction * needed_clear

    def _solve_palm_ik(self, whole_body, target_coords, use_base=None,
                        foot_sdf=None):
        """Solve whole-body IK for ``target_coords``.

        ``target_coords`` carries a fully-determined orientation (see
        ``_mirror_target_rotation``) rather than leaving any axis free, so
        this first asks for an exact match on all 3 axes
        (``rotation_axis=True``).  ``target_coords`` is drawn in the viewer
        as the IK target frame regardless of whether the solve succeeds,
        so a bad target is visible even when IK fails.

        If the full 3-axis solve doesn't converge, this does *not* drop
        straight to position-only: it first retries with
        ``rotation_axis='y'``.  skrobot's legacy ``rotation_axis`` string
        names the axis to leave *unconstrained*, so ``'y'`` frees the
        local X/Z rotation-error components while still forcing the local
        Y axis -- ``r/l_eef_grasp_link``'s dorsum->palm direction, i.e.
        exactly the axis that has to match the human palm's normal for
        the hand to press flush against it (see the module comment above
        ``_mirror_target_rotation``) -- to converge exactly.  What's left
        free is only the roll around that axis (finger direction / thumb
        side), which matters far less for making contact than the palm
        normal does.  Only if that also fails does it drop to
        ``rotation_axis=False`` (position-only, no orientation control at
        all) as the last resort.

        ``stop=200`` (default 50) and ``revert_if_fail=False``: this
        reorientation can be large (the palm normal can point off to the
        side, not just roughly at the camera -- see
        fake_people_pose_estimator_ros.py's handshake-like
        ``~present_hand`` pose) and needs more iterations than the default
        budget; with ``revert_if_fail`` at its default of True the solver
        snaps back to the seed on every non-improving iteration, which
        empirically gets it stuck at the seed instead of working through
        the reorientation at all.

        ``use_base`` is passed through to ``inverse_kinematics`` as-is
        (e.g. ``'planar'`` to let the cart move in the IK solve); left at
        its default (``None``) the base is not included. skrobot's
        ``inverse_kinematics`` has no working collision-avoidance hook of
        its own (its ``check_collision``/``obstacles`` kwargs are unused
        dead code), so when ``use_base`` did let the base move and the
        solved cart position ends up too close to ``foot_sdf`` (see
        ``_clear_cart_from_foot``), the cart is pushed clear of it and the
        arm is re-solved with the cart held fixed at that pushed-out
        position (rather than snapping all the way back to no base motion
        at all).
        """
        if self.scene is not None:
            self.scene.update_ik_target(target_coords)

        def try_ik(base_choice):
            """Returns (solved, rotation_axis) -- rotation_axis is whichever
            of True/'y'/False converged, or None if all three failed."""
            for rotation_axis in (True, 'y', False):
                try:
                    res = whole_body.inverse_kinematics(
                        target_coords, rotation_axis=rotation_axis,
                        stop=200, revert_if_fail=False, use_base=base_choice)
                except Exception as e:
                    rospy.logwarn("IK failed (rotation_axis=%r): %s",
                                  rotation_axis, e)
                    res = False
                if res is not False:
                    rospy.loginfo("IK converged with rotation_axis=%r "
                                  "(use_base=%r).", rotation_axis, base_choice)
                    return True, rotation_axis
                rospy.logwarn("IK did not converge with rotation_axis=%r "
                              "(use_base=%r); %s.", rotation_axis, base_choice,
                              "trying a looser rotation_axis"
                              if rotation_axis is not False
                              else "giving up on this target_coords")
            return False, None

        # 最終的にどの (rotation_axis, use_base) で解けた/解けなかったかを
        # _report_reach 経由で _last_report に残す (dataset generator の
        # log.jsonl 出力用)。
        solved, rotation_axis = try_ik(use_base)
        final_use_base = use_base

        if solved and use_base is not None:
            pushed_xy = self._clear_cart_from_foot(foot_sdf)
            if pushed_xy is not None:
                rospy.logwarn(
                    "IK moved the cart to within %.0f mm of the human's "
                    "feet; pushing it clear and re-solving the arm with "
                    "the cart fixed there.",
                    CART_AVOIDANCE_SAFETY_MARGIN * 1000.0)
                theta, _, _ = matrix2ypr(self.robot.rotation)
                self.robot.newcoords(Coordinates(
                    pos=[pushed_xy[0], pushed_xy[1], 0.0],
                    rot=rpy_matrix(theta, 0.0, 0.0)))
                solved, rotation_axis = try_ik(None)
                final_use_base = None

        self._last_ik_solve = {
            'solved': solved,
            'rotation_axis': rotation_axis if solved else None,
            'use_base': final_use_base if solved else None,
        }
        return solved

    def _solve_palm_ik_candidates(self, whole_body, candidates, coords_attr,
                                   use_base=None, foot_sdf=None):
        """``candidates`` (``PalmIkCandidate`` のリスト, 通常は ±90度の 2 つ)
        を順に試し、IK が解けた最初の候補を返す.

        ``coords_attr`` は各候補から取り出す座標の属性名 (``'approach_
        coords'`` / ``'press_coords'``)。どちらも解けなければ最後に試した
        候補のまま (フォールバック姿勢) を返す -- ``_solve_palm_ik`` の
        「解けなくても描画・続行はする」という従来の挙動を踏襲する。

        Returns
        -------
        (candidate, solved) : 使う候補と、それが実際に解けたかどうか。
        """
        for i, candidate in enumerate(candidates):
            target_coords = getattr(candidate, coords_attr)
            if self._solve_palm_ik(
                    whole_body, target_coords, use_base=use_base,
                    foot_sdf=foot_sdf):
                rospy.loginfo(
                    'IK solved with a %.0f deg mirrored turn.',
                    candidate.turn_deg)
                return candidate, True
            rospy.logwarn(
                'IK could not solve with a %.0f deg mirrored turn%s.',
                candidate.turn_deg,
                '; trying the next candidate' if i + 1 < len(candidates)
                else '')
        return candidates[-1], False

    def _move_avoiding_human(self, whole_body, av_start, duration):
        """``av_start`` から ``whole_body`` の現在の関節角 (直前に
        ``_solve_palm_ik`` が解いた姿勢) まで、人体を避ける経路を通って
        実際に動かす。

        経路は ``sqp_plan_trajectory`` (SQP, see collision_free_trajectory.py
        example) で ``self._locked_person`` の骨格を円柱近似した SDF
        (``_human_obstacle_sdf``) を障害物として最適化する。骨格が無い
        (fake 推定など) か最適化が失敗した場合は、これまで通り直接補間して
        動かす。

        ベース (waist/lifter 以外の台車) は対象にしない -- 実機への指令は
        ``angle_vector`` (関節角のみ) で、台車を動かすものではないので、
        経路計画も実際に送る関節だけを対象にすれば指令と一致する。
        """
        joint_list = list(whole_body.joint_list)
        av_goal = get_robot_config(self.robot, joint_list, with_base=False)

        human_sdf = self._human_sdf

        def move_directly():
            set_robot_config(self.robot, joint_list, av_goal)
            self.ri.angle_vector(self.robot.angle_vector(), duration)
            self.ri.wait_interpolation()

        if human_sdf is None:
            move_directly()
            return

        checker = SweptSphereSdfCollisionChecker(human_sdf, self.robot)
        for link in whole_body.link_list:
            if link.collision_mesh is None:
                continue
            try:
                checker.add_collision_link(link)
            except Exception as e:
                rospy.logwarn('skipping %s for human-collision checking: '
                              '%s', link.name, e)

        try:
            av_seq = sqp_plan_trajectory(
                checker, av_start, av_goal, joint_list,
                HUMAN_AVOIDANCE_WAYPOINTS,
                safety_margin=HUMAN_AVOIDANCE_SAFETY_MARGIN)
        except Exception as e:
            rospy.logwarn('human-avoiding trajectory planning failed (%s); '
                          'moving directly instead', e)
            move_directly()
            return

        step_duration = duration / len(av_seq)
        for av in av_seq:
            set_robot_config(self.robot, joint_list, av)
            self.ri.angle_vector(self.robot.angle_vector(), step_duration)
            self.ri.wait_interpolation()

    def control_loop(self, event):
        if self.state == "NODDING":
            self.robot.angle_vector(self.ri.angle_vector())
            # Nod down
            self.robot.neck_p_joint.joint_angle(0.4)
            self.ri.angle_vector(self.robot.angle_vector(), 1.0)
            self.ri.wait_interpolation()

            # Nod up (look forward)
            self.robot.neck_p_joint.joint_angle(-0.2)
            self.ri.angle_vector(self.robot.angle_vector(), 1.0)
            self.ri.wait_interpolation()

            self.state = "REACHING"
            rospy.loginfo("Nod finished, reaching toward the palm...")

        elif self.state == "REACHING":
            self.robot.angle_vector(self.ri.angle_vector())

            cz = self.target_palm_center[2]

            # Torso up/down (lifter) to roughly match hand height. This is
            # just a seed pose -- the whole-body IK below (the arm matching
            # ~hand_side, e.g. rarm_whole_body) is free to further adjust
            # the lifter *and* the waist yaw/pitch
            # joints (waist_y_joint, waist_p_joint) to reach low targets,
            # e.g. a seated person's hand, without moving the base.
            #
            # knee_joint/ankle_joint form a parallelogram lifter (two 0.25m
            # links, see lifter.urdf.xacro) driven by equal-and-opposite
            # angles so the torso stays level while it crouches. Relative to
            # its *current* angle, height given up is
            # 0.5 * (cos(current_angle) - cos(angle)) [m] -- not linear in
            # the angle (flat near angle=current_angle, steepening as it
            # crouches further), so invert that relation for the angle
            # instead of scaling the height gap directly. knee_joint's
            # range is negative ([-1.5707, 0]) and ankle_joint's is
            # positive ([0, 1.5707]) -- passing the same positive "amount"
            # to both as this used to do got silently clamped to 0 on one
            # of them every round, so the lifter never actually moved.
            #
            # Reference height is the current waist (body_link) height
            # minus 0.1 m -- at reset_pose that's ~0.90m (body_link sits at
            # ~1.00m), so a target at or above that height needs no crouch
            # beyond whatever the lifter is already at (drop=0 below keeps
            # the current angle exactly, rather than snapping to fully
            # extended).
            waist_z = self.robot.body_link.worldpos()[2]
            current_lifter_angle = -self.robot.knee_joint.joint_angle()
            drop = np.clip(waist_z - 0.1 - cz, 0.0, None)
            # == -knee_joint.min_angle; the two ranges mirror each other.
            max_lifter_angle = self.robot.ankle_joint.max_angle
            cos_arg = np.clip(
                math.cos(current_lifter_angle) - 2.0 * float(drop),
                math.cos(max_lifter_angle), 1.0)
            lifter_angle = math.acos(cos_arg)
            try:
                self.robot.knee_joint.joint_angle(-lifter_angle)
                if hasattr(self.robot, 'ankle_joint'):
                    self.robot.ankle_joint.joint_angle(lifter_angle)
            except AttributeError:
                pass

            # Look at the palm being touched.
            self._look_at(self.target_palm_center)

            rospy.loginfo("Adjusting posture and gaze first...")
            self.ri.angle_vector(self.robot.angle_vector(), 2.0)
            self.ri.wait_interpolation()

            rospy.loginfo("Extending arm toward the palm...")
            self.robot.angle_vector(self.ri.angle_vector())

            # Fallback posture (natural elbow position) used as the IK seed.
            # (wrist_p/wrist_r are clamped to this arm's actual joint
            # limits -- +-5deg / -85..+25deg for the left arm, +-5deg /
            # -25..+85deg for the right -- the original +0.2/+1.5 were both
            # out of range and silently clipped by skrobot.)
            #
            # shoulder_r/shoulder_y/wrist_r mirror in sign between the two
            # arms (see aero_upper_typef.urdf.xacro's joint limits, which
            # mirror the same way); shoulder_p/elbow/wrist_p/wrist_y don't.
            mirror = 1.0 if self.arm == 'l' else -1.0
            whole_body = getattr(self.robot, '{}arm_whole_body'.format(self.arm))
            # whole_body is a sub-chain RobotModel (lifter + arm links only,
            # see Aero._limb/_lifter_links) that doesn't include base_link,
            # so none of its own links have parent_link=None and IK's
            # automatic root-link scan (used by use_base='planar' below to
            # decide where to attach the virtual cart joint) fails. Point
            # it at the real root explicitly instead.
            whole_body.root_link = self.robot.base_link

            # Captured here (synced from the real/interpolated arm at line
            # ~924, before the fallback seed posture below overwrites it in
            # the model) so _move_avoiding_human can later plan a path from
            # where the arm actually is, not from the IK seed.
            joint_list = list(whole_body.joint_list)
            av_start_approach = get_robot_config(
                self.robot, joint_list, with_base=False)

            if self.same_hand:
                # 向かい合って握手のように前へ伸ばす自然な肘の構え。
                getattr(self.robot, '{}_shoulder_p_joint'.format(self.arm)) \
                    .joint_angle(-0.4)
                getattr(self.robot, '{}_shoulder_r_joint'.format(self.arm)) \
                    .joint_angle(0.2 * mirror)
                getattr(self.robot, '{}_shoulder_y_joint'.format(self.arm)) \
                    .joint_angle(0.5 * mirror)
                getattr(self.robot, '{}_elbow_joint'.format(self.arm)) \
                    .joint_angle(-1.2)
                getattr(self.robot, '{}_wrist_y_joint'.format(self.arm)) \
                    .joint_angle(0.0)
                getattr(self.robot, '{}_wrist_p_joint'.format(self.arm)) \
                    .joint_angle(0.087)
                getattr(self.robot, '{}_wrist_r_joint'.format(self.arm)) \
                    .joint_angle(0.436 * mirror)
            else:
                # 同じ方向を向いて横に並び、手をつなぐ構え。目標は体の正面
                # ではなく横 (人間の側) にあるので、上の「前へ伸ばす」種を
                # そのまま使うと種と目標の姿勢差が大きすぎて IK が迷走しや
                # すい (round_pause 前後で見えなくなる不具合の一因)。肩を
                # 前へ挙げる (shoulder_p) 代わりに横へ開き (shoulder_r) 、
                # 手首もひねらないニュートラルな構えにしておく。
                # (経験的な初期値。実機/シミュレータでの収束率を見ながら
                # 調整すること -- shoulder_r/y の可動域は shoulder_p/elbow/
                # wrist_p/wrist_y ほど厳密に把握していない。)
                getattr(self.robot, '{}_shoulder_p_joint'.format(self.arm)) \
                    .joint_angle(0.0)
                getattr(self.robot, '{}_shoulder_r_joint'.format(self.arm)) \
                    .joint_angle(0.8 * mirror)
                getattr(self.robot, '{}_shoulder_y_joint'.format(self.arm)) \
                    .joint_angle(0.0)
                getattr(self.robot, '{}_elbow_joint'.format(self.arm)) \
                    .joint_angle(-1.0)
                getattr(self.robot, '{}_wrist_y_joint'.format(self.arm)) \
                    .joint_angle(0.0)
                getattr(self.robot, '{}_wrist_p_joint'.format(self.arm)) \
                    .joint_angle(0.087)
                getattr(self.robot, '{}_wrist_r_joint'.format(self.arm)) \
                    .joint_angle(0.0)

            # ~same_hand が False (人間と反対の手で繋ぐ) のときは、ロボット
            # は人間と向かい合うのではなく同じ方向を向く。回転だけを種にし
            # て位置は変えないと、目標が体の正面ではなくほぼ真後ろになって
            # しまい、whole-body IK (use_base='planar' で base_link 自体も
            # 動かせる) の探索が不安定になりやすい (round_pause 前後でロボ
            # ットが見えなくなる不具合の一因だった)。そこで向きだけでなく
            # 立ち位置も、人間の正面方向 (肩のラインから推定,
            # _human_facing_xy) の真横へずらした姿勢を種にする。人間の正面
            # 方向が分からなければ (肩が見えていなければ) 何もせず、従来通
            # り向かい合う想定の種のまま IK に任せる。
            #
            # base_link は 'planar' のときだけモデル内で動く仮想関節で、実
            # 機へは角度指令 (angle_vector) しか送らない (_move_avoiding_
            # human 冒頭のコメント参照) ので、ここでの移動・回転はあくまで
            # IK の種であり、実機の台車を動かすものではない。
            if not self.same_hand:
                facing = self._human_facing_xy(self._locked_person) \
                    if self._locked_person is not None else None
                if facing is None:
                    rospy.logwarn(
                        "same_hand=False: could not tell which way the "
                        "human is facing (shoulders not visible); using "
                        "the default (face-to-face) IK seed instead.")
                else:
                    yaw = math.atan2(facing[1], facing[0])
                    # 人間の「左」方向。ロボットが左腕で触れるなら人間の
                    # 右側に、右腕で触れるなら人間の左側に立ちたいので、
                    # 目標位置からこの方向へ立ち位置をずらす符号は腕の左右
                    # で反転する。
                    human_left = np.array([-facing[1], facing[0]])
                    side = 1.0 if self.arm == 'r' else -1.0
                    target_xy = np.asarray(
                        self._ik_targets.pos[:2], dtype=np.float64)
                    base_xy = target_xy \
                        + side * SAME_DIRECTION_STANDOFF * human_left
                    self.robot.base_link.newcoords(Coordinates(
                        pos=[base_xy[0], base_xy[1], 0.0],
                        rot=rpy_matrix(yaw, 0.0, 0.0)))

            # Each candidate carries a differently-turned mirrored handshake
            # orientation (see MIRROR_TURN_CANDIDATES_DEG /
            # _mirror_target_rotation) -- solved exactly on all 3 axes, see
            # _solve_palm_ik. Try them in order and keep whichever one IK
            # actually reaches; the press pose below reuses that same
            # candidate so the hand doesn't twist between approach and
            # press.
            approach_candidate, approach_ok = self._solve_palm_ik_candidates(
                whole_body, self._ik_targets.candidates, 'approach_coords',
                use_base='planar', foot_sdf=self._human_foot_sdf)
            if not approach_ok:
                rospy.logwarn(
                    "IK could not reach the approach pose with either "
                    "mirrored turn; using fallback posture.")
            self.target_hand_rot = approach_candidate.hand_rot

            # Whole-body IK may have turned the waist to reach the target,
            # which rotates the neck with it -- re-aim after solving so
            # the head is actually looking at the hand once the arm gets
            # there, not just where it happened to be looking before.
            self._look_at(self.target_palm_center)

            self._report_reach('approach', self.target_palm_pos)
            self._move_avoiding_human(whole_body, av_start_approach, 2.0)

            # Second, slower motion: press past the approach pose so the
            # hand actually sinks into the palm rather than stopping just
            # short of it.
            rospy.loginfo("Pressing into the palm...")
            self.robot.angle_vector(self.ri.angle_vector())

            embed_coords = approach_candidate.press_coords

            av_start_press = get_robot_config(
                self.robot, joint_list, with_base=False)

            if not self._solve_palm_ik(whole_body, embed_coords):
                rospy.logwarn(
                    "IK could not reach the press pose; staying at the "
                    "approach pose.")

            # Same re-aim as after the approach solve -- keep looking at
            # the hand (not the embed point under the skin) while pressing.
            self._look_at(self.target_palm_center)

            self._report_reach('press', embed_coords.worldpos())
            self._move_avoiding_human(whole_body, av_start_press, 1.5)

            rospy.loginfo("Finished palm contact behavior sequence.")
            # 接近と押し込みが終わったので描画の更新も止める。viewer は
            # 開いたままなので、最終姿勢を回して確かめられる。
            self.ri.frozen = True
            self.state = "DONE"

        elif self.state == "DONE":
            pass


if __name__ == '__main__':
    try:
        HumanPalmContactBehavior()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
