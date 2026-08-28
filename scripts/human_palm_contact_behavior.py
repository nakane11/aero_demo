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
                                         ロボットは同じ側の手で触れる (``R``
                                         なら右手、``L`` なら左手)
                                         (``~hand`` は推定クラスの
                                         ``~hand/enable`` と衝突するので使わない)
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
from skrobot.coordinates.math import rotate_vector

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

# 首を動かしてから次の判断までの待ち [s]。関節が動いている途中の角度で
# 目標を決めないための間。
NECK_SETTLE_TIME = 2.5

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


def _mirror_target_rotation(rot):
    """Handshake-mirror a palm-target rotation (e.g. ``PalmPlane.rot``).

    ``rot``'s +X is the finger direction and +Z is the thumb<->pinky width
    axis, both copied straight from the human's fitted palm (see
    ``palm_plane.fit_palm_plane``); +Y already faces the robot's palm
    toward the human's.  Two things need fixing before this is a plausible
    handshake instead of a plain copy of the human's own frame:

    * Roll around the finger axis: copying +Z as-is would give the robot
      the *same* roll as the human -- i.e. the same tilt away from a
      plain vertical, thumb-up handshake -- instead of a mirrored one.
      Two people shaking hands face each other as mirror images: if the
      human tilts N degrees off vertical one way, the robot should tilt N
      degrees off vertical the *other* way (both measured from the same
      "vertical, thumb-up" reference for this approach direction), so
      e.g. a human tilted 45 deg one way and a robot mirrored 45 deg the
      other way end up 90 deg apart rather than lined up.
    * Finger direction: the human's +X points from their wrist toward
      their fingertips, i.e. roughly away from their own body.  Copying
      it as the robot's +X would point the robot's fingertips the same
      way -- past the human's hand, away from them -- whereas in an
      actual handshake each hand's fingers point back across the other's
      palm toward *their* wrist.  So the robot's +X is the human's +X
      negated (a 180 deg turn about the shared +Y, which leaves +Y --
      "face the human" -- untouched).
    """
    x_axis = _unit(rot[:, 0])
    if x_axis is None:
        return rot

    up = np.array([0.0, 0.0, 1.0])
    # Invariant under x_axis -> -x_axis (it only depends on x_axis via its
    # projection matrix), so it doubles as the reference for the
    # finger-reversed axis used below.
    up_perp = _unit(up - float(np.dot(up, x_axis)) * x_axis)
    if up_perp is None:
        # Finger axis ~vertical: no well-defined "vertical, thumb-up"
        # reference to mirror around, so keep the human's own orientation
        # rather than doing something arbitrary.
        return rot

    z_axis_in = rot[:, 2]
    theta = math.atan2(
        float(np.dot(x_axis, np.cross(up_perp, z_axis_in))),
        float(np.dot(up_perp, z_axis_in)))

    z_axis = _unit(rotate_vector(up_perp, -theta, x_axis))
    if z_axis is None:
        return rot
    y_axis = np.cross(z_axis, x_axis)
    # 180 deg about +Y: fingers point back toward the human's wrist
    # instead of past their hand, while +Y (facing the human) is
    # untouched.
    return np.column_stack([-x_axis, y_axis, -z_axis])


# 人間の掌平面 (palm_plane.fit_palm_plane の結果) を IK ターゲットへ変換した
# もの。pos/rot は人間自身の掌の位置・姿勢 (contact_target/plane.rot その
# まま、_lock_target が描画やデータセット生成のために残しておく分)、
# hand_rot はロボットの手が向くべき向き (_mirror_target_rotation 参照)、
# center/normal は gaze/lifter が使う掌中心とその法線。
PalmIkTargets = namedtuple('PalmIkTargets', [
    'pos', 'rot', 'hand_rot', 'center', 'normal',
    'approach_coords', 'press_coords'])


def palm_plane_to_ik_targets(plane):
    """人間の掌平面から、ロボットが IK で解く approach/press ターゲット
    (skrobot ``Coordinates``, base_link frame) を作る.

    ``approach_coords`` はまず触れに行く位置、``press_coords`` はそこから
    ``EMBED_DEPTH`` だけ掌の中へ押し込んだ位置で、向きはどちらも
    ``hand_rot`` (人間の掌姿勢を握手のように鏡写しした向き) で揃える。
    """
    pos = palm_plane.contact_target(plane)
    hand_rot = _mirror_target_rotation(plane.rot)
    embed_pos = plane.center - plane.normal * EMBED_DEPTH
    return PalmIkTargets(
        pos=pos.tolist(),
        rot=plane.rot,
        hand_rot=hand_rot,
        center=plane.center.tolist(),
        normal=plane.normal.tolist(),
        approach_coords=Coordinates(pos=pos.tolist(), rot=hand_rot),
        press_coords=Coordinates(pos=embed_pos.tolist(), rot=hand_rot))


def create_pose_source(name):
    """``~source`` で選んだ推定クラスのインスタンスを作る."""
    name = str(name).lower()
    if name == 'fake':
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
        # ここでいう「手」は差し出す人間側の手。ロボットは同じ側の手で触れる
        # (人が右手を出したらロボットは右手で、左手を出したら左手で)。
        self.hand = str(rospy.get_param('~hand_side', 'R')).upper()[:1]
        self.arm = 'r' if self.hand == 'R' else 'l'
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
        self.state_start_time = rospy.Time.now()
        # 平面フィットに失敗し続けている時間の計測用 (_track_with_neck 参照)。
        # None なら「まだ失敗が始まっていない」。
        self.plane_fit_timeout = rospy.get_param('~plane_fit_timeout', 4.0)
        self._plane_fit_fail_since = None
        # 直近の approach/press の結果 ("approach"/"press" -> (distance
        # [m], reached))。human_palm_contact_behavior_loop.py が周回ごとの
        # 成否をまとめるのに使う。
        self._last_report = {}

        # 姿勢はトピックではなく推定クラスのインスタンスから受け取る。
        # 関節点は estimator 側で base_link 相対に変換済みなので、ここでは
        # TF を引かない。
        self.source = create_pose_source(self.source_name)
        self._source_stopped = False
        self.pose_thread = threading.Thread(target=self.pose_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()

        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        rospy.loginfo("Human Palm Contact Behavior initialized "
                      "(source=%s human_hand=%sHand robot_arm=%sarm "
                      "robot_interface=%s). Waiting for human...",
                      self.source_name, self.hand, self.arm, self.use_ri)
        if self.source_name == 'real':
            rospy.loginfo("Note: requires the pose estimator's hand tracking "
                          "(~hand/enable:=true) and the depth/color/info "
                          "topics remapped onto ~input, ~input/depth and "
                          "~input/info.")

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
        self._lock_target(plane, palm_points, now)

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

    def _lock_target(self, plane, palm_points, now):
        # target_palm_pos/target_hand_rot 等は palm_plane_to_ik_targets が
        # 人間の掌平面から作った IK ターゲット。target_palm_rot だけは
        # 人間自身の掌姿勢そのもの (ロボット側の向きではない) を残す --
        # データセット生成側が人間の指方向の矢印を描くのに使う。
        self._ik_targets = palm_plane_to_ik_targets(plane)
        self.target_palm_pos = self._ik_targets.pos
        self.target_palm_rot = self._ik_targets.rot
        self.target_palm_center = self._ik_targets.center
        self.target_palm_normal = self._ik_targets.normal
        self.target_hand_rot = self._ik_targets.hand_rot

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
        self._last_report[label] = (distance, reached)
        if reached:
            rospy.loginfo('%s pose: %.1f mm from the target',
                          label, distance * 1000.0)
        else:
            rospy.logwarn('%s pose: %.1f mm from the target -- IK did not '
                          'reach it', label, distance * 1000.0)

    def _solve_palm_ik(self, whole_body, target_coords, use_base=None):
        """Solve whole-body IK for ``target_coords``.

        ``target_coords`` carries a fully-determined orientation (see
        ``_mirror_target_rotation``) rather than leaving any axis free, so
        this asks for an exact match on all 3 axes (``rotation_axis=True``).
        ``target_coords`` is drawn in the viewer as the IK target frame
        regardless of whether the solve succeeds, so a bad target is
        visible even when IK fails.  Falls back to position-only if the
        full orientation doesn't converge.

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
        its default (``None``) the base is not included.
        """
        if self.scene is not None:
            self.scene.update_ik_target(target_coords)
        for rotation_axis in (True, False):
            try:
                res = whole_body.inverse_kinematics(
                    target_coords, rotation_axis=rotation_axis,
                    stop=200, revert_if_fail=False, use_base=use_base)
            except Exception as e:
                rospy.logwarn("IK failed (rotation_axis=%r): %s",
                              rotation_axis, e)
                res = False
            if res is not False:
                return True
        return False

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

            target_coords = self._ik_targets.approach_coords

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

            # target_coords carries the mirrored handshake orientation
            # (see _mirror_target_rotation) -- solved exactly on all 3
            # axes, see _solve_palm_ik.
            if not self._solve_palm_ik(
                    whole_body, target_coords, use_base='planar'):
                rospy.logwarn(
                    "IK could not reach the approach pose; using fallback "
                    "posture.")

            # Whole-body IK may have turned the waist to reach the target,
            # which rotates the neck with it -- re-aim after solving so
            # the head is actually looking at the hand once the arm gets
            # there, not just where it happened to be looking before.
            self._look_at(self.target_palm_center)

            self._report_reach('approach', self.target_palm_pos)
            self.ri.angle_vector(self.robot.angle_vector(), 2.0)
            self.ri.wait_interpolation()

            # Second, slower motion: press past the approach pose so the
            # hand actually sinks into the palm rather than stopping just
            # short of it.
            rospy.loginfo("Pressing into the palm...")
            self.robot.angle_vector(self.ri.angle_vector())

            embed_coords = self._ik_targets.press_coords

            if not self._solve_palm_ik(whole_body, embed_coords):
                rospy.logwarn(
                    "IK could not reach the press pose; staying at the "
                    "approach pose.")

            # Same re-aim as after the approach solve -- keep looking at
            # the hand (not the embed point under the skin) while pressing.
            self._look_at(self.target_palm_center)

            self._report_reach('press', embed_coords.worldpos())
            self.ri.angle_vector(self.robot.angle_vector(), 1.5)
            self.ri.wait_interpolation()

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
