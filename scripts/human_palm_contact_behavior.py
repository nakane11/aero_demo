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
``~hand_side`` (str, default ``R``)      触る手、``R`` か ``L``
                                         (``~hand`` は推定クラスの
                                         ``~hand/enable`` と衝突するので使わない)
``~min_score`` (float, default 0.1)
``~viewer`` (str, default ``trimesh``)   ``trimesh`` / ``viser`` / ``none``
``~open_browser`` (bool, default false)  ``~viewer: viser`` のときブラウザを
                                         自動で開くか
``~draw_skeleton`` (bool, default true)  骨格を線で描くか
``~viewer_width`` / ``~viewer_height`` (int, default 960 / 720)
``~base_frame`` (str, default base_link)
``~output_frame`` (str, default ``~base_frame``)
"""

import math
import os
import sys
import threading

import numpy as np
import rospy
from visualization_msgs.msg import MarkerArray

from skrobot.coordinates import Coordinates

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
        self.hand = str(rospy.get_param('~hand_side', 'R')).upper()[:1]
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
            rospy.logwarn('~use_robot_interface is false: only the model '
                          'moves, the real robot is never commanded')

        self.scene = self._init_scene()
        self.ri = _DrawingRobotInterface(self.robot, self.scene, ri)

        self.last_neck_cmd_time = rospy.Time.now()

        # State machine variables
        self.state = "WAITING"
        self.target_palm_pos = None       # approach target, base_link frame
        self.target_palm_rot = None       # 3x3 rotation matrix, base_link frame
        self.target_palm_center = None    # for gaze/lifter, base_link frame
        self.target_palm_normal = None    # unit vector, base_link frame
        self.state_start_time = rospy.Time.now()

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
                      "(source=%s hand=%sHand robot_interface=%s). "
                      "Waiting for human...",
                      self.source_name, self.hand, self.use_ri)
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
        scene = PalmPlaneScene(
            viewer=viewer_name,
            resolution=(rospy.get_param('~viewer_width', 960),
                        rospy.get_param('~viewer_height', 720)),
            draw_skeleton=rospy.get_param('~draw_skeleton', True))
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
            return palm_plane.fit_palm_plane(points), points, person
        return None, {}, None

    def _draw_pose(self, result, plane, palm_points):
        if self.scene is None:
            return
        if not self.scene.update_skeleton(result.people):
            rospy.logwarn('cannot draw the skeleton; disabling it')
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
            rospy.loginfo_throttle(
                2.0, "Human detected, but the palm plane could not be "
                "fitted (%d of %s landmarks). Looking down...",
                len(palm_points), list(palm_plane.PLANE_LANDMARKS))
            target_p = 0.3
        else:
            target_p = current_p
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
        self.target_palm_pos = palm_plane.contact_target(plane).tolist()
        self.target_palm_rot = plane.rot
        self.target_palm_center = plane.center.tolist()
        self.target_palm_normal = plane.normal.tolist()

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
    def _report_reach(self, label, target):
        """IK が解けた姿勢で手先が目標からどれだけ離れているかを出す.

        IK が収束しなかったときは fallback 姿勢のまま指令が出るので、届いた
        つもりで空振りしていないかはここを見て判断する。
        """
        distance = float(np.linalg.norm(
            self.robot.larm_end_coords.worldpos()
            - np.asarray(target, dtype=np.float64)))
        if distance <= 0.02:
            rospy.loginfo('%s pose: %.1f mm from the target',
                          label, distance * 1000.0)
        else:
            rospy.logwarn('%s pose: %.1f mm from the target -- IK did not '
                          'reach it', label, distance * 1000.0)

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

            cx, cy, cz = self.target_palm_center

            # Torso up/down (lifter) to roughly match hand height. This is
            # just a seed pose -- the whole-body IK below (larm_whole_body)
            # is free to further adjust the lifter *and* the waist yaw/pitch
            # joints (waist_y_joint, waist_p_joint) to reach low targets,
            # e.g. a seated person's hand, without moving the base.
            lifter_amount = np.clip((1.0 - cz) * 1.5, 0.0, 1.0)
            try:
                self.robot.knee_joint.joint_angle(lifter_amount)
                if hasattr(self.robot, 'ankle_joint'):
                    self.robot.ankle_joint.joint_angle(-lifter_amount)
            except AttributeError:
                pass

            # Look at the palm being touched.
            neck_yaw = math.atan2(cy, cx)
            neck_pitch = math.atan2(cz - 1.2, math.hypot(cx, cy))
            self.robot.neck_y_joint.joint_angle(np.clip(neck_yaw, -1.5, 1.5))
            self.robot.neck_p_joint.joint_angle(np.clip(-neck_pitch, -0.3, 0.5))

            rospy.loginfo("Adjusting posture and gaze first...")
            self.ri.angle_vector(self.robot.angle_vector(), 2.0)
            self.ri.wait_interpolation()

            rospy.loginfo("Extending arm toward the palm...")
            self.robot.angle_vector(self.ri.angle_vector())

            target_coords = Coordinates(
                pos=self.target_palm_pos, rot=self.target_palm_rot)

            # Fallback posture (natural elbow position) used as the IK seed.
            self.robot.l_shoulder_p_joint.joint_angle(-0.4)
            self.robot.l_shoulder_r_joint.joint_angle(0.2)
            self.robot.l_shoulder_y_joint.joint_angle(0.5)
            self.robot.l_elbow_joint.joint_angle(-1.2)
            self.robot.l_wrist_y_joint.joint_angle(0.0)
            self.robot.l_wrist_p_joint.joint_angle(0.2)
            self.robot.l_wrist_r_joint.joint_angle(1.5)

            try:
                res = self.robot.larm_whole_body.inverse_kinematics(
                    target_coords, rotation_axis='yz')
                if res is False:
                    res = self.robot.larm_whole_body.inverse_kinematics(
                        target_coords, rotation_axis=False)
            except Exception as e:
                rospy.logwarn(f"IK failed: {e}. Using fallback posture.")

            self._report_reach('approach', self.target_palm_pos)
            self.ri.angle_vector(self.robot.angle_vector(), 2.0)
            self.ri.wait_interpolation()

            # Second, slower motion: press past the approach pose so the
            # hand actually sinks into the palm rather than stopping just
            # short of it.
            rospy.loginfo("Pressing into the palm...")
            self.robot.angle_vector(self.ri.angle_vector())

            center = np.array(self.target_palm_center)
            normal = np.array(self.target_palm_normal)
            embed_pos = (center - normal * EMBED_DEPTH).tolist()
            embed_coords = Coordinates(
                pos=embed_pos, rot=self.target_palm_rot)

            try:
                res = self.robot.larm_whole_body.inverse_kinematics(
                    embed_coords, rotation_axis='yz')
                if res is False:
                    res = self.robot.larm_whole_body.inverse_kinematics(
                        embed_coords, rotation_axis=False)
            except Exception as e:
                rospy.logwarn(f"IK failed: {e}. Staying at the approach pose.")

            self._report_reach('press', embed_pos)
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
