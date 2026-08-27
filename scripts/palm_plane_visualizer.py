#!/usr/bin/env python3
"""Visualise the fitted human palm plane in a scikit-robot viewer.

A test harness for ``human_palm_contact_behavior.py``: it runs exactly the
same palm-plane estimation (``palm_plane.fit_palm_plane``) on the poses it
gets from a pose estimator instance, but it never opens a controller
interface, so the robot cannot move.

Use it to check, before letting the robot reach, that

* the plane sits flat on the human palm,
* the green normal arrow points back at the robot (not into the person),
* the blue/red approach and press targets land where you expect,
* (``~source: fake`` only) the parts of the body the camera did *not*
  capture -- drawn faintly, alongside a faint pyramid for the camera's
  own field of view when ``~filter_by_fov`` is set true -- look like what
  ``~present_hand`` / the standing position should actually produce.

``human_palm_contact_behavior.py`` draws the same scene (via
``aero_demo.palm_plane_view``) while actually reaching, so this node is only
needed when you want the estimation on its own, with nothing commanded.

Pose input
----------
トピックは購読しない。``~source`` で選んだ推定クラスのインスタンスを自分で
持ち、``wait_for_result()`` で 1 フレームずつ受け取る。

``~source: real`` (既定)
    ``people_pose_estimator_ros.RosPeoplePoseEstimator``
    … カメラの画像トピックを購読する本物の推定。
``~source: fake``
    ``fake_people_pose_estimator_ros.FakeRosPeoplePoseEstimator``
    … カメラ無しで偽の姿勢を生成する。カメラも bag も MediaPipe も無い
      環境で描画だけ確かめたいときに使う。

どちらのクラスも同じ ``EstimationResult`` を返し、関節点は
``result.frame_id`` (既定 base_link) 相対なので、この可視化ノードは TF を
引かない。推定クラスのパラメータ (``~hand/enable``, ``~output_frame``,
``~rate``, ``~seed`` ...) はこのノードのプライベート名前空間から読まれる。
手のランドマークが要るので ``~hand/enable`` は既定で true にしてある。

Visualisation
-------------
描くものは ``aero_demo.palm_plane_view.PalmPlaneScene`` を見ること
(RViz の MarkerArray と同じ内容 + 骨格 + ロボットモデル)。

Parameters
----------
``~source`` (str, default ``real``)      ``real`` か ``fake``
``~hand_side`` (str, default ``R``)      追跡する手、``R`` か ``L``
                                         (``~hand`` は推定クラスの
                                         ``~hand/enable`` と衝突するので使わない)
``~min_score`` (float, default 0.1)
``~rate_limit`` (float, default 0.0)     描画更新の最小間隔 [s] (0 = 毎フレーム)
``~draw_skeleton`` (bool, default true)  骨格を線で描くか。検出できな
                                         かった関節も (fake 推定のときだけ)
                                         色を薄くして併せて描く
``~draw_camera_frustum`` (bool, default true)  カメラの画角を表す薄い
                                         四角すいを optical_frame から
                                         伸ばして描くか。``~source: fake``
                                         かつ ``~filter_by_fov`` が false
                                         (既定) のときは、画角の外でも
                                         関節が落ちず画角を考慮していない
                                         ので、この設定によらず描かない
``~draw_robot`` (bool, default true)     Aero のモデルを base_link に置くか
``~use_hand`` (bool, default true)       手付きの URDF を読むか
``~viewer`` (str, default ``trimesh``)   ``trimesh`` (X のウィンドウ) か
                                         ``viser`` (ブラウザ表示)。X が
                                         pyglet のウィンドウを開けない環境
                                         (WSLg など) では ``viser`` を使う。
``~open_browser`` (bool, default false)  ``~viewer: viser`` のときブラウザを
                                         自動で開くか
``~viewer_width`` / ``~viewer_height`` (int, default 960 / 720)
                                         ``~viewer: trimesh`` のときだけ有効
"""

import os
import sys

import numpy as np
import rospy

from aero_demo import palm_plane
from aero_demo.palm_plane_view import PalmPlaneScene

# catkin_install_python が devel space に置くのは実体ではなく exec() で
# 中継する relay script なので、それを import してもクラスがモジュールの
# 名前空間に入らない (ImportError になる)。同じ scripts/ にある推定クラス
# は実ファイル側から import する。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def create_pose_source(name):
    """``~source`` で選んだ推定クラスのインスタンスを作る.

    ``human_palm_contact_behavior.py`` も同じ選び方をするので、そちらから
    import して使う。
    """
    name = str(name).lower()
    if name == 'fake':
        from fake_people_pose_estimator_ros import FakeRosPeoplePoseEstimator
        return FakeRosPeoplePoseEstimator()
    if name == 'real':
        from people_pose_estimator_ros import RosPeoplePoseEstimator
        return RosPeoplePoseEstimator()
    raise ValueError(
        "~source must be 'real' or 'fake', got '{}'".format(name))


def load_aero(use_hand=True, reset_pose=True):
    """Aero のモデルを読む (読めなければ None を返す).

    ``~use_hand`` が true だと ``aero_with_feetech_hand.urdf`` が要る。
    """
    try:
        from skrobot.models import Aero
        robot = Aero(use_hand=use_hand)
    except Exception as e:
        rospy.logwarn('cannot load the Aero model (%s); drawing without it', e)
        return None
    if reset_pose:
        robot.reset_pose()
    return robot


def open_scene(scene, viewer_name):
    """scene を表示し、描けない状態なら理由を出す."""
    if scene.open(open_browser=rospy.get_param('~open_browser', False)):
        return True
    rospy.logfatal(
        'the trimesh viewer window did not open within 10 s, so nothing can '
        'be drawn (this happens when X cannot map a pyglet window, e.g. on '
        'WSLg). Re-run with _viewer:=viser to draw in a browser instead.')
    return False


class PalmPlaneVisualizer(object):
    def __init__(self):
        # 名前は ~hand ではなく ~hand_side。推定クラスが読む ~hand/enable を
        # 立てると param server 上の ~hand が dict になり、両者が衝突する。
        self.hand = str(rospy.get_param('~hand_side', 'R')).upper()[:1]
        self.min_score = rospy.get_param('~min_score', 0.1)
        self.rate_limit = rospy.get_param('~rate_limit', 0.0)

        # 手のひら平面には手のランドマークが要る。推定クラスは同じ private
        # 名前空間からパラメータを読むので、既定値をここで立てておく。
        rospy.set_param('~hand/enable',
                        bool(rospy.get_param('~hand/enable', True)))

        self.source_name = str(rospy.get_param('~source', 'real')).lower()
        self.source = create_pose_source(self.source_name)

        self.last_draw = rospy.Time(0)
        self.stats = {'frames': 0, 'fitted': 0}

        self.viewer_name = str(rospy.get_param('~viewer', 'trimesh')).lower()
        draw_camera = rospy.get_param('~draw_camera_frustum', True)
        if self.source_name == 'fake' \
                and not rospy.get_param('~filter_by_fov', False):
            # fake の既定では画角の外でも関節を落とさない (~filter_by_fov
            # 参照) = 画角を考慮していないので、四角すいを描いても意味が
            # ない。
            draw_camera = False
        self.scene = PalmPlaneScene(
            viewer=self.viewer_name,
            resolution=(rospy.get_param('~viewer_width', 960),
                        rospy.get_param('~viewer_height', 720)),
            draw_skeleton=rospy.get_param('~draw_skeleton', True),
            draw_camera=draw_camera)
        if rospy.get_param('~draw_robot', True):
            robot = load_aero(rospy.get_param('~use_hand', True))
            if robot is not None:
                self.scene.add_robot(robot)
        open_scene(self.scene, self.viewer_name)

        rospy.loginfo('palm_plane_visualizer: source=%s hand=%sHand '
                      '(visualisation only, the robot is never commanded)',
                      self.source_name, self.hand)
        rospy.Timer(rospy.Duration(5.0), self.report)

    def report(self, _event):
        n, k = self.stats['frames'], self.stats['fitted']
        if n == 0:
            rospy.logwarn_throttle(10.0, 'no pose result yet from the %s '
                                   'estimator', self.source_name)
            return
        rospy.loginfo('palm plane fitted in %d/%d frames (%.0f%%)',
                      k, n, 100.0 * k / n)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def spin(self):
        while not rospy.is_shutdown():
            result = self.source.wait_for_result(timeout=1.0)
            if result is None:
                rospy.logwarn_throttle(
                    5.0, 'no pose result from the %s estimator',
                    self.source_name)
                continue

            now = rospy.Time.now()
            if self.rate_limit > 0.0 and \
                    (now - self.last_draw).to_sec() < self.rate_limit:
                continue
            self.last_draw = now

            self._draw(result)
            self.scene.redraw()

    def _draw(self, result):
        if not self.scene.update_skeleton(result.people):
            rospy.logwarn('cannot draw the skeleton; disabling it')
        self.scene.update_camera(
            result.camera_intrinsics, result.camera_width,
            result.camera_height, result.camera_pose)

        # 法線は観測者 (= result.frame_id の原点、既定では base_link の
        # ロボット自身) の方を向かせる。
        viewpoint = np.zeros(3)
        for person in result.people:
            self.stats['frames'] += 1
            points = palm_plane.collect_palm_points(
                person, hand=self.hand, min_score=self.min_score)
            plane = palm_plane.fit_palm_plane(points, viewpoint=viewpoint)
            if plane is None:
                rospy.loginfo_throttle(
                    2.0, 'palm plane not fitted: only %d of %s landmarks '
                    '(need %d non-collinear)', len(points),
                    list(palm_plane.PLANE_LANDMARKS),
                    palm_plane.MIN_PALM_POINTS)
                continue

            self.stats['fitted'] += 1
            self.scene.update_plane(plane, points)
            rospy.loginfo_throttle(
                1.0, 'palm plane in %s: used=%s rms=%.1fmm center=(%.3f, '
                '%.3f, %.3f) normal=(%.2f, %.2f, %.2f)',
                result.frame_id, plane.used, plane.rms * 1000.0,
                plane.center[0], plane.center[1], plane.center[2],
                plane.normal[0], plane.normal[1], plane.normal[2])
            # Only the first person with a usable palm is drawn.
            return

        # 誰の手のひらも取れなかったフレームでは古い描画を消す
        self.scene.hide_plane()


if __name__ == '__main__':
    rospy.init_node('palm_plane_visualizer')
    try:
        visualizer = PalmPlaneVisualizer()
    except ValueError as e:
        rospy.logfatal('%s', e)
        sys.exit(1)
    visualizer.spin()
