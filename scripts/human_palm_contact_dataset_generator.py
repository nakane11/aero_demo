#!/usr/bin/env python3
"""Generate a dataset of contact-attempt snapshots against random fake people.

Runs ``human_palm_contact_behavior.py``'s reach-and-touch sequence against
``~num_people`` (default 100) freshly-sampled ``~source:=fake`` people (new
body, standing position and offered-arm angles each round -- see
``fake_people_pose_estimator_ros.py``), exactly like
``human_palm_contact_behavior_loop.py`` does. At the moment each round
reaches ``DONE`` it captures a single snapshot of the robot at its final
pose (the "contact moment") and saves it as a PNG, then moves on to the
next person.

The snapshot is captured through ``skrobot.viewers.ViserViewer``: the robot
is drawn once into a ``viser`` scene (a websocket server, not a native GL
window -- no ``DISPLAY``/``Xvfb`` needed), and each round's image is pulled
via a connected browser client's ``ClientHandle.get_render()`` (the render
itself happens client-side in the browser's WebGL context; the server just
requests a frame and receives the pixels back over the websocket). This
means **a browser must be pointed at the printed viser URL and left open
for the whole run** -- ``__init__`` prints the URL and blocks
(``~client_wait_timeout``, default 120s) until it sees a connected client
before the first round starts.

Whether the round's IK actually reached the palm (``approach`` and ``press``
both within 2 cm of their targets, the same criterion
``human_palm_contact_behavior_loop.py`` uses to report "contact OK") decides
which of two subdirectories the image goes to:

  ``<output_dir>/success/``   IK reached both the approach and press pose
  ``<output_dir>/fail/``      IK missed either one (image still saved, at
                              whatever pose the fallback posture ended up in)

Each image is named ``person_<round>_seed<seed>.png``, where ``<seed>`` is
the ``~seed`` the round's ``FakeRosPeoplePoseEstimator`` auto-picked (see
``fake_people_pose_estimator_ros.py``) -- passing that same value back as
``~seed`` with ``~num_people:=1`` resamples the exact same person, so any
round can be reproduced on its own.

``<output_dir>/log.jsonl`` records one JSON line per round for offline
analysis: the seed, the IK targets (approach/press position + orientation),
how close IK actually got (``_report_reach``'s per-target distance and
reached flag), and the robot's resulting hand pose and full joint-angle
vector at the moment the snapshot was taken.

Camera framing
--------------
The camera is a side elevation (looking along the base frame's Y axis, i.e.
perpendicular to the direction the robot reaches in), not the
``sample_render.py``-style three-quarter view, because the two framing
constraints asked for are naturally each one image axis of a side view:

* vertical: fills the frame top-to-bottom with the taller of the robot's
  own extent (its head, ``head_end_coords``, down to its wheeled lifter
  base, ``base_link``) and the tracked person's extent (every visible
  landmark's Z range, i.e. roughly head-to-feet) -- so neither body gets
  clipped at the top/bottom edge regardless of which one happens to be
  taller or standing higher/lower (a bit of padding is added on top of
  that).
* horizontal: the frame is centered on the midpoint between the human (the
  last tracked person's Neck/Nose, i.e. where they stood) and the robot's
  reaching hand (``rarm_end_coords``/``larm_end_coords``, wherever IK left
  it, reached or not) -- but only *centered*, not fit; the horizontal span
  is whatever the vertical-fit distance happens to show.

Wrist -> fingertip arrows
--------------------------
Each snapshot also draws two arrows from the wrist toward the fingertips:
one for the human's estimated hand (``target_palm_rot``'s local +X axis,
i.e. ``palm_plane.fit_palm_plane``'s ``finger_dir``, the plane fit from the
tracked landmarks) and one for the robot's actual hand pose
(``r/l_eef_grasp_link``'s local +X axis, the same wrist -> fingertip axis --
see the frame-orientation comment in ``aero_demo.palm_plane.fit_palm_plane``),
so a mismatch between where the human's fingers pointed and where the
robot's hand ended up aiming is visible in the frame itself, not just in
the numeric report.

Required parameters
--------------------
``~source`` is forced to ``fake`` (this script only makes sense against the
fake estimator -- see ``human_palm_contact_behavior_loop.py``'s docstring
for why) and ``~viewer`` to ``none`` (drawing is this script's own offscreen
snapshot, not the interactive scene) regardless of what is passed on the
command line. ``~present_hand`` defaults to ``~hand_side`` (so the fake
person actually offers a hand for the robot to reach for) unless already
set.

Parameters
----------
All of ``human_palm_contact_behavior.py``'s parameters (``~hand_side``,
``~use_hand``, ``~min_score``, ``~base_frame``, ``~output_frame``, the fake
estimator's ``~distance_range`` / ``~present_*_deg_range`` / etc.), plus

``~num_people`` (int, default 100)       何人分生成するか。
``~output_dir`` (str, default ``~/palm_contact_dataset``)
                                          ``success/`` / ``fail/`` を作る
                                          出力先ディレクトリ。
``~image_width`` / ``~image_height`` (int, default 960 / 720)
``~open_browser`` (bool, default false)  viser の URL を自動でブラウザで
                                          開くか (``~viewer:=viser`` の他
                                          スクリプトと同じ意味)。閉じたまま
                                          でも URL はログに出るので手動で
                                          開いてもよい。
``~client_wait_timeout`` (float, default 120.0)
                                          起動時、最初のブラウザが繋がる
                                          まで待つ最大秒数。
``~client_reconnect_timeout`` (float, default 20.0)
                                          周回の途中でブラウザが切れた
                                          とき、再接続を待つ最大秒数
                                          (超えたらその周は失敗として
                                          スキップし次へ進む)。
``~render_timeout`` (float, default 20.0)
                                          1 枚の ``get_render()`` 応答を
                                          待つ最大秒数 (繋がったままだが
                                          応答が返らないクライアントで
                                          無限に待たないため)。
``~round_pause`` (float, default 0.2)    次の周回のサンプリングを始める前に
                                          置く待ち時間 [s]。
``~neck_settle_time`` (float, default 0.3)
                                          ``human_palm_contact_behavior``の
                                          ``NECK_SETTLE_TIME`` (既定 2.5s) を
                                          この値に短縮する。実機に指令を出す
                                          わけではないので、100 人分回すのに
                                          待つ理由が薄い。
"""

import json
import os
import sys
import threading
import traceback

import numpy as np
import rospy

from skrobot.coordinates import Coordinates

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import human_palm_contact_behavior as _hpcb
from human_palm_contact_behavior import HumanPalmContactBehavior

# Vertical field of view [deg] passed to ClientHandle.get_render() -- fixed
# here so the head/cart framing math in _camera_pose and the actual
# rendered frame agree on what "fits the frame" means.
CAMERA_YFOV_DEG = 55.0
# Extra headroom above the head / below the cart so neither is clipped
# right at the image edge [m].
VERTICAL_PAD = 0.15
# Safety margin on top of the exact tan(fov/2) fit distance.
DISTANCE_PAD = 1.08

# Wrist -> fingertip arrows: length is the tail-to-tip span (tail sits at
# the wrist/palm point), the rest are the shaft/head proportions of that
# span.
ARROW_LENGTH = 0.15
ARROW_SHAFT_RADIUS = 0.004
ARROW_HEAD_RADIUS = 0.011
ARROW_HEAD_LENGTH = 0.045
COLOR_HUMAN_FINGER = [0, 200, 255, 255]   # matches palm_plane_view.COLOR_FINGER
COLOR_ROBOT_FINGER = [255, 140, 0, 255]   # distinct from the ok/fail sphere's
                                           # green/red and the human arrow's
                                           # cyan


def _make_direction_arrow(base, direction, color):
    """A shaft+cone arrow of a fixed length, tail-to-tip along ``direction``.

    ``base`` is where the tail starts (the wrist/palm point); the
    arrowhead lands ``ARROW_LENGTH`` further along ``direction`` -- i.e.
    the arrow points *out of* ``base`` toward the fingertips, matching
    "wrist -> fingertip" rather than pointing into the palm from behind.

    skrobot has no ready-made directional-arrow primitive (only
    ``Cylinder``/``Cone``, both authored pointing along +Z -- see
    ``palm_plane_view.rotation_from_z``), so the shaft and head are built
    as separate trimeshes, translated into place along +Z, and merged
    into one ``Link`` the same way ``skrobot.model.primitives.Axis``
    composites its three axis cylinders.
    """
    import trimesh
    from skrobot.model import Link
    from aero_demo.palm_plane_view import rotation_from_z, set_color, unit

    d = unit(direction)
    if d is None:
        return None

    shaft_length = ARROW_LENGTH - ARROW_HEAD_LENGTH
    shaft = trimesh.creation.cylinder(
        radius=ARROW_SHAFT_RADIUS, height=shaft_length, sections=12)
    shaft.apply_translation([0.0, 0.0, shaft_length / 2.0])
    head = trimesh.creation.cone(
        radius=ARROW_HEAD_RADIUS, height=ARROW_HEAD_LENGTH, sections=12)
    head.apply_translation([0.0, 0.0, shaft_length])
    mesh = shaft + head

    tail = np.asarray(base, dtype=np.float64)
    link = Link(pos=tail, rot=rotation_from_z(d), visual_mesh=mesh)
    return set_color(link, color)


class _InstantRobotInterface(_hpcb._DrawingRobotInterface):
    """Jump straight to each commanded pose instead of animating toward it.

    The base class's ``_interpolate`` sleeps in small steps so an
    interactive viewer can show the motion; this script never draws in
    between, so that sleeping only slows down 100 rounds' worth of nodding/
    reaching/pressing for no visible benefit. IK still runs exactly the
    same (this only replaces the *display* interpolation that happens
    after IK has already moved ``self.robot``).
    """

    def _interpolate(self, goal, duration):
        self._actual = goal.copy()
        self.robot.angle_vector(self._actual)


class HumanPalmContactDatasetGenerator(HumanPalmContactBehavior):
    """DONE に着くたびオフスクリーンで 1 枚撮り、success/fail に振り分ける版."""

    def __init__(self):
        # Called with the exact same name/arguments the base class's own
        # __init__ uses, so that call becomes a harmless no-op -- this lets
        # us force ~source/~viewer/~present_hand (below) before the base
        # class reads them, without duplicating its whole __init__.
        rospy.init_node('human_palm_contact_behavior')

        self.num_people = int(rospy.get_param('~num_people', 100))
        self.output_dir = os.path.expanduser(
            rospy.get_param('~output_dir', '~/palm_contact_dataset'))
        self.round_pause = rospy.get_param('~round_pause', 0.2)
        self.img_w = int(rospy.get_param('~image_width', 960))
        self.img_h = int(rospy.get_param('~image_height', 720))

        # This script only makes sense against randomly-sampled fake
        # people (see human_palm_contact_behavior_loop.py's docstring for
        # why ~source:=real would not do what "100 random people" asks
        # for), and it draws its own offscreen snapshot instead of the
        # interactive scene -- so both are forced regardless of what was
        # passed on the command line.
        rospy.set_param('~source', 'fake')
        rospy.set_param('~viewer', 'none')
        hand_side = str(rospy.get_param('~hand_side', 'R')).upper()[:1]
        rospy.set_param('~hand_side', hand_side)
        if not str(rospy.get_param('~present_hand', '')).strip():
            rospy.set_param('~present_hand', hand_side)

        _hpcb.NECK_SETTLE_TIME = rospy.get_param('~neck_settle_time', 0.3)

        self.round_count = 0
        self.results = []
        self._latest_person = None
        # Set by _start_next_round from the fresh pose source's own
        # ~seed=-1 auto-pick (FakeRosPeoplePoseEstimator.seed) -- carried
        # into both the image filename and the log line so any single
        # round can be reproduced later via ~seed:=<that value>
        # ~num_people:=1.
        self.current_seed = None
        # super().__init__() starts the 0.1s control_loop timer on its own
        # background thread before returning here -- with the instant robot
        # interface a whole round can finish fast enough to race the rest
        # of this __init__ (building the offscreen renderer etc.), so
        # control_loop no-ops until this flips True at the end of __init__.
        self._ready = False

        super(HumanPalmContactDatasetGenerator, self).__init__()

        # Replace the (possibly display-interpolating) robot interface the
        # base class built with the instant one -- see
        # _InstantRobotInterface. self.scene is always None here (~viewer
        # forced to 'none'), so there is nothing for it to have been
        # drawing anyway.
        self.ri = _InstantRobotInterface(self.robot, None, None)

        for sub in ('success', 'fail'):
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)
        self.log_path = os.path.join(self.output_dir, 'log.jsonl')

        self._init_viser_renderer()
        rospy.on_shutdown(self._cleanup_renderer)

        rospy.loginfo(
            "Generating %d people (hand=%sHand robot_arm=%sarm) -> %s "
            "(success/ + fail/)", self.num_people, self.hand, self.arm,
            self.output_dir)
        self._ready = True

    # ------------------------------------------------------------------
    # viser-backed rendering
    # ------------------------------------------------------------------
    def _init_viser_renderer(self):
        """Open the viser server, add the robot, and block for a client.

        ``ViserViewer`` is a websocket server, not a native GL window, so
        this needs no ``DISPLAY``/``Xvfb``. What it does need is an actual
        browser connected: ``ClientHandle.get_render()`` (used in
        ``_save_snapshot``) renders in that browser's WebGL context and
        ships the pixels back over the socket, so nothing can be captured
        before one client has connected.
        """
        from skrobot.viewers import ViserViewer

        self._viewer = ViserViewer()
        self._viewer.add(self.robot)
        self._viewer.show(
            open_browser=rospy.get_param('~open_browser', False))

        timeout = rospy.get_param('~client_wait_timeout', 120.0)
        rospy.loginfo(
            'Waiting up to %.0fs for a browser to connect to the viser '
            'URL above (open it if it was not opened automatically) -- '
            'no snapshot can be captured until one does...', timeout)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while not rospy.is_shutdown() and not self._viewer._server.get_clients():
            if rospy.Time.now() > deadline:
                raise RuntimeError(
                    'No viser client connected within {:.0f}s. Open the '
                    'printed URL in a browser and re-run (or raise '
                    '~client_wait_timeout).'.format(timeout))
            rospy.sleep(0.5)
        if not rospy.is_shutdown():
            rospy.loginfo('Viser client connected, starting.')

    def _cleanup_renderer(self):
        viewer = getattr(self, '_viewer', None)
        if viewer is not None:
            viewer.close()

    def _get_client(self):
        """Return a connected client, tolerating a brief reconnect gap.

        A dropped tab (network blip, browser throttling a background tab,
        someone bumping the window) shouldn't sink the whole batch: wait
        up to ``~client_reconnect_timeout`` for a client to (re)appear
        before giving up this one round.
        """
        timeout = rospy.get_param('~client_reconnect_timeout', 20.0)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        clients = self._viewer._server.get_clients()
        while not clients and not rospy.is_shutdown():
            if rospy.Time.now() > deadline:
                raise RuntimeError(
                    'no viser client connected for {:.0f}s; keep the '
                    'browser tab open for the whole batch (or raise '
                    '~client_reconnect_timeout)'.format(timeout))
            rospy.sleep(0.5)
            clients = self._viewer._server.get_clients()
        # Always the most-recently-connected client, in case of a
        # reconnect (a stale ClientHandle from a dropped connection would
        # otherwise wait forever for a render response that never comes).
        return list(clients.values())[-1]

    def _camera_view(self, human_points, hand_pos):
        """Side-elevation (position, look_at, up) framed per the module
        docstring.

        Returned as position/look_at/up_direction -- the same three
        values ``ViserViewer._apply_camera_view`` assigns to
        ``client.camera`` -- rather than a raw quaternion, so viser's own
        ``CameraHandle._update_wxyz`` derives the orientation. Hand-rolling
        the quaternion here would have to independently match viser's
        camera-basis convention (OpenCV-style: look direction is +Z, up is
        -Y, right is +X -- not the OpenGL "camera looks down -Z" convention
        most other skrobot viewers use), which is exactly what went wrong
        the first time this was written directly as a wxyz.

        ``human_points`` is every visible landmark of the tracked person
        (not just one reference point): centering on the bounding-box
        middle of the whole skeleton plus the robot's hand, rather than
        just averaging two single points (the person's Neck and the
        hand), keeps the frame tight around where the actual content is
        instead of leaving one side full of blank space whenever the
        person's body happens to sit off to one side of their own Neck
        point (the skeleton has no rendered volume of its own to fill
        that space the way the robot's mesh does).
        """
        head_pos = self.robot.head_end_coords.worldpos()
        cart_pos = self.robot.base_link.worldpos()
        # The vertical fit has to cover both bodies' top-to-bottom extent,
        # not just the robot's -- a tracked person can stand taller than
        # the robot's head or have their feet below the cart, and either
        # would otherwise get clipped at the top/bottom edge.
        human_z = np.atleast_2d(human_points)[:, 2]
        z_lo = min(float(head_pos[2]), float(cart_pos[2]),
                   float(human_z.min())) - VERTICAL_PAD
        z_hi = max(float(head_pos[2]), float(cart_pos[2]),
                   float(human_z.max())) + VERTICAL_PAD

        # cart_pos anchors how far back the robot's own body reaches (the
        # arm's shoulder sits close to it) -- without it the center would
        # sit only between the human and the hand, and the robot's torso
        # behind the hand would consistently spill off one side of the
        # frame with nothing balancing it on the other.
        content = np.vstack([np.atleast_2d(human_points), [hand_pos],
                             [cart_pos]])
        content_lo = content.min(axis=0)
        content_hi = content.max(axis=0)

        center = np.array([
            (float(content_lo[0]) + float(content_hi[0])) / 2.0,
            (float(content_lo[1]) + float(content_hi[1])) / 2.0,
            (z_lo + z_hi) / 2.0,
        ])

        # Look along the base frame's Y axis (side view): image-vertical
        # tracks world Z (height), image-horizontal tracks world X (how
        # far the robot reaches toward the person) -- see module
        # docstring for why this axis choice matches both framing asks at
        # once instead of needing perspective-aware cropping.
        forward = np.array([0.0, -1.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        # z_lo/z_hi (both bodies' vertical extent, above) drive the
        # distance; human_points/hand_pos otherwise only steer `center`
        # (horizontal centering, not framing) -- see module docstring.
        fov_v = np.deg2rad(CAMERA_YFOV_DEG)
        half_v = (z_hi - z_lo) / 2.0
        distance = (half_v / np.tan(fov_v / 2.0)) * DISTANCE_PAD

        position = center - forward * distance
        return position, center, up

    def _save_snapshot(self, ok):
        import imageio.v3 as iio
        from skrobot.model.primitives import Axis, LineString, Sphere
        from aero_demo.palm_plane_view import (
            BASE_ORIGIN_AXIS_LENGTH, BASE_ORIGIN_AXIS_RADIUS, bone_color,
            dim_color)

        hand_coords = getattr(self.robot, '{}arm_end_coords'.format(self.arm))
        hand_pos = hand_coords.worldpos()
        human_points = None
        if self._latest_person is not None and self._latest_person.positions:
            human_points = self._latest_person.positions
        if not human_points:
            ref = self.target_palm_center or hand_pos
            human_points = [ref]

        markers = []

        # 台車 (base_link) の初期位置 = ワールド原点 (_start_next_round が
        # 毎周回 reset_pose() の直後に明示的に戻すので常にここ)。approach
        # の IK は use_base='planar' で台車を動かせるので、動いたかどうか
        # がこの大きめの座標軸とロボット自身の台車の位置を見比べればわかる。
        markers.append(Axis(axis_radius=BASE_ORIGIN_AXIS_RADIUS,
                            axis_length=BASE_ORIGIN_AXIS_LENGTH))

        target_color = [50, 220, 50, 255] if ok else [220, 50, 50, 255]
        target_sphere = Sphere(radius=0.02, color=target_color)
        target_sphere.newcoords(Coordinates(pos=hand_pos))
        markers.append(target_sphere)

        # Wrist -> fingertip arrows -- see the module docstring's "Wrist ->
        # fingertip arrows" section. Human: local +X of the plane fit from
        # the tracked landmarks. Robot: r/l_eef_grasp_link's local +X axis
        # (the same wrist -> fingertip axis, see
        # aero_demo.palm_plane.fit_palm_plane).
        if self.target_palm_center is not None \
                and self.target_palm_rot is not None:
            human_finger_dir = np.asarray(self.target_palm_rot)[:, 0]
            human_arrow = _make_direction_arrow(
                self.target_palm_center, human_finger_dir,
                COLOR_HUMAN_FINGER)
            if human_arrow is not None:
                markers.append(human_arrow)
        robot_finger_dir = hand_coords.worldrot().dot(np.array([1.0, 0.0, 0.0]))
        robot_arrow = _make_direction_arrow(
            hand_pos, robot_finger_dir, COLOR_ROBOT_FINGER)
        if robot_arrow is not None:
            markers.append(robot_arrow)

        # Same colored-per-body-part skeleton lines
        # human_palm_contact_behavior_loop.py's viewer draws (see
        # aero_demo.palm_plane_view.PalmPlaneScene.update_skeleton) --
        # bones dimmed the same way for landmarks that were tracked but
        # dropped (~source:=fake's ground truth for what the camera would
        # not have seen), instead of this script's own plain gray dots.
        if self._latest_person is not None:
            for bone in getattr(self._latest_person, 'bones', []):
                line = LineString(
                    np.asarray([bone.start_point, bone.end_point],
                              dtype=np.float64),
                    color=bone_color(bone.name))
                markers.append(line)
            for bone in getattr(self._latest_person, 'hidden_bones', []):
                line = LineString(
                    np.asarray([bone.start_point, bone.end_point],
                              dtype=np.float64),
                    color=dim_color(bone_color(bone.name)))
                markers.append(line)

        for m in markers:
            self._viewer.add(m)
        self._viewer.redraw()

        position, look_at, up = self._camera_view(human_points, hand_pos)
        client = self._get_client()
        try:
            # Set position/look_at/up_direction (not a hand-rolled wxyz --
            # see _camera_view) the same way ViserViewer.set_camera does,
            # so CameraHandle derives the orientation itself; get_render()
            # then reads that state back when wxyz/position are omitted.
            client.camera.position = position
            client.camera.look_at = look_at
            client.camera.up_direction = up
            # timeout bounds a client that stays listed in get_clients()
            # (e.g. a half-open connection the server hasn't reaped yet)
            # but never actually answers -- without it this can block
            # forever, taking the rest of the batch down with it.
            render_timeout = rospy.get_param('~render_timeout', 20.0)
            image = client.get_render(
                self.img_h, self.img_w,
                fov=np.deg2rad(CAMERA_YFOV_DEG), transport_format='png',
                timeout=render_timeout)
        finally:
            for m in markers:
                self._viewer.delete(m)

        sub = 'success' if ok else 'fail'
        seed_str = 'na' if self.current_seed is None else str(self.current_seed)
        path = os.path.join(
            self.output_dir, sub,
            'person_{:04d}_seed{}.png'.format(self.round_count, seed_str))
        iio.imwrite(path, image)
        return path

    def _log_round(self, ok, image_path):
        """Append one JSON line with everything needed to inspect/reproduce
        this round after the fact: the sampled ``~seed``, the IK targets
        (approach and press), how close IK actually got
        (``_report_reach``'s per-label distance/reached), and the robot's
        resulting hand pose and full joint-angle vector.

        Written with a plain ``open(..., 'a')`` per round (no long-lived
        file handle) so a crash partway through the batch still leaves
        every already-finished round's line on disk, matching how images
        are already written incrementally to ``success/``/``fail/``.
        """
        from skrobot.coordinates.math import matrix2quaternion

        hand_coords = getattr(self.robot, '{}arm_end_coords'.format(self.arm))
        hand_pos = hand_coords.worldpos()
        hand_quat = matrix2quaternion(hand_coords.worldrot())

        press_pos = None
        if self.target_palm_center is not None \
                and self.target_palm_normal is not None:
            press_pos = (np.array(self.target_palm_center)
                        - np.array(self.target_palm_normal) * _hpcb.EMBED_DEPTH
                        ).tolist()

        def _reach(label):
            report = self._last_report.get(label)
            if report is None:
                return None
            distance, reached = report
            return {'distance_m': distance, 'reached': bool(reached)}

        record = {
            'round': self.round_count,
            'seed': self.current_seed,
            'ok': ok,
            'hand_side': self.hand,
            'arm': self.arm,
            'image_path': image_path,
            'target': {
                'palm_center': self.target_palm_center,
                'palm_normal': self.target_palm_normal,
                'approach_pos': self.target_palm_pos,
                'approach_quat': matrix2quaternion(
                    self.target_palm_rot).tolist()
                    if self.target_palm_rot is not None else None,
                'press_pos': press_pos,
            },
            'ik_result': {
                'approach': _reach('approach'),
                'press': _reach('press'),
            },
            'robot': {
                'hand_pos': hand_pos.tolist(),
                'hand_quat': hand_quat.tolist(),
                'joint_names': [j.name for j in self.robot.joint_list],
                'joint_angle_vector':
                    np.asarray(self.robot.angle_vector()).tolist(),
            },
        }
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

    # ------------------------------------------------------------------
    # tracking the current person (for the horizontal framing center)
    # ------------------------------------------------------------------
    def _best_palm(self, result):
        plane, palm_points, person = super(
            HumanPalmContactDatasetGenerator, self)._best_palm(result)
        if person is not None:
            self._latest_person = person
        return plane, palm_points, person

    # ------------------------------------------------------------------
    # state machine: intercept the DONE -> (restart) transition, same
    # trick human_palm_contact_behavior_loop.py uses.
    # ------------------------------------------------------------------
    def control_loop(self, event):
        if not self._ready:
            # __init__ hasn't finished building the offscreen renderer yet
            # -- see the _ready comment in __init__.
            return

        if self.state == "RESTARTING":
            self._start_next_round()
            return

        super(HumanPalmContactDatasetGenerator, self).control_loop(event)

        if self.state == "DONE":
            # round_count is incremented exactly once here, regardless of
            # whether the round below succeeds or raises -- _finish_round
            # used to bump it at its own top, and a second bump in the
            # except branch double-counted every failed round (100 rounds
            # attempted but only e.g. 98 images total, one round short per
            # failure) toward ~num_people, ending the batch early.
            self.round_count += 1
            # rospy.Timer runs every tick on the same background thread and
            # that thread's whole future is over the moment one callback
            # raises uncaught -- no more rounds would ever run, silently.
            # A single bad snapshot (disk full, one weird pose, a viser
            # client hiccup, ...) shouldn't take the rest of the batch
            # down with it.
            try:
                self._finish_round()
            except Exception:
                rospy.logerr('round %d failed, skipping it:\n%s',
                             self.round_count, traceback.format_exc())
                self.results.append(False)
                self._last_report = {}

            if self.round_count >= self.num_people:
                n_ok = sum(1 for r in self.results if r)
                rospy.loginfo(
                    'Generated %d/%d images (%d contact OK) in %s. '
                    'Shutting down.', len(self.results), self.round_count,
                    n_ok, self.output_dir)
                rospy.signal_shutdown('dataset complete')
            self.state = "RESTARTING"

    def _finish_round(self):
        approach = self._last_report.get('approach')
        press = self._last_report.get('press')
        ok = bool(approach and approach[1] and press and press[1])

        path = self._save_snapshot(ok)
        self._log_round(ok, path)
        self.results.append(ok)
        n_ok = sum(1 for r in self.results if r)
        rospy.loginfo(
            'person %d/%d: %s -> %s (%d/%d OK so far)', self.round_count,
            self.num_people, 'contact OK' if ok else 'FAILED', path,
            n_ok, self.round_count)

        self._last_report = {}

    def _start_next_round(self):
        if self.round_pause > 0.0:
            rospy.sleep(self.round_pause)
        if rospy.is_shutdown():
            return

        # reset_pose() only resets joint angles -- the previous round's
        # approach IK (use_base='planar') can have translated/rotated
        # base_link itself, which reset_pose() leaves untouched, so
        # recenter it explicitly here too.
        self.robot.reset_pose()
        self.robot.base_link.newcoords(Coordinates())
        self.ri.frozen = False
        self.ri.angle_vector(self.robot.angle_vector(), 0.0)

        self.target_palm_pos = None
        self.target_palm_rot = None
        self.target_hand_rot = None
        self.target_palm_center = None
        self.target_palm_normal = None
        self.last_neck_cmd_time = rospy.Time.now()
        self._latest_person = None

        self.state = "WAITING"
        self.state_start_time = rospy.Time.now()
        self.source = _hpcb.create_pose_source(self.source_name)
        # FakeRosPeoplePoseEstimator picks (and logs) its own ~seed=-1
        # random seed in __init__ -- grab it here so this round's image
        # name / log line can point back to the exact person it sampled.
        self.current_seed = getattr(self.source, 'seed', None)
        self._source_stopped = False
        self.pose_thread = threading.Thread(target=self.pose_loop)
        self.pose_thread.daemon = True
        self.pose_thread.start()


if __name__ == '__main__':
    try:
        HumanPalmContactDatasetGenerator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
