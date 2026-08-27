#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""カメラ入力を持たず、偽の人物姿勢を生成して返すクラス.

people_pose_estimator_ros.RosPeoplePoseEstimator の入力なし版。
向こうが「カラー画像 + 深度画像 + CameraInfo を購読して推定する」のに対し、
こちらは何も購読せず ``~rate`` [Hz] で偽の姿勢を生成する。カメラも bag も
MediaPipe も要らないので、下流ノード (human_tracker, palm_plane,
gaze/following behaviors ...) を人が居ない状態でテストできる。

publish は一切行わない。結果は本物と同じ ``EstimationResult`` で

  * ``wait_for_result(timeout)`` … 新しい結果が来るまで待って受け取る
  * ``last_result`` … 最後の結果をその場で参照する

で受け取る。3 次元の関節点は既定で ``~output_frame`` (既定 base_link) 相対
で返す。実際にどの座標系かは ``EstimationResult.frame_id`` を見ること。
偽の姿勢は元々ロボット座標系 (x 前, y 左, z 上) で作っているので、本物と
違って TF は引かず、``~camera_height`` / ``~camera_forward`` /
``~camera_pitch_deg`` から決まる既知の変換をそのまま使う。既定値は Aero の
URDF と ``launch/decompress.launch`` の head_link -> camera_link から
計算した実機の値 (首を人に向けた neck_p = 31 deg のときの姿勢)。

Assumptions
-----------
* Exactly one person, standing in front of the robot, facing the camera,
  with both arms hanging down at the sides.  With ``~present_hand`` set to
  ``R`` or ``L`` that arm is instead held out towards the robot with the
  fingertips pointing down and the index-finger side of the hand facing
  the robot (like an offered handshake), which is the posture
  ``palm_plane_visualizer.py`` / ``human_palm_contact_behavior.py``
  are meant to be tested against.
* The person stands within arm's reach (``~distance_range``, default
  0.75-0.95 m), so the offered palm can be touched without driving the
  base.  At that distance the camera only sees the upper body: the legs,
  and often the hips, fall below the bottom of the image.
* With ``~hand/enable`` (default true) the 21 MediaPipe hand landmarks of
  both hands are generated as ``RHand0``..``RHand20`` / ``LHand*``, so
  ``palm_plane.py`` can fit a plane to the wrist + knuckle landmarks.
* The person only moves slightly (postural sway, breathing, small yaw
  oscillation, micro arm swing).
* Body proportions, standing position, motion parameters and (when
  ``~present_hand`` is set) the shoulder/elbow/wrist angles of the offered
  arm are randomized once at start-up so that every run looks like a
  different person offering their hand differently (``~seed`` で固定できる。
  角度の範囲は ``~present_shoulder_elevation_deg_range`` などで調整できる)。
* Not every keypoint is visible from a camera.  Two independent dropout
  layers reproduce this:
    1. MediaPipe visibility: a per-joint, slowly varying score.  Joints
       below ``~min_visibility`` are dropped from every result (as in the
       real estimator).  Joints projected outside the image are *not*
       dropped for this reason by default (``~filter_by_fov``, default
       False) -- staying in frame isn't the thing under test right now;
       set it True to also drop those, matching the real estimator.
    2. Depth holes: joints that have no valid depth are additionally
       dropped from ``EstimationResult.people`` only, exactly like the real
       estimator does when the depth image has no return for that pixel.
  On top of that, the whole detection (and each hand) is intermittently
  lost for a few frames, so consumers see results with no person in them.
  Whatever gets dropped by any of the above still keeps its ground-truth
  3D position in ``Person3D.hidden_positions`` / ``hidden_bones`` (see
  ``aero_demo.palm_plane_view.PalmPlaneScene``, which draws it faintly).
"""

import math
import random
import threading

import numpy as np

import rospy

# EstimationResult は本物 (people_pose_estimator_ros.py) と共有する
from aero_demo.people_pose_types import (Bone, CameraIntrinsics,
                                         EstimationResult, Person3D)

try:
    import cv2
    HAS_CV = True
except ImportError:  # EstimationResult.image is optional
    HAS_CV = False


# Hand landmark model, MediaPipe order (0 wrist, 1-4 thumb, 5-8 index,
# 9-12 middle, 13-16 ring, 17-20 pinky), in units of hand length.
# Axes: u = wrist -> finger tips, v = thumb side, n = palm facing direction.
# The n component encodes the slight curl of a relaxed hanging hand.
HAND_LOCAL = np.array([
    [0.00,  0.00, 0.00],   # 0  wrist
    [0.11,  0.13, 0.02],   # 1  thumb CMC
    [0.25,  0.26, 0.05],   # 2  thumb MCP
    [0.36,  0.34, 0.08],   # 3  thumb IP
    [0.45,  0.40, 0.10],   # 4  thumb tip
    [0.47,  0.17, 0.00],   # 5  index MCP
    [0.66,  0.18, 0.04],   # 6  index PIP
    [0.78,  0.18, 0.10],   # 7  index DIP
    [0.88,  0.18, 0.15],   # 8  index tip
    [0.48,  0.05, 0.00],   # 9  middle MCP
    [0.69,  0.05, 0.04],   # 10 middle PIP
    [0.82,  0.05, 0.11],   # 11 middle DIP
    [0.93,  0.05, 0.17],   # 12 middle tip
    [0.46, -0.08, 0.00],   # 13 ring MCP
    [0.65, -0.09, 0.04],   # 14 ring PIP
    [0.77, -0.10, 0.11],   # 15 ring DIP
    [0.87, -0.10, 0.16],   # 16 ring tip
    [0.43, -0.21, 0.00],   # 17 pinky MCP
    [0.57, -0.23, 0.03],   # 18 pinky PIP
    [0.67, -0.24, 0.08],   # 19 pinky DIP
    [0.74, -0.25, 0.12],   # 20 pinky tip
])

# Probability that a hand landmark has no valid depth.  Roughly reproduces
# the per-landmark rates measured on real data (see palm_plane.py header:
# wrist / middle MCP / ring MCP ~100%, pinky MCP 97%, thumb CMC 88%,
# index MCP 79%).
HAND_DEPTH_HOLE = {0: 0.00, 9: 0.00, 13: 0.00, 17: 0.03,
                   1: 0.12, 5: 0.21, 4: 0.18, 8: 0.18, 12: 0.16, 16: 0.16,
                   20: 0.18}
HAND_DEPTH_HOLE_DEFAULT = 0.08

# Nominal MediaPipe visibility of each body joint for a person standing
# upright in front of the camera.  Legs are frequently out of the image or
# occluded by furniture, wrists are noisy because they are thin and moving.
BASE_VISIBILITY = {
    "Nose": 0.96, "Neck": 0.97,
    "RShoulder": 0.95, "LShoulder": 0.95,
    "RElbow": 0.86, "LElbow": 0.86,
    "RWrist": 0.76, "LWrist": 0.76,
    "RHip": 0.82, "LHip": 0.82,
    "RKnee": 0.60, "LKnee": 0.60,
    "RAnkle": 0.45, "LAnkle": 0.45,
    "REye": 0.90, "LEye": 0.90,
    "REar": 0.72, "LEar": 0.72,
}


class _Osc(object):
    """Sine oscillator with random amplitude / frequency / phase."""

    def __init__(self, rng, amp, freq_range=(0.08, 0.35)):
        self.amp = amp * rng.uniform(0.5, 1.0)
        self.freq = rng.uniform(*freq_range)
        self.phase = rng.uniform(0.0, 2.0 * math.pi)

    def __call__(self, t):
        return self.amp * math.sin(2.0 * math.pi * self.freq * t + self.phase)


class FakeRosPeoplePoseEstimator(object):
    """入力なしで人物の 3 次元姿勢を生成する (publish なし).

    購読するトピックは無い。``~rate`` [Hz] のタイマで結果を更新する。

    Examples
    --------
    >>> rospy.init_node('fake_people_pose_estimator')
    >>> node = FakeRosPeoplePoseEstimator()
    >>> while not rospy.is_shutdown():
    ...     result = node.wait_for_result(timeout=1.0)
    ...     if result is None:
    ...         continue
    ...     for person in result.people:
    ...         # person.positions は result.frame_id (既定 base_link) 相対
    ...         rospy.loginfo('%s', person.position_of('Neck'))
    """

    # Same definitions as people_pose_estimator.PeoplePoseEstimator
    limb_sequence = [[2, 1], [1, 16], [1, 15], [6, 18], [3, 17],
                     [2, 3], [2, 6], [3, 4], [4, 5], [6, 7],
                     [7, 8], [2, 9], [9, 10], [10, 11], [2, 12],
                     [12, 13], [13, 14], [15, 17], [16, 18]]

    index2limbname = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                      "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
                      "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                      "LEye", "REar", "LEar", "Bkg"]

    index2handname = ["RHand{}".format(i) for i in range(21)] + \
                     ["LHand{}".format(i) for i in range(21)]

    hand_sequence = [[0, 1],   [1, 2],   [2, 3],   [3, 4],
                     [0, 5],   [5, 6],   [6, 7],   [7, 8],
                     [0, 9],   [9, 10],  [10, 11], [11, 12],
                     [0, 13],  [13, 14], [14, 15], [15, 16],
                     [0, 17],  [17, 18], [18, 19], [19, 20]]

    def __init__(self, start=True):
        """
        Parameters
        ----------
        start : bool
            False なら生成を開始しない。後から ``start()`` を呼ぶ。
        """
        self.rate = rospy.get_param('~rate', 15.0)
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        # 仮想カメラの frame_id (EstimationResult.camera_frame_id)
        self.camera_frame_id = rospy.get_param(
            '~frame_id', 'camera_color_optical_frame')
        # 結果の関節点をこのフレームで返す。空文字ならカメラ座標系のまま。
        self.output_frame = rospy.get_param('~output_frame', self.base_frame)

        self.use_hand = rospy.get_param('~hand/enable', True)
        # '' : both arms hanging down, 'R'/'L' : that hand held out towards
        # the robot, fingertips down and the index-finger side facing the
        # robot (handshake-like), which is what
        # palm_plane_visualizer.py / human_palm_contact_behavior.py expect.
        self.present_hand = str(
            rospy.get_param('~present_hand', '')).upper()[:1]
        # ~present_hand の腕の構え方 (肩の挙上/開き、肘の曲げ、手首の曲げ/
        # ひねり) を deg で一様分布から引く範囲。個体ごとに 1 回だけ
        # _sample_person で引く。既定値は「肩を斜め前に上げ、肘を軽く
        # 伸ばし、手首で微調整する」握手のような構えのまわりに収まる範囲。
        # ~present_wrist_roll_deg_range は掌の法線 (どちらを向いて差し出す
        # か) を大きく変える -- 掌を正面から横向きまで広く振って、フィット
        # した法線に沿って近づく human_palm_contact_behavior 側が実際に
        # どこまで追従できるかを確かめるためのもの。
        self.present_shoulder_elevation_range = rospy.get_param(
            '~present_shoulder_elevation_deg_range', [0.0, 85.0])
        self.present_shoulder_azimuth_range = rospy.get_param(
            '~present_shoulder_azimuth_deg_range', [-20.0, 20.0])
        self.present_elbow_flex_range = rospy.get_param(
            '~present_elbow_flex_deg_range', [-10.0, 30.0])
        self.present_wrist_pitch_range = rospy.get_param(
            '~present_wrist_pitch_deg_range', [-30.0, 10.0])
        self.present_wrist_roll_range = rospy.get_param(
            '~present_wrist_roll_deg_range', [-90.0, 90.0])
        # 手のランドマークを Person3D.bones にも繋ぐ ("RHand0->RHand1" など)。
        # 本物は体のボーンしか返さないので、可視化用の拡張。
        self.include_hand_bones = rospy.get_param('~include_hand_bones', True)
        self.min_visibility = rospy.get_param('~min_visibility', 0.5)
        self.min_joints = rospy.get_param('~min_joints', 6)
        # 画角の外に出た関節を見えないものとして落とすか。今のところ画角に
        # 収まるかどうかは検証したいポイントではないので既定 False (落とさ
        # ない) -- 画角外でも ~min_visibility 相応の可視性さえあれば
        # people に入る。True にすると本物同様、画角の外は無条件で見えない
        # 扱いになる (viewer の四角すいと実際に落ちる関節を対応させたい
        # ときなど)。
        self.filter_by_fov = rospy.get_param('~filter_by_fov', False)

        # virtual camera.  Aero's camera is an Orbbec Femto Bolt (see
        # https://github.com/nakane11/OrbbecSDK_ROS1/tree/aero,
        # launch/femto_bolt.launch); its color sensor's FOV is
        # H 80 deg x V 51 deg (Orbbec's published spec, e.g. 80x51 deg at
        # its native 3840x2160) with square pixels, so fx == fy and the
        # ratio fx/width (== fy/height, up to the H/V rounding) holds at
        # any resolution.  Modelled here at 1280x720 (16:9, one of the
        # sensor's supported color resolutions):
        #   fx = fy = (1280/2) / tan(80 deg / 2) = 762.7
        # (720x... 51 deg backs out to within ~1 deg of this, consistent
        # with both being rounded from the same square-pixel fx).
        self.width = int(rospy.get_param('~image_width', 1280))
        self.height = int(rospy.get_param('~image_height', 720))
        self.intrinsics = CameraIntrinsics(
            fx=rospy.get_param('~fx', 762.7),
            fy=rospy.get_param('~fy', 762.7),
            cx=rospy.get_param('~cx', self.width / 2.0),
            cy=rospy.get_param('~cy', self.height / 2.0))
        # 仮想カメラの置き方。カメラは head_link に付いているので
        # (launch/decompress.launch の head_link -> camera_link static
        # transform) base_link の真上ではなく、高さも向きも neck_p_joint で
        # 変わる。Aero の URDF から計算した値は
        #   neck_p =  0 deg : 高さ 1.685 m / 前方 0.066 m / 光軸 -1.42 deg
        #   neck_p = 25 deg : 高さ 1.635 m / 前方 0.156 m / 光軸 23.58 deg
        #   neck_p = 31 deg : 高さ 1.618 m / 前方 0.174 m / 光軸 29.58 deg
        #   neck_p = 40 deg : 高さ 1.589 m / 前方 0.197 m / 光軸 38.58 deg
        # (光軸の下向き角はいつも neck_p - 1.42 deg)。
        #
        # 既定は neck_p = 31 deg。reset_pose の 25 deg より少し下を向いた姿勢に
        # してあるのは、~distance_range まで人が近づくと人体が垂直画角
        # (pitch +- 21.3 deg) に収まりきらず、首の角度で何が見えるかが大きく
        # 変わるため。300 フレーム x 5 seed の実測で
        #   neck_p = 28 deg : 人物 5/5 seed, 体の関節 6.8/12, 手のひら 38 %
        #   neck_p = 31 deg : 人物 5/5 seed, 体の関節 7.7/12, 手のひら 35 %
        #   neck_p = 34 deg : 人物 4/5 seed, 体の関節 8.4/12, 手のひら 38 %
        #   neck_p = 40 deg : 人物 3/5 seed, 体の関節 9.2/12, 手のひら 33 %
        # と、下を向くほど骨格は多く見えるが、首と鼻が画角から外れて人物ごと
        # 検出されない seed が出る (推定側が Neck か Nose を要求するため)。
        # 全 seed で人物が見えるのは 28 / 31 deg で、骨格がより見えるのは 31。
        # 実機も neck_tracker が首を人に向けるので、人を見ている姿勢のほうが
        # 実態に近い。実推定はこの変換を TF から引くので、これらは fake 専用。
        # 注意: 上の 300 フレーム x 5 seed の実測値は、旧デフォルトの仮想カメラ
        # (640x480, fx=fy=615, 水平画角 55 deg 相当) かつ画角の外を無条件で
        # 見えない扱いにしていた頃のもの。今は画角を Femto Bolt 実機に合わせて
        # H 80 / V 51 deg に広げ、かつ ~filter_by_fov の既定を False にして
        # 画角の外という理由だけでは関節を落とさなくした (上のコメント参照)
        # ので、実際にはもっと視野に収まりやすくなっているはず -- 再計測
        # するまでは目安として読むこと。
        self.camera_height = rospy.get_param('~camera_height', 1.618)
        self.camera_forward = rospy.get_param('~camera_forward', 0.174)
        self.camera_pitch = math.radians(
            rospy.get_param('~camera_pitch_deg', 29.58))
        # EstimationResult.image に骨格を描いた画像を入れる (入力画像が無いので
        # 黒画像に描く)。本物の image に相当するもの。
        self.draw_image = rospy.get_param('~draw_image', True) and HAS_CV

        # Where the person stands (metres, robot frame: x forward, y left).
        # Close enough that Aero can touch the offered palm without driving
        # the base: the arm reaches a palm at x = 0.30..0.60 m, and at these
        # distances the fitted palm centre lands at x ~ 0.44..0.60 m.  The
        # lateral spread also used to matter for staying inside the virtual
        # camera's 80 deg horizontal field of view (Femto Bolt spec, see
        # ~fx above), but with ~filter_by_fov defaulting to False that no
        # longer drops hand landmarks by itself.  Fix ~seed when a
        # repeatable pose is needed.
        self.distance_range = rospy.get_param('~distance_range', [0.80, 1.00])
        self.lateral_range = rospy.get_param('~lateral_range', [-0.15, 0.15])
        self.height_range = rospy.get_param('~height_range', [1.55, 1.85])

        # dropout / noise
        self.joint_noise = rospy.get_param('~joint_noise', 0.004)
        self.depth_hole_prob = rospy.get_param('~depth_hole_prob', 0.03)
        self.person_lost_prob = rospy.get_param('~person_lost_prob', 0.01)
        self.person_recover_prob = rospy.get_param('~person_recover_prob', 0.5)
        self.hand_lost_prob = rospy.get_param('~hand_lost_prob', 0.05)
        self.hand_recover_prob = rospy.get_param('~hand_recover_prob', 0.3)

        seed = rospy.get_param('~seed', -1)
        if seed < 0:
            seed = random.randrange(1 << 30)
        self.seed = seed
        rospy.loginfo('fake_people_pose_estimator: seed=%d' % seed)
        self.rng = random.Random(seed)
        self.nprng = np.random.RandomState(seed & 0xffffffff)

        self._sample_person()

        self.person_detected = True
        self.hand_detected = {'R': True, 'L': True}

        self.timer = None
        self.start_time = rospy.Time.now()
        self._condition = threading.Condition()
        self._last_result = None

        rospy.on_shutdown(self.clean_up)

        if start:
            self.start()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self):
        """``~rate`` [Hz] での生成を開始する (本物の subscribe に相当)."""
        if self.timer is not None:
            return
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate), self._cb)

    def stop(self):
        """生成を止める (本物の unsubscribe に相当)."""
        if self.timer is not None:
            self.timer.shutdown()
            self.timer = None

    def clean_up(self):
        self.stop()
        with self._condition:
            # shutdown 時に wait_for_result を待たせ続けない
            self._condition.notify_all()

    @property
    def last_result(self):
        """最後に生成した ``EstimationResult`` (未生成なら None)."""
        with self._condition:
            return self._last_result

    def wait_for_result(self, timeout=None):
        """新しい結果が来るまで待ち、それを返す.

        呼び出し時点で保持している結果は無視するので、同じフレームが 2 度
        返ることはない。timeout [s] 以内に来なければ None を返す。
        timeout=None なら来るまで待つ (shutdown 時も None が返る)。
        """
        with self._condition:
            current = self._last_result
            if self._condition.wait_for(
                    lambda: self._last_result is not current, timeout):
                return self._last_result
            return None

    # ------------------------------------------------------------------
    # timer callback (本物の画像コールバックに相当)
    # ------------------------------------------------------------------
    def _cb(self, event):
        now = rospy.Time.now()
        t = (now - self.start_time).to_sec()

        people_joint_positions = self._generate(t)
        people = self._to_people_3d(people_joint_positions)

        if self.output_frame and self.output_frame != self.camera_frame_id:
            for person in people:
                self._apply_transform(person, self.base_from_camera)
            output_frame_id = self.output_frame
            camera_pose = self.base_from_camera
        else:
            output_frame_id = self.camera_frame_id
            camera_pose = np.eye(4)
        # 骨は変換後の点から作るので出力座標系になる
        for person in people:
            person.bones, person.hidden_bones = self._create_bones(person)

        image = None
        if self.draw_image:
            image = self.draw_joints(self.create_canvas(),
                                     people_joint_positions)

        self._set_result(EstimationResult(
            stamp=now,
            frame_id=output_frame_id,
            camera_frame_id=self.camera_frame_id,
            image=image,
            joint_positions=people_joint_positions,
            people=people,
            camera_intrinsics=self.intrinsics,
            camera_width=self.width,
            camera_height=self.height,
            camera_pose=camera_pose))

    def _set_result(self, result):
        """結果を保持し、``wait_for_result`` の待ち手を起こす."""
        with self._condition:
            self._last_result = result
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # random person / motion model
    # ------------------------------------------------------------------
    def _sample_person(self):
        rng = self.rng
        h = rng.uniform(*self.height_range)

        # anthropometric ratios with a bit of individual variation
        def r(ratio, spread=0.06):
            return ratio * rng.uniform(1.0 - spread, 1.0 + spread)

        self.body = dict(
            height=h,
            h_shoulder=h * r(0.818, 0.02),
            h_hip=h * r(0.530, 0.03),
            h_knee=h * r(0.285, 0.03),
            h_ankle=h * r(0.039, 0.10),
            h_nose=h * r(0.936, 0.01),
            h_eye=h * r(0.945, 0.01),
            h_ear=h * r(0.938, 0.01),
            shoulder_width=h * r(0.245, 0.08),
            hip_width=h * r(0.185, 0.08),
            upper_arm=h * r(0.186, 0.06),
            forearm=h * r(0.146, 0.06),
            hand_length=h * r(0.108, 0.06),
            head_depth=h * r(0.052, 0.08),
            eye_gap=h * r(0.033, 0.08),
            ear_gap=h * r(0.075, 0.08),
            # arms are not perfectly vertical: elbows/wrists hang slightly
            # away from and in front of the trunk
            elbow_out=rng.uniform(0.00, 0.035),
            wrist_out=rng.uniform(-0.02, 0.03),
            elbow_fwd=rng.uniform(0.01, 0.05),
            wrist_fwd=rng.uniform(0.04, 0.11),
        )

        self.stand = dict(
            x=rng.uniform(*self.distance_range),
            y=rng.uniform(*self.lateral_range),
            yaw=math.radians(rng.uniform(-15.0, 15.0)),  # 0 = facing camera
        )

        self.motion = dict(
            sway_x=_Osc(rng, 0.025),
            sway_y=_Osc(rng, 0.030),
            bob=_Osc(rng, 0.012),
            yaw=_Osc(rng, math.radians(5.0)),
            breath=_Osc(rng, 0.006, (0.20, 0.35)),
            arm_r=_Osc(rng, 0.025),
            arm_l=_Osc(rng, 0.025),
            head_yaw=_Osc(rng, math.radians(7.0)),
            head_pitch=_Osc(rng, 0.015),
        )

        # ~present_hand の腕の構え (shoulder/elbow/wrist の角度, deg)。
        # _body_positions がこれを使って毎フレームの姿勢を作る。人が変わる
        # たびに 1 回だけ引き直すので、その人が差し出す構え自体は動作中
        # 一定 (揺れ・呼吸などの微小な動きだけが乗る)。consumer 側
        # (human_palm_contact_behavior_loop.py) がこの値を viewer に出す。
        self.presented_arm_angles = None
        if self.present_hand in ('R', 'L'):
            self.presented_arm_angles = dict(
                shoulder_elevation_deg=rng.uniform(
                    *self.present_shoulder_elevation_range),
                shoulder_azimuth_deg=rng.uniform(
                    *self.present_shoulder_azimuth_range),
                elbow_flex_deg=rng.uniform(*self.present_elbow_flex_range),
                wrist_pitch_deg=rng.uniform(*self.present_wrist_pitch_range),
                wrist_roll_deg=rng.uniform(*self.present_wrist_roll_range),
            )

        # per-joint visibility bias (individual differences, camera setup,
        # clothing ...) and a slow flicker
        self.vis_bias = {}
        self.vis_osc = {}
        for name, base in BASE_VISIBILITY.items():
            self.vis_bias[name] = min(1.0, base * rng.uniform(0.75, 1.15))
            self.vis_osc[name] = _Osc(rng, 0.13, (0.05, 0.5))
        if self.present_hand in ('R', 'L'):
            # a hand held out towards the camera is tracked much better
            # than a wrist hanging beside the body
            for joint in ('Elbow', 'Wrist'):
                name = self.present_hand + joint
                self.vis_bias[name] = min(1.0, 0.95 * rng.uniform(0.97, 1.03))

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _body_positions(self, t):
        """Return {limb_name: xyz} in the robot frame (x fwd, y left, z up).

        Also returns the two hand coordinate frames so that hand landmarks
        can be attached to the wrists.
        """
        b = self.body
        m = self.motion
        s = self.stand

        breath = m['breath'](t)
        bob = m['bob'](t)
        yaw = s['yaw'] + m['yaw'](t)

        # human local frame: xh forward (face direction), yh left, zh up.
        # yaw == 0 means the person looks straight at the camera (-x).
        psi = math.pi + yaw
        c, sn = math.cos(psi), math.sin(psi)
        xh = np.array([c, sn, 0.0])
        yh = np.array([-sn, c, 0.0])
        zh = np.array([0.0, 0.0, 1.0])
        origin = np.array([s['x'] + m['sway_x'](t),
                           s['y'] + m['sway_y'](t),
                           bob])

        def p(fwd, left, up):
            return origin + xh * fwd + yh * left + zh * up

        sw = b['shoulder_width'] / 2.0
        hw = b['hip_width'] / 2.0
        h_sh = b['h_shoulder'] + breath
        h_el = h_sh - b['upper_arm']
        h_wr = h_el - b['forearm']
        head_yaw = m['head_yaw'](t)
        head_pitch = m['head_pitch'](t)
        swing_r = m['arm_r'](t)
        swing_l = m['arm_l'](t)

        pos = {}
        pos['Neck'] = p(0.0, 0.0, h_sh)
        pos['LShoulder'] = p(0.0, sw, h_sh)
        pos['RShoulder'] = p(0.0, -sw, h_sh)
        pos['LHip'] = p(0.0, hw, b['h_hip'])
        pos['RHip'] = p(0.0, -hw, b['h_hip'])
        pos['LKnee'] = p(0.0, hw * 0.85, b['h_knee'])
        pos['RKnee'] = p(0.0, -hw * 0.85, b['h_knee'])
        pos['LAnkle'] = p(0.0, hw * 0.8, b['h_ankle'])
        pos['RAnkle'] = p(0.0, -hw * 0.8, b['h_ankle'])

        # head: nose in front, eyes slightly in front, ears at the sides.
        # a small extra head yaw makes the face turn a little.
        hc, hs = math.cos(head_yaw), math.sin(head_yaw)
        xf = xh * hc + yh * hs
        yf = -xh * hs + yh * hc
        head = origin + zh * b['h_nose']

        def ph(fwd, left, up):
            return head + xf * fwd + yf * left + zh * up

        pos['Nose'] = ph(b['head_depth'] * 1.7, 0.0, head_pitch)
        pos['LEye'] = ph(b['head_depth'] * 1.1, b['eye_gap'],
                         b['h_eye'] - b['h_nose'] + head_pitch)
        pos['REye'] = ph(b['head_depth'] * 1.1, -b['eye_gap'],
                         b['h_eye'] - b['h_nose'] + head_pitch)
        pos['LEar'] = ph(-b['head_depth'] * 0.3, b['ear_gap'],
                         b['h_ear'] - b['h_nose'] + head_pitch)
        pos['REar'] = ph(-b['head_depth'] * 0.3, -b['ear_gap'],
                         b['h_ear'] - b['h_nose'] + head_pitch)

        # arms.  ``sign`` is +1 for the left arm (the human's left is +yh).
        hand_frames = {}
        for hand, sign, swing in (('R', -1.0, swing_r), ('L', 1.0, swing_l)):
            if hand == self.present_hand:
                # hand held out towards the robot, like an offered
                # handshake, built from randomized shoulder/elbow/wrist
                # angles (self.presented_arm_angles, see _sample_person) so
                # different people offer the hand at different angles
                # instead of always the same fixed template.
                pa = self.presented_arm_angles
                elev = math.radians(pa['shoulder_elevation_deg'])
                azim = math.radians(pa['shoulder_azimuth_deg'])
                flex = math.radians(pa['elbow_flex_deg'])
                wpitch = math.radians(pa['wrist_pitch_deg'])
                wroll = math.radians(pa['wrist_roll_deg'])

                shoulder = pos['{}Shoulder'.format(hand)]
                # hinge axis for the elbow/wrist bend -- roughly the body's
                # left-right axis, so a positive elbow_flex/wrist_pitch
                # bends the same perceived way (forearm/hand swinging up
                # toward the shoulder) on both arms.
                hinge = yh * sign

                # shoulder_elevation: 0 deg = arm hanging straight down,
                # 90 deg = raised to horizontal. shoulder_azimuth: 0 deg =
                # raised straight toward the camera, +-90 deg = swung out
                # to the side. Elevation is kept below horizontal by
                # ~present_shoulder_elevation_deg_range's default so the
                # wrist doesn't get pushed above the camera's vertical FOV
                # at the close ~distance_range this person stands at.
                raise_dir = self._unit(
                    xh * math.cos(azim) + hinge * math.sin(azim))
                upper_dir = self._unit(
                    -zh * math.cos(elev) + raise_dir * math.sin(elev))
                elbow = shoulder + b['upper_arm'] * upper_dir + xh * 0.3 * swing

                # elbow_flex: 0 deg = forearm continues the upper arm;
                # positive bends it back up toward the shoulder.
                fdir = self._unit(self._rotate(upper_dir, hinge, flex))
                wrist = elbow + b['forearm'] * fdir

                # wrist_pitch: bend of the hand relative to the forearm
                # (0 deg = fingers continue the forearm direction).
                u = self._unit(self._rotate(fdir, hinge, wpitch))
                # baseline palm normal (wrist_roll == 0 deg): perpendicular
                # to both the hinge axis and the finger direction, so the
                # index-finger edge of the hand faces the robot -- same
                # convention as the relaxed hanging arm below (not the
                # palm itself, which is what actually exercises whether a
                # consumer approaches along the *fitted* palm normal
                # instead of assuming the palm always faces the camera
                # head-on).  wrist_roll then rolls the palm around the
                # finger axis, so different people present the hand at
                # different palm angles.
                n0 = self._unit(np.cross(hinge, u))
                n = self._unit(self._rotate(n0, u, wroll))
            else:
                # relaxed hanging arm: fingers down, palm facing the thigh
                elbow = p(b['elbow_fwd'] + swing,
                          sign * (sw + b['elbow_out']), h_el)
                wrist = p(b['wrist_fwd'] + 1.6 * swing,
                          sign * (sw + b['wrist_out']), h_wr)
                u = -zh                      # fingers point down
                n = -yh * sign               # palm faces the body
            pos['{}Elbow'.format(hand)] = elbow
            pos['{}Wrist'.format(hand)] = wrist
            # v (thumb direction) follows from the handedness of the hand:
            # the (u, v, n) triple is left-handed for the right hand and
            # right-handed for the left one, so the same landmark table
            # produces correctly mirrored hands.
            v = np.cross(u, n) if hand == 'R' else np.cross(n, u)
            hand_frames[hand] = (wrist, u, v, n)
        return pos, hand_frames

    @staticmethod
    def _unit(v):
        return v / np.linalg.norm(v)

    @staticmethod
    def _rotate(v, axis, angle):
        """Rodrigues' rotation formula: rotate ``v`` by ``angle`` [rad]
        around the unit vector ``axis``."""
        c, s = math.cos(angle), math.sin(angle)
        return v * c + np.cross(axis, v) * s + axis * axis.dot(v) * (1.0 - c)

    def _hand_bone_pairs(self, hand):
        """Bones of one hand, wrist -> Hand0 plus the 20 finger links."""
        pairs = [('{}Wrist'.format(hand), '{}Hand0'.format(hand))]
        pairs += [('{}Hand{}'.format(hand, c[0]),
                   '{}Hand{}'.format(hand, c[1]))
                  for c in self.hand_sequence]
        return pairs

    def _hand_positions(self, hand, frame):
        wrist, u, v, n = frame
        scale = self.body['hand_length']
        basis = np.vstack([u, v, n])           # (3, 3)
        pts = wrist + scale * HAND_LOCAL.dot(basis)
        return {'{}Hand{}'.format(hand, i): pts[i] for i in range(len(pts))}

    @property
    def camera_basis(self):
        """base_frame から見たカメラ光学系の (x, y, z) 軸."""
        cp, sp = math.cos(self.camera_pitch), math.sin(self.camera_pitch)
        x_dir = np.array([0.0, -1.0, 0.0])     # optical x = right
        y_dir = np.array([-sp, 0.0, -cp])      # optical y = down
        z_dir = np.array([cp, 0.0, -sp])       # optical z = forward
        return x_dir, y_dir, z_dir

    @property
    def camera_origin(self):
        """base_frame から見たカメラの位置."""
        return np.array([self.camera_forward, 0.0, self.camera_height])

    @property
    def base_from_camera(self):
        """カメラ光学系 -> base_frame の 4x4 同次変換行列.

        本物は TF から引くが、こちらは仮想カメラの置き方が既知なので
        ``~camera_height`` / ``~camera_forward`` / ``~camera_pitch_deg``
        から直接作れる。
        """
        x_dir, y_dir, z_dir = self.camera_basis
        matrix = np.eye(4)
        matrix[:3, :3] = np.vstack([x_dir, y_dir, z_dir]).T
        matrix[:3, 3] = self.camera_origin
        return matrix

    def _to_camera(self, p):
        """Robot frame (x fwd, y left, z up) -> camera optical frame."""
        q = p - self.camera_origin
        x_dir, y_dir, z_dir = self.camera_basis
        return np.array([q.dot(x_dir), q.dot(y_dir), q.dot(z_dir)])

    def _project(self, pc):
        """Camera 座標系の点をピクセル座標へ投影する.

        カメラの後ろ (``pc[2] <= 0.05``) は投影しようがないので None。
        画角の外に出た場合はそれでも (u, v) を返す -- 画角内かどうかは
        ``~filter_by_fov`` に従って ``_make_joint`` 側が判断する。
        """
        if pc[2] <= 0.05:
            return None
        intr = self.intrinsics
        u = pc[0] * intr.fx / pc[2] + intr.cx
        v = pc[1] * intr.fy / pc[2] + intr.cy
        return u, v

    def _in_frame(self, u, v):
        return 0.0 <= u < self.width and 0.0 <= v < self.height

    @staticmethod
    def _apply_transform(person, transform):
        """person の全関節点 (可視 + 非可視) を transform の座標系へ移す (破壊的)."""
        matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)

        def move(points):
            if not points:
                return points
            pts = np.asarray(points, dtype=np.float64)   # (N, 3)
            homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
            return list(homogeneous.dot(matrix.T)[:, :3])

        person.positions = move(person.positions)
        person.hidden_positions = move(person.hidden_positions)

    # ------------------------------------------------------------------
    # visibility / detection state
    # ------------------------------------------------------------------
    def _visibility(self, name, t, yaw_total):
        score = self.vis_bias[name] + self.vis_osc[name](t) \
            + self.nprng.normal(0.0, 0.05)
        # the ear/eye on the far side gets occluded when the head turns away
        if name in ('REar', 'LEar', 'REye', 'LEye'):
            sign = 1.0 if name.startswith('L') else -1.0
            score -= max(0.0, sign * math.sin(yaw_total)) * 1.2
        return float(min(1.0, score))

    def _update_detection_state(self):
        if self.person_detected:
            if self.rng.random() < self.person_lost_prob:
                self.person_detected = False
        else:
            if self.rng.random() < self.person_recover_prob:
                self.person_detected = True
        for hand in ('R', 'L'):
            if self.hand_detected[hand]:
                if self.rng.random() < self.hand_lost_prob:
                    self.hand_detected[hand] = False
            else:
                if self.rng.random() < self.hand_recover_prob:
                    self.hand_detected[hand] = True

    # ------------------------------------------------------------------
    # 2D generation (本物の PeoplePoseEstimator.estimate に相当)
    # ------------------------------------------------------------------
    def _generate(self, t):
        """時刻 t [s] の画像座標の関節位置を生成する.

        Returns
        -------
        list of list of dict
            人物ごとに dict(limb=str, x=float, y=float, score=float) のリスト。
            score が負の関節は未検出。3 次元位置とその有効性は 'cam' /
            'has_depth' として同じ dict に入る。
            人物がロストしているフレームでは空リストを返す。
        """
        self._update_detection_state()
        if not self.person_detected:
            return []

        pos, hand_frames = self._body_positions(t)
        yaw_total = self.stand['yaw'] + self.motion['yaw'](t) \
            + self.motion['head_yaw'](t)

        joints = []
        for limb in self.index2limbname:
            if limb == 'Bkg':
                joints.append(dict(limb=limb, x=0, y=0, score=-1))
                continue
            score = self._visibility(limb, t, yaw_total)
            joints.append(self._make_joint(limb, pos[limb], score))

        if self.use_hand:
            for hand in ('R', 'L'):
                names = ['{}Hand{}'.format(hand, i) for i in range(21)]
                # computed unconditionally (even if the whole hand is
                # lost this frame) so _make_joint can still attach the
                # ground-truth ``cam`` position for the viewer's faint
                # "not detected" drawing.
                hpos = self._hand_positions(hand, hand_frames[hand])
                if not self.hand_detected[hand]:
                    joints += [self._make_joint(name, hpos[name], -1.0)
                               for name in names]
                    continue
                for i, name in enumerate(names):
                    # MediaPipe hand landmarks carry no visibility, the
                    # real estimator reports score 1.0 for all of them
                    joints.append(self._make_joint(
                        name, hpos[name], 1.0,
                        depth_hole=HAND_DEPTH_HOLE.get(
                            i, HAND_DEPTH_HOLE_DEFAULT)))
        return [joints]

    def _make_joint(self, limb, p_robot, score, depth_hole=None):
        """Project one joint and decide visibility / depth validity."""
        noisy = p_robot + self.nprng.normal(0.0, self.joint_noise, 3)
        pc = self._to_camera(noisy)
        uv = self._project(pc)
        out_of_frame = uv is not None and self.filter_by_fov \
            and not self._in_frame(*uv)
        if uv is None or out_of_frame or score < self.min_visibility:
            # behind the camera, outside the image (only when
            # ~filter_by_fov), or below the visibility threshold ->
            # dropped by the real estimator as well.  ``cam`` is kept
            # anyway (unlike the real estimator, this one has ground
            # truth for joints it drops) purely so the viewer can draw
            # them faintly; estimate_3d never looks at it when score < 0.
            x, y = uv if uv is not None else (0, 0)
            return dict(limb=limb, x=x, y=y, score=-1, cam=pc)
        if depth_hole is None:
            depth_hole = self.depth_hole_prob
        has_depth = self.rng.random() >= depth_hole
        return dict(limb=limb, x=uv[0], y=uv[1], score=score,
                    cam=pc, has_depth=has_depth)

    # ------------------------------------------------------------------
    # 3D generation (本物の PeoplePoseEstimator.estimate_3d に相当)
    # ------------------------------------------------------------------
    def _to_people_3d(self, people_joint_positions):
        """カメラ座標系の ``Person3D`` を作る (骨はまだ張らない).

        本物の推定と同じ ``positions`` (可視 + 深度あり) に加えて、
        ``hidden_positions`` に画角の外・可視性不足・深度欠測で落ちた
        関節の真値を入れる (fake だけが知っている情報)。viewer で薄く
        描くためだけのもので、``palm_plane`` などの実際の判断ロジックは
        今まで通り ``positions`` / ``limb_names`` しか見ない。
        """
        people = []
        for person_joint_positions in people_joint_positions:
            person = Person3D()
            for joint_pos in person_joint_positions:
                # 深度画像が無いので、深度の欠測は has_depth で模擬する
                if joint_pos['score'] < 0 or not joint_pos['has_depth']:
                    if 'cam' in joint_pos:
                        person.hidden_limb_names.append(joint_pos['limb'])
                        person.hidden_positions.append(
                            np.asarray(joint_pos['cam'], dtype=np.float64))
                    continue
                person.limb_names.append(joint_pos['limb'])
                person.scores.append(joint_pos['score'])
                person.positions.append(
                    np.asarray(joint_pos['cam'], dtype=np.float64))

            # 本物と同じく Neck か Nose と最低限の関節数を要求する
            if person.position_of('Neck') is None \
                    and person.position_of('Nose') is None:
                continue
            if len(person.positions) < self.min_joints:
                continue
            people.append(person)
        return people

    def _create_bones(self, person):
        """可視の骨 (``bones``) と、片端以上が非可視の骨 (``hidden_bones``)."""
        pairs = [(self.index2limbname[c[0] - 1], self.index2limbname[c[1] - 1])
                 for c in self.limb_sequence]
        if self.use_hand and self.include_hand_bones:
            for hand in ('R', 'L'):
                pairs += self._hand_bone_pairs(hand)

        def lookup(name):
            if name in person.limb_names:
                return person.positions[person.limb_names.index(name)], True
            if name in person.hidden_limb_names:
                idx = person.hidden_limb_names.index(name)
                return person.hidden_positions[idx], False
            return None, False

        bones, hidden_bones = [], []
        for j1_name, j2_name in pairs:
            p1, visible1 = lookup(j1_name)
            p2, visible2 = lookup(j2_name)
            if p1 is None or p2 is None:
                continue
            bone = Bone(name='{}->{}'.format(j1_name, j2_name),
                       start_point=p1, end_point=p2)
            (bones if (visible1 and visible2) else hidden_bones).append(bone)
        return bones, hidden_bones

    # ------------------------------------------------------------------
    # visualization
    # ------------------------------------------------------------------
    def create_canvas(self):
        """描画用の黒画像を作る (偽推定には入力画像が無いため)."""
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def draw_joints(self, img, people_joint_positions):
        """2 次元関節位置を img へ描き込む (img は破壊的に変更される)."""
        if not HAS_CV:
            raise RuntimeError('cv2 is not available, cannot draw joints')
        n = len(self.index2limbname) - 1
        for person in people_joint_positions:
            for conn in self.limb_sequence:
                j1, j2 = person[conn[0] - 1], person[conn[1] - 1]
                if j1['score'] < 0 or j2['score'] < 0:
                    continue
                cv2.line(img, (int(j1['x']), int(j1['y'])),
                         (int(j2['x']), int(j2['y'])), (0, 255, 255), 2)
            if self.use_hand:
                by_name = {jt['limb']: jt for jt in person}
                for hand in ('R', 'L'):
                    for n1, n2 in self._hand_bone_pairs(hand):
                        j1, j2 = by_name.get(n1), by_name.get(n2)
                        if j1 is None or j2 is None \
                                or j1['score'] < 0 or j2['score'] < 0:
                            continue
                        cv2.line(img, (int(j1['x']), int(j1['y'])),
                                 (int(j2['x']), int(j2['y'])),
                                 (0, 160, 255), 1)
            for i, jt in enumerate(person):
                if jt['score'] < 0:
                    continue
                r, g, b = self.hsv2rgb(1.0 * min(i, n - 1) / n)
                radius = 4 if i < n else 2
                cv2.circle(img, (int(jt['x']), int(jt['y'])), radius,
                           (b * 255, g * 255, r * 255), thickness=-1)
        return img

    @staticmethod
    def hsv2rgb(h, s=1.0, v=1.0):
        """matplotlib 無しの hsv カラーマップ (本物の cmap('hsv') 相当)."""
        i = int(h * 6.0) % 6
        f = h * 6.0 - int(h * 6.0)
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        return [(v, t, p), (q, v, p), (p, v, t),
                (p, q, v), (t, p, v), (v, p, q)][i]


if __name__ == '__main__':
    rospy.init_node('fake_people_pose_estimator')

    node = FakeRosPeoplePoseEstimator()
    while not rospy.is_shutdown():
        result = node.wait_for_result(timeout=1.0)
        if result is None:
            rospy.logwarn_throttle(5.0, 'no result generated')
            continue
        for person in result.people:
            neck = person.position_of('Neck')
            if neck is None:
                continue
            rospy.loginfo_throttle(
                1.0, 'Neck at (%.2f, %.2f, %.2f) in %s'
                % (neck[0], neck[1], neck[2], result.frame_id))
