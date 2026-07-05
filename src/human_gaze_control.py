#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
import numpy as np
import rospy
import threading

from skrobot.models import Aero
from skrobot.interfaces.ros import AeroROSRobotInterface
import tf
from std_msgs.msg import Bool
from geometry_msgs.msg import PointStamped
from jsk_recognition_msgs.msg import PeoplePoseArray

class HumanGazeControl(object):
    def __init__(self):
        rospy.loginfo("Initializing HumanGazeControl node...")

        # Load robot model and ROS interface
        self.robot = Aero()
        self.ri = AeroROSRobotInterface(self.robot)
        
        # Sync robot model state with the physical/simulated robot
        self.robot.angle_vector(self.ri.angle_vector())

        # Parameters
        self.gaze_interval = rospy.get_param('~gaze_interval', 5.0)  # Gaze check every 5 seconds
        self.max_angular_velocity = rospy.get_param('~max_angular_velocity', 0.5)  # rad/s
        self.min_confidence = rospy.get_param('~min_confidence', 0.3)  # visibility threshold
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.max_height_diff = rospy.get_param('~max_height_diff', 1.5)  # 20cm height difference

        # Gaze targets parameters
        # Negative is right, positive is left for yaw joints
        self.initial_gaze_yaw = rospy.get_param('~initial_gaze_yaw', -0.5)  # Default: -0.5 rad (approx -30 deg, right side)
        self.initial_gaze_pitch = rospy.get_param('~initial_gaze_pitch', 0.3)  # Default: 0.3 rad (approx 17 deg, looking down)
        self.deep_look_angle = rospy.get_param('~deep_look_angle', 1.2)  # Additional angle for deep look (approx 69 deg)
        self.deep_speed_scale = rospy.get_param('~deep_speed_scale', 0.5)  # Gaze speed multiplier for deep look (default 0.5 = half speed)
        self.nod_amplitude = rospy.get_param('~nod_amplitude', 0.15)  # Pitch down angle for nod (default 0.15 rad, approx 8.5 deg)
        self.nod_duration = rospy.get_param('~nod_duration', 0.25)  # Duration of nod down (default 0.25s)
        self.gaze_hold_duration = rospy.get_param('~gaze_hold_duration', 0.5)  # Duration to hold gaze on human (default 0.5s)
        self.sweep_margin = rospy.get_param('~sweep_margin', 0.25)  # Margin to add to sweep angle (default 0.25 rad)

        # Tracking state
        self.lock = threading.Lock()
        self.last_known_yaw = None
        self.last_known_pitch = None
        self.last_seen_time = None
        self.has_detected_human = False
        self.first_human_height = None
        self.last_valid_pose_time = None
        self.tf_listener = tf.TransformListener()

        # Current target tracking yaw/pitch (to know which direction to deep look)
        self.current_target_yaw = 0.0
        self.current_target_pitch = 0.0

        # State machine states: "FORWARD", "MOVING_TO_GAZE", "WAIT_CAPTURE", "MOVING_TO_DEEP_GAZE", "WAIT_DEEP_CAPTURE", "MOVING_TO_FORWARD"
        self.state = "FORWARD"
        self.state_start_time = rospy.Time.now()
        self.motion_duration = 0.0
        self.gaze_timer_start = rospy.Time.now()
        self.lost_pub = rospy.Publisher('~lost', Bool, queue_size=1, latch=True)
        self.is_lost = True
        self.lost_pub.publish(Bool(data=True))

        # Subscribe to people pose
        self.pose_sub = rospy.Subscriber('~pose', PeoplePoseArray, self.pose_callback, queue_size=1)

        # Trigger initial look immediately
        self.transition_to_initial_look()

        rospy.loginfo("HumanGazeControl node initialized successfully.")

    def pose_callback(self, msg):
        if not msg.poses:
            return

        # Find the first pose (typically the most prominent person)
        target_pose = msg.poses[0]

        # Look for "Neck" first, fallback to "Nose"
        neck_idx = -1
        nose_idx = -1
        for idx, name in enumerate(target_pose.limb_names):
            if name == "Neck" and target_pose.scores[idx] > self.min_confidence:
                neck_idx = idx
            elif name == "Nose" and target_pose.scores[idx] > self.min_confidence:
                nose_idx = idx

        best_idx = neck_idx if neck_idx != -1 else nose_idx
        if best_idx == -1:
            return

        joint_pos = target_pose.poses[best_idx].position
        x = joint_pos.x
        y = joint_pos.y
        z = joint_pos.z

        # Transform human position to base_frame using TF to get height (independent of head orientation)
        camera_point = PointStamped()
        camera_point.header = msg.header
        camera_point.point = joint_pos

        try:
            self.tf_listener.waitForTransform(
                self.base_frame,
                camera_point.header.frame_id,
                camera_point.header.stamp,
                rospy.Duration(1.0)
            )
            base_point = self.tf_listener.transformPoint(self.base_frame, camera_point)
            base_z = base_point.point.z
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"TF transform in gaze control failed: {str(e)}")
            return

        now = rospy.Time.now()

        # Reset height baseline if no valid person seen for 5.0 seconds
        if self.last_valid_pose_time is not None:
            if (now - self.last_valid_pose_time).to_sec() > 5.0 or self.is_lost:
                self.first_human_height = None

        if self.first_human_height is None:
            self.first_human_height = base_z
            self.last_valid_pose_time = now
        else:
            # Update baseline with a low-pass filter
            self.last_valid_pose_time = now
            self.first_human_height = 0.95 * self.first_human_height + 0.05 * base_z

        # Calculate yaw and pitch in camera frame
        # If z is close to 0, it's a 2D pose (pixel coordinates), so use pixel fallback
        if z > 1e-3:
            detected_yaw = math.atan2(x, z)
            detected_pitch = math.atan2(y, z)
        else:
            # Fallback to pixel-based approximation (assume 640x480 resolution)
            # Center is (320, 240). FOV horizontal is ~60 deg (~1.05 rad), vertical is ~45 deg (~0.8 rad)
            detected_yaw = (x - 320.0) / 320.0 * (1.05 / 2.0)
            detected_pitch = (y - 240.0) / 240.0 * (0.8 / 2.0)

        # Get robot's current joint state
        self.robot.angle_vector(self.ri.angle_vector())
        current_waist_y = self.robot.waist_y_joint.joint_angle()
        current_neck_y = self.robot.neck_y_joint.joint_angle()
        current_neck_p = self.robot.neck_p_joint.joint_angle()

        # Compute human target angles in absolute base frame
        # waist/neck angle signs: positive = left, negative = right
        # detected_yaw in camera: positive = right, negative = left
        calculated_yaw = math.atan2(base_point.point.y, base_point.point.x)

        with self.lock:
            self.last_known_yaw = calculated_yaw
            self.last_known_pitch = current_neck_p + detected_pitch
            self.last_seen_time = rospy.Time.now()
            self.has_detected_human = True

            # If we are searching for the human, and we just successfully captured them
            if self.state in ["MOVING_TO_INITIAL", "WAIT_INITIAL_CAPTURE", "GAZE_INITIAL_SWEEP", "WAIT_INITIAL_SWEEP", "GAZE_DEEP_SWEEP", "WAIT_DEEP_SWEEP"]:
                rospy.loginfo("Human captured in camera frame. Returning to forward posture.")
                if self.is_lost:
                    self.is_lost = False
                    self.lost_pub.publish(Bool(data=False))
                self.transition_to_moving_to_forward()

    def transition_to_initial_look(self):
        self.state = "MOVING_TO_INITIAL"
        self.state_start_time = rospy.Time.now()

        # Initial look target (all the way to the deep look angle)
        direction = np.sign(self.initial_gaze_yaw) if self.initial_gaze_yaw != 0.0 else -1.0
        yaw = self.initial_gaze_yaw + direction * self.deep_look_angle
        pitch = self.initial_gaze_pitch

        rospy.loginfo(f"Node started. Turning to initial deep gaze direction (yaw={yaw:.2f} rad, pitch={pitch:.2f} rad) and waiting for human.")

        self._send_gaze_command(yaw, pitch, self.max_angular_velocity)

    def transition_to_moving_to_forward(self):
        self.state = "MOVING_TO_FORWARD"
        self.state_start_time = rospy.Time.now()

        # Command waist and head back to forward (0.0) positions in 1.0s
        self.robot.angle_vector(self.ri.angle_vector())
        self.robot.waist_y_joint.joint_angle(0.0)
        self.robot.neck_y_joint.joint_angle(0.0)
        self.robot.neck_p_joint.joint_angle(0.0)

        self.motion_duration = 1.0
        self.ri.angle_vector(self.robot.angle_vector(), self.motion_duration)

    def transition_to_gaze_sweep(self):
        # Determine target gaze positions
        if not self.has_detected_human:
            base_yaw = self.initial_gaze_yaw
            base_pitch = self.initial_gaze_pitch
            rospy.loginfo(f"[Stage 1] No human seen yet. Performing initial gaze check to default direction (yaw={base_yaw:.2f} rad).")
        else:
            margin = self.sweep_margin * np.sign(self.last_known_yaw) if self.last_known_yaw != 0.0 else 0.0
            base_yaw = self.last_known_yaw + margin
            base_pitch = self.last_known_pitch
            rospy.loginfo(f"[Stage 1] Searching for human around last known location with margin (yaw={base_yaw:.2f} rad).")

        self.current_target_yaw = base_yaw
        self.current_target_pitch = base_pitch

        self.state = "GAZE_INITIAL_SWEEP"
        self.state_start_time = rospy.Time.now()

        # 1.5 seconds to turn to the last known direction
        self._send_gaze_command(base_yaw, base_pitch, duration=1.5)

    def _send_gaze_command(self, target_yaw, target_pitch, duration=None, max_vel=None):
        # Distribute yaw angle in a natural human-like balance (40% waist, 60% neck)
        target_waist_y = 0.4 * target_yaw
        target_neck_y = 0.6 * target_yaw
        target_neck_p = target_pitch

        # Enforce joint limits
        target_waist_y = np.clip(target_waist_y, -0.785, 0.785)  # waist_y_joint: min=-0.785, max=0.785
        target_neck_y = target_yaw - target_waist_y
        target_neck_y = np.clip(target_neck_y, -0.873, 0.873)  # neck_y_joint: min=-0.873, max=0.873
        target_neck_p = np.clip(target_neck_p, -0.349, 0.960)  # neck_p_joint: min=-0.349, max=0.960

        # Don't show log when returning to forward (all joints at 0)
        if not (target_waist_y == 0.0 and target_neck_y == 0.0 and target_neck_p == 0.0):
            rospy.loginfo(f"Commanding joint angles -> waist_y: {target_waist_y:.2f}, neck_y: {target_neck_y:.2f}, neck_p: {target_neck_p:.2f}")

        # Update robot model
        self.robot.angle_vector(self.ri.angle_vector())
        self.robot.waist_y_joint.joint_angle(target_waist_y)
        self.robot.neck_y_joint.joint_angle(target_neck_y)
        self.robot.neck_p_joint.joint_angle(target_neck_p)

        if duration is not None:
            self.motion_duration = duration
        else:
            # Compute duration dynamically based on the angular distance to move (human-like speed profiles)
            current_waist_y = self.robot.waist_y_joint.joint_angle()
            current_neck_y = self.robot.neck_y_joint.joint_angle()
            diff = max(abs(target_waist_y - current_waist_y), abs(target_neck_y - current_neck_y))
            self.motion_duration = max(0.8, diff / max_vel)

        # Send angle vector to the robot interface
        self.ri.angle_vector(self.robot.angle_vector(), self.motion_duration)

    def run(self):
        rate = rospy.Rate(10) # 10Hz
        while not rospy.is_shutdown():
            with self.lock:
                now = rospy.Time.now()

                if self.state == "FORWARD":
                    if (now - self.gaze_timer_start).to_sec() >= self.gaze_interval:
                        self.gaze_timer_start = now
                        self.transition_to_gaze_sweep()

                elif self.state == "MOVING_TO_INITIAL":
                    # Wait until the initial gaze motion completes, then wait for capture
                    if (now - self.state_start_time).to_sec() >= self.motion_duration:
                        self.state = "WAIT_INITIAL_CAPTURE"
                        self.state_start_time = now
                        rospy.loginfo("Reached initial gaze direction. Standing still and waiting for human to join.")

                elif self.state == "WAIT_INITIAL_CAPTURE":
                    # Do nothing, wait indefinitely until pose_callback triggers a transition
                    pass

                elif self.state == "GAZE_INITIAL_SWEEP":
                    # Wait until initial sweep completes, then wait at that position
                    if (now - self.state_start_time).to_sec() >= self.motion_duration:
                        self.state = "WAIT_INITIAL_SWEEP"
                        self.state_start_time = now
                        self.motion_duration = 0.5

                elif self.state == "WAIT_INITIAL_SWEEP":
                    # If not found after 0.5s, turn further deep in 1.0s
                    if (now - self.state_start_time).to_sec() >= self.motion_duration:
                        direction = np.sign(self.current_target_yaw) if self.current_target_yaw != 0.0 else -1.0
                        deep_yaw = self.current_target_yaw + direction * self.deep_look_angle
                        deep_pitch = self.current_target_pitch

                        self.state = "GAZE_DEEP_SWEEP"
                        self.state_start_time = now
                        rospy.loginfo(f"[Stage 2] Human not found in Stage 1. Performing deep sweep further out (yaw={deep_yaw:.2f} rad).")
                        self._send_gaze_command(deep_yaw, deep_pitch, duration=1.0)

                elif self.state == "GAZE_DEEP_SWEEP":
                    # Wait until deep sweep completes, then wait at the deep angle
                    if (now - self.state_start_time).to_sec() >= self.motion_duration:
                        self.state = "WAIT_DEEP_SWEEP"
                        self.state_start_time = now
                        self.motion_duration = 0.5
                elif self.state == "WAIT_DEEP_SWEEP":
                    # If still not found after 0.5s, stop and reset to initial search look
                    if (now - self.state_start_time).to_sec() >= self.motion_duration:
                        rospy.loginfo("Deep sweep completed: human still not seen. Resetting to initial search look.")
                        if not self.is_lost:
                             self.is_lost = True
                             self.lost_pub.publish(Bool(data=True))
                        self.transition_to_initial_look()



                elif self.state == "MOVING_TO_FORWARD":
                    # Wait until the return motion to forward posture completes (1.0s), then hold
                    if (now - self.state_start_time).to_sec() >= self.motion_duration:
                        self.state = "FORWARD_HOLD"
                        self.state_start_time = now
                        self.motion_duration = 1.0

                elif self.state == "FORWARD_HOLD":
                    # Hold the forward posture for 1.0s, then resume normal forward tracking
                    if (now - self.state_start_time).to_sec() >= self.motion_duration:
                        self.state = "FORWARD"
                        self.state_start_time = now
                        self.gaze_timer_start = now

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('human_gaze_control')
    node = HumanGazeControl()
    node.run()
