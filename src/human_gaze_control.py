#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
import numpy as np
import rospy

from skrobot.models import Aero
from skrobot.interfaces.ros import AeroROSRobotInterface
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

        # Gaze targets parameters
        # Negative is right, positive is left for yaw joints
        self.initial_gaze_yaw = rospy.get_param('~initial_gaze_yaw', -0.5)  # Default: -0.5 rad (approx -30 deg, right side)
        self.initial_gaze_pitch = rospy.get_param('~initial_gaze_pitch', 0.3)  # Default: 0.3 rad (approx 17 deg, looking down)
        self.deep_look_angle = rospy.get_param('~deep_look_angle', 1.0)  # Additional angle for deep look (approx 57 deg)
        self.deep_speed_scale = rospy.get_param('~deep_speed_scale', 0.5)  # Gaze speed multiplier for deep look (default 0.5 = half speed)
        self.nod_amplitude = rospy.get_param('~nod_amplitude', 0.15)  # Pitch down angle for nod (default 0.15 rad, approx 8.5 deg)
        self.nod_duration = rospy.get_param('~nod_duration', 0.25)  # Duration of nod down (default 0.25s)

        # Tracking state
        self.last_known_yaw = None
        self.last_known_pitch = None
        self.last_seen_time = None
        self.has_detected_human = False

        # Current target tracking yaw/pitch (to know which direction to deep look)
        self.current_target_yaw = 0.0
        self.current_target_pitch = 0.0

        # State machine states: "FORWARD", "MOVING_TO_GAZE", "WAIT_CAPTURE", "MOVING_TO_DEEP_GAZE", "WAIT_DEEP_CAPTURE", "MOVING_TO_FORWARD"
        self.state = "FORWARD"
        self.state_start_time = rospy.Time.now()
        self.motion_duration = 0.0
        self.gaze_timer_start = rospy.Time.now()

        # Subscribe to people pose
        self.pose_sub = rospy.Subscriber('~pose', PeoplePoseArray, self.pose_callback, queue_size=1)

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
        self.last_known_yaw = current_waist_y + current_neck_y - detected_yaw
        self.last_known_pitch = current_neck_p + detected_pitch
        self.last_seen_time = rospy.Time.now()
        self.has_detected_human = True

        # If we are sweeping or waiting at the end, and we just successfully captured them
        if self.state in ["GAZE_SWEEP", "WAIT_FINAL"]:
            rospy.loginfo("Human captured in camera frame. Nodding and returning to forward posture.")
            self.transition_to_nod_down()

    def transition_to_nod_down(self):
        self.state = "NOD_DOWN"
        self.state_start_time = rospy.Time.now()

        # Update robot model to current state
        self.robot.angle_vector(self.ri.angle_vector())

        # Keep current waist and neck yaw, but pitch head down by nod_amplitude
        current_neck_p = self.robot.neck_p_joint.joint_angle()
        target_neck_p = np.clip(current_neck_p + self.nod_amplitude, -0.349, 0.960)
        self.robot.neck_p_joint.joint_angle(target_neck_p)

        self.motion_duration = self.nod_duration
        self.ri.angle_vector(self.robot.angle_vector(), self.motion_duration)

    def transition_to_moving_to_forward(self):
        self.state = "MOVING_TO_FORWARD"
        self.state_start_time = rospy.Time.now()

        # Command waist and head back to forward (0.0) positions
        self.robot.angle_vector(self.ri.angle_vector())
        self.robot.waist_y_joint.joint_angle(0.0)
        self.robot.neck_y_joint.joint_angle(0.0)
        self.robot.neck_p_joint.joint_angle(0.0)

        self.motion_duration = 1.5
        self.ri.angle_vector(self.robot.angle_vector(), self.motion_duration)

    def transition_to_gaze_sweep(self):
        # Determine target gaze positions
        if not self.has_detected_human:
            # If no human has been seen yet, use initial gaze parameters
            base_yaw = self.initial_gaze_yaw
            base_pitch = self.initial_gaze_pitch
            rospy.loginfo(f"No human seen yet. Performing initial gaze check to default direction (yaw={base_yaw:.2f} rad, pitch={base_pitch:.2f} rad).")
        else:
            # Check if the last seen time is too old (e.g. older than 15 seconds)
            time_since_seen = (rospy.Time.now() - self.last_seen_time).to_sec()
            if time_since_seen > 15.0:
                rospy.logwarn_throttle(5, f"Human has not been seen for {time_since_seen:.1f}s. Skipping gaze.")
                return
            base_yaw = self.last_known_yaw
            base_pitch = self.last_known_pitch

        # For a continuous sweep, the target is the full deep look angle!
        direction = np.sign(base_yaw) if base_yaw != 0.0 else -1.0
        deep_yaw = base_yaw + direction * self.deep_look_angle
        deep_pitch = base_pitch

        self.current_target_yaw = deep_yaw
        self.current_target_pitch = deep_pitch

        self.state = "GAZE_SWEEP"
        self.state_start_time = rospy.Time.now()

        # Perform the sweep continuously at a slower velocity
        self._send_gaze_command(deep_yaw, deep_pitch, self.max_angular_velocity * self.deep_speed_scale)

    def _send_gaze_command(self, target_yaw, target_pitch, max_vel):
        # Distribute yaw angle in a natural human-like balance (40% waist, 60% neck)
        target_waist_y = 0.4 * target_yaw
        target_neck_y = 0.6 * target_yaw
        target_neck_p = target_pitch

        # Enforce joint limits
        target_waist_y = np.clip(target_waist_y, -0.785, 0.785)  # waist_y_joint: min=-0.785, max=0.785
        target_neck_y = target_yaw - target_waist_y
        target_neck_y = np.clip(target_neck_y, -0.873, 0.873)  # neck_y_joint: min=-0.873, max=0.873
        target_neck_p = np.clip(target_neck_p, -0.349, 0.960)  # neck_p_joint: min=-0.349, max=0.960

        rospy.loginfo(f"Commanding joint angles -> waist_y: {target_waist_y:.2f}, neck_y: {target_neck_y:.2f}, neck_p: {target_neck_p:.2f}")

        # Update robot model
        self.robot.angle_vector(self.ri.angle_vector())
        self.robot.waist_y_joint.joint_angle(target_waist_y)
        self.robot.neck_y_joint.joint_angle(target_neck_y)
        self.robot.neck_p_joint.joint_angle(target_neck_p)

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
            now = rospy.Time.now()

            if self.state == "FORWARD":
                if (now - self.gaze_timer_start).to_sec() >= self.gaze_interval:
                    self.gaze_timer_start = now
                    self.transition_to_gaze_sweep()

            elif self.state == "GAZE_SWEEP":
                # If motion completes without any detection along the way, wait a short moment at the end
                if (now - self.state_start_time).to_sec() >= self.motion_duration:
                    self.state = "WAIT_FINAL"
                    self.state_start_time = now

            elif self.state == "WAIT_FINAL":
                # Timeout after 1.0 second of waiting at the end of the sweep
                if (now - self.state_start_time).to_sec() >= 1.0:
                    rospy.loginfo("Sweep completed: human still not seen. Returning to forward posture.")
                    self.transition_to_moving_to_forward()

            elif self.state == "NOD_DOWN":
                # Wait until the nod-down motion completes, then return forward
                if (now - self.state_start_time).to_sec() >= self.motion_duration:
                    self.transition_to_moving_to_forward()

            elif self.state == "MOVING_TO_FORWARD":
                # Wait until the return motion to forward posture completes
                if (now - self.state_start_time).to_sec() >= self.motion_duration:
                    self.state = "FORWARD"
                    self.state_start_time = now
                    # Reset the gaze timer start so that the next check happens gaze_interval seconds after returning
                    self.gaze_timer_start = now

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('human_gaze_control')
    node = HumanGazeControl()
    node.run()
