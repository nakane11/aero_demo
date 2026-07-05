#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import rospy
import tf

from std_msgs.msg import Bool
from geometry_msgs.msg import Twist, PointStamped
from jsk_recognition_msgs.msg import PeoplePoseArray

class HumanFollowingControl(object):
    def __init__(self):
        rospy.loginfo("Initializing HumanFollowingControl node...")

        # Parameters
        self.pose_topic = rospy.get_param('~pose_topic', '/people_pose_estimation_mediapipe/pose')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.min_confidence = rospy.get_param('~min_confidence', 0.3)
        self.v_max = rospy.get_param('~v_max', 0.1)  # m/s (default 0.1 for testing)
        self.max_acceleration = rospy.get_param('~max_acceleration', 0.3)  # m/s^2
        self.control_rate = rospy.get_param('~control_rate', 10.0)  # Hz
        self.timeout_duration = rospy.get_param('~timeout_duration', 20.0)  # Failsafe timeout (seconds)
        self.lost_topic = rospy.get_param('~lost_topic', '/human_gaze_control/lost')
        self.kp = rospy.get_param('~kp', 0.5)  # Proportional gain for side-by-side following
        self.v_bias = rospy.get_param('~v_bias', 0.15)  # Nominal walking speed (m/s)

        # Tracking state
        self.human_x = None
        self.last_pose_time = None
        self.human_lost = True
        self.current_vel_x = 0.0

        # Setup TF listener
        self.tf_listener = tf.TransformListener()

        # Publisher & Subscriber
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.pose_sub = rospy.Subscriber(self.pose_topic, PeoplePoseArray, self.pose_callback, queue_size=1)
        self.lost_sub = rospy.Subscriber(self.lost_topic, Bool, self.lost_callback, queue_size=1)

        # Start control loop timer
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)

        rospy.loginfo("HumanFollowingControl node initialized.")

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

        # Prepare camera-frame point
        camera_point = PointStamped()
        camera_point.header = msg.header
        camera_point.point = target_pose.poses[best_idx].position

        try:
            # Transform human position to base_frame (e.g. base_link) using TF
            self.tf_listener.waitForTransform(
                self.base_frame,
                camera_point.header.frame_id,
                rospy.Time(0),
                rospy.Duration(1.0)
            )
            base_point = self.tf_listener.transformPoint(self.base_frame, camera_point)

            # Store the longitudinal position relative to the base_frame (positive = in front, negative = behind)
            self.human_x = base_point.point.x
            self.last_pose_time = rospy.Time.now()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(5, f"TF transformation failed: {str(e)}")

    def lost_callback(self, msg):
        self.human_lost = msg.data

    def control_loop(self, event):
        now = rospy.Time.now()
        dt = 1.0 / self.control_rate
        v_target = 0.0

        # Propagate human position using constant velocity model (dead reckoning) between pose updates
        if self.last_pose_time is not None and not self.human_lost:
            time_since_pose = (now - self.last_pose_time).to_sec()
            if time_since_pose > dt:
                # relative velocity = human absolute speed - robot speed
                self.human_x += (self.v_bias - self.current_vel_x) * dt

        # 1. Safety Timeout & Lost Check
        if self.human_lost:
            rospy.logwarn_throttle(10, "Gaze control reports human is lost. Stopping robot.")
            v_target = 0.0
        elif self.last_pose_time is not None and (now - self.last_pose_time).to_sec() > self.timeout_duration:
            rospy.logwarn_throttle(5, "Failsafe safety timeout triggered (no pose for 20s). Stopping robot.")
            v_target = 0.0
        elif self.human_x is None:
            v_target = 0.0
        else:
            x = self.human_x

            # 2. Proportional velocity control with feed-forward (target offset x = 0.0 for side-by-side)
            v_target = self.v_bias + self.kp * x
            v_target = np.clip(v_target, 0.0, self.v_max)

        # 3. Apply smooth acceleration/deceleration limits
        dt = 1.0 / self.control_rate
        dv_limit = self.max_acceleration * dt

        diff = v_target - self.current_vel_x
        self.current_vel_x += np.clip(diff, -dv_limit, dv_limit)

        # Log velocity details (throttled to 1.0s to avoid console flooding)
        if self.human_lost or self.human_x is None:
            rospy.loginfo_throttle(1.0, f"Robot velocity: {self.current_vel_x:.3f} m/s (Human lost)")
        else:
            rospy.loginfo_throttle(1.0, f"Robot velocity: {self.current_vel_x:.3f} m/s | target: {v_target:.3f} m/s | human_dist: {self.human_x:.2f} m")

        # 4. Publish velocity command (x-direction only)
        twist_msg = Twist()
        twist_msg.linear.x = self.current_vel_x
        twist_msg.linear.y = 0.0
        twist_msg.linear.z = 0.0
        twist_msg.angular.x = 0.0
        twist_msg.angular.y = 0.0
        twist_msg.angular.z = 0.0

        self.cmd_vel_pub.publish(twist_msg)

if __name__ == '__main__':
    rospy.init_node('human_following_control')
    node = HumanFollowingControl()
    rospy.spin()
