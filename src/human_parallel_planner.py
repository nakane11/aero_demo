#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
import tf
import math
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

class HumanParallelPlanner:
    def __init__(self):
        rospy.loginfo("Initializing HumanParallelPlanner node...")

        # Parameters
        self.odom_topic = rospy.get_param('~human_state_topic', '/human_tracker/human_state')
        self.cmd_vel_topic = rospy.get_param('~cmd_vel_topic', '/cmd_vel')
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')

        # Target relative position in human's coordinate frame (X: forward, Y: left)
        # Default: 1.0m to the right (Y = -1.0)
        self.target_x = rospy.get_param('~target_x', 0.0)
        self.target_y = rospy.get_param('~target_y', -1.0)

        # Control gains
        self.kp_pos = rospy.get_param('~kp_pos', 0.4)  # Position feedback gain (lowered for safety)
        self.kp_yaw = rospy.get_param('~kp_yaw', 0.8)  # Yaw feedback gain (lowered for safety)

        # Velocity limits
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 0.4)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 0.3)
        self.max_accel = rospy.get_param('~max_accel', 0.2)

        self.control_rate = rospy.get_param('~control_rate', 10.0) # Hz

        # State variables
        self.human_global_x = None
        self.human_global_y = None
        self.human_vel_x = 0.0
        self.human_vel_y = 0.0
        self.human_yaw = 0.0
        self.last_update_time = None

        self.current_cmd = Twist()

        # TF Listener for robot's own position
        self.tf_listener = tf.TransformListener()

        # Publisher & Subscriber
        self.cmd_vel_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.human_state_cb, queue_size=1)

        # Control loop timer
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)

        rospy.loginfo("HumanParallelPlanner initialized. Target relative position: X=%.2f, Y=%.2f", self.target_x, self.target_y)

    def human_state_cb(self, msg):
        # Extract human position and velocity from Odometry (in global_frame)
        self.human_global_x = msg.pose.pose.position.x
        self.human_global_y = msg.pose.pose.position.y

        self.human_vel_x = msg.twist.twist.linear.x
        self.human_vel_y = msg.twist.twist.linear.y

        # Calculate human yaw based on velocity vector
        speed = math.hypot(self.human_vel_x, self.human_vel_y)
        if speed > 0.01:
            self.human_yaw = math.atan2(self.human_vel_y, self.human_vel_x)

        self.last_update_time = rospy.Time.now()

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def control_loop(self, event):
        if self.human_global_x is None:
            return

        # Failsafe: if human is lost for too long (e.g. 2 seconds), stop moving
        if (rospy.Time.now() - self.last_update_time).to_sec() > 2.0:
            self.stop_robot()
            rospy.logwarn_throttle(2.0, "[ParallelPlanner] Human data timeout. Stopping robot.")
            return

        # 1. Get robot's current position and yaw in the global frame
        try:
            # We get the transform from map -> base_link
            (trans, rot) = self.tf_listener.lookupTransform(self.global_frame, self.base_frame, rospy.Time(0))
            robot_x = trans[0]
            robot_y = trans[1]
            _, _, robot_yaw = tf.transformations.euler_from_quaternion(rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, "[ParallelPlanner] Waiting for TF (%s -> %s): %s", self.global_frame, self.base_frame, str(e))
            self.stop_robot()
            return

        # 2. Calculate the target position for the robot in the global frame
        # Rotate the relative target coordinates by the human's heading
        target_global_x = self.human_global_x + self.target_x * math.cos(self.human_yaw) - self.target_y * math.sin(self.human_yaw)
        target_global_y = self.human_global_y + self.target_x * math.sin(self.human_yaw) + self.target_y * math.cos(self.human_yaw)

        # 3. Position Error (Global)
        error_x_global = target_global_x - robot_x
        error_y_global = target_global_y - robot_y

        # 4. Calculate desired robot velocity in GLOBAL frame (Feedforward + Feedback)
        human_speed = math.hypot(self.human_vel_x, self.human_vel_y)

        # Deadband threshold: if human speed is very small (e.g. < 0.15 m/s), assume they are standing still
        if human_speed < 0.15:
            # If human stops, disable feedback and freeze
            desired_vx_global = 0.0
            desired_vy_global = 0.0
        else:
            # Feedforward: human's velocity + Feedback: Kp * Error
            desired_vx_global = self.human_vel_x + self.kp_pos * error_x_global
            desired_vy_global = self.human_vel_y + self.kp_pos * error_y_global

        # 5. Transform desired global velocity into robot's LOCAL frame (base_link)
        # Since it's an omni-wheel robot, we can directly command vx and vy.
        cmd_vx = desired_vx_global * math.cos(-robot_yaw) - desired_vy_global * math.sin(-robot_yaw)
        cmd_vy = desired_vx_global * math.sin(-robot_yaw) + desired_vy_global * math.cos(-robot_yaw)

        # 6. Calculate desired angular velocity (Yaw control)
        if human_speed < 0.15:
            cmd_wz = 0.0
        else:
            # For side-by-side walking, face the same direction as the human
            target_yaw = self.human_yaw
            error_yaw = self.normalize_angle(target_yaw - robot_yaw)
            cmd_wz = self.kp_yaw * error_yaw

        # 7. Apply limits (Velocity and Acceleration constraints)
        dt = 1.0 / self.control_rate
        max_dv = self.max_accel * dt

        # Clip linear velocity absolute values
        speed = math.hypot(cmd_vx, cmd_vy)
        if speed > self.max_linear_vel:
            cmd_vx = (cmd_vx / speed) * self.max_linear_vel
            cmd_vy = (cmd_vy / speed) * self.max_linear_vel

        cmd_wz = np.clip(cmd_wz, -self.max_angular_vel, self.max_angular_vel)

        # Apply acceleration limits smoothly
        self.current_cmd.linear.x += np.clip(cmd_vx - self.current_cmd.linear.x, -max_dv, max_dv)
        self.current_cmd.linear.y += np.clip(cmd_vy - self.current_cmd.linear.y, -max_dv, max_dv)
        #self.current_cmd.angular.z += np.clip(cmd_wz - self.current_cmd.angular.z, -max_dv * 2.0, max_dv * 2.0) # Allow faster angular acceleration

        # Publish command
        self.cmd_vel_pub.publish(self.current_cmd)

        rospy.loginfo_throttle(0.5, "[ParallelPlanner - ACTIVE] Cmd -> vx: %+.2f, vy: %+.2f, wz: %+.2f",
                                self.current_cmd.linear.x, self.current_cmd.linear.y, self.current_cmd.angular.z)

        rospy.logdebug_throttle(1.0, "[ParallelPlanner] Target:(%.2f, %.2f) Error:(%.2f, %.2f)",
                                target_global_x, target_global_y, error_x_global, error_y_global)

    def stop_robot(self):
        self.current_cmd = Twist()
        # Publish command
        self.cmd_vel_pub.publish(self.current_cmd)
        rospy.loginfo_throttle(2.0, "[ParallelPlanner - ACTIVE] Robot STOP commanded.")


if __name__ == '__main__':
    rospy.init_node('human_parallel_planner')
    try:
        HumanParallelPlanner()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
