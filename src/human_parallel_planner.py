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
        # 0.6m to the left (Y = 0.6), since human is on the right of the robot.
        self.target_x = rospy.get_param('~target_x', 0.0)
        self.target_y = rospy.get_param('~target_y', 0.6)

        # State Machine: WAITING (wait for human to come to right side) -> TRACKING (parallel walk)
        self.state = "WAITING"

        # Control gains
        self.kp_pos = rospy.get_param('~kp_pos', 0.8)  # Position feedback gain (lowered for safety)
        self.kp_yaw = rospy.get_param('~kp_yaw', 0.8)  # Yaw feedback gain (lowered for safety)

        # Velocity limits
        self.max_linear_vel = rospy.get_param('~max_linear_vel', 0.5)
        self.max_angular_vel = rospy.get_param('~max_angular_vel', 0.3)
        self.max_accel = rospy.get_param('~max_accel', 0.4)

        self.control_rate = rospy.get_param('~control_rate', 10.0) # Hz

        # State variables
        self.human_global_x = None
        self.human_global_y = None
        self.human_vel_x = 0.0
        self.human_vel_y = 0.0
        self.human_yaw = 0.0
        self.last_update_time = None
        self.start_time = None

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
        if self.start_time is None:
            self.start_time = rospy.Time.now()

        if self.human_global_x is None:
            return

        # 安全のため、起動後10秒間は制御を開始しない
        if (rospy.Time.now() - self.start_time).to_sec() < 10.0:
            self.stop_robot()
            rospy.logwarn_throttle(2.0, "[ParallelPlanner] Waiting for 10 seconds safety delay...")
            return

        # Failsafe: if human is lost for too long (e.g. 0.5 seconds), stop moving immediately
        if (rospy.Time.now() - self.last_update_time).to_sec() > 0.5:
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

        # 1.5 State Machine Check
        if self.state == "WAITING":
            self.stop_robot()
            hx_global = self.human_global_x - robot_x
            hy_global = self.human_global_y - robot_y

            # Transform human position to robot's local frame
            hx_local = hx_global * math.cos(-robot_yaw) - hy_global * math.sin(-robot_yaw)
            hy_local = hx_global * math.sin(-robot_yaw) + hy_global * math.cos(-robot_yaw)

            # Check if human is to the right (hy_local < -0.3) and roughly alongside (-1.0 < hx_local < 1.0)
            if hy_local < -0.3 and -1.0 < hx_local < 1.0:
                rospy.loginfo("[ParallelPlanner] Human arrived on the right side! Switching to TRACKING state.")
                self.state = "TRACKING"
            else:
                rospy.loginfo_throttle(2.0, "[ParallelPlanner] WAITING for human to come to the right side... (local pos: x=%.2f, y=%.2f)", hx_local, hy_local)
                return

        # 2. Calculate the target position for the robot in the global frame
        human_speed = math.hypot(self.human_vel_x, self.human_vel_y)

        # 人間は前進のみで、ロボットの回転も無効化しているため、
        # 人間が後ずさりした時に速度ベクトルから計算されるヨー角が180度反転して
        # 目標位置が右側にフリップする（突っ込んでくる）のを完全に防ぎます。
        current_human_yaw = robot_yaw

        # Rotate the relative target coordinates by the human's heading
        target_global_x = self.human_global_x + self.target_x * math.cos(current_human_yaw) - self.target_y * math.sin(current_human_yaw)
        target_global_y = self.human_global_y + self.target_x * math.sin(current_human_yaw) + self.target_y * math.cos(current_human_yaw)

        # 3. Position Error (Global)
        error_x_global = target_global_x - robot_x
        error_y_global = target_global_y - robot_y

        # 4. Calculate desired robot velocity in GLOBAL frame (Feedforward + Feedback)
        human_speed = math.hypot(self.human_vel_x, self.human_vel_y)

        # Apply deadzone to position error to avoid micro-adjustments
        deadzone_radius = 0.20 # 20cm
        dist_error = math.hypot(error_x_global, error_y_global)

        if dist_error < deadzone_radius:
            # 目標位置から20cm以内なら、微調整をやめて人間の速度(フィードフォワード)のみで滑らかに走る
            fb_x = 0.0
            fb_y = 0.0
        else:
            # 20cm以上離れた場合は、その超過分に対してフィードバックをかける（急加速を防ぐため徐々に強くする）
            scale = (dist_error - deadzone_radius) / dist_error
            fb_x = self.kp_pos * error_x_global * scale
            fb_y = self.kp_pos * error_y_global * scale

        if human_speed < 0.15:
            # 人間が止まった場合、目標位置に追いつくためのフィードバックのみを適用し、フィードフォワードは0にする
            desired_vx_global = fb_x
            desired_vy_global = fb_y
        else:
            # Feedforward: human's velocity + Feedback: Kp * Error (with deadzone)
            desired_vx_global = self.human_vel_x + fb_x
            desired_vy_global = self.human_vel_y + fb_y
        # 5. Transform desired global velocity into robot's LOCAL frame (base_link)
        # Since it's an omni-wheel robot, we can directly command vx and vy.
        cmd_vx = desired_vx_global * math.cos(-robot_yaw) - desired_vy_global * math.sin(-robot_yaw)
        cmd_vy = desired_vx_global * math.sin(-robot_yaw) + desired_vy_global * math.cos(-robot_yaw)

        # 6. Calculate desired angular velocity (Yaw control)
        # 安全のため、現在は回転を無効化（人間は前進のみを前提）
        cmd_wz = 0.0

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
        self.current_cmd.angular.z += np.clip(cmd_wz - self.current_cmd.angular.z, -max_dv * 2.0, max_dv * 2.0)

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
