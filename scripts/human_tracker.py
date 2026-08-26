#!/usr/bin/env python3
import rospy
import tf
import math
import numpy as np
from jsk_recognition_msgs.msg import PeoplePoseArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Quaternion, PoseStamped, Point
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan

class HumanTracker:
    def __init__(self):
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.target_joint = rospy.get_param('~target_joint', 'Neck')
        self.pose_topic = rospy.get_param('~pose_topic', '/people_pose_estimation_mediapipe/pose')

        self.max_human_speed = rospy.get_param('~max_human_speed', 1.0)

        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.odom_pub = rospy.Publisher('~human_state', Odometry, queue_size=1)
        self.marker_pub = rospy.Publisher('~marker', Marker, queue_size=1)

        self.state = np.zeros(4) # [x, y, vx, vy]
        self.P = np.eye(4)
        self.q_scale = rospy.get_param('~kf_q_scale', 2.0)
        self.r_scale = rospy.get_param('~kf_r_scale', 0.05)
        self.is_initialized = False
        self.last_yaw = 0.0

        self.last_obs_time = rospy.Time.now()
        self.last_body_obs_time = rospy.Time.now()

        self.last_scan = None
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_cb, queue_size=1)

        self.pose_sub = rospy.Subscriber(
            self.pose_topic,
            PeoplePoseArray,
            self.pose_cb,
            queue_size=1
        )

        self.head_pose_sub = rospy.Subscriber(
            '/head_pose_estimation/output/head_pose',
            PoseStamped,
            self.head_pose_cb,
            queue_size=1
        )

        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_cb)
        rospy.loginfo("Human Tracker initialized. Vision-primary tracking active.")

    def scan_cb(self, msg):
        self.last_scan = msg

    def head_pose_cb(self, msg):
        # Fallback to head if body hasn't been seen for 0.5s
        if (rospy.Time.now() - self.last_body_obs_time).to_sec() > 0.5:
            self.process_vision_pose(msg.header, msg.pose.position, is_body=False)

    def pose_cb(self, msg):
        for person in msg.poses:
            target_idx = -1
            if "Neck" in person.limb_names:
                idx = person.limb_names.index("Neck")
                if person.scores[idx] > 0.1: target_idx = idx

            if target_idx == -1 and "Nose" in person.limb_names:
                idx = person.limb_names.index("Nose")
                if person.scores[idx] > 0.1: target_idx = idx

            if target_idx == -1:
                best_score = 0.1
                for i, score in enumerate(person.scores):
                    if score > best_score:
                        best_score = score
                        target_idx = i

            if target_idx != -1:
                self.process_vision_pose(msg.header, person.poses[target_idx].position, is_body=True)
                break

    def process_vision_pose(self, header, position, is_body=True):
        current_time = header.stamp
        if current_time == rospy.Time(0):
            current_time = rospy.Time.now()

        pt = PointStamped()
        pt.header = header
        pt.point = position

        try:
            try:
                self.tf_listener.waitForTransform(self.global_frame, header.frame_id, current_time, rospy.Duration(0.05))
                pt.header.stamp = current_time
            except tf.Exception:
                self.tf_listener.waitForTransform(self.global_frame, header.frame_id, rospy.Time(0), rospy.Duration(0.05))
                pt.header.stamp = rospy.Time(0)

            pt_global = self.tf_listener.transformPoint(self.global_frame, pt)
            target_global = np.array([pt_global.point.x, pt_global.point.y])

            best_z = target_global

            # LiDAR Fusion
            if self.last_scan is not None:
                scan = self.last_scan
                try:
                    self.tf_listener.waitForTransform(scan.header.frame_id, self.global_frame, rospy.Time(0), rospy.Duration(0.05))
                    tg_pt = PointStamped()
                    tg_pt.header.frame_id = self.global_frame
                    tg_pt.point.x = target_global[0]
                    tg_pt.point.y = target_global[1]
                    tg_laser = self.tf_listener.transformPoint(scan.header.frame_id, tg_pt)

                    tx = tg_laser.point.x
                    ty = tg_laser.point.y
                    target_r = math.hypot(tx, ty)
                    target_theta = math.atan2(ty, tx)

                    valid_points = []
                    angle = scan.angle_min
                    for r in scan.ranges:
                        if scan.range_min < r < scan.range_max:
                            if abs(angle - target_theta) < math.radians(20) and abs(r - target_r) < 0.8:
                                valid_points.append((r, angle))
                        angle += scan.angle_increment

                    if valid_points:
                        best_r = min([p[0] for p in valid_points])
                        points_near_min = [p for p in valid_points if abs(p[0] - best_r) < 0.15]
                        avg_r = np.mean([p[0] for p in points_near_min])
                        avg_angle = np.mean([p[1] for p in points_near_min])

                        px = avg_r * math.cos(avg_angle)
                        py = avg_r * math.sin(avg_angle)

                        leg_pt = PointStamped()
                        leg_pt.header.frame_id = scan.header.frame_id
                        leg_pt.point.x = px
                        leg_pt.point.y = py
                        leg_global = self.tf_listener.transformPoint(self.global_frame, leg_pt)
                        best_z = np.array([leg_global.point.x, leg_global.point.y])
                except tf.Exception:
                    pass

            if not self.is_initialized:
                rospy.loginfo("[human_tracker] Initialized tracker at x=%.2f, y=%.2f", best_z[0], best_z[1])
                self.state = np.array([best_z[0], best_z[1], 0.0, 0.0])
                self.P = np.eye(4)
                self.is_initialized = True
                self.last_obs_time = current_time
                if is_body:
                    self.last_body_obs_time = current_time
            else:
                dt = (current_time - self.last_obs_time).to_sec()
                if dt > 0:
                    time_since_last_valid = (rospy.Time.now() - self.last_obs_time).to_sec()

                    F = np.array([
                        [1.0, 0.0, dt, 0.0],
                        [0.0, 1.0, 0.0, dt],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ])
                    pred_state = F @ self.state

                    jump_dist = np.hypot(best_z[0] - pred_state[0], best_z[1] - pred_state[1])
                    max_allowed_jump = 2.0 * dt + 0.5

                    if time_since_last_valid > 3.0:
                        rospy.loginfo("[human_tracker] Re-acquiring target after %.2fs. Resetting state.", time_since_last_valid)
                        self.state = np.array([best_z[0], best_z[1], 0.0, 0.0])
                        self.P = np.eye(4)
                        self.last_obs_time = current_time
                        if is_body:
                            self.last_body_obs_time = current_time
                    elif jump_dist > max_allowed_jump:
                        rospy.logwarn_throttle(1.0, "[human_tracker] Ignored outlier (jump: %.2fm)", jump_dist)
                    else:
                        Q = np.array([
                            [dt**4/4, 0, dt**3/2, 0],
                            [0, dt**4/4, 0, dt**3/2],
                            [dt**3/2, 0, dt**2, 0],
                            [0, dt**3/2, 0, dt**2]
                        ]) * self.q_scale

                        self.state = pred_state
                        self.P = F @ self.P @ F.T + Q

                        H = np.array([
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]
                        ])
                        R = np.eye(2) * self.r_scale

                        y = best_z - H @ self.state
                        S = H @ self.P @ H.T + R
                        K = self.P @ H.T @ np.linalg.inv(S)

                        self.state = self.state + K @ y
                        self.P = (np.eye(4) - K @ H) @ self.P

                        self.last_obs_time = current_time
                        if is_body:
                            self.last_body_obs_time = current_time

            self.publish_state()

        except tf.Exception as e:
            rospy.logwarn_throttle(2.0, "TF Error: %s", str(e))

    def timer_cb(self, event):
        if not self.is_initialized:
            return

        dt = (rospy.Time.now() - self.last_obs_time).to_sec()

        # Try to use LiDAR tracking if vision is lost for a short while
        if dt > 0.2 and self.last_scan is not None:
            pred_x = self.state[0] + self.state[2] * dt
            pred_y = self.state[1] + self.state[3] * dt

            scan = self.last_scan
            try:
                self.tf_listener.waitForTransform(scan.header.frame_id, self.global_frame, rospy.Time(0), rospy.Duration(0.05))
                tg_pt = PointStamped()
                tg_pt.header.frame_id = self.global_frame
                tg_pt.point.x = pred_x
                tg_pt.point.y = pred_y
                tg_laser = self.tf_listener.transformPoint(scan.header.frame_id, tg_pt)

                tx = tg_laser.point.x
                ty = tg_laser.point.y
                target_r = math.hypot(tx, ty)
                target_theta = math.atan2(ty, tx)

                valid_points = []
                angle = scan.angle_min
                for r in scan.ranges:
                    if scan.range_min < r < scan.range_max:
                        if abs(angle - target_theta) < math.radians(20) and abs(r - target_r) < 0.8:
                            valid_points.append((r, angle))
                    angle += scan.angle_increment

                if valid_points:
                    best_r = min([p[0] for p in valid_points])
                    points_near_min = [p for p in valid_points if abs(p[0] - best_r) < 0.15]
                    avg_r = np.mean([p[0] for p in points_near_min])
                    avg_angle = np.mean([p[1] for p in points_near_min])

                    px = avg_r * math.cos(avg_angle)
                    py = avg_r * math.sin(avg_angle)

                    leg_pt = PointStamped()
                    leg_pt.header.frame_id = scan.header.frame_id
                    leg_pt.point.x = px
                    leg_pt.point.y = py
                    leg_global = self.tf_listener.transformPoint(self.global_frame, leg_pt)

                    best_z = np.array([leg_global.point.x, leg_global.point.y])

                    # Update KF with LiDAR measurement
                    F = np.array([
                        [1.0, 0.0, dt, 0.0],
                        [0.0, 1.0, 0.0, dt],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ])
                    pred_state = F @ self.state

                    Q = np.array([
                        [dt**4/4, 0, dt**3/2, 0],
                        [0, dt**4/4, 0, dt**3/2],
                        [dt**3/2, 0, dt**2, 0],
                        [0, dt**3/2, 0, dt**2]
                    ]) * self.q_scale

                    self.P = F @ self.P @ F.T + Q
                    self.state = pred_state

                    H = np.array([
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]
                    ])
                    R = np.eye(2) * self.r_scale

                    y = best_z - H @ self.state
                    S = H @ self.P @ H.T + R
                    K = self.P @ H.T @ np.linalg.inv(S)

                    self.state = self.state + K @ y
                    self.P = (np.eye(4) - K @ H) @ self.P

                    self.last_obs_time = rospy.Time.now()
                    dt = 0.0 # reset dt since we just updated
            except tf.Exception:
                pass

        if dt > 3.0:
            rospy.logwarn_throttle(2.0, "[human_tracker] Target lost for >3.0s. Hiding.")
            self.is_initialized = False
            self.publish_state()
        elif dt > 0.5:
            self.state[2] *= 0.8
            self.state[3] *= 0.8
            self.publish_state()

    def publish_state(self):
        if not self.is_initialized:
            marker = Marker()
            marker.header.frame_id = self.global_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "human_velocity"
            marker.id = 0
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)
            return

        x, y, vx, vy = self.state

        speed = math.hypot(vx, vy)
        if speed > self.max_human_speed:
            scale = self.max_human_speed / speed
            vx *= scale
            vy *= scale
            speed = self.max_human_speed
            self.state[2] = vx
            self.state[3] = vy

        # 速度が遅い（停止中）時は、前回向いていた方向をキープする
        if speed > 0.15:
            self.last_yaw = math.atan2(vy, vx)

        q = tf.transformations.quaternion_from_euler(0, 0, self.last_yaw)

        self.tf_broadcaster.sendTransform((x, y, 0), q, rospy.Time.now(), "human_link", self.global_frame)

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.global_frame
        odom.child_frame_id = "human_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.orientation = Quaternion(*q)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        self.odom_pub.publish(odom)

        marker = Marker()
        marker.header = odom.header
        marker.ns = "human_velocity"
        marker.id = 0

        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0

        display_vx = vx
        display_vy = vy
        if speed <= 0.05:
            # Draw a very small arrow in the last known direction
            display_vx = 0.05 * math.cos(self.last_yaw)
            display_vy = 0.05 * math.sin(self.last_yaw)

        p_start = Point(x=x, y=y, z=0.1)
        p_end = Point(x=x + display_vx * 1.0, y=y + display_vy * 1.0, z=0.1)
        marker.points = [p_start, p_end]

        marker.scale.x = 0.025
        marker.scale.y = 0.075
        marker.scale.z = 0.075

        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        self.marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('human_tracker')
    try:
        HumanTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
