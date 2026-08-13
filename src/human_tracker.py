#!/usr/bin/env python3
import rospy
import tf
import math
import numpy as np
from jsk_recognition_msgs.msg import PeoplePoseArray
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Quaternion, PoseStamped
from visualization_msgs.msg import Marker

class HumanTracker:
    def __init__(self):
        self.global_frame = rospy.get_param('~global_frame', 'map')
        self.target_joint = rospy.get_param('~target_joint', 'Neck')
        self.pose_topic = rospy.get_param('~pose_topic', '/people_pose_estimation_mediapipe/pose')

        self.tf_listener = tf.TransformListener()
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.odom_pub = rospy.Publisher('~human_state', Odometry, queue_size=1)
        self.marker_pub = rospy.Publisher('~marker', Marker, queue_size=1)

        # Kalman filter initialization
        # State: [x, y, vx, vy]
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1.0 # Initial covariance

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Decrease observation noise, increase process noise (especially for velocity) to make it very snappy
        self.R = np.eye(2) * 0.05
        self.Q = np.diag([0.1, 0.1, 1.0, 1.0])

        self.last_time = rospy.Time.now()
        self.last_obs_time = rospy.Time.now()
        self.last_body_obs_time = rospy.Time.now()
        self.is_initialized = False

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

        # Timer for prediction during occlusion and continuous publishing
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_cb)
        rospy.loginfo("Human Tracker initialized. Tracking joint: %s in frame: %s", self.target_joint, self.global_frame)

    def predict(self, dt):
        F = np.array([[1, 0, dt,  0],
                      [0, 1,  0, dt],
                      [0, 0,  1,  0],
                      [0, 0,  0,  1]])
        self.state = np.dot(F, self.state)
        self.P = np.dot(F, np.dot(self.P, F.T)) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.state)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))

        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        rospy.loginfo_throttle(1.0, "[human_tracker] Filter updated. New state: x=%.2f, y=%.2f, vx=%.2f, vy=%.2f",
                               self.state[0], self.state[1], self.state[2], self.state[3])

    def pose_cb(self, msg):
        # Use system time consistently to avoid negative dt caused by camera/inference latency
        current_time = rospy.Time.now()

        best_z = None
        for person in msg.poses:
            target_idx = -1
            if "Neck" in person.limb_names:
                idx = person.limb_names.index("Neck")
                if person.scores[idx] > 0:
                    target_idx = idx

            if target_idx == -1 and "Nose" in person.limb_names:
                idx = person.limb_names.index("Nose")
                if person.scores[idx] > 0:
                    target_idx = idx

            if target_idx != -1:
                pose = person.poses[target_idx]

                pt = PointStamped()
                pt.header = msg.header
                pt.point = pose.position

                try:
                    self.tf_listener.waitForTransform(self.global_frame, pt.header.frame_id, rospy.Time(0), rospy.Duration(0.1))
                    pt.header.stamp = rospy.Time(0)
                    pt_global = self.tf_listener.transformPoint(self.global_frame, pt)
                    best_z = np.array([pt_global.point.x, pt_global.point.y])
                    break
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.logwarn_throttle(2.0, "TF Error in human_tracker (check if '%s' exists): %s", self.global_frame, str(e))
                    continue

        if best_z is not None:
            if not self.is_initialized:
                rospy.loginfo("[human_tracker] Initializing tracker at x=%.2f, y=%.2f", best_z[0], best_z[1])
                self.state = np.array([best_z[0], best_z[1], 0.0, 0.0])
                self.last_time = current_time
                self.is_initialized = True
            else:
                dt = (current_time - self.last_time).to_sec()
                if dt > 0:
                    self.predict(dt)
                    self.update(best_z)
                    self.last_time = current_time
                    self.last_obs_time = current_time
                    self.last_body_obs_time = current_time
                else:
                    rospy.logwarn_throttle(1.0, "[human_tracker] Received message with dt <= 0 (%.4f). Is the camera frozen or time not updating?", dt)
        else:
            rospy.loginfo_throttle(2.0, "[human_tracker] No valid target (Neck/Nose) found, or TF failed silently in this frame.")

    def head_pose_cb(self, msg):
        current_time = rospy.Time.now()

        # Use face fallback immediately if the primary body tracker is lost (0.1s = approx 1 frame delay)
        time_since_body = (current_time - self.last_body_obs_time).to_sec()
        if time_since_body < 0.1:
            return

        try:
            # Transform face position to global_frame
            self.tf_listener.waitForTransform(self.global_frame, msg.header.frame_id, rospy.Time(0), rospy.Duration(0.1))

            # Create a PointStamped for the face position
            from geometry_msgs.msg import PointStamped
            pt = PointStamped()
            pt.header = msg.header
            pt.header.stamp = rospy.Time(0) # Use latest TF
            pt.point = msg.pose.position

            base_point = self.tf_listener.transformPoint(self.global_frame, pt)

            # Extract 2D floor position
            best_z = np.array([base_point.point.x, base_point.point.y])

            if not self.is_initialized:
                rospy.loginfo("[human_tracker] Initializing tracker from FACE at x=%.2f, y=%.2f", best_z[0], best_z[1])
                self.state = np.array([best_z[0], best_z[1], 0.0, 0.0])
                self.last_time = current_time
                self.is_initialized = True
            else:
                dt = (current_time - self.last_time).to_sec()
                if dt > 0:
                    self.predict(dt)
                    self.update(best_z)
                    self.last_time = current_time
                    self.last_obs_time = current_time

            rospy.logdebug_throttle(1.0, "[human_tracker] Using FACE fallback for tracking.")

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, "TF Error in face fallback: %s", str(e))

    def timer_cb(self, event):
        if not self.is_initialized:
            rospy.loginfo_throttle(2.0, "[human_tracker] Waiting for initialization...")
            return

        current_time = rospy.Time.now()
        dt = (current_time - self.last_time).to_sec()
        time_since_obs = (current_time - self.last_obs_time).to_sec()

        if dt > 0.05 and dt < 2.0:
            self.predict(dt)
            self.last_time = current_time

        # If no real observation for more than 0.5 seconds, forcefully decay velocity to prevent fly-away
        if time_since_obs > 0.5:
            self.state[2] *= 0.5
            self.state[3] *= 0.5

        self.publish_state()

    def publish_state(self):
        x, y, vx, vy = self.state

        speed = math.hypot(vx, vy)
        yaw = 0.0
        if speed > 0.01: # Lowered threshold to see minor movements
            yaw = math.atan2(vy, vx)

        q = tf.transformations.quaternion_from_euler(0, 0, yaw)

        self.tf_broadcaster.sendTransform(
            (x, y, 0),
            q,
            rospy.Time.now(),
            "human_link",
            self.global_frame
        )

        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.global_frame
        odom.child_frame_id = "human_link"

        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = Quaternion(*q)

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy

        self.odom_pub.publish(odom)

        # Publish Marker for visualization (arrow pointing in velocity direction)
        marker = Marker()
        marker.header = odom.header
        marker.ns = "human_velocity"
        marker.id = 0

        if speed > 0.05:
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            # Reset pose to identity when using points
            marker.pose.orientation.w = 1.0

            # Start point (tail) is exactly the human's position, raised slightly (z=0.1)
            from geometry_msgs.msg import Point
            p_start = Point(x=x, y=y, z=0.1)
            # End point (head) is scaled by velocity (1.0 means 1 m/s = 1 meter length), raised slightly
            p_end = Point(x=x + vx * 1.0, y=y + vy * 1.0, z=0.1)

            marker.points = [p_start, p_end]

            # For points-based arrows: x=shaft diameter, y=head diameter, z=head length (halved size)
            marker.scale.x = 0.025
            marker.scale.y = 0.075
            marker.scale.z = 0.075

            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            self.marker_pub.publish(marker)
        else:
            # If speed is too slow, hide the arrow to avoid drawing a 0-length arrow
            marker.action = Marker.DELETE
            self.marker_pub.publish(marker)

if __name__ == '__main__':
    rospy.init_node('human_tracker')
    try:
        HumanTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
