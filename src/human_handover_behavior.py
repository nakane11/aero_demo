#!/usr/bin/env python3
import rospy
import tf
import math
import numpy as np
from jsk_recognition_msgs.msg import PeoplePoseArray
from geometry_msgs.msg import PointStamped, Twist, Point, Quaternion, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from skrobot.models import Aero
from skrobot.interfaces.ros import AeroROSRobotInterface
from skrobot.coordinates import Coordinates

class HumanHandoverBehavior:
    def __init__(self):
        rospy.init_node('human_handover_behavior')
        self.tf_listener = tf.TransformListener()
        
        self.marker_pub = rospy.Publisher('~target_markers', MarkerArray, queue_size=1)
        
        rospy.loginfo("Initializing robot model for full body control...")
        self.robot = Aero()
        self.ri = AeroROSRobotInterface(self.robot)
        self.robot.angle_vector(self.ri.angle_vector())

        self.last_neck_cmd_time = rospy.Time.now()
        
        # State machine variables
        self.state = "WAITING"
        self.target_lhand = None
        self.target_face_map = None
        self.target_wrist_map = None
        self.target_base = None
        self.state_start_time = rospy.Time.now()
        
        self.pose_sub = rospy.Subscriber(
            '/people_pose_estimation_mediapipe/pose',
            PeoplePoseArray,
            self.pose_cb,
            queue_size=1
        )
        
        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)
        
        rospy.loginfo("Human Handover Behavior initialized. Waiting for human...")

    def pose_cb(self, msg):
        if self.state != "WAITING":
            return

        now = rospy.Time.now()
        
        if (now - self.last_neck_cmd_time).to_sec() < 2.5:
            return
            
        for person in msg.poses:
            if "Nose" in person.limb_names:
                idx_nose = person.limb_names.index("Nose")
                if person.scores[idx_nose] > 0.1:
                    nose_pos = person.poses[idx_nose].position
                    
                    has_wrist = False
                    if "RWrist" in person.limb_names:
                        idx_wrist = person.limb_names.index("RWrist")
                        if person.scores[idx_wrist] > 0.1:
                            has_wrist = True
                            wrist_pos = person.poses[idx_wrist].position
                    
                    try:
                        time = rospy.Time(0)
                        self.tf_listener.waitForTransform("base_link", msg.header.frame_id, time, rospy.Duration(0.05))
                        self.tf_listener.waitForTransform("map", msg.header.frame_id, time, rospy.Duration(0.05))
                        
                        pn = PointStamped()
                        pn.header.frame_id = msg.header.frame_id
                        pn.header.stamp = time
                        pn.point = nose_pos
                        pn_base = self.tf_listener.transformPoint("base_link", pn)
                        pn_map = self.tf_listener.transformPoint("map", pn)
                        
                        hx, hy, hz = pn_base.point.x, pn_base.point.y, pn_base.point.z
                        neck_yaw = math.atan2(hy, hx)
                        
                        current_p = self.robot.neck_p_joint.joint_angle()
                        current_y = self.robot.neck_y_joint.joint_angle()
                        
                        if not has_wrist:
                            rospy.loginfo_throttle(2.0, "Nose detected, but RWrist is missing. Looking down...")
                            target_p = 0.3 
                        else:
                            target_p = current_p
                            
                        target_y = np.clip(neck_yaw, -1.5, 1.5)
                        
                        if abs(current_p - target_p) > 0.1 or abs(current_y - target_y) > 0.1:
                            self.robot.angle_vector(self.ri.angle_vector())
                            self.robot.neck_y_joint.joint_angle(target_y)
                            self.robot.neck_p_joint.joint_angle(target_p)
                            self.ri.angle_vector(self.robot.angle_vector(), 1.5)
                            self.last_neck_cmd_time = now
                            return
                        
                        if has_wrist:
                            pw = PointStamped()
                            pw.header.frame_id = msg.header.frame_id
                            pw.header.stamp = time
                            pw.point = wrist_pos
                            pw_base = self.tf_listener.transformPoint("base_link", pw)
                            pw_map = self.tf_listener.transformPoint("map", pw)
                            
                            self.target_face_map = pn_map
                            self.target_wrist_map = pw_map
                            
                            # Calculate hand target
                            wx, wy, wz = pw_base.point.x, pw_base.point.y, pw_base.point.z
                            dist = math.hypot(hx, hy)
                            if dist > 0.01:
                                f_x = -hx / dist
                                f_y = -hy / dist
                                theta_prime = math.atan2(f_y, f_x)
                                
                                target_lhand_x = wx + 0.1 * math.cos(theta_prime)
                                target_lhand_y = wy + 0.1 * math.sin(theta_prime)
                                target_lhand_z = wz + 0.05
                                
                                self.target_lhand = [target_lhand_x, target_lhand_y, target_lhand_z]
                                
                                # Publish markers
                                marker_array = MarkerArray()
                                
                                def create_marker(m_id, color, pos, frame="base_link", m_type=Marker.SPHERE, quat=None):
                                    m = Marker()
                                    m.header.frame_id = frame
                                    m.header.stamp = rospy.Time.now()
                                    m.ns = "targets"
                                    m.id = m_id
                                    m.type = m_type
                                    m.action = Marker.ADD
                                    m.pose.position.x = pos[0]
                                    m.pose.position.y = pos[1]
                                    m.pose.position.z = pos[2]
                                    
                                    if quat is not None:
                                        m.pose.orientation.x = quat[0]
                                        m.pose.orientation.y = quat[1]
                                        m.pose.orientation.z = quat[2]
                                        m.pose.orientation.w = quat[3]
                                    else:
                                        m.pose.orientation.w = 1.0
                                        
                                    if m_type == Marker.ARROW:
                                        m.scale.x = 0.3  # Shaft length
                                        m.scale.y = 0.05 # Shaft width
                                        m.scale.z = 0.05 # Head width
                                    else:
                                        m.scale.x = 0.1
                                        m.scale.y = 0.1
                                        m.scale.z = 0.1
                                        
                                    m.color.r = color[0]
                                    m.color.g = color[1]
                                    m.color.b = color[2]
                                    m.color.a = 1.0
                                    return m
                                    
                                # Calculate target base dynamically in base_link
                                btheta = -2.35  # -135 deg (右斜め後ろ)
                                bx = wx - 0.30
                                by = wy + 0.35
                                
                                # Convert target base to map frame
                                target_pose_base = PoseStamped()
                                target_pose_base.header.frame_id = "base_link"
                                target_pose_base.header.stamp = time
                                target_pose_base.pose.position.x = bx
                                target_pose_base.pose.position.y = by
                                q = tf.transformations.quaternion_from_euler(0, 0, btheta)
                                target_pose_base.pose.orientation = Quaternion(*q)
                                
                                try:
                                    target_pose_map = self.tf_listener.transformPose("map", target_pose_base)
                                    px = target_pose_map.pose.position.x
                                    py = target_pose_map.pose.position.y
                                    mq = target_pose_map.pose.orientation
                                    euler = tf.transformations.euler_from_quaternion([mq.x, mq.y, mq.z, mq.w])
                                    ptheta = euler[2]
                                    self.target_base = [px, py, ptheta]
                                except tf.Exception as e:
                                    rospy.logwarn("Failed to transform target base to map: %s", e)
                                    continue

                                # Base target arrow in map frame
                                q_base = tf.transformations.quaternion_from_euler(0, 0, ptheta)
                                marker_array.markers.append(create_marker(0, (1, 1, 0), [px, py, 0.0], "map", Marker.ARROW, q_base)) # Yellow Arrow: Base Target
                                
                                # Human wrist and hand target in base_link frame
                                marker_array.markers.append(create_marker(1, (0, 1, 0), [wx, wy, wz])) # Green: Wrist
                                marker_array.markers.append(create_marker(2, (0, 0, 1), self.target_lhand)) # Blue: Hand Target
                                
                                self.marker_pub.publish(marker_array)
                                
                                rospy.loginfo("Found target! Target locked. Executing handover sequence...")
                                self.state = "NODDING"
                                self.state_start_time = rospy.Time.now()
                            
                        break # Only track the first valid person
                    except tf.Exception as e:
                        rospy.logwarn_throttle(2.0, f"TF Error: {e}")

    def control_loop(self, event):
        if self.state == "NODDING":
            self.robot.angle_vector(self.ri.angle_vector())
            # Nod down
            self.robot.neck_p_joint.joint_angle(0.4)
            self.ri.angle_vector(self.robot.angle_vector(), 1.0)
            self.ri.wait_interpolation()
            
            # Nod up (look forward)
            self.robot.neck_p_joint.joint_angle(-0.2)
            self.ri.angle_vector(self.robot.angle_vector(), 1.0)
            self.ri.wait_interpolation()
            
            self.state = "MOVING_BASE"
            rospy.loginfo("Nod finished, starting base movement with move_to...")

        elif self.state == "MOVING_BASE":
            px, py, ptheta = self.target_base
            target_coords = Coordinates(pos=[px, py, 0.0])
            target_coords.rotate(ptheta, 'z')
            
            rospy.loginfo("Moving base to x=%.2f, y=%.2f, theta=%.2f in map...", px, py, ptheta)
            self.ri.move_to(target_coords, wait=True, frame_id="map")
            
            self.state = "ADJUSTING_BODY"
            rospy.loginfo("Base movement done. Proceeding to adjust body...")
                
        elif self.state == "ADJUSTING_BODY":
            self.robot.angle_vector(self.ri.angle_vector())
            
            # Use TF to transform targets from map to the new base_link
            try:
                time = rospy.Time(0)
                self.tf_listener.waitForTransform("base_link", "map", time, rospy.Duration(0.5))
                
                self.target_face_map.header.stamp = time
                self.target_wrist_map.header.stamp = time
                
                pn_new_base = self.tf_listener.transformPoint("base_link", self.target_face_map)
                pw_new_base = self.tf_listener.transformPoint("base_link", self.target_wrist_map)
                
                hx, hy, hz = pn_new_base.point.x, pn_new_base.point.y, pn_new_base.point.z
                wx, wy, wz = pw_new_base.point.x, pw_new_base.point.y, pw_new_base.point.z
            except tf.Exception as e:
                rospy.logwarn("Failed to transform targets from map to new base_link: %s", e)
                return
            
            # Recalculate hand target in new frame
            dist = math.hypot(hx, hy)
            if dist > 0.01:
                f_x = -hx / dist
                f_y = -hy / dist
                theta_prime = math.atan2(f_y, f_x)
                self.target_lhand = [
                    wx + 0.1 * math.cos(theta_prime),
                    wy + 0.1 * math.sin(theta_prime),
                    wz + 0.05
                ]
            
            # 1. Bend upper body slightly left
            self.robot.waist_y_joint.joint_angle(0.8) # Twist left
            
            # 2. Torso up/down (lifter) to match hand height
            lifter_amount = np.clip((1.0 - wz) * 1.5, 0.0, 1.0)
            try:
                self.robot.knee_joint.joint_angle(lifter_amount)
                if hasattr(self.robot, 'ankle_joint'):
                    self.robot.ankle_joint.joint_angle(-lifter_amount)
            except AttributeError:
                pass
            
            # 3. Look at human's face (using new coordinates)
            # Look at halfway between face and hand for natural handover gaze
            neck_yaw = math.atan2(hy, hx)
            target_look_z = (hz + wz) / 2.0
            neck_pitch = math.atan2(target_look_z - 1.2, math.hypot(hx, hy))
            self.robot.neck_y_joint.joint_angle(np.clip(neck_yaw, -1.5, 1.5))
            self.robot.neck_p_joint.joint_angle(np.clip(-neck_pitch, -0.3, 0.5))

            rospy.loginfo("Adjusting body posture and gaze first...")
            self.ri.angle_vector(self.robot.angle_vector(), 2.0)
            self.ri.wait_interpolation()
            
            rospy.loginfo("Extending arm...")
            self.robot.angle_vector(self.ri.angle_vector())
            
            target_coords = Coordinates(pos=self.target_lhand)
            
            target_yaw = math.atan2(self.target_lhand[1], self.target_lhand[0])
            target_coords.rotate(target_yaw, 'z')
            target_coords.rotate(-1.57, 'x') # Roll -90 degrees so palm faces UP
            
            # Fallback posture (palm up, natural elbow)
            self.robot.l_shoulder_p_joint.joint_angle(-0.4)
            self.robot.l_shoulder_r_joint.joint_angle(0.2)
            self.robot.l_shoulder_y_joint.joint_angle(0.5)
            self.robot.l_elbow_joint.joint_angle(-1.2) # Bend elbow more
            self.robot.l_wrist_y_joint.joint_angle(0.0)
            self.robot.l_wrist_p_joint.joint_angle(0.2)
            self.robot.l_wrist_r_joint.joint_angle(1.5) # Positive for palm up
            
            try:
                # IK
                res = self.robot.larm.inverse_kinematics(target_coords, rotation_axis='yz')
                if res is False:
                    res = self.robot.larm.inverse_kinematics(target_coords, rotation_axis=False)
            except Exception as e:
                rospy.logwarn(f"IK failed: {e}. Using fallback posture.")

            self.ri.angle_vector(self.robot.angle_vector(), 2.0)
            self.ri.wait_interpolation()
            
            rospy.loginfo("Finished handover behavior sequence.")
            self.state = "DONE"
            
        elif self.state == "DONE":
            pass

if __name__ == '__main__':
    try:
        HumanHandoverBehavior()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
