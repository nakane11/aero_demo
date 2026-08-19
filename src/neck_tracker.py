#!/usr/bin/env python3
import rospy
import tf
import math
from skrobot.models import Aero 
from skrobot.interfaces.ros import AeroROSRobotInterface 

class NeckTracker:
    def __init__(self):
        rospy.init_node('neck_tracker')
        
        self.tf_listener = tf.TransformListener()
        
        rospy.loginfo("Initializing robot model...")
        self.robot_model = Aero() 
        self.ri = AeroROSRobotInterface(self.robot_model) 

        self.robot_model.init_pose()
        self.robot_model.r_wrist_r_joint.joint_angle(0.3)
        self.robot_model.r_shoulder_r_joint.joint_angle(-0.3)
        self.robot_model.neck_p_joint.joint_angle(0.3)
        self.robot_model.neck_y_joint.joint_angle(-1.0)
        self.ri.angle_vector(robot_model.angle_vector(),5)
        self.ri.wait_interpolation()
        # Sync with current robot state
        self.robot_model.angle_vector(self.ri.angle_vector())
        self.last_yaw = self.robot_model.neck_y_joint.joint_angle()
        self.tracking_target_yaw = self.last_yaw
        
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_cb)
        rospy.loginfo("Neck tracker started. Waiting for human_link TF...")
        
    def timer_cb(self, event):
        try:
            time = rospy.Time(0)
            base_frame = "base_link"
            if not self.tf_listener.canTransform(base_frame, "human_link", time):
                if self.tf_listener.canTransform("leg_base_link", "human_link", time):
                    base_frame = "leg_base_link"
                else:
                    return
                
            self.tf_listener.waitForTransform(base_frame, "human_link", time, rospy.Duration(0.05))
            trans, rot = self.tf_listener.lookupTransform(base_frame, "human_link", time)
            
            hx, hy, hz = trans
            
            # Calculate raw target yaw angle to face the human
            raw_target_yaw = math.atan2(hy, hx)
            
            # Clamp the yaw angle to prevent the neck from turning beyond its absolute limits (+/- 1.5 rad)
            raw_target_yaw = max(min(raw_target_yaw, 1.5), -1.5)
            
            # Threshold to prevent constant micro-movements (deadband)
            # If human moves more than 0.2 rad (approx 11.5 deg) from current target, set new target
            if abs(raw_target_yaw - self.tracking_target_yaw) > 0.2:
                self.tracking_target_yaw = raw_target_yaw
            
            # Apply Exponential Smoothing (Low-pass filter) to smoothly move towards the fixed tracking target
            alpha = 0.6 
            self.last_yaw = self.last_yaw + alpha * (self.tracking_target_yaw - self.last_yaw)
            
            # Only send commands if there is still a meaningful difference
            if abs(self.tracking_target_yaw - self.last_yaw) > 0.01:
                self.robot_model.neck_y_joint.joint_angle(self.last_yaw)
                # Pitch further down (increased from 0.1 to 0.4) so the camera sees more of the body
                self.robot_model.neck_p_joint.joint_angle(0.4) 
                
                # Send command to robot with 0.15s duration (slightly longer than 0.1s loop to connect motions smoothly)
                self.ri.angle_vector(self.robot_model.angle_vector(), 0.15)
            
        except tf.Exception as e:
            pass

if __name__ == '__main__':
    try:
        NeckTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
