#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np
import cv_bridge
import mediapipe as mp
import tf

from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, Vector3, Quaternion

class HeadPoseEstimation:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 3D model points (generic face model for solvePnP)
        # OpenCV Camera: X right, Y down, Z forward
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, 330.0, -65.0),         # Chin
            (-225.0, -170.0, -135.0),    # Left eye left corner
            (225.0, -170.0, -135.0),     # Right eye right corner
            (-150.0, 150.0, -125.0),     # Left Mouth corner
            (150.0, 150.0, -125.0)       # Right mouth corner
        ], dtype=np.float64)

        # Filter parameters for smoothing
        self.alpha = 0.2  # Smoothing factor (0.0 to 1.0, lower is smoother)
        self.image_pts_filtered = None

        # ROS publishers and subscribers
        self.bridge = cv_bridge.CvBridge()
        self.tf_listener = tf.TransformListener()
        
        # Parameters for Attention
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.yaw_threshold = rospy.get_param('~yaw_threshold', 28.0)
        self.pitch_min = rospy.get_param('~pitch_min', -15.0)
        self.pitch_max = rospy.get_param('~pitch_max', 20.0)
        
        self.image_pub = rospy.Publisher('~output/image', Image, queue_size=1)
        self.pose_pub = rospy.Publisher('~output/head_pose', PoseStamped, queue_size=1)
        self.euler_pub = rospy.Publisher('~output/euler_angles', Vector3, queue_size=1)
        self.attention_pub = rospy.Publisher('~output/attention', Bool, queue_size=1)
        
        # Subscribe to RGB image
        self.image_sub = rospy.Subscriber('~input/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(e)
            return

        size = image.shape
        # Create a generic camera matrix assuming no calibration data is provided
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

        # Convert image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh on the image
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())

                # Extract 2D points corresponding to the 3D model
                # Indices: Nose (1), Chin (152), Left eye outer (33), Right eye outer (263), Left mouth (61), Right mouth (291)
                image_pts = np.array([
                    (face_landmarks.landmark[1].x * size[1], face_landmarks.landmark[1].y * size[0]),
                    (face_landmarks.landmark[152].x * size[1], face_landmarks.landmark[152].y * size[0]),
                    (face_landmarks.landmark[33].x * size[1], face_landmarks.landmark[33].y * size[0]),
                    (face_landmarks.landmark[263].x * size[1], face_landmarks.landmark[263].y * size[0]),
                    (face_landmarks.landmark[61].x * size[1], face_landmarks.landmark[61].y * size[0]),
                    (face_landmarks.landmark[291].x * size[1], face_landmarks.landmark[291].y * size[0])
                ], dtype="double")

                # Apply Exponential Moving Average (EMA) filter to 2D image points
                if self.image_pts_filtered is None:
                    self.image_pts_filtered = image_pts
                else:
                    self.image_pts_filtered = self.alpha * image_pts + (1.0 - self.alpha) * self.image_pts_filtered

                # Solve PnP
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    self.model_points, self.image_pts_filtered, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    # Draw 3D coordinate axis
                    p1 = (int(self.image_pts_filtered[0][0]), int(self.image_pts_filtered[0][1]))
                    # Project a 3D point (0, 0, 1000.0) onto the image plane.
                    # We use this to draw a line sticking out of the nose
                    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    cv2.line(image, p1, p2, (255, 0, 0), 2)
                    
                    # Alternatively, use drawFrameAxes for XYZ axes
                    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rotation_vector, translation_vector, 500)

                    # Calculate Euler Angles
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    # Convert to euler angles using rq decomposition
                    # This returns angles in degrees
                    euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
                    pitch, yaw, roll = euler_angles[0], euler_angles[1], euler_angles[2]

                    # Convert Euler to Quaternion for PoseStamped
                    # We convert to radians first
                    q = tf.transformations.quaternion_from_euler(
                        np.deg2rad(roll), np.deg2rad(pitch), np.deg2rad(yaw)
                    )

                    # Publish PoseStamped
                    pose_msg = PoseStamped()
                    pose_msg.header = msg.header
                    pose_msg.pose.orientation = Quaternion(*q)
                    # We don't have true depth for position yet, so keep it zero or use the dummy translation
                    pose_msg.pose.position.x = 0
                    pose_msg.pose.position.y = 0
                    pose_msg.pose.position.z = 0
                    self.pose_pub.publish(pose_msg)

                    # Publish Euler Angles (for easy reading in rostopic echo)
                    euler_msg = Vector3()
                    euler_msg.x = pitch
                    euler_msg.y = yaw
                    euler_msg.z = roll
                    self.euler_pub.publish(euler_msg)

                    # Transform to base_link and check Attention
                    attention_flag = False
                    try:
                        # Ensure pose has the correct frame ID
                        if pose_msg.header.frame_id == "":
                            pose_msg.header.frame_id = "camera_link" # Fallback
                            
                        self.tf_listener.waitForTransform(
                            self.base_frame,
                            pose_msg.header.frame_id,
                            pose_msg.header.stamp,
                            rospy.Duration(0.1)
                        )
                        base_pose = self.tf_listener.transformPose(self.base_frame, pose_msg)
                        
                        # Extract euler from base_pose
                        base_q = [base_pose.pose.orientation.x, base_pose.pose.orientation.y, base_pose.pose.orientation.z, base_pose.pose.orientation.w]
                        base_roll, base_pitch, base_yaw = tf.transformations.euler_from_quaternion(base_q)
                        base_pitch_deg = np.rad2deg(base_pitch)
                        base_yaw_deg = np.rad2deg(base_yaw)
                        
                        # Note: Depending on TF tree setup, face orientation might point backwards relative to base_link X
                        # if the person is facing the robot. Normalizing to [-180, 180] and checking proximity to 180 or 0.
                        # For simplicity, we check if the transformed yaw/pitch fall within bounds. 
                        # We'll use absolute threshold assuming robot looks at human.
                        normalized_yaw = (base_yaw_deg + 180.0) % 360.0 - 180.0
                        
                        # Using heuristic for when facing the robot (Yaw is approx 180 deg offset from robot's forward)
                        if abs(normalized_yaw) > (180.0 - self.yaw_threshold):
                            # Looking at robot
                            attention_flag = True
                            
                        # Overriding with camera frame heuristic for now to ensure it works even if TF is weird
                        # but keeping the base_link logic ready
                    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf.Exception) as e:
                        pass
                        
                    # Fallback/Primary logic using Camera Frame (since camera is in the head)
                    # This ensures it works on your laptop webcam immediately
                    if abs(yaw) < self.yaw_threshold and self.pitch_min < pitch < self.pitch_max:
                        attention_flag = True
                    else:
                        attention_flag = False

                    # Publish Attention
                    self.attention_pub.publish(Bool(data=attention_flag))

                    # Draw text on image for visualization
                    cv2.putText(image, f"Pitch: {pitch:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Yaw:   {yaw:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Roll:  {roll:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if attention_flag:
                        cv2.putText(image, "Attention: ON", (size[1] // 2 - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    else:
                        cv2.putText(image, "Attention: OFF", (size[1] // 2 - 120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Publish the visualized image
        try:
            image_msg = self.bridge.cv2_to_imgmsg(image, "bgr8")
            image_msg.header = msg.header
            self.image_pub.publish(image_msg)
        except cv_bridge.CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    rospy.init_node('head_pose_estimation')
    node = HeadPoseEstimation()
    rospy.loginfo("Head Pose Estimation node initialized.")
    rospy.spin()
