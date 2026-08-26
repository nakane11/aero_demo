#!/usr/bin/env python3
import rospy
import tf
import math
import numpy as np
from jsk_recognition_msgs.msg import PeoplePoseArray
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import MarkerArray
from skrobot.models import Aero
from skrobot.interfaces.ros import AeroROSRobotInterface
from skrobot.coordinates import Coordinates

import palm_plane
from palm_plane import EMBED_DEPTH

# Reference point for "where the human is", used only for gaze.  Nose is
# preferred but was published in just 26.5% of a 34-frame sample, while
# Neck reached 100%, so fall back to Neck rather than dropping the frame.
HUMAN_REF_LIMBS = ('Nose', 'Neck')


class HumanPalmContactBehavior:
    def __init__(self):
        rospy.init_node('human_palm_contact_behavior')
        self.tf_listener = tf.TransformListener()

        self.marker_pub = rospy.Publisher('~target_markers', MarkerArray, queue_size=1)

        rospy.loginfo("Initializing robot model for full body control...")
        self.robot = Aero()
        self.ri = AeroROSRobotInterface(self.robot)
        self.robot.angle_vector(self.ri.angle_vector())

        self.last_neck_cmd_time = rospy.Time.now()

        # State machine variables
        self.state = "WAITING"
        self.target_palm_pos = None       # approach target, base_link frame
        self.target_palm_rot = None       # 3x3 rotation matrix, base_link frame
        self.target_palm_center = None    # for gaze/lifter, base_link frame
        self.target_palm_normal = None    # unit vector, base_link frame
        self.state_start_time = rospy.Time.now()

        self.pose_sub = rospy.Subscriber(
            '/people_pose_estimation_mediapipe/pose',
            PeoplePoseArray,
            self.pose_cb,
            queue_size=1
        )

        self.timer = rospy.Timer(rospy.Duration(0.1), self.control_loop)

        rospy.loginfo("Human Palm Contact Behavior initialized. Waiting for human...")
        rospy.loginfo("Note: requires the pose estimator's hand tracking "
                       "(~hand/enable:=true) and depth (~with_depth:=true) to be on.")

    def pose_cb(self, msg):
        if self.state != "WAITING":
            return

        now = rospy.Time.now()

        if (now - self.last_neck_cmd_time).to_sec() < 2.5:
            return

        for person in msg.poses:
            human_ref = None
            for limb in HUMAN_REF_LIMBS:
                if limb in person.limb_names:
                    j = person.limb_names.index(limb)
                    if person.scores[j] > 0.1:
                        human_ref = person.poses[j].position
                        break
            if human_ref is None:
                continue

            try:
                time = rospy.Time(0)
                self.tf_listener.waitForTransform(
                    "base_link", msg.header.frame_id, time, rospy.Duration(0.05))

                def to_base(point_msg):
                    p = PointStamped()
                    p.header.frame_id = msg.header.frame_id
                    p.header.stamp = time
                    p.point = point_msg
                    p_base = self.tf_listener.transformPoint("base_link", p)
                    return np.array([p_base.point.x, p_base.point.y, p_base.point.z])

                # Fit the palm plane to whichever palm landmarks arrived (see
                # palm_plane.py for why this is a least-squares fit and not a
                # single cross product over a fixed quadruple).
                palm_points = palm_plane.collect_palm_points(
                    person, hand='R', min_score=0.1, transform=to_base)
                plane = palm_plane.fit_palm_plane(palm_points)
                has_palm = plane is not None

                hx, hy, hz = to_base(human_ref)
                neck_yaw = math.atan2(hy, hx)

                current_p = self.robot.neck_p_joint.joint_angle()
                current_y = self.robot.neck_y_joint.joint_angle()

                if not has_palm:
                    rospy.loginfo_throttle(
                        2.0, "Human detected, but the palm plane could not be "
                        "fitted (%d of %s landmarks). Looking down...",
                        len(palm_points), list(palm_plane.PLANE_LANDMARKS))
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

                if not has_palm:
                    continue

                self.target_palm_pos = palm_plane.contact_target(plane).tolist()
                self.target_palm_rot = plane.rot
                self.target_palm_center = plane.center.tolist()
                self.target_palm_normal = plane.normal.tolist()

                # Same visualisation the standalone palm_plane_visualizer
                # publishes, so what you verified in RViz is what the robot
                # is about to reach for.
                marker_array = palm_plane.palm_plane_markers(
                    plane, "base_link", stamp=now, ns="targets",
                    label="target")
                marker_array.markers.extend(palm_plane.palm_landmark_markers(
                    palm_points, "base_link", stamp=now, ns="targets",
                    used=plane.used).markers)
                self.marker_pub.publish(marker_array)

                rospy.loginfo(
                    "Found palm! used=%s rms=%.1fmm. Target locked. "
                    "Executing contact sequence...",
                    plane.used, plane.rms * 1000.0)
                self.state = "NODDING"
                self.state_start_time = rospy.Time.now()

                break  # Only track the first valid person
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

            self.state = "REACHING"
            rospy.loginfo("Nod finished, reaching toward the palm...")

        elif self.state == "REACHING":
            self.robot.angle_vector(self.ri.angle_vector())

            cx, cy, cz = self.target_palm_center

            # Torso up/down (lifter) to roughly match hand height. This is
            # just a seed pose -- the whole-body IK below (larm_whole_body)
            # is free to further adjust the lifter *and* the waist yaw/pitch
            # joints (waist_y_joint, waist_p_joint) to reach low targets,
            # e.g. a seated person's hand, without moving the base.
            lifter_amount = np.clip((1.0 - cz) * 1.5, 0.0, 1.0)
            try:
                self.robot.knee_joint.joint_angle(lifter_amount)
                if hasattr(self.robot, 'ankle_joint'):
                    self.robot.ankle_joint.joint_angle(-lifter_amount)
            except AttributeError:
                pass

            # Look at the palm being touched.
            neck_yaw = math.atan2(cy, cx)
            neck_pitch = math.atan2(cz - 1.2, math.hypot(cx, cy))
            self.robot.neck_y_joint.joint_angle(np.clip(neck_yaw, -1.5, 1.5))
            self.robot.neck_p_joint.joint_angle(np.clip(-neck_pitch, -0.3, 0.5))

            rospy.loginfo("Adjusting posture and gaze first...")
            self.ri.angle_vector(self.robot.angle_vector(), 2.0)
            self.ri.wait_interpolation()

            rospy.loginfo("Extending arm toward the palm...")
            self.robot.angle_vector(self.ri.angle_vector())

            target_coords = Coordinates(pos=self.target_palm_pos, rot=self.target_palm_rot)

            # Fallback posture (natural elbow position) used as the IK seed.
            self.robot.l_shoulder_p_joint.joint_angle(-0.4)
            self.robot.l_shoulder_r_joint.joint_angle(0.2)
            self.robot.l_shoulder_y_joint.joint_angle(0.5)
            self.robot.l_elbow_joint.joint_angle(-1.2)
            self.robot.l_wrist_y_joint.joint_angle(0.0)
            self.robot.l_wrist_p_joint.joint_angle(0.2)
            self.robot.l_wrist_r_joint.joint_angle(1.5)

            try:
                res = self.robot.larm_whole_body.inverse_kinematics(target_coords, rotation_axis='yz')
                if res is False:
                    res = self.robot.larm_whole_body.inverse_kinematics(target_coords, rotation_axis=False)
            except Exception as e:
                rospy.logwarn(f"IK failed: {e}. Using fallback posture.")

            self.ri.angle_vector(self.robot.angle_vector(), 2.0)
            self.ri.wait_interpolation()

            # Second, slower motion: press past the approach pose so the
            # hand actually sinks into the palm rather than stopping just
            # short of it.
            rospy.loginfo("Pressing into the palm...")
            self.robot.angle_vector(self.ri.angle_vector())

            center = np.array(self.target_palm_center)
            normal = np.array(self.target_palm_normal)
            embed_pos = (center - normal * EMBED_DEPTH).tolist()
            embed_coords = Coordinates(pos=embed_pos, rot=self.target_palm_rot)

            try:
                res = self.robot.larm_whole_body.inverse_kinematics(embed_coords, rotation_axis='yz')
                if res is False:
                    res = self.robot.larm_whole_body.inverse_kinematics(embed_coords, rotation_axis=False)
            except Exception as e:
                rospy.logwarn(f"IK failed: {e}. Staying at the approach pose.")

            self.ri.angle_vector(self.robot.angle_vector(), 1.5)
            self.ri.wait_interpolation()

            rospy.loginfo("Finished palm contact behavior sequence.")
            self.state = "DONE"

        elif self.state == "DONE":
            pass


if __name__ == '__main__':
    try:
        HumanPalmContactBehavior()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
