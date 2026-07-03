#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import rospy
import cv2
import numpy as np
import cv_bridge
import message_filters
import mediapipe as mp
import matplotlib
matplotlib.use('Agg') # Prevent GUI issues
import matplotlib.cm

from jsk_topic_tools import ConnectionBasedTransport
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion, Vector3
from jsk_recognition_msgs.msg import PeoplePoseArray, PeoplePose, HumanSkeletonArray, HumanSkeleton, Segment
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

class PeoplePoseEstimationMediaPipe(ConnectionBasedTransport):
    limb_sequence = [[ 2,  1], [ 1, 16], [ 1, 15], [ 6, 18], [ 3, 17],
                     [ 2,  3], [ 2,  6], [ 3,  4], [ 4,  5], [ 6,  7],
                     [ 7,  8], [ 2,  9], [ 9, 10], [10, 11], [ 2, 12],
                     [12, 13], [13, 14], [15, 17], [16, 18]]

    index2limbname = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                      "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
                      "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                      "LEye", "REar", "LEar", "Bkg"]

    index2handname = ["RHand{}".format(i) for i in range(21)] + \
                     ["LHand{}".format(i) for i in range(21)]

    hand_sequence = [[0, 1],   [1, 2],   [2, 3],   [3, 4],
                     [0, 5],   [5, 6],   [6, 7],   [7, 8],
                     [0, 9],   [9, 10],  [10, 11], [11, 12],
                     [0, 13],  [13, 14], [14, 15], [15, 16],
                     [0, 17],  [17, 18], [18, 19], [19, 20],]

    def __init__(self):
        super(self.__class__, self).__init__()
        self.with_depth = rospy.get_param('~with_depth', False)
        self.use_hand = rospy.get_param('~hand/enable', False)
        self.min_detection_confidence = rospy.get_param('~min_detection_confidence', 0.5)
        self.min_tracking_confidence = rospy.get_param('~min_tracking_confidence', 0.5)
        self.min_visibility = rospy.get_param('~min_visibility', 0.5)
        self.min_joints = rospy.get_param('~min_joints', 6)
        self.max_z_diff = rospy.get_param('~max_z_diff', 1.0)

        # Initialize MediaPipe Solutions
        if self.use_hand:
            self.holistic = mp.solutions.holistic.Holistic(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
        else:
            self.pose = mp.solutions.pose.Pose(
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )

        rospy.on_shutdown(self.clean_up)

        # ROS Advertisements
        self.image_pub = self.advertise('~output', Image, queue_size=1)
        self.pose_pub = self.advertise('~pose', PeoplePoseArray, queue_size=1)
        self.sub_info = None

        if self.with_depth is True:
            self.pose_2d_pub = self.advertise('~pose_2d', PeoplePoseArray, queue_size=1)
            self.skeleton_pub = self.advertise('~skeleton', HumanSkeletonArray, queue_size=1)
            self.marker_pub = self.advertise('~markers', MarkerArray, queue_size=1)

    def clean_up(self):
        if self.use_hand:
            self.holistic.close()
        else:
            self.pose.close()

    @property
    def visualize(self):
        return self.image_pub.get_num_connections() > 0

    def subscribe(self):
        if self.with_depth:
            queue_size = rospy.get_param('~queue_size', 10)
            sub_img = message_filters.Subscriber(
                '~input', Image, queue_size=1, buff_size=2**24)
            sub_depth = message_filters.Subscriber(
                '~input/depth', Image, queue_size=1, buff_size=2**24)
            self.subs = [sub_img, sub_depth]

            sync_cam_info = rospy.get_param("~sync_camera_info", False)
            if sync_cam_info:
                sub_info = message_filters.Subscriber(
                    '~input/info', CameraInfo, queue_size=1, buff_size=2**24)
                self.subs.append(sub_info)
            else:
                self.sub_info = rospy.Subscriber(
                    '~input/info', CameraInfo, self._cb_cam_info)

            if rospy.get_param('~approximate_sync', True):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            
            if sync_cam_info:
                sync.registerCallback(self._cb_with_depth_info)
            else:
                self.camera_info_msg = None
                sync.registerCallback(self._cb_with_depth)
        else:
            sub_img = rospy.Subscriber(
                '~input', Image, self._cb, queue_size=1, buff_size=2**24)
            self.subs = [sub_img]

    def unsubscribe(self):
        for sub in self.subs:
            sub.unregister()
        if self.sub_info is not None:
            self.sub_info.unregister()
            self.sub_info = None

    def _cb_cam_info(self, msg):
        self.camera_info_msg = msg
        self.sub_info.unregister()
        self.sub_info = None
        rospy.loginfo("Received camera info")

    def _cb_with_depth(self, img_msg, depth_msg):
        if self.camera_info_msg is None:
            return
        self._cb_with_depth_info(img_msg, depth_msg, self.camera_info_msg)

    def _cb_with_depth_info(self, img_msg, depth_msg, camera_info_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        depth_img = br.imgmsg_to_cv2(depth_msg, 'passthrough')
        if depth_msg.encoding == '16UC1':
            depth_img = np.asarray(depth_img, dtype=np.float32)
            depth_img /= 1000  # convert metric: mm -> m
        elif depth_msg.encoding != '32FC1':
            rospy.logerr('Unsupported depth encoding: %s' % depth_msg.encoding)

        people_joint_positions = self.estimate(img)

        people_pose_msg = PeoplePoseArray()
        people_pose_msg.header = img_msg.header
        people_pose_2d_msg = self._create_2d_people_pose_array_msgs(
            people_joint_positions,
            img_msg.header)
        skeleton_msgs = HumanSkeletonArray(header=img_msg.header)

        # calculate xyz-position
        fx = camera_info_msg.K[0]
        fy = camera_info_msg.K[4]
        cx = camera_info_msg.K[2]
        cy = camera_info_msg.K[5]
        for person_joint_positions in people_joint_positions:
            pose_msg = PeoplePose()
            skeleton_msg = HumanSkeleton(header=img_msg.header)
            for joint_pos in person_joint_positions:
                if joint_pos['score'] < 0:
                    continue
                if 0 <= joint_pos['y'] < depth_img.shape[0] and\
                   0 <= joint_pos['x'] < depth_img.shape[1]:
                    z = float(depth_img[int(joint_pos['y'])][int(joint_pos['x'])])
                else:
                    continue
                if np.isnan(z) or z <= 0:
                    continue
                x = (joint_pos['x'] - cx) * z / fx
                y = (joint_pos['y'] - cy) * z / fy
                pose_msg.limb_names.append(joint_pos['limb'])
                pose_msg.scores.append(joint_pos['score'])
                pose_msg.poses.append(Pose(position=Point(x=x, y=y, z=z),
                                           orientation=Quaternion(w=1)))
            # Filter out non-human detections (Method 2 & Method 3)
            z_values = [p.position.z for p in pose_msg.poses]
            if len(pose_msg.poses) < self.min_joints:
                continue
            if len(z_values) > 0:
                z_diff = max(z_values) - min(z_values)
                if z_diff > self.max_z_diff:
                    continue

            people_pose_msg.poses.append(pose_msg)

            for i, conn in enumerate(self.limb_sequence):
                j1_name = self.index2limbname[conn[0] - 1]
                j2_name = self.index2limbname[conn[1] - 1]
                if j1_name not in pose_msg.limb_names \
                        or j2_name not in pose_msg.limb_names:
                    continue
                j1_index = pose_msg.limb_names.index(j1_name)
                j2_index = pose_msg.limb_names.index(j2_name)
                bone_name = '{}->{}'.format(j1_name, j2_name)
                bone = Segment(
                    start_point=pose_msg.poses[j1_index].position,
                    end_point=pose_msg.poses[j2_index].position)
                skeleton_msg.bones.append(bone)
                skeleton_msg.bone_names.append(bone_name)
            skeleton_msgs.skeletons.append(skeleton_msg)

        # calculate MarkerArray for RViz visualization
        marker_array = MarkerArray()
        try:
            cmap = matplotlib.colormaps.get_cmap('hsv')
        except AttributeError:
            cmap = matplotlib.cm.get_cmap('hsv')

        # Add joint spheres
        marker_id = 0
        for person_idx, pose_msg in enumerate(people_pose_msg.poses):
            for joint_idx, (limb_name, score, pose) in enumerate(zip(pose_msg.limb_names, pose_msg.scores, pose_msg.poses)):
                marker = Marker()
                marker.header = img_msg.header
                marker.ns = "joints"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose = pose
                marker.scale = Vector3(x=0.06, y=0.06, z=0.06)
                try:
                    i = self.index2limbname.index(limb_name)
                except ValueError:
                    i = 0
                rgba = cmap(1. * i / (len(self.index2limbname) - 1))
                marker.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=1.0)
                marker.lifetime = rospy.Duration(0.5)
                marker_array.markers.append(marker)

        # Add bone line list
        for person_idx, skeleton_msg in enumerate(skeleton_msgs.skeletons):
            marker = Marker()
            marker.header = img_msg.header
            marker.ns = "bones"
            marker.id = person_idx
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.scale = Vector3(x=0.02, y=0, z=0)
            marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8) # semi-transparent yellow
            marker.lifetime = rospy.Duration(0.5)
            for bone in skeleton_msg.bones:
                marker.points.append(bone.start_point)
                marker.points.append(bone.end_point)
            if len(marker.points) > 0:
                marker_array.markers.append(marker)

        self.pose_2d_pub.publish(people_pose_2d_msg)
        self.pose_pub.publish(people_pose_msg)
        self.skeleton_pub.publish(skeleton_msgs)
        self.marker_pub.publish(marker_array)

        if self.visualize:
            vis_img = self._draw_joints(img, people_joint_positions)
            vis_msg = br.cv2_to_imgmsg(vis_img, encoding='bgr8')
            vis_msg.header.stamp = img_msg.header.stamp
            self.image_pub.publish(vis_msg)

    def _cb(self, img_msg):
        br = cv_bridge.CvBridge()
        img = br.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        people_joint_positions = self.estimate(img)

        people_pose_msg = self._create_2d_people_pose_array_msgs(
            people_joint_positions,
            img_msg.header)

        self.pose_pub.publish(people_pose_msg)

        if self.visualize:
            vis_img = self._draw_joints(img, people_joint_positions)
            vis_msg = br.cv2_to_imgmsg(vis_img, encoding='bgr8')
            vis_msg.header.stamp = img_msg.header.stamp
            self.image_pub.publish(vis_msg)

    def _create_2d_people_pose_array_msgs(self, people_joint_positions, header):
        people_pose_msg = PeoplePoseArray(header=header)
        for person_joint_positions in people_joint_positions:
            pose_msg = PeoplePose()
            for joint_pos in person_joint_positions:
                if joint_pos['score'] < 0:
                    continue
                pose_msg.limb_names.append(joint_pos['limb'])
                pose_msg.scores.append(joint_pos['score'])
                pose_msg.poses.append(Pose(position=Point(x=joint_pos['x'],
                                                          y=joint_pos['y'],
                                                          z=0),
                                           orientation=Quaternion(w=1)))
            people_pose_msg.poses.append(pose_msg)
        return people_pose_msg

    def estimate(self, bgr_img):
        h, w, _ = bgr_img.shape
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        if self.use_hand:
            results = self.holistic.process(rgb_img)
            pose_landmarks = results.pose_landmarks
            left_hand_landmarks = results.left_hand_landmarks
            right_hand_landmarks = results.right_hand_landmarks
        else:
            results = self.pose.process(rgb_img)
            pose_landmarks = results.pose_landmarks
            left_hand_landmarks = None
            right_hand_landmarks = None

        if not pose_landmarks:
            return []

        people_joint_positions = []
        person_joint_positions = []
        landmarks = pose_landmarks.landmark

        mp_indices = {
            "Nose": 0,
            "RShoulder": 12,
            "RElbow": 14,
            "RWrist": 16,
            "LShoulder": 11,
            "LElbow": 13,
            "LWrist": 15,
            "RHip": 24,
            "RKnee": 26,
            "RAnkle": 28,
            "LHip": 23,
            "LKnee": 25,
            "LAnkle": 27,
            "REye": 5,
            "LEye": 2,
            "REar": 8,
            "LEar": 7
        }

        for limb_name in self.index2limbname:
            if limb_name == "Bkg":
                person_joint_positions.append(dict(limb=limb_name, x=0, y=0, score=-1))
            elif limb_name == "Neck":
                l_sh = landmarks[11]
                r_sh = landmarks[12]
                score = min(l_sh.visibility, r_sh.visibility)
                if score > self.min_visibility:
                    person_joint_positions.append(dict(
                        limb=limb_name,
                        x=(l_sh.x + r_sh.x) / 2.0 * w,
                        y=(l_sh.y + r_sh.y) / 2.0 * h,
                        score=score
                    ))
                else:
                    person_joint_positions.append(dict(limb=limb_name, x=0, y=0, score=-1))
            else:
                idx = mp_indices[limb_name]
                lm = landmarks[idx]
                if lm.visibility > self.min_visibility:
                    person_joint_positions.append(dict(
                        limb=limb_name,
                        x=lm.x * w,
                        y=lm.y * h,
                        score=lm.visibility
                    ))
                else:
                    person_joint_positions.append(dict(limb=limb_name, x=0, y=0, score=-1))

        if self.use_hand:
            # RHand
            if right_hand_landmarks:
                for idx, lm in enumerate(right_hand_landmarks.landmark):
                    limb_name = "RHand{}".format(idx)
                    person_joint_positions.append(dict(
                        limb=limb_name,
                        x=lm.x * w,
                        y=lm.y * h,
                        score=1.0
                    ))
            else:
                for idx in range(21):
                    limb_name = "RHand{}".format(idx)
                    person_joint_positions.append(dict(limb=limb_name, x=0, y=0, score=-1))

            # LHand
            if left_hand_landmarks:
                for idx, lm in enumerate(left_hand_landmarks.landmark):
                    limb_name = "LHand{}".format(idx)
                    person_joint_positions.append(dict(
                        limb=limb_name,
                        x=lm.x * w,
                        y=lm.y * h,
                        score=1.0
                    ))
            else:
                for idx in range(21):
                    limb_name = "LHand{}".format(idx)
                    person_joint_positions.append(dict(limb=limb_name, x=0, y=0, score=-1))

        people_joint_positions.append(person_joint_positions)
        return people_joint_positions

    def _draw_joints(self, img, people_joint_positions):
        all_peaks = [[] for _ in range(len(self.index2limbname) - 1)]
        for person_joint_positions in people_joint_positions:
            for i in range(len(self.index2limbname) - 1):
                jt = person_joint_positions[i]
                if jt['score'] >= 0:
                    all_peaks[i].append((jt['x'], jt['y']))

        try:
            cmap = matplotlib.colormaps.get_cmap('hsv')
        except AttributeError:
            cmap = matplotlib.cm.get_cmap('hsv')

        if all_peaks:
            # keypoints
            n = len(self.index2limbname)-1
            for i in range(len(self.index2limbname)-1):
                rgba = np.array(cmap(1. * i / n))
                color = rgba[:3] * 255
                for j in range(len(all_peaks[i])):
                    cv2.circle(img, (int(all_peaks[i][j][0]), int(
                        all_peaks[i][j][1])), 4, color, thickness=-1)

        # connections
        stickwidth = 4
        for joint_positions in people_joint_positions:
            n = len(self.limb_sequence)
            for i, conn in enumerate(self.limb_sequence):
                rgba = np.array(cmap(1. * i / n))
                color = rgba[:3] * 255
                j1, j2 = joint_positions[conn[0]-1], joint_positions[conn[1]-1]
                if j1['score'] < 0 or j2['score'] < 0:
                    continue
                cx, cy = int((j1['x'] + j2['x']) / 2.), int((j1['y'] + j2['y']) / 2.)
                dx, dy = j1['x'] - j2['x'], j1['y'] - j2['y']
                length = np.linalg.norm([dx, dy])
                angle = int(np.degrees(np.arctan2(dy, dx)))
                polygon = cv2.ellipse2Poly((cx, cy), (int(length / 2.), stickwidth),
                                           angle, 0, 360, 1)
                top = max(0, np.min(polygon[:,1]))
                left = max(0, np.min(polygon[:,0]))
                bottom = min(img.shape[0], np.max(polygon[:,1]))
                right = min(img.shape[1], np.max(polygon[:,0]))
                if top >= bottom or left >= right:
                    continue
                roi = img[top:bottom,left:right]
                roi2 = roi.copy()
                cv2.fillConvexPoly(roi2, polygon - np.array([left, top]), color)
                cv2.addWeighted(roi, 0.4, roi2, 0.6, 0.0, dst=roi)

        # for hand
        if self.use_hand:
            offset = len(self.limb_sequence)
            for joint_positions in people_joint_positions:
                n = len(joint_positions[offset:])
                for i, jt in enumerate(joint_positions[offset:]):
                    if jt['score'] < 0.0:
                        continue
                    rgba = np.array(cmap(1. * i / n))
                    color = rgba[:3] * 255
                    cv2.circle(img, (int(jt['x']), int(jt['y'])),
                               2, color, thickness=-1)

            for joint_positions in people_joint_positions:
                offset = len(self.limb_sequence)
                n = len(self.hand_sequence)
                for _ in range(2):
                    # for both hands
                    for i, conn in enumerate(self.hand_sequence):
                        rgba = np.array(cmap(1. * i / n))
                        color = rgba[:3] * 255
                        j1 = joint_positions[offset + conn[0]]
                        j2 = joint_positions[offset + conn[1]]
                        if j1['score'] < 0 or j2['score'] < 0:
                            continue
                        cx, cy = int((j1['x'] + j2['x']) / 2.), int((j1['y'] + j2['y']) / 2.)
                        dx, dy = j1['x'] - j2['x'], j1['y'] - j2['y']
                        length = np.linalg.norm([dx, dy])
                        angle = int(np.degrees(np.arctan2(dy, dx)))
                        polygon = cv2.ellipse2Poly((cx, cy), (int(length / 2.), stickwidth),
                                                   angle, 0, 360, 1)
                        top = max(0, np.min(polygon[:,1]))
                        left = max(0, np.min(polygon[:,0]))
                        bottom = min(img.shape[0], np.max(polygon[:,1]))
                        right = min(img.shape[1], np.max(polygon[:,0]))
                        if top >= bottom or left >= right:
                            continue
                        roi = img[top:bottom,left:right]
                        roi2 = roi.copy()
                        cv2.fillConvexPoly(roi2, polygon - np.array([left, top]), color)
                        cv2.addWeighted(roi, 0.4, roi2, 0.6, 0.0, dst=roi)
                    #
                    offset += int(len(self.index2handname) / 2)

        return img

if __name__ == '__main__':
    rospy.init_node('people_pose_estimation_mediapipe')
    PeoplePoseEstimationMediaPipe()
    rospy.spin()
