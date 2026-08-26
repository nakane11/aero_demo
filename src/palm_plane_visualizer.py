#!/usr/bin/env python3
"""Visualise the fitted human palm plane in RViz -- no robot motion.

A test harness for ``human_palm_contact_behavior.py``: it runs exactly the
same palm-plane estimation (``palm_plane.fit_palm_plane``) on the live pose
topic and publishes the result as markers, but it never loads the robot
model or opens a controller interface, so the robot cannot move.

Use it to check, before letting the robot reach, that

* the plane sits flat on the human palm,
* the green normal arrow points back at the robot (not into the person),
* the blue/red approach and press targets land where you expect.

Topics
------
``~input`` (jsk_recognition_msgs/PeoplePoseArray)
    Pose input; the launch file remaps it to
    ``/people_pose_estimation_mediapipe/pose``.
``~markers`` (visualization_msgs/MarkerArray)
    plate, normal arrow, finger-direction arrow, fitted landmarks,
    approach/press targets and a text readout of the fit quality.

Parameters
----------
``~target_frame`` (str, default ``base_link``)
    Frame the plane is computed and drawn in.  Falls back to the pose
    message's own frame if the transform is unavailable, so the node is
    still useful without the robot's TF tree.
``~hand`` (str, default ``R``)
    Which hand to track, ``R`` or ``L``.
``~min_score`` (float, default 0.1)
``~rate_limit`` (float, default 0.0)
    Minimum seconds between published updates (0 = every message).
"""

import numpy as np
import rospy
import tf
from geometry_msgs.msg import PointStamped
from jsk_recognition_msgs.msg import PeoplePoseArray
from visualization_msgs.msg import MarkerArray

import palm_plane


class PalmPlaneVisualizer(object):
    def __init__(self):
        self.target_frame = rospy.get_param('~target_frame', 'base_link')
        self.hand = rospy.get_param('~hand', 'R').upper()[:1]
        self.min_score = rospy.get_param('~min_score', 0.1)
        self.rate_limit = rospy.get_param('~rate_limit', 0.0)

        self.tf_listener = tf.TransformListener()
        self.marker_pub = rospy.Publisher('~markers', MarkerArray,
                                          queue_size=1)
        self.last_pub = rospy.Time(0)
        self.stats = {'msgs': 0, 'fitted': 0}

        self.sub = rospy.Subscriber('~input', PeoplePoseArray,
                                    self.pose_cb, queue_size=1)

        rospy.loginfo('palm_plane_visualizer: input=%s target_frame=%s '
                      'hand=%sHand (visualisation only, the robot is never '
                      'commanded)', rospy.resolve_name('~input'),
                      self.target_frame, self.hand)
        rospy.Timer(rospy.Duration(5.0), self.report)

    def report(self, _event):
        n, k = self.stats['msgs'], self.stats['fitted']
        if n == 0:
            rospy.logwarn_throttle(
                10.0, 'no pose messages yet on %s',
                rospy.resolve_name('~input'))
            return
        rospy.loginfo('palm plane fitted in %d/%d frames (%.0f%%)',
                      k, n, 100.0 * k / n)

    def _make_transform(self, source_frame):
        """Return f(Point)->xyz in target_frame, or None if TF is missing."""
        stamp = rospy.Time(0)
        try:
            self.tf_listener.waitForTransform(
                self.target_frame, source_frame, stamp, rospy.Duration(0.1))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException,
                tf.ExtrapolationException) as e:
            rospy.logwarn_throttle(
                5.0, 'no TF %s <- %s (%s); drawing in the camera frame '
                'instead', self.target_frame, source_frame, e)
            return None

        def transform(point):
            ps = PointStamped()
            ps.header.frame_id = source_frame
            ps.header.stamp = stamp
            ps.point = point
            try:
                out = self.tf_listener.transformPoint(self.target_frame, ps)
            except tf.Exception as e:
                rospy.logwarn_throttle(5.0, 'TF error: %s', e)
                return None
            return np.array([out.point.x, out.point.y, out.point.z])

        return transform

    def pose_cb(self, msg):
        now = rospy.Time.now()
        if self.rate_limit > 0.0 and \
                (now - self.last_pub).to_sec() < self.rate_limit:
            return

        transform = self._make_transform(msg.header.frame_id)
        if transform is None:
            frame = msg.header.frame_id
            # Without TF the normal-sign heuristic ("point back at the
            # robot") is meaningless in base_link terms, but the camera
            # origin is a reasonable stand-in for the observer.
            viewpoint = np.zeros(3)
        else:
            frame = self.target_frame
            viewpoint = np.zeros(3)

        for person in msg.poses:
            self.stats['msgs'] += 1
            points = palm_plane.collect_palm_points(
                person, hand=self.hand, min_score=self.min_score,
                transform=transform)
            plane = palm_plane.fit_palm_plane(points, viewpoint=viewpoint)
            if plane is None:
                rospy.loginfo_throttle(
                    2.0, 'palm plane not fitted: only %d of %s landmarks '
                    '(need %d non-collinear)', len(points),
                    list(palm_plane.PLANE_LANDMARKS),
                    palm_plane.MIN_PALM_POINTS)
                continue

            self.stats['fitted'] += 1
            arr = palm_plane.palm_plane_markers(
                plane, frame, stamp=now,
                lifetime=rospy.Duration(0.5),
                label='{}Hand'.format(self.hand))
            arr.markers.extend(palm_plane.palm_landmark_markers(
                points, frame, stamp=now, lifetime=rospy.Duration(0.5),
                used=plane.used).markers)
            self.marker_pub.publish(arr)
            self.last_pub = now
            rospy.loginfo_throttle(
                1.0, 'palm plane: used=%s rms=%.1fmm center=(%.3f, %.3f, '
                '%.3f) normal=(%.2f, %.2f, %.2f)',
                plane.used, plane.rms * 1000.0,
                plane.center[0], plane.center[1], plane.center[2],
                plane.normal[0], plane.normal[1], plane.normal[2])
            # Only the first person with a usable palm is drawn.
            break


if __name__ == '__main__':
    rospy.init_node('palm_plane_visualizer')
    PalmPlaneVisualizer()
    rospy.spin()
