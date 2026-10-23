"""表示ツール (``scripts/handshake_viewer.py``) の ROS 側: 実機ノード
(``run_camera_pipeline_test.py``) の状態・計画・骨格・画像と、実機の関節角・台車位置を
受け取り、ARM/RESET のサービスを呼ぶ。

``--ros`` のときだけ import する (合成骨格の表示は ROS 無しで動かすため)。
"""

import json
import threading

import numpy as np

import rospy
import tf2_ros
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aero_demo.ros_camera_utils import imgmsg_to_ndarray
from aero_demo.ros_camera_utils import transform_to_matrix

ODOM_FRAME = 'odom'


class RosBridge(object):
    """受け取った最新の値を持つ (コールバックは rospy のスレッドから来る)。"""

    def __init__(self, node, joint_states_topic='/joint_states',
                 base_frame='base_link', image_topic=None):
        """``node``: 実機ノードの名前 (``~status`` などの名前空間)。
        ``image_topic``: カメラ画像のトピック (None ならノードの ``~skeleton_image``)。"""
        rospy.init_node('handshake_viewer', anonymous=True,
                        disable_signals=True)
        self.node = node.rstrip('/')
        self.base_frame = base_frame
        self._lock = threading.Lock()
        self._plan = None
        self._plan_version = 0
        self._status = None
        self._skeleton = None
        self._image = None
        self._image_version = 0
        self._joint_angles = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self._subs = [
            rospy.Subscriber(self.node + '/plan', String, self._on_plan,
                             queue_size=1),
            rospy.Subscriber(self.node + '/status', String, self._on_status,
                             queue_size=1),
            rospy.Subscriber(self.node + '/skeleton', String,
                             self._on_skeleton, queue_size=1),
            rospy.Subscriber(image_topic or self.node + '/skeleton_image',
                             Image, self._on_image, queue_size=1,
                             buff_size=2 ** 24),
            rospy.Subscriber(joint_states_topic, JointState,
                             self._on_joint_states, queue_size=1),
        ]

    def _on_plan(self, msg):
        plan = json.loads(msg.data)
        with self._lock:
            self._plan = plan if plan.get('attempt') is not None else None
            self._plan_version += 1

    def _on_status(self, msg):
        status = json.loads(msg.data)
        with self._lock:
            self._status = (status, rospy.get_time())

    def _on_skeleton(self, msg):
        with self._lock:
            self._skeleton = json.loads(msg.data)

    def _on_image(self, msg):
        image = imgmsg_to_ndarray(msg, desired_encoding='bgr8')
        with self._lock:
            self._image = image
            self._image_version += 1

    def _on_joint_states(self, msg):
        angles = dict(zip(msg.name, msg.position))
        with self._lock:
            if self._joint_angles is None:
                self._joint_angles = angles
            else:
                # 一部の関節だけのメッセージでも他の関節を消さない。
                self._joint_angles.update(angles)

    def plan(self):
        """``(版, 計画 dict or None)``。版は計画を受け取るたびに増える。"""
        with self._lock:
            return self._plan_version, self._plan

    def status(self):
        """最新の ``~status`` (dict) と受信からの経過秒 (無ければ ``(None, None)``)。"""
        with self._lock:
            if self._status is None:
                return None, None
            status, received = self._status
        return status, rospy.get_time() - received

    def live_skeleton(self):
        """最新の ``~skeleton`` の骨格 (base_link 系、無ければ None)。"""
        with self._lock:
            skeleton = self._skeleton
        if (skeleton is None or skeleton.get('frame') != 'base_link'
                or skeleton.get('joint_positions') is None):
            return None
        return skeleton['joint_positions']

    def image(self):
        """``(版, BGR 画像 or None)``。"""
        with self._lock:
            return self._image_version, self._image

    @staticmethod
    def now():
        """ROS の今の時刻 [秒] (bag の再生中は bag の時刻)。"""
        return rospy.get_time()

    def joint_angles(self):
        with self._lock:
            return None if self._joint_angles is None else dict(
                self._joint_angles)

    def odom_from_base(self):
        """最新の odom <- base_link の 4x4 (引けなければ None)。"""
        try:
            transform = self.tf_buffer.lookup_transform(
                ODOM_FRAME, self.base_frame, rospy.Time(0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            return None
        return transform_to_matrix(transform.transform)

    def call(self, service):
        """実機ノードのサービス (``arm``/``reset``) を呼び、``(成否, 文)`` を返す。"""
        name = '{}/{}'.format(self.node, service)
        try:
            rospy.wait_for_service(name, timeout=1.0)
            response = rospy.ServiceProxy(name, Trigger)()
        except (rospy.ROSException, rospy.ServiceException) as e:
            return False, str(e)
        return response.success, response.message

    @staticmethod
    def world_from_base(odom_from_base, real_pose_ref, origin=None):
        """実機の台車を、計画の系 (``real_pose_ref`` の時点の base_link を原点)
        に置いた 4x4。``real_pose_ref`` が無ければ今の台車を原点にする。"""
        world = np.eye(4) if origin is None else np.asarray(origin)
        if real_pose_ref is None or odom_from_base is None:
            return world
        return world @ np.linalg.inv(np.asarray(real_pose_ref)) \
            @ odom_from_base

    def shutdown(self):
        rospy.signal_shutdown('handshake_viewer closed')
