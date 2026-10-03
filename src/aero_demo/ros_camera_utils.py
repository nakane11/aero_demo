#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""ROS ノード共通のヘルパー: Image <-> numpy 変換、TF の行列化・解決.

Image の変換は cv_bridge の代替 (apt の cv_bridge は NumPy 2.x で動かない)。
"""

import numpy as np
import rospy
import tf2_ros
from sensor_msgs.msg import Image

# sensor_msgs/Image -> numpy 変換用の dtype/チャンネル数テーブル。
IMGMSG_DTYPE_CHANNELS = {
    'bgr8': (np.uint8, 3),
    'rgb8': (np.uint8, 3),
    'mono8': (np.uint8, 1),
    '8UC1': (np.uint8, 1),
    '16UC1': (np.uint16, 1),
    '32FC1': (np.float32, 1),
}


def imgmsg_to_ndarray(msg, desired_encoding=None):
    """``sensor_msgs/Image`` を numpy 配列へ変換する (cv_bridge の代替)."""
    if msg.encoding not in IMGMSG_DTYPE_CHANNELS:
        raise ValueError('Unsupported image encoding: {}'.format(msg.encoding))
    dtype, channels = IMGMSG_DTYPE_CHANNELS[msg.encoding]
    dtype = np.dtype(dtype).newbyteorder('>' if msg.is_bigendian else '<')
    arr = np.frombuffer(msg.data, dtype=dtype)
    shape = (msg.height, msg.width, channels) if channels > 1 else (msg.height, msg.width)
    arr = arr.reshape(shape)

    if desired_encoding is not None and desired_encoding != msg.encoding:
        if {desired_encoding, msg.encoding} == {'bgr8', 'rgb8'}:
            arr = arr[..., ::-1]
        else:
            raise ValueError(
                'Cannot convert image encoding {} -> {}'.format(
                    msg.encoding, desired_encoding))
    return np.ascontiguousarray(arr)


def ndarray_to_imgmsg(arr, encoding, header):
    """numpy 配列を ``sensor_msgs/Image`` に変換する."""
    dtype, channels = IMGMSG_DTYPE_CHANNELS[encoding]
    arr = np.ascontiguousarray(arr, dtype=dtype)
    msg = Image()
    msg.header = header
    msg.height, msg.width = arr.shape[0], arr.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = msg.width * channels * np.dtype(dtype).itemsize
    msg.data = arr.tobytes()
    return msg


def transform_to_matrix(transform):
    """``geometry_msgs/Transform`` を 4x4 の同次変換行列にする."""
    t = transform.translation
    q = transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix = np.eye(4)
    matrix[:3, :3] = rot
    matrix[:3, 3] = [t.x, t.y, t.z]
    return matrix


def lookup_camera_to_base(tf_buffer, base_frame, header):
    """``header`` (画像の frame_id/stamp) から base_frame への TF を引く.

    stamp で引けなければ (マシン間の時計ずれ等) 現在時刻で引き直す。
    """
    try:
        return tf_buffer.lookup_transform(
            base_frame, header.frame_id, header.stamp,
            rospy.Duration(0.2))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
           tf2_ros.ExtrapolationException):
        pass  # 想定内なので警告しない
    try:
        return tf_buffer.lookup_transform(
            base_frame, header.frame_id, rospy.Time.now(),
            rospy.Duration(0.2))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
           tf2_ros.ExtrapolationException) as e:
        rospy.logwarn_throttle(
            5.0, 'TF lookup failed (%s -> %s): %s',
            header.frame_id, base_frame, e)
        return None


def lookup_frame_position(tf_buffer, base_frame, frame_id, fallback_position,
                          warn_label=''):
    """``base_frame`` から見た ``frame_id`` 原点の位置 [x, y, z] を最新の TF で返す.

    引けなければ警告して ``fallback_position`` を返す。``warn_label`` は
    警告ログの接頭辞。
    """
    try:
        transform = tf_buffer.lookup_transform(
            base_frame, frame_id, rospy.Time(0))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
           tf2_ros.ExtrapolationException) as e:
        rospy.logwarn_throttle(
            5.0, '%sロボット手先の TF (%s -> %s) がまだ引けません (%s)。'
            'フォールバック値 %s に切り替えます。',
            warn_label, frame_id, base_frame, e, tuple(fallback_position))
        return np.asarray(fallback_position, dtype=np.float64)
    t = transform.transform.translation
    return np.array([t.x, t.y, t.z], dtype=np.float64)
