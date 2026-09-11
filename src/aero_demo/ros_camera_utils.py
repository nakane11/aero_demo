#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``sensor_msgs/Image`` <-> numpy 変換、TF <-> 4x4 行列変換、camera->base の
TF 解決といった、``scripts/ros/`` 配下の複数のノード
(``run_camera_pipeline_test.py``/``record_palm_offer_clips.py``) が共通で
使うヘルパー。

``imgmsg_to_ndarray``/``ndarray_to_imgmsg`` は cv_bridge の代替 (cv_bridge は
システム (apt) 由来のバイナリで、ビルド時の NumPy 1.x の C-API を静的に
埋め込んでいるため NumPy 2.x 実行時に ImportError/AttributeError
(``_ARRAY_API not found``) を起こす。ここで使うのは bgr8/rgb8/mono8/16UC1/
32FC1 だけなので、cv_bridge に頼らず ``Image.data`` を直接 numpy 配列に
変換する)。
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
    """numpy 配列を ``sensor_msgs/Image`` に変換する
    (``imgmsg_to_ndarray`` の逆、cv_bridge の代替)."""
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

    まず画像の stamp ちょうどの TF を試み、それが (バッファに無い/
    extrapolation エラー等で) 引けなければ最新の TF (``rospy.Time(0)``)
    にフォールバックする。後者は画像とTFの時刻が厳密には一致しない
    (カメラ画像を出しているマシンと TF を配信しているマシンの間で
    システムクロックがズレていると、``ExtrapolationException`` が
    毎回発生してこの経路に入り続ける -- その場合は根本的には NTP 等で
    クロックを同期するべきだが、応急的にこのフォールバックでテストを
    続けられるようにしてある)。
    """
    try:
        return tf_buffer.lookup_transform(
            base_frame, header.frame_id, header.stamp,
            rospy.Duration(0.2))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
           tf2_ros.ExtrapolationException):
        # マシン間の時刻ズレで毎フレーム発生しうる想定内のフォールバック
        # なので、警告は出さず黙って最新の TF にフォールバックする
        # (それでも引けない場合だけ下の except で警告する)。
        pass
    try:
        # tf2 では Time(0) は「時刻 0」であり tf とは違って「最新」を
        # 意味しない。最新を取得するには現在時刻を渡す必要がある。
        return tf_buffer.lookup_transform(
            base_frame, header.frame_id, rospy.Time.now(),
            rospy.Duration(0.2))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
           tf2_ros.ExtrapolationException) as e:
        rospy.logwarn_throttle(
            5.0, 'TF lookup failed (%s -> %s): %s',
            header.frame_id, base_frame, e)
        return None
