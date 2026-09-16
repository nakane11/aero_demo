#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""MediaPipe 形式の骨格 (``{関節名: 座標}`` / ``PeoplePoseEstimator`` の
2D ランドマーク) を viser の線分、または OpenCV 画像への重ね描きにする
共通処理。``draw_random_human_poses.py``/``scripts/ros/
run_camera_pipeline_test.py``/``scripts/ros/record_palm_offer_clips.py``
で共有する。rospy には依存しない。
"""

import cv2

from aero_demo import palm_plane_view
from aero_demo.people_pose_types import Bone

# 骨格の関節同士のつながり (関節名のペア)。
# PeoplePoseEstimator.limb_sequence/index2limbname/hand_sequence と同じ
# 骨格のつながりを、名前のペアとして書き下したもの。
BODY_BONE_PAIRS = [
    ('Neck', 'Nose'), ('Nose', 'LEye'), ('Nose', 'REye'),
    ('LShoulder', 'LEar'), ('RShoulder', 'REar'),
    ('Neck', 'RShoulder'), ('Neck', 'LShoulder'),
    ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
    ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
    ('Neck', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
    ('Neck', 'LHip'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
    ('REye', 'REar'), ('LEye', 'LEar'),
]
# 手のランドマーク (MediaPipe の並び) 同士のつながり。
HAND_SEQUENCE = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
# 腕の手首と手ランドマーク index 0 (手首) の接続。
HAND_WRIST_PAIRS = [('RWrist', 'RHand0'), ('LWrist', 'LHand0')]
BONE_NAME_PAIRS = BODY_BONE_PAIRS + HAND_WRIST_PAIRS + [
    ('{}Hand{}'.format(side, a), '{}Hand{}'.format(side, b))
    for side in ('R', 'L') for a, b in HAND_SEQUENCE]

# 手繋ぎに使うと判定された手を見分けるための色 (画像描画用, BGR)。
OFFERED_HAND_BGR = (0, 0, 255)


def fill_missing_wrist_from_hand(positions):
    """手首 (``RWrist``/``LWrist``) が未検出でも、Hand モデルの手首
    ランドマーク (``RHand0``/``LHand0``) が検出できていればその位置を
    手首として補って返す (辞書のコピー、``positions`` 自体は書き換えない)。

    Pose モデルの手首と Hand モデルの手首は別々に検出されるランドマーク
    なので、人にカメラから見て手が体の陰に隠れる等で Pose 側の手首だけ
    未検出になっても Hand 側は検出できていることがある。これを補わずに
    描画すると手のランドマークだけが肘から浮いて見えてしまう。
    """
    filled = dict(positions)
    for wrist_name, hand_wrist_name in (('RWrist', 'RHand0'),
                                        ('LWrist', 'LHand0')):
        if wrist_name not in filled and hand_wrist_name in filled:
            filled[wrist_name] = filled[hand_wrist_name]
    return filled


def build_skeleton_links(joint_positions, hand_colors=None):
    """骨格を部位ごとに色分けした線 (``skrobot.model.primitives.
    LineString``) のリストにする (viser 表示用)。

    Parameters
    ----------
    joint_positions : dict
        関節名 -> ``np.ndarray([x, y, z])``。
    hand_colors : dict or None
        ``{'R': rgba, 'L': rgba}``。渡すと手のランドマークのボーン
        (``RHand*``/``LHand*``) だけこの色で描く (手繋ぎに使うと判定
        された手を見分けるため)。``None`` なら部位ごとの既定色
        (``palm_plane_view.COLOR_BONES``) のまま。
    """
    joint_positions = fill_missing_wrist_from_hand(joint_positions)
    links = []
    for start_name, end_name in BONE_NAME_PAIRS:
        if start_name not in joint_positions or end_name not in joint_positions:
            continue
        bone = Bone(name='{}->{}'.format(start_name, end_name),
                   start_point=joint_positions[start_name],
                   end_point=joint_positions[end_name])
        color = palm_plane_view.bone_color(bone.name)
        group = palm_plane_view.bone_group(bone.name)
        if hand_colors is not None and group in ('rhand', 'lhand'):
            color = hand_colors['R' if group == 'rhand' else 'L']
        links.append(palm_plane_view.bone_line(bone, color))
    return links


def draw_skeleton_overlay(color_bgr, joints_2d, offered_side=None):
    """カメラ画像 (BGR) に、検出できた 2D 関節位置を重ねて描いた画像を
    返す (元の ``color_bgr`` は書き換えない)。

    Parameters
    ----------
    color_bgr : np.ndarray
    joints_2d : list of dict
        ``PeoplePoseEstimator.estimate``/``estimate_3d`` が返す 1 人分の
        ``[{"limb": str, "x": float, "y": float, "score": float}, ...]``
        (画像座標、score < 0 は未検出)。
    offered_side : str or None
        ``'R'``/``'L'`` を渡すと、差し出し手と判定された側の骨格を赤
        (``OFFERED_HAND_BGR``) で描く。``None`` なら部位ごとの既定色。
    """
    overlay = color_bgr.copy()
    positions = {j['limb']: (int(round(j['x'])), int(round(j['y'])))
                for j in joints_2d if j['score'] >= 0}
    positions = fill_missing_wrist_from_hand(positions)

    def is_offered_joint(name):
        return offered_side is not None and name.startswith(offered_side + 'Hand')

    def is_offered_bone(start_name, end_name):
        return is_offered_joint(start_name) or is_offered_joint(end_name)

    for start_name, end_name in BONE_NAME_PAIRS:
        if start_name not in positions or end_name not in positions:
            continue
        if is_offered_bone(start_name, end_name):
            bgr = OFFERED_HAND_BGR
        else:
            color = palm_plane_view.bone_color(
                '{}->{}'.format(start_name, end_name))
            bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.line(overlay, positions[start_name], positions[end_name],
                 bgr, 2, cv2.LINE_AA)
    for name, point in positions.items():
        dot_bgr = OFFERED_HAND_BGR if is_offered_joint(name) else (255, 255, 255)
        cv2.circle(overlay, point, 3, dot_bgr, -1, cv2.LINE_AA)
    return overlay
