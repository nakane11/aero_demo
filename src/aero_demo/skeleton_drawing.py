#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""MediaPipe 形式の骨格を viser の線分、または画像への重ね描きにする."""

import cv2

from aero_demo import palm_plane_view
from aero_demo.people_pose_types import Bone

# 骨格のつながり (関節名のペア)
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
HAND_SEQUENCE = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
HAND_WRIST_PAIRS = [('RWrist', 'RHand0'), ('LWrist', 'LHand0')]
BONE_NAME_PAIRS = BODY_BONE_PAIRS + HAND_WRIST_PAIRS + [
    ('{}Hand{}'.format(side, a), '{}Hand{}'.format(side, b))
    for side in ('R', 'L') for a, b in HAND_SEQUENCE]

# 差し出し手の色 (BGR)
OFFERED_HAND_BGR = (0, 0, 255)


def fill_missing_wrist_from_hand(positions):
    """未検出の RWrist/LWrist を RHand0/LHand0 で補ったコピーを返す."""
    filled = dict(positions)
    for wrist_name, hand_wrist_name in (('RWrist', 'RHand0'),
                                        ('LWrist', 'LHand0')):
        if wrist_name not in filled and hand_wrist_name in filled:
            filled[wrist_name] = filled[hand_wrist_name]
    return filled


def build_skeleton_links(joint_positions, hand_colors=None):
    """骨格を部位ごとに色分けした ``LineString`` のリストにする.

    ``hand_colors`` ({'R': rgba, 'L': rgba}) を渡すと手のボーンをその色にする。
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
    """BGR 画像に 1 人分の 2D 関節 (``estimate`` の形式) を重ねた複製を返す.

    ``offered_side`` ('R'/'L') の手は ``OFFERED_HAND_BGR`` で描く。
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
