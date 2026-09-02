#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""人間の掌の位置姿勢 JSON (``estimate_palm_poses.py`` の出力) を入力とし、
ベース移動型ロボット (Aero) が全身 IK (台車の平面移動を含む) を解いて手を
繋ぐ姿勢を求め、その結果 (関節角・台車位置・手先姿勢) を JSON として保存
する。

``human_palm_contact_behavior.py`` (ROS ノード。カメラからリアルタイムに
掌を追跡し、実機を実際に動かす) の IK 周りのロジックを参考にしたオフライン
版。``rospy`` は import せず、``estimate_palm_poses.py`` が保存した JSON を
そのまま読んで 1 回だけ IK を解く。同ファイルにあった、以下のような実機を
安全に動かすための機能は持たないシンプル版であることに注意:

* 人体を障害物とした干渉回避 (経路計画) はしない。
* IK が 3 軸厳密な解 (``rotation_axis=True``) で収束しないとき、条件を
  段階的に緩めて (``'y'`` -> ``False``) 再試行することはしない。
* 台車が人間の足元に寄りすぎていないかの押し出し処理はしない。

一方、向きを ±90 度ずらした候補を順に試す処理 (``human_palm_contact_
behavior.py`` の ``MIRROR_TURN_CANDIDATES_DEG`` と同じ考え方) だけは残して
ある -- 平面フィットの誤差次第で、特定の向きのままだと IK が解けないことが
あるため。

対象にするのは、``estimate_palm_poses.py`` が「人がこの手を差し出して
いる」と判定した手 (掌 JSON の ``offered_hand`` が ``'L'`` / ``'R'``) を
持つ人物だけ。``offered_hand`` が ``null`` (どちらの手も差し出していない
と判定された) 人物は IK を解かない -- 差し出していない手を掴みに行く姿勢
はそもそも実機で取りたい姿勢ではないため。ただし対象外の人物についても、
入力と同じファイル名で ``offered_hand`` が ``null`` / ``solved`` が
``false`` の JSON (IK の結果は持たない) を書き出す。
``view_handshake_poses.py`` が骨格 JSON と突き合わせて「対象外」と表示
できるようにするため。

使うロボットの腕は既定で人間の手の反対側 (人の左手ならロボットの右腕、
``--robot-arm`` で上書きできる)。同じ側の手で向き合う握手ではなく、人間と
同じ方向を向いて反対側の手で繋ぐ想定 (``human_palm_contact_behavior.py``
の ``~same_hand=False`` に相当) なので、掌の向きは鏡写しにせずそのまま
使う。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo estimate_palm_poses.py
    rosrun aero_demo solve_palm_ik.py

(いずれも --input-dir/--output-dir を省略すると、scripts/ 直下の
random_human_poses/ -> random_palm_poses/ -> random_handshake_poses/ を
共通の入出力先として自動的につながる)
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo.palm_plane import CONTACT_OFFSET  # noqa: E402  (パス追加後に import)

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import matrix2ypr  # noqa: E402
from skrobot.models import Aero  # noqa: E402

# 掌のローカル +Y (甲->掌方向) まわりにこの角度ずつ向きをずらした候補を
# 順に試し、IK が解けた最初のものを採用する。0 度 (掌の向きをそのまま
# 使う) を最初に試し、それで解けなければ ±90 度を試す
# (human_palm_contact_behavior.py の MIRROR_TURN_CANDIDATES_DEG と同じ
# 考え方。この用途では鏡写しはしないので 0 度も候補に含めている)。
TURN_CANDIDATES_DEG = (0.0, 90.0, -90.0)

# 人間の手 (掌 JSON の ``offered_hand``) に対して既定で使うロボットの腕
# (``--robot-arm auto``)。向かい合う握手ではなく、人間と同じ方向を向いて
# 反対側の手で繋ぐ想定なので、人の手とは反対側の腕を使う。
DEFAULT_ROBOT_ARM = {'L': 'r', 'R': 'l'}


def _turn_about_y(rot, turn_deg):
    """``rot`` の局所 +X/+Z を、+Y 軸まわりに ``turn_deg`` 度だけ回す
    (+Y はそのまま)。``human_palm_contact_behavior._mirror_target_
    rotation`` の回転部分と同じ計算。"""
    x_axis, y_axis, z_axis = rot[:, 0], rot[:, 1], rot[:, 2]
    phi = math.radians(turn_deg)
    turned_x = math.cos(phi) * x_axis + math.sin(phi) * z_axis
    turned_z = -math.sin(phi) * x_axis + math.cos(phi) * z_axis
    return np.column_stack([turned_x, y_axis, turned_z])


def _correct_grasp_frame(rot, arm):
    """左腕用に +Y/+Z を反転する.

    ``l_eef_grasp_link`` は ``r_eef_grasp_link`` に対して +X (指方向)
    まわりに 180 度ずれている (URDF が左右ミラーで作られているため) ので、
    右腕用に組んだ ``rot`` を左腕で使うにはこの補正が要る。詳細は
    ``human_palm_contact_behavior._correct_grasp_frame`` を参照。
    """
    if arm != 'l':
        return rot
    return np.column_stack([rot[:, 0], -rot[:, 1], -rot[:, 2]])


def palm_to_target_rots(palm, robot_arm):
    """掌の位置姿勢 JSON (``estimate_palm_poses.PalmPoseEstimator`` の
    出力の 1 手分) から、ロボットの手先座標系 (``{arm}_eef_grasp_link``,
    +X=指方向, +Y=甲->掌方向, +Z=+X×+Y) で表した目標姿勢の候補群
    (``TURN_CANDIDATES_DEG`` の数だけ) を返す。

    向かい合う握手ではなく、人間と同じ方向を向いて反対側の手で繋ぐ想定
    (``human_palm_contact_behavior.py`` の ``~same_hand=False``) なので、
    指方向 (+X) は鏡写しにせずそのまま使う。掌の法線 (``y_axis``, 手の甲
    ->掌方向, 体の外側を向く) に対し、ロボットの掌は人間の掌に正対する
    向きにしたいので、ロボットの +Y は ``-y_axis``。
    """
    x_axis = np.asarray(palm['x_axis'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    y_axis = -normal
    z_axis = np.cross(x_axis, y_axis)
    base_rot = np.column_stack([x_axis, y_axis, z_axis])
    return [_correct_grasp_frame(_turn_about_y(base_rot, deg), robot_arm)
           for deg in TURN_CANDIDATES_DEG]


def solve_palm_ik(robot, palm, robot_arm):
    """1 人分の掌の位置姿勢に対して全身 IK (台車の平面移動を含む) を解く.

    Returns
    -------
    dict
        ``target`` (bool, IK の対象にした人物か。この関数の戻り値は常に
        ``True``。対象外の人物には代わりに ``not_target_result`` の
        戻り値を保存する), ``solved`` (bool), ``turn_deg`` (解けた候補の
        角度。解けなければ最後に試した候補の角度), ``target_position``/``target_rot``
        (IK に渡した目標), ``hand_position``/``hand_rot`` (IK 後の実際の
        手先姿勢), ``base_position``/``base_yaw`` (IK 後の台車の位置・
        向き), ``joint_names``/``joint_angle_vector`` (IK 後の全関節角)。
    """
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    # 掌の少し手前 (法線方向, palm_plane.contact_target と同じ考え方) を
    # 目標位置にする。
    target_pos = position + normal * CONTACT_OFFSET

    robot.reset_pose()
    robot.base_link.newcoords(Coordinates())

    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    # whole_body (リフター+腕だけの部分モデル) は base_link を含まないので、
    # use_base='planar' が仮想関節を挿す root を明示しないと自動探索に
    # 失敗する (human_palm_contact_behavior.py の同じ処理を参照)。
    whole_body.root_link = robot.base_link

    # 向かい合わず、人間と同じ方向を向いて手を繋ぐ構え (肩を横に開き、
    # 手首はひねらないニュートラルな姿勢) を IK の種にする -- 種のまま
    # だと目標との姿勢差が大きすぎて IK が迷走しやすいため
    # (human_palm_contact_behavior.py の ~same_hand=False の種と同じ)。
    mirror = 1.0 if robot_arm == 'l' else -1.0
    getattr(robot, '{}_shoulder_p_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_shoulder_r_joint'.format(robot_arm)) \
        .joint_angle(0.8 * mirror)
    getattr(robot, '{}_shoulder_y_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_elbow_joint'.format(robot_arm)).joint_angle(-1.0)
    getattr(robot, '{}_wrist_y_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_wrist_p_joint'.format(robot_arm)).joint_angle(0.087)
    getattr(robot, '{}_wrist_r_joint'.format(robot_arm)).joint_angle(0.0)

    solved = False
    turn_deg = TURN_CANDIDATES_DEG[-1]
    target_rot = None
    for deg, rot in zip(TURN_CANDIDATES_DEG,
                        palm_to_target_rots(palm, robot_arm)):
        target_rot = rot
        target_coords = Coordinates(pos=target_pos.tolist(), rot=rot)
        result = whole_body.inverse_kinematics(
            target_coords, rotation_axis=True, use_base='planar',
            stop=200, revert_if_fail=True)
        if result is not False:
            solved = True
            turn_deg = deg
            break

    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    # use_base='planar' moves base_link itself (see whole_body.root_link
    # above), but that motion is a temporary virtual joint spliced in above
    # base_link during the solve -- robot.translation/rotation (the
    # RobotModel's own root coordinate) do not track it, only base_link's
    # own worldcoords do, so read the base pose from there.
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())

    return dict(
        target=True,
        solved=solved,
        turn_deg=turn_deg,
        target_position=[float(v) for v in target_pos],
        target_rot=[[float(v) for v in row] for row in target_rot],
        hand_position=[float(v) for v in hand_coords.worldpos()],
        hand_rot=[[float(v) for v in row] for row in hand_coords.worldrot()],
        base_position=[float(v) for v in robot.base_link.worldpos()],
        base_yaw=float(yaw),
        joint_names=[j.name for j in robot.joint_list],
        joint_angle_vector=[float(v) for v in robot.angle_vector()],
    )


def not_target_result(offered_hand, reason):
    """IK の対象外だった人物のための結果 dict.

    IK は解かないので関節角・台車位置は持たず、``solved`` は ``False``、
    ``target`` が ``False`` になる。``view_handshake_poses.py`` は
    ``target`` を見て「対象外」と表示する (``solve_palm_ik`` が返す
    通常の結果と同じファイル名で保存される)。

    Parameters
    ----------
    offered_hand : str or None
        掌 JSON の ``offered_hand`` (対象外なので通常は ``None``)。
    reason : str
        対象外にした理由 (``'no_offered_hand'`` / ``'no_palm'``)。
    """
    return dict(target=False, solved=False, offered_hand=offered_hand,
                robot_arm=None, not_target_reason=reason)


def load_palm_json(path):
    """``estimate_palm_poses.save_json`` が保存した 1 人分の JSON を読む."""
    with open(path) as f:
        return json.load(f)


def save_json(result, path):
    """``solve_palm_ik`` の戻り値を JSON として保存する."""
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)


def iter_palm_files(input_dir, pattern='*.json'):
    """``input_dir`` 内の掌の位置姿勢 JSON をファイル名順に列挙する."""
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def main():
    parser = argparse.ArgumentParser(
        description='掌の位置姿勢 JSON (estimate_palm_poses.py の出力) を '
                    '入力とし、ベース移動型ロボットが全身 IK を解いて手を '
                    '繋ぐ姿勢を求め、JSON として保存する。IK を解くのは '
                    '掌推定が「手を差し出している」と判定した '
                    '(offered_hand が L/R の) 人物だけで、null の人物には '
                    'IK の結果を持たない JSON (target: false) を書き出す。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_palm_poses'),
        help='掌の位置姿勢 JSON の入力ディレクトリ (既定は '
            'estimate_palm_poses.py の既定の出力先と同じ '
            'random_palm_poses/。test_generate_and_estimate_palm_poses.py '
            'が書き出した test_palm_pose_pipeline/palms/ を使う場合は '
            'このオプションで指定する)。')
    parser.add_argument(
        '--output-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='IK の結果 JSON の保存先ディレクトリ (既定は '
            'random_handshake_poses/。入力と同じファイル名で保存するので、'
            'どの人物の結果かは入力ディレクトリの対応するファイルと '
            '突き合わせられる)。')
    parser.add_argument(
        '--robot-arm', choices=['auto', 'r', 'l'], default='auto',
        help='使うロボットの腕。既定 (auto) は人間の手の反対側 '
            '(人の左手ならロボットの右腕) -- 向かい合わず、人間と同じ '
            '方向を向いて反対側の手で繋ぐ想定のため。')
    args = parser.parse_args()

    files = iter_palm_files(args.input_dir)
    if not files:
        print('{} に掌の位置姿勢 JSON が見つかりません。先に '
              'estimate_palm_poses.py を実行してください。'.format(
                  args.input_dir))
        return

    os.makedirs(args.output_dir, exist_ok=True)
    # r/l_eef_grasp_link (IK が使う手先フレーム) は手あり/なし両方の URDF に
    # ある (Aero.__init__ 参照) ので、指の関節が要らないこのスクリプトでは
    # 手なしモデルを使う。
    robot = Aero(use_hand=False)

    n_solved = 0
    n_total = 0
    n_not_target = 0
    for i, path in enumerate(files):
        palms = load_palm_json(path)
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        # IK を解くのは、掌推定が「この手を差し出している」と判定した
        # (offered_hand が 'L'/'R' になった) 人物だけ。null (どちらの手も
        # 差し出していない) の人物は対象外として、IK の結果を持たない
        # JSON だけ書き出す (view_handshake_poses.py がそれを見て「対象
        # 外」と表示する)。
        human_hand = palms.get('offered_hand')
        palm = palms.get(human_hand) if human_hand in ('L', 'R') else None
        if palm is None:
            reason = ('no_palm' if human_hand in ('L', 'R')
                      else 'no_offered_hand')
            save_json(not_target_result(human_hand, reason), out_path)
            n_not_target += 1
            print('[{}/{}] {} -> {} (not target: {})'.format(
                i + 1, len(files), os.path.basename(path), out_path, reason))
            continue

        robot_arm = (DEFAULT_ROBOT_ARM[human_hand]
                     if args.robot_arm == 'auto' else args.robot_arm)
        result = solve_palm_ik(robot, palm, robot_arm)
        result['offered_hand'] = human_hand
        result['robot_arm'] = robot_arm
        n_total += 1
        n_solved += int(result['solved'])
        save_json(result, out_path)
        print('[{}/{}] {} -> {} ({}Hand -> {}arm, {})'.format(
            i + 1, len(files), os.path.basename(path), out_path,
            human_hand, robot_arm,
            'solved' if result['solved'] else 'NOT solved'))

    print('{}/{} solved (対象外 {} 人: 掌推定の offered_hand が null 等)。'
          .format(n_solved, n_total, n_not_target))


if __name__ == '__main__':
    main()
