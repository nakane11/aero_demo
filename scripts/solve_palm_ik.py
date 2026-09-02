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

IK は 1 人ずつ逐次に解くのではなく、``batch_inverse_kinematics``
(複数の目標姿勢 × 複数初期値を並列に解くバッチ IK) で **ロボットの腕
(``r``/``l``) ごとに 1 回だけ** 解く。全人物 × 全 ``TURN_CANDIDATES_DEG``
候補の目標姿勢をあらかじめ集めて 1 バッチにまとめ、1 目標あたり
``--attempts-per-pose`` 個の初期値 (attempt 0 が下記の「肩を開いた種の
姿勢」、残りは関節範囲の一様乱数) から同時に解く。逐次版に比べて速く、
初期値を振る分だけ解ける人物も増える (66 人で 12.3 秒 / 6 人 ->
2.9 秒 / 64 人)。``use_base`` を指定するとソルバキャッシュが効かない
(仮想リンクが毎回作り直されるため) ので、呼び出し回数を最小にするこの
まとめ方が前提。

台車の移動範囲は ``batch_inverse_kinematics`` の ``base_limits``
(``--base-x-range``/``--base-y-range``/``--base-yaw-range``) で明示的に
指定する。バッチ IK は ``base_limits`` を渡さないと非有限リミットを
±π に丸めてしまい、台車が暗黙に原点 ±3.14 m に拘束されるため、既定でも
明示的な箱を渡している。

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

# 台車 (use_base='planar' の仮想関節) の既定の移動範囲。IK 開始時の台車
# 位置を原点とした [x, y, yaw] の (下限, 上限)。乱数初期値もこの範囲から
# 引かれるので、実機で現実的な範囲に絞ったほうが解けやすい。
DEFAULT_BASE_X_RANGE = (-0.2, 2.0)
DEFAULT_BASE_Y_RANGE = (-1.5, 1.5)
DEFAULT_BASE_YAW_RANGE = (-math.pi / 2.0, math.pi / 2.0)

# 1 目標姿勢あたりに振る初期値の数 (バッチ IK の attempts_per_pose)。
# attempt 0 は seed_arm_pose の種の姿勢、残りは関節範囲の一様乱数。
DEFAULT_ATTEMPTS_PER_POSE = 16


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


def palm_target_position(palm):
    """掌の少し手前 (法線方向, ``palm_plane.contact_target`` と同じ考え方)
    を IK の目標位置として返す。"""
    position = np.asarray(palm['position'], dtype=np.float64)
    normal = np.asarray(palm['y_axis'], dtype=np.float64)
    return position + normal * CONTACT_OFFSET


def seed_arm_pose(robot, robot_arm):
    """バッチ IK の attempt 0 に使う「種の姿勢」をロボットに作る.

    向かい合わず、人間と同じ方向を向いて手を繋ぐ構え (肩を横に開き、
    手首はひねらないニュートラルな姿勢)。種のままだと目標との姿勢差が
    大きすぎて IK が迷走しやすいため (``human_palm_contact_behavior.py``
    の ``~same_hand=False`` の種と同じ)。台車も原点に戻す
    (``base_limits`` はこの位置を基準にした範囲になる)。

    ``robot.newcoords`` と ``base_link.newcoords`` の両方を単位座標系に
    戻しているのは、Aero では ``root_link`` が ``base_link`` そのもので、
    バッチ IK の解を ``robot.newcoords(base_pose)`` で反映すると
    ``base_link`` のローカル座標が単位でないとずれるため。
    """
    robot.reset_pose()
    robot.newcoords(Coordinates())
    robot.base_link.newcoords(Coordinates())
    mirror = 1.0 if robot_arm == 'l' else -1.0
    getattr(robot, '{}_shoulder_p_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_shoulder_r_joint'.format(robot_arm)) \
        .joint_angle(0.8 * mirror)
    getattr(robot, '{}_shoulder_y_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_elbow_joint'.format(robot_arm)).joint_angle(-1.0)
    getattr(robot, '{}_wrist_y_joint'.format(robot_arm)).joint_angle(0.0)
    getattr(robot, '{}_wrist_p_joint'.format(robot_arm)).joint_angle(0.087)
    getattr(robot, '{}_wrist_r_joint'.format(robot_arm)).joint_angle(0.0)


def build_ik_tasks(palms_list, robot_arms):
    """人物ごとの掌の位置姿勢から、バッチ IK に渡す目標姿勢の一覧を作る.

    Parameters
    ----------
    palms_list : list[dict]
        IK の対象にする人物の掌 (``palm``) の位置姿勢。
    robot_arms : list[str]
        ``palms_list`` と同じ並びで、それぞれに使うロボットの腕
        (``'r'``/``'l'``)。

    Returns
    -------
    dict[str, list[tuple]]
        腕ごとの ``(person_index, candidate_index, target_coords)`` の
        一覧。``candidate_index`` は ``TURN_CANDIDATES_DEG`` の添字
        (小さいほど優先したい候補)。
    """
    tasks = {'r': [], 'l': []}
    for person_index, (palm, arm) in enumerate(zip(palms_list, robot_arms)):
        target_pos = palm_target_position(palm)
        for candidate_index, rot in enumerate(
                palm_to_target_rots(palm, arm)):
            tasks[arm].append((
                person_index, candidate_index,
                Coordinates(pos=target_pos.tolist(), rot=rot)))
    return tasks


def solve_palm_ik_batch(robot, tasks, robot_arm,
                        attempts_per_pose=DEFAULT_ATTEMPTS_PER_POSE,
                        base_limits=None, backend=None):
    """1 本の腕について、全人物 × 全候補の目標姿勢をバッチ IK で解く.

    ``robot`` の腕は ``seed_arm_pose`` で種の姿勢にしてから
    ``batch_inverse_kinematics`` を 1 回だけ呼ぶ (``use_base`` 付きの
    呼び出しはソルバキャッシュが効かないので、呼び出し回数を最小に
    するのが速さの前提)。バッチ IK 自体はロボットを動かさないので、
    戻り値は「解を反映するための材料」であり、``robot`` は呼び出し後も
    種の姿勢のまま。

    Returns
    -------
    dict[tuple, tuple]
        ``(person_index, candidate_index)`` -> ``(angle_vector,
        base_pose, target_coords)``。IK が解けた組み合わせだけを含む。
    """
    if not tasks:
        return {}
    seed_arm_pose(robot, robot_arm)
    whole_body = getattr(robot, '{}arm_whole_body'.format(robot_arm))
    move_target = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    angle_vectors, base_poses, success_flags, _ = \
        robot.batch_inverse_kinematics(
            target_coords=[coords for _, _, coords in tasks],
            move_target=move_target,
            link_list=whole_body.link_list,
            position_mask=True, rotation_mask=True,
            stop=200, thre=0.001, rthre=np.deg2rad(1.0),
            initial_angles='current',
            attempts_per_pose=attempts_per_pose,
            backend=backend,
            use_base='planar', base_limits=base_limits)
    solutions = {}
    for (person_index, candidate_index, coords), angle_vector, base_pose, ok \
            in zip(tasks, angle_vectors, base_poses, success_flags):
        if ok:
            solutions[(person_index, candidate_index)] = (
                angle_vector, base_pose, coords)
    return solutions


def pick_solution(solutions, person_index):
    """1 人分の解のうち、候補インデックスが最小 (優先度が高い) ものを選ぶ.

    バッチ IK は ``TURN_CANDIDATES_DEG`` の全候補を同時に解くので、
    解けた中で添字が最小のものを採ることで、逐次版の「0 度を優先し、
    解けなければ ±90 度」という優先順位がそのまま保たれる。

    Returns
    -------
    tuple or None
        ``(candidate_index, angle_vector, base_pose, target_coords)``。
        どの候補も解けなければ ``None``。
    """
    candidates = sorted(
        candidate_index for (pi, candidate_index) in solutions
        if pi == person_index)
    if not candidates:
        return None
    candidate_index = candidates[0]
    angle_vector, base_pose, coords = solutions[
        (person_index, candidate_index)]
    return candidate_index, angle_vector, base_pose, coords


def solved_result(robot, robot_arm, target_pos, target_rot, candidate_index,
                  angle_vector, base_pose):
    """採用した解をロボットに反映し、結果 dict を組む.

    バッチ IK はロボットを動かさないので、``angle_vector`` と
    ``base_pose`` を実際に反映してから手先・台車の姿勢を読み直す。
    Aero は ``root_link`` が ``base_link`` なので台車の姿勢は
    ``robot.newcoords`` で入る (``seed_arm_pose`` の注記を参照)。
    """
    robot.angle_vector(angle_vector)
    robot.newcoords(base_pose)
    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    return dict(
        target=True,
        solved=True,
        turn_deg=TURN_CANDIDATES_DEG[candidate_index],
        target_position=[float(v) for v in target_pos],
        target_rot=[[float(v) for v in row] for row in target_rot],
        hand_position=[float(v) for v in hand_coords.worldpos()],
        hand_rot=[[float(v) for v in row] for row in hand_coords.worldrot()],
        base_position=[float(v) for v in robot.base_link.worldpos()],
        base_yaw=float(yaw),
        joint_names=[j.name for j in robot.joint_list],
        joint_angle_vector=[float(v) for v in robot.angle_vector()],
    )


def unsolved_result(robot, robot_arm, target_pos, target_rot):
    """どの候補も解けなかった人物のための結果 dict.

    逐次版が ``revert_if_fail=True`` で種の姿勢に戻してから結果を読んで
    いたのと同じになるよう、種の姿勢 (台車は原点) を反映してから手先・
    台車の姿勢を読む。``turn_deg``/``target_rot`` も逐次版と同じく最後に
    試した候補のものにする。
    """
    seed_arm_pose(robot, robot_arm)
    hand_coords = getattr(robot, '{}arm_end_coords'.format(robot_arm))
    yaw, _, _ = matrix2ypr(robot.base_link.worldrot())
    return dict(
        target=True,
        solved=False,
        turn_deg=TURN_CANDIDATES_DEG[-1],
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
    ``target`` を見て「対象外」と表示する (通常の結果と同じファイル名で
    保存される)。

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
    """IK の結果 dict を JSON として保存する."""
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)


def iter_palm_files(input_dir, pattern='*.json'):
    """``input_dir`` 内の掌の位置姿勢 JSON をファイル名順に列挙する."""
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def main():
    parser = argparse.ArgumentParser(
        description='掌の位置姿勢 JSON (estimate_palm_poses.py の出力) を '
                    '入力とし、ベース移動型ロボットが全身 IK をバッチで '
                    '解いて手を繋ぐ姿勢を求め、JSON として保存する。IK を '
                    '解くのは掌推定が「手を差し出している」と判定した '
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
    parser.add_argument(
        '--attempts-per-pose', type=int,
        default=DEFAULT_ATTEMPTS_PER_POSE,
        help='1 つの目標姿勢に対して振る初期値の数 (バッチ IK の '
            'attempts_per_pose)。1 個目は肩を開いた種の姿勢、残りは '
            '関節範囲の一様乱数。増やすと解ける人物が増えるが遅くなる '
            '(既定 {})。'.format(DEFAULT_ATTEMPTS_PER_POSE))
    parser.add_argument(
        '--backend', choices=['auto', 'numpy', 'jax'], default='auto',
        help='バッチ IK のバックエンド。既定 (auto) は jax が入って '
            'いれば jax、無ければ numpy。')
    parser.add_argument(
        '--base-x-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_X_RANGE),
        help='台車の前後方向 (x) の移動範囲 [m]。IK 開始時の台車位置を '
            '原点とする (既定 {} {})。'.format(*DEFAULT_BASE_X_RANGE))
    parser.add_argument(
        '--base-y-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_Y_RANGE),
        help='台車の左右方向 (y) の移動範囲 [m] (既定 {} {})。'.format(
            *DEFAULT_BASE_Y_RANGE))
    parser.add_argument(
        '--base-yaw-range', type=float, nargs=2, metavar=('MIN', 'MAX'),
        default=list(DEFAULT_BASE_YAW_RANGE),
        help='台車の向き (yaw) の範囲 [rad] (既定 {:.4f} {:.4f})。'.format(
            *DEFAULT_BASE_YAW_RANGE))
    parser.add_argument(
        '--seed', type=int, default=None,
        help='バッチ IK の乱数初期値に使う numpy の乱数シード。指定すると '
            '実行ごとに同じ解が得られる (既定は指定なし)。')
    args = parser.parse_args()

    files = iter_palm_files(args.input_dir)
    if not files:
        print('{} に掌の位置姿勢 JSON が見つかりません。先に '
              'estimate_palm_poses.py を実行してください。'.format(
                  args.input_dir))
        return

    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    # r/l_eef_grasp_link (IK が使う手先フレーム) は手あり/なし両方の URDF に
    # ある (Aero.__init__ 参照) ので、指の関節が要らないこのスクリプトでは
    # 手なしモデルを使う。
    robot = Aero(use_hand=False)

    base_limits = [tuple(args.base_x_range), tuple(args.base_y_range),
                   tuple(args.base_yaw_range)]
    backend = None if args.backend == 'auto' else args.backend

    # まず全ファイルを読んで、IK の対象になる人物 (掌推定が「この手を
    # 差し出している」と判定した人物) だけを集める。対象外の人物は IK の
    # 結果を持たない JSON をこの場で書き出す (view_handshake_poses.py が
    # それを見て「対象外」と表示する)。
    entries = []  # ファイル順の [(path, human_hand, robot_arm, palm) or None]
    n_not_target = 0
    for path in files:
        palms = load_palm_json(path)
        human_hand = palms.get('offered_hand')
        palm = palms.get(human_hand) if human_hand in ('L', 'R') else None
        if palm is None:
            reason = ('no_palm' if human_hand in ('L', 'R')
                      else 'no_offered_hand')
            save_json(not_target_result(human_hand, reason),
                      os.path.join(args.output_dir, os.path.basename(path)))
            n_not_target += 1
            entries.append((path, human_hand, None, None, reason))
            continue
        robot_arm = (DEFAULT_ROBOT_ARM[human_hand]
                     if args.robot_arm == 'auto' else args.robot_arm)
        entries.append((path, human_hand, robot_arm, palm, None))

    targets = [(path, human_hand, robot_arm, palm)
               for path, human_hand, robot_arm, palm, _ in entries
               if palm is not None]

    # 全人物 × 全候補の目標姿勢を腕ごとにまとめ、腕ごとに 1 回だけ
    # バッチ IK を呼ぶ (use_base 付きの呼び出しはキャッシュが効かないので、
    # ファイル単位で呼ぶと再 JIT/再構築で逆に遅くなる)。
    tasks = build_ik_tasks([palm for _, _, _, palm in targets],
                           [arm for _, _, arm, _ in targets])
    solutions = {}
    for robot_arm in ('r', 'l'):
        solutions[robot_arm] = solve_palm_ik_batch(
            robot, tasks[robot_arm], robot_arm,
            attempts_per_pose=args.attempts_per_pose,
            base_limits=base_limits, backend=backend)

    # 解を人物ごとに集約して書き出す。進捗表示は逐次版と同じ形式。
    person_index = 0
    n_solved = 0
    n_total = 0
    for i, (path, human_hand, robot_arm, palm, reason) in enumerate(entries):
        out_path = os.path.join(args.output_dir, os.path.basename(path))
        if palm is None:
            print('[{}/{}] {} -> {} (not target: {})'.format(
                i + 1, len(files), os.path.basename(path), out_path, reason))
            continue

        target_pos = palm_target_position(palm)
        rots = palm_to_target_rots(palm, robot_arm)
        picked = pick_solution(solutions[robot_arm], person_index)
        if picked is None:
            result = unsolved_result(robot, robot_arm, target_pos, rots[-1])
        else:
            candidate_index, angle_vector, base_pose, _ = picked
            result = solved_result(
                robot, robot_arm, target_pos, rots[candidate_index],
                candidate_index, angle_vector, base_pose)
        result['offered_hand'] = human_hand
        result['robot_arm'] = robot_arm
        person_index += 1
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
