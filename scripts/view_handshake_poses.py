#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``solve_palm_ik.py`` が出力したロボットの位置姿勢・関節角度 JSON
(``random_handshake_poses/human_xxx.json``) と、対応する
SMPL モデル (``random_human_poses/human_xxx.json`` の
``smpl.pose``/``betas``/``root_pos``/``gender``) をファイル名で突き合わせ
て読み込み、scikit-robot の viser ビューアで並べて表示する。

``draw_random_human_poses.py`` と違い、ビューアには SMPL メッシュとロボット
モデルの 2 つだけを表示する (骨格線・掌 Axis・ランドマーク球・ワールド
Axis・カメラは描かない)。ただし ``solve_palm_ik.py`` が干渉回避の障害物
として使ったのと同じ人体の近似ジオメトリ (``solve_palm_ik.human_body_
obstacles`` の Cylinder) は、SMPL メッシュに重ねて半透明
(``COLLISION_OBSTACLE_COLOR``) で表示する -- 解けなかった/危なかった
姿勢が体のどの部位のせいか目で見て確認できるようにするため。同様に、
``solve_palm_ik.py`` が干渉回避に使ったのと同じロボット自身の近似ジオメトリ
(box/cylinder/sphere のプリミティブ形状。``solve_palm_ik.apply_collision_
model`` / ``view_aero_collision_model.py`` と同じ ``skrobot.urdf.
convert_meshes_to_primitives`` で生成・キャッシュした URDF) も、
``view_aero_collision_model.py`` と全く同じ方法でもう一体の
``skrobot.model.RobotModel`` として読み込み、通常のロボットモデル (不透明)
に重ねて半透明 (``ROBOT_COLLISION_LINK_COLOR``、``human_palm_contact_
behavior.py`` が ``aero_demo.palm_plane_view.PalmPlaneScene`` 経由で人体
メッシュを半透明に描くのと同じ ``aero_demo.palm_plane_view.set_color`` を
使い、viser でも alpha が effective になるようにしてある) で表示する
(``build_robot_collision_overlay`` 参照)。viser 画面には
Back/Next ボタンに加え Good/Bad ボタンも表示され、押すと表示中の IK 結果が
正しいかどうかの判定結果 (``human_label``: ``true``: Good/``false``: Bad)
が対応する handshakes ディレクトリの JSON に書き込まれる (Next と同様に
次の人物へ進む)。書き込まれたラベルは viser 画面のテキストパネルにも表示
される。共通の Back/Next/Good/Bad ボタンや判定結果の読み書きは
``draw_random_human_poses.py`` と共有の ``aero_demo.viewer_nav`` を使う。

人間は、握手のときに実際そうするように、ロボットが触れている手先を見て
いる姿勢で描く。骨格 JSON の ``smpl.pose`` は顔の向きも乱数で決まって
いるので、``look_at_pose`` が SMPL の首 (``NECK``) と頭 (``HEAD``) だけを
回して、顔の正面がロボットの手先 (IK 結果 JSON の ``hand_position``。
IK が解けなかった人物では触れようとしていた ``target_position``) を向く
ようにしてから描画する。ロボットの首は ``solve_palm_ik.py`` が保存した
関節角 (IK は首を動かさないので ``reset_pose`` の値) をそのまま反映する
だけで、手先を見るようには動かさない。

``solve_palm_ik.py`` が IK の対象外にした人物 (掌推定の ``offered_hand``
が ``null``、つまりどちらの手も差し出していないと判定された人物。JSON の
``target`` が ``false``) は IK の結果が無いので、ビューアには表示せず
読み飛ばす。

IK が失敗した人物 (``solved`` が ``false``) では、どちらの手に合わせよう
としていたのかを確認できるように、人間の差し出した手 (``offered_hand``)
とロボットが使った腕 (``robot_arm``) をテキストパネルと標準出力に出す。
さらに、どれくらい届いていないのかが目で見て分かるように、IK の目標姿勢
(JSON の ``target_position``/``target_rot``) と、解けなかったときの手先
姿勢 (``hand_position``/``hand_rot``。``solve_palm_ik.unsolved_result``
が種の姿勢で読んだもの) を Axis としてビューアに描く (目標のほうが長い
Axis)。IK が成功した人物では手先が目標に一致しているので描かない。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py --num-samples 100
    rosrun aero_demo solve_palm_ik.py
    rosrun aero_demo view_handshake_poses.py

viser はブラウザで表示するビューアなので、実行するとブラウザが開く
(WSLg 環境などでは自動で開く)。ブラウザが自動で開かない場合は、標準出力
に表示される URL を手動で開くこと。画面下の Back/Next/Good/Bad ボタンで
人物の切り替えと判定を行う。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import trimesh

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from aero_demo import smpl_body  # noqa: E402  (パス追加後に import)
from aero_demo import viewer_nav  # noqa: E402
from aero_demo.palm_plane_view import set_color as set_translucent_color  # noqa: E402,E501

from generate_random_human_poses import load_smpl_models  # noqa: E402
from solve_palm_ik import human_body_obstacles  # noqa: E402
from solve_palm_ik import load_skeleton_json as load_joint_positions  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.model import Link  # noqa: E402
from skrobot.model import RobotModel  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

from view_aero_collision_model import build_collision_model_urdf  # noqa: E402

# SMPL メッシュの肌色 (RGBA, 0-255)。draw_random_human_poses.py と違い
# 人物ごとにランダムにはしない (このビューアは IK 結果の確認が目的で、
# 見た目のバリエーションは不要なため)。
SKIN_COLOR = [180, 130, 110, 255]

# solve_palm_ik.human_body_obstacles が作る干渉回避ジオメトリ (Cylinder)
# を表示する色 (RGBA, 0-255)。半透明にして SMPL メッシュ越しでも
# 円柱の位置関係が見えるようにする (alpha が 255 未満)。
COLLISION_OBSTACLE_COLOR = [80, 140, 220, 90]

# solve_palm_ik.py が干渉回避に使ったのと同じロボット自身の近似ジオメトリ
# (box/cylinder/sphere のプリミティブ形状。build_robot_collision_overlay
# 参照) を表示する色 (RGBA, 0-255)。人体側の COLLISION_OBSTACLE_COLOR
# (青系) と見分けられるよう、こちらは橙系にしてある。
ROBOT_COLLISION_LINK_COLOR = [220, 140, 80, 90]

# IK が失敗したときに描く Axis の大きさ [m]。Axis の色は 3 軸の RGB で
# 固定なので目標と手先を色では区別できない。代わりに長さで区別する
# (長いほうが目標、短いほうが実際の手先)。
TARGET_AXIS_LENGTH = 0.12
TARGET_AXIS_RADIUS = 0.005
HAND_AXIS_LENGTH = 0.06
HAND_AXIS_RADIUS = 0.005

# 視線 (顔の正面) の回転を SMPL の首 (NECK) と頭 (HEAD) に分ける割合。
# 首だけを回すと顔だけでなく肩の付け根近くから大きく曲がって見えるので、
# 首と頭で半分ずつ持つ。
GAZE_NECK_RATIO = 0.5

# 胸 (SMPL の首の親関節) の正面から顔の正面を離せる最大角度 [deg]。人間の
# 首がありえない角度までねじれて見えるのを防ぐ (実際の人間の首の可動域は
# 左右 70-80 度、下 60 度ほど)。生成された人物はほぼ全員この範囲内に手を
# 差し出しているので、普通は効かない安全弁。届かない向きでは顔が手先を
# 向ききらないだけで、それ以上は回さない。
GAZE_MAX_ANGLE_DEG = 80.0

# 視線合わせを何回繰り返すか。首を回すと頭 (視線の始点) 自身も動くので、
# 1 回では手先の方向から少しずれる。動いた頭の位置で解き直すことで、
# 2 回でほぼ収束する (残差は 1 度以下)。
GAZE_ITERATIONS = 2

# SMPL の関節はすべて静止姿勢で回転が単位行列なので、頭の「正面」は
# 静止姿勢の体の正面と同じ (ロボット座標系の +x, smpl_body の PERM に
# よる軸対応で SMPL ローカルの +z)。
GAZE_FORWARD_AXIS = np.array([1.0, 0.0, 0.0])


def load_skeleton_json(path):
    """``generate_random_human_poses.build_person_json`` が保存した骨格
    JSON から SMPL のパラメータだけを読む."""
    with open(path) as f:
        data = json.load(f)
    smpl = data['smpl']
    return dict(
        gender=smpl['gender'],
        betas=np.asarray(smpl['betas'], dtype=np.float64),
        pose=np.asarray(smpl['pose'], dtype=np.float64),
        root_pos=np.asarray(smpl['root_pos'], dtype=np.float64))


def load_handshake_json(path):
    """``solve_palm_ik.save_json`` が保存した IK 結果 JSON を読む."""
    with open(path) as f:
        return json.load(f)


def smpl_world_rots(model, pose):
    """SMPL の各関節のワールド (ロボット座標系) 回転行列 ``(24, 3, 3)``.

    ``smpl_body.forward_world`` は頂点と関節位置しか返さないので、視線を
    合わせるのに要る「頭が今どちらを向いているか」をここで計算する。
    ``pose`` の各要素は SMPL ローカル座標系の親関節相対 axis-angle なので、
    ロボット座標系の回転に直して (``smpl_body.PERM`` による軸の対応、
    ``to_smpl_rotation`` の逆) 根元から掛けていく。root_rot は
    ``build_smpl_mesh`` (``forward_world`` の既定) と同じく単位行列。
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(24, 3)
    world_rots = np.zeros((24, 3, 3))
    for i in range(24):
        local = smpl_body.PERM.dot(smpl_body.rodrigues(pose[i])).dot(
            smpl_body.PERM.T)
        parent = model.parent[i]
        world_rots[i] = local if parent < 0 \
            else world_rots[parent].dot(local)
    return world_rots


def limit_direction(base, direction, max_angle):
    """``direction`` を ``base`` から ``max_angle`` [rad] 以内に丸める.

    視線の回転量ではなく目標の「方向」を丸めるので、``look_at_pose`` の
    繰り返しは丸めた向きに収束する (回転量を毎回クリップすると、次の
    繰り返しが残差を足してクリップを打ち消してしまう)。
    """
    angle = np.arccos(np.clip(float(np.dot(base, direction)), -1.0, 1.0))
    if angle <= max_angle:
        return direction
    axis = np.cross(base, direction)
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        # 真後ろ (180 度): 回す軸が決まらないので諦めて元の向きのまま。
        return base
    return smpl_body.rodrigues(axis / norm * max_angle).dot(base)


def look_at_pose(model, person, target_position,
                 neck_ratio=GAZE_NECK_RATIO,
                 max_angle_deg=GAZE_MAX_ANGLE_DEG):
    """人間が ``target_position`` を見るように首/頭を回した pose を返す.

    握手のように手を触れ合わせる動作では、人間は触れている手先を見ている
    のが自然なので、``generate_random_human_poses.py`` が乱数で決めた顔の
    向き (骨格 JSON の ``smpl.pose``) を、ロボットの手先 (触れている手先)
    を向くように上書きして描画する。動かすのは人間の首/頭だけで、ロボット
    の首は ``solve_palm_ik.py`` が保存した関節角のまま。

    顔の正面 (``GAZE_FORWARD_AXIS``) を頭の関節位置から
    ``target_position`` へ向ける最小回転を求め、それを首 (``NECK``) と
    頭 (``HEAD``) に ``neck_ratio`` : ``1 - neck_ratio`` で分けて入れる
    (2 関節は同じ軸まわりに回すので、合成すると狙った回転になる)。首から
    上以外の関節は触らないので、体の姿勢は乱数生成されたまま。首を回すと
    視線の始点である頭の関節自身も動くので、``GAZE_ITERATIONS`` 回だけ
    解き直して残差を詰める (実測 0.1 度以下まで収束する)。

    Parameters
    ----------
    model : smpl_body.SmplModel
    person : dict
        ``load_skeleton_json`` の戻り値 (``pose``/``betas``/``root_pos``)。
    target_position : array_like
        見てほしい点 (ロボット座標系)。ロボットの手先位置を渡す想定。
    neck_ratio : float, optional
        視線の回転を首と頭に分ける割合 (既定 ``GAZE_NECK_RATIO``)。
    max_angle_deg : float, optional
        胸の正面から顔の正面を離せる最大角度 [deg]
        (既定 ``GAZE_MAX_ANGLE_DEG``)。

    Returns
    -------
    pose : ndarray(24, 3)
        首/頭だけ差し替えた新しい pose (``person['pose']`` は変更しない)。
    """
    pose = np.array(person['pose'], dtype=np.float64).reshape(24, 3)
    target_position = np.asarray(target_position, dtype=np.float64)
    max_angle = np.deg2rad(max_angle_deg)

    for _ in range(GAZE_ITERATIONS):
        _vertices, joints = smpl_body.forward_world(
            model, pose, person['betas'], person['root_pos'])
        world_rots = smpl_world_rots(model, pose)
        forward = world_rots[smpl_body.HEAD].dot(GAZE_FORWARD_AXIS)
        # 首をひねれる限界は胸 (首の親関節) の正面から測る -- 首/頭より
        # 下は動かさないので、この向きは繰り返しても変わらない。
        chest_forward = world_rots[model.parent[smpl_body.NECK]].dot(
            GAZE_FORWARD_AXIS)

        direction = target_position - joints[smpl_body.HEAD]
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            break
        direction = limit_direction(
            chest_forward, direction / norm, max_angle)
        axis_angle = smpl_body.mat_to_axis_angle(
            smpl_body.rotation_between(forward, direction))
        if np.linalg.norm(axis_angle) < 1e-9:
            break

        # 首 -> 頭の順に、ワールドでの回転を親の座標系に移して入れる。
        # 首を回すと頭の親 (首) のワールド回転も変わるので、頭の分は
        # 「首を回した後」の首のワールド回転を親として計算する。
        parent_world = world_rots[model.parent[smpl_body.NECK]]
        accumulated = np.eye(3)
        for joint_index, ratio in ((smpl_body.NECK, neck_ratio),
                                   (smpl_body.HEAD, 1.0 - neck_ratio)):
            accumulated = smpl_body.rodrigues(
                axis_angle * ratio).dot(accumulated)
            new_world = accumulated.dot(world_rots[joint_index])
            pose[joint_index] = smpl_body.mat_to_axis_angle(
                smpl_body.to_smpl_rotation(parent_world.T.dot(new_world)))
            parent_world = new_world

    return pose


def build_smpl_mesh(model, person, pose=None):
    """保存済みの SMPL pose/betas/root_pos からメッシュを作る
    (``smpl_body.forward_world`` を使うのは draw_random_human_poses.py と
    同じ)。

    ``pose`` を渡すと ``person['pose']`` の代わりにそれを使う
    (``look_at_pose`` が首/頭を差し替えた pose を描くため)。
    """
    if pose is None:
        pose = person['pose']
    vertices, _joints = smpl_body.forward_world(
        model, pose, person['betas'], person['root_pos'])
    mesh = trimesh.Trimesh(vertices=vertices, faces=model.f, process=False)
    mesh.visual.face_colors = SKIN_COLOR
    return mesh


def is_target(handshake):
    """``solve_palm_ik.py`` が IK の対象にした人物かどうか.

    対象外 (掌推定の ``offered_hand`` が ``null`` で、どちらの手も差し
    出していないと判定された人物) の JSON は ``target`` が ``false`` で、
    IK の結果 (関節角・台車位置) を持たない (``solve_palm_ik.
    not_target_result``)。このビューアは対象外の人物を表示しない。
    ``target`` キーを持たない JSON は、キーが無かった頃の
    solve_palm_ik.py が IK を解いた結果なので対象として扱う。
    """
    return bool(handshake.get('target', True))


def hand_text(handshake):
    """どちらの手に合わせようとしたかを viser の画面に出すための文字列.

    IK が失敗したときに、人間のどちらの手 (掌推定の ``offered_hand``:
    ``'L'``/``'R'``) を握手の相手として狙い、ロボットのどちらの腕
    (``robot_arm``: ``'l'``/``'r'``) で解こうとしていたのかを示す。
    """
    offered = handshake.get('offered_hand')
    robot_arm = handshake.get('robot_arm')
    return '人間の {} 手 -> ロボットの {}arm'.format(
        {'L': '左', 'R': '右'}.get(offered, '不明 ({})'.format(offered)),
        robot_arm if robot_arm is not None else '不明')


def pose_coords(handshake, pos_key, rot_key):
    """IK 結果 JSON の位置 (``pos_key``) と回転行列 (``rot_key``) から
    ``Coordinates`` を作る。どちらかが無ければ ``None`` を返す
    (``target`` が ``false`` の JSON や、これらのキーを持たなかった頃の
    solve_palm_ik.py の出力のため)。"""
    position = handshake.get(pos_key)
    rot = handshake.get(rot_key)
    if position is None or rot is None:
        return None
    return Coordinates(pos=np.asarray(position, dtype=np.float64),
                       rot=np.asarray(rot, dtype=np.float64))


def gaze_target_position(handshake):
    """人間に見せる点 (``look_at_pose`` に渡す注視点) を IK 結果から選ぶ.

    IK が解けた人物では、実際にロボットが触れている手先
    (``hand_position``) を見る。解けなかった人物の手先は種の姿勢のまま
    (ロボットの体の近く) でどこにも触れていないので、代わりに触れよう
    としていた点 (``target_position``、人間の掌の少し手前) を見る --
    どちらの手を狙っていたのかを目で追えるようにするため。どちらのキーも
    無い JSON (これらのキーを持たなかった頃の solve_palm_ik.py の出力) は
    ``None`` を返し、乱数生成された顔の向きをそのまま使う。
    """
    key = 'hand_position' if handshake.get('solved') else 'target_position'
    position = handshake.get(key)
    if position is None:
        return None
    return np.asarray(position, dtype=np.float64)


def pose_error_text(target_coords, hand_coords):
    """目標姿勢と手先姿勢のずれ (位置 [m] と向き [deg]) の文字列.

    向きのずれは相対回転 ``target^-1 * hand`` の回転角 (軸は問わない)。
    """
    diff = hand_coords.worldpos() - target_coords.worldpos()
    rel = np.dot(target_coords.worldrot().T, hand_coords.worldrot())
    # 数値誤差で arccos の定義域を外れることがあるのでクリップする。
    angle = np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))
    return '位置ずれ {:.3f} m, 向きずれ {:.1f} deg'.format(
        float(np.linalg.norm(diff)), float(np.rad2deg(angle)))


def apply_robot_pose(robot, handshake):
    """``solve_palm_ik`` の戻り値 (関節角・台車位置姿勢) をロボットモデル
    に反映する.

    ``joint_angle_vector``/``joint_names`` は ``solve_palm_ik.py`` が
    ``use_hand=False`` (指関節なし) のロボットで解いた際の ``robot.joint_list``
    の角度なので、指関節ありのロボット (``--no-hand`` を付けない既定の表示
    モデル) とは ``joint_list`` の要素数・並びが異なる。そのため
    ``robot.angle_vector`` にそのまま渡さず、``joint_names`` で名前を突き
    合わせて該当する関節だけ角度を反映する (指関節は初期姿勢のまま)。
    台車の位置・向きは ``base_position``/``base_yaw`` に別で保存されている
    ので、あわせて反映する (``solve_palm_ik.solve_palm_ik`` 参照)。
    """
    robot.reset_pose()
    name_to_angle = dict(zip(
        handshake['joint_names'], handshake['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.base_link.newcoords(Coordinates(
        pos=handshake['base_position'],
        rot=rpy_matrix(handshake['base_yaw'], 0.0, 0.0)))


def build_robot_collision_overlay(robot, primitive_type=None,
                                  force_convert=False):
    """``solve_palm_ik.py`` が干渉回避に使ったのと同じロボット自身の近似
    ジオメトリを、``view_aero_collision_model.py`` と全く同じ方法でもう
    一体の ``RobotModel`` として読み込む。

    ``build_collision_model_urdf`` (``view_aero_collision_model.py`` /
    ``solve_palm_ik.apply_collision_model`` と共通) が ``robot.urdf_path``
    から生成した box/cylinder/sphere のプリミティブ近似 URDF をキャッシュ
    する (既に生成済みならそれを再利用し、``force_convert`` を指定した
    ときだけ作り直す) ので、``skr convert-urdf-to-primitives`` で見える
    のと同じ形状がそのまま overlay になる (``solve_palm_ik.py`` の IK が
    球近似で使っているジオメトリの元でもある)。``robot`` 自体は変更しない。

    読み込んだ ``collision_robot`` は ``robot`` と全く同じ URDF (ジオメトリ
    以外) から作られるため、リンク・関節の構成は ``robot`` と同一
    (``use_hand`` の値によらず ``robot.urdf_path`` を使うので、指ありなし
    どちらのモデルでも動く)。呼び出し側は毎フレーム
    ``sync_robot_collision_overlay`` で ``robot`` の現在の姿勢に追従させる。

    Returns
    -------
    skrobot.model.RobotModel
        半透明 (``ROBOT_COLLISION_LINK_COLOR``) に色付け済みの overlay。
    """
    collision_urdf_path = build_collision_model_urdf(
        robot.urdf_path, primitive_type=primitive_type, force=force_convert)
    collision_robot = RobotModel()
    collision_robot.load_urdf_file(
        str(collision_urdf_path), include_mimic_joints=False)
    for link in collision_robot.link_list:
        set_translucent_color(link, ROBOT_COLLISION_LINK_COLOR)
    return collision_robot


def sync_robot_collision_overlay(collision_robot, robot):
    """``build_robot_collision_overlay`` が返した overlay を ``robot`` の
    現在の姿勢 (関節角・台車位置姿勢) に追従させる.

    ``collision_robot`` は ``robot`` と同じ URDF から作られているので
    (``build_robot_collision_overlay`` 参照)、関節の構成は完全に一致し、
    ``angle_vector`` をそのままコピーできる。台車の位置姿勢は
    ``robot.base_link.copy_worldcoords()`` を ``collision_robot`` に
    反映する (Aero は ``root_link`` が ``base_link`` そのものなので、
    ``newcoords`` で台車の移動も含めて反映される。``seed_arm_pose`` の
    注記を参照)。
    """
    collision_robot.angle_vector(robot.angle_vector())
    collision_robot.newcoords(robot.base_link.copy_worldcoords())


def iter_common_names(skeleton_dir, handshake_dir):
    """``skeleton_dir``/``handshake_dir`` の両方に存在するファイル名
    (basename) をファイル名順に列挙する.

    IK の対象外だった人物 (``is_target`` が ``False``) はビューアに表示
    しないので、ここで読み飛ばす。
    """
    skeleton_names = {os.path.basename(p) for p in
                      glob.glob(os.path.join(skeleton_dir, '*.json'))}
    handshake_names = {os.path.basename(p) for p in
                       glob.glob(os.path.join(handshake_dir, '*.json'))}
    return sorted(
        name for name in skeleton_names & handshake_names
        if is_target(load_handshake_json(os.path.join(handshake_dir, name))))


def main():
    parser = argparse.ArgumentParser(
        description='solve_palm_ik.py が出力したロボットの位置姿勢・関節'
                    '角度と、対応する SMPL モデルを viser で表示する '
                    '(ビューアに表示するのは SMPL モデルとロボットモデル'
                    'だけ)。')
    parser.add_argument(
        '--skeleton-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='SMPL pose/betas/root_pos を持つ骨格 JSON のディレクトリ '
            '(既定は generate_random_human_poses.py の既定の出力先と '
            '同じ random_human_poses/)。')
    parser.add_argument(
        '--handshake-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_handshake_poses'),
        help='solve_palm_ik.py が出力した JSON のディレクトリ (既定は '
            'solve_palm_ik.py の既定の出力先と同じ '
            'random_handshake_poses/。skeleton-dir と同じファイル名で '
            '対応させる)。')
    parser.add_argument(
        '--model-path', type=str,
        default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicmodel_m_lbs_10_207_0_v1.0.0.pkl'),
        help='SMPL (男性) モデル .pkl のパス。')
    parser.add_argument(
        '--female-model-path', type=str,
        default=os.path.expanduser(
            '~/SMPL_python_v.1.0.0/smpl/models/'
            'basicModel_f_lbs_10_207_0_v1.0.0.pkl'),
        help='SMPL (女性) モデル .pkl のパス (無ければ男性モデルのみ使う)。')
    parser.add_argument(
        '--no-hand', dest='use_hand', action='store_false',
        help='指関節なしの URDF (solve_palm_ik.py と同じ手なしモデル) を '
            '使う。既定では指関節ありの URDF (aero_with_feetech_hand) を '
            '使い、手先にハンドを表示する。')
    parser.set_defaults(use_hand=True)
    parser.add_argument('--client-wait-timeout', type=float, default=30.0,
                        help='ブラウザクライアント接続を待つ 1 回あたりの'
                             '秒数 (繰り返し待つ)。')
    parser.add_argument('--no-open-browser', action='store_true',
                        help='ブラウザの自動起動を無効にする '
                             '(URL を自分で開く場合)。')
    parser.add_argument(
        '--collision-primitive-type', choices=['box', 'cylinder', 'sphere'],
        default=None,
        help='ロボット自身の干渉モデル (オレンジ色の球, '
            'ROBOT_COLLISION_LINK_COLOR) の元になるジオメトリを、指定した '
            '形状に全リンク強制変換する (solve_palm_ik.py / '
            'view_aero_collision_model.py の --collision-primitive-type / '
            '--primitive-type と同じ。既定 (未指定) はリンクごとに自動選択 '
            '-- solve_palm_ik.py を同じオプションで実行した場合はそれと '
            '揃える)。')
    parser.add_argument(
        '--force-convert-collision-model', action='store_true',
        help='ロボット自身の干渉モデル (プリミティブ近似 URDF) のキャッシュ '
            'を使わず毎回作り直す (solve_palm_ik.py / '
            'view_aero_collision_model.py の --force-convert-collision-model '
            '/ --force-convert と同じ)。')
    args = parser.parse_args()

    names = iter_common_names(args.skeleton_dir, args.handshake_dir)
    if not names:
        print('{} と {} の両方に対応する IK 対象のファイルが見つかりません。'
              '先に solve_palm_ik.py を実行してください。'.format(
                  args.skeleton_dir, args.handshake_dir))
        return

    models_by_gender = dict(
        load_smpl_models(args.model_path, args.female_model_path))

    # r/l_eef_grasp_link (solve_palm_ik.py が使う手先フレーム) は手あり/
    # なし両方の URDF にあるので、IK 結果自体は --no-hand でも変わらない。
    # 既定では見た目のために手ありモデルを使う。
    robot = Aero(use_hand=args.use_hand)

    viewer = ViserViewer(draw_grid=True)
    # Back/Next/Good/Bad ボタンは、ロボットモデルを追加するより前に作る。
    # ViserViewer は RobotModel を add() すると "Joint Angles" フォルダ
    # (関節ごとのスライダー) を自動で GUI パネルに追加してしまい
    # (skrobot.viewers._viser.ViserViewer._ensure_gui_initialized/_add_
    # joint_sliders)、Aero は関節数が多いためこのボタン群を先に追加
    # しないと大量のスライダーの下に埋もれて見えなくなる
    # (draw_random_human_poses.py はロボットモデルを表示しないため
    # この問題が起きない)。
    nav = viewer_nav.ManualNav(viewer)
    label_text = viewer._server.gui.add_markdown('')

    # 干渉回避用の半透明モデル (人体の障害物 Cylinder ``current_obstacle_
    # links`` とロボットの干渉モデル overlay ``robot_collision_overlay``)
    # の表示/非表示をまとめて切り替えるチェックボックス。viser の
    # SceneNodeHandle は ``.visible`` を持つので、delete/add をやり直さず
    # 既に追加済みのオブジェクトの表示だけを切り替えられる
    # (``ViserViewer.add`` はハンドルを返さないので
    # ``viewer._linkid_to_handle`` から引く)。
    show_collision_models_checkbox = viewer._server.gui.add_checkbox(
        '干渉回避用モデルの表示', initial_value=True)

    def set_link_visible(link, visible):
        viewer._linkid_to_handle[str(id(link))].visible = visible

    @show_collision_models_checkbox.on_update
    def _on_toggle_collision_models(_):  # noqa: ANN001
        visible = show_collision_models_checkbox.value
        for link in robot_collision_overlay.link_list:
            set_link_visible(link, visible)
        for obstacle_link in current_obstacle_links:
            set_link_visible(obstacle_link, visible)

    viewer.add(robot)
    # solve_palm_ik.py の IK が干渉回避に使ったのと同じプリミティブ近似
    # ジオメトリ (view_aero_collision_model.py と同じもの) を、人物を
    # またいで一度だけ作って重ねて表示する (毎フレーム作り直す必要はなく、
    # sync_robot_collision_overlay で robot の姿勢に追従させるだけでよい)。
    robot_collision_overlay = build_robot_collision_overlay(
        robot,
        primitive_type=args.collision_primitive_type,
        force_convert=args.force_convert_collision_model)
    viewer.add(robot_collision_overlay)
    viewer.show(open_browser=not args.no_open_browser)
    viewer_nav.wait_for_client(viewer, args.client_wait_timeout)
    # 人物は常に原点で +x 方向を向いて生成されるので、+x 側から -x 方向を
    # 見るカメラ (draw_random_human_poses.py と同じ視点) で人物を正面から
    # 見ることになる (ロボットは人物の正面に立つので、カメラと人物の間に
    # 入る)。
    viewer_nav.set_front_view(viewer)

    # IK が失敗した人物のときだけ表示する目標姿勢/手先姿勢の Axis。
    # draw_random_human_poses.py の掌 Axis と同じく、あらかじめ 1 組だけ
    # 作っておいて座標を更新して viewer に足す/外す (毎回作り直さない)。
    target_axis = Axis(axis_length=TARGET_AXIS_LENGTH,
                       axis_radius=TARGET_AXIS_RADIUS)
    hand_axis = Axis(axis_length=HAND_AXIS_LENGTH,
                     axis_radius=HAND_AXIS_RADIUS)
    axes_added = False

    current_mesh_link = None
    current_obstacle_links = []
    i = 0
    while 0 <= i < len(names):
        name = names[i]
        skeleton_path = os.path.join(args.skeleton_dir, name)
        handshake_path = os.path.join(args.handshake_dir, name)
        person = load_skeleton_json(skeleton_path)
        handshake = load_handshake_json(handshake_path)

        model = models_by_gender.get(
            person['gender'], models_by_gender['male'])
        # 触れている手先を人間も見ているように、乱数生成された顔の向きを
        # 上書きして描く (ロボットの首は apply_robot_pose が保存された
        # 関節角をそのまま反映するだけ)。
        gaze_target = gaze_target_position(handshake)
        pose = None if gaze_target is None \
            else look_at_pose(model, person, gaze_target)
        mesh = build_smpl_mesh(model, person, pose)
        link = Link(visual_mesh=mesh, name='smpl_human')
        if current_mesh_link is not None:
            viewer.delete(current_mesh_link)
        viewer.add(link)
        current_mesh_link = link

        # solve_palm_ik.py が干渉回避の障害物として使ったのと同じ人体の
        # 近似ジオメトリ (体幹・頭部・四肢・掌・指すべて Cylinder) を、
        # SMPL メッシュに重ねて半透明で表示する
        # (COLLISION_OBSTACLE_COLOR)。解けなかった/危なかった姿勢が
        # どの部位のせいか目で見て確認できるようにするため。
        for obstacle_link in current_obstacle_links:
            viewer.delete(obstacle_link)
        current_obstacle_links = human_body_obstacles(
            load_joint_positions(skeleton_path))
        for obstacle_link in current_obstacle_links:
            set_translucent_color(obstacle_link, COLLISION_OBSTACLE_COLOR)
            viewer.add(obstacle_link)
            # 新しく追加した障害物は既定で表示状態になるので、チェック
            # ボックスで非表示にされていたら合わせる。
            set_link_visible(obstacle_link,
                             show_collision_models_checkbox.value)

        apply_robot_pose(robot, handshake)
        # 通常のロボットモデル (不透明) に重ねた半透明の干渉モデル overlay
        # も、動いた robot の現在の姿勢 (関節角・台車位置姿勢) に追従させる。
        sync_robot_collision_overlay(robot_collision_overlay, robot)

        # IK が失敗したときは、どちらの手に合わせようとしていたのかが
        # 分かるように人間の手とロボットの腕もあわせて出す。
        # 失敗したときは、どこに届かなかったのかが目で見て分かるように
        # 目標姿勢 (長い Axis) と手先姿勢 (短い Axis) も描く。
        target_coords = pose_coords(
            handshake, 'target_position', 'target_rot')
        hand_coords = pose_coords(handshake, 'hand_position', 'hand_rot')
        show_axes = (not handshake.get('solved')
                     and target_coords is not None
                     and hand_coords is not None)
        if show_axes:
            target_axis.newcoords(target_coords)
            hand_axis.newcoords(hand_coords)
            if not axes_added:
                viewer.add(target_axis)
                viewer.add(hand_axis)
                axes_added = True
        elif axes_added:
            viewer.delete(target_axis)
            viewer.delete(hand_axis)
            axes_added = False

        if handshake.get('solved'):
            status = 'solved'
        else:
            status = 'NOT solved ({})'.format(hand_text(handshake))
        # ずれの数値は 1 行に収まらないので、標準出力の 1 行 (status) には
        # 入れずテキストパネルにだけ出す。
        detail = ''
        if show_axes:
            detail = '\n\n目標 Axis (長い方, 長さ {:.2f} m) と手先 Axis ' \
                '(短い方, 長さ {:.2f} m): {}'.format(
                    TARGET_AXIS_LENGTH, HAND_AXIS_LENGTH,
                    pose_error_text(target_coords, hand_coords))
        label_text.content = '**{}** ({}/{})  IK: {}{}\n\n{}'.format(
            name, i + 1, len(names), status, detail,
            viewer_nav.format_label_text(handshake.get('human_label')))

        viewer.redraw()
        print('[{}/{}] displayed {} ({})'.format(
            i + 1, len(names), name, status))

        direction, label = nav.wait(viewer)
        if direction == 0:
            print('ブラウザクライアントが切断されました。中断します。')
            break
        if label is not viewer_nav.NOT_PRESSED:
            viewer_nav.save_label(handshake_path, label)
            print('  -> {} として {} に記録しました。'.format(
                'Good' if label else 'Bad', handshake_path))
        i = max(0, i + direction)

    viewer.close()


if __name__ == '__main__':
    main()
