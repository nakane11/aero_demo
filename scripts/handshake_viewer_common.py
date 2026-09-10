#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``view_handshake_poses.py``/``view_handshake_motion.py``/``scripts/ros/
run_camera_pipeline_test.py`` の 3 つの viser ビューアが、それぞれ独立に
(ほぼ同一のコードで) 持っていた次の処理をまとめる共通モジュール。

* ``solve_palm_ik.py``/``plan_handshake_motion.py`` が干渉回避に使ったのと
  同じロボット自身の近似ジオメトリ (box/cylinder/sphere) を、表示用の
  ロボットモデルに重ねて半透明で表示する overlay の構築・追従
  (``build_robot_collision_overlay``/``sync_robot_collision_overlay``)
* ``solve_palm_ik.collision_pairs_min_distance`` と同じ厳密な形状・許容
  誤差で、実際に貫通しているリンクの組み合わせをすべて列挙する事後検証
  (``colliding_link_pairs``) と、その結果をテキストパネル用の文字列にする
  (``collision_pairs_text``)
* ``solve_palm_ik.py``/``plan_handshake_motion.py`` が出力した関節角・
  台車位置姿勢を表示用ロボットに反映する (``apply_robot_pose``/
  ``apply_waypoint_pose``)
* 経路の最後に、後処理判定 (``post_process`` = 掌へのわずかな押し込み) まで
  の補間フレームを表示専用で追加する (``build_display_waypoints``)
* viser の ``SceneNodeHandle`` の表示/非表示を切り替える
  (``set_link_visible``)

3 つのビューアはいずれも表示専用の派生 (IK/軌道計画そのものではない) で、
差分は「SMPL メッシュを描くかどうか」「waypoint スライダーがあるかどうか」
「合成骨格か実カメラの骨格か」といった上位の構成だけなので、上記の下請け
処理はこのモジュールに一本化する。

``colliding_link_pairs`` は ``solve_palm_ik.py`` (``scripts/`` 直下) の
``human_obstacle_names`` に依存するため、``scripts/`` を ``sys.path`` に
含めた状態で import すること
(``view_handshake_poses.py``/``view_handshake_motion.py`` は自分自身が
``scripts/`` にあるため素の import で足りる。``scripts/ros/`` 以下からは
``run_camera_pipeline_test.py`` が既に行っている ``scripts/`` の sys.path
追加で足りる)。
"""

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import solve_palm_ik as spik  # noqa: E402
from view_aero_collision_model import build_collision_model_urdf  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402
from skrobot.model import RobotModel  # noqa: E402

from aero_demo.palm_plane_view import set_color as set_translucent_color  # noqa: E402,E501


# ロボット自身の干渉回避用近似ジオメトリ (box/cylinder/sphere のプリミティブ
# 形状。build_robot_collision_overlay 参照) を、通常のロボットモデル
# (不透明) に重ねて表示する色 (RGBA, 0-255)。
ROBOT_COLLISION_LINK_COLOR = [220, 140, 80, 90]

# solve_palm_ik.human_body_obstacles が作る人体側の干渉回避用近似ジオメトリ
# (Cylinder) を表示する色 (RGBA, 0-255)。ロボット側の
# ROBOT_COLLISION_LINK_COLOR (橙系) と見分けられるよう青系にしてある。
HUMAN_COLLISION_OBSTACLE_COLOR = [80, 140, 220, 90]

# 経路の最後に表示専用で追加する、後処理判定 (post_process = 掌への
# 押し込み) までの補間フレーム数 (build_display_waypoints 参照)。
# plan_handshake_motion.py はこの区間を経路として計画・検証しない (接触
# そのものが目的の動きで、経路上の干渉検証にはなじまないため)。
PRESS_IN_DISPLAY_WAYPOINTS = 5


def set_link_visible(viewer, link, visible):
    """``viewer.add`` 済みの ``link`` の表示/非表示を切り替える.

    人物切り替え/RESET によるリンクの削除・再作成と、チェックボックスの
    ``on_update`` (viser の GUI コールバックは別スレッドで実行される) が
    競合すると、既に削除されて ``viewer._linkid_to_handle`` に存在しない
    リンクを渡されることがある。その場合は何もしない (どうせ表示すべき
    対象ではない)。
    """
    handle = viewer._linkid_to_handle.get(str(id(link)))
    if handle is not None:
        handle.visible = visible


def build_robot_collision_overlay(robot, primitive_type=None,
                                  force_convert=False):
    """``solve_palm_ik.py``/``plan_handshake_motion.py`` が干渉回避に
    使ったのと同じロボット自身の近似ジオメトリを、``view_aero_collision_
    model.py`` と全く同じ方法でもう一体の ``skrobot.model.RobotModel``
    として読み込む。

    ``build_collision_model_urdf`` (``view_aero_collision_model.py``/
    ``solve_palm_ik.apply_collision_model`` と共通) が ``robot.urdf_path``
    から生成した box/cylinder/sphere のプリミティブ近似 URDF をキャッシュ
    する (既に生成済みならそれを再利用し、``force_convert`` を指定した
    ときだけ作り直す) ので、``skr convert-urdf-to-primitives`` で見えるの
    と同じ形状がそのまま overlay になる。``robot`` 自体は変更しない。

    読み込んだ overlay は ``robot`` と全く同じ URDF (ジオメトリ以外) から
    作られるため、リンク・関節の構成は ``robot`` と同一になる
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

    関節名で突き合わせて反映するので、``collision_robot`` と ``robot`` の
    関節構成 (要素数・並び) が完全に一致していなくても動作する (例:
    指なしの ``self.robot`` で解いた IK 結果を、指ありの表示用ロボットの
    overlay に反映する場合。一致していなくても余分な関節は無視されるだけ)。
    台車の位置姿勢は ``robot.base_link.copy_worldcoords()`` を
    ``collision_robot`` に反映する (Aero は ``root_link`` が ``base_link``
    そのものなので、``newcoords`` で台車の移動も含めて反映される)。
    """
    name_to_angle = {joint.name: joint.joint_angle()
                     for joint in robot.joint_list}
    for joint in collision_robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    collision_robot.newcoords(robot.base_link.copy_worldcoords())


def colliding_link_pairs(robot, pairs, obstacle_links,
                         tolerance=spik.DEFAULT_COLLISION_VERIFY_TOLERANCE):
    """``solve_palm_ik.collision_pairs_min_distance`` (``solve_palm_ik.py``
    の事後検証。IK の収束判定が見ない干渉ペナルティの残差を、厳密な形状
    (``collision_mesh`` の頂点そのもの。勾配降下法内部が使う粗い球近似では
    ない) で採用前にチェックする処理) と同じ考え方 (``collision_mesh`` の
    頂点同士の最短距離) で、``pairs`` (``solve_palm_ik.build_collision_
    verification_pairs`` が作る自己干渉・人体との干渉の総当たりの組み合わせ)
    の中から実際に貫通している組み合わせを**すべて**列挙する
    (``collision_pairs_min_distance`` は最小距離しか返さないため、表示用に
    ここで作り直す)。

    人体側は ``solve_palm_ik.human_capsules`` の解析的な (線分, 半径) では
    なく、画面に表示している ``obstacle_links`` (``solve_palm_ik.human_body_
    obstacles`` が返す ``Cylinder`` そのもの) をそのまま使う。掌
    (``R_palm``/``L_palm``) のように解析的な捉え方 (球近似) と実際の表示
    形状 (掌面に沿った平たい円柱) が一致しない部位があり、見た目は貫通して
    いるのに数値上は貫通していないと判定される (またはその逆の) 食い違いが
    起きうるため、「画面に見えている半透明メッシュ」を判定にもそのまま使う
    ことでこの食い違いを無くす。

    ロボット側リンクの各頂点が、その ``obstacle_links[other]`` (常に
    ``skrobot.model.primitives.Cylinder``、``solve_palm_ik.human_body_
    obstacles`` 参照) の中にどれだけ入り込んでいるかを、円柱自身の
    ``radius``/``height`` から解析的に求める (単純な頂点同士の最短距離だと、
    指のように細いリンクが障害物の「表面」にかすらず内部深くへ潜り込んだ
    場合に、頂点同士は互いの表面近くまで来ないため貫通を見逃してしまう。
    ``trimesh.proximity.signed_distance`` でも同じことは求まるが、この
    ペア数 (ロボットの全リンク × 人体セグメント数) でリアルタイムに使うには
    汎用メッシュのレイキャストは遅すぎるため、円柱に特化した解析式にする)。
    ロボットの自己干渉 (``other`` が ``Link``) は従来通り頂点同士の最短
    距離のまま (自己干渉ペアはどちらも薄いリンク同士がほとんどで、この
    見逃しが実質問題にならないため)。

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        干渉ジオメトリ (プリミティブ近似済みの ``collision_mesh``) を持つ、
        現在の姿勢のロボット (通常は ``build_robot_collision_overlay`` が
        返した overlay を ``sync_robot_collision_overlay`` で同期した後の
        もの)。
    pairs : list of (Link, Link) or (Link, int)
        ``build_collision_verification_pairs`` の戻り値。2 要素目が ``int``
        なら ``human_obstacle_names()`` の人体セグメントとの組み合わせ、
        ``Link`` ならロボット自身の自己干渉の組み合わせ。
    obstacle_links : list of Link or empty
        ``solve_palm_ik.human_body_obstacles(joint_positions)`` の戻り値
        (``human_obstacle_names()`` と同じ順序の ``Cylinder`` のリスト。
        画面に表示中のものをそのまま渡す想定)。空 (``[]``) なら人体との
        干渉ペア (``other`` が ``int``) は判定できないので読み飛ばす
        (自己干渉ペアは判定する)。

    Returns
    -------
    list of (str, str, str, float)
        ``(種別, リンク A の名前, リンク B の名前 (人体セグメントなら
        human_obstacle_names() の名前), 距離 [m])`` のリスト。種別は
        ``'self'`` (自己干渉) / ``'human'`` (人体との干渉)。距離が負なほど
        深く貫通している。貫通していない (``dist >= -tolerance``) 組み合わせ
        は含めない。貫通が深い順に並べる。
    """
    if not pairs:
        return []
    obstacle_names = spik.human_obstacle_names()
    world_vertices_by_link = {}

    def _world_vertices(link):
        if link not in world_vertices_by_link:
            local = np.asarray(link.collision_mesh.vertices, dtype=np.float64)
            world_vertices_by_link[link] = (
                local @ link.worldrot().T + link.worldpos())
        return world_vertices_by_link[link]

    colliding = []
    for link_a, other in pairs:
        verts_a = _world_vertices(link_a)
        if isinstance(other, int):
            if not obstacle_links:
                continue
            obstacle = obstacle_links[other]
            # obstacle は常に Cylinder (ローカル Z 軸が円柱の高さ方向、
            # 原点中心) なので、verts_a をローカル座標系へ変換した上で
            # 円柱の radius/height から直接、中に入り込んだ深さ (内側なら
            # 正) を求める。
            local_pts = ((verts_a - obstacle.worldpos())
                        @ obstacle.worldrot())
            radial = np.linalg.norm(local_pts[:, :2], axis=1)
            axial = np.abs(local_pts[:, 2])
            depth = float(np.minimum(
                obstacle.radius - radial,
                obstacle.height / 2.0 - axial).max())
            dist = -depth
            kind, name_b = 'human', obstacle_names[other]
        else:
            verts_b = _world_vertices(other)
            dist = float(np.linalg.norm(
                verts_a[:, np.newaxis, :] - verts_b[np.newaxis, :, :],
                axis=-1).min())
            kind, name_b = 'self', other.name
        if dist < -tolerance:
            colliding.append((kind, link_a.name, name_b, dist))
    colliding.sort(key=lambda item: item[3])
    return colliding


def collision_pairs_text(colliding, label='干渉'):
    """``colliding_link_pairs`` の戻り値を、viser のテキストパネルに出す
    ための文字列にする (自己干渉/人体との干渉を分けて列挙する)。

    ``label`` は見出しに使う語句 (既定 ``'干渉'``)。``run_camera_pipeline_
    test.py`` のように、IK 自体は指なしで解いていて画面表示の事後検証だけ
    指先まで含めている場合など、見出しでその旨を区別したいときに使う
    (例: ``label='指先まで含めた事後検証'``)。
    """
    self_pairs = [c for c in colliding if c[0] == 'self']
    human_pairs = [c for c in colliding if c[0] == 'human']
    if not colliding:
        return '{}: なし (自己干渉・人体との干渉ともに検出されていません)'.format(
            label)
    lines = ['{}: {} 件 (自己干渉 {} 件, 人体との干渉 {} 件)'.format(
        label, len(colliding), len(self_pairs), len(human_pairs))]
    for _, name_a, name_b, dist in self_pairs:
        lines.append('- [自己干渉] `{}` - `{}` ({:.4f} m 貫通)'.format(
            name_a, name_b, -dist))
    for _, name_a, name_b, dist in human_pairs:
        lines.append('- [対人干渉] `{}` - `{}` ({:.4f} m 貫通)'.format(
            name_a, name_b, -dist))
    return '\n\n'.join(lines)


def apply_robot_pose(robot, result, use_post_process=False):
    """``solve_palm_ik`` の戻り値 (関節角・台車位置姿勢) を ``robot`` に
    反映する.

    ``result['joint_names']``/``joint_angle_vector`` は ``solve_palm_ik.py``
    が ``use_hand=False`` (指関節なし) のロボットで解いた際の
    ``robot.joint_list`` の角度なので、指関節ありのロボット (見た目のための
    表示用モデル) とは ``joint_list`` の要素数・並びが異なりうる。そのため
    ``robot.angle_vector`` にそのまま渡さず、``joint_names`` で名前を突き
    合わせて該当する関節だけ角度を反映する (指関節は初期姿勢のまま)。台車の
    位置・向きは ``base_position``/``base_yaw`` に別で保存されているので、
    あわせて反映する (``solve_palm_ik.solve_palm_ik`` 参照)。

    Parameters
    ----------
    result : dict
        ``solve_palm_ik.py`` が保存した IK 結果 (``joint_names``/
        ``joint_angle_vector``/``base_position``/``base_yaw``/
        ``post_process`` を持つ dict)。
    use_post_process : bool, optional
        ``True`` のとき、``solve_palm_ik.solve_post_process`` が解いた
        後処理後の姿勢 (``result['post_process']`` -- 掌に押し付ける位置
        まで詰め、自分の手を見るよう首も向けた姿勢) を反映する。既定
        (``False``) は従来通り後処理前の姿勢。``post_process`` が無い
        (後処理判定に失敗した/この機能追加前の solve_palm_ik.py が書き
        出した/IK 自体が解けなかった) 結果では、``use_post_process`` が
        ``True`` でも後処理前の姿勢にフォールバックする。
    """
    source = result
    if use_post_process and result.get('post_process') is not None:
        source = result['post_process']
    robot.reset_pose()
    name_to_angle = dict(zip(
        source['joint_names'], source['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.base_link.newcoords(Coordinates(
        pos=source['base_position'],
        rot=rpy_matrix(source['base_yaw'], 0.0, 0.0)))


def apply_waypoint_pose(robot, joint_names, waypoints, index):
    """``waypoints[index]`` (台車位置姿勢・全身の関節角) を ``robot`` に
    反映する.

    ``apply_robot_pose`` と同じパターン -- ``joint_names``/
    ``waypoints[...]['joint_angle_vector']`` は ``plan_handshake_motion.py``
    が指なしロボットで計画した際の関節角なので、名前で突き合わせて該当
    する関節だけ反映する (指関節は ``reset_pose`` の初期姿勢のまま)。
    """
    wp = waypoints[index]
    robot.reset_pose()
    name_to_angle = dict(zip(joint_names, wp['joint_angle_vector']))
    for joint in robot.joint_list:
        if joint.name in name_to_angle:
            joint.joint_angle(name_to_angle[joint.name])
    robot.base_link.newcoords(Coordinates(
        pos=wp['base_position'], rot=rpy_matrix(wp['base_yaw'], 0.0, 0.0)))


def build_display_waypoints(motion, result, n_press_in=PRESS_IN_DISPLAY_WAYPOINTS):
    """``motion['waypoints']`` (``plan_handshake_motion.py`` が計画・検証
    した経路) に、``result['post_process']`` (``solve_palm_ik.py`` の後処理
    判定: 実際に掌へわずかにめり込む位置まで腕を詰め、首を人間の手へ向ける)
    までの補間フレームを表示用に追加する。

    ``plan_handshake_motion.py`` はこの区間を経路として計画・検証しない
    (接触そのものが目的の動きで、経路上の干渉検証にはなじまないため) ので、
    あくまで見た目のための表示専用フレームであり、
    ``waypoint_min_distances`` による検証の対象ではない。

    ``post_process`` が無い (後処理判定が全ての候補で失敗し、後処理前の
    まま採用された/IK 自体が解けなかった) 場合は ``motion['waypoints']``
    をそのまま返す。

    Returns
    -------
    (waypoints, n_approach)
        ``waypoints`` は表示用の waypoint リスト。``n_approach`` は
        ``motion['waypoints']`` の個数 (この添字以降が表示専用の後処理
        フレーム、``waypoint_min_distances`` による検証の対象外)。
    """
    waypoints = list(motion['waypoints'])
    n_approach = len(waypoints)
    post = result.get('post_process')
    if post is None:
        return waypoints, n_approach

    joint_names = motion['joint_names']
    last_wp = waypoints[-1]
    start_vec = np.asarray(last_wp['joint_angle_vector'], dtype=np.float64)
    post_name_to_angle = dict(zip(post['joint_names'],
                                  post['joint_angle_vector']))
    end_vec = np.array([post_name_to_angle.get(name, start_vec[i])
                        for i, name in enumerate(joint_names)])
    base_start = np.array([last_wp['base_position'][0],
                           last_wp['base_position'][1], last_wp['base_yaw']])
    base_end = np.array([post['base_position'][0], post['base_position'][1],
                         post['base_yaw']])

    for t in np.linspace(0.0, 1.0, n_press_in + 1)[1:]:
        angle_vec = start_vec + (end_vec - start_vec) * t
        base_vec = base_start + (base_end - base_start) * t
        waypoints.append(dict(
            base_position=[float(base_vec[0]), float(base_vec[1]), 0.0],
            base_yaw=float(base_vec[2]),
            joint_angle_vector=[float(v) for v in angle_vec],
        ))
    return waypoints, n_approach
