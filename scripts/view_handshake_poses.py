#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``solve_palm_ik.py`` が出力したロボットの位置姿勢・関節角度 JSON
(``random_handshake_poses/human_xxx.json``) と、対応する
SMPL モデル (``random_human_poses/human_xxx.json`` の
``smpl.pose``/``betas``/``root_pos``/``gender``) をファイル名で突き合わせ
て読み込み、scikit-robot の viser ビューアで並べて表示する。

``draw_random_human_poses.py`` と違い、ビューアには SMPL メッシュとロボット
モデルの 2 つだけを表示する (骨格線・掌 Axis・ランドマーク球・ワールド
Axis・カメラは描かない)。viser 画面には Back/Next ボタンに加え Good/Bad
ボタンも表示され、押すと表示中の IK 結果が正しいかどうかの判定結果
(``human_label``: ``true``: Good/``false``: Bad) が対応する handshakes
ディレクトリの JSON に書き込まれる (Next と同様に次の人物へ進む)。書き
込まれたラベルは viser 画面のテキストパネルにも表示される。共通の
Back/Next/Good/Bad ボタンや判定結果の読み書きは ``draw_random_human_
poses.py`` と共有の ``aero_demo.viewer_nav`` を使う。

``solve_palm_ik.py`` が IK の対象外にした人物 (掌推定の ``offered_hand``
が ``null``、つまりどちらの手も差し出していないと判定された人物。JSON の
``target`` が ``false``) は IK の結果が無いので、ビューアには表示せず
読み飛ばす。

IK が失敗した人物 (``solved`` が ``false``) では、どちらの手に合わせよう
としていたのかを確認できるように、人間の差し出した手 (``offered_hand``)
とロボットが使った腕 (``robot_arm``) をテキストパネルと標準出力に出す。

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

from generate_random_human_poses import load_smpl_models  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.coordinates.math import rpy_matrix  # noqa: E402
from skrobot.model import Link  # noqa: E402
from skrobot.models import Aero  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

# SMPL メッシュの肌色 (RGBA, 0-255)。draw_random_human_poses.py と違い
# 人物ごとにランダムにはしない (このビューアは IK 結果の確認が目的で、
# 見た目のバリエーションは不要なため)。
SKIN_COLOR = [180, 130, 110, 255]


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


def build_smpl_mesh(model, person):
    """保存済みの SMPL pose/betas/root_pos からメッシュを作る
    (``smpl_body.forward_world`` を使うのは draw_random_human_poses.py と
    同じ)。"""
    vertices, _joints = smpl_body.forward_world(
        model, person['pose'], person['betas'], person['root_pos'])
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
    viewer.add(robot)
    viewer.show(open_browser=not args.no_open_browser)
    viewer_nav.wait_for_client(viewer, args.client_wait_timeout)

    current_mesh_link = None
    i = 0
    while 0 <= i < len(names):
        name = names[i]
        skeleton_path = os.path.join(args.skeleton_dir, name)
        handshake_path = os.path.join(args.handshake_dir, name)
        person = load_skeleton_json(skeleton_path)
        handshake = load_handshake_json(handshake_path)

        model = models_by_gender.get(
            person['gender'], models_by_gender['male'])
        mesh = build_smpl_mesh(model, person)
        link = Link(visual_mesh=mesh, name='smpl_human')
        if current_mesh_link is not None:
            viewer.delete(current_mesh_link)
        viewer.add(link)
        current_mesh_link = link

        apply_robot_pose(robot, handshake)

        # IK が失敗したときは、どちらの手に合わせようとしていたのかが
        # 分かるように人間の手とロボットの腕もあわせて出す。
        if handshake.get('solved'):
            status = 'solved'
        else:
            status = 'NOT solved ({})'.format(hand_text(handshake))
        label_text.content = '**{}** ({}/{})  IK: {}\n\n{}'.format(
            name, i + 1, len(names), status,
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
