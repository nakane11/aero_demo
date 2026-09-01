#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``generate_random_human_poses.py`` が出力した JSON (SMPL の人モデルと、
そこから作った骨格の両方を持つ) を読み込み、SMPL の人体メッシュと骨格を
scikit-robot の viser ビューアで重ねて表示する。``estimate_palm_poses.py``
が出力した対応する掌の位置姿勢 JSON (``--palm-dir``、既定では骨格と同じ
ファイル名で ``random_palm_poses/`` に入っているもの) があれば、左右の掌
の位置姿勢を Axis として併せて描画する (無ければ黙ってスキップする)。

SMPL メッシュは JSON に保存済みの ``smpl.pose``/``smpl.betas``/``smpl.
root_pos``/``smpl.gender`` から ``aero_demo.smpl_body.forward_world`` で
直接組み立てる。骨格 (``generate_random_human_poses.RandomSkeletonGen
erator`` が作ったもの、既に SMPL と同じ関節位置から作られているので手の
位置もメッシュの手と一致する) は、``human_palm_contact_behavior.py`` が
使っているのと同じ骨格描画 (``aero_demo.palm_plane_view`` の
``bone_line``/``bone_color``, 部位ごとに色分けした線) で重ねて描く。以前
あった「(SMPL に依存しない) 骨格から SMPL の姿勢を推定し直す」処理
(``aero_demo.smpl_body.retarget_and_pose``) はもう要らないので、この
ファイルはそれを一切使わない。

SMPL のモデルファイル自体はライセンス上リポジトリに同梱されていないので、
呼び出し側がローカルパスを渡す (既定値は smpl_body.py と同じ
``~/SMPL_python_v.1.0.0/smpl/models/`` 以下)。

Usage
-----
    rosrun aero_demo generate_random_human_poses.py \
        --num-samples 100 --output-dir /tmp/random_human_poses
    rosrun aero_demo estimate_palm_poses.py \
        --input-dir /tmp/random_human_poses \
        --output-dir /tmp/random_palm_poses
    rosrun aero_demo draw_random_human_poses.py \
        --input-dir /tmp/random_human_poses \
        --palm-dir /tmp/random_palm_poses

viser はブラウザで表示するビューアなので、実行するとブラウザが開く
(WSLg 環境などでは自動で開く)。ブラウザが自動で開かない場合は、標準出力
に表示される URL を手動で開くこと。画像は viser に接続したブラウザ
クライアントの画面をそのまま ``get_render`` で読み出して保存するので、
最初の 1 クライアントが接続するまで待つ (``--output-dir`` を指定した
ときだけ保存する)。
"""

import argparse
import glob
import json
import math
import os
import sys
import threading
import time

import numpy as np
import trimesh

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from aero_demo import palm_plane  # noqa: E402  (パス追加後に import)
from aero_demo import palm_plane_view  # noqa: E402
from aero_demo import smpl_body  # noqa: E402
from aero_demo.people_pose_types import Bone  # noqa: E402

from generate_random_human_poses import load_smpl_models  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.model import Link  # noqa: E402
from skrobot.model import Sphere  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

# 掌の平面フィットに使ったランドマーク (手首 + 知節/MCP) を示す点の色。
# palm_plane_view.py の rhand/lhand の配色 (COLOR_BONES) に合わせ、左右を
# 見分けられるようにする。
HAND_POINT_RADIUS = 0.006
HAND_POINT_COLOR = {'R': [255, 60, 60, 255], 'L': [60, 120, 255, 255]}

# 骨格の関節同士のつながり (関節名のペア)。people_pose_estimator.
# PeoplePoseEstimator の limb_sequence/index2limbname と同じ骨格の
# つながりを、名前のペアとして書き下したもの (MediaPipe/mediapipe 無しで
# 使えるよう、この描画専用ファイルに複製してある -- fake_people_pose_
# estimator_ros.py の hand_sequence も同様に複製する方針, module
# docstring 参照)。
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
# 手のランドマーク (MediaPipe の並び) 同士のつながり。people_pose_
# estimator.PeoplePoseEstimator.hand_sequence と同じ。
HAND_SEQUENCE = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
BONE_NAME_PAIRS = BODY_BONE_PAIRS + [
    ('{}Hand{}'.format(side, a), '{}Hand{}'.format(side, b))
    for side in ('R', 'L') for a, b in HAND_SEQUENCE]

# ワールド座標系 (x=前, y=左, z=上) の -x 方向を向くカメラの姿勢
# (骨格生成側で人物は常に原点・+x 方向を向いて配置されるので、これで
# 人物を正面から見ることになる)。わずかに見下ろすよう、少し高い位置から
# 下向きに傾ける。
_CAMERA_DISTANCE = 2.5
_CAMERA_HEIGHT = 1.6
_CAMERA_TILT_DOWN_DEG = 15.0


def _front_view_camera_transform():
    """人物を原点から +x 方向 (正面) に、わずかに見下ろして見るカメラの
    世界姿勢 (4x4) を作る."""
    tilt = math.radians(_CAMERA_TILT_DOWN_DEG)
    forward = np.array([-math.cos(tilt), 0.0, -math.sin(tilt)])
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = -forward
    transform[:3, 3] = [_CAMERA_DISTANCE, 0.0, _CAMERA_HEIGHT]
    return transform


def load_person_json(path):
    """``generate_random_human_poses.build_person_json`` が保存した 1 人分
    の JSON を読む.

    Returns
    -------
    dict
        ``joint_positions`` (関節名 -> ``np.ndarray([x, y, z])``, ロボット
        座標系), ``height`` (身長 [m]), ``gender`` (str), ``betas``
        ((10,) ndarray), ``pose`` ((24, 3) ndarray), ``root_pos``
        ((3,) ndarray) を持つ dict。
    """
    with open(path) as f:
        data = json.load(f)
    skeleton = data['skeleton']
    smpl = data['smpl']
    return dict(
        joint_positions={name: np.asarray(xyz, dtype=np.float64)
                         for name, xyz in skeleton['joint_positions'].items()},
        height=float(skeleton['height']),
        gender=smpl['gender'],
        betas=np.asarray(smpl['betas'], dtype=np.float64),
        pose=np.asarray(smpl['pose'], dtype=np.float64),
        root_pos=np.asarray(smpl['root_pos'], dtype=np.float64))


def iter_pose_files(input_dir, pattern='*.json'):
    """``input_dir`` 内の骨格 JSON をファイル名順に列挙する."""
    return sorted(glob.glob(os.path.join(input_dir, pattern)))


def load_palm_json(path):
    """``estimate_palm_poses.py`` が保存した 1 人分の掌 JSON を読む.

    Parameters
    ----------
    path : str
        ``estimate_palm_poses.PalmPoseEstimator.estimate`` の戻り値を
        そのまま保存した JSON のパス。ファイルが無ければ ``None``。

    Returns
    -------
    dict or None
        ``{'R': (position, rot) or None, 'L': (position, rot) or None}``。
        ``position`` は ``np.ndarray(3,)``、``rot`` は ``np.ndarray(3,3)``
        (skrobot の ``Coordinates(rot=...)`` にそのまま渡せる)。
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    palms = {}
    for side in ('R', 'L'):
        palm = data.get(side)
        if palm is None:
            palms[side] = None
            continue
        palms[side] = (np.asarray(palm['position'], dtype=np.float64),
                       np.asarray(palm['rot'], dtype=np.float64))
    return palms


SKIN_ALPHA = 150  # SMPL メッシュを半透明にするための alpha (0-255)。


def random_skin_color(rng):
    """人物ごとに見た目を変えるためのランダムな肌色 (RGBA, 0-255).

    SMPL メッシュ自体を半透明にしたいので、alpha は ``SKIN_ALPHA`` 固定に
    する (色 (RGB) だけ人物ごとにランダムにする)。
    """
    base = np.array([0.55, 0.40, 0.32])
    variation = rng.uniform(-0.18, 0.20, size=3)
    rgb = np.clip(base + variation, 0.05, 0.95)
    return [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255),
            SKIN_ALPHA]


def build_mesh(model, person, skin_color):
    """保存済みの SMPL pose/betas/root_pos から SMPL メッシュを作る.

    ``generate_random_human_poses.RandomSmplHumanGenerator`` が生成した
    ときと同じ ``aero_demo.smpl_body.forward_world`` (root の向きは常に
    単位行列, scale は SMPL 自身の betas が表す実寸のまま) を使うので、
    骨格 (``person['joint_positions']``) とメッシュは常に同じ関節位置を
    共有する。

    Parameters
    ----------
    model : smpl_body.SmplModel
        ``person['gender']`` に対応するモデル。
    person : dict
        ``load_person_json`` の戻り値。
    skin_color : list of int
        ``random_skin_color`` が返す RGBA (0-255)。呼び出し側で 1 人に
        つき 1 回だけ引いて使い回す (Back で戻ったときに毎回見た目が
        変わらないようにするため)。

    Returns
    -------
    trimesh.Trimesh
    """
    vertices, _joints = smpl_body.forward_world(
        model, person['pose'], person['betas'], person['root_pos'])
    mesh = trimesh.Trimesh(vertices=vertices, faces=model.f, process=False)
    mesh.visual.face_colors = skin_color
    if len(skin_color) >= 4 and skin_color[3] < 255:
        # face_colors だけでは viser (glTF エクスポートが既定で
        # alphaMode: OPAQUE になる) 側で不透明に見えてしまうので、
        # palm_plane_view.set_color と同様に alphaMode='BLEND' の
        # PBRMaterial を明示的に付ける。
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[c / 255.0 for c in skin_color],
                alphaMode='BLEND'))
    return mesh


def build_skeleton_links(joint_positions):
    """骨格を部位ごとに色分けした線 (``skrobot.model.primitives.
    LineString``) のリストにする.

    ``human_palm_contact_behavior.py`` (``aero_demo.palm_plane_view.
    PalmPlaneScene``) が SMPL メッシュに重ねて骨格を描くのと同じ
    ``palm_plane_view.bone_line``/``bone_color`` を使うので、見た目
    (部位ごとの色, ``palm_plane_view.COLOR_BONES``) も同じになる。
    """
    links = []
    for start_name, end_name in BONE_NAME_PAIRS:
        if start_name not in joint_positions or end_name not in joint_positions:
            continue
        bone = Bone(name='{}->{}'.format(start_name, end_name),
                   start_point=joint_positions[start_name],
                   end_point=joint_positions[end_name])
        links.append(palm_plane_view.bone_line(
            bone, palm_plane_view.bone_color(bone.name)))
    return links


def wait_for_client(viewer, timeout):
    """viser に最低 1 つブラウザクライアントが接続するまで待つ."""
    print('viser のブラウザ画面が接続するまで待っています '
          '(タイムアウト {:.0f} 秒ごとに再度待機します)...'.format(timeout))
    while True:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if viewer._server.get_clients():
                print('クライアントが接続しました。')
                return
            time.sleep(0.2)
        print('ブラウザクライアントがまだ接続していません。上に表示された '
              'URL を手動で開いてください (Ctrl-C で中断できます)。')


class ManualNav(object):
    """viser の GUI に Back/Next ボタンを追加し、押された向きを ``wait()``
    で受け取れるようにする (``--advance-mode manual`` 用)。

    Back/Next のどちらを押しても同じ ``threading.Event`` を立てるだけの
    単純な仕組みで、直近に押されたボタンの向きだけを覚える (連打しても
    最後の 1 回分しか進まない/戻らない)。
    """

    def __init__(self, viewer):
        self._event = threading.Event()
        self._direction = 0
        back = viewer._server.gui.add_button('Back')
        next_ = viewer._server.gui.add_button('Next')

        @back.on_click
        def _on_back(_):  # noqa: ANN001  (viser の GuiEvent は型を問わない)
            self._direction = -1
            self._event.set()

        @next_.on_click
        def _on_next(_):  # noqa: ANN001
            self._direction = 1
            self._event.set()

    def wait(self, viewer):
        """Back/Next が押されるまで待ち、向き (``-1``/``+1``) を返す.

        ブラウザクライアントが切断されたら ``0`` を返す。
        """
        self._event.clear()
        while not self._event.is_set():
            if not viewer._server.get_clients():
                return 0
            time.sleep(0.05)
        return self._direction


def wait_for_advance(viewer, nav, pause):
    """次に表示する人物への向きを決める.

    ``nav`` (``ManualNav``) が渡されていれば (``--advance-mode
    manual``)、Back/Next ボタンが押されるまで待って ``-1``/``+1`` を
    返す。渡されていなければ (``--advance-mode auto``) ``pause`` 秒だけ
    待って常に ``+1`` を返す (これまでと同じ動作)。ブラウザクライアント
    が切断されていれば ``None`` を返す。
    """
    if nav is None:
        time.sleep(pause)
        if not viewer._server.get_clients():
            return None
        return 1
    direction = nav.wait(viewer)
    return direction if direction != 0 else None


def main():
    parser = argparse.ArgumentParser(
        description='generate_random_human_poses.py が出力した SMPL の '
                    '人モデル・骨格 JSON を読み込み、viser で表示する。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_human_poses'),
        help='人物 JSON の入力ディレクトリ。')
    parser.add_argument(
        '--palm-dir', type=str,
        default=os.path.join(_THIS_DIR, 'random_palm_poses'),
        help='estimate_palm_poses.py が出力した掌の位置姿勢 JSON の入力 '
             'ディレクトリ (骨格 JSON と同じファイル名で対応させる)。存在 '
             'しないファイルは黙ってスキップする (掌の描画なし)。')
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
        '--output-dir', type=str, default=None,
        help='指定すると、表示した各姿勢の画像もこのディレクトリに保存する。')
    parser.add_argument('--image-width', type=int, default=800)
    parser.add_argument('--image-height', type=int, default=600)
    parser.add_argument('--seed', type=int, default=None,
                        help='肌色などの見た目に使う乱数シード。')
    parser.add_argument('--client-wait-timeout', type=float, default=30.0,
                        help='ブラウザクライアント接続を待つ 1 回あたりの'
                             '秒数 (繰り返し待つ)。')
    parser.add_argument('--no-open-browser', action='store_true',
                        help='ブラウザの自動起動を無効にする '
                             '(URL を自分で開く場合)。')
    parser.add_argument('--pause', type=float, default=0.15,
                        help='姿勢を切り替えてから撮影するまでの待機秒数 '
                             '(ブラウザ側の描画が反映されるのを待つ、'
                             '--advance-mode auto のときだけ使う)。')
    parser.add_argument(
        '--advance-mode', choices=['auto', 'manual'], default='manual',
        help='manual (既定): viser 画面の Back/Next ボタンを押すまで '
             '切り替えずに待つ。auto: --pause 秒ごとに自動で次の人物に '
             '切り替える (Back はできない)。')
    args = parser.parse_args()

    pose_files = iter_pose_files(args.input_dir)
    if not pose_files:
        print('{} に人物 JSON が見つかりません。先に '
              'generate_random_human_poses.py を実行してください。'.format(
                  args.input_dir))
        return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    models_by_gender = dict(
        load_smpl_models(args.model_path, args.female_model_path))
    # 見た目 (肌色) だけあらかじめ 1 回だけ引いておく。Back で同じ人物に
    # 戻ったときに毎回見た目が変わらないようにするため (build_mesh をその
    # 都度呼んでも、この選択済みの値を使い回す)。SMPL のモデル (男性/女性)
    # は人物ごとの JSON に保存済みの ``gender`` で決まるので、ここでは
    # 引かない。
    skin_colors = [random_skin_color(rng) for _ in pose_files]

    viewer = ViserViewer(draw_grid=True)
    viewer.add(Axis(axis_length=0.1, axis_radius=0.004))
    # 左右の掌の Axis はあらかじめ 1 組だけ作っておき、掌 JSON が見つかった
    # 人物のときだけ座標を更新して viewer に足す/居なければ外す (毎回作り
    # 直さない)。
    palm_axes = {'R': Axis(axis_length=0.05, axis_radius=0.003),
                'L': Axis(axis_length=0.05, axis_radius=0.003)}
    palm_axes_added = {'R': False, 'L': False}
    # 推定に使った手のランドマーク (wrist + MCP) を色付きの球で表示する。
    # 骨格 JSON に含まれる関節そのものなので、掌 JSON の有無によらず
    # (推定が None になった側でも) 見えている点はそのまま描く。
    hand_point_spheres = {
        (side, i): Sphere(radius=HAND_POINT_RADIUS,
                          color=HAND_POINT_COLOR[side])
        for side in ('R', 'L') for i in palm_plane.PLANE_LANDMARKS}
    hand_point_added = {key: False for key in hand_point_spheres}
    viewer.show(open_browser=not args.no_open_browser)
    wait_for_client(viewer, args.client_wait_timeout)
    viewer.set_camera(coords_or_transform=_front_view_camera_transform())

    nav = None
    if args.advance_mode == 'manual':
        nav = ManualNav(viewer)

    image_module = None
    if args.output_dir:
        from PIL import Image
        image_module = Image

    current_link = None
    current_skeleton_links = []
    visited = set()
    i = 0
    while 0 <= i < len(pose_files):
        path = pose_files[i]
        person = load_person_json(path)
        joints = person['joint_positions']
        model = models_by_gender.get(person['gender'],
                                     models_by_gender['male'])
        mesh = build_mesh(model, person, skin_colors[i])

        link = Link(visual_mesh=mesh, name='human_{:03d}'.format(i))
        if current_link is not None:
            viewer.delete(current_link)
        viewer.add(link)
        current_link = link

        for old_link in current_skeleton_links:
            viewer.delete(old_link)
        current_skeleton_links = build_skeleton_links(joints)
        for skeleton_link in current_skeleton_links:
            viewer.add(skeleton_link)

        palm_path = os.path.join(args.palm_dir, os.path.basename(path))
        palms = load_palm_json(palm_path)
        for side, axis in palm_axes.items():
            palm = palms.get(side) if palms else None
            if palm is None:
                if palm_axes_added[side]:
                    viewer.delete(axis)
                    palm_axes_added[side] = False
                continue
            position, rot = palm
            axis.newcoords(Coordinates(pos=position, rot=rot))
            if not palm_axes_added[side]:
                viewer.add(axis)
                palm_axes_added[side] = True

        for (side, idx), sphere in hand_point_spheres.items():
            key = '{}Hand{}'.format(side, idx)
            present = key in joints
            if present:
                sphere.newcoords(Coordinates(pos=joints[key]))
            if present and not hand_point_added[(side, idx)]:
                viewer.add(sphere)
                hand_point_added[(side, idx)] = True
            elif not present and hand_point_added[(side, idx)]:
                viewer.delete(sphere)
                hand_point_added[(side, idx)] = False

        viewer.redraw()

        visited.add(i)
        print('[{}/{}] displayed {}'.format(i + 1, len(pose_files), path))

        clients = list(viewer._server.get_clients().values())
        if clients and image_module is not None:
            image = clients[0].camera.get_render(
                args.image_height, args.image_width, transport_format='jpeg')
            out_path = os.path.join(
                args.output_dir, 'human_{:03d}.jpg'.format(i))
            image_module.fromarray(image).save(out_path)

        direction = wait_for_advance(viewer, nav, args.pause)
        if direction is None:
            print('ブラウザクライアントが切断されました。中断します。')
            break
        # 先頭で Back を押しても終了しない (0 未満にはしない) よう下限を
        # クランプする。末尾で Next を押した場合は (auto で最後まで
        # 表示し終わったときと同じく) そのままループを抜けて終了する。
        i = max(0, i + direction)

    print('{} / {} 体を表示しました。'.format(len(visited), len(pose_files)))
    viewer.close()


if __name__ == '__main__':
    main()
