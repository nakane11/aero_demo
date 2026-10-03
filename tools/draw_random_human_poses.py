#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``generate_random_human_poses.py`` の人物 JSON (SMPL メッシュ + 骨格) と
``estimate_palm_poses.py`` の掌 JSON を viser で重ねて表示する。

``offered_hand`` で選ばれた手を赤、他方を白で描く (掌 JSON が無ければ左右で
色分け)。manual モードでは Right/Left/Null ボタンで人手判定 ``human_label``
を掌 JSON に書き込む。SMPL モデルは同梱されないので ``--model-path`` 等で渡す。

Usage
-----
    python3 tools/draw_random_human_poses.py \\
        --input-dir /tmp/random_human_poses \\
        --palm-dir /tmp/random_palm_poses
"""

import argparse
import json
import os
import sys

import numpy as np
import trimesh

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.join(_THIS_DIR, '..', 'scripts')
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from aero_demo import json_io  # noqa: E402  (パス追加後に import)
from aero_demo import palm_plane  # noqa: E402
from aero_demo import palm_plane_view  # noqa: E402
from aero_demo import skeleton_drawing  # noqa: E402
from aero_demo import smpl_body  # noqa: E402
from aero_demo import viewer_nav  # noqa: E402

from generate_random_human_poses import load_smpl_models  # noqa: E402

from skrobot.coordinates import Coordinates  # noqa: E402
from skrobot.model import Axis  # noqa: E402
from skrobot.model import Link  # noqa: E402
from skrobot.model import Sphere  # noqa: E402
from skrobot.viewers import ViserViewer  # noqa: E402

# 掌フィットに使ったランドマーク (手首 + MCP) の点。掌 JSON が無いときの左右の色。
HAND_POINT_RADIUS = 0.006
HAND_POINT_COLOR = {'R': [255, 60, 60, 255], 'L': [60, 120, 255, 255]}

# offered_hand に選ばれた手 / 選ばれなかった手の色。
COLOR_OFFERED_HAND = palm_plane_view.COLOR_BONES['rhand']
COLOR_NOT_OFFERED_HAND = [255, 255, 255, 255]

# 人手判定ボタン。値は offered_hand と同じ ('R'/'L'/None)。
OFFERED_HAND_BUTTONS = [('Right', 'R'), ('Left', 'L'), ('Null', None)]
OFFERED_HAND_LABEL_NAMES = {'R': 'Right', 'L': 'Left', None: 'Null'}

def load_person_json(path):
    """1 人分の人物 JSON を読む (``joint_positions`` はロボット座標系、身長 [m])."""
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


iter_pose_files = json_io.iter_json_files


def load_palm_json(path):
    """1 人分の掌 JSON を読む (無ければ ``None``).

    Returns ``{'R': (position, rot) or None, 'L': ..., 'offered_hand': ...}``。
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    palms = {'offered_hand': data.get('offered_hand')}
    for side in ('R', 'L'):
        palm = data.get(side)
        if palm is None:
            palms[side] = None
            continue
        palms[side] = (np.asarray(palm['position'], dtype=np.float64),
                       np.asarray(palm['rot'], dtype=np.float64))
    return palms


def offered_hand_colors(palms):
    """``offered_hand`` から ``{'R': rgba, 'L': rgba}`` を返す (``palms`` が None なら None)."""
    if palms is None:
        return None
    offered = palms.get('offered_hand')
    return {side: (COLOR_OFFERED_HAND if side == offered
                   else COLOR_NOT_OFFERED_HAND)
            for side in ('R', 'L')}


SKIN_ALPHA = 150  # SMPL メッシュを半透明にするための alpha (0-255)。


def random_skin_color(rng):
    """ランダムな肌色 (RGBA, 0-255、alpha は ``SKIN_ALPHA`` 固定)."""
    base = np.array([0.55, 0.40, 0.32])
    variation = rng.uniform(-0.18, 0.20, size=3)
    rgb = np.clip(base + variation, 0.05, 0.95)
    return [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255),
            SKIN_ALPHA]


def build_mesh(model, person, skin_color):
    """保存済みの SMPL pose/betas/root_pos から ``trimesh.Trimesh`` を作る."""
    vertices, _joints = smpl_body.forward_world(
        model, person['pose'], person['betas'], person['root_pos'])
    mesh = trimesh.Trimesh(vertices=vertices, faces=model.f, process=False)
    mesh.visual.face_colors = skin_color
    if len(skin_color) >= 4 and skin_color[3] < 255:
        # face_colors だけでは viser で不透明になるので BLEND を明示する。
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[c / 255.0 for c in skin_color],
                alphaMode='BLEND'))
    return mesh


def main():
    parser = argparse.ArgumentParser(
        description='generate_random_human_poses.py が出力した SMPL の '
                    '人モデル・骨格 JSON を読み込み、viser で表示する。')
    parser.add_argument(
        '--input-dir', type=str,
        default=os.path.join(_SCRIPTS_DIR, 'random_human_poses'),
        help='人物 JSON の入力ディレクトリ。')
    parser.add_argument(
        '--palm-dir', type=str,
        default=os.path.join(_SCRIPTS_DIR, 'random_palm_poses'),
        help='掌 JSON のディレクトリ (骨格と同じファイル名、無ければ省略)。')
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
                        help='クライアント接続を待つ 1 回あたりの秒数。')
    parser.add_argument('--no-open-browser', action='store_true',
                        help='ブラウザを自動で開かない。')
    parser.add_argument('--pause', type=float, default=0.15,
                        help='auto モードで次の人物へ進むまでの秒数。')
    parser.add_argument(
        '--advance-mode', choices=['auto', 'manual'], default='manual',
        help='manual: Back/Next ボタンで送る。auto: --pause 秒ごとに送る。')
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
    # Back で戻っても見た目が変わらないよう肌色は先に引いておく。
    skin_colors = [random_skin_color(rng) for _ in pose_files]

    viewer = ViserViewer(draw_grid=True)
    viewer.add(Axis(axis_length=0.1, axis_radius=0.004))
    palm_axes = {'R': Axis(axis_length=0.05, axis_radius=0.003),
                'L': Axis(axis_length=0.05, axis_radius=0.003)}
    palm_axes_added = {'R': False, 'L': False}
    # 手のランドマーク (wrist + MCP) の球。表示中の色を覚えて変化時だけ塗り直す。
    hand_point_spheres = {
        (side, i): Sphere(radius=HAND_POINT_RADIUS,
                          color=HAND_POINT_COLOR[side])
        for side in ('R', 'L') for i in palm_plane.PLANE_LANDMARKS}
    hand_point_added = {key: False for key in hand_point_spheres}
    hand_point_color_shown = {(side, i): HAND_POINT_COLOR[side]
                              for side, i in hand_point_spheres}
    viewer.show(open_browser=not args.no_open_browser)
    viewer_nav.wait_for_client(viewer, args.client_wait_timeout)
    viewer_nav.set_front_view(viewer)

    nav = None
    if args.advance_mode == 'manual':
        nav = viewer_nav.ManualNav(viewer, buttons=OFFERED_HAND_BUTTONS)
    # 表示中の人物の human_label。
    label_text = viewer._server.gui.add_markdown('')

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

        palm_path = os.path.join(args.palm_dir, os.path.basename(path))
        palms = load_palm_json(palm_path)
        hand_colors = offered_hand_colors(palms)

        for old_link in current_skeleton_links:
            viewer.delete(old_link)
        current_skeleton_links = skeleton_drawing.build_skeleton_links(
            joints, hand_colors)
        for skeleton_link in current_skeleton_links:
            viewer.add(skeleton_link)

        label_text.content = viewer_nav.format_label_text(
            viewer_nav.load_label(palm_path, default=viewer_nav.UNLABELED),
            title='掌ラベル', value_names=OFFERED_HAND_LABEL_NAMES)
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
            color = (HAND_POINT_COLOR[side] if hand_colors is None
                     else hand_colors[side])
            if present:
                sphere.newcoords(Coordinates(pos=joints[key]))
            # 色は add 時にしか反映されないので、変わるときは外して足し直す。
            if hand_point_added[(side, idx)] and (
                    not present
                    or hand_point_color_shown[(side, idx)] != color):
                viewer.delete(sphere)
                hand_point_added[(side, idx)] = False
            if present and not hand_point_added[(side, idx)]:
                palm_plane_view.set_color(sphere, color)
                viewer.add(sphere)
                hand_point_added[(side, idx)] = True
                hand_point_color_shown[(side, idx)] = color

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

        direction, label = viewer_nav.wait_for_advance(viewer, nav, args.pause)
        if direction is None:
            print('ブラウザクライアントが切断されました。中断します。')
            break
        if label is not viewer_nav.NOT_PRESSED:
            viewer_nav.save_label(palm_path, label)
            print('  -> {} として {} に記録しました。'.format(
                OFFERED_HAND_LABEL_NAMES[label], palm_path))
        # 先頭での Back は終了しない。末尾での Next は終了。
        i = max(0, i + direction)

    print('{} / {} 体を表示しました。'.format(len(visited), len(pose_files)))
    viewer.close()


if __name__ == '__main__':
    main()
