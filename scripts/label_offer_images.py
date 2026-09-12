#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``extract_skeletons_from_bag.py`` (または ``record_palm_offer_clips.
py``) が保存した骨格重畳画像を 1 枚ずつブラウザ (viser の GUI パネル) に
表示し、Right/Left/Null ボタンで「実際にはどちらの手を差し出しているか」
の人手ラベル (``human_label``) を対応する掌 JSON に書き込む。

``draw_random_human_poses.py`` と同じ ``aero_demo.viewer_nav`` の
ナビゲーション (Back/Next/判定ボタン、``human_label`` の読み書き) を使う
が、あちらが SMPL メッシュ + 3D 骨格を viser のシーンに描くのに対し、
こちらは 2D の骨格重畳 PNG (``extract_skeletons_from_bag.py`` の
``images/``、または ``record_palm_offer_clips.py`` のスナップショット)
をそのまま GUI パネルの画像として表示するだけの軽量版 (3D シーンは使わ
ない)。自動判定 (掌 JSON の ``offered_hand``) もテキストパネルに表示する
ので、人手判定との一致/不一致がその場で分かる。

Usage
-----
    python3 scripts/label_offer_images.py \
        --image-dir /tmp/offer_dataset/images \
        --palm-dir /tmp/offer_dataset/palms

ラベル付けが終わったら、そのまま ``tune_offer_selector.py`` の
``--skeleton-dir``/``--palm-dir`` に対応する ``skeletons/``/``palms/`` を
渡せる。
"""

import argparse
import glob
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_SRC_DIR = os.path.join(_THIS_DIR, '..', 'src')
if _PKG_SRC_DIR not in sys.path:
    sys.path.insert(0, _PKG_SRC_DIR)

from aero_demo import viewer_nav  # noqa: E402

from skrobot.viewers import ViserViewer  # noqa: E402

OFFERED_HAND_BUTTONS = [('Right', 'R'), ('Left', 'L'), ('Null', None)]
OFFERED_HAND_LABEL_NAMES = {'R': 'Right', 'L': 'Left', None: 'Null'}


def load_image_rgb(path):
    """PNG を RGB の ndarray として読む (``viser`` の ``add_image`` は
    RGB を期待する。骨格重畳画像は cv2 (BGR) で保存されているので変換
    する)。"""
    import cv2
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError('画像を読み込めません: {}'.format(path))
    return np.ascontiguousarray(bgr[:, :, ::-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--image-dir', type=str, required=True,
        help='骨格重畳 PNG のディレクトリ (extract_skeletons_from_bag.py '
            'の images/ 相当)。')
    parser.add_argument(
        '--palm-dir', type=str, required=True,
        help='対応する掌 JSON のディレクトリ (画像と同じファイル名 (拡張子'
            '違い) で対応させる、human_label をここに書き込む)。')
    parser.add_argument(
        '--pattern', type=str, default='*.png',
        help='画像ファイルの glob パターン (既定 *.png)。')
    parser.add_argument('--client-wait-timeout', type=float, default=30.0)
    parser.add_argument('--no-open-browser', action='store_true')
    args = parser.parse_args()

    image_paths = sorted(glob.glob(os.path.join(args.image_dir,
                                                 args.pattern)))
    if not image_paths:
        print('{} に画像が見つかりません。先に extract_skeletons_from_'
              'bag.py (または record_palm_offer_clips.py) を実行してくだ'
              'さい。'.format(args.image_dir))
        return

    viewer = ViserViewer(draw_grid=False)
    viewer.show(open_browser=not args.no_open_browser)
    viewer_nav.wait_for_client(viewer, args.client_wait_timeout)

    image_handle = None
    label_text = viewer._server.gui.add_markdown('')
    auto_text = viewer._server.gui.add_markdown('')
    progress_text = viewer._server.gui.add_markdown('')
    nav = viewer_nav.ManualNav(viewer, buttons=OFFERED_HAND_BUTTONS)

    def _palm_path(image_path):
        name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(args.palm_dir, name + '.json')

    def _auto_label_text(palm_path):
        if not os.path.exists(palm_path):
            return '**自動判定:** (掌 JSON なし)'
        import json
        with open(palm_path) as f:
            data = json.load(f)
        offered = data.get('offered_hand')
        return '**自動判定 (offered_hand):** {}'.format(
            OFFERED_HAND_LABEL_NAMES.get(offered, offered))

    visited = set()
    i = 0
    while 0 <= i < len(image_paths):
        image_path = image_paths[i]
        palm_path = _palm_path(image_path)
        image = load_image_rgb(image_path)

        if image_handle is None:
            image_handle = viewer._server.gui.add_image(
                image, label=os.path.basename(image_path))
        else:
            image_handle.image = image

        label_text.content = viewer_nav.format_label_text(
            viewer_nav.load_label(palm_path, default=viewer_nav.UNLABELED),
            title='人手ラベル', value_names=OFFERED_HAND_LABEL_NAMES)
        auto_text.content = _auto_label_text(palm_path)
        progress_text.content = '**進捗:** {} / {} ({})'.format(
            i + 1, len(image_paths), os.path.basename(image_path))

        visited.add(i)
        print('[{}/{}] displayed {}'.format(i + 1, len(image_paths),
                                             image_path))

        direction, label = nav.wait(viewer)
        if direction == 0:
            print('ブラウザクライアントが切断されました。中断します。')
            break
        if label is not viewer_nav.NOT_PRESSED:
            viewer_nav.save_label(palm_path, label)
            print('  -> {} として {} に記録しました。'.format(
                OFFERED_HAND_LABEL_NAMES[label], palm_path))
        i = max(0, i + direction)

    print('{} / {} 枚を表示しました。'.format(len(visited), len(image_paths)))
    viewer.close()


if __name__ == '__main__':
    main()
