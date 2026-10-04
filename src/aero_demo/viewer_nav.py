#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""viser ビューアで人物を 1 つずつ切り替えて表示するための GUI ナビゲーション
(Back/Next と判定ボタン) と、判定結果の JSON 読み書き.
"""

import json
import math
import os
import threading
import time

import numpy as np


# 原点で +x を向く人物を正面からやや見下ろすカメラ
CAMERA_DISTANCE = 2.5
CAMERA_HEIGHT = 1.6
CAMERA_TILT_DOWN_DEG = 15.0


# 判定ボタンが押されていない (Back/Next)。判定値に None があり得るため別に用意。
NOT_PRESSED = object()

# ``load_label`` の default 用: 未判定 (None 判定と区別するため)。
UNLABELED = object()


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


def front_view_camera_transform():
    """人物を正面からやや見下ろすカメラの世界姿勢 (4x4, 列が右/上/後ろ)."""
    tilt = math.radians(CAMERA_TILT_DOWN_DEG)
    forward = np.array([-math.cos(tilt), 0.0, -math.sin(tilt)])
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, 0] = right
    transform[:3, 1] = up
    transform[:3, 2] = -forward
    transform[:3, 3] = [CAMERA_DISTANCE, 0.0, CAMERA_HEIGHT]
    return transform


def set_front_view(viewer):
    """``viewer`` のカメラを人物の正面に合わせる."""
    viewer.set_camera(coords_or_transform=front_view_camera_transform())


class ManualNav(object):
    """viser の GUI に Back/Next と判定ボタンを追加し、``wait()`` で結果を返す.

    判定ボタンを押すと Next と同様に進み、その値も返す。
    """

    def __init__(self, viewer, buttons=None):
        """``buttons``: 判定ボタンの (表示テキスト, 値) のリスト (既定 Good/Bad)."""
        if buttons is None:
            buttons = [('Good', True), ('Bad', False)]
        self._event = threading.Event()
        self._direction = 0
        self._label = NOT_PRESSED
        back = viewer._server.gui.add_button('Back')
        next_ = viewer._server.gui.add_button('Next')

        @back.on_click
        def _on_back(_):  # noqa: ANN001  (viser の GuiEvent は型を問わない)
            self._direction = -1
            self._label = NOT_PRESSED
            self._event.set()

        @next_.on_click
        def _on_next(_):  # noqa: ANN001
            self._direction = 1
            self._label = NOT_PRESSED
            self._event.set()

        for text, value in buttons:
            button = viewer._server.gui.add_button(text)

            @button.on_click
            def _on_judge(_, value=value):  # noqa: ANN001
                self._direction = 1
                self._label = value
                self._event.set()

    def wait(self, viewer):
        """ボタンが押されるまで待ち ``(direction (-1/+1), label)`` を返す.

        切断されたら ``(0, NOT_PRESSED)``。
        """
        self._event.clear()
        while not self._event.is_set():
            if not viewer._server.get_clients():
                return 0, NOT_PRESSED
            time.sleep(0.05)
        return self._direction, self._label


def wait_for_advance(viewer, nav, pause):
    """``(direction, label)`` を返す. nav が None なら pause 秒待って自動送り.

    切断されていれば ``(None, NOT_PRESSED)``。
    """
    if nav is None:
        time.sleep(pause)
        if not viewer._server.get_clients():
            return None, NOT_PRESSED
        return 1, NOT_PRESSED
    direction, label = nav.wait(viewer)
    if direction == 0:
        return None, NOT_PRESSED
    return direction, label


def save_label(json_path, label):
    """判定結果を JSON の ``human_label`` に書き込む (既存の内容は保つ)."""
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}
    data['human_label'] = label
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_label(json_path, default=None):
    """``save_label`` の結果を読む (無ければ ``default``).

    判定値に None があり得るなら ``default=UNLABELED`` を渡すこと。
    """
    if not os.path.exists(json_path):
        return default
    with open(json_path) as f:
        data = json.load(f)
    return data.get('human_label', default)


def format_label_text(label, title='ラベル', value_names=None):
    """判定結果を GUI 表示用の文字列にする (value_names に無ければ未判定)."""
    if value_names is None:
        value_names = {True: 'Good', False: 'Bad'}
    if label in value_names:
        return '**{}:** {}'.format(title, value_names[label])
    return '**{}:** (未判定)'.format(title)
