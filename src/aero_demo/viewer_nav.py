#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``draw_random_human_poses.py``/``view_handshake_poses.py`` など、viser
ビューアで人物 (姿勢) を 1 つずつ切り替えて表示するスクリプトが共通で使う
GUI ナビゲーション (Back/Next/Good/Bad ボタン) と、Good/Bad の判定結果を
JSON に読み書きする処理をまとめたもの。
"""

import json
import os
import threading
import time


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
    """viser の GUI に Back/Next/Good/Bad ボタンを追加し、押された結果を
    ``wait()`` で受け取れるようにする。

    Back/Next はどちらも同じ ``threading.Event`` を立てて向きだけを伝える
    (直近に押されたボタンの向きだけを覚える。連打しても最後の 1 回分しか
    進まない/戻らない)。Good/Bad は表示中の姿勢が正しいかどうかの人手
    judgment を表し、押すと (Next と同様に) 次の人物へ進みつつ、判定結果
    (``True``: Good, ``False``: Bad) も一緒に伝える。
    """

    def __init__(self, viewer):
        self._event = threading.Event()
        self._direction = 0
        self._label = None
        back = viewer._server.gui.add_button('Back')
        next_ = viewer._server.gui.add_button('Next')
        good = viewer._server.gui.add_button('Good')
        bad = viewer._server.gui.add_button('Bad')

        @back.on_click
        def _on_back(_):  # noqa: ANN001  (viser の GuiEvent は型を問わない)
            self._direction = -1
            self._label = None
            self._event.set()

        @next_.on_click
        def _on_next(_):  # noqa: ANN001
            self._direction = 1
            self._label = None
            self._event.set()

        @good.on_click
        def _on_good(_):  # noqa: ANN001
            self._direction = 1
            self._label = True
            self._event.set()

        @bad.on_click
        def _on_bad(_):  # noqa: ANN001
            self._direction = 1
            self._label = False
            self._event.set()

    def wait(self, viewer):
        """Back/Next/Good/Bad のいずれかが押されるまで待つ.

        ``(direction, label)`` を返す。``direction`` は ``-1``/``+1``、
        ``label`` は Good/Bad が押されたときだけ ``True``/``False`` (Back/
        Next のときは ``None``)。ブラウザクライアントが切断されたら
        ``(0, None)`` を返す。
        """
        self._event.clear()
        while not self._event.is_set():
            if not viewer._server.get_clients():
                return 0, None
            time.sleep(0.05)
        return self._direction, self._label


def wait_for_advance(viewer, nav, pause):
    """次に表示する人物への向きと、Good/Bad の判定結果を決める.

    ``nav`` (``ManualNav``) が渡されていれば、Back/Next/Good/Bad ボタンが
    押されるまで待って ``(direction, label)`` を返す (``label`` は Good/Bad
    のときだけ ``True``/``False``)。渡されていなければ ``pause`` 秒だけ
    待って常に ``(1, None)`` を返す (自動送り)。ブラウザクライアントが
    切断されていれば ``(None, None)`` を返す。
    """
    if nav is None:
        time.sleep(pause)
        if not viewer._server.get_clients():
            return None, None
        return 1, None
    direction, label = nav.wait(viewer)
    if direction == 0:
        return None, None
    return direction, label


def save_label(json_path, label, key='human_label'):
    """Good/Bad ボタンで判定した結果を JSON に書き込む.

    既存の JSON があればその内容を保ったまま ``key`` (既定
    ``human_label``, ``True``: Good/``False``: Bad) を追記する。JSON が
    無ければ、判定結果だけを持つ JSON を新規に作る。
    """
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}
    data[key] = label
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_label(json_path, key='human_label'):
    """``save_label`` が書き込んだ判定結果を読む.

    JSON が無い、もしくは ``key`` が無ければ (まだ Good/Bad が押されて
    いなければ) ``None`` を返す。
    """
    if not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        data = json.load(f)
    return data.get(key)


def format_label_text(label, title='ラベル'):
    """``load_label``/Good・Bad ボタンの結果を GUI 表示用の文字列にする."""
    if label is True:
        return '**{}:** Good'.format(title)
    if label is False:
        return '**{}:** Bad'.format(title)
    return '**{}:** (未判定)'.format(title)
