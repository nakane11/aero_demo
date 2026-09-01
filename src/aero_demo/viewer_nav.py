#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""``draw_random_human_poses.py``/``view_handshake_poses.py`` など、viser
ビューアで人物 (姿勢) を 1 つずつ切り替えて表示するスクリプトが共通で使う
GUI ナビゲーション (Back/Next ボタンと、用途ごとに異なる判定ボタン) と、
判定結果を JSON に読み書きする処理をまとめたもの。判定ボタンは
``ManualNav`` の ``buttons`` で差し替えられる (``view_handshake_poses.py``
の Good/Bad (``True``/``False``), ``draw_random_human_poses.py`` の
Right/Left/Null (``'R'``/``'L'``/``None``, ``estimate_palm_poses.
PalmPoseEstimator.estimate`` の ``offered_hand`` と同じ値) など)。
"""

import json
import os
import threading
import time


# ``ManualNav.wait()``/``wait_for_advance`` が Back/Next (判定なし) を
# 表したいときに使うセンチネル。判定ボタンの値には ``True``/``False``
# だけでなく ``None`` (Null 判定, offered_hand の「差し出していない」に
# 対応) もあり得るので、「判定ボタンが押されていない」を ``None`` では
# 表せない。
NOT_PRESSED = object()

# ``load_label`` が「JSON にまだ判定結果が無い」ことを表すために使える
# センチネル (``default`` に渡す)。判定値そのものに ``None`` (Null 判定)
# があり得る場合、既定の ``default=None`` では「未判定」と「Null 判定
# 済み」を区別できないため。
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


class ManualNav(object):
    """viser の GUI に Back/Next ボタンと判定ボタンを追加し、押された結果を
    ``wait()`` で受け取れるようにする。

    Back/Next はどちらも同じ ``threading.Event`` を立てて向きだけを伝える
    (直近に押されたボタンの向きだけを覚える。連打しても最後の 1 回分しか
    進まない/戻らない)。判定ボタン (``buttons``) は表示中の姿勢についての
    人手 judgment を表し、押すと (Next と同様に) 次の人物へ進みつつ、判定
    結果 (対応する値) も一緒に伝える。
    """

    def __init__(self, viewer, buttons=None):
        """
        Parameters
        ----------
        buttons : list of (str, object) or None
            判定ボタンの ``(表示テキスト, 値)`` のリスト。既定
            (``None``) は ``[('Good', True), ('Bad', False)]``
            (``view_handshake_poses.py`` の IK 判定)。
            ``draw_random_human_poses.py`` は差し出している手の人手判定
            用に ``[('Right', 'R'), ('Left', 'L'), ('Null', None)]``
            (``estimate_palm_poses.py`` の ``offered_hand`` と同じ値)
            を渡す。
        """
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
        """Back/Next/判定ボタンのいずれかが押されるまで待つ.

        ``(direction, label)`` を返す。``direction`` は ``-1``/``+1``、
        ``label`` は判定ボタンが押されたときだけそのボタンの値 (Back/Next
        のときは :data:`NOT_PRESSED`)。ブラウザクライアントが切断されたら
        ``(0, NOT_PRESSED)`` を返す。
        """
        self._event.clear()
        while not self._event.is_set():
            if not viewer._server.get_clients():
                return 0, NOT_PRESSED
            time.sleep(0.05)
        return self._direction, self._label


def wait_for_advance(viewer, nav, pause):
    """次に表示する人物への向きと、判定ボタンの結果を決める.

    ``nav`` (``ManualNav``) が渡されていれば、Back/Next/判定ボタンが押さ
    れるまで待って ``(direction, label)`` を返す (``label`` は判定ボタンが
    押されたときだけその値、Back/Next のときは :data:`NOT_PRESSED`)。
    渡されていなければ ``pause`` 秒だけ待って常に ``(1, NOT_PRESSED)`` を
    返す (自動送り)。ブラウザクライアントが切断されていれば
    ``(None, NOT_PRESSED)`` を返す。
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


def save_label(json_path, label, key='human_label'):
    """判定ボタンで判定した結果を JSON に書き込む.

    既存の JSON があればその内容を保ったまま ``key`` (既定
    ``human_label``) を追記する。JSON が無ければ、判定結果だけを持つ
    JSON を新規に作る。
    """
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}
    data[key] = label
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_label(json_path, key='human_label', default=None):
    """``save_label`` が書き込んだ判定結果を読む.

    JSON が無い、もしくは ``key`` が無ければ (まだ判定ボタンが押されて
    いなければ) ``default`` を返す。判定値そのものに ``None`` (Null 判定)
    があり得る呼び出し元は、「未判定」と区別できるよう
    ``default=UNLABELED`` を渡すこと。
    """
    if not os.path.exists(json_path):
        return default
    with open(json_path) as f:
        data = json.load(f)
    return data.get(key, default)


def format_label_text(label, title='ラベル', value_names=None):
    """``load_label``/判定ボタンの結果を GUI 表示用の文字列にする.

    Parameters
    ----------
    value_names : dict or None
        判定値 -> 表示テキスト。既定 (``None``) は
        ``{True: 'Good', False: 'Bad'}``。``label`` がここに無ければ
        (``NOT_PRESSED``/``UNLABELED`` を含め) 「未判定」と表示する。
    """
    if value_names is None:
        value_names = {True: 'Good', False: 'Bad'}
    if label in value_names:
        return '**{}:** {}'.format(title, value_names[label])
    return '**{}:** (未判定)'.format(title)
