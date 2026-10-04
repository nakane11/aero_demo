#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""手ありの Aero URDF (aero_with_feetech_hand.urdf) を ROS 無しで読めるようにする.

scikit-robot は ``package://`` を解決できないと URDF の祖先ディレクトリを
探すので、キャッシュディレクトリ直下に ``feetech_hand`` へのリンクを作る。
"""

import os
import os.path as osp
import shutil

from skrobot.data import aero_urdfpath
from skrobot.data import get_cache_dir

FEETECH_HAND_DIR_ENV = 'FEETECH_HAND_DIR'


def _feetech_hand_source_dir():
    """feetech_hand パッケージ (urdf/meshes を含む) のディレクトリを返す。"""
    env = os.environ.get(FEETECH_HAND_DIR_ENV)
    if env:
        return osp.abspath(osp.expanduser(env))
    # 同じワークスペースの src/ 直下の兄弟パッケージという前提。
    workspace_src = osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', '..'))
    return osp.join(workspace_src, 'feetech_hand')


def ensure_feetech_hand_urdf_cached():
    """手あり URDF とメッシュをキャッシュに置き、URDF のパスを返す."""
    source_dir = _feetech_hand_source_dir()
    source_urdf = osp.join(source_dir, 'urdf', 'aero_with_feetech_hand.urdf')
    if not osp.exists(source_urdf):
        raise FileNotFoundError(
            "feetech_hand パッケージが見つかりません ('{}' が存在しません)。"
            " aero_demo と同じワークスペースの src/ 直下に feetech_hand を"
            " clone するか、{} 環境変数でそのディレクトリを指定してください。"
            .format(source_urdf, FEETECH_HAND_DIR_ENV))

    # 初回は aero_description が自動ダウンロードされる。
    target_urdf = aero_urdfpath(use_hand=True)

    if not osp.exists(target_urdf) or (
            osp.getmtime(source_urdf) > osp.getmtime(target_urdf)):
        shutil.copy2(source_urdf, target_urdf)

    # package://feetech_hand/... を解決させるためのリンク
    cache_dir = get_cache_dir()
    feetech_hand_link = osp.join(cache_dir, 'feetech_hand')
    if osp.islink(feetech_hand_link):
        if osp.realpath(feetech_hand_link) != osp.realpath(source_dir):
            os.remove(feetech_hand_link)
            os.symlink(source_dir, feetech_hand_link)
    elif not osp.exists(feetech_hand_link):
        os.symlink(source_dir, feetech_hand_link)

    return target_urdf


def load_aero(use_hand=True, **kwargs):
    """``skrobot.models.Aero`` を、手ありの場合は事前準備をしてから作る。"""
    from skrobot.models import Aero

    if use_hand:
        ensure_feetech_hand_urdf_cached()
    return Aero(use_hand=use_hand, **kwargs)
