#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""手ありの Aero URDF (aero_with_feetech_hand.urdf) を、ROS を source
していなくても読み込めるように準備する。

``aero_with_feetech_hand.urdf`` はメッシュを
``package://feetech_hand/meshes/...`` から参照する。scikit-robot は
``package://`` を ROS (``ROS_PACKAGE_PATH``/ament) で解決できないとき、
URDF 自身のディレクトリから上へ辿って同名の相対パスを探す
(``skrobot.utils.urdf.search_up``)。``aero_description`` の分はこの
フォールバックだけで解決できている (展開したアーカイブのトップディレクトリが
たまたま ``aero_description`` という名前で、URDF の祖先ディレクトリに
なっているため)。同じ仕組みが ``feetech_hand`` でも働くように、
scikit-robot のキャッシュディレクトリ (``aero_description`` の親)
直下に ``feetech_hand`` パッケージへのシンボリックリンクを作る。
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
    # aero_demo と feetech_hand は同じワークスペースの src/ 直下にある
    # 兄弟パッケージという前提 (README のセットアップ手順と同じ前提)。
    workspace_src = osp.abspath(
        osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', '..'))
    return osp.join(workspace_src, 'feetech_hand')


def ensure_feetech_hand_urdf_cached():
    """手あり URDF とそのメッシュを、ROS 非依存で解決できる場所に置く。

    Returns
    -------
    str
        ``skrobot.data.aero_urdfpath(use_hand=True)`` が返すのと同じ、
        キャッシュ内の ``aero_with_feetech_hand.urdf`` のパス。
    """
    source_dir = _feetech_hand_source_dir()
    source_urdf = osp.join(source_dir, 'urdf', 'aero_with_feetech_hand.urdf')
    if not osp.exists(source_urdf):
        raise FileNotFoundError(
            "feetech_hand パッケージが見つかりません ('{}' が存在しません)。"
            " aero_demo と同じワークスペースの src/ 直下に feetech_hand を"
            " clone するか、{} 環境変数でそのディレクトリを指定してください。"
            .format(source_urdf, FEETECH_HAND_DIR_ENV))

    # aero_description (typeJSK/urdf/aero_nohand.urdf 等) をキャッシュに
    # 用意する (初回はここで自動ダウンロードされる)。
    target_urdf = aero_urdfpath(use_hand=True)

    if not osp.exists(target_urdf) or (
            osp.getmtime(source_urdf) > osp.getmtime(target_urdf)):
        shutil.copy2(source_urdf, target_urdf)

    # aero_description と同じ仕組みで package://feetech_hand/... が
    # 解決できるよう、キャッシュ直下に feetech_hand へのリンクを作る。
    cache_dir = get_cache_dir()
    feetech_hand_link = osp.join(cache_dir, 'feetech_hand')
    if osp.islink(feetech_hand_link):
        if osp.realpath(feetech_hand_link) != osp.realpath(source_dir):
            os.remove(feetech_hand_link)
            os.symlink(source_dir, feetech_hand_link)
    elif not osp.exists(feetech_hand_link):
        os.symlink(source_dir, feetech_hand_link)

    return target_urdf


def load_aero(use_hand=True, *args, **kwargs):
    """``skrobot.models.Aero`` を、手ありの場合は事前準備をしてから作る。"""
    from skrobot.models import Aero

    if use_hand:
        ensure_feetech_hand_urdf_cached()
    return Aero(use_hand=use_hand, *args, **kwargs)
