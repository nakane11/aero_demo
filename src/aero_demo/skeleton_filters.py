#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""3 次元関節位置の時系列を平滑化する、ROS 非依存のフィルタ群.

``scripts/filter_skeleton_data.py`` が ``scripts/ros/record_skeleton_data.py``
で録った生データにオフラインで適用し、パラメータ (window/mincutoff/beta 等)
を比較検討するために使う。``run_camera_pipeline_test.py`` はこのモジュールの
``OneEuroFilter`` をオンライン (フレームごとに ``update`` を呼ぶ) でそのまま
使っている -- 実データで比較した結果、``MedianFilter`` (旧
``run_camera_pipeline_test._JointSmoother`` と同じ移動中央値のアルゴリズム)
よりも跳びを抑えつつ追従の遅れが小さかったため。


Examples
--------
>>> f = MedianFilter(window=3)
>>> f.update({'Neck': [0.0, 0.0, 1.0]}, t=0.0)
{'Neck': array([0., 0., 1.])}
"""

import numpy as np

__all__ = ['MedianFilter', 'MovingAverageFilter', 'OneEuroFilter',
          'FILTERS', 'make_filter']


class MedianFilter(object):
    """関節名・成分ごとに、直近 ``window`` フレームの中央値を返す."""

    def __init__(self, window=3):
        self.window = max(1, int(window))
        self._history = {}
        self._frame_key = None

    def reset(self):
        self._history = {}
        self._frame_key = None

    def update(self, joint_positions, t=None, frame_key=None):
        """``t`` は他フィルタとインタフェースを揃えるためだけの引数で、
        このフィルタ自体は使わない (フレーム数だけで window を数える)。
        """
        if frame_key != self._frame_key:
            self._history = {}
            self._frame_key = frame_key
        filtered = {}
        for name, pos in joint_positions.items():
            history = self._history.setdefault(name, [])
            history.append(np.asarray(pos, dtype=np.float64))
            if len(history) > self.window:
                del history[0]
            filtered[name] = np.median(np.stack(history, axis=0), axis=0)
        for name in list(self._history):
            if name not in joint_positions:
                del self._history[name]
        return filtered


class MovingAverageFilter(object):
    """関節名・成分ごとに、直近 ``window`` フレームの平均を返す.

    ``MedianFilter`` と違い単発の外れ値にも引かれるが、その分位相ずれ
    (追従の遅れ) が小さく素直な平滑化になる。外れ値混入が少ないデータで
    ジッタだけを抑えたい場合に向く。
    """

    def __init__(self, window=5):
        self.window = max(1, int(window))
        self._history = {}
        self._frame_key = None

    def reset(self):
        self._history = {}
        self._frame_key = None

    def update(self, joint_positions, t=None, frame_key=None):
        if frame_key != self._frame_key:
            self._history = {}
            self._frame_key = frame_key
        filtered = {}
        for name, pos in joint_positions.items():
            history = self._history.setdefault(name, [])
            history.append(np.asarray(pos, dtype=np.float64))
            if len(history) > self.window:
                del history[0]
            filtered[name] = np.mean(np.stack(history, axis=0), axis=0)
        for name in list(self._history):
            if name not in joint_positions:
                del self._history[name]
        return filtered


class _OneEuroScalar(object):
    """1 次元の One Euro Filter (Casiez et al., 2012)."""

    def __init__(self, mincutoff, beta, dcutoff):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self._t_prev is None or t <= self._t_prev:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x
        dt = t - self._t_prev
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


class OneEuroFilter(object):
    """関節名・軸 (x, y, z) ごとに独立な One Euro Filter を持つフィルタ.

    ``MedianFilter``/``MovingAverageFilter`` の window (フレーム数) と違い、
    実時間 (``t``, 秒) に基づいて減衰するため、フレームレートが変動しても
    同じ設定で使える。``mincutoff`` を下げると静止時のジッタが減り、
    ``beta`` を上げると速い動きへの追従の遅れが減る (Casiez et al. 2012 の
    推奨どおり、まず beta=0 で mincutoff を決め、次に動き出しのラグを見ながら
    beta を上げるとよい)。
    """

    def __init__(self, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self._scalars = {}  # name -> [_OneEuroScalar x3]
        self._frame_key = None

    def reset(self):
        self._scalars = {}
        self._frame_key = None

    def update(self, joint_positions, t=None, frame_key=None):
        if t is None:
            raise ValueError('OneEuroFilter.update() には t (秒) が必要です')
        if frame_key != self._frame_key:
            self._scalars = {}
            self._frame_key = frame_key
        filtered = {}
        for name, pos in joint_positions.items():
            scalars = self._scalars.setdefault(name, [
                _OneEuroScalar(self.mincutoff, self.beta, self.dcutoff)
                for _ in range(3)])
            filtered[name] = np.array(
                [scalars[i](float(pos[i]), t) for i in range(3)])
        for name in list(self._scalars):
            if name not in joint_positions:
                del self._scalars[name]
        return filtered


FILTERS = {
    'median': MedianFilter,
    'moving_average': MovingAverageFilter,
    'one_euro': OneEuroFilter,
}


def make_filter(name, **kwargs):
    """``FILTERS`` の名前からフィルタを作る (CLI からの生成用)."""
    try:
        cls = FILTERS[name]
    except KeyError:
        raise ValueError('未知のフィルタ {!r} (選べるのは {})'.format(
            name, sorted(FILTERS)))
    return cls(**kwargs)
