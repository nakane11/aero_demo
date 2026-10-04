#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""3 次元関節位置の時系列を平滑化するフィルタ.

Examples
--------
>>> f = OneEuroFilter()
>>> f.update({'Neck': [0.0, 0.0, 1.0]}, t=0.0)
{'Neck': array([0., 0., 1.])}
"""

import numpy as np

__all__ = ['OneEuroFilter']


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
    """関節名・軸ごとの One Euro Filter (``update`` に時刻 t [s] が必要).

    mincutoff を下げると静止時のジッタが減り、beta を上げると追従遅れが減る。
    既定値は run_camera_pipeline_test.py の既定と同じ。
    """

    def __init__(self, mincutoff=0.5, beta=0.3, dcutoff=1.0):
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
