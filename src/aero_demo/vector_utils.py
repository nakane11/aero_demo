#!/usr/bin/env python3
"""Vector helpers shared across the human-pose / palm-pose pipeline."""

import numpy as np


def unit(v, fallback=None):
    """Normalize ``v``; return ``fallback`` if its norm is ~0."""
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return fallback
    return v / n


def rotate(v, axis, angle):
    """Rotate ``v`` by ``angle`` [rad] around the unit vector ``axis``."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)
