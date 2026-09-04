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
    """Rodrigues' rotation formula: rotate ``v`` by ``angle`` [rad] around
    the unit vector ``axis``."""
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c)


def rotation_from_z(z_axis):
    """Rotation matrix whose +Z column points along ``z_axis``."""
    z = unit(z_axis)
    if z is None:
        return np.eye(3)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(z[2])) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    x = unit(np.cross(ref, z))
    if x is None:
        return np.eye(3)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])
