#!/usr/bin/env python3
"""Least-squares palm plane estimation from MediaPipe hand landmarks.

Fits a plane by SVD to whichever of the wrist + MCP knuckles are present.
Depends only on numpy.
"""

from collections import namedtuple

import numpy as np

from aero_demo.vector_utils import unit as _unit

# MediaPipe hand-landmark indices ("RHand0".."RHand20" / "LHand0"..).
WRIST_INDEX = 0
MCP_INDICES = (5, 9, 13, 17)          # index / middle / ring / pinky knuckles

# Only these are rigid with the palm (fingers and thumb leave the plane).
PLANE_LANDMARKS = (WRIST_INDEX,) + MCP_INDICES

# Knuckle order from thumb side to pinky side; used to resolve the normal sign.
MCP_LATERAL_RANK = {5: 1.5, 9: 0.5, 13: -0.5, 17: -1.5}

MIN_PALM_POINTS = 3
# Reject near-collinear point sets (S1/S0 below this).
MIN_SPAN_RATIO = 0.15

# Palm centre position between wrist (0.0) and knuckle row (1.0).
PALM_CENTER_MCP_WEIGHT = 0.5


PalmPlane = namedtuple('PalmPlane', [
    'center',       # (3,) mid-palm point
    'normal',       # (3,) unit normal, pointing out of the palm
    'rot',          # (3,3) Aero eef_grasp_link axes: +X = fingers,
                    # +Y = -normal (robot palm faces the human's), +Z = X x Y
    'finger_dir',   # (3,) unit vector wrist -> knuckles, in the plane
    'used',         # sorted list of landmark indices used for the fit
    'rms',          # RMS distance of those points from the plane [m]
    'span',         # 2nd singular value
    'span_ratio',   # S1/S0
])


def fit_palm_plane(points, hand='R'):
    """Fit a plane to the palm landmarks by SVD.

    ``hand`` ('R'/'L') is needed to resolve the normal sign from the knuckle
    layout. With fewer than 2 knuckles the palm is assumed to face the origin.

    Returns
    -------
    PalmPlane or None
        ``None`` when there are too few points or they are near-collinear.
    """
    idxs = sorted(points)
    if len(idxs) < MIN_PALM_POINTS:
        return None

    pts = np.array([points[i] for i in idxs], dtype=np.float64)
    centroid = pts.mean(axis=0)
    centred = pts - centroid

    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    if len(s) < 3 or s[0] < 1e-9:
        return None
    span_ratio = float(s[1] / s[0])
    if span_ratio < MIN_SPAN_RATIO:
        return None

    normal = _unit(vt[2])
    if normal is None:
        return None

    rms = float(np.sqrt(np.mean((centred @ normal) ** 2)))

    mcps = [i for i in idxs if i in MCP_INDICES]
    if mcps:
        mcp_center = np.mean([points[i] for i in mcps], axis=0)
    else:
        mcp_center = centroid
    if WRIST_INDEX in points and mcps:
        wrist = points[WRIST_INDEX]
        center = ((1.0 - PALM_CENTER_MCP_WEIGHT) * wrist
                  + PALM_CENTER_MCP_WEIGHT * mcp_center)
        finger_dir = _unit(mcp_center - wrist)
    else:
        center = mcp_center
        finger_dir = None

    if finger_dir is None:
        # Fall back to any axis not parallel to the normal.
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(normal[2])) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        finger_dir = _unit(ref - float(np.dot(ref, normal)) * normal)
        if finger_dir is None:
            return None

    x_axis = _unit(finger_dir - float(np.dot(finger_dir, normal)) * normal)
    if x_axis is None:
        return None

    # Resolve the normal sign from the knuckle layout and handedness, which
    # works regardless of viewing angle.
    lateral = np.zeros(3)
    n_ranked = 0
    for i, rank in MCP_LATERAL_RANK.items():
        if i in points:
            lateral += rank * (points[i] - centroid)
            n_ranked += 1
    v_axis = _unit(lateral - float(np.dot(lateral, x_axis)) * x_axis) \
        if n_ranked >= 2 else None
    if v_axis is not None:
        anatomical_normal = (np.cross(v_axis, x_axis) if hand == 'R'
                             else np.cross(x_axis, v_axis))
        if float(np.dot(normal, anatomical_normal)) < 0.0:
            normal = -normal
    else:
        # Fallback: assume the palm faces the origin.
        if float(np.dot(normal, -center)) < 0.0:
            normal = -normal

    # Robot palm (+Y of eef_grasp_link) faces the human's: +Y = -normal.
    y_axis = -normal
    z_axis = np.cross(x_axis, y_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])

    return PalmPlane(center=center, normal=normal, rot=rot,
                     finger_dir=x_axis, used=idxs, rms=rms,
                     span=float(s[1]), span_ratio=span_ratio)
