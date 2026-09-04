#!/usr/bin/env python3
"""Least-squares palm plane estimation from MediaPipe hand landmarks.

Used by ``estimate_palm_poses.py`` (offered-hand estimation) and
``aero_demo.palm_plane_view`` (drawing the result in a scikit-robot
viewer), so both agree on exactly what "the palm plane" means.

A plane is fit by SVD to whichever of ``PLANE_LANDMARKS`` (wrist + MCP
knuckles) is present, rather than requiring a fixed quadruple and a single
cross product, since not every landmark is reliably tracked; this both
raises the effective hit rate and averages out per-landmark jitter.  The
module depends only on numpy (no ROS messages, no skrobot).
"""

from collections import namedtuple

import numpy as np

from aero_demo.vector_utils import unit as _unit

# --- MediaPipe hand-landmark indices -------------------------------------
# published by people_pose_estimation_mediapipe.py as "RHand0".."RHand20"
# (and "LHand0".."LHand20") when ~hand/enable is true.
WRIST_INDEX = 0
MCP_INDICES = (5, 9, 13, 17)          # index / middle / ring / pinky knuckles

# Landmarks used for the plane fit: the wrist plus the four knuckles.  These
# are the only hand landmarks rigidly attached to the palm -- everything
# distal to an MCP swings out of the palm plane when a finger bends, and the
# thumb CMC (1) leaves the plane as the thumb opposes, so both are excluded.
PLANE_LANDMARKS = (WRIST_INDEX,) + MCP_INDICES

# Relative position of each knuckle across the palm, index (thumb) side to
# pinky side -- MediaPipe's landmark layout is anatomically fixed (index
# MCP is always on the thumb side, pinky MCP always on the little-finger
# side) regardless of which way the hand happens to be turned, so this
# ordering -- unlike "which side faces the robot" -- is a reliable way to
# tell the palm side from the back of the hand.  Values only need the
# right relative order/spacing, not any particular unit.
MCP_LATERAL_RANK = {5: 1.5, 9: 0.5, 13: -0.5, 17: -1.5}

# --- fit quality gates ---------------------------------------------------
# A plane needs 3 non-collinear points.
MIN_PALM_POINTS = 3
# Reject near-collinear point sets: the second singular value carries the
# in-plane width of the point cloud, so a small S1/S0 means the landmarks
# lie almost on a line and the normal direction is numerically meaningless.
MIN_SPAN_RATIO = 0.15

# --- contact geometry ----------------------------------------------------
# Where along the wrist -> knuckles axis we call "the centre of the palm".
# 0.0 = at the wrist, 1.0 = at the knuckle row, 0.5 = mid-palm (the fleshy
# part you would actually touch).
PALM_CENTER_MCP_WEIGHT = 0.5

# How far in front of the palm surface (along +normal) the robot's hand is
# aimed for the initial approach, so it stops just off the skin.
CONTACT_OFFSET = 0.02
# How far past the surface (along -normal) it then presses, so the hand
# actually sinks into the palm instead of stopping short of it.
EMBED_DEPTH = 0.01


PalmPlane = namedtuple('PalmPlane', [
    'center',       # (3,) mid-palm point
    'normal',       # (3,) unit normal, pointing out of the palm (see
                    # fit_palm_plane's anatomical sign resolution)
    'rot',          # (3,3) rotation matrix matching Aero's r/l_eef_grasp_link
                    # axes: local +X = fingers (wrist->fingertip), +Y = the
                    # dorsum->palm axis set to -normal (so the robot's palm
                    # faces the human's), +Z = thumb<->pinky width
    'finger_dir',   # (3,) unit vector wrist -> knuckles, in the plane
    'used',         # sorted list of landmark indices used for the fit
    'rms',          # RMS distance of those points from the fitted plane [m]
    'span',         # in-plane extent of the point cloud (2nd sing. value)
    'span_ratio',   # S1/S0, the conditioning measure gated by MIN_SPAN_RATIO
])


def collect_palm_points(person, hand='R', min_score=0.1,
                        indices=PLANE_LANDMARKS):
    """Pull the palm landmarks out of one person's pose.

    Parameters
    ----------
    person : people_pose_types.Person3D
        推定クラス (people_pose_estimator_ros / fake_people_pose_estimator_ros)
        が返す ``EstimationResult.people`` の 1 人分。点は既に
        ``EstimationResult.frame_id`` (既定 base_link) 相対なので、ここでは
        座標変換をしない。
    hand : str
        ``'R'`` or ``'L'`` -- which hand's landmarks to read.
    min_score : float
        Landmarks at or below this confidence are ignored.
    indices : iterable of int
        Landmark indices to look for.

    Returns
    -------
    dict
        ``{landmark_index: (3,) ndarray}`` for the landmarks that were
        present and confident.
    """
    names = list(person.limb_names)
    points = {}
    for i in indices:
        key = '{}Hand{}'.format(hand, i)
        if key not in names:
            continue
        j = names.index(key)
        if person.scores[j] <= min_score:
            continue
        points[i] = np.asarray(person.positions[j], dtype=np.float64)
    return points


def fit_palm_plane(points, hand='R', viewpoint=None):
    """Fit a plane to the palm landmarks by SVD.

    Parameters
    ----------
    points : dict
        ``{landmark_index: (3,) ndarray}`` as returned by
        :func:`collect_palm_points`.  At least :data:`MIN_PALM_POINTS`
        entries are required.
    hand : str
        ``'R'`` or ``'L'`` -- which hand ``points`` came from.  Needed to
        resolve the fitted normal's palm/back-of-hand sign from the
        knuckles' own left-right layout (see below); get this wrong and
        every plane from this call comes out back-to-front.
    viewpoint : (3,) array_like or None
        Fallback sign reference, used only when too few knuckles survived
        tracking to tell the palm side from the layout itself (see below).
        Defaults to the frame origin (``base_link``, i.e. the robot).

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

    # Columns of V are the principal directions of the point cloud; the one
    # with the smallest singular value is the plane normal.
    _u, s, vt = np.linalg.svd(centred, full_matrices=False)
    if len(s) < 3 or s[0] < 1e-9:
        return None
    span_ratio = float(s[1] / s[0])
    if span_ratio < MIN_SPAN_RATIO:
        # Points are essentially on a line -- the normal would be arbitrary.
        return None

    normal = _unit(vt[2])
    if normal is None:
        return None

    # Signed distances of the fitted points from the plane.
    rms = float(np.sqrt(np.mean((centred @ normal) ** 2)))

    # --- palm centre: blend the wrist toward the knuckle row -------------
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

    # +X = fingers (wrist -> knuckle row), in the plane.  Computed before
    # resolving the normal's sign because this projection is sign-of-
    # -normal-independent (dot(finger_dir, normal) * normal is the same
    # for +-normal), and the anatomical sign check right below needs it.
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

    # --- orient the normal using the hand's own anatomy -------------------
    # The SVD normal's sign is ambiguous (palm or back of hand), and can't
    # be resolved by guessing "whichever side faces the robot" -- that
    # breaks down whenever the palm is presented edge-on to the robot
    # rather than flat-on. Instead we use what's anatomically fixed
    # regardless of viewing angle: MediaPipe's knuckle layout, where index
    # MCP (5) is always on the thumb side and pinky MCP (17) always on the
    # little-finger side (MCP_LATERAL_RANK). A lateral axis built from that
    # ordering, combined with the fixed left/right handedness of an R or L
    # hand, pins down the normal's sign (v = u x n for R, v = n x u for L,
    # inverted for n below) without needing to know where the robot is.
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
        # Fewer than 2 knuckles with distinct thumb/pinky ranks survived
        # tracking -- not enough to tell the palm side from the layout, so
        # fall back to a "faces the robot" guess.
        if viewpoint is None:
            viewpoint = np.zeros(3)
        viewpoint = np.asarray(viewpoint, dtype=np.float64)
        if float(np.dot(normal, viewpoint - center)) < 0.0:
            normal = -normal

    # --- build a full frame: +X = fingers, +Y = palm normal --------------
    # Matches Aero's r/l_eef_grasp_link axes: local +X is wrist->fingertip,
    # local +Y is dorsum->palm, +Z (thumb<->pinky) completes the
    # right-handed frame. A robot palm pressed against the human's must
    # face them, i.e. point opposite the human normal, so local +Y = -normal.
    y_axis = -normal
    z_axis = np.cross(x_axis, y_axis)
    rot = np.column_stack([x_axis, y_axis, z_axis])

    return PalmPlane(center=center, normal=normal, rot=rot,
                     finger_dir=x_axis, used=idxs, rms=rms,
                     span=float(s[1]), span_ratio=span_ratio)


def contact_target(plane, offset=CONTACT_OFFSET):
    """Approach point: just off the palm surface, along +normal."""
    return plane.center + plane.normal * offset


def embed_target(plane, depth=EMBED_DEPTH):
    """Press point: inside the palm, along -normal."""
    return plane.center - plane.normal * depth


