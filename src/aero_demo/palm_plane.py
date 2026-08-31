#!/usr/bin/env python3
"""Least-squares palm plane estimation from MediaPipe hand landmarks.

Shared by ``human_palm_contact_behavior.py`` (which reaches for the palm)
and ``palm_plane_visualizer.py`` (which only draws the result in a
scikit-robot viewer), so both agree on exactly what "the palm plane" means.

Why a least-squares fit instead of one cross product
----------------------------------------------------
The original behavior required RHand{0,5,9,17} (wrist + index/middle/pinky
MCP) and took a single cross product.  Measured against a 34-frame sample
of ``/people_pose_estimation_mediapipe/pose``, that AND-condition held in
only 76.5% of frames: RHand5 (index MCP) is published in just 79.4%, and
the thumb landmarks are worse still.  The palm-fixed landmarks rank

    RHand0  wrist       100.0%
    RHand9  middle MCP  100.0%
    RHand13 ring MCP    100.0%
    RHand17 pinky MCP    97.1%
    RHand1  thumb CMC    88.2%
    RHand5  index MCP    79.4%

and at least 3 of them were present in 100% of frames.  So instead of
demanding a fixed quadruple, we fit a plane by SVD to whichever of
``PLANE_LANDMARKS`` arrived, which both raises the hit rate to ~100% and
averages out per-landmark jitter.

The module deliberately depends only on numpy and ROS messages (no
skrobot), so the fitting itself runs on a machine without the robot model.
The RViz marker helpers below are kept for ``human_palm_contact_behavior``
and for anyone who still wants markers; the visualiser draws the same
geometry with skrobot's viewer instead.
"""

import math
from collections import namedtuple

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

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

LANDMARK_LABELS = {
    0: 'wrist', 1: 'thumb_cmc', 5: 'index_mcp', 9: 'middle_mcp',
    13: 'ring_mcp', 17: 'pinky_mcp',
}

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


def _unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return np.asarray(v, dtype=np.float64) / n


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
    # The SVD normal's sign is ambiguous (palm or back of hand -- both
    # surfaces are parallel to the same fitted plane), and it must not be
    # resolved by guessing "whichever side faces the robot": that breaks
    # down whenever the palm is presented edge-on / sideways to the robot
    # rather than flat-on (exactly the case for a natural handshake
    # posture, see fake_people_pose_estimator_ros.py's
    # ~present_wrist_roll_deg_range), where a small pose change can flip
    # which side that heuristic calls "facing the robot".
    #
    # What is anatomically fixed regardless of viewing angle is MediaPipe's
    # knuckle layout: index MCP (5) is always on the thumb side and pinky
    # MCP (17) always on the little-finger side (MCP_LATERAL_RANK).  A
    # lateral axis built from that ordering, combined with the fixed
    # left/right handedness of an R or L hand, pins down the normal's sign
    # without needing to know where the robot is at all.
    #
    # This mirrors fake_people_pose_estimator_ros.py's own hand frame
    # convention (u=finger_dir, v=thumb-side, n=palm normal), where the
    # (u, v, n) triple is left-handed for the right hand and right-handed
    # for the left one -- see that module's ``_body_positions`` -- i.e.
    # v = u x n for R, v = n x u for L, which inverted for n gives the
    # formulas below.
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
        # fall back to the old "faces the robot" guess.
        if viewpoint is None:
            viewpoint = np.zeros(3)
        viewpoint = np.asarray(viewpoint, dtype=np.float64)
        if float(np.dot(normal, viewpoint - center)) < 0.0:
            normal = -normal

    # --- build a full frame: +X = fingers, +Y = palm normal --------------
    # Checked directly against Aero's r/l_eef_grasp_link frame (built by
    # loading the URDF in skrobot and probing it -- see the analysis in
    # scripts/human_palm_contact_behavior.py's IK-target comment): local
    # +X is the wrist -> fingertip direction (confirmed by how far along
    # +X the fingertip links sit), and local +Y is the *dorsum -> palm*
    # direction -- confirmed by driving the finger-curl joints and
    # watching the fingertips move toward +Y as they close, which is
    # exactly the side "the palm" is on.  +Z (thumb <-> pinky width)
    # completes the right-handed frame.
    #
    # A robot palm pressed flat against the human's must face the human,
    # i.e. point roughly opposite the human normal (which itself points
    # out of the human's palm -- see above).  So the robot's local +Y
    # (dorsum -> palm) must equal -normal, which is the same thing as
    # saying local -Y (palm -> dorsum, "back of the hand") equals +normal.
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


def _matrix_to_quaternion_xyzw(m):
    """3x3 rotation matrix -> [x, y, z, w] (ROS message order)."""
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return [x, y, z, w]


def _base_marker(frame_id, stamp, ns, marker_id, m_type, lifetime):
    m = Marker()
    m.header.frame_id = frame_id
    if stamp is not None:
        m.header.stamp = stamp
    m.ns = ns
    m.id = marker_id
    m.type = m_type
    m.action = Marker.ADD
    m.pose.orientation.w = 1.0
    if lifetime is not None:
        m.lifetime = lifetime
    return m


def palm_plane_markers(plane, frame_id, stamp=None, ns='palm_plane',
                       lifetime=None, plate_size=0.12,
                       contact_offset=CONTACT_OFFSET,
                       embed_depth=EMBED_DEPTH, label=''):
    """Build a MarkerArray visualising a fitted palm plane.

    Contents
    --------
    0 : translucent plate lying in the fitted plane (the palm surface)
    1 : green arrow along the palm normal, from the palm centre
    2 : cyan arrow along the finger direction (in-plane, for orientation)
    3 : white spheres at the landmarks that were actually fitted
    4 : blue sphere at the approach target (centre + offset * normal)
    5 : red sphere at the press target (centre - depth * normal)
    6 : text with the fit diagnostics
    """
    arr = MarkerArray()

    quat = _matrix_to_quaternion_xyzw(plane.rot)

    # 0: the fitted plane, drawn as a thin plate.  Local +Y is -normal
    # (see palm_plane.fit_palm_plane), so the flat dimensions are x and z.
    plate = _base_marker(frame_id, stamp, ns, 0, Marker.CUBE, lifetime)
    plate.pose.position = Point(*plane.center)
    plate.pose.orientation.x = quat[0]
    plate.pose.orientation.y = quat[1]
    plate.pose.orientation.z = quat[2]
    plate.pose.orientation.w = quat[3]
    plate.scale.x = plate_size
    plate.scale.y = 0.002
    plate.scale.z = plate_size
    plate.color = ColorRGBA(r=1.0, g=0.55, b=0.15, a=0.55)
    arr.markers.append(plate)

    # 1: palm normal.
    normal_arrow = _base_marker(frame_id, stamp, ns, 1, Marker.ARROW, lifetime)
    normal_arrow.points = [Point(*plane.center),
                           Point(*(plane.center + plane.normal * 0.15))]
    normal_arrow.scale.x = 0.008   # shaft diameter
    normal_arrow.scale.y = 0.02    # head diameter
    normal_arrow.scale.z = 0.03    # head length
    normal_arrow.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)
    arr.markers.append(normal_arrow)

    # 2: finger direction, so the frame's roll is visible too.
    finger_arrow = _base_marker(frame_id, stamp, ns, 2, Marker.ARROW, lifetime)
    finger_arrow.points = [Point(*plane.center),
                           Point(*(plane.center + plane.finger_dir * 0.08))]
    finger_arrow.scale.x = 0.005
    finger_arrow.scale.y = 0.014
    finger_arrow.scale.z = 0.02
    finger_arrow.color = ColorRGBA(r=0.0, g=0.8, b=1.0, a=1.0)
    arr.markers.append(finger_arrow)

    # 4: approach target.
    approach = _base_marker(frame_id, stamp, ns, 4, Marker.SPHERE, lifetime)
    approach.pose.position = Point(*contact_target(plane, contact_offset))
    approach.scale.x = approach.scale.y = approach.scale.z = 0.025
    approach.color = ColorRGBA(r=0.2, g=0.4, b=1.0, a=1.0)
    arr.markers.append(approach)

    # 5: press-into-the-palm target.
    press = _base_marker(frame_id, stamp, ns, 5, Marker.SPHERE, lifetime)
    press.pose.position = Point(*embed_target(plane, embed_depth))
    press.scale.x = press.scale.y = press.scale.z = 0.02
    press.color = ColorRGBA(r=1.0, g=0.2, b=0.2, a=1.0)
    arr.markers.append(press)

    # 6: diagnostics.
    text = _base_marker(frame_id, stamp, ns, 6, Marker.TEXT_VIEW_FACING,
                        lifetime)
    text.pose.position = Point(*(plane.center + plane.normal * 0.18))
    text.scale.z = 0.03
    text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    used = ','.join(LANDMARK_LABELS.get(i, str(i)) for i in plane.used)
    text.text = '{}n={} [{}]\nrms={:.1f}mm span={:.0f}mm ratio={:.2f}'.format(
        label + '\n' if label else '',
        len(plane.used), used, plane.rms * 1000.0,
        plane.span * 1000.0, plane.span_ratio)
    arr.markers.append(text)
    return arr


def palm_landmark_markers(points, frame_id, stamp=None, ns='palm_plane',
                          marker_id=3, lifetime=None, used=None):
    """Spheres at the raw palm landmarks (fitted ones white, others grey)."""
    arr = MarkerArray()
    if not points:
        return arr
    used = set(used or points.keys())
    fitted = _base_marker(frame_id, stamp, ns, marker_id, Marker.SPHERE_LIST,
                          lifetime)
    fitted.scale.x = fitted.scale.y = fitted.scale.z = 0.015
    fitted.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
    for i in sorted(points):
        if i in used:
            fitted.points.append(Point(*points[i]))
    if fitted.points:
        arr.markers.append(fitted)
    return arr


def delete_all_marker(frame_id, ns='palm_plane'):
    """A DELETEALL marker, to clear stale visualisation."""
    arr = MarkerArray()
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.action = Marker.DELETEALL
    arr.markers.append(m)
    return arr
