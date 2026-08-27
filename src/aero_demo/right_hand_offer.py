#!/usr/bin/env python3
"""Solve IK so Aero's *right* hand offers itself toward a detected palm.

Shared by a future ``human_hand_offer_behavior.py`` ROS node (mirroring how
``palm_plane.py`` is shared by ``human_palm_contact_behavior.py`` and
``palm_plane_visualizer.py``) and ``samples/run_right_hand_offer_ik.py``
(which only checks reachability offline, in a skrobot-only process with no
robot, no ROS master, no viewer).

"差し出す" (offer/hold out the hand) vs. ``palm_plane.contact_target``
---------------------------------------------------------------------
``palm_plane.contact_target`` / ``embed_target`` describe the *human's*
palm surface and a point just off it / pressed into it, for a hand that
approaches to touch.  Offering a hand is a different, gentler geometry:
Aero should stop at a comfortable stand-off in front of the human's palm,
with its own palm facing theirs, and wait -- not push in.  Hence a larger
:data:`OFFER_DISTANCE` (no press stage) and its own target-building
function here, built out of the same ``PalmPlane`` (center + normal +
rot) rather than a new detector.

Frame contract
--------------
``plane`` must already be expressed in the robot's ``base_link`` (or
whatever frame the ``robot`` argument's root link uses) -- exactly what
``people_pose_estimator_ros.RosPeoplePoseEstimator`` (``~output_frame``)
already TF-transforms to for the live robot.  This module does no frame
conversion itself, same as ``human_palm_contact_behavior.py``.

Hand-less URDF
--------------
:func:`build_robot` defaults to ``use_hand=False``.  The with-hand URDF
that :class:`skrobot.models.Aero` asks for by default
(``aero_with_feetech_hand.urdf``, wired up in the local
``scikit-robot`` checkout to come from ``feetech_hand/urdf/``) is not
part of the upstream ``aero_description`` archive that
``skrobot.data.aero_urdfpath`` downloads into ``~/.skrobot``, so
``Aero()`` raises ``ValueError`` on a machine that has not separately
installed that file.  The hand-less URDF has no such dependency and
still exposes the same ``r_eef_grasp_link`` / ``rarm_end_coords`` this
module targets, so it is the reliable default for IK-only use (no
fingers are needed to hold out an open hand).  Pass ``use_hand=True``
once that URDF is available.
"""

from collections import namedtuple

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.models import Aero

__all__ = ['OFFER_DISTANCE', 'IK_ATTEMPTS', 'OfferResult',
          'build_robot', 'offer_target_coords', 'solve_right_hand_offer']

# How far in front of the human's palm (along the palm's +normal, i.e. back
# toward the robot) Aero's hand stops.  Comfortably clear of the palm --
# about a handshake's distance -- unlike palm_plane.CONTACT_OFFSET (0.02 m)
# which is calibrated for a hand that is about to touch.
OFFER_DISTANCE = 0.15

# IK is attempted with progressively fewer degrees of freedom / orientation
# constraints, same fallback idea as human_palm_contact_behavior.py's
# ``rotation_mask='xz'`` -> ``rotation_mask=True`` -> ``rotation_mask=False``
# retry, extended with an arm-only tier in case the lifter/waist of
# *_whole_body pull the solver into a posture it cannot resolve for this
# particular target.  ``rotation_mask`` (not the legacy ``rotation_axis``,
# whose axis letters mean "ignore", inverted from what they look like --
# see skrobot.coordinates.math) so 'xz' means "constrain X
# (offer_target_coords.rot's finger axis) and Z (thumb<->pinky width),
# leaving Y (the palm-normal axis the target is actually built around --
# see palm_plane.fit_palm_plane) free for wrist roll".  That is the
# physically meaningful constraint (align the palm, don't care about
# roll about it), so it is tried before the fully-constrained ``True``.
# (limb_attr, rotation_mask) tried in order; first success wins.
IK_ATTEMPTS = (
    ('rarm_whole_body', 'xz'),
    ('rarm_whole_body', True),
    ('rarm_whole_body', False),
    ('rarm', 'xz'),
    ('rarm', True),
    ('rarm', False),
)


OfferResult = namedtuple('OfferResult', [
    'success',        # bool
    'limb',           # str or None -- which IK_ATTEMPTS entry succeeded
    'rotation_mask',  # the matching rotation_mask, or None
    'target',         # skrobot.coordinates.Coordinates, the IK goal
    'reached',        # skrobot.coordinates.Coordinates, rarm_end_coords after solving
    'position_error', # float [m], |reached - target|
    'angle_vector',   # np.ndarray, robot.angle_vector() after solving
])


def build_robot(use_hand=False):
    """Construct the Aero model used for IK. See module docstring."""
    return Aero(use_hand=use_hand)


def offer_target_coords(plane, distance=OFFER_DISTANCE):
    """Where Aero's right hand should stop to hold itself out to ``plane``.

    Parameters
    ----------
    plane : aero_demo.palm_plane.PalmPlane
        A fitted human palm plane, in the robot's base frame.
    distance : float
        Stand-off from the palm centre along +normal [m].

    Returns
    -------
    skrobot.coordinates.Coordinates
        ``rot`` reuses ``plane.rot`` (local +X = fingers, +Y = -normal,
        +Z = width -- see ``palm_plane.fit_palm_plane``), the same
        convention ``human_palm_contact_behavior.py`` feeds straight into
        ``Coordinates(rot=...)`` -- the hand's local +Y is its
        dorsum->palm direction, so this orientation lands Aero's palm
        facing the human's, without extra rotation bookkeeping here.
    """
    pos = np.asarray(plane.center, dtype=np.float64) \
        + np.asarray(plane.normal, dtype=np.float64) * distance
    return Coordinates(pos=pos, rot=np.asarray(plane.rot, dtype=np.float64))


def solve_right_hand_offer(robot, plane, distance=OFFER_DISTANCE,
                           attempts=IK_ATTEMPTS, seed_reset_pose=True):
    """Solve IK so ``robot``'s right hand reaches :func:`offer_target_coords`.

    Parameters
    ----------
    robot : skrobot.models.Aero
    plane : aero_demo.palm_plane.PalmPlane
        In ``robot``'s base frame (see module docstring).
    seed_reset_pose : bool
        Seed the solver from ``robot.reset_pose()`` first.  ``reset_pose``
        is guaranteed within every joint's limits on both URDF variants
        (unlike the hand-tuned fallback postures in
        ``human_palm_contact_behavior.py`` / ``human_handover_behavior.py``,
        which were measured against the with-hand URDF's wider wrist_r
        range and can be out of range on the hand-less one -- see
        :func:`build_robot`), and it is a reasonable non-singular starting
        pose for the solver either way.

    Returns
    -------
    OfferResult
        ``success=False`` means every entry in ``attempts`` returned
        ``False``; ``robot`` is left at whatever the last attempt reached
        (mirrors the existing behaviors, which keep moving even after a
        logged IK failure) and ``reached``/``position_error`` describe that
        final, unsolved pose.
    """
    if seed_reset_pose:
        robot.reset_pose()

    target = offer_target_coords(plane, distance)

    for limb_attr, rotation_mask in attempts:
        limb = getattr(robot, limb_attr)
        try:
            # stop=200 (default 50) / revert_if_fail=False: a fully
            # orientation-constrained target this far from reset_pose's
            # seed needs more iterations than the default budget, and
            # skrobot's default revert_if_fail=True snaps back to the
            # seed on any non-improving iteration, which empirically
            # gets the solver stuck at the seed instead of working
            # through the reorientation (see
            # human_palm_contact_behavior.py for the same fix, arrived
            # at the same way).
            res = limb.inverse_kinematics(
                target, rotation_mask=rotation_mask,
                stop=200, revert_if_fail=False)
        except Exception:
            res = False
        reached = robot.rarm_end_coords.copy_worldcoords()
        error = float(np.linalg.norm(reached.worldpos() - target.worldpos()))
        if res is not False:
            return OfferResult(
                success=True, limb=limb_attr, rotation_mask=rotation_mask,
                target=target, reached=reached, position_error=error,
                angle_vector=robot.angle_vector())

    return OfferResult(
        success=False, limb=None, rotation_mask=None,
        target=target, reached=reached, position_error=error,
        angle_vector=robot.angle_vector())
