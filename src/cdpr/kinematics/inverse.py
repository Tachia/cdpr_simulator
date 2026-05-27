r"""Inverse kinematics: pose :math:`\to` cable lengths.

For a cable connecting world anchor :math:`\mathbf{a}_i` to platform attachment
:math:`\mathbf{b}_i` (expressed in the platform body frame), the cable vector
under platform pose :math:`(\mathbf{p}, \mathbf{R})` is

.. math::

    \mathbf{l}_i = \mathbf{a}_i - (\mathbf{p} + \mathbf{R}\,\mathbf{b}_i),

and the cable length is :math:`L_i = \lVert \mathbf{l}_i \rVert`. The unit
vector :math:`\hat{\mathbf{u}}_i = \mathbf{l}_i / L_i` points from the platform
attachment toward the base anchor and is the direction in which the cable
pulls the platform.

All routines here accept either a single ``Pose`` or a batched ``Pose`` whose
position has shape ``(N, 3)``; the leading axis is preserved in the output.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cdpr.core.frames import Pose
from cdpr.core.numerics import safe_normalize
from cdpr.geometry.robot import Robot, RobotGeometry


def _geometry(obj: Robot | RobotGeometry) -> RobotGeometry:
    return obj.geometry if isinstance(obj, Robot) else obj


def cable_vectors(pose: Pose, robot: Robot | RobotGeometry) -> NDArray[np.float64]:
    r"""Cable vectors :math:`\mathbf{l}_i` from platform attachment to base anchor.

    Returns shape ``(m, 3)`` for a single pose, or ``(N, m, 3)`` for a batched
    pose where ``N`` is the leading axis of ``pose.position``.
    """
    g = _geometry(robot)
    # World-frame attachment points: R b_i + p
    b_world = pose.rotation.apply(g.attachments) + pose.position[..., np.newaxis, :]
    return g.anchors - b_world


def cable_lengths(pose: Pose, robot: Robot | RobotGeometry) -> NDArray[np.float64]:
    r"""Cable lengths :math:`L_i = \lVert \mathbf{l}_i \rVert`, shape ``(m,)`` or ``(N, m)``."""
    return np.linalg.norm(cable_vectors(pose, robot), axis=-1)


def cable_unit_vectors(
    pose: Pose,
    robot: Robot | RobotGeometry,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return ``(unit_vectors, lengths)`` for every cable.

    The unit vectors :math:`\hat{\mathbf{u}}_i = \mathbf{l}_i / L_i` point from
    the platform attachment toward the base anchor. For zero-length cables the
    unit vector is set to zero (a degenerate configuration; downstream
    Jacobian code already treats this as singular).
    """
    L = cable_vectors(pose, robot)
    return safe_normalize(L)


def inverse_kinematics(pose: Pose, robot: Robot | RobotGeometry) -> NDArray[np.float64]:
    """Canonical IK entry point. Identical to :func:`cable_lengths`.

    Kept as a separate, named function because the rest of the framework, the
    examples, and the dissertation refer to "inverse kinematics" --- having
    the name in the API matters more than avoiding a one-line wrapper.
    """
    return cable_lengths(pose, robot)
