r"""Wrench-feasible workspace (WFW) test.

A pose lies in the wrench-feasible workspace *for a particular external
wrench* :math:`\mathbf{w}_\text{ext}` if a tension vector in the cable bound
box :math:`[\mathbf{t}_\text{min}, \mathbf{t}_\text{max}]` exists satisfying

.. math::

    \mathbf{W}(\mathbf{q})\,\mathbf{t} = -\,\mathbf{w}_\text{ext}.

For a *set* of external wrenches the pose must satisfy the condition for
every wrench in the set; the standard reduction (Bouchard, Gosselin, Moore
2010) tests only the vertices of the wrench polytope when the set is itself
a polytope, which makes the typical "gravity plus a payload at an unknown
position in some box" workspace question tractable.

The single-wrench test below delegates to the LP feasibility routine in the
statics layer. Multi-wrench tests are intentionally not provided as a single
opaque helper --- they depend on the wrench set's specific structure, and
inlining the vertex enumeration in research scripts is clearer than hiding
it here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from cdpr.core.frames import Pose, Wrench
from cdpr.geometry.robot import Robot
from cdpr.kinematics.jacobian import structure_matrix
from cdpr.statics.tension import is_wrench_feasible


def is_in_wfw(
    pose: Pose,
    robot: Robot,
    external_wrench: Wrench | ArrayLike,
) -> bool:
    """Single-wrench WFW test for a pose against a fixed external wrench."""
    limits = robot.require_limits()
    W = structure_matrix(pose, robot)
    if isinstance(external_wrench, Wrench):
        w_vec = external_wrench.data[: robot.dof]
    else:
        w_vec = np.asarray(external_wrench, dtype=np.float64).reshape(-1)
        if w_vec.size != robot.dof:
            raise ValueError(
                f"external_wrench length {w_vec.size} does not match robot dof {robot.dof}"
            )
    return is_wrench_feasible(W, w_vec, limits.t_min, limits.t_max)
