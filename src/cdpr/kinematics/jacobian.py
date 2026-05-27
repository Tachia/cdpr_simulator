r"""Structure matrix and Jacobian analysis.

For a spatial 6-DOF CDPR with :math:`m` cables, the *structure matrix*
:math:`\mathbf{W} \in \mathbb{R}^{6 \times m}` collects the wrenches that unit
cable tensions exert on the platform. Column :math:`i` is

.. math::

    \mathbf{w}_i \;=\; \begin{bmatrix} \hat{\mathbf{u}}_i \\
        (\mathbf{R}\,\mathbf{b}_i) \times \hat{\mathbf{u}}_i
        \end{bmatrix},

so the platform wrench produced by tension vector :math:`\mathbf{t}` is
:math:`\mathbf{w}_p = \mathbf{W}\,\mathbf{t}`, and static equilibrium against
an external wrench :math:`\mathbf{w}_\text{ext}` requires

.. math::

    \mathbf{W}\,\mathbf{t} = -\,\mathbf{w}_\text{ext}, \qquad
    \mathbf{t} \in [\mathbf{t}_\text{min}, \mathbf{t}_\text{max}].

For planar / translational CDPRs (``dof < 6``), the structure matrix is the
corresponding row subset. This module returns the full :math:`6 \times m`
form for ``dof == 6`` and the leading ``dof`` rows otherwise; the convention
matches how the statics layer assembles the equilibrium system.

The relation to the velocity Jacobian is :math:`\mathbf{J} = -\mathbf{W}^\top`
(see Pott 2018, §5.2). We expose only :math:`\mathbf{W}` here because every
downstream consumer in this framework (tension distribution, workspace
analysis, force-closure) is wrench-side.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cdpr.core.exceptions import SingularConfiguration
from cdpr.core.frames import Pose
from cdpr.geometry.robot import Robot, RobotGeometry
from cdpr.kinematics.inverse import cable_unit_vectors, _geometry


def structure_matrix(pose: Pose, robot: Robot | RobotGeometry) -> NDArray[np.float64]:
    r"""Compute :math:`\mathbf{W}(\mathbf{q}) \in \mathbb{R}^{\text{dof} \times m}` for a single pose.

    See module docstring for the convention. For ``dof < 6`` the leading
    ``dof`` rows of the spatial matrix are returned --- consistent with the
    statics layer, which truncates external wrenches to the same DOF set.
    """
    if pose.is_batched:
        raise ValueError(
            "structure_matrix expects a single pose; use structure_matrix_batch for a stack."
        )

    g = _geometry(robot)
    U, _ = cable_unit_vectors(pose, g)               # (m, 3)
    Rb = pose.rotation.apply(g.attachments)          # (m, 3) -- R b_i in world frame
    moments = np.cross(Rb, U)                        # (m, 3)
    W = np.concatenate([U.T, moments.T], axis=0)     # (6, m)
    return W if g.dof == 6 else W[: g.dof, :]


def structure_matrix_batch(
    poses: Pose, robot: Robot | RobotGeometry
) -> NDArray[np.float64]:
    """Vectorised :func:`structure_matrix` over a batch of poses.

    Returns shape ``(N, dof, m)`` where ``N`` is the leading axis of
    ``poses.position``. Use this when sweeping a trajectory or a workspace
    grid; the per-pose loop is significantly slower for large ``N``.
    """
    if not poses.is_batched:
        return structure_matrix(poses, robot)[np.newaxis]

    g = _geometry(robot)
    U, _ = cable_unit_vectors(poses, g)                       # (N, m, 3)
    Rb = poses.rotation.apply(g.attachments)                  # (N, m, 3)
    moments = np.cross(Rb, U, axis=-1)                        # (N, m, 3)
    W_full = np.concatenate(
        [np.swapaxes(U, -1, -2), np.swapaxes(moments, -1, -2)], axis=-2
    )                                                          # (N, 6, m)
    return W_full if g.dof == 6 else W_full[..., : g.dof, :]


def condition_number(
    pose: Pose,
    robot: Robot | RobotGeometry,
    *,
    raise_if_singular: bool = False,
    threshold: float = 1e8,
) -> float:
    r"""2-norm condition number :math:`\kappa_2(\mathbf{W}^\top\mathbf{W})^{1/2}`.

    Large values flag a near-singular configuration, where small wrench
    perturbations require large tension changes; the standard CDPR
    interpretation is loss of force closure or proximity to a workspace
    boundary. When ``raise_if_singular`` is true and the condition number
    exceeds ``threshold``, raises :class:`SingularConfiguration` with the
    measured value attached.
    """
    W = structure_matrix(pose, robot)
    s = np.linalg.svd(W, compute_uv=False)
    if s[-1] <= 0:
        cond = float("inf")
    else:
        cond = float(s[0] / s[-1])
    if raise_if_singular and cond > threshold:
        raise SingularConfiguration(
            f"Structure matrix is near-singular (cond={cond:.3e} > {threshold:.0e})",
            condition_number=cond,
        )
    return cond
