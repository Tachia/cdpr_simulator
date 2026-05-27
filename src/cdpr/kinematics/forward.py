r"""Forward kinematics: cable lengths :math:`\to` platform pose.

The spatial CDPR forward-kinematics problem has no closed form: given
:math:`m` measured cable lengths :math:`L_i^\star` we seek a pose
:math:`(\mathbf{p}, \mathbf{R})` satisfying

.. math::

    f_i(\mathbf{p}, \boldsymbol{\theta}) \;=\;
    \lVert \mathbf{a}_i - \mathbf{p} - \mathbf{R}(\boldsymbol{\theta})\,\mathbf{b}_i \rVert
      - L_i^\star
    \;=\; 0, \qquad i = 1, \dots, m .

For a fully constrained / redundantly actuated robot (:math:`m \geq n+1`) this
is an overdetermined system; we minimise the squared residual using a
trust-region Levenberg--Marquardt step. The orientation is parameterised by
its rotation vector :math:`\boldsymbol{\theta} \in \mathbb{R}^3` (axis times
angle), which is minimal and avoids the unit-quaternion constraint at the
cost of a coordinate singularity at :math:`\lVert \boldsymbol{\theta} \rVert = \pi`;
in the trust-region setting this is harmless provided the initial guess is
within a few radians of the solution, which is the regime forward kinematics
is typically called in (tracking, controller inner loops).

Because FK is locally but not globally unique for many CDPR geometries, an
initial guess is required. Callers tracking a trajectory should pass the
previous step's pose; callers solving cold should use a geometrically
sensible seed (e.g. the centroid of cable anchors with identity rotation).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose
from cdpr.geometry.robot import Robot, RobotGeometry
from cdpr.kinematics.inverse import _geometry, cable_lengths


def forward_kinematics(
    target_lengths: ArrayLike,
    robot: Robot | RobotGeometry,
    initial_guess: Pose,
    *,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    max_iter: int = 100,
    return_diagnostics: bool = False,
) -> Pose | tuple[Pose, dict[str, float]]:
    r"""Solve for the platform pose given measured cable lengths.

    Parameters
    ----------
    target_lengths:
        Measured cable lengths :math:`L_i^\star`, shape ``(m,)``.
    robot:
        Robot or geometry; ``robot.dof`` controls the parameter count
        (3 for translational, 6 for spatial).
    initial_guess:
        Seed pose. For tracking applications, the previous solved pose.
    ftol, xtol, max_iter:
        Forwarded to ``scipy.optimize.least_squares`` (Levenberg--Marquardt).
    return_diagnostics:
        If true, also return a dict with residual norm, iteration count,
        condition number of the final Jacobian, and convergence status.

    Returns
    -------
    pose:
        Solved :class:`Pose`. If diagnostics are requested, a ``(pose, dict)``
        tuple instead.

    Notes
    -----
    SciPy's ``least_squares`` with ``method='lm'`` uses MINPACK's LMDIF, which
    requires the residual count :math:`m` to be at least the parameter count
    (3 or 6). This is automatically satisfied by the
    :class:`RobotGeometry` invariant :math:`m \geq n + 1`.
    """
    g = _geometry(robot)
    L_star = np.asarray(target_lengths, dtype=np.float64)
    if L_star.shape != (g.n_cables,):
        raise ValueError(f"target_lengths must have shape ({g.n_cables},); got {L_star.shape}")

    if g.dof == 3:
        x0 = np.asarray(initial_guess.position, dtype=np.float64).copy()

        def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
            pose = Pose(position=x, rotation=Rotation.identity())
            return cable_lengths(pose, g) - L_star

    elif g.dof == 6:
        p0 = np.asarray(initial_guess.position, dtype=np.float64)
        theta0 = initial_guess.rotation.as_rotvec()
        x0 = np.concatenate([p0, theta0])

        def residual(x: NDArray[np.float64]) -> NDArray[np.float64]:
            pose = Pose(position=x[:3], rotation=Rotation.from_rotvec(x[3:]))
            return cable_lengths(pose, g) - L_star

    else:
        raise NotImplementedError(
            f"Forward kinematics for dof={g.dof} is not implemented; "
            "currently only 3-DOF translational and 6-DOF spatial CDPRs are supported."
        )

    result = least_squares(
        residual,
        x0,
        method="lm",
        ftol=ftol,
        xtol=xtol,
        max_nfev=max_iter * (len(x0) + 1),
    )

    if g.dof == 3:
        solved = Pose(position=result.x, rotation=Rotation.identity())
    else:
        solved = Pose(position=result.x[:3], rotation=Rotation.from_rotvec(result.x[3:]))

    if return_diagnostics:
        # Final Jacobian conditioning for diagnostics.
        J = result.jac
        s = np.linalg.svd(J, compute_uv=False)
        cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
        return solved, {
            "residual_norm": float(np.linalg.norm(result.fun)),
            "n_iter": int(result.nfev),
            "jacobian_condition": cond,
            "status": int(result.status),
            "message": result.message,
        }
    return solved
