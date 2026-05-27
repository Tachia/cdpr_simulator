r"""Receding-horizon model-predictive controller (linear, translational).

A purposely small MPC: predict the platform's centre-of-mass motion N
steps into the future under a constant-mass linearised point-mass model

.. math::

    \mathbf{x} = \begin{bmatrix}\mathbf{p} \\ \mathbf{v}\end{bmatrix},
    \qquad
    \mathbf{x}_{k+1} = \mathbf{A}\,\mathbf{x}_k + \mathbf{B}\,\mathbf{u}_k,

with :math:`\mathbf{A} = \begin{bmatrix} \mathbf{I} & \Delta t\,\mathbf{I} \\
\mathbf{0} & \mathbf{I}\end{bmatrix}` and
:math:`\mathbf{B} = \tfrac{1}{m}\begin{bmatrix} \tfrac{\Delta t^2}{2}\,\mathbf{I} \\
\Delta t\,\mathbf{I}\end{bmatrix}`. The optimiser is
:func:`scipy.optimize.minimize` (BFGS, no constraints) --- cable bounds
are enforced downstream by the tension-distribution QP, the same way
PD and CT controllers leave them to the static solver.

The orientation channel is left to the feedback term (the cdpr core's
tension-distribution call still satisfies the angular wrench), so the
controller returns a force-only contribution on top of which a small
PD torque is added. This is enough to demonstrate MPC tracking at
dissertation-grade quality on translational tasks; replacing the
solver with CVXPY / qpOASES is one swap away when a constrained
formulation is wanted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from cdpr.control.base import as_gain_matrix, orientation_error

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.core.frames import Pose, Twist, Wrench
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot


@dataclass(slots=True)
class MPCController:
    """Linear receding-horizon MPC over translational dynamics."""

    horizon: int = 8
    dt: float = 5e-3
    Q_pos: float | NDArray[np.float64] = 1.0e3
    Q_vel: float | NDArray[np.float64] = 1.0e1
    R_force: float | NDArray[np.float64] = 1.0e-2
    P_terminal: float | NDArray[np.float64] = 1.0e4
    # Orientation feedback: a plain PD on top of the translational MPC.
    Kp_rot: float | NDArray[np.float64] = 100.0
    Kd_rot: float | NDArray[np.float64] = 10.0
    gravity_compensation: bool = True
    cancel_external: bool = True

    _last_solution: NDArray[np.float64] = field(default_factory=lambda: np.zeros((1, 3)), repr=False)

    def __call__(
        self,
        *,
        state: "PlatformState",
        reference_pose: "Pose",
        reference_twist: "Twist",
        reference_accel: "tuple[NDArray[np.float64], NDArray[np.float64]] | None",
        t: float,
        robot: "Robot",
        gravity: NDArray[np.float64],
        external: "Wrench",
    ) -> "Wrench":
        from cdpr.core.frames import Wrench

        inertia = robot.require_inertia()
        m = float(inertia.mass)
        dt = self.dt
        N = self.horizon

        # Discrete-time dynamics for the 6-D translational state per axis.
        # Decoupled axes -> solve one 1-D MPC per axis and stack the result.
        Q_pos = float(np.atleast_1d(self.Q_pos).mean())
        Q_vel = float(np.atleast_1d(self.Q_vel).mean())
        R_u = float(np.atleast_1d(self.R_force).mean())
        P_term = float(np.atleast_1d(self.P_terminal).mean())

        # Per-axis solve.
        u_seq_axes: list[NDArray[np.float64]] = []
        for axis in range(3):
            x0 = np.array([
                state.pose.position[axis],
                state.velocity.linear[axis],
            ])
            # Reference along the prediction horizon: constant-velocity hold of
            # (reference_pose, reference_twist) starting at t.
            p_ref0 = reference_pose.position[axis]
            v_ref = reference_twist.linear[axis]
            ks = np.arange(1, N + 1)
            x_ref_traj = np.column_stack([p_ref0 + v_ref * dt * ks,
                                          np.full(N, v_ref)])
            u_init = self._last_solution[:, axis] if self._last_solution.shape[0] == N else np.zeros(N)
            u_opt = _solve_axis_mpc(
                x0=x0, x_ref=x_ref_traj,
                u_init=u_init,
                dt=dt, mass=m,
                Q_pos=Q_pos, Q_vel=Q_vel, R_u=R_u, P_term=P_term,
            )
            u_seq_axes.append(u_opt)
        u_seq = np.column_stack(u_seq_axes)
        self._last_solution = u_seq

        # Apply the first control action.
        F = u_seq[0]                                        # (3,) force
        if self.gravity_compensation:
            F = F - m * gravity                             # cable opposes gravity
        if self.cancel_external:
            F = F - external.force

        # Orientation PD on top.
        Kp_R = as_gain_matrix(self.Kp_rot)
        Kd_R = as_gain_matrix(self.Kd_rot)
        e_o = orientation_error(reference_pose, state.pose)
        e_w = reference_twist.angular - state.velocity.angular
        Tau = Kp_R @ e_o + Kd_R @ e_w
        if self.cancel_external:
            Tau = Tau - external.torque

        return Wrench.from_parts(F, Tau)


# ---------------------------------------------------------------------------
# 1-D MPC solver (per axis)
# ---------------------------------------------------------------------------

def _solve_axis_mpc(
    x0: NDArray[np.float64],
    x_ref: NDArray[np.float64],
    u_init: NDArray[np.float64],
    *,
    dt: float, mass: float,
    Q_pos: float, Q_vel: float, R_u: float, P_term: float,
) -> NDArray[np.float64]:
    """One-dimensional unconstrained MPC; returns the optimal control sequence."""
    N = x_ref.shape[0]

    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([0.5 * dt * dt / mass, dt / mass])

    def predict(u_seq: NDArray[np.float64]) -> NDArray[np.float64]:
        x = x0.copy()
        traj = np.empty((N, 2))
        for k in range(N):
            x = A @ x + B * u_seq[k]
            traj[k] = x
        return traj

    def cost(u_seq: NDArray[np.float64]) -> float:
        x_traj = predict(u_seq)
        err = x_traj - x_ref
        stage = Q_pos * err[:-1, 0] ** 2 + Q_vel * err[:-1, 1] ** 2
        terminal = P_term * (err[-1, 0] ** 2) + Q_vel * (err[-1, 1] ** 2)
        return float(stage.sum() + terminal + R_u * (u_seq ** 2).sum())

    result = minimize(cost, u_init, method="BFGS",
                      options={"gtol": 1e-6, "maxiter": 50})
    return np.asarray(result.x, dtype=np.float64)
