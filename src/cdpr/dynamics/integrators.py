r"""Time-step integrators for the rigid-body platform state.

Two integrators are provided:

* :func:`semi_implicit_step` -- a symplectic (semi-implicit) Euler step.
  Velocities advance first using the current acceleration, then positions
  advance using the *new* velocities. First-order accurate, but the
  symplectic structure gives it dramatically better energy behaviour than
  explicit Euler at the same step size, and it is the standard choice for
  long-duration rigid-body simulations.

* :func:`rk4_step` -- classical fourth-order Runge--Kutta. Four wrench
  evaluations per step. Higher accuracy when the wrench is smooth in time;
  use this for verification studies against analytic solutions and for
  short trajectories where accuracy beats throughput.

Both integrators update the rotation via the matrix exponential of
:math:`\boldsymbol{\omega}\,\Delta t`, implemented through
``Rotation.from_rotvec``. This preserves :math:`\mathbf{R} \in SO(3)`
exactly --- no quaternion renormalisation drift accumulates.

The integrators consume a ``wrench_fn(state, t) -> Wrench`` callable rather
than a fixed wrench. This decouples the dynamics from any specific control
policy and lets the same loop drive an open-loop CDPR, a PD controller, a
PPO policy, or a PINN-augmented inverse-dynamics scheme.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.dynamics.rigid_body import PlatformState, rigid_body_acceleration
from cdpr.geometry.robot import PlatformInertia


WrenchFn = Callable[[PlatformState, float], Wrench]


class IntegratorStep(Protocol):
    """Type protocol for the step functions exposed in this module."""

    def __call__(
        self,
        state: PlatformState,
        t: float,
        dt: float,
        wrench_fn: WrenchFn,
        inertia: PlatformInertia,
    ) -> PlatformState: ...


def _advance_pose(pose: Pose, v: np.ndarray, omega: np.ndarray, dt: float) -> Pose:
    """Pose update :math:`(\\mathbf{p} + \\mathbf{v}\\Delta t,\\, \\exp([\\boldsymbol\\omega]\\Delta t)\\,\\mathbf{R})`."""
    new_position = pose.position + v * dt
    rotvec = omega * dt
    new_rotation = Rotation.from_rotvec(rotvec) * pose.rotation
    return Pose(position=new_position, rotation=new_rotation)


def semi_implicit_step(
    state: PlatformState,
    t: float,
    dt: float,
    wrench_fn: WrenchFn,
    inertia: PlatformInertia,
) -> PlatformState:
    """Symplectic Euler step. One wrench evaluation per step."""
    w = wrench_fn(state, t)
    a_lin, a_ang = rigid_body_acceleration(state, w, inertia)

    v_new = state.velocity.linear + a_lin * dt
    omega_new = state.velocity.angular + a_ang * dt

    # Positions advance using the new velocities (this is what makes it semi-implicit).
    pose_new = _advance_pose(state.pose, v_new, omega_new, dt)
    return PlatformState(pose=pose_new, velocity=Twist.from_parts(v_new, omega_new))


def rk4_step(
    state: PlatformState,
    t: float,
    dt: float,
    wrench_fn: WrenchFn,
    inertia: PlatformInertia,
) -> PlatformState:
    """Classical 4th-order Runge--Kutta step. Four wrench evaluations per step.

    The rotation increment is integrated through the rotation vector
    :math:`\\boldsymbol{\\omega}\\,\\Delta t`. This is exact for constant
    :math:`\\boldsymbol{\\omega}` and second-order accurate when
    :math:`\\boldsymbol{\\omega}` varies, which combined with the RK4 step for
    the velocity gives overall fourth-order behaviour for the translational
    state and effectively fourth-order behaviour for the rotational state
    provided :math:`\\boldsymbol{\\omega}` does not change direction wildly
    within a step.
    """
    def derivatives(s: PlatformState, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        w = wrench_fn(s, time)
        a_lin, a_ang = rigid_body_acceleration(s, w, inertia)
        return s.velocity.linear, s.velocity.angular, a_lin, a_ang

    # Stage 1
    v1, omega1, a1, alpha1 = derivatives(state, t)

    # Stage 2: state advanced by dt/2 with stage-1 derivatives
    s2 = PlatformState(
        pose=_advance_pose(state.pose, v1, omega1, dt / 2),
        velocity=Twist.from_parts(
            state.velocity.linear + a1 * dt / 2,
            state.velocity.angular + alpha1 * dt / 2,
        ),
    )
    v2, omega2, a2, alpha2 = derivatives(s2, t + dt / 2)

    # Stage 3: dt/2 with stage-2 derivatives
    s3 = PlatformState(
        pose=_advance_pose(state.pose, v2, omega2, dt / 2),
        velocity=Twist.from_parts(
            state.velocity.linear + a2 * dt / 2,
            state.velocity.angular + alpha2 * dt / 2,
        ),
    )
    v3, omega3, a3, alpha3 = derivatives(s3, t + dt / 2)

    # Stage 4: dt with stage-3 derivatives
    s4 = PlatformState(
        pose=_advance_pose(state.pose, v3, omega3, dt),
        velocity=Twist.from_parts(
            state.velocity.linear + a3 * dt,
            state.velocity.angular + alpha3 * dt,
        ),
    )
    v4, omega4, a4, alpha4 = derivatives(s4, t + dt)

    v_avg = (v1 + 2 * v2 + 2 * v3 + v4) / 6
    omega_avg = (omega1 + 2 * omega2 + 2 * omega3 + omega4) / 6
    a_avg = (a1 + 2 * a2 + 2 * a3 + a4) / 6
    alpha_avg = (alpha1 + 2 * alpha2 + 2 * alpha3 + alpha4) / 6

    pose_new = _advance_pose(state.pose, v_avg, omega_avg, dt)
    velocity_new = Twist.from_parts(
        state.velocity.linear + a_avg * dt,
        state.velocity.angular + alpha_avg * dt,
    )
    return PlatformState(pose=pose_new, velocity=velocity_new)
