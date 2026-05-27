r"""Feedforward + feedback controller composition.

A common dissertation experiment is "show how much an analytic
feedforward improves a pure-PD baseline". Rather than writing the
addition by hand in every script, this module composes two
:class:`Controller` instances: a feedforward (typically an inverse-
dynamics term that depends on the reference acceleration) and a
feedback term (PD, MPC, learned residual) that closes the loop on the
state error.

The composed controller returns

.. math::

    \mathbf{w}_\text{cable} \;=\; \mathbf{w}_\text{cable}^\text{ff}
        \;+\; \mathbf{w}_\text{cable}^\text{fb},

where each contribution is whatever its controller decides. Gravity
compensation lives in whichever controller wants to claim it --- pass
``gravity_compensation=False`` to the feedback controller to avoid
double-counting it when the feedforward already includes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.control.base import Controller
    from cdpr.core.frames import Pose, Twist, Wrench
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot


@dataclass(slots=True)
class FeedforwardPlusFeedback:
    """Wrench-additive composition of two controllers."""

    feedforward: "Controller"
    feedback: "Controller"

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

        # Feedforward sees no external disturbance --- it doesn't have a
        # state-error signal to fight it with. The feedback term handles
        # disturbance rejection through the error.
        zero_ext = Wrench(np.zeros(6))
        w_ff = self.feedforward(
            state=state,
            reference_pose=reference_pose,
            reference_twist=reference_twist,
            reference_accel=reference_accel,
            t=t,
            robot=robot,
            gravity=gravity,
            external=zero_ext,
        )
        w_fb = self.feedback(
            state=state,
            reference_pose=reference_pose,
            reference_twist=reference_twist,
            reference_accel=reference_accel,
            t=t,
            robot=robot,
            gravity=gravity,
            external=external,
        )
        return w_ff + w_fb


# ---------------------------------------------------------------------------
# Pure feedforward inverse-dynamics term (analytic, no state feedback)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class InverseDynamicsFeedforward:
    """Analytic computed-torque term using *only* the reference acceleration.

    Returns the cable wrench needed to drive the rigid body along the
    reference trajectory in the absence of disturbance. Pair with a
    feedback controller via :class:`FeedforwardPlusFeedback` to get a
    standard "FF + FB" tracking law without the feedback term having to
    re-derive the inertia coupling.

    Mathematically::

        F  = m * a_ref_lin - m * g
        Tau = I_W * a_ref_ang + omega x I_W omega
    """

    include_coriolis: bool = True
    cancel_gravity: bool = True

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
        a_lin = reference_accel[0] if reference_accel is not None else np.zeros(3)
        a_ang = reference_accel[1] if reference_accel is not None else np.zeros(3)

        F = inertia.mass * a_lin
        if self.cancel_gravity:
            F = F - inertia.mass * gravity

        R = state.pose.rotation.as_matrix()
        I_world = R @ inertia.inertia @ R.T
        Tau = I_world @ a_ang
        if self.include_coriolis:
            omega = state.velocity.angular
            Tau = Tau + np.cross(omega, I_world @ omega)

        return Wrench.from_parts(F, Tau)
