r"""Computed-torque controller (inverse-dynamics feedback linearisation).

The rigid-body equations of motion for the platform are

.. math::

    m\,\dot{\mathbf{v}} \;=\; \mathbf{F}_\text{cable} + m\,\mathbf{g}
        + \mathbf{F}_\text{ext},
    \qquad
    \mathbf{I}_W\,\dot{\boldsymbol\omega}
        + \boldsymbol\omega \times \mathbf{I}_W\,\boldsymbol\omega
        \;=\; \boldsymbol\tau_\text{cable} + \boldsymbol\tau_\text{ext},

so the cable wrench that produces a desired acceleration
:math:`(\dot{\mathbf{v}}_d, \dot{\boldsymbol\omega}_d)` is

.. math::

    \mathbf{F}_\text{cable} \;=\; m\,\dot{\mathbf{v}}_d - m\,\mathbf{g} - \mathbf{F}_\text{ext},
    \qquad
    \boldsymbol\tau_\text{cable} \;=\; \mathbf{I}_W\,\dot{\boldsymbol\omega}_d
        + \boldsymbol\omega \times \mathbf{I}_W\,\boldsymbol\omega - \boldsymbol\tau_\text{ext}.

The desired acceleration combines a reference feedforward (from the
trajectory) with a PD correction on the pose error:

.. math::

    \dot{\mathbf{v}}_d = \dot{\mathbf{v}}_\text{ref}
        + \mathbf{K}_p\,(\mathbf{p}_\text{ref} - \mathbf{p})
        + \mathbf{K}_d\,(\mathbf{v}_\text{ref} - \mathbf{v}),

and similarly for the angular acceleration with
:math:`\log_{SO(3)}(\mathbf{R}_\text{ref}\mathbf{R}^{\top})` as the
orientation error. Gains here have units of
:math:`s^{-2}` and :math:`s^{-1}` --- distinct from the PD law, which uses
force-per-error gains directly.

Reference acceleration is consumed when supplied (Trajectory provides it
automatically); when ``reference_accel`` is ``None`` the law degenerates
to a feedback-only computed-torque law (no feedforward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cdpr.control.base import as_gain_matrix, orientation_error

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.core.frames import Pose, Twist, Wrench
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot


@dataclass(slots=True)
class ComputedTorqueController:
    """Inverse-dynamics feedback-linearisation control law."""

    Kp_pos: float | NDArray[np.float64] = 100.0
    Kd_pos: float | NDArray[np.float64] = 20.0
    Kp_rot: float | NDArray[np.float64] = 100.0
    Kd_rot: float | NDArray[np.float64] = 20.0
    cancel_external: bool = True

    _Kp_pos: NDArray[np.float64] = field(init=False, repr=False)
    _Kd_pos: NDArray[np.float64] = field(init=False, repr=False)
    _Kp_rot: NDArray[np.float64] = field(init=False, repr=False)
    _Kd_rot: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._Kp_pos = as_gain_matrix(self.Kp_pos)
        self._Kd_pos = as_gain_matrix(self.Kd_pos)
        self._Kp_rot = as_gain_matrix(self.Kp_rot)
        self._Kd_rot = as_gain_matrix(self.Kd_rot)

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
        e_p = reference_pose.position - state.pose.position
        e_v = reference_twist.linear - state.velocity.linear
        e_o = orientation_error(reference_pose, state.pose)
        e_w = reference_twist.angular - state.velocity.angular

        a_ref_lin = reference_accel[0] if reference_accel is not None else np.zeros(3)
        a_ref_ang = reference_accel[1] if reference_accel is not None else np.zeros(3)

        a_des_lin = a_ref_lin + self._Kp_pos @ e_p + self._Kd_pos @ e_v
        a_des_ang = a_ref_ang + self._Kp_rot @ e_o + self._Kd_rot @ e_w

        # Linear: F_cable = m * a_des - m * g  (cable supplies the difference)
        F = inertia.mass * (a_des_lin - gravity)

        # Angular: tau_cable = I_W * alpha_des + omega x I_W omega
        R = state.pose.rotation.as_matrix()
        I_world = R @ inertia.inertia @ R.T
        omega = state.velocity.angular
        Tau = I_world @ a_des_ang + np.cross(omega, I_world @ omega)

        if self.cancel_external:
            F = F - external.force
            Tau = Tau - external.torque

        return Wrench.from_parts(F, Tau)
