r"""Pose-regulating PD controller for the CDPR platform.

The control law produces the cable wrench

.. math::

    \mathbf{F}_\text{cable} \;=\; \mathbf{K}_p\,(\mathbf{p}_\text{ref} - \mathbf{p})
        \;+\; \mathbf{K}_d\,(\mathbf{v}_\text{ref} - \mathbf{v})
        \;+\; \underbrace{(-m\,\mathbf{g})}_{\text{gravity comp}}
        \;-\; \mathbf{F}_\text{external},

.. math::

    \boldsymbol\tau_\text{cable} \;=\; \mathbf{K}_p^\text{rot}\,
            \log_{SO(3)}(\mathbf{R}_\text{ref}\,\mathbf{R}^{\top})
        \;+\; \mathbf{K}_d^\text{rot}\,(\boldsymbol\omega_\text{ref} - \boldsymbol\omega)
        \;-\; \boldsymbol\tau_\text{external}.

Gain matrices are accepted as scalars (treated as ``g * I``), 3-vectors
(treated as the diagonal), or full ``3 x 3`` matrices. The default-constructed
controller is *not* tuned for any particular robot --- gains must be set
explicitly. Sensible starting points for the IPAnema-class are
``Kp_pos = 2_000 N/m``, ``Kd_pos = 200 N s/m``,
``Kp_rot = 100 N m/rad``, ``Kd_rot = 20 N m s/rad``, but a per-robot retune
is expected.
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
class PDController:
    """Pose-regulating PD with optional gravity compensation."""

    Kp_pos: float | NDArray[np.float64] = 1.0
    Kd_pos: float | NDArray[np.float64] = 0.1
    Kp_rot: float | NDArray[np.float64] = 1.0
    Kd_rot: float | NDArray[np.float64] = 0.1
    gravity_compensation: bool = True
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

        e_p = reference_pose.position - state.pose.position
        e_v = reference_twist.linear - state.velocity.linear
        e_o = orientation_error(reference_pose, state.pose)
        e_w = reference_twist.angular - state.velocity.angular

        F = self._Kp_pos @ e_p + self._Kd_pos @ e_v
        Tau = self._Kp_rot @ e_o + self._Kd_rot @ e_w

        if self.gravity_compensation and robot.inertia is not None:
            F = F - robot.inertia.mass * gravity

        if self.cancel_external:
            F = F - external.force
            Tau = Tau - external.torque

        return Wrench.from_parts(F, Tau)
