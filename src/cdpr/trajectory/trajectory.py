r"""Compose a geometric :class:`Path` with a :class:`TimeScaling`.

The result is a callable ``Trajectory(t) -> Pose`` plus matching ``twist`` and
``acceleration`` methods, all related by the chain rule:

.. math::

    \dot{\mathbf{p}}(t) = \frac{d\mathbf{p}}{ds}\,\dot s(t),
    \qquad
    \ddot{\mathbf{p}}(t) = \frac{d^2\mathbf{p}}{ds^2}\,\dot s(t)^2
                          + \frac{d\mathbf{p}}{ds}\,\ddot s(t).

A trajectory built this way can be handed directly to
:func:`cdpr.dynamics.simulator.simulate` as its ``reference_pose`` argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cdpr.core.frames import Pose, Twist
from cdpr.trajectory.paths import Path
from cdpr.trajectory.scaling import TimeScaling


@dataclass(slots=True, frozen=True)
class Trajectory:
    """A path + a time scaling, evaluated at any time ``t``."""

    path: Path
    scaling: TimeScaling

    @property
    def duration(self) -> float:
        return self.scaling.duration

    def pose(self, t: float) -> Pose:
        s = float(self.scaling.s(t))
        return self.path.pose(s)

    def __call__(self, t: float) -> Pose:
        return self.pose(t)

    def twist(self, t: float) -> Twist:
        s = float(self.scaling.s(t))
        sd = float(self.scaling.s_dot(t))
        dp_ds, omega_ds = self.path.velocity(s)
        return Twist.from_parts(dp_ds * sd, omega_ds * sd)

    def acceleration(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = float(self.scaling.s(t))
        sd = float(self.scaling.s_dot(t))
        sdd = float(self.scaling.s_ddot(t))
        dp_ds, omega_ds = self.path.velocity(s)
        ddp_ds, _ = self.path.acceleration(s)
        a_lin = ddp_ds * sd * sd + dp_ds * sdd
        a_ang = omega_ds * sdd  # angular acceleration if path's angular velocity is constant in s
        return a_lin, a_ang
