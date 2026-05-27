r"""Newton--Euler equations of motion for a free rigid body.

The platform state is :math:`\mathbf{x} = (\mathbf{p}, \mathbf{R}, \mathbf{v},
\boldsymbol{\omega})`, with :math:`(\mathbf{p}, \mathbf{R})` the world-frame pose
and :math:`(\mathbf{v}, \boldsymbol{\omega})` the world-frame translational and
rotational velocity of the platform body origin. Given an external wrench
:math:`\mathbf{w} = (\mathbf{f}, \boldsymbol{\tau})` applied at that origin,
the acceleration is

.. math::

    \dot{\mathbf{v}} \;=\; \frac{1}{m}\,\mathbf{f},
    \qquad
    \dot{\boldsymbol{\omega}} \;=\; \mathbf{I}_W^{-1}\bigl(\boldsymbol{\tau}
        \;-\; \boldsymbol{\omega} \times \mathbf{I}_W\,\boldsymbol{\omega}\bigr),

with :math:`\mathbf{I}_W = \mathbf{R}\,\mathbf{I}_B\,\mathbf{R}^\top` the
world-frame inertia tensor. Gravity, if present, is part of
:math:`\mathbf{f}` --- :func:`~cdpr.dynamics.simulator.simulate` adds it for
convenience but the low-level routines treat it as just another external
force.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.geometry.robot import PlatformInertia


@dataclass(slots=True)
class PlatformState:
    """Mutable container for the platform's full kinematic state."""

    pose: Pose
    velocity: Twist

    @classmethod
    def at_rest(cls, pose: Pose) -> "PlatformState":
        return cls(pose=pose, velocity=Twist(np.zeros(6)))


def rigid_body_acceleration(
    state: PlatformState,
    wrench: Wrench,
    inertia: PlatformInertia,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    r"""Return ``(linear_acceleration, angular_acceleration)`` in world frame.

    Implements the Newton--Euler equations directly; no coupling assumptions
    are made about how the wrench was produced (gravity, cables, contact).
    """
    R = state.pose.rotation.as_matrix()
    I_world = R @ inertia.inertia @ R.T
    omega = state.velocity.angular
    f = wrench.force
    tau = wrench.torque

    a_lin = f / inertia.mass
    a_ang = np.linalg.solve(I_world, tau - np.cross(omega, I_world @ omega))
    return a_lin, a_ang
