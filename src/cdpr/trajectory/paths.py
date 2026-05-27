r"""Geometric paths :math:`s \in [0, 1] \mapsto \mathrm{SE}(3)`.

Each path implements ``pose(s)``, ``velocity(s)``, ``acceleration(s)`` where
the derivatives are taken with respect to the path parameter :math:`s`. The
:class:`~cdpr.trajectory.trajectory.Trajectory` composer combines these with
a :class:`~cdpr.trajectory.scaling.TimeScaling` via the chain rule so that
controllers receive proper time derivatives.

The default orientation policy on translational paths (line, circle,
Lissajous) is *identity* --- if the path itself does not impose an
orientation, the platform points the same way at every :math:`s`. Override
by composing with an explicit orientation path or by setting
``end_rotation`` on the line path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation, Slerp

from cdpr.core.frames import Pose


@runtime_checkable
class Path(Protocol):
    """Protocol every geometric path implements."""

    def pose(self, s: float) -> Pose: ...
    def velocity(self, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def acceleration(self, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...


# ---------------------------------------------------------------------------
# Line in position, SLERP in orientation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LinearPath:
    """Linear interpolation in position; SLERP between two end rotations.

    The platform translates along :math:`\\mathbf{p}(s) = (1 - s)\\,\\mathbf{p}_0 + s\\,\\mathbf{p}_1`
    while orientation slerps between ``start_rotation`` and ``end_rotation``.
    Translational velocity in the path parameter is constant; orientation
    velocity is too, on the unit-quaternion great-circle.
    """

    start: ArrayLike
    end: ArrayLike
    start_rotation: Rotation = field(default_factory=Rotation.identity)
    end_rotation: Rotation = field(default_factory=Rotation.identity)
    _slerp: Slerp = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=np.float64).reshape(3)
        self.end = np.asarray(self.end, dtype=np.float64).reshape(3)
        rots = Rotation.concatenate([self.start_rotation, self.end_rotation])
        self._slerp = Slerp([0.0, 1.0], rots)

    def pose(self, s: float) -> Pose:
        s = float(np.clip(s, 0.0, 1.0))
        return Pose(position=(1 - s) * self.start + s * self.end, rotation=self._slerp(s))

    def velocity(self, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Constant in s for both translation and (great-circle) rotation.
        dp = self.end - self.start
        # Angular velocity is the rotation vector from start to end (constant great-circle rate).
        relative = self.end_rotation * self.start_rotation.inv()
        omega = relative.as_rotvec()
        return dp, omega

    def acceleration(self, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return np.zeros(3), np.zeros(3)


# ---------------------------------------------------------------------------
# Circle in a plane
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CircularPath:
    r"""Circular path in the plane through ``center`` orthogonal to ``axis``.

    Parameterised so :math:`s = 0` and :math:`s = 1` correspond to the start
    and end angular positions. For a full revolution set
    ``angle_span = 2*pi``.
    """

    center: ArrayLike
    radius: float
    axis: ArrayLike = (0.0, 0.0, 1.0)
    angle_start: float = 0.0
    angle_span: float = 2 * np.pi
    rotation: Rotation = field(default_factory=Rotation.identity)
    _u: NDArray[np.float64] = field(init=False, repr=False)
    _v: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        n = np.asarray(self.axis, dtype=np.float64).reshape(3)
        n = n / np.linalg.norm(n)
        # In-plane basis: choose any vector not parallel to n, project out, normalise.
        tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = tmp - np.dot(tmp, n) * n
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)
        self._u, self._v = u, v

    def pose(self, s: float) -> Pose:
        theta = self.angle_start + s * self.angle_span
        p = self.center + self.radius * (np.cos(theta) * self._u + np.sin(theta) * self._v)
        return Pose(position=p, rotation=self.rotation)

    def velocity(self, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        theta = self.angle_start + s * self.angle_span
        dp_dtheta = self.radius * (-np.sin(theta) * self._u + np.cos(theta) * self._v)
        return dp_dtheta * self.angle_span, np.zeros(3)

    def acceleration(self, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        theta = self.angle_start + s * self.angle_span
        d2p_dtheta2 = -self.radius * (np.cos(theta) * self._u + np.sin(theta) * self._v)
        return d2p_dtheta2 * self.angle_span**2, np.zeros(3)


# ---------------------------------------------------------------------------
# Lissajous figure
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LissajousPath:
    r"""3D Lissajous figure with per-axis amplitudes, frequencies and phases.

    .. math::

        \mathbf{p}(s) = \mathbf{c}
            + \begin{bmatrix}
                A_x \sin(2\pi f_x s + \varphi_x) \\
                A_y \sin(2\pi f_y s + \varphi_y) \\
                A_z \sin(2\pi f_z s + \varphi_z)
              \end{bmatrix}

    A common CDPR benchmark trajectory; published results often use
    :math:`(f_x, f_y, f_z) = (1, 2, 0)` or :math:`(3, 2, 1)` for spatial
    versions.
    """

    center: ArrayLike
    amplitudes: ArrayLike = (1.0, 1.0, 0.0)
    frequencies: ArrayLike = (1.0, 2.0, 0.0)
    phases: ArrayLike = (0.0, np.pi / 2, 0.0)
    rotation: Rotation = field(default_factory=Rotation.identity)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.amplitudes = np.asarray(self.amplitudes, dtype=np.float64).reshape(3)
        self.frequencies = np.asarray(self.frequencies, dtype=np.float64).reshape(3)
        self.phases = np.asarray(self.phases, dtype=np.float64).reshape(3)

    def _phase_args(self, s: float) -> NDArray[np.float64]:
        return 2 * np.pi * self.frequencies * s + self.phases

    def pose(self, s: float) -> Pose:
        phi = self._phase_args(s)
        return Pose(
            position=self.center + self.amplitudes * np.sin(phi),
            rotation=self.rotation,
        )

    def velocity(self, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        phi = self._phase_args(s)
        omega = 2 * np.pi * self.frequencies
        return self.amplitudes * omega * np.cos(phi), np.zeros(3)

    def acceleration(self, s: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        phi = self._phase_args(s)
        omega = 2 * np.pi * self.frequencies
        return -self.amplitudes * omega * omega * np.sin(phi), np.zeros(3)
