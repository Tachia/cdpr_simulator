r"""Rigid-body kinematic primitives: SE(3) poses, twists, and wrenches.

Conventions
-----------

* Right-handed coordinate frames throughout.
* A pose :math:`(\mathbf{p}, \mathbf{R}) \in SE(3)` maps a point expressed in
  the body frame to the world frame via :math:`\mathbf{x}_W = \mathbf{R}\,\mathbf{x}_B + \mathbf{p}`.
* Twists and wrenches use the **linear-first** stacking
  :math:`\boldsymbol{\xi} = (\mathbf{v}^\top, \boldsymbol{\omega}^\top)^\top`
  matching Pott, *Cable-Driven Parallel Robots* (Springer, 2018, Ch. 5).
* Quaternions follow SciPy's scalar-last :math:`(x, y, z, w)` convention; the
  ``Pose`` helpers accept and return that form when working with quaternion
  representations explicitly.

The objects defined here are deliberately small. They wrap SciPy's
``Rotation`` (which is well-tested, vectorised, and avoids us reimplementing
quaternion normalisation) and add the few operations that the rest of the
framework actually needs: composition, inversion, point/vector transformation,
adjoint transport of twists and wrenches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# so(3) <-> R^3 helpers
# ---------------------------------------------------------------------------

def hat(omega: ArrayLike) -> NDArray[np.float64]:
    r"""Skew-symmetric matrix :math:`[\boldsymbol{\omega}]_\times` of a 3-vector.

    Supports a single 3-vector ``(3,)`` or a stack ``(..., 3)``; the trailing
    two axes of the result are the :math:`3\times 3` skew matrices.
    """
    w = np.asarray(omega, dtype=np.float64)
    if w.shape[-1] != 3:
        raise ValueError(f"hat() expects last axis of size 3, got shape {w.shape}")
    zero = np.zeros(w.shape[:-1], dtype=np.float64)
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    return np.stack(
        [
            np.stack([zero, -wz, wy], axis=-1),
            np.stack([wz, zero, -wx], axis=-1),
            np.stack([-wy, wx, zero], axis=-1),
        ],
        axis=-2,
    )


def vee(skew: ArrayLike) -> NDArray[np.float64]:
    r"""Inverse of :func:`hat`: extract the 3-vector from a skew matrix.

    No symmetry check is performed; the antisymmetric part is read directly so
    callers can pass slightly non-skew matrices produced by finite-difference
    code without an exception.
    """
    S = np.asarray(skew, dtype=np.float64)
    if S.shape[-2:] != (3, 3):
        raise ValueError(f"vee() expects last two axes (3, 3), got shape {S.shape}")
    return np.stack([S[..., 2, 1], S[..., 0, 2], S[..., 1, 0]], axis=-1)


# ---------------------------------------------------------------------------
# SE(3) pose
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Pose:
    r"""Rigid-body pose :math:`(\mathbf{p}, \mathbf{R}) \in SE(3)`.

    The translation lives in ``position`` (shape ``(3,)`` or a leading batch
    ``(..., 3)``) and the orientation in ``rotation`` (a SciPy ``Rotation``
    which natively handles batching). The two must broadcast compatibly.
    """

    position: NDArray[np.float64]
    rotation: Rotation = field(default_factory=lambda: Rotation.identity())

    # --- construction ----------------------------------------------------

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=np.float64)
        if self.position.shape[-1] != 3:
            raise ValueError(
                f"Pose.position last axis must be 3, got shape {self.position.shape}"
            )

    @classmethod
    def identity(cls) -> Self:
        return cls(position=np.zeros(3), rotation=Rotation.identity())

    @classmethod
    def from_matrix(cls, T: ArrayLike) -> Self:
        T = np.asarray(T, dtype=np.float64)
        if T.shape[-2:] != (4, 4):
            raise ValueError(f"Expected (..., 4, 4) homogeneous matrix, got {T.shape}")
        return cls(position=T[..., :3, 3], rotation=Rotation.from_matrix(T[..., :3, :3]))

    @classmethod
    def from_quaternion(cls, position: ArrayLike, quat_xyzw: ArrayLike) -> Self:
        return cls(position=np.asarray(position), rotation=Rotation.from_quat(quat_xyzw))

    @classmethod
    def from_euler(
        cls, position: ArrayLike, angles: ArrayLike, seq: str = "xyz", degrees: bool = False
    ) -> Self:
        return cls(
            position=np.asarray(position),
            rotation=Rotation.from_euler(seq, angles, degrees=degrees),
        )

    @classmethod
    def from_axis_angle(cls, position: ArrayLike, rotvec: ArrayLike) -> Self:
        return cls(position=np.asarray(position), rotation=Rotation.from_rotvec(rotvec))

    # --- inspection ------------------------------------------------------

    @property
    def matrix(self) -> NDArray[np.float64]:
        R = self.rotation.as_matrix()
        T = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
        T[..., :3, :3] = R
        T[..., :3, 3] = self.position
        T[..., 3, 3] = 1.0
        return T

    @property
    def quaternion_xyzw(self) -> NDArray[np.float64]:
        return self.rotation.as_quat()

    @property
    def is_batched(self) -> bool:
        return self.position.ndim > 1

    # --- algebra ---------------------------------------------------------

    def inverse(self) -> Pose:
        Rinv = self.rotation.inv()
        return Pose(position=-Rinv.apply(self.position), rotation=Rinv)

    def compose(self, other: Pose) -> Pose:
        r"""Group product :math:`T = T_\text{self} \cdot T_\text{other}`."""
        return Pose(
            position=self.position + self.rotation.apply(other.position),
            rotation=self.rotation * other.rotation,
        )

    def __matmul__(self, other: Pose) -> Pose:
        return self.compose(other)

    # --- transforming points / vectors / spatial quantities --------------

    @overload
    def transform_point(self, point: ArrayLike) -> NDArray[np.float64]: ...
    @overload
    def transform_point(self, point: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def transform_point(self, point: ArrayLike) -> NDArray[np.float64]:
        r""":math:`\mathbf{x}_W = \mathbf{R}\,\mathbf{x}_B + \mathbf{p}`."""
        p = np.asarray(point, dtype=np.float64)
        return self.rotation.apply(p) + self.position

    def transform_vector(self, vec: ArrayLike) -> NDArray[np.float64]:
        r"""Rotate a free vector (no translation): :math:`\mathbf{v}_W = \mathbf{R}\,\mathbf{v}_B`."""
        return self.rotation.apply(np.asarray(vec, dtype=np.float64))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        rv = self.rotation.as_rotvec()
        return f"Pose(p={np.asarray(self.position).round(4).tolist()}, rotvec={np.round(rv, 4).tolist()})"


# ---------------------------------------------------------------------------
# Spatial 6-vectors (twist, wrench)
# ---------------------------------------------------------------------------

def _check_six(vec: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    if vec.shape[-1] != 6:
        raise ValueError(f"{name} expects last axis of size 6, got shape {vec.shape}")
    return vec


@dataclass(slots=True, frozen=True)
class Twist:
    r"""Spatial velocity :math:`\boldsymbol{\xi} = (\mathbf{v}^\top, \boldsymbol{\omega}^\top)^\top \in \mathbb{R}^6`.

    The default stacking is *linear-first*. Use :attr:`linear` and
    :attr:`angular` to access the parts without remembering the order.
    """

    data: NDArray[np.float64]

    def __post_init__(self) -> None:
        d = np.asarray(self.data, dtype=np.float64)
        _check_six(d, "Twist.data")
        object.__setattr__(self, "data", d)

    @classmethod
    def from_parts(cls, linear: ArrayLike, angular: ArrayLike) -> Twist:
        return cls(np.concatenate([np.asarray(linear), np.asarray(angular)], axis=-1))

    @property
    def linear(self) -> NDArray[np.float64]:
        return self.data[..., :3]

    @property
    def angular(self) -> NDArray[np.float64]:
        return self.data[..., 3:]

    def transform(self, T: Pose) -> Twist:
        r"""Map this twist through pose ``T`` using the adjoint of :math:`SE(3)`.

        .. math::

            \operatorname{Ad}_T \boldsymbol{\xi}
            = \begin{bmatrix} \mathbf{R} & [\mathbf{p}]_\times \mathbf{R} \\ \mathbf{0} & \mathbf{R} \end{bmatrix}
              \boldsymbol{\xi}.
        """
        R = T.rotation.as_matrix()
        v = R @ self.linear + np.cross(T.position, R @ self.angular)
        w = R @ self.angular
        return Twist.from_parts(v, w)


@dataclass(slots=True, frozen=True)
class Wrench:
    r"""Spatial force :math:`\mathbf{w} = (\mathbf{f}^\top, \boldsymbol{\tau}^\top)^\top \in \mathbb{R}^6`.

    Linear-first stacking (force, torque). Multiplication by a tension vector
    in the statics layer yields the platform wrench produced by the cables.
    """

    data: NDArray[np.float64]

    def __post_init__(self) -> None:
        d = np.asarray(self.data, dtype=np.float64)
        _check_six(d, "Wrench.data")
        object.__setattr__(self, "data", d)

    @classmethod
    def from_parts(cls, force: ArrayLike, torque: ArrayLike) -> Wrench:
        return cls(np.concatenate([np.asarray(force), np.asarray(torque)], axis=-1))

    @classmethod
    def gravity(cls, mass: float, g: ArrayLike = (0.0, 0.0, -9.81)) -> Wrench:
        g = np.asarray(g, dtype=np.float64)
        return cls.from_parts(mass * g, np.zeros(3))

    @property
    def force(self) -> NDArray[np.float64]:
        return self.data[..., :3]

    @property
    def torque(self) -> NDArray[np.float64]:
        return self.data[..., 3:]

    def transform(self, T: Pose) -> Wrench:
        r"""Co-adjoint transport of the wrench through pose ``T``.

        Wrenches transform with :math:`\operatorname{Ad}_T^{-\top}`, which for
        ``T = (p, R)`` evaluates to

        .. math::

            \begin{bmatrix} \mathbf{R} & \mathbf{0} \\ [\mathbf{p}]_\times \mathbf{R} & \mathbf{R} \end{bmatrix}.
        """
        R = T.rotation.as_matrix()
        f = R @ self.force
        tau = R @ self.torque + np.cross(T.position, f)
        return Wrench.from_parts(f, tau)

    def __add__(self, other: Wrench) -> Wrench:
        return Wrench(self.data + other.data)

    def __sub__(self, other: Wrench) -> Wrench:
        return Wrench(self.data - other.data)

    def __neg__(self) -> Wrench:
        return Wrench(-self.data)
