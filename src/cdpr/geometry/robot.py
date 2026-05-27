r"""Robot configuration: geometric anchors, platform attachments, physical limits.

A CDPR is described by:

* a set of ``m`` base anchors :math:`\mathbf{a}_i \in \mathbb{R}^3` expressed in
  the world frame --- the cable exit points on the fixed structure;
* a set of ``m`` platform attachments :math:`\mathbf{b}_i \in \mathbb{R}^3`
  expressed in the platform's body frame;
* the platform DOF count :math:`n \in \{3, 4, 5, 6\}` --- 3 for a point-mass
  translational CDPR, 6 for a fully spatial one. The number of cables ``m``
  must satisfy :math:`m \geq n + 1` for fully constrained operation.

This module deliberately keeps geometry, inertia, cable limits, and cable
material properties as separate dataclasses. The kinematics layer only needs
``RobotGeometry``; statics adds ``CableLimits``; dynamics adds
``PlatformInertia``; the elastic-catenary cable model adds ``CableProperties``.
Splitting them keeps unit tests and module imports narrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdpr.core.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Geometric configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class RobotGeometry:
    """Pure geometric description of a CDPR.

    Parameters
    ----------
    anchors:
        ``(m, 3)`` array of base-frame anchor positions.
    attachments:
        ``(m, 3)`` array of platform-frame cable attachment points.
    dof:
        Platform degrees of freedom. Must be 3, 4, 5, or 6.
    name:
        Human-readable identifier used in diagnostics and figures.
    """

    anchors: NDArray[np.float64]
    attachments: NDArray[np.float64]
    dof: int = 6
    name: str = "cdpr"

    def __post_init__(self) -> None:
        a = np.asarray(self.anchors, dtype=np.float64)
        b = np.asarray(self.attachments, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ConfigurationError(f"anchors must have shape (m, 3); got {a.shape}")
        if b.shape != a.shape:
            raise ConfigurationError(
                f"attachments shape {b.shape} does not match anchors shape {a.shape}"
            )
        if self.dof not in (2, 3, 4, 5, 6):
            raise ConfigurationError(f"dof must be in {{2,3,4,5,6}}; got {self.dof}")
        if a.shape[0] < self.dof + 1:
            raise ConfigurationError(
                f"A fully constrained {self.dof}-DOF CDPR requires at least "
                f"{self.dof + 1} cables; got {a.shape[0]}."
            )
        object.__setattr__(self, "anchors", a)
        object.__setattr__(self, "attachments", b)

    @property
    def n_cables(self) -> int:
        return int(self.anchors.shape[0])

    @property
    def redundancy(self) -> int:
        """Degree of actuation redundancy, :math:`r = m - n`."""
        return self.n_cables - self.dof

    @classmethod
    def from_arrays(
        cls,
        anchors: ArrayLike,
        attachments: ArrayLike,
        *,
        dof: int = 6,
        name: str = "cdpr",
    ) -> Self:
        return cls(
            anchors=np.asarray(anchors, dtype=np.float64),
            attachments=np.asarray(attachments, dtype=np.float64),
            dof=dof,
            name=name,
        )

    def bounding_box(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Axis-aligned bounding box of the base anchor set."""
        return self.anchors.min(axis=0), self.anchors.max(axis=0)


# ---------------------------------------------------------------------------
# Physical properties
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class PlatformInertia:
    """Rigid-body inertia of the moving platform, expressed at its body frame.

    Parameters
    ----------
    mass:
        Total mass [kg].
    com:
        Centre of mass in the platform body frame [m]. Defaults to the body
        origin, which is the convention assumed elsewhere in the framework.
    inertia:
        ``(3, 3)`` inertia tensor about the body origin [kg m^2]. Defaults to
        a diagonal sphere-equivalent tensor; replace with the real value for
        physically meaningful dynamic simulations.
    """

    mass: float
    com: NDArray[np.float64] = field(default_factory=lambda: np.zeros(3))
    inertia: NDArray[np.float64] = field(default_factory=lambda: np.eye(3) * 1e-3)

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ConfigurationError(f"mass must be positive; got {self.mass}")
        com = np.asarray(self.com, dtype=np.float64).reshape(3)
        I = np.asarray(self.inertia, dtype=np.float64)
        if I.shape != (3, 3):
            raise ConfigurationError(f"inertia must be (3, 3); got {I.shape}")
        if not np.allclose(I, I.T, atol=1e-10):
            raise ConfigurationError("inertia tensor must be symmetric")
        eigs = np.linalg.eigvalsh(I)
        if (eigs <= 0.0).any():
            raise ConfigurationError(f"inertia tensor must be positive-definite; eigs={eigs}")
        object.__setattr__(self, "com", com)
        object.__setattr__(self, "inertia", I)

    @classmethod
    def point_mass(cls, mass: float) -> Self:
        return cls(mass=mass, com=np.zeros(3), inertia=np.eye(3) * 1e-6)


@dataclass(slots=True, frozen=True)
class CableLimits:
    """Per-cable lower / upper tension bounds and (optional) maximum length.

    Stored as arrays of length ``m`` so heterogeneous cable assemblies can
    be modelled (e.g. one heavy lifting cable plus several thinner tendons).
    """

    t_min: NDArray[np.float64]
    t_max: NDArray[np.float64]
    l_max: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        tmin = np.asarray(self.t_min, dtype=np.float64)
        tmax = np.asarray(self.t_max, dtype=np.float64)
        if tmin.shape != tmax.shape:
            raise ConfigurationError(
                f"t_min shape {tmin.shape} != t_max shape {tmax.shape}"
            )
        if (tmin < 0).any():
            raise ConfigurationError("t_min must be non-negative (cables push? no.)")
        if (tmax <= tmin).any():
            raise ConfigurationError("t_max must exceed t_min for every cable")
        object.__setattr__(self, "t_min", tmin)
        object.__setattr__(self, "t_max", tmax)
        if self.l_max is not None:
            lmax = np.asarray(self.l_max, dtype=np.float64)
            if lmax.shape != tmin.shape:
                raise ConfigurationError("l_max shape must match t_min / t_max")
            if (lmax <= 0).any():
                raise ConfigurationError("l_max must be positive")
            object.__setattr__(self, "l_max", lmax)

    @classmethod
    def uniform(cls, n_cables: int, t_min: float = 10.0, t_max: float = 1000.0) -> Self:
        return cls(t_min=np.full(n_cables, t_min), t_max=np.full(n_cables, t_max))


@dataclass(slots=True, frozen=True)
class CableProperties:
    """Material / cross-section data needed by the elastic and sagging models.

    All quantities are per-cable arrays of length ``m`` so heterogeneous
    cable choices are first-class. Defaults correspond to a 4 mm steel
    aircraft cable (Young's modulus ~110 GPa, linear density ~0.07 kg/m).
    """

    youngs_modulus: NDArray[np.float64]
    cross_section: NDArray[np.float64]
    linear_density: NDArray[np.float64]
    unstretched_length: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        E = np.asarray(self.youngs_modulus, dtype=np.float64)
        A = np.asarray(self.cross_section, dtype=np.float64)
        rho = np.asarray(self.linear_density, dtype=np.float64)
        if E.shape != A.shape or A.shape != rho.shape:
            raise ConfigurationError("youngs_modulus, cross_section, linear_density must agree in shape")
        if (E <= 0).any() or (A <= 0).any() or (rho < 0).any():
            raise ConfigurationError("E, A must be positive and rho non-negative")
        object.__setattr__(self, "youngs_modulus", E)
        object.__setattr__(self, "cross_section", A)
        object.__setattr__(self, "linear_density", rho)

    @classmethod
    def steel_aircraft_cable(cls, n_cables: int, diameter_m: float = 4e-3) -> Self:
        E = np.full(n_cables, 110e9)
        A = np.full(n_cables, np.pi * (diameter_m / 2.0) ** 2)
        rho = np.full(n_cables, 7850.0 * np.pi * (diameter_m / 2.0) ** 2)
        return cls(youngs_modulus=E, cross_section=A, linear_density=rho)

    @property
    def axial_stiffness(self) -> NDArray[np.float64]:
        """Per-cable :math:`EA` product (axial stiffness)."""
        return self.youngs_modulus * self.cross_section


# ---------------------------------------------------------------------------
# Assembled robot
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Robot:
    """A CDPR description that bundles geometry, inertia, and cable data.

    Only :attr:`geometry` is required. The remaining fields default to neutral
    values that let kinematics-only code run without forcing the user to
    invent inertia or cable material parameters.
    """

    geometry: RobotGeometry
    inertia: PlatformInertia | None = None
    limits: CableLimits | None = None
    cable_properties: CableProperties | None = None

    # convenience pass-throughs --------------------------------------------------

    @property
    def n_cables(self) -> int:
        return self.geometry.n_cables

    @property
    def dof(self) -> int:
        return self.geometry.dof

    @property
    def redundancy(self) -> int:
        return self.geometry.redundancy

    @property
    def name(self) -> str:
        return self.geometry.name

    @property
    def anchors(self) -> NDArray[np.float64]:
        return self.geometry.anchors

    @property
    def attachments(self) -> NDArray[np.float64]:
        return self.geometry.attachments

    # factory --------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        anchors: ArrayLike,
        attachments: ArrayLike,
        *,
        dof: int = 6,
        name: str = "cdpr",
        mass: float | None = None,
        t_min: float = 10.0,
        t_max: float = 1000.0,
    ) -> Self:
        """Convenience constructor mirroring :py:meth:`RobotGeometry.from_arrays`.

        Optional ``mass``, ``t_min``, ``t_max`` populate the inertia and
        cable-limit blocks with uniform defaults. Heterogeneous configurations
        should be built explicitly by composing the dataclasses.
        """
        geom = RobotGeometry.from_arrays(anchors, attachments, dof=dof, name=name)
        return cls(
            geometry=geom,
            inertia=PlatformInertia(mass=mass) if mass is not None else None,
            limits=CableLimits.uniform(geom.n_cables, t_min=t_min, t_max=t_max),
        )

    def require_inertia(self) -> PlatformInertia:
        if self.inertia is None:
            raise ConfigurationError(
                f"Robot {self.name!r} has no inertia set; dynamics requires PlatformInertia."
            )
        return self.inertia

    def require_limits(self) -> CableLimits:
        if self.limits is None:
            raise ConfigurationError(
                f"Robot {self.name!r} has no cable limits set; statics requires CableLimits."
            )
        return self.limits

    def require_cable_properties(self) -> CableProperties:
        if self.cable_properties is None:
            raise ConfigurationError(
                f"Robot {self.name!r} has no CableProperties; elastic/sagging models require them."
            )
        return self.cable_properties
