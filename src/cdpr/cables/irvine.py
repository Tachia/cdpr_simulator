r"""Irvine catenary constitutive law (static sag-aware).

Each cable carries its own self-weight and hangs in the plane spanned by
the chord and gravity. The two-equation Irvine inverse problem from
:mod:`cdpr.cables.sagging` solves for the cable force components
:math:`(H, V)` at the lower (platform) end given chord geometry, cable
unstretched length, axial stiffness :math:`EA`, and weight per unit
unstretched length :math:`w = \mu g`.

Per the Phase-7 directive we expose the *scalar* tension at the platform
end (:math:`T_i = \sqrt{H_i^2 + V_i^2}`) and apply it along the chord
direction --- the simplification noted in the directive's SQCK formula.
The full elastic-catenary tangent direction at the lower end is
recoverable from :meth:`IrvineModel.diagnostics`; it agrees with the
chord direction in the small-sag limit and deviates slightly under
heavy self-weight.

This mode has no velocity dependence --- cable damping is the Kelvin--Voigt
mode's job. Per the directive's exclusivity rule, the Irvine mode never
injects damping internally.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdpr.cables.base import CableModel
from cdpr.cables.kelvin_voigt import _broadcast_per_cable
from cdpr.cables.sagging import sagging_cable

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot


@dataclass(slots=True)
class IrvineModel(CableModel):
    """Static elastic-catenary cable law."""

    mode_name = "irvine"

    linear_density: ArrayLike = 0.07         # kg/m (4 mm steel cable default)
    youngs_modulus: ArrayLike = 1.1e11        # Pa
    cross_section: ArrayLike = 1.26e-5        # m^2
    gravity_magnitude: float = 9.81
    n_cables: int | None = None

    _rho: NDArray[np.float64] = field(init=False, repr=False,
                                       default_factory=lambda: np.zeros(0))
    _E: NDArray[np.float64] = field(init=False, repr=False,
                                     default_factory=lambda: np.zeros(0))
    _A: NDArray[np.float64] = field(init=False, repr=False,
                                     default_factory=lambda: np.zeros(0))

    # --- parameters --------------------------------------------------

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "linear_density": np.asarray(self.linear_density).tolist(),
            "youngs_modulus": np.asarray(self.youngs_modulus).tolist(),
            "cross_section": np.asarray(self.cross_section).tolist(),
            "gravity_magnitude": float(self.gravity_magnitude),
        }

    # --- main constitutive evaluation --------------------------------

    def tension(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._ensure_arrays(robot.n_cables)
        rest = np.asarray(rest_lengths, dtype=np.float64).reshape(-1)
        _, b_world, _ = self._cable_geometry(robot, state)

        T = np.empty(robot.n_cables, dtype=np.float64)
        gravity_vec = (0.0, 0.0, -self.gravity_magnitude)
        for i in range(robot.n_cables):
            sol = sagging_cable(
                anchor_upper=robot.anchors[i],
                anchor_lower=b_world[i],
                unstretched_length=float(rest[i]),
                axial_stiffness=float(self._E[i] * self._A[i]),
                linear_weight=float(self._rho[i] * self.gravity_magnitude),
                gravity=gravity_vec,
            )
            # Scalar tension at the platform end (the "lower" attachment).
            T[i] = 0.0 if sol.is_slack else sol.tension_lower
        return np.maximum(T, 0.0)

    def effective_length(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Arc length along the cable (longer than chord under sag)."""
        self._ensure_arrays(robot.n_cables)
        rest = np.asarray(rest_lengths, dtype=np.float64).reshape(-1)
        _, b_world, _ = self._cable_geometry(robot, state)

        L = np.empty(robot.n_cables, dtype=np.float64)
        gravity_vec = (0.0, 0.0, -self.gravity_magnitude)
        for i in range(robot.n_cables):
            sol = sagging_cable(
                anchor_upper=robot.anchors[i],
                anchor_lower=b_world[i],
                unstretched_length=float(rest[i]),
                axial_stiffness=float(self._E[i] * self._A[i]),
                linear_weight=float(self._rho[i] * self.gravity_magnitude),
                gravity=gravity_vec,
            )
            L[i] = sol.arc_length
        return L

    # --- diagnostics --------------------------------------------------

    def diagnostics(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Adds sag and per-end-tension reporting on top of the base summary."""
        self._ensure_arrays(robot.n_cables)
        rest = np.asarray(rest_lengths, dtype=np.float64).reshape(-1)
        _, b_world, _ = self._cable_geometry(robot, state)

        gravity_vec = (0.0, 0.0, -self.gravity_magnitude)
        sags = np.empty(robot.n_cables, dtype=np.float64)
        T_low = np.empty(robot.n_cables, dtype=np.float64)
        T_top = np.empty(robot.n_cables, dtype=np.float64)
        slack = np.zeros(robot.n_cables, dtype=bool)
        arc_lengths = np.empty(robot.n_cables, dtype=np.float64)
        for i in range(robot.n_cables):
            sol = sagging_cable(
                anchor_upper=robot.anchors[i],
                anchor_lower=b_world[i],
                unstretched_length=float(rest[i]),
                axial_stiffness=float(self._E[i] * self._A[i]),
                linear_weight=float(self._rho[i] * self.gravity_magnitude),
                gravity=gravity_vec,
            )
            sags[i] = sol.sag_max
            T_low[i] = sol.tension_lower
            T_top[i] = sol.tension_upper
            slack[i] = sol.is_slack
            arc_lengths[i] = sol.arc_length

        return {
            "mode": self.mode_name,
            "tension_min": float(T_low.min()),
            "tension_max": float(T_low.max()),
            "tension_mean": float(T_low.mean()),
            "tension_upper_max": float(T_top.max()),
            "sag_max": float(sags.max()),
            "sag_mean": float(sags.mean()),
            "n_slack": int(slack.sum()),
            "n_cables": int(robot.n_cables),
            "max_arc_minus_chord": float(np.max(arc_lengths - rest)),
        }

    # --- internals ---------------------------------------------------

    def _ensure_arrays(self, n: int) -> None:
        if self._rho.size != n:
            self._rho = _broadcast_per_cable(self.linear_density, n, "linear_density")
            self._E = _broadcast_per_cable(self.youngs_modulus, n, "youngs_modulus")
            self._A = _broadcast_per_cable(self.cross_section, n, "cross_section")
            self.n_cables = n
