r"""SQCK hybrid cable law: Irvine static baseline + Kelvin--Voigt damping.

Per the Phase-7 directive, per cable :math:`i`,

.. math::

    T_i(t) \;=\; T_{\text{Irvine}, i}(t)
        \;+\; \frac{\eta A}{L_{0,i}}\,\hat{\mathbf{u}}_i^\top \dot{\mathbf{p}}(t).

The first term is the static catenary tension at the platform end (the
:class:`~cdpr.cables.irvine.IrvineModel` result); the second is the
velocity-projected damping correction with the same coefficient as the
Kelvin--Voigt model. The cable force on the platform is
:math:`\mathbf{F}_i = T_i\,\hat{\mathbf{u}}_i` along the chord direction.

This is a *single* selectable mode --- not an automatic fallback
composition of the other two. The implementation references the same
Irvine solver as :class:`IrvineModel` but stays self-contained so the
factory + diagnostics never have to introspect a hybrid's internals.
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
class SQCKHybridModel(CableModel):
    """Static catenary tension plus velocity-projected damping correction."""

    mode_name = "sqck_hybrid"

    # Irvine block.
    linear_density: ArrayLike = 0.07
    youngs_modulus: ArrayLike = 1.1e11
    cross_section: ArrayLike = 1.26e-5
    gravity_magnitude: float = 9.81

    # Kelvin--Voigt damping block (same η, A as the dynamic mode).
    viscous_coefficient: ArrayLike = 5.0e8

    n_cables: int | None = None

    _rho: NDArray[np.float64] = field(init=False, repr=False,
                                       default_factory=lambda: np.zeros(0))
    _E: NDArray[np.float64] = field(init=False, repr=False,
                                     default_factory=lambda: np.zeros(0))
    _A: NDArray[np.float64] = field(init=False, repr=False,
                                     default_factory=lambda: np.zeros(0))
    _eta: NDArray[np.float64] = field(init=False, repr=False,
                                       default_factory=lambda: np.zeros(0))

    # --- parameters --------------------------------------------------

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "linear_density": np.asarray(self.linear_density).tolist(),
            "youngs_modulus": np.asarray(self.youngs_modulus).tolist(),
            "cross_section": np.asarray(self.cross_section).tolist(),
            "viscous_coefficient": np.asarray(self.viscous_coefficient).tolist(),
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

        T_irvine, _ = self._irvine_components(robot, state, rest)
        # Damping correction --- shares the dL/dt projection with KV.
        dL_dt = self._length_rates(robot, state)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(rest > 1e-12, self._eta * self._A / rest, 0.0)
        T_damp = c * dL_dt

        T = T_irvine + T_damp
        return np.maximum(T, 0.0)

    def effective_length(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Arc length from the Irvine solver (sag-aware)."""
        self._ensure_arrays(robot.n_cables)
        rest = np.asarray(rest_lengths, dtype=np.float64).reshape(-1)
        _, arc = self._irvine_components(robot, state, rest)
        return arc

    # --- diagnostics --------------------------------------------------

    def diagnostics(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> dict[str, Any]:
        rest = np.asarray(rest_lengths, dtype=np.float64).reshape(-1)
        T_irvine, _ = self._irvine_components(robot, state, rest)
        dL_dt = self._length_rates(robot, state)
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(rest > 1e-12, self._eta * self._A / rest, 0.0)
        T_damp = c * dL_dt
        T_total = np.maximum(T_irvine + T_damp, 0.0)

        return {
            "mode": self.mode_name,
            "tension_min": float(T_total.min()),
            "tension_max": float(T_total.max()),
            "tension_mean": float(T_total.mean()),
            "T_irvine_mean": float(T_irvine.mean()),
            "T_damping_mean": float(T_damp.mean()),
            "T_damping_max_abs": float(np.max(np.abs(T_damp))),
            "max_abs_Ldot": float(np.max(np.abs(dL_dt))),
            "n_slack": int((T_total <= 1e-9).sum()),
            "n_cables": int(robot.n_cables),
        }

    # --- internals ---------------------------------------------------

    def _ensure_arrays(self, n: int) -> None:
        if self._rho.size != n:
            self._rho = _broadcast_per_cable(self.linear_density, n, "linear_density")
            self._E = _broadcast_per_cable(self.youngs_modulus, n, "youngs_modulus")
            self._A = _broadcast_per_cable(self.cross_section, n, "cross_section")
            self._eta = _broadcast_per_cable(self.viscous_coefficient, n, "viscous_coefficient")
            self.n_cables = n

    def _irvine_components(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(T_lower_irvine, arc_length)`` per cable."""
        self._ensure_arrays(robot.n_cables)
        rest = np.asarray(rest_lengths, dtype=np.float64).reshape(-1)
        _, b_world, _ = self._cable_geometry(robot, state)
        T_low = np.empty(robot.n_cables, dtype=np.float64)
        arc = np.empty(robot.n_cables, dtype=np.float64)
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
            T_low[i] = 0.0 if sol.is_slack else sol.tension_lower
            arc[i] = sol.arc_length
        return T_low, arc
