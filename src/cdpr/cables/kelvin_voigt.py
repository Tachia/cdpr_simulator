r"""Kelvin--Voigt cable constitutive law.

Per cable :math:`i`,

.. math::

    T_i(t) \;=\; k_i\,\delta L_i(t) \;+\; c_i\,\dot{\delta L}_i(t),
    \qquad
    k_i = \frac{EA}{L_{0,i}}, \quad c_i = \frac{\eta A}{L_{0,i}},

with :math:`\delta L_i = L_i - L_{0,i}` and
:math:`\dot{\delta L}_i = \dot L_i` (the rest length is per-step constant
within the integrator). Non-negative tension is enforced (a cable that
would push goes slack and contributes zero force); the stretch sign and
slack flag are reported through :meth:`KelvinVoigtModel.diagnostics` so
downstream consumers can see the transition without re-evaluating.

Analytic Jacobian: :math:`\partial T_i / \partial \delta L_i = k_i` while
the cable is taut, zero while slack; :math:`\partial T_i / \partial \dot L_i = c_i`
while taut, zero while slack. Both are exposed via
:meth:`KelvinVoigtModel.tension_jacobian`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdpr.cables.base import CableModel

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot


def _broadcast_per_cable(value: ArrayLike, n_cables: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        return np.full(n_cables, float(arr[0]))
    if arr.size != n_cables:
        raise ValueError(
            f"{name} must be a scalar or have length n_cables={n_cables}; got size {arr.size}"
        )
    return arr


@dataclass(slots=True)
class KelvinVoigtModel(CableModel):
    """Dynamic elastic-damping cable law."""

    mode_name = "kelvin_voigt"

    youngs_modulus: ArrayLike = 1.1e11      # Pa (default: steel)
    cross_section: ArrayLike = 1.26e-5      # m^2 (default: 4 mm wire-rope)
    viscous_coefficient: ArrayLike = 5.0e8  # Pa.s (representative)
    n_cables: int | None = None             # cached on first call

    # Internal: per-cable broadcast arrays, materialised lazily so the
    # caller can construct the model before they know the robot.
    _E: NDArray[np.float64] = field(init=False, repr=False, default_factory=lambda: np.zeros(0))
    _A: NDArray[np.float64] = field(init=False, repr=False, default_factory=lambda: np.zeros(0))
    _eta: NDArray[np.float64] = field(init=False, repr=False, default_factory=lambda: np.zeros(0))

    # --- parameters --------------------------------------------------

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "youngs_modulus": np.asarray(self.youngs_modulus).tolist(),
            "cross_section": np.asarray(self.cross_section).tolist(),
            "viscous_coefficient": np.asarray(self.viscous_coefficient).tolist(),
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
        if rest.shape[0] != robot.n_cables:
            raise ValueError(
                f"rest_lengths shape {rest.shape} != n_cables {robot.n_cables}"
            )

        L = self.effective_length(robot, state, rest)
        delta = L - rest
        dL_dt = self._length_rates(robot, state)

        # Per-cable stiffness and damping (rest length appears in the denominator).
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(rest > 1e-12, self._E * self._A / rest, 0.0)
            c = np.where(rest > 1e-12, self._eta * self._A / rest, 0.0)

        T = k * delta + c * dL_dt
        # Cables can only pull: clip to zero on slack / compression.
        return np.maximum(T, 0.0)

    def tension_jacobian(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Per-cable :math:`(\\partial T / \\partial \\delta L, \\partial T / \\partial \\dot L)`.

        Returns a dict with keys ``"dT_dDeltaL"`` and ``"dT_dLdot"`` --- each
        a length-``m`` array. Both are zero where the cable is slack.
        """
        self._ensure_arrays(robot.n_cables)
        rest = np.asarray(rest_lengths, dtype=np.float64).reshape(-1)
        taut = (self.tension(robot, state, rest) > 1e-9).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            k = np.where(rest > 1e-12, self._E * self._A / rest, 0.0)
            c = np.where(rest > 1e-12, self._eta * self._A / rest, 0.0)
        return {
            "dT_dDeltaL": taut * k,
            "dT_dLdot": taut * c,
        }

    # --- diagnostics --------------------------------------------------

    def diagnostics(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> dict[str, Any]:
        # NOTE: do not use ``super().diagnostics(...)`` here. The class
        # carries ``@dataclass(slots=True)``, and Python's dataclass
        # mechanism replaces the original class object with a freshly
        # constructed slotted one. The ``__class__`` cell that zero-arg
        # ``super()`` captures at compile time still refers to the
        # *pre-replacement* class, but ``self`` is an instance of the
        # *post-replacement* class --- so the runtime check
        # ``isinstance(self, __class__)`` fails and raises
        # ``TypeError: super(type, obj): obj must be an instance or
        # subtype of type``. Calling the parent method by name sidesteps
        # the issue entirely and is correct here because ``CableModel``
        # is the single base class (no diamond inheritance).
        base = CableModel.diagnostics(self, robot, state, rest_lengths)
        dL_dt = self._length_rates(robot, state)
        base.update({
            "max_abs_Ldot": float(np.max(np.abs(dL_dt))),
            "Ldot_mean": float(dL_dt.mean()),
            "Ldot_min": float(dL_dt.min()),
            "Ldot_max": float(dL_dt.max()),
        })
        return base

    # --- internals ---------------------------------------------------

    def _ensure_arrays(self, n: int) -> None:
        if self._E.size != n:
            self._E = _broadcast_per_cable(self.youngs_modulus, n, "youngs_modulus")
            self._A = _broadcast_per_cable(self.cross_section, n, "cross_section")
            self._eta = _broadcast_per_cable(self.viscous_coefficient, n, "viscous_coefficient")
            self.n_cables = n
