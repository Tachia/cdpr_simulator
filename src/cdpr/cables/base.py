r"""Cable-model base layer.

Two things live here:

* :class:`CableSolution` --- the per-cable evaluation outcome used by the
  Phase-1 per-cable helpers (``massless_cable``, ``elastic_cable``,
  ``sagging_cable``).
* :class:`CableModel` --- a strict constitutive-model abstract base class
  introduced in Phase 7. Subclasses implement one of the three mutually
  exclusive constitutive laws (Kelvin--Voigt, Irvine, SQCK hybrid) and
  drive the simulator through a uniform interface.

Conventions in :class:`CableModel`
----------------------------------

For each cable :math:`i` connecting world anchor
:math:`\mathbf{a}_i \in \mathbb{R}^3` to platform attachment
:math:`\mathbf{b}_i \in \mathbb{R}^3` (in the platform's body frame) when
the platform is at pose :math:`(\mathbf{p}, \mathbf{R})`:

* World attachment :math:`\mathbf{B}_i = \mathbf{p} + \mathbf{R}\,\mathbf{b}_i`,
* Cable chord vector :math:`\mathbf{l}_i = \mathbf{a}_i - \mathbf{B}_i`,
* Chord length :math:`L_i = \lVert \mathbf{l}_i \rVert`,
* Unit vector :math:`\hat{\mathbf{u}}_i = \mathbf{l}_i / L_i`
  **pointing from platform toward the anchor** --- this is the direction
  in which the cable pulls the platform.
* Cable force on the platform: :math:`\mathbf{F}_i = T_i\,\hat{\mathbf{u}}_i`.

Note on the directive's notation: the directive defines
:math:`\hat{\mathbf{u}}_i = (\mathbf{p} - \mathbf{a}_i)/\lVert \mathbf{p} - \mathbf{a}_i \rVert`
(anchor-to-platform). With that convention, the *physical* force on the
platform is :math:`-T_i\,\hat{\mathbf{u}}_i`. We keep the unit vector
pointing **toward the anchor** internally because that's the direction
the cable actually pulls; the rate-of-length formula then becomes
:math:`\dot L_i = -\hat{\mathbf{u}}_i^\top \dot{\mathbf{B}}_i`. The two
conventions are equivalent and the directive's
:math:`\dot L_i = \hat{\mathbf{u}}_i^\top \dot{\mathbf{p}}` formula
matches when interpreted with its own sign convention.

For the 6-DOF case the attachment-point velocity is
:math:`\dot{\mathbf{B}}_i = \mathbf{v} + \boldsymbol{\omega} \times (\mathbf{R}\,\mathbf{b}_i)`,
so

.. math::

    \dot L_i = -\,\hat{\mathbf{u}}_i^\top
        \bigl(\mathbf{v} + \boldsymbol{\omega} \times (\mathbf{R}\,\mathbf{b}_i)\bigr).

The directive's point-mass formula falls out for ``b_i = 0``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.core.frames import Wrench
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot


# ---------------------------------------------------------------------------
# Per-cable evaluation outcome (Phase 1)
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class CableSolution:
    """Outcome of evaluating one cable model at one configuration.

    Used by the per-cable function helpers in
    :mod:`cdpr.cables.massless`, :mod:`cdpr.cables.elastic`,
    :mod:`cdpr.cables.sagging`. The Phase-7 :class:`CableModel` returns
    vector quantities directly and does not produce :class:`CableSolution`
    objects --- the two layers exist side by side.
    """

    force_on_platform: NDArray[np.float64]
    tension_lower: float
    tension_upper: float
    chord_length: float
    arc_length: float
    sag_max: float = 0.0
    axial_strain: float = 0.0
    is_slack: bool = False
    diagnostics: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Constitutive-model abstract base (Phase 7)
# ---------------------------------------------------------------------------

class CableModel(ABC):
    """Strict abstract base for the three exclusive cable constitutive laws.

    Subclasses set :attr:`mode_name` to one of ``"kelvin_voigt"``,
    ``"irvine"``, ``"sqck_hybrid"`` and implement :meth:`tension`. The
    default :meth:`effective_length`, :meth:`force_vector`, and
    :meth:`platform_wrench` derive from the chord geometry; sagging-style
    models override :meth:`effective_length` to return the arc length.

    The interface is deliberately stateless --- the model object holds
    only its calibration parameters. State (pose + velocity) and the
    instantaneous rest lengths are passed in at every call. This keeps
    the model thread-safe and lets the same instance drive a CDPR core
    simulation, an external backend, and an identification routine
    without coordination.
    """

    mode_name: ClassVar[str] = "abstract"

    # --- declared by subclasses --------------------------------------

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Calibration parameters in a serialisable dict."""

    @abstractmethod
    def tension(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Per-cable tension magnitude (size m), clamped to be non-negative.

        Cables physically cannot push. Subclasses should produce
        non-negative tension by construction (slack returns zero).
        """

    # --- defaults provided here --------------------------------------

    def effective_length(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Cable length used by the constitutive law.

        Default: geometric chord length (taut straight cable). Sagging
        models override this with the cable's arc length.
        """
        _, _, L = self._cable_geometry(robot, state)
        return L

    def stretch(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r""":math:`\delta L_i = L_i - L_{0,i}`."""
        return self.effective_length(robot, state, rest_lengths) - rest_lengths

    def force_vector(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Total cable force on the platform (3-vector).

        :math:`\mathbf{F}_\text{total} = \sum_i T_i\,\hat{\mathbf{u}}_i`
        with :math:`\hat{\mathbf{u}}_i` pointing platform :math:`\to` anchor.
        """
        T = self.tension(robot, state, rest_lengths)
        u, _, _ = self._cable_geometry(robot, state)
        return (T[:, None] * u).sum(axis=0)

    def platform_wrench(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> "Wrench":
        r"""Total cable wrench on the platform body origin.

        Force = :math:`\sum_i T_i\,\hat{\mathbf{u}}_i`;
        Torque = :math:`\sum_i (\mathbf{R}\,\mathbf{b}_i) \times (T_i\,\hat{\mathbf{u}}_i)`.
        """
        from cdpr.core.frames import Wrench
        T = self.tension(robot, state, rest_lengths)
        u, b_world, _ = self._cable_geometry(robot, state)
        forces = T[:, None] * u                                  # (m, 3)
        Rb = state.pose.rotation.apply(robot.attachments)        # (m, 3)
        torques = np.cross(Rb, forces)                            # (m, 3)
        return Wrench.from_parts(forces.sum(axis=0), torques.sum(axis=0))

    def diagnostics(
        self,
        robot: "Robot",
        state: "PlatformState",
        rest_lengths: NDArray[np.float64],
    ) -> dict[str, Any]:
        """Lightweight per-step diagnostics summary."""
        T = self.tension(robot, state, rest_lengths)
        L = self.effective_length(robot, state, rest_lengths)
        delta = L - rest_lengths
        slack = (T <= 1e-9)
        return {
            "mode": self.mode_name,
            "tension_min": float(T.min()),
            "tension_max": float(T.max()),
            "tension_mean": float(T.mean()),
            "stretch_min": float(delta.min()),
            "stretch_max": float(delta.max()),
            "n_slack": int(slack.sum()),
            "n_cables": int(T.shape[0]),
        }

    # --- shared geometric helpers ------------------------------------

    def _cable_geometry(
        self, robot: "Robot", state: "PlatformState",
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""Return ``(unit_vectors, B_world, lengths)`` for every cable.

        ``B_world`` are the world-frame attachment points. Unit vectors
        point from each platform attachment toward its anchor (the
        physical direction in which the cable pulls).
        """
        b_world = state.pose.rotation.apply(robot.attachments) + state.pose.position
        cable_vec = robot.anchors - b_world                       # (m, 3)
        L = np.linalg.norm(cable_vec, axis=-1)
        L_safe = np.where(L > 1e-12, L, 1.0)
        u = cable_vec / L_safe[:, None]
        return u, b_world, L

    def _length_rates(
        self, robot: "Robot", state: "PlatformState",
    ) -> NDArray[np.float64]:
        r"""Per-cable :math:`\dot L_i = -\hat{\mathbf{u}}_i^\top \dot{\mathbf{B}}_i`.

        Equivalent to the directive's
        :math:`\hat{\mathbf{u}}_i^\top \dot{\mathbf{p}}` under the sign
        convention noted in the module docstring, with full 6-DOF angular
        velocity coupling at the attachment point.
        """
        u, _, _ = self._cable_geometry(robot, state)
        Rb = state.pose.rotation.apply(robot.attachments)         # (m, 3)
        v_attach = state.velocity.linear + np.cross(
            state.velocity.angular, Rb,
        )                                                          # (m, 3)
        # dL/dt = -u^T v_attach (length grows as platform moves away from anchor)
        return -np.einsum("ij,ij->i", u, v_attach)

    # --- model identity ---------------------------------------------

    def __repr__(self) -> str:                                # pragma: no cover - cosmetic
        return f"{type(self).__name__}(mode={self.mode_name!r}, params={self.parameters})"
