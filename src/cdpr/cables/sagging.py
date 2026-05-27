r"""Irvine elastic-catenary cable model.

The cable is modelled as a perfectly flexible elastic rod that supports its
own distributed weight. With unstretched length :math:`L_0`, weight per unit
unstretched length :math:`w` (so :math:`w = \rho_\ell\,g` for linear density
:math:`\rho_\ell`), and axial stiffness :math:`EA`, the cable hangs in the
plane spanned by the chord and gravity. In that plane, with the lower end
(platform attachment) at the origin, :math:`x` horizontal toward the upper
end's horizontal projection, and :math:`z` vertical (upward), the upper end
:math:`(x_h, z_h)` is related to the force the cable exerts on the platform
:math:`(H, V)` --- :math:`H` horizontal in :math:`+x`, :math:`V` vertical in
:math:`+z` (so :math:`V > 0` means the cable pulls the platform *up*) --- by
the two Irvine equations

.. math::

    x_h \;=\; \frac{H\,L_0}{EA}
       \;+\; \frac{H}{w}\bigl[\operatorname{arcsinh}\!\tfrac{V + wL_0}{H}
                            \;-\; \operatorname{arcsinh}\!\tfrac{V}{H}\bigr],

.. math::

    z_h \;=\; \frac{V\,L_0}{EA}
       \;+\; \frac{w\,L_0^2}{2\,EA}
       \;+\; \frac{1}{w}\Bigl[\sqrt{H^2 + (V+wL_0)^2}
                              \;-\; \sqrt{H^2 + V^2}\Bigr].

The CDPR cable problem is the *inverse* of these: given :math:`(x_h, z_h)`,
:math:`L_0`, :math:`EA`, :math:`w`, solve for :math:`(H, V)`. The system is
two equations in two unknowns and is solved here with a damped Newton step
backed by SciPy's ``fsolve``, started from the straight-elastic
approximation. The Jacobian is provided in closed form, which both
accelerates convergence and removes the worst conditioning issues of
finite-difference Jacobians near vertical-cable degeneracies.

References
----------
Irvine, H. M. *Cable Structures*, MIT Press, 1981.
Kozak, K., Zhou, Q., Wang, J. *Static analysis of cable-driven manipulators
with non-negligible cable mass*, IEEE Trans. Robotics 22(3), 2006.
Such, M., Jimenez-Octavio, J. R., Carnicero, A., Lopez-Garcia, O. *An
approach based on the catenary equation to deal with static analysis of
three-dimensional cable structures*, Engineering Structures 31(9), 2009.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import fsolve

from cdpr.cables.base import CableSolution


# Small numerical floor used to avoid the vertical-cable singularity in the
# Irvine equations when ``H`` is near zero. Values below this magnitude get
# pushed back to ``+H_FLOOR``; the resulting solution is then post-checked.
_H_FLOOR = 1e-6


@dataclass(frozen=True, slots=True)
class _CableFrame:
    """Local 2D cable plane defined by the chord and gravity."""

    origin: NDArray[np.float64]      # world-frame lower-end position
    x_hat: NDArray[np.float64]       # world-frame +x_local direction
    z_hat: NDArray[np.float64]       # world-frame +z_local direction (against gravity)
    xh: float                        # upper-end horizontal coordinate
    zh: float                        # upper-end vertical coordinate

    def to_world(self, fx_local: float, fz_local: float) -> NDArray[np.float64]:
        return fx_local * self.x_hat + fz_local * self.z_hat


def _build_cable_frame(
    anchor_upper: NDArray[np.float64],
    anchor_lower: NDArray[np.float64],
    gravity: NDArray[np.float64],
) -> _CableFrame:
    """Construct the 2D plane in which the cable hangs.

    The +z_local axis points opposite to gravity; +x_local is the unit
    component of the chord vector in the plane perpendicular to gravity. For
    a perfectly vertical chord the +x_local direction is degenerate; we then
    choose an arbitrary horizontal axis since the resulting Irvine system is
    1-D (vertical cable) and the choice does not affect the platform force.
    """
    chord = anchor_upper - anchor_lower
    z_hat = -gravity / np.linalg.norm(gravity)        # unit "up"
    chord_horizontal = chord - np.dot(chord, z_hat) * z_hat
    h_norm = float(np.linalg.norm(chord_horizontal))
    if h_norm > 1e-12:
        x_hat = chord_horizontal / h_norm
    else:
        # Vertical chord -- pick any horizontal axis perpendicular to z_hat.
        tmp = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_hat = tmp - np.dot(tmp, z_hat) * z_hat
        x_hat = x_hat / np.linalg.norm(x_hat)
    xh = float(np.dot(chord, x_hat))
    zh = float(np.dot(chord, z_hat))
    return _CableFrame(origin=anchor_lower, x_hat=x_hat, z_hat=z_hat, xh=xh, zh=zh)


def _irvine_residual_and_jac(
    HV: NDArray[np.float64],
    L0: float,
    EA: float,
    w: float,
    xh: float,
    zh: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Residual ``(xh, zh)_target - (xh, zh)_predicted`` and its analytic Jacobian."""
    H, V = float(HV[0]), float(HV[1])
    if abs(H) < _H_FLOOR:
        H = _H_FLOOR if H >= 0 else -_H_FLOOR

    V_top = V + w * L0
    T_low = np.hypot(H, V)
    T_top = np.hypot(H, V_top)

    asinh_top = np.arcsinh(V_top / H)
    asinh_low = np.arcsinh(V / H)

    xh_pred = H * L0 / EA + (H / w) * (asinh_top - asinh_low)
    zh_pred = V * L0 / EA + w * L0**2 / (2 * EA) + (T_top - T_low) / w

    res = np.array([xh - xh_pred, zh - zh_pred])

    # Jacobian of (xh_pred, zh_pred) w.r.t. (H, V).
    # d/dH asinh(u/H) = -u/(H * sqrt(H^2 + u^2))  ; with u = V_top or V
    dxh_dH = (
        L0 / EA
        + (asinh_top - asinh_low) / w
        + (H / w) * (-V_top / (H * T_top) + V / (H * T_low))
    )
    # d/dV asinh((V+wL0)/H) = (1/H)/sqrt(1+((V+wL0)/H)^2) = 1/T_top  ; similar for V/H
    dxh_dV = (H / w) * (1.0 / T_top - 1.0 / T_low)

    dzh_dH = (H / w) * (1.0 / T_top - 1.0 / T_low)
    dzh_dV = L0 / EA + (V_top / T_top - V / T_low) / w

    # We are returning  res = target - pred, so the Jacobian of res is  -d pred / d HV.
    J = -np.array([[dxh_dH, dxh_dV], [dzh_dH, dzh_dV]])
    return res, J


def _initial_guess(L0: float, EA: float, xh: float, zh: float) -> NDArray[np.float64]:
    """Straight-elastic-rod approximation, used as the Newton warm start."""
    L_c = float(np.hypot(xh, zh))
    strain = max((L_c - L0) / L0, 1e-4)        # bias to a positive guess
    T = EA * strain
    if L_c > 0:
        return np.array([T * xh / L_c, T * zh / L_c])
    return np.array([_H_FLOOR, max(EA * 1e-4, 1.0)])


def sagging_cable(
    anchor_upper: ArrayLike,
    anchor_lower: ArrayLike,
    unstretched_length: float,
    axial_stiffness: float,
    linear_weight: float,
    *,
    gravity: ArrayLike = (0.0, 0.0, -9.81),
    xtol: float = 1e-10,
    max_iter: int = 100,
) -> CableSolution:
    r"""Solve the Irvine elastic-catenary inverse problem for one cable.

    Parameters
    ----------
    anchor_upper, anchor_lower:
        World-frame endpoints. ``anchor_upper`` is the fixed structure
        anchor; ``anchor_lower`` is the platform attachment.
    unstretched_length:
        :math:`L_0`, the natural cable length when no tension is applied.
    axial_stiffness:
        :math:`EA` product.
    linear_weight:
        :math:`w`, cable weight per unit unstretched length [N/m]. For a
        linear density :math:`\rho_\ell` [kg/m] under gravity magnitude
        :math:`g`, this is :math:`\rho_\ell\,g`.
    gravity:
        World-frame gravity 3-vector [m/s^2]. Defaults to the conventional
        :math:`(0, 0, -9.81)`.

    Returns
    -------
    CableSolution
        Force vector in *world* coordinates, scalar tensions at both ends,
        chord and arc lengths, and a sag estimate. The diagnostics dict
        carries the local-frame :math:`(H, V)` plus the Newton solver's
        final residual norm.

    Notes
    -----
    Convergence is generally fast (3--6 Newton steps) when the chord is
    longer than :math:`L_0`. For chords shorter than :math:`L_0` the cable
    is slack; the solver returns a zero-tension solution rather than
    iterating into a meaningless region.
    """
    a_up = np.asarray(anchor_upper, dtype=np.float64)
    a_lo = np.asarray(anchor_lower, dtype=np.float64)
    g = np.asarray(gravity, dtype=np.float64)
    L0 = float(unstretched_length)
    EA = float(axial_stiffness)
    w = float(linear_weight)

    frame = _build_cable_frame(a_up, a_lo, g)
    L_chord = float(np.linalg.norm(a_up - a_lo))

    # Slack short-circuit: a perfectly straight unstretched cable already
    # reaches further than the chord, so no tension can be developed.
    if L_chord <= L0:
        return CableSolution(
            force_on_platform=np.zeros(3),
            tension_lower=0.0, tension_upper=0.0,
            chord_length=L_chord, arc_length=L0,
            sag_max=0.0, axial_strain=(L_chord - L0) / L0,
            is_slack=True,
            diagnostics={"H": 0.0, "V": 0.0, "residual": 0.0},
        )

    guess = _initial_guess(L0, EA, frame.xh, frame.zh)

    def residual_only(HV: NDArray[np.float64]) -> NDArray[np.float64]:
        r, _ = _irvine_residual_and_jac(HV, L0, EA, w, frame.xh, frame.zh)
        return r

    def jacobian_only(HV: NDArray[np.float64]) -> NDArray[np.float64]:
        _, J = _irvine_residual_and_jac(HV, L0, EA, w, frame.xh, frame.zh)
        return J

    sol, info, ier, _msg = fsolve(
        residual_only,
        guess,
        fprime=jacobian_only,
        full_output=True,
        xtol=xtol,
        maxfev=max_iter * 3,
    )
    H, V = float(sol[0]), float(sol[1])
    res_norm = float(np.linalg.norm(info["fvec"]))

    if ier != 1 and res_norm > 1e-4 * max(1.0, L_chord):
        # Solver did not converge to a meaningful tolerance; report slack
        # rather than a misleading "solution". This is the conservative
        # choice for downstream wrench computations.
        return CableSolution(
            force_on_platform=np.zeros(3),
            tension_lower=0.0, tension_upper=0.0,
            chord_length=L_chord, arc_length=L0,
            is_slack=True,
            diagnostics={"H": H, "V": V, "residual": res_norm, "ier": float(ier)},
        )

    T_low = float(np.hypot(H, V))
    T_top = float(np.hypot(H, V + w * L0))

    # Force the cable exerts on the platform, mapped from cable plane to world.
    F_world = frame.to_world(H, V)

    # Sag estimate (peak perpendicular distance from chord). Closed form for a
    # parabolic approximation is (w * L_chord^2) / (8 * H_chord_horizontal),
    # which is accurate to ~1% for shallow sags and a useful diagnostic
    # without sampling the cable shape.
    sag_max = (w * frame.xh**2) / (8.0 * H) if H > _H_FLOOR else 0.0

    # Mean axial strain along the cable (Irvine's exact arc-length expression).
    arc_length = L0 + (1.0 / EA) * (V * L0 + 0.5 * w * L0**2)
    mean_strain = (arc_length - L0) / L0

    return CableSolution(
        force_on_platform=F_world,
        tension_lower=T_low,
        tension_upper=T_top,
        chord_length=L_chord,
        arc_length=arc_length,
        sag_max=float(abs(sag_max)),
        axial_strain=mean_strain,
        is_slack=False,
        diagnostics={"H": H, "V": V, "residual": res_norm},
    )
