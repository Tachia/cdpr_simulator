r"""Dissertation 8-cable spatial CDPR.

Exact verbatim of the geometry specified in the directive. Lives in the
canonical robots package so every consumer --- the CLI, the API, the
Streamlit GUI, the Phase-2 replay / RL workflows --- sees it as a
first-class robot.

The anchor positions are asymmetric in the horizontal plane (the four
top corners follow one pattern, the four bottom corners follow the
mirror pattern). This produces the criss-cross spatial topology
common in academic CDPR demonstrators.

The platform is a 0.3 m cubic end-effector with attachment points at
the 8 corners ``(\pm 0.15, \pm 0.15, \pm 0.15)``. Default mass is
1 kg with solid-cube inertia ``I = \frac{1}{6} m s^2`` per principal
axis; payload mass is added on top.

Default tension limits are :math:`5 \le T_i \le 500\,\mathrm{N}`. These
are the directive's specification and CAN be overridden via the
``t_min`` / ``t_max`` parameters --- the geometry alone does not care
about the tension bounds.
"""

from __future__ import annotations

import numpy as np

from cdpr.geometry.robot import (
    CableLimits,
    CableProperties,
    PlatformInertia,
    Robot,
    RobotGeometry,
)


# Directive anchor positions (8 fixed points on the frame, m).
_ANCHORS = np.array(
    [
        [-0.535, -0.755, +1.309],
        [+0.755, -0.525, +1.309],
        [-0.755, -0.525,  0.000],
        [+0.535, -0.755,  0.000],
        [-0.755, +0.525, +1.309],
        [+0.535, +0.755, +1.309],
        [-0.535, +0.755,  0.000],
        [+0.755, +0.525,  0.000],
    ],
    dtype=np.float64,
)

# Cubic 0.3 m end-effector --- attachments at (±0.15, ±0.15, ±0.15).
_HALF_SIDE = 0.15
_ATTACHMENTS = np.array(
    [
        [+_HALF_SIDE, +_HALF_SIDE, -_HALF_SIDE],
        [-_HALF_SIDE, +_HALF_SIDE, -_HALF_SIDE],
        [-_HALF_SIDE, -_HALF_SIDE, -_HALF_SIDE],
        [+_HALF_SIDE, -_HALF_SIDE, -_HALF_SIDE],
        [+_HALF_SIDE, +_HALF_SIDE, +_HALF_SIDE],
        [-_HALF_SIDE, +_HALF_SIDE, +_HALF_SIDE],
        [-_HALF_SIDE, -_HALF_SIDE, +_HALF_SIDE],
        [+_HALF_SIDE, -_HALF_SIDE, +_HALF_SIDE],
    ],
    dtype=np.float64,
)


def dissertation_8cable(
    *,
    payload_mass: float = 0.0,
    t_min: float = 5.0,
    t_max: float = 500.0,
    base_mass: float = 1.0,
    cable_diameter_m: float = 3e-3,
    name: str = "dissertation-8cable",
) -> Robot:
    """Build the directive's 8-cable spatial CDPR.

    Parameters
    ----------
    payload_mass:
        Extra mass added to the 1 kg base cube, in kg.
    t_min, t_max:
        Per-cable tension bounds in Newtons. Both must be non-negative and
        ``t_max > t_min``. Defaults reproduce the directive (5-500 N) but
        any values are accepted --- the geometry happily hosts a 50-5000 N
        industrial scenario or a 0.5-50 N tabletop scenario.
    base_mass:
        Mass of the cubic platform itself (excluding payload), in kg.
        Default 1 kg per the directive.
    cable_diameter_m:
        Steel aircraft cable diameter for the material model (relevant
        only for the elastic and sagging cable models, ignored in the
        massless / Kelvin-Voigt / inverse-dynamics paths).
    name:
        Cosmetic label used by the visualiser and reports.
    """
    mass = float(base_mass) + float(payload_mass)
    # Solid cube inertia about the centre of mass.
    side = 2.0 * _HALF_SIDE
    I = (1.0 / 6.0) * mass * side ** 2
    geometry = RobotGeometry(
        anchors=_ANCHORS.copy(),
        attachments=_ATTACHMENTS.copy(),
        dof=6,
        name=name,
    )
    inertia = PlatformInertia(
        mass=mass,
        com=np.zeros(3),
        inertia=np.diag([I, I, I]),
    )
    return Robot(
        geometry=geometry,
        inertia=inertia,
        limits=CableLimits.uniform(8, t_min=float(t_min), t_max=float(t_max)),
        cable_properties=CableProperties.steel_aircraft_cable(8, diameter_m=cable_diameter_m),
    )
