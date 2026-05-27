r"""Canonical CDPR geometries.

Four configurations are provided:

* :func:`point_mass_3d` -- a 4-cable, 3-DOF translational robot with anchors
  at the corners of a cube. The simplest CDPR with redundant actuation, and
  the standard pedagogical example.
* :func:`planar_translational` -- a 4-cable, 2-DOF planar robot in the
  :math:`xy`-plane with anchors at the corners of a square. Useful for
  workspace visualisation and quick controller sanity checks.
* :func:`ipanema_class` -- an 8-cable, 6-DOF spatial robot following the
  IPAnema topology: cuboid base frame, smaller cuboid platform, cables
  crossed corner-to-opposite-corner. The standard "real robot" benchmark.
* :func:`cogiro_class` -- a 15 m span large-workspace 8-cable robot in the
  CoGiRo topology, with cable masses tuned so the Irvine sagging model
  produces visible deviations from the chord. Used for sagging-cable
  studies.
"""

from __future__ import annotations

import numpy as np

from cdpr.geometry.robot import CableLimits, CableProperties, PlatformInertia, Robot, RobotGeometry


def point_mass_3d(*, name: str = "point-mass-3d") -> Robot:
    """4-cable 3-DOF translational CDPR with cables in a regular-tetrahedron pattern.

    Cables run from the four vertices of a regular tetrahedron inscribed in
    a unit cube to the platform at the cube's centre. This arrangement is
    fully wrench-closure-capable at the origin --- the four unit vectors
    positively span :math:`\\mathbb{R}^3`. Suspended (all-above-platform)
    geometries are workspace-feasible against gravity but not in WCW at the
    centre, and are the right baseline for :func:`ipanema_class` instead.
    """
    # Four vertices of a regular tetrahedron inscribed in a 2-edge cube.
    anchors = np.array(
        [
            [+1.0, +1.0, +1.0],
            [+1.0, -1.0, -1.0],
            [-1.0, +1.0, -1.0],
            [-1.0, -1.0, +1.0],
        ]
    )
    attachments = np.zeros_like(anchors)
    geometry = RobotGeometry(anchors=anchors, attachments=attachments, dof=3, name=name)
    return Robot(
        geometry=geometry,
        inertia=PlatformInertia.point_mass(mass=1.0),
        limits=CableLimits.uniform(geometry.n_cables, t_min=1.0, t_max=200.0),
    )


def planar_translational(*, side: float = 2.0, name: str = "planar-trans") -> Robot:
    """4-cable 2-DOF planar CDPR; anchors at the corners of a square in the xy-plane."""
    a = side / 2.0
    anchors = np.array(
        [
            [+a, +a, 0.0],
            [-a, +a, 0.0],
            [-a, -a, 0.0],
            [+a, -a, 0.0],
        ]
    )
    attachments = np.zeros_like(anchors)
    geometry = RobotGeometry(anchors=anchors, attachments=attachments, dof=2, name=name)
    return Robot(
        geometry=geometry,
        inertia=PlatformInertia.point_mass(mass=0.5),
        limits=CableLimits.uniform(geometry.n_cables, t_min=1.0, t_max=100.0),
    )


def ipanema_class(*, frame: tuple[float, float, float] = (3.0, 2.0, 2.0),
                  platform: tuple[float, float, float] = (0.5, 0.3, 0.2),
                  name: str = "ipanema-class") -> Robot:
    """8-cable 6-DOF IPAnema-topology CDPR.

    Anchors at the corners of the base cuboid; attachments at the corners of
    the platform cuboid; each cable runs from a base corner to the
    diagonally opposite platform corner --- the cross-connected topology
    that gives the IPAnema family its workspace shape.
    """
    fx, fy, fz = (s / 2 for s in frame)
    px, py, pz = (s / 2 for s in platform)
    # Cables: each base corner attaches to the platform corner with matching
    # horizontal sign but the OPPOSITE vertical sign. This non-symmetric
    # routing avoids the degenerate parallel-to-radial configuration that
    # makes the moment block of the structure matrix vanish at home pose.
    base_corners = np.array(
        [
            [+fx, +fy, +fz], [-fx, +fy, +fz], [-fx, -fy, +fz], [+fx, -fy, +fz],
            [+fx, +fy, -fz], [-fx, +fy, -fz], [-fx, -fy, -fz], [+fx, -fy, -fz],
        ]
    )
    platform_corners = np.array(
        [
            [+px, +py, -pz], [-px, +py, -pz], [-px, -py, -pz], [+px, -py, -pz],
            [+px, +py, +pz], [-px, +py, +pz], [-px, -py, +pz], [+px, -py, +pz],
        ]
    )
    geometry = RobotGeometry(anchors=base_corners, attachments=platform_corners, dof=6, name=name)
    mass = 5.0
    inertia = PlatformInertia(
        mass=mass,
        com=np.zeros(3),
        inertia=np.diag([mass * (py**2 + pz**2) / 3,
                         mass * (px**2 + pz**2) / 3,
                         mass * (px**2 + py**2) / 3]),
    )
    return Robot(
        geometry=geometry,
        inertia=inertia,
        limits=CableLimits.uniform(8, t_min=20.0, t_max=2000.0),
        cable_properties=CableProperties.steel_aircraft_cable(8, diameter_m=3e-3),
    )


def cogiro_class(*, frame: tuple[float, float, float] = (15.0, 11.0, 6.0),
                 platform: tuple[float, float, float] = (1.2, 0.7, 0.4),
                 name: str = "cogiro-class") -> Robot:
    """Large 8-cable 6-DOF CDPR in the CoGiRo topology, with sag-relevant masses."""
    fx, fy, fz = (s / 2 for s in frame)
    px, py, pz = (s / 2 for s in platform)
    base_corners = np.array(
        [
            [+fx, +fy, +fz], [-fx, +fy, +fz], [-fx, -fy, +fz], [+fx, -fy, +fz],
            [+fx, +fy, -fz], [-fx, +fy, -fz], [-fx, -fy, -fz], [+fx, -fy, -fz],
        ]
    )
    platform_corners = np.array(
        [
            [+px, +py, -pz], [-px, +py, -pz], [-px, -py, -pz], [+px, -py, -pz],
            [+px, +py, +pz], [-px, +py, +pz], [-px, -py, +pz], [+px, -py, +pz],
        ]
    )
    geometry = RobotGeometry(anchors=base_corners, attachments=platform_corners, dof=6, name=name)
    mass = 100.0
    inertia = PlatformInertia(
        mass=mass,
        com=np.zeros(3),
        inertia=np.diag([mass * (py**2 + pz**2) / 3,
                         mass * (px**2 + pz**2) / 3,
                         mass * (px**2 + py**2) / 3]),
    )
    return Robot(
        geometry=geometry,
        inertia=inertia,
        limits=CableLimits.uniform(8, t_min=50.0, t_max=20000.0),
        cable_properties=CableProperties.steel_aircraft_cable(8, diameter_m=8e-3),
    )
