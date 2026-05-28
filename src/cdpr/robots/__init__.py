"""Reference CDPR geometries used in examples and tests.

These are canonical configurations rather than calibrated reproductions of
specific published robots --- the anchor / attachment numbers are chosen to
match the topology and scale of well-known designs without claiming
millimetre-accurate parity. For high-fidelity studies, replace the numbers
with the calibration values from your own robot.
"""

from cdpr.robots.catalog import (
    cogiro_class,
    ipanema_class,
    planar_translational,
    point_mass_3d,
)
from cdpr.robots.dissertation import dissertation_8cable

__all__ = [
    "point_mass_3d",
    "planar_translational",
    "ipanema_class",
    "cogiro_class",
    "dissertation_8cable",
]
