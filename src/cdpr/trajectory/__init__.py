"""Trajectory generation: geometric paths and time-scalings.

A *path* maps a normalised path parameter :math:`s \\in [0, 1]` to an SE(3)
pose; a *time-scaling* maps the time :math:`t \\in [0, T]` to :math:`s(t)`.
Composing the two gives a callable :math:`t \\mapsto \\mathrm{Pose}` plus its
first and second time derivatives that the dynamics simulator and any
controller can consume directly.

Keeping path geometry and time scaling separate is what lets the same
circular path be retimed (constant velocity, quintic ease-in/out, jerk
limited) without reimplementing the geometry.
"""

from cdpr.trajectory.paths import CircularPath, LinearPath, LissajousPath, Path
from cdpr.trajectory.scaling import (
    LinearScaling,
    QuinticScaling,
    TimeScaling,
    TrapezoidalScaling,
)
from cdpr.trajectory.trajectory import Trajectory

__all__ = [
    "Path",
    "LinearPath",
    "CircularPath",
    "LissajousPath",
    "TimeScaling",
    "LinearScaling",
    "QuinticScaling",
    "TrapezoidalScaling",
    "Trajectory",
]
