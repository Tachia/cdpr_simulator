"""Cable-Driven Parallel Robot computational framework.

The top-level package re-exports the most frequently used primitives so callers
can write::

    from cdpr import Pose, Robot, inverse_kinematics, structure_matrix

without having to know the internal layout. Less-used names live in their
submodules and should be imported from there.
"""

from __future__ import annotations

from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.geometry.robot import Robot
from cdpr.kinematics.inverse import inverse_kinematics, cable_lengths
from cdpr.kinematics.jacobian import structure_matrix

__all__ = [
    "Pose",
    "Twist",
    "Wrench",
    "Robot",
    "inverse_kinematics",
    "cable_lengths",
    "structure_matrix",
]

__version__ = "0.1.0.dev0"
