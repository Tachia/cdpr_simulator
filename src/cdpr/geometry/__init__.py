"""CDPR configuration: anchors, platform attachments, robot assembly."""

from cdpr.geometry.robot import (
    CableLimits,
    CableProperties,
    PlatformInertia,
    Robot,
    RobotGeometry,
)

__all__ = [
    "Robot",
    "RobotGeometry",
    "PlatformInertia",
    "CableLimits",
    "CableProperties",
]
