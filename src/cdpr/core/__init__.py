"""Foundational mathematical primitives shared across the framework.

The submodules here have no dependency on any other ``cdpr`` package; they
implement rigid-body kinematic objects (poses, twists, wrenches), light
numerical helpers, and exception types. Anything CDPR-specific (cable models,
anchors, tension distribution) lives elsewhere.
"""

from cdpr.core.exceptions import (
    ConfigurationError,
    InfeasibleTensionError,
    MissingAdapterDependency,
    SingularConfiguration,
)
from cdpr.core.frames import Pose, Twist, Wrench, hat, vee

__all__ = [
    "Pose",
    "Twist",
    "Wrench",
    "hat",
    "vee",
    "ConfigurationError",
    "InfeasibleTensionError",
    "MissingAdapterDependency",
    "SingularConfiguration",
]
