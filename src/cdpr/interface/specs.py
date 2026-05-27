r"""Request / specification models shared by the FastAPI and Streamlit fronts.

Both interfaces accept structured "what robot, what trajectory, what
settings" descriptions. Centralising those descriptions here (with
dataclasses --- *not* Pydantic, to keep the import-cost of the core
specifications down) means the API and the GUI cannot drift apart.

The natural-language layer mentioned in the Phase-2 directive is the obvious
next consumer of these specs: it would parse a free-text experiment
description into one of these dataclasses and hand it off to the same
:func:`build_robot` / :func:`build_trajectory` helpers below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


RobotName = Literal["point_mass_3d", "planar_translational", "ipanema_class", "cogiro_class"]
TrajectoryKind = Literal["hold", "line", "circle", "lissajous"]
PlotKind = Literal[
    "position", "velocity", "angular_velocity",
    "cable_lengths", "cable_tensions",
    "tracking_error", "condition_number",
    "trajectory_xy", "trajectory_xz", "trajectory_yz",
]


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrajectorySpec:
    """Generic, kind-discriminated trajectory descriptor.

    ``params`` carries the kind-specific fields (e.g. ``{"start": [...], "end":
    [...]}`` for ``"line"``). The :func:`build_trajectory` helper validates
    and constructs the concrete Phase-1 trajectory object.
    """

    kind: TrajectoryKind = "hold"
    duration: float = 1.0
    params: dict = field(default_factory=dict)


@dataclass(slots=True)
class SimulationRequest:
    """Everything the simulator needs to run a forward CDPR experiment."""

    robot: RobotName = "ipanema_class"
    trajectory: TrajectorySpec = field(default_factory=TrajectorySpec)
    duration: float = 1.0
    dt: float = 2e-3
    integrator: Literal["rk4", "semi_implicit"] = "rk4"
    tension_objective: Literal["min_norm", "centered", "preferred"] = "centered"
    payload_mass: float = 0.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass(slots=True)
class WorkspaceRequest:
    """Workspace scan over a regular position grid."""

    robot: RobotName = "ipanema_class"
    xlim: tuple[float, float] = (-0.6, 0.6)
    ylim: tuple[float, float] = (-0.4, 0.4)
    zlim: tuple[float, float] = (-0.4, 0.4)
    resolution: int = 10
    kind: Literal["wcw", "wfw"] = "wcw"


# ---------------------------------------------------------------------------
# Builders that turn specs into Phase-1 objects
# ---------------------------------------------------------------------------

def build_robot(name: RobotName, *, payload_mass: float = 0.0):
    """Instantiate one of the reference robots; optionally add payload mass."""
    from cdpr.robots import (
        cogiro_class, ipanema_class, planar_translational, point_mass_3d,
    )
    factories = {
        "point_mass_3d": point_mass_3d,
        "planar_translational": planar_translational,
        "ipanema_class": ipanema_class,
        "cogiro_class": cogiro_class,
    }
    if name not in factories:
        raise ValueError(f"Unknown robot name: {name!r}. Choose from {list(factories)}.")
    robot = factories[name]()
    if payload_mass > 0 and robot.inertia is not None:
        from cdpr.geometry.robot import PlatformInertia
        robot.inertia = PlatformInertia(
            mass=robot.inertia.mass + payload_mass,
            com=robot.inertia.com,
            inertia=robot.inertia.inertia,
        )
    return robot


def build_trajectory(spec: TrajectorySpec):
    """Construct a Phase-1 :class:`Trajectory` from a spec.

    The ``hold`` kind returns a constant-pose callable rather than a true
    trajectory, since :func:`cdpr.dynamics.simulator.simulate` accepts any
    ``t -> Pose`` for its ``reference_pose`` argument.
    """
    from scipy.spatial.transform import Rotation
    from cdpr.core.frames import Pose
    from cdpr.trajectory.paths import CircularPath, LinearPath, LissajousPath
    from cdpr.trajectory.scaling import QuinticScaling
    from cdpr.trajectory.trajectory import Trajectory

    if spec.kind == "hold":
        held = np.asarray(spec.params.get("position", [0.0, 0.0, 0.0]), dtype=float)
        rotvec = np.asarray(spec.params.get("rotvec", [0.0, 0.0, 0.0]), dtype=float)
        pose = Pose(position=held, rotation=Rotation.from_rotvec(rotvec))
        return lambda t: pose

    scaling = QuinticScaling(duration=spec.duration)
    if spec.kind == "line":
        path = LinearPath(
            start=spec.params.get("start", [0, 0, 0]),
            end=spec.params.get("end", [0.2, 0, 0]),
        )
    elif spec.kind == "circle":
        path = CircularPath(
            center=spec.params.get("center", [0, 0, 0]),
            radius=float(spec.params.get("radius", 0.2)),
            axis=spec.params.get("axis", [0, 0, 1]),
            angle_span=float(spec.params.get("angle_span", 2 * np.pi)),
        )
    elif spec.kind == "lissajous":
        path = LissajousPath(
            center=spec.params.get("center", [0, 0, 0]),
            amplitudes=spec.params.get("amplitudes", [0.3, 0.2, 0.0]),
            frequencies=spec.params.get("frequencies", [1.0, 2.0, 0.0]),
            phases=spec.params.get("phases", [0.0, np.pi / 2, 0.0]),
        )
    else:
        raise ValueError(f"Unknown trajectory kind: {spec.kind!r}")
    return Trajectory(path=path, scaling=scaling)
