r"""Closed-loop benchmark scenarios.

A :class:`Scenario` is a self-contained, immutable description of one
experiment --- the equivalent of "one line in a result table". The
:func:`scenario_hash` helper produces a short SHA-256 digest of the
fully-resolved scenario, suitable as a deterministic identifier for
result directories.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

from cdpr.core.frames import Pose

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.control.base import Controller
    from cdpr.geometry.robot import Robot
    from cdpr.trajectory.trajectory import Trajectory


@dataclass(slots=True, frozen=True)
class Scenario:
    """Immutable benchmark scenario.

    Phase-7 addition: ``cable_model`` selects one of the three exclusive
    constitutive laws (Kelvin--Voigt, Irvine, SQCK hybrid). When set, the
    simulator follows the constitutive evaluation path (rest lengths
    from reference IK, tension from the law); when ``None``, the
    Phase 1-6 tension-distribution path runs unchanged.
    """

    name: str
    robot: "Robot"
    trajectory: "Trajectory | Callable[[float], Pose] | None"
    controller: "Controller | None"
    duration: float
    dt: float = 2e-3
    seed: int = 0
    initial_pose_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    notes: str = ""
    # Free-form tags for grouping results downstream (e.g. backend, controller name).
    tags: dict = field(default_factory=dict)
    # Phase 7: the active constitutive law. Must be a CableModel instance
    # or None (= no constitutive law; default Phase 1-6 path).
    cable_model: "object | None" = None

    # --- derived properties ------------------------------------------

    def initial_pose(self) -> Pose:
        """Where the platform starts: trajectory start + small offset."""
        if self.trajectory is None:
            base = Pose(position=np.zeros(3), rotation=_identity_rotation())
        elif hasattr(self.trajectory, "pose"):
            base = self.trajectory.pose(0.0)
        else:
            base = self.trajectory(0.0)
        offset = np.asarray(self.initial_pose_offset, dtype=np.float64)
        return Pose(position=base.position + offset, rotation=base.rotation)

    def describe(self) -> dict:
        """Stable dict description used for hashing + serialization."""
        return {
            "name": self.name,
            "robot": _robot_description(self.robot),
            "trajectory": _trajectory_description(self.trajectory),
            "controller": _controller_description(self.controller),
            "cable_model": _cable_model_description(self.cable_model),
            "duration": float(self.duration),
            "dt": float(self.dt),
            "seed": int(self.seed),
            "initial_pose_offset": list(self.initial_pose_offset),
            "notes": self.notes,
            "tags": _coerce_jsonable(self.tags),
        }


def _identity_rotation():
    from scipy.spatial.transform import Rotation
    return Rotation.identity()


# ---------------------------------------------------------------------------
# Description helpers
# ---------------------------------------------------------------------------

def _robot_description(robot) -> dict:
    return {
        "name": robot.geometry.name,
        "dof": robot.geometry.dof,
        "n_cables": robot.n_cables,
        "anchors": robot.anchors.tolist(),
        "attachments": robot.attachments.tolist(),
        "mass": float(robot.inertia.mass) if robot.inertia is not None else None,
    }


def _trajectory_description(traj) -> dict:
    if traj is None:
        return {"kind": "none"}
    cls = type(traj).__name__
    out: dict = {"kind": cls}
    # Trajectory composer has .path and .scaling --- recurse one level.
    if hasattr(traj, "path") and hasattr(traj, "scaling"):
        out["path"] = type(traj.path).__name__
        out["scaling"] = type(traj.scaling).__name__
        out["duration"] = float(getattr(traj.scaling, "duration", 0.0))
    return out


def _cable_model_description(model) -> dict:
    """Stable description of an optional constitutive cable model."""
    if model is None:
        return {"mode": "none"}
    return {
        "mode": getattr(model, "mode_name", type(model).__name__),
        "parameters": _coerce_jsonable(getattr(model, "parameters", {})),
    }


def _controller_description(ctrl) -> dict:
    if ctrl is None:
        return {"kind": "none"}
    name = type(ctrl).__name__
    out: dict = {"kind": name}
    # Common scalar/array gain attributes worth recording.
    for attr in ("Kp_pos", "Kd_pos", "Kp_rot", "Kd_rot", "horizon"):
        if hasattr(ctrl, attr):
            val = getattr(ctrl, attr)
            if isinstance(val, np.ndarray):
                out[attr] = val.tolist()
            else:
                out[attr] = float(val) if isinstance(val, (int, float)) else val
    return out


def _coerce_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return repr(obj)


def scenario_hash(scenario: Scenario, length: int = 12) -> str:
    """Short SHA-256 digest of the scenario description.

    Two scenarios with identical physics and identical numerics produce
    the same hash; appears in result directory names so subsequent runs
    overwrite the same artifact set rather than accumulating duplicates.
    """
    payload = json.dumps(scenario.describe(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]
