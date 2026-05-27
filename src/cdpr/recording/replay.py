r"""Read an on-disk recording back into memory.

The :func:`load_experiment` entry point returns an :class:`Experiment`,
which mirrors the field layout of :class:`cdpr.dynamics.simulator.SimulationResult`
so that downstream code (plotting, animation) can consume either
interchangeably. The :class:`Experiment` also exposes the metadata block
and the reproducibility manifest, which the bare ``SimulationResult``
does not carry.

A separate helper, :func:`robot_from_snapshot`, rebuilds the
:class:`cdpr.geometry.robot.Robot` instance from the JSON snapshot --- this
is what makes recordings genuinely portable. A reader on a different
machine can reconstruct the robot and re-run analyses without needing the
original construction code.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.geometry.robot import Robot


@dataclass(slots=True)
class Experiment:
    """In-memory representation of a recorded experiment.

    The time-series fields are 2-D NumPy arrays indexed as ``[step, axis]``,
    matching :class:`cdpr.dynamics.simulator.SimulationResult` so that the
    same plotting code works for live simulations and for replays.
    """

    time: NDArray[np.float64]                 # (T,)
    positions: NDArray[np.float64]            # (T, 3)
    quaternions_xyzw: NDArray[np.float64]     # (T, 4)
    linear_velocities: NDArray[np.float64]    # (T, 3)
    angular_velocities: NDArray[np.float64]   # (T, 3)
    cable_tensions: NDArray[np.float64]       # (T, m)
    cable_lengths: NDArray[np.float64]        # (T, m)

    condition_numbers: NDArray[np.float64]    # (T,) -- NaN if not recorded
    tension_residuals: NDArray[np.float64]    # (T,) -- NaN if not recorded
    infeasible_steps: list[int]

    metadata: dict[str, object]
    manifest: dict[str, object]
    root: Path


# ---------------------------------------------------------------------------
# Robot reconstruction
# ---------------------------------------------------------------------------

def robot_from_snapshot(snapshot: dict[str, object]) -> "Robot":
    """Rebuild a :class:`Robot` from a :class:`RobotSnapshot` dict."""
    from cdpr.geometry.robot import (
        CableLimits, CableProperties, PlatformInertia, Robot, RobotGeometry,
    )
    geom = RobotGeometry(
        anchors=np.asarray(snapshot["anchors"], dtype=np.float64),
        attachments=np.asarray(snapshot["attachments"], dtype=np.float64),
        dof=int(snapshot["dof"]),
        name=str(snapshot["name"]),
    )
    inertia = None
    if snapshot.get("mass") is not None:
        inertia = PlatformInertia(
            mass=float(snapshot["mass"]),
            inertia=np.asarray(snapshot["inertia"], dtype=np.float64),
        )
    limits = None
    if snapshot.get("t_min") is not None and snapshot.get("t_max") is not None:
        limits = CableLimits(
            t_min=np.asarray(snapshot["t_min"], dtype=np.float64),
            t_max=np.asarray(snapshot["t_max"], dtype=np.float64),
        )
    props = None
    if snapshot.get("cable_youngs_modulus") is not None:
        props = CableProperties(
            youngs_modulus=np.asarray(snapshot["cable_youngs_modulus"], dtype=np.float64),
            cross_section=np.asarray(snapshot["cable_cross_section"], dtype=np.float64),
            linear_density=np.asarray(snapshot["cable_linear_density"], dtype=np.float64),
        )
    return Robot(geometry=geom, inertia=inertia, limits=limits, cable_properties=props)


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def _read_timeseries_csv(path: Path, n_cables: int) -> dict[str, NDArray[np.float64]]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    arr = np.array(rows, dtype=np.float64)
    cols = {name: arr[:, i] for i, name in enumerate(header)}

    def stack(prefix: str, components: list[str]) -> NDArray[np.float64]:
        return np.column_stack([cols[f"{prefix}{c}"] for c in components])

    return {
        "time": cols["time"],
        "positions": stack("pos_", ["x", "y", "z"]),
        "quaternions_xyzw": stack("quat_", ["x", "y", "z", "w"]),
        "linear_velocities": stack("vel_", ["x", "y", "z"]),
        "angular_velocities": stack("omega_", ["x", "y", "z"]),
        "cable_tensions": np.column_stack([cols[f"tension_{i + 1}"] for i in range(n_cables)]),
        "cable_lengths": np.column_stack([cols[f"length_{i + 1}"] for i in range(n_cables)]),
    }


def _read_diagnostics_csv(path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    arr_str = np.array(rows)
    cols = {name: arr_str[:, i] for i, name in enumerate(header)}
    cond = np.asarray(cols.get("condition_number", []), dtype=np.float64) if rows else np.zeros(0)
    resid = np.asarray(cols.get("tension_residual", []), dtype=np.float64) if rows else np.zeros(0)
    flag = np.asarray(cols.get("infeasible", []), dtype=int) if rows else np.zeros(0, dtype=int)
    infeasible_steps = [k for k, v in enumerate(flag) if v]
    return cond, resid, infeasible_steps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_experiment(root: str | Path) -> Experiment:
    """Load a recording directory back into an :class:`Experiment`."""
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    n_cables = int(metadata["robot"]["n_cables"])
    series = _read_timeseries_csv(root / "timeseries.csv", n_cables)
    cond, resid, infeasible = _read_diagnostics_csv(root / "diagnostics.csv")

    return Experiment(
        time=series["time"],
        positions=series["positions"],
        quaternions_xyzw=series["quaternions_xyzw"],
        linear_velocities=series["linear_velocities"],
        angular_velocities=series["angular_velocities"],
        cable_tensions=series["cable_tensions"],
        cable_lengths=series["cable_lengths"],
        condition_numbers=cond,
        tension_residuals=resid,
        infeasible_steps=infeasible,
        metadata=metadata,
        manifest=manifest,
        root=root,
    )
