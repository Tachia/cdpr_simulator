r"""Recorder: turns a Phase-1 :class:`SimulationResult` into an on-disk experiment.

Usage::

    from cdpr.recording import record_simulation, ExperimentLog

    rec = record_simulation(
        robot=ipanema_class(),
        result=sim_result,
        out_dir="runs/2026-05-26_circle-track",
        title="Circle tracking, dt=2ms, centered tension",
        seed=42,
    )
    print(rec.root)   # Path("runs/2026-05-26_circle-track")

The :class:`ExperimentLog` class is also exposed for streaming use --- when
the caller has a controller loop that emits states one at a time and wants
to commit them incrementally.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cdpr.recording.schema import (
    DIAGNOSTIC_COLUMNS,
    ExperimentMetadata,
    ReproducibilityManifest,
    RobotSnapshot,
    SimulationSettings,
)

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.dynamics.simulator import SimulationResult
    from cdpr.geometry.robot import Robot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_revision(cwd: Path) -> str | None:
    """Return the current git revision short hash, if cwd is inside a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True, cwd=cwd,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _snapshot_robot(robot: "Robot") -> RobotSnapshot:
    g = robot.geometry
    snap = RobotSnapshot(
        name=g.name,
        dof=g.dof,
        n_cables=g.n_cables,
        anchors=g.anchors.tolist(),
        attachments=g.attachments.tolist(),
    )
    if robot.inertia is not None:
        snap.mass = float(robot.inertia.mass)
        snap.inertia = robot.inertia.inertia.tolist()
    if robot.limits is not None:
        snap.t_min = robot.limits.t_min.tolist()
        snap.t_max = robot.limits.t_max.tolist()
    if robot.cable_properties is not None:
        snap.cable_youngs_modulus = robot.cable_properties.youngs_modulus.tolist()
        snap.cable_cross_section = robot.cable_properties.cross_section.tolist()
        snap.cable_linear_density = robot.cable_properties.linear_density.tolist()
    return snap


def _make_manifest(seed: int | None, cwd: Path) -> ReproducibilityManifest:
    from cdpr import __version__ as cdpr_version
    import numpy
    import scipy
    return ReproducibilityManifest(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        cdpr_version=cdpr_version,
        python_version=sys.version.split()[0],
        numpy_version=numpy.__version__,
        scipy_version=scipy.__version__,
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        seed=seed,
        git_revision=_git_revision(cwd),
    )


# ---------------------------------------------------------------------------
# Time series and diagnostics CSV writers
# ---------------------------------------------------------------------------

def _timeseries_header(n_cables: int) -> list[str]:
    return (
        ["time",
         "pos_x", "pos_y", "pos_z",
         "quat_x", "quat_y", "quat_z", "quat_w",
         "vel_x", "vel_y", "vel_z",
         "omega_x", "omega_y", "omega_z"]
        + [f"tension_{i + 1}" for i in range(n_cables)]
        + [f"length_{i + 1}" for i in range(n_cables)]
    )


def _write_timeseries(path: Path, result: "SimulationResult") -> None:
    n_cables = result.cable_tensions.shape[1]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_timeseries_header(n_cables))
        for k in range(len(result.time)):
            row = [
                f"{result.time[k]:.10e}",
                *[f"{x:.10e}" for x in result.positions[k]],
                *[f"{x:.10e}" for x in result.quaternions_xyzw[k]],
                *[f"{x:.10e}" for x in result.linear_velocities[k]],
                *[f"{x:.10e}" for x in result.angular_velocities[k]],
                *[f"{x:.10e}" for x in result.cable_tensions[k]],
                *[f"{x:.10e}" for x in result.cable_lengths[k]],
            ]
            w.writerow(row)


def _write_diagnostics(
    path: Path,
    times: NDArray[np.float64],
    *,
    infeasible_steps: list[int],
    condition_numbers: NDArray[np.float64] | None,
    tension_residuals: NDArray[np.float64] | None,
) -> None:
    infeasible = np.zeros(len(times), dtype=bool)
    for k in infeasible_steps:
        if 0 <= k < len(infeasible):
            infeasible[k] = True
    cond = condition_numbers if condition_numbers is not None else np.full_like(times, np.nan)
    resid = tension_residuals if tension_residuals is not None else np.full_like(times, np.nan)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(DIAGNOSTIC_COLUMNS)
        for k in range(len(times)):
            w.writerow([
                f"{times[k]:.10e}",
                f"{cond[k]:.10e}",
                int(infeasible[k]),
                f"{resid[k]:.10e}",
            ])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ExperimentLog:
    """Reference to a recorded experiment on disk."""

    root: Path
    metadata_path: Path
    timeseries_path: Path
    diagnostics_path: Path
    manifest_path: Path


def record_simulation(
    robot: "Robot",
    result: "SimulationResult",
    out_dir: str | Path,
    *,
    title: str,
    tags: dict[str, object] | None = None,
    seed: int | None = None,
    duration: float | None = None,
    dt: float | None = None,
    integrator: str = "rk4",
    tension_objective: str = "centered",
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    reference_trajectory: str | None = None,
    notes: str | None = None,
    condition_numbers: NDArray[np.float64] | None = None,
    tension_residuals: NDArray[np.float64] | None = None,
    mirror_to_supabase: bool | None = None,
    supabase_metrics: dict[str, float] | None = None,
) -> ExperimentLog:
    """Persist a complete simulation as a recording directory.

    ``duration`` and ``dt`` default to values inferred from ``result.time``;
    pass them explicitly when the recorded run came from a custom loop.

    ``condition_numbers`` and ``tension_residuals``, if provided, are
    written into ``diagnostics.csv``. Computing them here is intentionally
    not automatic --- the user may already have them from a controller
    callback, and re-running the structure-matrix SVD for every recorded
    frame would double the cost of a fresh recording.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    metadata_path = root / "metadata.json"
    timeseries_path = root / "timeseries.csv"
    diagnostics_path = root / "diagnostics.csv"
    manifest_path = root / "manifest.json"

    # Derive duration/dt from the result if not given.
    if duration is None:
        duration = float(result.time[-1] - result.time[0]) if len(result.time) > 1 else 0.0
    if dt is None:
        dt = float(result.time[1] - result.time[0]) if len(result.time) > 1 else 0.0

    metadata = ExperimentMetadata(
        experiment_id=str(uuid.uuid4()),
        title=title,
        robot=_snapshot_robot(robot),
        simulation=SimulationSettings(
            duration=duration, dt=dt, integrator=integrator,
            tension_objective=tension_objective, gravity=gravity,
            reference_trajectory=reference_trajectory, notes=notes,
        ),
        tags=dict(tags or {}),
    )
    manifest = _make_manifest(seed=seed, cwd=root)

    # Determinism: sort keys + indent=2 so byte-equal runs produce byte-equal files.
    metadata_path.write_text(json.dumps(metadata.to_dict(), indent=2, sort_keys=True))
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    _write_timeseries(timeseries_path, result)
    _write_diagnostics(
        diagnostics_path,
        result.time,
        infeasible_steps=result.infeasible_steps,
        condition_numbers=condition_numbers,
        tension_residuals=tension_residuals,
    )

    log = ExperimentLog(
        root=root,
        metadata_path=metadata_path,
        timeseries_path=timeseries_path,
        diagnostics_path=diagnostics_path,
        manifest_path=manifest_path,
    )

    # Optional Supabase mirror: writes a single row to the experiments
    # table when SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set in the
    # environment (or when ``mirror_to_supabase=True`` forces an attempt).
    # The local artefact bundle is the source of truth --- a failed
    # mirror never raises and never blocks the return.
    should_mirror = mirror_to_supabase
    if should_mirror is None:
        from cdpr.storage.supabase import supabase_available
        should_mirror = supabase_available()
    if should_mirror:
        try:
            from cdpr.storage.supabase import mirror_experiment
            mirror_experiment(log, metrics=supabase_metrics, extra_tags=dict(tags or {}))
        except Exception:                                       # pragma: no cover - mirror is best-effort
            pass

    return log
