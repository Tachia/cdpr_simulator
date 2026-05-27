r"""Execute an :class:`ExperimentConfig` to disk."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cdpr.adapters import available_backends
from cdpr.benchmarks.suite import BenchmarkRun, run_scenario
from cdpr.experiments.bundle import ExperimentBundle
from cdpr.experiments.config import ExperimentConfig


# ---------------------------------------------------------------------------
# Reproducibility manifest
# ---------------------------------------------------------------------------

def _git_revision(cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, capture_output=True, text=True, cwd=cwd,
        )
        return out.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_manifest(config: ExperimentConfig, root: Path) -> dict:
    from cdpr import __version__ as cdpr_version
    import numpy as _np
    import scipy as _sp
    return {
        "experiment_name": config.name,
        "config_hash": config.config_hash(),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": config.seed,
        "cdpr_version": cdpr_version,
        "python_version": sys.version.split()[0],
        "numpy_version": _np.__version__,
        "scipy_version": _sp.__version__,
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "host": platform.node(),
        "git_revision": _git_revision(root),
        "n_scenarios": len(config.scenarios),
        "backends_requested": list(config.backends),
        "tags": config.tags,
        "notes": config.notes,
    }


# ---------------------------------------------------------------------------
# Per-run serialisation
# ---------------------------------------------------------------------------

def _write_run(run: BenchmarkRun, root: Path, write_full_timeseries: bool) -> Path:
    run_id = f"{run.scenario_name}_{run.backend}_{run.scenario_hash}"
    out = root / "runs" / run_id
    out.mkdir(parents=True, exist_ok=True)

    (out / "metrics.json").write_text(
        json.dumps({
            "scenario_name": run.scenario_name,
            "scenario_hash": run.scenario_hash,
            "backend": run.backend,
            **run.metrics.to_dict(),
        }, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if write_full_timeseries:
        _write_timeseries_csv(out / "timeseries.csv", run)
        _write_reference_csv(out / "reference.csv", run)

    return out


def _write_timeseries_csv(path: Path, run: BenchmarkRun) -> None:
    r = run.result
    n_cables = r.cable_tensions.shape[1]
    header = (
        ["time", "pos_x", "pos_y", "pos_z",
         "quat_x", "quat_y", "quat_z", "quat_w",
         "vel_x", "vel_y", "vel_z",
         "omega_x", "omega_y", "omega_z"]
        + [f"tension_{i + 1}" for i in range(n_cables)]
        + [f"length_{i + 1}" for i in range(n_cables)]
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in range(len(r.time)):
            row = [
                f"{r.time[k]:.10e}",
                *[f"{x:.10e}" for x in r.positions[k]],
                *[f"{x:.10e}" for x in r.quaternions_xyzw[k]],
                *[f"{x:.10e}" for x in r.linear_velocities[k]],
                *[f"{x:.10e}" for x in r.angular_velocities[k]],
                *[f"{x:.10e}" for x in r.cable_tensions[k]],
                *[f"{x:.10e}" for x in r.cable_lengths[k]],
            ]
            w.writerow(row)


def _write_reference_csv(path: Path, run: BenchmarkRun) -> None:
    header = ["time", "ref_x", "ref_y", "ref_z",
              "ref_quat_x", "ref_quat_y", "ref_quat_z", "ref_quat_w"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in range(len(run.result.time)):
            row = [
                f"{run.result.time[k]:.10e}",
                *[f"{x:.10e}" for x in run.reference_positions[k]],
                *[f"{x:.10e}" for x in run.reference_quaternions[k]],
            ]
            w.writerow(row)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_experiment(config: ExperimentConfig) -> ExperimentBundle:
    """Execute every scenario / backend pair and write a reproducible bundle."""
    np.random.seed(config.seed)

    root = config.output_root / f"{config.name}_{config.config_hash()}"
    root.mkdir(parents=True, exist_ok=True)
    runs_dir = root / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Write static metadata up front --- partial failures still leave the
    # manifest pointing at what was intended.
    manifest = _build_manifest(config, root)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    config_path = root / "config.json"
    config_path.write_text(json.dumps(config.describe(), indent=2, sort_keys=True),
                           encoding="utf-8")

    backends_path = root / "backends.json"
    backends_path.write_text(json.dumps(available_backends(), indent=2, sort_keys=True),
                             encoding="utf-8")

    # Execute every (scenario, backend) pair.
    run_records: list[dict] = []
    runs: list[BenchmarkRun] = []
    for scenario in config.scenarios:
        for backend in config.backends:
            run = run_scenario(scenario, backend)
            runs.append(run)
            run_path = _write_run(run, root, config.write_full_timeseries)
            run_records.append({
                "id": run_path.name,
                "metrics": run.metrics.to_dict(),
                "scenario_name": run.scenario_name,
                "backend": run.backend,
            })

    # Top-level metrics summary --- one row per (scenario, backend).
    (root / "metrics_summary.json").write_text(
        json.dumps(run_records, indent=2, sort_keys=True), encoding="utf-8",
    )

    report_dir: Path | None = None
    if config.write_bundle_report:
        from cdpr.reports.bundle import write_bundle_report
        report_dir = root / "report"
        write_bundle_report(runs, report_dir, title=config.name)

    return ExperimentBundle(
        root=root,
        manifest_path=manifest_path,
        config_path=config_path,
        backends_path=backends_path,
        runs_dir=runs_dir,
        report_dir=report_dir,
        run_records=run_records,
    )
