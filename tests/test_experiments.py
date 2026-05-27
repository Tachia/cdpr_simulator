"""Experiment bundle: config hashing, manifest, run record persistence."""

from __future__ import annotations

import json

import numpy as np

from cdpr.benchmarks import Scenario
from cdpr.control import PDController
from cdpr.experiments import ExperimentConfig, load_bundle, run_experiment
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def _config(tmp_path, robot, write_bundle_report=False):
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.1, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=0.2),
    )
    scen = Scenario(
        name="circle", robot=robot, trajectory=traj,
        controller=PDController(Kp_pos=2000.0, Kd_pos=200.0,
                                Kp_rot=50.0, Kd_rot=5.0),
        duration=0.2, dt=2e-3, seed=1,
    )
    return ExperimentConfig(
        name="smoke",
        scenarios=[scen],
        backends=["cdpr"],
        output_root=tmp_path,
        seed=42,
        write_bundle_report=write_bundle_report,
        write_full_timeseries=True,
    )


def test_config_hash_deterministic(ipanema, tmp_path):
    c1 = _config(tmp_path, ipanema)
    c2 = _config(tmp_path, ipanema)
    assert c1.config_hash() == c2.config_hash()


def test_run_experiment_creates_bundle(ipanema, tmp_path):
    bundle = run_experiment(_config(tmp_path, ipanema))
    assert bundle.manifest_path.exists()
    assert bundle.config_path.exists()
    assert bundle.backends_path.exists()
    assert bundle.runs_dir.exists()
    assert len(bundle.run_records) == 1

    manifest = json.loads(bundle.manifest_path.read_text())
    for key in ("experiment_name", "config_hash", "cdpr_version",
                "python_version", "n_scenarios"):
        assert key in manifest


def test_load_bundle_round_trips(ipanema, tmp_path):
    bundle = run_experiment(_config(tmp_path, ipanema))
    loaded = load_bundle(bundle.root)
    assert loaded.root == bundle.root
    assert len(loaded.run_records) == len(bundle.run_records)


def test_full_timeseries_written(ipanema, tmp_path):
    bundle = run_experiment(_config(tmp_path, ipanema))
    run_dir = bundle.runs_dir.iterdir().__next__()
    ts_path = run_dir / "timeseries.csv"
    ref_path = run_dir / "reference.csv"
    assert ts_path.exists()
    assert ref_path.exists()
    # Header sanity check.
    head = ts_path.read_text().splitlines()[0]
    for col in ("time", "pos_x", "quat_w", "tension_1"):
        assert col in head


def test_experiment_with_bundle_report_writes_summary(ipanema, tmp_path):
    bundle = run_experiment(_config(tmp_path, ipanema, write_bundle_report=True))
    assert bundle.report_dir is not None
    assert (bundle.report_dir / "summary.md").exists()
    assert (bundle.report_dir / "tables" / "comparison.csv").exists()
    assert (bundle.report_dir / "tables" / "comparison.tex").exists()
