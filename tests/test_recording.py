"""Recording round-trip and comparison."""

from __future__ import annotations

import json

import numpy as np

from cdpr.recording import (
    compare,
    load_experiment,
    record_simulation,
)
from cdpr.recording.replay import robot_from_snapshot


def test_record_then_load_yields_matching_arrays(short_sim, ipanema, tmp_path):
    log = record_simulation(
        robot=ipanema, result=short_sim, out_dir=tmp_path / "run1",
        title="hold-against-gravity smoke", seed=42,
    )
    assert log.metadata_path.exists()
    assert log.timeseries_path.exists()
    assert log.diagnostics_path.exists()
    assert log.manifest_path.exists()

    exp = load_experiment(tmp_path / "run1")
    assert np.allclose(exp.time, short_sim.time)
    assert np.allclose(exp.positions, short_sim.positions, atol=1e-9)
    assert np.allclose(exp.cable_tensions, short_sim.cable_tensions, atol=1e-9)
    assert np.allclose(exp.cable_lengths, short_sim.cable_lengths, atol=1e-9)


def test_manifest_is_deterministic_when_re_serialised(short_sim, ipanema, tmp_path):
    """Two records with the same seed should produce identical manifest JSON
    *byte-content* except for ``created_at`` (which always advances)."""
    log_a = record_simulation(
        robot=ipanema, result=short_sim, out_dir=tmp_path / "a",
        title="t", seed=7,
    )
    log_b = record_simulation(
        robot=ipanema, result=short_sim, out_dir=tmp_path / "b",
        title="t", seed=7,
    )
    mani_a = json.loads(log_a.manifest_path.read_text())
    mani_b = json.loads(log_b.manifest_path.read_text())
    mani_a.pop("created_at"); mani_b.pop("created_at")
    mani_a.pop("git_revision", None); mani_b.pop("git_revision", None)
    assert mani_a == mani_b


def test_robot_from_snapshot_round_trip(short_sim, ipanema, tmp_path):
    log = record_simulation(
        robot=ipanema, result=short_sim, out_dir=tmp_path / "r",
        title="snapshot test",
    )
    exp = load_experiment(log.root)
    rebuilt = robot_from_snapshot(exp.metadata["robot"])
    assert rebuilt.n_cables == ipanema.n_cables
    assert rebuilt.dof == ipanema.dof
    assert np.allclose(rebuilt.anchors, ipanema.anchors)
    assert np.allclose(rebuilt.attachments, ipanema.attachments)


def test_compare_identical_experiments_yields_zero_error(short_sim, ipanema, tmp_path):
    log = record_simulation(
        robot=ipanema, result=short_sim, out_dir=tmp_path / "same",
        title="compare-itself",
    )
    exp = load_experiment(log.root)
    report = compare(exp, exp)
    assert report.position.rms == 0.0
    assert report.cable_tension.peak == 0.0


def test_compare_shifted_position_has_expected_rms(short_sim, ipanema, tmp_path):
    """Shift one experiment's positions by 0.1 m on x; RMS should equal 0.1."""
    log_a = record_simulation(
        robot=ipanema, result=short_sim, out_dir=tmp_path / "a",
        title="reference", seed=1,
    )
    log_b = record_simulation(
        robot=ipanema, result=short_sim, out_dir=tmp_path / "b",
        title="shifted", seed=1,
    )
    exp_a = load_experiment(log_a.root)
    exp_b = load_experiment(log_b.root)
    exp_b.positions[:] += np.array([0.1, 0.0, 0.0])

    report = compare(exp_a, exp_b)
    assert abs(report.position.rms - 0.1) < 1e-9
