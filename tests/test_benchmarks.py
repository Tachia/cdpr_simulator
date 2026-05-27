"""Scenario hashing, suite runner, multi-backend execution."""

from __future__ import annotations

import numpy as np
import pytest

from cdpr.benchmarks import (
    BenchmarkSuite,
    Scenario,
    run_scenario,
    scenario_hash,
)
from cdpr.control import PDController
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def _make_scenario(robot, *, name: str = "circle"):
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.1, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=0.3),
    )
    return Scenario(
        name=name, robot=robot,
        trajectory=traj,
        controller=PDController(Kp_pos=2000.0, Kd_pos=200.0,
                                Kp_rot=50.0, Kd_rot=5.0),
        duration=0.3, dt=2e-3, seed=42,
    )


def test_scenario_hash_is_deterministic(ipanema):
    s1 = _make_scenario(ipanema)
    s2 = _make_scenario(ipanema)
    assert scenario_hash(s1) == scenario_hash(s2)


def test_scenario_hash_differs_on_seed(ipanema):
    s1 = _make_scenario(ipanema)
    s2 = Scenario(
        name=s1.name, robot=s1.robot, trajectory=s1.trajectory,
        controller=s1.controller, duration=s1.duration, dt=s1.dt,
        seed=s1.seed + 1,
    )
    assert scenario_hash(s1) != scenario_hash(s2)


def test_run_scenario_cdpr_backend_returns_metrics(ipanema):
    run = run_scenario(_make_scenario(ipanema), backend="cdpr")
    assert run.backend == "cdpr"
    assert run.metrics.n_samples > 0
    assert run.metrics.tracking_error_rms < 0.1
    assert run.metrics.runtime_s > 0.0


def test_benchmark_suite_runs_one_scenario_two_backends(ipanema):
    """Smoke: cdpr only (mujoco optional). When MuJoCo is available we run
    both and verify two BenchmarkRun objects are produced."""
    backends: list = ["cdpr"]
    try:
        import mujoco  # noqa: F401
        backends.append("mujoco")
    except ImportError:
        pass

    suite = BenchmarkSuite(
        scenarios=[_make_scenario(ipanema)],
        backends=backends,
    )
    runs = suite.run()
    assert len(runs) == len(backends)
    backend_names = {r.backend for r in runs}
    assert backend_names == set(backends)


def test_metrics_dict_round_trip(ipanema):
    run = run_scenario(_make_scenario(ipanema), backend="cdpr")
    d = run.metrics.to_dict()
    for key in ("tracking_error_rms", "feasibility_rate",
                "condition_number_max", "runtime_s"):
        assert key in d
        assert isinstance(d[key], (int, float))
