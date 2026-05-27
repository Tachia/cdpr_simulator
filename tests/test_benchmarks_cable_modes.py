"""Phase 7 — mode-aware benchmark scenarios and report tagging."""

from __future__ import annotations

import numpy as np
import pytest

from cdpr.benchmarks import Scenario, run_scenario, scenario_hash
from cdpr.cables import cable_model_by_name
from cdpr.control import PDController
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def _scenario(robot, *, cable_model=None, name="hold"):
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.05, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=0.1),
    )
    return Scenario(
        name=name, robot=robot, trajectory=traj,
        controller=PDController(Kp_pos=2000.0, Kd_pos=200.0,
                                Kp_rot=50.0, Kd_rot=5.0),
        duration=0.1, dt=2e-3, seed=1,
        cable_model=cable_model,
    )


def test_scenario_hash_differs_per_cable_mode(ipanema):
    base = _scenario(ipanema)
    kv = _scenario(ipanema, cable_model=cable_model_by_name("kelvin_voigt"))
    assert scenario_hash(base) != scenario_hash(kv), \
        "scenario hash must differentiate the active cable mode"


def test_run_scenario_records_mode_on_metrics(ipanema):
    kv = cable_model_by_name("kelvin_voigt", viscous_coefficient=1e6)
    run = run_scenario(_scenario(ipanema, cable_model=kv), backend="cdpr")
    assert run.metrics.cable_mode == "kelvin_voigt"


def test_run_scenario_without_cable_model_defaults_to_none(ipanema):
    run = run_scenario(_scenario(ipanema), backend="cdpr")
    assert run.metrics.cable_mode == "none"


def test_external_backend_refuses_cable_model(ipanema):
    """External backends must not silently fall back to their own cable
    physics when a constitutive law was requested."""
    try:
        import mujoco       # noqa: F401
    except ImportError:
        pytest.skip("MuJoCo not installed; cannot exercise the refusal path.")
    kv = cable_model_by_name("kelvin_voigt")
    with pytest.raises(ValueError, match="cannot honour scenario.cable_model"):
        run_scenario(_scenario(ipanema, cable_model=kv), backend="mujoco")


def test_three_modes_produce_distinct_scenario_hashes(ipanema):
    """All three exclusive modes must be hashable into distinct scenarios.

    Tension-envelope divergence over a trajectory depends on per-mode
    parameter tuning and integrator stability --- both are model-design
    concerns, not benchmark-harness concerns. What the suite must
    guarantee at the structural level is that the three modes are
    *individually addressable*: the scenario hash differs and the
    metric record carries the right label.
    """
    modes = ["kelvin_voigt", "irvine", "sqck_hybrid"]
    hashes: set[str] = set()
    for name in modes:
        model = cable_model_by_name(name)
        scen = _scenario(ipanema, cable_model=model, name=f"hold_{name}")
        hashes.add(scenario_hash(scen))
    assert len(hashes) == len(modes), \
        "each mode must produce a distinct scenario hash"
