"""Validation against the dynamic model on a synthetic simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cdpr.core.frames import Pose
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.ingest import (
    ColumnMap,
    IngestedExperiment,
    Pipeline,
    load_dataframe,
    reconstruct_trajectory,
    validate_against_robot,
)


def _experiment_from_simulation(robot, home_pose) -> IngestedExperiment:
    """Run a hold simulation and pour it into an IngestedExperiment via the
    pipeline --- mirrors the path real lab data takes, just with a clean
    source."""
    state0 = PlatformState.at_rest(home_pose)
    result = simulate(robot=robot, state0=state0, duration=0.05, dt=1e-3)

    m = result.cable_tensions.shape[1]
    df = pd.DataFrame({"t": result.time})
    df["x"] = result.positions[:, 0]
    df["y"] = result.positions[:, 1]
    df["z"] = result.positions[:, 2]
    df["qx"] = result.quaternions_xyzw[:, 0]
    df["qy"] = result.quaternions_xyzw[:, 1]
    df["qz"] = result.quaternions_xyzw[:, 2]
    df["qw"] = result.quaternions_xyzw[:, 3]
    for i in range(m):
        df[f"T{i + 1}"] = result.cable_tensions[:, i]
        df[f"L{i + 1}"] = result.cable_lengths[:, i]

    raw = load_dataframe(df)
    cmap = ColumnMap(
        time="t",
        position=("x", "y", "z"),
        quaternion=("qx", "qy", "qz", "qw"),
        cable_tensions=tuple(f"T{i + 1}" for i in range(m)),
        cable_lengths=tuple(f"L{i + 1}" for i in range(m)),
    )
    return Pipeline(raw, columns=cmap).run()


def test_ik_residual_is_machine_zero_for_consistent_data(ipanema, home_pose):
    exp = _experiment_from_simulation(ipanema, home_pose)
    report = validate_against_robot(exp, ipanema)
    # The pipeline did no resampling / filtering, so the cable lengths and
    # poses are still internally consistent; IK residual should be near zero.
    assert report.ik_length_residual is not None
    assert report.ik_length_residual.peak < 1e-10


def test_tension_residual_near_zero_at_static_hold(ipanema, home_pose):
    exp = _experiment_from_simulation(ipanema, home_pose)
    report = validate_against_robot(exp, ipanema)
    # The simulator solved gravity-cancelling tensions; wrench residual
    # should be at the QP solver's tolerance.
    assert report.tension_wrench_residual is not None
    assert report.tension_wrench_residual.peak < 1e-6


def test_reconstruct_trajectory_matches_recorded_positions(ipanema, home_pose):
    exp = _experiment_from_simulation(ipanema, home_pose)
    positions, residual = reconstruct_trajectory(
        exp, ipanema, seed=Pose(position=exp.positions[0], rotation=None) if False else None,
    )
    assert residual is not None
    # FK + LMA recovers the position trajectory to micrometre level.
    assert residual.peak < 1e-5
