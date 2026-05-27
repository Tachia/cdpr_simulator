"""Streaming generator: shape, equivalence to batch, infeasibility propagation."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import StreamStep, iter_simulation, simulate


def test_iter_simulation_yields_initial_plus_n_steps(ipanema, home_pose):
    duration, dt = 0.02, 1e-3
    n_steps = int(np.ceil(duration / dt))
    state0 = PlatformState.at_rest(home_pose)

    samples = list(iter_simulation(ipanema, state0, duration, dt))
    assert len(samples) == n_steps + 1
    # The first sample is the initial state at t = 0.
    assert samples[0].step == 0
    assert samples[0].time == 0.0
    # Times strictly increase by dt thereafter.
    for k, s in enumerate(samples):
        assert s.step == k
        assert s.time == pytest.approx(k * dt, abs=1e-12)


def test_iter_simulation_matches_simulate_array_for_array(ipanema, home_pose):
    duration, dt = 0.05, 2e-3
    state0 = PlatformState.at_rest(home_pose)

    batch = simulate(ipanema, state0, duration, dt)
    samples = list(iter_simulation(ipanema, state0, duration, dt))

    assert len(samples) == len(batch.time)
    for k, s in enumerate(samples):
        assert np.allclose(s.state.pose.position, batch.positions[k])
        assert np.allclose(s.state.pose.quaternion_xyzw, batch.quaternions_xyzw[k])
        assert np.allclose(s.cable_tensions, batch.cable_tensions[k])
        assert np.allclose(s.cable_lengths, batch.cable_lengths[k])


def test_stream_step_has_pose_and_velocity_accessors(ipanema, home_pose):
    samples = list(iter_simulation(ipanema, PlatformState.at_rest(home_pose),
                                   0.01, 1e-3))
    s = samples[0]
    assert isinstance(s, StreamStep)
    assert s.pose is s.state.pose
    assert s.velocity is s.state.velocity


def test_iter_simulation_infeasible_flag_set_on_singular_pose(point_mass_robot):
    """A pose far outside the WCW should trigger the infeasible recorder."""
    # Move the platform 100 m up; cables can't reach.
    pose = Pose(position=np.array([0.0, 0.0, 100.0]), rotation=Rotation.identity())
    state0 = PlatformState.at_rest(pose)

    samples = list(iter_simulation(point_mass_robot, state0, duration=0.005, dt=1e-3))
    # At least one step (probably all) should be flagged infeasible.
    assert any(s.infeasible for s in samples)
