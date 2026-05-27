"""Trajectory composition: paths, scalings, finite-difference derivative checks."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cdpr.trajectory.paths import CircularPath, LinearPath, LissajousPath
from cdpr.trajectory.scaling import LinearScaling, QuinticScaling, TrapezoidalScaling
from cdpr.trajectory.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Time scalings
# ---------------------------------------------------------------------------

def test_quintic_endpoint_conditions():
    s = QuinticScaling(duration=2.0)
    assert s.s(0.0) == 0.0
    assert s.s(2.0) == pytest.approx(1.0)
    assert s.s_dot(0.0) == 0.0
    assert s.s_dot(2.0) == pytest.approx(0.0, abs=1e-12)
    assert s.s_ddot(0.0) == 0.0
    assert s.s_ddot(2.0) == pytest.approx(0.0, abs=1e-12)


def test_quintic_monotonic_interior():
    s = QuinticScaling(duration=1.0)
    ts = np.linspace(0.01, 0.99, 50)
    sds = s.s_dot(ts)
    assert (sds > 0).all()


def test_trapezoidal_velocity_matches_finite_difference():
    sc = TrapezoidalScaling(duration=3.0, accel_time=0.5)
    ts = np.linspace(0.0, 3.0, 1001)
    s_vals = sc.s(ts)
    sd_finite = np.gradient(s_vals, ts)
    sd_analytic = sc.s_dot(ts)
    # Skip phase boundaries (acceleration jumps) AND the two endpoints
    # (np.gradient falls back to one-sided differences there).
    mask = (ts > 0.05) & (ts < 2.95)
    for boundary in (0.5, 2.5):
        mask &= np.abs(ts - boundary) > 0.02
    assert np.allclose(sd_finite[mask], sd_analytic[mask], atol=1e-3)


def test_linear_scaling_basic():
    s = LinearScaling(duration=2.0)
    assert s.s(1.0) == 0.5
    assert s.s_dot(0.5) == 0.5
    assert s.s_ddot(0.5) == 0.0


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def test_linear_path_endpoints():
    p = LinearPath(start=[0, 0, 0], end=[1, 2, 3])
    assert np.allclose(p.pose(0.0).position, [0, 0, 0])
    assert np.allclose(p.pose(1.0).position, [1, 2, 3])
    assert np.allclose(p.pose(0.5).position, [0.5, 1.0, 1.5])


def test_circular_path_revolves_in_plane():
    p = CircularPath(center=[0, 0, 0], radius=1.0, axis=[0, 0, 1])
    assert np.allclose(p.pose(0.0).position, [1.0, 0.0, 0.0])
    assert np.allclose(p.pose(0.25).position, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(p.pose(0.5).position, [-1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(p.pose(1.0).position, [1.0, 0.0, 0.0], atol=1e-12)


def test_circular_path_velocity_perpendicular_to_radius():
    p = CircularPath(center=[0, 0, 0], radius=2.0, axis=[0, 0, 1])
    s = 0.13
    pos = p.pose(s).position
    v, _ = p.velocity(s)
    # Velocity along the path should be tangent (perpendicular to radius in plane).
    assert abs(np.dot(pos, v)) < 1e-12


def test_lissajous_path_evaluable():
    p = LissajousPath(center=[0, 0, 0])
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        pose = p.pose(s)
        assert pose.position.shape == (3,)


# ---------------------------------------------------------------------------
# Trajectory composition
# ---------------------------------------------------------------------------

def test_trajectory_derivative_matches_finite_difference():
    path = LinearPath(start=[0, 0, 0], end=[1, 0, 0])
    scaling = QuinticScaling(duration=2.0)
    traj = Trajectory(path=path, scaling=scaling)
    # Dense grid: with N samples, np.gradient has O(dt^2) error and the
    # quintic's third derivative peaks near ~10 -- aim for ~1e-3 accuracy.
    ts = np.linspace(0.1, 1.9, 401)
    positions = np.array([traj.pose(t).position for t in ts])
    velocities_finite = np.gradient(positions, ts, axis=0)
    velocities_analytic = np.array([traj.twist(t).linear for t in ts])
    # Skip the first and last samples (one-sided gradient).
    assert np.allclose(velocities_finite[2:-2], velocities_analytic[2:-2], atol=2e-4)


def test_trajectory_callable_returns_pose():
    traj = Trajectory(
        path=LinearPath(start=[0, 0, 0], end=[1, 1, 1]),
        scaling=QuinticScaling(duration=1.0),
    )
    pose = traj(0.5)
    assert pose.position.shape == (3,)
    assert isinstance(pose.rotation, Rotation)
