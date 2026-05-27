"""Rigid-body dynamics and time integration."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.dynamics.integrators import rk4_step, semi_implicit_step
from cdpr.dynamics.rigid_body import PlatformState, rigid_body_acceleration
from cdpr.dynamics.simulator import simulate
from cdpr.geometry.robot import PlatformInertia


def test_acceleration_from_gravity_only():
    inertia = PlatformInertia(mass=2.0)
    pose = Pose(position=np.zeros(3), rotation=Rotation.identity())
    state = PlatformState.at_rest(pose)
    w = Wrench.gravity(mass=2.0, g=(0.0, 0.0, -9.81))
    a_lin, a_ang = rigid_body_acceleration(state, w, inertia)
    assert np.allclose(a_lin, [0, 0, -9.81])
    assert np.allclose(a_ang, 0.0)


def test_rk4_free_fall_matches_analytic():
    """For a free body in gravity, position after t equals p0 + v0 t + 0.5 g t^2."""
    inertia = PlatformInertia(mass=1.0)
    pose0 = Pose(position=np.zeros(3), rotation=Rotation.identity())
    state = PlatformState.at_rest(pose0)
    g = -9.81
    wrench_fn = lambda s, t: Wrench.from_parts([0, 0, g], np.zeros(3))

    dt = 1e-3
    n_steps = 1000
    t_final = n_steps * dt
    for _ in range(n_steps):
        state = rk4_step(state, 0.0, dt, wrench_fn, inertia)
    expected_z = 0.5 * g * t_final**2
    assert state.pose.position[2] == pytest.approx(expected_z, rel=1e-6)
    assert state.velocity.linear[2] == pytest.approx(g * t_final, rel=1e-6)


def test_semi_implicit_pure_rotation_keeps_rotation_in_so3():
    """Free spin with constant angular velocity must keep R orthogonal."""
    inertia = PlatformInertia(mass=1.0, inertia=np.eye(3))
    pose = Pose(position=np.zeros(3), rotation=Rotation.identity())
    omega = np.array([0.0, 0.0, 1.0])
    state = PlatformState(pose=pose, velocity=Twist.from_parts(np.zeros(3), omega))
    wrench_fn = lambda s, t: Wrench(np.zeros(6))  # no external wrench

    for _ in range(500):
        state = semi_implicit_step(state, 0.0, 0.01, wrench_fn, inertia)

    R = state.pose.rotation.as_matrix()
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_simulate_holds_static_equilibrium(point_mass_robot):
    """A platform commanded to its starting pose should stay there (within ms)."""
    pose0 = Pose(position=np.array([0.0, 0.0, 0.0]), rotation=Rotation.identity())
    state0 = PlatformState.at_rest(pose0)

    result = simulate(
        point_mass_robot,
        state0,
        duration=0.05,
        dt=1e-3,
        reference_pose=lambda t: pose0,
    )
    # Final position should still be near origin (gravity is cancelled by cable tensions).
    assert np.linalg.norm(result.positions[-1] - pose0.position) < 1e-3
    assert (result.cable_tensions[-1] > 0).all()
    assert not result.infeasible_steps


def test_rk4_and_semi_implicit_agree_to_first_order():
    """At small dt, RK4 and semi-implicit should agree on a smooth problem.

    The position drift between the two methods for constant gravity is
    g*t*dt/2 (RK4 is exact for the quadratic; semi-implicit accumulates a
    half-step bias). At dt=1e-4 over t=0.01 s with g~10, the bias is ~5e-6 m,
    so the allowed tolerance is comfortably above this.
    """
    inertia = PlatformInertia(mass=1.0)
    pose = Pose(position=np.zeros(3), rotation=Rotation.identity())
    state_a = PlatformState.at_rest(pose)
    state_b = PlatformState.at_rest(pose)
    wrench_fn = lambda s, t: Wrench.from_parts([0, 0, -9.81], np.zeros(3))

    dt = 1e-4
    for _ in range(100):
        state_a = rk4_step(state_a, 0.0, dt, wrench_fn, inertia)
        state_b = semi_implicit_step(state_b, 0.0, dt, wrench_fn, inertia)
    assert np.allclose(state_a.pose.position, state_b.pose.position, atol=1e-4)
