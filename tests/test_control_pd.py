"""PD controller behaviour.

Two pillars:

1. *Without* a controller, the platform under a step disturbance drifts.
2. *With* a PD controller, the platform recovers toward the reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cdpr.control import PDController
from cdpr.core.frames import Pose, Twist
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate


def _displaced_state(home_pose, dx_m: float = 0.05) -> PlatformState:
    p = home_pose.position + np.array([dx_m, 0.0, 0.0])
    pose = Pose(position=p, rotation=home_pose.rotation)
    return PlatformState(pose=pose, velocity=Twist(np.zeros(6)))


def test_pd_returns_wrench_with_right_shape(ipanema, home_pose):
    pd = PDController(Kp_pos=100.0, Kd_pos=10.0, Kp_rot=20.0, Kd_rot=2.0)
    state = _displaced_state(home_pose, dx_m=0.05)
    w = pd(
        state=state, reference_pose=home_pose,
        reference_twist=Twist(np.zeros(6)), reference_accel=None,
        t=0.0, robot=ipanema, gravity=np.array([0, 0, -9.81]),
        external=ipanema.inertia and (
            __import__("cdpr.core.frames", fromlist=["Wrench"]).Wrench(np.zeros(6))
        ),
    )
    assert w.data.shape == (6,)


def test_pd_pulls_displaced_platform_back_toward_reference(ipanema, home_pose):
    """Start displaced, run with PD locked on home; final |p| should shrink."""
    pd = PDController(Kp_pos=2000.0, Kd_pos=200.0, Kp_rot=50.0, Kd_rot=5.0)
    state0 = _displaced_state(home_pose, dx_m=0.05)

    result = simulate(
        robot=ipanema, state0=state0,
        duration=0.5, dt=2e-3,
        reference=lambda t: home_pose,
        controller=pd,
    )
    initial_err = np.linalg.norm(state0.pose.position - home_pose.position)
    final_err = np.linalg.norm(result.positions[-1] - home_pose.position)
    # Recovery: final error at least 80% smaller than the initial step.
    assert final_err < 0.2 * initial_err


def test_simulator_without_controller_does_not_track_moving_reference(
    ipanema, home_pose,
):
    """Sanity baseline: with no controller, the platform ignores a moving reference."""
    state0 = PlatformState.at_rest(home_pose)
    moving = lambda t: Pose(
        position=np.array([0.1 * t, 0.0, 0.0]),
        rotation=Rotation.identity(),
    )
    result = simulate(
        robot=ipanema, state0=state0,
        duration=0.3, dt=2e-3,
        reference=moving,
        controller=None,
    )
    # The platform should remain near origin (gravity-comp hold).
    drift = np.linalg.norm(result.positions[-1])
    assert drift < 1e-3
