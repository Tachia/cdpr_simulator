"""Computed-torque controller behaviour: feedback linearisation + feedforward."""

from __future__ import annotations

import numpy as np

from cdpr.control import ComputedTorqueController
from cdpr.core.frames import Pose
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def test_computed_torque_tracks_small_circle_below_centimeter(ipanema, home_pose):
    """A computed-torque law with reference acceleration should track a small
    smooth trajectory tightly.

    With Kp = (omega_n)^2, Kd = 2 zeta omega_n, omega_n ~ 30 rad/s,
    the closed-loop bandwidth comfortably exceeds the trajectory's content
    so steady-state tracking error should sit well below the circle radius.
    """
    omega_n = 30.0
    zeta = 1.0
    ct = ComputedTorqueController(
        Kp_pos=omega_n ** 2,
        Kd_pos=2 * zeta * omega_n,
        Kp_rot=omega_n ** 2,
        Kd_rot=2 * zeta * omega_n,
    )
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.05, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=2.0),
    )
    # Match the platform's initial state to the trajectory start so we
    # measure tracking error, not initial-condition mismatch.
    state0 = PlatformState.at_rest(traj.pose(0.0))
    result = simulate(
        robot=ipanema, state0=state0,
        duration=2.0, dt=1e-3,
        reference=traj,
        controller=ct,
    )

    ref_positions = np.array([traj(t).position for t in result.time])
    err = np.linalg.norm(result.positions - ref_positions, axis=1)
    # On a 50 mm circle, computed-torque + feedforward should give
    # sub-millimetre tracking everywhere.
    assert err.max() < 1e-3, f"peak err {err.max():.4e}"
    assert err.mean() < 1e-4, f"mean err {err.mean():.4e}"


def test_computed_torque_holds_against_external_force(ipanema, home_pose):
    """Apply a constant downward 50 N external force; CT should hold position."""
    from cdpr.core.frames import Wrench

    ct = ComputedTorqueController(Kp_pos=400.0, Kd_pos=40.0,
                                  Kp_rot=400.0, Kd_rot=40.0)

    def disturb(state, t):
        return Wrench.from_parts([0.0, 0.0, -50.0], np.zeros(3))

    state0 = PlatformState.at_rest(home_pose)
    result = simulate(
        robot=ipanema, state0=state0,
        duration=0.5, dt=1e-3,
        reference=lambda t: home_pose,
        controller=ct,
        external_wrench=disturb,
    )
    drift = np.linalg.norm(result.positions[-1] - home_pose.position)
    assert drift < 5e-3, f"drift under disturbance: {drift:.4e} m"
