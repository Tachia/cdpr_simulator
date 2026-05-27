"""MPC controller smoke + tracking on a short trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from cdpr.control import MPCController
from cdpr.core.frames import Pose, Wrench
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.trajectory.paths import LinearPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def test_mpc_returns_finite_wrench(ipanema, home_pose):
    """A single call to MPC should produce a finite, correctly-shaped wrench."""
    from cdpr.core.frames import Twist
    mpc = MPCController(horizon=4, dt=5e-3, Q_pos=1e3, Q_vel=1e1,
                        R_force=1e-2, P_terminal=1e4)
    state = PlatformState.at_rest(home_pose)
    w = mpc(
        state=state,
        reference_pose=home_pose,
        reference_twist=Twist(np.zeros(6)),
        reference_accel=None,
        t=0.0, robot=ipanema,
        gravity=np.array([0.0, 0.0, -9.81]),
        external=Wrench(np.zeros(6)),
    )
    assert w.data.shape == (6,)
    assert np.all(np.isfinite(w.data))


def test_mpc_holds_against_gravity(ipanema, home_pose):
    """Static hold under MPC: drift over 0.2 s should stay small."""
    mpc = MPCController(horizon=6, dt=5e-3, Q_pos=2e3, Q_vel=20.0,
                        R_force=1e-3, P_terminal=1e4)
    state0 = PlatformState.at_rest(home_pose)
    result = simulate(
        robot=ipanema, state0=state0,
        duration=0.2, dt=5e-3,
        reference=lambda t: home_pose,
        controller=mpc,
    )
    final_err = np.linalg.norm(result.positions[-1] - home_pose.position)
    assert final_err < 5e-3, f"MPC hold drift {final_err:.4e} m"


def test_mpc_tracks_linear_motion(ipanema, home_pose):
    """A 5 cm linear motion in 0.4 s --- MPC should reduce tracking error
    well below the displacement."""
    target = Pose(position=home_pose.position + np.array([0.05, 0.0, 0.0]),
                  rotation=home_pose.rotation)
    traj = Trajectory(
        path=LinearPath(start=home_pose.position, end=target.position),
        scaling=QuinticScaling(duration=0.4),
    )
    mpc = MPCController(horizon=6, dt=5e-3, Q_pos=4e3, Q_vel=80.0,
                        R_force=1e-3, P_terminal=2e4)
    result = simulate(
        robot=ipanema, state0=PlatformState.at_rest(traj.pose(0.0)),
        duration=0.4, dt=5e-3, reference=traj, controller=mpc,
    )
    ref_positions = np.array([traj(t).position for t in result.time])
    err = np.linalg.norm(result.positions - ref_positions, axis=1)
    assert err.max() < 0.05 * 0.5         # below half the displacement
