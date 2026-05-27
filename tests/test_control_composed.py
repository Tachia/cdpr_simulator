"""FF + FB composition and the inverse-dynamics feedforward."""

from __future__ import annotations

import numpy as np

from cdpr.control import (
    FeedforwardPlusFeedback,
    InverseDynamicsFeedforward,
    PDController,
)
from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def test_inverse_dynamics_feedforward_cancels_gravity(ipanema, home_pose):
    """At zero reference acceleration, the FF term equals minus gravity wrench."""
    ff = InverseDynamicsFeedforward()
    state = PlatformState.at_rest(home_pose)
    g = np.array([0.0, 0.0, -9.81])
    w = ff(
        state=state, reference_pose=home_pose,
        reference_twist=Twist(np.zeros(6)),
        reference_accel=(np.zeros(3), np.zeros(3)),
        t=0.0, robot=ipanema, gravity=g,
        external=Wrench(np.zeros(6)),
    )
    expected_F = -ipanema.inertia.mass * g
    assert np.allclose(w.force, expected_F)
    assert np.allclose(w.torque, 0.0)


def test_composed_controller_sum_property(ipanema, home_pose):
    """FF+FB composition returns the additive sum of the two wrenches."""
    ff = InverseDynamicsFeedforward()
    fb = PDController(Kp_pos=2000.0, Kd_pos=200.0, Kp_rot=50.0, Kd_rot=5.0,
                      gravity_compensation=False)
    composed = FeedforwardPlusFeedback(feedforward=ff, feedback=fb)

    state = PlatformState.at_rest(home_pose)
    g = np.array([0.0, 0.0, -9.81])
    w_ff = ff(
        state=state, reference_pose=home_pose,
        reference_twist=Twist(np.zeros(6)),
        reference_accel=(np.zeros(3), np.zeros(3)),
        t=0.0, robot=ipanema, gravity=g, external=Wrench(np.zeros(6)),
    )
    w_fb = fb(
        state=state, reference_pose=home_pose,
        reference_twist=Twist(np.zeros(6)),
        reference_accel=(np.zeros(3), np.zeros(3)),
        t=0.0, robot=ipanema, gravity=g, external=Wrench(np.zeros(6)),
    )
    w_composed = composed(
        state=state, reference_pose=home_pose,
        reference_twist=Twist(np.zeros(6)),
        reference_accel=(np.zeros(3), np.zeros(3)),
        t=0.0, robot=ipanema, gravity=g, external=Wrench(np.zeros(6)),
    )
    assert np.allclose(w_composed.data, w_ff.data + w_fb.data)


def test_ff_plus_pd_tracks_circle_well(ipanema):
    """FF+FB on a small circle should give clearly better tracking than
    PD alone --- this is the dissertation talking point the composer exists
    for."""
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.1, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=1.0),
    )
    state0 = PlatformState.at_rest(traj.pose(0.0))
    # PD only.
    pd = PDController(Kp_pos=2000.0, Kd_pos=200.0, Kp_rot=50.0, Kd_rot=5.0)
    res_pd = simulate(robot=ipanema, state0=state0, duration=1.0, dt=2e-3,
                      reference=traj, controller=pd)
    # FF + PD (gravity comp on the FF side, PD without it).
    ff = InverseDynamicsFeedforward()
    pd_no_grav = PDController(Kp_pos=2000.0, Kd_pos=200.0,
                              Kp_rot=50.0, Kd_rot=5.0,
                              gravity_compensation=False)
    composed = FeedforwardPlusFeedback(feedforward=ff, feedback=pd_no_grav)
    res_ff_pd = simulate(robot=ipanema, state0=state0, duration=1.0, dt=2e-3,
                         reference=traj, controller=composed)

    ref = np.array([traj(t).position for t in res_pd.time])
    err_pd = np.linalg.norm(res_pd.positions - ref, axis=1).mean()
    err_composed = np.linalg.norm(res_ff_pd.positions - ref, axis=1).mean()
    assert err_composed < err_pd, (
        f"FF+PD ({err_composed:.4e}) should outperform PD ({err_pd:.4e})"
    )
