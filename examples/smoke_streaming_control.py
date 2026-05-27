"""Streaming + closed-loop control demo.

Three runs on the IPAnema-class robot, tracking the same 0.3 m circular
trajectory:

* Open-loop (no controller). The platform gravity-compensates at its
  initial pose and ignores the reference. Tracking error equals the
  reference radius.

* PD pose regulator. Force-per-error gains; works without a model of the
  inertia tensor; tracks within centimetres on this trajectory.

* Computed-torque with feedforward acceleration. Uses the trajectory's
  analytic :math:`\\dot{v}_{\\text{ref}}` and :math:`\\dot{\\omega}_{\\text{ref}}`;
  tracks within fractions of a millimetre.

The third run also exercises :func:`iter_simulation` directly --- the
streaming generator that the live animator and any RL loop will be
consuming.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.spatial.transform import Rotation

from cdpr.control import ComputedTorqueController, PDController
from cdpr.core.frames import Pose
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import iter_simulation, simulate
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def main() -> None:
    robot = ipanema_class()
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.3, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=2.0),
    )
    duration, dt = 2.0, 2e-3

    # All runs start at the trajectory's initial pose so we measure tracking
    # error rather than start-up transient.
    state0 = PlatformState.at_rest(traj.pose(0.0))
    ref_grid = np.array([traj(t).position for t in np.arange(0.0, duration + dt / 2, dt)])

    def tracking_error(positions: np.ndarray) -> tuple[float, float]:
        err = np.linalg.norm(positions - ref_grid[: len(positions)], axis=1)
        return float(err.mean()), float(err.max())

    # --- 1) Open-loop --------------------------------------------------
    open_loop = simulate(
        robot=robot, state0=state0, duration=duration, dt=dt,
        reference=traj, controller=None,
    )
    mean_ol, max_ol = tracking_error(open_loop.positions)
    print(f"[open-loop]      mean = {mean_ol:.4e} m   max = {max_ol:.4e} m")

    # --- 2) PD ---------------------------------------------------------
    pd = PDController(Kp_pos=4000.0, Kd_pos=400.0, Kp_rot=200.0, Kd_rot=20.0)
    pd_run = simulate(
        robot=robot, state0=state0, duration=duration, dt=dt,
        reference=traj, controller=pd,
    )
    mean_pd, max_pd = tracking_error(pd_run.positions)
    print(f"[PD regulator]   mean = {mean_pd:.4e} m   max = {max_pd:.4e} m")

    # --- 3) Computed-torque via the streaming generator ----------------
    omega_n, zeta = 40.0, 1.0
    ct = ComputedTorqueController(
        Kp_pos=omega_n ** 2, Kd_pos=2 * zeta * omega_n,
        Kp_rot=omega_n ** 2, Kd_rot=2 * zeta * omega_n,
    )
    positions: list[np.ndarray] = []
    n_infeasible = 0
    last_t = 0.0
    for sample in iter_simulation(
        robot, state0, duration, dt,
        reference=traj, controller=ct,
    ):
        positions.append(sample.state.pose.position.copy())
        if sample.infeasible:
            n_infeasible += 1
        last_t = sample.time
    positions_arr = np.array(positions)
    mean_ct, max_ct = tracking_error(positions_arr)
    print(f"[computed-torque] mean = {mean_ct:.4e} m   max = {max_ct:.4e} m"
          f"   (yielded {len(positions)} samples, last t = {last_t:.3f} s, "
          f"infeasible = {n_infeasible})")

    # Quick assertions so a regression here is loud.
    assert max_ol > 0.4, "open-loop tracking error should be ~radius"
    assert max_pd < 5e-2, "PD should track to centimetres"
    assert max_ct < 1e-3, "computed-torque should track to sub-mm"
    print("\nAll three modes behaved as expected.")


if __name__ == "__main__":
    main()
