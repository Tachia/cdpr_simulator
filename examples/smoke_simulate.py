"""Phase 1 end-to-end smoke run.

Exercises kinematics + statics + dynamics + trajectory + workspace on the
IPAnema-class reference robot. Useful as a one-command "does the whole stack
still work" check; the unit tests cover each layer individually.

Run with:  python examples/smoke_simulate.py
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose, Wrench
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.kinematics.inverse import cable_lengths
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory
from cdpr.workspace.feasible import is_in_wfw


def main() -> None:
    robot = ipanema_class()
    pose0 = Pose(position=np.zeros(3), rotation=Rotation.identity())
    state0 = PlatformState.at_rest(pose0)

    # 1) Open-loop static hold against gravity (commanded pose == platform pose).
    hold = simulate(robot, state0, duration=0.05, dt=1e-3, reference_pose=lambda t: pose0)
    print(f"[hold] final position drift:  {np.linalg.norm(hold.positions[-1] - pose0.position):.2e} m")
    print(f"[hold] cable tensions in [{hold.cable_tensions[-1].min():.1f}, "
          f"{hold.cable_tensions[-1].max():.1f}] N")

    # 2) Trajectory inverse-kinematics sweep (no dynamic integration -- IK only).
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.4, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=2.0),
    )
    ts = np.linspace(0.0, 2.0, 41)
    lengths_by_t = np.array([cable_lengths(traj(t), robot) for t in ts])
    print(f"[traj] cable lengths swept across circle:  "
          f"min={lengths_by_t.min():.3f}, max={lengths_by_t.max():.3f} m")

    # 3) Wrench-feasibility sample along the same trajectory.
    gravity = Wrench.from_parts([0.0, 0.0, -robot.inertia.mass * 9.81], np.zeros(3))
    feasible = sum(int(is_in_wfw(traj(t), robot, gravity)) for t in ts)
    print(f"[wfw]  feasible against gravity at {feasible}/{len(ts)} sampled poses")


if __name__ == "__main__":
    main()
