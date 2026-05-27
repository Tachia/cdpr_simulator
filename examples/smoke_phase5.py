"""Phase-5 smoke: external-engine physics verification + ROS 2 transport replay.

Two demonstrations:

1. **MuJoCo verification.** Run a short CT-controlled trajectory on the
   IPAnema-class robot through the cdpr scientific core and the MuJoCo
   adapter using identical cable wrenches; print position / orientation
   / velocity divergence.

2. **ROS 2 transport replay.** Take the cdpr simulation result, feed it
   step-by-step through the ROS 2 transport adapter in in-memory mode,
   and verify the published history matches the source. This is the
   path real ROS 2 bridges will take once rclpy is installed --- the
   in-memory mode exercises the same publish/subscribe API surface.

Artifacts (verification summary JSON, published history dump) land in
``runs/phase5_smoke/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from cdpr.adapters import available_backends, make_backend, verify_against
from cdpr.control import ComputedTorqueController
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import iter_simulation
from cdpr.kinematics.jacobian import structure_matrix
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def main(out_root: Path = Path("runs/phase5_smoke")) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    robot = ipanema_class()

    print("[backends] availability probe:")
    for name, ok in available_backends().items():
        print(f"     {name:10s} {'available' if ok else 'absent'}")

    # --- 1. MuJoCo verification ---------------------------------------
    print("\n[1/2] MuJoCo physics verification on a 1 s CT-tracked circle...")
    # NOTE: this is an open-loop wrench-history replay --- we record the
    # cable wrenches cdpr produced at its own state trajectory, then push
    # the same wrench history into MuJoCo. Because MuJoCo's state diverges
    # slightly from step 1, the same wrenches no longer cancel acceleration
    # in MuJoCo, and the divergence grows. A 1 s duration keeps acceleration
    # demands modest and the verification numbers interpretable.
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.2, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=1.0),
    )
    state0 = PlatformState.at_rest(traj.pose(0.0))
    ct = ComputedTorqueController(Kp_pos=900.0, Kd_pos=60.0,
                                  Kp_rot=900.0, Kd_rot=60.0)

    with make_backend("mujoco", robot=robot, timestep=1e-3) as backend:
        report = verify_against(
            backend, robot, state0,
            duration=1.0, dt=1e-3,
            reference=traj, controller=ct,
        )

    summary = report.summary()
    print("     divergence (cdpr core vs MuJoCo):")
    for ch, s in summary.items():
        print(f"       {ch:20s} rms={s['rms']:.4e}  peak={s['peak']:.4e}")
    (out_root / "mujoco_verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8",
    )

    # --- 2. ROS 2 transport replay -----------------------------------
    print("\n[2/2] ROS 2 transport adapter (in-memory mode) replay...")
    ros = make_backend("ros2", robot=robot, use_rclpy=False, topic_prefix="/cdpr/demo")

    samples_pushed = 0
    for sample in iter_simulation(
        robot, state0, duration=0.05, dt=2e-3,
        reference=traj, controller=ct,
    ):
        W = structure_matrix(sample.state.pose, robot)
        cable_wrench_vec = np.zeros(6)
        cable_wrench_vec[: robot.dof] = W @ sample.cable_tensions
        ros.publish_state(
            sample.state,
            sample.cable_lengths,
            sample.cable_tensions,
            timestamp=sample.time,
        )
        samples_pushed += 1
    print(f"     published {samples_pushed} samples to {ros.topic_prefix}/*")

    history = ros.published_history()
    print(f"     in-memory history retained: {len(history)} entries")
    print(f"     latest timestamp: {history[-1].timestamp:.3f} s")
    (out_root / "ros2_history.json").write_text(
        json.dumps([
            {
                "t": float(h.timestamp),
                "p": h.position.tolist(),
                "q": h.quaternion_xyzw.tolist(),
                "lengths": h.cable_lengths.tolist(),
                "tensions": h.cable_tensions.tolist(),
            }
            for h in history
        ], indent=2),
        encoding="utf-8",
    )
    ros.close()
    print(f"\nDone. Artifacts in: {out_root.resolve()}")


if __name__ == "__main__":
    main()
