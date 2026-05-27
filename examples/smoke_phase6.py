"""Phase-6 end-to-end dissertation experiment.

Four threads, each producing artifacts that drop straight into a thesis
chapter or appendix:

1. **Closed-loop benchmark.** Three controllers (PD, computed-torque,
   FF+PD) on an IPAnema-class circle; cdpr core + MuJoCo as the two
   physics backends. Produces tracking / tension / runtime metrics.

2. **Identification.** Apply a known anchor perturbation to a robot,
   generate synthetic ``(pose, length)`` measurements, run the
   :mod:`cdpr.identification` solver, recover the perturbation, write
   a before/after residual report.

3. **Experiment bundle.** Wrap the closed-loop benchmark in an
   :class:`ExperimentConfig`, run it through
   :func:`cdpr.experiments.run_experiment`, produce a reproducible
   bundle with manifest, per-run CSVs, and the full report.

4. **Manifest readback.** Reload the bundle from disk and print the
   summary --- the path a reviewer would take.

Run with:  python examples/smoke_phase6.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.spatial.transform import Rotation

from cdpr.benchmarks import Scenario
from cdpr.control import (
    ComputedTorqueController,
    FeedforwardPlusFeedback,
    InverseDynamicsFeedforward,
    PDController,
)
from cdpr.core.frames import Pose
from cdpr.experiments import ExperimentConfig, load_bundle, run_experiment
from cdpr.geometry.robot import Robot, RobotGeometry
from cdpr.identification import (
    IdentifiableGroup,
    IdentifiableParameters,
    IdentificationProblem,
    apply_result,
    identify,
)
from cdpr.kinematics.inverse import cable_lengths
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def _controller_set():
    pd = PDController(Kp_pos=2000.0, Kd_pos=200.0, Kp_rot=50.0, Kd_rot=5.0)
    ct = ComputedTorqueController(Kp_pos=900.0, Kd_pos=60.0,
                                  Kp_rot=900.0, Kd_rot=60.0)
    ff = InverseDynamicsFeedforward()
    pd_no_grav = PDController(Kp_pos=2000.0, Kd_pos=200.0, Kp_rot=50.0,
                              Kd_rot=5.0, gravity_compensation=False)
    ff_pd = FeedforwardPlusFeedback(feedforward=ff, feedback=pd_no_grav)
    return {"pd": pd, "computed_torque": ct, "ff_plus_pd": ff_pd}


# ---------------------------------------------------------------------------
# 1) Closed-loop benchmark
# ---------------------------------------------------------------------------

def _closed_loop_scenarios(robot):
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.2, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=1.0),
    )
    scenarios: list[Scenario] = []
    for name, controller in _controller_set().items():
        scenarios.append(Scenario(
            name=f"circle_{name}", robot=robot, trajectory=traj,
            controller=controller, duration=1.0, dt=2e-3, seed=1,
            tags={"controller": name, "trajectory": "circle", "radius_m": 0.2},
        ))
    return scenarios


# ---------------------------------------------------------------------------
# 2) Identification
# ---------------------------------------------------------------------------

def _identification_demo(robot, out_root: Path) -> dict:
    rng = np.random.default_rng(13)
    true_da = rng.uniform(-3e-3, 3e-3, size=(robot.n_cables, 3))

    # Build a "true" perturbed robot for synthetic data generation.
    truth_geom = RobotGeometry(
        anchors=robot.anchors + true_da,
        attachments=robot.attachments,
        dof=robot.dof, name="ipanema_truth",
    )
    truth = Robot(geometry=truth_geom, inertia=robot.inertia,
                  limits=robot.limits)

    positions = rng.uniform(-0.25, 0.25, size=(40, 3))
    rotvecs = rng.uniform(-0.1, 0.1, size=(40, 3))
    quats = Rotation.from_rotvec(rotvecs).as_quat()
    L_measured = np.array([
        cable_lengths(Pose(position=p, rotation=Rotation.from_quat(q)), truth)
        for p, q in zip(positions, quats, strict=True)
    ])

    params = IdentifiableParameters(
        groups=(IdentifiableGroup.ANCHOR_OFFSETS,),
        n_cables=robot.n_cables,
    )
    problem = IdentificationProblem(
        robot=robot, parameters=params,
        positions=positions, quaternions_xyzw=quats,
        measured_lengths=L_measured,
    )
    result = identify(problem)
    fitted_robot = apply_result(problem, result)

    da_fit = params.anchor_offsets(result.fitted_vector)
    out = {
        "n_samples": int(positions.shape[0]),
        "initial_residual_rms_m": result.initial_residual_rms,
        "final_residual_rms_m": result.final_residual_rms,
        "initial_residual_peak_m": result.initial_residual_peak,
        "final_residual_peak_m": result.final_residual_peak,
        "iterations": result.n_iterations,
        "converged": result.converged,
        "anchor_offset_rms_error_m": float(
            np.sqrt(np.mean((da_fit - true_da) ** 2))
        ),
        "fitted_robot_name": fitted_robot.name,
    }
    (out_root / "identification.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(out_root: Path = Path("runs/phase6_smoke")) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    robot = ipanema_class()

    # Identification first (fast, self-contained).
    print("[1/4] Identification on synthetic perturbed IPAnema...")
    ident = _identification_demo(robot, out_root)
    print(f"     residual RMS: {ident['initial_residual_rms_m']:.4e} m -> "
          f"{ident['final_residual_rms_m']:.4e} m  "
          f"({ident['iterations']} LM steps, anchor RMS error "
          f"{ident['anchor_offset_rms_error_m']:.4e} m)")

    # Decide which backends to include based on availability.
    backends: list[str] = ["cdpr"]
    try:
        import mujoco  # noqa: F401
        backends.append("mujoco")
    except ImportError:
        print("     (MuJoCo not installed; skipping backend comparison)")

    print(f"[2/4] Closed-loop benchmark across backends {backends}...")
    config = ExperimentConfig(
        name="phase6_dissertation_demo",
        scenarios=_closed_loop_scenarios(robot),
        backends=backends,                                          # type: ignore[arg-type]
        output_root=out_root,
        seed=2026,
        tags={"phase": "6", "trajectory": "circle"},
        notes="Phase 6 end-to-end smoke: PD / CT / FF+PD on cdpr core (and MuJoCo if available).",
        write_full_timeseries=True,
        write_bundle_report=True,
    )

    print("[3/4] Running experiment (with full bundle report)...")
    bundle = run_experiment(config)
    print(f"     bundle root: {bundle.root}")
    print(f"     {len(bundle.run_records)} runs recorded")
    for rec in bundle.run_records:
        m = rec["metrics"]
        print(f"       {rec['id']:60s}  "
              f"track RMS={m['tracking_error_rms']:.3e}  "
              f"peak={m['tracking_error_peak']:.3e}  "
              f"runtime={m['runtime_s']:.3f} s")

    print("[4/4] Reloading bundle from disk (the reviewer path)...")
    reloaded = load_bundle(bundle.root)
    print(f"     loaded {len(reloaded.run_records)} runs; "
          f"report dir = {reloaded.report_dir}")

    print(f"\nDone. Bundle at: {bundle.root.resolve()}")


if __name__ == "__main__":
    main()
