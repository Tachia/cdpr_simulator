r"""Phase-7 smoke: three exclusive cable constitutive laws, side by side.

Holds the IPAnema-class platform at the workspace centre under each of
the three mutually exclusive cable models (Kelvin--Voigt, Irvine, SQCK
hybrid) and reports the per-cable tension envelope each law predicts at
that single configuration.

This is the right place to confirm at a glance that:
* the factory builds each mode by name,
* tension statistics differ between modes for the same geometry, and
* nothing leaks across modes (each diagnostics record names exactly one).

A "single-pose comparison" (rather than a closed-loop scenario sweep) is
the most direct dissertation talking point for these constitutive laws
--- the difference between Kelvin--Voigt and Irvine at a given pose is a
pure modelling claim, independent of any controller or integrator.

A second section uses :class:`cdpr.cables.sweep_modes` to run the
comparison across a small simulated trajectory at the cdpr core level,
producing the per-step tension envelope under each mode for the same
state series.

Artifacts (mode comparison JSON, per-mode tension trace plots) land in
``runs/phase7_smoke/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from cdpr.cables import (
    IrvineModel,
    KelvinVoigtModel,
    SQCKHybridModel,
    available_modes,
    cable_model_by_name,
    sweep_modes,
)
from cdpr.control import PDController
from cdpr.core.frames import Pose
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.kinematics.inverse import cable_lengths
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def _summarise(name: str, T: np.ndarray) -> dict:
    return {
        "mode": name,
        "n_cables": int(T.size),
        "tension_min_N": float(T.min()),
        "tension_max_N": float(T.max()),
        "tension_mean_N": float(T.mean()),
    }


def _single_pose_comparison(robot, pose) -> dict:
    """Evaluate each mode at one workspace centre pose."""
    state = PlatformState.at_rest(pose)
    # All three models share a per-cable axial stiffness; for the
    # comparison we pick parameters that produce stable, comparable numbers:
    # softer steel-like properties at moderate self-weight.
    common = dict(
        youngs_modulus=5e9, cross_section=1.0e-5,
    )
    rest = cable_lengths(pose, robot)
    # Apply a small uniform preload by shrinking rest lengths by 1 mm.
    rest_preload = rest - 1e-3

    summaries: dict[str, dict] = {}

    kv = KelvinVoigtModel(**common, viscous_coefficient=1e7)
    T = kv.tension(robot, state, rest_preload)
    summaries["kelvin_voigt"] = _summarise("kelvin_voigt", T)

    irv = IrvineModel(linear_density=0.07, **common)
    T = irv.tension(robot, state, rest_preload)
    summaries["irvine"] = _summarise("irvine", T)

    hyb = SQCKHybridModel(linear_density=0.07, viscous_coefficient=1e7, **common)
    T = hyb.tension(robot, state, rest_preload)
    summaries["sqck_hybrid"] = _summarise("sqck_hybrid", T)

    return summaries


def _mode_overlay_along_trajectory(robot, out_root: Path) -> dict:
    """Run *one* simulation (default tension-distribution path), then sweep
    the three constitutive laws over its recorded state series.

    This is the canonical Phase-7 comparison output: same trajectory, same
    platform state at every step --- only the constitutive law differs.
    The result is a directly comparable per-mode tension envelope.
    """
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.15, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=0.6),
    )
    pd = PDController(Kp_pos=2000.0, Kd_pos=200.0, Kp_rot=50.0, Kd_rot=5.0)
    state0 = PlatformState.at_rest(traj.pose(0.0))
    result = simulate(
        robot=robot, state0=state0,
        duration=0.6, dt=2e-3,
        reference=traj, controller=pd,
    )

    # Reconstruct per-step states + rest lengths from IK on the reference.
    from scipy.spatial.transform import Rotation
    states: list[PlatformState] = []
    from cdpr.core.frames import Twist
    for k in range(len(result.time)):
        states.append(PlatformState(
            pose=Pose(
                position=result.positions[k],
                rotation=Rotation.from_quat(result.quaternions_xyzw[k]),
            ),
            velocity=Twist.from_parts(
                result.linear_velocities[k], result.angular_velocities[k],
            ),
        ))
    rest_history = np.array([
        cable_lengths(traj(t), robot) for t in result.time
    ]) - 1e-3                                            # 1 mm preload

    models = {
        "kelvin_voigt": KelvinVoigtModel(
            youngs_modulus=5e9, cross_section=1e-5, viscous_coefficient=1e7,
        ),
        "irvine": IrvineModel(
            linear_density=0.07, youngs_modulus=5e9, cross_section=1e-5,
        ),
        "sqck_hybrid": SQCKHybridModel(
            linear_density=0.07, youngs_modulus=5e9, cross_section=1e-5,
            viscous_coefficient=1e7,
        ),
    }
    comp = sweep_modes(models, robot, states, rest_history, result.time)

    # Quick mode-overlay plot of mean tension.
    import matplotlib.pyplot as plt
    from cdpr.viz.style import CDPR_CABLE_COLORS, apply_paper_style
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for k, (name, d) in enumerate(comp.modes.items()):
        ax.plot(d.time, d.tension_mean,
                color=CDPR_CABLE_COLORS[k], label=name)
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"mean cable tension $\bar T(t)$ [N]")
    ax.set_title("Three exclusive cable constitutive laws --- same state trajectory")
    ax.legend(loc="best", frameon=False)
    fig_path = out_root / "mode_overlay_mean_tension.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "modes_summary": comp.summary(),
        "figure": str(fig_path.name),
        "n_samples": int(len(result.time)),
    }


def main(out_root: Path = Path("runs/phase7_smoke")) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    robot = ipanema_class()

    print(f"[available modes] {available_modes()}")

    # 1. Single-pose tension envelope under each mode.
    pose0 = Pose(position=np.zeros(3),
                 rotation=__import__("scipy.spatial.transform",
                                     fromlist=["Rotation"]).Rotation.identity())
    print("\n[1/3] Single-pose comparison at the workspace centre...")
    single = _single_pose_comparison(robot, pose0)
    for name in ("kelvin_voigt", "irvine", "sqck_hybrid"):
        s = single[name]
        print(f"   {name:14s} T in [{s['tension_min_N']:.2f}, "
              f"{s['tension_max_N']:.2f}] N, mean {s['tension_mean_N']:.2f}")
    (out_root / "single_pose_comparison.json").write_text(
        json.dumps(single, indent=2), encoding="utf-8",
    )

    # 2. Constitutive-law sweep along a recorded trajectory.
    print("\n[2/3] Cross-mode tension envelope along a tracked circle...")
    overlay = _mode_overlay_along_trajectory(robot, out_root)
    for name, vals in overlay["modes_summary"].items():
        print(f"   {name:14s} tension_mean={vals['tension_mean']:7.2f} N, "
              f"peak={vals['tension_peak']:7.2f} N, slack_steps={vals['slack_steps']}")
    (out_root / "mode_overlay.json").write_text(
        json.dumps(overlay, indent=2), encoding="utf-8",
    )

    # 3. Quick sanity: simulator path records the active mode.
    print("\n[3/3] Simulator records the active mode label...")
    for name in ("kelvin_voigt", "irvine", "sqck_hybrid"):
        model = cable_model_by_name(name)
        result = simulate(
            robot=robot,
            state0=PlatformState.at_rest(pose0),
            duration=0.005, dt=1e-3,
            reference=lambda t: pose0,
            cable_model=model,
        )
        assert result.cable_model_name == name, (
            f"simulator failed to record {name!r}; got {result.cable_model_name!r}"
        )
        print(f"   simulate(cable_model={name!r}) -> cable_model_name={result.cable_model_name!r}")

    print(f"\nDone. Artifacts in: {out_root.resolve()}")


if __name__ == "__main__":
    main()
