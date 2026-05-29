r"""Built-in CDPR_SIMULATOR examples — five selectable demonstrations.

The list below is the single source of truth for both the PowerShell
CLI (:mod:`scripts.run_example`) and the Streamlit GUI's example
selector. Adding a new example here makes it discoverable in both
interfaces without touching either.

Phase 1 — physics-based motion simulation
-----------------------------------------

* ``circle``   — 3-DoF planar circular tracking on the directive robot.
* ``spiral``   — **6-DoF** helical sweep with continuous yaw + pitch
                 oscillation, demonstrating full SE(3) tracking.
* ``mshape``   — Pick-and-place letter-M task with vertex dwells and
                 discrete gripper yaw at pick / place events.

Phase 2 — data-driven training / analysis
------------------------------------------

* ``train``    — Single-model PINN fit on a Phase-1 CSV (auto-runs
                 ``circle`` first when no input is supplied).
* ``compare``  — Multi-model bench: replay vs MLP vs PINN vs PPO vs SAC
                 on the same dataset.

Each entry carries everything the runner needs: a phase id, a human-
readable title and description, the expected output directory, and
either a ``runner`` callable (Phase 1) or a ``deps + subprocess argv``
recipe (Phase 2).
"""

from __future__ import annotations

import json
import os
import platform as _platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Headless plotting before any matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np                                                  # noqa: E402
from scipy.spatial.transform import Rotation                        # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Local helpers (shared with the train/compare scripts).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _csv_io import RobotSpec                                       # noqa: E402

from cdpr.core.frames import Pose                                   # noqa: E402
from cdpr.control.pd import PDController                            # noqa: E402
from cdpr.dynamics.rigid_body import PlatformState                  # noqa: E402
from cdpr.dynamics.simulator import simulate                        # noqa: E402
from cdpr.robots.dissertation import dissertation_8cable            # noqa: E402


# ---------------------------------------------------------------------------
# Custom reference trajectories used by the spiral and M-shape examples.
# The simulator accepts any callable ``t -> Pose`` as a reference, so these
# do not need to be wrapped in the heavier Trajectory class — at the cost
# of zero feedforward velocity (PD still tracks with realistic lag, which
# is itself a useful demonstration).
# ---------------------------------------------------------------------------

def helical6_pose(
    t: float,
    *,
    duration: float = 15.0,
    r_start: float = 0.015,
    r_end: float = 0.035,
    z_center: float = 0.65,
    z_amp: float = 0.04,
    n_revs: float = 3.0,
    yaw_amp_rad: float = 0.30,                                      # ±17°
    pitch_amp_rad: float = 0.10,                                    # ±6°
    roll_amp_rad: float = 0.05,                                     # ±3°
) -> Pose:
    """6-DoF helical reference: position spirals outward + up while the
    platform tilts on all three Euler axes.

    Orientation amplitudes are **bounded sinusoids** rather than
    monotonic rotations --- without an orientation feedforward, PD
    cannot track an angle that keeps growing; bounded oscillation lets
    the angular error stay inside the cables' wrench-feasibility
    envelope while still exercising all six rigid-body DoFs.
    """
    s = float(np.clip(t / duration, 0.0, 1.0))
    s_smooth = 6 * s ** 5 - 15 * s ** 4 + 10 * s ** 3
    r = r_start + (r_end - r_start) * s_smooth
    z = z_center - z_amp + 2 * z_amp * s_smooth
    theta = 2.0 * np.pi * n_revs * s_smooth
    pos = np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=np.float64)
    # Bounded orientation oscillation --- one full back-and-forth per
    # revolution of the spiral, phase-shifted between axes to keep the
    # demonstration visually rich.
    yaw   = yaw_amp_rad   * np.sin(theta)
    pitch = pitch_amp_rad * np.sin(theta + np.pi / 3)
    roll  = roll_amp_rad  * np.sin(theta + 2 * np.pi / 3)
    rot = Rotation.from_euler("xyz", [roll, pitch, yaw])
    return Pose(position=pos, rotation=rot)


def mshape_pose(
    t: float,
    *,
    vertices: list[tuple[float, float, float]] | None = None,
    segment_time: float = 1.8,
    dwell_time: float = 0.7,
    gripper_yaw_amp: float = 0.52,                                  # ~30°
) -> Pose:
    """Pick-and-place letter-M reference with explicit dwells at each
    vertex and a discrete gripper yaw during the dwell (simulating
    pick / place actuation)."""
    if vertices is None:
        # Five vertices inside the dissertation robot's feasible
        # workspace (r ≲ 0.05 m around z = 0.65 m).
        vertices = [
            (-0.030, -0.025, 0.650),  # V1 bottom-left (start)
            (-0.030, +0.025, 0.650),  # V2 top-left
            ( 0.000, -0.010, 0.650),  # V3 middle dip
            (+0.030, +0.025, 0.650),  # V4 top-right
            (+0.030, -0.025, 0.650),  # V5 bottom-right (end)
        ]
    V = [np.asarray(v, dtype=np.float64) for v in vertices]
    n_segs = len(V) - 1
    cycle = segment_time + dwell_time
    total = dwell_time + n_segs * cycle

    t = float(np.clip(t, 0.0, total))
    if t < dwell_time:
        # Initial dwell at V[0] --- "approach pick".
        return Pose(position=V[0], rotation=Rotation.identity())

    elapsed = t - dwell_time
    seg_idx = int(elapsed // cycle)
    if seg_idx >= n_segs:
        return Pose(position=V[-1], rotation=Rotation.identity())
    local = elapsed - seg_idx * cycle
    v_start, v_end = V[seg_idx], V[seg_idx + 1]

    if local < segment_time:
        # Motion phase with quintic smoothing.
        s = local / segment_time
        s_smooth = 6 * s ** 5 - 15 * s ** 4 + 10 * s ** 3
        pos = v_start + (v_end - v_start) * s_smooth
        yaw = 0.0
    else:
        # Dwell at v_end with a sinusoidal gripper-yaw "pick" gesture
        # to demonstrate orientation actuation under load.
        pos = v_end
        u = (local - segment_time) / dwell_time
        yaw = gripper_yaw_amp * np.sin(np.pi * u)
    return Pose(position=pos, rotation=Rotation.from_euler("xyz", [0.0, 0.0, yaw]))


def mshape_total_duration(*, segment_time: float = 1.8, dwell_time: float = 0.7,
                          n_vertices: int = 5) -> float:
    return dwell_time + (n_vertices - 1) * (segment_time + dwell_time)


# ---------------------------------------------------------------------------
# Plot bundle --- the 13-figure pack the directive expects from each
# Phase-1 run. Lives here (instead of being imported from run_simulation.py)
# so the examples module is fully self-contained.
# ---------------------------------------------------------------------------

def _render_phase1_plots(out_dir: Path, result, robot, reference):
    try:
        from cdpr.viz.style import apply_paper_style
        apply_paper_style()
    except Exception:                                              # pragma: no cover
        pass
    import matplotlib.pyplot as plt
    from cdpr.viz import plots2d
    from cdpr.viz.scene import SceneOptions, render_scene

    figs_made: list[str] = []

    # Hard cap on matplotlib canvas dimensions. The directive's max is
    # 10000 x 10000; we go below that to leave margin for tight_layout.
    MAX_PIXELS = 8000

    def _save(name: str, fn) -> None:
        path = out_dir / f"{name}.png"
        try:
            fig = fn()
            # Validate canvas size BEFORE savefig. Matplotlib crashes
            # with cryptic 'image size too large' errors when figsize x
            # dpi exceeds a few hundred million pixels --- which can
            # happen when somebody passes log-scaled outliers or a 12 h
            # sim through a per-step plot.
            dpi = 160
            w_in, h_in = fig.get_size_inches()
            w_px, h_px = int(w_in * dpi), int(h_in * dpi)
            if w_px > MAX_PIXELS or h_px > MAX_PIXELS:
                # Downscale dpi proportionally rather than dropping the
                # figure entirely.
                scale = MAX_PIXELS / max(w_px, h_px)
                dpi = max(50, int(dpi * scale))
                print(f"  [{name:22s}] WARN: canvas {w_px}x{h_px} > {MAX_PIXELS}, "
                      f"reducing dpi to {dpi}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            figs_made.append(name)
            print(f"  [{name:22s}] -> {path.resolve()}")
        except Exception as exc:
            # Defensive: even with the size guard, never let one plot
            # bring down the others. The directive says: never crash.
            try:
                plt.close("all")
            except Exception:                                       # pragma: no cover
                pass
            print(f"  [{name:22s}] FAILED  {type(exc).__name__}: {exc}")

    _save("position",         lambda: plots2d.plot_position(result))
    _save("velocity",         lambda: plots2d.plot_velocity(result))
    _save("angular_velocity", lambda: plots2d.plot_angular_velocity(result))

    def _accel_plot():
        t = np.asarray(result.time); v = np.asarray(result.linear_velocities)
        a = np.gradient(v, t, axis=0)
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for k, lbl in enumerate(("a_x", "a_y", "a_z")):
            ax.plot(t, a[:, k], label=fr"${lbl}$")
        ax.set_xlabel(r"time $t$ [s]")
        ax.set_ylabel(r"acceleration [m s$^{-2}$]")
        ax.legend(loc="best"); ax.set_title("Translational acceleration")
        return fig
    _save("acceleration", _accel_plot)

    _save("cable_tensions",   lambda: plots2d.plot_cable_tensions(result, robot=robot))
    _save("cable_lengths",    lambda: plots2d.plot_cable_lengths(result))

    def _stretch_plot():
        L = np.asarray(result.cable_lengths)
        head = max(1, int(0.01 * L.shape[0])); L0 = L[:head].mean(axis=0)
        d = L - L0[None, :]
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for i in range(L.shape[1]):
            ax.plot(result.time, d[:, i] * 1e3, label=f"cable {i+1}")
        ax.set_xlabel(r"time $t$ [s]")
        ax.set_ylabel(r"stretch $\Delta L$ [mm]")
        ax.legend(loc="best", ncol=2, fontsize=8)
        ax.set_title("Per-cable elongation vs initial length")
        return fig
    _save("cable_stretch", _stretch_plot)

    if reference is not None:
        _save("tracking_error", lambda: plots2d.plot_tracking_error(result, reference))

        def _rms_plot():
            t = np.asarray(result.time)
            ref = np.asarray([reference(tt).position for tt in t])
            err = np.linalg.norm(np.asarray(result.positions) - ref, axis=1)
            sq_cumsum = np.cumsum(err ** 2)
            rms = np.sqrt(sq_cumsum / np.arange(1, len(err) + 1))
            fig, ax = plt.subplots(figsize=(6.0, 3.2))
            ax.plot(t, rms * 1e3, color="C3")
            ax.set_xlabel(r"time $t$ [s]")
            ax.set_ylabel("cumulative RMS error [mm]")
            ax.set_title("Tracking error — cumulative RMS evolution")
            return fig
        _save("rms_error_evolution", _rms_plot)

    _save("condition_number", lambda: plots2d.plot_condition_number(result, robot))

    if reference is not None:
        ref_xyz = np.asarray([reference(t).position for t in result.time])
    else:
        ref_xyz = None
    for plane in ("xy", "xz", "yz"):
        _save(
            f"trajectory_{plane}",
            lambda p=plane: plots2d.plot_trajectory_projection(
                result.positions, plane=p, reference=ref_xyz,
            ),
        )

    # Orientation evolution (especially relevant for the 6-DoF spiral
    # and the M-shape's gripper actuation).
    def _orientation_plot():
        from scipy.spatial.transform import Rotation as _R
        quat = np.asarray(result.quaternions_xyzw)
        eul = _R.from_quat(quat).as_euler("xyz", degrees=True)
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for k, lbl in enumerate(("roll", "pitch", "yaw")):
            ax.plot(result.time, eul[:, k], label=lbl)
        ax.set_xlabel(r"time $t$ [s]")
        ax.set_ylabel("orientation [deg]")
        ax.legend(loc="best")
        ax.set_title("End-effector orientation evolution")
        return fig
    _save("orientation", _orientation_plot)

    _save("scene_3d", lambda: render_scene(
        robot,
        Pose(
            position=result.positions[-1],
            rotation=Rotation.from_quat(result.quaternions_xyzw[-1]),
        ),
        options=SceneOptions(tension_heatmap=True),
        tensions=result.cable_tensions[-1],
        trajectory_positions=result.positions,
    ))

    return figs_made


# ---------------------------------------------------------------------------
# CSV writer (mirror of scripts/run_simulation.save_timeseries_csv, kept
# in sync so the Phase-2 examples can consume the output directly).
# ---------------------------------------------------------------------------

def _save_csv(out_dir: Path, result, reference=None,
              t_min: float = 0.0, t_max: float = 0.0) -> Path:
    path = out_dir / "timeseries.csv"
    t = np.asarray(result.time)
    positions = np.asarray(result.positions)
    quats = np.asarray(result.quaternions_xyzw)
    lin_vel = np.asarray(result.linear_velocities)
    ang_vel = np.asarray(result.angular_velocities)
    lengths = np.asarray(result.cable_lengths)
    tensions = np.asarray(result.cable_tensions)
    n_cables = tensions.shape[1]

    if reference is not None:
        ref_pos = np.array([reference(tt).position for tt in t])
        err = np.linalg.norm(positions - ref_pos, axis=1)
    else:
        ref_pos = np.full_like(positions, np.nan)
        err = np.full(len(t), np.nan)

    if t_max > t_min:
        feasible = ((tensions >= t_min - 1e-6) & (tensions <= t_max + 1e-6)).all(axis=1).astype(int)
    else:
        feasible = np.ones(len(t), dtype=int)
    infeasible_set = set(int(i) for i in result.infeasible_steps)
    qp_infeas = np.array([1 if i in infeasible_set else 0 for i in range(len(t))], dtype=int)

    headers = ["t"] + ["px", "py", "pz"] + ["qx", "qy", "qz", "qw"]
    headers += ["vx", "vy", "vz", "wx", "wy", "wz"]
    headers += ["px_ref", "py_ref", "pz_ref", "track_err"]
    headers += [f"L{i+1}" for i in range(n_cables)]
    headers += [f"T{i+1}" for i in range(n_cables)]
    headers += ["feasible", "infeasible_qp"]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for k in range(len(t)):
            row = [
                f"{t[k]:.6f}",
                *(f"{v:.6f}" for v in positions[k]),
                *(f"{v:.6f}" for v in quats[k]),
                *(f"{v:.6f}" for v in lin_vel[k]),
                *(f"{v:.6f}" for v in ang_vel[k]),
                *(f"{v:.6f}" for v in ref_pos[k]),
                f"{err[k]:.6e}",
                *(f"{v:.6f}" for v in lengths[k]),
                *(f"{v:.6f}" for v in tensions[k]),
                str(int(feasible[k])), str(int(qp_infeas[k])),
            ]
            f.write(",".join(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Phase-1 example runner: build robot + controller + reference, simulate,
# render the 13-figure pack, write CSV + manifest + feasibility report.
# ---------------------------------------------------------------------------

@dataclass
class Phase1Run:
    name: str
    title: str
    description: str
    out_dir: Path
    samples: int = 0
    runtime_s: float = 0.0
    tension_range_N: tuple[float, float] = (0.0, 0.0)
    tracking_rms_m: float = 0.0
    figures: list[str] = field(default_factory=list)


def _run_phase1(
    *,
    name: str,
    title: str,
    description: str,
    reference: Callable[[float], Pose],
    duration: float,
    dt: float,
    kp_pos: float,
    kp_rot: float,
    out_dir: Path,
    trajectory_spec: dict | None = None,
) -> Phase1Run:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Phase-1 example: {name} ===")
    print(f"  title       : {title}")
    print(f"  description : {description}")
    print(f"  out_dir     : {out_dir.resolve()}")
    robot = dissertation_8cable(payload_mass=0.0, t_min=5.0, t_max=500.0)
    kd_pos = 2.0 * float(np.sqrt(kp_pos))
    kd_rot = 2.0 * float(np.sqrt(kp_rot))
    controller = PDController(
        Kp_pos=kp_pos, Kd_pos=kd_pos, Kp_rot=kp_rot, Kd_rot=kd_rot,
        gravity_compensation=True, cancel_external=True,
    )
    p0 = reference(0.0).position
    state0 = PlatformState.at_rest(
        Pose(position=p0, rotation=reference(0.0).rotation),
    )
    print(f"  duration    : {duration:.3f} s   dt: {dt:.4f} s")

    t_start = time.perf_counter()
    result = simulate(
        robot=robot, state0=state0,
        duration=duration, dt=dt,
        reference=reference, controller=controller,
        tension_objective="centered", gravity=(0.0, 0.0, -9.81),
    )
    runtime_s = time.perf_counter() - t_start
    samples = len(result.time)
    tens = np.asarray(result.cable_tensions)
    t_min, t_max = float(robot.limits.t_min[0]), float(robot.limits.t_max[0])
    n_violations = int(((tens < t_min - 1e-6) | (tens > t_max + 1e-6)).sum())
    print(
        f"  done        : {samples} samples in {runtime_s:.2f} s, "
        f"tensions [{float(tens.min()):.2f}, {float(tens.max()):.2f}] N, "
        f"violations {n_violations}, infeasible {len(result.infeasible_steps)}"
    )

    # Tracking error metrics.
    ref_pos = np.array([reference(t).position for t in result.time])
    err = np.linalg.norm(np.asarray(result.positions) - ref_pos, axis=1)
    rms_track = float(np.sqrt(np.mean(err ** 2)))

    # Feasibility report.
    feas = {
        "t_min_N": t_min, "t_max_N": t_max,
        "tension_min_observed_N": float(tens.min()),
        "tension_max_observed_N": float(tens.max()),
        "tension_mean_N": float(tens.mean()),
        "qp_infeasible_steps": list(map(int, result.infeasible_steps)),
        "bound_violation_count": n_violations,
        "samples": int(samples),
        "tracking_error_rms_m": rms_track,
        "tracking_error_peak_m": float(err.max()),
        "tracking_error_final_m": float(err[-1]),
    }
    (out_dir / "feasibility.json").write_text(
        json.dumps(feas, indent=2), encoding="utf-8",
    )

    csv_path = _save_csv(out_dir, result, reference=reference,
                         t_min=t_min, t_max=t_max)
    print(f"  [timeseries.csv   ] -> {csv_path.resolve()}")
    figs = _render_phase1_plots(out_dir, result, robot, reference)

    # Manifest (compatible with the Phase-2 readers).
    manifest = {
        "example": name,
        "title": title,
        "description": description,
        "git_hash": _git_describe(),
        "python": sys.version.split()[0],
        "platform": _platform.platform(),
        "controller": "pd",
        "controller_gains": {"kp_pos": kp_pos, "kp_rot": kp_rot,
                             "kd_pos": kd_pos, "kd_rot": kd_rot},
        "feasibility": feas,
        "robot_spec": RobotSpec.from_robot(robot).to_dict(),
        # ``request`` block lets train_from_csv.run_replay rebuild the
        # trajectory for catalog kinds (circle). For the custom
        # callables used by spiral / mshape the trajectory_spec is
        # marked ``custom_callable`` and replay/RL fall back to their
        # "skip cleanly with reason" stub --- which is the honest
        # behaviour since we cannot reconstruct a closed-form
        # reference for the inline functions.
        "request": {
            "robot": "dissertation_8cable",
            "payload_mass": 0.0,
            "duration": duration, "dt": dt,
            "tension_objective": "centered",
            "gravity": [0.0, 0.0, -9.81],
            "trajectory": trajectory_spec or {
                "kind": "custom_callable",
                "duration": duration, "params": {},
            },
        },
        "samples": samples,
        "runtime_s": round(runtime_s, 3),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )
    print(f"  [feasibility.json ] -> {(out_dir / 'feasibility.json').resolve()}")
    print(f"  [manifest.json    ] -> {(out_dir / 'manifest.json').resolve()}")

    return Phase1Run(
        name=name, title=title, description=description,
        out_dir=out_dir, samples=samples, runtime_s=runtime_s,
        tension_range_N=(float(tens.min()), float(tens.max())),
        tracking_rms_m=rms_track, figures=figs,
    )


def _git_describe() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, text=True,
        ).strip()
    except Exception:                                              # pragma: no cover
        return "unknown"


# ---------------------------------------------------------------------------
# Phase-1 example builders
# ---------------------------------------------------------------------------

def example_circle(out_dir: Path) -> Phase1Run:
    """Example 1 — circular tracking, 3-DoF planar reference."""
    # Reuse the proven dissertation-circle preset's trajectory analytically:
    # quintic time-scaling for smooth start/stop over 2 revolutions of
    # a 0.05 m horizontal circle at z = 0.65 m.
    def _circle_pose(t: float, *, T: float = 12.566) -> Pose:
        s = float(np.clip(t / T, 0.0, 1.0))
        s_smooth = 6 * s ** 5 - 15 * s ** 4 + 10 * s ** 3
        theta = 2.0 * np.pi * 2.0 * s_smooth
        pos = np.array([0.05 * np.cos(theta), 0.05 * np.sin(theta), 0.65])
        return Pose(position=pos, rotation=Rotation.identity())
    return _run_phase1(
        name="circle",
        title="Circular trajectory tracking",
        description="Horizontal 0.05 m radius circle traced twice. "
                    "Demonstrates closed-loop PD tracking with feasible "
                    "tensions inside the 5-500 N envelope.",
        reference=_circle_pose,
        duration=12.566, dt=1e-3,                                   # 4π/(0.1/0.05) s for v=0.1 m/s tangential
        kp_pos=400.0, kp_rot=100.0,
        out_dir=out_dir,
        # Catalog-recoverable: replay / PPO / SAC can rebuild this
        # trajectory from the manifest alone.
        trajectory_spec={
            "kind": "circle",
            "duration": 12.566,
            "params": {
                "center": [0.0, 0.0, 0.65],
                "radius": 0.05,
                "axis": [0.0, 0.0, 1.0],
                "angle_span": float(4 * np.pi),
            },
        },
    )


def example_spiral(out_dir: Path) -> Phase1Run:
    """Example 2 — 6-DoF helical reference.

    Yaw rotates 2 full turns over the run and pitch oscillates ±~9 deg
    so all six rigid-body DoFs see non-trivial motion. The ``orientation``
    plot in the artifact bundle makes this visually obvious.
    """
    def _ref(t: float) -> Pose:
        return helical6_pose(t)                                     # uses module defaults
    return _run_phase1(
        name="spiral",
        title="6-DoF helical spiral (full SE(3) tracking)",
        description="3 spiral revolutions with bounded roll / pitch / yaw "
                    "oscillation (±3-17°); demonstrates orientation "
                    "tracking under closed-loop PD with 5-500 N bounds.",
        reference=_ref,
        duration=15.0, dt=1e-3,
        kp_pos=300.0, kp_rot=80.0,
        out_dir=out_dir,
    )


def example_mshape(out_dir: Path) -> Phase1Run:
    """Example 3 — pick-and-place letter-M task with vertex dwells and
    discrete gripper actuation at each pick / place event."""
    duration = mshape_total_duration()
    def _ref(t: float) -> Pose:
        return mshape_pose(t)
    return _run_phase1(
        name="mshape",
        title="Pick-and-place M-task",
        description=f"Trace the letter 'M' through 5 vertices with "
                    f"{int(duration)} s total of motion + dwells. "
                    "Gripper yaw at each dwell simulates pick/place "
                    "actuation under load.",
        reference=_ref,
        duration=duration, dt=1e-3,
        kp_pos=350.0, kp_rot=90.0,
        out_dir=out_dir,
    )


# ---------------------------------------------------------------------------
# Phase-2 example runner: invokes the existing train_from_csv / compare_models
# scripts as subprocesses with pre-baked argument lists.
# ---------------------------------------------------------------------------

def _phase2_input_csv(dependency: str, dep_out_root: Path) -> Path:
    """Locate the CSV produced by a Phase-1 dependency, generating it if
    missing. Default dependency is the circle example."""
    csv = dep_out_root / "timeseries.csv"
    if csv.exists():
        return csv
    print(f"[phase2] dependency CSV missing: {csv}")
    print(f"[phase2] auto-running Phase-1 example '{dependency}' to generate it…")
    if dependency == "circle":
        example_circle(dep_out_root)
    elif dependency == "spiral":
        example_spiral(dep_out_root)
    elif dependency == "mshape":
        example_mshape(dep_out_root)
    else:
        raise ValueError(f"Unknown Phase-1 dependency: {dependency!r}")
    if not csv.exists():
        raise FileNotFoundError(f"Phase-1 run did not produce {csv}")
    return csv


def example_train(out_dir: Path, *, dependency: str = "circle",
                  dep_out_root: Path | None = None) -> dict:
    """Example 4 — single-model PINN training on a Phase-1 CSV."""
    dep_out_root = dep_out_root or (_REPO_ROOT / "out" / "example-circle")
    csv = _phase2_input_csv(dependency, dep_out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Phase-2 example: train (PINN) ===")
    print(f"  input  : {csv}")
    print(f"  out    : {out_dir.resolve()}")
    cmd = [
        sys.executable, str(Path(__file__).parent / "train_from_csv.py"),
        "--input", str(csv),
        "--model", "pinn",
        "--epochs", "120",
        "--batch-size", "64",
        "--learning-rate", "1e-3",
        "--out", str(out_dir),
    ]
    return _run_subprocess("train", cmd, out_dir)


def example_compare(out_dir: Path, *, dependency: str = "circle",
                    dep_out_root: Path | None = None) -> dict:
    """Example 5 — multi-model comparison on a Phase-1 CSV."""
    dep_out_root = dep_out_root or (_REPO_ROOT / "out" / "example-circle")
    csv = _phase2_input_csv(dependency, dep_out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Phase-2 example: compare (replay/mlp/pinn/ppo/sac) ===")
    print(f"  input  : {csv}")
    print(f"  out    : {out_dir.resolve()}")
    cmd = [
        sys.executable, str(Path(__file__).parent / "compare_models.py"),
        "--input", str(csv),
        "--out",   str(out_dir),
        "--models", "replay", "mlp", "pinn", "ppo", "sac",
        "--epochs", "60",
        "--rl-steps", "2000",
        "--eval-episodes", "2",
    ]
    return _run_subprocess("compare", cmd, out_dir)


def _run_subprocess(label: str, cmd: list[str], out_dir: Path) -> dict:
    print(f"  cmd    : {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    (out_dir / f"{label}_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    (out_dir / f"{label}_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    print(f"  done   : exit {proc.returncode}, {dt:.1f} s")
    return {"label": label, "returncode": proc.returncode, "wall_time_s": round(dt, 2)}


# ---------------------------------------------------------------------------
# Registry --- the single source of truth used by the CLI and the GUI.
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, dict] = {
    "circle": {
        "phase": 1,
        "title": "Circular trajectory tracking",
        "description": ("3-DoF planar circle at z = 0.65 m, 0.05 m radius, "
                        "2 revolutions with smooth start/stop."),
        "out_dir": "example-circle",
        "runner": example_circle,
    },
    "spiral": {
        "phase": 1,
        "title": "6-DoF helical spiral",
        "description": ("Full SE(3) tracking: 3-revolution spiral with "
                        "platform yaw and pitch oscillation."),
        "out_dir": "example-spiral",
        "runner": example_spiral,
    },
    "mshape": {
        "phase": 1,
        "title": "Pick-and-place M-task",
        "description": ("Trace letter 'M' through 5 vertices with dwells "
                        "and discrete gripper actuation at each pick/place."),
        "out_dir": "example-mshape",
        "runner": example_mshape,
    },
    "train": {
        "phase": 2,
        "title": "Data-driven PINN training",
        "description": ("Train PINN inverse-dynamics on the circle CSV; "
                        "if missing, runs the circle example first."),
        "out_dir": "example-train",
        "depends_on": "circle",
        "runner": example_train,
    },
    "compare": {
        "phase": 2,
        "title": "Multi-model comparison",
        "description": ("Bench replay / MLP / PINN / PPO / SAC on the "
                        "same dataset and rank by RMSE."),
        "out_dir": "example-compare",
        "depends_on": "circle",
        "runner": example_compare,
    },
}


def list_examples() -> list[dict]:
    return [
        {"name": k, **{kk: vv for kk, vv in v.items() if kk != "runner"}}
        for k, v in EXAMPLES.items()
    ]
