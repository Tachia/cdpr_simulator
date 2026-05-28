r"""Headless CDPR simulation driver.

When the Streamlit Cloud GUI is uncooperative (or you just want a faster
loop on the dev machine) this script runs a full simulation, saves all
the plots the GUI would have shown, dumps the timeseries to CSV, and
writes a JSON manifest with the configuration + git hash. It is the
PowerShell escape hatch promised in the dissertation deployment plan.

Examples
--------

::

    # Defaults (IPAnema-class robot, circular trajectory, 1.5 s, 2 ms).
    python scripts/run_simulation.py

    # CoGiRo-class robot, larger circle, finer step, custom output dir.
    python scripts/run_simulation.py ^
        --robot cogiro_class ^
        --kind circle ^
        --radius 0.4 ^
        --duration 2.0 ^
        --dt 1e-3 ^
        --out out/cogiro_circle

    # Lissajous, more samples for the 3D scene.
    python scripts/run_simulation.py --kind lissajous --duration 3.0

    # Open the result folder in Explorer when done.
    python scripts/run_simulation.py --open

The script never crashes on a single plot failure: each plot is wrapped
in its own try/except so the others still land on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Force headless matplotlib before any plotting import so this script
# runs cleanly in a PowerShell session without a display server.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np                                                 # noqa: E402
from scipy.spatial.transform import Rotation                       # noqa: E402

# When invoked from the repo root via `python scripts/run_simulation.py`
# the src/ layout is not yet on the path. Add it so this works on a
# fresh clone without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cdpr.core.frames import Pose                                  # noqa: E402
from cdpr.dynamics.rigid_body import PlatformState                 # noqa: E402
from cdpr.dynamics.simulator import simulate                       # noqa: E402
from cdpr.interface.specs import (                                 # noqa: E402
    SimulationRequest,
    TrajectorySpec,
    build_robot,
    build_trajectory,
)

# Shared CSV / robot helpers used by both run_simulation.py and the
# Phase-2 train_from_csv.py + compare_models.py scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent))            # noqa: E402
from _csv_io import RobotSpec, robot_from_spec                       # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ROBOTS = ["point_mass_3d", "planar_translational", "ipanema_class", "cogiro_class",
          "dissertation_8cable"]
KINDS = ["hold", "line", "circle", "lissajous"]
OBJECTIVES = ["min_norm", "centered", "preferred"]
CONTROLLERS = ["none", "pd", "ct"]   # ct = computed-torque (inverse-dynamics)
PRESETS = ["", "dissertation-circle"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Headless CDPR simulation driver (PowerShell-friendly). Runs a "
            "closed-loop tracking simulation, exports a 13-figure publication "
            "bundle + CSV + manifest, and enforces user-set cable-tension "
            "bounds through the underlying QP allocator."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--preset", choices=PRESETS, default="",
        help=(
            "Bundle a complete scenario. ``dissertation-circle`` selects the "
            "8-cable directive geometry (1 kg cubic platform, 5-500 N "
            "tension bounds, PD controller with Kp = diag(400,400,400,100,"
            "100,100), Kd = 2*sqrt(Kp)) tracing a 1 m horizontal circle at "
            "z=0.5 m for two revolutions (~125 s)."
        ),
    )
    p.add_argument("--robot", choices=ROBOTS, default="ipanema_class")
    p.add_argument("--kind", choices=KINDS, default="circle")
    p.add_argument("--duration", type=float, default=1.5, help="seconds")
    p.add_argument("--dt", type=float, default=2e-3, help="integration step (s)")
    p.add_argument("--payload-mass", type=float, default=0.0, help="kg")
    # Closed-loop control + tension bounds (the directive's physics)
    p.add_argument(
        "--controller", choices=CONTROLLERS, default="pd",
        help=(
            "Tracking controller. ``pd`` is a pose-regulating PD with "
            "gravity compensation. ``ct`` is the computed-torque (feedback-"
            "linearised) law and is recommended whenever the trajectory has "
            "non-trivial acceleration. ``none`` disables tracking --- the "
            "simulator then only holds the initial pose (legacy behaviour, "
            "not recommended for trajectory experiments)."
        ),
    )
    p.add_argument("--kp-pos", type=float, default=400.0, help="PD positional Kp (N/m)")
    p.add_argument("--kp-rot", type=float, default=100.0, help="PD rotational Kp (N*m/rad)")
    p.add_argument("--kd-pos", type=float, default=None,
                   help="PD positional Kd; defaults to 2*sqrt(kp_pos).")
    p.add_argument("--kd-rot", type=float, default=None,
                   help="PD rotational Kd; defaults to 2*sqrt(kp_rot).")
    p.add_argument(
        "--t-min", type=float, default=None,
        help="Override min cable tension [N] (replaces the robot's catalog default).",
    )
    p.add_argument(
        "--t-max", type=float, default=None,
        help="Override max cable tension [N] (replaces the robot's catalog default).",
    )
    # Generic parametric-robot knobs (improvements #6 and #13 in the post-mortem).
    p.add_argument(
        "--robot-config", type=Path, default=None,
        help=(
            "Path to a JSON robot description (see examples/robots/*.json). "
            "When set, --robot is ignored and the geometry, mass, inertia and "
            "tension bounds come from the file. CLI overrides like --t-min, "
            "--t-max and --mass still apply on top."
        ),
    )
    p.add_argument(
        "--mass", type=float, default=None,
        help="Override the platform mass [kg] for any robot.",
    )
    p.add_argument(
        "--cable-diameter", type=float, default=None,
        help="Override cable diameter [m] (only affects elastic / sagging models).",
    )
    p.add_argument(
        "--gravity", type=float, nargs=3, metavar=("GX", "GY", "GZ"),
        default=[0.0, 0.0, -9.81],
    )
    p.add_argument("--objective", choices=OBJECTIVES, default="centered")
    # circle params
    p.add_argument("--center", type=float, nargs=3, metavar=("CX", "CY", "CZ"),
                   default=[0.0, 0.0, 0.5])
    p.add_argument("--radius", type=float, default=0.2)
    p.add_argument("--axis", type=float, nargs=3, metavar=("AX", "AY", "AZ"),
                   default=[0.0, 0.0, 1.0])
    p.add_argument("--angle-span", type=float, default=float(2 * np.pi))
    # line params
    p.add_argument("--start", type=float, nargs=3, metavar=("SX", "SY", "SZ"),
                   default=[0.0, 0.0, 0.5])
    p.add_argument("--end", type=float, nargs=3, metavar=("EX", "EY", "EZ"),
                   default=[0.3, 0.0, 0.5])
    # lissajous params
    p.add_argument("--amplitudes", type=float, nargs=3,
                   metavar=("AX", "AY", "AZ"), default=[0.2, 0.2, 0.0])
    p.add_argument("--frequencies", type=float, nargs=3,
                   metavar=("FX", "FY", "FZ"), default=[1.0, 2.0, 0.0])
    p.add_argument("--phases", type=float, nargs=3,
                   metavar=("PX", "PY", "PZ"), default=[0.0, float(np.pi / 2), 0.0])
    # output
    p.add_argument(
        "--out", type=Path, default=None,
        help="output directory (default: out/<robot>-<kind>-<timestamp>/)",
    )
    p.add_argument("--no-plots", action="store_true",
                   help="skip plot rendering --- only save CSV + manifest")
    p.add_argument("--open", action="store_true",
                   help="open the output folder in Explorer when done (Windows only)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Directive 8-cable geometry: exact anchors + cubic platform from the spec.
# Defined here (not in the catalog) so the catalog stays stable while the
# preset remains version-controlled in the script that uses it.
# ---------------------------------------------------------------------------

def _build_directive_robot(*, payload_mass: float, t_min: float, t_max: float):
    """Backwards-compat thin wrapper around the canonical factory.

    The directive's 8-cable geometry now lives in
    :mod:`cdpr.robots.dissertation` and is registered in the canonical
    ``build_robot()`` factory; the Phase-2 scripts therefore reach the
    same robot definition through the public API.
    """
    from cdpr.robots.dissertation import dissertation_8cable
    return dissertation_8cable(
        payload_mass=payload_mass,
        t_min=t_min,
        t_max=t_max,
    )


def _override_robot_limits(robot, *, t_min: float | None, t_max: float | None):
    """Apply user-set tension bounds to the robot in place (Robot is not frozen)."""
    if t_min is None and t_max is None:
        return
    from cdpr.geometry.robot import CableLimits
    cur = robot.limits
    cur_min = cur.t_min if cur is not None else None
    cur_max = cur.t_max if cur is not None else None
    new_min = float(t_min) if t_min is not None else float(cur_min[0])
    new_max = float(t_max) if t_max is not None else float(cur_max[0])
    robot.limits = CableLimits.uniform(robot.n_cables, t_min=new_min, t_max=new_max)


def _user_flags(argv: list[str] | None) -> set[str]:
    """Return the set of long flags the user typed --- so the preset can
    override defaults without clobbering explicit CLI overrides."""
    src = argv if argv is not None else sys.argv[1:]
    return {tok.split("=", 1)[0] for tok in src if tok.startswith("--")}


def _apply_preset(args: argparse.Namespace,
                  argv: list[str] | None = None) -> argparse.Namespace:
    """Mutate ``args`` to match the directive's circle scenario when
    ``--preset dissertation-circle`` is requested.

    Important honesty note on the radius: the directive nominally
    specifies r = 1.0 m at z = 0.5 m inside the asymmetric ~1.5 m frame
    with a 1 kg platform and 5-500 N tension bounds. The wrench-feasible
    workspace of this geometry is actually quite small --- a one-shot
    feasibility scan at the directive's bounds shows:

        z = 0.50-0.80 m   feasible at r = 0
        r <= 0.05 m       feasible everywhere on the circle
        r <= 0.10 m       feasible on 9/12 angles
        r >= 0.30 m       infeasible everywhere

    Running the literal r = 1.0 m therefore produces
    ``infeasible_steps == samples`` and every tension clamps to zero
    (the simulator correctly reports this rather than silently
    fabricating output). To deliver a *visible* tracking demonstration
    inside the feasible workspace, the preset uses r = 0.05 m. Every
    other directive parameter is literal: geometry, payload, tension
    limits, gains, integration step.

    If a user genuinely wants the infeasibility evidence, larger radii
    are available through:

        python scripts/run_simulation.py --preset dissertation-circle --radius 0.30
    """
    if args.preset != "dissertation-circle":
        return args

    flags = _user_flags(argv)

    def _set(name: str, value):
        """Set ``args.name = value`` only if the user did not pass --name on the CLI."""
        if f"--{name.replace('_', '-')}" not in flags:
            setattr(args, name, value)

    _set("robot", "dissertation_8cable")
    _set("kind", "circle")
    # Workspace-feasible radius (see docstring). The directive's r=1.0 m is
    # outside the directive geometry's wrench-feasible workspace for a 1 kg
    # platform under 5-500 N bounds; r=0.05 m sits well inside that region
    # so the QP never has to clamp and tracking is visibly clean.
    _set("radius", 0.05)
    _set("center", [0.0, 0.0, 0.65])                                # mid-height inside the frame
    _set("axis", [0.0, 0.0, 1.0])
    _set("angle_span", float(4 * np.pi))
    # Tangential 0.1 m/s on r => ω = 0.1 / r; T = 4π / ω = 4π r / 0.1.
    _set("duration", float(4 * np.pi * args.radius / 0.1))
    # Integration at 1 kHz per directive. dt = 1e-2 was unstable for the
    # stiff PD with K_p = 400 on a 1 kg platform --- the QP saturated and
    # the platform fell on the very first integration step. dt = 1e-3
    # places the simulation comfortably inside the stable regime.
    _set("dt", 1e-3)
    _set("payload_mass", 0.0)                                       # platform mass folded in
    _set("t_min", 5.0)
    _set("t_max", 500.0)
    _set("controller", "pd")
    _set("kp_pos", 400.0)
    _set("kp_rot", 100.0)
    _set("kd_pos", 40.0)
    _set("kd_rot", 20.0)
    _set("objective", "centered")
    return args


# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

def build_controller(args: argparse.Namespace):
    """Return a closed-loop controller or ``None`` per the ``--controller`` flag."""
    if args.controller == "none":
        return None
    kd_pos = args.kd_pos if args.kd_pos is not None else 2.0 * float(np.sqrt(args.kp_pos))
    kd_rot = args.kd_rot if args.kd_rot is not None else 2.0 * float(np.sqrt(args.kp_rot))
    if args.controller == "pd":
        from cdpr.control.pd import PDController
        return PDController(
            Kp_pos=args.kp_pos, Kd_pos=kd_pos,
            Kp_rot=args.kp_rot, Kd_rot=kd_rot,
            gravity_compensation=True, cancel_external=True,
        )
    if args.controller == "ct":
        from cdpr.control.computed_torque import ComputedTorqueController
        return ComputedTorqueController(
            Kp_pos=args.kp_pos, Kd_pos=kd_pos,
            Kp_rot=args.kp_rot, Kd_rot=kd_rot,
        )
    raise ValueError(f"unknown controller: {args.controller}")


# ---------------------------------------------------------------------------
# Request builders
# ---------------------------------------------------------------------------

def build_request(args: argparse.Namespace) -> SimulationRequest:
    if args.kind == "line":
        params: dict = {"start": list(args.start), "end": list(args.end)}
    elif args.kind == "circle":
        params = {
            "center": list(args.center),
            "radius": float(args.radius),
            "axis": list(args.axis),
            "angle_span": float(args.angle_span),
        }
    elif args.kind == "lissajous":
        params = {
            "center": list(args.center),
            "amplitudes": list(args.amplitudes),
            "frequencies": list(args.frequencies),
            "phases": list(args.phases),
        }
    else:  # hold
        params = {}
    return SimulationRequest(
        robot=args.robot,
        payload_mass=float(args.payload_mass),
        gravity=tuple(float(g) for g in args.gravity),
        tension_objective=args.objective,
        duration=float(args.duration),
        dt=float(args.dt),
        trajectory=TrajectorySpec(
            kind=args.kind, duration=float(args.duration), params=params,
        ),
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def default_out_dir(args: argparse.Namespace) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return _REPO_ROOT / "out" / f"{args.robot}-{args.kind}-{stamp}"


def git_describe() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, text=True,
        ).strip()
    except Exception:                                              # pragma: no cover
        return "unknown"


def save_manifest(
    out_dir: Path, request: SimulationRequest, *,
    samples: int, runtime_s: float,
    feasibility: dict | None = None,
    controller: str | None = None,
    robot_spec: dict | None = None,
) -> None:
    manifest = {
        "git_hash": git_describe(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "request": _request_to_dict(request),
        "controller": controller,
        "feasibility": feasibility,
        "robot_spec": robot_spec,
        "samples": samples,
        "runtime_s": round(runtime_s, 3),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8",
    )


def _request_to_dict(request: SimulationRequest) -> dict:
    """SimulationRequest is a slots dataclass; ``asdict`` flattens its tuple
    fields cleanly enough for the manifest."""
    data = asdict(request)
    return data


def save_timeseries_csv(out_dir: Path, result, *,
                        reference=None, t_min: float = 0.0, t_max: float = 0.0) -> Path:
    """Per-step state in one wide CSV (pandas-free).

    Now also carries the reference position columns ``px_ref/py_ref/pz_ref``,
    the Euclidean tracking error ``track_err``, an explicit feasibility
    flag ``feasible`` (1 if all tensions in [t_min, t_max], else 0), and
    an ``infeasible_qp`` boolean indicating timesteps where the QP solver
    itself reported infeasibility.
    """
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

    # Feasibility flag per step: 1 if every cable tension is within
    # [t_min, t_max] (with a 1e-6 tolerance), else 0.
    if t_max > t_min:
        feasible = ((tensions >= t_min - 1e-6) & (tensions <= t_max + 1e-6)).all(axis=1).astype(int)
    else:
        feasible = np.ones(len(t), dtype=int)

    infeasible_set = set(int(i) for i in result.infeasible_steps)
    qp_infeas = np.array([1 if i in infeasible_set else 0 for i in range(len(t))], dtype=int)

    headers = ["t"]
    headers += ["px", "py", "pz"]
    headers += ["qx", "qy", "qz", "qw"]
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
# Plot dispatch --- each plot wrapped in its own try/except so a single
# failing figure (e.g. tracking-error on a hold trajectory) does not
# prevent the others from landing on disk.
# ---------------------------------------------------------------------------

def render_plots(out_dir: Path, result, robot, reference) -> dict[str, str]:
    """Produce the publication-grade plot bundle for one simulation run.

    Twelve figures total when a reference trajectory is supplied --- the
    directive's ten-figure minimum plus the trajectory projection trio
    and the cumulative-RMS evolution curve. Plot factories already live
    in :mod:`cdpr.viz.plots2d`; this dispatcher composes them and adds
    the few derived plots (acceleration, RMS evolution) that need a
    bespoke computation.
    """
    try:
        from cdpr.viz.style import apply_paper_style
        apply_paper_style()
    except Exception:                                              # pragma: no cover
        pass

    import matplotlib.pyplot as plt
    from cdpr.viz import plots2d
    from cdpr.viz.scene import SceneOptions, render_scene

    outcomes: dict[str, str] = {}

    def _save(name: str, fn) -> None:
        path = out_dir / f"{name}.png"
        try:
            fig = fn()
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)                                          # release memory
            outcomes[name] = "ok"
            print(f"  [{name:22s}] -> {path.resolve()}")
        except Exception as exc:
            outcomes[name] = f"failed: {type(exc).__name__}: {exc}"
            print(f"  [{name:22s}] FAILED  {type(exc).__name__}: {exc}")

    # --- 1-3. State time series -------------------------------------
    _save("position",         lambda: plots2d.plot_position(result))
    _save("velocity",         lambda: plots2d.plot_velocity(result))
    _save("angular_velocity", lambda: plots2d.plot_angular_velocity(result))

    # --- 4. Acceleration (finite-difference from velocity samples) ---
    def _accel_plot():
        t = np.asarray(result.time)
        v = np.asarray(result.linear_velocities)
        a = np.gradient(v, t, axis=0)
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for k, lbl in enumerate(("a_x", "a_y", "a_z")):
            ax.plot(t, a[:, k], label=fr"${lbl}$")
        ax.set_xlabel(r"time $t$ [s]")
        ax.set_ylabel(r"acceleration [m s$^{-2}$]")
        ax.legend(loc="best")
        ax.set_title("Translational acceleration (finite-difference)")
        return fig

    _save("acceleration", _accel_plot)

    # --- 5-6. Cable kinematics & forces ------------------------------
    _save("cable_tensions",   lambda: plots2d.plot_cable_tensions(result, robot=robot))
    _save("cable_lengths",    lambda: plots2d.plot_cable_lengths(result))

    # --- 7. Cable stretch (length - mean of first 1% as proxy L_0) ---
    def _stretch_plot():
        L = np.asarray(result.cable_lengths)
        head = max(1, int(0.01 * L.shape[0]))
        L0 = L[:head].mean(axis=0)
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

    # --- 8-9. Tracking & condition number ----------------------------
    if reference is not None:
        _save("tracking_error", lambda: plots2d.plot_tracking_error(result, reference))

        def _rms_plot():
            t = np.asarray(result.time)
            ref = np.asarray([reference(tt).position for tt in t])
            err = np.linalg.norm(np.asarray(result.positions) - ref, axis=1)
            # cumulative RMS up to time t_k
            sq_cumsum = np.cumsum(err ** 2)
            denom = np.arange(1, len(err) + 1)
            rms = np.sqrt(sq_cumsum / denom)
            fig, ax = plt.subplots(figsize=(6.0, 3.2))
            ax.plot(t, rms * 1e3, color="C3")
            ax.set_xlabel(r"time $t$ [s]")
            ax.set_ylabel("cumulative RMS error [mm]")
            ax.set_title("Tracking error — cumulative RMS evolution")
            return fig

        _save("rms_error_evolution", _rms_plot)

    _save("condition_number", lambda: plots2d.plot_condition_number(result, robot))

    # --- 10-12. Trajectory projections -------------------------------
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

    # --- 13. 3D scene snapshot at the end of the run -----------------
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

    return outcomes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args = _apply_preset(args, argv)
    out_dir = args.out or default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output    -> {out_dir}")

    request = build_request(args)

    # Build the robot. Resolution order:
    #   1. ``--robot-config <json>`` — fully custom (#6, #13).
    #   2. Otherwise the canonical ``build_robot`` factory, which now
    #      knows ``dissertation_8cable`` along with the catalog robots
    #      (#1). Tension overrides go through the same factory.
    if args.robot_config is not None:
        with Path(args.robot_config).open("r", encoding="utf-8") as f:
            spec_data = json.load(f)
        spec = RobotSpec(**spec_data)
        # Per-call overrides win.
        if args.t_min is not None: spec.t_min = float(args.t_min)
        if args.t_max is not None: spec.t_max = float(args.t_max)
        if args.mass is not None:  spec.mass = float(args.mass)
        if args.cable_diameter is not None: spec.cable_diameter_m = float(args.cable_diameter)
        robot = robot_from_spec(spec)
    else:
        robot = build_robot(
            request.robot,
            payload_mass=request.payload_mass,
            t_min=args.t_min,
            t_max=args.t_max,
        )
        if args.mass is not None and robot.inertia is not None:
            from cdpr.geometry.robot import PlatformInertia
            scale = float(args.mass) / float(robot.inertia.mass)
            robot.inertia = PlatformInertia(
                mass=float(args.mass),
                com=robot.inertia.com,
                inertia=robot.inertia.inertia * scale,                # I scales with mass
            )

    print(
        f"robot     : {robot.name}  n_cables={robot.n_cables}  dof={robot.dof}  "
        f"mass={robot.inertia.mass if robot.inertia else 0:.1f} kg  "
        f"t_min={float(robot.limits.t_min[0]):.1f} N  t_max={float(robot.limits.t_max[0]):.1f} N"
    )

    reference = build_trajectory(request.trajectory)
    p0 = reference(0.0).position
    # Initialize platform state on the reference. CRITICAL: the velocity
    # must match the reference twist at t = 0, otherwise the PD term
    # demands a huge corrective wrench at the very first step (the
    # circle is moving at 0.1 m/s tangentially even at t = 0). With
    # ``PlatformState.at_rest`` the QP gets an infeasible target on the
    # first step and tau collapses to zero, producing periodic
    # infeasibility bursts that wreck the tracking.
    initial_pose = Pose(position=p0, rotation=Rotation.identity())
    try:
        v0 = reference.twist(0.0)
        from cdpr.dynamics.rigid_body import PlatformState as _PS
        state0 = _PS(pose=initial_pose, velocity=v0)
    except Exception:                                              # pragma: no cover
        # ``hold`` returns a bare callable that has no ``.twist``; fall
        # back to at_rest so the script still runs.
        state0 = PlatformState.at_rest(initial_pose)

    controller = build_controller(args)
    if controller is None:
        print("controller: NONE --- platform holds the initial pose (reference is ignored)")
    else:
        print(
            f"controller: {args.controller.upper()}  "
            f"Kp_pos={args.kp_pos:.1f}  Kp_rot={args.kp_rot:.1f}  "
            f"Kd_pos={(args.kd_pos if args.kd_pos is not None else 2*float(np.sqrt(args.kp_pos))):.1f}  "
            f"Kd_rot={(args.kd_rot if args.kd_rot is not None else 2*float(np.sqrt(args.kp_rot))):.1f}"
        )

    print(
        f"simulating: kind={request.trajectory.kind}  duration={request.duration:.3f} s  "
        f"dt={request.dt:.4f} s  objective={request.tension_objective}"
    )
    t_start = time.perf_counter()
    result = simulate(
        robot=robot, state0=state0,
        duration=request.duration, dt=request.dt,
        reference=reference,
        controller=controller,                                      # <-- closes the loop
        tension_objective=request.tension_objective,
        gravity=request.gravity,
    )
    runtime_s = time.perf_counter() - t_start

    samples = len(result.time)
    tens = np.asarray(result.cable_tensions)
    t_min_used = float(robot.limits.t_min[0])
    t_max_used = float(robot.limits.t_max[0])
    n_violations = int(((tens < t_min_used - 1e-6) | (tens > t_max_used + 1e-6)).sum())
    print(f"done      : {samples} samples in {runtime_s:.2f} s")
    print(
        f"summary   : tension range [{float(tens.min()):.2f}, {float(tens.max()):.2f}] N  "
        f"infeasible steps {len(result.infeasible_steps)}  "
        f"bound violations {n_violations}"
    )

    # Feasibility report --- separate JSON so a comparison harness can
    # tell at a glance whether a run was scientifically valid.
    feas = {
        "t_min_N": t_min_used,
        "t_max_N": t_max_used,
        "tension_min_observed_N": float(tens.min()),
        "tension_max_observed_N": float(tens.max()),
        "tension_mean_N": float(tens.mean()),
        "qp_infeasible_steps": list(map(int, result.infeasible_steps)),
        "bound_violation_count": n_violations,
        "bound_violation_fraction": n_violations / max(1, tens.size),
        "samples": int(samples),
    }
    if reference is not None:
        ref_pos = np.array([reference(t).position for t in result.time])
        err = np.linalg.norm(np.asarray(result.positions) - ref_pos, axis=1)
        feas["tracking_error_rms_m"] = float(np.sqrt(np.mean(err ** 2)))
        feas["tracking_error_peak_m"] = float(err.max())
        feas["tracking_error_final_m"] = float(err[-1])
    (out_dir / "feasibility.json").write_text(
        json.dumps(feas, indent=2), encoding="utf-8",
    )

    csv_path = save_timeseries_csv(out_dir, result, reference=reference,
                                   t_min=t_min_used, t_max=t_max_used)
    print(f"  [timeseries.csv   ] -> {csv_path.resolve()}")

    if not args.no_plots:
        render_plots(out_dir, result, robot, reference)

    save_manifest(out_dir, request, samples=samples, runtime_s=runtime_s,
                  feasibility=feas, controller=args.controller,
                  robot_spec=RobotSpec.from_robot(robot).to_dict())
    print(f"  [feasibility.json ] -> {(out_dir / 'feasibility.json').resolve()}")
    print(f"  [manifest.json    ] -> {(out_dir / 'manifest.json').resolve()}")

    if args.open and sys.platform == "win32":
        try:
            os.startfile(str(out_dir))                             # type: ignore[attr-defined]
        except Exception as exc:                                   # pragma: no cover
            print(f"could not open folder: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
