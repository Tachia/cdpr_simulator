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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ROBOTS = ["point_mass_3d", "planar_translational", "ipanema_class", "cogiro_class"]
KINDS = ["hold", "line", "circle", "lissajous"]
OBJECTIVES = ["min_norm", "centered", "preferred"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless CDPR simulation driver (PowerShell-friendly).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--robot", choices=ROBOTS, default="ipanema_class")
    p.add_argument("--kind", choices=KINDS, default="circle")
    p.add_argument("--duration", type=float, default=1.5, help="seconds")
    p.add_argument("--dt", type=float, default=2e-3, help="integration step (s)")
    p.add_argument("--payload-mass", type=float, default=0.0, help="kg")
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


def save_manifest(out_dir: Path, request: SimulationRequest, samples: int, runtime_s: float) -> None:
    manifest = {
        "git_hash": git_describe(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "request": _request_to_dict(request),
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


def save_timeseries_csv(out_dir: Path, result) -> Path:
    """Per-step state in one wide CSV --- pandas-free so the script has
    one less dependency on a fresh clone."""
    path = out_dir / "timeseries.csv"
    t = np.asarray(result.time)
    positions = np.asarray(result.positions)
    quats = np.asarray(result.quaternions_xyzw)
    lin_vel = np.asarray(result.linear_velocities)
    ang_vel = np.asarray(result.angular_velocities)
    lengths = np.asarray(result.cable_lengths)
    tensions = np.asarray(result.cable_tensions)
    n_cables = tensions.shape[1]
    headers = ["t"]
    headers += ["px", "py", "pz"]
    headers += ["qx", "qy", "qz", "qw"]
    headers += ["vx", "vy", "vz", "wx", "wy", "wz"]
    headers += [f"L{i+1}" for i in range(n_cables)]
    headers += [f"T{i+1}" for i in range(n_cables)]
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for k in range(len(t)):
            row = [
                f"{t[k]:.6f}",
                *(f"{v:.6f}" for v in positions[k]),
                *(f"{v:.6f}" for v in quats[k]),
                *(f"{v:.6f}" for v in lin_vel[k]),
                *(f"{v:.6f}" for v in ang_vel[k]),
                *(f"{v:.6f}" for v in lengths[k]),
                *(f"{v:.6f}" for v in tensions[k]),
            ]
            f.write(",".join(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# Plot dispatch --- each plot wrapped in its own try/except so a single
# failing figure (e.g. tracking-error on a hold trajectory) does not
# prevent the others from landing on disk.
# ---------------------------------------------------------------------------

def render_plots(out_dir: Path, result, robot, reference) -> dict[str, str]:
    try:
        from cdpr.viz.style import apply_paper_style
        apply_paper_style()
    except Exception:                                              # pragma: no cover
        pass

    from cdpr.viz import plots2d
    from cdpr.viz.scene import SceneOptions, render_scene

    outcomes: dict[str, str] = {}

    def _save(name: str, fn) -> None:
        path = out_dir / f"{name}.png"
        try:
            fig = fn()
            fig.savefig(path, dpi=160, bbox_inches="tight")
            outcomes[name] = "ok"
            print(f"  [{name:18s}] -> {path.resolve()}")
        except Exception as exc:
            outcomes[name] = f"failed: {type(exc).__name__}: {exc}"
            print(f"  [{name:18s}] FAILED  {type(exc).__name__}: {exc}")

    _save("position",         lambda: plots2d.plot_position(result))
    _save("cable_tensions",   lambda: plots2d.plot_cable_tensions(result, robot=robot))
    _save("cable_lengths",    lambda: plots2d.plot_cable_lengths(result))
    if reference is not None:
        _save("tracking_error", lambda: plots2d.plot_tracking_error(result, reference))
    _save("condition_number", lambda: plots2d.plot_condition_number(result, robot))
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
    out_dir = args.out or default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output -> {out_dir}")

    request = build_request(args)
    robot = build_robot(request.robot, payload_mass=request.payload_mass)
    reference = build_trajectory(request.trajectory)
    p0 = reference(0.0).position
    state0 = PlatformState.at_rest(Pose(position=p0, rotation=Rotation.identity()))

    print(
        f"simulating: robot={request.robot} kind={request.trajectory.kind} "
        f"duration={request.duration}s dt={request.dt}s"
    )
    t_start = time.perf_counter()
    result = simulate(
        robot=robot, state0=state0,
        duration=request.duration, dt=request.dt,
        reference=reference,
        tension_objective=request.tension_objective,
        gravity=request.gravity,
    )
    runtime_s = time.perf_counter() - t_start

    samples = len(result.time)
    print(f"done   : {samples} samples in {runtime_s:.2f} s")
    print(
        f"summary: max tension {float(np.max(result.cable_tensions)):.1f} N, "
        f"min tension {float(np.min(result.cable_tensions)):.1f} N, "
        f"infeasible steps {len(result.infeasible_steps)}"
    )

    csv_path = save_timeseries_csv(out_dir, result)
    print(f"  [timeseries.csv   ] -> {csv_path.resolve()}")

    if not args.no_plots:
        render_plots(out_dir, result, robot, reference)

    save_manifest(out_dir, request, samples=samples, runtime_s=runtime_s)
    print(f"  [manifest.json    ] -> {(out_dir / 'manifest.json').resolve()}")

    if args.open and sys.platform == "win32":
        try:
            os.startfile(str(out_dir))                             # type: ignore[attr-defined]
        except Exception as exc:                                   # pragma: no cover
            print(f"could not open folder: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
