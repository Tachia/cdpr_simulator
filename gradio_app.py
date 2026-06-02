"""Gradio frontend for CDPR_SIMULATOR --- the recommended hosted demo.

Designed for **Hugging Face Spaces** (free 16 GB / 2 vCPU CPU Space).
The architectural rationale lives in ``docs/frontend-architecture.md``;
this file is the production code that backs it.

The app exposes three tabs that mirror the directive's two phases:

    1. **Built-in examples**   --- the same registry used by
       ``scripts/run_example.py`` and the Streamlit GUI. One click runs
       the example and the produced figures stream into the page.
    2. **Custom Phase-1**      --- pick robot + trajectory + controller
       (PD or MPC) + tension bounds, hit *Run simulation*, get the full
       14-figure bundle and CSV download link. A chat box at the top of
       this tab parses free-text descriptions through ``cdpr.llm`` and
       pre-fills the form, so users can either describe what they want
       in English or configure parameters by hand.
    3. **Upload CSV / Phase-2** --- upload any CSV (auto-mapped through
       the alias table) and run PINN / replay / compare in-app.

Local development::

    pip install gradio
    python gradio_app.py
    # http://127.0.0.1:7860

Hugging Face Spaces deployment::

    1. Create a Space with SDK = Gradio, Hardware = CPU-Basic.
    2. Push this file and ``requirements.txt`` to the Space repo.
    3. The Space rebuilds and the URL becomes the hosted demo.
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

# Headless matplotlib BEFORE any plotting import (HF workers have no display).
os.environ.setdefault("MPLBACKEND", "Agg")

# Lift the src/ layout onto sys.path so a clean clone runs out of the box.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# scripts/ is where the examples registry and the shared CSV / robot
# helpers live; both the Streamlit GUI and the Gradio GUI consume them.
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import gradio as gr                                                  # noqa: E402
import numpy as np                                                   # noqa: E402
from scipy.spatial.transform import Rotation                         # noqa: E402

from cdpr.core.frames import Pose                                    # noqa: E402
from cdpr.control.pd import PDController                             # noqa: E402
from cdpr.control.mpc import MPCController                           # noqa: E402
from cdpr.dynamics.rigid_body import PlatformState                   # noqa: E402
from cdpr.dynamics.simulator import simulate                         # noqa: E402
from cdpr.interface.specs import (                                   # noqa: E402
    SimulationRequest,
    TrajectorySpec,
    build_robot,
    build_trajectory,
)

# Shared CSV/robot helpers --- the alias mapping, URL ingest, robot
# reconstruction from manifest, schema report.
from _csv_io import load_csv_any, split_canonical_blocks             # noqa: E402

# The 5-example registry --- shared with the CLI and Streamlit.
from examples import EXAMPLES, list_examples                         # noqa: E402


# Internal build id --- written into manifests and the Console panel
# for traceability; intentionally not surfaced in the page header.
BUILD_ID = "gradio-2026-05-30-b"


# ---------------------------------------------------------------------------
# Styling
#
# A reserved, instrument-style look: slate gradient header, monospace
# accents on parameter names, left-bordered field cards. No animations,
# no glass, no rainbows. The aim is for the page to read like a
# measurement instrument's web UI, not a SaaS landing page.
# ---------------------------------------------------------------------------

_CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }

#cdpr-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
    color: #f8fafc;
    padding: 1.4rem 1.8rem;
    border-radius: 0.6rem;
    margin: 0 0 1rem 0;
    box-shadow: 0 1px 4px rgba(15, 23, 42, 0.18);
}
#cdpr-header h1 {
    color: #ffffff !important;
    margin: 0 0 0.35rem 0 !important;
    font-weight: 600;
    font-size: 1.65rem;
    letter-spacing: -0.012em;
}
#cdpr-header .lead {
    color: #cbd5e1 !important;
    margin: 0 !important;
    line-height: 1.55;
    font-size: 0.97rem;
}
#cdpr-header .lead b { color: #f1f5f9; }
#cdpr-header .lead code {
    background: rgba(255, 255, 255, 0.13);
    color: #f1f5f9;
    padding: 0 0.35em;
    border-radius: 0.25em;
    font-size: 0.86em;
}

.cdpr-howto {
    background: #f8fafc;
    border-left: 3px solid #2c5282;
    padding: 0.7rem 1rem;
    border-radius: 0 0.35rem 0.35rem 0;
    margin: 0 0 1.2rem 0;
    font-size: 0.93rem;
    color: #334155;
}
.cdpr-howto strong { color: #1e3a5f; }
.cdpr-howto em { color: #475569; font-style: normal; font-weight: 500; }

.cdpr-chat-card {
    border: 1px solid #e2e8f0;
    background: #fafbfc;
    border-radius: 0.5rem;
    padding: 0.6rem 0.75rem;
    margin-bottom: 1rem;
}
.cdpr-chat-card h4 {
    color: #1e3a5f;
    margin: 0 0 0.4rem 0;
    font-weight: 600;
    font-size: 1rem;
}
.cdpr-chat-hint {
    color: #64748b;
    font-size: 0.85rem;
    margin: 0 0 0.6rem 0;
}

.tab-nav button.selected {
    border-bottom-color: #2c5282 !important;
    color: #1e3a5f !important;
}
.tab-nav button { font-weight: 500 !important; }

#cdpr-footer {
    margin-top: 1.8rem;
    padding-top: 0.9rem;
    border-top: 1px solid #e2e8f0;
    font-size: 0.85rem;
    color: #64748b;
}
#cdpr-footer code { color: #1e3a5f; }
#cdpr-footer a { color: #2c5282; }
"""


_HEADER_HTML = """
<div id='cdpr-header'>
  <h1>CDPR Simulator</h1>
  <p class='lead'>
    A research-grade computational platform for <b>Cable-Driven Parallel
    Robots</b>: inverse and forward kinematics, the wrench-mapping and
    tension-distribution QP, coupled rigid-body dynamics, three
    constitutive cable models (massless &middot; Kelvin&ndash;Voigt
    &middot; Irvine catenary, plus a hybrid), feedback control via
    <code>PD</code> or receding-horizon <code>MPC</code>, parametric
    trajectories with jerk-limited time-scaling, and a Phase-2 data
    layer that ingests CSV experiment logs to train
    PINN&nbsp;/&nbsp;MLP&nbsp;/&nbsp;PPO&nbsp;/&nbsp;SAC models for
    inverse-dynamics learning and sim-to-data comparison.
  </p>
</div>
"""


_HOWTO_HTML = """
<div class='cdpr-howto'>
  <strong>How to use this app.</strong>
  <em>(1) Built-in examples</em> &mdash; five one-click runs covering
  the three Phase-1 trajectories and two Phase-2 data-driven flows.
  <em>(2) Custom Phase-1</em> &mdash; describe what you want in plain
  English (the chat box parses it through the LLM layer and pre-fills
  the form) <strong>or</strong> set parameters by hand; pick
  <strong>PD</strong> or <strong>MPC</strong>; press
  <strong>Run simulation</strong>.
  <em>(3) Upload CSV / Phase-2</em> &mdash; drop any experiment log and
  train or compare the five models on it.
</div>
"""


# ---------------------------------------------------------------------------
# Phase-1 --- Built-in examples tab helpers
# ---------------------------------------------------------------------------

def _run_example_via_subprocess(name: str) -> tuple[str, list[str], dict]:
    """Shell out to scripts/run_example.py so the in-Python process
    stays clean (matplotlib + torch + sb3 imports leak memory across
    runs otherwise). Returns (log, image_paths, metrics_dict)."""
    out_root = _ROOT / "out" / EXAMPLES[name]["out_dir"]
    cmd = [sys.executable, str(_SCRIPTS / "run_example.py"), "--name", name]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        return (
            f"FAILED (exit {proc.returncode}) in {dt:.1f}s\n\n--- stderr ---\n"
            + (proc.stderr or "")[-4000:],
            [],
            {},
        )

    # Collect images and metrics.
    pngs = sorted(out_root.glob("*.png")) if out_root.exists() else []
    metrics = {}
    feas = out_root / "feasibility.json"
    if feas.exists():
        metrics = json.loads(feas.read_text(encoding="utf-8"))
    manifest = out_root / "manifest.json"
    if manifest.exists():
        man = json.loads(manifest.read_text(encoding="utf-8"))
        metrics["_manifest_summary"] = {
            k: man.get(k) for k in ("git_hash", "samples", "runtime_s",
                                     "controller", "example", "title")
        }
    log = (
        f"[done] example '{name}' in {dt:.1f} s\n"
        f"output : {out_root}\n"
        f"figures: {len(pngs)}\n"
        f"build  : {BUILD_ID}\n\n"
        + (proc.stdout or "")[-2000:]
    )
    return log, [str(p) for p in pngs], metrics


def _examples_choices() -> list[str]:
    return [f"[{e['name']}] {e['title']}" for e in list_examples()]


def _example_name_from_label(label: str) -> str:
    return label.split("]", 1)[0].strip("[").strip()


def _example_description(label: str) -> str:
    name = _example_name_from_label(label)
    entry = EXAMPLES[name]
    deps = (f"\n\n*Depends on*: example `{entry['depends_on']}` "
            "(auto-run if its CSV is missing).") if entry.get("depends_on") else ""
    return (f"**Phase {entry['phase']} --- {entry['title']}**\n\n"
            f"{entry['description']}{deps}")


# ---------------------------------------------------------------------------
# Phase-1 --- Custom simulation
# ---------------------------------------------------------------------------

def _run_custom_simulation(
    robot_name: str, kind: str, duration: float, dt: float,
    payload_mass: float, gravity_on: bool, objective: str,
    t_min: float, t_max: float,
    # Controller selection + gains
    controller_kind: str,
    kp_pos: float, kp_rot: float,
    mpc_horizon: float, mpc_q_pos: float, mpc_q_vel: float,
    mpc_r_force: float, mpc_p_terminal: float,
    # Trajectory params (kind-specific; unused entries are ignored)
    circle_center: str, circle_radius: float, circle_axis: str, circle_angle_span: float,
    line_start: str, line_end: str,
    lj_center: str, lj_amps: str, lj_freqs: str, lj_phases: str,
) -> tuple[str, list[str], str]:
    """Run a custom Phase-1 simulation in-process and return
    ``(status_text, image_paths, csv_download_path)``."""
    try:
        # Build kind-specific params.
        def _xyz(s: str, default: list[float]) -> list[float]:
            try:
                parts = [p.strip() for p in str(s).split(",") if p.strip()]
                return [float(p) for p in parts] if len(parts) == 3 else default
            except Exception:
                return default

        if kind == "line":
            params: dict = {
                "start": _xyz(line_start, [0.0, 0.0, 0.5]),
                "end":   _xyz(line_end,   [0.05, 0.0, 0.5]),
            }
        elif kind == "circle":
            params = {
                "center": _xyz(circle_center, [0.0, 0.0, 0.65]),
                "radius": float(circle_radius),
                "axis":   _xyz(circle_axis, [0.0, 0.0, 1.0]),
                "angle_span": float(circle_angle_span),
            }
        elif kind == "lissajous":
            params = {
                "center":      _xyz(lj_center, [0.0, 0.0, 0.65]),
                "amplitudes":  _xyz(lj_amps,   [0.03, 0.03, 0.0]),
                "frequencies": _xyz(lj_freqs,  [1.0, 2.0, 0.0]),
                "phases":      _xyz(lj_phases, [0.0, np.pi / 2, 0.0]),
            }
        else:
            params = {}

        gravity_vec = (0.0, 0.0, -9.81 if gravity_on else 0.0)
        request = SimulationRequest(
            robot=robot_name,
            payload_mass=float(payload_mass),
            gravity=gravity_vec,
            tension_objective=objective,
            duration=float(duration),
            dt=float(dt),
            trajectory=TrajectorySpec(
                kind=kind, duration=float(duration), params=params,
            ),
        )

        robot = build_robot(
            request.robot, payload_mass=request.payload_mass,
            t_min=float(t_min) if t_min > 0 else None,
            t_max=float(t_max) if t_max > 0 else None,
        )
        reference = build_trajectory(request.trajectory)
        p0 = reference(0.0).position
        state0 = PlatformState.at_rest(Pose(position=p0, rotation=Rotation.identity()))

        # ------------------- controller construction --------------------
        # PD = the safe default; MPC = receding-horizon translational
        # solver on top of a PD orientation loop (cdpr.control.mpc).
        # Cable bounds are enforced downstream by the tension-distribution
        # QP regardless of the controller choice.
        if str(controller_kind).upper() == "MPC":
            kd_rot = 2.0 * float(np.sqrt(kp_rot))
            controller = MPCController(
                horizon=int(mpc_horizon) if mpc_horizon and mpc_horizon > 0 else 8,
                dt=float(dt),
                Q_pos=float(mpc_q_pos),
                Q_vel=float(mpc_q_vel),
                R_force=float(mpc_r_force),
                P_terminal=float(mpc_p_terminal),
                Kp_rot=float(kp_rot),
                Kd_rot=kd_rot,
                gravity_compensation=True,
                cancel_external=True,
            )
            controller_label = (
                f"MPC(N={int(controller.horizon)}, "
                f"Q_pos={mpc_q_pos:g}, Q_vel={mpc_q_vel:g}, "
                f"R_u={mpc_r_force:g}, P_term={mpc_p_terminal:g}; "
                f"orientation PD Kp={kp_rot:g}, Kd={kd_rot:g})"
            )
        else:
            kd_pos = 2.0 * float(np.sqrt(kp_pos))
            kd_rot = 2.0 * float(np.sqrt(kp_rot))
            controller = PDController(
                Kp_pos=kp_pos, Kd_pos=kd_pos, Kp_rot=kp_rot, Kd_rot=kd_rot,
                gravity_compensation=True, cancel_external=True,
            )
            controller_label = (
                f"PD(Kp_pos={kp_pos:g}, Kd_pos={kd_pos:g}, "
                f"Kp_rot={kp_rot:g}, Kd_rot={kd_rot:g})"
            )

        t0 = time.perf_counter()
        result = simulate(
            robot=robot, state0=state0,
            duration=request.duration, dt=request.dt,
            reference=reference, controller=controller,
            tension_objective=request.tension_objective,
            gravity=request.gravity,
        )
        dt_run = time.perf_counter() - t0

        # Write outputs + render plots into a unique timestamped dir.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = _ROOT / "out" / f"gradio-custom-{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Reuse the example runner's plot bundle for consistency.
        from examples import _render_phase1_plots, _save_csv
        csv_path = _save_csv(
            out_dir, result, reference=reference,
            t_min=float(robot.limits.t_min[0]),
            t_max=float(robot.limits.t_max[0]),
        )
        figs = _render_phase1_plots(out_dir, result, robot, reference)

        # Extra figure: dense reference vs realised trajectory in 3D.
        # Makes shape-level differences (circle vs figure-eight vs line)
        # unambiguous at a glance. Built from the reference callable and
        # the actual result.positions of THIS run --- no caching.
        _render_reference_overlay(
            out_dir, reference, result, request.trajectory, controller_label,
        )

        tens = np.asarray(result.cable_tensions)
        ref_pos = np.array([reference(t).position for t in result.time])
        err = np.linalg.norm(np.asarray(result.positions) - ref_pos, axis=1)
        metrics = {
            "samples": int(len(result.time)),
            "wall_time_s": round(dt_run, 3),
            "tension_min_N": float(tens.min()),
            "tension_max_N": float(tens.max()),
            "infeasible_steps": int(len(result.infeasible_steps)),
            "tracking_rms_mm": float(np.sqrt(np.mean(err ** 2)) * 1e3),
            "tracking_peak_mm": float(err.max() * 1e3),
        }
        manifest_path = _write_run_manifest(
            out_dir, request, result, controller_label,
            kp_pos=kp_pos, kp_rot=kp_rot,
            mpc_horizon=mpc_horizon, mpc_q_pos=mpc_q_pos, mpc_q_vel=mpc_q_vel,
            mpc_r_force=mpc_r_force, mpc_p_terminal=mpc_p_terminal,
            t_min=t_min, t_max=t_max, gravity_on=gravity_on,
            metrics=metrics,
        )
        # SHA1 of the realised position trajectory --- two runs that share
        # this fingerprint are bit-for-bit identical. Two runs that differ
        # anywhere in inputs produce different fingerprints, which proves
        # the figures cannot have come from a cached previous run.
        import hashlib
        fp_pos = hashlib.sha1(np.asarray(result.positions).tobytes()).hexdigest()[:12]

        status = (
            f"[done] {len(result.time)} samples in {dt_run:.1f} s\n"
            f"run id       : {stamp}\n"
            f"controller   : {controller_label}\n"
            f"trajectory   : {kind} {dict(params)!r}\n"
            f"tensions     : [{tens.min():.2f}, {tens.max():.2f}] N\n"
            f"infeasible   : {len(result.infeasible_steps)} steps\n"
            f"tracking     : RMS {metrics['tracking_rms_mm']:.2f} mm  "
            f"peak {metrics['tracking_peak_mm']:.2f} mm\n"
            f"position hash: {fp_pos}   (different every distinct run)\n"
            f"manifest     : {manifest_path.name}\n"
            f"figures      : {len(figs) + 1} PNGs in {out_dir}\n"
            f"build        : {BUILD_ID}\n"
        )
        png_paths = sorted(str(p) for p in out_dir.glob("*.png"))
        return status, png_paths, str(csv_path)
    except Exception as exc:
        return (f"ERROR: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                [], "")


# ---------------------------------------------------------------------------
# Per-run audit artefacts: manifest + reference overlay
# ---------------------------------------------------------------------------
# Two artefacts that make every run independently auditable:
#
#  * ``run_manifest.json`` --- every input parameter + the resulting
#    metrics + a SHA1 of the realised trajectory. If two runs share a
#    fingerprint, they ARE the same simulation; if the fingerprint
#    differs, every figure built from result.positions necessarily
#    differs too.
#  * ``reference_vs_actual_3d.png`` --- the high-resolution reference
#    trajectory (from the analytic callable) overlaid with the actual
#    realised trajectory (from the simulation). Shape-level differences
#    (circle vs figure-eight vs line vs hold) jump out instantly.


def _write_run_manifest(
    out_dir: Path, request: SimulationRequest, result, controller_label: str,
    *, kp_pos: float, kp_rot: float,
    mpc_horizon: float, mpc_q_pos: float, mpc_q_vel: float,
    mpc_r_force: float, mpc_p_terminal: float,
    t_min: float, t_max: float, gravity_on: bool,
    metrics: dict,
) -> Path:
    """Write a per-run JSON manifest. The fingerprint at the bottom is
    the SHA1 of the realised trajectory; two runs with the same
    fingerprint are bit-for-bit identical."""
    import hashlib
    positions = np.asarray(result.positions)
    tensions = np.asarray(result.cable_tensions)
    manifest = {
        "build_id": BUILD_ID,
        "run_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "robot": request.robot,
        "payload_mass_kg": float(request.payload_mass),
        "gravity_on": bool(gravity_on),
        "tension_objective": request.tension_objective,
        "tension_bounds_N": [float(t_min), float(t_max)],
        "duration_s": float(request.duration),
        "dt_s": float(request.dt),
        "trajectory": {
            "kind": request.trajectory.kind,
            "params": request.trajectory.params,
        },
        "controller": {
            "label": controller_label,
            "kp_pos_pd": float(kp_pos),
            "kp_rot_shared": float(kp_rot),
            "mpc": {
                "horizon": int(mpc_horizon),
                "Q_pos": float(mpc_q_pos),
                "Q_vel": float(mpc_q_vel),
                "R_force": float(mpc_r_force),
                "P_terminal": float(mpc_p_terminal),
            },
        },
        "metrics": metrics,
        "fingerprint": {
            "positions_sha1": hashlib.sha1(positions.tobytes()).hexdigest(),
            "tensions_sha1":  hashlib.sha1(tensions.tobytes()).hexdigest(),
            "samples":        int(positions.shape[0]),
            "cables":         int(tensions.shape[1]),
        },
    }
    path = out_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _render_reference_overlay(
    out_dir: Path, reference, result, traj_spec, controller_label: str,
) -> None:
    """Render reference_vs_actual_3d.png + reference_vs_actual_xy.png.

    The reference path is sampled at HIGH resolution from the analytic
    ``reference(t)`` callable, so the curve looks smooth regardless of
    how coarse the simulation dt is. The actual realised trajectory
    comes from ``result.positions``. Both are plotted in the same axes
    so the shape difference is unmistakable."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D                            # noqa: F401

    try:
        from cdpr.viz.style import apply_paper_style
        apply_paper_style()
    except Exception:                                                  # pragma: no cover
        pass

    t = np.asarray(result.time)
    if t.size < 2:
        return
    # Sample the reference at 10x the simulation rate (capped at 4000
    # samples so the figure stays light).
    n_ref = int(min(4000, max(400, 10 * t.size)))
    t_ref = np.linspace(float(t[0]), float(t[-1]), n_ref)
    ref_xyz = np.asarray([reference(tt).position for tt in t_ref])
    act_xyz = np.asarray(result.positions)

    fig = plt.figure(figsize=(8.2, 5.2))
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2],
              "-", color="C0", lw=1.5, label="reference")
    ax3.plot(act_xyz[:, 0], act_xyz[:, 1], act_xyz[:, 2],
              "--", color="C3", lw=1.3, label="actual")
    ax3.set_xlabel("x [m]"); ax3.set_ylabel("y [m]"); ax3.set_zlabel("z [m]")
    ax3.legend(loc="best", fontsize=8)
    ax3.set_title(f"Reference vs actual ({traj_spec.kind})")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(ref_xyz[:, 0], ref_xyz[:, 1], "-",
              color="C0", lw=1.5, label="reference")
    ax2.plot(act_xyz[:, 0], act_xyz[:, 1], "--",
              color="C3", lw=1.3, label="actual")
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    ax2.set_aspect("equal", adjustable="datalim")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("xy projection")

    fig.suptitle(
        f"Run fingerprint stamp --- controller = {controller_label[:60]}",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "reference_vs_actual.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Phase-1 --- Conversational simulation builder (chat box)
# ---------------------------------------------------------------------------
# Parses a free-text description through cdpr.llm.simulation_builder and
# emits gr.update() values for every parameter on the form, so the user
# can review what the LLM inferred and tweak it before pressing Run. If
# the LLM layer raises, the chat reports the failure and leaves the form
# untouched --- the "never crash" rule from the directive.


# 16 trajectory/dynamics fields + 1 controller_kind + 2 group-visibility
# toggles (pd_group, mpc_group) = 19 form-side outputs from the chat
# handler. _no_form_changes() supplies the matching "do nothing" tuple
# for the empty-message + parser-failure branches.
_FORM_FIELD_COUNT = 19


def _no_form_changes() -> tuple[Any, ...]:
    return tuple(gr.update() for _ in range(_FORM_FIELD_COUNT))


def _infer_controller(text: str) -> str | None:
    """Heuristic: scan the description for an explicit controller hint
    and return 'MPC' / 'PD' / None. Used so the chat box can flip the
    controller radio when the user says 'use MPC' --- the LLM layer
    parses trajectory and robot, but controller choice is a frontend
    concern, not a SimulationRequest field."""
    s = (text or "").lower()
    # MPC patterns
    if ("mpc" in s
            or "model predictive" in s
            or "predictive control" in s
            or "receding horizon" in s
            or "receding-horizon" in s):
        return "MPC"
    # PD patterns --- require word boundaries so 'pd' inside other words
    # doesn't trigger a false positive.
    if ("pd controller" in s
            or "proportional derivative" in s
            or "proportional-derivative" in s):
        return "PD"
    return None


def _xyz_str(value: Any, default: str) -> str:
    """Convert a 3-vector (any iterable) into the comma-separated
    string format the form widgets expect."""
    try:
        seq = list(value)
        if len(seq) == 3:
            return ",".join(f"{float(x):g}" for x in seq)
    except (TypeError, ValueError):
        pass
    return default


def _chat_to_form(message: str, history: list[dict[str, str]] | None):
    """Parse the user's English description into a SimulationRequest
    and return chatbot + form updates."""
    msg = (message or "").strip()
    hist: list[dict[str, str]] = list(history or [])
    if not msg:
        return (hist, "", *_no_form_changes())

    hist.append({"role": "user", "content": msg})

    try:
        from cdpr.llm.simulation_builder import describe_to_request
        result = describe_to_request(msg)
    except Exception as exc:                                        # noqa: BLE001
        hist.append({
            "role": "assistant",
            "content": (
                f"Could not parse that description "
                f"(`{type(exc).__name__}`: {exc}). "
                "The form below is unchanged --- you can configure it by "
                "hand and press **Run simulation**."
            ),
        })
        return (hist, "", *_no_form_changes())

    req = result.request
    params = dict(req.trajectory.params or {})
    kind = req.trajectory.kind

    # Controller hint comes from the original message text (the LLM
    # parser deliberately doesn't put controller in SimulationRequest
    # --- it is a frontend choice). Falling back to PD when unspecified
    # keeps the UI deterministic and matches the form's default.
    controller_hint = _infer_controller(msg) or "PD"
    is_mpc = controller_hint == "MPC"

    # Build assistant reply summarising what was parsed.
    bullets = [
        f"- robot: `{req.robot}`",
        f"- trajectory: `{kind}`",
        f"- controller: `{controller_hint}`",
        f"- duration: `{req.duration:g} s`  &middot;  dt: `{req.dt:g} s`",
        f"- payload: `{req.payload_mass:g} kg`",
        f"- objective: `{req.tension_objective}`",
    ]
    for key, val in params.items():
        bullets.append(f"  - `{key}`: `{val}`")

    lines = [
        (f"**Parsed by provider `{result.provider or 'echo'}` "
         f"(model `{result.model or 'n/a'}`)** "
         f"&mdash; confidence `{result.confidence}`."),
        "",
        "**Configuration**",
        *bullets,
    ]
    if result.follow_up_questions:
        lines += ["", "**I would clarify**",
                  *[f"- {q}" for q in result.follow_up_questions]]
    if result.notes:
        lines += ["", "*Notes*", *[f"- {n}" for n in result.notes]]
    lines += ["",
              "Form fields below have been updated. Edit if needed, "
              "then press **Run simulation**."]
    hist.append({"role": "assistant", "content": "\n".join(lines)})

    # Emit form updates. We always update every group's defaults so the
    # user can switch trajectory kind after the parse without losing
    # context --- the params dict for the chosen kind contributes, and
    # the others get sensible fallbacks. The last three updates flip
    # the controller radio and the conditional PD/MPC gain groups; in
    # Gradio 6 a programmatic value change does not re-fire the radio's
    # .change() listener, so we set both visibilities here explicitly.
    return (
        hist,
        "",                                              # clear chat input
        gr.update(value=req.robot),                      # robot_name
        gr.update(value=kind),                           # kind
        gr.update(value=float(req.duration)),            # duration
        gr.update(value=float(req.dt)),                  # dt_input
        gr.update(value=float(req.payload_mass)),        # payload_mass
        gr.update(value=req.tension_objective),          # objective
        gr.update(value=_xyz_str(params.get("start"),       "0,0,0.5")),
        gr.update(value=_xyz_str(params.get("end"),         "0.05,0,0.5")),
        gr.update(value=_xyz_str(params.get("center"),      "0,0,0.65")),
        gr.update(value=float(params.get("radius", 0.05))),
        gr.update(value=_xyz_str(params.get("axis"),        "0,0,1")),
        gr.update(value=float(params.get("angle_span", 4 * np.pi))),
        gr.update(value=_xyz_str(params.get("center"),      "0,0,0.65")),
        gr.update(value=_xyz_str(params.get("amplitudes"),  "0.03,0.03,0")),
        gr.update(value=_xyz_str(params.get("frequencies"), "1,2,0")),
        gr.update(value=_xyz_str(params.get("phases"),      "0,1.5708,0")),
        gr.update(value=controller_hint),                # controller_kind
        gr.update(visible=not is_mpc),                   # pd_group
        gr.update(visible=is_mpc),                       # mpc_group
    )


# ---------------------------------------------------------------------------
# Phase-2 --- Upload CSV and analyse tab
# ---------------------------------------------------------------------------

def _analyse_uploaded_csv(file_obj, model: str, epochs: int) -> tuple[str, list[str]]:
    """Upload a CSV --> run train_from_csv as a subprocess --> display
    the resulting figures.

    Subprocess isolation keeps torch / sb3 imports from accumulating
    across calls (which the Streamlit version cannot do safely)."""
    if file_obj is None:
        return "Upload a CSV (or paste a URL) first.", []
    try:
        src = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", file_obj)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = _ROOT / "out" / f"gradio-{model}-{stamp}"

        cmd = [
            sys.executable, str(_SCRIPTS / "train_from_csv.py"),
            "--input", str(src),
            "--model", model,
            "--epochs", str(int(epochs)),
            "--out", str(out_dir),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
        dt = time.perf_counter() - t0
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")

        log = f"[{model}] exit={proc.returncode} in {dt:.1f} s\n"
        log += (proc.stdout or "")[-2500:]
        if proc.returncode != 0:
            log += "\n\n--- stderr (last 2 KB) ---\n" + (proc.stderr or "")[-2000:]

        pngs = sorted(str(p) for p in out_dir.glob("*.png")) if out_dir.exists() else []
        metrics = out_dir / "metrics.json"
        if metrics.exists():
            log += "\n\n--- metrics.json ---\n" + metrics.read_text(encoding="utf-8")
        return log, pngs
    except Exception as exc:
        return (f"ERROR: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                [])


def _compare_uploaded_csv(file_obj, epochs: int, rl_steps: int) -> tuple[str, list[str], str]:
    """Run the multi-model compare on an uploaded CSV. Heavy --- only
    suitable for the HF Spaces 16 GB workers, not the Streamlit Cloud
    1 GB worker (which is why this exists in Gradio in the first place)."""
    if file_obj is None:
        return "Upload a CSV first.", [], "{}"
    try:
        src = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", file_obj)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = _ROOT / "out" / f"gradio-compare-{stamp}"
        cmd = [
            sys.executable, str(_SCRIPTS / "compare_models.py"),
            "--input", str(src),
            "--out",   str(out_dir),
            "--models", "replay", "mlp", "pinn", "ppo", "sac",
            "--epochs", str(int(epochs)),
            "--rl-steps", str(int(rl_steps)),
            "--eval-episodes", "2",
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
        dt = time.perf_counter() - t0
        log = f"[compare] exit={proc.returncode} in {dt:.1f} s\n"
        log += (proc.stdout or "")[-2000:]
        if proc.returncode != 0:
            log += "\n\n--- stderr (last 2 KB) ---\n" + (proc.stderr or "")[-2000:]
        pngs = sorted(str(p) for p in (out_dir.glob("compare_*.png"))) if out_dir.exists() else []
        ranking_path = out_dir / "ranking.json"
        ranking_json = ranking_path.read_text(encoding="utf-8") if ranking_path.exists() else "{}"
        return log, pngs, ranking_json
    except Exception as exc:
        return (f"ERROR: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                [], "{}")


# ---------------------------------------------------------------------------
# UI assembly
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    # Gradio 6.0 moved ``theme`` and ``css`` from the Blocks() constructor
    # to ``launch(theme=..., css=...)``; we pass both via launch() below.
    with gr.Blocks(
        title="CDPR Simulator",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(_HEADER_HTML)
        gr.HTML(_HOWTO_HTML)

        # ===================================================================
        # Tab 1 --- Built-in examples
        # ===================================================================
        with gr.Tab("Built-in examples"):
            with gr.Row():
                ex_dropdown = gr.Dropdown(
                    choices=_examples_choices(),
                    value=_examples_choices()[0],
                    label="Pick a built-in example",
                )
            ex_description = gr.Markdown(_example_description(_examples_choices()[0]))
            ex_dropdown.change(
                fn=lambda lbl: _example_description(lbl),
                inputs=[ex_dropdown], outputs=[ex_description],
            )
            ex_run = gr.Button("Run example", variant="primary")
            ex_log = gr.Textbox(label="Console", lines=12, interactive=False)
            ex_gallery = gr.Gallery(label="Figures", columns=2, height="600px")
            ex_metrics = gr.JSON(label="feasibility.json / manifest summary")

            def _do_example(label):
                return _run_example_via_subprocess(_example_name_from_label(label))

            ex_run.click(
                fn=_do_example,
                inputs=[ex_dropdown],
                outputs=[ex_log, ex_gallery, ex_metrics],
            )

        # ===================================================================
        # Tab 2 --- Custom Phase-1 simulation (with chat box + MPC)
        # ===================================================================
        with gr.Tab("Custom Phase-1 simulation"):

            # ----- Conversational simulation builder ----------------------
            with gr.Group(elem_classes=["cdpr-chat-card"]):
                gr.HTML(
                    "<h4>Describe what you want to simulate</h4>"
                    "<p class='cdpr-chat-hint'>"
                    "The LLM layer parses your description into a "
                    "<code>SimulationRequest</code> and pre-fills the form "
                    "below. The chain falls through providers in order "
                    "&mdash; <b>Gemini</b> first; if it 429-rate-limits or "
                    "is unavailable, <b>OpenRouter / DeepSeek</b> takes the "
                    "next attempt; if that fails too, the <b>echo</b> stub "
                    "falls back to keyword-based intent detection. "
                    "Set <code>GEMINI_API_KEY</code> and "
                    "<code>OPENROUTER_API_KEY</code> as Space secrets to "
                    "enable both. See <code>docs/llm-providers.md</code>."
                    "</p>"
                )
                # Gradio 6.x: messages format is the default; ``buttons``
                # supersedes the old ``show_copy_button`` flag.
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=260,
                    buttons=["copy", "copy_all"],
                )
                with gr.Row():
                    chat_input = gr.Textbox(
                        placeholder=(
                            "e.g. 'Simulate an 8-cable CDPR carrying 5 kg "
                            "following a horizontal circle of radius 5 cm "
                            "for 10 seconds.'"
                        ),
                        show_label=False, lines=2, scale=8,
                    )
                    chat_send = gr.Button("Parse", variant="primary", scale=1)
                with gr.Row():
                    chat_clear = gr.Button("Clear conversation", variant="secondary")

            # ----- Manual form ---------------------------------------------
            # Four collapsible accordions, modelled on a file-tree: click
            # the heading to expand, click again to collapse so a
            # configured form takes one line each. Each accordion starts
            # CLOSED --- the chat box drives most parameter setting; users
            # who prefer manual configuration open the sections they need.

            with gr.Accordion("Robot & dynamics", open=False):
                robot_name = gr.Dropdown(
                    ["point_mass_3d", "planar_translational",
                     "ipanema_class", "cogiro_class", "dissertation_8cable"],
                    value="dissertation_8cable", label="Robot",
                )
                with gr.Row():
                    payload_mass = gr.Number(value=0.0, label="Payload mass [kg]",
                                              minimum=0.0, step=0.1)
                    gravity_on = gr.Checkbox(value=True, label="Apply gravity")
                objective = gr.Radio(
                    ["min_norm", "centered", "preferred"],
                    value="centered", label="Tension distribution objective",
                )
                with gr.Row():
                    t_min = gr.Number(value=5.0, label="Tension min [N]",
                                       minimum=0.0)
                    t_max = gr.Number(value=500.0, label="Tension max [N]",
                                       minimum=0.0)

            with gr.Accordion("Trajectory", open=False):
                kind = gr.Dropdown(
                    ["hold", "line", "circle", "lissajous"],
                    value="circle", label="Trajectory kind",
                )
                with gr.Accordion("Line parameters", open=False):
                    line_start = gr.Textbox(value="0,0,0.5",
                                            label="start (x,y,z) [m]")
                    line_end = gr.Textbox(value="0.05,0,0.5",
                                          label="end   (x,y,z) [m]")
                with gr.Accordion("Circle parameters", open=False):
                    circle_center = gr.Textbox(value="0,0,0.65",
                                                label="center (x,y,z) [m]")
                    with gr.Row():
                        circle_radius = gr.Number(value=0.05, label="radius [m]",
                                                  minimum=0.0, maximum=2.0)
                        circle_angle_span = gr.Number(
                            value=float(4 * np.pi), label="angle span [rad]",
                            minimum=0.1,
                        )
                    circle_axis = gr.Textbox(value="0,0,1", label="axis (x,y,z)")
                with gr.Accordion("Lissajous (figure-eight) parameters",
                                  open=False):
                    lj_center = gr.Textbox(value="0,0,0.65", label="center [m]")
                    lj_amps = gr.Textbox(value="0.03,0.03,0.0",
                                         label="amplitudes (x,y,z) [m]")
                    lj_freqs = gr.Textbox(value="1.0,2.0,0.0",
                                          label="frequencies (x,y,z) [rad/s]")
                    lj_phases = gr.Textbox(value="0,1.5708,0",
                                            label="phases (x,y,z) [rad]")

            with gr.Accordion("Time", open=False):
                with gr.Row():
                    duration = gr.Number(value=12.566, label="Duration [s]",
                                         minimum=0.05, maximum=300.0)
                    dt_input = gr.Number(value=1e-3, label="dt [s]",
                                         minimum=1e-4, maximum=5e-2)

            with gr.Accordion("Controller", open=False):
                controller_kind = gr.Radio(
                    ["PD", "MPC"], value="PD",
                    label="Family",
                    info=("PD is the robust default. MPC is a linear "
                          "receding-horizon solver over translational "
                          "dynamics with a PD orientation loop on top "
                          "(cdpr.control.mpc); cable bounds are enforced "
                          "by the tension-distribution QP either way."),
                )
                # Orientation PD gains are SHARED --- both PD and MPC use
                # them for the orientation channel.
                kp_rot = gr.Number(value=100.0,
                                    label="Kp_rot (orientation; shared between PD and MPC)",
                                    minimum=0.0)
                # PD-only group: translational position gain.
                with gr.Group(visible=True) as pd_group:
                    gr.Markdown("**PD --- translational gain**")
                    kp_pos = gr.Number(value=400.0,
                                        label="Kp_pos  (Kd_pos = 2·sqrt(Kp_pos))",
                                        minimum=0.0)
                # MPC-only group: horizon + QP weights.
                with gr.Group(visible=False) as mpc_group:
                    gr.Markdown(
                        "**MPC --- horizon and quadratic weights**  \n"
                        "*State* x = [p, v] (3+3). *Stage cost* = "
                        "Q_pos·||p−p_ref||² + Q_vel·||v−v_ref||² + R·||u||². "
                        "*Terminal* = P·||p−p_ref||²."
                    )
                    with gr.Row():
                        mpc_horizon = gr.Number(value=8, label="Horizon N (steps)",
                                                 minimum=2, maximum=50, step=1)
                        mpc_q_pos = gr.Number(value=2.0e3, label="Q_pos",
                                               minimum=0.0)
                        mpc_q_vel = gr.Number(value=2.0e1, label="Q_vel",
                                               minimum=0.0)
                    with gr.Row():
                        mpc_r_force = gr.Number(value=1.0e-3,
                                                 label="R (force penalty)",
                                                 minimum=0.0)
                        mpc_p_terminal = gr.Number(value=1.0e4,
                                                    label="P (terminal)",
                                                    minimum=0.0)

            def _toggle_controller(choice: str):
                is_mpc = (choice or "").upper() == "MPC"
                return gr.update(visible=not is_mpc), gr.update(visible=is_mpc)

            controller_kind.change(
                fn=_toggle_controller,
                inputs=[controller_kind],
                outputs=[pd_group, mpc_group],
            )

            # ----- Run + Reset + outputs ---------------------------------
            with gr.Row():
                run_btn = gr.Button("Run simulation", variant="primary", scale=3)
                reset_btn = gr.Button("Reset all", variant="secondary", scale=1)
            status_text = gr.Textbox(label="Run summary", lines=9, interactive=False)
            with gr.Row():
                fig_gallery = gr.Gallery(label="Figures", columns=2, height="600px")
            csv_out = gr.File(label="Download timeseries.csv")

            run_btn.click(
                fn=_run_custom_simulation,
                inputs=[
                    robot_name, kind, duration, dt_input,
                    payload_mass, gravity_on, objective,
                    t_min, t_max,
                    controller_kind,
                    kp_pos, kp_rot,
                    mpc_horizon, mpc_q_pos, mpc_q_vel,
                    mpc_r_force, mpc_p_terminal,
                    circle_center, circle_radius, circle_axis, circle_angle_span,
                    line_start, line_end,
                    lj_center, lj_amps, lj_freqs, lj_phases,
                ],
                outputs=[status_text, fig_gallery, csv_out],
            )

            # ----- Reset handler -----------------------------------------
            # Reset clears the gallery / status / CSV download and pushes
            # every form widget back to its factory default. The chat
            # history is intentionally preserved --- it documents the
            # research session.
            def _reset_form():
                return (
                    # outputs first (cleared)
                    "",                                              # status_text
                    None,                                            # fig_gallery
                    None,                                            # csv_out
                    # robot & dynamics
                    "dissertation_8cable",                           # robot_name
                    0.0,                                             # payload_mass
                    True,                                            # gravity_on
                    "centered",                                      # objective
                    5.0,                                             # t_min
                    500.0,                                           # t_max
                    # trajectory
                    "circle",                                        # kind
                    "0,0,0.5",                                       # line_start
                    "0.05,0,0.5",                                    # line_end
                    "0,0,0.65",                                      # circle_center
                    0.05,                                            # circle_radius
                    "0,0,1",                                         # circle_axis
                    float(4 * np.pi),                                # circle_angle_span
                    "0,0,0.65",                                      # lj_center
                    "0.03,0.03,0.0",                                 # lj_amps
                    "1.0,2.0,0.0",                                   # lj_freqs
                    "0,1.5708,0",                                    # lj_phases
                    # time
                    12.566,                                          # duration
                    1e-3,                                            # dt_input
                    # controller
                    "PD",                                            # controller_kind
                    400.0,                                           # kp_pos
                    100.0,                                           # kp_rot
                    8,                                               # mpc_horizon
                    2.0e3,                                           # mpc_q_pos
                    2.0e1,                                           # mpc_q_vel
                    1.0e-3,                                          # mpc_r_force
                    1.0e4,                                           # mpc_p_terminal
                    # group visibility
                    gr.update(visible=True),                         # pd_group
                    gr.update(visible=False),                        # mpc_group
                )

            reset_btn.click(
                fn=_reset_form,
                inputs=None,
                outputs=[
                    status_text, fig_gallery, csv_out,
                    robot_name, payload_mass, gravity_on, objective, t_min, t_max,
                    kind,
                    line_start, line_end,
                    circle_center, circle_radius, circle_axis, circle_angle_span,
                    lj_center, lj_amps, lj_freqs, lj_phases,
                    duration, dt_input,
                    controller_kind, kp_pos, kp_rot,
                    mpc_horizon, mpc_q_pos, mpc_q_vel,
                    mpc_r_force, mpc_p_terminal,
                    pd_group, mpc_group,
                ],
            )

            # Chat wiring --- must come AFTER the form widgets and the
            # controller group definitions exist. The trailing three
            # outputs (controller_kind, pd_group, mpc_group) let a chat
            # prompt that mentions 'MPC' flip the radio AND reveal the
            # MPC fields in one round-trip.
            chat_outputs = [
                chatbot, chat_input,
                robot_name, kind, duration, dt_input,
                payload_mass, objective,
                line_start, line_end,
                circle_center, circle_radius, circle_axis, circle_angle_span,
                lj_center, lj_amps, lj_freqs, lj_phases,
                controller_kind, pd_group, mpc_group,
            ]
            chat_send.click(
                fn=_chat_to_form,
                inputs=[chat_input, chatbot],
                outputs=chat_outputs,
            )
            chat_input.submit(
                fn=_chat_to_form,
                inputs=[chat_input, chatbot],
                outputs=chat_outputs,
            )
            chat_clear.click(
                fn=lambda: ([], ""),
                inputs=None,
                outputs=[chatbot, chat_input],
            )

        # ===================================================================
        # Tab 3 --- Upload CSV / Phase-2
        # ===================================================================
        with gr.Tab("Upload CSV / Phase-2"):
            gr.Markdown(
                "Drop a CSV that follows the `scripts/run_simulation.py` "
                "layout (or any CSV whose columns can be auto-mapped --- "
                "see `docs/csv-schema.md`). The Gradio worker has enough "
                "memory to run all five models; the Streamlit Cloud worker "
                "does not --- which is the point of this alternative "
                "frontend."
            )
            upload = gr.File(
                label="Upload timeseries.csv (or any compatible CSV)",
                file_types=[".csv"],
            )

            with gr.Tab("Single model"):
                with gr.Row():
                    model = gr.Radio(
                        ["replay", "mlp", "pinn", "ppo", "sac"],
                        value="pinn", label="Model",
                    )
                    epochs = gr.Number(value=80, label="Epochs (mlp/pinn)",
                                       minimum=1, maximum=10000)
                analyse_btn = gr.Button("Run", variant="primary")
                analyse_log = gr.Textbox(label="Console", lines=20, interactive=False)
                analyse_gallery = gr.Gallery(label="Figures", columns=2,
                                              height="500px")
                analyse_btn.click(
                    fn=_analyse_uploaded_csv,
                    inputs=[upload, model, epochs],
                    outputs=[analyse_log, analyse_gallery],
                )

            with gr.Tab("Compare 5 models"):
                with gr.Row():
                    cmp_epochs = gr.Number(value=60, label="Epochs (mlp/pinn)",
                                            minimum=1, maximum=10000)
                    cmp_rl_steps = gr.Number(value=2000, label="RL train steps",
                                              minimum=0, maximum=10000000)
                cmp_btn = gr.Button("Run multi-model comparison", variant="primary")
                cmp_log = gr.Textbox(label="Console", lines=20, interactive=False)
                cmp_gallery = gr.Gallery(label="Compare figures",
                                          columns=2, height="500px")
                cmp_ranking = gr.JSON(label="ranking.json")
                cmp_btn.click(
                    fn=_compare_uploaded_csv,
                    inputs=[upload, cmp_epochs, cmp_rl_steps],
                    outputs=[cmp_log, cmp_gallery, cmp_ranking],
                )

        gr.HTML(
            "<div id='cdpr-footer'>"
            "<em>Architectural rationale</em>: "
            "<code>docs/frontend-architecture.md</code> &middot; "
            "<em>Terminal guide</em>: <code>docs/terminal-execution.md</code> "
            "&middot; <em>Examples</em>: <code>docs/examples.md</code> "
            "&middot; <em>LLM setup</em>: <code>docs/llm-providers.md</code> "
            "&middot; <em>Sources</em>: "
            "<a href='https://github.com/Tachia/cdpr_simulator' target='_blank'>"
            "github.com/Tachia/cdpr_simulator</a>"
            "</div>"
        )
    return demo


# Pre-warm the LLM provider chain at module load time, BEFORE the
# Blocks are built. This pays the GeminiProvider / OpenRouterProvider
# construction cost (a few hundred ms on a cold HF worker) once at boot
# instead of on the user's first chat. The function is best-effort and
# silently moves on if a provider can't be built (no API key, etc.).
try:
    from cdpr.llm.simulation_builder import prewarm as _llm_prewarm
    _llm_prewarm()
except Exception:                                                    # pragma: no cover
    pass


# Module-level ``demo`` --- required by Hugging Face Spaces, which does
# ``from gradio_app import demo`` (or ``from app import demo`` via the
# tiny app.py shim). Building it here at import time keeps the launch
# path identical between ``python gradio_app.py`` (local) and HF's
# auto-launch.
demo = build_ui()


if __name__ == "__main__":
    # Local launch. ``server_name=0.0.0.0`` makes the server reachable
    # from the LAN (HF Spaces requires it; on the dev machine 127.0.0.1
    # is the same address served at http://127.0.0.1:7860).
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
        # Gradio 6.0 expects theme + css here, not on Blocks().
        # Soft is the closest match to the previous look.
        theme=gr.themes.Soft(),
        css=_CSS,
        # share=True would give a temporary tunnel URL --- uncomment
        # if you need a public link without going through HF Spaces.
        # share=True,
    )
