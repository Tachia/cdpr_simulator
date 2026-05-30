"""Gradio frontend for CDPR_SIMULATOR — the recommended hosted demo.

Designed for **Hugging Face Spaces** (free 16 GB / 2 vCPU CPU Space).
The architectural rationale lives in ``docs/frontend-architecture.md``;
this file is the production code that backs it.

Why Gradio rather than Streamlit (in one sentence):

    Gradio is REST-style by default, file uploads survive corporate
    firewalls, the HF Spaces worker has 16 GB of RAM (vs Streamlit
    Cloud's 1 GB), and per-call state isolation means a slider tick
    never re-runs the entire page.

The app exposes three tabs that mirror the directive's two phases:

    1. **Run a built-in example**   — the same registry used by
       ``scripts/run_example.py`` and the Streamlit GUI. One click
       runs the example and the produced figures stream into the page.
    2. **Custom Phase-1 simulation** — pick robot + trajectory +
       controller + tension bounds, hit *Run simulation*, get the
       full 14-figure bundle and CSV download link.
    3. **Phase-2 CSV training / comparison** — upload any CSV
       (auto-mapped through the alias table) and run PINN / replay /
       compare in-app.

Local development:

    pip install gradio   # if not already on the cdpr extras
    python gradio_app.py
    # http://127.0.0.1:7860

Hugging Face Spaces deployment:

    1. Create a Space with SDK = Gradio, Hardware = CPU-Basic.
    2. Upload this file and ``requirements-gradio.txt`` at the root.
    3. The Space rebuilds; the URL becomes the canonical hosted demo.
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


BUILD_ID = "gradio-2026-05-29-a"


# ---------------------------------------------------------------------------
# Phase-1 — Built-in examples tab
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
        f"figures: {len(pngs)}\n\n"
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
    deps = f"\n\n*Depends on*: example `{entry['depends_on']}` "\
           "(auto-run if its CSV is missing)." if entry.get("depends_on") else ""
    return (f"**Phase {entry['phase']} — {entry['title']}**\n\n"
            f"{entry['description']}{deps}")


# ---------------------------------------------------------------------------
# Phase-1 — Custom simulation tab
# ---------------------------------------------------------------------------

def _run_custom_simulation(
    robot_name: str, kind: str, duration: float, dt: float,
    payload_mass: float, gravity_on: bool, objective: str,
    t_min: float, t_max: float, kp_pos: float, kp_rot: float,
    # circle params
    circle_center: str, circle_radius: float, circle_axis: str, circle_angle_span: float,
    # line params
    line_start: str, line_end: str,
    # lissajous params
    lj_center: str, lj_amps: str, lj_freqs: str, lj_phases: str,
) -> tuple[str, list[str], str]:
    """Run a custom Phase-1 simulation in-process and return
    (status_text, image_paths, csv_download_path)."""
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
                "end":   _xyz(line_end,   [0.3, 0.0, 0.5]),
            }
        elif kind == "circle":
            params = {
                "center": _xyz(circle_center, [0.0, 0.0, 0.5]),
                "radius": float(circle_radius),
                "axis":   _xyz(circle_axis, [0.0, 0.0, 1.0]),
                "angle_span": float(circle_angle_span),
            }
        elif kind == "lissajous":
            params = {
                "center":      _xyz(lj_center, [0.0, 0.0, 0.5]),
                "amplitudes":  _xyz(lj_amps,   [0.2, 0.2, 0.0]),
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
        kd_pos = 2.0 * float(np.sqrt(kp_pos))
        kd_rot = 2.0 * float(np.sqrt(kp_rot))
        controller = PDController(
            Kp_pos=kp_pos, Kd_pos=kd_pos, Kp_rot=kp_rot, Kd_rot=kd_rot,
            gravity_compensation=True, cancel_external=True,
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

        # Write outputs + render plots into a unique temp dir.
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

        tens = np.asarray(result.cable_tensions)
        ref_pos = np.array([reference(t).position for t in result.time])
        err = np.linalg.norm(np.asarray(result.positions) - ref_pos, axis=1)
        status = (
            f"[done] {len(result.time)} samples in {dt_run:.1f} s\n"
            f"tensions   : [{float(tens.min()):.2f}, {float(tens.max()):.2f}] N\n"
            f"infeasible : {len(result.infeasible_steps)} steps\n"
            f"tracking   : RMS {float(np.sqrt(np.mean(err**2)))*1e3:.2f} mm  "
            f"peak {float(err.max())*1e3:.2f} mm\n"
            f"figures    : {len(figs)} PNGs in {out_dir}\n"
        )
        png_paths = sorted(str(p) for p in out_dir.glob("*.png"))
        return status, png_paths, str(csv_path)
    except Exception as exc:
        return (f"ERROR: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                [], "")


# ---------------------------------------------------------------------------
# Phase-2 — Upload CSV and analyse tab
# ---------------------------------------------------------------------------

def _analyse_uploaded_csv(file_obj, model: str, epochs: int) -> tuple[str, list[str]]:
    """Upload a CSV → run train_from_csv as a subprocess → display
    the resulting figures.

    Subprocess isolation keeps torch / sb3 imports from accumulating
    across calls (which the Streamlit version cannot do safely)."""
    if file_obj is None:
        return "Upload a CSV (or paste a URL) first.", []
    try:
        # Resolve the input path. Gradio gives us either a tempfile
        # path or a NamedString; both work as strings.
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
    """Run the multi-model compare on an uploaded CSV. Heavy — only
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
    # Gradio 6.0 moved ``theme`` from the Blocks() constructor to
    # ``launch(theme=...)``. We pass theme via launch() below; the
    # Blocks-level params here are the ones that still work.
    with gr.Blocks(
        title="CDPR Simulator",
        analytics_enabled=False,
    ) as demo:
        gr.Markdown(
            f"# CDPR Simulator — Gradio frontend\n"
            f"*Build `{BUILD_ID}` · scientific-core import OK · "
            f"runs locally and on Hugging Face Spaces (16 GB CPU worker).*\n\n"
            "Three tabs cover the full directive:\n\n"
            "1. **Built-in examples** — five one-click demonstrations.\n"
            "2. **Custom simulation** — Phase-1 with full robot / "
            "trajectory / controller choice.\n"
            "3. **Upload CSV** — Phase-2 PINN/MLP/replay/PPO/SAC training "
            "and comparison."
        )

        # ----- Tab 1: examples ---------------------------------------------
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

        # ----- Tab 2: custom Phase-1 simulation ----------------------------
        with gr.Tab("Custom Phase-1 simulation"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Robot & dynamics")
                    robot_name = gr.Dropdown(
                        ["point_mass_3d", "planar_translational",
                         "ipanema_class", "cogiro_class", "dissertation_8cable"],
                        value="dissertation_8cable", label="Robot",
                    )
                    payload_mass = gr.Number(value=0.0, label="Payload mass [kg]",
                                              minimum=0.0, step=0.1)
                    gravity_on = gr.Checkbox(value=True, label="Apply gravity")
                    objective = gr.Radio(
                        ["min_norm", "centered", "preferred"],
                        value="centered", label="Tension distribution objective",
                    )
                    t_min = gr.Number(value=5.0, label="Tension min [N]", minimum=0.0)
                    t_max = gr.Number(value=500.0, label="Tension max [N]", minimum=0.0)

                    gr.Markdown("### Controller (PD)")
                    kp_pos = gr.Number(value=400.0, label="Kp_pos", minimum=0.0)
                    kp_rot = gr.Number(value=100.0, label="Kp_rot", minimum=0.0)

                    gr.Markdown("### Time")
                    duration = gr.Number(value=12.566, label="Duration [s]",
                                         minimum=0.05, maximum=300.0)
                    dt_input = gr.Number(value=1e-3, label="dt [s]",
                                         minimum=1e-4, maximum=5e-2)

                with gr.Column(scale=1):
                    gr.Markdown("### Trajectory")
                    kind = gr.Dropdown(
                        ["hold", "line", "circle", "lissajous"],
                        value="circle", label="Trajectory kind",
                    )
                    gr.Markdown("**Line params**")
                    line_start = gr.Textbox(value="0,0,0.5",
                                            label="start (x,y,z) [m]")
                    line_end = gr.Textbox(value="0.05,0,0.5",
                                          label="end   (x,y,z) [m]")
                    gr.Markdown("**Circle params**")
                    circle_center = gr.Textbox(value="0,0,0.65",
                                                label="center (x,y,z) [m]")
                    circle_radius = gr.Number(value=0.05, label="radius [m]",
                                              minimum=0.0, maximum=2.0)
                    circle_axis = gr.Textbox(value="0,0,1", label="axis (x,y,z)")
                    circle_angle_span = gr.Number(
                        value=float(4 * np.pi), label="angle span [rad]", minimum=0.1,
                    )
                    gr.Markdown("**Lissajous params**")
                    lj_center = gr.Textbox(value="0,0,0.65", label="center [m]")
                    lj_amps = gr.Textbox(value="0.03,0.03,0.0",
                                         label="amplitudes (x,y,z) [m]")
                    lj_freqs = gr.Textbox(value="1.0,2.0,0.0",
                                          label="frequencies (x,y,z) [rad/s]")
                    lj_phases = gr.Textbox(value="0,1.5708,0",
                                            label="phases (x,y,z) [rad]")

            run_btn = gr.Button("Run simulation", variant="primary")
            status_text = gr.Textbox(label="Run summary", lines=8, interactive=False)
            with gr.Row():
                fig_gallery = gr.Gallery(label="Figures", columns=2, height="600px")
            csv_out = gr.File(label="Download timeseries.csv")

            run_btn.click(
                fn=_run_custom_simulation,
                inputs=[
                    robot_name, kind, duration, dt_input,
                    payload_mass, gravity_on, objective,
                    t_min, t_max, kp_pos, kp_rot,
                    circle_center, circle_radius, circle_axis, circle_angle_span,
                    line_start, line_end,
                    lj_center, lj_amps, lj_freqs, lj_phases,
                ],
                outputs=[status_text, fig_gallery, csv_out],
            )

        # ----- Tab 3: upload CSV & analyse ---------------------------------
        with gr.Tab("Upload CSV / Phase-2"):
            gr.Markdown(
                "Drop a CSV that follows the `scripts/run_simulation.py` "
                "layout (or any CSV whose columns can be auto-mapped — see "
                "`docs/csv-schema.md`). The Gradio worker has enough memory "
                "to run all five models; the Streamlit Cloud worker does not "
                "— which is the point of this alternative frontend."
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

        gr.Markdown(
            "---\n*Architectural rationale*: `docs/frontend-architecture.md` · "
            "*Terminal guide*: `docs/terminal-execution.md` · "
            "*Examples*: `docs/examples.md` · "
            "*Sources*: [github.com/Tachia/cdpr_simulator]"
            "(https://github.com/Tachia/cdpr_simulator)"
        )
    return demo


# Module-level ``demo`` — required by Hugging Face Spaces, which does
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
        # Gradio 6.0 expects theme here, not on Blocks().
        # Soft is the closest match to the previous look.
        theme=gr.themes.Soft(),
        # share=True would give a temporary tunnel URL --- uncomment
        # if you need a public link without going through HF Spaces.
        # share=True,
    )
