"""Plotly Dash frontend for CDPR_SIMULATOR — the recommended *local
research dashboard*.

Why Dash (in one sentence):

    For a dissertation-grade scientific workload — long simulations,
    14-figure plot bundles, large CSV uploads, parameter sliders that
    should not re-run the whole page, and PINN / PPO / SAC training
    that should stream progress — Dash's **callback model** (no script
    reruns), **native interactive Plotly figures** (zoom / hover /
    crosshair work for free), and **dcc.Store client-side state**
    eliminate three of the four categories of Streamlit failure we
    have been fighting.

This file is the production research dashboard. Gradio (gradio_app.py)
remains the recommended one-click public demo on Hugging Face Spaces;
the two coexist — they share the same cdpr scientific core and the
same on-disk artifact contract (out/<id>/timeseries.csv + manifest
.json + 14 PNGs).

Local development:

    pip install -e ".[viz,data,api,dash]"
    python dash_app.py
    # http://127.0.0.1:8050

Hugging Face Spaces (Docker SDK, 16 GB worker):

    Make a Space with SDK = Docker, hardware = CPU-Basic, then push
    this file and Dockerfile.dash. See docs/multi-frontend.md.

Render / Fly.io / Railway:

    Dash is a Flask app under the hood. Standard WSGI deployment.
"""

from __future__ import annotations

import base64
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

# Headless matplotlib before any plotting import.
os.environ.setdefault("MPLBACKEND", "Agg")

# Path setup — same pattern as gradio_app.py so a clean clone works.
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_SCRIPTS = _ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from dash import (                                                  # noqa: E402
    Dash,
    Input,
    Output,
    State,
    callback,
    callback_context,
    dcc,
    html,
    no_update,
)
import numpy as np                                                  # noqa: E402
from scipy.spatial.transform import Rotation                        # noqa: E402

from cdpr.core.frames import Pose                                   # noqa: E402
from cdpr.control.pd import PDController                            # noqa: E402
from cdpr.dynamics.rigid_body import PlatformState                  # noqa: E402
from cdpr.dynamics.simulator import simulate                        # noqa: E402
from cdpr.interface.specs import (                                  # noqa: E402
    SimulationRequest,
    TrajectorySpec,
    build_robot,
    build_trajectory,
)

from _csv_io import load_csv_any, split_canonical_blocks            # noqa: E402
from examples import EXAMPLES, list_examples                        # noqa: E402


BUILD_ID = "dash-2026-05-29-a"


# ---------------------------------------------------------------------------
# Plotly figure builders --- the core of "why Dash wins for science".
# Every plot is a native Plotly figure: zoom, pan, hover, crosshair,
# range-slider, png/svg export all work without writing any of that
# code. Matplotlib PNGs in Gradio / Streamlit cannot do this.
# ---------------------------------------------------------------------------

def _plot_position(result, reference) -> dict:
    import plotly.graph_objects as go
    t = np.asarray(result.time)
    p = np.asarray(result.positions)
    ref_p = np.array([reference(tt).position for tt in t]) if reference else None
    fig = go.Figure()
    colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    labels = ("x", "y", "z")
    for k in range(3):
        fig.add_trace(go.Scatter(
            x=t, y=p[:, k],
            name=f"{labels[k]} actual", line=dict(color=colors[k]),
            hovertemplate=f"<b>{labels[k]}</b> %{{y:.4f}} m @ t = %{{x:.3f}} s<extra></extra>",
        ))
        if ref_p is not None:
            fig.add_trace(go.Scatter(
                x=t, y=ref_p[:, k],
                name=f"{labels[k]} reference", line=dict(color=colors[k], dash="dash"),
                opacity=0.55,
            ))
    fig.update_layout(
        title="Platform position (actual vs reference)",
        xaxis_title="time t [s]", yaxis_title="position [m]",
        hovermode="x unified", template="plotly_white",
        height=380, margin=dict(l=60, r=20, t=50, b=50),
    )
    return fig


def _plot_cable_tensions(result, robot) -> dict:
    import plotly.graph_objects as go
    t = np.asarray(result.time)
    T = np.asarray(result.cable_tensions)
    t_min = float(robot.limits.t_min[0])
    t_max = float(robot.limits.t_max[0])
    fig = go.Figure()
    for i in range(T.shape[1]):
        fig.add_trace(go.Scatter(
            x=t, y=T[:, i], name=f"cable {i+1}",
            hovertemplate=f"<b>T{i+1}</b> %{{y:.2f}} N @ t = %{{x:.3f}} s<extra></extra>",
        ))
    fig.add_hline(y=t_min, line=dict(color="grey", dash="dot"),
                  annotation_text=f"T_min = {t_min:.0f} N", annotation_position="bottom right")
    fig.add_hline(y=t_max, line=dict(color="grey", dash="dot"),
                  annotation_text=f"T_max = {t_max:.0f} N", annotation_position="top right")
    fig.update_layout(
        title="Cable tensions vs time (range-slider enabled — try it)",
        xaxis_title="time t [s]", yaxis_title="cable tension [N]",
        hovermode="x unified", template="plotly_white",
        height=380, margin=dict(l=60, r=20, t=50, b=50),
        xaxis=dict(rangeslider=dict(visible=True), type="linear"),
    )
    return fig


def _plot_trajectory_xy(result, reference) -> dict:
    import plotly.graph_objects as go
    p = np.asarray(result.positions)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p[:, 0], y=p[:, 1], name="actual",
        mode="lines", line=dict(width=2, color="#1f77b4"),
    ))
    if reference is not None:
        ref_p = np.array([reference(t).position for t in result.time])
        fig.add_trace(go.Scatter(
            x=ref_p[:, 0], y=ref_p[:, 1], name="reference",
            mode="lines", line=dict(width=2, color="#ff7f0e", dash="dash"),
        ))
    fig.update_layout(
        title="XY trajectory (zoom + pan with mouse)",
        xaxis_title="x [m]", yaxis_title="y [m]",
        template="plotly_white",
        height=420, margin=dict(l=60, r=20, t=50, b=50),
        yaxis=dict(scaleanchor="x", scaleratio=1),               # equal aspect
    )
    return fig


def _plot_tracking_error(result, reference) -> dict:
    import plotly.graph_objects as go
    t = np.asarray(result.time)
    ref_p = np.array([reference(tt).position for tt in t]) if reference else None
    if ref_p is None:
        fig = {"data": [], "layout": {"title": "no reference --- nothing to track"}}
        return fig
    err = np.linalg.norm(np.asarray(result.positions) - ref_p, axis=1)
    rms = np.sqrt(np.cumsum(err ** 2) / np.arange(1, len(err) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=err * 1e3, name="instantaneous |error|", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=t, y=rms * 1e3, name="cumulative RMS", line=dict(color="#9467bd")))
    fig.update_layout(
        title="Tracking error evolution",
        xaxis_title="time t [s]", yaxis_title="error [mm]",
        hovermode="x unified", template="plotly_white",
        height=320, margin=dict(l=60, r=20, t=50, b=50),
        yaxis=dict(type="log"),
    )
    return fig


def _plot_3d_scene(result, robot) -> dict:
    """A real 3D Plotly scene of the platform path and cable anchors —
    the kind of thing matplotlib in Gradio / Streamlit can't make
    interactive."""
    import plotly.graph_objects as go
    p = np.asarray(result.positions)
    anchors = np.asarray(robot.anchors)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=p[:, 0], y=p[:, 1], z=p[:, 2],
        mode="lines",
        line=dict(color="#1f77b4", width=4),
        name="EE path",
    ))
    fig.add_trace(go.Scatter3d(
        x=anchors[:, 0], y=anchors[:, 1], z=anchors[:, 2],
        mode="markers+text",
        marker=dict(size=6, color="black"),
        text=[f"A{i+1}" for i in range(len(anchors))],
        textposition="top center",
        name="anchors",
    ))
    # Final-step cable segments to give spatial context.
    last = p[-1]
    for i, a in enumerate(anchors):
        fig.add_trace(go.Scatter3d(
            x=[a[0], last[0]], y=[a[1], last[1]], z=[a[2], last[2]],
            mode="lines", line=dict(color="rgba(100,100,100,0.4)", width=2),
            showlegend=False, hoverinfo="skip",
        ))
    fig.update_layout(
        title="3D scene — drag to rotate, scroll to zoom",
        scene=dict(
            xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]",
            aspectmode="data",
        ),
        template="plotly_white",
        height=500, margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# In-process simulation runner (Phase-1)
# ---------------------------------------------------------------------------

def _do_simulation(
    robot_name: str, kind: str, duration: float, dt: float,
    payload_mass: float, gravity_on: bool, objective: str,
    t_min: float, t_max: float, kp_pos: float, kp_rot: float,
    circle_center: str, circle_radius: float, circle_angle_span: float,
) -> dict:
    def _xyz(s: str, default: list[float]) -> list[float]:
        try:
            parts = [p.strip() for p in str(s).split(",") if p.strip()]
            return [float(p) for p in parts] if len(parts) == 3 else default
        except Exception:
            return default
    if kind == "circle":
        params: dict = {
            "center": _xyz(circle_center, [0.0, 0.0, 0.65]),
            "radius": float(circle_radius),
            "axis": [0.0, 0.0, 1.0],
            "angle_span": float(circle_angle_span),
        }
    elif kind == "line":
        params = {"start": [0, 0, 0.5], "end": [0.05, 0, 0.5]}
    elif kind == "lissajous":
        params = {"center": [0, 0, 0.65], "amplitudes": [0.03, 0.03, 0],
                  "frequencies": [1.0, 2.0, 0], "phases": [0, np.pi/2, 0]}
    else:
        params = {}
    request = SimulationRequest(
        robot=robot_name,
        payload_mass=float(payload_mass),
        gravity=(0.0, 0.0, -9.81 if gravity_on else 0.0),
        tension_objective=objective,
        duration=float(duration),
        dt=float(dt),
        trajectory=TrajectorySpec(kind=kind, duration=float(duration), params=params),
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
    result = simulate(
        robot=robot, state0=state0,
        duration=request.duration, dt=request.dt,
        reference=reference, controller=controller,
        tension_objective=request.tension_objective,
        gravity=request.gravity,
    )
    # Save artifacts to disk.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = _ROOT / "out" / f"dash-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from examples import _save_csv
        csv_path = _save_csv(out_dir, result, reference=reference,
                              t_min=float(robot.limits.t_min[0]),
                              t_max=float(robot.limits.t_max[0]))
    except Exception as exc:
        csv_path = None
        print(f"warning: could not save CSV: {exc}", file=sys.stderr)
    return {
        "robot": robot, "reference": reference, "result": result,
        "out_dir": str(out_dir), "csv_path": str(csv_path) if csv_path else None,
    }


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

def build_app() -> Dash:
    app = Dash(
        __name__,
        title="CDPR Simulator — Research Dashboard",
        update_title=None,
        suppress_callback_exceptions=True,
    )

    examples_catalog = list_examples()
    example_options = [
        {"label": f"[{e['name']}] {e['title']} (phase {e['phase']})",
         "value": e["name"]}
        for e in examples_catalog
    ]

    app.layout = html.Div([
        # Top banner --- visible build id so users can confirm freshness.
        html.Div([
            html.H2("CDPR Simulator — Research Dashboard"),
            html.Span(
                f"Build {BUILD_ID} · Dash + Plotly + cdpr core · "
                "interactive plots: zoom, pan, hover, range-slider, png/svg export",
                style={"color": "#555", "fontSize": "0.9em"},
            ),
        ], style={"padding": "12px 24px", "borderBottom": "1px solid #ddd"}),

        # Two-column layout: controls on the left, plot tabs on the right.
        html.Div([
            html.Div([
                html.H4("Phase 1 — simulate"),

                html.Label("Robot"),
                dcc.Dropdown(
                    id="robot",
                    options=[{"label": r, "value": r} for r in
                             ("point_mass_3d", "planar_translational",
                              "ipanema_class", "cogiro_class", "dissertation_8cable")],
                    value="dissertation_8cable",
                ),

                html.Label("Trajectory kind"),
                dcc.Dropdown(
                    id="kind",
                    options=[{"label": k, "value": k} for k in
                             ("hold", "line", "circle", "lissajous")],
                    value="circle",
                ),

                html.Label("Duration [s]"),
                dcc.Input(id="duration", type="number", value=12.566,
                          step=0.5, min=0.05, max=300.0, style={"width": "100%"}),

                html.Label("dt [s]"),
                dcc.Input(id="dt", type="number", value=1e-3,
                          step=1e-4, min=1e-4, max=5e-2, style={"width": "100%"}),

                html.Label("Circle: center (x,y,z)"),
                dcc.Input(id="circle_center", type="text", value="0,0,0.65",
                          style={"width": "100%"}),
                html.Label("Circle: radius [m]"),
                dcc.Input(id="circle_radius", type="number", value=0.05,
                          step=0.005, min=0.0, max=2.0, style={"width": "100%"}),
                html.Label("Circle: angle span [rad]"),
                dcc.Input(id="circle_angle_span", type="number",
                          value=float(4 * np.pi), step=0.1, min=0.1,
                          style={"width": "100%"}),

                html.Label("Payload mass [kg]"),
                dcc.Input(id="payload_mass", type="number", value=0.0,
                          step=0.1, min=0.0, style={"width": "100%"}),

                html.Label("Tension min [N]"),
                dcc.Input(id="t_min", type="number", value=5.0,
                          step=1.0, min=0.0, style={"width": "100%"}),
                html.Label("Tension max [N]"),
                dcc.Input(id="t_max", type="number", value=500.0,
                          step=10.0, min=0.0, style={"width": "100%"}),

                html.Label("Controller Kp_pos"),
                dcc.Input(id="kp_pos", type="number", value=400.0,
                          step=10.0, min=0.0, style={"width": "100%"}),
                html.Label("Controller Kp_rot"),
                dcc.Input(id="kp_rot", type="number", value=100.0,
                          step=10.0, min=0.0, style={"width": "100%"}),

                html.Label("Gravity"),
                dcc.Checklist(
                    id="gravity_on",
                    options=[{"label": " Apply gravity", "value": "on"}],
                    value=["on"],
                ),

                html.Label("Tension objective"),
                dcc.RadioItems(
                    id="objective",
                    options=[{"label": v, "value": v} for v in
                             ("min_norm", "centered", "preferred")],
                    value="centered",
                ),

                html.Br(),
                html.Button("Run simulation", id="btn_run", n_clicks=0,
                            style={"width": "100%", "padding": "10px",
                                   "backgroundColor": "#1f77b4",
                                   "color": "white", "border": "none",
                                   "borderRadius": "4px",
                                   "fontSize": "1em", "cursor": "pointer"}),

                html.Div(id="run_summary",
                          style={"marginTop": "12px", "fontSize": "0.9em",
                                 "color": "#222"}),

                html.Hr(),
                html.H4("Built-in examples"),
                dcc.Dropdown(
                    id="example_pick",
                    options=example_options,
                    value=examples_catalog[0]["name"],
                ),
                html.Div(id="example_description",
                          style={"fontSize": "0.85em", "color": "#444",
                                 "margin": "8px 0"}),
                html.Button("Run example (subprocess)", id="btn_example", n_clicks=0,
                            style={"width": "100%", "padding": "8px",
                                   "backgroundColor": "#2ca02c", "color": "white",
                                   "border": "none", "borderRadius": "4px",
                                   "cursor": "pointer"}),
                html.Div(id="example_log",
                          style={"marginTop": "8px", "fontSize": "0.85em",
                                 "fontFamily": "monospace", "color": "#222",
                                 "maxHeight": "180px", "overflowY": "auto"}),

                html.Hr(),
                html.H4("Phase 2 — upload CSV"),
                dcc.Upload(
                    id="upload_csv",
                    children=html.Div(["Drop a timeseries.csv here or ",
                                       html.A("click to browse")]),
                    style={"border": "1px dashed #aaa", "padding": "12px",
                           "borderRadius": "4px", "textAlign": "center"},
                    multiple=False,
                ),
                html.Label("Phase-2 model"),
                dcc.RadioItems(
                    id="phase2_model",
                    options=[{"label": m, "value": m}
                             for m in ("replay", "mlp", "pinn", "ppo", "sac")],
                    value="pinn",
                ),
                html.Label("Epochs (mlp / pinn)"),
                dcc.Input(id="phase2_epochs", type="number", value=80,
                          min=1, max=10000, style={"width": "100%"}),
                html.Button("Run Phase-2", id="btn_phase2", n_clicks=0,
                            style={"width": "100%", "padding": "8px",
                                   "marginTop": "8px",
                                   "backgroundColor": "#9467bd", "color": "white",
                                   "border": "none", "borderRadius": "4px",
                                   "cursor": "pointer"}),
                html.Div(id="phase2_log",
                          style={"marginTop": "8px", "fontSize": "0.85em",
                                 "fontFamily": "monospace", "color": "#222",
                                 "maxHeight": "180px", "overflowY": "auto"}),
            ], style={"width": "320px", "padding": "16px",
                      "borderRight": "1px solid #ddd",
                      "overflowY": "auto", "height": "calc(100vh - 60px)"}),

            html.Div([
                dcc.Tabs(id="plot_tabs", value="tab_position", children=[
                    dcc.Tab(label="Position",       value="tab_position"),
                    dcc.Tab(label="Cable tensions", value="tab_tensions"),
                    dcc.Tab(label="XY trajectory",  value="tab_xy"),
                    dcc.Tab(label="Tracking error", value="tab_error"),
                    dcc.Tab(label="3D scene",       value="tab_3d"),
                ]),
                html.Div(id="plot_container", style={"padding": "12px"}),
            ], style={"flex": "1", "overflowY": "auto",
                      "height": "calc(100vh - 60px)"}),
        ], style={"display": "flex", "flexDirection": "row"}),

        # Client-side state ---  the "session_state reliability problem"
        # disappears here because the entire run sits as a JSON blob in
        # the browser's memory, not on the server.
        dcc.Store(id="store_result_meta", data={}),
    ], style={"fontFamily": "system-ui, -apple-system, sans-serif",
              "margin": 0, "padding": 0})

    # -- Example description --------------------------------------------------
    @app.callback(
        Output("example_description", "children"),
        Input("example_pick", "value"),
    )
    def _example_desc(name):
        if not name:
            return ""
        e = EXAMPLES[name]
        deps = (f" · depends on '{e['depends_on']}' (auto-runs if missing)"
                if e.get("depends_on") else "")
        return f"{e['description']}{deps}"

    # -- Built-in example runner ---------------------------------------------
    @app.callback(
        Output("example_log", "children"),
        Output("store_result_meta", "data", allow_duplicate=True),
        Input("btn_example", "n_clicks"),
        State("example_pick", "value"),
        prevent_initial_call=True,
    )
    def _run_example(n_clicks, name):
        if not n_clicks or not name:
            return no_update, no_update
        cmd = [sys.executable, str(_SCRIPTS / "run_example.py"), "--name", name]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
        dt = time.perf_counter() - t0
        if proc.returncode != 0:
            return f"[FAILED rc={proc.returncode}] {dt:.1f}s\n{(proc.stderr or '')[-1500:]}", no_update
        out_dir = _ROOT / "out" / EXAMPLES[name]["out_dir"]
        log = f"[done] '{name}' in {dt:.1f}s → {out_dir}"
        return log, {"source": "example", "name": name, "out_dir": str(out_dir)}

    # -- Phase-1 inline runner -----------------------------------------------
    @app.callback(
        Output("run_summary", "children"),
        Output("store_result_meta", "data", allow_duplicate=True),
        Output("plot_container", "children", allow_duplicate=True),
        Input("btn_run", "n_clicks"),
        State("robot", "value"),
        State("kind", "value"),
        State("duration", "value"),
        State("dt", "value"),
        State("payload_mass", "value"),
        State("gravity_on", "value"),
        State("objective", "value"),
        State("t_min", "value"),
        State("t_max", "value"),
        State("kp_pos", "value"),
        State("kp_rot", "value"),
        State("circle_center", "value"),
        State("circle_radius", "value"),
        State("circle_angle_span", "value"),
        State("plot_tabs", "value"),
        prevent_initial_call=True,
    )
    def _run_inline(n_clicks, robot, kind, duration, dt, payload, gravity_on,
                     objective, t_min, t_max, kp_pos, kp_rot,
                     cc, cr, ca, current_tab):
        if not n_clicks:
            return no_update, no_update, no_update
        try:
            t0 = time.perf_counter()
            run = _do_simulation(
                robot, kind, duration, dt, payload,
                "on" in (gravity_on or []), objective,
                t_min, t_max, kp_pos, kp_rot, cc, cr, ca,
            )
            elapsed = time.perf_counter() - t0
            result = run["result"]
            tens = np.asarray(result.cable_tensions)
            ref_pos = np.array([run["reference"](t).position for t in result.time])
            err = np.linalg.norm(np.asarray(result.positions) - ref_pos, axis=1)
            summary = (
                f"✓ {len(result.time)} samples in {elapsed:.1f}s · "
                f"tensions [{float(tens.min()):.1f}, {float(tens.max()):.1f}] N · "
                f"tracking RMS {float(np.sqrt(np.mean(err**2)))*1e3:.2f} mm · "
                f"out → {run['out_dir']}"
            )
            CURRENT_RESULT["last"] = run
            return summary, {"source": "inline", "out_dir": run["out_dir"]}, \
                   _render_tab(current_tab, run)
        except Exception as exc:
            return (f"✗ ERROR: {type(exc).__name__}: {exc}", no_update,
                    html.Pre(traceback.format_exc(), style={"color": "#a00"}))

    # -- Plot tab switcher (the killer feature: switching tabs does not
    # -- re-run the simulation; only the affected plot is drawn) -----------
    @app.callback(
        Output("plot_container", "children"),
        Input("plot_tabs", "value"),
        State("store_result_meta", "data"),
    )
    def _switch_tab(tab, meta):
        run = CURRENT_RESULT.get("last")
        if run is None:
            return html.Div("Run a simulation or an example to see plots.",
                            style={"padding": "32px", "color": "#888",
                                   "textAlign": "center"})
        return _render_tab(tab, run)

    # -- Phase 2 inline runner (subprocess) ---------------------------------
    @app.callback(
        Output("phase2_log", "children"),
        Input("btn_phase2", "n_clicks"),
        State("upload_csv", "contents"),
        State("upload_csv", "filename"),
        State("phase2_model", "value"),
        State("phase2_epochs", "value"),
        prevent_initial_call=True,
    )
    def _run_phase2(n_clicks, contents, filename, model, epochs):
        if not n_clicks or not contents:
            return "Upload a CSV first."
        _content_type, content_b64 = contents.split(",", 1)
        data_bytes = base64.b64decode(content_b64)
        tmp_path = _ROOT / "out" / f"_dash_upload_{int(time.time())}.csv"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(data_bytes)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        out_dir = _ROOT / "out" / f"dash-{model}-{stamp}"
        cmd = [
            sys.executable, str(_SCRIPTS / "train_from_csv.py"),
            "--input", str(tmp_path),
            "--model", model,
            "--epochs", str(int(epochs)),
            "--out", str(out_dir),
        ]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
        dt = time.perf_counter() - t0
        log = f"[{model}] exit={proc.returncode} in {dt:.1f}s\n"
        log += (proc.stdout or "")[-1500:]
        if proc.returncode != 0:
            log += "\n--- stderr ---\n" + (proc.stderr or "")[-1500:]
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            log += "\n--- metrics.json ---\n" + metrics_path.read_text(encoding="utf-8")
        return log

    return app


# Module-level cache for the last simulation result. Dash's normal
# pattern would put it in dcc.Store, but our SimulationResult is a
# rich dataclass with numpy arrays (and a Trajectory reference
# callable), neither of which JSON-serialises cleanly. The cache
# is per-process and per-worker, which is exactly what we want.
CURRENT_RESULT: dict = {}


def _render_tab(tab: str, run: dict) -> Any:
    """Build the Plotly figure for the active tab. Switching tabs only
    redraws this one container --- the simulation is NOT re-run, which
    is the central reason Dash wins for this workload."""
    result = run["result"]
    robot = run["robot"]
    reference = run["reference"]
    if tab == "tab_position":
        fig = _plot_position(result, reference)
    elif tab == "tab_tensions":
        fig = _plot_cable_tensions(result, robot)
    elif tab == "tab_xy":
        fig = _plot_trajectory_xy(result, reference)
    elif tab == "tab_error":
        fig = _plot_tracking_error(result, reference)
    elif tab == "tab_3d":
        fig = _plot_3d_scene(result, robot)
    else:
        fig = {"data": [], "layout": {"title": "Unknown tab"}}
    return dcc.Graph(figure=fig, config={"toImageButtonOptions":
                                          {"format": "png", "scale": 2}})


if __name__ == "__main__":
    app = build_app()
    port = int(os.environ.get("DASH_PORT", "8050"))
    host = os.environ.get("DASH_HOST", "127.0.0.1")
    print(f"[dash] CDPR Simulator research dashboard on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
