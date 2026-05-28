r"""Streamlit research console.

Launch with::

    pip install 'cdpr[gui,viz]'
    streamlit run streamlit_app.py

Free-tier-aware design
----------------------

Streamlit Community Cloud runs each app in a ~1 GB sandbox. The previous
revision rendered every plot tab on every script rerun --- six
matplotlib figures including a 3D scene --- which OOM-killed the worker
right after the simulate() call finished, producing a blank page with
no Python-level traceback.

Two changes prevent that:

1. **Sidebar lives inside an :func:`st.form`.** Streamlit only reruns
   the script when the user explicitly clicks *Apply parameters* (the
   form's submit button) or *Run simulation*. Editing a slider mid-
   form no longer triggers a rerun.

2. **Only the active result tab renders a figure.** Streamlit's
   :func:`st.tabs` runs the body of *every* tab on every rerun, but we
   gate each tab's plot behind a per-tab "Render" button + a session-
   state cache. The 3D scene tab in particular is hidden behind an
   explicit opt-in because it is the heaviest figure in the framework.

The console also forces ``MPLBACKEND=Agg`` upstream in
``streamlit_app.py`` (must be set before any matplotlib import).

Every exception inside :func:`render` and around each plot is caught
and surfaced via :func:`st.exception`; previous versions silently
swallowed errors and showed a blank page.
"""

from __future__ import annotations

import os
import traceback

try:
    import streamlit as st
except ImportError as exc:
    raise ImportError(
        "The Streamlit console needs Streamlit. Install with:  pip install 'cdpr[gui]'"
    ) from exc

# matplotlib backend must be selected before any plotting import. The
# top-level streamlit_app.py sets MPLBACKEND=Agg at process start; here
# we also call matplotlib.use() defensively in case the GUI is invoked
# through a different entry-point.
import matplotlib                                              # noqa: E402
matplotlib.use("Agg", force=True)

import io                                                      # noqa: E402
import sys                                                     # noqa: E402

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402
from scipy.spatial.transform import Rotation                   # noqa: E402

from cdpr.core.frames import Pose                              # noqa: E402
from cdpr.dynamics.rigid_body import PlatformState             # noqa: E402
from cdpr.dynamics.simulator import simulate                   # noqa: E402
from cdpr.interface.specs import (                             # noqa: E402
    SimulationRequest,
    TrajectorySpec,
    build_robot,
    build_trajectory,
)


# Frugal mode (free Streamlit Cloud tier) trims the default workload.
FRUGAL = bool(int(os.environ.get("CDPR_GUI_FRUGAL", "0")))


def _log(msg: str) -> None:
    """Write to stderr so it appears in the Streamlit Cloud manage-app log
    even when the in-app UI is dead."""
    print(f"[cdpr-gui] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def render() -> None:
    if not st.session_state.get("_page_config_set"):
        st.set_page_config(page_title="CDPR research console", layout="wide")
        st.session_state["_page_config_set"] = True

    try:
        st.title("CDPR research console")
        st.caption(
            "Configure parameters in the sidebar form, click **Apply parameters**, "
            "then click **Run simulation**. Plots render on demand to keep memory low."
        )

        request = _request_form()
        _action_bar(request)
        _results_panel()
        _upload_panel()

    except Exception as exc:
        _log(f"render() crashed: {type(exc).__name__}: {exc}")
        st.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
        st.exception(exc)
        st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Sidebar form
# ---------------------------------------------------------------------------

def _request_form() -> SimulationRequest:
    """All parameter widgets live inside a single form so Streamlit
    only re-runs the script on Apply."""
    default_duration = 0.5 if FRUGAL else 1.5
    default_dt = 5e-3 if FRUGAL else 2e-3

    with st.sidebar:
        with st.form("params_form", clear_on_submit=False):
            st.header("Experiment")
            robot_name = st.selectbox(
                "Robot",
                ["ipanema_class", "cogiro_class", "point_mass_3d", "planar_translational"],
                index=0,
                key="robot_name",
            )
            payload = st.number_input(
                "Payload mass [kg]", min_value=0.0, value=0.0, step=0.5, key="payload",
            )

            st.divider()
            st.header("Trajectory")
            kind = st.selectbox(
                "Kind", ["hold", "line", "circle", "lissajous"], index=2, key="kind",
            )
            duration = st.number_input(
                "Duration [s]", min_value=0.05, max_value=5.0,
                value=default_duration, step=0.05, key="duration",
            )
            dt = st.number_input(
                "Time step dt [s]", min_value=1e-4, max_value=5e-2,
                value=default_dt, step=1e-3, format="%.4f", key="dt",
            )

            st.subheader("Line parameters")
            line_start = st.text_input("start (x,y,z) [m]", "0,0,0", key="line_start")
            line_end = st.text_input("end   (x,y,z) [m]", "0.3,0,0", key="line_end")

            st.subheader("Circle parameters")
            circle_radius = st.slider(
                "radius [m]", 0.05, 1.0, 0.3, step=0.05, key="circle_radius",
            )
            circle_axis = st.selectbox(
                "axis", ["z", "y", "x"], index=0, key="circle_axis",
            )

            st.subheader("Lissajous amplitudes [m]")
            liss_ax = st.slider("Ax", 0.0, 1.0, 0.3, step=0.05, key="liss_ax")
            liss_ay = st.slider("Ay", 0.0, 1.0, 0.2, step=0.05, key="liss_ay")
            liss_az = st.slider("Az", 0.0, 1.0, 0.0, step=0.05, key="liss_az")
            st.subheader("Lissajous frequencies [-]")
            liss_fx = st.slider("fx", 1.0, 5.0, 1.0, step=0.5, key="liss_fx")
            liss_fy = st.slider("fy", 1.0, 5.0, 2.0, step=0.5, key="liss_fy")
            liss_fz = st.slider("fz", 0.0, 5.0, 0.0, step=0.5, key="liss_fz")

            st.divider()
            st.header("Solver")
            integrator = st.radio(
                "Integrator", ["rk4", "semi_implicit"], horizontal=True, key="integrator",
            )
            objective = st.radio(
                "Tension objective", ["centered", "min_norm", "preferred"],
                horizontal=True, key="objective",
            )

            applied = st.form_submit_button(
                "Apply parameters", type="secondary", use_container_width=True,
            )

    if kind == "line":
        params: dict = {
            "start": _parse_xyz(line_start, default=[0.0, 0.0, 0.0]),
            "end": _parse_xyz(line_end, default=[0.3, 0.0, 0.0]),
        }
    elif kind == "circle":
        axis_vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[circle_axis]
        params = {
            "center": [0.0, 0.0, 0.0],
            "radius": float(circle_radius),
            "axis": axis_vec,
            "angle_span": 2 * np.pi,
        }
    elif kind == "lissajous":
        params = {
            "center": [0.0, 0.0, 0.0],
            "amplitudes": [float(liss_ax), float(liss_ay), float(liss_az)],
            "frequencies": [float(liss_fx), float(liss_fy), float(liss_fz)],
            "phases": [0.0, np.pi / 2, 0.0],
        }
    else:  # hold
        params = {}

    if applied:
        st.toast("Parameters applied.")

    return SimulationRequest(
        robot=robot_name,                                                       # type: ignore[arg-type]
        trajectory=TrajectorySpec(kind=kind, duration=duration, params=params),  # type: ignore[arg-type]
        duration=duration,
        dt=dt,
        integrator=integrator,                                                  # type: ignore[arg-type]
        tension_objective=objective,                                            # type: ignore[arg-type]
        payload_mass=payload,
    )


def _parse_xyz(text: str, *, default: list[float]) -> list[float]:
    try:
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        if len(parts) != 3:
            return list(default)
        return [float(p) for p in parts]
    except (ValueError, TypeError):
        return list(default)


# ---------------------------------------------------------------------------
# Action bar
# ---------------------------------------------------------------------------

def _action_bar(request: SimulationRequest) -> None:
    cols = st.columns([1, 1, 6])
    if cols[0].button("Run simulation", type="primary", key="btn_run"):
        try:
            _log(
                f"Run simulation: robot={request.robot} kind={request.trajectory.kind} "
                f"duration={request.duration} dt={request.dt}"
            )
            with st.spinner("Integrating..."):
                robot = build_robot(request.robot, payload_mass=request.payload_mass)
                ref = build_trajectory(request.trajectory)
                initial_pose = Pose(position=np.zeros(3), rotation=Rotation.identity())
                result = simulate(
                    robot=robot,
                    state0=PlatformState.at_rest(initial_pose),
                    duration=request.duration,
                    dt=request.dt,
                    reference_pose=ref,
                    integrator=request.integrator,
                    tension_objective=request.tension_objective,
                    gravity=request.gravity,
                )
            _log(f"Simulation finished: {len(result.time)} samples")
            st.session_state["last_result"] = result
            st.session_state["last_robot"] = robot
            st.session_state["last_reference"] = ref
            # Invalidate cached figures from a previous run.
            st.session_state.pop("_figure_cache", None)
            st.toast(f"Simulation finished ({len(result.time)} samples).")
        except Exception as exc:
            _log(f"Simulation crashed: {type(exc).__name__}: {exc}")
            st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
    if cols[1].button("Clear cached run", key="btn_clear"):
        for key in ("last_result", "last_robot", "last_reference", "_figure_cache"):
            st.session_state.pop(key, None)
        st.toast("Cleared.")


# ---------------------------------------------------------------------------
# Results panel — lazy per-tab rendering
# ---------------------------------------------------------------------------

def _results_panel() -> None:
    result = st.session_state.get("last_result")
    robot = st.session_state.get("last_robot")
    reference = st.session_state.get("last_reference")
    if result is None or robot is None:
        st.info(
            "Configure parameters in the sidebar, click **Apply parameters**, "
            "then click **Run simulation**."
        )
        return

    # Top-line summary --- cheap, render unconditionally.
    summary_cols = st.columns(4)
    summary_cols[0].metric("samples", len(result.time))
    summary_cols[1].metric(
        "max |p|", f"{np.linalg.norm(result.positions, axis=1).max():.3f} m",
    )
    summary_cols[2].metric(
        "tension range",
        f"{result.cable_tensions.min():.0f} – {result.cable_tensions.max():.0f} N",
    )
    summary_cols[3].metric("infeasible steps", len(result.infeasible_steps))

    # Per-tab lazy plot rendering. Each tab has a Render button + a cached
    # figure in session_state, so:
    #   - on the first run, nothing renders until the user asks for a plot;
    #   - once rendered, the figure is reused across reruns (no recompute);
    #   - the 3D scene tab is opt-in only because it is the heaviest.
    tab_labels = [
        "Position", "Cable tensions", "Cable lengths",
        "Tracking error", "Condition number", "3D scene (heavy)",
    ]
    tabs = st.tabs(tab_labels)

    _lazy_tab(tabs[0], "position",        lambda: _plot_position(result))
    _lazy_tab(tabs[1], "cable_tensions",  lambda: _plot_cable_tensions(result, robot))
    _lazy_tab(tabs[2], "cable_lengths",   lambda: _plot_cable_lengths(result))
    _lazy_tab(tabs[3], "tracking_error",  lambda: _plot_tracking_error(result, reference))
    _lazy_tab(tabs[4], "condition_number", lambda: _plot_condition_number(result, robot))
    _lazy_tab(tabs[5], "scene_3d",        lambda: _plot_scene_3d(result, robot), warn_heavy=True)


def _lazy_tab(tab, key: str, render_fn, *, warn_heavy: bool = False) -> None:
    cache = st.session_state.setdefault("_figure_cache", {})
    with tab:
        if warn_heavy:
            st.caption(
                "The 3D scene is the heaviest figure and can OOM the free Streamlit "
                "Cloud worker. Click below to render."
            )
        if key in cache:
            st.pyplot(cache[key], clear_figure=False)
            if st.button(f"Re-render {key}", key=f"rerender_{key}"):
                cache.pop(key, None)
                st.rerun()
        else:
            if st.button(f"Render {key}", key=f"render_{key}", type="primary"):
                try:
                    _log(f"Rendering {key}...")
                    fig = render_fn()
                    cache[key] = fig
                    _log(f"Rendered {key}.")
                    st.rerun()
                except Exception as exc:
                    _log(f"Render {key} failed: {type(exc).__name__}: {exc}")
                    st.error(f"Could not render {key}: {type(exc).__name__}: {exc}")
                    st.exception(exc)


# ---------------------------------------------------------------------------
# Plot factories — imported lazily so the module load cost stays small
# ---------------------------------------------------------------------------

def _plot_position(result):
    import matplotlib.pyplot as plt
    from cdpr.viz import plots2d
    from cdpr.viz.style import apply_paper_style
    apply_paper_style()
    fig = plots2d.plot_position(result)
    return fig


def _plot_cable_tensions(result, robot):
    from cdpr.viz import plots2d
    from cdpr.viz.style import apply_paper_style
    apply_paper_style()
    return plots2d.plot_cable_tensions(result, robot=robot)


def _plot_cable_lengths(result):
    from cdpr.viz import plots2d
    from cdpr.viz.style import apply_paper_style
    apply_paper_style()
    return plots2d.plot_cable_lengths(result)


def _plot_tracking_error(result, reference):
    if reference is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no reference trajectory configured",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    from cdpr.viz import plots2d
    from cdpr.viz.style import apply_paper_style
    apply_paper_style()
    return plots2d.plot_tracking_error(result, reference)


def _plot_condition_number(result, robot):
    from cdpr.viz import plots2d
    from cdpr.viz.style import apply_paper_style
    apply_paper_style()
    return plots2d.plot_condition_number(result, robot)


def _plot_scene_3d(result, robot):
    from cdpr.viz.scene import SceneOptions, render_scene
    from cdpr.viz.style import apply_paper_style
    apply_paper_style()
    snapshot_pose = Pose(
        position=result.positions[-1],
        rotation=Rotation.from_quat(result.quaternions_xyzw[-1]),
    )
    # Keep the trajectory subsampled to control memory footprint.
    traj = result.positions
    if len(traj) > 200:
        traj = traj[:: max(1, len(traj) // 200)]
    return render_scene(
        robot, snapshot_pose,
        options=SceneOptions(tension_heatmap=True),
        tensions=result.cable_tensions[-1],
        trajectory_positions=traj,
    )


# ---------------------------------------------------------------------------
# Upload panel
# ---------------------------------------------------------------------------

def _upload_panel() -> None:
    st.divider()
    st.subheader("Experiment upload")
    uploaded = st.file_uploader(
        "Drop a CSV / XLSX experimental log",
        type=["csv", "xlsx", "xls"],
        help="The file is parsed with pandas and the head is shown below.",
        key="upload_file",
    )
    if uploaded is None:
        return
    try:
        raw = uploaded.read()
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
        st.write(f"Loaded **{uploaded.name}** with shape {df.shape}.")
        st.dataframe(df.head(50), height=300)
    except Exception as exc:
        _log(f"Upload parse failed: {type(exc).__name__}: {exc}")
        st.error(f"Failed to parse uploaded file: {type(exc).__name__}: {exc}")
        st.exception(exc)


# Streamlit executes the module body when launched via `streamlit run`.
render()
