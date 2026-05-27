r"""Streamlit research console.

Launch with::

    pip install 'cdpr[gui,viz]'
    streamlit run streamlit_app.py
    # or directly:
    streamlit run src/cdpr/interface/gui.py

Layout: the sidebar holds a single :class:`st.form` containing every
parameter widget. Streamlit only re-runs the script when the user hits
*Apply parameters*, which eliminates the "page goes blank after touching
a slider" failure mode that intermediate reruns introduced (slider
widgets coming in and out of existence as the trajectory kind changed
could leave the script in an inconsistent state on Streamlit Cloud's
free runtime). All widgets carry explicit ``key=`` so the form state
persists cleanly across reruns.

Every exception inside :func:`render` is caught and re-displayed
through :func:`streamlit.exception`; previous versions silently
swallowed errors and left a blank page.
"""

from __future__ import annotations

try:
    import streamlit as st
except ImportError as exc:
    raise ImportError(
        "The Streamlit console needs Streamlit. Install with:  pip install 'cdpr[gui]'"
    ) from exc

import io
import traceback

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.interface.specs import (
    SimulationRequest,
    TrajectorySpec,
    build_robot,
    build_trajectory,
)
from cdpr.viz import plots2d
from cdpr.viz.scene import SceneOptions, render_scene
from cdpr.viz.style import apply_dissertation_style


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------

def render() -> None:
    # set_page_config must be the first Streamlit call --- guard against
    # repeated calls (Streamlit raises on duplicate config in some versions).
    if not st.session_state.get("_page_config_set"):
        st.set_page_config(page_title="CDPR research console", layout="wide")
        st.session_state["_page_config_set"] = True

    try:
        apply_dissertation_style()

        st.title("CDPR research console")
        st.caption(
            "Front-end for the cdpr scientific core. Configure parameters in "
            "the sidebar form, click Apply, then Run simulation."
        )

        request = _request_form()
        _action_bar(request)
        _results_panel()
        _upload_panel()

    except Exception as exc:
        # Surface the traceback in-app instead of a silent blank page.
        st.error(f"Unhandled exception in console: {type(exc).__name__}: {exc}")
        st.exception(exc)
        st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Sidebar form
# ---------------------------------------------------------------------------

def _request_form() -> SimulationRequest:
    """All parameter widgets live inside a single form so Streamlit
    only re-runs the script on Apply --- not on every keystroke."""
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
                "Duration [s]", min_value=0.1, value=1.5, step=0.1, key="duration",
            )
            dt = st.number_input(
                "Time step dt [s]", min_value=1e-4, max_value=1e-1,
                value=2e-3, step=1e-3, format="%.4f", key="dt",
            )

            # Kind-specific parameters --- ALL widgets are always rendered,
            # we just read the ones we need based on `kind`. This avoids
            # the "widget appears/disappears between reruns" footgun.
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

            # Submit button. Until this is pressed, the form's widgets retain
            # their values without re-running the rest of the script.
            applied = st.form_submit_button(
                "Apply parameters", type="secondary", use_container_width=True,
            )

    # Build params dict based on the active kind.
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
    """Best-effort '0.1, 0.2, 0.3' → [0.1, 0.2, 0.3]; falls back to default
    on any parse failure (empty field while user is editing, etc.)."""
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
            st.session_state["last_result"] = result
            st.session_state["last_robot"] = robot
            st.session_state["last_reference"] = ref
            st.toast("Simulation finished.")
        except Exception as exc:
            st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
    if cols[1].button("Clear cached run", key="btn_clear"):
        for key in ("last_result", "last_robot", "last_reference"):
            st.session_state.pop(key, None)
        st.toast("Cleared.")


# ---------------------------------------------------------------------------
# Results panel
# ---------------------------------------------------------------------------

def _results_panel() -> None:
    result = st.session_state.get("last_result")
    robot = st.session_state.get("last_robot")
    reference = st.session_state.get("last_reference")
    if result is None or robot is None:
        st.info("Configure parameters in the sidebar, click **Apply parameters**, "
                "then click **Run simulation**.")
        return

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

    tabs = st.tabs([
        "3D scene", "Cable tensions", "Cable lengths", "Position",
        "Tracking error", "Condition number",
    ])

    try:
        with tabs[0]:
            snapshot_pose = Pose(
                position=result.positions[-1],
                rotation=Rotation.from_quat(result.quaternions_xyzw[-1]),
            )
            fig = render_scene(
                robot, snapshot_pose,
                options=SceneOptions(tension_heatmap=True),
                tensions=result.cable_tensions[-1],
                trajectory_positions=result.positions,
            )
            st.pyplot(fig, clear_figure=True)
        with tabs[1]:
            st.pyplot(plots2d.plot_cable_tensions(result, robot=robot), clear_figure=True)
        with tabs[2]:
            st.pyplot(plots2d.plot_cable_lengths(result), clear_figure=True)
        with tabs[3]:
            st.pyplot(plots2d.plot_position(result), clear_figure=True)
        with tabs[4]:
            if reference is None:
                st.warning("Tracking error requires a reference trajectory.")
            else:
                st.pyplot(plots2d.plot_tracking_error(result, reference), clear_figure=True)
        with tabs[5]:
            st.pyplot(plots2d.plot_condition_number(result, robot), clear_figure=True)
    except Exception as exc:
        st.error(f"Failed to render results: {type(exc).__name__}: {exc}")
        st.exception(exc)


# ---------------------------------------------------------------------------
# Upload panel
# ---------------------------------------------------------------------------

def _upload_panel() -> None:
    st.divider()
    st.subheader("Experiment upload")
    uploaded = st.file_uploader(
        "Drop a CSV / XLSX experimental log",
        type=["csv", "xlsx", "xls"],
        help="The file is parsed with pandas and the head is shown below; the "
             "schema is left to the experimenter --- this surface is a "
             "quick-look, not a sanitiser.",
        key="upload_file",
    )
    if uploaded is None:
        return
    try:
        raw = uploaded.read()
        df: pd.DataFrame
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw))
        st.write(f"Loaded **{uploaded.name}** with shape {df.shape}.")
        st.dataframe(df.head(50), height=300)
    except Exception as exc:
        st.error(f"Failed to parse uploaded file: {type(exc).__name__}: {exc}")
        st.exception(exc)


# Streamlit executes the module body when launched via `streamlit run`.
render()
