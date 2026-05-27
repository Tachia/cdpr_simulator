r"""Streamlit research console.

Launch with::

    pip install 'cdpr[gui,viz]'
    streamlit run -m cdpr.interface.gui
    # or:
    python -m streamlit run src/cdpr/interface/gui.py

The console exposes four panels:

1. **Robot & trajectory** -- pick a reference robot, choose a trajectory kind
   (hold / line / circle / Lissajous), set duration and step size.
2. **Simulation** -- trigger ``cdpr.dynamics.simulator.simulate``, show summary
   statistics and an inline 3D scene snapshot.
3. **Analysis** -- toggle which 2D plots to render against the most recent
   simulation; plots are drawn lazily on demand.
4. **Experiment upload** -- accept a CSV / XLSX experimental log and overlay
   it against the simulation.

The session state caches the most recent :class:`SimulationResult` so that
toggling a plot does not re-run the integration.
"""

from __future__ import annotations

try:
    import streamlit as st
except ImportError as exc:
    raise ImportError(
        "The Streamlit console needs Streamlit. Install with:  pip install 'cdpr[gui]'"
    ) from exc

import io

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
# Layout
# ---------------------------------------------------------------------------

def render() -> None:
    st.set_page_config(page_title="CDPR research console", layout="wide")
    apply_dissertation_style()

    st.title("CDPR research console")
    st.caption(
        "Front-end for the cdpr scientific core. All physics lives in the core; "
        "this surface only selects parameters, triggers runs, and renders results."
    )

    request = _request_panel()
    _action_bar(request)
    _results_panel()
    _upload_panel()


def _request_panel() -> SimulationRequest:
    with st.sidebar:
        st.header("Experiment")
        robot_name = st.selectbox(
            "Robot", ["ipanema_class", "cogiro_class", "point_mass_3d", "planar_translational"],
            index=0,
        )
        payload = st.number_input("Payload mass [kg]", min_value=0.0, value=0.0, step=0.5)
        st.divider()

        st.header("Trajectory")
        kind = st.selectbox("Kind", ["hold", "line", "circle", "lissajous"], index=2)
        duration = st.number_input("Duration [s]", min_value=0.1, value=1.5, step=0.1)
        dt = st.number_input("Time step dt [s]", min_value=1e-4, max_value=1e-1,
                             value=2e-3, step=1e-3, format="%.4f")

        params: dict = {}
        if kind == "line":
            params["start"] = list(st.text_input("start (x,y,z)", "0,0,0").split(","))
            params["end"] = list(st.text_input("end   (x,y,z)", "0.3,0,0").split(","))
            params["start"] = [float(x) for x in params["start"]]
            params["end"] = [float(x) for x in params["end"]]
        elif kind == "circle":
            r = st.slider("radius [m]", 0.05, 1.0, 0.3, step=0.05)
            axis = st.selectbox("axis", ["z", "y", "x"], index=0)
            axis_vec = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
            params = {"center": [0, 0, 0], "radius": r, "axis": axis_vec,
                      "angle_span": 2 * np.pi}
        elif kind == "lissajous":
            params = {
                "center": [0, 0, 0],
                "amplitudes": [
                    st.slider("Ax", 0.0, 1.0, 0.3, step=0.05),
                    st.slider("Ay", 0.0, 1.0, 0.2, step=0.05),
                    st.slider("Az", 0.0, 1.0, 0.0, step=0.05),
                ],
                "frequencies": [
                    st.slider("fx", 1.0, 5.0, 1.0, step=0.5),
                    st.slider("fy", 1.0, 5.0, 2.0, step=0.5),
                    st.slider("fz", 0.0, 5.0, 0.0, step=0.5),
                ],
                "phases": [0.0, np.pi / 2, 0.0],
            }

        st.divider()
        st.header("Solver")
        integrator = st.radio("Integrator", ["rk4", "semi_implicit"], horizontal=True)
        objective = st.radio("Tension objective", ["centered", "min_norm", "preferred"],
                             horizontal=True)

    return SimulationRequest(
        robot=robot_name,                                                       # type: ignore[arg-type]
        trajectory=TrajectorySpec(kind=kind, duration=duration, params=params),  # type: ignore[arg-type]
        duration=duration,
        dt=dt,
        integrator=integrator,                                                  # type: ignore[arg-type]
        tension_objective=objective,                                            # type: ignore[arg-type]
        payload_mass=payload,
    )


def _action_bar(request: SimulationRequest) -> None:
    cols = st.columns([1, 1, 6])
    if cols[0].button("Run simulation", type="primary"):
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
    if cols[1].button("Clear cached run"):
        for key in ("last_result", "last_robot", "last_reference"):
            st.session_state.pop(key, None)
        st.toast("Cleared.")


def _results_panel() -> None:
    result = st.session_state.get("last_result")
    robot = st.session_state.get("last_robot")
    reference = st.session_state.get("last_reference")
    if result is None or robot is None:
        st.info("Configure parameters in the sidebar and click *Run simulation*.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("samples", len(result.time))
    summary_cols[1].metric("max |p|", f"{np.linalg.norm(result.positions, axis=1).max():.3f} m")
    summary_cols[2].metric(
        "tension range",
        f"{result.cable_tensions.min():.0f} – {result.cable_tensions.max():.0f} N",
    )
    summary_cols[3].metric("infeasible steps", len(result.infeasible_steps))

    tabs = st.tabs(["3D scene", "Cable tensions", "Cable lengths", "Position",
                    "Tracking error", "Condition number"])

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


def _upload_panel() -> None:
    st.divider()
    st.subheader("Experiment upload")
    uploaded = st.file_uploader(
        "Drop a CSV / XLSX experimental log",
        type=["csv", "xlsx", "xls"],
        help="The file is parsed with pandas and shown below; "
             "the schema is left to the experimenter --- "
             "this surface is a quick-look, not a sanitiser.",
    )
    if uploaded is None:
        return
    raw = uploaded.read()
    df: pd.DataFrame
    if uploaded.name.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        df = pd.read_csv(io.BytesIO(raw))
    st.write(f"Loaded **{uploaded.name}** with shape {df.shape}.")
    st.dataframe(df.head(50), height=300)


# Streamlit executes the module body when launched via `streamlit run`.
render()
