r"""Streamlit research console for the CDPR framework.

Launch with::

    pip install 'cdpr[gui,viz]'
    streamlit run streamlit_app.py

Design notes (why the page used to go blank on every rerun)
-----------------------------------------------------------

Streamlit Community Cloud runs each app in a ~1 GB sandbox. Two
distinct failure modes hit earlier revisions:

1. **Run-simulation OOM.** The previous results panel rendered six
   matplotlib figures --- including a 3D scene with the full trajectory
   --- on every script rerun. The worker was killed before the WebSocket
   could deliver the first frame, so the user saw a blank page with no
   Python-level traceback. Fix: per-tab lazy rendering. Each plot has
   a "Render" button and its figure is cached in ``st.session_state``
   for the lifetime of the simulation result.

2. **Any-rerun blank.** Widgets that carry both ``value=`` and ``key=``
   trigger a warning if ``st.session_state[key]`` was already set by a
   previous run with a different value (e.g. after a redeploy that
   changed the default). On Cloud's free worker a deep enough warning
   chain (Streamlit emits one *per* widget) inflates the WebSocket
   payload past its limit and the page disconnects silently. Fix:
   pre-populate every default through :func:`st.session_state.setdefault`
   *before* widgets are created, and pass only ``key=`` to the widget
   (no ``value=`` / ``index=``).

Everything inside :func:`render` runs under a top-level try/except.
Any uncaught exception is shown via :func:`st.exception` plus a
``traceback`` block --- so the user always sees the failure cause
rather than a blank page.

A short ``_log()`` helper writes to stderr so the Streamlit Cloud
"manage app" log carries breadcrumbs even when the in-app UI is gone.
"""

from __future__ import annotations

# ---- Streamlit & matplotlib are imported FIRST so backend pinning happens
# ---- before any plotting module sees matplotlib for the first time.

import os
import sys
import traceback

try:
    import streamlit as st
except ImportError as exc:                                       # pragma: no cover
    raise ImportError(
        "The Streamlit console needs Streamlit. Install with:  pip install 'cdpr[gui]'"
    ) from exc

import matplotlib                                                # noqa: E402
matplotlib.use("Agg", force=True)

import io                                                        # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy.spatial.transform import Rotation                     # noqa: E402

from cdpr.core.frames import Pose                                # noqa: E402
from cdpr.dynamics.rigid_body import PlatformState               # noqa: E402
from cdpr.dynamics.simulator import simulate                     # noqa: E402
from cdpr.interface.specs import (                               # noqa: E402
    SimulationRequest,
    TrajectorySpec,
    build_robot,
    build_trajectory,
)


# Build-id banner: lets the user tell instantly whether Streamlit Cloud
# is serving the latest commit or a stale cached worker. Bump when the
# behavioural contract of this file changes.
BUILD_ID = "gui-2026-05-28-d"

# Frugal mode trims the *simulator* workload to keep the integration
# cheap on the free tier (1 GB / 1 vCPU). Plot rendering is already
# fully lazy, so this knob only affects how long simulate() runs.
FRUGAL = bool(int(os.environ.get("CDPR_GUI_FRUGAL", "1")))


def _log(msg: str) -> None:
    """Write to stderr so it surfaces in Streamlit Cloud's manage-app log."""
    print(f"[cdpr-gui] {BUILD_ID} :: {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# set_page_config must be the first Streamlit call in the process and is
# allowed to run exactly once. Use a module-level flag (not session_state)
# so it survives any session_state.clear() in the Reset button.
# ---------------------------------------------------------------------------

_PAGE_CONFIG_DONE = False


def _ensure_page_config() -> None:
    global _PAGE_CONFIG_DONE
    if _PAGE_CONFIG_DONE:
        return
    try:
        st.set_page_config(
            page_title="CDPR research console",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception as exc:                                      # pragma: no cover
        # Streamlit raises on duplicate set_page_config; treat as benign.
        _log(f"set_page_config skipped: {type(exc).__name__}: {exc}")
    _PAGE_CONFIG_DONE = True


# ---------------------------------------------------------------------------
# Default state setup --- runs once per session before any widget exists.
# ---------------------------------------------------------------------------

def _seed_defaults() -> None:
    """Populate :mod:`st.session_state` with widget defaults.

    We seed *before* creating any widget so we never have to pass both
    ``value=`` and ``key=`` simultaneously --- the source of the
    repeated 'value-and-state-may-differ' warnings that previously
    inflated the rerun WebSocket payload past Cloud's limit.

    Keys here mirror :class:`SimulationRequest` / :class:`TrajectorySpec`
    field semantics exactly. Robot identifiers use the canonical names
    from :mod:`cdpr.interface.specs` (``point_mass_3d`` etc.) so
    :func:`build_robot` accepts them without translation.
    """
    defaults: dict[str, object] = {
        # SimulationRequest
        "robot": "ipanema_class",
        "payload_mass": 0.0,
        "gravity_on": True,
        "objective": "centered",
        "duration": 0.5 if FRUGAL else 1.5,
        "dt": 5e-3 if FRUGAL else 2e-3,
        # TrajectorySpec
        "kind": "circle",
        # line params
        "line_start": "0,0,0.5",
        "line_end": "0.3,0,0.5",
        # circle params --- build_trajectory expects center / radius /
        # axis / angle_span (NOT period). Defaults trace one full revolution.
        "circle_center": "0,0,0.5",
        "circle_radius": 0.2,
        "circle_axis": "0,0,1",
        "circle_angle_span": float(2 * np.pi),
        # lissajous params --- build_trajectory expects amplitudes /
        # frequencies / phases (plural).
        "lissajous_center": "0,0,0.5",
        "lissajous_amplitudes": "0.2,0.2,0.0",
        "lissajous_frequencies": "1.0,2.0,0.0",
        "lissajous_phases": "0,1.5708,0",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ---------------------------------------------------------------------------
# Top-level entry --- imported & called by streamlit_app.py at script start.
# ---------------------------------------------------------------------------

def render() -> None:
    _ensure_page_config()
    _seed_defaults()

    # Wrap *everything* below in a top-level try/except so any failure
    # is reported in-app rather than blanking the page.
    try:
        _log("render() begin")

        # Visible build banner + reset escape hatch.
        top = st.columns([6, 1])
        top[0].title("CDPR research console")
        top[0].caption(
            f"Build {BUILD_ID}. Configure parameters in the sidebar, click "
            "**Run simulation**, then use the **Render** button inside each "
            "result tab. Plot rendering is lazy to fit the free Streamlit "
            "Cloud worker."
        )
        if top[1].button("Reset session", key="btn_reset"):
            for k in list(st.session_state.keys()):
                if k != "_page_config_done":
                    del st.session_state[k]
            st.rerun()

        request = _request_sidebar()
        _action_bar(request)
        _results_panel()
        _upload_panel()
        _diagnostics_expander()

        _log("render() end")

    except Exception as exc:
        _log(f"render() crashed: {type(exc).__name__}: {exc}")
        st.error(f"Unhandled exception in console: {type(exc).__name__}: {exc}")
        st.exception(exc)
        st.code(traceback.format_exc(), language="text")


# ---------------------------------------------------------------------------
# Sidebar widgets (NO st.form --- the form added before did not fix the
# blank-page-on-rerun symptom and made debugging harder). Every widget
# has only `key=`; their initial values came from _seed_defaults().
# ---------------------------------------------------------------------------

def _request_sidebar() -> SimulationRequest:
    """Build a :class:`SimulationRequest` from the sidebar widget state.

    Widget keys map 1:1 to :class:`SimulationRequest` /
    :class:`TrajectorySpec` field semantics. Robot names use the
    canonical identifiers from :mod:`cdpr.interface.specs` (``ipanema_class``
    rather than ``ipanema``) so the result is consumable by the same
    :func:`build_robot` helper the API uses.
    """
    with st.sidebar:
        st.header("Experiment")
        st.selectbox(
            "Robot",
            ["point_mass_3d", "planar_translational", "ipanema_class", "cogiro_class"],
            key="robot",
            help="Reference robot from cdpr.robots. Point-mass is the simplest "
                 "(translation-only); IPAnema-class and CoGiRo-class are 6-DOF.",
        )
        st.number_input(
            "Payload mass [kg]", min_value=0.0, step=0.1, key="payload_mass",
            help="Extra mass added to the platform's nominal inertia.",
        )
        st.checkbox(
            "Apply gravity", key="gravity_on",
            help="If off, the simulator runs with zero gravitational acceleration.",
        )
        st.radio(
            "Tension distribution objective",
            ["min_norm", "centered", "preferred"],
            horizontal=True, key="objective",
        )

        st.header("Time")
        st.number_input(
            "Duration [s]",
            min_value=0.05, max_value=5.0, step=0.05, key="duration",
        )
        st.number_input(
            "Time step dt [s]",
            min_value=1e-4, max_value=5e-2, step=1e-3,
            format="%.4f", key="dt",
        )

        st.header("Trajectory")
        st.selectbox(
            "Kind", ["hold", "line", "circle", "lissajous"], key="kind",
        )

        st.subheader("Line")
        st.text_input("start (x,y,z) [m]", key="line_start")
        st.text_input("end   (x,y,z) [m]", key="line_end")

        st.subheader("Circle")
        st.text_input("center (x,y,z) [m]", key="circle_center")
        st.number_input(
            "radius [m]", min_value=0.0, step=0.05, key="circle_radius",
        )
        st.text_input("axis (x,y,z)", key="circle_axis")
        st.number_input(
            "angle span [rad]", min_value=0.1, step=0.1,
            key="circle_angle_span",
        )

        st.subheader("Lissajous")
        st.text_input("center (x,y,z) [m]", key="lissajous_center")
        st.text_input("amplitudes (x,y,z) [m]", key="lissajous_amplitudes")
        st.text_input("frequencies (x,y,z) [rad/s]", key="lissajous_frequencies")
        st.text_input("phases (x,y,z) [rad]", key="lissajous_phases")

    # Build the kind-specific params dict. Keys here MUST match what
    # cdpr.interface.specs.build_trajectory consumes:
    #   line     -> start, end
    #   circle   -> center, radius, axis, angle_span
    #   lissajous-> center, amplitudes, frequencies, phases
    kind = st.session_state["kind"]
    if kind == "line":
        params: dict = {
            "start": _parse_xyz(st.session_state["line_start"], default=[0.0, 0.0, 0.5]),
            "end":   _parse_xyz(st.session_state["line_end"],   default=[0.3, 0.0, 0.5]),
        }
    elif kind == "circle":
        params = {
            "center": _parse_xyz(st.session_state["circle_center"], default=[0.0, 0.0, 0.5]),
            "radius": float(st.session_state["circle_radius"]),
            "axis":   _parse_xyz(st.session_state["circle_axis"],   default=[0.0, 0.0, 1.0]),
            "angle_span": float(st.session_state["circle_angle_span"]),
        }
    elif kind == "lissajous":
        params = {
            "center":      _parse_xyz(st.session_state["lissajous_center"],      default=[0.0, 0.0, 0.5]),
            "amplitudes":  _parse_xyz(st.session_state["lissajous_amplitudes"],  default=[0.2, 0.2, 0.0]),
            "frequencies": _parse_xyz(st.session_state["lissajous_frequencies"], default=[1.0, 2.0, 0.0]),
            "phases":      _parse_xyz(st.session_state["lissajous_phases"],      default=[0.0, np.pi / 2, 0.0]),
        }
    else:  # hold
        params = {}

    duration = float(st.session_state["duration"])
    gravity_vec = (0.0, 0.0, -9.81) if bool(st.session_state["gravity_on"]) else (0.0, 0.0, 0.0)

    return SimulationRequest(
        robot=st.session_state["robot"],
        payload_mass=float(st.session_state["payload_mass"]),
        gravity=gravity_vec,
        tension_objective=st.session_state["objective"],
        duration=duration,
        dt=float(st.session_state["dt"]),
        trajectory=TrajectorySpec(kind=kind, duration=duration, params=params),
    )


def _parse_xyz(text: str, *, default: list[float]) -> list[float]:
    try:
        parts = [p.strip() for p in str(text).split(",") if p.strip()]
        if len(parts) != 3:
            return list(default)
        return [float(p) for p in parts]
    except (TypeError, ValueError):
        return list(default)


# ---------------------------------------------------------------------------
# Action bar --- Run simulation + Clear
# ---------------------------------------------------------------------------

def _action_bar(request: SimulationRequest) -> None:
    cols = st.columns([1, 1, 6])
    if cols[0].button("Run simulation", type="primary", key="btn_run"):
        try:
            _log(
                f"Run: robot={request.robot} kind={request.trajectory.kind} "
                f"duration={request.duration} dt={request.dt}"
            )
            with st.spinner("Integrating…"):
                robot = build_robot(request.robot, payload_mass=request.payload_mass)
                ref = build_trajectory(request.trajectory)
                # PlatformState.at_rest(pose) is the canonical factory.
                # Pose has no from_translation() --- use the constructor
                # with Rotation.identity() for a position-only start.
                p0 = ref(0.0).position
                state0 = PlatformState.at_rest(
                    Pose(position=p0, rotation=Rotation.identity())
                )
                result = simulate(
                    robot=robot,
                    state0=state0,
                    duration=request.duration,
                    dt=request.dt,
                    reference=ref,
                    tension_objective=request.tension_objective,
                    gravity=request.gravity,
                )
            _log(f"Sim ok: {len(result.time)} samples")
            st.session_state["last_result"] = result
            st.session_state["last_robot"] = robot
            st.session_state["last_reference"] = ref
            st.session_state.pop("_figure_cache", None)
            st.toast(f"Simulation finished ({len(result.time)} samples).")
        except Exception as exc:
            _log(f"Sim crashed: {type(exc).__name__}: {exc}")
            st.error(f"Simulation failed: {type(exc).__name__}: {exc}")
            st.exception(exc)
    if cols[1].button("Clear cached run", key="btn_clear"):
        for k in ("last_result", "last_robot", "last_reference", "_figure_cache"):
            st.session_state.pop(k, None)
        st.toast("Cleared.")


# ---------------------------------------------------------------------------
# Results panel --- lazy per-tab rendering.
# ---------------------------------------------------------------------------

def _results_panel() -> None:
    result = st.session_state.get("last_result")
    robot = st.session_state.get("last_robot")
    reference = st.session_state.get("last_reference")
    if result is None or robot is None:
        st.info("Configure parameters in the sidebar, then click **Run simulation**.")
        return

    cols = st.columns(4)
    cols[0].metric("samples", len(result.time))
    cols[1].metric("duration [s]", f"{float(result.time[-1]):.3f}")
    cols[2].metric(
        "max tension [N]",
        f"{float(np.max(result.cable_tensions)):.1f}",
    )
    cols[3].metric("infeasible steps", len(result.infeasible_steps))

    tab_labels = [
        "Position", "Cable tensions", "Cable lengths",
        "Tracking error", "Condition number", "3D scene (heavy)",
    ]
    tabs = st.tabs(tab_labels)
    _lazy_tab(tabs[0], "position",         lambda: _plot_position(result))
    _lazy_tab(tabs[1], "cable_tensions",   lambda: _plot_cable_tensions(result, robot))
    _lazy_tab(tabs[2], "cable_lengths",    lambda: _plot_cable_lengths(result))
    _lazy_tab(tabs[3], "tracking_error",   lambda: _plot_tracking_error(result, reference))
    _lazy_tab(tabs[4], "condition_number", lambda: _plot_condition_number(result, robot))
    _lazy_tab(tabs[5], "scene_3d",         lambda: _plot_scene_3d(result, robot), warn_heavy=True)


def _lazy_tab(tab, key: str, render_fn, *, warn_heavy: bool = False) -> None:
    cache = st.session_state.setdefault("_figure_cache", {})
    with tab:
        if warn_heavy:
            st.caption(
                "The 3D scene is the heaviest figure --- click below to render."
            )
        if key in cache:
            st.pyplot(cache[key], clear_figure=False)
            if st.button(f"Re-render {key}", key=f"rerender_{key}"):
                cache.pop(key, None)
                st.rerun()
            return
        if st.button(f"Render {key}", key=f"render_{key}", type="primary"):
            try:
                _log(f"Render {key}…")
                fig = render_fn()
                cache[key] = fig
                _log(f"Render {key} ok")
                st.rerun()
            except Exception as exc:
                _log(f"Render {key} crashed: {type(exc).__name__}: {exc}")
                st.error(f"Could not render {key}: {type(exc).__name__}: {exc}")
                st.exception(exc)


# ---------------------------------------------------------------------------
# Plot factories --- imported lazily to keep module-load cheap.
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    try:
        from cdpr.viz.style import apply_paper_style
        apply_paper_style()
    except Exception as exc:                                      # pragma: no cover
        _log(f"apply_paper_style skipped: {type(exc).__name__}: {exc}")


def _plot_position(result):
    _apply_style()
    from cdpr.viz import plots2d
    return plots2d.plot_position(result)


def _plot_cable_tensions(result, robot):
    _apply_style()
    from cdpr.viz import plots2d
    return plots2d.plot_cable_tensions(result, robot=robot)


def _plot_cable_lengths(result):
    _apply_style()
    from cdpr.viz import plots2d
    return plots2d.plot_cable_lengths(result)


def _plot_tracking_error(result, reference):
    if reference is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no reference trajectory configured",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    _apply_style()
    from cdpr.viz import plots2d
    return plots2d.plot_tracking_error(result, reference)


def _plot_condition_number(result, robot):
    _apply_style()
    from cdpr.viz import plots2d
    return plots2d.plot_condition_number(result, robot)


def _plot_scene_3d(result, robot):
    _apply_style()
    from cdpr.viz.scene import SceneOptions, render_scene
    snapshot_pose = Pose(
        position=result.positions[-1],
        rotation=Rotation.from_quat(result.quaternions_xyzw[-1]),
    )
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
# CSV / XLSX peek
# ---------------------------------------------------------------------------

def _upload_panel() -> None:
    st.divider()
    st.subheader("Experimental log")
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


# ---------------------------------------------------------------------------
# Diagnostics expander --- visible breadcrumb of what state Streamlit
# thinks it has. Helpful when the page goes blank: open the expander on
# the previous run to inspect what the worker saw.
# ---------------------------------------------------------------------------

def _diagnostics_expander() -> None:
    with st.expander("Diagnostics", expanded=False):
        try:
            import streamlit as _st_mod
            st.write({
                "build_id": BUILD_ID,
                "streamlit_version": getattr(_st_mod, "__version__", "?"),
                "matplotlib_backend": matplotlib.get_backend(),
                "frugal_mode": FRUGAL,
                "python": sys.version.split()[0],
                "session_state_keys": sorted(
                    k for k in st.session_state.keys()
                    if not k.startswith("_")
                ),
                "cached_figures": list(
                    (st.session_state.get("_figure_cache") or {}).keys()
                ),
            })
        except Exception as exc:                                  # pragma: no cover
            st.warning(f"diagnostics unavailable: {exc}")


# ---------------------------------------------------------------------------
# Entry point --- importing this module under `streamlit run` immediately
# kicks off the render. Streamlit replays the script on every interaction.
# ---------------------------------------------------------------------------

render()
