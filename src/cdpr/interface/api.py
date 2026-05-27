r"""FastAPI service: JSON-in, JSON-out CDPR simulation and analysis.

Run with::

    pip install 'cdpr[api,viz]'
    uvicorn cdpr.interface.api:app --host 0.0.0.0 --port 8000

Endpoints
---------

================  ====================================================================
``GET /health``   Liveness probe.
``GET /robots``   List the available reference robots, with their cable and DOF counts.
``POST /simulate`` Run a forward simulation; returns time series + summary statistics.
``POST /workspace`` Scan a translational workspace at a fixed orientation.
``POST /plot``     Render a single 2D analysis plot for a recorded experiment;
                  returns the figure as base64-encoded PNG.
================  ====================================================================

All endpoints accept dataclass-shaped JSON bodies that mirror the structures
in :mod:`cdpr.interface.specs`; FastAPI auto-generates Pydantic models from
those dataclasses, so adding a field there propagates here automatically.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import numpy as np

from cdpr.interface.specs import (
    PlotKind,
    SimulationRequest,
    WorkspaceRequest,
    build_robot,
    build_trajectory,
)

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.dynamics.simulator import SimulationResult


def _require_fastapi():
    try:
        import fastapi  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The HTTP service needs FastAPI. Install with:  pip install 'cdpr[api]'"
        ) from exc


_require_fastapi()
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel               # noqa: E402

app = FastAPI(
    title="cdpr",
    version="0.1.0",
    summary="HTTP service for the CDPR computational framework",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summarise(result: "SimulationResult") -> dict[str, object]:
    pos = np.linalg.norm(result.positions, axis=-1)
    return {
        "n_samples": int(len(result.time)),
        "duration": float(result.time[-1] - result.time[0]) if len(result.time) > 1 else 0.0,
        "position_magnitude": {
            "mean": float(pos.mean()), "max": float(pos.max()), "min": float(pos.min()),
        },
        "cable_tension_range": {
            "min": float(result.cable_tensions.min()),
            "max": float(result.cable_tensions.max()),
        },
        "infeasible_steps": list(map(int, result.infeasible_steps)),
    }


def _serialise_result(result: "SimulationResult") -> dict[str, object]:
    return {
        "time": result.time.tolist(),
        "positions": result.positions.tolist(),
        "quaternions_xyzw": result.quaternions_xyzw.tolist(),
        "linear_velocities": result.linear_velocities.tolist(),
        "angular_velocities": result.angular_velocities.tolist(),
        "cable_tensions": result.cable_tensions.tolist(),
        "cable_lengths": result.cable_lengths.tolist(),
        "infeasible_steps": list(map(int, result.infeasible_steps)),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict[str, object]:
    """Service descriptor returned when the root URL is hit in a browser.

    Lists the available endpoints + points the visitor at the OpenAPI
    docs. The actual API surface is the other ``@app.<method>`` routes
    below --- this is a friendly landing page, not the API itself.
    """
    return {
        "name": "cdpr",
        "version": app.version,
        "summary": app.summary,
        "endpoints": {
            "health":     "GET  /health",
            "robots":     "GET  /robots",
            "simulate":   "POST /simulate",
            "workspace":  "POST /workspace",
            "plot":       "POST /plot",
            "openapi":    "GET  /openapi.json",
            "swagger_ui": "GET  /docs",
            "redoc":      "GET  /redoc",
        },
        "repository": "https://github.com/Tachia/cdpr_simulator",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/robots")
def list_robots() -> list[dict[str, object]]:
    """Return the catalog of available reference robots."""
    from cdpr.robots import cogiro_class, ipanema_class, planar_translational, point_mass_3d
    factories = {
        "point_mass_3d": point_mass_3d,
        "planar_translational": planar_translational,
        "ipanema_class": ipanema_class,
        "cogiro_class": cogiro_class,
    }
    out: list[dict[str, object]] = []
    for name, fn in factories.items():
        r = fn()
        out.append({
            "name": name,
            "human_name": r.name,
            "dof": r.dof,
            "n_cables": r.n_cables,
            "redundancy": r.redundancy,
            "has_inertia": r.inertia is not None,
            "has_limits": r.limits is not None,
        })
    return out


class SimulationResponse(BaseModel):
    summary: dict
    series: dict | None = None


@app.post("/simulate", response_model=SimulationResponse)
def simulate_endpoint(req: SimulationRequest, full: bool = True) -> SimulationResponse:
    """Run one forward simulation.

    Set ``full=false`` to return only the summary block (no time series),
    which is useful when the caller only wants infeasibility flags or
    aggregate statistics from a parameter sweep.
    """
    from cdpr.core.frames import Pose
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.dynamics.simulator import simulate
    from scipy.spatial.transform import Rotation

    robot = build_robot(req.robot, payload_mass=req.payload_mass)
    reference = build_trajectory(req.trajectory)
    initial_pose = Pose(position=np.zeros(3), rotation=Rotation.identity())
    try:
        result = simulate(
            robot=robot,
            state0=PlatformState.at_rest(initial_pose),
            duration=req.duration,
            dt=req.dt,
            reference_pose=reference,
            integrator=req.integrator,
            tension_objective=req.tension_objective,
            gravity=req.gravity,
        )
    except Exception as exc:                          # pragma: no cover - surfaced as 4xx/5xx
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return SimulationResponse(
        summary=_summarise(result),
        series=_serialise_result(result) if full else None,
    )


class WorkspaceResponse(BaseModel):
    xs: list[float]
    ys: list[float]
    zs: list[float]
    mask_flat: list[int]
    n_inside: int
    fraction_inside: float
    estimated_volume: float


@app.post("/workspace", response_model=WorkspaceResponse)
def workspace_endpoint(req: WorkspaceRequest) -> WorkspaceResponse:
    from cdpr.core.frames import Wrench
    from cdpr.workspace.grid import scan_translational_workspace

    robot = build_robot(req.robot)
    wrench = (
        Wrench.from_parts([0, 0, -robot.inertia.mass * 9.81], np.zeros(3))
        if req.kind == "wfw" and robot.inertia is not None else None
    )
    grid = scan_translational_workspace(
        robot, xlim=req.xlim, ylim=req.ylim, zlim=req.zlim,
        resolution=req.resolution, kind=req.kind, external_wrench=wrench,
    )
    return WorkspaceResponse(
        xs=grid.xs.tolist(),
        ys=grid.ys.tolist(),
        zs=grid.zs.tolist(),
        mask_flat=grid.mask.astype(int).flatten().tolist(),
        n_inside=int(grid.n_inside),
        fraction_inside=float(grid.fraction_inside),
        estimated_volume=float(grid.estimated_volume()),
    )


class PlotRequest(BaseModel):
    log_root: str
    kind: PlotKind = "cable_tensions"


class PlotResponse(BaseModel):
    kind: str
    format: str = "png"
    base64: str


@app.post("/plot", response_model=PlotResponse)
def plot_endpoint(req: PlotRequest) -> PlotResponse:
    """Render a single analytic plot from a recorded experiment."""
    from cdpr.recording import load_experiment
    from cdpr.recording.replay import robot_from_snapshot
    from cdpr.viz import plots2d
    from cdpr.viz.export import figure_to_png_bytes
    from cdpr.viz.style import apply_paper_style

    apply_paper_style()
    exp = load_experiment(req.log_root)
    robot = robot_from_snapshot(exp.metadata["robot"])

    # The plotting layer accepts SimulationResult-shaped inputs; the
    # Experiment dataclass mirrors that shape, so it works as-is.
    handlers = {
        "position": lambda: plots2d.plot_position(exp),
        "velocity": lambda: plots2d.plot_velocity(exp),
        "angular_velocity": lambda: plots2d.plot_angular_velocity(exp),
        "cable_lengths": lambda: plots2d.plot_cable_lengths(exp),
        "cable_tensions": lambda: plots2d.plot_cable_tensions(exp, robot=robot),
        "condition_number": lambda: plots2d.plot_condition_number(exp, robot),
        "trajectory_xy": lambda: plots2d.plot_trajectory_projection(exp.positions, plane="xy"),
        "trajectory_xz": lambda: plots2d.plot_trajectory_projection(exp.positions, plane="xz"),
        "trajectory_yz": lambda: plots2d.plot_trajectory_projection(exp.positions, plane="yz"),
    }
    if req.kind not in handlers:
        raise HTTPException(status_code=400, detail=f"Unsupported plot kind: {req.kind}")
    fig = handlers[req.kind]()
    png = figure_to_png_bytes(fig)
    return PlotResponse(kind=req.kind, base64=base64.b64encode(png).decode())
