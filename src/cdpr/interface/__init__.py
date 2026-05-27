"""Human-and-machine interfaces to the CDPR framework.

Two surfaces are provided:

* :mod:`cdpr.interface.api` -- a FastAPI service that exposes simulation,
  workspace analysis, and plotting as HTTP endpoints. Returns JSON
  everywhere; figures are inlined as base64 PNG. Install with
  ``pip install 'cdpr[api]'`` and run with ``uvicorn cdpr.interface.api:app``.
* :mod:`cdpr.interface.gui` -- a Streamlit research console with structured
  parameter selectors, a simulation trigger, live plot pane, and experiment
  upload. Install with ``pip install 'cdpr[gui]'`` and run with
  ``streamlit run -m cdpr.interface.gui``.

Both surfaces are *consumers* of the scientific core --- no physics lives
in this package.
"""

from cdpr.interface.specs import (
    PlotKind,
    RobotName,
    SimulationRequest,
    TrajectorySpec,
    WorkspaceRequest,
    build_robot,
    build_trajectory,
)

__all__ = [
    "RobotName",
    "PlotKind",
    "TrajectorySpec",
    "SimulationRequest",
    "WorkspaceRequest",
    "build_robot",
    "build_trajectory",
]
