"""Schema dataclasses for experiment recordings.

Three top-level objects are persisted:

* :class:`ExperimentMetadata` -- robot configuration, simulation settings, an
  arbitrary tag dict. Stored as ``metadata.json``.
* :class:`ReproducibilityManifest` -- versions, seed, git hash, run timestamp.
  Stored as ``manifest.json``.
* Per-step diagnostics columns -- listed in :data:`DIAGNOSTIC_COLUMNS`,
  stored as ``diagnostics.csv``.

Time series themselves (``timeseries.csv``) are written as fixed-schema
columns rather than via this module --- their column structure depends on
``n_cables`` and we want the CSV header to make that explicit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# Diagnostics CSV columns. Adding a column means adding it here and writing
# it from the recorder; the replay loader reads whatever it finds.
DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "time",
    "condition_number",
    "infeasible",
    "tension_residual",
)


@dataclass(slots=True)
class RobotSnapshot:
    """Numeric snapshot of a :class:`cdpr.geometry.robot.Robot`.

    Stored as part of :class:`ExperimentMetadata`. Sufficient to rebuild the
    robot via :func:`cdpr.recording.replay.robot_from_snapshot`.
    """

    name: str
    dof: int
    n_cables: int
    anchors: list[list[float]]
    attachments: list[list[float]]
    mass: float | None = None
    inertia: list[list[float]] | None = None
    t_min: list[float] | None = None
    t_max: list[float] | None = None
    cable_youngs_modulus: list[float] | None = None
    cable_cross_section: list[float] | None = None
    cable_linear_density: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SimulationSettings:
    """Numerical settings used to drive a simulation."""

    duration: float
    dt: float
    integrator: str = "rk4"
    tension_objective: str = "centered"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    reference_trajectory: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["gravity"] = list(self.gravity)
        return d


@dataclass(slots=True)
class ExperimentMetadata:
    """Top-level metadata block written to ``metadata.json``."""

    experiment_id: str
    title: str
    robot: RobotSnapshot
    simulation: SimulationSettings
    tags: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "schema_version": self.schema_version,
            "robot": self.robot.to_dict(),
            "simulation": self.simulation.to_dict(),
            "tags": self.tags,
        }


@dataclass(slots=True)
class ReproducibilityManifest:
    """Reproducibility data written to ``manifest.json``.

    Fields default to neutral values so a partially-filled manifest is still
    valid JSON; callers should populate as much as is available.
    """

    created_at: str
    cdpr_version: str
    python_version: str
    numpy_version: str
    scipy_version: str
    platform: str
    seed: int | None = None
    git_revision: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
