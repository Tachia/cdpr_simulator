r"""Per-run metrics computed against a reference trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.dynamics.simulator import SimulationResult


@dataclass(slots=True)
class BenchmarkMetrics:
    """Closed-loop benchmark summary metrics."""

    n_samples: int
    duration: float
    runtime_s: float

    tracking_error_rms: float
    tracking_error_peak: float
    orientation_error_rms_deg: float
    orientation_error_peak_deg: float

    velocity_error_rms: float

    control_effort_rms: float                  # ||cable tension||_2 mean over time
    cable_tension_peak: float
    cable_tension_min: float

    feasibility_rate: float                    # fraction of feasible steps
    condition_number_max: float                # peak structure-matrix condition
    condition_number_median: float

    # Phase 7: which constitutive law produced these metrics; "none" means
    # the default Phase 1-6 tension-distribution path.
    cable_mode: str = "none"

    def to_dict(self) -> dict[str, float]:
        return {
            "n_samples": int(self.n_samples),
            "duration": self.duration,
            "runtime_s": self.runtime_s,
            "tracking_error_rms": self.tracking_error_rms,
            "tracking_error_peak": self.tracking_error_peak,
            "orientation_error_rms_deg": self.orientation_error_rms_deg,
            "orientation_error_peak_deg": self.orientation_error_peak_deg,
            "velocity_error_rms": self.velocity_error_rms,
            "control_effort_rms": self.control_effort_rms,
            "cable_tension_peak": self.cable_tension_peak,
            "cable_tension_min": self.cable_tension_min,
            "feasibility_rate": self.feasibility_rate,
            "condition_number_max": self.condition_number_max,
            "condition_number_median": self.condition_number_median,
            "cable_mode": self.cable_mode,
        }


def compute_metrics(
    result: "SimulationResult",
    reference_positions: NDArray[np.float64],
    reference_quaternions: NDArray[np.float64],
    reference_velocities: NDArray[np.float64],
    runtime_s: float,
    robot,
    *,
    cable_mode: str = "none",
) -> BenchmarkMetrics:
    """Reduce a :class:`SimulationResult` plus a reference series to
    one row of dissertation-table-ready metrics."""
    from cdpr.core.frames import Pose
    from scipy.spatial.transform import Rotation
    from cdpr.kinematics.jacobian import structure_matrix

    err_p = np.linalg.norm(result.positions - reference_positions, axis=1)

    # Quaternion angular distance, robust to the double-cover sign ambiguity.
    dot = np.abs(np.sum(result.quaternions_xyzw * reference_quaternions, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    err_o_deg = np.rad2deg(2.0 * np.arccos(dot))

    err_v = np.linalg.norm(result.linear_velocities - reference_velocities, axis=1)

    tens = result.cable_tensions
    effort = np.linalg.norm(tens, axis=1)

    n = len(result.time)
    feasibility_rate = 1.0 - (len(result.infeasible_steps) / max(n, 1))

    # Condition number swept along the trajectory.
    cond = np.empty(n)
    for k in range(n):
        pose = Pose(
            position=result.positions[k],
            rotation=Rotation.from_quat(result.quaternions_xyzw[k]),
        )
        W = structure_matrix(pose, robot)
        s = np.linalg.svd(W, compute_uv=False)
        cond[k] = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    finite = cond[np.isfinite(cond)]

    return BenchmarkMetrics(
        n_samples=int(n),
        duration=float(result.time[-1] - result.time[0]) if n > 1 else 0.0,
        runtime_s=float(runtime_s),
        tracking_error_rms=float(np.sqrt(np.mean(err_p ** 2))),
        tracking_error_peak=float(err_p.max()),
        orientation_error_rms_deg=float(np.sqrt(np.mean(err_o_deg ** 2))),
        orientation_error_peak_deg=float(err_o_deg.max()),
        velocity_error_rms=float(np.sqrt(np.mean(err_v ** 2))),
        control_effort_rms=float(np.sqrt(np.mean(effort ** 2))),
        cable_tension_peak=float(tens.max()),
        cable_tension_min=float(tens.min()),
        feasibility_rate=float(feasibility_rate),
        condition_number_max=float(finite.max()) if finite.size else float("nan"),
        condition_number_median=float(np.median(finite)) if finite.size else float("nan"),
        cable_mode=cable_mode,
    )
