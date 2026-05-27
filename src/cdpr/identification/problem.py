r"""Identification problem assembly and solver.

Given a sequence of recorded ``(pose, cable_lengths)`` samples and the
nominal :class:`Robot`, build the residual function

.. math::

    r_{k,i}(\boldsymbol\phi) \;=\;
        L_{k,i}^\text{recorded}
        \;-\; \bigl\lVert (\mathbf{a}_i + \Delta\mathbf{a}_i)
                    - \mathbf{p}_k
                    - \mathbf{R}_k\,(\mathbf{b}_i + \Delta\mathbf{b}_i)
              \bigr\rVert
        \;-\; \Delta L_i,

and solve the bounded nonlinear least-squares system

.. math::

    \min_{\boldsymbol\phi \in [\boldsymbol\phi_\text{lo}, \boldsymbol\phi_\text{hi}]}
        \sum_{k, i} r_{k,i}(\boldsymbol\phi)^2

via :func:`scipy.optimize.least_squares` (trust-region reflective, the
standard choice for problems with box constraints).

The :func:`apply_result` helper produces a new :class:`Robot` with the
fitted perturbations baked in, ready for downstream simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from cdpr.identification.parameters import IdentifiableParameters

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.geometry.robot import Robot


# ---------------------------------------------------------------------------
# Problem / Result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IdentificationProblem:
    """Data + parametrisation handed to :func:`identify`."""

    robot: "Robot"
    parameters: IdentifiableParameters
    positions: NDArray[np.float64]              # (T, 3)
    quaternions_xyzw: NDArray[np.float64]       # (T, 4)
    measured_lengths: NDArray[np.float64]       # (T, m)
    sample_weights: NDArray[np.float64] | None = None  # (T,) optional per-sample weight

    def __post_init__(self) -> None:
        if self.positions.shape[0] != self.quaternions_xyzw.shape[0]:
            raise ValueError("positions / quaternions length mismatch")
        if self.positions.shape[0] != self.measured_lengths.shape[0]:
            raise ValueError("positions / measured_lengths length mismatch")
        m = self.robot.n_cables
        if self.measured_lengths.shape[1] != m:
            raise ValueError(
                f"measured_lengths has {self.measured_lengths.shape[1]} cables; "
                f"robot has {m}"
            )
        if self.parameters.n_cables != m:
            raise ValueError(
                f"IdentifiableParameters claims {self.parameters.n_cables} cables; "
                f"robot has {m}"
            )


@dataclass(slots=True)
class IdentificationResult:
    """Output of :func:`identify`."""

    fitted_vector: NDArray[np.float64]
    initial_residual_rms: float
    final_residual_rms: float
    initial_residual_peak: float
    final_residual_peak: float
    n_iterations: int
    converged: bool
    message: str
    cost_history: list[float] = field(default_factory=list)

    def parameter_changes(self, params: IdentifiableParameters) -> dict[str, NDArray[np.float64]]:
        """Per-group view of the fitted perturbation."""
        return {
            "anchor_offsets":      params.anchor_offsets(self.fitted_vector),
            "attachment_offsets":  params.attachment_offsets(self.fitted_vector),
            "cable_length_offsets": params.cable_length_offsets(self.fitted_vector),
        }


# ---------------------------------------------------------------------------
# Residual function
# ---------------------------------------------------------------------------

def _residuals(x: NDArray[np.float64], problem: IdentificationProblem) -> NDArray[np.float64]:
    p = problem.parameters
    robot = problem.robot

    da = p.anchor_offsets(x)                                 # (m, 3)
    db = p.attachment_offsets(x)                             # (m, 3)
    dL = p.cable_length_offsets(x)                           # (m,)

    anchors = robot.anchors + da                             # (m, 3)
    attachments_body = robot.attachments + db                # (m, 3)

    T = problem.positions.shape[0]
    m = robot.n_cables
    res = np.empty((T, m), dtype=np.float64)

    for k in range(T):
        R = Rotation.from_quat(problem.quaternions_xyzw[k]).as_matrix()
        b_world = (R @ attachments_body.T).T + problem.positions[k]
        L_model = np.linalg.norm(anchors - b_world, axis=-1) + dL
        res[k] = problem.measured_lengths[k] - L_model

    if problem.sample_weights is not None:
        res = res * np.sqrt(problem.sample_weights)[:, None]

    return res.ravel()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def identify(
    problem: IdentificationProblem,
    *,
    xtol: float = 1e-10,
    ftol: float = 1e-10,
    max_iter: int = 200,
) -> IdentificationResult:
    """Solve the identification problem and return diagnostics."""
    p = problem.parameters
    x0 = p.initial_vector()
    lo, hi = p.bounds_vectors()

    # Initial residual for the "before fit" comparison.
    r0 = _residuals(x0, problem)
    initial_rms = float(np.sqrt(np.mean(r0 ** 2)))
    initial_peak = float(np.max(np.abs(r0)))

    cost_log: list[float] = []

    def fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
        r = _residuals(x, problem)
        cost_log.append(float(0.5 * np.dot(r, r)))
        return r

    result = least_squares(
        fn, x0,
        bounds=(lo, hi),
        method="trf",
        xtol=xtol, ftol=ftol,
        max_nfev=max_iter * (len(x0) + 1),
    )

    r_final = _residuals(result.x, problem)
    final_rms = float(np.sqrt(np.mean(r_final ** 2)))
    final_peak = float(np.max(np.abs(r_final)))

    return IdentificationResult(
        fitted_vector=np.asarray(result.x, dtype=np.float64),
        initial_residual_rms=initial_rms,
        final_residual_rms=final_rms,
        initial_residual_peak=initial_peak,
        final_residual_peak=final_peak,
        n_iterations=int(result.nfev),
        converged=bool(result.success),
        message=str(result.message),
        cost_history=cost_log,
    )


# ---------------------------------------------------------------------------
# Apply the fitted parameters to a Robot
# ---------------------------------------------------------------------------

def apply_result(
    problem: IdentificationProblem,
    result: IdentificationResult,
) -> "Robot":
    """Return a new :class:`Robot` with the identified perturbations applied."""
    from cdpr.geometry.robot import Robot, RobotGeometry

    p = problem.parameters
    da = p.anchor_offsets(result.fitted_vector)
    db = p.attachment_offsets(result.fitted_vector)
    new_anchors = problem.robot.anchors + da
    new_attachments = problem.robot.attachments + db
    new_geom = RobotGeometry(
        anchors=new_anchors,
        attachments=new_attachments,
        dof=problem.robot.dof,
        name=f"{problem.robot.name}_identified",
    )
    return Robot(
        geometry=new_geom,
        inertia=problem.robot.inertia,
        limits=problem.robot.limits,
        cable_properties=problem.robot.cable_properties,
    )
