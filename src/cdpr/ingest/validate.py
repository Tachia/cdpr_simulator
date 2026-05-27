r"""Validate an ingested experiment against the framework's dynamic model.

Three checks are exposed:

* **IK residual** (:func:`ik_residual`) -- pose history :math:`\to` predicted
  cable lengths; compare to the recorded lengths. Surfaces calibration
  errors and mocap-frame misalignment.
* **FK reconstruction** (:func:`reconstruct_trajectory`) -- cable-length
  history :math:`\to` pose history via Phase-1 forward kinematics. Used
  when the experiment only logged winch-side encoders.
* **Tension consistency** (:func:`tension_residual`) -- recorded tensions
  vs the platform wrench they would actually produce
  :math:`\mathbf{W}(\mathbf{q})\,\boldsymbol\tau`; large residuals indicate
  a mass / external-load mismatch.

The aggregated :class:`ValidationReport` packages all three for the
preprocessing report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose
from cdpr.ingest.containers import IngestedExperiment

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.geometry.robot import Robot


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ChannelResidual:
    """Per-step residual magnitude on one channel."""

    name: str
    rms: float
    peak: float
    trace: NDArray[np.float64]


@dataclass(slots=True)
class ValidationReport:
    """Aggregated validation diagnostics."""

    n_samples: int
    ik_length_residual: ChannelResidual | None = None
    tension_wrench_residual: ChannelResidual | None = None
    reconstructed_position_residual: ChannelResidual | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, dict[str, float] | None]:
        def fmt(ch: ChannelResidual | None) -> dict[str, float] | None:
            if ch is None:
                return None
            return {"rms": float(ch.rms), "peak": float(ch.peak)}
        return {
            "ik_length_residual": fmt(self.ik_length_residual),
            "tension_wrench_residual": fmt(self.tension_wrench_residual),
            "reconstructed_position_residual": fmt(self.reconstructed_position_residual),
        }


def _stats(name: str, per_step_error: NDArray[np.float64]) -> ChannelResidual:
    return ChannelResidual(
        name=name,
        rms=float(np.sqrt(np.mean(per_step_error ** 2))),
        peak=float(np.max(np.abs(per_step_error))),
        trace=per_step_error,
    )


def _poses_iter(experiment: IngestedExperiment):
    if experiment.positions is None:
        raise ValueError("Validation requires experimental position data.")
    quats = experiment.quaternions_xyzw
    for k in range(len(experiment.time)):
        rot = Rotation.from_quat(quats[k]) if quats is not None else Rotation.identity()
        yield Pose(position=experiment.positions[k], rotation=rot)


# ---------------------------------------------------------------------------
# Validation primitives
# ---------------------------------------------------------------------------

def ik_residual(experiment: IngestedExperiment, robot: "Robot") -> ChannelResidual:
    r"""Inverse-kinematics residual on recorded cable lengths.

    For each recorded pose, compute the model's predicted cable lengths
    and subtract from the recorded vector. The residual norm at step
    :math:`k` is :math:`\lVert \mathbf{L}_k^\text{rec} - \mathbf{L}_k^\text{IK} \rVert`;
    persistent non-zero values indicate that the cables' anchor / attachment
    geometry in the calibration file disagrees with the lab measurements.
    """
    from cdpr.kinematics.inverse import cable_lengths

    if experiment.cable_lengths is None:
        raise ValueError("ik_residual needs experimental cable_lengths.")
    n = len(experiment.time)
    err = np.empty(n)
    for k, pose in enumerate(_poses_iter(experiment)):
        L_model = cable_lengths(pose, robot)
        L_rec = experiment.cable_lengths[k]
        err[k] = float(np.linalg.norm(L_rec - L_model))
    return _stats("cable_length_residual [m]", err)


def tension_residual(
    experiment: IngestedExperiment,
    robot: "Robot",
    *,
    external_force: NDArray[np.float64] = np.array([0.0, 0.0, 0.0]),
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
) -> ChannelResidual:
    r"""Equilibrium residual :math:`\lVert \mathbf{W}(\mathbf{q})\boldsymbol\tau
    + \mathbf{w}_\text{ext} \rVert`.

    A platform in static equilibrium with the recorded tensions should
    satisfy the wrench-balance equation. Treating gravity (and an optional
    constant external force) as the load, residual norm at each step
    measures how far the cable model is from explaining the recorded
    tensions. Large peaks flag mass-calibration errors, payload changes,
    or controller transients.
    """
    from cdpr.kinematics.jacobian import structure_matrix

    if experiment.cable_tensions is None:
        raise ValueError("tension_residual needs experimental cable_tensions.")
    inertia = robot.require_inertia()
    g = np.asarray(gravity, dtype=np.float64)
    w_ext = np.zeros(6)
    w_ext[:3] = inertia.mass * g + np.asarray(external_force, dtype=np.float64)

    n = len(experiment.time)
    err = np.empty(n)
    for k, pose in enumerate(_poses_iter(experiment)):
        W = structure_matrix(pose, robot)
        tau = experiment.cable_tensions[k]
        residual_vec = W @ tau + w_ext[: robot.dof]
        err[k] = float(np.linalg.norm(residual_vec))
    return _stats("wrench_balance_residual [N or N·m]", err)


def reconstruct_trajectory(
    experiment: IngestedExperiment, robot: "Robot",
    *, seed: Pose | None = None,
) -> tuple[NDArray[np.float64], ChannelResidual | None]:
    """Recover pose trajectory from recorded cable lengths via Phase-1 FK.

    Returns ``(positions, residual)`` where ``positions`` is an ``(T, 3)``
    array of reconstructed positions. When the experiment already contains
    a position channel the residual against it is also computed; otherwise
    the residual return is ``None``.

    FK is locally unique but globally branched. The first sample is
    solved from ``seed`` (defaults to the centroid of the anchor set);
    each subsequent sample uses the previous solution as the warm start,
    so the resulting trajectory stays on a single FK branch.
    """
    from cdpr.kinematics.forward import forward_kinematics

    if experiment.cable_lengths is None:
        raise ValueError("reconstruct_trajectory needs experimental cable_lengths.")
    if seed is None:
        anchor_centroid = robot.anchors.mean(axis=0)
        seed = Pose(position=anchor_centroid * 0.0, rotation=Rotation.identity())

    n = len(experiment.time)
    positions = np.empty((n, 3))
    current = seed
    for k in range(n):
        L = experiment.cable_lengths[k]
        current = forward_kinematics(L, robot, initial_guess=current)
        positions[k] = current.position

    residual: ChannelResidual | None = None
    if experiment.positions is not None:
        err = np.linalg.norm(positions - experiment.positions, axis=1)
        residual = _stats("reconstruction_position_error [m]", err)

    return positions, residual


# ---------------------------------------------------------------------------
# Aggregated validation
# ---------------------------------------------------------------------------

def validate_against_robot(
    experiment: IngestedExperiment, robot: "Robot",
    *,
    external_force: NDArray[np.float64] | None = None,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    reconstruct: bool = False,
) -> ValidationReport:
    """Run every applicable validation check and bundle the results.

    Checks that require channels the experiment doesn't have are simply
    skipped (with a note in :attr:`ValidationReport.notes`). The
    ``reconstruct`` flag is off by default because the FK loop costs an
    LMA solve per sample.
    """
    report = ValidationReport(n_samples=int(len(experiment.time)))

    if experiment.positions is not None and experiment.cable_lengths is not None:
        report.ik_length_residual = ik_residual(experiment, robot)
    else:
        report.notes.append("IK residual skipped: missing positions and/or cable_lengths.")

    if experiment.positions is not None and experiment.cable_tensions is not None and robot.inertia is not None:
        report.tension_wrench_residual = tension_residual(
            experiment, robot,
            external_force=external_force if external_force is not None else np.zeros(3),
            gravity=gravity,
        )
    else:
        report.notes.append(
            "Tension residual skipped: missing positions, tensions, or robot inertia."
        )

    if reconstruct and experiment.cable_lengths is not None:
        _, recon_residual = reconstruct_trajectory(experiment, robot)
        report.reconstructed_position_residual = recon_residual
    return report
