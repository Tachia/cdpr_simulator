r"""Benchmark harness: run several controllers on the same task and compare.

The intended use case is the dissertation result section "How does our
learned controller compare to the analytic baselines?" Build a
:class:`Benchmark` with a robot, a reference trajectory, and a dict of
named controllers; call :meth:`run`; receive a :class:`BenchmarkReport`
with per-controller tracking metrics and (optionally) a side-by-side
position-error plot.

Controllers may be:
* analytic (``PDController``, ``ComputedTorqueController``),
* open-loop (``None`` --- gravity compensation only, the no-tracking
  baseline),
* learned (any callable returning a desired cable wrench --- e.g. wrap
  an :class:`InverseDynamicsPINN` with :class:`LearnedInverseDynamics`).

Sim-to-data comparison: pass an :class:`IngestedExperiment` as a
"reference" controller and the harness will treat it as an
observational baseline rather than running a simulation, so the same
metrics can be computed against real lab data alongside the simulated
controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from numpy.typing import NDArray

from cdpr.core.frames import Pose, Wrench
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import SimulationResult, simulate

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.control.base import Controller
    from cdpr.geometry.robot import Robot
    from cdpr.ingest.containers import IngestedExperiment
    from cdpr.learn.datasets import Normalizer
    from cdpr.trajectory.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ControllerOutcome:
    """Aggregate per-controller metrics from one benchmark run."""

    name: str
    mean_position_error: float
    peak_position_error: float
    rms_tension: float
    peak_tension: float
    infeasible_steps: int
    result: SimulationResult | None = None


@dataclass(slots=True)
class BenchmarkReport:
    outcomes: list[ControllerOutcome] = field(default_factory=list)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "name": o.name,
                "mean_position_error": o.mean_position_error,
                "peak_position_error": o.peak_position_error,
                "rms_tension": o.rms_tension,
                "peak_tension": o.peak_tension,
                "infeasible_steps": o.infeasible_steps,
            }
            for o in self.outcomes
        ]

    def best_by_mean_error(self) -> ControllerOutcome:
        return min(self.outcomes, key=lambda o: o.mean_position_error)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Benchmark:
    robot: "Robot"
    trajectory: "Trajectory"
    duration: float
    dt: float = 2e-3
    initial_pose_at_t0: bool = True

    def run(
        self,
        controllers: dict[str, "Controller | None"],
        *,
        keep_results: bool = False,
    ) -> BenchmarkReport:
        """Run every controller and collect tracking / tension metrics.

        Setting ``keep_results=True`` retains the underlying
        :class:`SimulationResult` on each :class:`ControllerOutcome` so
        callers can plot or further analyse the trajectories. Off by
        default to keep the report lightweight.
        """
        initial_pose = self.trajectory.pose(0.0) if self.initial_pose_at_t0 else Pose(
            position=np.zeros(3), rotation=self.trajectory.pose(0.0).rotation,
        )
        state0 = PlatformState.at_rest(initial_pose)

        outcomes: list[ControllerOutcome] = []
        for name, ctrl in controllers.items():
            result = simulate(
                robot=self.robot, state0=state0,
                duration=self.duration, dt=self.dt,
                reference=self.trajectory,
                controller=ctrl,
            )
            ref_positions = np.array([self.trajectory(t).position for t in result.time])
            err = np.linalg.norm(result.positions - ref_positions, axis=1)
            outcomes.append(ControllerOutcome(
                name=name,
                mean_position_error=float(err.mean()),
                peak_position_error=float(err.max()),
                rms_tension=float(np.sqrt(np.mean(result.cable_tensions ** 2))),
                peak_tension=float(result.cable_tensions.max()),
                infeasible_steps=len(result.infeasible_steps),
                result=result if keep_results else None,
            ))
        return BenchmarkReport(outcomes=outcomes)


# ---------------------------------------------------------------------------
# Learned-controller adapter
# ---------------------------------------------------------------------------

class LearnedInverseDynamics:
    """Wrap a trained inverse-dynamics model as a :class:`Controller`.

    The model maps ``(p, q, v, omega, a_des_lin, a_des_ang)`` to a cable
    tension vector :math:`\\hat{\\boldsymbol\\tau}`; the controller then
    realises that as a cable wrench :math:`\\mathbf{W}(\\mathbf{q})\\hat{\\boldsymbol\\tau}`
    at the *current* pose. Designed to slot into
    :func:`cdpr.dynamics.simulator.simulate` as the ``controller=`` argument.

    Normalisation: most training pipelines z-score both inputs and
    outputs. Pass the input normaliser as ``normalizer`` and the
    *target* normaliser as ``target_normalizer`` so the wrapper applies
    the correct inverse-transform before computing the cable wrench ---
    without this the model's predictions live in normalised units and
    produce nonsense wrenches.

    A PD term may be added on top via ``feedback_kp`` / ``feedback_kd`` ---
    the standard feedforward-plus-feedback recipe for using a learned
    inverse-dynamics model in closed loop.
    """

    def __init__(
        self,
        model,
        *,
        normalizer: "Normalizer | None" = None,
        target_normalizer: "Normalizer | None" = None,
        feedback_kp: float | NDArray[np.float64] = 0.0,
        feedback_kd: float | NDArray[np.float64] = 0.0,
    ) -> None:
        from cdpr.control.base import as_gain_matrix
        self.model = model
        self.normalizer = normalizer
        self.target_normalizer = target_normalizer
        self.Kp = as_gain_matrix(feedback_kp)
        self.Kd = as_gain_matrix(feedback_kd)

    def __call__(
        self,
        *,
        state: PlatformState,
        reference_pose: Pose,
        reference_twist,
        reference_accel,
        t: float,
        robot,
        gravity: NDArray[np.float64],
        external: Wrench,
    ) -> Wrench:
        from cdpr.kinematics.jacobian import structure_matrix
        from cdpr.learn._lazy import require_torch
        torch = require_torch()

        # Assemble the feature vector in the canonical TrajectoryDataset layout.
        p = state.pose.position
        q = state.pose.quaternion_xyzw
        v = state.velocity.linear
        w = state.velocity.angular
        a_lin = reference_accel[0] if reference_accel is not None else np.zeros(3)
        a_ang = reference_accel[1] if reference_accel is not None else np.zeros(3)
        x_np = np.concatenate([p, q, v, w, a_lin, a_ang]).reshape(1, -1)
        if self.normalizer is not None:
            x_np = self.normalizer.transform(x_np)

        with torch.no_grad():
            x_t = torch.as_tensor(x_np, dtype=torch.float32)
            tau_hat = self.model(x_t).cpu().numpy().reshape(-1)

        # Denormalise the prediction if the model was trained on z-scored targets.
        if self.target_normalizer is not None:
            tau_hat = self.target_normalizer.inverse_transform(tau_hat[None, :])[0]

        # Realise that tension as a platform wrench at the current pose.
        W = structure_matrix(state.pose, robot)
        cable_wrench_vec = np.zeros(6)
        cable_wrench_vec[: robot.dof] = W @ tau_hat

        # Optional PD feedback on the pose error (additive cable wrench).
        e_p = reference_pose.position - p
        e_v = reference_twist.linear - v
        cable_wrench_vec[:3] = cable_wrench_vec[:3] + self.Kp @ e_p + self.Kd @ e_v

        return Wrench(cable_wrench_vec)
