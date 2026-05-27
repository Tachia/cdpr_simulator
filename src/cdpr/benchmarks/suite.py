r"""Run scenarios across backends.

For the cdpr backend, :func:`run_scenario` delegates to
:func:`cdpr.dynamics.simulate` directly. For an external backend
(currently MuJoCo), the same controller and tension distribution run in
the cdpr namespace, but the wrench is applied to the backend and the
backend advances physics --- this is the closed-loop counterpart to
:func:`cdpr.adapters.verify_against` (which is open-loop wrench replay).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from numpy.typing import NDArray

from cdpr.benchmarks.metrics import BenchmarkMetrics, compute_metrics
from cdpr.benchmarks.scenario import Scenario, scenario_hash
from cdpr.core.exceptions import InfeasibleTensionError, SingularConfiguration
from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import SimulationResult, simulate
from cdpr.kinematics.jacobian import structure_matrix
from cdpr.statics.tension import TensionObjective, tension_distribution

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.adapters.base import PhysicsBackend


BackendKind = Literal["cdpr", "mujoco", "pybullet"]


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BenchmarkRun:
    """One (scenario, backend) result."""

    scenario_name: str
    scenario_hash: str
    backend: str
    metrics: BenchmarkMetrics
    result: SimulationResult                                # full time series for figures
    reference_positions: NDArray[np.float64]
    reference_quaternions: NDArray[np.float64]
    # The robot the scenario ran on. Stored so downstream consumers
    # (figure generation, identification on the resulting log) can pull it
    # without re-threading the scenario object.
    robot: "object | None" = None


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def _reference_callables(scenario: Scenario):
    """Return ``(pose_fn, twist_fn, accel_fn)`` from the scenario reference."""
    traj = scenario.trajectory
    if traj is None:
        zero_pose = scenario.initial_pose()
        return (
            lambda t: zero_pose,
            lambda t: Twist(np.zeros(6)),
            lambda t: (np.zeros(3), np.zeros(3)),
        )
    if hasattr(traj, "twist") and hasattr(traj, "acceleration"):
        return traj.pose, traj.twist, traj.acceleration
    # Pose-only callable: zero twist / accel.
    return (
        traj,
        lambda t: Twist(np.zeros(6)),
        lambda t: (np.zeros(3), np.zeros(3)),
    )


def _reference_series(scenario: Scenario, time_grid: NDArray[np.float64]):
    pose_fn, twist_fn, _ = _reference_callables(scenario)
    p = np.array([pose_fn(t).position for t in time_grid])
    q = np.array([pose_fn(t).quaternion_xyzw for t in time_grid])
    v = np.array([twist_fn(t).linear for t in time_grid])
    return p, q, v


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def _run_cdpr(scenario: Scenario) -> tuple[SimulationResult, float]:
    state0 = PlatformState.at_rest(scenario.initial_pose())
    start = time.perf_counter()
    result = simulate(
        robot=scenario.robot,
        state0=state0,
        duration=scenario.duration,
        dt=scenario.dt,
        reference=scenario.trajectory,
        controller=scenario.controller,
        cable_model=scenario.cable_model,
    )
    runtime = time.perf_counter() - start
    return result, runtime


def _run_external_backend(
    scenario: Scenario, backend_name: BackendKind,
) -> tuple[SimulationResult, float]:
    """Closed loop with cdpr's control law + tension distribution + backend
    physics integration."""
    from cdpr.adapters import make_backend, AdapterCapability

    robot = scenario.robot
    limits = robot.require_limits()
    inertia = robot.require_inertia()
    g_vec = np.array([0.0, 0.0, -9.81])
    gravity_wrench = Wrench.from_parts(inertia.mass * g_vec, np.zeros(3))

    pose_fn, twist_fn, accel_fn = _reference_callables(scenario)

    n_steps = int(np.ceil(scenario.duration / scenario.dt))
    times = np.zeros(n_steps + 1)
    positions = np.zeros((n_steps + 1, 3))
    quaternions = np.zeros((n_steps + 1, 4))
    lin_vel = np.zeros((n_steps + 1, 3))
    ang_vel = np.zeros((n_steps + 1, 3))
    tensions = np.zeros((n_steps + 1, robot.n_cables))
    lengths = np.zeros((n_steps + 1, robot.n_cables))
    infeasible: list[int] = []

    backend = make_backend(backend_name, robot=robot, timestep=scenario.dt)
    required = (
        AdapterCapability.SET_POSE | AdapterCapability.READ_STATE
        | AdapterCapability.STEP_PHYSICS | AdapterCapability.APPLY_WRENCH
    )
    if (backend.capabilities & required) != required:
        backend.close()
        raise ValueError(
            f"Backend {backend_name!r} lacks required capabilities for closed-loop run."
        )

    start = time.perf_counter()
    try:
        backend.set_pose(scenario.initial_pose())
        state = backend.read_state()

        def _record(idx: int, s: PlatformState, t: float, tau: NDArray[np.float64], bad: bool):
            times[idx] = t
            positions[idx] = s.pose.position
            quaternions[idx] = s.pose.quaternion_xyzw
            lin_vel[idx] = s.velocity.linear
            ang_vel[idx] = s.velocity.angular
            tensions[idx] = tau
            b_world = s.pose.rotation.apply(robot.attachments) + s.pose.position
            lengths[idx] = np.linalg.norm(robot.anchors - b_world, axis=-1)
            if bad:
                infeasible.append(idx)

        # Initial sample.
        _record(0, state, 0.0, np.zeros(robot.n_cables), False)

        t = 0.0
        for k in range(n_steps):
            ref_pose = pose_fn(t)
            ref_twist = twist_fn(t)
            try:
                ref_accel = accel_fn(t)
            except Exception:
                ref_accel = (np.zeros(3), np.zeros(3))

            # Controller -> desired cable wrench
            if scenario.controller is None:
                desired_cable_wrench = -gravity_wrench
            else:
                desired_cable_wrench = scenario.controller(
                    state=state,
                    reference_pose=ref_pose,
                    reference_twist=ref_twist,
                    reference_accel=ref_accel,
                    t=t,
                    robot=robot,
                    gravity=g_vec,
                    external=Wrench(np.zeros(6)),
                )

            W = structure_matrix(state.pose, robot)
            bad = False
            try:
                tau = tension_distribution(
                    W,
                    -desired_cable_wrench.data[: robot.dof],
                    limits.t_min, limits.t_max,
                    objective=TensionObjective.CENTERED,
                )
            except (InfeasibleTensionError, SingularConfiguration):
                tau = np.zeros(robot.n_cables)
                bad = True

            cable_wrench_vec = np.zeros(6)
            cable_wrench_vec[: robot.dof] = W @ tau
            backend.apply_wrench(Wrench(cable_wrench_vec))
            backend.step(scenario.dt)
            t += scenario.dt
            state = backend.read_state()
            _record(k + 1, state, t, tau, bad)
        runtime = time.perf_counter() - start
    finally:
        backend.close()

    result = SimulationResult(
        time=times, positions=positions, quaternions_xyzw=quaternions,
        linear_velocities=lin_vel, angular_velocities=ang_vel,
        cable_tensions=tensions, cable_lengths=lengths,
        infeasible_steps=infeasible,
    )
    return result, runtime


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_scenario(scenario: Scenario, backend: BackendKind = "cdpr") -> BenchmarkRun:
    """Run a scenario through the specified backend; return a full record."""
    np.random.seed(scenario.seed)
    if backend == "cdpr":
        result, runtime = _run_cdpr(scenario)
    else:
        if scenario.cable_model is not None:
            # Phase 7 directive: the constitutive law lives in the cdpr core.
            # External backends act as physics integrators / renderers only;
            # they cannot consume a Python-side CableModel callable, so
            # combining them with a cable_model would silently fall back to
            # the backend's own (different) cable physics. Refuse loudly.
            raise ValueError(
                f"Backend {backend!r} cannot honour scenario.cable_model "
                f"(mode {scenario.cable_model.mode_name!r}); only the "
                f"'cdpr' backend evaluates the constitutive law. Run the "
                f"backend comparison without a cable_model, or run the "
                f"constitutive law through 'cdpr' separately."
            )
        result, runtime = _run_external_backend(scenario, backend)

    ref_p, ref_q, ref_v = _reference_series(scenario, result.time)
    mode = (
        scenario.cable_model.mode_name
        if scenario.cable_model is not None else "none"
    )
    metrics = compute_metrics(
        result, ref_p, ref_q, ref_v, runtime, scenario.robot,
        cable_mode=mode,
    )
    return BenchmarkRun(
        scenario_name=scenario.name,
        scenario_hash=scenario_hash(scenario),
        backend=backend,
        metrics=metrics,
        result=result,
        reference_positions=ref_p,
        reference_quaternions=ref_q,
        robot=scenario.robot,
    )


@dataclass(slots=True)
class BenchmarkSuite:
    """Run multiple scenarios across multiple backends."""

    scenarios: list[Scenario] = field(default_factory=list)
    backends: list[BackendKind] = field(default_factory=lambda: ["cdpr"])

    def run(self) -> list[BenchmarkRun]:
        runs: list[BenchmarkRun] = []
        for scenario in self.scenarios:
            for backend in self.backends:
                runs.append(run_scenario(scenario, backend))
        return runs
