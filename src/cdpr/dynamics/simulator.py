r"""End-to-end CDPR forward simulation, in batch and streaming form.

Two entry points consume the same per-step logic:

* :func:`iter_simulation` is a *generator* yielding one :class:`StreamStep`
  per integration step (including the initial sample at :math:`t = 0`). Use
  it for live visualisation, hardware-in-the-loop loops, RL rollouts, or
  any context where the caller wants to interleave the integrator with
  other work.

* :func:`simulate` is the batch wrapper: it pre-allocates the result buffers
  and consumes :func:`iter_simulation` into a :class:`SimulationResult`.
  The two functions share their physics path, so any improvement (a new
  controller, a different integrator, an external-wrench disturbance)
  propagates to both immediately.

Closed-loop control is optional. When a ``controller=`` argument is
supplied, the controller computes the wrench it wants the cables to
deliver, and the tension solver finds the closest feasible tension vector;
the actual cable wrench applied to the platform is the structure matrix at
the *current* state times the chosen tensions --- *not* at the reference
pose. The earlier feedforward-at-reference variant was numerically
inconsistent (cable wrench computed at one pose, applied at another); we
keep the same default for the controller-less case (gravity compensation
only), but the structure matrix is now always evaluated at the integrated
state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterator, Literal

import numpy as np
from numpy.typing import NDArray

from cdpr.core.exceptions import InfeasibleTensionError, SingularConfiguration
from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.dynamics.integrators import (
    IntegratorStep,
    rk4_step,
    semi_implicit_step,
)
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.geometry.robot import Robot
from cdpr.kinematics.jacobian import structure_matrix
from cdpr.statics.tension import TensionObjective, tension_distribution

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.control.base import Controller
    from cdpr.trajectory.trajectory import Trajectory


# ---------------------------------------------------------------------------
# Per-step record
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StreamStep:
    """One sample from :func:`iter_simulation`.

    Captures everything an online consumer (live animator, RL agent, logger)
    typically needs: the integrator's current state, the cable tensions and
    lengths that produced it, and a flag telling whether the QP failed at
    this step.
    """

    step: int
    time: float
    state: PlatformState
    cable_tensions: NDArray[np.float64]
    cable_lengths: NDArray[np.float64]
    infeasible: bool

    @property
    def pose(self) -> Pose:
        return self.state.pose

    @property
    def velocity(self) -> Twist:
        return self.state.velocity


# ---------------------------------------------------------------------------
# Batch result
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SimulationResult:
    """Container for a complete forward-simulation run."""

    time: NDArray[np.float64]
    positions: NDArray[np.float64]            # (T, 3)
    quaternions_xyzw: NDArray[np.float64]     # (T, 4)
    linear_velocities: NDArray[np.float64]    # (T, 3)
    angular_velocities: NDArray[np.float64]   # (T, 3)
    cable_tensions: NDArray[np.float64]       # (T, m)
    cable_lengths: NDArray[np.float64]        # (T, m)
    infeasible_steps: list[int] = field(default_factory=list)
    # Phase 7: which constitutive law drove this run; ``None`` means the
    # default tension-distribution path was used. Populated by simulate()
    # when ``cable_model`` is supplied.
    cable_model_name: str | None = None
    cable_model_parameters: dict | None = None
    cable_diagnostics: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reference adapter
# ---------------------------------------------------------------------------

_PoseFn = Callable[[float], Pose]
_TwistFn = Callable[[float], Twist]
_AccelFn = Callable[[float], "tuple[NDArray[np.float64], NDArray[np.float64]]"]


def _bind_reference(
    reference: "Trajectory | _PoseFn | None",
) -> tuple[_PoseFn | None, _TwistFn | None, _AccelFn | None]:
    """Normalise the ``reference`` argument into three time-keyed callables.

    A :class:`Trajectory` gives pose / twist / acceleration; a bare callable
    is treated as pose-only (the controller layer falls back to zero twist
    and zero acceleration in that case); ``None`` yields all three as
    ``None``.
    """
    if reference is None:
        return None, None, None
    from cdpr.trajectory.trajectory import Trajectory
    if isinstance(reference, Trajectory):
        return reference.pose, reference.twist, reference.acceleration
    return reference, None, None


# ---------------------------------------------------------------------------
# Cable-length cache (avoids importing inverse.py just for an inline norm)
# ---------------------------------------------------------------------------

def _cable_lengths(state: PlatformState, robot: Robot) -> NDArray[np.float64]:
    b_world = state.pose.rotation.apply(robot.attachments) + state.pose.position
    return np.linalg.norm(robot.anchors - b_world, axis=-1)


# ---------------------------------------------------------------------------
# wrench_fn factory --- shared between batch and streaming paths
# ---------------------------------------------------------------------------

def _make_wrench_function(
    robot: Robot,
    *,
    controller: "Controller | None",
    reference_pose_fn: _PoseFn | None,
    reference_twist_fn: _TwistFn | None,
    reference_accel_fn: _AccelFn | None,
    external_wrench_fn: Callable[[PlatformState, float], Wrench] | None,
    gravity_vec: NDArray[np.float64],
    gravity_wrench: Wrench,
    tension_objective: TensionObjective | str,
    raise_on_infeasible: bool,
    dt: float,
    infeasible_recorder: list[int],
    tensions_holder: dict[str, NDArray[np.float64]],
):
    """Build the closure handed to the integrator.

    The closure also writes the last-computed tension vector into
    ``tensions_holder`` so the step recorder can read it without re-solving
    the QP.
    """
    limits = robot.require_limits()

    zero_lin = np.zeros(3)
    zero_ang = np.zeros(3)

    def desired_cable_wrench(state: PlatformState, t: float, external: Wrench) -> Wrench:
        if controller is None:
            # No controller --- cables compensate gravity and any external load.
            # Cable wrench = -gravity - external  (so that net = 0).
            return -gravity_wrench - external

        ref_pose = reference_pose_fn(t) if reference_pose_fn is not None else state.pose
        ref_twist = reference_twist_fn(t) if reference_twist_fn is not None else Twist.from_parts(zero_lin, zero_ang)
        ref_accel = reference_accel_fn(t) if reference_accel_fn is not None else None

        return controller(
            state=state,
            reference_pose=ref_pose,
            reference_twist=ref_twist,
            reference_accel=ref_accel,
            t=t,
            robot=robot,
            gravity=gravity_vec,
            external=external,
        )

    def wrench_fn(state: PlatformState, t: float) -> Wrench:
        external = (
            external_wrench_fn(state, t) if external_wrench_fn is not None
            else Wrench(np.zeros(6))
        )
        target_cable_wrench = desired_cable_wrench(state, t, external)

        W = structure_matrix(state.pose, robot)            # at CURRENT pose (correct)
        try:
            tau = tension_distribution(
                W,
                -target_cable_wrench.data[: robot.dof],     # tension_distribution solves W tau = -w_ext
                limits.t_min,
                limits.t_max,
                objective=tension_objective,
            )
        except (InfeasibleTensionError, SingularConfiguration):
            if raise_on_infeasible:
                raise
            infeasible_recorder.append(int(round(t / dt)))
            tau = np.zeros(robot.n_cables)

        tensions_holder["last"] = tau

        cable_wrench_vec = np.zeros(6)
        cable_wrench_vec[: robot.dof] = W @ tau
        return Wrench(cable_wrench_vec) + gravity_wrench + external

    return wrench_fn


def _make_constitutive_wrench_function(
    robot: Robot,
    *,
    cable_model,
    reference_pose_fn: _PoseFn | None,
    external_wrench_fn: Callable[[PlatformState, float], Wrench] | None,
    gravity_wrench: Wrench,
    initial_state: PlatformState,
    tensions_holder: dict[str, NDArray[np.float64]],
    diagnostics_holder: dict[str, dict],
):
    """Wrench function for the Phase-7 constitutive-law path.

    Rest lengths come from inverse kinematics on the *reference* pose at
    each step (open-loop tracking). When no reference is supplied, the
    rest lengths are frozen at the initial state's cable lengths --- the
    platform then holds whatever constitutive equilibrium it starts in.
    """
    from cdpr.kinematics.inverse import cable_lengths as _ik

    if reference_pose_fn is None:
        initial_rest = _ik(initial_state.pose, robot)

        def _rest_at(_t: float) -> NDArray[np.float64]:
            return initial_rest
    else:
        def _rest_at(t: float) -> NDArray[np.float64]:
            return _ik(reference_pose_fn(t), robot)

    def wrench_fn(state: PlatformState, t: float) -> Wrench:
        rest = _rest_at(t)
        cable_w = cable_model.platform_wrench(robot, state, rest)
        external = (
            external_wrench_fn(state, t) if external_wrench_fn is not None
            else Wrench(np.zeros(6))
        )

        tensions_holder["last"] = cable_model.tension(robot, state, rest)
        diagnostics_holder["last"] = cable_model.diagnostics(robot, state, rest)
        return cable_w + gravity_wrench + external

    return wrench_fn


# ---------------------------------------------------------------------------
# Streaming entry point
# ---------------------------------------------------------------------------

def iter_simulation(
    robot: Robot,
    state0: PlatformState,
    duration: float,
    dt: float,
    *,
    reference: "Trajectory | _PoseFn | None" = None,
    reference_pose: _PoseFn | None = None,                   # legacy alias for reference
    controller: "Controller | None" = None,
    external_wrench: Callable[[PlatformState, float], Wrench] | None = None,
    integrator: Literal["rk4", "semi_implicit"] = "rk4",
    tension_objective: TensionObjective | str = TensionObjective.CENTERED,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    raise_on_infeasible: bool = False,
    cable_model: "object | None" = None,
) -> Iterator[StreamStep]:
    r"""Generator yielding one :class:`StreamStep` per integration step.

    Parameters
    ----------
    cable_model:
        Optional :class:`cdpr.cables.CableModel`. When supplied, the
        simulator switches to the *constitutive* path: rest lengths come
        from inverse kinematics on the reference at each step, the
        constitutive law produces the cable wrench directly, and the
        Phase-1 tension-distribution QP is bypassed. The ``controller``
        argument is *not* consulted in this mode --- closed-loop control
        on a constitutive law operates at the rest-length level and
        belongs in a separate controller class. Default (``None``)
        preserves the Phase 1-6 tension-distribution behaviour.

    The first yield is the initial sample at :math:`t = 0`. Subsequent
    yields advance time by ``dt`` until ``duration`` is exhausted.

    Other parameters
    ----------------
    reference, reference_pose, controller, external_wrench, integrator,
    tension_objective, gravity, raise_on_infeasible:
        See the Phase 1-6 behaviour above. With ``cable_model=None`` they
        behave exactly as before.
    """
    inertia = robot.require_inertia()

    n_steps = int(np.ceil(duration / dt))
    g_vec = np.asarray(gravity, dtype=np.float64)
    f_gravity = inertia.mass * g_vec
    gravity_wrench = Wrench.from_parts(f_gravity, np.zeros(3))

    bound_ref = reference if reference is not None else reference_pose
    ref_pose_fn, ref_twist_fn, ref_accel_fn = _bind_reference(bound_ref)

    infeasible_steps: list[int] = []
    tensions_holder: dict[str, NDArray[np.float64]] = {"last": np.zeros(robot.n_cables)}
    diagnostics_holder: dict[str, dict] = {"last": {}}

    if cable_model is None:
        robot.require_limits()
        wrench_fn = _make_wrench_function(
            robot,
            controller=controller,
            reference_pose_fn=ref_pose_fn,
            reference_twist_fn=ref_twist_fn,
            reference_accel_fn=ref_accel_fn,
            external_wrench_fn=external_wrench,
            gravity_vec=g_vec,
            gravity_wrench=gravity_wrench,
            tension_objective=tension_objective,
            raise_on_infeasible=raise_on_infeasible,
            dt=dt,
            infeasible_recorder=infeasible_steps,
            tensions_holder=tensions_holder,
        )
    else:
        wrench_fn = _make_constitutive_wrench_function(
            robot,
            cable_model=cable_model,
            reference_pose_fn=ref_pose_fn,
            external_wrench_fn=external_wrench,
            gravity_wrench=gravity_wrench,
            initial_state=state0,
            tensions_holder=tensions_holder,
            diagnostics_holder=diagnostics_holder,
        )

    step_fn: IntegratorStep = rk4_step if integrator == "rk4" else semi_implicit_step

    # --- Initial sample -------------------------------------------------
    state = state0
    _ = wrench_fn(state, 0.0)                               # primes the holders
    yield StreamStep(
        step=0,
        time=0.0,
        state=state,
        cable_tensions=tensions_holder["last"].copy(),
        cable_lengths=_cable_lengths(state, robot),
        infeasible=0 in infeasible_steps,
    )

    # --- Integration loop ----------------------------------------------
    t = 0.0
    for k in range(n_steps):
        state = step_fn(state, t, dt, wrench_fn, inertia)
        t += dt
        yield StreamStep(
            step=k + 1,
            time=t,
            state=state,
            cable_tensions=tensions_holder["last"].copy(),
            cable_lengths=_cable_lengths(state, robot),
            infeasible=(k + 1) in infeasible_steps,
        )


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------

def simulate(
    robot: Robot,
    state0: PlatformState,
    duration: float,
    dt: float,
    *,
    reference: "Trajectory | _PoseFn | None" = None,
    reference_pose: _PoseFn | None = None,
    controller: "Controller | None" = None,
    external_wrench: Callable[[PlatformState, float], Wrench] | None = None,
    integrator: Literal["rk4", "semi_implicit"] = "rk4",
    tension_objective: TensionObjective | str = TensionObjective.CENTERED,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    raise_on_infeasible: bool = False,
    cable_model: "object | None" = None,
) -> SimulationResult:
    """Batch wrapper: consume :func:`iter_simulation` into a :class:`SimulationResult`.

    Parameter semantics match :func:`iter_simulation`. The optional
    ``cable_model`` selects one of the Phase-7 constitutive laws; when
    supplied, the result records the active mode and a per-step
    diagnostics trace.
    """
    from cdpr.kinematics.inverse import cable_lengths as _ik

    n_steps = int(np.ceil(duration / dt))
    time = np.zeros(n_steps + 1)
    positions = np.zeros((n_steps + 1, 3))
    quaternions = np.zeros((n_steps + 1, 4))
    lin_vel = np.zeros((n_steps + 1, 3))
    ang_vel = np.zeros((n_steps + 1, 3))
    tensions = np.zeros((n_steps + 1, robot.n_cables))
    lengths = np.zeros((n_steps + 1, robot.n_cables))
    infeasible_steps: list[int] = []
    cable_diagnostics: list[dict] = [] if cable_model is not None else []

    # Build per-step rest-length closure so the recorder can pull
    # cable_model diagnostics without re-running the constitutive solve.
    bound_ref = reference if reference is not None else reference_pose
    if cable_model is not None:
        ref_pose_fn, _, _ = _bind_reference(bound_ref)
        if ref_pose_fn is None:
            initial_rest = _ik(state0.pose, robot)
            def _rest(_t: float):
                return initial_rest
        else:
            def _rest(t: float):
                return _ik(ref_pose_fn(t), robot)

    for sample in iter_simulation(
        robot, state0, duration, dt,
        reference=reference,
        reference_pose=reference_pose,
        controller=controller,
        external_wrench=external_wrench,
        integrator=integrator,
        tension_objective=tension_objective,
        gravity=gravity,
        raise_on_infeasible=raise_on_infeasible,
        cable_model=cable_model,
    ):
        k = sample.step
        time[k] = sample.time
        positions[k] = sample.state.pose.position
        quaternions[k] = sample.state.pose.quaternion_xyzw
        lin_vel[k] = sample.state.velocity.linear
        ang_vel[k] = sample.state.velocity.angular
        tensions[k] = sample.cable_tensions
        lengths[k] = sample.cable_lengths
        if sample.infeasible:
            infeasible_steps.append(k)
        if cable_model is not None:
            cable_diagnostics.append(
                cable_model.diagnostics(robot, sample.state, _rest(sample.time))
            )

    return SimulationResult(
        time=time,
        positions=positions,
        quaternions_xyzw=quaternions,
        linear_velocities=lin_vel,
        angular_velocities=ang_vel,
        cable_tensions=tensions,
        cable_lengths=lengths,
        infeasible_steps=infeasible_steps,
        cable_model_name=getattr(cable_model, "mode_name", None),
        cable_model_parameters=(
            cable_model.parameters if cable_model is not None else None
        ),
        cable_diagnostics=cable_diagnostics,
    )
