r"""Gymnasium environment around the CDPR scientific core.

:class:`CDPREnv` exposes the standard ``reset / step / render / close``
contract. One physical step takes ``dt`` seconds; the integrator is the
Phase-1 :func:`cdpr.dynamics.integrators.rk4_step`, the wrench-from-action
mapping is either direct cable tensions or a desired wrench solved
through tension distribution. Either way nothing about the physics is
re-implemented here --- the env is a thin adapter on top of Phase 1.

Observation space (default, ``observe_absolute=False``):
* position error (3)
* orientation error in :math:`SO(3)` log map (3)
* linear velocity error (3)
* angular velocity error (3)
* previous action

Set ``observe_absolute=True`` to additionally include the absolute pose
and reference pose, which is what most published CDPR-RL papers feed
their policies. Either observation is a flat 1-D vector to make Stable-
Baselines3 happy without custom feature extractors.

Action space:
* ``"wrench"`` -- 6 components normalised to ``[-1, 1]`` and scaled by
  ``wrench_scale`` (per-axis), then realised as cable tensions via
  :func:`tension_distribution` at the current pose. The cable wrench
  actually applied to the platform is :math:`\mathbf{W}(\mathbf{q})\boldsymbol\tau`,
  *not* the commanded wrench --- whenever the QP hits a bound, the
  difference shows up as tracking error and is the agent's problem.
* ``"tension"`` -- ``n_cables`` components normalised to ``[-1, 1]`` and
  mapped affinely onto ``[t_min, t_max]`` per cable.

Termination / truncation:
* The episode terminates if the position error exceeds ``max_error_m``
  (the platform has left the controllable region).
* It truncates after ``horizon`` integration steps.

Determinism: ``reset(seed=...)`` seeds NumPy's per-instance generator;
the integrator itself has no internal randomness, so the trajectory is
fully reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from cdpr.core.exceptions import InfeasibleTensionError, SingularConfiguration
from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.dynamics.integrators import rk4_step, semi_implicit_step
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.kinematics.jacobian import structure_matrix
from cdpr.learn._lazy import require_gymnasium
from cdpr.learn.rewards import RewardSum
from cdpr.statics.tension import TensionObjective, tension_distribution

# Resolve Gymnasium at module import. This module lives in the learning
# extra; importing it without Gymnasium installed is a programming error.
require_gymnasium()
import gymnasium as gym                                      # noqa: E402

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.geometry.robot import Robot


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CDPREnvConfig:
    """Static configuration of a :class:`CDPREnv`."""

    dt: float = 5e-3
    horizon: int = 400
    action_mode: Literal["wrench", "tension"] = "wrench"
    wrench_scale: NDArray[np.float64] | None = None    # default: per-dof reasonable scale
    integrator: Literal["rk4", "semi_implicit"] = "rk4"
    tension_objective: TensionObjective | str = TensionObjective.CENTERED
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    max_error_m: float = 1.0
    observe_absolute: bool = False


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------

class CDPREnv(gym.Env):
    """Gymnasium environment around the CDPR forward simulator.

    Subclasses :class:`gymnasium.Env` so that Stable-Baselines3 and other
    consumers recognise it natively. Importing this module imports
    Gymnasium; the lazy-import guard in :mod:`cdpr.learn._lazy` raises a
    helpful install hint if it is missing.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        robot: "Robot",
        reference_factory: Callable[[int | None], Callable[[float], Pose]] | None = None,
        *,
        config: CDPREnvConfig | None = None,
        reward: RewardSum | None = None,
        initial_pose: Pose | None = None,
    ) -> None:
        super().__init__()
        self.spaces = gym.spaces

        self.robot = robot
        self.cfg = config or CDPREnvConfig()
        self.reward = reward or RewardSum.tracking_default()
        self.inertia = robot.require_inertia()
        self.limits = robot.require_limits()

        self._reference_factory = reference_factory or _default_reference_factory
        self._reference: Callable[[float], Pose] | None = None
        self._initial_pose = initial_pose or Pose(position=np.zeros(3), rotation=Rotation.identity())

        # Wrench scale per dof: position [N] axes get 200 N, torque axes 20 N·m by default.
        if self.cfg.wrench_scale is None:
            self.cfg.wrench_scale = np.array(
                [200.0, 200.0, 200.0, 20.0, 20.0, 20.0][: robot.dof]
            )

        self.action_space, self.observation_space = self._build_spaces()

        self._state: PlatformState | None = None
        self._step_count = 0
        self._t = 0.0
        self._previous_action: NDArray[np.float64] | None = None
        self._np_random = np.random.default_rng()

        self._integrator = rk4_step if self.cfg.integrator == "rk4" else semi_implicit_step
        self._gravity_vec = np.asarray(self.cfg.gravity, dtype=np.float64)
        self._gravity_wrench = Wrench.from_parts(self.inertia.mass * self._gravity_vec, np.zeros(3))

    # --- space construction ----------------------------------------------

    def _build_spaces(self):
        # Action
        if self.cfg.action_mode == "wrench":
            n_action = self.robot.dof
        else:
            n_action = self.robot.n_cables
        action_space = self.spaces.Box(
            low=-1.0, high=1.0, shape=(n_action,), dtype=np.float32,
        )

        # Observation: errors + previous action, optionally absolute pose/ref
        base = 3 + 3 + 3 + 3                  # err_p, err_o, err_v, err_w
        absolute = 7 + 7 if self.cfg.observe_absolute else 0  # pose + ref pose (pos+quat)
        n_obs = base + n_action + absolute
        observation_space = self.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32,
        )
        return action_space, observation_space

    # --- standard API -----------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        self._reference = self._reference_factory(seed)
        self._t = 0.0
        self._step_count = 0
        self._previous_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self._state = PlatformState.at_rest(self._initial_pose)
        return self._observation(), {"reference_pose": self._reference(0.0)}

    def step(self, action: NDArray[np.float32]):
        assert self._state is not None and self._reference is not None, "Call reset() first."

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Build wrench_fn closure for this step.
        wrench_fn, diag_holder = self._make_wrench_fn(action)

        # Integrate one step.
        new_state = self._integrator(
            self._state, self._t, self.cfg.dt, wrench_fn, self.inertia,
        )
        t_next = self._t + self.cfg.dt

        # Reward at the new state.
        ref_pose = self._reference(t_next)
        ref_twist = _finite_difference_twist(self._reference, t_next, self.cfg.dt)
        r = self.reward(
            pose=new_state.pose,
            velocity=new_state.velocity,
            reference_pose=ref_pose,
            reference_velocity=ref_twist,
            action=action.astype(np.float64),
            previous_action=self._previous_action.astype(np.float64),
            tensions=diag_holder["tensions"],
            infeasible=diag_holder["infeasible"],
        )

        # Step bookkeeping.
        self._state = new_state
        self._t = t_next
        self._step_count += 1
        self._previous_action = action.copy()

        # Termination conditions.
        err = float(np.linalg.norm(new_state.pose.position - ref_pose.position))
        terminated = bool(err > self.cfg.max_error_m)
        truncated = bool(self._step_count >= self.cfg.horizon)

        info: dict[str, object] = {
            "tracking_error": err,
            "tensions": diag_holder["tensions"],
            "infeasible": diag_holder["infeasible"],
            "reward_decomposition": self.reward.decomposition(),
        }
        return self._observation(), float(r), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass

    # --- helpers ----------------------------------------------------------

    def _observation(self) -> NDArray[np.float32]:
        assert self._state is not None and self._reference is not None
        ref_pose = self._reference(self._t)
        ref_twist = _finite_difference_twist(self._reference, self._t, self.cfg.dt)
        err_p = ref_pose.position - self._state.pose.position
        err_o = (ref_pose.rotation * self._state.pose.rotation.inv()).as_rotvec()
        err_v = ref_twist.linear - self._state.velocity.linear
        err_w = ref_twist.angular - self._state.velocity.angular

        parts = [err_p, err_o, err_v, err_w, self._previous_action]
        if self.cfg.observe_absolute:
            parts.append(self._state.pose.position)
            parts.append(self._state.pose.quaternion_xyzw)
            parts.append(ref_pose.position)
            parts.append(ref_pose.quaternion_xyzw)
        return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])

    def _make_wrench_fn(self, action: NDArray[np.float32]):
        diag = {"tensions": np.zeros(self.robot.n_cables), "infeasible": False}

        if self.cfg.action_mode == "tension":
            t_min = self.limits.t_min
            t_max = self.limits.t_max
            # Map [-1, 1] -> [t_min, t_max] per cable.
            tau = 0.5 * (action.astype(np.float64) + 1.0) * (t_max - t_min) + t_min
            tau = np.clip(tau, t_min, t_max)
            diag["tensions"] = tau

            def wrench_fn(state: PlatformState, _t: float) -> Wrench:
                W = structure_matrix(state.pose, self.robot)
                cable_wrench_vec = np.zeros(6)
                cable_wrench_vec[: self.robot.dof] = W @ tau
                return Wrench(cable_wrench_vec) + self._gravity_wrench
        else:
            scale = self.cfg.wrench_scale
            desired = action.astype(np.float64) * scale     # cable wrench the agent wants

            def wrench_fn(state: PlatformState, _t: float) -> Wrench:
                W = structure_matrix(state.pose, self.robot)
                try:
                    tau = tension_distribution(
                        W, -desired, self.limits.t_min, self.limits.t_max,
                        objective=self.cfg.tension_objective,
                    )
                    diag["infeasible"] = False
                except (InfeasibleTensionError, SingularConfiguration):
                    tau = np.zeros(self.robot.n_cables)
                    diag["infeasible"] = True
                diag["tensions"] = tau
                cable_wrench_vec = np.zeros(6)
                cable_wrench_vec[: self.robot.dof] = W @ tau
                return Wrench(cable_wrench_vec) + self._gravity_wrench

        return wrench_fn, diag


# ---------------------------------------------------------------------------
# Reference utilities
# ---------------------------------------------------------------------------

def _default_reference_factory(seed: int | None) -> Callable[[float], Pose]:
    """Trivial default: hold at the origin with identity orientation."""
    pose = Pose(position=np.zeros(3), rotation=Rotation.identity())
    return lambda t: pose


def _finite_difference_twist(
    reference: Callable[[float], Pose], t: float, dt: float,
) -> Twist:
    """Numerical twist of a reference callable; cheap and avoids forcing the
    caller to provide an analytic Trajectory."""
    t_back = max(0.0, t - dt)
    p1 = reference(t)
    p0 = reference(t_back)
    if t == t_back:
        return Twist(np.zeros(6))
    lin = (p1.position - p0.position) / dt
    ang = ((p1.rotation * p0.rotation.inv()).as_rotvec()) / dt
    return Twist.from_parts(lin, ang)
