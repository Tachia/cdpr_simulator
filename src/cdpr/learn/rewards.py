r"""Composable rewards for the CDPR Gymnasium environment.

Five primitive shapes are provided; combine them with :class:`RewardSum`
to build the policy objective you want. The standard recipe for a
tracking task is

.. math::

    r(s, a) \;=\; -\,w_p\,\lVert \mathbf{p}_\text{ref} - \mathbf{p} \rVert^2
        \;-\; w_v\,\lVert \mathbf{v}_\text{ref} - \mathbf{v} \rVert^2
        \;-\; w_a\,\lVert \mathbf{a} - \mathbf{a}_\text{prev} \rVert^2
        \;-\; w_T\,\lVert \boldsymbol\tau \rVert^2
        \;-\; w_x\,\mathbf{1}\!\left[\text{infeasible}\right],

and the building blocks below cover every term. Each reward records its
*last* unweighted value on the instance, which the environment surfaces
in the ``info`` dict so callers can plot the reward decomposition over
training without re-running the components.

Sign convention: every reward returns a value that the agent should
**maximise**. The primitives below are therefore non-positive (they're
distances or costs returned negated).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.core.frames import Pose, Twist


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Reward(Protocol):
    """A component that returns a scalar reward given the env context."""

    name: str
    weight: float
    last_value: float

    def __call__(
        self,
        *,
        pose: "Pose",
        velocity: "Twist",
        reference_pose: "Pose",
        reference_velocity: "Twist",
        action: NDArray[np.float64],
        previous_action: NDArray[np.float64] | None,
        tensions: NDArray[np.float64] | None,
        infeasible: bool,
    ) -> float: ...


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PoseTracking:
    r"""Negative squared Euclidean tracking error in position (and optionally orientation).

    The orientation term uses the rotation-vector log map of
    :math:`\mathbf{R}_\text{ref}\mathbf{R}^{\top}`, weighted independently
    from the translational term.
    """

    name: str = "pose_tracking"
    weight: float = 1.0
    orientation_weight: float = 0.1
    last_value: float = 0.0

    def __call__(self, *, pose, velocity, reference_pose, reference_velocity,
                 action, previous_action, tensions, infeasible):
        e_p = reference_pose.position - pose.position
        e_o = (reference_pose.rotation * pose.rotation.inv()).as_rotvec()
        val = -float(np.dot(e_p, e_p)) - self.orientation_weight * float(np.dot(e_o, e_o))
        self.last_value = val
        return val


@dataclass(slots=True)
class VelocityTracking:
    name: str = "velocity_tracking"
    weight: float = 0.1
    last_value: float = 0.0

    def __call__(self, *, pose, velocity, reference_pose, reference_velocity,
                 action, previous_action, tensions, infeasible):
        e_v = reference_velocity.linear - velocity.linear
        e_w = reference_velocity.angular - velocity.angular
        val = -float(np.dot(e_v, e_v) + np.dot(e_w, e_w))
        self.last_value = val
        return val


@dataclass(slots=True)
class ActionSmoothness:
    """Penalise large action increments (a Bode-style derivative penalty).

    Encourages controllers that don't slam the cables. Particularly useful
    for RL agents that otherwise discover bang-bang tension policies.
    """

    name: str = "action_smoothness"
    weight: float = 0.01
    last_value: float = 0.0

    def __call__(self, *, pose, velocity, reference_pose, reference_velocity,
                 action, previous_action, tensions, infeasible):
        if previous_action is None:
            self.last_value = 0.0
            return 0.0
        d = action - previous_action
        val = -float(np.dot(d, d))
        self.last_value = val
        return val


@dataclass(slots=True)
class TensionCost:
    r"""Penalise the L2 norm of the realised cable tension vector."""

    name: str = "tension_cost"
    weight: float = 1e-5
    last_value: float = 0.0

    def __call__(self, *, pose, velocity, reference_pose, reference_velocity,
                 action, previous_action, tensions, infeasible):
        if tensions is None:
            self.last_value = 0.0
            return 0.0
        val = -float(np.dot(tensions, tensions))
        self.last_value = val
        return val


@dataclass(slots=True)
class InfeasibilityPenalty:
    """Constant negative reward whenever the tension QP failed.

    Strong default weight because an infeasible solution means the cables
    cannot produce the commanded wrench --- the platform is effectively in
    free fall during that step. Making the agent learn to avoid this is
    cheaper than letting it discover that by reward shaping alone.
    """

    name: str = "infeasibility_penalty"
    weight: float = 10.0
    last_value: float = 0.0

    def __call__(self, *, pose, velocity, reference_pose, reference_velocity,
                 action, previous_action, tensions, infeasible):
        val = -1.0 if infeasible else 0.0
        self.last_value = val
        return val


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RewardSum:
    """Weighted sum of reward components with per-component bookkeeping."""

    components: list[Reward] = field(default_factory=list)
    last_total: float = 0.0

    def __call__(self, **kwargs) -> float:
        total = 0.0
        for c in self.components:
            total += c.weight * c(**kwargs)
        self.last_total = total
        return total

    def decomposition(self) -> dict[str, float]:
        """Per-component unweighted last value, plus weighted contributions."""
        out: dict[str, float] = {"total": self.last_total}
        for c in self.components:
            out[c.name] = c.last_value
            out[f"{c.name}_weighted"] = c.weight * c.last_value
        return out

    @classmethod
    def tracking_default(cls) -> "RewardSum":
        """Sensible starting reward for a pose-tracking RL task."""
        return cls(components=[
            PoseTracking(weight=1.0, orientation_weight=0.1),
            VelocityTracking(weight=0.05),
            ActionSmoothness(weight=0.001),
            TensionCost(weight=1e-7),
            InfeasibilityPenalty(weight=5.0),
        ])
