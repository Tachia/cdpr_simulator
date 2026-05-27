r"""Cross-engine physics verification.

Drive both the cdpr scientific core and a :class:`PhysicsBackend` adapter
with the *same wrench history* and report the divergence of the two
state trajectories. This is the canonical "do two different integrators
agree?" check that gives confidence the cdpr core's Newton-Euler RK4 is
not silently fighting the engineering convention of an external engine.

The workflow:

1. Run the cdpr simulator with whatever controller / trajectory the
   experiment uses, recording (a) the platform state at every step and
   (b) the *total* wrench actually applied to the platform at every step
   (cable wrench from tension distribution plus gravity plus any
   external disturbance).
2. Reset the backend to the same initial state.
3. For each recorded step, hand the backend the same wrench and ask it
   to advance physics by ``dt``.
4. Diff the resulting state series against the cdpr reference.

Position, orientation, and velocity divergences are reported separately
so calibration mismatches (orientation drift due to a quaternion-order
bug, say) are distinguishable from genuine integrator differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose, Twist, Wrench
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import iter_simulation
from cdpr.kinematics.jacobian import structure_matrix
from cdpr.statics.tension import TensionObjective, tension_distribution

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.adapters.base import PhysicsBackend
    from cdpr.control.base import Controller
    from cdpr.geometry.robot import Robot


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class VerificationReport:
    """Pairwise divergence between cdpr core and a backend, per channel."""

    time: NDArray[np.float64]
    cdpr_positions: NDArray[np.float64]
    backend_positions: NDArray[np.float64]
    cdpr_quaternions: NDArray[np.float64]
    backend_quaternions: NDArray[np.float64]
    cdpr_velocities: NDArray[np.float64]
    backend_velocities: NDArray[np.float64]

    @property
    def position_error(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.cdpr_positions - self.backend_positions, axis=1)

    @property
    def orientation_error_deg(self) -> NDArray[np.float64]:
        dot = np.abs(np.sum(self.cdpr_quaternions * self.backend_quaternions, axis=1))
        dot = np.clip(dot, -1.0, 1.0)
        return np.rad2deg(2.0 * np.arccos(dot))

    @property
    def velocity_error(self) -> NDArray[np.float64]:
        return np.linalg.norm(self.cdpr_velocities - self.backend_velocities, axis=1)

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            "position_m":     _stat(self.position_error),
            "orientation_deg": _stat(self.orientation_error_deg),
            "velocity_m_per_s": _stat(self.velocity_error),
        }


def _stat(arr: NDArray[np.float64]) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(arr ** 2))),
        "peak": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


# ---------------------------------------------------------------------------
# Verification routine
# ---------------------------------------------------------------------------

def verify_against(
    backend: "PhysicsBackend",
    robot: "Robot",
    initial_state: PlatformState,
    duration: float,
    dt: float,
    *,
    reference=None,
    controller: "Controller | None" = None,
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    tension_objective: TensionObjective | str = TensionObjective.CENTERED,
) -> VerificationReport:
    r"""Run the same scenario through cdpr and a backend; report divergence.

    The cdpr simulator runs first, producing a wrench history. The backend
    is then driven through the same wrench history --- this isolates the
    integration divergence from any difference in *what* force is applied.

    Both runs use identical ``dt`` and start from ``initial_state``. The
    backend must declare ``AdapterCapability.STEP_PHYSICS``,
    ``APPLY_WRENCH``, ``SET_POSE``, and ``READ_STATE`` (the verification
    routine asserts this up front).
    """
    from cdpr.adapters.base import AdapterCapability

    required = (
        AdapterCapability.SET_POSE
        | AdapterCapability.READ_STATE
        | AdapterCapability.STEP_PHYSICS
        | AdapterCapability.APPLY_WRENCH
    )
    if (backend.capabilities & required) != required:
        missing = (required & ~backend.capabilities).name
        raise ValueError(
            f"Backend {backend.name!r} lacks required capabilities: {missing}"
        )

    # We record only the *cable* wrench, not the cable+gravity total. Both
    # the cdpr simulator and the backend apply gravity themselves through
    # their own configured gravity vector; injecting gravity via
    # apply_wrench on top of that would double-count it.

    cdpr_states: list[PlatformState] = []
    cable_wrenches: list[Wrench] = []

    for sample in iter_simulation(
        robot, initial_state, duration, dt,
        reference=reference, controller=controller,
        gravity=gravity, tension_objective=tension_objective,
    ):
        cdpr_states.append(sample.state)
        W = structure_matrix(sample.state.pose, robot)
        cable_wrench_vec = np.zeros(6)
        cable_wrench_vec[: robot.dof] = W @ sample.cable_tensions
        cable_wrenches.append(Wrench(cable_wrench_vec))

    n = len(cdpr_states)
    cdpr_pos = np.array([s.pose.position for s in cdpr_states])
    cdpr_quat = np.array([s.pose.quaternion_xyzw for s in cdpr_states])
    cdpr_vel = np.array([s.velocity.linear for s in cdpr_states])

    # --- 2) backend trajectory -------------------------------------------
    backend.set_pose(initial_state.pose)
    backend_states: list[PlatformState] = [backend.read_state()]
    for k in range(n - 1):
        backend.apply_wrench(cable_wrenches[k])
        backend.step(dt)
        backend_states.append(backend.read_state())

    backend_pos = np.array([s.pose.position for s in backend_states])
    backend_quat = np.array([s.pose.quaternion_xyzw for s in backend_states])
    backend_vel = np.array([s.velocity.linear for s in backend_states])

    times = np.array([k * dt for k in range(n)])
    return VerificationReport(
        time=times,
        cdpr_positions=cdpr_pos,
        backend_positions=backend_pos,
        cdpr_quaternions=cdpr_quat,
        backend_quaternions=backend_quat,
        cdpr_velocities=cdpr_vel,
        backend_velocities=backend_vel,
    )
