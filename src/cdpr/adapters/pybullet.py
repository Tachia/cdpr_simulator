r"""PyBullet adapter.

PyBullet has no native concept of cables, so this adapter uses the same
strategy as the MuJoCo one: PyBullet integrates the platform as a free
rigid body, the cdpr core computes the cable wrench, and the wrench is
fed into PyBullet through ``applyExternalForce`` /
``applyExternalTorque``. Cable lengths are computed geometrically from
the current pose --- :meth:`read_cable_lengths` returns the same numbers
the cdpr core would, but reading them from the adapter is convenient
when the rest of the experiment is already going through it.

Quaternion convention: PyBullet uses ``(x, y, z, w)`` --- matching the
framework's :class:`Pose`, so no conversion is needed (unlike MuJoCo).

Note: PyBullet wheels are not yet available for Python 3.14 at the time
of writing; the adapter installs cleanly anyway because of the lazy
import guard, and runs once a 3.14 wheel ships.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cdpr.adapters.base import AdapterCapability, PhysicsBackend
from cdpr.core.exceptions import MissingAdapterDependency

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.core.frames import Pose, Wrench
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot

try:
    import pybullet as pb
    import pybullet_data                                    # noqa: F401
    IS_AVAILABLE = True
except ImportError:
    pb = None                                               # type: ignore[assignment]
    IS_AVAILABLE = False


def _require() -> None:
    if not IS_AVAILABLE:
        raise MissingAdapterDependency(
            backend="pybullet",
            install_hint="pip install 'cdpr[adapters-pybullet]'  (or  pip install pybullet)",
        )


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class PyBulletAdapter(PhysicsBackend):
    """PyBullet rigid-body backend with cdpr-supplied cable wrench."""

    name = "pybullet"

    @property
    def capabilities(self) -> AdapterCapability:
        return (
            AdapterCapability.LOAD_ROBOT
            | AdapterCapability.SET_POSE
            | AdapterCapability.READ_STATE
            | AdapterCapability.STEP_PHYSICS
            | AdapterCapability.APPLY_WRENCH
            | AdapterCapability.READ_CABLE_LENGTHS
        )

    def __init__(
        self,
        robot: "Robot",
        *,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        timestep: float = 1e-3,
        gui: bool = False,
    ) -> None:
        _require()
        self._gravity = gravity
        self._timestep = timestep
        self._gui = gui
        self._client: int | None = None
        self._platform_uid: int | None = None
        self._pending_wrench = np.zeros(6, dtype=np.float64)
        super().__init__(robot)

    # --- lifecycle ---------------------------------------------------

    def load_robot(self, robot: "Robot") -> None:
        self.robot = robot
        inertia = robot.require_inertia()
        mode = pb.GUI if self._gui else pb.DIRECT
        self._client = pb.connect(mode)
        pb.setGravity(*self._gravity, physicsClientId=self._client)
        pb.setTimeStep(self._timestep, physicsClientId=self._client)
        pb.setRealTimeSimulation(0, physicsClientId=self._client)

        # Create the platform as a free rigid body. We don't need a collision
        # shape (cables don't collide with anything), only an inertial shape.
        I = inertia.inertia
        diag = np.diag(I).tolist()
        # PyBullet expects principal-axis inertia; for symmetric tensors the
        # diagonal is a sensible approximation when off-diagonals are zero.
        if not np.allclose(I, np.diag(diag), atol=1e-9):
            # Diagonalise to satisfy PyBullet, but warn through the docstring
            # rather than at runtime --- realistic CDPR platforms are
            # designed with diagonal inertia in the body frame.
            eigvals, _ = np.linalg.eigh(I)
            diag = eigvals.tolist()

        self._platform_uid = pb.createMultiBody(
            baseMass=inertia.mass,
            baseInertialFramePosition=list(inertia.com),
            baseInertialFrameOrientation=[0.0, 0.0, 0.0, 1.0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client,
        )
        # PyBullet doesn't expose the inertia tensor through createMultiBody
        # directly; we set it via changeDynamics.
        pb.changeDynamics(
            self._platform_uid, -1,
            mass=inertia.mass,
            localInertiaDiagonal=diag,
            physicsClientId=self._client,
        )

    def close(self) -> None:
        if self._client is not None and pb is not None:
            try:
                pb.disconnect(physicsClientId=self._client)
            except Exception:                               # pragma: no cover - defensive
                pass
            self._client = None

    # --- pose ---------------------------------------------------------

    def set_pose(self, pose: "Pose") -> None:
        if self._platform_uid is None:
            raise RuntimeError("pybullet adapter not initialised")
        pb.resetBasePositionAndOrientation(
            self._platform_uid,
            list(pose.position),
            list(pose.quaternion_xyzw),
            physicsClientId=self._client,
        )
        pb.resetBaseVelocity(
            self._platform_uid,
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=self._client,
        )

    def read_state(self) -> "PlatformState":
        if self._platform_uid is None:
            raise RuntimeError("pybullet adapter not initialised")
        from scipy.spatial.transform import Rotation
        from cdpr.core.frames import Pose, Twist
        from cdpr.dynamics.rigid_body import PlatformState

        pos, quat = pb.getBasePositionAndOrientation(
            self._platform_uid, physicsClientId=self._client,
        )
        lin, ang = pb.getBaseVelocity(
            self._platform_uid, physicsClientId=self._client,
        )
        return PlatformState(
            pose=Pose(
                position=np.asarray(pos, dtype=np.float64),
                rotation=Rotation.from_quat(np.asarray(quat, dtype=np.float64)),
            ),
            velocity=Twist.from_parts(
                np.asarray(lin, dtype=np.float64),
                np.asarray(ang, dtype=np.float64),
            ),
        )

    # --- wrench -------------------------------------------------------

    def apply_wrench(self, wrench: "Wrench") -> None:
        # PyBullet clears external forces after every stepSimulation, so we
        # stash and re-apply inside step() for fidelity across substeps.
        self._pending_wrench[:3] = wrench.force
        self._pending_wrench[3:] = wrench.torque

    # --- step ---------------------------------------------------------

    def step(self, dt: float) -> None:
        if self._platform_uid is None:
            raise RuntimeError("pybullet adapter not initialised")
        substeps = max(int(round(dt / self._timestep)), 1)
        for _ in range(substeps):
            pb.applyExternalForce(
                self._platform_uid, -1,
                forceObj=self._pending_wrench[:3].tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=pb.LINK_FRAME,
                physicsClientId=self._client,
            )
            pb.applyExternalTorque(
                self._platform_uid, -1,
                torqueObj=self._pending_wrench[3:].tolist(),
                flags=pb.LINK_FRAME,
                physicsClientId=self._client,
            )
            pb.stepSimulation(physicsClientId=self._client)

    # --- cable lengths -------------------------------------------------

    def read_cable_lengths(self) -> NDArray[np.float64]:
        if self._platform_uid is None:
            raise RuntimeError("pybullet adapter not initialised")
        # Compute geometrically from the current pose.
        state = self.read_state()
        b_world = state.pose.rotation.apply(self.robot.attachments) + state.pose.position
        return np.linalg.norm(self.robot.anchors - b_world, axis=-1)


Adapter = PyBulletAdapter
