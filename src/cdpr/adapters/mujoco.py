r"""MuJoCo adapter.

Builds an MJCF model programmatically from a :class:`Robot` definition:

* every base anchor becomes a worldbody ``<site>``,
* the platform is a single free-body with mass and inertia taken from
  :class:`PlatformInertia`,
* every platform attachment becomes a body-local ``<site>``,
* each cable is a ``<tendon><spatial>`` connecting the two sites --- the
  tendon is visualised and its length is accessible from
  :meth:`MuJoCoAdapter.read_cable_lengths`, but it carries no actuator
  force; cable forces are applied externally as a free-body wrench via
  :meth:`MuJoCoAdapter.apply_wrench`. This split keeps tension
  computation in the cdpr core (where the QP-based tension distribution
  lives) and uses MuJoCo purely as a rigid-body integrator + renderer.

Quaternion convention note: MuJoCo stores quaternions as ``(w, x, y, z)``;
the framework's :class:`Pose` uses SciPy's ``(x, y, z, w)``. Conversions
happen inside :meth:`set_pose` and :meth:`read_state`.

Usage::

    from cdpr.adapters import make_backend
    with make_backend("mujoco", robot=ipanema_class()) as backend:
        backend.set_pose(initial_pose)
        for k in range(n_steps):
            backend.apply_wrench(wrench)        # F + tau on platform body origin
            backend.step(dt)
            state = backend.read_state()
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
    import mujoco
    IS_AVAILABLE = True
except ImportError:
    mujoco = None                                           # type: ignore[assignment]
    IS_AVAILABLE = False


def _require() -> None:
    if not IS_AVAILABLE:
        raise MissingAdapterDependency(
            backend="mujoco",
            install_hint="pip install 'cdpr[adapters-mujoco]'  (or  pip install mujoco)",
        )


# ---------------------------------------------------------------------------
# MJCF generation
# ---------------------------------------------------------------------------

def build_mjcf(robot: "Robot", *,
               gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
               timestep: float = 1e-3) -> str:
    """Return the MJCF XML for a CDPR :class:`Robot`."""
    inertia = robot.require_inertia()
    a = robot.anchors
    b = robot.attachments
    g_str = " ".join(f"{v:.6g}" for v in gravity)
    com_str = " ".join(f"{v:.6g}" for v in inertia.com)

    # MuJoCo's fullinertia order: M11 M22 M33 M12 M13 M23
    I = inertia.inertia
    fullinertia = f"{I[0,0]:.6g} {I[1,1]:.6g} {I[2,2]:.6g} {I[0,1]:.6g} {I[0,2]:.6g} {I[1,2]:.6g}"

    # Crude bounding box of the platform attachments for the visual geom.
    bbox = np.max(np.abs(b), axis=0)
    if (bbox < 1e-6).all():                                 # point-mass robot
        bbox = np.array([0.02, 0.02, 0.02])
    half = bbox * 0.5
    geom_size_str = " ".join(f"{max(v, 0.01):.6g}" for v in half)

    anchor_sites = "\n    ".join(
        f'<site name="anchor_{i}" pos="{a[i,0]:.6g} {a[i,1]:.6g} {a[i,2]:.6g}" '
        f'size="0.02" rgba="0.2 0.2 0.2 1"/>' for i in range(robot.n_cables)
    )
    attach_sites = "\n      ".join(
        f'<site name="attach_{i}" pos="{b[i,0]:.6g} {b[i,1]:.6g} {b[i,2]:.6g}" '
        f'size="0.015" rgba="0.85 0.32 0.15 1"/>' for i in range(robot.n_cables)
    )
    tendons = "\n    ".join(
        f'<spatial name="cable_{i}" width="0.004" rgba="0.1 0.45 0.7 0.9">'
        f'<site site="attach_{i}"/><site site="anchor_{i}"/></spatial>'
        for i in range(robot.n_cables)
    )

    xml = f"""<mujoco model="cdpr_{robot.name}">
  <option timestep="{timestep:.6g}" integrator="RK4" gravity="{g_str}"/>
  <visual>
    <quality shadowsize="2048"/>
    <map zfar="100"/>
  </visual>
  <worldbody>
    {anchor_sites}
    <body name="platform" pos="0 0 0">
      <freejoint name="platform_free"/>
      <inertial pos="{com_str}" mass="{inertia.mass:.6g}" fullinertia="{fullinertia}"/>
      {attach_sites}
      <geom name="platform_geom" type="box" size="{geom_size_str}"
            rgba="0.30 0.55 0.85 0.35" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <tendon>
    {tendons}
  </tendon>
</mujoco>
"""
    return xml


# ---------------------------------------------------------------------------
# Quaternion conversions
# ---------------------------------------------------------------------------

def _quat_xyzw_to_wxyz(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """SciPy (x, y, z, w)  ->  MuJoCo (w, x, y, z)."""
    return np.array([q[3], q[0], q[1], q[2]])


def _quat_wxyz_to_xyzw(q: NDArray[np.float64]) -> NDArray[np.float64]:
    """MuJoCo (w, x, y, z)  ->  SciPy (x, y, z, w)."""
    return np.array([q[1], q[2], q[3], q[0]])


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class MuJoCoAdapter(PhysicsBackend):
    """MuJoCo rigid-body backend with cdpr-supplied cable wrench."""

    name = "mujoco"

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
        xml_override: str | None = None,
    ) -> None:
        _require()
        self._gravity = gravity
        self._timestep = timestep
        self._xml_override = xml_override
        self.mjModel = None
        self.mjData = None
        self._platform_body_id: int | None = None
        self._cable_tendon_ids: list[int] = []
        self._dof_index_lin: slice | None = None
        super().__init__(robot)                             # invokes load_robot

    # --- lifecycle ---------------------------------------------------

    def load_robot(self, robot: "Robot") -> None:
        xml = self._xml_override or build_mjcf(
            robot, gravity=self._gravity, timestep=self._timestep,
        )
        self.xml = xml
        self.mjModel = mujoco.MjModel.from_xml_string(xml)
        self.mjData = mujoco.MjData(self.mjModel)
        self.robot = robot

        # Cache the platform body id and cable tendon ids by name.
        self._platform_body_id = int(mujoco.mj_name2id(
            self.mjModel, mujoco.mjtObj.mjOBJ_BODY, "platform",
        ))
        if self._platform_body_id < 0:
            raise RuntimeError("mujoco model has no 'platform' body --- MJCF build failed.")
        self._cable_tendon_ids = [
            int(mujoco.mj_name2id(self.mjModel, mujoco.mjtObj.mjOBJ_TENDON, f"cable_{i}"))
            for i in range(robot.n_cables)
        ]

    def close(self) -> None:
        # MuJoCo objects own native buffers; drop our references and the
        # garbage collector frees them.
        self.mjModel = None
        self.mjData = None

    # --- pose ---------------------------------------------------------

    def set_pose(self, pose: "Pose") -> None:
        if self.mjData is None:
            raise RuntimeError("mujoco adapter not initialised")
        # Free joint qpos layout: x y z qw qx qy qz
        self.mjData.qpos[:3] = pose.position
        self.mjData.qpos[3:7] = _quat_xyzw_to_wxyz(pose.quaternion_xyzw)
        self.mjData.qvel[:6] = 0.0
        mujoco.mj_forward(self.mjModel, self.mjData)

    def read_state(self) -> "PlatformState":
        if self.mjData is None:
            raise RuntimeError("mujoco adapter not initialised")
        from scipy.spatial.transform import Rotation
        from cdpr.core.frames import Pose, Twist
        from cdpr.dynamics.rigid_body import PlatformState

        pos = self.mjData.qpos[:3].copy()
        quat_xyzw = _quat_wxyz_to_xyzw(self.mjData.qpos[3:7].copy())
        # qvel for a free joint: first 3 are linear velocity in WORLD frame,
        # next 3 are angular velocity in WORLD frame (MuJoCo docs:
        # https://mujoco.readthedocs.io/en/stable/computation/index.html#cdof).
        vel_lin = self.mjData.qvel[:3].copy()
        vel_ang = self.mjData.qvel[3:6].copy()
        return PlatformState(
            pose=Pose(position=pos, rotation=Rotation.from_quat(quat_xyzw)),
            velocity=Twist.from_parts(vel_lin, vel_ang),
        )

    # --- forces -------------------------------------------------------

    def apply_wrench(self, wrench: "Wrench") -> None:
        if self.mjData is None or self._platform_body_id is None:
            raise RuntimeError("mujoco adapter not initialised")
        # xfrc_applied is shape (nbody, 6) in (Fx Fy Fz Tx Ty Tz), in WORLD frame.
        self.mjData.xfrc_applied[self._platform_body_id, :3] = wrench.force
        self.mjData.xfrc_applied[self._platform_body_id, 3:] = wrench.torque

    # --- step ---------------------------------------------------------

    def step(self, dt: float) -> None:
        if self.mjData is None or self.mjModel is None:
            raise RuntimeError("mujoco adapter not initialised")
        # Honour the caller's dt: temporarily override the model timestep,
        # then take however many sub-steps the configured timestep needs.
        substeps = max(int(round(dt / self.mjModel.opt.timestep)), 1)
        for _ in range(substeps):
            mujoco.mj_step(self.mjModel, self.mjData)

    # --- cable lengths -------------------------------------------------

    def read_cable_lengths(self) -> NDArray[np.float64]:
        if self.mjData is None:
            raise RuntimeError("mujoco adapter not initialised")
        # ten_length is shape (ntendon,) in metres.
        lengths = np.empty(len(self._cable_tendon_ids), dtype=np.float64)
        for k, tid in enumerate(self._cable_tendon_ids):
            lengths[k] = float(self.mjData.ten_length[tid])
        return lengths


# The name the registry / factory expects.
Adapter = MuJoCoAdapter
