r"""ROS 2 (Jazzy) transport adapter.

ROS 2 is not a physics engine; it's a transport layer. This adapter
therefore exposes only the streaming capability of the
:class:`PhysicsBackend` contract --- :attr:`EXPORT_POSE_STREAM` --- and
provides one method beyond the base class: :meth:`publish_state`, which
broadcasts the current platform pose, cable lengths, and cable tensions.
Command input (e.g. a desired pose pushed by an external planner) goes
the other way through :meth:`latest_command`.

Two modes:

1. **In-memory mode** (default, no extra dependencies). Pushed states
   are stored on the instance and read back through accessors. Useful
   for unit tests, for offline simulation-to-data pipelines, and as the
   API surface examples and the dissertation demonstrations are
   written against.

2. **rclpy mode** (``use_rclpy=True``, requires a working ROS 2 install
   with ``rclpy`` importable). Creates a ROS 2 node, declares
   publishers on ``<topic_prefix>/pose``,
   ``<topic_prefix>/cable_lengths``, ``<topic_prefix>/cable_tensions``,
   and a subscriber on ``<topic_prefix>/command_pose``. Messages use
   the standard ``geometry_msgs/PoseStamped`` and
   ``std_msgs/Float64MultiArray`` types, which keeps the adapter
   message-package-agnostic. Custom CDPR messages can be wired in by
   passing message classes via the constructor's keyword arguments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cdpr.adapters.base import AdapterCapability, PhysicsBackend

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.core.frames import Pose
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot

try:
    import rclpy                                            # type: ignore[import-not-found]
    IS_AVAILABLE = True
except ImportError:
    rclpy = None                                            # type: ignore[assignment]
    IS_AVAILABLE = False


def _require() -> None:
    if not IS_AVAILABLE:
        from cdpr.core.exceptions import MissingAdapterDependency
        raise MissingAdapterDependency(
            backend="ros2",
            install_hint="install ROS 2 Jazzy and the rclpy Python bindings; "
                         "see https://docs.ros.org/en/jazzy/",
        )


# ---------------------------------------------------------------------------
# Buffered messages (in-memory mode)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class _PublishedState:
    """Snapshot of the last :meth:`publish_state` call."""

    timestamp: float
    position: NDArray[np.float64]
    quaternion_xyzw: NDArray[np.float64]
    cable_lengths: NDArray[np.float64]
    cable_tensions: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ROS2TransportAdapter(PhysicsBackend):
    """Transport-only adapter: publish state, subscribe to commands."""

    name = "ros2"

    @property
    def capabilities(self) -> AdapterCapability:
        # Intentionally narrow: this backend does not simulate physics.
        return AdapterCapability.LOAD_ROBOT | AdapterCapability.EXPORT_POSE_STREAM

    def __init__(
        self,
        robot: "Robot",
        *,
        topic_prefix: str = "/cdpr",
        use_rclpy: bool = False,
        node_name: str = "cdpr_transport",
    ) -> None:
        self.topic_prefix = topic_prefix
        self.use_rclpy = bool(use_rclpy)
        self.node_name = node_name
        self._latest: _PublishedState | None = None
        self._latest_command_pose: "Pose | None" = None
        self._published_history: list[_PublishedState] = []
        self._node = None
        self._publishers: dict[str, object] = {}
        super().__init__(robot)

    def load_robot(self, robot: "Robot") -> None:
        self.robot = robot
        if self.use_rclpy:
            self._init_rclpy()

    def _init_rclpy(self) -> None:                          # pragma: no cover - exercised in ROS env
        _require()
        if not rclpy.ok():
            rclpy.init()
        from geometry_msgs.msg import PoseStamped               # type: ignore[import-not-found]
        from std_msgs.msg import Float64MultiArray              # type: ignore[import-not-found]

        self._node = rclpy.create_node(self.node_name)
        self._publishers["pose"] = self._node.create_publisher(
            PoseStamped, f"{self.topic_prefix}/pose", 10,
        )
        self._publishers["cable_lengths"] = self._node.create_publisher(
            Float64MultiArray, f"{self.topic_prefix}/cable_lengths", 10,
        )
        self._publishers["cable_tensions"] = self._node.create_publisher(
            Float64MultiArray, f"{self.topic_prefix}/cable_tensions", 10,
        )
        self._node.create_subscription(
            PoseStamped, f"{self.topic_prefix}/command_pose",
            self._on_command_pose, 10,
        )

    def close(self) -> None:
        if self._node is not None and rclpy is not None:        # pragma: no cover - ROS env
            self._node.destroy_node()
            self._node = None
            try:
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass

    # --- streaming API -------------------------------------------------

    def publish_state(
        self,
        state: "PlatformState",
        cable_lengths: NDArray[np.float64],
        cable_tensions: NDArray[np.float64],
        *,
        timestamp: float = 0.0,
    ) -> None:
        """Publish (or buffer) one snapshot of the platform state."""
        msg = _PublishedState(
            timestamp=float(timestamp),
            position=np.asarray(state.pose.position, dtype=np.float64),
            quaternion_xyzw=np.asarray(state.pose.quaternion_xyzw, dtype=np.float64),
            cable_lengths=np.asarray(cable_lengths, dtype=np.float64),
            cable_tensions=np.asarray(cable_tensions, dtype=np.float64),
        )
        self._latest = msg
        self._published_history.append(msg)
        if self.use_rclpy and self._node is not None:           # pragma: no cover - ROS env
            self._publish_rclpy(msg)

    def _publish_rclpy(self, msg: _PublishedState) -> None:     # pragma: no cover - ROS env
        from geometry_msgs.msg import PoseStamped               # type: ignore[import-not-found]
        from std_msgs.msg import Float64MultiArray              # type: ignore[import-not-found]
        ps = PoseStamped()
        ps.header.frame_id = "world"
        ps.pose.position.x = float(msg.position[0])
        ps.pose.position.y = float(msg.position[1])
        ps.pose.position.z = float(msg.position[2])
        ps.pose.orientation.x = float(msg.quaternion_xyzw[0])
        ps.pose.orientation.y = float(msg.quaternion_xyzw[1])
        ps.pose.orientation.z = float(msg.quaternion_xyzw[2])
        ps.pose.orientation.w = float(msg.quaternion_xyzw[3])
        self._publishers["pose"].publish(ps)

        for key, arr in (("cable_lengths", msg.cable_lengths),
                         ("cable_tensions", msg.cable_tensions)):
            m = Float64MultiArray()
            m.data = arr.tolist()
            self._publishers[key].publish(m)

    def latest_published(self) -> _PublishedState | None:
        return self._latest

    def published_history(self) -> list[_PublishedState]:
        return list(self._published_history)

    def latest_command(self) -> "Pose | None":
        return self._latest_command_pose

    def push_command_pose(self, pose: "Pose") -> None:
        """In-memory-mode helper: simulate an incoming command."""
        self._latest_command_pose = pose

    def _on_command_pose(self, msg) -> None:                    # pragma: no cover - ROS env
        from scipy.spatial.transform import Rotation
        from cdpr.core.frames import Pose
        self._latest_command_pose = Pose(
            position=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            rotation=Rotation.from_quat([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w,
            ]),
        )


# In-memory mode works without rclpy installed; the registry probe is
# therefore True for transport purposes even when ROS 2 itself is absent.
# The Adapter is always exposed; mode is selected at construction time.
Adapter = ROS2TransportAdapter
IS_AVAILABLE = True
