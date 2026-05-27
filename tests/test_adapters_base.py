"""Refactored adapter base contract and registry behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from cdpr.adapters import (
    AdapterCapability,
    PhysicsBackend,
    available_backends,
    make_backend,
)
from cdpr.core.exceptions import MissingAdapterDependency


class _NoOpBackend(PhysicsBackend):
    """Minimal subclass used to exercise the abstract base."""

    name = "noop"

    @property
    def capabilities(self) -> AdapterCapability:
        return AdapterCapability.LOAD_ROBOT | AdapterCapability.READ_STATE

    def load_robot(self, robot) -> None:
        self.robot = robot


def test_subclass_must_set_capabilities():
    class _Bad(PhysicsBackend):
        name = "bad"
        # forgot to override `capabilities`

        def load_robot(self, robot):
            pass

    with pytest.raises(TypeError):
        _Bad(robot=None)


def test_unsupported_methods_raise_with_capability_name(point_mass_robot):
    backend = _NoOpBackend(robot=point_mass_robot)
    assert backend.has(AdapterCapability.READ_STATE)
    with pytest.raises(NotImplementedError, match="STEP_PHYSICS"):
        backend.step(0.01)
    with pytest.raises(NotImplementedError, match="APPLY_WRENCH"):
        backend.apply_wrench(None)


def test_context_manager_calls_close(point_mass_robot):
    closed = {"hit": False}

    class _Track(_NoOpBackend):
        name = "track"
        def close(self):
            closed["hit"] = True

    with _Track(robot=point_mass_robot) as backend:
        assert isinstance(backend, _Track)
    assert closed["hit"] is True


def test_available_backends_returns_bool_for_each(point_mass_robot):
    status = available_backends()
    assert set(status) == {"ros2", "gazebo", "mujoco", "pybullet", "isaac_sim"}
    for k, v in status.items():
        assert isinstance(v, bool), f"{k}: {v!r}"


def test_make_backend_on_unavailable_raises(point_mass_robot):
    # gazebo is intentionally not available even when the module imports.
    with pytest.raises(MissingAdapterDependency):
        make_backend("gazebo", robot=point_mass_robot)


def test_ros2_transport_in_memory_mode(point_mass_robot):
    """ROS 2 in-memory mode should always work, even without rclpy."""
    from scipy.spatial.transform import Rotation
    from cdpr.core.frames import Pose, Twist
    from cdpr.dynamics.rigid_body import PlatformState

    backend = make_backend("ros2", robot=point_mass_robot, use_rclpy=False)
    assert AdapterCapability.EXPORT_POSE_STREAM in backend.capabilities

    state = PlatformState(
        pose=Pose(position=np.array([0.1, 0.0, 0.0]), rotation=Rotation.identity()),
        velocity=Twist(np.zeros(6)),
    )
    backend.publish_state(state, np.ones(4) * 1.7, np.ones(4) * 50.0, timestamp=0.5)
    latest = backend.latest_published()
    assert latest is not None
    assert latest.timestamp == 0.5
    assert np.allclose(latest.position, [0.1, 0.0, 0.0])
    assert np.allclose(latest.cable_tensions, 50.0)
    backend.close()
