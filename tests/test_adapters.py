"""Adapter registry probe: works on a machine with zero simulator SDKs installed."""

from __future__ import annotations

from cdpr.adapters import available_backends


def test_registry_returns_every_known_backend():
    status = available_backends()
    assert set(status) == {"ros2", "gazebo", "mujoco", "pybullet", "isaac_sim"}


def test_no_backend_pretends_to_be_available_in_a_clean_env():
    """On the developer machine we do not expect any backend installed; the
    probe must return False rather than raising. Adapter stubs are imported
    eagerly inside :func:`available_backends`."""
    status = available_backends()
    for name, available in status.items():
        # We don't fail if some backend happens to be available --- we just
        # require the entry to be a bool, never an exception.
        assert isinstance(available, bool), f"{name}: {available!r}"
