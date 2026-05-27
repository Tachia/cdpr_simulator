r"""Gazebo Harmonic adapter (stub).

Status: the adapter is *not* implemented in Phase 5 because Gazebo
Harmonic requires a system-wide install (apt / dnf) and a sourced
``GZ_SIM_RESOURCE_PATH``; that's outside the scope of a pip-installable
extra. The interface, the install hint, and the registry probe are in
place so a follow-up patch can drop in the real implementation without
churning the rest of the framework.

The expected implementation pattern is:

1. Generate an SDF model from a :class:`Robot` (mirroring what
   :func:`cdpr.adapters.mujoco.build_mjcf` does for MuJoCo): worldbody
   anchor frames, a single platform link, kinematic joints, cable
   tendons drawn as visual-only lines.
2. Use ``gz-transport`` to spawn the SDF into a running ``gz sim``
   instance and to publish/subscribe wrench commands, pose feedback,
   and tendon length feedback.
3. Implement :meth:`set_pose` via the ``/world/<name>/set_pose`` service.
4. Implement :meth:`step` by calling ``WorldControl::step`` over the
   transport.

Until the real version lands, instantiating this backend raises
:class:`cdpr.core.exceptions.MissingAdapterDependency` with the install
hint below.
"""

from __future__ import annotations

from cdpr.adapters.base import AdapterCapability, PhysicsBackend
from cdpr.core.exceptions import MissingAdapterDependency

try:
    import gz.transport13                                   # type: ignore[import-not-found]  # noqa: F401
    _GZ_TRANSPORT_PRESENT = True
except ImportError:
    _GZ_TRANSPORT_PRESENT = False


# This stub is structurally "present" (it can be imported) but
# "not available" for instantiation because the bridge is unimplemented.
IS_AVAILABLE = False


def _require() -> None:
    raise MissingAdapterDependency(
        backend="gazebo",
        install_hint=(
            "install Gazebo Harmonic + its Python bindings "
            "(https://gazebosim.org/docs/harmonic/install) and wait for the "
            "cdpr Gazebo adapter implementation to ship."
        ),
    )


class GazeboAdapter(PhysicsBackend):
    """Placeholder so the registry can report progress toward this backend."""

    name = "gazebo"

    @property
    def capabilities(self) -> AdapterCapability:
        return AdapterCapability.NONE

    def __init__(self, *_args, **_kwargs) -> None:
        _require()

    def load_robot(self, robot) -> None:                    # pragma: no cover - unreachable
        _require()


Adapter = GazeboAdapter
