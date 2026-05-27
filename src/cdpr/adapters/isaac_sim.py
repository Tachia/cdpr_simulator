r"""Isaac Sim adapter (stub).

Status: not implemented in Phase 5. Isaac Sim ships through NVIDIA's
Omniverse launcher, not pip, and its Python bindings (``omni.isaac.kit``)
require an Omniverse-managed Python interpreter --- mixing it with the
framework's pip environment is non-trivial. The interface is reserved
here so that, once the install side is solved, only one module needs to
change.

Planned implementation outline:

1. Generate a USD scene from a :class:`Robot`: world frame, base anchor
   transforms, articulation root for the platform, cable rope assets
   between each anchor / attachment pair (USD's :class:`PhysxRope` is
   the natural primitive here).
2. Stand up a single :class:`isaacsim.SimulationApp`; spawn the USD
   into it; read back via the Isaac core APIs.
3. Capability set will be a superset of the MuJoCo adapter's (Isaac Sim
   provides photorealistic rendering, optional sensor synthesis, and
   parallel environment cloning for RL --- those would extend
   :class:`AdapterCapability` rather than being hidden behind opaque
   methods).
"""

from __future__ import annotations

from cdpr.adapters.base import AdapterCapability, PhysicsBackend
from cdpr.core.exceptions import MissingAdapterDependency

try:
    import omni.isaac.kit                                   # type: ignore[import-not-found]  # noqa: F401
    _ISAAC_KIT_PRESENT = True
except ImportError:
    _ISAAC_KIT_PRESENT = False

IS_AVAILABLE = False


def _require() -> None:
    raise MissingAdapterDependency(
        backend="isaac_sim",
        install_hint=(
            "install Isaac Sim via the NVIDIA Omniverse launcher "
            "(https://docs.isaacsim.omniverse.nvidia.com/) and wait for the "
            "cdpr Isaac Sim adapter implementation to ship."
        ),
    )


class IsaacSimAdapter(PhysicsBackend):
    """Placeholder."""

    name = "isaac_sim"

    @property
    def capabilities(self) -> AdapterCapability:
        return AdapterCapability.NONE

    def __init__(self, *_args, **_kwargs) -> None:
        _require()

    def load_robot(self, robot) -> None:                    # pragma: no cover - unreachable
        _require()


Adapter = IsaacSimAdapter
