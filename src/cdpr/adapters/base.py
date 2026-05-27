r"""Abstract base class, capability flags, and registry for external-physics adapters.

The scientific core is fully self-contained. Adapters here let the same
:class:`Robot`, :class:`Pose`, :class:`Wrench`, and trajectory objects be
pushed into an external physics engine (MuJoCo, PyBullet) or onto a ROS 2
transport for hardware-in-the-loop integration. The framework continues
working with zero adapters installed; only the registry probe changes its
output.

Lifecycle::

    with make_backend("mujoco", robot=ipanema_class()) as backend:
        backend.set_pose(initial_pose)
        for k in range(steps):
            backend.apply_wrench(W)
            backend.step(dt)
            state = backend.read_state()

The context-manager form guarantees ``close()`` runs even if the body
throws. Adapters that aren't capable of an operation declare it through
:attr:`PhysicsBackend.capabilities`; calling an unsupported method raises
:class:`NotImplementedError` with a message naming the backend and the
missing capability flag --- callers should consult capabilities first
rather than catch the exception.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Flag, auto
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.core.frames import Pose, Wrench
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot
    from numpy.typing import NDArray


BackendName = Literal["ros2", "gazebo", "mujoco", "pybullet", "isaac_sim"]


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------

class AdapterCapability(Flag):
    """Optional features an adapter may declare."""

    NONE = 0
    LOAD_ROBOT = auto()
    SET_POSE = auto()
    READ_STATE = auto()
    STEP_PHYSICS = auto()
    APPLY_WRENCH = auto()
    APPLY_CABLE_TENSIONS = auto()
    READ_CABLE_LENGTHS = auto()
    EXPORT_POSE_STREAM = auto()


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PhysicsBackend(ABC):
    """Contract every external-physics adapter satisfies.

    Subclasses must:
    * set :attr:`name` to one of the registered backend names,
    * declare :attr:`capabilities` (a non-empty :class:`AdapterCapability`),
    * implement :meth:`load_robot` (called from ``__init__``),
    * implement whichever lifecycle methods their capabilities advertise.

    The class is a plain :class:`abc.ABC` --- no dataclass --- so subclasses
    are free to define their own state (e.g. native handles, observation
    buffers) without fighting decorator ordering.
    """

    name: BackendName

    def __init__(self, robot: "Robot") -> None:
        if not isinstance(self.capabilities, AdapterCapability):
            raise TypeError(
                f"{type(self).__name__} must set capabilities to an AdapterCapability."
            )
        if AdapterCapability.LOAD_ROBOT in self.capabilities:
            self.load_robot(robot)

    # --- declared by subclasses --------------------------------------

    @property
    @abstractmethod
    def capabilities(self) -> AdapterCapability: ...

    @abstractmethod
    def load_robot(self, robot: "Robot") -> None: ...

    # --- lifecycle (concrete defaults raise NotImplementedError) -----

    def set_pose(self, pose: "Pose") -> None:
        self._unsupported("set_pose", AdapterCapability.SET_POSE)

    def read_state(self) -> "PlatformState":
        self._unsupported("read_state", AdapterCapability.READ_STATE)

    def step(self, dt: float) -> None:
        self._unsupported("step", AdapterCapability.STEP_PHYSICS)

    def apply_wrench(self, wrench: "Wrench") -> None:
        self._unsupported("apply_wrench", AdapterCapability.APPLY_WRENCH)

    def apply_cable_tensions(self, tensions: "NDArray") -> None:
        self._unsupported("apply_cable_tensions", AdapterCapability.APPLY_CABLE_TENSIONS)

    def read_cable_lengths(self) -> "NDArray":
        self._unsupported("read_cable_lengths", AdapterCapability.READ_CABLE_LENGTHS)

    def close(self) -> None:
        """Release any native resources. Idempotent."""
        pass

    # --- context manager ---------------------------------------------

    def __enter__(self) -> "PhysicsBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- helpers ------------------------------------------------------

    def _unsupported(self, op: str, flag: AdapterCapability) -> None:
        raise NotImplementedError(
            f"{self.name!r} adapter does not implement {op!r} "
            f"(missing capability {flag.name})."
        )

    def has(self, flag: AdapterCapability) -> bool:
        """Test for a capability flag."""
        return flag in self.capabilities


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

_BACKENDS: dict[BackendName, str] = {
    "ros2":      "cdpr.adapters.ros2",
    "gazebo":    "cdpr.adapters.gazebo",
    "mujoco":    "cdpr.adapters.mujoco",
    "pybullet":  "cdpr.adapters.pybullet",
    "isaac_sim": "cdpr.adapters.isaac_sim",
}


def available_backends() -> dict[BackendName, bool]:
    """Probe every registered backend without raising on missing SDKs."""
    import importlib
    status: dict[BackendName, bool] = {}
    for name, module_path in _BACKENDS.items():
        try:
            mod = importlib.import_module(module_path)
            status[name] = bool(getattr(mod, "IS_AVAILABLE", False))
        except Exception:
            status[name] = False
    return status


def make_backend(name: BackendName, *, robot: "Robot", **kwargs: Any) -> PhysicsBackend:
    """Construct the named backend, importing it lazily.

    Raises :class:`cdpr.core.exceptions.MissingAdapterDependency` if the
    backend module declares ``IS_AVAILABLE = False``.
    """
    import importlib
    from cdpr.core.exceptions import MissingAdapterDependency

    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend {name!r}; choose from {list(_BACKENDS)}.")
    module = importlib.import_module(_BACKENDS[name])
    if not getattr(module, "IS_AVAILABLE", False):
        # Each stub module exposes a `_require` that raises with a hint.
        require = getattr(module, "_require", None)
        if require is not None:
            require()                                       # always raises
        raise MissingAdapterDependency(name, "see adapter module for install hint")
    cls = getattr(module, "Adapter", None)
    if cls is None:
        raise NotImplementedError(
            f"{name!r} backend reports IS_AVAILABLE=True but defines no Adapter class."
        )
    return cls(robot=robot, **kwargs)
