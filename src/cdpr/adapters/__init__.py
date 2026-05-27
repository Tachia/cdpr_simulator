r"""Optional external-physics adapter layer (Phase-2 interface, Phase-5 implementations).

The framework's scientific core is fully self-contained --- it computes
kinematics, statics, dynamics, workspaces, and trajectories without any
external simulator. Adapters here let the *same* :class:`Robot`,
:class:`Pose`, :class:`Wrench`, and trajectory objects be pushed into an
external physics engine for high-fidelity rendering (Gazebo Harmonic, Isaac
Sim) or large-scale RL acceleration (MuJoCo, PyBullet), or be exposed as
ROSĀ 2 topics for hardware-in-the-loop integration.

What this module ships in Phase 2:

* :class:`PhysicsBackend` -- the abstract base class every adapter must
  implement. Methods are typed against Phase-1 objects, never against
  third-party SDK types --- the adapter is responsible for the translation.
* A registry-style discovery mechanism (:func:`available_backends`).
* Five stub modules (``ros2``, ``gazebo``, ``mujoco``, ``pybullet``,
  ``isaac_sim``) that import lazily and raise
  :class:`cdpr.core.exceptions.MissingAdapterDependency` until the matching
  backend package is installed.

What this module does *not* ship: the actual simulator bridges. Those will
land in Phase 5 once the controller, RL, and PINN layers are in place ---
adapter design without consumers is a guessing game.
"""

from cdpr.adapters.base import (
    AdapterCapability,
    BackendName,
    PhysicsBackend,
    available_backends,
    make_backend,
)
from cdpr.adapters.verify import VerificationReport, verify_against

__all__ = [
    "PhysicsBackend",
    "AdapterCapability",
    "BackendName",
    "available_backends",
    "make_backend",
    "verify_against",
    "VerificationReport",
]
