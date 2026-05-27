r"""Controller protocol and shared helpers.

A controller is a callable with the signature ::

    controller(
        *,
        state: PlatformState,
        reference_pose: Pose,
        reference_twist: Twist,
        reference_accel: tuple[NDArray, NDArray] | None,
        t: float,
        robot: Robot,
        gravity: NDArray,
        external: Wrench,
    ) -> Wrench

returning the wrench it wants the cables to apply. Anything matching that
signature is a controller --- the :class:`Controller` Protocol exists so
that user code and the simulator can type against it.

Helpers in this module are the few small primitives the two shipped
controllers share. The most important one is :func:`orientation_error`,
which produces a 3-vector rotation-vector error :math:`\log_{SO(3)}
(\mathbf{R}_{\text{ref}} \mathbf{R}^{\top})`. That's the canonical
"how far apart are two orientations" used by orientation-PD laws; we expose
it separately so it can be reused outside the shipped controllers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.core.frames import Pose, Twist, Wrench
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot


@runtime_checkable
class Controller(Protocol):
    """Callable that maps a state and reference into a desired cable wrench."""

    def __call__(
        self,
        *,
        state: "PlatformState",
        reference_pose: "Pose",
        reference_twist: "Twist",
        reference_accel: "tuple[NDArray[np.float64], NDArray[np.float64]] | None",
        t: float,
        robot: "Robot",
        gravity: NDArray[np.float64],
        external: "Wrench",
    ) -> "Wrench": ...


def orientation_error(reference: "Pose", actual: "Pose") -> NDArray[np.float64]:
    r"""Return :math:`\boldsymbol\theta_e = \log_{SO(3)}(\mathbf{R}_\text{ref}\,\mathbf{R}^{\top})`.

    The result is a 3-vector whose direction is the axis of rotation and
    whose magnitude is the angle. Using the rotvec representation avoids
    the unit-quaternion sign ambiguity (SciPy's ``Rotation.as_rotvec()``
    returns a canonical vector in :math:`[-\pi, \pi]` magnitude).
    """
    relative = reference.rotation * actual.rotation.inv()
    return relative.as_rotvec()


def as_gain_matrix(gain: float | NDArray[np.float64], dim: int = 3) -> NDArray[np.float64]:
    """Promote a scalar gain to a diagonal matrix; pass arrays through as-is.

    Accepts:
    * a scalar (treated as ``gain * I``)
    * a 1-D length-``dim`` array (treated as the diagonal)
    * a ``dim``-by-``dim`` matrix (used directly)
    """
    g = np.asarray(gain, dtype=np.float64)
    if g.ndim == 0:
        return np.eye(dim) * float(g)
    if g.ndim == 1:
        if g.shape[0] != dim:
            raise ValueError(f"1-D gain must have length {dim}; got {g.shape}")
        return np.diag(g)
    if g.shape != (dim, dim):
        raise ValueError(f"2-D gain must be {dim}x{dim}; got {g.shape}")
    return g
