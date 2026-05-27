"""Ideal massless cable: straight line, infinite axial stiffness, no sag.

This is the model assumed implicitly by the kinematics structure matrix and
by most published tension-distribution work. It exists here as a first-class
:class:`~cdpr.cables.base.CableSolution` returner so callers can swap models
through the same interface without branching.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdpr.cables.base import CableSolution


def massless_cable(
    anchor_upper: ArrayLike,
    anchor_lower: ArrayLike,
    *,
    tension: float = 0.0,
) -> CableSolution:
    r"""Evaluate an ideal massless cable between two world-frame points.

    Parameters
    ----------
    anchor_upper:
        World-frame anchor on the fixed structure.
    anchor_lower:
        World-frame attachment on the moving platform.
    tension:
        Tension magnitude the cable is carrying. Optional; only affects
        :attr:`force_on_platform`. The kinematics layer is independent of
        this value, so leaving it zero is the right thing when the caller
        only needs the chord length.

    Notes
    -----
    The force on the platform is :math:`T\,\hat{\mathbf{u}}` where
    :math:`\hat{\mathbf{u}} = (\mathbf{a}_\text{upper} - \mathbf{a}_\text{lower}) / L_\text{chord}`
    points from the platform attachment toward the base anchor.
    """
    a_up: NDArray[np.float64] = np.asarray(anchor_upper, dtype=np.float64)
    a_lo: NDArray[np.float64] = np.asarray(anchor_lower, dtype=np.float64)
    chord = a_up - a_lo
    L = float(np.linalg.norm(chord))
    if L <= 0.0:
        return CableSolution(
            force_on_platform=np.zeros(3),
            tension_lower=0.0,
            tension_upper=0.0,
            chord_length=0.0,
            arc_length=0.0,
            is_slack=True,
        )
    u = chord / L
    return CableSolution(
        force_on_platform=tension * u,
        tension_lower=float(tension),
        tension_upper=float(tension),
        chord_length=L,
        arc_length=L,
        is_slack=(tension <= 0.0),
    )
