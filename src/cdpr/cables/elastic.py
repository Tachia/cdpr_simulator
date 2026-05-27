r"""Linear-elastic axial cable.

The cable runs straight along the chord but obeys Hooke's law axially:

.. math::

    L_\text{stretched} = L_0\,\bigl(1 + T / (EA)\bigr),

so for a given chord length :math:`L_c` and unstretched length :math:`L_0`,
the tension is

.. math::

    T = EA \cdot \frac{L_c - L_0}{L_0}, \quad T \geq 0.

The cable cannot push, so any configuration with :math:`L_c < L_0` is
reported as slack with zero tension. This is the right model for short,
stiff cables where sag is negligible but stretch is not (e.g. several-metre
spans on a stiff steel rope).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdpr.cables.base import CableSolution


def elastic_cable(
    anchor_upper: ArrayLike,
    anchor_lower: ArrayLike,
    unstretched_length: float,
    axial_stiffness: float,
) -> CableSolution:
    """Evaluate the axial-spring cable model.

    Parameters
    ----------
    anchor_upper:
        World-frame anchor on the fixed structure.
    anchor_lower:
        World-frame platform attachment.
    unstretched_length:
        :math:`L_0`, the natural cable length when no tension is applied.
    axial_stiffness:
        :math:`EA` product (Young's modulus times cross-sectional area).
    """
    a_up: NDArray[np.float64] = np.asarray(anchor_upper, dtype=np.float64)
    a_lo: NDArray[np.float64] = np.asarray(anchor_lower, dtype=np.float64)
    chord = a_up - a_lo
    L_c = float(np.linalg.norm(chord))
    if L_c <= 0.0:
        return CableSolution(
            force_on_platform=np.zeros(3),
            tension_lower=0.0, tension_upper=0.0,
            chord_length=0.0, arc_length=0.0, is_slack=True,
        )
    u = chord / L_c
    strain = (L_c - unstretched_length) / unstretched_length
    if strain <= 0.0:
        return CableSolution(
            force_on_platform=np.zeros(3),
            tension_lower=0.0, tension_upper=0.0,
            chord_length=L_c, arc_length=unstretched_length,
            axial_strain=strain, is_slack=True,
        )
    T = axial_stiffness * strain
    return CableSolution(
        force_on_platform=T * u,
        tension_lower=T, tension_upper=T,
        chord_length=L_c, arc_length=L_c,
        axial_strain=strain, is_slack=False,
    )
