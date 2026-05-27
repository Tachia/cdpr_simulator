r"""Parameter parametrisation for CDPR identification.

A flat decision vector is assembled from selected groups; each group
knows its own size, its bounds, and how to read its slice back out into
a meaningful structure (per-cable offset, per-cable rest-length bias,
scalar mass).

This module deliberately avoids dynamic / inertial parameters. Anchor
positions, attachment positions, and cable rest-length offsets are all
identifiable from purely kinematic data --- recorded poses and recorded
cable lengths --- which is the only data most CDPR labs reliably
collect. Mass and friction identification needs synchronised tension
measurements and is left for a follow-up module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Identifiable groups
# ---------------------------------------------------------------------------

class IdentifiableGroup(str, Enum):
    """Which parameter groups the identification is allowed to perturb."""

    ANCHOR_OFFSETS = "anchor_offsets"          # 3 * m parameters
    ATTACHMENT_OFFSETS = "attachment_offsets"  # 3 * m parameters
    CABLE_LENGTH_OFFSETS = "cable_length_offsets"  # m parameters


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ParameterBounds:
    """Symmetric box bounds around the nominal value, per group.

    Defaults assume the nominal anchor / attachment positions are accurate
    to within 10 mm and cable encoders to within 5 mm --- typical for a
    factory-calibrated CDPR. Tighten for finer estimates, loosen when the
    calibration is known to be poor.
    """

    anchor_offset_m: float = 0.01
    attachment_offset_m: float = 0.01
    cable_length_offset_m: float = 0.005


# ---------------------------------------------------------------------------
# Decision-vector packing
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IdentifiableParameters:
    """Map between a flat decision vector and structured parameter groups."""

    groups: tuple[IdentifiableGroup, ...]
    n_cables: int
    bounds: ParameterBounds = field(default_factory=ParameterBounds)

    def size_per_group(self) -> dict[IdentifiableGroup, int]:
        return {
            IdentifiableGroup.ANCHOR_OFFSETS:        3 * self.n_cables,
            IdentifiableGroup.ATTACHMENT_OFFSETS:    3 * self.n_cables,
            IdentifiableGroup.CABLE_LENGTH_OFFSETS:  self.n_cables,
        }

    def total_size(self) -> int:
        sizes = self.size_per_group()
        return sum(sizes[g] for g in self.groups)

    def slices(self) -> dict[IdentifiableGroup, slice]:
        sizes = self.size_per_group()
        out: dict[IdentifiableGroup, slice] = {}
        start = 0
        for g in self.groups:
            out[g] = slice(start, start + sizes[g])
            start += sizes[g]
        return out

    def initial_vector(self) -> NDArray[np.float64]:
        """Zero perturbation from the nominal robot."""
        return np.zeros(self.total_size(), dtype=np.float64)

    def bounds_vectors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        sizes = self.size_per_group()
        lo = np.empty(self.total_size(), dtype=np.float64)
        hi = np.empty(self.total_size(), dtype=np.float64)
        offset = 0
        bound_map = {
            IdentifiableGroup.ANCHOR_OFFSETS: self.bounds.anchor_offset_m,
            IdentifiableGroup.ATTACHMENT_OFFSETS: self.bounds.attachment_offset_m,
            IdentifiableGroup.CABLE_LENGTH_OFFSETS: self.bounds.cable_length_offset_m,
        }
        for g in self.groups:
            n = sizes[g]
            lo[offset : offset + n] = -bound_map[g]
            hi[offset : offset + n] = bound_map[g]
            offset += n
        return lo, hi

    # --- structured readers -------------------------------------------

    def anchor_offsets(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        sl = self.slices().get(IdentifiableGroup.ANCHOR_OFFSETS)
        if sl is None:
            return np.zeros((self.n_cables, 3))
        return x[sl].reshape(self.n_cables, 3)

    def attachment_offsets(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        sl = self.slices().get(IdentifiableGroup.ATTACHMENT_OFFSETS)
        if sl is None:
            return np.zeros((self.n_cables, 3))
        return x[sl].reshape(self.n_cables, 3)

    def cable_length_offsets(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        sl = self.slices().get(IdentifiableGroup.CABLE_LENGTH_OFFSETS)
        if sl is None:
            return np.zeros(self.n_cables)
        return x[sl]
