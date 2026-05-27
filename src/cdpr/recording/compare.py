r"""Compare two experiments time-step by time-step.

The two experiments must share the same number of cables; their time
vectors are aligned by linear interpolation of the second onto the first.
The :class:`ComparisonReport` carries per-channel root-mean-square and peak
errors plus a per-step error trace, so callers can both quote summary
statistics and plot the time series of where the two diverged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.recording.replay import Experiment


@dataclass(slots=True)
class ChannelStats:
    """Per-channel RMS and peak error.

    ``rms`` is :math:`\\sqrt{\\langle \\lVert \\mathbf{a} - \\mathbf{b} \\rVert^2 \\rangle_t}`,
    ``peak`` is :math:`\\max_t \\lVert \\mathbf{a}(t) - \\mathbf{b}(t) \\rVert`.
    """

    name: str
    rms: float
    peak: float
    trace: NDArray[np.float64]    # per-step error magnitude


@dataclass(slots=True)
class ComparisonReport:
    """Pairwise diagnostics between two :class:`Experiment` instances."""

    time: NDArray[np.float64]
    position: ChannelStats
    orientation_deg: ChannelStats
    linear_velocity: ChannelStats
    cable_tension: ChannelStats
    cable_length: ChannelStats

    def summary(self) -> dict[str, dict[str, float]]:
        return {
            ch.name: {"rms": float(ch.rms), "peak": float(ch.peak)}
            for ch in (self.position, self.orientation_deg, self.linear_velocity,
                       self.cable_tension, self.cable_length)
        }


def _resample(t_src: NDArray[np.float64], y_src: NDArray[np.float64],
              t_target: NDArray[np.float64]) -> NDArray[np.float64]:
    """Linear interpolation of each column of ``y_src`` onto ``t_target``."""
    if y_src.ndim == 1:
        return np.interp(t_target, t_src, y_src)
    return np.column_stack([np.interp(t_target, t_src, y_src[:, j]) for j in range(y_src.shape[1])])


def _quat_angular_distance_deg(qa: NDArray[np.float64], qb: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-step rotation angle between two quaternion stacks, in degrees.

    Uses :math:`\\theta = 2\\arccos(|q_a \\cdot q_b|)`. Robust to the
    double-cover ambiguity via the absolute value.
    """
    dot = np.abs(np.sum(qa * qb, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))


def _stats(name: str, per_step_error: NDArray[np.float64]) -> ChannelStats:
    return ChannelStats(
        name=name,
        rms=float(np.sqrt(np.mean(per_step_error**2))),
        peak=float(np.max(per_step_error)),
        trace=per_step_error,
    )


def compare(a: "Experiment", b: "Experiment") -> ComparisonReport:
    """Pairwise comparison of two experiments on the common time interval.

    The time grid of ``a`` is the reference; ``b`` is linearly interpolated
    onto it. Both experiments must have the same number of cables.
    """
    if a.cable_tensions.shape[1] != b.cable_tensions.shape[1]:
        raise ValueError(
            f"cable count mismatch: {a.cable_tensions.shape[1]} vs {b.cable_tensions.shape[1]}"
        )

    t = a.time
    # Restrict to the overlap.
    t_lo = max(a.time[0], b.time[0])
    t_hi = min(a.time[-1], b.time[-1])
    mask = (t >= t_lo) & (t <= t_hi)
    t = t[mask]

    pos_a = a.positions[mask]
    pos_b = _resample(b.time, b.positions, t)
    quat_a = a.quaternions_xyzw[mask]
    quat_b = _resample(b.time, b.quaternions_xyzw, t)
    vel_a = a.linear_velocities[mask]
    vel_b = _resample(b.time, b.linear_velocities, t)
    ten_a = a.cable_tensions[mask]
    ten_b = _resample(b.time, b.cable_tensions, t)
    len_a = a.cable_lengths[mask]
    len_b = _resample(b.time, b.cable_lengths, t)

    pos_err = np.linalg.norm(pos_a - pos_b, axis=-1)
    rot_err = _quat_angular_distance_deg(quat_a, quat_b)
    vel_err = np.linalg.norm(vel_a - vel_b, axis=-1)
    ten_err = np.linalg.norm(ten_a - ten_b, axis=-1)
    len_err = np.linalg.norm(len_a - len_b, axis=-1)

    return ComparisonReport(
        time=t,
        position=_stats("position [m]", pos_err),
        orientation_deg=_stats("orientation [deg]", rot_err),
        linear_velocity=_stats("velocity [m/s]", vel_err),
        cable_tension=_stats("cable tension [N]", ten_err),
        cable_length=_stats("cable length [m]", len_err),
    )
