r"""Time scalings :math:`s(t)` that retime a unit-parameter path.

A scaling exposes three methods --- ``s(t)``, ``s_dot(t)``, ``s_ddot(t)`` ---
so trajectories can hand both pose and pose-derivatives to a controller. All
implementations are vectorised: pass either a scalar or a NumPy array of
times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@runtime_checkable
class TimeScaling(Protocol):
    """Common interface for time-scaling functions."""

    duration: float

    def s(self, t: ArrayLike) -> NDArray[np.float64]: ...
    def s_dot(self, t: ArrayLike) -> NDArray[np.float64]: ...
    def s_ddot(self, t: ArrayLike) -> NDArray[np.float64]: ...


@dataclass(slots=True, frozen=True)
class LinearScaling:
    r"""Constant-rate scaling :math:`s(t) = t/T`.

    Velocity is constant; acceleration is zero in the interior with Dirac
    spikes at the endpoints. Useful for analytical baselines but rarely
    used for executed CDPR motions.
    """
    duration: float

    def s(self, t: ArrayLike) -> NDArray[np.float64]:
        return np.clip(np.asarray(t, dtype=np.float64) / self.duration, 0.0, 1.0)

    def s_dot(self, t: ArrayLike) -> NDArray[np.float64]:
        tt = np.asarray(t, dtype=np.float64)
        return np.where((tt >= 0) & (tt <= self.duration), 1.0 / self.duration, 0.0)

    def s_ddot(self, t: ArrayLike) -> NDArray[np.float64]:
        return np.zeros_like(np.asarray(t, dtype=np.float64))


@dataclass(slots=True, frozen=True)
class QuinticScaling:
    r"""Quintic polynomial :math:`s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5`, :math:`\tau = t/T`.

    The standard choice for smooth point-to-point motion: zero velocity and
    zero acceleration at both endpoints, monotonic interior. Maximum
    velocity is :math:`15/(8T)` of the path length per second; peak
    acceleration occurs at :math:`\tau = 1/2`.
    """
    duration: float

    def s(self, t: ArrayLike) -> NDArray[np.float64]:
        tau = np.clip(np.asarray(t, dtype=np.float64) / self.duration, 0.0, 1.0)
        return 10 * tau**3 - 15 * tau**4 + 6 * tau**5

    def s_dot(self, t: ArrayLike) -> NDArray[np.float64]:
        T = self.duration
        tau = np.clip(np.asarray(t, dtype=np.float64) / T, 0.0, 1.0)
        return (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / T

    def s_ddot(self, t: ArrayLike) -> NDArray[np.float64]:
        T = self.duration
        tau = np.clip(np.asarray(t, dtype=np.float64) / T, 0.0, 1.0)
        return (60 * tau - 180 * tau**2 + 120 * tau**3) / (T * T)


@dataclass(slots=True, frozen=True)
class TrapezoidalScaling:
    r"""Symmetric trapezoidal-velocity (bang--coast--bang) scaling.

    Phases of duration :math:`(t_a, T - 2t_a, t_a)` accelerate at constant
    :math:`\dot{s}_\text{max}/t_a`, coast at :math:`\dot{s}_\text{max}`, then
    decelerate symmetrically. For a unit-length path the cruise rate is
    determined by :math:`\dot{s}_\text{max} = 1/(T - t_a)`. Setting
    :math:`t_a = T/2` recovers a triangular velocity profile (no coasting).

    The acceleration is piecewise constant and discontinuous; for jerk-bound
    motions a higher-order scaling is appropriate, but this version is the
    correct one for replicating classical CDPR trajectory benchmarks.
    """
    duration: float
    accel_time: float

    def __post_init__(self) -> None:
        if not 0.0 < self.accel_time <= self.duration / 2.0:
            raise ValueError(
                "accel_time must satisfy 0 < t_a <= duration/2; "
                f"got {self.accel_time} for duration {self.duration}"
            )

    def _v_peak(self) -> float:
        return 1.0 / (self.duration - self.accel_time)

    def s(self, t: ArrayLike) -> NDArray[np.float64]:
        T, t_a = self.duration, self.accel_time
        v = self._v_peak()
        a = v / t_a
        tt = np.clip(np.asarray(t, dtype=np.float64), 0.0, T)
        out = np.empty_like(tt)
        m1 = tt < t_a
        m2 = (tt >= t_a) & (tt <= T - t_a)
        m3 = tt > T - t_a
        out[m1] = 0.5 * a * tt[m1] ** 2
        out[m2] = v * (tt[m2] - 0.5 * t_a)
        td = T - tt[m3]
        out[m3] = 1.0 - 0.5 * a * td * td
        return out

    def s_dot(self, t: ArrayLike) -> NDArray[np.float64]:
        T, t_a = self.duration, self.accel_time
        v = self._v_peak()
        a = v / t_a
        tt = np.clip(np.asarray(t, dtype=np.float64), 0.0, T)
        out = np.empty_like(tt)
        m1 = tt < t_a
        m2 = (tt >= t_a) & (tt <= T - t_a)
        m3 = tt > T - t_a
        out[m1] = a * tt[m1]
        out[m2] = v
        out[m3] = a * (T - tt[m3])
        return out

    def s_ddot(self, t: ArrayLike) -> NDArray[np.float64]:
        T, t_a = self.duration, self.accel_time
        v = self._v_peak()
        a = v / t_a
        tt = np.asarray(t, dtype=np.float64)
        out = np.zeros_like(tt)
        out[(tt > 0) & (tt < t_a)] = a
        out[(tt > T - t_a) & (tt < T)] = -a
        return out
