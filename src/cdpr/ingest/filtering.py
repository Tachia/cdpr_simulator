r"""Signal filtering for experimental data.

Two filters are exposed:

* :func:`lowpass_butterworth` -- a zero-phase Butterworth low-pass via
  :func:`scipy.signal.filtfilt`. Zero-phase means the filter is applied
  forwards and backwards, so no group-delay distortion is introduced.
  This is the right choice for offline post-processing where causality is
  not required, which is exactly the case for ingest.
* :func:`savitzky_golay` -- polynomial-fit smoothing with optional
  derivative extraction. Preserves higher-order moments (useful when the
  same filtered series is going to be differentiated to recover velocity
  or acceleration) and has explicit control over window length and
  polynomial order.

Both filters require a uniform time grid, so call :func:`resample_uniform`
first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

from cdpr.ingest.containers import ColumnMap

if TYPE_CHECKING:                                           # pragma: no cover
    import pandas as pd


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _uniform_dt(time: np.ndarray, *, atol: float = 1e-6) -> float:
    diffs = np.diff(time)
    dt = float(diffs.mean())
    if dt <= 0:
        raise ValueError("Non-positive time step in filter input.")
    if diffs.std() > atol * dt:
        raise ValueError(
            f"Filtering requires a uniform time grid; "
            f"time-step std/mean = {diffs.std() / dt:.3e}. "
            "Call resample_uniform() first."
        )
    return dt


def _filterable_columns(columns: ColumnMap) -> set[str]:
    """Column names eligible for filtering (everything mapped except time)."""
    out: set[str] = set()
    for ch, cols in columns.assigned_columns().items():
        if ch.value == "time":
            continue
        out.update(cols)
    return out


# ---------------------------------------------------------------------------
# Butterworth low-pass
# ---------------------------------------------------------------------------

def lowpass_butterworth(
    frame: "pd.DataFrame",
    columns: ColumnMap,
    *,
    cutoff_hz: float,
    order: int = 4,
    only_channels: list[str] | None = None,
) -> tuple["pd.DataFrame", dict[str, object]]:
    """Zero-phase Butterworth low-pass at ``cutoff_hz`` (Hz).

    The cutoff is the half-amplitude frequency; ``order=4`` gives a
    24 dB/octave roll-off and is a sensible default for laboratory motion
    capture. ``only_channels`` restricts the filtering to specific column
    names (e.g. apply to positions but not to tensions).
    """
    import pandas as pd

    if columns.time is None:
        raise ValueError("lowpass_butterworth requires ColumnMap.time to be set.")
    t = frame[columns.time].to_numpy(dtype=np.float64)
    dt = _uniform_dt(t)
    fs = 1.0 / dt
    nyq = 0.5 * fs
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        raise ValueError(
            f"cutoff_hz must lie in (0, {nyq:.3f}) for fs={fs:.3f} Hz; got {cutoff_hz}."
        )

    b, a = butter(order, cutoff_hz / nyq, btype="low")
    target = set(only_channels) if only_channels else _filterable_columns(columns)

    out = frame.copy()
    actually_filtered: list[str] = []
    for c in target:
        if c not in out.columns:
            continue
        x = out[c].to_numpy(dtype=np.float64)
        # filtfilt requires len(x) > 3 * max(len(a), len(b)).
        if len(x) > 3 * max(len(a), len(b)):
            out[c] = filtfilt(b, a, x)
            actually_filtered.append(c)

    return out, {
        "cutoff_hz": cutoff_hz,
        "order": order,
        "fs": fs,
        "filtered_columns": sorted(actually_filtered),
        "skipped": sorted(set(target) - set(actually_filtered)),
    }


# ---------------------------------------------------------------------------
# Savitzky-Golay
# ---------------------------------------------------------------------------

def savitzky_golay(
    frame: "pd.DataFrame",
    columns: ColumnMap,
    *,
    window_length: int,
    polyorder: int = 3,
    deriv: int = 0,
    only_channels: list[str] | None = None,
) -> tuple["pd.DataFrame", dict[str, object]]:
    """Savitzky--Golay polynomial smoothing / differentiation.

    ``deriv=0`` smooths in place. ``deriv=1`` writes the first time
    derivative of each filtered column into a new ``"<name>_d1"`` column
    (and similarly ``"_d2"`` for ``deriv=2``); this is the right way to
    extract velocity / acceleration from noisy position data because the
    polynomial fit damps the high-frequency content that finite
    differencing would amplify.
    """
    if columns.time is None:
        raise ValueError("savitzky_golay requires ColumnMap.time to be set.")
    if window_length <= polyorder:
        raise ValueError("window_length must exceed polyorder.")
    if window_length % 2 == 0:
        raise ValueError("window_length must be odd (scipy convention).")

    t = frame[columns.time].to_numpy(dtype=np.float64)
    dt = _uniform_dt(t)
    target = set(only_channels) if only_channels else _filterable_columns(columns)

    out = frame.copy()
    written: list[str] = []
    for c in target:
        if c not in out.columns or len(out) < window_length:
            continue
        y = out[c].to_numpy(dtype=np.float64)
        smoothed = savgol_filter(y, window_length, polyorder, deriv=deriv, delta=dt)
        if deriv == 0:
            out[c] = smoothed
            written.append(c)
        else:
            new_col = f"{c}_d{deriv}"
            out[new_col] = smoothed
            written.append(new_col)

    return out, {
        "window_length": window_length,
        "polyorder": polyorder,
        "deriv": deriv,
        "dt": dt,
        "written": sorted(written),
    }
