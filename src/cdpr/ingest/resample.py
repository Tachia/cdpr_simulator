r"""Timestamp alignment and uniform-grid resampling.

Most labs deliver time-stamped samples at irregular intervals --- a 240 Hz
nominal motion-capture stream actually arrives with frame-to-frame deltas
distributed between 3 ms and 6 ms. Phase-1 dynamics expects a uniform
time step; this module re-interpolates the data onto a regular grid.

Numeric channels are interpolated with cubic splines (smoother
derivatives, matters for downstream finite-difference velocity / wrench
analysis) by default. Quaternion channels, when identified by the
:class:`ColumnMap`, are interpolated with **SLERP** via
:class:`scipy.spatial.transform.Slerp` --- linear interpolation of the
four scalar components is not a valid rotation interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

from cdpr.ingest.containers import ColumnMap

if TYPE_CHECKING:                                           # pragma: no cover
    import pandas as pd


def _suggest_dt(t: np.ndarray) -> float:
    diffs = np.diff(t)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        raise ValueError("resample needs at least two distinct timestamps")
    return float(np.median(diffs))


def resample_uniform(
    frame: "pd.DataFrame",
    columns: ColumnMap,
    *,
    dt: float | None = None,
    method: Literal["cubic", "linear"] = "cubic",
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple["pd.DataFrame", dict[str, object]]:
    """Resample the DataFrame onto a uniform time grid.

    Parameters
    ----------
    dt:
        Target time step. When ``None``, set to the median of the raw
        sample-to-sample deltas --- the "natural" rate of the data.
    method:
        ``"cubic"`` (default) uses a natural cubic spline. ``"linear"`` is
        the safer choice when the input is itself piecewise linear (e.g.
        zero-order-hold encoder reads) or has obvious step discontinuities.
    t_start, t_end:
        Output grid bounds. Default to the input's first and last
        timestamps.
    """
    import pandas as pd

    if columns.time is None:
        raise ValueError("resample_uniform requires ColumnMap.time to be set.")

    t_col = columns.time
    t_raw = frame[t_col].to_numpy(dtype=np.float64)
    order = np.argsort(t_raw)
    if not np.all(order == np.arange(len(order))):
        frame = frame.iloc[order].reset_index(drop=True)
        t_raw = t_raw[order]

    t_start_v = float(t_raw[0]) if t_start is None else float(t_start)
    t_end_v = float(t_raw[-1]) if t_end is None else float(t_end)
    dt_v = float(dt) if dt is not None else _suggest_dt(t_raw)
    n = max(int(round((t_end_v - t_start_v) / dt_v)) + 1, 2)
    t_grid = t_start_v + dt_v * np.arange(n)

    interpolated: dict[str, np.ndarray] = {t_col: t_grid}

    # Drop the time column from "things to interpolate" --- and skip any
    # non-numeric column, which will simply be dropped from the resampled
    # output (the cleaning layer has already left only what we care about).
    target_cols = [
        c for c in frame.columns
        if c != t_col and np.issubdtype(frame[c].dtype, np.number)
    ]
    quat_cols = columns.quaternion

    # SLERP the quaternion if present --- handle it before the generic loop
    # so the same column names don't get linearly interpolated as well.
    if quat_cols is not None and all(c in target_cols for c in quat_cols):
        q_raw = frame[list(quat_cols)].to_numpy(dtype=np.float64)
        slerp = Slerp(t_raw, Rotation.from_quat(q_raw))
        clipped = np.clip(t_grid, t_raw[0], t_raw[-1])
        q_grid = slerp(clipped).as_quat()
        for k, c in enumerate(quat_cols):
            interpolated[c] = q_grid[:, k]
            target_cols.remove(c)

    # Numeric channels.
    for c in target_cols:
        y = frame[c].to_numpy(dtype=np.float64)
        if method == "cubic" and len(t_raw) >= 4:
            spline = CubicSpline(t_raw, y, extrapolate=False)
            yi = spline(t_grid)
            # Edge fill: clip then re-evaluate so we don't carry NaNs out.
            clipped = np.clip(t_grid, t_raw[0], t_raw[-1])
            yi = spline(clipped)
        else:
            yi = np.interp(t_grid, t_raw, y)
        interpolated[c] = yi

    out = pd.DataFrame({c: interpolated[c] for c in [t_col, *target_cols, *(quat_cols or ())]})
    diagnostics = {
        "dt": dt_v,
        "n_in": int(len(frame)),
        "n_out": int(n),
        "t_start": t_start_v,
        "t_end": t_end_v,
        "method": method,
        "slerp_quaternion": quat_cols is not None,
    }
    return out, diagnostics
