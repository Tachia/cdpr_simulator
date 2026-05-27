r"""Cleaning operations: NaN handling, deduplication, robust outlier detection.

All operations are pure functions taking ``(DataFrame, ColumnMap)`` and
returning ``(new DataFrame, diagnostics dict)``. They never mutate their
inputs. The Pipeline composes them and tags each call with a
:class:`StepRecord`.

For outlier detection we use a per-channel **median absolute deviation**
(MAD) test, which is the standard robust alternative to z-scoring (the
mean and standard deviation are themselves vulnerable to outliers, so a
contaminated sensor stream would flag almost nothing under a z-score test
that uses the same data). The scaled MAD :math:`\hat\sigma = 1.4826\,
\mathrm{MAD}` is a consistent estimator of the Gaussian standard deviation
under uncontaminated data, so a 3.5-MAD threshold is roughly the same
"unusualness" level as a 3.5-sigma threshold but ten times more robust to
contamination (Leys et al., *JESP* 2013).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from cdpr.ingest.containers import ColumnMap

if TYPE_CHECKING:                                           # pragma: no cover
    import pandas as pd


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def drop_nan_rows(
    frame: "pd.DataFrame", columns: ColumnMap,
) -> tuple["pd.DataFrame", dict[str, int]]:
    """Drop rows where any mapped column contains NaN.

    Unmapped columns are ignored --- they may contain auxiliary instrument
    data the user does not care about.
    """
    cols = [c for cols_for_ch in columns.assigned_columns().values() for c in cols_for_ch]
    if not cols:
        return frame, {"dropped": 0}
    mask = ~frame[cols].isna().any(axis=1)
    out = frame.loc[mask].reset_index(drop=True)
    return out, {"dropped": int((~mask).sum())}


def interpolate_nans(
    frame: "pd.DataFrame", columns: ColumnMap, *, method: str = "linear",
) -> tuple["pd.DataFrame", dict[str, int]]:
    """Fill NaN gaps in mapped columns by interpolation along time.

    Requires ``columns.time`` to be set. The method string is forwarded to
    :meth:`pandas.DataFrame.interpolate`; ``"linear"`` is the safe default
    for laboratory motion-capture gaps (a few missing frames).
    """
    if columns.time is None:
        raise ValueError("interpolate_nans requires ColumnMap.time to be set.")
    cols = [c for cols_for_ch in columns.assigned_columns().values() for c in cols_for_ch]
    cols = [c for c in cols if c != columns.time]
    out = frame.copy()
    n_before = int(out[cols].isna().sum().sum())
    out[cols] = out[cols].interpolate(method=method, limit_direction="both")
    n_after = int(out[cols].isna().sum().sum())
    return out, {"nans_before": n_before, "nans_after": n_after}


# ---------------------------------------------------------------------------
# Duplicate timestamps
# ---------------------------------------------------------------------------

def deduplicate_timestamps(
    frame: "pd.DataFrame",
    columns: ColumnMap,
    *,
    strategy: Literal["first", "mean"] = "mean",
) -> tuple["pd.DataFrame", dict[str, int]]:
    """Drop or average rows that share an identical timestamp.

    Many tracking systems occasionally emit duplicate frames; averaging
    preserves the signal estimate, ``first`` is a faster fallback.
    """
    if columns.time is None:
        raise ValueError("deduplicate_timestamps requires ColumnMap.time to be set.")
    import pandas as pd
    t = columns.time
    if strategy == "first":
        out = frame.drop_duplicates(subset=[t], keep="first").reset_index(drop=True)
    else:
        out = frame.groupby(t, as_index=False, sort=True).mean(numeric_only=True)
    return out.reset_index(drop=True), {
        "rows_before": int(len(frame)),
        "rows_after": int(len(out)),
    }


# ---------------------------------------------------------------------------
# MAD outlier flagging
# ---------------------------------------------------------------------------

def _mad_z(series: np.ndarray) -> np.ndarray:
    """Robust z-score using the scaled median absolute deviation."""
    median = np.median(series)
    mad = np.median(np.abs(series - median))
    if mad == 0:
        # Fall back to the IQR-based MAD; if still zero the channel is constant.
        q75, q25 = np.percentile(series, [75, 25])
        mad = (q75 - q25) / 1.349 or 1.0
    return (series - median) / (1.4826 * mad)


def remove_outliers_mad(
    frame: "pd.DataFrame",
    columns: ColumnMap,
    *,
    threshold: float = 3.5,
    action: Literal["drop", "nan"] = "nan",
) -> tuple["pd.DataFrame", dict[str, object]]:
    """Per-channel MAD outlier detection.

    Each numeric mapped column is robustly z-scored; samples whose absolute
    score exceeds ``threshold`` are flagged. With ``action="nan"`` the
    flagged samples are set to NaN (so a later
    :func:`interpolate_nans` call can fill them); with ``action="drop"``
    the whole row is removed.
    """
    out = frame.copy()
    flagged_per_column: dict[str, int] = {}

    cols: list[str] = []
    for cols_for_ch in columns.assigned_columns().values():
        cols.extend(c for c in cols_for_ch if c != columns.time)

    flag_mask = np.zeros(len(out), dtype=bool)
    for c in cols:
        x = out[c].to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(x)
        if finite.sum() < 4:
            flagged_per_column[c] = 0
            continue
        z = np.full_like(x, np.nan)
        z[finite] = _mad_z(x[finite])
        mask_c = np.abs(z) > threshold
        flagged_per_column[c] = int(mask_c.sum())
        if action == "nan":
            out.loc[mask_c, c] = np.nan
        else:
            flag_mask |= mask_c

    if action == "drop":
        kept = ~flag_mask
        out = out.loc[kept].reset_index(drop=True)
        rows_dropped = int(flag_mask.sum())
    else:
        rows_dropped = 0

    return out, {
        "threshold": threshold,
        "flagged_per_column": flagged_per_column,
        "rows_dropped": rows_dropped,
        "action": action,
    }
