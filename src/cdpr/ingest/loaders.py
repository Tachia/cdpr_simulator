r"""File-format readers.

All four supported formats land in the same :class:`RawDataset`:

* :func:`load_csv` -- pandas reader with sane defaults. Comments and blank
  lines are skipped; the first non-comment row becomes the header.
* :func:`load_xlsx` -- spreadsheet loader; pass ``sheet=`` to pick a named
  sheet other than the first.
* :func:`load_txt` -- whitespace-delimited reader with header detection.
* :func:`load_json` -- two layouts supported: a list-of-records (each row
  is one observation, keys become columns) and a columnar dict (each top-
  level key is a column whose value is the column's array).
* :func:`load_dataframe` -- adapter for callers that already hold a
  pandas DataFrame (e.g. fetched from a database or constructed in tests).

Pandas is required for all of these. It ships with the ``data`` extra
(``pip install 'cdpr[data]'``); :func:`_require_pandas` raises a clear
install hint when it is missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from cdpr.ingest.containers import RawDataset

if TYPE_CHECKING:                                           # pragma: no cover
    import pandas as pd


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "The ingest layer needs pandas. Install with:  pip install 'cdpr[data]'"
        ) from exc
    return pd


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

def load_dataframe(
    frame: "pd.DataFrame", *, source_path: str | Path | None = None, format: str = "dataframe"
) -> RawDataset:
    """Wrap an in-memory pandas DataFrame as a RawDataset.

    Useful for tests, for ingestion from a database, or when the caller has
    already done their own loading.
    """
    return RawDataset(
        frame=frame,
        source_path=Path(source_path) if source_path else None,
        format=format,
        n_rows_raw=int(len(frame)),
        header=[str(c) for c in frame.columns],
    )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def load_csv(
    path: str | Path,
    *,
    comment: str = "#",
    sep: str = ",",
    decimal: str = ".",
    header: int | None = 0,
) -> RawDataset:
    """Load a CSV file. ``header=None`` for headerless files."""
    pd = _require_pandas()
    path = Path(path)
    frame = pd.read_csv(
        path,
        comment=comment,
        sep=sep,
        decimal=decimal,
        header=header,
        skip_blank_lines=True,
    )
    return RawDataset(
        frame=frame,
        source_path=path,
        format="csv",
        n_rows_raw=int(len(frame)),
        header=[str(c) for c in frame.columns],
    )


# ---------------------------------------------------------------------------
# TXT (whitespace-delimited)
# ---------------------------------------------------------------------------

def load_txt(
    path: str | Path,
    *,
    comment: str = "#",
    header: int | None = 0,
) -> RawDataset:
    """Load a whitespace-delimited text file."""
    pd = _require_pandas()
    path = Path(path)
    frame = pd.read_csv(
        path,
        comment=comment,
        sep=r"\s+",
        header=header,
        engine="python",
        skip_blank_lines=True,
    )
    return RawDataset(
        frame=frame,
        source_path=path,
        format="txt",
        n_rows_raw=int(len(frame)),
        header=[str(c) for c in frame.columns],
    )


# ---------------------------------------------------------------------------
# XLSX
# ---------------------------------------------------------------------------

def load_xlsx(
    path: str | Path,
    *,
    sheet: str | int = 0,
    header: int | None = 0,
) -> RawDataset:
    """Load one sheet from an Excel workbook."""
    pd = _require_pandas()
    path = Path(path)
    frame = pd.read_excel(path, sheet_name=sheet, header=header)
    return RawDataset(
        frame=frame,
        source_path=path,
        format="xlsx",
        n_rows_raw=int(len(frame)),
        header=[str(c) for c in frame.columns],
    )


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def load_json(path: str | Path) -> RawDataset:
    """Load a JSON experiment log.

    Two layouts are accepted:

    * **List of records.** A JSON array whose elements are objects; each
      object becomes one row, keys become columns.
    * **Columnar dict.** A JSON object whose values are arrays of equal
      length; keys become column names.

    Anything else raises ``ValueError``.
    """
    pd = _require_pandas()
    path = Path(path)
    obj = json.loads(path.read_text())

    if isinstance(obj, list):
        if not obj:
            raise ValueError(f"{path}: empty JSON list")
        if not all(isinstance(r, dict) for r in obj):
            raise ValueError(f"{path}: JSON list must contain objects")
        frame = pd.DataFrame.from_records(obj)
    elif isinstance(obj, dict):
        lengths = {k: (len(v) if isinstance(v, list) else 1) for k, v in obj.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"{path}: columnar JSON has inconsistent column lengths: {lengths}"
            )
        frame = pd.DataFrame(obj)
    else:
        raise ValueError(f"{path}: JSON root must be a list or an object, got {type(obj).__name__}")

    return RawDataset(
        frame=frame,
        source_path=path,
        format="json",
        n_rows_raw=int(len(frame)),
        header=[str(c) for c in frame.columns],
    )
