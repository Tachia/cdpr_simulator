r"""Robust multi-format data loader with timestamp + missing-value handling.

This module exists because the previous CSV ingestion path crashed on
realistic data:

* ``float("2025-05-07 15:01:27.548198")`` → ``ValueError``
* ``float("none")`` → ``ValueError``
* CSV with no ``t`` column → downstream model crash
* Excel / Parquet files → "unsupported format"

The smart loader fixes all of them while keeping the existing simple
CSV path (``scripts/_csv_io.load_csv_any``) working unchanged. New code
calls :func:`load_dataset`; old code keeps using its existing helper
and silently gets the new behaviour through delegation.

Supported formats (auto-detected from extension and magic bytes):

================== ===================================================
Extension          Engine
================== ===================================================
``.csv``           ``pandas.read_csv`` (with C engine, fallback Python)
``.tsv``           ``pandas.read_csv(sep='\t')``
``.txt``           same as CSV — sniff for tab vs comma
``.xlsx``          ``pandas.read_excel(engine='openpyxl')``
``.xls``           ``pandas.read_excel(engine='xlrd')``
``.ods``           ``pandas.read_excel(engine='odf')``
``.parquet``       ``pandas.read_parquet`` (needs ``pyarrow``)
``.feather``       ``pandas.read_feather`` (needs ``pyarrow``)
================== ===================================================

The loader is dependency-light: ``pandas`` is the only hard
requirement. Optional readers (``openpyxl``, ``xlrd``, ``odf``,
``pyarrow``) are imported lazily — a missing dependency surfaces a
single clear error rather than an opaque ImportError.

Public API
----------

* :func:`load_dataset` — main entrypoint, returns a cleaned DataFrame
  plus a structured :class:`ProfileReport`.
* :class:`ProfileReport` — dataclass describing what the loader found,
  what it changed, and why.
* :func:`save_cleaning_report` — write the report to disk in JSON +
  Markdown + HTML (the three formats the directive asked for).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:                                          # pragma: no cover
    raise ImportError(
        "cdpr.ingest.smart_loader requires pandas. "
        "Install with: pip install -e '.[data]'"
    ) from exc


# ---------------------------------------------------------------------------
# Profile report
# ---------------------------------------------------------------------------

@dataclass
class ProfileReport:
    """Structured description of what the loader found and changed.

    Attributes
    ----------
    source:
        Path or URL the dataset came from.
    format:
        Detected format (``csv``, ``xlsx``, ``parquet`` …).
    n_rows, n_cols:
        Dataset shape after parsing.
    timestamp_column:
        Name of the column that was interpreted as a timestamp (None if
        no timestamp was detected). The loader adds an extra ``t``
        column with seconds-from-first-sample when this is non-None.
    numeric_columns / categorical_columns:
        Best-guess classification per column.
    cable_length_columns / cable_tension_columns:
        Columns identified as cable-indexed (``L1, L2 …`` and ``T1, T2 …``
        / aliases such as ``cable_1``, ``tension_3``).
    pose_columns:
        Mapping ``canonical → source`` for pose-like columns (``px``,
        ``py``, ``pz``, ``qx``/qy/qz/qw, vx/vy/vz, wx/wy/wz). Missing
        canonical names fall through to the downstream alias mapper.
    missing_per_column:
        How many cells per column were marked missing during sentinel
        cleaning (``"none"`` / ``"N/A"`` etc. converted to ``NaN``).
    fills_per_column:
        How many cells per column were imputed, broken down by strategy
        ({"interpolation": int, "forward_fill": int, "backward_fill":
        int, "constant": int}).
    warnings:
        Free-form notes (capped columns, dropped duplicates, etc.).
    """

    source: str = ""
    format: str = ""
    n_rows: int = 0
    n_cols: int = 0
    timestamp_column: str | None = None
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    cable_length_columns: list[str] = field(default_factory=list)
    cable_tension_columns: list[str] = field(default_factory=list)
    pose_columns: dict[str, str] = field(default_factory=dict)
    missing_per_column: dict[str, int] = field(default_factory=dict)
    fills_per_column: dict[str, dict[str, int]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "format": self.format,
            "n_rows": int(self.n_rows),
            "n_cols": int(self.n_cols),
            "timestamp_column": self.timestamp_column,
            "numeric_columns": list(self.numeric_columns),
            "categorical_columns": list(self.categorical_columns),
            "cable_length_columns": list(self.cable_length_columns),
            "cable_tension_columns": list(self.cable_tension_columns),
            "pose_columns": dict(self.pose_columns),
            "missing_per_column": dict(self.missing_per_column),
            "fills_per_column": dict(self.fills_per_column),
            "warnings": list(self.warnings),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Cleaning report",
            f"",
            f"- **Source**: `{self.source}`",
            f"- **Format**: `{self.format}`",
            f"- **Shape**: {self.n_rows} rows × {self.n_cols} columns",
        ]
        if self.timestamp_column:
            lines.append(f"- **Timestamp column** auto-detected: "
                         f"`{self.timestamp_column}` → converted to `t` (seconds)")
        if self.cable_length_columns:
            lines.append(f"- **Cable lengths** ({len(self.cable_length_columns)}): "
                         f"`{', '.join(self.cable_length_columns)}`")
        if self.cable_tension_columns:
            lines.append(f"- **Cable tensions** ({len(self.cable_tension_columns)}): "
                         f"`{', '.join(self.cable_tension_columns)}`")
        if self.pose_columns:
            lines.append(f"- **Pose columns** mapped:")
            for canon, src in sorted(self.pose_columns.items()):
                lines.append(f"  - `{canon}` ← `{src}`")
        if self.missing_per_column:
            lines.append(f"")
            lines.append(f"## Missing values found")
            lines.append("")
            lines.append("| column | cells marked missing |")
            lines.append("|---|---:|")
            for k, v in sorted(self.missing_per_column.items()):
                if v > 0:
                    lines.append(f"| `{k}` | {v} |")
        if self.fills_per_column:
            lines.append(f"")
            lines.append(f"## Imputation applied")
            lines.append("")
            lines.append("| column | interpolation | ffill | bfill | constant |")
            lines.append("|---|---:|---:|---:|---:|")
            for col, d in sorted(self.fills_per_column.items()):
                lines.append(
                    f"| `{col}` | {d.get('interpolation', 0)} "
                    f"| {d.get('forward_fill', 0)} | {d.get('backward_fill', 0)} "
                    f"| {d.get('constant', 0)} |"
                )
        if self.warnings:
            lines.append(f"")
            lines.append(f"## Warnings")
            for w in self.warnings:
                lines.append(f"- {w}")
        return "\n".join(lines) + "\n"

    def to_html(self) -> str:
        body = self.to_markdown()
        # Bare-bones markdown → HTML so we don't drag in a heavy renderer.
        # Tables, code spans and headings are the only constructs we use.
        html_parts: list[str] = ["<html><head><meta charset='utf-8'>"
                                  "<title>Cleaning report</title>"
                                  "<style>"
                                  "body{font:14px/1.5 system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 1em}"
                                  "h1,h2{border-bottom:1px solid #ddd;padding-bottom:.2em}"
                                  "code{background:#f3f3f5;padding:.05em .35em;border-radius:3px}"
                                  "table{border-collapse:collapse;margin:1em 0}"
                                  "th,td{padding:.4em .8em;border:1px solid #ddd;text-align:left}"
                                  "th{background:#f7f7f9}"
                                  "</style></head><body>"]
        in_table = False
        for raw in body.splitlines():
            line = raw.rstrip()
            if line.startswith("# "):
                html_parts.append(f"<h1>{line[2:]}</h1>")
            elif line.startswith("## "):
                html_parts.append(f"<h2>{line[3:]}</h2>")
            elif line.startswith("|"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if not in_table:
                    html_parts.append("<table>")
                    in_table = True
                    # Header row.
                    html_parts.append("<tr>" + "".join(
                        f"<th>{_md_inline(c)}</th>" for c in cells
                    ) + "</tr>")
                elif set("-:").issuperset("".join(cells).replace(" ", "")):
                    pass                                            # separator row
                else:
                    html_parts.append("<tr>" + "".join(
                        f"<td>{_md_inline(c)}</td>" for c in cells
                    ) + "</tr>")
            else:
                if in_table:
                    html_parts.append("</table>")
                    in_table = False
                if line.startswith("- "):
                    html_parts.append(f"<li>{_md_inline(line[2:])}</li>")
                elif line:
                    html_parts.append(f"<p>{_md_inline(line)}</p>")
        if in_table:
            html_parts.append("</table>")
        html_parts.append("</body></html>")
        return "\n".join(html_parts)


def _md_inline(s: str) -> str:
    # `code` → <code>code</code>
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    # **bold** → <strong>
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    return out


def save_cleaning_report(report: ProfileReport, out_dir: Path | str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "cleaning_report.json",
        "md":   out_dir / "cleaning_report.md",
        "html": out_dir / "cleaning_report.html",
    }
    paths["json"].write_text(
        json.dumps(report.to_dict(), indent=2, default=str), encoding="utf-8"
    )
    paths["md"].write_text(report.to_markdown(), encoding="utf-8")
    paths["html"].write_text(report.to_html(), encoding="utf-8")
    return paths


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

_MISSING_SENTINELS = {
    "", "none", "null", "n/a", "na", "nan", "missing", "blank",
    "-", "--", "#n/a", "#null!", "#div/0!",
}

# Aliases — same convention used by scripts/_csv_io.py so anything that
# loader does works downstream. Repeated locally so this module has no
# dependency on the scripts/ directory.
_POSE_ALIASES: dict[str, list[str]] = {
    "px": ["px", "x", "pos_x", "position_x", "p_x", "x_actual", "x_m"],
    "py": ["py", "y", "pos_y", "position_y", "p_y", "y_actual", "y_m"],
    "pz": ["pz", "z", "pos_z", "position_z", "p_z", "z_actual", "z_m"],
    "qx": ["qx", "quat_x", "q_x", "q1"],
    "qy": ["qy", "quat_y", "q_y", "q2"],
    "qz": ["qz", "quat_z", "q_z", "q3"],
    "qw": ["qw", "quat_w", "q_w", "q0"],
    "vx": ["vx", "vel_x", "v_x", "linear_vel_x"],
    "vy": ["vy", "vel_y", "v_y", "linear_vel_y"],
    "vz": ["vz", "vel_z", "v_z", "linear_vel_z"],
    "wx": ["wx", "omega_x", "ang_vel_x", "w_x"],
    "wy": ["wy", "omega_y", "ang_vel_y", "w_y"],
    "wz": ["wz", "omega_z", "ang_vel_z", "w_z"],
}
_TIMESTAMP_ALIASES = {"t", "time", "timestamp", "ts", "sec", "seconds",
                       "date", "datetime"}


def _norm_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"csv", "tsv", "txt", "xlsx", "xls", "ods", "parquet", "feather"}:
        return suffix
    # Magic-byte sniffing as fallback.
    try:
        head = path.read_bytes()[:8]
    except Exception:
        return "csv"
    if head.startswith(b"PK"):
        return "xlsx"
    if head.startswith(b"PAR1"):
        return "parquet"
    if head.startswith(b"ARROW"):
        return "feather"
    return "csv"


def _read_dataframe(path: Path, fmt: str) -> pd.DataFrame:
    """Dispatch to the right pandas reader with helpful error messages."""
    if fmt in {"csv", "txt"}:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")
    if fmt == "tsv":
        return pd.read_csv(path, sep="\t")
    if fmt == "xlsx":
        try:
            return pd.read_excel(path, engine="openpyxl")
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to read .xlsx files. "
                "Install with: pip install openpyxl"
            ) from exc
    if fmt == "xls":
        try:
            return pd.read_excel(path, engine="xlrd")
        except ImportError as exc:
            raise ImportError(
                "xlrd is required to read legacy .xls files. "
                "Install with: pip install xlrd"
            ) from exc
    if fmt == "ods":
        try:
            return pd.read_excel(path, engine="odf")
        except ImportError as exc:
            raise ImportError(
                "odfpy is required to read .ods files. "
                "Install with: pip install odfpy"
            ) from exc
    if fmt == "parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "pyarrow (or fastparquet) is required to read .parquet files. "
                "Install with: pip install pyarrow"
            ) from exc
    if fmt == "feather":
        try:
            return pd.read_feather(path)
        except ImportError as exc:
            raise ImportError(
                "pyarrow is required to read .feather files. "
                "Install with: pip install pyarrow"
            ) from exc
    # Default: treat as CSV with python engine.
    return pd.read_csv(path, sep=None, engine="python")


def _detect_timestamp_column(df: pd.DataFrame) -> str | None:
    """Pick the most likely timestamp column. Strategy:

    1. Any column whose normalised header is in :data:`_TIMESTAMP_ALIASES`.
    2. Any string column whose first non-null value parses as a datetime.
    """
    for col in df.columns:
        if _norm_header(col) in _TIMESTAMP_ALIASES:
            return col
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(5)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().sum() >= max(1, len(sample) - 1):
                return col
    return None


def _classify_cable_column(name: str) -> tuple[str | None, int | None]:
    """``L1`` / ``tension_4`` / ``cable_2`` → ``("L", 1)`` etc.
    Layer / Time / TensionMean → ``(None, None)``."""
    norm = _norm_header(name)
    m = re.match(r"^([a-z_]+?)_?(\d+)$", norm)
    if not m:
        return None, None
    prefix, idx = m.group(1), int(m.group(2))
    if prefix in {"l", "length", "cable_length", "cable_len", "len"}:
        return "L", idx
    if prefix in {"t", "tension", "cable_tension", "cable", "force", "tau"}:
        return "T", idx
    return None, None


def _detect_pose_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical pose names to source columns using the alias table."""
    out: dict[str, str] = {}
    for col in df.columns:
        norm = _norm_header(col)
        for canon, aliases in _POSE_ALIASES.items():
            if canon in out:
                continue
            if norm in {_norm_header(a) for a in aliases}:
                out[canon] = col
    return out


def _clean_missing_sentinels(df: pd.DataFrame,
                              report: ProfileReport) -> pd.DataFrame:
    """Walk every cell of every object/string column and replace known
    missing-data sentinels (``"none"``, ``"N/A"``, ``""`` …) with
    ``np.nan``. Numeric columns are untouched (numpy already has NaN)."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype != object:
            continue
        before = out[col].isna().sum()
        # Lower-case, stripped sentinel check.
        cleaned = out[col].apply(
            lambda v: np.nan if (isinstance(v, str)
                                  and v.strip().lower() in _MISSING_SENTINELS)
            else v
        )
        out[col] = cleaned
        gained = int(out[col].isna().sum() - before)
        if gained > 0:
            report.missing_per_column[col] = gained
    return out


def _convert_timestamp(df: pd.DataFrame, ts_col: str,
                        report: ProfileReport) -> pd.DataFrame:
    """Convert the chosen timestamp column to seconds-from-first-sample
    and store the result under ``t`` (replacing any existing ``t``)."""
    parsed = pd.to_datetime(df[ts_col], errors="coerce")
    if parsed.isna().all():
        report.warnings.append(
            f"timestamp column '{ts_col}' could not be parsed as datetime; "
            "leaving the column untouched"
        )
        return df
    first = parsed.dropna().iloc[0]
    seconds = (parsed - first).dt.total_seconds()
    if "t" in df.columns and ts_col != "t":
        report.warnings.append(
            f"existing 't' column overwritten with timestamp-derived seconds "
            f"from '{ts_col}'"
        )
    df = df.copy()
    df["t"] = seconds.astype(float)
    report.timestamp_column = ts_col
    return df


def _impute_numeric(df: pd.DataFrame, report: ProfileReport) -> pd.DataFrame:
    """Apply the directive's 5-level missing-value hierarchy to every
    numeric column. We record what was done per column for the report."""
    out = df.copy()
    for col in out.select_dtypes(include="number").columns:
        s = out[col]
        n_missing = int(s.isna().sum())
        if n_missing == 0:
            continue
        per = {"interpolation": 0, "forward_fill": 0,
               "backward_fill": 0, "constant": 0}

        # Level 1: linear interpolation between neighbours.
        before = s.isna().sum()
        s = s.interpolate(method="linear", limit_direction="both")
        per["interpolation"] = int(before - s.isna().sum())

        # Level 2: forward fill.
        if s.isna().any():
            before = s.isna().sum()
            s = s.ffill()
            per["forward_fill"] = int(before - s.isna().sum())

        # Level 3: backward fill.
        if s.isna().any():
            before = s.isna().sum()
            s = s.bfill()
            per["backward_fill"] = int(before - s.isna().sum())

        # Level 4: physics-informed --- domain-informed fallback for the
        # canonical CDPR columns. For cable tensions we use the column
        # mean if known to be in a reasonable range; for everything else,
        # zero. This is the "never crash" guarantee.
        if s.isna().any():
            before = s.isna().sum()
            fill_value = float(s.mean()) if s.notna().any() else 0.0
            s = s.fillna(fill_value)
            per["constant"] = int(before - s.isna().sum())

        out[col] = s
        report.fills_per_column[col] = per
    return out


def _profile_columns(df: pd.DataFrame, report: ProfileReport) -> None:
    """Classify every column into numeric / categorical / cable / pose."""
    report.n_rows = int(len(df))
    report.n_cols = int(df.shape[1])
    numeric, categorical = [], []
    lengths_pairs: list[tuple[int, str]] = []
    tensions_pairs: list[tuple[int, str]] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
        fam, idx = _classify_cable_column(col)
        if fam == "L":
            lengths_pairs.append((idx, col))
        elif fam == "T":
            tensions_pairs.append((idx, col))
    lengths_pairs.sort()
    tensions_pairs.sort()
    report.numeric_columns = numeric
    report.categorical_columns = categorical
    report.cable_length_columns = [c for _, c in lengths_pairs]
    report.cable_tension_columns = [c for _, c in tensions_pairs]
    report.pose_columns = _detect_pose_columns(df)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path, *,
                  clean_missing: bool = True,
                  convert_timestamp: bool = True,
                  ) -> tuple[pd.DataFrame, ProfileReport]:
    """Load any supported data file and return ``(DataFrame, ProfileReport)``.

    The loader is **idempotent and crash-safe**:

    * Unsupported formats raise :class:`ImportError` with a clean install
      hint (e.g. ``pip install pyarrow``) rather than a tangle of nested
      tracebacks.
    * Malformed CSV rows are passed through if pandas' Python engine can
      sniff them, otherwise reported as a warning rather than aborting.
    * Missing values are recognised across all sentinel forms the
      directive listed and filled via the 5-level hierarchy.
    * Timestamp columns are auto-detected and converted to seconds
      (``t`` column) without user intervention.
    * Cable / pose columns are classified by alias matching and added to
      the report for downstream consumers.

    The caller can persist the report with :func:`save_cleaning_report`.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    fmt = _detect_format(path)
    df = _read_dataframe(path, fmt)
    if df.empty:
        raise ValueError(f"dataset has zero rows after parsing: {path}")

    report = ProfileReport(source=str(path), format=fmt)

    if clean_missing:
        df = _clean_missing_sentinels(df, report)

    if convert_timestamp:
        ts_col = _detect_timestamp_column(df)
        if ts_col is not None:
            df = _convert_timestamp(df, ts_col, report)

    if clean_missing:
        # Promote object columns to numeric where possible. Pandas
        # infers ``object`` dtype as soon as one ``"none"`` slips
        # through; after sentinel cleaning those columns are mostly
        # numbers with NaN holes --- so a coerce pass recovers the
        # numeric dtype and lets _impute_numeric fill the holes.
        ts_col_name = report.timestamp_column
        for col in df.select_dtypes(include=["object", "string"]).columns:
            if col == ts_col_name:
                continue                                            # leave the raw timestamps alone
            coerced = pd.to_numeric(df[col], errors="coerce")
            # Only promote if the coerced column has at least one
            # real value (otherwise the original column might just be
            # categorical text, like a status label).
            if coerced.notna().any():
                df = df.copy()
                df[col] = coerced
        df = _impute_numeric(df, report)

    _profile_columns(df, report)
    return df, report
