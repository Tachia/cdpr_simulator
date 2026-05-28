r"""Shared CSV ingestion and robot reconstruction for the Phase-2 scripts.

The directive's tag-line for this layer is "any CSV from anywhere, any
robot, any tension bounds". Two functions matter externally:

* :func:`load_csv_any` --- resolve a path or URL to a local file, parse
  it (numpy + csv, no pandas hard-dep), normalise the column names,
  apply alias mapping, and return a column-keyed dict of arrays plus a
  :class:`SchemaReport` saying which canonical columns were found,
  inferred, or missing.
* :func:`rebuild_robot` --- given a sibling ``manifest.json`` (or an
  explicit override file), reconstruct exactly the robot that produced
  the data, even when its geometry is not in the catalog (e.g. the
  dissertation 8-cable, or a user-supplied JSON config).

Everything is dependency-light so the helpers stay usable when the
heavy extras (torch / sb3) are not installed.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Canonical schema --- the columns the simulator writes, and the aliases
# we accept when ingesting third-party data. The dict maps canonical name
# to a list of acceptable aliases (case-insensitive).
# ---------------------------------------------------------------------------

CANONICAL_ALIASES: dict[str, list[str]] = {
    "t":      ["t", "time", "timestamp", "ts", "sec", "seconds"],
    "px":     ["px", "x", "pos_x", "position_x", "p_x", "x_actual", "x_m"],
    "py":     ["py", "y", "pos_y", "position_y", "p_y", "y_actual", "y_m"],
    "pz":     ["pz", "z", "pos_z", "position_z", "p_z", "z_actual", "z_m"],
    "qx":     ["qx", "quat_x", "q_x", "q1"],
    "qy":     ["qy", "quat_y", "q_y", "q2"],
    "qz":     ["qz", "quat_z", "q_z", "q3"],
    "qw":     ["qw", "quat_w", "q_w", "q0"],
    "vx":     ["vx", "vel_x", "v_x", "vel_lin_x", "linear_vel_x"],
    "vy":     ["vy", "vel_y", "v_y", "vel_lin_y", "linear_vel_y"],
    "vz":     ["vz", "vel_z", "v_z", "vel_lin_z", "linear_vel_z"],
    "wx":     ["wx", "omega_x", "ang_vel_x", "w_x"],
    "wy":     ["wy", "omega_y", "ang_vel_y", "w_y"],
    "wz":     ["wz", "omega_z", "ang_vel_z", "w_z"],
    "px_ref": ["px_ref", "x_ref", "x_des", "pos_x_desired"],
    "py_ref": ["py_ref", "y_ref", "y_des", "pos_y_desired"],
    "pz_ref": ["pz_ref", "z_ref", "z_des", "pos_z_desired"],
}

# Cable-indexed prefixes --- members of these families must end in a digit.
CABLE_LENGTH_PREFIXES = ("l", "length", "cable_length", "cable_len", "len")
# ``cable`` lives in the tension family because the most common
# convention in CDPR papers is ``cable_N`` -> per-cable tension /
# force. Explicit length columns are nearly always named
# ``length_N`` or ``L_N``.
CABLE_TENSION_PREFIXES = ("t", "tension", "cable_tension", "cable", "force", "tau")


@dataclass
class SchemaReport:
    """What the loader actually found versus what it expected."""

    canonical_mapping: dict[str, str] = field(default_factory=dict)  # canonical -> source column
    length_columns: list[str] = field(default_factory=list)
    tension_columns: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    n_samples: int = 0
    source: str = ""                                                 # path or URL

    def summary(self) -> str:
        lines = [
            f"source         = {self.source}",
            f"samples        = {self.n_samples}",
            f"cable_lengths  = {len(self.length_columns)}  ({', '.join(self.length_columns) or '<none>'})",
            f"cable_tensions = {len(self.tension_columns)}  ({', '.join(self.tension_columns) or '<none>'})",
            f"mapped         = {len(self.canonical_mapping)} canonical columns",
        ]
        if self.missing_required:
            lines.append(f"MISSING        = {', '.join(self.missing_required)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# URL / path resolution
# ---------------------------------------------------------------------------

def _looks_like_url(s: str) -> bool:
    """Cheap discriminator between a filesystem path and a URL."""
    try:
        p = urllib.parse.urlparse(str(s))
    except Exception:
        return False
    return p.scheme in {"http", "https"}


def resolve_csv_input(input_str: str) -> Path:
    """Resolve a CSV reference to a local :class:`Path`.

    Accepts:
      * a plain local path (returned as-is after expansion),
      * an ``http://`` / ``https://`` URL --- downloaded to a temp file
        (caller is responsible for the temp file lifetime; we mark it as
        read-only and leave it on disk so the user can re-use it).
    """
    s = str(input_str).strip()
    if _looks_like_url(s):
        return _download_to_tempfile(s)
    p = Path(os.path.expanduser(s))
    if not p.exists():
        raise FileNotFoundError(f"CSV not found at: {p}")
    return p


def _download_to_tempfile(url: str) -> Path:
    """Download the URL to a stable temp file and return its path.

    We do not delete the temp file --- keeping it lets the user re-run
    Phase-2 workflows without paying the download cost twice.
    """
    print(f"[csv_io] fetching {url} …", file=sys.stderr)
    suffix = Path(urllib.parse.urlparse(url).path).suffix or ".csv"
    fd, tmp_path = tempfile.mkstemp(prefix="cdpr_csv_", suffix=suffix)
    os.close(fd)
    try:
        # Some hosts reject the default urllib User-Agent.
        req = urllib.request.Request(
            url, headers={"User-Agent": "cdpr-csv-loader/1.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
            out.write(resp.read())
    except Exception as exc:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass
        raise RuntimeError(f"could not download {url!r}: {exc}") from exc
    print(f"[csv_io] saved to {tmp_path}", file=sys.stderr)
    return Path(tmp_path)


# ---------------------------------------------------------------------------
# Column header normalisation
# ---------------------------------------------------------------------------

def _norm_header(name: str) -> str:
    """Lower-case and strip non-alphanumerics so ``Position X`` and
    ``position_x`` collapse to the same alias key."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _build_alias_index() -> dict[str, str]:
    """Map every normalised alias to its canonical name."""
    index: dict[str, str] = {}
    for canon, aliases in CANONICAL_ALIASES.items():
        for a in aliases:
            index[_norm_header(a)] = canon
    return index


_ALIAS_INDEX = _build_alias_index()


def _classify_cable_column(raw: str) -> tuple[str | None, int | None]:
    """Return ``("L", k)`` / ``("T", k)`` if ``raw`` is a cable-indexed
    column (length or tension family), ``(None, None)`` otherwise.

    The classifier requires a trailing integer --- so ``Layer``,
    ``Time``, and other accidental ``L*`` / ``T*`` columns are rejected.
    """
    norm = _norm_header(raw)
    m = re.match(r"^([a-z_]+?)_?(\d+)$", norm)
    if not m:
        return None, None
    prefix, idx = m.group(1), int(m.group(2))
    if prefix in CABLE_LENGTH_PREFIXES:
        return "L", idx
    if prefix in CABLE_TENSION_PREFIXES:
        return "T", idx
    return None, None


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_csv_rows(path: Path) -> tuple[list[str], np.ndarray]:
    """Read header + rows, tolerating leading whitespace and BOM."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV: {path}") from exc
        header = [h.strip() for h in header]
        data_rows: list[list[float]] = []
        for row_no, row in enumerate(reader, start=2):
            if not row or all(not str(c).strip() for c in row):
                continue
            try:
                data_rows.append([float(c) if str(c).strip() else float("nan") for c in row])
            except ValueError as exc:
                raise ValueError(
                    f"could not parse CSV row {row_no} as floats: {row} ({exc})"
                ) from exc
        if not data_rows:
            raise ValueError(f"CSV has a header but no data rows: {path}")
    arr = np.asarray(data_rows, dtype=np.float64)
    return header, arr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

REQUIRED_CANONICAL = ["t", "px", "py", "pz"]


def load_csv_any(
    input_str: str,
    *,
    overrides: dict[str, str] | None = None,
) -> tuple[dict[str, np.ndarray], SchemaReport]:
    """Resolve, read, and column-map any CSV into the canonical layout.

    Parameters
    ----------
    input_str:
        Local path *or* HTTP(S) URL.
    overrides:
        Optional explicit ``{canonical_name: source_column}`` mapping
        applied AFTER alias auto-detection. Useful when a CSV uses
        domain-specific column names that aren't in
        :data:`CANONICAL_ALIASES`.

    Returns
    -------
    columns:
        Dict keyed by canonical names. Cable families are returned as
        ``L1, L2, ...`` and ``T1, T2, ...`` (renamed from whatever the
        source called them). Original source-only columns are also
        passed through under their original names so downstream code
        can inspect them.
    report:
        :class:`SchemaReport` describing what was mapped and what is
        missing. ``report.missing_required`` lists the canonical names
        that are NOT present after mapping --- callers should refuse
        the workflow if any of :data:`REQUIRED_CANONICAL` are missing.
    """
    path = resolve_csv_input(input_str)
    header, data = _parse_csv_rows(path)

    # Build the canonical -> source mapping using aliases + user overrides.
    canonical_mapping: dict[str, str] = {}
    used_source_columns: set[str] = set()
    for col in header:
        canon = _ALIAS_INDEX.get(_norm_header(col))
        if canon and canon not in canonical_mapping:
            canonical_mapping[canon] = col
            used_source_columns.add(col)
    if overrides:
        for canon, src in overrides.items():
            if src in header:
                canonical_mapping[canon] = src
                used_source_columns.add(src)

    # Classify cable-indexed columns.
    length_pairs: list[tuple[int, str]] = []
    tension_pairs: list[tuple[int, str]] = []
    for col in header:
        family, idx = _classify_cable_column(col)
        if family == "L":
            length_pairs.append((idx, col))
        elif family == "T":
            tension_pairs.append((idx, col))
    length_pairs.sort()
    tension_pairs.sort()

    # Materialise the column-keyed dict.
    columns: dict[str, np.ndarray] = {}
    for i, col in enumerate(header):
        columns[col] = data[:, i]
    # Add canonical aliases (point at the same array).
    for canon, src in canonical_mapping.items():
        columns[canon] = columns[src]
    # Cable families: renumber to L1..Lm / T1..Tm.
    for new_idx, (_, src) in enumerate(length_pairs, start=1):
        columns[f"L{new_idx}"] = columns[src]
    for new_idx, (_, src) in enumerate(tension_pairs, start=1):
        columns[f"T{new_idx}"] = columns[src]

    missing = [c for c in REQUIRED_CANONICAL if c not in canonical_mapping]

    report = SchemaReport(
        canonical_mapping=canonical_mapping,
        length_columns=[f"L{i}" for i in range(1, len(length_pairs) + 1)],
        tension_columns=[f"T{i}" for i in range(1, len(tension_pairs) + 1)],
        missing_required=missing,
        n_samples=int(data.shape[0]),
        source=str(path),
    )
    return columns, report


def split_canonical_blocks(columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Pack the canonical blocks expected by downstream consumers."""
    t = columns["t"]
    n = len(t)
    pos = np.column_stack([columns[c] for c in ("px", "py", "pz")])
    quat = (
        np.column_stack([columns[c] for c in ("qx", "qy", "qz", "qw")])
        if all(c in columns for c in ("qx", "qy", "qz", "qw"))
        else np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    )
    lin_v = (
        np.column_stack([columns[c] for c in ("vx", "vy", "vz")])
        if all(c in columns for c in ("vx", "vy", "vz"))
        else _finite_diff(pos, t)
    )
    ang_v = (
        np.column_stack([columns[c] for c in ("wx", "wy", "wz")])
        if all(c in columns for c in ("wx", "wy", "wz"))
        else np.zeros_like(lin_v)
    )
    L_cols = [k for k in columns if k.startswith("L") and k[1:].isdigit()]
    L_cols.sort(key=lambda s: int(s[1:]))
    T_cols = [k for k in columns if k.startswith("T") and k[1:].isdigit()]
    T_cols.sort(key=lambda s: int(s[1:]))
    lengths = (
        np.column_stack([columns[c] for c in L_cols]) if L_cols else np.zeros((n, 0))
    )
    tensions = (
        np.column_stack([columns[c] for c in T_cols]) if T_cols else np.zeros((n, 0))
    )
    return {
        "time": t,
        "positions": pos,
        "quaternions_xyzw": quat,
        "linear_velocities": lin_v,
        "angular_velocities": ang_v,
        "cable_lengths": lengths,
        "cable_tensions": tensions,
    }


def _finite_diff(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(arr, t, axis=0)


# ---------------------------------------------------------------------------
# Robot reconstruction --- manifest-aware, catalog-aware, JSON-aware.
# ---------------------------------------------------------------------------

@dataclass
class RobotSpec:
    """Plain-dict description of a robot, suitable for JSON round-trip."""

    name: str
    n_cables: int
    dof: int
    anchors: list[list[float]]
    attachments: list[list[float]]
    mass: float
    inertia: list[list[float]]                                       # 3x3
    com: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    t_min: float = 5.0
    t_max: float = 500.0
    cable_diameter_m: float = 3e-3

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_cables": self.n_cables,
            "dof": self.dof,
            "anchors": self.anchors,
            "attachments": self.attachments,
            "mass": self.mass,
            "inertia": self.inertia,
            "com": self.com,
            "t_min": self.t_min,
            "t_max": self.t_max,
            "cable_diameter_m": self.cable_diameter_m,
        }

    @classmethod
    def from_robot(cls, robot, *, name: str | None = None) -> "RobotSpec":
        anchors = robot.anchors.tolist()
        attachments = robot.attachments.tolist()
        mass = float(robot.inertia.mass) if robot.inertia is not None else 0.0
        inertia = (
            robot.inertia.inertia.tolist()
            if robot.inertia is not None
            else np.eye(3).tolist()
        )
        com = (
            robot.inertia.com.tolist()
            if robot.inertia is not None
            else [0.0, 0.0, 0.0]
        )
        t_min = float(robot.limits.t_min[0]) if robot.limits is not None else 0.0
        t_max = float(robot.limits.t_max[0]) if robot.limits is not None else 1.0
        return cls(
            name=name or robot.name,
            n_cables=int(robot.n_cables),
            dof=int(robot.dof),
            anchors=anchors,
            attachments=attachments,
            mass=mass,
            inertia=inertia,
            com=com,
            t_min=t_min,
            t_max=t_max,
        )


def robot_from_spec(spec: RobotSpec):
    """Materialise a :class:`cdpr.geometry.robot.Robot` from a JSON spec.

    Lazy imports keep this module light when only the CSV helpers are
    needed."""
    from cdpr.geometry.robot import (
        CableLimits, CableProperties, PlatformInertia, Robot, RobotGeometry,
    )
    anchors = np.asarray(spec.anchors, dtype=np.float64)
    attachments = np.asarray(spec.attachments, dtype=np.float64)
    geometry = RobotGeometry(
        anchors=anchors,
        attachments=attachments,
        dof=int(spec.dof),
        name=spec.name,
    )
    inertia = PlatformInertia(
        mass=float(spec.mass),
        com=np.asarray(spec.com, dtype=np.float64),
        inertia=np.asarray(spec.inertia, dtype=np.float64),
    )
    limits = CableLimits.uniform(
        spec.n_cables, t_min=float(spec.t_min), t_max=float(spec.t_max),
    )
    return Robot(
        geometry=geometry,
        inertia=inertia,
        limits=limits,
        cable_properties=CableProperties.steel_aircraft_cable(
            spec.n_cables, diameter_m=float(spec.cable_diameter_m),
        ),
    )


def robot_from_manifest_or_catalog(
    manifest: dict | None,
    *,
    robot_config_path: str | Path | None = None,
):
    """Best-effort robot reconstruction.

    Resolution order:
      1. If ``robot_config_path`` is given, load it as a :class:`RobotSpec`
         and return ``robot_from_spec``. Highest priority --- explicit
         user override.
      2. If the manifest carries a ``robot_spec`` block (new schema),
         reconstruct from that.
      3. If the manifest carries only ``request.robot`` (legacy schema),
         fall back to :func:`cdpr.interface.specs.build_robot`. This now
         knows ``dissertation_8cable`` too thanks to the factory update.
      4. Raise :class:`ValueError` with a clear message --- the caller
         should turn that into a friendly "skipping replay/RL" stub.
    """
    if robot_config_path:
        with Path(robot_config_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return robot_from_spec(RobotSpec(**data))
    if manifest:
        spec_data = manifest.get("robot_spec")
        if spec_data:
            return robot_from_spec(RobotSpec(**spec_data))
        req = manifest.get("request") or {}
        name = req.get("robot")
        if name:
            from cdpr.interface.specs import build_robot
            return build_robot(
                name,
                payload_mass=float(req.get("payload_mass") or 0.0),
                t_min=(manifest.get("feasibility") or {}).get("t_min_N"),
                t_max=(manifest.get("feasibility") or {}).get("t_max_N"),
            )
    raise ValueError(
        "Could not reconstruct the robot: no robot_config_path was supplied "
        "and the manifest carries no robot description. Replay / RL "
        "workflows need this to be set --- pass --robot-config <path.json> "
        "or use a CSV produced by scripts/run_simulation.py (which writes "
        "the robot spec into manifest.json)."
    )
