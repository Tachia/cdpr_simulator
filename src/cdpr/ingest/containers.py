r"""Core data containers shared by the loaders, the pipeline, and the validator.

* :class:`RawDataset` -- thin wrapper around a pandas :class:`DataFrame`
  plus the metadata we always want around (origin path, byte size, raw
  header, format tag). The DataFrame is preserved as-is; *no* schema is
  imposed at load time --- the ColumnMap layer handles that.

* :class:`Channel` -- enum of the semantic data channels the framework
  knows about. Used as keys in :class:`ColumnMap` and as the dictionary
  keys for the cleaned outputs produced by the pipeline.

* :class:`ColumnMap` -- a mapping from semantic channel name to the
  *actual* column names in the raw data, plus an autodetection routine
  that tries common aliases. Building it by hand is the right thing for
  unusual lab formats; the autodetector handles 80% of cases.

* :class:`StepRecord` -- single entry in the pipeline's audit log. Each
  cleaning / resampling / filtering operation appends one of these.

* :class:`IngestedExperiment` -- the pipeline's final output. Its field
  layout mirrors :class:`cdpr.recording.replay.Experiment` so the same
  plotting code consumes both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    import pandas as pd


# ---------------------------------------------------------------------------
# Channel registry
# ---------------------------------------------------------------------------

class Channel(str, Enum):
    """Semantic categories of CDPR experimental data.

    Membership here is what defines "the framework knows what to do with
    this column". Adding a new channel means adding it both here and in
    the cleaning / resampling code paths.
    """

    TIME = "time"
    POSITION = "position"                    # (x, y, z)
    QUATERNION = "quaternion"                # (x, y, z, w), SciPy convention
    EULER = "euler"                          # (a, b, c) -- requires euler_order
    LINEAR_VELOCITY = "linear_velocity"
    ANGULAR_VELOCITY = "angular_velocity"
    CABLE_LENGTHS = "cable_lengths"
    CABLE_TENSIONS = "cable_tensions"


# Default alias table the autodetector tries. Keys are channels; values are
# tuples of (column-name regexes, expected component count). The regexes are
# matched case-insensitively against the raw header strings.
_ALIAS_TABLE: dict[Channel, list[tuple[str, ...]]] = {
    Channel.TIME: [("time", "t", "timestamp", "time_s")],
    Channel.POSITION: [
        ("x", "y", "z"),
        ("pos_x", "pos_y", "pos_z"),
        ("position_x", "position_y", "position_z"),
        ("px", "py", "pz"),
    ],
    Channel.QUATERNION: [
        ("qx", "qy", "qz", "qw"),
        ("quat_x", "quat_y", "quat_z", "quat_w"),
        ("quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w"),
    ],
    Channel.EULER: [
        ("roll", "pitch", "yaw"),
        ("euler_x", "euler_y", "euler_z"),
    ],
    Channel.LINEAR_VELOCITY: [
        ("vx", "vy", "vz"),
        ("vel_x", "vel_y", "vel_z"),
        ("velocity_x", "velocity_y", "velocity_z"),
    ],
    Channel.ANGULAR_VELOCITY: [
        ("wx", "wy", "wz"),
        ("omega_x", "omega_y", "omega_z"),
        ("angular_vel_x", "angular_vel_y", "angular_vel_z"),
    ],
}


# ---------------------------------------------------------------------------
# RawDataset
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RawDataset:
    """Loader output. The DataFrame is the source of truth; metadata is descriptive."""

    frame: "pd.DataFrame"
    source_path: Path | None
    format: str
    n_rows_raw: int
    header: list[str]
    loaded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        # Defensive copy so the user can't mutate our reference by accident.
        # pandas DataFrame copy is cheap by default (CoW in pandas 2.x).
        self.frame = self.frame.copy()

    @property
    def columns(self) -> list[str]:
        return list(self.frame.columns)

    def describe(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "format": self.format,
            "n_rows_raw": int(self.n_rows_raw),
            "n_columns": len(self.header),
            "header": self.header,
            "loaded_at": self.loaded_at,
        }


# ---------------------------------------------------------------------------
# ColumnMap
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ColumnMap:
    """Maps semantic channels to *actual* column names in the raw data.

    Build directly with the channel keys you have::

        ColumnMap(
            time="t",
            position=("x", "y", "z"),
            cable_lengths=("L1", "L2", "L3", "L4"),
        )

    or call :meth:`autodetect` on a :class:`RawDataset` to try common aliases.
    """

    time: str | None = None
    position: tuple[str, str, str] | None = None
    quaternion: tuple[str, str, str, str] | None = None
    euler: tuple[str, str, str] | None = None
    euler_order: str = "xyz"
    linear_velocity: tuple[str, str, str] | None = None
    angular_velocity: tuple[str, str, str] | None = None
    cable_lengths: tuple[str, ...] | None = None
    cable_tensions: tuple[str, ...] | None = None

    def channel_columns(self, channel: Channel) -> tuple[str, ...] | None:
        """Return the column names mapped to a channel, or ``None``."""
        match channel:
            case Channel.TIME:
                return (self.time,) if self.time is not None else None
            case Channel.POSITION:
                return self.position
            case Channel.QUATERNION:
                return self.quaternion
            case Channel.EULER:
                return self.euler
            case Channel.LINEAR_VELOCITY:
                return self.linear_velocity
            case Channel.ANGULAR_VELOCITY:
                return self.angular_velocity
            case Channel.CABLE_LENGTHS:
                return self.cable_lengths
            case Channel.CABLE_TENSIONS:
                return self.cable_tensions

    def assigned_columns(self) -> dict[Channel, tuple[str, ...]]:
        out: dict[Channel, tuple[str, ...]] = {}
        for ch in Channel:
            cols = self.channel_columns(ch)
            if cols is not None:
                out[ch] = cols
        return out

    # --- autodetection -------------------------------------------------

    @classmethod
    def autodetect(cls, raw: RawDataset) -> "ColumnMap":
        """Best-effort guess at the channel mapping from the raw header.

        Cable channels are detected via prefix patterns ``T1, T2, ...``
        (tension) and ``L1, L2, ...`` (length); arbitrary digit suffixes
        are accepted to match common lab conventions.
        """
        lowered = {col.lower(): col for col in raw.columns}

        def find(group: tuple[str, ...]) -> tuple[str, ...] | None:
            cols: list[str] = []
            for alias in group:
                if alias in lowered:
                    cols.append(lowered[alias])
            return tuple(cols) if len(cols) == len(group) else None

        time_col: str | None = None
        for alias_group in _ALIAS_TABLE[Channel.TIME]:
            for alias in alias_group:
                if alias in lowered:
                    time_col = lowered[alias]
                    break
            if time_col is not None:
                break

        def first_match(channel: Channel) -> tuple[str, ...] | None:
            for group in _ALIAS_TABLE.get(channel, []):
                hit = find(group)
                if hit is not None:
                    return hit
            return None

        return cls(
            time=time_col,
            position=first_match(Channel.POSITION),
            quaternion=first_match(Channel.QUATERNION),
            euler=first_match(Channel.EULER),
            linear_velocity=first_match(Channel.LINEAR_VELOCITY),
            angular_velocity=first_match(Channel.ANGULAR_VELOCITY),
            cable_lengths=_find_numbered(raw.columns, prefixes=("l", "len", "length", "cable_l")),
            cable_tensions=_find_numbered(raw.columns, prefixes=("t", "ten", "tension", "cable_t")),
        )

    # --- validation -----------------------------------------------------

    def validate(self, raw: RawDataset) -> None:
        cols_in_data = set(raw.columns)
        for channel, cols in self.assigned_columns().items():
            missing = [c for c in cols if c not in cols_in_data]
            if missing:
                raise KeyError(
                    f"ColumnMap channel {channel.value!r} references missing columns: {missing}"
                )


def _find_numbered(
    columns: list[str], prefixes: tuple[str, ...]
) -> tuple[str, ...] | None:
    """Find ``prefix1, prefix2, ...`` patterns in the header.

    Used by the autodetector for cable channels --- different labs use
    ``T1..T8``, ``Ten01..Ten08``, ``tension_1..tension_8``, etc.
    """
    import re
    candidates: list[tuple[int, str]] = []
    for col in columns:
        low = col.lower()
        for prefix in prefixes:
            m = re.fullmatch(rf"{re.escape(prefix)}_?(\d+)", low)
            if m:
                # Tension prefix "t" must not swallow plain "time" --- a
                # single bare 't' has no number, so this is safe; we still
                # guard against ambiguous one-character prefixes by
                # requiring the digit suffix.
                candidates.append((int(m.group(1)), col))
                break
    if not candidates:
        return None
    candidates.sort()
    return tuple(name for _, name in candidates)


# ---------------------------------------------------------------------------
# Step record
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class StepRecord:
    """One entry in the pipeline's audit log."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    rows_before: int = 0
    rows_after: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IngestedExperiment
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class IngestedExperiment:
    """Clean, time-aligned experimental data.

    Field layout mirrors :class:`cdpr.recording.replay.Experiment` so the
    same plotting, comparison, and report functions consume both. Fields
    that the source data did not contain are ``None`` rather than zero
    arrays --- downstream code should check before indexing.
    """

    time: NDArray[np.float64]                    # (T,)
    positions: NDArray[np.float64] | None        # (T, 3)
    quaternions_xyzw: NDArray[np.float64] | None # (T, 4)
    linear_velocities: NDArray[np.float64] | None
    angular_velocities: NDArray[np.float64] | None
    cable_lengths: NDArray[np.float64] | None
    cable_tensions: NDArray[np.float64] | None

    columns: ColumnMap
    source: dict[str, Any]                       # RawDataset.describe()
    steps: list[StepRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- convenience properties expected by the viz layer ----------------

    @property
    def infeasible_steps(self) -> list[int]:
        # Experimental data carries no solver feasibility flag; return empty.
        return []

    @property
    def condition_numbers(self) -> NDArray[np.float64]:
        return np.full(len(self.time), np.nan, dtype=np.float64)

    def has(self, channel: Channel) -> bool:
        attr = {
            Channel.TIME: "time",
            Channel.POSITION: "positions",
            Channel.QUATERNION: "quaternions_xyzw",
            Channel.LINEAR_VELOCITY: "linear_velocities",
            Channel.ANGULAR_VELOCITY: "angular_velocities",
            Channel.CABLE_LENGTHS: "cable_lengths",
            Channel.CABLE_TENSIONS: "cable_tensions",
        }.get(channel)
        if attr is None:
            return False
        return getattr(self, attr, None) is not None
