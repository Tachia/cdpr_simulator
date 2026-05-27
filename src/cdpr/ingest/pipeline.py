r"""Pipeline orchestrator.

The :class:`Pipeline` glues the cleaning, resampling, filtering, and
unit-conversion modules into a chainable, recorded sequence. The
constructor takes a :class:`RawDataset` and a :class:`ColumnMap`; each
operation returns ``self`` so calls compose; :meth:`run` materialises the
result into an :class:`IngestedExperiment`.

Steps are accumulated *lazily* in a deferred queue --- nothing executes
until :meth:`run`. That means the operation list reads as the operation
*description*; the run produces both the data and the per-step
:class:`StepRecord` history. This separation makes the pipeline (a)
serialisable (the description is a list of dicts) and (b) easy to retry
or modify without redoing the load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from cdpr.ingest.cleaning import (
    deduplicate_timestamps,
    drop_nan_rows,
    interpolate_nans,
    remove_outliers_mad,
)
from cdpr.ingest.containers import (
    Channel,
    ColumnMap,
    IngestedExperiment,
    RawDataset,
    StepRecord,
)
from cdpr.ingest.filtering import lowpass_butterworth, savitzky_golay
from cdpr.ingest.resample import resample_uniform
from cdpr.ingest.units import ChannelScale, convert_units, transform_frame

if TYPE_CHECKING:                                           # pragma: no cover
    import pandas as pd


_OperationFn = Callable[["pd.DataFrame", ColumnMap], tuple["pd.DataFrame", dict[str, Any]]]


@dataclass(slots=True)
class _PendingStep:
    name: str
    parameters: dict[str, Any]
    fn: _OperationFn


@dataclass(slots=True)
class Pipeline:
    """Fluent recorder for a sequence of ingest operations."""

    raw: RawDataset
    columns: ColumnMap
    pending: list[_PendingStep] = field(default_factory=list)

    # --- construction helpers -----------------------------------------

    def __post_init__(self) -> None:
        # Validate that the column map exists in the raw header; surfacing
        # missing-column errors here instead of inside a buried step keeps
        # the traceback short.
        self.columns.validate(self.raw)

    def _queue(self, name: str, parameters: dict[str, Any], fn: _OperationFn) -> "Pipeline":
        self.pending.append(_PendingStep(name=name, parameters=parameters, fn=fn))
        return self

    # --- cleaning ------------------------------------------------------

    def drop_nan(self) -> "Pipeline":
        """Drop rows where any mapped column is NaN."""
        return self._queue("drop_nan", {}, drop_nan_rows)

    def interpolate_missing(self, *, method: str = "linear") -> "Pipeline":
        """Linearly interpolate missing values along the time axis."""
        return self._queue(
            "interpolate_missing", {"method": method},
            lambda f, c: interpolate_nans(f, c, method=method),
        )

    def deduplicate_timestamps(
        self, *, strategy: Literal["first", "mean"] = "mean",
    ) -> "Pipeline":
        """Collapse or drop duplicate-timestamped rows."""
        return self._queue(
            "deduplicate_timestamps", {"strategy": strategy},
            lambda f, c: deduplicate_timestamps(f, c, strategy=strategy),
        )

    def remove_outliers(
        self, *, method: Literal["mad"] = "mad", threshold: float = 3.5,
        action: Literal["drop", "nan"] = "nan",
    ) -> "Pipeline":
        """Robust outlier flagging.

        With ``action="nan"`` the flagged samples are replaced by NaN so a
        subsequent :meth:`interpolate_missing` call can fill them.
        """
        if method != "mad":
            raise ValueError(f"Unsupported outlier method: {method!r}")
        return self._queue(
            "remove_outliers",
            {"method": method, "threshold": threshold, "action": action},
            lambda f, c: remove_outliers_mad(f, c, threshold=threshold, action=action),
        )

    # --- resampling ----------------------------------------------------

    def resample(
        self, *, dt: float | None = None,
        method: Literal["cubic", "linear"] = "cubic",
        t_start: float | None = None, t_end: float | None = None,
    ) -> "Pipeline":
        """Interpolate onto a uniform time grid."""
        return self._queue(
            "resample",
            {"dt": dt, "method": method, "t_start": t_start, "t_end": t_end},
            lambda f, c: resample_uniform(
                f, c, dt=dt, method=method, t_start=t_start, t_end=t_end,
            ),
        )

    # --- filtering -----------------------------------------------------

    def lowpass(
        self, *, cutoff_hz: float, order: int = 4,
        only_channels: list[str] | None = None,
    ) -> "Pipeline":
        return self._queue(
            "lowpass_butterworth",
            {"cutoff_hz": cutoff_hz, "order": order, "only_channels": only_channels},
            lambda f, c: lowpass_butterworth(
                f, c, cutoff_hz=cutoff_hz, order=order, only_channels=only_channels,
            ),
        )

    def savgol(
        self, *, window_length: int, polyorder: int = 3, deriv: int = 0,
        only_channels: list[str] | None = None,
    ) -> "Pipeline":
        return self._queue(
            "savitzky_golay",
            {"window_length": window_length, "polyorder": polyorder,
             "deriv": deriv, "only_channels": only_channels},
            lambda f, c: savitzky_golay(
                f, c, window_length=window_length, polyorder=polyorder,
                deriv=deriv, only_channels=only_channels,
            ),
        )

    # --- units / frames ------------------------------------------------

    def convert_units(
        self, *,
        position_scale: float | None = None,
        cable_length_scale: float | None = None,
        tension_scale: float | None = None,
        time_scale: float | None = None,
        euler_deg_to_rad: bool = False,
    ) -> "Pipeline":
        """Apply unit-conversion to the standard channels.

        Each ``*_scale`` argument is a scalar broadcast across the
        corresponding channel; pass arrays through :func:`convert_units`
        directly if you need per-component scaling.
        """
        return self._queue(
            "convert_units",
            {
                "position_scale": position_scale,
                "cable_length_scale": cable_length_scale,
                "tension_scale": tension_scale,
                "time_scale": time_scale,
                "euler_deg_to_rad": euler_deg_to_rad,
            },
            lambda f, c: convert_units(
                f, c,
                position=ChannelScale(scale=position_scale) if position_scale else None,
                cable_lengths=ChannelScale(scale=cable_length_scale) if cable_length_scale else None,
                cable_tensions=ChannelScale(scale=tension_scale) if tension_scale else None,
                time=ChannelScale(scale=time_scale) if time_scale else None,
                euler_deg_to_rad=euler_deg_to_rad,
            ),
        )

    def transform_world_frame(
        self, *, rotation: Rotation | None = None, translation: list[float] | None = None,
    ) -> "Pipeline":
        """Rigid transform of positions / velocities / orientations."""
        return self._queue(
            "transform_frame",
            {
                "rotation_rotvec": (rotation.as_rotvec().tolist() if rotation else None),
                "translation": translation or [0.0, 0.0, 0.0],
            },
            lambda f, c: transform_frame(
                f, c, rotation=rotation, translation=translation or [0.0, 0.0, 0.0],
            ),
        )

    # --- execution -----------------------------------------------------

    def run(self) -> IngestedExperiment:
        """Apply every queued operation and produce the cleaned experiment."""
        df = self.raw.frame
        steps: list[StepRecord] = []
        for step in self.pending:
            rows_before = int(len(df))
            df, diagnostics = step.fn(df, self.columns)
            steps.append(StepRecord(
                name=step.name,
                parameters=step.parameters,
                rows_before=rows_before,
                rows_after=int(len(df)),
                diagnostics=diagnostics,
            ))
        return _materialise(df, self.columns, self.raw, steps)


# ---------------------------------------------------------------------------
# DataFrame -> IngestedExperiment
# ---------------------------------------------------------------------------

def _materialise(
    df: "pd.DataFrame",
    columns: ColumnMap,
    raw: RawDataset,
    steps: list[StepRecord],
) -> IngestedExperiment:
    """Project the cleaned DataFrame into the structured experiment object."""
    if columns.time is None:
        raise ValueError("Cannot materialise IngestedExperiment without a time channel.")
    time = df[columns.time].to_numpy(dtype=np.float64)

    def stack(cols: tuple[str, ...] | None) -> np.ndarray | None:
        if cols is None:
            return None
        return df[list(cols)].to_numpy(dtype=np.float64)

    positions = stack(columns.position)
    quaternions = stack(columns.quaternion)

    # Euler -> quaternion if quaternion was not present but euler was.
    if quaternions is None and columns.euler is not None:
        ang = df[list(columns.euler)].to_numpy(dtype=np.float64)
        quaternions = Rotation.from_euler(columns.euler_order, ang).as_quat()

    return IngestedExperiment(
        time=time,
        positions=positions,
        quaternions_xyzw=quaternions,
        linear_velocities=stack(columns.linear_velocity),
        angular_velocities=stack(columns.angular_velocity),
        cable_lengths=stack(columns.cable_lengths),
        cable_tensions=stack(columns.cable_tensions),
        columns=columns,
        source=raw.describe(),
        steps=steps,
    )
