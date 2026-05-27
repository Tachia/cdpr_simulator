"""Experimental data ingestion pipeline.

Takes a raw CSV / XLSX / TXT / JSON experimental log and produces an
:class:`IngestedExperiment` --- structurally compatible with
:class:`cdpr.recording.replay.Experiment`, so every downstream plotter,
report writer, and comparison helper already in the framework consumes it
unchanged.

Typical usage::

    from cdpr.ingest import Pipeline, load_csv, ColumnMap

    raw = load_csv("trial_03.csv")
    columns = ColumnMap.autodetect(raw)               # or build by hand
    pipeline = (
        Pipeline(raw, columns=columns)
        .drop_nan()
        .deduplicate_timestamps()
        .remove_outliers(method="mad", threshold=3.5)
        .resample(dt=1e-3)
        .lowpass(cutoff_hz=25.0)
        .convert_units(position_scale=1e-3)           # mm -> m
    )
    experiment = pipeline.run()

    from cdpr.ingest.validate import validate_against_robot
    diagnostics = validate_against_robot(experiment, my_robot)

Pipeline operations are *recorded*. Calling :meth:`Pipeline.report` returns
a :class:`PreprocessingReport` describing exactly what was done; calling
:func:`write_preprocessing_report` serialises it to Markdown + JSON for
the dissertation appendix.
"""

from cdpr.ingest.containers import (
    Channel,
    ColumnMap,
    IngestedExperiment,
    RawDataset,
    StepRecord,
)
from cdpr.ingest.loaders import (
    load_csv,
    load_dataframe,
    load_json,
    load_txt,
    load_xlsx,
)
from cdpr.ingest.pipeline import Pipeline
from cdpr.ingest.report import PreprocessingReport, write_preprocessing_report
from cdpr.ingest.validate import (
    ValidationReport,
    reconstruct_trajectory,
    validate_against_robot,
)

__all__ = [
    "Channel",
    "ColumnMap",
    "RawDataset",
    "IngestedExperiment",
    "StepRecord",
    "Pipeline",
    "load_csv",
    "load_xlsx",
    "load_txt",
    "load_json",
    "load_dataframe",
    "validate_against_robot",
    "reconstruct_trajectory",
    "ValidationReport",
    "PreprocessingReport",
    "write_preprocessing_report",
]
