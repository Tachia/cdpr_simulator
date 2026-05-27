"""Publication-grade output generation.

Consumes :class:`Experiment` (replayed) or :class:`SimulationResult`
objects plus optional :class:`ComparisonReport` instances and produces:

* LaTeX-ready captioned figures (PDF/SVG/PNG via :mod:`cdpr.viz.export`)
* Structured tables in CSV and LaTeX
* Markdown experiment summaries that drop into a paper / dissertation

The :class:`ReproducibilityManifest` lives in :mod:`cdpr.recording.schema`
and is referenced from here without duplication.
"""

from cdpr.reports.bundle import write_bundle_report
from cdpr.reports.figures import CaptionedFigure, save_captioned_figure
from cdpr.reports.summary import write_markdown_summary
from cdpr.reports.tables import (
    cable_summary_table,
    summary_table_csv,
    summary_table_latex,
)

__all__ = [
    "CaptionedFigure",
    "save_captioned_figure",
    "cable_summary_table",
    "summary_table_csv",
    "summary_table_latex",
    "write_markdown_summary",
    "write_bundle_report",
]
