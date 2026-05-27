r"""Preprocessing report.

Generates a one-experiment Markdown document and an accompanying machine-
readable JSON describing exactly what the ingest pipeline did to the
data --- which steps ran, what they removed or modified, and (if
:func:`validate_against_robot` was called) what the validation residuals
were. Both files are deterministic given the same inputs, which is what
makes them appendix-quality artefacts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from cdpr.ingest.containers import IngestedExperiment, StepRecord

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.ingest.validate import ValidationReport


@dataclass(slots=True)
class PreprocessingReport:
    """In-memory representation of the ingest report."""

    title: str
    source: dict[str, Any]
    columns: dict[str, Any]
    n_samples_in: int
    n_samples_out: int
    steps: list[StepRecord] = field(default_factory=list)
    validation: dict[str, Any] | None = None
    statistics: dict[str, Any] = field(default_factory=dict)


def _step_to_dict(s: StepRecord) -> dict[str, Any]:
    return {
        "name": s.name,
        "parameters": s.parameters,
        "rows_before": s.rows_before,
        "rows_after": s.rows_after,
        "diagnostics": s.diagnostics,
    }


def _channel_stats(arr: np.ndarray | None) -> dict[str, float] | None:
    if arr is None or arr.size == 0:
        return None
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr[:, None]
    return {
        "mean": float(np.nanmean(flat)),
        "std": float(np.nanstd(flat)),
        "min": float(np.nanmin(flat)),
        "max": float(np.nanmax(flat)),
    }


def _statistics(experiment: IngestedExperiment) -> dict[str, Any]:
    return {
        "duration_s": float(experiment.time[-1] - experiment.time[0]) if len(experiment.time) > 1 else 0.0,
        "position": _channel_stats(experiment.positions),
        "linear_velocity": _channel_stats(experiment.linear_velocities),
        "angular_velocity": _channel_stats(experiment.angular_velocities),
        "cable_lengths": _channel_stats(experiment.cable_lengths),
        "cable_tensions": _channel_stats(experiment.cable_tensions),
    }


def build_preprocessing_report(
    experiment: IngestedExperiment,
    *,
    title: str,
    validation: "ValidationReport | None" = None,
) -> PreprocessingReport:
    columns_dict = {
        "time": experiment.columns.time,
        "position": list(experiment.columns.position or ()),
        "quaternion": list(experiment.columns.quaternion or ()),
        "linear_velocity": list(experiment.columns.linear_velocity or ()),
        "angular_velocity": list(experiment.columns.angular_velocity or ()),
        "cable_lengths": list(experiment.columns.cable_lengths or ()),
        "cable_tensions": list(experiment.columns.cable_tensions or ()),
    }
    n_in = int(experiment.source.get("n_rows_raw", 0))
    return PreprocessingReport(
        title=title,
        source=experiment.source,
        columns=columns_dict,
        n_samples_in=n_in,
        n_samples_out=int(len(experiment.time)),
        steps=experiment.steps,
        validation=validation.summary() if validation is not None else None,
        statistics=_statistics(experiment),
    )


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _markdown(report: PreprocessingReport) -> str:
    lines: list[str] = [f"# {report.title}", ""]
    lines += [
        "## Source",
        "",
        f"- **Path:** `{report.source.get('source_path') or '(in-memory)'}`",
        f"- **Format:** {report.source.get('format')}",
        f"- **Rows raw:** {report.source.get('n_rows_raw')}",
        f"- **Columns ({report.source.get('n_columns')}):** "
        f"`{', '.join(report.source.get('header', []))}`",
        f"- **Loaded at:** {report.source.get('loaded_at')}",
        "",
    ]

    lines += ["## Column map", ""]
    for name, value in report.columns.items():
        if value:
            lines.append(f"- **{name}:** `{value if isinstance(value, str) else ', '.join(value)}`")
    lines.append("")

    lines += ["## Pipeline", ""]
    if not report.steps:
        lines += ["_No steps recorded._", ""]
    for k, step in enumerate(report.steps, start=1):
        lines += [
            f"### Step {k}: `{step.name}`",
            "",
            f"- rows {step.rows_before} → {step.rows_after}",
        ]
        if step.parameters:
            lines.append(f"- parameters: `{step.parameters}`")
        if step.diagnostics:
            lines.append(f"- diagnostics: `{step.diagnostics}`")
        lines.append("")

    lines += ["## Output statistics", ""]
    stats = report.statistics
    lines += [
        f"- **Duration:** {stats.get('duration_s', 0.0):.4f} s",
        f"- **Samples:** {report.n_samples_in} → {report.n_samples_out}",
        "",
    ]
    for name in ("position", "linear_velocity", "angular_velocity",
                 "cable_lengths", "cable_tensions"):
        block = stats.get(name)
        if not block:
            continue
        lines.append(
            f"- **{name}:** mean {block['mean']:.4g}, std {block['std']:.4g}, "
            f"range [{block['min']:.4g}, {block['max']:.4g}]"
        )
    lines.append("")

    if report.validation is not None:
        lines += ["## Validation against the dynamic model", ""]
        for k, v in report.validation.items():
            if v is None:
                lines.append(f"- **{k}:** _not applicable_")
            else:
                lines.append(f"- **{k}:** RMS {v['rms']:.4g}, peak {v['peak']:.4g}")
        lines.append("")

    return "\n".join(lines)


def _json_payload(report: PreprocessingReport) -> dict[str, Any]:
    return {
        "title": report.title,
        "source": report.source,
        "columns": report.columns,
        "n_samples_in": report.n_samples_in,
        "n_samples_out": report.n_samples_out,
        "steps": [_step_to_dict(s) for s in report.steps],
        "statistics": report.statistics,
        "validation": report.validation,
    }


def write_preprocessing_report(
    experiment: IngestedExperiment,
    out_dir: str | Path,
    *,
    title: str = "Preprocessing report",
    validation: "ValidationReport | None" = None,
) -> dict[str, Path]:
    """Write ``report.md`` and ``report.json`` for an ingested experiment.

    Returns a dict mapping ``"md"`` / ``"json"`` to the written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = build_preprocessing_report(experiment, title=title, validation=validation)

    md_path = out / "report.md"
    json_path = out / "report.json"
    # Force UTF-8 explicitly so the arrow / mathematical symbols in the
    # template survive on Windows hosts whose default codepage is cp1252.
    md_path.write_text(_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(_json_payload(report), indent=2, sort_keys=True),
                         encoding="utf-8")
    return {"md": md_path, "json": json_path}
