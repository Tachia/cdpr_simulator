r"""On-disk experiment bundle and loader."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ExperimentBundle:
    """Reference to a complete on-disk experiment artifact set.

    Directory layout::

        <output_root>/<name>_<config_hash>/
            manifest.json             reproducibility manifest
            config.json               full experiment configuration
            backends.json             availability probe at run time
            runs/
                <run_id>/
                    metrics.json
                    timeseries.csv         (if write_full_timeseries)
                    reference.csv          (if write_full_timeseries)
            report/                   (if write_bundle_report)
                figures/, tables/, summary.md
    """

    root: Path
    manifest_path: Path
    config_path: Path
    backends_path: Path
    runs_dir: Path
    report_dir: Path | None = None
    run_records: list[dict[str, Any]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "n_runs": len(self.run_records),
            "report": str(self.report_dir) if self.report_dir else None,
        }


def load_bundle(root: str | Path) -> ExperimentBundle:
    """Load a previously-written bundle from disk.

    The returned :class:`ExperimentBundle` carries the on-disk paths;
    use the JSON files directly to re-instantiate scenarios in user
    code if a full re-run is needed.
    """
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json under {root!r}; not an experiment bundle.")
    config_path = root / "config.json"
    backends_path = root / "backends.json"
    runs_dir = root / "runs"
    report_dir = root / "report" if (root / "report").exists() else None

    run_records: list[dict[str, Any]] = []
    if runs_dir.exists():
        for run_path in sorted(runs_dir.iterdir()):
            metrics_path = run_path / "metrics.json"
            if metrics_path.exists():
                rec = {"id": run_path.name, "metrics": json.loads(metrics_path.read_text())}
                run_records.append(rec)

    return ExperimentBundle(
        root=root,
        manifest_path=manifest_path,
        config_path=config_path,
        backends_path=backends_path,
        runs_dir=runs_dir,
        report_dir=report_dir,
        run_records=run_records,
    )
