"""Preprocessing report file output."""

from __future__ import annotations

import json

import pandas as pd

from cdpr.ingest import (
    ColumnMap,
    Pipeline,
    load_dataframe,
    write_preprocessing_report,
)


def test_report_writes_markdown_and_json(tmp_path):
    df = pd.DataFrame({
        "t": [0.0, 0.1, 0.2, 0.3, 0.4],
        "x": [0.0, 0.1, 0.2, 0.3, 0.4],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
        "z": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    raw = load_dataframe(df, source_path="test", format="memory")
    cmap = ColumnMap(time="t", position=("x", "y", "z"))
    pipeline = (
        Pipeline(raw, columns=cmap)
        .drop_nan()
        .resample(dt=0.05, method="linear")
    )
    exp = pipeline.run()

    paths = write_preprocessing_report(exp, tmp_path, title="Smoke test report")
    assert paths["md"].exists()
    assert paths["json"].exists()

    md = paths["md"].read_text()
    for header in ("# Smoke test report", "## Source", "## Column map",
                   "## Pipeline", "## Output statistics"):
        assert header in md

    payload = json.loads(paths["json"].read_text())
    assert payload["title"] == "Smoke test report"
    assert payload["n_samples_in"] == 5
    assert payload["n_samples_out"] == 9       # 0..0.4 inclusive at dt=0.05
    assert len(payload["steps"]) == 2
