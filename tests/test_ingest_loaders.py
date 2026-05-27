"""Loader smoke tests for every supported format."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cdpr.ingest import (
    Channel,
    ColumnMap,
    load_csv,
    load_dataframe,
    load_json,
    load_txt,
    load_xlsx,
)


@pytest.fixture
def synthetic_log(tmp_path):
    """Build a small synthetic experimental log in every supported format."""
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "t": np.linspace(0.0, 0.4, n),
        "x": rng.standard_normal(n) * 0.1,
        "y": rng.standard_normal(n) * 0.1,
        "z": rng.standard_normal(n) * 0.1,
        "T1": 100 + rng.standard_normal(n),
        "T2": 100 + rng.standard_normal(n),
        "T3": 100 + rng.standard_normal(n),
        "T4": 100 + rng.standard_normal(n),
    })
    csv_path = tmp_path / "log.csv"
    txt_path = tmp_path / "log.txt"
    xlsx_path = tmp_path / "log.xlsx"
    json_path = tmp_path / "log.json"
    df.to_csv(csv_path, index=False)
    df.to_csv(txt_path, index=False, sep=" ")
    df.to_excel(xlsx_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="list")))
    return df, csv_path, txt_path, xlsx_path, json_path


def test_load_csv(synthetic_log):
    df_ref, csv_path, *_ = synthetic_log
    raw = load_csv(csv_path)
    assert raw.format == "csv"
    assert raw.n_rows_raw == len(df_ref)
    assert set(raw.columns) == set(df_ref.columns)


def test_load_txt(synthetic_log):
    _, _, txt_path, *_ = synthetic_log
    raw = load_txt(txt_path)
    assert raw.format == "txt"
    assert "t" in raw.columns


def test_load_xlsx(synthetic_log):
    _, _, _, xlsx_path, _ = synthetic_log
    raw = load_xlsx(xlsx_path)
    assert raw.format == "xlsx"
    assert "T1" in raw.columns


def test_load_json_columnar(synthetic_log):
    _, _, _, _, json_path = synthetic_log
    raw = load_json(json_path)
    assert raw.format == "json"
    assert "x" in raw.columns


def test_load_json_records(tmp_path):
    records = [{"t": 0.0, "x": 1.0}, {"t": 0.1, "x": 2.0}]
    p = tmp_path / "rec.json"
    p.write_text(json.dumps(records))
    raw = load_json(p)
    assert raw.n_rows_raw == 2


def test_load_dataframe_passthrough():
    df = pd.DataFrame({"t": [0.0, 0.1], "x": [1.0, 2.0]})
    raw = load_dataframe(df, source_path="virtual.csv", format="custom")
    assert raw.format == "custom"
    assert raw.n_rows_raw == 2


def test_columnmap_autodetect_finds_position_and_tensions(synthetic_log):
    df_ref, csv_path, *_ = synthetic_log
    raw = load_csv(csv_path)
    cmap = ColumnMap.autodetect(raw)
    assert cmap.time == "t"
    assert cmap.position == ("x", "y", "z")
    assert cmap.cable_tensions == ("T1", "T2", "T3", "T4")
    assigned = cmap.assigned_columns()
    assert Channel.TIME in assigned
    assert Channel.POSITION in assigned
    assert Channel.CABLE_TENSIONS in assigned
