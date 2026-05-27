"""Pipeline: cleaning, resampling, filtering, units, recorded step log."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cdpr.ingest import ColumnMap, Pipeline, load_dataframe


def _make_noisy_log(rng: np.random.Generator, n: int = 200) -> pd.DataFrame:
    """A 200-sample log with NaNs, duplicates, outliers, and uneven dt."""
    t = np.sort(rng.uniform(0, 1.0, size=n))             # non-uniform timestamps
    x = 0.1 * np.sin(2 * np.pi * 1.5 * t) + 1e-3 * rng.standard_normal(n)
    y = 0.05 * np.cos(2 * np.pi * 0.7 * t)
    z = np.linspace(0.0, 0.05, n)
    T1 = 100 + 5 * rng.standard_normal(n)

    df = pd.DataFrame({"t": t, "x": x, "y": y, "z": z, "T1": T1})
    # Inject a handful of NaNs.
    nan_idx = rng.choice(n, size=5, replace=False)
    df.loc[nan_idx, "x"] = np.nan
    # Duplicate one timestamp.
    df = pd.concat([df, df.iloc[[10]]], ignore_index=True).sort_values("t").reset_index(drop=True)
    # Insert a clear outlier in x.
    df.loc[50, "x"] = 100.0
    return df


def test_pipeline_runs_full_chain():
    rng = np.random.default_rng(42)
    df = _make_noisy_log(rng)
    raw = load_dataframe(df, source_path="memory", format="memory")
    cmap = ColumnMap(time="t", position=("x", "y", "z"), cable_tensions=("T1",))

    pipeline = (
        Pipeline(raw, columns=cmap)
        .deduplicate_timestamps(strategy="mean")
        .remove_outliers(method="mad", threshold=4.0, action="nan")
        .interpolate_missing(method="linear")
        .resample(dt=5e-3, method="linear", t_start=0.0, t_end=1.0)
        .lowpass(cutoff_hz=10.0, order=4)
    )
    exp = pipeline.run()

    assert len(exp.steps) == 5
    # Explicit [0, 1] grid at dt=5e-3 -> 201 samples.
    assert exp.positions is not None
    assert len(exp.time) == 201
    assert exp.positions.shape == (201, 3)
    # No NaNs survive.
    assert not np.isnan(exp.positions).any()
    assert not np.isnan(exp.cable_tensions).any()
    # The outlier should be gone after MAD + interpolation.
    assert np.abs(exp.positions[:, 0]).max() < 1.0


def test_pipeline_validates_columnmap_on_construction():
    df = pd.DataFrame({"t": [0.0, 1.0], "x": [0.0, 1.0]})
    raw = load_dataframe(df)
    bad = ColumnMap(time="t", position=("x", "y", "z"))   # y, z missing
    with pytest.raises(KeyError):
        Pipeline(raw, columns=bad)


def test_pipeline_step_records_track_row_count_changes():
    df = pd.DataFrame({
        "t": [0.0, 0.0, 0.1, 0.2, 0.3],     # 0.0 duplicated
        "x": [1.0, 1.0, 2.0, np.nan, 3.0],
    })
    raw = load_dataframe(df)
    cmap = ColumnMap(time="t", position=("x", "x", "x"))   # nonsense map; OK for this test
    # Use a sensible single-column map instead.
    cmap = ColumnMap(time="t", cable_lengths=("x",))
    pipeline = (
        Pipeline(raw, columns=cmap)
        .drop_nan()
        .deduplicate_timestamps(strategy="first")
    )
    exp = pipeline.run()
    # drop_nan removes one row, dedupe collapses two duplicates.
    assert exp.steps[0].rows_after == 4
    assert exp.steps[1].rows_after == 3


def test_resample_with_quaternion_uses_slerp():
    from scipy.spatial.transform import Rotation

    # Two timestamps with a 90 deg rotation between them.
    # scipy expects shape (N, len(seq)) so a single-axis sequence needs (N, 1).
    rots = Rotation.from_euler("z", np.array([[0.0], [90.0]]), degrees=True)
    q = rots.as_quat()
    df = pd.DataFrame({
        "t": [0.0, 1.0],
        "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3],
    })
    raw = load_dataframe(df)
    cmap = ColumnMap(time="t", quaternion=("qx", "qy", "qz", "qw"))
    exp = Pipeline(raw, columns=cmap).resample(dt=0.5, method="linear").run()
    # At t=0.5, SLERP should give a 45 deg z-rotation.
    mid = Rotation.from_quat(exp.quaternions_xyzw[1]).as_euler("xyz", degrees=True)
    assert mid[2] == pytest.approx(45.0, abs=1e-9)
