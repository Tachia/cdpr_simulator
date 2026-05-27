"""Phase-3 end-to-end smoke: a realistic dirty log -> cleaned experiment.

Generates a synthetic 2-second IPAnema-class trial with:
* irregular timestamps (jittered around 200 Hz),
* a handful of dropped samples (NaN gaps),
* a single timestamp duplicated by a tracker hiccup,
* an outlier spike on cable T1,
* additive Gaussian sensor noise on every channel,
* positions in MILLIMETRES (the typical mocap output unit).

The pipeline:
1. loads the CSV,
2. de-duplicates the timestamp collision,
3. removes the outlier (MAD flag + NaN replacement),
4. interpolates the NaN gaps,
5. resamples onto a uniform 500 Hz grid,
6. low-passes at 30 Hz,
7. converts position from mm to m.

Then validates the result against the IPAnema-class robot, writes the
preprocessing report, and prints summary residuals.

Artifacts land in ``runs/phase3_smoke/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.ingest import (
    ColumnMap,
    Pipeline,
    load_csv,
    validate_against_robot,
    write_preprocessing_report,
)
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def _generate_dirty_log(out_csv: Path, robot, traj, *, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    # Simulate a ground-truth trajectory.
    state0 = PlatformState.at_rest(traj.pose(0.0))
    sim = simulate(robot=robot, state0=state0, duration=2.0, dt=2e-3, reference=traj)

    # Resample onto irregular timestamps: 200 Hz nominal with +/- 1 ms jitter.
    nominal = np.arange(0.0, 2.0, 1.0 / 200.0)
    jitter = rng.uniform(-0.001, 0.001, size=nominal.shape)
    t_jittered = np.clip(nominal + jitter, sim.time[0], sim.time[-1])
    # Interpolate ground truth onto these timestamps for the synthetic measurement.
    def interp(arr):
        return np.column_stack([np.interp(t_jittered, sim.time, arr[:, j])
                                for j in range(arr.shape[1])])
    pos_mm = 1000.0 * interp(sim.positions)             # millimetres
    quat = interp(sim.quaternions_xyzw)
    tens = interp(sim.cable_tensions)
    lens_mm = 1000.0 * interp(sim.cable_lengths)

    # Add Gaussian sensor noise.
    pos_mm += rng.normal(scale=0.5, size=pos_mm.shape)  # 0.5 mm = realistic mocap noise
    tens += rng.normal(scale=2.0, size=tens.shape)      # ~2 N tension sensor noise

    # Inject artefacts:
    # ... drop five samples by inserting NaN in position
    drop_idx = rng.choice(len(t_jittered), size=5, replace=False)
    pos_mm[drop_idx, 0] = np.nan
    # ... duplicate one timestamp
    t_jittered = np.concatenate([t_jittered, [t_jittered[50]]])
    pos_mm = np.vstack([pos_mm, pos_mm[[50]]])
    quat = np.vstack([quat, quat[[50]]])
    tens = np.vstack([tens, tens[[50]]])
    lens_mm = np.vstack([lens_mm, lens_mm[[50]]])
    # ... insert one outlier on T1
    spike = 30
    tens[spike, 0] = 5_000.0

    # Assemble the dataframe in lab format.
    df = pd.DataFrame({"t": t_jittered})
    for k, ax in enumerate("xyz"):
        df[ax] = pos_mm[:, k]
    for k, c in enumerate("xyzw"):
        df[f"q{c}"] = quat[:, k]
    for k in range(tens.shape[1]):
        df[f"T{k + 1}"] = tens[:, k]
        df[f"L{k + 1}"] = lens_mm[:, k]
    df = df.sort_values("t").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main(out_root: Path = Path("runs/phase3_smoke")) -> None:
    robot = ipanema_class()
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.25, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=2.0),
    )

    raw_path = out_root / "trial.csv"
    print(f"[1/5] Generating dirty CSV at {raw_path}")
    _generate_dirty_log(raw_path, robot, traj)

    print("[2/5] Loading and auto-detecting columns...")
    raw = load_csv(raw_path)
    cmap = ColumnMap.autodetect(raw)
    # Autodetect picks up positions and quaternions; for cable channels we
    # know the lab layout (T1..T8 / L1..L8), so set them explicitly.
    cmap.cable_tensions = tuple(f"T{i + 1}" for i in range(robot.n_cables))
    cmap.cable_lengths = tuple(f"L{i + 1}" for i in range(robot.n_cables))

    print("[3/5] Running pipeline...")
    pipeline = (
        Pipeline(raw, columns=cmap)
        .deduplicate_timestamps(strategy="mean")
        .remove_outliers(method="mad", threshold=4.0, action="nan")
        .interpolate_missing(method="linear")
        .resample(dt=2e-3, method="cubic", t_start=0.0, t_end=2.0)
        .lowpass(cutoff_hz=30.0, order=4)
        .convert_units(position_scale=1e-3, cable_length_scale=1e-3)
    )
    experiment = pipeline.run()
    print(f"     -> {experiment.source['n_rows_raw']} raw samples "
          f"-> {len(experiment.time)} cleaned samples")

    print("[4/5] Validating against the IPAnema-class model...")
    validation = validate_against_robot(experiment, robot, reconstruct=False)
    if validation.ik_length_residual:
        v = validation.ik_length_residual
        print(f"     IK length residual:  RMS={v.rms:.4e} m, peak={v.peak:.4e} m")
    if validation.tension_wrench_residual:
        v = validation.tension_wrench_residual
        print(f"     Tension wrench residual:  RMS={v.rms:.4e}, peak={v.peak:.4e}")

    print("[5/5] Writing preprocessing report...")
    paths = write_preprocessing_report(
        experiment, out_root, title="Phase 3 smoke: dirty IPAnema trial",
        validation=validation,
    )
    print(f"     markdown: {paths['md']}")
    print(f"     json:     {paths['json']}")
    print(f"\nDone. Artifacts in: {out_root.resolve()}")


if __name__ == "__main__":
    main()
