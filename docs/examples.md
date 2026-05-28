# Built-in examples

Five selectable demonstrations cover the directive's two operational
phases. They are accessible identically from PowerShell and the
Streamlit GUI; the registry lives in `scripts/examples.py`.

| ID | Phase | DoF | Title | Description |
|---|---|---|---|---|
| `circle`  | 1 | 3 | Circular trajectory tracking | 0.05 m radius horizontal circle, 2 revolutions, ≈12.6 s, quintic smoothing. |
| `spiral`  | 1 | **6** | 6-DoF helical spiral | 3 spiral revs at z = 0.65 m ± 0.04 m, **roll / pitch / yaw oscillation** (±3 / ±6 / ±17 °), ≈15 s. |
| `mshape`  | 1 | 3+1 | Pick-and-place M-task | Letter-M trace through 5 vertices with dwells + discrete gripper yaw at each pick/place. |
| `train`   | 2 |   | Data-driven PINN training | 120-epoch PINN fit on the `circle` CSV (auto-generates it if missing). |
| `compare` | 2 |   | Multi-model comparison | Bench replay vs MLP vs PINN vs PPO vs SAC on the same dataset. |

## Decision on the 6-DoF showcase

The directive asked which of the three Phase-1 examples should
incorporate all six rigid-body DoFs. The **spiral** is the natural
choice:

* The position is already coupled in 3D space (x, y, z),
* Adding **roll / pitch / yaw oscillation** at distinct phase offsets
  exercises the three rotational DoFs without breaking workspace
  feasibility,
* The circle stays a clean 3-DoF planar demo (best fit for "first
  introduction to closed-loop tracking"), and the M-shape uses
  discrete gripper yaw at pick/place events (rather than continuous
  orientation tracking) to demonstrate payload-handling logic.

Orientation amplitudes are deliberately **bounded sinusoids**
(±17° yaw, ±6° pitch, ±3° roll) rather than monotonic rotations.
Without an orientation feedforward, a PD controller cannot keep up
with a continuously growing angle; bounded oscillation lets the
angular error stay inside the cables' wrench-feasibility envelope
under the 5–500 N tension limits while still exercising all six DoFs
visibly. The `orientation` plot in every example artifact bundle
makes this explicit.

## PowerShell usage

```powershell
# List all five with descriptions.
python scripts\run_example.py --list

# Run a single example end-to-end. Output lands under out/example-<name>/.
python scripts\run_example.py --name circle  --open
python scripts\run_example.py --name spiral  --open
python scripts\run_example.py --name mshape  --open
python scripts\run_example.py --name train   --open
python scripts\run_example.py --name compare --open
```

The Phase-2 examples (`train`, `compare`) check for the Phase-1 CSV
they depend on (`out/example-circle/timeseries.csv`) and auto-run the
`circle` example to generate it if missing.

## Streamlit GUI usage

The sidebar carries a "Built-in examples" panel at the top of the main
page. Pick an example from the dropdown:

* the description, target output directory, and required PowerShell
  command are shown,
* the **PowerShell command is always copy-paste ready** (works on a
  local terminal regardless of where the Streamlit app is running),
* on **local Streamlit only**, a **"Run inline (local only)"** button
  runs the example as a subprocess and displays the produced figures
  inline. Inline execution is disabled on Streamlit Cloud because the
  free-tier worker is too small for the full plot bundle.

## Output layout

Each Phase-1 example produces under `out/example-<id>/`:

| File | What |
|---|---|
| `timeseries.csv` | Full state log (Phase-2-ready) |
| `manifest.json` | Git hash, controller, robot spec, feasibility, trajectory recipe (catalog kind where reconstructible, `custom_callable` otherwise) |
| `feasibility.json` | t_min/t_max, observed tension range, infeasibility list, tracking RMS/peak/final |
| `position.png`, `velocity.png`, `angular_velocity.png`, `acceleration.png` | Per-axis state plots |
| `cable_tensions.png`, `cable_lengths.png`, `cable_stretch.png` | Per-cable diagnostics with bound shading |
| `tracking_error.png`, `rms_error_evolution.png` | Reference-vs-actual error and cumulative RMS |
| `condition_number.png` | Structure-matrix conditioning |
| `trajectory_xy.png`, `trajectory_xz.png`, `trajectory_yz.png` | 2D projections with reference overlay |
| `orientation.png` | Roll/pitch/yaw vs time (visible 6-DoF demonstration for `spiral`) |
| `scene_3d.png` | Final 3D scene with anchors, cables, platform, tension heatmap, trajectory trace |

That's **14 figures per Phase-1 run**, well above the directive's
≥10-figure minimum.

Each Phase-2 example produces under its `out/example-<id>/` (or per-
model subdirectories for `compare`):

| File | What |
|---|---|
| `loss.png`, `pred_vs_truth.png`, `residuals.png` | Training/eval curves |
| `metrics.json` | RMSE, MAE, per-cable error breakdown, train/val loss history |
| `manifest.json` | Hyper-params + git hash + python version |
| `compare_rmse.png`, `compare_runtime.png`, `compare_table.png`, `compare_overview.png` | Multi-model bar charts and ranking table (compare only) |
| `ranking.json`, `compare_metrics.json` | Sorted ranking with `ok` / `failed` / `skipped` status (compare only) |

## Reproducibility

All five examples are deterministic given the registered seeds
(supervised models use `--seed 0`; the RL evals use `seed=0+episode`).
Re-running `python scripts/run_example.py --name <id>` on the same
checkout produces byte-equivalent CSVs and reproducible figures.

The manifest carries the `robot_spec` block from improvement #2 of
the v17 sweep, so Phase-2 replay/PPO/SAC reconstruct the directive
robot exactly. Spiral and mshape use inline Python callables for
their references (full SE(3) tracking and piecewise pick-and-place
respectively); their manifests therefore mark `trajectory.kind` as
`custom_callable` and replay/RL Phase-2 workflows skip them cleanly
with a documented reason — supervised models (`mlp`, `pinn`) run on
all three Phase-1 CSVs because they only need the timeseries.
