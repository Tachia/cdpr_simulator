# CDPR Simulator — runbook

Two execution modes, two workflows each. The CLI is the guaranteed
fallback whenever the browser path misbehaves.

```
                +----------------------------------------+
                |          You — PowerShell or browser    |
                +----------------+------------------------+
                                 |
        +------------------------+--------------------------+
        |                                                   |
        v                                                   v
 +-------------+                                +---------------------+
 | PowerShell  |                                | Browser (Streamlit) |
 | CLI scripts |                                |   + FastAPI on Render|
 +------+------+                                +----------+----------+
        |                                                  |
        v                                                  v
 +-------------+        +----------------+         +-----------------+
 | scripts/    |        | streamlit_app  |         | cdpr.interface  |
 | run_simu    |        | gui.py         |         | .api (FastAPI)  |
 | train_csv   |        | upload panel   |         | /simulate /plot |
 | compare     |        |                |         | /workspace      |
 +-----+-------+        +--------+-------+         +--------+--------+
       |                         |                          |
       +---------+---------------+--------------------------+
                 |
                 v
        +------------------+
        | cdpr scientific  |   <-- single source of truth (numpy/scipy)
        | core             |       195 tests, both 3.11 and 3.12 green
        +------------------+
```

## Mode 1 — defined-trajectory simulation

### PowerShell

```powershell
# Defaults: ipanema_class, circle, 1.5 s, dt = 2 ms.
python scripts\run_simulation.py

# Full-circle 2-revolution sweep with the parameters from the directive.
python scripts\run_simulation.py `
    --robot ipanema_class `
    --kind circle `
    --radius 1.0 `
    --duration 125 `
    --dt 1e-3 `
    --center 0 0 0.5 `
    --open

# Any robot x any trajectory --- 4 x 4 combinations all validated.
python scripts\run_simulation.py --robot cogiro_class --kind lissajous --duration 30 --open
python scripts\run_simulation.py --robot planar_translational --kind line --duration 2
python scripts\run_simulation.py --robot point_mass_3d --kind hold --duration 1

# No plots --- CSV + manifest only (~5 s for a 125 s run).
python scripts\run_simulation.py --duration 125 --no-plots
```

Output (per run) under `out/<robot>-<kind>-<timestamp>/`:

| Artifact | Notes |
|---|---|
| `timeseries.csv` | t, px/py/pz, qx/qy/qz/qw, vx/vy/vz, wx/wy/wz, L1..Lm, T1..Tm |
| `manifest.json` | git hash + python + request + runtime |
| `position.png` | translational components |
| `velocity.png` | linear velocity components |
| `angular_velocity.png` | wx, wy, wz |
| `acceleration.png` | finite-difference from velocity |
| `cable_tensions.png` | per-cable, with bound shading |
| `cable_lengths.png` | per-cable length over time |
| `cable_stretch.png` | (L − L₀) per cable [mm] |
| `tracking_error.png` | ‖p − p_ref‖ |
| `rms_error_evolution.png` | cumulative RMS |
| `condition_number.png` | κ₂ of the structure matrix |
| `trajectory_xy.png` | actual + reference projection |
| `trajectory_xz.png` | actual + reference projection |
| `trajectory_yz.png` | actual + reference projection |
| `scene_3d.png` | end-of-run 3D scene with tension heatmap |

13 figure files per run for circle/line/lissajous; 12 for hold (no
tracking-error overlay).

### Browser

1. Open the live console: `https://cdprsimulator-a5u8bciz6tsnsxegg8zys2.streamlit.app`.
2. Configure the sidebar widgets (robot, kind, duration, dt, …).
3. Click **Run simulation** in the top action bar.
4. Each tab carries a **Render** button — figures are lazy because the
   free Streamlit Cloud worker has only ~1 GB. The 3D scene tab is
   marked *heavy* and is opt-in.
5. The **Diagnostics** expander at the bottom reports build id +
   Streamlit version + matplotlib backend + session-state keys.
6. If the page ever blanks: set the `CDPR_GUI_DIAG=1` secret in
   Streamlit Cloud → Settings → Secrets and reboot. The page swaps to
   a one-button counter that confirms whether the worker itself is
   healthy. While Streamlit Cloud is misbehaving, use the CLI above.

### FastAPI (deployed on Render)

```powershell
# Warm the worker (~50 s cold start the first time).
.\scripts\call_render.ps1 -Action health

# Run a simulation through the deployed API.
.\scripts\call_render.ps1 -Action simulate -Robot ipanema_class -Kind circle

# Render a plot server-side, save the returned PNG locally, open it.
.\scripts\call_render.ps1 -Action plot -PlotKind cable_tensions -OpenPng

# Workspace scan.
.\scripts\call_render.ps1 -Action workspace -Robot ipanema_class
```

Outputs land in `out\render-<timestamp>\`.

## Mode 2 — CSV-driven training / replay / comparison

### PowerShell

The same `timeseries.csv` produced in Mode 1 is the input here. Five
workflows; the ones whose extras are not installed exit cleanly with
the install hint (never with a silent traceback).

```powershell
# Analytic replay --- no extras needed, near-machine-precision against
# the recorded tensions.
python scripts\train_from_csv.py `
    --input out\<robot>-<kind>-<stamp>\timeseries.csv `
    --model replay

# Supervised MLP inverse-dynamics fit.
python scripts\train_from_csv.py `
    --input out\<stamp>\timeseries.csv `
    --model mlp --epochs 80

# Physics-informed variant.
python scripts\train_from_csv.py `
    --input out\<stamp>\timeseries.csv `
    --model pinn --epochs 80

# PPO evaluation (requires cdpr[rl]).
python scripts\train_from_csv.py `
    --input out\<stamp>\timeseries.csv `
    --model ppo --rl-steps 5000 --eval-episodes 3

# SAC evaluation.
python scripts\train_from_csv.py `
    --input out\<stamp>\timeseries.csv `
    --model sac --rl-steps 5000 --eval-episodes 3

# Cross-model comparison (runs each as a subprocess, ranks by RMSE).
python scripts\compare_models.py `
    --input out\<stamp>\timeseries.csv `
    --models replay mlp pinn `
    --out out\compare-<stamp> `
    --epochs 80
```

Per-model output under `out\<stamp>\<model>\` (or under
`out\compare-<stamp>\<model>\` when launched through the comparator):

* `loss.png`            — training / validation loss
* `pred_vs_truth.png`   — predicted vs recorded cable tensions
* `residuals.png`       — error histogram + per-cable boxplot
* `metrics.json`        — RMSE / MAE / per-cable error
* `manifest.json`       — git hash + python + hyper-params

Comparison root (`out\compare-<stamp>\`):

* `compare_rmse.png`    — per-model RMSE bar chart (log scale)
* `compare_runtime.png` — per-model wall time
* `compare_overview.png` — mean tension overlay vs ground truth
* `compare_table.png`   — model / RMSE / wall-time table
* `ranking.json`        — sorted ranking
* `compare_metrics.json` — full per-model metric blob

### Browser

The Streamlit upload panel now mirrors Mode 2 inline:

1. Run a Mode-1 simulation (CLI or browser); copy / drop the
   `timeseries.csv` into the **Phase 2 — upload an experimental log**
   block at the bottom of the page.
2. The CSV head appears as a 50-row preview.
3. Click **Quick PINN fit** — runs a 30-epoch supervised fit (~1 s
   on the deployed worker) and displays the loss + prediction overlay
   inline. RMSE / MAE land in the displayed metrics JSON.
4. Click **Quick replay** — replays the sidebar configuration's
   analytic trajectory against the CSV's tension columns and overlays
   them.
5. **Clear analysis** wipes the cached figures so the worker memory
   stays light.

## Local FastAPI + local Streamlit

```powershell
# Backend on http://localhost:8000
uvicorn cdpr.interface.api:app --reload --port 8000

# Console (in another terminal) on http://localhost:8501
streamlit run streamlit_app.py
```

When the local Streamlit is running, set `$env:CDPR_GUI_FRUGAL=0`
beforehand to lift the default-duration cap from 0.5 s to 1.5 s.

## Diagnostics that exist when things go wrong

| Symptom | First check | Recovery path |
|---|---|---|
| Streamlit Cloud page blank | Browser hard-refresh (Ctrl+F5) | Set `CDPR_GUI_DIAG=1` in Streamlit Cloud → Settings → Secrets, reboot. If the diagnostic page also blanks, the worker is unhealthy — use the CLI escape hatches above. |
| Render `/simulate` returns 502 | The free tier slept after 15 min idle | Wait ~50 s; the first `Action health` call wakes it. |
| CLI `--model pinn` says "torch not installed" | The dev install was viz/data only | `pip install -e ".[learn]"` |
| CLI `--model ppo` says "stable_baselines3 not installed" | Same | `pip install -e ".[rl]"` |
| CI flagged a test | GH Actions → latest run → expand "Run pytest" | The `-v --tb=short` flags surface the failing test name and traceback. |

## Service URLs

| Layer | URL |
|---|---|
| GitHub repo | <https://github.com/Tachia/cdpr_simulator> |
| Render FastAPI | <https://cdpr-api.onrender.com> (`/health`, `/robots`, `/simulate`, `/plot`, `/workspace`, `/docs`) |
| Cloudflare Pages docs | <https://cdpr-simulator.pages.dev> |
| Streamlit research console | <https://cdprsimulator-a5u8bciz6tsnsxegg8zys2.streamlit.app> |
| Supabase Postgres | `db.nohbtlhhisfiajjsbguy.supabase.co` (CLI only) |
| GitHub Actions CI | <https://github.com/Tachia/cdpr_simulator/actions> |
