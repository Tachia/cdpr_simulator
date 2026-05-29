# Universal terminal execution guide

This guide walks a brand-new user from a clean clone to a working
simulation, in any of the supported terminal environments:

| Shell / environment | Verified |
|---|---|
| Windows **PowerShell 5.1 & 7+** | ✅ |
| Windows **CMD (cmd.exe)** | ✅ |
| **Git Bash** for Windows | ✅ |
| Linux **bash** / **zsh** | ✅ |
| macOS **bash** / **zsh** | ✅ |
| **VSCode integrated terminal** | inherits from above |
| Remote **SSH** terminal | inherits from above |
| Cloud notebook terminals (Colab, Kaggle, HF Spaces) | ✅ |

The simulator is **pure Python** above NumPy and SciPy — every
verified path uses identical Python invocations. The only differences
between shells are how you set environment variables and chain
commands.

---

## Step 1 — Prerequisites

* **Python 3.10, 3.11, or 3.12** (CI is green on 3.11 and 3.12).
* **Git**.
* (Optional) a clean virtual environment.

Check what you have:

```powershell
# PowerShell / CMD
python --version
git --version
```

```bash
# bash / zsh
python3 --version
git --version
```

---

## Step 2 — Clone the repository

```powershell
# PowerShell
git clone https://github.com/Tachia/cdpr_simulator.git
cd cdpr_simulator
```

```bash
# bash / zsh
git clone https://github.com/Tachia/cdpr_simulator.git
cd cdpr_simulator
```

CMD users: same commands, no changes.

---

## Step 3 — Create and activate a virtual environment

A `venv` is recommended (keeps the project's deps separate from system
Python). It is not strictly required — system Python with the right
packages works too.

### PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked by execution policy:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Windows CMD

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### Linux / macOS / Git Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
```

After activation `which python` (bash) or `Get-Command python`
(PowerShell) should point inside `.venv`.

---

## Step 4 — Install the package

Pick the install profile matching what you want to do.

### Minimum (CLI simulation + plots)

```bash
# Works in every shell above:
python -m pip install --upgrade pip
python -m pip install -e ".[viz,data,api]"
```

### + Streamlit local GUI

```bash
python -m pip install -e ".[viz,data,api,gui]"
```

### + Gradio local GUI

```bash
python -m pip install -e ".[viz,data,api,gradio]"
```

### + Data-driven training (PINN / MLP / PPO / SAC)

```bash
python -m pip install -e ".[viz,data,api,learn,rl]"
```

### Everything

```bash
python -m pip install -e ".[all]"
# then individually add gui and/or gradio if you want them:
python -m pip install -e ".[gui,gradio]"
```

---

## Step 5 — Run a built-in example end-to-end

```powershell
# PowerShell
python scripts\run_example.py --list
python scripts\run_example.py --name circle --open
```

```bash
# bash / zsh / Git Bash / CMD
python scripts/run_example.py --list
python scripts/run_example.py --name circle
```

The example produces `out/example-circle/` containing 14 figures +
`timeseries.csv` + `manifest.json` + `feasibility.json`. The `--open`
flag (PowerShell only) opens the folder in Explorer.

Other examples:

```bash
python scripts/run_example.py --name spiral     # 6-DoF helical spiral
python scripts/run_example.py --name mshape     # pick-and-place letter M
python scripts/run_example.py --name train      # Phase-2 PINN
python scripts/run_example.py --name compare    # Phase-2 multi-model bench
```

---

## Step 6 — Run a custom Phase-1 simulation

Direct simulation with CLI flags:

```powershell
# PowerShell (back-tick line continuations)
python scripts\run_simulation.py `
    --robot dissertation_8cable `
    --kind circle --radius 0.05 --duration 12 --dt 1e-3 `
    --controller pd --kp-pos 400 --kp-rot 100 `
    --t-min 5 --t-max 500 `
    --out out\my-run --open
```

```bash
# bash / zsh (backslash continuations)
python scripts/run_simulation.py \
    --robot dissertation_8cable \
    --kind circle --radius 0.05 --duration 12 --dt 1e-3 \
    --controller pd --kp-pos 400 --kp-rot 100 \
    --t-min 5 --t-max 500 \
    --out out/my-run
```

Heavy industrial scenario from a JSON robot config:

```bash
python scripts/run_simulation.py \
    --robot-config examples/robots/industrial_heavy.json \
    --kind circle --radius 1.0 --duration 20 \
    --t-min 50 --t-max 5000 \
    --out out/industrial-circle
```

Run `python scripts/run_simulation.py --help` for the full flag list.

---

## Step 7 — Phase-2 CSV training / comparison

Single PINN fit on a CSV the simulator just produced:

```bash
python scripts/train_from_csv.py \
    --input out/example-circle/timeseries.csv \
    --model pinn --epochs 80
```

Single model from a **URL**:

```bash
python scripts/train_from_csv.py \
    --input "https://example.com/data.csv" \
    --model mlp --epochs 60
```

Multi-model comparison:

```bash
python scripts/compare_models.py \
    --input out/example-circle/timeseries.csv \
    --out  out/compare \
    --models replay mlp pinn ppo sac \
    --epochs 60 --rl-steps 2000 --eval-episodes 2
```

CSV with non-standard column names? Supply an explicit mapping:

```bash
python scripts/train_from_csv.py \
    --input data.csv --model pinn \
    --column-map "px=Position X,py=Position Y,pz=Position Z"
```

Interactive PowerShell wrapper (prompts for CSV + model menu):

```powershell
.\scripts\train_interactive.ps1
```

---

## Step 8 — Launch a GUI

### Streamlit (local dev)

```bash
streamlit run streamlit_app.py
# open http://localhost:8501
```

### Gradio (local — recommended for casual use)

```bash
python gradio_app.py
# open http://localhost:7860
```

### FastAPI backend (local)

```bash
uvicorn cdpr.interface.api:app --reload --port 8000
# open http://localhost:8000/docs
```

These all run in-process against the cdpr scientific core; you can
launch them concurrently in three terminals if you want all three
URLs alive at once.

---

## Step 9 — Diagnostics & troubleshooting

```bash
# Check the cdpr import + version + which Python loaded it
python -c "import cdpr; print(cdpr.__file__)"

# Run the test subset that CI also runs:
python -m pytest tests/ -q \
    --ignore=tests/test_learn_env.py \
    --ignore=tests/test_learn_supervised.py \
    --ignore=tests/test_learn_benchmark.py \
    --ignore=tests/test_learn_rl_factories.py \
    --ignore=tests/test_adapters_mujoco.py

# Probe which extras are installed
python -c "import importlib.util as u; print({k: bool(u.find_spec(k)) for k in ('torch','gymnasium','stable_baselines3','streamlit','gradio')})"

# Hit the deployed FastAPI for liveness
# PowerShell:
.\scripts\call_render.ps1 -Action health
# bash:
curl -s https://cdpr-api.onrender.com/health
```

Common pitfalls:

| Symptom | Cause | Fix |
|---|---|---|
| `python: command not found` on macOS | macOS ships `python3` not `python` | Use `python3` (or `alias python=python3` in your shell rc) |
| `Activate.ps1 cannot be loaded` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ModuleNotFoundError: cdpr` | Forgot `-e .` install | Re-run Step 4 |
| Cold first call to Render API takes ~50 s | Free-tier sleep after 15 min idle | First call wakes the worker; subsequent calls are 200-300 ms |
| `torch` not installed warning | Phase-2 model needs the `learn` extra | `pip install -e ".[learn,rl]"` |
| `tracking_error` plot fails with `\lVert` | Pre-2026-05-28 commit | `git pull` |

---

## One-page cheat sheet

```bash
# Clean clone to first sim in under 10 minutes.
git clone https://github.com/Tachia/cdpr_simulator.git
cd cdpr_simulator
python -m venv .venv
# Activate per your shell (see Step 3)
python -m pip install -e ".[viz,data,api]"
python scripts/run_example.py --name circle
ls out/example-circle/                # 14 PNGs + CSV + manifest
```

---

## Cross-references

* `docs/frontend-architecture.md` — why Gradio is the recommended hosted GUI
* `docs/examples.md` — the 5 built-in examples
* `docs/csv-schema.md` — accepted CSV layouts + alias table
* `docs/runbook.md` — operational runbook
