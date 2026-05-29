# Running the CDPR Simulator locally — PowerShell step-by-step

The local Gradio UI at `http://127.0.0.1:7860` is the primary
interface for daily work. Streamlit and Dash are independent
secondary interfaces. None of them depend on Hugging Face
deployment — the cloud Space is purely a public-demo convenience.

> **About `127.0.0.1:7860`** — this address is loopback only. It
> exists *while the Python process is running* in your terminal. Close
> the terminal and the page goes 404. For a temporary public URL
> without HF Spaces you can use `demo.launch(share=True)` inside
> `gradio_app.py` (the line is commented out for safety; uncomment
> when you want it).

## Prerequisites

* Python 3.10 / 3.11 / 3.12
* Git
* (Recommended) a virtual environment

## 1) Clone and enter the repo

```powershell
git clone https://github.com/Tachia/cdpr_simulator.git
cd cdpr_simulator
```

## 2) Create + activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# If activation is blocked by execution policy:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

After activation `where.exe python` should point inside `.venv`.

## 3) Install the package and the UI extras you want

```powershell
# Minimum (CLI + plots):
python -m pip install --upgrade pip
python -m pip install -e ".[viz,data,api]"

# + Gradio UI (recommended local interface):
python -m pip install -e ".[viz,data,api,gradio]"

# + Streamlit (secondary dashboard):
python -m pip install -e ".[viz,data,api,gui]"

# + Dash (interactive scientific dashboard):
python -m pip install -e ".[viz,data,api,dash]"

# + Phase-2 training (PINN / MLP / PPO / SAC):
python -m pip install -e ".[viz,data,api,learn,rl]"

# Everything at once:
python -m pip install -e ".[viz,data,api,gradio,gui,dash,learn,rl]"
```

## 4) Launch the primary interface (Gradio)

```powershell
python gradio_app.py
# Opens at http://127.0.0.1:7860/
```

That's it. The terminal must stay open — closing it kills the server.

You should see three tabs: **Built-in examples**, **Custom Phase-1
simulation**, **Upload CSV / Phase-2**. Output of any run lands in
`out/` and persists across launches.

## 5) Launch the secondary interfaces

Open additional PowerShell windows (the current one keeps Gradio
running):

```powershell
# Streamlit dashboard at http://localhost:8501
streamlit run streamlit_app.py
```

```powershell
# Dash research dashboard at http://127.0.0.1:8050
python dash_app.py
```

```powershell
# FastAPI backend at http://localhost:8000  (interactive docs at /docs)
uvicorn cdpr.interface.api:app --reload --port 8000
```

All four can run side-by-side on the same machine — they share the
same `cdpr` core and the same `out/` directory.

## 6) Run a simulation from the terminal (no UI)

```powershell
# List the 5 built-in examples:
python scripts\run_example.py --list

# Run any of them end-to-end:
python scripts\run_example.py --name circle --open
python scripts\run_example.py --name spiral --open
python scripts\run_example.py --name mshape --open
python scripts\run_example.py --name train  --open
python scripts\run_example.py --name compare --open
```

Each run produces `out/example-<name>/` with 14 figures +
`timeseries.csv` + `manifest.json` + `feasibility.json`. The `--open`
flag pops the folder in File Explorer on Windows.

## 7) Custom simulations from the terminal

```powershell
python scripts\run_simulation.py `
    --robot dissertation_8cable `
    --kind circle --radius 0.05 --duration 12 --dt 1e-3 `
    --controller pd --kp-pos 400 --kp-rot 100 `
    --t-min 5 --t-max 500 `
    --out out\my-run --open
```

See `python scripts\run_simulation.py --help` for every flag.

## 8) Phase-2 training and comparison

```powershell
# Train PINN on any timeseries.csv:
python scripts\train_from_csv.py `
    --input out\example-circle\timeseries.csv `
    --model pinn --epochs 80

# 5-model comparison:
python scripts\compare_models.py `
    --input out\example-circle\timeseries.csv `
    --out  out\compare `
    --models replay mlp pinn ppo sac `
    --epochs 60 --rl-steps 2000

# Interactive prompts (CSV path → model menu 1-7 → run):
.\scripts\train_interactive.ps1
```

## Where each interface fits

| Surface | URL while running | Best for |
|---|---|---|
| **Gradio** (`python gradio_app.py`) | `http://127.0.0.1:7860/` | One-click examples, file uploads, casual experimentation |
| **Dash** (`python dash_app.py`) | `http://127.0.0.1:8050/` | Interactive Plotly plots (zoom + hover), parameter tweaking that doesn't re-run the simulation on every tick |
| **Streamlit** (`streamlit run streamlit_app.py`) | `http://localhost:8501/` | Heavier dashboards, multi-page Streamlit-style interaction |
| **FastAPI** (`uvicorn …`) | `http://localhost:8000/docs` | Programmatic access from notebooks / curl / other apps |
| **CLI** (`python scripts\run_example.py …`) | terminal only | Reproducible batch runs, dissertation regeneration |

## When something doesn't work

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` |
| `ModuleNotFoundError: cdpr` | The package isn't installed in the active venv — re-run step 3 |
| Browser shows `Address can't be reached` at `127.0.0.1:7860` | The Python process is not running — start `python gradio_app.py` again |
| `Port 7860 is already in use` | Another Gradio is already running — close the other terminal, or change the port with `$env:GRADIO_SERVER_PORT="7861"; python gradio_app.py` |
| Old simulation results polluting `out/` | `Remove-Item out -Recurse` — outputs are reproducible from `python scripts\run_example.py --name <id>` |
| Streamlit blanks on Cloud | Expected on the free 1 GB worker — use the local Gradio / Dash UIs instead. See [docs/frontend-architecture-revised.md](frontend-architecture-revised.md) for the architectural rationale. |

## Quick reference card

```powershell
# Clean clone to first simulation in under 10 minutes:
git clone https://github.com/Tachia/cdpr_simulator.git
cd cdpr_simulator
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[viz,data,api,gradio]"
python gradio_app.py
# Open http://127.0.0.1:7860/
```

## Cross-references

* [docs/hf-deployment.md](hf-deployment.md) — push to a Hugging Face Space without the HF CLI
* [docs/terminal-execution.md](terminal-execution.md) — bash / zsh / CMD variants of the commands above
* [docs/examples.md](examples.md) — what each built-in example does
* [docs/frontend-architecture-revised.md](frontend-architecture-revised.md) — why each interface exists
