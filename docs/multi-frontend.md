# Multi-frontend topology and deployment guide

How the cdpr scientific core feeds multiple coexisting frontends, and
how to deploy each.

## Topology

```
                ┌──────────────────────────────────────┐
                │   cdpr core (Python, NumPy/SciPy)    │   one source of truth
                │   src/cdpr/…                         │   195 tests
                └────────────┬─────────────────────────┘
                             │ Python import
        ┌────────────────────┼────────────────────────────────┐
        │                    │                                │
        ▼                    ▼                                ▼
 PowerShell / bash CLI   FastAPI backend                Examples registry
 scripts/*.py            cdpr.interface.api             scripts/examples.py
 (run_simulation,        (deployed on Render)           (5 built-in demos)
  train_from_csv,
  compare_models,
  run_example,
  train_interactive.ps1)
        │                    │                                │
        └────────────────────┼──────────────────┐             │
                             │                  │             │
              ┌──────────────┼────────┐         │             │
              ▼              ▼        ▼         ▼             ▼
        Streamlit (local)  Gradio  Dash (next)  curl     all GUIs read
        streamlit_app.py   gradio_app.py        users     EXAMPLES from
                                                          examples.py
```

**Properties:**

1. **Each frontend imports cdpr directly.** No HTTP boundary required
   between the GUI and the simulator. Each GUI runs the simulator
   in-process, then renders the artifacts.
2. **Disk is the shared artifact contract.** Every successful run
   writes `out/<id>/timeseries.csv`, `manifest.json`,
   `feasibility.json`, plus 14 PNGs. Any other frontend can read those.
3. **FastAPI is the optional network boundary.** Clients without
   Python (curl, mobile, JS) hit `/simulate`, `/plot`, `/workspace`
   instead of importing.
4. **The examples registry is the only state the GUIs share** (via
   `scripts/examples.py`). Adding a new example surfaces it
   simultaneously in the CLI, Streamlit, and Gradio.

## Deployment surfaces

| Surface | What runs there | URL |
|---|---|---|
| GitHub (sources) | All code | github.com/Tachia/cdpr_simulator |
| Render | FastAPI backend (Docker) | cdpr-api.onrender.com |
| Cloudflare Pages | Static docs landing | cdpr-simulator.pages.dev |
| Streamlit Cloud | Streamlit GUI (experimental — see frontend-architecture.md) | cdprsimulator-…streamlit.app |
| **Hugging Face Spaces** | **Gradio GUI (recommended)** | huggingface.co/spaces/<user>/cdpr-simulator |
| Supabase | Postgres + storage (optional) | db.nohbtlhhisfiajjsbguy.supabase.co |
| Local dev box | All of the above (concurrently if desired) | localhost |

## How to deploy the new Gradio frontend to Hugging Face Spaces

This is the recommended hosted demonstration URL going forward.

### One-time setup

1. Make a free account at <https://huggingface.co>.
2. New → **Space**:
   * Owner: your username
   * Space name: `cdpr-simulator`
   * License: MIT
   * Space SDK: **Gradio**
   * Space hardware: **CPU basic** (free; 16 GB RAM, 2 vCPU)
   * Visibility: Public (or Private if you prefer)
3. After creation, get the URL of the form:
   `https://huggingface.co/spaces/<your-username>/cdpr-simulator`.

### Upload the files

From your local checkout:

```powershell
# PowerShell
git remote add space https://huggingface.co/spaces/<your-username>/cdpr-simulator
git push space main
```

```bash
# bash / zsh
git remote add space https://huggingface.co/spaces/<your-username>/cdpr-simulator
git push space main
```

HF will pick up `gradio_app.py` and `requirements-gradio.txt` and
build the Space automatically (3–6 minutes).

### What HF needs to find at the repo root

The repo already contains everything required:

```
gradio_app.py             # Gradio app entry-point
requirements-gradio.txt   # HF Spaces' pip requirements file
src/cdpr/…                # the scientific core (installed by '.' in requirements)
scripts/examples.py       # the 5-example registry
scripts/_csv_io.py        # shared CSV loader
pyproject.toml            # hatchling build descriptor
```

### Verify the deployment

After the Space rebuilds, open its URL. You should see:

* "CDPR Simulator — Gradio frontend" header with the build banner,
* three tabs: **Built-in examples**, **Custom Phase-1 simulation**,
  **Upload CSV / Phase-2**.
* The example dropdown lists the five built-in demos.
* Clicking **Run example** on `circle` produces 14 figures (60-100 s
  on the free CPU Space).

The same UI runs identically against `python gradio_app.py` on a
local box.

## How to run Streamlit locally (the supported Streamlit mode)

```bash
streamlit run streamlit_app.py
# open http://localhost:8501
```

The Streamlit Cloud URL stays live but is now tagged
"experimental" — see `docs/frontend-architecture.md` for why.

## How to drive the FastAPI backend without any GUI

```powershell
# PowerShell
.\scripts\call_render.ps1 -Action simulate -Kind circle
.\scripts\call_render.ps1 -Action plot -PlotKind cable_tensions -OpenPng
```

```bash
# bash: warm the worker, then simulate
curl -s https://cdpr-api.onrender.com/health
curl -s -X POST https://cdpr-api.onrender.com/simulate \
    -H "Content-Type: application/json" \
    -d '{"robot":"ipanema_class","duration":1.5,"dt":2e-3,
         "trajectory":{"kind":"circle","duration":1.5,
                       "params":{"center":[0,0,0.5],"radius":0.2,
                                 "axis":[0,0,1],"angle_span":6.283}}}' \
    | python -m json.tool | head -40
```

## Running multiple frontends concurrently on the same machine

Three separate terminals:

```bash
# Terminal 1 — Streamlit
streamlit run streamlit_app.py            # localhost:8501

# Terminal 2 — Gradio
python gradio_app.py                      # localhost:7860

# Terminal 3 — FastAPI
uvicorn cdpr.interface.api:app --reload --port 8000   # localhost:8000

# Terminal 4 — CLI for everything else
python scripts/run_example.py --list
```

All four interfaces import the same `cdpr` package. There is no
state collision because each writes to a unique
`out/<id>-<timestamp>/` folder.

## Picking the right frontend for the right task

| You want to … | Use … |
|---|---|
| Demo the project to a non-Python audience | Hugging Face Gradio Space |
| Debug your own simulation with full plot interactivity | Streamlit local |
| Sweep 100 simulations programmatically | PowerShell CLI |
| Call from a mobile app or non-Python client | FastAPI on Render |
| Get one quick sim result with bookmarkable settings | Gradio local |
| Compare 5 ML models on uploaded data | Gradio local or CLI |
| Run a long PINN training over lunch | PowerShell CLI |

## Cross-references

* `docs/frontend-architecture.md` — full architectural assessment + comparison matrix
* `docs/terminal-execution.md` — clean-clone-to-running-sim universal guide
* `docs/examples.md` — built-in examples registry
* `docs/csv-schema.md` — accepted CSV layouts + alias table
* `docs/runbook.md` — operational runbook (pre-Gradio)
* `docs/deployment-status.md` — current deployment health snapshot
