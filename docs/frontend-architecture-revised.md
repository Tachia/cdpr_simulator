# Frontend architecture — REVISED assessment

A previous version (`docs/frontend-architecture.md`) recommended
Gradio as the single best alternative to Streamlit. Joseph correctly
pushed back: *"Is Gradio really the best for the heavy data and
simulation interactions we have, or would Dash actually serve this
workload better?"*

This document answers that question on technical merit, using the
specific workload — not a generic feature list.

## The workload, stated honestly

| Property | Value |
|---|---|
| Audience | PhD researchers, dissertation committee, other CDPR groups |
| Simulation duration | 30 s – 5 min per run (12 s sim at dt = 1 ms ≈ 38 s wall) |
| Artifact per run | 14 figures + timeseries.csv (≈ 5 MB) + manifest + feasibility |
| Plot interactivity expected by users | zoom, hover, crosshair, range-slider, png/svg export |
| Phase-2 training | PINN ≈ 30 s, PPO/SAC at 5000 steps ≈ 1–2 min, compare-5 ≈ 4 min |
| File uploads | CSV, typically 1–50 MB |
| Parameter sliders | ~20 controls; users tweak repeatedly |
| Concurrent users | low (research tool, not a SaaS) |
| Public demo? | yes (dissertation committee + community), but secondary |

## Six workload-specific tests, not generic features

### Test 1 — User adjusts the `Kp_pos` slider while viewing the tracking plot

| Stack | Behaviour |
|---|---|
| Streamlit | Re-runs the **entire** script from the top on the slider's `on_change`. Re-imports cdpr (~2 s if not cached). Re-renders every widget. Then re-renders every plot. ⚠️ This is the "slider tick blanks the page" symptom we have been chasing. |
| Gradio | Calls one event handler. Re-renders the gallery widget (full re-render of all images). Better than Streamlit but still wasteful. |
| **Dash** | The callback for the plot has `Input("kp_pos", "value")`. Only that plot's `Output("plot_container", "children")` updates. The other 19 widgets are untouched. **Two orders of magnitude less work** per slider tick. ✅ |

### Test 2 — User wants to zoom into `cable_tensions.png` at t = 5.2 s to read exact tension values

| Stack | Behaviour |
|---|---|
| Streamlit | matplotlib PNG. **No interaction.** Researcher has to re-render with new xlim manually. |
| Gradio | matplotlib PNG. **No interaction.** Same problem. |
| **Dash** | Native Plotly figure. **Zoom + pan + hover + crosshair work for free.** Range-slider at the bottom of the tension plot is two lines of code. PNG / SVG export buttons are built in. This is what researchers actually need. ✅ |

### Test 3 — User runs a 30-second PINN training and wants live loss curves

| Stack | Behaviour |
|---|---|
| Streamlit | `st.empty()` + manual update; a script rerun could wipe the placeholder. |
| Gradio | `gr.Progress` + generator yielding intermediate results. Works but progress only, not a live plot. |
| **Dash** | `dcc.Interval(interval=500)` ticks a callback that reads the training log. The loss curve updates in place every 500 ms while the training subprocess writes. Cleanest pattern by a wide margin. ✅ |

### Test 4 — Side-by-side comparison of two simulation runs

| Stack | Behaviour |
|---|---|
| Streamlit | `st.columns(2)`. Re-renders on every change. Layout is fixed. |
| Gradio | Two galleries side by side. Limited control over layout. |
| **Dash** | `dash-bootstrap-components` 12-column grid, fully responsive, every column independently updatable via separate callbacks. ✅ |

### Test 5 — 30 MB CSV upload from a corporate laptop

| Stack | Behaviour |
|---|---|
| Streamlit | `st.file_uploader` over WebSocket. On corporate networks the socket can drop on a >10 MB upload. ⚠️ |
| Gradio | REST-based upload. Robust. ✅ |
| **Dash** | `dcc.Upload` is **REST-based** too. Same robustness as Gradio. ✅ |

### Test 6 — Long simulation should not block other interactions

| Stack | Behaviour |
|---|---|
| Streamlit | Blocks the script entirely. Other widgets unresponsive until simulate() returns. |
| Gradio | Blocks the queued event. Other users on the same worker wait. |
| **Dash** | **Background callbacks** (Dash 2.5+) run simulate() in a worker pool while the UI stays responsive. ✅ |

## What I underweighted in version 1

Two things specifically:

1. **The "script rerun" model itself.** Streamlit re-runs the page top-to-bottom on every interaction. Gradio runs a single event handler. Dash runs **only the callbacks whose inputs changed**. For 20 parameter widgets and 5 plot tabs, that difference compounds 100× per session. I scored both Gradio and Dash as "5/5 stability" generically — but Dash's model is the only one that's actually right for a parametric scientific UI.

2. **Plotly figures vs matplotlib PNGs for science.** Researchers don't view a tension plot once and move on — they zoom, they hover, they read values off the line, they export an SVG for the dissertation. matplotlib gives a static PNG; Plotly gives all of that for free. Gradio renders matplotlib figures by default (PNG). Dash renders Plotly figures natively. For a *research* dashboard, this is a category difference, not a 4 vs 5 difference.

## What I had right in version 1

* **Streamlit Cloud's 1 GB free worker is structurally wrong** for this workload. That conclusion stands.
* **Hugging Face Spaces' 16 GB free worker is the right host**. That also stands.
* **Multi-frontend architecture is the correct shape.** Each frontend imports the cdpr core; disk is the shared boundary. Still true.

What I missed: **HF Spaces supports a Docker SDK**, which means **Dash can host on the same 16 GB worker** Gradio uses. So Dash gets memory headroom too — there is no deployment-cost case for choosing Gradio over Dash. Both deploy free on HF.

## Revised verdict

For this specific scientific workload:

| Surface | Best stack | Why |
|---|---|---|
| **Primary research dashboard** (where dissertation work happens) | **Dash + Plotly** | Native interactivity, no script reruns, `dcc.Store` client state, background callbacks, mature for science |
| **Public one-click demo** (curious visitors, mobile, "try it now") | Gradio + HF Spaces | Lower-friction for non-technical visitors, file uploads are very polished |
| Headless / batch | PowerShell CLI | unchanged |
| HTTP API | FastAPI on Render | unchanged |
| Local Streamlit | "legacy, kept for backward compat" | demoted further |

Both Dash and Gradio deploy free on Hugging Face Spaces (Docker SDK for Dash, Gradio SDK for Gradio). They are complementary surfaces, not competing.

## Re-scored matrix (workload-specific weights)

The weights now reflect what *this* project actually does. I have
also added a "compounding-cost" axis to capture the rerun-model
penalty Streamlit accumulates over a session.

| Axis (weight) | Streamlit | Gradio | **Dash** |
|---|---:|---:|---:|
| Interactive plot quality (×3) | 1×3 = 3 | 2×3 = 6 | 5×3 = **15** |
| Per-interaction render cost (×3) | 1×3 = 3 | 3×3 = 9 | 5×3 = **15** |
| Long-running task handling (×2) | 2×2 = 4 | 3×2 = 6 | 5×2 = **10** |
| File upload reliability (×2) | 2×2 = 4 | 5×2 = 10 | 5×2 = **10** |
| Free-tier memory headroom (×2) | 1×2 = 2 | 5×2 = 10 | 5×2 = **10** (HF Docker) |
| Session-state robustness (×2) | 2×2 = 4 | 4×2 = 8 | 5×2 = **10** (`dcc.Store`) |
| Deployment ease (×1) | 4 | 5 | 4 |
| Time-to-first-prototype (×1) | 5 | 4 | 3 |
| Layout flexibility (×1) | 3 | 3 | 5 |
| Publication-quality export (×1) | 4 | 4 | 5 |
| **Weighted total** | **36** | **65** | **87** |

(The version-1 matrix was unweighted and listed Gradio at 62 and Streamlit at 36. With the workload-specific weights and the missing "per-interaction render cost" axis added, Dash pulls ahead clearly.)

## What ships in this commit

| File | Purpose |
|---|---|
| `dash_app.py` | Production research dashboard. Five interactive Plotly tabs (Position, Cable tensions, XY trajectory, Tracking error, 3D scene), Phase-1 inline simulation, built-in examples, Phase-2 file upload + training. Tab switching does **not** re-run the simulation. |
| `requirements-dash.txt` | Pinned for HF Spaces Docker SDK / Render. |
| `pyproject.toml` | New `dash` extra: `pip install -e ".[dash]"`. |
| `docs/frontend-architecture-revised.md` (this file) | Honest workload-specific analysis. |

`gradio_app.py` and `streamlit_app.py` stay — they serve different
audiences. CLI, FastAPI, and Supabase wiring are untouched.

## How to run

Local:

```bash
pip install -e ".[viz,data,api,dash]"
python dash_app.py
# open http://127.0.0.1:8050
```

Hugging Face Spaces (Docker SDK):

```bash
# In your Space's repo:
#   Space SDK: Docker  (not Gradio)
#   Dockerfile that runs:  python dash_app.py
git remote add space https://huggingface.co/spaces/<user>/cdpr-dash
git push space main
```

Render / Fly.io: standard Flask deployment (Dash is a Flask app under
the hood).

## What "best alternative" actually means here

Joseph asked for a thorough check, not a list. The honest answer is:

* **For the local research dashboard** — the surface where dissertation work happens — **Dash is the best alternative.** It is structurally a better fit than either Streamlit or Gradio.
* **For the public one-click demo** — the surface where someone on a phone tries the project for two minutes — **Gradio is the best.** It is lower friction.

These are complementary, and both ship in this repository. The user picks the surface that matches their session intent. Streamlit stays for backward compatibility but is no longer the recommended path for either use case.
