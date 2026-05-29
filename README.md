---
title: CDPR Simulator
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.15.2
app_file: app.py
pinned: false
---

# cdpr — Cable-Driven Parallel Robot research framework

A hybrid physics-driven and data-driven computational framework for **Cable-Driven Parallel Robots**, developed as the software contribution of an Innopolis PhD dissertation.

This is a research instrument — not a simulator clone, not a wrapper around an existing engine. The scientific core is self-contained NumPy / SciPy / SymPy; PyTorch and Stable-Baselines3 are optional learning extras; MuJoCo / PyBullet / ROS 2 / Gazebo / Isaac Sim are pluggable adapters. The core runs without any of them.

## What's in the box (Phases 1–7)

| layer | package | content |
| --- | --- | --- |
| Core | `cdpr.core`, `cdpr.geometry`, `cdpr.kinematics`, `cdpr.statics`, `cdpr.cables`, `cdpr.dynamics`, `cdpr.workspace`, `cdpr.trajectory`, `cdpr.robots` | SE(3) primitives, robot configuration, inverse / forward kinematics, structure matrix, tension distribution QP, three per-cable physics models (massless / linear-elastic / Irvine catenary), Newton–Euler + RK4 / semi-implicit integrators, WCW + WFW workspace tests, parametric trajectories, IPAnema-class and CoGiRo-class reference catalogues |
| Control | `cdpr.control` | PD pose regulator, computed-torque feedback linearisation, MPC stub, FF+FB composer |
| Cable constitutive laws (Phase 7) | `cdpr.cables` (extended) | three **mutually exclusive** runtime laws — `KelvinVoigtModel`, `IrvineModel`, `SQCKHybridModel` — selectable by name through `cable_model_by_name()`; integrated into `simulate()`, benchmarks, and reports without disturbing the Phase 1-6 default path |
| Visualisation | `cdpr.viz` | 2D analytics, 3D scene with tension heatmap and singularity indicator, animation (MP4 / GIF), workspace volume rendering, publication-grade Matplotlib style |
| Data ingestion | `cdpr.ingest` | CSV / XLSX / TXT / JSON loaders, MAD outlier detection, SLERP-correct resampling, Butterworth + Savitzky–Golay filters, validation against the dynamic model |
| Reporting | `cdpr.recording`, `cdpr.reports` | recordings with reproducibility manifest, captioned LaTeX figures, booktabs tables, Markdown summaries, full benchmark-bundle generator |
| Learning | `cdpr.learn` | Gymnasium env, SB3 PPO / SAC / TD3 wrappers, supervised inverse-dynamics MLP and Newton–Euler-residual PINN |
| Identification | `cdpr.identification` | bounded nonlinear least-squares calibration of anchor / attachment / cable-length offsets |
| Benchmarks + experiments | `cdpr.benchmarks`, `cdpr.experiments` | reproducible scenario + multi-backend suite, deterministic experiment bundles with manifest |
| External-physics adapters | `cdpr.adapters` | MuJoCo (full), PyBullet (gated on wheel availability), ROS 2 transport (in-memory + rclpy modes), Gazebo / Isaac Sim stubs |
| Interface | `cdpr.interface` | FastAPI HTTP service, Streamlit research console |

**195 pytest tests pass** (52 s on a developer laptop); end-to-end smoke scripts exercise each phase (`examples/smoke_*.py`).

## Install

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[all]"            # everything (torch, sb3, fastapi, streamlit, ...)
```

Optional extras:

| extra | purpose |
| --- | --- |
| `viz` | Matplotlib + Pillow for figures + GIFs |
| `viz-extras` | Plotly + PyVista for richer 3D |
| `learn` | PyTorch + Gymnasium for supervised / PINN / env construction |
| `rl` | adds Stable-Baselines3 for PPO / SAC / TD3 |
| `data` | pandas + openpyxl for experimental log ingestion |
| `api` | FastAPI + Uvicorn + Pydantic for the HTTP service |
| `gui` | Streamlit for the research console |
| `adapters-mujoco` | MuJoCo backend |
| `adapters-pybullet` | PyBullet backend |

## Quick start

```python
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory
from cdpr.control import ComputedTorqueController
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate

robot = ipanema_class()
traj = Trajectory(
    path=CircularPath(center=(0, 0, 0), radius=0.25, axis=(0, 0, 1)),
    scaling=QuinticScaling(duration=1.5),
)
ct = ComputedTorqueController(Kp_pos=900.0, Kd_pos=60.0,
                              Kp_rot=900.0, Kd_rot=60.0)
state0 = PlatformState.at_rest(traj.pose(0.0))
result = simulate(robot=robot, state0=state0, duration=1.5, dt=2e-3,
                  reference=traj, controller=ct)
```

## Running the services

The framework ships with multiple coexisting human-facing surfaces.
Every one of them imports the same `cdpr` scientific core — see
[docs/multi-frontend.md](docs/multi-frontend.md) for the topology.

| Surface | Best for | Command |
|---|---|---|
| PowerShell / bash CLI | Reproducible runs, batch sweeps, headless work | `python scripts/run_example.py --name circle` |
| **FastAPI backend** (`cdpr.interface.api`) | JSON HTTP for any client | `uvicorn cdpr.interface.api:app --port 8000` |
| **Gradio GUI** (`gradio_app.py`) — **recommended hosted demo** | Free 16 GB Hugging Face Space, robust uploads | `python gradio_app.py` (local) |
| **Streamlit console** (`streamlit_app.py`) | Local research UI | `streamlit run streamlit_app.py` |

```bash
# All deps, then start any combination:
pip install -e ".[api,viz,data,gui,gradio,learn,rl]"

uvicorn cdpr.interface.api:app --port 8000      # API @ http://localhost:8000/docs
python gradio_app.py                            # Gradio @ http://localhost:7860
streamlit run streamlit_app.py                  # Streamlit @ http://localhost:8501
python scripts/run_example.py --list            # CLI (5 built-in examples)
```

For the universal terminal walk-through (PowerShell / CMD / Git Bash /
bash / zsh / VSCode / SSH / Colab) see
[docs/terminal-execution.md](docs/terminal-execution.md).

For the architectural rationale on why Gradio is now the recommended
hosted demo see
[docs/frontend-architecture.md](docs/frontend-architecture.md).

## Deployment (Phase 8 wiring)

The repo ships with deployment configuration for:

| service | file | purpose |
| --- | --- | --- |
| Render | `render.yaml` + `Dockerfile` | hosts the FastAPI backend |
| Streamlit Community Cloud | `.streamlit/config.toml` + entry point in `src/cdpr/interface/gui.py` | hosts the research console |
| Cloudflare Pages | `docs/` + `cloudflare-pages.toml` | hosts the static docs site |
| Supabase | `supabase/schema.sql` | persistent storage for experiment metadata and uploaded datasets (optional) |
| GitHub Actions | `.github/workflows/ci.yml` | lint + test on every push |

See [`docs/deployment.md`](docs/deployment.md) for the full deployment guide and the connection map. Required secrets are listed in [`.env.example`](.env.example).

## Layout

```
src/cdpr/
    adapters/        external-physics adapters (mujoco, pybullet, ros2, ...)
    benchmarks/      mode-aware scenario harness across backends
    cables/          per-cable physics + Phase-7 constitutive model classes
    control/         PD, computed-torque, MPC, FF+FB composer
    core/            SE(3) frames, twists, wrenches, numerical helpers
    dynamics/        Newton-Euler + integrators + simulate() driver
    experiments/     reproducible experiment bundles
    geometry/        anchors, platform attachments, Robot configuration
    identification/  bounded LSQ calibration of robot parameters
    ingest/          experimental data loaders + cleaning pipeline
    interface/       FastAPI service + Streamlit console
    kinematics/      IK, FK (LM), structure matrix
    learn/           Gymnasium env, SB3 wrappers, MLP + PINN inverse-dynamics
    recording/       experiment recorder + replay
    reports/         publication-grade figures, tables, Markdown bundles
    robots/          IPAnema-class, CoGiRo-class, point-mass, planar
    statics/         tension distribution QP + feasibility
    trajectory/      paths, scalings, Trajectory composer
    viz/             2D / 3D / animation
    workspace/       WCW, WFW, grid sampler
tests/               pytest test suite (195 tests)
examples/            end-to-end smoke scripts (smoke_phase1..7.py)
docs/                static docs site (Cloudflare Pages target)
supabase/            optional persistent-storage schema
.github/workflows/   CI
```

## Design rules

1. The scientific core depends only on NumPy / SciPy.
2. No external simulator is imported by core modules — adapters are lazy.
3. Three cable constitutive laws are runtime-exclusive (`kelvin_voigt`, `irvine`, `sqck_hybrid`); never blended automatically.
4. Everything is reproducible from a single config dict plus a seed.
5. All numerical routines accept a single pose or a batch.
6. Conventions follow Pott, *Cable-Driven Parallel Robots* (Springer, 2018) unless documented otherwise.

## Citing

If you use this framework in published work, please cite the dissertation (forthcoming) and the relevant module-level references in the source headers.
