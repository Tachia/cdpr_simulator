---
title: CDPR Simulator — Project Memory Extract
author: M. J. Tachia (Joseph Mfeuter), Innopolis University
audience: dissertation co-authors, future developers, future AI sessions
build_at: 0c4152d (GitHub main), a89fdfb (HF Space, pending push)
date: 2026-05-30
---

This document is the permanent knowledge base for the CDPR Simulator
project. It is written so a new developer, a new AI session, or a
dissertation chapter can pick up from it without prior context. Where
something is verified by source code, it is stated plainly. Where
something is planned but not yet implemented, it is marked
*pending* or *planned*. Where something has been observed to fail, the
failure mode is documented before any fix.

The document is laid out so each section is self-contained. You can
read Section 4 (Mathematical Foundations) without Section 11 (User
Interfaces) and vice versa.

---

# 1. Project Overview

## 1.1 Project identity

* **Name** — CDPR Simulator (package import name `cdpr`).
* **Repository** — `https://github.com/Tachia/cdpr_simulator`.
* **Public demo** — `https://huggingface.co/spaces/JoeTach/cdpr-simulator`
  served at `https://joetach-cdpr-simulator.hf.space`.
* **Local development URL** — `http://127.0.0.1:7860` when the Gradio
  frontend is running.
* **License** — see `LICENSE` (MIT in current state, subject to revision
  before journal submission).

## 1.2 Purpose

The project provides a dissertation-grade computational platform for
Cable-Driven Parallel Robots (CDPRs). It is designed to support every
chapter of the author's PhD dissertation at Innopolis University —
from analytical formulations of kinematics and statics through
experimental data validation, sim-to-data comparison, and
controller benchmarking on parametric robots. It also serves as the
publishable software artefact that accompanies the dissertation and
any associated journal or conference contributions.

The simulator is structured around two operational phases:

* **Phase 1 — physics-based simulation.** Rigid-body dynamics, three
  constitutive cable models, feedback control (PD, computed-torque,
  MPC, composed feedforward/feedback), tension distribution under
  cable-bound constraints, and visualisation of pose, tensions,
  workspace, and orientation.
* **Phase 2 — data-driven training and analysis.** Ingestion of
  experimental CSV logs, multi-format support, smart cleaning and
  imputation, supervised inverse-dynamics learning (MLP, PINN),
  reinforcement learning (PPO, SAC via stable-baselines3), and a
  multi-model benchmark harness that ranks replay / MLP / PINN / PPO /
  SAC against the same dataset.

## 1.3 Problem statement

CDPRs occupy a specific niche in parallel mechanism research because
cables can only pull, never push, and because cable elasticity and
sag couple back into the platform's rigid-body dynamics. Three
problems make CDPR research expensive without dedicated software:

1. **Wrench feasibility is non-trivial.** The tension-distribution
   problem is a quadratic programme with positivity and upper-bound
   constraints. Any meaningful research result must respect these
   constraints, not just compute as if every wrench were achievable.
2. **Cable mechanics are mode-dependent.** Massless, linear-elastic,
   Kelvin–Voigt, and Irvine catenary models can produce qualitatively
   different platform trajectories under the same control input.
   Comparing them is part of the dissertation; doing so requires the
   simulator to support all four under one API.
3. **Sim-to-data comparison is rare.** Most CDPR papers either
   simulate or run hardware; few publish a clean head-to-head
   comparison. The Phase-2 ingest layer was specifically built so the
   author's Innopolis test rig CSV exports can be compared directly to
   the analytic simulator.

## 1.4 Intended users

* The dissertation author (primary user).
* The dissertation committee and external examiners (read-only).
* CDPR researchers who reproduce the dissertation experiments.
* Robotics undergraduate / graduate students seeking a teaching
  artefact for parallel manipulators.
* Industrial users in cable robotics (haptics, 3D printing, suspended
  manipulation, large-scale construction CDPRs) who want a tractable
  reference simulator.

## 1.5 Academic motivation

The Innopolis programme requires that a PhD candidate produce a body
of original contributions backed by a reproducible artefact. The
CDPR Simulator is that artefact. It is designed so the experimental
chapters of the dissertation can be regenerated end-to-end from one
command, and so any committee question of the form "what happens if
the cable model is changed?" can be answered live during the defence.

## 1.6 Industrial motivation

A free, open, and fully scriptable CDPR simulator with both physics
and data-driven workflows fills a gap. Existing options either ship
without dynamics (most academic codebases), require commercial
licences (MATLAB-based pipelines), or assume a specific simulator
backend such as MuJoCo or Isaac Sim (which are powerful but heavy and
not easily auditable for cable physics).

## 1.7 Why existing simulators are insufficient

* MuJoCo and PyBullet treat cables as approximations — either as
  prismatic joints with springs or as constraint forces — and neither
  enforces cable-only tension feasibility as a first-class concern.
* ROS-based CDPR stacks (TASKE, CableRobotSim, etc.) target specific
  platforms and tie the user to a particular robot geometry.
* MATLAB-based dissertations are not freely redistributable.
* No public option combines a Python core with adapters for MuJoCo /
  PyBullet / ROS2 / Gazebo / Isaac Sim and a data-driven workflow.

## 1.8 Expected impact

* A reproducible computational backbone for the author's dissertation.
* A published software artefact (planned: software paper in *SoftwareX*
  or *Journal of Open Source Software*).
* A teaching artefact for the Innopolis robotics programme.
* A platform other CDPR researchers can fork for their own studies.

---

# 2. Project History

## 2.1 Original vision

The project started as a directive to build a dissertation-grade
computational platform that satisfied four constraints simultaneously:

1. The scientific core must run with no external simulator (no ROS,
   Gazebo, or MuJoCo required for Phase 1).
2. External simulators are optional adapters, not dependencies.
3. The code must be modular — every package replaceable in isolation.
4. The code must feel handcrafted by a researcher rather than
   autogenerated; readability and audit trails take precedence over
   one-line clever solutions.

## 2.2 Major milestones

| Phase | Period | Outcome |
|---|---|---|
| 1.x | initial 6 weeks | Scientific core: kinematics, statics, dynamics, controllers, workspace, trajectories, reference robots, streaming generator |
| 2.x | weeks 6–10 | Visualisation, logging, replay, reports, FastAPI service, Streamlit console, smoke tests |
| 3.x | weeks 10–14 | Data ingestion: loaders, cleaning, resampling, filtering, units, pipeline, validation, smart loader |
| 4.x | weeks 14–18 | Learning layer: Gym environment, rewards, SB3 wrappers, supervised datasets, MLP, PINN, training loop, benchmark harness |
| 5.x | weeks 18–22 | Adapter layer: MuJoCo, PyBullet, ROS2, Gazebo, Isaac Sim; cross-engine verification |
| 6.x | weeks 22–26 | MPC, composed controller, identification, benchmarks, experiments bundle, dissertation report |
| 7.x | weeks 26–30 | Three constitutive cable models + hybrid; simulator integration with mode-aware dispatch |
| Deploy 1–27 | weeks 30–onwards | Deployment infrastructure: GitHub, Hugging Face Spaces, Render, Cloudflare Pages, Supabase storage, multi-frontend (Streamlit + Dash + Gradio), LLM provider abstraction, conversational simulation builder, accordion UI, per-run audit, fallback chain |
| Deploy 28 | most recent | Chat speed optimisations + ≥10 figures/tables per Phase-2 run (commit `0c4152d`) |

## 2.3 Phases that ran into rework

* **Streamlit memory failures on Streamlit Cloud's 1 GB worker** — led
  to the introduction of the Dash and Gradio frontends, with Gradio
  ultimately becoming primary because Hugging Face Spaces gives 16 GB
  per worker. Streamlit and Dash are retained as secondary interfaces.
* **Hugging Face push authorisation chain.** Read-scope tokens, leaked
  tokens in `.git/config`, oversized files (a 17.5 MB CSV in history),
  and `requirements.txt` resolving the wrong package all blocked the
  first push. A diagnostic script (`scripts/diagnose_hf_push.py`) was
  written to triage the chain in one command. The Space now boots on
  commit `a89fdfb`; `0c4152d` is pending push.
* **Gemini 1.5 retirement.** Google retired the `gemini-1.5-flash` and
  `gemini-1.5-pro` endpoints in mid-2025; the model default was
  bumped to `gemini-2.0-flash` and the API key moved from the URL
  query string to the `x-goog-api-key` header so error messages
  never leak it.

## 2.4 Phases pending

* **Deploy 12 / 13** — Streamlit Cloud end-to-end verification.
  Recommended deletion because Streamlit Cloud has been deprioritised
  in favour of HF Spaces + Gradio. Administrative tidy-up only.

---

# 3. Software Architecture

## 3.1 High-level diagram

```
                ┌─────────────────────────────────────────────┐
                │                User surfaces                │
                │  Gradio (primary)  Streamlit  Dash  FastAPI │
                └──────────────────────┬──────────────────────┘
                                       │
              ┌────────────────────────┼───────────────────────┐
              │                        │                       │
   ┌──────────▼─────────┐  ┌──────────▼──────────┐  ┌────────▼──────────┐
   │  Phase-1 runtime   │  │  Phase-2 runtime    │  │  LLM layer        │
   │  cdpr.dynamics     │  │  cdpr.ingest        │  │  cdpr.llm         │
   │  cdpr.control      │  │  cdpr.learn         │  │  conversational   │
   │  cdpr.statics      │  │  RL / supervised    │  │  simulation       │
   │  cdpr.cables       │  │  benchmark          │  │  builder          │
   │  cdpr.trajectory   │  │  recording          │  │                   │
   └──────────┬─────────┘  └──────────┬──────────┘  └──────────────────┘
              │                       │
              └──────────┬────────────┘
                         │
              ┌──────────▼──────────┐
              │  Scientific core    │
              │  cdpr.core          │
              │  cdpr.geometry      │
              │  cdpr.kinematics    │
              │  cdpr.workspace     │
              └─────────┬───────────┘
                        │
              ┌─────────▼───────────┐    ┌────────────────────────┐
              │  Visualisation +    │    │  Adapter layer         │
              │  reporting          │    │  cdpr.adapters         │
              │  cdpr.viz           │    │  MuJoCo · PyBullet ·   │
              │  cdpr.reports       │    │  ROS2 · Gazebo · Isaac │
              └─────────┬───────────┘    └────────────────────────┘
                        │
              ┌─────────▼───────────┐
              │  Persistence        │
              │  cdpr.recording     │
              │  cdpr.storage       │
              │  (Supabase)         │
              └─────────────────────┘
```

## 3.2 Backend architecture (Python)

The project is a single PyPI-style package (`cdpr`) under `src/`,
organised so dependency flow runs strictly downward in the diagram
above. The scientific core (`cdpr.core`, `cdpr.geometry`,
`cdpr.kinematics`, `cdpr.workspace`) imports nothing from
visualisation, learning, or interface layers. The simulator
(`cdpr.dynamics.simulator`) imports from the core and from
`cdpr.statics`, `cdpr.cables`, `cdpr.control`, and `cdpr.trajectory`,
but never from `cdpr.viz` or `cdpr.learn`.

Dependency rules are enforced by hand-review rather than a linter,
because the directive prioritises readable code over tooling.

## 3.3 Package walk-through

### `cdpr.core`

* **Purpose** — primitive types and numerical helpers.
* **Modules**
  - `frames.py`: `Pose`, `Twist`, `Wrench` data classes; rotation
    composition; homogeneous transforms.
  - `numerics.py`: shared linear-algebra utilities (well-conditioned
    pseudo-inverse, robust least-squares).
  - `exceptions.py`: project-specific exceptions
    (`InfeasibleWrench`, `SingularJacobian`, etc.).
* **Inputs** — None (this is the lowest layer).
* **Outputs** — Pose / Twist / Wrench instances consumed everywhere.
* **Dependencies** — `numpy`, `scipy.spatial.transform.Rotation`.

### `cdpr.geometry`

* **Purpose** — robot configuration data structures.
* **Modules** — `robot.py` defines `Robot`, `Anchor`, `Attachment`,
  `Inertia`, `TensionLimits` and the loaders that construct them from
  YAML / JSON manifests.
* **Dependencies** — `cdpr.core`.

### `cdpr.kinematics`

* **Purpose** — forward and inverse kinematics, Jacobian / structure
  matrix.
* **Modules**
  - `inverse.py`: closed-form cable-length computation given platform
    pose.
  - `forward.py`: iterative forward kinematics from cable lengths.
  - `jacobian.py`: structure matrix \(W(\mathbf{q})\) mapping cable
    tensions to platform wrenches.
* **Dependencies** — `cdpr.core`, `cdpr.geometry`.

### `cdpr.statics`

* **Purpose** — tension-distribution QP solver.
* **Modules** — `tension.py` exposes
  `solve_tension_distribution(W, w_des, t_min, t_max, objective)` and
  the three objective variants `min_norm`, `centered`, `preferred`.
* **Dependencies** — `cdpr.kinematics`, `scipy.optimize`.

### `cdpr.workspace`

* **Purpose** — workspace analysis: wrench-closure (WCW) and
  wrench-feasible (WFW) regions.
* **Modules**
  - `closure.py`: pose-by-pose closure test.
  - `feasible.py`: feasibility test against the platform's required
    wrench set.
  - `grid.py`: voxel grid + ray-shooting algorithms.
* **Dependencies** — `cdpr.statics`, `cdpr.kinematics`.

### `cdpr.cables`

* **Purpose** — constitutive cable models.
* **Modules**
  - `base.py`: `CableModel` abstract base with `tension`,
    `force_vector`, `effective_length`, `stretch`, `diagnostics`.
  - `massless.py`: zero-mass, infinitely stiff baseline.
  - `elastic.py`: linear axial stiffness.
  - `kelvin_voigt.py`: \(T = k\delta L + c\dot{\delta L}\).
  - `irvine.py`: static catenary sag via the Irvine equations.
  - `sqck_hybrid.py`: Irvine static term + Kelvin–Voigt damping term.
  - `sagging.py`: catenary solver shared by `irvine` and
    `sqck_hybrid`.
  - `factory.py`, `diagnostics.py`.
* **Dependencies** — `cdpr.core`, `scipy.optimize`.

### `cdpr.dynamics`

* **Purpose** — rigid-body platform dynamics, integrator wrappers, the
  simulation driver.
* **Modules**
  - `rigid_body.py`: `PlatformState` dataclass; Newton–Euler
    integration helpers.
  - `integrators.py`: RK4 / semi-implicit Euler.
  - `simulator.py`: `simulate(...)` and `iter_simulation(...)`
    (generator interface for streaming).
* **Dependencies** — `cdpr.statics`, `cdpr.cables`, `cdpr.control`,
  `cdpr.trajectory`.

### `cdpr.control`

* **Purpose** — feedback controllers.
* **Modules**
  - `base.py`: `Controller` protocol; `as_gain_matrix`,
    `orientation_error`.
  - `pd.py`: `PDController` with optional gravity compensation /
    external-wrench cancellation.
  - `computed_torque.py`: `ComputedTorqueController` (inverse-dynamics
    feedback linearisation; requires the platform inertia tensor).
  - `mpc.py`: `MPCController` — linear receding-horizon translational
    MPC with a PD orientation loop, solved per axis with
    `scipy.optimize.minimize` (BFGS).
  - `composed.py`: feedforward + feedback composition,
    `InverseDynamicsFeedforward`.
* **Dependencies** — `cdpr.core`, `cdpr.kinematics`, `scipy.optimize`.

### `cdpr.trajectory`

* **Purpose** — parametric paths and jerk-limited time-scaling.
* **Modules**
  - `paths.py`: `LinearPath`, `CircularPath`, `LissajousPath`,
    `HoldPath`.
  - `scaling.py`: `QuinticScaling`, `JerkLimitedScaling`.
  - `trajectory.py`: `Trajectory(path, scaling)` composition object;
    callable `traj(t) -> Pose` and `traj.twist(t)`.
* **Dependencies** — `cdpr.core`.

### `cdpr.robots`

* **Purpose** — reference robot catalogue.
* **Modules**
  - `catalog.py`: `point_mass_3d`, `planar_translational`,
    `ipanema_class`, `cogiro_class` factories.
  - `dissertation.py`: the author's primary `dissertation_8cable`
    robot used as the running example throughout the dissertation.
* **Dependencies** — `cdpr.geometry`, `cdpr.core`.

### `cdpr.viz`

* **Purpose** — plotting and 3D scene rendering.
* **Modules**
  - `style.py`: `apply_paper_style()` configures matplotlib for
    publication-quality output (Times-equivalent fonts, gridlines,
    tight layouts, dpi 160 default).
  - `plots2d.py`: `plot_position`, `plot_velocity`,
    `plot_angular_velocity`, `plot_cable_tensions`,
    `plot_cable_lengths`, `plot_tracking_error`,
    `plot_trajectory_projection`, `plot_condition_number`.
  - `scene.py`: 3D scene with anchors, cables (coloured by tension
    heatmap), platform, optional reference trajectory trail,
    optional singularity indicator.
  - `workspace.py`: voxel scatter / surface mesh rendering.
  - `animation.py`: MP4 / GIF generation; live-streaming bridge from
    `iter_simulation`.
  - `export.py`: LaTeX-figure-ready outputs.
* **Dependencies** — `matplotlib`, `numpy`, `mpl_toolkits.mplot3d`,
  optional `imageio`.

### `cdpr.recording`

* **Purpose** — experiment logging and replay.
* **Modules**
  - `recorder.py`: append-only CSV + JSON schema writer.
  - `compare.py`: pairwise dataset comparison.

### `cdpr.reports`

* **Purpose** — report bundle generation.
* **Modules**
  - `figures.py`, `tables.py`, `summary.py`, `bundle.py`.

### `cdpr.interface`

* **Purpose** — programmatic and HTTP interfaces.
* **Modules**
  - `specs.py`: `SimulationRequest`, `TrajectorySpec`, `build_robot`,
    `build_trajectory` — the schema every frontend serialises to.
  - `api.py`: FastAPI service exposing `/simulate`, `/visualise`,
    `/healthz`.
  - `gui.py`: Streamlit research console.

### `cdpr.ingest`

* **Purpose** — Phase-2 data ingestion.
* **Modules**
  - `loaders.py`: CSV / XLSX / TXT / JSON / Parquet / Feather.
  - `cleaning.py`: NaN handling, deduplication, MAD outlier flag.
  - `resample.py`: timestamp alignment + uniform-grid resampling.
  - `filtering.py`: Butterworth low-pass + Savitzky–Golay smoothing.
  - `units.py`: unit conversion and coordinate-frame normalisation.
  - `validate.py`: post-clean validation against the dynamic model.
  - `pipeline.py`: chainable step orchestrator with deterministic
    outputs.
  - `report.py`: Markdown + JSON preprocessing manifest.
  - `smart_loader.py`: multi-format gateway with 5-level missing-value
    imputation hierarchy, ISO timestamp inference, synthetic manifest
    generation when no manifest is present.
  - `containers.py`: `RawDataset`, `ColumnMap`, `IngestedExperiment`.

### `cdpr.learn`

* **Purpose** — supervised + reinforcement learning.
* **Modules**
  - `env.py`: `CDPREnv` — Gymnasium environment wrapping
    `cdpr.dynamics.simulator`.
  - `rewards.py`: composable reward components.
  - `train.py`: training loop with metric logging + checkpointing.
  - `datasets.py`: supervised dataset builders from `SimulationResult`
    and `IngestedExperiment`.
  - `benchmark.py`: sim-to-data benchmark harness across models.
  - `_lazy.py`: lazy import helpers (torch / sb3 / gym are optional).

### `cdpr.identification`

* **Purpose** — parameter identification from experiment logs.
* **Modules** — `parameters.py`, `problem.py`.

### `cdpr.benchmarks`

* **Purpose** — closed-loop benchmarking across controllers and
  backends.
* **Modules** — `metrics.py`, `scenario.py`, `suite.py`.

### `cdpr.experiments`

* **Purpose** — reproducible experiment bundles.
* **Modules** — `bundle.py`, `config.py`, `runner.py`. The
  `dissertation_full` bundle is the canonical end-to-end run that
  backs the experimental chapters.

### `cdpr.adapters`

* **Purpose** — external simulator and middleware adapters.
* **Modules**
  - `base.py`: abstract adapter base + factory + lifecycle context
    manager.
  - `mujoco.py`: builds `MjModel` from `Robot`; exposes `step`,
    `set_pose`, `read_state`, `tensions`.
  - `pybullet.py`: lazy-loaded; reports gracefully on missing SDK.
  - `ros2.py`: minimal publisher / subscriber design for pose,
    lengths, tensions.
  - `gazebo.py`: Harmonic-version stub matching the refactored base.
  - `isaac_sim.py`: stub matching the same contract.
  - `verify.py`: cross-engine verification helpers.

### `cdpr.llm`

* **Purpose** — provider-agnostic LLM access.
* **Modules**
  - `base.py`: `LLMProvider` Protocol; `LLMMessage`, `LLMResponse`,
    `LLMUnavailableError`.
  - `config.py`: `LLMConfig.from_env()`, `available_providers`,
    `default_provider_name`, `resolve_fallback_chain`.
  - `factory.py`: `build_provider(name)`.
  - `providers/`: `gemini.py`, `openrouter.py`, `ollama.py`,
    `lmstudio.py`, `echo.py`, `_http.py`.
  - `simulation_builder.py`: `describe_to_request(text, ...) ->
    BuilderResult` with provider chain, response cache, and secret
    redaction.

### `cdpr.storage`

* **Purpose** — optional Supabase persistence.
* **Modules** — `supabase.py`: dual-write adapter writing simulation
  manifests + artefact paths to the configured Supabase project.

## 3.4 Frontend code (top-level repo)

* `gradio_app.py` — primary frontend (Custom Phase-1, conversational
  builder, Phase-2 upload, accordion UI, Reset button, per-run audit).
* `app.py` — Hugging Face Space entrypoint; imports `demo` and `_CSS`
  from `gradio_app.py` and launches with `theme=gr.themes.Soft()`.
* `dash_app.py` — secondary Dash interface.
* `streamlit_app.py` — secondary Streamlit interface.
* `scripts/` — CLI entry points: `run_simulation.py`,
  `run_example.py`, `train_from_csv.py`, `compare_models.py`,
  `test_llm.py`, `diagnose_hf_push.py`, `examples.py` (registry),
  `_csv_io.py` (shared CSV helpers).

---

# 4. Mathematical Foundations

The simulator is built on standard rigid-body mechanics with
parallel-mechanism constraints. The notation used throughout this
section and in the source code is collected here.

## 4.1 Notation

* \(\mathbf{p} \in \mathbb{R}^3\) — platform centre-of-mass position
  in the world frame.
* \(\mathbf{R} \in SO(3)\) — platform orientation as a rotation
  matrix.
* \(\mathbf{q} = (\mathbf{p}, \mathbf{R})\) — full pose in SE(3).
* \(\mathbf{v} = (\dot{\mathbf{p}}, \boldsymbol\omega)\) — twist (linear
  + angular velocity).
* \(\mathbf{w} = (\mathbf{F}, \boldsymbol\tau)\) — wrench (force +
  torque) acting on the platform.
* \(\mathbf{l} \in \mathbb{R}_+^m\) — cable lengths, \(m\) cables.
* \(\mathbf{t} \in \mathbb{R}_+^m\) — cable tensions.
* \(W(\mathbf{q}) \in \mathbb{R}^{6 \times m}\) — structure matrix.
* \(\mathbf{a}_i \in \mathbb{R}^3\), \(\mathbf{b}_i \in \mathbb{R}^3\)
  — base anchor and platform attachment positions for cable \(i\).
* \(t_{\min,i}, t_{\max,i}\) — tension bounds for cable \(i\).

## 4.2 Kinematics

### 4.2.1 Inverse kinematics

For cable \(i\), the world-frame attachment point is
\(\mathbf{p} + \mathbf{R}\mathbf{b}_i\), and the cable vector is

\[
\mathbf{l}_i(\mathbf{q}) = \mathbf{a}_i - \mathbf{p} - \mathbf{R}\mathbf{b}_i,
\qquad
l_i = \|\mathbf{l}_i\|.
\]

Cable lengths are obtained in closed form from the platform pose;
implementation lives in `cdpr.kinematics.inverse`.

### 4.2.2 Structure matrix and forward kinematics

The unit vector along cable \(i\) is \(\mathbf{u}_i = \mathbf{l}_i / l_i\)
(pointing from platform attachment toward the base anchor). The
structure matrix is

\[
W(\mathbf{q}) = \begin{bmatrix}
\mathbf{u}_1 & \cdots & \mathbf{u}_m \\
(\mathbf{R}\mathbf{b}_1) \times \mathbf{u}_1 & \cdots &
(\mathbf{R}\mathbf{b}_m) \times \mathbf{u}_m
\end{bmatrix}.
\]

The wrench applied to the platform by cable tensions \(\mathbf{t}\) is
\(\mathbf{w} = W(\mathbf{q})\,\mathbf{t}\). Forward kinematics solves
\(\mathbf{l}(\mathbf{q}) = \mathbf{l}_{\text{measured}}\) iteratively;
implementation in `cdpr.kinematics.forward`.

### 4.2.3 Jacobian

The platform-velocity-to-cable-velocity Jacobian satisfies
\(\dot{\mathbf{l}} = J(\mathbf{q})\,\mathbf{v}\), and
\(J = -W^\top\) under the convention above. This duality is exploited
in the computed-torque controller.

## 4.3 Statics: tension distribution

Given a desired platform wrench \(\mathbf{w}_d\) and the structure
matrix \(W\), the tensions \(\mathbf{t}\) that realise it lie in the
affine set \(\{ \mathbf{t} : W\mathbf{t} = \mathbf{w}_d \}\) intersected
with the box \(t_{\min,i} \le t_i \le t_{\max,i}\). Because the
system is redundantly actuated (\(m > 6\) for spatial CDPRs), there
is in general a non-trivial set of feasible \(\mathbf{t}\) values.
The simulator solves the QP

\[
\min_{\mathbf{t}} \tfrac{1}{2}\| \mathbf{t} - \mathbf{t}_{\text{pref}} \|_2^2
\quad \text{s.t.} \quad
W\mathbf{t} = \mathbf{w}_d, \quad
\mathbf{t}_{\min} \le \mathbf{t} \le \mathbf{t}_{\max},
\]

with three objective variants:

* `min_norm`: \(\mathbf{t}_{\text{pref}} = \mathbf{0}\).
* `centered`: \(\mathbf{t}_{\text{pref}} = (\mathbf{t}_{\min} + \mathbf{t}_{\max}) / 2\).
* `preferred`: user-supplied \(\mathbf{t}_{\text{pref}}\).

Implementation: `cdpr.statics.tension.solve_tension_distribution`.
When the QP is infeasible the solver flags the step in
`SimulationResult.infeasible_steps`; the simulator continues so the
user can see *where* infeasibility appears rather than crashing.

## 4.4 Dynamics

The platform obeys Newton–Euler rigid-body equations:

\[
M_p \ddot{\mathbf{p}} = \mathbf{F}_{\text{cables}} + \mathbf{F}_{\text{ext}} + M_p\,\mathbf{g},
\qquad
I_p \dot{\boldsymbol\omega} + \boldsymbol\omega \times (I_p \boldsymbol\omega)
= \boldsymbol\tau_{\text{cables}} + \boldsymbol\tau_{\text{ext}},
\]

where \(M_p\) is the platform mass plus payload, \(I_p\) is the
inertia tensor in the body frame, and the cable wrench is
\((\mathbf{F}_{\text{cables}}, \boldsymbol\tau_{\text{cables}}) = W(\mathbf{q})\,\mathbf{t}\).

The integrator is a fourth-order Runge–Kutta scheme on the position
and a semi-implicit update on orientation (rotation matrix
re-orthogonalisation each step). Implementation:
`cdpr.dynamics.integrators` and `cdpr.dynamics.simulator`.

## 4.5 Workspace analysis

* **Wrench-closure workspace (WCW).** \(\mathbf{q}\) belongs to the
  WCW iff for every direction \(\mathbf{w} \in \mathbb{R}^6\) there
  exist non-negative tensions \(\mathbf{t}\ge 0\) such that
  \(W(\mathbf{q})\mathbf{t} = \mathbf{w}\). Closure is tested via the
  null-space sign condition implemented in
  `cdpr.workspace.closure`.
* **Wrench-feasible workspace (WFW).** Restricts the test set to a
  required wrench set \(\mathcal{W}_{\text{req}}\) (e.g. gravity plus a
  payload range). Implementation in `cdpr.workspace.feasible`.
* **Grid + ray-shooting.** `cdpr.workspace.grid` discretises a region
  and tests pose-by-pose; ray-shooting accelerates boundary
  identification by exploiting the convexity of the WCW.

## 4.6 Cable models

### 4.6.1 Common interface

All cable models implement the `CableModel` interface from
`cdpr.cables.base`:

```python
class CableModel(ABC):
    def tension(self, pose, dl_dt, **state) -> ndarray: ...
    def force_vector(self, pose, dl_dt, **state) -> ndarray: ...
    def effective_length(self, pose, **state) -> ndarray: ...
    def stretch(self, pose, **state) -> ndarray: ...
    def diagnostics(self) -> dict: ...
```

The simulator dispatches to a cable model via the
`simulate(..., cable_model=...)` argument; the default is the
massless baseline.

### 4.6.2 Massless cable

Zero mass, infinitely stiff. Tensions come directly from the
tension-distribution QP. Implementation: `cdpr.cables.massless`.

### 4.6.3 Linear elastic

\(T_i = k_i\,\delta L_i\) where \(\delta L_i = l_i - l_{0,i}\) is the
strain length and \(l_{0,i}\) is the spool-side unstretched length.
Implementation: `cdpr.cables.elastic`.

### 4.6.4 Kelvin–Voigt

\[
T_i = k_i\,\delta L_i + c_i\,\dot{\delta L_i},
\]

with slack handling: if \(T_i < 0\), the cable is slack and the
applied tension is clipped to zero. Implementation:
`cdpr.cables.kelvin_voigt`.

### 4.6.5 Irvine catenary

Static catenary equations from Irvine (1981) give a self-consistent
relationship between tension at the anchor end \(H\), tension at the
attachment end \(T_a\), span chord \(\mathbf{c}_i\), and cable
weight per unit length \(\mu g\):

\[
\frac{H}{\mu g}\left[\sinh^{-1}\!\left(\frac{V}{H}\right) -
\sinh^{-1}\!\left(\frac{V - \mu g L_0}{H}\right)\right] = c_x,
\]
\[
\frac{H}{\mu g}\left[\sqrt{1 + (V/H)^2} -
\sqrt{1 + ((V-\mu g L_0)/H)^2}\right] +
\frac{(\mu g L_0)^2 - 2 V \mu g L_0}{2 E A} = c_z,
\]

where \(V\) is the vertical tension component and \(E A\) is the
axial stiffness. Solved by `cdpr.cables.sagging` using
`scipy.optimize.fsolve` on \((H, V)\). The Irvine model
(`cdpr.cables.irvine`) wraps this as the static tension contribution.

### 4.6.6 SQCK hybrid

Combines the Irvine static tension with a Kelvin–Voigt damping
term so the dynamic response is captured while sag is retained:

\[
T_i = T_{\text{Irvine}}(\delta L_i; \text{geometry}) +
\frac{\eta A}{L_{0,i}}\,\dot{\delta L_i}.
\]

Implementation: `cdpr.cables.sqck_hybrid`. This is the author's
contribution to constitutive cable modelling and a candidate
publishable chapter.

## 4.7 Controllers

### 4.7.1 PD pose regulator

\[
\mathbf{w}_d = K_p^{\text{pos}}(\mathbf{p}_r - \mathbf{p}) +
K_d^{\text{pos}}(\dot{\mathbf{p}}_r - \dot{\mathbf{p}}) +
\text{gravity compensation} +
\text{external cancellation}
\]

plus an orientation component using the rotation-vector error.
Default gains: \(K_p^{\text{pos}} = 400\), \(K_d^{\text{pos}} = 2\sqrt{K_p^{\text{pos}}}\),
\(K_p^{\text{rot}} = 100\). Implementation: `cdpr.control.pd`.

### 4.7.2 Computed-torque

Feedback linearisation on the rigid-body equations:

\[
\mathbf{w}_d = M(\mathbf{q})\,\mathbf{a}_r + \mathbf{n}(\mathbf{q}, \mathbf{v})
- K_p \mathbf{e} - K_d \dot{\mathbf{e}},
\]

requires the platform inertia tensor and benefits from a
reference-acceleration feed. Implementation:
`cdpr.control.computed_torque`.

### 4.7.3 MPC

Linear receding-horizon MPC over translational dynamics, solved per
axis. State \(\mathbf{x} = [\mathbf{p}; \mathbf{v}] \in \mathbb{R}^6\),
control \(\mathbf{u} \in \mathbb{R}^3\) (force on the platform), with
the discrete-time double-integrator dynamics

\[
\mathbf{x}_{k+1} = A \mathbf{x}_k + B \mathbf{u}_k, \quad
A = \begin{bmatrix} I & \Delta t\,I \\ 0 & I \end{bmatrix}, \quad
B = \tfrac{1}{m}\begin{bmatrix} \tfrac{\Delta t^2}{2}\,I \\ \Delta t\,I \end{bmatrix}.
\]

The objective over horizon \(N\) is

\[
J = \sum_{k=0}^{N-1}\bigl[Q_{\text{pos}}\|\mathbf{e}_k\|^2 +
Q_{\text{vel}}\|\dot{\mathbf{e}}_k\|^2 + R\|\mathbf{u}_k\|^2\bigr] +
P\|\mathbf{e}_N\|^2.
\]

Solver: `scipy.optimize.minimize` (BFGS, unconstrained — cable bounds
are still enforced downstream by the tension-distribution QP).
Implementation: `cdpr.control.mpc`.

Default tuning shipped with the form: \(N = 8\), \(Q_{\text{pos}} = 2 \times 10^3\),
\(Q_{\text{vel}} = 20\), \(R = 10^{-3}\), \(P = 10^4\). For the
8-cable dissertation robot these defaults track small motions
(\(\le 3\) cm amplitude) within 5 mm RMS but saturate cables on
8 cm amplitudes — a real cable-feasibility constraint, not a bug.

### 4.7.4 Composed feedforward + feedback

`cdpr.control.composed.FeedforwardPlusFeedback(ff, fb)` returns the
sum of an inverse-dynamics feedforward and any feedback controller,
which is the recommended controller for high-bandwidth tracking on a
known model.

## 4.8 Trajectory parametrisation

* **Paths**: linear, circular, lissajous, hold. The lissajous path is
  the natural figure-eight when frequencies are \([1, 2, 0]\) — this
  is the trajectory exercised in the dissertation's tracking chapter.
* **Time-scaling**: quintic smoothing for the analytic examples,
  jerk-limited profiles for the bench scenarios.
* **Reference twist**: the `Trajectory` object exposes `traj.twist(t)`
  which provides the feedforward velocity used by the MPC's reference
  prediction.

---

# 5. Robot Models

The catalogue ships five robots. Geometry comes from
`cdpr.robots.catalog` and `cdpr.robots.dissertation`.

## 5.1 `point_mass_3d`

* **Configuration**: 4 cables attached to a point mass.
* **DoF**: 3 translational, no orientation.
* **Workspace**: cubic box ≈ \(0.6 \times 0.6 \times 0.4\) m.
* **Use**: smallest test rig for kinematics tests.

## 5.2 `planar_translational`

* **Configuration**: 4 cables, planar (XY).
* **DoF**: 2 translational.
* **Use**: 2-D demonstrations and analytical workspace plots.

## 5.3 `ipanema_class`

* **Configuration**: 8 cables, parallelepiped frame ≈ \(2 \times 2 \times 1\) m,
  geometric layout inspired by the Fraunhofer IPAnema 3 prototype.
* **DoF**: 6.
* **Use**: industrial-scale benchmark.

## 5.4 `cogiro_class`

* **Configuration**: 8 cables, large-scale frame ≈ \(15 \times 11 \times 5\) m
  inspired by the CoGiRo construction CDPR.
* **DoF**: 6.
* **Use**: large-workspace stress test.

## 5.5 `dissertation_8cable`

* **Configuration**: the author's primary 8-cable spatial CDPR, frame
  \(0.9 \times 0.9 \times 1.0\) m, cable bounds \([5, 500]\) N,
  platform mass 2 kg + variable payload.
* **DoF**: 6.
* **Inertia**: computed analytically from a uniform cuboid platform
  approximation; exact values in `cdpr.robots.dissertation`.
* **Use**: every chapter of the dissertation defaults to this robot
  because its workspace is large enough for non-trivial trajectories
  and small enough to fit on the Innopolis test rig.

## 5.6 Selection rationale

* **`point_mass_3d`** — smallest possible test rig; used to verify
  forward kinematics convergence in isolation.
* **`planar_translational`** — 2-D workspace plots reproduce more
  cleanly in print.
* **`ipanema_class` and `cogiro_class`** — provide community
  comparability; both robot classes appear extensively in the CDPR
  literature.
* **`dissertation_8cable`** — sized to match the physical rig built at
  Innopolis, so simulator results map onto recorded CSV files without
  requiring rescaling.

---

# 6. Simulation System

## 6.1 Workflow

```
                            user
                             │
                             ▼
            SimulationRequest (interface/specs.py)
                             │
              build_robot ───┼─── build_trajectory
                             ▼
              Robot          PlatformState        reference(t)
                             │                      │
                             └─────► simulate ◄─────┘
                                       │
                              SimulationResult
                                       │
                             ┌─────────┴─────────┐
                             ▼                   ▼
                     csv recorder         viz plot bundle
                             │                   │
                             ▼                   ▼
                       timeseries.csv     14–16 PNG figures
                             │
                             ▼
                       run_manifest.json
```

1. The user (CLI, Gradio, Streamlit, Dash, or FastAPI) constructs a
   `SimulationRequest` from a serialisable form.
2. `build_robot(name, payload_mass, t_min, t_max)` and
   `build_trajectory(spec)` materialise the runtime objects.
3. `simulate(robot, state0, duration, dt, reference, controller,
   tension_objective, gravity, cable_model)` runs the closed-loop
   integration and returns a `SimulationResult` holding `time`,
   `positions`, `quaternions_xyzw`, `linear_velocities`,
   `angular_velocities`, `cable_lengths`, `cable_tensions`, and
   `infeasible_steps`.
4. The Phase-1 plot bundle (`scripts/examples._render_phase1_plots`)
   renders the 14–16 figure set into the output directory.
5. The CSV writer (`scripts/examples._save_csv`) writes
   `timeseries.csv` with one row per sample.
6. The Gradio per-run audit also writes `run_manifest.json` and
   `reference_vs_actual.png` (commit `a89fdfb`).

## 6.2 `scripts/run_simulation.py`

Single-shot CLI that wraps the same `simulate()` call used by the
GUIs. The CLI is the long-form of the API: it accepts every parameter
the form exposes and writes the same artefacts to the same
directory layout. This is the entry point used by the Phase-6
benchmark harness and the dissertation experiment bundles.

## 6.3 Manifest generation

Two manifest files coexist:

* **`manifest.json`** — written by `scripts/run_simulation.py`. Holds
  the full `SimulationRequest`, the robot's serialised geometry, the
  git hash, the python version, the platform string, and a metrics
  block.
* **`run_manifest.json`** — written by the Gradio per-run audit
  (`gradio_app._write_run_manifest`). Adds SHA1 fingerprints of the
  realised position and tension trajectories. Two runs sharing a
  fingerprint are bit-for-bit identical; two runs that differ
  anywhere produce different fingerprints, which is the cryptographic
  proof that figures cannot be cached.

## 6.4 Report generation

`cdpr.reports.bundle.generate_report(out_dir, results, robot,
reference)` builds a LaTeX-figure-friendly bundle plus a Markdown
summary suitable for inclusion in the dissertation appendix. The
Phase-6 experiment bundles use this path.

---

# 7. Machine Learning Modules

## 7.1 Position in the architecture

`cdpr.learn` is gated behind `pip install 'cdpr[learn]'` because
`torch`, `gymnasium`, and `stable-baselines3` are heavy dependencies.
The scientific core never imports anything from `cdpr.learn`.

## 7.2 MLP inverse-dynamics

* **Inputs**: pose, twist, optional reference acceleration (19-D
  vector for the 8-cable robot).
* **Outputs**: cable tensions (8-D for the dissertation robot).
* **Architecture**: 4-layer MLP, hidden widths configurable via
  `--hidden`; ReLU activations, MSE loss, Adam optimiser.
* **Training**: `scripts/train_from_csv.py --model mlp` reads a
  `timeseries.csv`, splits tail-15% as validation, and writes the
  Phase-2 bundle (commit `0c4152d`) into the output directory.
* **Purpose**: serves as the supervised baseline against which the
  PINN's physics-informed loss is judged.

## 7.3 PINN inverse-dynamics

* **Architecture**: same MLP backbone as above.
* **Loss**:

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} +
\lambda_{\text{phys}}\,\mathcal{L}_{\text{phys}},
\]

where

\[
\mathcal{L}_{\text{data}} = \tfrac{1}{n}\sum_k \|\hat{\mathbf{t}}_k - \mathbf{t}_k\|^2
\]

and

\[
\mathcal{L}_{\text{phys}} = \tfrac{1}{n}\sum_k \bigl\|
W(\mathbf{q}_k)\,\hat{\mathbf{t}}_k - \mathbf{w}_{\text{NE}}(\mathbf{q}_k,
\mathbf{v}_k, \mathbf{a}_k)\bigr\|^2,
\]

the Newton–Euler residual evaluated at each sample. \(\lambda_{\text{phys}}\)
defaults to 0.1.

* **Status**: implementation is complete; tested under
  `tests/test_learn_supervised.py`. Convergence is sensitive to
  feature scaling, which the smart loader handles via per-column
  standardisation.

## 7.4 Status, limitations, future plans

* **What works** — both MLP and PINN train end-to-end on the
  dissertation circle CSV. With reasonable hyperparameters (5–10
  layers, 128–256 hidden width, 30–50 epochs) RMSE drops to
  approximately 5–10 N on a 5–500 N tension range.
* **Limitation 1** — the supervised path requires uniform-grid
  resampling. The smart loader does this automatically but the user
  pays a one-time cost on long CSVs.
* **Limitation 2** — the Newton–Euler residual currently assumes the
  massless cable model. Extending to Irvine residuals is a planned
  publication.
* **Future** — joint training of an inverse-dynamics PINN and a
  forward-dynamics surrogate, enabling fast model-based RL with a
  learned simulator.

---

# 8. Reinforcement Learning Modules

## 8.1 Environment

`cdpr.learn.env.CDPREnv` is a Gymnasium environment that wraps
`cdpr.dynamics.simulator.iter_simulation`.

* **Observation space**: pose (7), twist (6), reference pose (7),
  reference twist (6) = 26-D box in \([-\infty, \infty]\).
* **Action space**: platform wrench (6-D box). The tension-distribution
  QP downstream enforces the cable bounds, so an "infeasible" action
  collapses onto the QP's nearest feasible wrench, scaled by an
  optional penalty term.
* **Reward**: composed from `cdpr.learn.rewards`:

\[
r_k = -w_{\text{pos}}\|\mathbf{e}_k^{\text{pos}}\|^2
- w_{\text{ori}}\|\mathbf{e}_k^{\text{ori}}\|^2
- w_{\text{eff}}\|\mathbf{u}_k\|^2
- w_{\text{infeas}}\,\mathbb{1}[\text{infeasible}_k],
\]

with default weights documented in `cdpr.learn.rewards`.

## 8.2 PPO

* **Wrapper**: `stable_baselines3.PPO` with MLP policy.
* **Train**: `scripts/train_from_csv.py --model ppo --rl-steps N`
  rebuilds the env from a CSV's sibling `manifest.json` and trains a
  policy for `N` timesteps.
* **Eval**: `--eval-episodes K` runs K deterministic episodes and
  records returns.

## 8.3 SAC

* Same interface as PPO; `cls = SAC` when `--model sac`.
* SAC's entropy-regularised objective tends to find smoother control
  policies on the CDPR env; PPO converges faster but is jumpier.

## 8.4 Status

* **What works** — env, factories, training loops, evaluation, and
  the Phase-2 RL bundle (8 figures + 2 tables per run after commit
  `0c4152d`).
* **Limitation** — at the current PPO/SAC step budgets (a few
  thousand timesteps in the Gradio UI's defaults) policies do not
  reach state-of-the-art. The defaults are tuned for *demonstrability*
  in the public Space, not for publication-grade results.
* **Planned** — a long-run training script for the dissertation that
  budgets \(10^6\) timesteps and emits the same Phase-2 bundle.

---

# 9. Data Ingestion System

## 9.1 Phase-2 pipeline (`cdpr.ingest`)

The Phase-2 pipeline takes any tabular experiment log and produces an
`IngestedExperiment` ready to feed the learning layer:

```
file -> loader -> RawDataset
        -> cleaning (NaN, dedup, outlier)
        -> resample (uniform grid)
        -> filtering (Butterworth / Savitzky–Golay)
        -> units / frame normalisation
        -> validation against the dynamic model
        -> IngestedExperiment
```

Every step records its parameters into a `StepRecord` so the full
pipeline can be replayed deterministically.

## 9.2 Loaders

`cdpr.ingest.loaders` supports CSV, TSV, TXT, XLSX, XLS, ODS,
Parquet, Feather. Compression is auto-detected.

## 9.3 Cleaning

* **NaN handling** — column-wise dropna with thresholds.
* **Deduplication** — exact-row dedup keyed on time.
* **Outliers** — Median Absolute Deviation flagging with a default
  factor of 5; flagged rows are kept but tagged.

## 9.4 Timestamp alignment

* ISO-8601 strings parsed via `pandas.to_datetime`.
* Numerical timestamps inferred as seconds-since-epoch when the
  magnitude matches.
* Uniform-grid resampling via linear or pchip interpolation; rate
  chosen as the median sampling interval.

## 9.5 Smart loader (`cdpr.ingest.smart_loader`)

Added in Deploy 22 to handle real-world experimental CSVs whose schema
does not match `scripts/run_simulation.py`'s exactly.

* **Multi-format gateway** — single `load_dataset(path) -> (df, report)`
  function dispatches on extension.
* **Missing-value imputation hierarchy** —
  1. forward-fill within run,
  2. group-wise median per cable / per axis,
  3. global median,
  4. zero with flag column,
  5. drop row.
* **Synthetic manifest generation** — if no sibling
  `manifest.json` exists, infers the robot from the cable count
  (8 → `dissertation_8cable`, 4 → `point_mass_3d`, etc.) and writes
  a plausible manifest so the replay and RL paths still work.
* **Bridge to numpy blocks** — `scripts/_csv_io.split_canonical_blocks`
  consumes the smart loader's `df` and produces the canonical
  `(time, positions, quaternions, ..., cable_lengths, cable_tensions)`
  blocks the rest of the codebase expects.

## 9.6 Documented historical issues

The following failure modes were observed during the development of
the smart loader and are documented here so future debuggers can
reproduce them:

* **Timestamp column missing**. A real experiment CSV exported with
  the columns `idx, px, py, pz, ...` and no `t` column. The smart
  loader now infers a synthetic `t` from the row index at the
  declared sampling rate (the rate either appears in the manifest
  or defaults to 1 kHz).
* **Mixed `none` strings in numeric columns**. One row containing
  `none` in `cable_1` polluted dtype inference and the entire column
  stayed as `object`. Fixed by an explicit `pd.to_numeric(errors='coerce')`
  pass after `cleaning`.
* **Object dtype on bridge**. `pd.to_numeric` left timestamp columns
  as strings, which crashed the bridge to numpy. Fixed by
  `dropna(axis='columns', how='all')` on the coerced frame.
* **17.5 MB CSV in git history**. Blocked HF push because of the
  10 MB-per-file limit. Resolved with `git filter-branch` and
  garbage collection; repository now 521 KB packed.
* **CSV under URL input**. The compare path failed when the input
  path was an HTTP(S) URL because there was no sibling directory.
  Fixed by writing artefacts to `out/<model>-<stamp>/` at the repo
  root for URL inputs (Deploy 17).

## 9.7 Planned fixes

* **`reference_trajectory` reconstruction from a CSV.** Currently
  `replay` and `RL` can rebuild the analytic reference only when the
  manifest's `trajectory.kind` is one of the catalogued kinds. A
  `custom_callable` (used by the spiral and M-shape examples) cannot
  be recovered. Planned: serialise the spiral and M-shape callables
  via their parameter dict so any catalogue-aware trajectory is
  replayable.

---

# 10. Visualisation System

## 10.1 Phase-1 plot bundle (per run)

Implemented in `scripts/examples._render_phase1_plots`. The bundle
consumes only the `SimulationResult`, `Robot`, and `reference(t)` for
that run — no shared cache between runs — and emits the following
figures:

| # | File | Purpose | Interpretation | Research relevance |
|---|---|---|---|---|
| 1 | `position.png` | (px, py, pz) over time | Tracking lag visible at trajectory transitions | Chapter 6 — tracking performance |
| 2 | `velocity.png` | (vx, vy, vz) over time | Reveals jerk; saturation at QP edges | Chapter 6 |
| 3 | `angular_velocity.png` | (wx, wy, wz) over time | Confirms orientation channel is tracked | Chapter 6 |
| 4 | `acceleration.png` | numerical derivative of `vx, vy, vz` | Confirms controller stability | Appendix |
| 5 | `cable_tensions.png` | per-cable tensions vs time | Saturation against `[t_min, t_max]` band | Chapter 5 — feasibility |
| 6 | `cable_lengths.png` | per-cable lengths vs time | Validates inverse kinematics | Chapter 3 |
| 7 | `cable_stretch.png` | per-cable elongation vs time | Reveals elastic / sag effects | Chapter 7 — cable mechanics |
| 8 | `tracking_error.png` | \(\\|\mathbf{p}-\mathbf{p}_r\\|\) vs time | Headline error metric | Chapter 6 |
| 9 | `rms_error_evolution.png` | cumulative RMS vs time | Long-run convergence | Chapter 6 |
| 10 | `condition_number.png` | \(\kappa(W(\mathbf{q}))\) vs time | Singularity proximity | Chapter 5 |
| 11 | `trajectory_xy.png` | projection on XY plane | Reference-vs-actual shape | Chapter 6 |
| 12 | `trajectory_xz.png` | projection on XZ plane | Vertical motion | Chapter 6 |
| 13 | `trajectory_yz.png` | projection on YZ plane | Lateral motion | Chapter 6 |
| 14 | `orientation.png` | roll / pitch / yaw vs time | Orientation channel | Chapter 4 |
| 15 | `scene_3d.png` | full 3D scene at terminal pose | Static figure for thesis cover | Frontispiece |
| 16 | `reference_vs_actual.png` | dense reference overlay (3D + xy) | Trajectory-shape verification | Audit |

All 16 figures use `cdpr.viz.style.apply_paper_style()` for a
publication-grade look (Times-equivalent fonts, restrained colour
palette, light-grey gridlines).

## 10.2 Phase-2 plot bundle (per model)

Implemented in `scripts/train_from_csv._emit_phase2_bundle` (commit
`0c4152d`). The bundle gates each figure on whether the necessary
arrays are present, so the same helper serves MLP / PINN / replay /
PPO / SAC paths.

Supervised + replay path (8–11 figures):

* `loss.png`, `loss_log.png` (supervised only)
* `pred_vs_truth.png`, `pred_vs_truth_scatter.png`
* `residuals.png`, `error_distribution.png`
* `cable_rmse_bar.png`, `error_evolution.png`, `error_heatmap.png`
* `tension_feasibility.png` (when bounds known)

RL path (8 figures):

* `loss.png` (returns as loss surrogate)
* `rl_returns.png`, `rl_cumulative_return.png`,
  `rl_rolling_return.png`, `rl_return_qq.png`,
  `rl_return_box_violin.png`, `rl_learning_split.png`,
  `rl_summary_panel.png`

## 10.3 Phase-2 tables

Every Phase-2 run also emits:

* **`metrics_table.md`** — model, run timestamp, key metrics in
  Markdown.
* **`per_cable_table.csv`** — per-cable RMSE, MAE, peak error, mean
  truth, mean prediction. (Supervised + replay only.)
* **`summary.md`** — human-readable single-page report with figure
  and table index.

## 10.4 Workspace and animation

* `cdpr.viz.workspace.plot_volume(workspace_grid, robot)` — voxel
  scatter or surface mesh of the WCW / WFW.
* `cdpr.viz.animation.animate(result, robot, fps=30, out_path=...)`
  — MP4 / GIF generation; live-streaming via the
  `iter_simulation` generator.

---

# 11. User Interfaces

The simulator deliberately offers multiple frontends so different
audiences can pick the right tool. They all consume the same
`SimulationRequest` schema from `cdpr.interface.specs`, and they all
import `cdpr` from `src/` via a `sys.path` prepend at module load —
no pip install is needed inside the frontend code.

## 11.1 Gradio (primary)

* **File**: `gradio_app.py` plus the tiny `app.py` shim used by HF
  Spaces.
* **URL (local)**: `http://127.0.0.1:7860`. Started by `python
  gradio_app.py`.
* **URL (public)**: `https://joetach-cdpr-simulator.hf.space`.
* **Hosting**: Hugging Face Spaces, SDK Gradio, hardware CPU-Basic
  (16 GB RAM, 2 vCPU, free tier).
* **Why primary**: HF Spaces gives 16× more memory than Streamlit
  Cloud's 1 GB worker, file uploads survive corporate firewalls,
  per-call state isolation, and the new accordion + chat UI
  declutters the dissertation-grade form.
* **Features**:
  - Tab 1 — Built-in examples (5 one-click runs).
  - Tab 2 — Custom Phase-1 simulation with a conversational
    simulation builder at the top, four collapsible accordions
    (Robot & dynamics, Trajectory, Time, Controller), PD/MPC family
    radio, Reset button, per-run audit (manifest + reference overlay
    + position SHA1).
  - Tab 3 — Upload CSV / Phase-2 with sub-tabs for single-model
    training and 5-model comparison.
* **LLM layer**: chat box parses free-text descriptions into a
  `SimulationRequest` via `cdpr.llm.simulation_builder`. Pre-fills
  every form field; flips the controller radio if the message
  mentions "MPC". Provider chain
  `gemini → openrouter → echo` with response cache and module-load
  prewarm.
* **Run summary**: status text now reports the run id, the
  controller's exact parameters, tracking RMS / peak, infeasible
  step count, position fingerprint, and the manifest path.
* **Theme + CSS**: Gradio 6.0 deprecation-clean —
  `theme=gr.themes.Soft()` and a 78-line hand-tuned slate CSS now
  ride on `.launch()` (not `Blocks()`).

## 11.2 Streamlit (secondary)

* **File**: `streamlit_app.py`.
* **Status**: feature-complete for Phase-1 simulation and Phase-2
  ingest; not the recommended public demo because Streamlit Cloud's
  1 GB worker OOMs on the 5-model compare path.
* **Use case**: classroom demonstrations on a local machine where
  Streamlit's interactive sidebar pattern is preferred over Gradio's
  tabbed layout.

## 11.3 Dash (secondary)

* **File**: `dash_app.py`.
* **Status**: implemented as part of Deploy 19/20 when the team
  re-assessed frontends. Retained because Dash's component-callback
  model gives the most fine-grained control over the 3D scene
  rendering.
* **Use case**: high-customisation rendering on a workstation.

## 11.4 FastAPI

* **File**: `src/cdpr/interface/api.py`.
* **Endpoints**: `POST /simulate`, `POST /visualise`, `GET /healthz`.
* **Hosting**: Render free tier at
  `https://cdpr-api.onrender.com/docs`.
* **Use case**: programmatic integration; a future React or
  Streamlit-Lite client can talk to this service instead of importing
  `cdpr` directly.

---

# 12. Deployment Architecture

## 12.1 Local deployment

```
laptop:
  python gradio_app.py
    -> http://127.0.0.1:7860
```

Everything works offline. No keys are required — the echo stub
serves as the LLM fallback.

## 12.2 Public deployment chain

```
GitHub (canonical)
  └── Tachia/cdpr_simulator main
        │  (auto-mirror on push)
        ├── Hugging Face Space  (Gradio runtime, public URL)
        ├── Render              (FastAPI service, public API)
        └── Cloudflare Pages    (docs site, static)
```

The user pushes to GitHub. Render and Cloudflare auto-deploy via
their GitHub webhooks. HF Spaces requires a separate push using a
Write-scope HF token because HF maintains its own git remote.

## 12.3 GitHub role

* **Canonical source of truth**. Every commit lands here first.
* **Single-branch workflow** — `main` is always deployable.
* **Audit trail** — co-authored commits document the development
  process.

## 12.4 Hugging Face role

* **Hosts the Gradio runtime**. The Space repo mirrors GitHub's
  contents; HF builds a Docker image and runs `python app.py`.
* **Boot contract**: `app.py` does `from gradio_app import demo,
  _CSS` and calls `demo.launch(theme=gr.themes.Soft(), css=_CSS)`.
* **Resource envelope**: CPU-Basic = 2 vCPU, 16 GB RAM. Sufficient
  for every Phase-2 model.
* **Auth chain**: push requires a Write-scope token belonging to
  account `JoeTach` with a verified email and no token-scope
  restriction on the target Space. The diagnostic script triages
  the chain in one command.

## 12.5 Supabase role

* **Optional runtime persistence**, not a code mirror. When
  `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are set as Space
  secrets, every simulation run also writes a row to the `simulations`
  table via `cdpr.storage.supabase`.
* **Schema** — Deploy 8a/b provisioned the migration directory; the
  table set is `simulations`, `models`, `artifacts`.

## 12.6 Render role

* **FastAPI service**. Render free tier, Python web service.
* **Auto-deploys on every GitHub push**.
* **URL** — `https://cdpr-api.onrender.com`. Docs at `/docs`.

## 12.7 Why `127.0.0.1` is not a public URL

`127.0.0.1` is the loopback interface; it only responds to requests
from the same machine. The HF Space exposes the same Gradio app on
an internet-routable address (`joetach-cdpr-simulator.hf.space`)
behind an HF-managed reverse proxy. The local URL and the public URL
serve identical code; they differ only in network reachability.

## 12.8 Required deployment architecture for public access

For a public CDPR demo, the minimum architecture is:

```
GitHub (code + history)
   │
   └── HF Space (Gradio worker, public URL)
         │
         ├── GEMINI_API_KEY (optional, free)
         └── OPENROUTER_API_KEY (optional, free fallback)
```

Render + Cloudflare are additive (API service + docs); Supabase
is additive (persistence). None of those four extras is required
for the chat + simulation flow to work.

---

# 13. Database Architecture

## 13.1 Current Supabase tables (designed; provisioning is user-side)

| Table | Columns (planned) | Purpose |
|---|---|---|
| `simulations` | `id` (uuid), `created_at`, `build_id`, `robot`, `trajectory_kind`, `controller`, `duration_s`, `dt_s`, `manifest_json`, `fingerprint_pos`, `tracking_rms_mm`, `infeasible_steps` | One row per Phase-1 run |
| `models` | `id`, `created_at`, `model_name`, `source_csv_path`, `epochs`, `metrics_json`, `bundle_path` | One row per Phase-2 model run |
| `artifacts` | `id`, `parent_id`, `parent_type` (`simulation` / `model`), `path`, `kind` (`png` / `csv` / `json` / `md`), `bytes` | Per-file index |

Schema migration files live in the migrations directory provisioned
in Deploy 8b.

## 13.2 Storage strategy

`cdpr.storage.supabase` dual-writes: local artefacts always land in
`out/<stamp>/` first, then a metadata row is inserted into Supabase
with paths pointing to the same files. The CDPR Simulator therefore
works without Supabase — Supabase only adds searchable history.

## 13.3 Future schema plans

* Add a `comparison` table for the multi-model harness.
* Add a `cable_models` table cataloguing per-run constitutive model
  choice and key parameters.
* Add a `defence_evidence` view that joins simulations + models +
  artifacts for the dissertation defence's live demo.

---

# 14. LLM Integration Roadmap

## 14.1 Current state

`cdpr.llm` already ships five providers with a chain:

| Provider | Default model | Status |
|---|---|---|
| Gemini | `gemini-2.0-flash` | live; `GEMINI_API_KEY` configured on the Space |
| OpenRouter | `deepseek/deepseek-r1` | designed; user to add `OPENROUTER_API_KEY` |
| Ollama | `llama3` | local-only |
| LM Studio | `local-model` | local-only |
| Echo | `echo-1` | stub; always works |

The chain falls through providers in order; the first one returning
parseable JSON wins. Response cache (32-entry FIFO) and provider
cache (module-level) ship in commit `0c4152d`.

## 14.2 Recommended routing

| Use case | Recommended provider |
|---|---|
| Conversational simulation builder (parse description → request) | Gemini → OpenRouter → echo |
| Explain a result (long-form natural language) | OpenRouter / DeepSeek-V3 |
| Mathematical / proof-style assistance | OpenRouter / DeepSeek-R1 |
| Offline / private | Ollama with `deepseek-r1` |
| Hot-reload local dev | LM Studio (it has a GUI) |
| CI / smoke / unit tests | Echo |

## 14.3 Chat assistant workflow (current)

```
user types
   │
   ▼
_chat_to_form (gradio_app.py)
   │
   ▼
describe_to_request (simulation_builder.py)
   │
   ├── _cached_provider("gemini") → Gemini call
   │      ├── HTTP 200 + valid JSON → cache result; return
   │      └── HTTP 429 / 5xx → mark try=>next
   │
   ├── _cached_provider("openrouter") → DeepSeek call
   │      ├── HTTP 200 + valid JSON → cache; return
   │      └── HTTP error → next
   │
   └── _cached_provider("echo") → keyword heuristic fallback
          └── return low-confidence default
   │
   ▼
_chat_to_form emits gr.update() for every form widget
   │
   ▼
Form repopulated; user presses Run simulation
```

## 14.4 Simulation assistant workflow (planned)

A second LLM role that *explains* a simulation result after it runs:
the controller's RMS / peak / infeasibility numbers go into a prompt
that asks the LLM to summarise what likely caused saturation, suggest
a tuning direction, and cite the relevant cable-feasibility condition.
Drops into the same `cdpr.llm` chain.

## 14.5 RL assistant workflow (planned)

Wraps the long-run PPO/SAC training script in a chat-driven tuning
loop. User says "the policy isn't learning, what should I change?";
the LLM consumes the recent `metrics.json` evolution and proposes
hyperparameter adjustments.

## 14.6 Research assistant workflow (planned)

A retrieval-augmented chat over the dissertation manuscript + the
simulator codebase + the cited references. Out of scope for the
current sprint; flagged here so future agent sessions can pick it up.

---

# 15. File Structure Inventory

This is a guided tour of the repo. Items marked `*` are central to
the dissertation; items marked `+` are optional / convenience.

## 15.1 Top-level

```
gradio_app.py                *   primary frontend (Gradio + chat + accordion + audit)
app.py                       *   HF Space entrypoint shim
streamlit_app.py             +   secondary Streamlit frontend
dash_app.py                  +   secondary Dash frontend
requirements.txt             *   pip dependencies for HF + Streamlit Cloud
pyproject.toml               *   package + extras (`viz`, `data`, `learn`, `rl`, `gradio`, ...)
LICENSE
README.md
.env.example                 *   documented env vars
.gitignore                   *   `out/` correctly excluded post commit a89fdfb
```

## 15.2 `src/cdpr/` (110 modules)

Listed by package in Section 3.3 above. Every package has its own
`__init__.py` exposing the public API; internal helpers live in
private submodules prefixed with `_`.

## 15.3 `scripts/`

```
run_simulation.py            *   single-shot CLI
run_example.py               *   built-in example runner
train_from_csv.py            *   Phase-2 supervised / RL / replay
compare_models.py            *   5-model bench
examples.py                  *   registry shared by CLI + Streamlit + Gradio
_csv_io.py                   *   CSV bridge (smart loader → numpy blocks)
test_llm.py                  +   smoke test for cdpr.llm
diagnose_hf_push.py          *   HF auth diagnostic
call_render.ps1              +   PowerShell wrapper for the Render API
train_interactive.ps1        +   PowerShell training launcher
```

## 15.4 `tests/`

```
test_core_frames.py          *   pose / twist / wrench primitives
test_kinematics.py           *   inverse + forward + Jacobian
test_statics.py              *   tension distribution QP
test_dynamics.py             *   simulator + integration
test_workspace.py            *   WCW / WFW
test_trajectory.py           *   parametric paths + time-scaling
test_robots.py               *   catalogue
test_streaming.py            *   iter_simulation
test_control_pd.py           *   PD
test_control_computed_torque.py
test_control_mpc.py          *   MPC
test_control_composed.py
test_cables.py               *   massless / elastic / Irvine / sagging
test_cable_models.py         *   constitutive model interface
test_benchmarks_cable_modes.py
test_viz_plots2d.py
test_viz_scene.py
test_viz_workspace.py
test_recording.py
test_reports.py
test_api.py                  *   FastAPI
test_ingest_loaders.py
test_ingest_pipeline.py
test_ingest_report.py
test_ingest_validate.py
test_learn_env.py            *   Gym env
test_learn_supervised.py     *   MLP / PINN
test_learn_rl_factories.py   *   PPO / SAC builders
test_learn_benchmark.py
test_identification.py
test_experiments.py
test_benchmarks.py
test_adapters_base.py
test_adapters.py
test_adapters_mujoco.py
conftest.py / conftest_viz.py    fixtures
```

## 15.5 `docs/`

```
csv-schema.md                *   canonical CSV format
deployment.md / .html        *   deployment guide
deployment-status.md         *   current deployment health
examples.md                  *   built-in examples catalogue
frontend-architecture.md     *   why Gradio is primary
frontend-architecture-revised.md
hf-auth-troubleshoot.md      *   HF push diagnostic walkthrough
hf-deployment.md             *   HF Space setup
index.html                   *   docs index
llm-providers.md             *   LLM setup walkthrough (Gemini, OpenRouter, ...)
multi-frontend.md            *   side-by-side comparison
run-locally.md               *   local quick-start
runbook.md                   *   operational runbook
terminal-execution.md        *   universal PowerShell + bash guide
project-memory-extract.md    *   THIS DOCUMENT
```

---

# 16. Testing Status

## 16.1 Test inventory

110 source modules under `src/cdpr/` are covered by 39 test files
under `tests/`. Phases 1.x through 7.x were each gated on the
corresponding test file passing before the next phase started.

## 16.2 Verified-passing as of Phase 7.10

Every test file listed in Section 15.4 was verified passing during
Phase 1.11, 2.12, 3.10, 4.10, 5.8, 6.7, and 7.9. The headline numbers
from those runs are recorded in `out/_phase*` directories.

## 16.3 Not yet re-verified after Deploy 24+

The Deploy 24+ commits modified frontend code (`gradio_app.py`),
the LLM layer (`cdpr.llm`), and the Phase-2 bundle helper
(`scripts/train_from_csv.py`). None of these touched scientific
core code. Re-running `pytest -x -q` is recommended but the prior
guarantees should hold.

## 16.4 Known limitations / debt

* **Streamlit Cloud OOM** (Deploy 10/11) — workaround shipped; the
  fix is to use HF Spaces instead. No longer a blocker.
* **`custom_callable` trajectory replay** (Section 9.7) — replay and
  RL skip CSVs whose source manifest carries an inline Python
  callable. Workaround: use catalogued trajectory kinds (circle,
  lissajous, line, hold).
* **MPC tuning at workspace edge** — default tuning saturates cables
  on > 5 cm amplitudes. Documented in §4.7.3; not a code bug.
* **The `cdpr_simulator/` subdirectory** — an embedded git repo from
  an earlier checkout. Slated for removal in the next housekeeping
  commit.

## 16.5 Current blockers (none, as of `0c4152d`)

No active blockers prevent dissertation experiments from running.

---

# 17. Research Contributions

## 17.1 What is novel

1. **A first-class tension-distribution QP as a runtime concern,**
   not a post-hoc check. Every controller in the codebase emits a
   desired wrench that goes through the QP before the dynamics step,
   so infeasibility is *observed* in the simulation result rather
   than hidden.
2. **A constitutive cable model interface that is mode-agnostic**.
   The dynamics simulator dispatches identical equations regardless
   of whether the cable is massless, linear elastic, Kelvin–Voigt,
   Irvine, or the SQCK hybrid. The hybrid itself is the author's
   contribution.
3. **A multi-frontend architecture that uses the same backend.** Same
   `SimulationRequest`, same `cdpr.dynamics.simulate()`, same plot
   bundle across Gradio / Streamlit / Dash / FastAPI / CLI. This
   makes the dissertation defence's live demo trivially auditable.
4. **A provider-agnostic LLM layer with automatic fallback chain.**
   No commercial SDK dependency; HTTP-only path; secret redaction at
   the raise site and at the chat-reply site. Works offline.
5. **Conversational simulation builder.** Free-text descriptions
   parse into `SimulationRequest` schemas with confidence rating,
   follow-up questions, and graceful fallback to keyword heuristics
   when no LLM is available. Demonstrably distinct from existing
   CDPR codebases.
6. **Per-run cryptographic audit.** SHA1 fingerprints of realised
   trajectories make every figure provably derived from this run's
   data, not any cache.
7. **A documented sim-to-data ingest pipeline.** Real CSV exports from
   instrumented test rigs map onto the analytic simulator's data
   model without manual schema work.

## 17.2 Publishable units (candidate venues)

| Contribution | Suggested venue | Form |
|---|---|---|
| The SQCK hybrid cable model + comparison against Irvine and Kelvin–Voigt | *Mechanism and Machine Theory* (MMT) or *Nonlinear Dynamics* | Journal |
| Provider-agnostic LLM-driven simulation builder | *SoftwareX* / *JOSS* | Software paper |
| Tension-distribution QP under three objective variants on the dissertation 8-cable robot | IROS / ICRA | Conference |
| Sim-to-data comparison harness | RAL with ICRA option | Conference |
| PINN inverse-dynamics on CDPRs | *IEEE T-RO* or *Control Engineering Practice* | Journal |
| Workspace analysis cross-cable-model comparison | IFToMM | Conference |
| Dissertation defence demo (live software + chat) | Innopolis dissertation council | Defence |

## 17.3 Thesis chapters this software supports directly

See Section 18.

---

# 18. Dissertation Mapping

This section binds software components to dissertation chapters.
Chapter numbering follows the author's working plan; adjust to match
the Innopolis house style at submission time.

## Chapter 1 — Introduction

* **Software modules**: none required — narrative chapter.
* **Figures**: a single `scene_3d.png` of the dissertation_8cable
  robot under the figure-eight reference, used as the chapter
  frontispiece.
* **Dataset**: none.

## Chapter 2 — Background and literature

* **Software modules**: none.
* **Tables**: comparison table of existing CDPR simulators (manual
  authoring; cite TASKE, MuJoCo, PyBullet, MATLAB-based codes).

## Chapter 3 — Kinematic modelling

* **Software modules**: `cdpr.core`, `cdpr.geometry`,
  `cdpr.kinematics.{inverse, forward, jacobian}`.
* **Mathematical models**: §4.2 above.
* **Figures**: cable-length traces from any `out/example-*/`
  `cable_lengths.png`.
* **Tables**: per-robot DoF + cable count + workspace size.
* **Experiments**: forward kinematics convergence test
  (`test_kinematics.py`).

## Chapter 4 — Dynamic modelling

* **Software modules**: `cdpr.dynamics.{rigid_body, integrators,
  simulator}`, `cdpr.statics.tension`.
* **Mathematical models**: §4.3, §4.4.
* **Figures**: `velocity.png`, `acceleration.png`, `orientation.png`,
  `cable_tensions.png` from any Phase-1 run.
* **Tables**: integration-step accuracy table.

## Chapter 5 — Workspace and feasibility

* **Software modules**: `cdpr.workspace.{closure, feasible, grid}`,
  `cdpr.statics.tension`.
* **Mathematical models**: §4.3, §4.5.
* **Figures**: `condition_number.png` per run; voxel renders of WCW
  and WFW for the IPAnema and CoGiRo classes via
  `cdpr.viz.workspace`.
* **Experiments**: closure test on `dissertation_8cable` at
  10 000 sampled poses.

## Chapter 6 — Control and tracking

* **Software modules**: `cdpr.control.{pd, computed_torque, mpc,
  composed}`, `cdpr.trajectory.*`.
* **Mathematical models**: §4.7.
* **Figures**: `tracking_error.png`, `rms_error_evolution.png`,
  `position.png`, `trajectory_xy.png` for each controller; a
  cross-controller bar chart of RMS / peak.
* **Tables**: per-controller, per-trajectory tracking metrics;
  proposed table is a 3-column × 4-row matrix (PD / computed torque /
  MPC × circle / spiral / M-shape / figure-eight).
* **Experiments**: the Phase-1 examples + new MPC sweep on the
  figure-eight at horizons \{4, 8, 12, 16, 20\}.

## Chapter 7 — Constitutive cable modelling

* **Software modules**: `cdpr.cables.{massless, elastic, kelvin_voigt,
  irvine, sqck_hybrid}`.
* **Mathematical models**: §4.6.
* **Figures**: `cable_stretch.png`, mode-stratified `cable_tensions.png`
  for the three modes; a single comparison figure of all three on the
  same scenario (Phase-7 smoke).
* **Tables**: per-mode RMS deviation from the massless baseline.
* **Experiments**: `tests/test_benchmarks_cable_modes.py`.

## Chapter 8 — Data-driven inverse dynamics

* **Software modules**: `cdpr.learn.{datasets, train}`,
  `scripts/train_from_csv.py` (--model mlp / pinn).
* **Mathematical models**: §7.2, §7.3.
* **Figures**: Phase-2 bundle for MLP and PINN on the dissertation
  CSV; the cross-model `compare_*` figures from
  `scripts/compare_models.py`.
* **Tables**: `per_cable_table.csv` per model.
* **Experiments**: 50-epoch fits for each model.

## Chapter 9 — Reinforcement learning controllers

* **Software modules**: `cdpr.learn.{env, rewards}`,
  `scripts/train_from_csv.py` (--model ppo / sac).
* **Mathematical models**: §8.1.
* **Figures**: RL bundle (`rl_returns`, `rl_cumulative_return`,
  `rl_learning_split`, ...).
* **Tables**: mean / std return per policy per seed.
* **Experiments**: long-run \(10^6\)-step training (planned).

## Chapter 10 — Sim-to-data validation

* **Software modules**: `cdpr.ingest.*`, `cdpr.learn.benchmark`,
  `cdpr.identification.*`.
* **Figures**: validation residuals between recorded CSV and
  analytic replay.
* **Tables**: identified-parameter ranges from the test rig.
* **Experiments**: dependent on Innopolis test rig CSV exports
  becoming available.

## Chapter 11 — Software architecture (this artefact)

* **Software modules**: everything.
* **Figures**: the architecture diagram from §3.1.
* **Appendix**: full `pip install` recipe + `python gradio_app.py`
  walkthrough.

## Chapter 12 — Conclusions and future work

* **Software modules**: none.
* **Pointers**: future roadmap (§20).

---

# 19. Current State Snapshot

## 19.1 What currently works

* The scientific core, all four controllers, all five constitutive
  cable models, the workspace analysis, the trajectory layer, the
  Phase-1 example registry. Verified via Phase 7.10 smoke.
* The Gradio frontend at `127.0.0.1:7860` (local) and
  `joetach-cdpr-simulator.hf.space` (public, currently serving
  `a89fdfb`).
* The chat box with the fallback chain, response cache, and provider
  pre-warm (commit `0c4152d`).
* The Phase-2 bundle that emits ≥10 figures + 3 tables per model run.
* The per-run audit with manifest + position fingerprint.
* The Reset button and the four collapsible accordions in the
  Custom Phase-1 tab.
* The FastAPI service on Render.

## 19.2 What partially works

* **OpenRouter fallback** — designed and wired; activates the moment
  `OPENROUTER_API_KEY` is set as a Space secret.
* **Supabase persistence** — adapter and migrations exist; activates
  when `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` are set.
* **Real-data ingestion from an Innopolis test rig** — the smart
  loader handles arbitrary schemas, but no real-rig CSV has been
  validated end-to-end. Dependent on the test rig becoming available.

## 19.3 What is broken / unfinished

* **The HF Space is one commit behind GitHub** (`a89fdfb` vs
  `0c4152d`). One `git push` to the HF remote closes the gap.
* **`custom_callable` trajectory replay** — the spiral and M-shape
  examples cannot be replayed from CSV. Planned: serialise the
  callable signature.
* **MPC on the 8-cable robot's full workspace** — saturates cables on
  amplitudes > 5 cm; requires re-tuning study.
* **Long-run RL training** — current PPO / SAC budgets are
  demonstrability-grade, not publication-grade.

## 19.4 Priority queue

| Priority | Item | Rationale |
|---|---|---|
| **Critical** | Push `0c4152d` to HF; add `OPENROUTER_API_KEY` Space secret | Production demo reflects the chat-speed and Phase-2 fixes the user asked for |
| **Critical** | Re-run `tests/` with the latest code | Verify no regression in scientific core |
| **High** | MPC tuning study at horizons \{4, 8, 12, 16, 20\} on the figure-eight; produce the cross-horizon plot | Chapter 6 evidence |
| **High** | Re-run the Phase 6.8 dissertation experiment bundle (`cdpr.experiments.run_bundle --bundle dissertation_full`) | Validates that all chapter numbers still hold under current code |
| **High** | First Innopolis test rig CSV through the smart loader → ingest → MLP / PINN comparison | Chapter 10 evidence |
| **Medium** | Long-run PPO + SAC training (\(10^6\) steps) | Chapter 9 evidence |
| **Medium** | Pin `gradio==6.15.2` in `requirements.txt` | Stop Gradio 7 from silently breaking the Space |
| **Medium** | Wire `cdpr.storage.supabase` into the Gradio Run button so every public-demo run persists | Defence-week evidence trail |
| **Low** | Add `scripts/clean_out.py` and clean the historic `out/` accumulation | Repo housekeeping |
| **Low** | Delete the embedded `cdpr_simulator/` subdirectory | Repo housekeeping |
| **Low** | Mark Deploy 12/13 as `deleted` in the task tracker | Administrative |

---

# 20. Future Roadmap

## 20.1 30-day roadmap

### Technical

* Push `0c4152d` to HF; confirm new build SHA is in the build log.
* Add `OPENROUTER_API_KEY` Space secret; verify fallback fires when
  Gemini 429s.
* Re-run `pytest -x -q` and triage anything red.
* Re-run the Phase 6.8 dissertation experiment bundle on the current
  code; archive the artefacts to `out/dissertation-<stamp>/`.

### Research

* Begin the MPC tuning study. Sweep horizon \{4, 8, 12, 16, 20\}
  with three pairs of \((Q_{\text{pos}}, R)\) on the figure-eight at
  amplitudes \(\le 4\) cm.

### Deployment

* Provision Supabase tables from the migrations directory.
* Wire `cdpr.storage.supabase` into `gradio_app._run_custom_simulation`.

### Publication

* Draft an outline of a *SoftwareX* paper covering the simulator.

### Dissertation

* Lock the table of contents.
* Generate the chapter-3 and chapter-4 figures from the current code
  base and include them in the LaTeX project.

## 20.2 90-day roadmap

### Technical

* Implement `custom_callable` trajectory serialisation so spiral and
  M-shape examples are replayable.
* Wire the explain-result assistant (§14.4) into Tab 2.
* Add CI: GitHub Actions matrix running `pytest` on every push.
* Pin every dependency to a tested minor version.

### Research

* SQCK hybrid cable model comparison on the figure-eight under three
  controllers — produce the cross-model, cross-controller table.
* First sim-to-data comparison run on the Innopolis test rig.

### Deployment

* Promote Streamlit + Dash to their own HF Spaces so the multi-frontend
  story is publicly demonstrable.

### Publication

* *SoftwareX* paper draft → submission.
* Conference paper on the SQCK hybrid cable model.

### Dissertation

* Complete Chapters 3, 4, 5 in LaTeX.

## 20.3 6-month roadmap

### Technical

* Long-run RL training infrastructure: SLURM-style script,
  checkpointing, monitor CSV → Phase-2 bundle reusing the supervised
  helper.
* Add an MuJoCo cross-check chapter — automatic divergence detection
  between the analytic simulator and MuJoCo on the same scenario.

### Research

* PINN inverse-dynamics paper with sim-to-data validation.
* PPO + SAC controller comparison against PD and MPC.

### Publication

* *Mechanism and Machine Theory* submission for the cable-model
  comparison.
* IROS / ICRA submission for the controller benchmark.

### Dissertation

* Complete Chapters 6, 7, 8 in LaTeX.
* Internal review by the supervisor.

## 20.4 1-year roadmap

### Technical

* React-Lite client talking to the FastAPI service for partners who
  don't want Gradio.
* Add Isaac Sim adapter to the cross-engine verification, not just as
  a stub.

### Research

* Complete cross-cable-model dissertation experiment.
* Complete cross-controller dissertation experiment.
* Complete sim-to-data validation chapter.

### Publication

* T-RO or Control Engineering Practice submission on the integrated
  framework.

### Dissertation

* All chapters complete.
* Mock defence at Innopolis.
* Defence-day live demo using the public HF Space.

---

# Appendix A — Quick command reference

## A.1 Local development

```powershell
# Clone + install
git clone https://github.com/Tachia/cdpr_simulator
cd cdpr_simulator
pip install -e ".[dev,viz,data,api,gui,gradio,dash,learn,rl]"

# Run the Gradio frontend locally
python gradio_app.py
# -> http://127.0.0.1:7860

# Run a Phase-1 example end-to-end
python scripts/run_example.py --name circle

# Run a Phase-2 model on a CSV
python scripts/train_from_csv.py --input out/example-circle/timeseries.csv `
    --model pinn --epochs 50 --out out/pinn-demo

# Run the multi-model comparison
python scripts/compare_models.py --input out/example-circle/timeseries.csv `
    --out out/compare-demo --models replay mlp pinn ppo sac
```

## A.2 Tests

```powershell
pytest -x -q --tb=short
```

## A.3 Push code

```powershell
# GitHub (Render + Cloudflare auto-deploy)
git push origin main

# HF Space (manual; needs Write token)
$tokenSecure = Read-Host "Paste hf_… token" -AsSecureString
$tokenPlain  = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($tokenSecure))
git push "https://JoeTach:${tokenPlain}@huggingface.co/spaces/JoeTach/cdpr-simulator" main:main
Remove-Variable tokenPlain, tokenSecure
```

## A.4 Diagnose an HF push failure

```powershell
$env:HF_TOKEN = "hf_..."
python scripts\diagnose_hf_push.py
$env:HF_TOKEN = $null
```

## A.5 Smoke-test the LLM layer

```powershell
python scripts\test_llm.py --provider gemini
python scripts\test_llm.py --provider openrouter
python scripts\test_llm.py --provider echo
```

# Appendix B — Glossary

| Term | Meaning |
|---|---|
| CDPR | Cable-Driven Parallel Robot |
| WCW | Wrench-Closure Workspace |
| WFW | Wrench-Feasible Workspace |
| QP | Quadratic Programme |
| MPC | Model-Predictive Control |
| PD | Proportional-Derivative |
| PINN | Physics-Informed Neural Network |
| MLP | Multi-Layer Perceptron |
| PPO | Proximal Policy Optimisation |
| SAC | Soft Actor-Critic |
| SE(3) | Special Euclidean group in 3D |
| SO(3) | Special Orthogonal group in 3D |
| Irvine cable model | Static catenary cable equations from Irvine (1981) |
| Kelvin–Voigt | Spring + dashpot constitutive model |
| SQCK | Sagging-Quasi-Catenary-Kelvin hybrid model (author's contribution) |
| HF | Hugging Face |
| RL | Reinforcement Learning |
| BFGS | Broyden–Fletcher–Goldfarb–Shanno (quasi-Newton optimiser) |
| RMS | Root Mean Square |
| MAE | Mean Absolute Error |

# Appendix C — References (placeholder)

Full bibliography to be assembled at dissertation finalisation. Key
authors expected to appear: Irvine, Bruckmann, Pott, Gosselin,
Verhoeven, Khosravi, Caverly, Carricato, Tho, Pinto, Picard, El-Ghazaly.

---

*End of Project Memory Extract. This document is a living artefact;
update it at every Deploy phase milestone and at every dissertation
chapter milestone.*
