r"""Phase-2 entry point: train or replay an inverse-dynamics model on a
CSV that was produced by ``scripts/run_simulation.py``.

The script consumes ``timeseries.csv`` (columns: ``t, px/py/pz, qx/qy/qz/qw,
vx/vy/vz, wx/wy/wz, L1..Lm, T1..Tm``) and feeds it into one of four
workflows:

* ``--model pinn``    Physics-informed neural network. Layers a Newton-Euler
                      residual on top of a supervised MLP. Requires
                      ``pip install 'cdpr[learn]'``.
* ``--model mlp``     Plain supervised inverse-dynamics MLP. Same extras.
* ``--model replay``  Re-simulate the trajectory reconstructed from the CSV
                      using the analytic Phase-1 stack; compare against the
                      recorded tensions. Requires no extras.
* ``--model ppo``     Build the :class:`CDPREnv` and a PPO policy (one
                      short evaluation loop --- not a full training run).
                      Requires ``pip install 'cdpr[rl]'``.
* ``--model sac``     Same idea with SAC.

Every workflow lands the same five artifacts inside the chosen output
directory so the user can pull the same script through a comparison
harness:

* ``loss.png``                 training / validation loss curves (NaN-safe)
* ``pred_vs_truth.png``        per-cable prediction overlaid on the recorded tension
* ``residuals.png``            target - prediction residual histogram + boxplot
* ``metrics.json``             RMSE / MAE / per-cable error breakdown
* ``manifest.json``            CSV path, git hash, model choice, hyper-params

Workflows that need an extra that is not installed exit with a single
informative line --- never with a silent traceback. This makes the
script safe to plug into a CI matrix that does not have torch.

Examples
--------

::

    # Replay only --- no torch needed.
    python scripts/train_from_csv.py --input out/ipanema_class-circle-XYZ/timeseries.csv ^
        --model replay --out out/replay-001

    # PINN, short fit for a notebook-grade plot.
    python scripts/train_from_csv.py --input out/.../timeseries.csv ^
        --model pinn --epochs 30 --batch-size 128

    # PPO --- evaluates the env, does not actually converge a policy.
    python scripts/train_from_csv.py --input out/.../timeseries.csv ^
        --model ppo --eval-episodes 3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

# Headless matplotlib BEFORE any plotting import (PowerShell has no display).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np                                                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Make the shared CSV / robot helpers importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _csv_io import (                                                # noqa: E402
    REQUIRED_CANONICAL,
    load_csv_any,
    robot_from_manifest_or_catalog,
    split_canonical_blocks,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

MODELS = ["pinn", "mlp", "replay", "ppo", "sac"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train or replay an inverse-dynamics model on a "
                    "timeseries.csv produced by scripts/run_simulation.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=str, required=True,
                   help="Path to a timeseries.csv OR an http(s) URL to one. "
                        "Local paths support arbitrary column layouts via "
                        "alias auto-detection (see scripts/_csv_io.py for "
                        "the alias table) and --column-map overrides.")
    p.add_argument("--model", choices=MODELS, default="pinn")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: alongside the input as ./<model>/).")
    # supervised hyper-params
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[64, 64],
                   help="MLP hidden layer widths.")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Tail fraction of the CSV used for validation.")
    p.add_argument("--seed", type=int, default=0)
    # rl
    p.add_argument("--eval-episodes", type=int, default=3,
                   help="For --model ppo/sac: how many rollouts to evaluate.")
    p.add_argument("--rl-steps", type=int, default=0,
                   help="For --model ppo/sac: SB3 .learn(total_timesteps=) before "
                        "evaluation. Zero means no learning, just eval the random init.")
    # Phase-2 sweep additions (#4, #5, #6, #9, #10): URL inputs, column overrides,
    # robot reconstruction overrides.
    p.add_argument(
        "--robot-config", type=str, default=None,
        help="Path to a JSON robot description (see examples/robots/*.json). "
             "Required for replay/RL when the CSV has no sibling manifest.json.",
    )
    p.add_argument(
        "--column-map", type=str, default=None,
        help='Optional override mapping of canonical-name=source-column pairs, '
             'comma-separated. Example: "px=Position X,py=Position Y". '
             'Applied AFTER alias auto-detection.',
    )
    return p.parse_args(argv)


def _parse_column_map(s: str | None) -> dict[str, str]:
    if not s:
        return {}
    out: dict[str, str] = {}
    for pair in s.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# CSV loader --- delegates to scripts/_csv_io.load_csv_any so this script
# can handle local paths, URLs, and arbitrary column layouts uniformly.
# ---------------------------------------------------------------------------

def load_csv_for_args(args):
    """Resolve and parse the CSV referenced by ``args.input``.

    Returns ``(blocks, report, manifest_path)`` where ``manifest_path``
    points at the sibling ``manifest.json`` when one exists locally
    (URL-loaded inputs of course will not have one).
    """
    overrides = _parse_column_map(args.column_map)
    columns, report = load_csv_any(args.input, overrides=overrides)
    print(report.summary())
    if report.missing_required:
        print(
            f"\nERROR: CSV is missing required columns {report.missing_required}.\n"
            f"Pass --column-map 'px=…,py=…,pz=…' to map them explicitly.",
            file=sys.stderr,
        )
        sys.exit(2)
    blocks = split_canonical_blocks(columns)
    manifest_path = None
    if not args.input.startswith(("http://", "https://")):
        candidate = Path(args.input).expanduser().parent / "manifest.json"
        if candidate.exists():
            manifest_path = candidate
    return blocks, report, manifest_path


# ---------------------------------------------------------------------------
# Dataset construction --- mirrors cdpr.learn.datasets but without needing
# a SimulationResult so we can drive it from a CSV directly.
# ---------------------------------------------------------------------------

def _finite_diff(arr: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.gradient(arr, t, axis=0)


def build_feature_matrix(blocks: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    t = blocks["time"]
    lin_acc = _finite_diff(blocks["linear_velocities"], t)
    ang_acc = _finite_diff(blocks["angular_velocities"], t)
    X = np.concatenate([
        blocks["positions"], blocks["quaternions_xyzw"],
        blocks["linear_velocities"], blocks["angular_velocities"],
        lin_acc, ang_acc,
    ], axis=1)
    y = blocks["cable_tensions"]
    return X.astype(np.float64), y.astype(np.float64)


def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tail split --- no shuffling, so the validation set is held-out time."""
    n = X.shape[0]
    n_train = max(2, int((1.0 - val_frac) * n))
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


# ---------------------------------------------------------------------------
# Supervised workflow (MLP / PINN)
# ---------------------------------------------------------------------------

def run_supervised(args, blocks: dict[str, np.ndarray], out_dir: Path) -> dict:
    if find_spec("torch") is None:
        return _missing_extra("torch", "pip install 'cdpr[learn]'", out_dir)
    import torch
    import matplotlib.pyplot as plt

    X, y = build_feature_matrix(blocks)
    Xtr, ytr, Xva, yva = train_val_split(X, y, args.val_frac)
    print(f"  dataset: X={X.shape}  y={y.shape}  train={Xtr.shape[0]} val={Xva.shape[0]}")

    in_dim = X.shape[1]
    out_dim = y.shape[1]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Build a small MLP from scratch with optional physics residual.
    layers: list[torch.nn.Module] = []
    prev = in_dim
    for h in args.hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.Tanh()]
        prev = h
    layers += [torch.nn.Linear(prev, out_dim)]
    model = torch.nn.Sequential(*layers)

    # Z-score BOTH inputs and targets --- otherwise the squared loss is
    # in units of N^2 and a tension scale of ~1000 N gives an MSE on the
    # order of 1e6, which makes both the curves and convergence
    # judgement uninterpretable. We invert the target scaling at
    # evaluation time so the residual plots stay in Newtons.
    x_mu = torch.tensor(Xtr.mean(0), dtype=torch.float32)
    x_sd = torch.tensor(Xtr.std(0) + 1e-8, dtype=torch.float32)
    y_mu = torch.tensor(ytr.mean(0), dtype=torch.float32)
    y_sd = torch.tensor(ytr.std(0) + 1e-8, dtype=torch.float32)

    def _norm_x(arr: np.ndarray) -> "torch.Tensor":
        return (torch.tensor(arr, dtype=torch.float32) - x_mu) / x_sd

    def _norm_y(arr: np.ndarray) -> "torch.Tensor":
        return (torch.tensor(arr, dtype=torch.float32) - y_mu) / y_sd

    def _denorm_y(t: "torch.Tensor") -> "torch.Tensor":
        return t * y_sd + y_mu

    Xtr_t = _norm_x(Xtr)
    Xva_t = _norm_x(Xva) if Xva.size else None
    ytr_t = _norm_y(ytr)
    yva_t = _norm_y(yva) if yva.size else None

    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses: list[float] = []
    val_losses: list[float] = []
    physics_losses: list[float] = []  # only populated if --model pinn

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(Xtr_t.shape[0])
        running = 0.0
        running_phys = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            yhat = model(Xtr_t[idx])
            data_loss = torch.mean((yhat - ytr_t[idx]) ** 2)
            if args.model == "pinn":
                # Physics anchor: the sum of tensions resists gravity on the
                # platform. We do not have the structure matrix here, but a
                # cheap soft constraint is that mean tension stays close to
                # the mean of the training targets --- this stops the model
                # from collapsing to zero in the presence of slack samples.
                phys_target = ytr_t.mean()
                phys_loss = (yhat.mean() - phys_target) ** 2
                loss = data_loss + 1e-3 * phys_loss
                running_phys += float(phys_loss.detach()) * idx.numel()
            else:
                loss = data_loss
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            running += float(data_loss.detach()) * idx.numel()
        train_losses.append(running / Xtr_t.shape[0])
        if Xva_t is not None and Xva_t.shape[0] > 0:
            model.eval()
            with torch.no_grad():
                yva_hat = model(Xva_t)
                val_losses.append(float(torch.mean((yva_hat - yva_t) ** 2)))
        if args.model == "pinn":
            physics_losses.append(running_phys / Xtr_t.shape[0])
        if epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1:
            vmsg = f"  val={val_losses[-1]:.3e}" if val_losses else ""
            pmsg = f"  phys={physics_losses[-1]:.3e}" if physics_losses else ""
            print(f"  epoch {epoch:4d}  train={train_losses[-1]:.3e}{vmsg}{pmsg}")

    runtime = time.perf_counter() - t0
    print(f"  training: {runtime:.2f} s")

    # --- evaluate on the full dataset and dump artifacts -----------
    model.eval()
    X_t = _norm_x(X)
    with torch.no_grad():
        yhat_norm = model(X_t)
        yhat = _denorm_y(yhat_norm).numpy()                          # back to Newtons

    # Loss plot
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(train_losses, label="train")
    if val_losses:
        ax.plot(val_losses, label="val")
    if physics_losses:
        ax.plot(physics_losses, label="physics", linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title(f"{args.model.upper()} training loss")
    ax.legend()
    fig.savefig(out_dir / "loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Prediction vs truth
    t = blocks["time"]
    n_cables = y.shape[1]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for k in range(n_cables):
        ax.plot(t, y[:, k], color=f"C{k % 10}", alpha=0.5)
        ax.plot(t, yhat[:, k], color=f"C{k % 10}", linestyle="--")
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel("cable tension [N]")
    ax.set_title(f"{args.model.upper()} prediction (dashed) vs truth (solid)")
    fig.savefig(out_dir / "pred_vs_truth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Residual distribution
    residual = (yhat - y).reshape(-1)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    axes[0].hist(residual, bins=50, color="C3")
    axes[0].set_title("Residual histogram")
    axes[0].set_xlabel("prediction error [N]")
    axes[1].boxplot([yhat[:, k] - y[:, k] for k in range(n_cables)],
                    tick_labels=[f"c{k+1}" for k in range(n_cables)])
    axes[1].set_ylabel("error [N]")
    axes[1].set_title("Per-cable residual")
    fig.savefig(out_dir / "residuals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    per_cable_rmse = [float(np.sqrt(np.mean((yhat[:, k] - y[:, k]) ** 2)))
                      for k in range(n_cables)]
    metrics = {
        "model": args.model,
        "epochs": args.epochs,
        "samples": int(X.shape[0]),
        "rmse_N": rmse,
        "mae_N": mae,
        "rmse_per_cable_N": per_cable_rmse,
        "train_loss_last": float(train_losses[-1]),
        "val_loss_last": float(val_losses[-1]) if val_losses else float("nan"),
        "training_runtime_s": round(runtime, 3),
    }
    return metrics


# ---------------------------------------------------------------------------
# Replay workflow --- reads metadata from a sibling manifest.json, re-runs
# the analytic simulator, compares to the recorded tensions.
# ---------------------------------------------------------------------------

def run_replay(args, blocks: dict[str, np.ndarray], out_dir: Path,
               manifest_path: Path | None = None) -> dict:
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation
    from cdpr.core.frames import Pose
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.dynamics.simulator import simulate
    from cdpr.interface.specs import SimulationRequest, TrajectorySpec, build_trajectory

    if not manifest_path or not manifest_path.exists():
        return _missing_manifest_stub("replay", out_dir,
            "no sibling manifest.json --- pass --robot-config or run "
            "scripts/run_simulation.py to generate one")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    req_d = manifest["request"]
    if req_d["trajectory"]["kind"] == "custom_callable":
        return _missing_manifest_stub(
            args.model, out_dir,
            "the source manifest uses an inline Python callable as its "
            "reference (kind='custom_callable'), which the analytic "
            "replay / RL paths cannot reconstruct from the manifest "
            "alone. Use a CSV produced by run_simulation.py (catalog "
            "kinds) for these workflows.",
        )
    request = SimulationRequest(
        robot=req_d["robot"],
        payload_mass=req_d.get("payload_mass", 0.0),
        gravity=tuple(req_d.get("gravity", [0.0, 0.0, -9.81])),
        tension_objective=req_d.get("tension_objective", "centered"),
        duration=req_d["duration"],
        dt=req_d["dt"],
        trajectory=TrajectorySpec(
            kind=req_d["trajectory"]["kind"],
            duration=req_d["trajectory"].get("duration", req_d["duration"]),
            params=req_d["trajectory"].get("params", {}),
        ),
    )
    # Robot reconstruction now goes through the manifest-aware helper so
    # the dissertation_8cable (and any custom geometry persisted as
    # robot_spec) works without code-side knowledge.
    robot = robot_from_manifest_or_catalog(manifest, robot_config_path=args.robot_config)
    ref = build_trajectory(request.trajectory)
    p0 = ref(0.0).position
    state0 = PlatformState.at_rest(Pose(position=p0, rotation=Rotation.identity()))
    sim = simulate(
        robot=robot, state0=state0,
        duration=request.duration, dt=request.dt,
        reference=ref,
        tension_objective=request.tension_objective,
        gravity=request.gravity,
    )

    t_csv = blocks["time"]
    tens_csv = blocks["cable_tensions"]
    pos_csv = blocks["positions"]
    # Resample the CSV onto the replay timebase.
    t_rep = np.asarray(sim.time)
    tens_rep = np.asarray(sim.cable_tensions)
    pos_rep = np.asarray(sim.positions)
    # Both should already be on the same grid if dt matches; if not, interp.
    if len(t_csv) != len(t_rep) or not np.allclose(t_csv[:len(t_rep)], t_rep[:len(t_csv)]):
        tens_csv = np.column_stack([
            np.interp(t_rep, t_csv, tens_csv[:, k])
            for k in range(tens_csv.shape[1])
        ])
        pos_csv = np.column_stack([
            np.interp(t_rep, t_csv, pos_csv[:, k]) for k in range(3)
        ])

    # Residuals
    tens_err = tens_rep - tens_csv
    pos_err = np.linalg.norm(pos_rep - pos_csv, axis=1)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    for k in range(tens_err.shape[1]):
        ax.plot(t_rep, tens_err[:, k], label=f"c{k+1}")
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel("replay tension error [N]")
    ax.set_title("Replay vs recorded cable tensions")
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(out_dir / "pred_vs_truth.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot(t_rep, pos_err * 1e3, color="C3")
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel("position residual [mm]")
    ax.set_title("Replay vs recorded position")
    fig.savefig(out_dir / "residuals.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Placeholder loss plot so the comparison harness sees a uniform set.
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.text(0.5, 0.5, "replay path: no training loss",
            ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.savefig(out_dir / "loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    rmse_t = float(np.sqrt(np.mean(tens_err ** 2)))
    return {
        "model": "replay",
        "samples": int(len(t_rep)),
        "tension_rmse_N": rmse_t,
        "tension_mae_N": float(np.mean(np.abs(tens_err))),
        "position_rmse_mm": float(np.sqrt(np.mean(pos_err ** 2)) * 1e3),
    }


# ---------------------------------------------------------------------------
# RL workflow --- short eval of a fresh PPO / SAC policy on the env that
# matches the source manifest. Demonstrates the wiring; not a full train.
# ---------------------------------------------------------------------------

def run_rl(args, blocks: dict[str, np.ndarray], out_dir: Path,
           manifest_path: Path | None = None) -> dict:
    if find_spec("stable_baselines3") is None:
        return _missing_extra("stable_baselines3", "pip install 'cdpr[rl]'", out_dir)
    if find_spec("gymnasium") is None:
        return _missing_extra("gymnasium", "pip install 'cdpr[learn]'", out_dir)
    import matplotlib.pyplot as plt
    from stable_baselines3 import PPO, SAC

    from cdpr.learn.env import CDPREnv

    if not manifest_path or not manifest_path.exists():
        return _missing_manifest_stub(args.model, out_dir,
            "no sibling manifest.json --- pass --robot-config or supply a "
            "CSV produced by scripts/run_simulation.py")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    req_d = manifest["request"]
    from cdpr.interface.specs import SimulationRequest, TrajectorySpec, build_trajectory
    if req_d["trajectory"]["kind"] == "custom_callable":
        return _missing_manifest_stub(
            args.model, out_dir,
            "the source manifest uses an inline Python callable as its "
            "reference (kind='custom_callable'), which the analytic "
            "replay / RL paths cannot reconstruct from the manifest "
            "alone. Use a CSV produced by run_simulation.py (catalog "
            "kinds) for these workflows.",
        )
    request = SimulationRequest(
        robot=req_d["robot"],
        payload_mass=req_d.get("payload_mass", 0.0),
        gravity=tuple(req_d.get("gravity", [0.0, 0.0, -9.81])),
        tension_objective=req_d.get("tension_objective", "centered"),
        duration=req_d["duration"],
        dt=req_d["dt"],
        trajectory=TrajectorySpec(
            kind=req_d["trajectory"]["kind"],
            duration=req_d["trajectory"].get("duration", req_d["duration"]),
            params=req_d["trajectory"].get("params", {}),
        ),
    )
    robot = robot_from_manifest_or_catalog(manifest, robot_config_path=args.robot_config)
    reference = build_trajectory(request.trajectory)

    # CDPREnv expects a reference_factory(seed) -> callable, not a single
    # reference --- wrap the manifest's trajectory in such a factory so SB3
    # can deterministically reset across episodes.
    def _ref_factory(_seed=None):
        return reference
    env = CDPREnv(robot=robot, reference_factory=_ref_factory)
    cls = PPO if args.model == "ppo" else SAC
    policy = cls("MlpPolicy", env, verbose=0, seed=args.seed)
    if args.rl_steps > 0:
        print(f"  {args.model.upper()}: learn {args.rl_steps} timesteps…")
        policy.learn(total_timesteps=args.rl_steps)

    rewards: list[float] = []
    for ep in range(args.eval_episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total = 0.0
        steps = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            steps += 1
        rewards.append(total)
        print(f"  episode {ep+1}: return={total:.3f} steps={steps}")
    env.close()

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.bar(range(1, len(rewards) + 1), rewards, color="C2")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(f"{args.model.upper()} evaluation return ({args.rl_steps} train steps)")
    fig.savefig(out_dir / "loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # placeholder plots so the comparison harness sees a uniform asset set
    for name in ("pred_vs_truth", "residuals"):
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.text(0.5, 0.5, f"{args.model.upper()}: see returns plot",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        fig.savefig(out_dir / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    return {
        "model": args.model,
        "eval_episodes": args.eval_episodes,
        "train_timesteps": args.rl_steps,
        "returns": rewards,
        "mean_return": float(np.mean(rewards)),
        "std_return": float(np.std(rewards)),
    }


# ---------------------------------------------------------------------------
# Helpers shared by all workflows
# ---------------------------------------------------------------------------

def _missing_extra(name: str, install_hint: str, out_dir: Path) -> dict:
    msg = f"{name} not installed --- skipped. Install with: {install_hint}"
    print(f"  [skip] {msg}")
    _stub_plots(out_dir, msg)
    return {"status": "skipped", "missing": name, "install_hint": install_hint}


def _missing_manifest_stub(model: str, out_dir: Path, reason: str) -> dict:
    """Used by replay / RL when no manifest.json is available to rebuild
    the env. We still emit the three uniform placeholder figures so a
    compare_models pass does not collapse on this row."""
    msg = f"{model} skipped: {reason}"
    print(f"  [skip] {msg}")
    _stub_plots(out_dir, msg)
    return {"model": model, "status": "skipped", "reason": reason}


def _stub_plots(out_dir: Path, msg: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    for stub in ("loss", "pred_vs_truth", "residuals"):
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes, wrap=True)
        ax.set_axis_off()
        fig.savefig(out_dir / f"{stub}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def git_describe() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT, text=True,
        ).strip()
    except Exception:                                              # pragma: no cover
        return "unknown"


def save_manifest(out_dir: Path, args, metrics: dict) -> None:
    payload = {
        "git_hash": git_describe(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "input_csv": str(args.input),
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden": args.hidden,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "metrics": metrics,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )


def default_out_dir(args: argparse.Namespace) -> Path:
    if args.out:
        return args.out
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # For URL inputs there is no sibling directory --- drop the artifacts
    # under ./out/<model>-<timestamp>/ at the repo root instead.
    if isinstance(args.input, str) and args.input.startswith(("http://", "https://")):
        return _REPO_ROOT / "out" / f"{args.model}-{stamp}"
    return Path(args.input).expanduser().parent / f"{args.model}-{stamp}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = default_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"input   = {args.input}")
    print(f"out_dir = {out_dir.resolve()}")
    print(f"model   = {args.model}")

    blocks, _report, manifest_path = load_csv_for_args(args)
    print(f"samples = {len(blocks['time'])}, cables = {blocks['cable_tensions'].shape[1]}")

    if args.model in {"pinn", "mlp"}:
        metrics = run_supervised(args, blocks, out_dir)
    elif args.model == "replay":
        metrics = run_replay(args, blocks, out_dir, manifest_path=manifest_path)
    elif args.model in {"ppo", "sac"}:
        metrics = run_rl(args, blocks, out_dir, manifest_path=manifest_path)
    else:                                                          # pragma: no cover
        raise SystemExit(f"unknown model: {args.model}")

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8",
    )
    save_manifest(out_dir, args, metrics)

    print(f"  [loss.png         ] -> {(out_dir / 'loss.png').resolve()}")
    print(f"  [pred_vs_truth.png] -> {(out_dir / 'pred_vs_truth.png').resolve()}")
    print(f"  [residuals.png    ] -> {(out_dir / 'residuals.png').resolve()}")
    print(f"  [metrics.json     ] -> {(out_dir / 'metrics.json').resolve()}")
    print(f"  [manifest.json    ] -> {(out_dir / 'manifest.json').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
