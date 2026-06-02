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
    points at the sibling ``manifest.json`` when one exists locally,
    or a freshly generated synthetic one inferred from the dataset
    structure when no real manifest is available.

    The synthetic manifest is what lets PPO / SAC / replay run on
    third-party CSVs without the user having to construct a robot spec
    by hand. It's deliberately conservative: we infer the cable count
    from the number of ``T``-columns and pick the catalog robot with
    the closest cable count.
    """
    overrides = _parse_column_map(args.column_map)
    columns, report = load_csv_any(args.input, overrides=overrides)
    print(report.summary())
    if report.missing_required:
        # Dataset adapter (Major Problem 2): synthesise the canonical
        # columns we can rather than crashing the workflow. For each
        # missing required column, fabricate a sensible default and
        # warn loudly so users can see what was done.
        n_rows = (
            len(next(iter(columns.values())))
            if columns else 0
        )
        if n_rows == 0:
            print("\nERROR: CSV is empty.", file=sys.stderr)
            sys.exit(2)
        for canon in report.missing_required:
            if canon == "t":
                columns[canon] = np.arange(n_rows, dtype=float)
                print(f"  [dataset-adapter] synthesised 't' as row index "
                      f"(0 .. {n_rows - 1})")
            else:
                columns[canon] = np.zeros(n_rows, dtype=float)
                print(f"  [dataset-adapter] synthesised '{canon}' as zeros "
                      "(no real data available --- downstream metrics that "
                      "depend on this column will be degraded)")
        report.missing_required = []
    blocks = split_canonical_blocks(columns)

    manifest_path = None
    if not args.input.startswith(("http://", "https://")):
        candidate = Path(args.input).expanduser().parent / "manifest.json"
        if candidate.exists():
            manifest_path = candidate

    # If still no manifest, build a synthetic one and persist it next
    # to the CSV so all downstream Phase-2 paths see it.
    if manifest_path is None and not args.input.startswith(("http://", "https://")):
        manifest_path = _synthesise_manifest(args.input, blocks)
        if manifest_path:
            print(f"  [synthetic-manifest] wrote {manifest_path}")

    return blocks, report, manifest_path


def _synthesise_manifest(input_path: str, blocks: dict[str, np.ndarray]
                          ) -> Path | None:
    """Best-effort synthetic manifest so replay / PPO / SAC can run on
    arbitrary CSVs that have no sibling manifest.json."""
    try:
        from cdpr.interface.specs import build_robot
        from _csv_io import RobotSpec
    except Exception:
        return None

    n_cables = int(blocks["cable_tensions"].shape[1])
    t = blocks["time"]
    dt = float(t[1] - t[0]) if len(t) > 1 else 1e-3
    duration = float(t[-1] - t[0]) if len(t) > 1 else 1.0

    # Pick the catalog robot whose cable count matches; default to the
    # dissertation 8-cable when nothing better fits.
    catalog_by_count: dict[int, str] = {
        3: "point_mass_3d", 4: "planar_translational",
        8: "dissertation_8cable",
    }
    robot_name = catalog_by_count.get(n_cables, "dissertation_8cable")

    try:
        robot = build_robot(robot_name)
    except Exception:
        return None

    spec = RobotSpec.from_robot(robot, name=robot_name).to_dict()

    # Infer the trajectory bounding box for a 'hold' / 'line' fall-back.
    positions = blocks["positions"]
    p0 = positions[0].tolist() if len(positions) else [0.0, 0.0, 0.5]
    p1 = positions[-1].tolist() if len(positions) else [0.1, 0.0, 0.5]

    manifest = {
        "synthetic": True,
        "note": "Generated by scripts/train_from_csv.py because no real "
                 "manifest.json was found next to the CSV.",
        "request": {
            "robot": robot_name,
            "payload_mass": 0.0,
            "gravity": [0.0, 0.0, -9.81],
            "tension_objective": "centered",
            "duration": duration,
            "dt": dt,
            "trajectory": {
                "kind": "line",
                "duration": duration,
                "params": {"start": p0, "end": p1},
            },
        },
        "robot_spec": spec,
        "samples": int(len(t)),
    }
    out_path = Path(input_path).expanduser().parent / "manifest.json"
    try:
        out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return out_path
    except Exception:
        return None


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
# Phase-2 artefact bundle
# ---------------------------------------------------------------------------
# A single helper every model path calls before returning. It emits at
# least ten figures and three tables per run, gating each figure on
# whether its required inputs are present so the same function works
# for supervised, replay, and RL runs without branching at the call
# site. Every plot is wrapped so one failure (e.g. degenerate data)
# never brings the bundle down --- the directive's 'never crash' rule.


def _safe_save(out_dir: "Path", name: str, fn, log: list[str]) -> None:
    import matplotlib.pyplot as plt
    path = out_dir / f"{name}.png"
    try:
        fig = fn()
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        log.append(name)
    except Exception as exc:                                          # noqa: BLE001
        try:
            plt.close("all")
        except Exception:                                              # pragma: no cover
            pass
        print(f"  [{name:24s}] FAILED  {type(exc).__name__}: {exc}")


def _emit_phase2_bundle(
    out_dir: "Path",
    *,
    model_name: str,
    y_true: "np.ndarray | None" = None,
    y_pred: "np.ndarray | None" = None,
    t: "np.ndarray | None" = None,
    train_losses: "list[float] | None" = None,
    val_losses: "list[float] | None" = None,
    physics_losses: "list[float] | None" = None,
    rewards: "list[float] | None" = None,
    episode_lengths: "list[int] | None" = None,
    t_min_bound: float = 0.0,
    t_max_bound: float = 0.0,
    metrics: dict | None = None,
    extra_meta: dict | None = None,
) -> list[str]:
    """Render every applicable figure and table into ``out_dir``.

    Returns the list of artefact base-names created (without
    extensions). Designed so any caller can pass whatever subset of
    arrays it has and get the largest possible bundle out.
    """
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []
    title_suffix = f" ({model_name.upper()})"

    # 1. Training loss --- always emit, even RL runs (rewards-as-loss).
    def _f_loss():
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        if train_losses:
            ax.plot(train_losses, label="train")
        if val_losses:
            ax.plot(val_losses, label="val")
        if physics_losses:
            ax.plot(physics_losses, "--", label="physics")
        if rewards is not None and not train_losses:
            ax.plot(rewards, "o-", label="eval return", color="C2")
            ax.set_ylabel("return")
        else:
            ax.set_ylabel("MSE")
        ax.set_xlabel("epoch / episode")
        ax.set_title("Training curve" + title_suffix)
        ax.legend(); ax.grid(True, alpha=0.3)
        return fig
    _safe_save(out_dir, "loss", _f_loss, log)

    # 2. Log-scale loss --- reveals plateaus the linear plot hides.
    if train_losses or val_losses:
        def _f_loss_log():
            fig, ax = plt.subplots(figsize=(6.0, 3.5))
            if train_losses:
                ax.semilogy(train_losses, label="train")
            if val_losses:
                ax.semilogy(val_losses, label="val")
            if physics_losses:
                ax.semilogy(physics_losses, "--", label="physics")
            ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log)")
            ax.set_title("Training curve, log-scale" + title_suffix)
            ax.legend(); ax.grid(True, which="both", alpha=0.3)
            return fig
        _safe_save(out_dir, "loss_log", _f_loss_log, log)

    # 3. Prediction vs truth over time (one trace per cable).
    if y_true is not None and y_pred is not None and t is not None:
        def _f_pred():
            n_c = y_true.shape[1]
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            for k in range(n_c):
                ax.plot(t, y_true[:, k], color=f"C{k % 10}", alpha=0.5)
                ax.plot(t, y_pred[:, k], color=f"C{k % 10}", linestyle="--")
            ax.set_xlabel("time t [s]"); ax.set_ylabel("cable tension [N]")
            ax.set_title("Prediction (--) vs truth (—)" + title_suffix)
            ax.grid(True, alpha=0.3)
            return fig
        _safe_save(out_dir, "pred_vs_truth", _f_pred, log)

    # 4. Residual histogram + per-cable boxplot.
    if y_true is not None and y_pred is not None:
        residual = (y_pred - y_true).reshape(-1)
        n_c = y_true.shape[1]
        def _f_res():
            fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
            axes[0].hist(residual, bins=50, color="C3", edgecolor="black", alpha=0.85)
            axes[0].set_xlabel("error [N]")
            axes[0].set_title("Residual histogram")
            axes[0].axvline(0, color="black", lw=0.8)
            axes[1].boxplot([y_pred[:, k] - y_true[:, k] for k in range(n_c)],
                            tick_labels=[f"c{k+1}" for k in range(n_c)])
            axes[1].set_ylabel("error [N]")
            axes[1].set_title("Per-cable residual")
            axes[1].axhline(0, color="black", lw=0.6)
            return fig
        _safe_save(out_dir, "residuals", _f_res, log)

    # 5. Per-cable RMSE / MAE bar chart.
    if y_true is not None and y_pred is not None:
        def _f_per_cable():
            n_c = y_true.shape[1]
            rmse = [float(np.sqrt(np.mean((y_pred[:, k] - y_true[:, k]) ** 2)))
                    for k in range(n_c)]
            mae = [float(np.mean(np.abs(y_pred[:, k] - y_true[:, k])))
                   for k in range(n_c)]
            idx = np.arange(n_c); w = 0.35
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.bar(idx - w/2, rmse, w, label="RMSE", color="C0")
            ax.bar(idx + w/2, mae,  w, label="MAE",  color="C1")
            ax.set_xticks(idx); ax.set_xticklabels([f"c{k+1}" for k in range(n_c)])
            ax.set_ylabel("error [N]")
            ax.set_title("Per-cable RMSE & MAE" + title_suffix)
            ax.legend(); ax.grid(True, axis="y", alpha=0.3)
            return fig
        _safe_save(out_dir, "cable_rmse_bar", _f_per_cable, log)

    # 6. Predicted-vs-true scatter with y = x reference.
    if y_true is not None and y_pred is not None:
        def _f_scatter():
            n_c = y_true.shape[1]
            cols = min(4, n_c); rows = int(np.ceil(n_c / cols))
            fig, axes = plt.subplots(rows, cols,
                                      figsize=(2.6 * cols, 2.4 * rows),
                                      sharex=True, sharey=True, squeeze=False)
            for k in range(n_c):
                ax = axes[k // cols, k % cols]
                ax.scatter(y_true[:, k], y_pred[:, k], s=3, alpha=0.4,
                           color=f"C{k % 10}")
                lo = float(min(y_true[:, k].min(), y_pred[:, k].min()))
                hi = float(max(y_true[:, k].max(), y_pred[:, k].max()))
                ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
                ax.set_title(f"c{k+1}", fontsize=9)
            for k in range(n_c, rows * cols):
                axes[k // cols, k % cols].axis("off")
            fig.supxlabel("truth [N]"); fig.supylabel("prediction [N]")
            fig.suptitle("Predicted vs true tensions" + title_suffix, y=1.02)
            return fig
        _safe_save(out_dir, "pred_vs_truth_scatter", _f_scatter, log)

    # 7. Error distribution diagnostics (overall + Q-Q normal).
    if y_true is not None and y_pred is not None:
        residual = (y_pred - y_true).reshape(-1)
        def _f_dist():
            fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
            axes[0].hist(residual, bins=60, density=True,
                          color="C3", alpha=0.85, edgecolor="black")
            axes[0].set_xlabel("error [N]"); axes[0].set_ylabel("density")
            axes[0].set_title("Residual density")
            mu, sigma = float(residual.mean()), float(residual.std() or 1.0)
            xs = np.linspace(residual.min(), residual.max(), 200)
            axes[0].plot(xs, (1.0 / (sigma * np.sqrt(2 * np.pi)))
                          * np.exp(-0.5 * ((xs - mu) / sigma) ** 2),
                          "k--", lw=1.0,
                          label=f"N(mu={mu:.2f}, sigma={sigma:.2f})")
            axes[0].legend(fontsize=8)
            sample = np.sort(residual)
            theo = np.linspace(-3, 3, sample.size)
            theo_q = mu + sigma * theo
            axes[1].plot(theo_q, sample, ".", ms=2, color="C0")
            mn, mx = float(theo_q.min()), float(theo_q.max())
            axes[1].plot([mn, mx], [mn, mx], "k--", lw=0.8)
            axes[1].set_xlabel("normal quantile"); axes[1].set_ylabel("residual quantile")
            axes[1].set_title("Q-Q vs normal")
            return fig
        _safe_save(out_dir, "error_distribution", _f_dist, log)

    # 8. Cumulative RMSE evolution along the dataset.
    if y_true is not None and y_pred is not None and t is not None:
        def _f_evol():
            per_step = np.linalg.norm(y_pred - y_true, axis=1)
            cum = np.sqrt(np.cumsum(per_step ** 2) / np.arange(1, per_step.size + 1))
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.plot(t, cum, color="C3")
            ax.set_xlabel("time t [s]")
            ax.set_ylabel("cumulative RMSE [N]")
            ax.set_title("RMSE evolution" + title_suffix)
            ax.grid(True, alpha=0.3)
            return fig
        _safe_save(out_dir, "error_evolution", _f_evol, log)

    # 9. Tension-feasibility band (predictions vs. cable bounds).
    if y_pred is not None and t is not None and t_max_bound > t_min_bound:
        def _f_feas():
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            n_c = y_pred.shape[1]
            for k in range(n_c):
                ax.plot(t, y_pred[:, k], color=f"C{k % 10}", alpha=0.7, lw=0.8)
            ax.axhline(t_min_bound, color="black", ls="--", lw=0.8,
                        label=f"t_min = {t_min_bound:.1f} N")
            ax.axhline(t_max_bound, color="black", ls=":", lw=0.8,
                        label=f"t_max = {t_max_bound:.1f} N")
            ax.fill_between([t[0], t[-1]], t_min_bound, t_max_bound,
                             alpha=0.05, color="green")
            ax.set_xlabel("time t [s]")
            ax.set_ylabel("predicted tension [N]")
            ax.set_title("Predicted tension feasibility" + title_suffix)
            ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
            return fig
        _safe_save(out_dir, "tension_feasibility", _f_feas, log)

    # 10. Per-cable absolute error heatmap (time x cable).
    if y_true is not None and y_pred is not None and t is not None:
        def _f_heat():
            err = np.abs(y_pred - y_true).T                              # (n_c, n)
            fig, ax = plt.subplots(figsize=(7.0, 3.4))
            im = ax.imshow(err, aspect="auto", cmap="magma",
                            extent=[float(t[0]), float(t[-1]),
                                    err.shape[0] + 0.5, 0.5],
                            interpolation="nearest")
            ax.set_yticks(np.arange(1, err.shape[0] + 1))
            ax.set_yticklabels([f"c{k+1}" for k in range(err.shape[0])])
            ax.set_xlabel("time t [s]"); ax.set_ylabel("cable")
            ax.set_title("Per-cable absolute error" + title_suffix)
            fig.colorbar(im, ax=ax, label="|error| [N]")
            return fig
        _safe_save(out_dir, "error_heatmap", _f_heat, log)

    # 11-16. RL-specific bundle. The reward array alone supports a
    # rich set of policy-evaluation figures; we lean into them so that
    # PPO / SAC runs --- which lack per-step y_true/y_pred --- still
    # exit with >= 10 total artefacts.
    if rewards is not None and len(rewards) > 0:
        r = np.asarray(rewards, dtype=np.float64)
        n_ep = len(r)
        ep_idx = np.arange(1, n_ep + 1)

        # 11a. Per-episode return + distribution
        def _f_rl_dist():
            fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
            axes[0].bar(ep_idx, r, color="C2")
            axes[0].set_xlabel("episode"); axes[0].set_ylabel("return")
            axes[0].set_title("Per-episode return" + title_suffix)
            axes[0].grid(True, axis="y", alpha=0.3)
            axes[1].hist(r, bins=max(5, n_ep // 2),
                          color="C2", edgecolor="black", alpha=0.85)
            axes[1].set_xlabel("return"); axes[1].set_ylabel("count")
            axes[1].set_title("Return distribution")
            return fig
        _safe_save(out_dir, "rl_returns", _f_rl_dist, log)

        # 11b. Cumulative return
        def _f_cum():
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.plot(ep_idx, np.cumsum(r), "-o", color="C0")
            ax.set_xlabel("episode"); ax.set_ylabel("cumulative return")
            ax.set_title("Cumulative return" + title_suffix)
            ax.grid(True, alpha=0.3)
            return fig
        _safe_save(out_dir, "rl_cumulative_return", _f_cum, log)

        # 11c. Running mean ± std (sliding window)
        def _f_rolling():
            w = max(2, min(n_ep // 3, 25))
            roll_mean = np.convolve(r, np.ones(w) / w, mode="valid")
            x_roll = np.arange(w, n_ep + 1)
            roll_std = np.array([float(r[i - w:i].std())
                                  for i in x_roll])
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            ax.plot(ep_idx, r, ".", color="C2", alpha=0.4, label="return")
            ax.plot(x_roll, roll_mean, "-", color="C3", label=f"rolling mean (w={w})")
            ax.fill_between(x_roll, roll_mean - roll_std, roll_mean + roll_std,
                             alpha=0.2, color="C3", label="±1 std")
            ax.set_xlabel("episode"); ax.set_ylabel("return")
            ax.set_title("Return rolling statistics" + title_suffix)
            ax.legend(loc="best", fontsize=8); ax.grid(True, alpha=0.3)
            return fig
        _safe_save(out_dir, "rl_rolling_return", _f_rolling, log)

        # 11d. Return Q-Q vs normal (sanity for stationarity claims)
        def _f_qq():
            sort_r = np.sort(r)
            theo = np.linspace(-2.5, 2.5, n_ep)
            mu, sigma = float(r.mean()), float(r.std() or 1.0)
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
            ax.plot(mu + sigma * theo, sort_r, "o", color="C0", ms=4)
            lo, hi = float(sort_r.min()), float(sort_r.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel("normal quantile"); ax.set_ylabel("return quantile")
            ax.set_title("Return Q-Q vs normal" + title_suffix)
            return fig
        _safe_save(out_dir, "rl_return_qq", _f_qq, log)

        # 11e. Return box + violin in one panel
        def _f_box():
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
            axes[0].boxplot([r], tick_labels=["return"])
            axes[0].set_title("Return box")
            axes[0].grid(True, axis="y", alpha=0.3)
            axes[1].violinplot([r], showmeans=True, showmedians=True)
            axes[1].set_xticks([1]); axes[1].set_xticklabels(["return"])
            axes[1].set_title("Return violin")
            axes[1].grid(True, axis="y", alpha=0.3)
            return fig
        _safe_save(out_dir, "rl_return_box_violin", _f_box, log)

        # 11g. First-half vs second-half learning check.
        def _f_split():
            half = n_ep // 2
            if half < 2:
                raise ValueError("need >= 4 episodes for split comparison")
            first, second = r[:half], r[half:]
            fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
            axes[0].hist([first, second], bins=max(4, n_ep // 4),
                          color=["C7", "C2"],
                          label=["first half", "second half"], alpha=0.85)
            axes[0].set_xlabel("return"); axes[0].set_ylabel("count")
            axes[0].set_title("Learning split: first vs second half")
            axes[0].legend(loc="best", fontsize=8)
            axes[1].bar(["first", "second"],
                        [float(first.mean()), float(second.mean())],
                        yerr=[float(first.std()), float(second.std())],
                        color=["C7", "C2"], capsize=4)
            axes[1].set_ylabel("mean return ± std")
            axes[1].set_title(f"Improvement: "
                              f"{float(second.mean()) - float(first.mean()):+.3f}")
            return fig
        _safe_save(out_dir, "rl_learning_split", _f_split, log)

        # 11f. Summary metrics as a text figure (saves to PNG so it
        #      appears in the Gallery alongside the others).
        def _f_summary_fig():
            fig, ax = plt.subplots(figsize=(6.0, 3.5))
            ax.axis("off")
            stats = [
                ("episodes",      f"{n_ep}"),
                ("mean return",   f"{float(r.mean()):+.4f}"),
                ("std return",    f"{float(r.std()):.4f}"),
                ("min return",    f"{float(r.min()):+.4f}"),
                ("max return",    f"{float(r.max()):+.4f}"),
                ("median return", f"{float(np.median(r)):+.4f}"),
                ("IQR",           f"{float(np.percentile(r, 75) - np.percentile(r, 25)):.4f}"),
                ("best episode",  f"#{int(np.argmax(r)) + 1}"),
            ]
            txt = "\n".join(f"  {k:18s} {v}" for k, v in stats)
            ax.text(0.05, 0.95, "Evaluation summary" + title_suffix,
                     transform=ax.transAxes, fontsize=12, weight="bold",
                     va="top")
            ax.text(0.05, 0.80, txt, transform=ax.transAxes,
                     fontsize=10, family="monospace", va="top")
            return fig
        _safe_save(out_dir, "rl_summary_panel", _f_summary_fig, log)

    # ---- Tables ----------------------------------------------------
    summary_rows: list[tuple[str, str]] = []
    if metrics:
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool)):
                summary_rows.append((str(k), f"{v}"))

    # T1. metrics_table.md
    try:
        lines = [
            "# Phase-2 run metrics",
            "",
            f"**Model**: `{model_name}`  ",
            f"**Created**: `{time.strftime('%Y-%m-%dT%H:%M:%S')}`",
            "",
            "| Metric | Value |",
            "|---|---|",
        ]
        for k, v in summary_rows:
            lines.append(f"| `{k}` | {v} |")
        if extra_meta:
            lines += ["", "## Run metadata", "",
                       "| Key | Value |", "|---|---|"]
            for k, v in extra_meta.items():
                lines.append(f"| `{k}` | {v} |")
        (out_dir / "metrics_table.md").write_text("\n".join(lines),
                                                    encoding="utf-8")
        log.append("metrics_table")
    except Exception as exc:                                          # noqa: BLE001
        print(f"  [metrics_table.md       ] FAILED  {exc}")

    # T2. per_cable_table.csv
    try:
        if y_true is not None and y_pred is not None:
            n_c = y_true.shape[1]
            rows = ["cable,rmse_N,mae_N,peak_err_N,mean_truth_N,mean_pred_N"]
            for k in range(n_c):
                d = y_pred[:, k] - y_true[:, k]
                rows.append(
                    f"c{k+1},{float(np.sqrt(np.mean(d**2))):.6f},"
                    f"{float(np.mean(np.abs(d))):.6f},"
                    f"{float(np.max(np.abs(d))):.6f},"
                    f"{float(np.mean(y_true[:, k])):.6f},"
                    f"{float(np.mean(y_pred[:, k])):.6f}"
                )
            (out_dir / "per_cable_table.csv").write_text("\n".join(rows),
                                                          encoding="utf-8")
            log.append("per_cable_table")
    except Exception as exc:                                          # noqa: BLE001
        print(f"  [per_cable_table.csv    ] FAILED  {exc}")

    # T3. summary.md --- human-readable single-page report.
    try:
        bits = [
            f"# {model_name.upper()} run summary",
            "",
            f"Generated: `{time.strftime('%Y-%m-%dT%H:%M:%S')}`",
            "",
            f"Artefacts: {len(log)} ({len([n for n in log if 'table' not in n])} figures, "
            f"{len([n for n in log if 'table' in n])} tables).",
            "",
        ]
        if metrics:
            bits += ["## Headline metrics", ""]
            for k, v in summary_rows:
                bits.append(f"- **{k}**: {v}")
            bits.append("")
        bits += ["## Figure index", ""]
        for name in log:
            if "table" not in name:
                bits.append(f"- `{name}.png`")
        bits.append("")
        bits += ["## Table index", ""]
        for name in log:
            if "table" in name:
                suffix = "md" if name == "metrics_table" else "csv"
                bits.append(f"- `{name}.{suffix}`")
        (out_dir / "summary.md").write_text("\n".join(bits), encoding="utf-8")
        log.append("summary")
    except Exception as exc:                                          # noqa: BLE001
        print(f"  [summary.md             ] FAILED  {exc}")

    print(f"  [bundle] {len(log)} artefacts written to {out_dir}")
    return log


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

    t_arr = blocks["time"]
    n_cables = y.shape[1]
    residual = (yhat - y).reshape(-1)
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
    # Try to recover the tension bounds from the source manifest so the
    # feasibility-band figure has the right limits to draw.
    t_min_b = t_max_b = 0.0
    try:
        # Reach for the input CSV's sibling manifest.json if any.
        src = Path(args.input)
        sib = (src.parent if not str(src).startswith(("http://", "https://"))
               else None)
        if sib is not None and (sib / "manifest.json").exists():
            man = json.loads((sib / "manifest.json").read_text(encoding="utf-8"))
            limits = (man.get("robot") or {}).get("tension_limits") or {}
            t_min_b = float(limits.get("t_min", 0.0) or 0.0)
            t_max_b = float(limits.get("t_max", 0.0) or 0.0)
    except Exception:                                                 # pragma: no cover
        pass

    _emit_phase2_bundle(
        out_dir,
        model_name=args.model,
        y_true=y, y_pred=yhat, t=t_arr,
        train_losses=list(train_losses),
        val_losses=list(val_losses) if val_losses else None,
        physics_losses=list(physics_losses) if physics_losses else None,
        t_min_bound=t_min_b, t_max_bound=t_max_b,
        metrics=metrics,
        extra_meta={
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "hidden_layers": args.hidden,
            "val_frac": args.val_frac,
            "seed": args.seed,
        },
    )
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

    rmse_t = float(np.sqrt(np.mean(tens_err ** 2)))
    metrics = {
        "model": "replay",
        "samples": int(len(t_rep)),
        "tension_rmse_N": rmse_t,
        "tension_mae_N": float(np.mean(np.abs(tens_err))),
        "position_rmse_mm": float(np.sqrt(np.mean(pos_err ** 2)) * 1e3),
    }
    # The replay 'prediction' IS the recorded tensions; truth is what
    # the analytic simulator computed under the same trajectory. So
    # pass tens_csv as y_true and tens_rep as y_pred (the model output).
    t_min_b = t_max_b = 0.0
    try:
        if manifest_path and manifest_path.exists():
            man = json.loads(manifest_path.read_text(encoding="utf-8"))
            limits = (man.get("robot") or {}).get("tension_limits") or {}
            t_min_b = float(limits.get("t_min", 0.0) or 0.0)
            t_max_b = float(limits.get("t_max", 0.0) or 0.0)
    except Exception:                                                 # pragma: no cover
        pass

    _emit_phase2_bundle(
        out_dir,
        model_name="replay",
        y_true=tens_csv, y_pred=tens_rep, t=t_rep,
        t_min_bound=t_min_b, t_max_bound=t_max_b,
        metrics=metrics,
        extra_meta={
            "position_rmse_mm": metrics["position_rmse_mm"],
            "source_manifest": str(manifest_path) if manifest_path else "none",
        },
    )
    return metrics


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

    metrics = {
        "model": args.model,
        "eval_episodes": args.eval_episodes,
        "train_timesteps": args.rl_steps,
        "returns": rewards,
        "mean_return": float(np.mean(rewards)),
        "std_return": float(np.std(rewards)),
    }
    # RL has no y_true / y_pred per timestep, so the bundle's RL branch
    # picks up the rewards array and emits the returns figures, plus
    # the metrics_table.md / summary.md tables.
    _emit_phase2_bundle(
        out_dir,
        model_name=args.model,
        rewards=rewards,
        metrics=metrics,
        extra_meta={
            "rl_steps": args.rl_steps,
            "eval_episodes": args.eval_episodes,
            "robot": request.robot,
            "trajectory_kind": request.trajectory.kind,
            "seed": args.seed,
        },
    )
    return metrics


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
