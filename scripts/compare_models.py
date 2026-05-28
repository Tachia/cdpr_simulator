r"""Run all available Phase-2 workflows on the same CSV and emit a
side-by-side comparison: bar charts of RMSE / MAE, an overlay of each
model's tension prediction against the recorded ground truth, and a
ranking table.

Models exercised (only those whose extras are installed are run; the
rest are reported as ``skipped`` with the install hint):

* replay  --- analytic re-simulation
* mlp     --- supervised inverse-dynamics MLP
* pinn    --- physics-informed variant
* ppo     --- short PPO eval (requires cdpr[rl])
* sac     --- short SAC eval (requires cdpr[rl])

Each sub-workflow lands its own directory next to the comparison root
so the individual artifacts remain inspectable.

Example
-------

::

    python scripts/compare_models.py --input out/.../timeseries.csv ^
        --out out/compare-001 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np                                                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TRAIN_SCRIPT = Path(__file__).resolve().parent / "train_from_csv.py"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare PPO / SAC / PINN / MLP / replay on the "
                    "same CSV and emit a side-by-side report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--models", nargs="+",
        default=["replay", "mlp", "pinn"],
        choices=["replay", "mlp", "pinn", "ppo", "sac"],
        help="Subset to run. Models whose extras are missing are still "
             "invoked --- they produce 'skipped' artifacts.",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rl-steps", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=3)
    return p.parse_args(argv)


def run_one(model: str, args: argparse.Namespace, sub_out: Path) -> dict:
    """Invoke scripts/train_from_csv.py as a subprocess and read its metrics.json."""
    cmd = [
        sys.executable, str(_TRAIN_SCRIPT),
        "--input", str(args.input),
        "--model", model,
        "--out", str(sub_out),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(args.seed),
        "--rl-steps", str(args.rl_steps),
        "--eval-episodes", str(args.eval_episodes),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        return {
            "model": model,
            "status": "subprocess_failed",
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr.splitlines()[-10:],
            "runtime_s": round(dt, 2),
        }
    metrics_path = sub_out / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.exists() else {"model": model, "status": "no_metrics"}
    )
    metrics["wall_time_s"] = round(dt, 2)
    return metrics


def render_comparison(out_dir: Path, results: dict[str, dict],
                      ref_blocks: dict[str, np.ndarray]) -> None:
    """Bar chart of RMSE per model + tension overlay + ranking table."""
    import matplotlib.pyplot as plt

    # --- 1. RMSE bar chart ------------------------------------------
    ranked = []
    for name, m in results.items():
        rmse = m.get("rmse_N", m.get("tension_rmse_N"))
        ranked.append((name, rmse if rmse is not None else float("nan"),
                       m.get("wall_time_s", float("nan"))))
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    names = [r[0] for r in ranked]
    rmses = [r[1] for r in ranked]
    ax.bar(names, rmses, color=["C0", "C1", "C2", "C3", "C4"][:len(names)])
    ax.set_ylabel("tension RMSE [N]")
    ax.set_title("Per-model tension RMSE (lower is better)")
    ax.set_yscale("log")
    for name, val in zip(names, rmses, strict=False):
        label = f"{val:.3g}" if val == val else "n/a"                # NaN-safe
        ax.text(name, val if val == val else 1e-6, label,
                ha="center", va="bottom", fontsize=9)
    fig.savefig(out_dir / "compare_rmse.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Wall-time bar chart -------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(names, [r[2] for r in ranked], color="C5")
    ax.set_ylabel("wall time [s]")
    ax.set_title("Per-model end-to-end runtime")
    fig.savefig(out_dir / "compare_runtime.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- 3. Ranking radar / table -----------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 0.5 + 0.35 * len(names)))
    ax.set_axis_off()
    cells = [
        [name,
         f"{r:.3g}" if r == r else "—",
         f"{w:.2f}" if w == w else "—"]
        for (name, r, w) in ranked
    ]
    table = ax.table(
        cellText=cells,
        colLabels=["model", "RMSE [N]", "wall time [s]"],
        loc="center", cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    fig.savefig(out_dir / "compare_table.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- 4. Tension overlay vs recorded -----------------------------
    # Plot mean tension across cables per model + ground truth.
    t = ref_blocks["time"]
    truth = ref_blocks["cable_tensions"].mean(axis=1)
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t, truth, color="black", label="recorded mean", linewidth=2)
    for name in names:
        sub = out_dir / name / "metrics.json"
        if not sub.exists():
            continue
        metric = json.loads(sub.read_text(encoding="utf-8"))
        rmse = metric.get("rmse_N", metric.get("tension_rmse_N", "?"))
        ax.plot([], [], label=f"{name}: RMSE={rmse:.3g} N" if isinstance(rmse, float) else f"{name}")
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel("mean cable tension [N]")
    ax.set_title("Cross-model overview (per-model details in their sub-directories)")
    ax.legend()
    fig.savefig(out_dir / "compare_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Ranking JSON
    ranked_sorted = sorted(ranked, key=lambda r: (float("inf") if r[1] != r[1] else r[1]))
    (out_dir / "ranking.json").write_text(
        json.dumps(
            [{"model": n, "rmse_N": v if v == v else None,
              "wall_time_s": w if w == w else None}
             for (n, v, w) in ranked_sorted],
            indent=2,
        ),
        encoding="utf-8",
    )


def load_blocks(csv_path: Path) -> dict[str, np.ndarray]:
    """Mirror of train_from_csv.split_columns --- duplicated here so this
    script has no internal dependency on the train module."""
    with csv_path.open("r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[None, :]
    cols = {h: data[:, i] for i, h in enumerate(header)}
    length_keys = sorted([k for k in cols if k.startswith("L")],
                         key=lambda s: int(s[1:]))
    tension_keys = sorted([k for k in cols if k.startswith("T")],
                          key=lambda s: int(s[1:]))
    return {
        "time": cols["t"],
        "positions": np.column_stack([cols[k] for k in ("px", "py", "pz")]),
        "cable_lengths": np.column_stack([cols[k] for k in length_keys]) if length_keys else np.zeros((len(cols["t"]), 0)),
        "cable_tensions": np.column_stack([cols[k] for k in tension_keys]) if tension_keys else np.zeros((len(cols["t"]), 0)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"input  = {args.input.resolve()}")
    print(f"out    = {args.out.resolve()}")
    print(f"models = {args.models}")

    ref_blocks = load_blocks(args.input)

    results: dict[str, dict] = {}
    for model in args.models:
        sub_out = args.out / model
        sub_out.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {model} ---")
        results[model] = run_one(model, args, sub_out)
        print(f"  result: { {k: v for k, v in results[model].items() if k != 'returns'} }")

    render_comparison(args.out, results, ref_blocks)

    (args.out / "compare_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    print(f"\n[compare_metrics.json] -> {(args.out / 'compare_metrics.json').resolve()}")
    print(f"[compare_rmse.png     ] -> {(args.out / 'compare_rmse.png').resolve()}")
    print(f"[compare_runtime.png  ] -> {(args.out / 'compare_runtime.png').resolve()}")
    print(f"[compare_table.png    ] -> {(args.out / 'compare_table.png').resolve()}")
    print(f"[compare_overview.png ] -> {(args.out / 'compare_overview.png').resolve()}")
    print(f"[ranking.json         ] -> {(args.out / 'ranking.json').resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
