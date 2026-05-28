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

# Shared loader (#3, #4, #5, #9 in the post-mortem).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _csv_io import load_csv_any                                     # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare PPO / SAC / PINN / MLP / replay on the "
                    "same CSV and emit a side-by-side report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=str, required=True,
                   help="Path or URL of a CSV (forwarded to train_from_csv.py).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--robot-config", type=str, default=None,
                   help="JSON robot description; forwarded to each per-model "
                        "subprocess so replay/RL work without manifest.json.")
    p.add_argument("--column-map", type=str, default=None,
                   help="Canonical-name=source-column overrides, comma-separated. "
                        "Forwarded to each per-model subprocess.")
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
    """Invoke scripts/train_from_csv.py as a subprocess and read its metrics.json.

    The full stderr is dropped to ``<sub_out>/stderr.txt`` regardless of
    success or failure --- makes debugging an "RMSE = blank" row in the
    comparison table a one-file lookup.
    """
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
    if args.robot_config:
        cmd += ["--robot-config", str(args.robot_config)]
    if args.column_map:
        cmd += ["--column-map", str(args.column_map)]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    sub_out.mkdir(parents=True, exist_ok=True)
    (sub_out / "stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
    (sub_out / "stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
    if proc.returncode != 0:
        return {
            "model": model,
            "status": "subprocess_failed",
            "returncode": proc.returncode,
            "stderr_path": str((sub_out / "stderr.txt").resolve()),
            "stderr_tail": (proc.stderr or "").splitlines()[-12:],
            "wall_time_s": round(dt, 2),
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
    """Bar chart of RMSE per model + tension overlay + ranking table.

    Models that failed or were skipped (missing extra, missing manifest)
    show up in the bar chart with an explicit annotation rather than
    silently dropping out of the comparison.
    """
    import matplotlib.pyplot as plt

    def _rmse(m: dict) -> float:
        for k in ("rmse_N", "tension_rmse_N", "rmse"):
            v = m.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return float("nan")

    def _status_note(m: dict) -> str:
        if m.get("status") == "subprocess_failed":
            return f"failed (rc={m.get('returncode','?')})"
        if m.get("status") == "skipped":
            return f"skipped: {m.get('missing') or m.get('reason') or 'unknown'}"
        if m.get("status") == "no_metrics":
            return "no metrics"
        return ""

    ranked = [
        (name, _rmse(m), m.get("wall_time_s", float("nan")), _status_note(m))
        for name, m in results.items()
    ]
    names = [r[0] for r in ranked]
    rmses = [r[1] for r in ranked]
    walls = [r[2] for r in ranked]
    notes = [r[3] for r in ranked]

    # --- 1. RMSE bar chart ------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(names, rmses, color=["C0", "C1", "C2", "C3", "C4"][:len(names)])
    ax.set_ylabel("tension RMSE [N]")
    ax.set_title("Per-model tension RMSE (lower is better)")
    ax.set_yscale("log")
    for name, val, note in zip(names, rmses, notes, strict=False):
        label = f"{val:.3g}" if val == val else (note or "—")
        ax.text(name, val if val == val else 1e-6, label,
                ha="center", va="bottom", fontsize=9)
    fig.savefig(out_dir / "compare_rmse.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- 2. Wall-time bar chart -------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.bar(names, walls, color="C5")
    ax.set_ylabel("wall time [s]")
    ax.set_title("Per-model end-to-end runtime")
    fig.savefig(out_dir / "compare_runtime.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # --- 3. Ranking table (now with a status column) ----------------
    fig, ax = plt.subplots(figsize=(8.0, 0.5 + 0.4 * len(names)))
    ax.set_axis_off()
    cells = [
        [
            name,
            f"{r:.3g}" if r == r else "—",
            f"{w:.2f}" if w == w else "—",
            note or "ok",
        ]
        for (name, r, w, note) in ranked
    ]
    table = ax.table(
        cellText=cells,
        colLabels=["model", "RMSE [N]", "wall time [s]", "status"],
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

    # Ranking JSON --- include the status note so a 5-model run with two
    # failures and three real numbers still reads cleanly.
    ranked_sorted = sorted(ranked, key=lambda r: (float("inf") if r[1] != r[1] else r[1]))
    (out_dir / "ranking.json").write_text(
        json.dumps(
            [{"model": n, "rmse_N": v if v == v else None,
              "wall_time_s": w if w == w else None,
              "status": note or "ok"}
             for (n, v, w, note) in ranked_sorted],
            indent=2,
        ),
        encoding="utf-8",
    )


def load_blocks(csv_input: str, *, overrides: dict[str, str] | None = None
                ) -> dict[str, np.ndarray]:
    """Load the canonical blocks via the shared CSV ingester.

    Handles local paths and ``http(s)://`` URLs identically. The numeric-
    suffix filter prevents the ``int('ayer')`` crash that previously
    occurred on external CSVs with columns like ``Layer`` or ``Time``.
    """
    columns, _report = load_csv_any(csv_input, overrides=overrides)
    t = columns["t"]
    pos_keys = ("px", "py", "pz")
    L_cols = sorted([k for k in columns if k.startswith("L") and k[1:].isdigit()],
                    key=lambda s: int(s[1:]))
    T_cols = sorted([k for k in columns if k.startswith("T") and k[1:].isdigit()],
                    key=lambda s: int(s[1:]))
    return {
        "time": t,
        "positions": np.column_stack([columns[k] for k in pos_keys])
                     if all(k in columns for k in pos_keys) else np.zeros((len(t), 3)),
        "cable_lengths": np.column_stack([columns[k] for k in L_cols])
                         if L_cols else np.zeros((len(t), 0)),
        "cable_tensions": np.column_stack([columns[k] for k in T_cols])
                          if T_cols else np.zeros((len(t), 0)),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"input  = {args.input}")
    print(f"out    = {args.out.resolve()}")
    print(f"models = {args.models}")

    overrides: dict[str, str] = {}
    if args.column_map:
        for pair in str(args.column_map).split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                overrides[k.strip()] = v.strip()
    ref_blocks = load_blocks(args.input, overrides=overrides)

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
