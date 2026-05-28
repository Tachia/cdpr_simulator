r"""Auto-generated dissertation report bundle.

Takes a list of :class:`BenchmarkRun` (typically the output of
:meth:`BenchmarkSuite.run`) and emits a chapter-ready artifact set:

* ``report/figures/<scenario>_<backend>_tracking_error.{pdf,png}``
* ``report/figures/<scenario>_<backend>_cable_tensions.{pdf,png}``
* ``report/figures/<scenario>_<backend>_position.{pdf,png}``
* ``report/figures/<scenario>_<backend>_condition_number.{pdf,png}``
* ``report/tables/comparison.{csv,tex}``  --- one row per (scenario, backend)
* ``report/summary.md``  --- prose summary suitable for the appendix

The figures use the publication style; the tables emit booktabs LaTeX
ready to ``\input``; the markdown links to every artifact so it can be
shipped as a single self-contained directory.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.benchmarks.suite import BenchmarkRun


def _lazy_matplotlib():
    from cdpr.viz._lazy import require_matplotlib
    require_matplotlib()


# ---------------------------------------------------------------------------
# Figure helpers (one figure per run, named deterministically)
# ---------------------------------------------------------------------------

def _figure_paths(out_dir: Path, run: "BenchmarkRun", kind: str) -> tuple[Path, Path]:
    mode = run.metrics.cable_mode
    stem = f"{run.scenario_name}_{run.backend}_{mode}_{kind}"
    return out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"


def _emit_run_figures(run: "BenchmarkRun", out_dir: Path) -> list[Path]:
    """Render the canonical figures for one run.

    The condition-number figure requires ``run.robot`` to be populated
    (the benchmark suite does this automatically); if absent, it is
    skipped silently.
    """
    _lazy_matplotlib()
    import matplotlib.pyplot as plt
    from cdpr.viz import plots2d
    from cdpr.viz.style import apply_paper_style

    apply_paper_style()
    written: list[Path] = []

    # 1) Tracking error
    err = np.linalg.norm(run.result.positions - run.reference_positions, axis=1)
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(run.result.time, err, color="#0072B2")
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"$\| \mathbf{p}(t) - \mathbf{p}_\mathrm{ref}(t) \|$ [m]")
    ax.set_yscale("log")
    pdf, png = _figure_paths(out_dir, run, "tracking_error")
    fig.savefig(pdf); fig.savefig(png, dpi=300); plt.close(fig)
    written += [pdf, png]

    # 2) Cable tensions
    fig = plots2d.plot_cable_tensions(run.result, robot=run.robot)
    pdf, png = _figure_paths(out_dir, run, "cable_tensions")
    fig.savefig(pdf); fig.savefig(png, dpi=300); plt.close(fig)
    written += [pdf, png]

    # 3) Position vs reference, per-axis overlay
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for axis_idx, (label, color) in enumerate(
        zip(("x", "y", "z"), ("#0072B2", "#E69F00", "#009E73"))
    ):
        ax.plot(run.result.time, run.result.positions[:, axis_idx],
                color=color, label=f"actual ${label}$")
        ax.plot(run.result.time, run.reference_positions[:, axis_idx],
                color=color, linestyle="--", linewidth=0.8,
                label=f"ref ${label}$")
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"position [m]")
    ax.legend(ncols=2, fontsize=7, frameon=False)
    pdf, png = _figure_paths(out_dir, run, "position")
    fig.savefig(pdf); fig.savefig(png, dpi=300); plt.close(fig)
    written += [pdf, png]

    # 4) Condition number --- needs the geometric model.
    if run.robot is not None:
        from cdpr.core.frames import Pose
        from cdpr.kinematics.jacobian import structure_matrix
        from scipy.spatial.transform import Rotation
        kappa = np.empty(len(run.result.time))
        for k in range(len(run.result.time)):
            pose = Pose(
                position=run.result.positions[k],
                rotation=Rotation.from_quat(run.result.quaternions_xyzw[k]),
            )
            W = structure_matrix(pose, run.robot)
            s = np.linalg.svd(W, compute_uv=False)
            kappa[k] = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
        fig, ax = plt.subplots(figsize=(6.0, 3.0))
        ax.plot(run.result.time, kappa, color="#D55E00")
        ax.set_xlabel(r"time $t$ [s]")
        ax.set_ylabel(r"$\kappa_2(\mathbf{W})$")
        ax.set_yscale("log")
        pdf, png = _figure_paths(out_dir, run, "condition_number")
        fig.savefig(pdf); fig.savefig(png, dpi=300); plt.close(fig)
        written += [pdf, png]

    return written


# ---------------------------------------------------------------------------
# Cross-mode comparison figure
# ---------------------------------------------------------------------------

def _emit_mode_comparison_figures(
    runs: list["BenchmarkRun"], out_dir: Path,
) -> list[Path]:
    """One overlay figure per ``(scenario, backend)`` that ran in multiple modes.

    Shows mean cable tension over time for every active mode --- the
    standard dissertation comparison "how does the constitutive law
    change the tension envelope?".
    """
    _lazy_matplotlib()
    import matplotlib.pyplot as plt
    from cdpr.viz.style import CDPR_CABLE_COLORS, apply_paper_style

    apply_paper_style()
    written: list[Path] = []

    # Group by (scenario_name, backend); only emit when >= 2 distinct modes.
    grouped: dict[tuple[str, str], list[BenchmarkRun]] = {}
    for r in runs:
        grouped.setdefault((r.scenario_name, r.backend), []).append(r)

    for (scen, backend), bucket in grouped.items():
        modes = {r.metrics.cable_mode for r in bucket}
        if len(modes) < 2:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 3.2))
        for k, r in enumerate(sorted(bucket, key=lambda x: x.metrics.cable_mode)):
            mean_tau = r.result.cable_tensions.mean(axis=1)
            ax.plot(
                r.result.time, mean_tau,
                color=CDPR_CABLE_COLORS[k % len(CDPR_CABLE_COLORS)],
                label=r.metrics.cable_mode,
            )
        ax.set_xlabel(r"time $t$ [s]")
        ax.set_ylabel(r"mean cable tension $\bar T(t)$ [N]")
        ax.set_title(f"{scen} / {backend} --- cable-mode comparison")
        ax.legend(loc="best", frameon=False)
        stem = f"{scen}_{backend}_mode_comparison"
        pdf = out_dir / f"{stem}.pdf"
        png = out_dir / f"{stem}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=300); plt.close(fig)
        written += [pdf, png]
    return written


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _write_csv(runs: list["BenchmarkRun"], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "backend", "cable_mode",
            "tracking_rms_m", "tracking_peak_m",
            "orientation_rms_deg", "orientation_peak_deg",
            "control_effort_rms_N",
            "cable_tension_peak_N", "cable_tension_min_N",
            "feasibility_rate", "kappa_max", "kappa_median",
            "runtime_s",
        ])
        for r in runs:
            m = r.metrics
            w.writerow([
                r.scenario_name, r.backend, m.cable_mode,
                f"{m.tracking_error_rms:.6e}", f"{m.tracking_error_peak:.6e}",
                f"{m.orientation_error_rms_deg:.6e}", f"{m.orientation_error_peak_deg:.6e}",
                f"{m.control_effort_rms:.6e}",
                f"{m.cable_tension_peak:.6e}", f"{m.cable_tension_min:.6e}",
                f"{m.feasibility_rate:.6e}", f"{m.condition_number_max:.6e}",
                f"{m.condition_number_median:.6e}", f"{m.runtime_s:.4e}",
            ])


def _write_latex(runs: list["BenchmarkRun"], path: Path, caption: str, label: str) -> None:
    lines: list[str] = [
        r"\begin{table}[!htb]",
        r"  \centering",
        fr"  \caption{{{caption}}}",
        fr"  \label{{{label}}}",
        r"  \begin{tabular}{lllrrrrrr}",
        r"    \toprule",
        r"    scenario & backend & cable mode & tracking RMS [m] & peak [m] & "
        r"orientation RMS [deg] & feasibility & $\kappa_2$ max & runtime [s] \\",
        r"    \midrule",
    ]
    for r in runs:
        m = r.metrics
        # Escape underscores for LaTeX. Hoist the replace() calls into
        # locals --- f-string expressions cannot contain a backslash on
        # Python <3.12 (PEP 701 lifted the restriction only in 3.12).
        scenario_str = r.scenario_name.replace("_", r"\_")
        backend_str = r.backend.replace("_", r"\_")
        mode_str = m.cable_mode.replace("_", r"\_")
        lines.append(
            fr"    {scenario_str} & {backend_str} & {mode_str} & "
            fr"{m.tracking_error_rms:.3g} & {m.tracking_error_peak:.3g} & "
            fr"{m.orientation_error_rms_deg:.3g} & {m.feasibility_rate:.3g} & "
            fr"{m.condition_number_max:.3g} & {m.runtime_s:.3g} \\"
        )
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _summary_markdown(runs: list["BenchmarkRun"], title: str, fig_dir: Path) -> str:
    lines: list[str] = [f"# {title}", ""]
    lines += [
        "## Overview",
        "",
        f"- **Runs:** {len(runs)}",
        f"- **Scenarios:** {sorted({r.scenario_name for r in runs})}",
        f"- **Backends:** {sorted({r.backend for r in runs})}",
        f"- **Cable modes:** {sorted({r.metrics.cable_mode for r in runs})}",
        "",
        "## Metrics summary",
        "",
        "| scenario | backend | cable mode | tracking RMS [m] | peak [m] | feasibility | runtime [s] |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in runs:
        m = r.metrics
        lines.append(
            f"| {r.scenario_name} | {r.backend} | {m.cable_mode} | "
            f"{m.tracking_error_rms:.3g} | {m.tracking_error_peak:.3g} | "
            f"{m.feasibility_rate:.3g} | {m.runtime_s:.3g} |"
        )
    lines += ["", "## Figures", ""]
    for r in runs:
        mode = r.metrics.cable_mode
        prefix = f"{r.scenario_name}_{r.backend}_{mode}"
        lines += [
            f"### {prefix}",
            "",
            f"![Tracking error](figures/{prefix}_tracking_error.png)",
            "",
            f"![Cable tensions](figures/{prefix}_cable_tensions.png)",
            "",
            f"![Position vs reference](figures/{prefix}_position.png)",
            "",
            f"![Condition number](figures/{prefix}_condition_number.png)",
            "",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_bundle_report(
    runs: list["BenchmarkRun"],
    out_dir: str | Path,
    *,
    title: str = "CDPR benchmark report",
) -> dict[str, Path]:
    """Emit figures, tables, and markdown for a benchmark report bundle.

    The :class:`BenchmarkRun` records carry their own robot reference
    (the suite populates it from the scenario), so this function does
    not need a separate robot argument. Runs constructed by hand
    without ``robot`` will skip the condition-number figure.
    """
    out = Path(out_dir)
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for r in runs:
        paths = _emit_run_figures(r, fig_dir)
        written.update({p.name: p for p in paths})

    # Cross-mode tension overlay --- only meaningful when at least two cable
    # modes ran on the same (scenario, backend) pair.
    mode_overlay_paths = _emit_mode_comparison_figures(runs, fig_dir)
    written.update({p.name: p for p in mode_overlay_paths})

    csv_path = tab_dir / "comparison.csv"
    tex_path = tab_dir / "comparison.tex"
    _write_csv(runs, csv_path)
    _write_latex(runs, tex_path,
                 caption="Closed-loop benchmark comparison across scenarios and backends.",
                 label="tab:bench")
    written["comparison.csv"] = csv_path
    written["comparison.tex"] = tex_path

    md_path = out / "summary.md"
    md_path.write_text(_summary_markdown(runs, title, fig_dir), encoding="utf-8")
    written["summary.md"] = md_path

    # Machine-readable summary too.
    (out / "metrics.json").write_text(
        json.dumps([
            {"scenario": r.scenario_name, "backend": r.backend,
             "metrics": r.metrics.to_dict()}
            for r in runs
        ], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return written
