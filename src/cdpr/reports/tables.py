r"""Structured tables for reports: CSV and LaTeX.

Produces booktabs-style LaTeX so the output drops cleanly into a manuscript
(``\\usepackage{booktabs}`` plus the snippet from :func:`summary_table_latex`
and no further tweaks).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.recording.replay import Experiment


# ---------------------------------------------------------------------------
# Channel summaries (one row per channel)
# ---------------------------------------------------------------------------

def _channel_rows(exp: "Experiment") -> list[tuple[str, float, float, float, float]]:
    """Compute (name, mean, std, min, max) for the main scalar channels."""
    pos_norm = np.linalg.norm(exp.positions, axis=-1)
    vel_norm = np.linalg.norm(exp.linear_velocities, axis=-1)
    ang_norm = np.linalg.norm(exp.angular_velocities, axis=-1)
    rows: list[tuple[str, float, float, float, float]] = [
        (r"position magnitude [m]",
         float(pos_norm.mean()), float(pos_norm.std()),
         float(pos_norm.min()), float(pos_norm.max())),
        (r"linear speed [m/s]",
         float(vel_norm.mean()), float(vel_norm.std()),
         float(vel_norm.min()), float(vel_norm.max())),
        (r"angular speed [rad/s]",
         float(ang_norm.mean()), float(ang_norm.std()),
         float(ang_norm.min()), float(ang_norm.max())),
    ]
    if not np.isnan(exp.condition_numbers).all():
        kappa = exp.condition_numbers[np.isfinite(exp.condition_numbers)]
        if kappa.size:
            rows.append((
                r"structure-matrix $\kappa_2(\mathbf{W})$",
                float(kappa.mean()), float(kappa.std()),
                float(kappa.min()), float(kappa.max()),
            ))
    return rows


def summary_table_csv(exp: "Experiment", path: str | Path) -> Path:
    """Write a per-channel summary CSV with columns: channel, mean, std, min, max."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "mean", "std", "min", "max"])
        for row in _channel_rows(exp):
            w.writerow([row[0]] + [f"{x:.6e}" for x in row[1:]])
    return p


def summary_table_latex(
    exp: "Experiment", path: str | Path, *, label: str = "tab:summary", caption: str = "",
) -> Path:
    """Write a booktabs LaTeX table with the same channel summary."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = _channel_rows(exp)
    lines: list[str] = [
        r"\begin{table}[!htb]",
        r"  \centering",
        fr"  \caption{{{caption}}}",
        fr"  \label{{{label}}}",
        r"  \begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    channel & mean & std & min & max \\",
        r"    \midrule",
    ]
    for name, mean, std, lo, hi in rows:
        lines.append(
            fr"    {name} & {mean:.4g} & {std:.4g} & {lo:.4g} & {hi:.4g} \\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    p.write_text("\n".join(lines))
    return p


# ---------------------------------------------------------------------------
# Cable-level table (one row per cable)
# ---------------------------------------------------------------------------

def cable_summary_table(
    exp: "Experiment", path: str | Path, *, fmt: str = "csv",
    label: str = "tab:cables", caption: str = "",
) -> Path:
    """Per-cable summary (tension and length statistics) in CSV or LaTeX."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    m = exp.cable_tensions.shape[1]
    cable_rows: list[tuple[int, float, float, float, float, float]] = []
    for i in range(m):
        T = exp.cable_tensions[:, i]
        L = exp.cable_lengths[:, i]
        cable_rows.append((
            i + 1,
            float(T.mean()), float(T.min()), float(T.max()),
            float(L.mean()), float(L.max() - L.min()),
        ))

    if fmt == "csv":
        with p.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cable", "T_mean [N]", "T_min [N]", "T_max [N]",
                        "L_mean [m]", "L_range [m]"])
            for row in cable_rows:
                w.writerow([row[0]] + [f"{x:.6e}" for x in row[1:]])
        return p

    if fmt == "latex":
        lines: list[str] = [
            r"\begin{table}[!htb]",
            r"  \centering",
            fr"  \caption{{{caption}}}",
            fr"  \label{{{label}}}",
            r"  \begin{tabular}{rrrrrr}",
            r"    \toprule",
            r"    cable $i$ & $\bar T_i$ [N] & $T_{i,\min}$ [N] & $T_{i,\max}$ [N]"
            r" & $\bar \ell_i$ [m] & $\ell_{i,\max} - \ell_{i,\min}$ [m] \\",
            r"    \midrule",
        ]
        for i, t_mean, t_min, t_max, l_mean, l_range in cable_rows:
            lines.append(
                fr"    {i} & {t_mean:.4g} & {t_min:.4g} & {t_max:.4g} & "
                fr"{l_mean:.4g} & {l_range:.4g} \\"
            )
        lines += [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
        p.write_text("\n".join(lines))
        return p

    raise ValueError(f"Unsupported table format {fmt!r}; use 'csv' or 'latex'.")
