r"""Markdown experiment summary --- intended as a "drop into a paper / lab
notebook" artefact.

The structure deliberately mirrors a methods-and-results section:

1. **Experiment** -- title, ID, tag dict.
2. **Robot** -- name, DOF, cable count, mass, anchor / attachment dimensions.
3. **Simulation settings** -- duration, dt, integrator, tension objective.
4. **Reproducibility** -- versions, seed, git revision, run timestamp.
5. **Results** -- key scalar statistics (position bounds, tension bounds,
   condition-number summary), infeasible-step count.
6. **Figures** -- optional list of relative paths to figure files.

Rendering to PDF is left to the user's preferred Markdown engine (Pandoc
is the natural choice for academic workflows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.recording.replay import Experiment


def write_markdown_summary(
    exp: "Experiment",
    path: str | Path,
    *,
    title: str | None = None,
    figures: list[tuple[str, str]] | None = None,
) -> Path:
    """Write a Markdown summary file describing the experiment.

    ``figures``, if provided, is a list of ``(caption, relative_path)``
    pairs; each becomes an ``![]()`` Markdown image entry. Captions support
    LaTeX math --- they're left verbatim for the Markdown processor to
    handle.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    md = _build_markdown(exp, title=title, figures=figures or [])
    p.write_text(md)
    return p


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _build_markdown(
    exp: "Experiment", *, title: str | None, figures: list[tuple[str, str]]
) -> str:
    meta = exp.metadata
    sim = meta.get("simulation", {})
    robot = meta.get("robot", {})

    head = title or meta.get("title", "CDPR experiment")
    lines: list[str] = [f"# {head}", ""]

    # 1. Experiment block
    lines += [
        "## Experiment",
        "",
        f"- **ID:** `{meta.get('experiment_id', '?')}`",
        f"- **Title:** {meta.get('title', '?')}",
    ]
    tags = meta.get("tags", {})
    if tags:
        lines.append("- **Tags:**")
        for k, v in sorted(tags.items()):
            lines.append(f"  - `{k}`: {v}")
    lines.append("")

    # 2. Robot
    anchors = np.asarray(robot.get("anchors", []), dtype=float)
    lines += [
        "## Robot",
        "",
        f"- **Name:** {robot.get('name', '?')}",
        f"- **DOF:** {robot.get('dof', '?')}",
        f"- **Cables:** {robot.get('n_cables', '?')}",
    ]
    if robot.get("mass") is not None:
        lines.append(f"- **Platform mass:** {robot['mass']:.3f} kg")
    if anchors.size:
        bbox = np.ptp(anchors, axis=0)
        lines.append(
            f"- **Frame extent:** {bbox[0]:.2f} × {bbox[1]:.2f} × {bbox[2]:.2f} m"
        )
    lines.append("")

    # 3. Simulation settings
    lines += [
        "## Simulation",
        "",
        f"- **Duration:** {sim.get('duration', float('nan')):.3f} s",
        f"- **Time step:** {sim.get('dt', float('nan')):.4g} s",
        f"- **Integrator:** {sim.get('integrator', '?')}",
        f"- **Tension objective:** {sim.get('tension_objective', '?')}",
        f"- **Gravity:** {sim.get('gravity', '?')}",
    ]
    if sim.get("reference_trajectory"):
        lines.append(f"- **Reference trajectory:** {sim['reference_trajectory']}")
    if sim.get("notes"):
        lines.append(f"- **Notes:** {sim['notes']}")
    lines.append("")

    # 4. Reproducibility
    mani = exp.manifest
    lines += [
        "## Reproducibility",
        "",
        f"- **Run at:** {mani.get('created_at', '?')}",
        f"- **cdpr:** {mani.get('cdpr_version', '?')}  "
        f"| **Python:** {mani.get('python_version', '?')}  "
        f"| **NumPy:** {mani.get('numpy_version', '?')}  "
        f"| **SciPy:** {mani.get('scipy_version', '?')}",
        f"- **Platform:** {mani.get('platform', '?')}",
    ]
    if mani.get("seed") is not None:
        lines.append(f"- **Seed:** {mani['seed']}")
    if mani.get("git_revision"):
        lines.append(f"- **Git revision:** `{mani['git_revision']}`")
    lines.append("")

    # 5. Results
    pos_norm = np.linalg.norm(exp.positions, axis=-1)
    vel_norm = np.linalg.norm(exp.linear_velocities, axis=-1)
    T = exp.cable_tensions
    lines += [
        "## Results",
        "",
        f"- **Samples recorded:** {len(exp.time)}",
        f"- **Position magnitude:** mean {pos_norm.mean():.4g} m, "
        f"max {pos_norm.max():.4g} m",
        f"- **Linear speed:** mean {vel_norm.mean():.4g} m/s, "
        f"max {vel_norm.max():.4g} m/s",
        f"- **Cable tension range:** {T.min():.4g} N to {T.max():.4g} N",
        f"- **Infeasible steps:** {len(exp.infeasible_steps)} of {len(exp.time)}",
    ]
    if not np.isnan(exp.condition_numbers).all():
        finite = exp.condition_numbers[np.isfinite(exp.condition_numbers)]
        if finite.size:
            lines.append(
                f"- **Structure-matrix κ₂(W):** median {np.median(finite):.3g}, "
                f"max {finite.max():.3g}"
            )
    lines.append("")

    # 6. Figures
    if figures:
        lines += ["## Figures", ""]
        for caption, rel_path in figures:
            lines.append(f"![{caption}]({rel_path})")
            lines.append("")

    # Raw manifest as a fenced block so the reader can see everything.
    lines += [
        "## Raw manifest",
        "",
        "```json",
        json.dumps(mani, indent=2, sort_keys=True),
        "```",
        "",
    ]
    return "\n".join(lines)
