r"""Publication-grade plot styling.

Two presets are provided:

* :func:`apply_paper_style` -- compact, two-column journal figures. Times-style
  serif font (falls back to DejaVu Serif if unavailable), 9-pt axis labels,
  thin lines, vector-friendly defaults. Suitable for IEEE / Elsevier
  templates.
* :func:`apply_dissertation_style` -- larger, single-column dissertation
  figures. 11-pt labels, slightly heavier strokes, generous tick padding.

Both presets are *idempotent* and *additive* on top of the user's existing
rcParams, so they can be applied and reverted with :func:`reset_style`
without polluting global state.

The CDPR-specific colour cycles below are picked to remain legible when
collapsed to grayscale (Wong 2011 palette plus a sand-coloured accent for
the cable-tension heatmap). Cable indices follow a categorical cycle; cable
tensions get a sequential map (``viridis`` reversed) so that "more tension"
reads as warmer.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from cdpr.viz._lazy import require_matplotlib


# Wong's colour-blind-friendly categorical palette (Nature Methods 8, 441, 2011).
CDPR_CABLE_COLORS: tuple[str, ...] = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#999999",  # neutral gray
)

# Sequential colormap used by tension heatmaps. "viridis_r" keeps the
# perceptual ordering while putting bright yellow at the *low* tension end,
# which reads as "loose" --- the conventional cable-engineering intuition.
CDPR_TENSION_CMAP = "viridis_r"


def apply_paper_style() -> None:
    mpl = require_matplotlib()
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "lines.markersize": 3.0,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.4,
        "axes.grid": True,
        "axes.prop_cycle": mpl.cycler(color=list(CDPR_CABLE_COLORS)),
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,                # embed Type 42 (TrueType) so PDF is editable
        "ps.fonttype": 42,
    })


def apply_dissertation_style() -> None:
    mpl = require_matplotlib()
    apply_paper_style()
    mpl.rcParams.update({
        "font.size": 11.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 12.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "xtick.major.pad": 4.0,
        "ytick.major.pad": 4.0,
    })


def reset_style() -> None:
    mpl = require_matplotlib()
    mpl.rcdefaults()


@contextmanager
def styled(preset: str = "paper") -> Iterator[None]:
    """Context manager: apply a preset for the body, restore afterwards.

    Useful for one-off figures inside notebooks or test runs that should not
    permanently change rcParams.
    """
    mpl = require_matplotlib()
    saved = mpl.rcParams.copy()
    try:
        if preset == "paper":
            apply_paper_style()
        elif preset == "dissertation":
            apply_dissertation_style()
        else:
            raise ValueError(f"Unknown style preset: {preset!r}")
        yield
    finally:
        mpl.rcParams.update(saved)
