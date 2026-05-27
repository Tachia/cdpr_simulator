"""Export utilities for visualisation outputs.

Two helpers, both kept thin: :func:`save_figure` writes a Matplotlib figure
to disk, dispatching on the file extension; :func:`figure_to_png_bytes`
renders a figure into an in-memory PNG buffer (used by the FastAPI service
to return images as base64 without ever touching disk).

The dispatch table below is the source of truth for "what does PhaseĀ 2
consider an exportable figure format". Adding a format means adding a row
here, not editing call sites.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                           # pragma: no cover
    from matplotlib.figure import Figure


_SUPPORTED_FIGURE_FORMATS = {".png", ".svg", ".pdf", ".eps", ".jpg", ".jpeg"}


def save_figure(fig: "Figure", path: str | Path, **savefig_kwargs: object) -> Path:
    """Save a figure; format inferred from the file extension.

    Any keyword arguments are forwarded to ``Figure.savefig``. The publication
    style preset already sets ``dpi`` and ``bbox`` defaults, so most callers
    pass nothing extra.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix not in _SUPPORTED_FIGURE_FORMATS:
        raise ValueError(
            f"Unsupported figure extension {suffix!r}; supported: "
            f"{sorted(_SUPPORTED_FIGURE_FORMATS)}"
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, **savefig_kwargs)
    return p


def figure_to_png_bytes(fig: "Figure", *, dpi: int = 150) -> bytes:
    """Render a figure to PNG and return the raw bytes."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    return buffer.getvalue()
