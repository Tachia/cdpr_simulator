r"""Captioned figure export for inclusion in LaTeX manuscripts.

A :class:`CaptionedFigure` is a lightweight pairing of a Matplotlib figure
with its caption, label, and the inclusion stanza that a LaTeX document
should use to bring it in. :func:`save_captioned_figure` writes both the
figure file (PDF + PNG side by side --- PDF for the manuscript, PNG for
slide decks and previews) and an accompanying ``.tex`` snippet that the
user can ``\input`` from their paper without copy-pasting captions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                           # pragma: no cover
    from matplotlib.figure import Figure


@dataclass(slots=True)
class CaptionedFigure:
    """A figure plus its scholarly metadata."""

    figure: "Figure"
    caption: str
    label: str
    width_textwidth: float = 0.95           # \\includegraphics[width=<this>\\textwidth]


_TEX_TEMPLATE = r"""\begin{{figure}}[!htb]
  \centering
  \includegraphics[width={width:.3f}\textwidth]{{{filename}}}
  \caption{{{caption}}}
  \label{{{label}}}
\end{{figure}}
"""


def save_captioned_figure(
    cf: CaptionedFigure,
    out_dir: str | Path,
    *,
    stem: str | None = None,
    also_png: bool = True,
) -> dict[str, Path]:
    """Write the figure as PDF (+ optional PNG) plus a LaTeX inclusion snippet.

    Returns a dict mapping the written role (``"pdf"``, ``"png"``, ``"tex"``)
    to its on-disk path. The PDF and PNG share the same basename, derived
    from ``stem`` if supplied or from ``cf.label`` otherwise.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = stem or _safe_stem(cf.label)

    pdf_path = out / f"{base}.pdf"
    cf.figure.savefig(pdf_path)
    written: dict[str, Path] = {"pdf": pdf_path}

    if also_png:
        png_path = out / f"{base}.png"
        cf.figure.savefig(png_path, dpi=300)
        written["png"] = png_path

    tex_path = out / f"{base}.tex"
    tex_path.write_text(
        _TEX_TEMPLATE.format(
            width=cf.width_textwidth,
            filename=base + ".pdf",
            caption=cf.caption,
            label=cf.label,
        )
    )
    written["tex"] = tex_path
    return written


def _safe_stem(label: str) -> str:
    """Turn a LaTeX label like ``fig:tension-track`` into a safe filename stem."""
    return label.replace(":", "_").replace("/", "_").replace(" ", "_")
