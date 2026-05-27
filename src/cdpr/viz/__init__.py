"""Visualisation layer for the CDPR framework.

Strict consumer-only relationship to the scientific core: nothing in this
package re-implements physics or kinematics. Inputs come from Phase 1 types
(``Pose``, ``Robot``, ``SimulationResult``, ``GridResult``, ``Trajectory``)
and outputs are :class:`matplotlib.figure.Figure` objects plus export-ready
files (PNG / SVG / PDF / MP4 / GIF).

Matplotlib is required (``pip install 'cdpr[viz]'``). Plotly / PyVista are
optional ``viz-extras`` for the few rich-3D helpers that benefit from them;
the core 3D rendering uses Matplotlib so that headless CI servers work out
of the box.

All entry-point modules below import lazily --- importing :mod:`cdpr.viz`
itself does *not* import Matplotlib, so the scientific core remains usable
on systems without it installed.
"""

from cdpr.viz._lazy import require_matplotlib

__all__ = ["require_matplotlib"]
