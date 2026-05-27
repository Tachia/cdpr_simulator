r"""Rendering for :class:`cdpr.workspace.grid.GridResult` voxel masks.

Two styles are supported:

* :func:`render_workspace_scatter` -- one Matplotlib 3D figure with the
  voxels drawn as a scatter cloud, coloured by inclusion. Lightweight; fine
  up to a few :math:`100^3` voxels before redraw stalls.
* :func:`add_workspace` -- the same scatter, but added to a *caller-supplied*
  ``Axes3D``. This is what :func:`cdpr.viz.scene.render_scene` uses internally
  to overlay the workspace under the robot.

For very large grids the right tool is an isosurface extraction
(marching cubes), which produces a single mesh and renders interactively.
A PyVista-based implementation is the natural Phase-5 follow-up; we keep
this module dependency-free beyond Matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cdpr.viz._lazy import require_matplotlib

if TYPE_CHECKING:                                           # pragma: no cover
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from cdpr.workspace.grid import GridResult


def _grid_points(grid: "GridResult") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, Y, Z = np.meshgrid(grid.xs, grid.ys, grid.zs, indexing="ij")
    return X, Y, Z, grid.mask


def add_workspace(
    ax: "Axes3D",
    grid: "GridResult",
    *,
    inside_color: str = "#2CA02C",
    inside_alpha: float = 0.18,
    point_size: float = 8.0,
) -> None:
    """Add a voxel-cloud workspace overlay to an existing 3D axis."""
    X, Y, Z, mask = _grid_points(grid)
    if mask.any():
        ax.scatter(
            X[mask], Y[mask], Z[mask],
            s=point_size, c=inside_color, alpha=inside_alpha, depthshade=False,
            label=f"{grid.kind.upper()} ({mask.sum()} voxels)",
        )


def render_workspace_scatter(
    grid: "GridResult",
    *,
    inside_color: str = "#2CA02C",
    inside_alpha: float = 0.2,
    show_outside: bool = False,
    outside_alpha: float = 0.03,
) -> "Figure":
    """Render a workspace grid as a standalone 3D scatter figure."""
    require_matplotlib()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(6.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    X, Y, Z, mask = _grid_points(grid)

    if show_outside and (~mask).any():
        ax.scatter(X[~mask], Y[~mask], Z[~mask],
                   s=5.0, c="0.6", alpha=outside_alpha, depthshade=False)
    if mask.any():
        ax.scatter(X[mask], Y[mask], Z[mask],
                   s=10.0, c=inside_color, alpha=inside_alpha, depthshade=False,
                   label=f"{grid.kind.upper()} ({mask.sum()} voxels)")
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    ax.set_title(f"{grid.kind.upper()} workspace, "
                 f"orientation rotvec = {np.round(grid.orientation.as_rotvec(), 3).tolist()}")
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:                                  # pragma: no cover
        pass
    ax.legend(loc="upper right")
    return fig
