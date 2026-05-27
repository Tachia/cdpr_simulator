r"""Three-dimensional rendering of the CDPR scene.

A *scene* here means the geometric snapshot of the robot at one pose:
base anchors, the moving platform, and the cables connecting them. Optional
overlays add a tension colormap, a singularity indicator (drawn from the
condition number of the structure matrix), and the trajectory ribbon.

The renderer is built on Matplotlib's :class:`mpl_toolkits.mplot3d.Axes3D`
because that ships with the standard install and runs headless. Richer
fly-through interaction is delegated to the optional PyVista helper in
:mod:`cdpr.viz.scene_pyvista` (not implemented in Phase 2 --- only the
matplotlib renderer is required by the directive).

Coupling rule: this module imports from ``cdpr.core``, ``cdpr.geometry``,
and ``cdpr.kinematics`` for *reading* only (poses, anchors, cable
endpoints, structure matrix). It does not call into ``cdpr.dynamics`` or
``cdpr.statics``; if a tension array is wanted, the caller passes one in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdpr.viz._lazy import require_matplotlib
from cdpr.viz.style import CDPR_CABLE_COLORS, CDPR_TENSION_CMAP

if TYPE_CHECKING:                                           # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from cdpr.core.frames import Pose
    from cdpr.geometry.robot import Robot


# ---------------------------------------------------------------------------
# Scene rendering options
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SceneOptions:
    """Switches controlling what the renderer draws.

    Defaults give a clean publication-grade still (anchors as small dots,
    cables as thin lines, platform as a wireframe). Set ``tension_heatmap``
    to True and pass ``tensions=...`` to :func:`render_scene` to recolour
    the cables by tension.
    """

    show_anchors: bool = True
    show_platform: bool = True
    show_cables: bool = True
    show_world_axes: bool = True
    show_platform_axes: bool = True
    show_workspace: bool = False                 # only if a GridResult is passed
    tension_heatmap: bool = False                # only meaningful with tensions=...
    singularity_indicator: bool = True
    cable_linewidth: float = 1.0
    anchor_size: float = 20.0
    platform_alpha: float = 0.25
    elev: float = 22.0
    azim: float = -55.0


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def render_scene(
    robot: "Robot",
    pose: "Pose",
    *,
    options: SceneOptions | None = None,
    tensions: ArrayLike | None = None,
    trajectory_positions: ArrayLike | None = None,
    workspace_grid: "object | None" = None,
    ax: "Axes3D | None" = None,
) -> "Figure":
    r"""Render the CDPR at a single pose.

    Parameters
    ----------
    robot:
        Robot configuration (anchors and platform attachments come from here).
    pose:
        Platform pose to draw.
    options:
        Style switches; see :class:`SceneOptions`.
    tensions:
        Optional ``(m,)`` array. When provided together with
        ``options.tension_heatmap=True``, cables are coloured by tension via
        the :data:`~cdpr.viz.style.CDPR_TENSION_CMAP` colormap.
    trajectory_positions:
        Optional ``(N, 3)`` array drawn as a thin ribbon behind the platform.
        Used for "where has the platform been" overlays in playback.
    workspace_grid:
        Optional :class:`cdpr.workspace.grid.GridResult` rendered as a
        scatter cloud when ``options.show_workspace`` is true. Delegated to
        :func:`cdpr.viz.workspace.add_workspace` so we don't import the
        workspace module here.
    ax:
        Existing 3D axis to draw into. If ``None``, a new figure is built.
    """
    opts = options or SceneOptions()
    require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection  # noqa: F401

    if ax is None:
        fig = plt.figure(figsize=(6.0, 5.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    anchors = np.asarray(robot.anchors, dtype=np.float64)
    attachments_world = pose.rotation.apply(robot.attachments) + pose.position

    # --- Anchors ----------------------------------------------------------
    if opts.show_anchors:
        ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2],
                   s=opts.anchor_size, color="0.2", depthshade=False, label="anchors")

    # --- Cables -----------------------------------------------------------
    if opts.show_cables:
        segments = [
            np.stack([anchors[i], attachments_world[i]]) for i in range(robot.n_cables)
        ]
        if opts.tension_heatmap and tensions is not None:
            t = np.asarray(tensions, dtype=np.float64).reshape(-1)
            if t.shape[0] != robot.n_cables:
                raise ValueError(
                    f"tensions length {t.shape[0]} != n_cables {robot.n_cables}"
                )
            from matplotlib import colormaps
            from matplotlib.colors import Normalize
            t_lo = float(robot.limits.t_min.min()) if robot.limits is not None else float(t.min())
            t_hi = float(robot.limits.t_max.max()) if robot.limits is not None else float(t.max())
            norm = Normalize(vmin=t_lo, vmax=t_hi)
            cmap = colormaps[CDPR_TENSION_CMAP]
            colors = [cmap(norm(ti)) for ti in t]
            lc = Line3DCollection(segments, colors=colors, linewidths=opts.cable_linewidth + 0.4)
            ax.add_collection3d(lc)
            mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(t)
            cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.07)
            cbar.set_label(r"cable tension $T_i$ [N]")
        else:
            colors = [CDPR_CABLE_COLORS[i % len(CDPR_CABLE_COLORS)] for i in range(robot.n_cables)]
            lc = Line3DCollection(segments, colors=colors, linewidths=opts.cable_linewidth)
            ax.add_collection3d(lc)

    # --- Platform --------------------------------------------------------
    if opts.show_platform:
        ax.scatter(
            attachments_world[:, 0], attachments_world[:, 1], attachments_world[:, 2],
            s=opts.anchor_size * 1.5, color=CDPR_CABLE_COLORS[3], depthshade=False,
            label="platform attachments",
        )
        if robot.n_cables >= 3:
            _draw_platform_hull(ax, attachments_world, alpha=opts.platform_alpha)

    # --- World axes ------------------------------------------------------
    if opts.show_world_axes:
        _draw_triad(ax, np.zeros(3), np.eye(3), length=_scene_scale(anchors) * 0.08, alpha=0.6)

    # --- Platform body axes ----------------------------------------------
    if opts.show_platform_axes:
        R = pose.rotation.as_matrix()
        _draw_triad(ax, pose.position, R, length=_scene_scale(anchors) * 0.05, alpha=1.0)

    # --- Trajectory ribbon -----------------------------------------------
    if trajectory_positions is not None:
        traj = np.asarray(trajectory_positions, dtype=np.float64)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                color=CDPR_CABLE_COLORS[2], linewidth=0.8, alpha=0.7, label="trajectory")

    # --- Workspace overlay ----------------------------------------------
    if opts.show_workspace and workspace_grid is not None:
        from cdpr.viz.workspace import add_workspace
        add_workspace(ax, workspace_grid)

    # --- Singularity indicator -------------------------------------------
    if opts.singularity_indicator:
        from cdpr.kinematics.jacobian import condition_number
        kappa = condition_number(pose, robot)
        _annotate_singularity(ax, kappa)

    _set_equal_3d_box(ax, anchors)
    ax.view_init(elev=opts.elev, azim=opts.azim)
    ax.set_xlabel(r"$x$ [m]")
    ax.set_ylabel(r"$y$ [m]")
    ax.set_zlabel(r"$z$ [m]")
    return fig


# ---------------------------------------------------------------------------
# Drawing primitives (private)
# ---------------------------------------------------------------------------

def _draw_triad(
    ax: "Axes3D", origin: NDArray[np.float64], R: NDArray[np.float64],
    length: float, alpha: float,
) -> None:
    colors = ("#D62728", "#2CA02C", "#1F77B4")  # x, y, z
    for k in range(3):
        end = origin + length * R[:, k]
        ax.plot(
            [origin[0], end[0]], [origin[1], end[1]], [origin[2], end[2]],
            color=colors[k], linewidth=1.5, alpha=alpha,
        )


def _draw_platform_hull(
    ax: "Axes3D", attachments_world: NDArray[np.float64], *, alpha: float
) -> None:
    """Draw the platform as the convex hull of its attachment points.

    For three points this is a triangle; for four or more we tessellate the
    convex hull. This is purely a visual representation --- it doesn't
    assume anything about the platform's actual physical shape.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    pts = attachments_world
    if pts.shape[0] < 3:
        return

    if pts.shape[0] == 3:
        faces = [pts]
    else:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            faces = [pts[s] for s in hull.simplices]
        except Exception:
            # Fall back to a fan triangulation if the hull is degenerate
            # (e.g., all points coplanar): pivot on point 0.
            faces = [pts[[0, i, i + 1]] for i in range(1, pts.shape[0] - 1)]

    coll = Poly3DCollection(faces, alpha=alpha, facecolor=CDPR_CABLE_COLORS[3],
                            edgecolor=CDPR_CABLE_COLORS[3])
    ax.add_collection3d(coll)


def _annotate_singularity(ax: "Axes3D", kappa: float) -> None:
    if not np.isfinite(kappa):
        text, color = r"$\kappa(\mathbf{W}) = \infty$  (singular)", "#D62728"
    elif kappa > 1e6:
        text, color = fr"$\kappa(\mathbf{{W}}) = {kappa:.2e}$  (near-singular)", "#D62728"
    elif kappa > 1e3:
        text, color = fr"$\kappa(\mathbf{{W}}) = {kappa:.1f}$  (ill-conditioned)", "#F0E442"
    else:
        text, color = fr"$\kappa(\mathbf{{W}}) = {kappa:.1f}$", "#1F77B4"
    ax.text2D(0.02, 0.96, text, transform=ax.transAxes,
              fontsize=8, color=color, verticalalignment="top",
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85, lw=0.5))


def _scene_scale(anchors: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(anchors.max(axis=0) - anchors.min(axis=0)))


def _set_equal_3d_box(ax: "Axes3D", anchors: NDArray[np.float64]) -> None:
    """Make the 3D box cubical so the geometry isn't distorted.

    Matplotlib's default 3D aspect is "auto", which stretches the cable
    rendering arbitrarily. We compute the largest extent across axes and
    set all three limits to that, centred on the anchor centroid.
    """
    lo = anchors.min(axis=0)
    hi = anchors.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.55 * float((hi - lo).max())
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:                                  # pragma: no cover
        pass
