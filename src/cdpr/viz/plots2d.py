r"""Two-dimensional scientific plots derived from Phase-1 outputs.

Every function in this module takes a :class:`cdpr.dynamics.simulator.SimulationResult`
(or another Phase-1 data object) and returns a Matplotlib :class:`Figure`.
The figures are styled by the active preset in :mod:`cdpr.viz.style` --- the
caller is expected to wrap calls in :func:`cdpr.viz.style.styled` when they
need a non-default look, rather than us mutating ``rcParams`` here.

Composition pattern: pass ``ax=None`` (default) to get a fresh figure; pass an
existing :class:`Axes` to draw into a sub-panel of a multi-figure layout.

Export: each call site can use :func:`cdpr.viz.export.save_figure`, or just
``fig.savefig(path)`` --- the publication style takes care of dpi and
margins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cdpr.viz._lazy import require_matplotlib
from cdpr.viz.style import CDPR_CABLE_COLORS

if TYPE_CHECKING:                                           # pragma: no cover
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from cdpr.dynamics.simulator import SimulationResult
    from cdpr.geometry.robot import Robot
    from cdpr.workspace.grid import GridResult


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _axes(ax: "Axes | None" = None, **figkw: object) -> tuple["Figure", "Axes"]:
    """Resolve an axis: return ``(fig, ax)`` whether the caller passed one or not."""
    require_matplotlib()
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(**figkw)
        return fig, ax
    return ax.figure, ax


def _cable_color(i: int) -> str:
    return CDPR_CABLE_COLORS[i % len(CDPR_CABLE_COLORS)]


# ---------------------------------------------------------------------------
# Time series of platform state
# ---------------------------------------------------------------------------

def plot_position(result: "SimulationResult", *, ax: "Axes | None" = None) -> "Figure":
    """Platform position components vs time."""
    fig, ax = _axes(ax, figsize=(6.0, 3.2))
    ax.plot(result.time, result.positions[:, 0], label=r"$x$", color=CDPR_CABLE_COLORS[0])
    ax.plot(result.time, result.positions[:, 1], label=r"$y$", color=CDPR_CABLE_COLORS[1])
    ax.plot(result.time, result.positions[:, 2], label=r"$z$", color=CDPR_CABLE_COLORS[2])
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"position [m]")
    ax.legend(loc="best")
    return fig


def plot_velocity(result: "SimulationResult", *, ax: "Axes | None" = None) -> "Figure":
    """Platform translational velocity components vs time."""
    fig, ax = _axes(ax, figsize=(6.0, 3.2))
    for i, (lbl, col) in enumerate(zip(("v_x", "v_y", "v_z"), CDPR_CABLE_COLORS, strict=False)):
        ax.plot(result.time, result.linear_velocities[:, i], label=fr"${lbl}$", color=col)
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"velocity [m s$^{-1}$]")
    ax.legend(loc="best")
    return fig


def plot_angular_velocity(result: "SimulationResult", *, ax: "Axes | None" = None) -> "Figure":
    """Platform angular velocity components vs time."""
    fig, ax = _axes(ax, figsize=(6.0, 3.2))
    for i, (lbl, col) in enumerate(zip(
        (r"\omega_x", r"\omega_y", r"\omega_z"), CDPR_CABLE_COLORS, strict=False
    )):
        ax.plot(result.time, result.angular_velocities[:, i], label=fr"${lbl}$", color=col)
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"angular velocity [rad s$^{-1}$]")
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------------------
# Cable-level time series
# ---------------------------------------------------------------------------

def plot_cable_lengths(result: "SimulationResult", *, ax: "Axes | None" = None) -> "Figure":
    """Per-cable length vs time. One line per cable."""
    fig, ax = _axes(ax, figsize=(6.0, 3.5))
    m = result.cable_lengths.shape[1]
    for i in range(m):
        ax.plot(result.time, result.cable_lengths[:, i],
                label=fr"$\ell_{{{i + 1}}}$", color=_cable_color(i))
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"cable length $\ell_i$ [m]")
    if m <= 8:
        ax.legend(ncols=min(m, 4), loc="best", frameon=False)
    return fig


def plot_cable_tensions(
    result: "SimulationResult",
    *,
    robot: "Robot | None" = None,
    show_bounds: bool = True,
    ax: "Axes | None" = None,
) -> "Figure":
    r"""Per-cable tension vs time.

    When ``robot`` is provided and it carries :class:`CableLimits`, dashed
    horizontal lines mark :math:`t_\min` and :math:`t_\max` --- the standard
    "did we hit a bound" diagnostic for tension-distribution traces.
    """
    fig, ax = _axes(ax, figsize=(6.0, 3.5))
    m = result.cable_tensions.shape[1]
    for i in range(m):
        ax.plot(result.time, result.cable_tensions[:, i],
                label=fr"$T_{{{i + 1}}}$", color=_cable_color(i))
    if show_bounds and robot is not None and robot.limits is not None:
        ax.axhline(float(robot.limits.t_min.max()), linestyle="--", color="0.4", linewidth=0.6,
                   label=r"$T_\mathrm{min}$")
        ax.axhline(float(robot.limits.t_max.min()), linestyle="--", color="0.4", linewidth=0.6,
                   label=r"$T_\mathrm{max}$")
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"cable tension $T_i$ [N]")
    if m <= 8:
        ax.legend(ncols=min(m, 4), loc="best", frameon=False)
    return fig


# ---------------------------------------------------------------------------
# Tracking and conditioning diagnostics
# ---------------------------------------------------------------------------

def plot_tracking_error(
    result: "SimulationResult",
    reference: Callable[[float], "object"],
    *,
    ax: "Axes | None" = None,
) -> "Figure":
    r"""Euclidean position tracking error :math:`\lVert \mathbf{p}(t) - \mathbf{p}_\text{ref}(t) \rVert`.

    ``reference`` is any callable :math:`t \mapsto \text{Pose}` --- a
    :class:`cdpr.trajectory.trajectory.Trajectory` is the typical choice, but
    a recorded experimental log indexed by time also works.
    """
    fig, ax = _axes(ax, figsize=(6.0, 3.0))
    ref_positions = np.array([reference(t).position for t in result.time])
    err = np.linalg.norm(result.positions - ref_positions, axis=1)
    ax.plot(result.time, err, color=CDPR_CABLE_COLORS[3])
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"$\lVert \mathbf{p}(t) - \mathbf{p}_\mathrm{ref}(t) \rVert$ [m]")
    ax.set_yscale("log")
    return fig


def plot_condition_number(
    result: "SimulationResult",
    robot: "Robot",
    *,
    ax: "Axes | None" = None,
) -> "Figure":
    r"""Structure-matrix condition number :math:`\kappa_2(\mathbf{W})` vs time.

    Spikes flag transient proximity to a singular configuration. Computed
    per-frame by re-running :func:`cdpr.kinematics.jacobian.structure_matrix`
    on the recorded pose history, which is why this function takes the robot.
    """
    from scipy.spatial.transform import Rotation
    from cdpr.core.frames import Pose
    from cdpr.kinematics.jacobian import structure_matrix

    fig, ax = _axes(ax, figsize=(6.0, 3.0))
    n = len(result.time)
    kappa = np.empty(n)
    for k in range(n):
        pose = Pose(position=result.positions[k],
                    rotation=Rotation.from_quat(result.quaternions_xyzw[k]))
        W = structure_matrix(pose, robot)
        s = np.linalg.svd(W, compute_uv=False)
        kappa[k] = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    ax.plot(result.time, kappa, color=CDPR_CABLE_COLORS[5])
    ax.set_xlabel(r"time $t$ [s]")
    ax.set_ylabel(r"condition number $\kappa_2(\mathbf{W})$")
    ax.set_yscale("log")
    return fig


# ---------------------------------------------------------------------------
# 2D projections of 3D trajectories
# ---------------------------------------------------------------------------

def plot_trajectory_projection(
    positions: ArrayLike,
    *,
    plane: str = "xy",
    reference: ArrayLike | None = None,
    ax: "Axes | None" = None,
) -> "Figure":
    """Project a 3D trajectory onto one of the principal planes."""
    fig, ax = _axes(ax, figsize=(4.5, 4.5))
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3); got {p.shape}")
    axes_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    i, j = axes_map[plane]
    ax.plot(p[:, i], p[:, j], color=CDPR_CABLE_COLORS[0], label="actual")
    if reference is not None:
        r = np.asarray(reference, dtype=np.float64)
        ax.plot(r[:, i], r[:, j], color=CDPR_CABLE_COLORS[1],
                linestyle="--", label="reference")
        ax.legend()
    ax.set_xlabel(f"${plane[0]}$ [m]")
    ax.set_ylabel(f"${plane[1]}$ [m]")
    ax.set_aspect("equal", adjustable="datalim")
    return fig


# ---------------------------------------------------------------------------
# Workspace slice
# ---------------------------------------------------------------------------

def plot_workspace_slice(
    grid: "GridResult",
    *,
    axis: str = "z",
    index: int = 0,
    ax: "Axes | None" = None,
) -> "Figure":
    """Render a 2D slice of a :class:`GridResult` as an image."""
    fig, ax = _axes(ax, figsize=(4.5, 4.5))
    coord_axes = {"x": (grid.ys, grid.zs, 0), "y": (grid.xs, grid.zs, 1), "z": (grid.xs, grid.ys, 2)}
    h_axis, v_axis, slice_dim = coord_axes[axis]
    if slice_dim == 0:
        sl = grid.mask[index, :, :]
    elif slice_dim == 1:
        sl = grid.mask[:, index, :]
    else:
        sl = grid.mask[:, :, index]
    ax.imshow(
        sl.T,
        origin="lower",
        extent=(h_axis[0], h_axis[-1], v_axis[0], v_axis[-1]),
        aspect="equal",
        cmap="Greens",
        alpha=0.7,
        interpolation="nearest",
    )
    horiz_lbl, vert_lbl = [c for c in ("x", "y", "z") if c != axis]
    ax.set_xlabel(f"${horiz_lbl}$ [m]")
    ax.set_ylabel(f"${vert_lbl}$ [m]")
    ax.set_title(f"{grid.kind.upper()} slice at {axis} = "
                 f"{[grid.xs, grid.ys, grid.zs][slice_dim][index]:.3f} m")
    return fig
