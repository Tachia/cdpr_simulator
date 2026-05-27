r"""Voxel-grid workspace sampling.

Maps a regular 3D grid of platform positions (at a fixed orientation) to a
boolean array telling whether each voxel lies in a chosen workspace type.
This is the workhorse used by the visualisation layer to render workspace
volumes and by quantitative dissertation studies comparing two robot
configurations or two cable models.

For studies of orientational workspaces, sweep this routine over a sequence
of fixed orientations rather than building a higher-dimensional grid --- the
combinatorics of 6-D sampling are unforgiving, and orientational sweeps are
what published results actually report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose, Wrench
from cdpr.geometry.robot import Robot
from cdpr.workspace.closure import is_in_wcw
from cdpr.workspace.feasible import is_in_wfw


@dataclass(slots=True)
class GridResult:
    """Container for a workspace scan over a regular position grid."""

    xs: NDArray[np.float64]                  # (Nx,)
    ys: NDArray[np.float64]                  # (Ny,)
    zs: NDArray[np.float64]                  # (Nz,)
    mask: NDArray[np.bool_]                  # (Nx, Ny, Nz)
    orientation: Rotation
    kind: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.mask.shape

    @property
    def n_inside(self) -> int:
        return int(self.mask.sum())

    @property
    def fraction_inside(self) -> float:
        return float(self.n_inside / self.mask.size)

    def voxel_volume(self) -> float:
        dx = self.xs[1] - self.xs[0] if len(self.xs) > 1 else 0.0
        dy = self.ys[1] - self.ys[0] if len(self.ys) > 1 else 0.0
        dz = self.zs[1] - self.zs[0] if len(self.zs) > 1 else 0.0
        return float(dx * dy * dz)

    def estimated_volume(self) -> float:
        """Riemann-sum estimate of the contained workspace volume."""
        return self.n_inside * self.voxel_volume()


def scan_translational_workspace(
    robot: Robot,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    resolution: int | tuple[int, int, int],
    *,
    kind: Literal["wcw", "wfw"] = "wcw",
    orientation: Rotation | None = None,
    external_wrench: Wrench | NDArray[np.float64] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> GridResult:
    """Scan a regular 3D position grid at a fixed orientation.

    Parameters
    ----------
    robot:
        Robot definition.
    xlim, ylim, zlim:
        Inclusive endpoints of the position grid along each world axis.
    resolution:
        Voxel count along each axis. Pass an int for a uniform grid or a
        tuple for axis-specific resolutions.
    kind:
        ``"wcw"`` (default) for wrench-closure or ``"wfw"`` for
        wrench-feasibility against ``external_wrench``.
    orientation:
        Fixed platform orientation across the grid. Defaults to identity.
    external_wrench:
        Required when ``kind == "wfw"``. Either a :class:`Wrench` or a raw
        ``dof``-vector.
    progress:
        Optional ``(i, total)`` callback called every plane sweep. Useful
        for long scans on large grids.
    """
    if isinstance(resolution, int):
        nx = ny = nz = resolution
    else:
        nx, ny, nz = resolution

    xs = np.linspace(*xlim, nx)
    ys = np.linspace(*ylim, ny)
    zs = np.linspace(*zlim, nz)
    R = orientation if orientation is not None else Rotation.identity()
    mask = np.zeros((nx, ny, nz), dtype=bool)

    if kind == "wfw" and external_wrench is None:
        raise ValueError("WFW scan requires an external_wrench argument.")

    total = nx
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                pose = Pose(position=np.array([x, y, z]), rotation=R)
                if kind == "wcw":
                    mask[i, j, k] = is_in_wcw(pose, robot)
                else:
                    mask[i, j, k] = is_in_wfw(pose, robot, external_wrench)  # type: ignore[arg-type]
        if progress is not None:
            progress(i + 1, total)

    return GridResult(xs=xs, ys=ys, zs=zs, mask=mask, orientation=R, kind=kind)
