"""Workspace renderer smoke tests."""

from __future__ import annotations

from matplotlib.figure import Figure

from cdpr.viz.plots2d import plot_workspace_slice
from cdpr.viz.workspace import render_workspace_scatter
from cdpr.workspace.grid import scan_translational_workspace


def test_render_workspace_scatter(point_mass_robot):
    grid = scan_translational_workspace(
        point_mass_robot,
        xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5),
        resolution=5, kind="wcw",
    )
    fig = render_workspace_scatter(grid)
    assert isinstance(fig, Figure)


def test_plot_workspace_slice(point_mass_robot):
    grid = scan_translational_workspace(
        point_mass_robot,
        xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5),
        resolution=5, kind="wcw",
    )
    fig = plot_workspace_slice(grid, axis="z", index=2)
    assert isinstance(fig, Figure)
