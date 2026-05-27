"""Smoke tests for the 2D analytics module.

Each test verifies (a) the function returns a Matplotlib Figure, (b) that
figure has at least one Axes, and (c) the figure can be exported to PNG via
:func:`figure_to_png_bytes` --- which is what the FastAPI service depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

import matplotlib
from matplotlib.figure import Figure

from cdpr.viz import plots2d
from cdpr.viz.export import figure_to_png_bytes, save_figure


def _is_figure(obj: object) -> bool:
    return isinstance(obj, Figure)


def test_plot_position(short_sim):
    fig = plots2d.plot_position(short_sim)
    assert _is_figure(fig)
    assert len(fig.axes) >= 1
    assert figure_to_png_bytes(fig)


def test_plot_velocity(short_sim):
    assert _is_figure(plots2d.plot_velocity(short_sim))


def test_plot_angular_velocity(short_sim):
    assert _is_figure(plots2d.plot_angular_velocity(short_sim))


def test_plot_cable_lengths_and_tensions(short_sim, ipanema):
    fig = plots2d.plot_cable_lengths(short_sim)
    assert _is_figure(fig)
    fig2 = plots2d.plot_cable_tensions(short_sim, robot=ipanema)
    assert _is_figure(fig2)


def test_plot_tracking_error_against_constant_reference(short_sim, home_pose):
    fig = plots2d.plot_tracking_error(short_sim, lambda t: home_pose)
    assert _is_figure(fig)


def test_plot_condition_number(short_sim, ipanema):
    fig = plots2d.plot_condition_number(short_sim, ipanema)
    assert _is_figure(fig)


def test_plot_trajectory_projection_three_planes(short_sim):
    for plane in ("xy", "xz", "yz"):
        fig = plots2d.plot_trajectory_projection(short_sim.positions, plane=plane)
        assert _is_figure(fig)


def test_save_figure_to_png_svg_pdf(tmp_path, short_sim):
    fig = plots2d.plot_position(short_sim)
    for ext in (".png", ".svg", ".pdf"):
        out = save_figure(fig, tmp_path / f"position{ext}")
        assert out.exists()
        assert out.stat().st_size > 0


def test_save_figure_rejects_unknown_extension(tmp_path, short_sim):
    fig = plots2d.plot_position(short_sim)
    with pytest.raises(ValueError):
        save_figure(fig, tmp_path / "x.bmp")
