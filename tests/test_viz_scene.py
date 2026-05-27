"""Smoke tests for 3D scene rendering, animation, and frame stepping."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.spatial.transform import Rotation

from cdpr.core.frames import Pose
from cdpr.viz.animation import FrameStepper, animate_playback, stream_from_result
from cdpr.viz.scene import SceneOptions, render_scene


def test_render_scene_default_options(ipanema, home_pose):
    fig = render_scene(ipanema, home_pose)
    assert isinstance(fig, Figure)
    # 3D axes have z; sanity-check by reading the label.
    assert fig.axes[0].get_zlabel() == r"$z$ [m]"


def test_render_scene_with_tension_heatmap(short_sim, ipanema):
    pose = Pose(position=short_sim.positions[-1],
                rotation=Rotation.from_quat(short_sim.quaternions_xyzw[-1]))
    fig = render_scene(
        ipanema, pose,
        options=SceneOptions(tension_heatmap=True),
        tensions=short_sim.cable_tensions[-1],
    )
    assert isinstance(fig, Figure)


def test_render_scene_rejects_mismatched_tensions(ipanema, home_pose):
    with pytest.raises(ValueError):
        render_scene(
            ipanema, home_pose,
            options=SceneOptions(tension_heatmap=True),
            tensions=np.zeros(3),         # ipanema has 8 cables
        )


def test_frame_stepper_advances(short_sim, ipanema):
    stepper = FrameStepper(short_sim, ipanema)
    fig0 = stepper.draw(0)
    assert isinstance(fig0, Figure)
    assert stepper.current_frame == 0
    stepper.step(1)
    assert stepper.current_frame == 1


def test_frame_stepper_out_of_range(short_sim, ipanema):
    stepper = FrameStepper(short_sim, ipanema)
    with pytest.raises(IndexError):
        stepper.draw(10_000)


def test_animate_playback_returns_animation(short_sim, ipanema):
    anim = animate_playback(short_sim, ipanema, fps=15, trail=5)
    # FuncAnimation has a frame_seq attribute we can probe without rendering.
    assert anim is not None
    assert anim._fig is not None


def test_stream_from_result_yields_one_sample_per_step(short_sim):
    samples = list(stream_from_result(short_sim))
    assert len(samples) == len(short_sim.time)
    assert samples[0].pose.position.shape == (3,)
