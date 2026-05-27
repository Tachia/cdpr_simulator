r"""Animation of the CDPR scene.

Three modes match the Phase-2 directive:

* **Playback** -- :func:`animate_playback` consumes a
  :class:`~cdpr.dynamics.simulator.SimulationResult` and returns a
  :class:`matplotlib.animation.FuncAnimation` ready to display or export.
* **Frame stepping** -- :class:`FrameStepper` redraws a chosen frame on demand,
  useful for debug inspection (Jupyter or interactive scripts).
* **Live streaming** -- :class:`LiveAnimator` accepts incremental state updates
  via :meth:`LiveAnimator.push` and refreshes the figure each call. Wire it
  into a custom integration loop that yields states.

Export goes through :func:`save_animation`, which dispatches MP4 (requires
the user's system ``ffmpeg``) or GIF (uses Pillow, already in the ``viz``
extra) from the file extension.

Decoupling: the animation routines never call into ``cdpr.dynamics``. They
either consume an already-computed :class:`SimulationResult` or accept
hand-fed states. The integration loop and the visualisation never cross.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from cdpr.viz._lazy import require_matplotlib
from cdpr.viz.scene import SceneOptions, render_scene

if TYPE_CHECKING:                                           # pragma: no cover
    from matplotlib.animation import FuncAnimation
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from cdpr.core.frames import Pose
    from cdpr.dynamics.simulator import SimulationResult, StreamStep
    from cdpr.geometry.robot import Robot


# ---------------------------------------------------------------------------
# Helpers shared across the three modes
# ---------------------------------------------------------------------------

def _pose_at(result: "SimulationResult", k: int) -> "Pose":
    """Reconstruct a Pose object from the ``k``-th sample of a result."""
    from cdpr.core.frames import Pose
    return Pose(
        position=result.positions[k],
        rotation=Rotation.from_quat(result.quaternions_xyzw[k]),
    )


def _make_axes() -> tuple["Figure", "Axes3D"]:
    require_matplotlib()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6.0, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    return fig, ax


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def animate_playback(
    result: "SimulationResult",
    robot: "Robot",
    *,
    fps: int = 30,
    trail: int = 100,
    options: SceneOptions | None = None,
) -> "FuncAnimation":
    r"""Build a Matplotlib animation of a simulated trajectory.

    Parameters
    ----------
    result, robot:
        Phase-1 simulation output and the robot it ran on.
    fps:
        Output frame rate. The number of animation frames equals the
        number of samples in ``result`` --- ``dt_sim`` controls realism,
        ``fps`` controls playback speed of the saved file.
    trail:
        Number of past positions kept as a fading trajectory ribbon. Set
        to 0 to disable; set to a large value to show the full path.
    options:
        Forwarded to :func:`render_scene`. The animation will reuse these
        options on every frame (no per-frame mutation), so tweaks like
        ``tension_heatmap=True`` apply uniformly.
    """
    require_matplotlib()
    import matplotlib.animation as manim

    opts = options or SceneOptions()
    fig, ax = _make_axes()

    def draw_frame(k: int) -> tuple[object, ...]:
        ax.cla()
        pose = _pose_at(result, k)
        trail_lo = max(0, k - trail) if trail else k
        traj_slice = result.positions[trail_lo : k + 1] if trail else None
        tensions = result.cable_tensions[k] if opts.tension_heatmap else None
        render_scene(
            robot, pose,
            options=opts,
            tensions=tensions,
            trajectory_positions=traj_slice,
            ax=ax,
        )
        ax.set_title(f"t = {result.time[k]:.3f} s")
        return ()

    anim = manim.FuncAnimation(
        fig,
        draw_frame,
        frames=len(result.time),
        interval=1000.0 / fps,
        blit=False,
        repeat=False,
    )
    return anim


def save_animation(anim: "FuncAnimation", path: str | Path, *, fps: int = 30) -> Path:
    """Save an animation as MP4 (ffmpeg) or GIF (Pillow), inferred from extension."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suffix = p.suffix.lower()
    if suffix == ".mp4":
        import matplotlib.animation as manim
        if not manim.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "MP4 export needs ffmpeg on PATH. Install ffmpeg or save to .gif instead."
            )
        anim.save(p, writer="ffmpeg", fps=fps)
    elif suffix == ".gif":
        # Pillow ships with the [viz] extra; no system binary needed.
        anim.save(p, writer="pillow", fps=fps)
    else:
        raise ValueError(f"Unsupported animation extension {suffix!r}; use .mp4 or .gif")
    return p


# ---------------------------------------------------------------------------
# Frame stepping
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FrameStepper:
    """Stateful renderer for scientific frame-by-frame inspection.

    Usage::

        stepper = FrameStepper(result, robot)
        stepper.draw(120)        # render the 120th frame
        stepper.draw(121)        # advance one frame
        stepper.export("frame.png")
    """

    result: "SimulationResult"
    robot: "Robot"
    options: SceneOptions = field(default_factory=lambda: SceneOptions(tension_heatmap=True))
    _fig: "Figure | None" = None
    _ax: "Axes3D | None" = None
    current_frame: int = 0

    def _ensure_axes(self) -> None:
        if self._fig is None or self._ax is None:
            self._fig, self._ax = _make_axes()

    def draw(self, frame: int) -> "Figure":
        if not 0 <= frame < len(self.result.time):
            raise IndexError(
                f"frame {frame} out of range [0, {len(self.result.time)})"
            )
        self._ensure_axes()
        assert self._ax is not None and self._fig is not None
        self._ax.cla()
        pose = _pose_at(self.result, frame)
        tensions = self.result.cable_tensions[frame] if self.options.tension_heatmap else None
        render_scene(
            self.robot, pose,
            options=self.options,
            tensions=tensions,
            ax=self._ax,
        )
        self._ax.set_title(f"frame {frame}  /  t = {self.result.time[frame]:.3f} s")
        self.current_frame = frame
        return self._fig

    def step(self, delta: int = 1) -> "Figure":
        return self.draw(self.current_frame + delta)

    def export(self, path: str | Path, **savefig_kwargs: object) -> Path:
        from cdpr.viz.export import save_figure
        if self._fig is None:
            raise RuntimeError("Call draw() before export().")
        return save_figure(self._fig, path, **savefig_kwargs)


# ---------------------------------------------------------------------------
# Live streaming
# ---------------------------------------------------------------------------

class LiveAnimator:
    """Incrementally update a 3D scene as :class:`StreamStep` samples arrive.

    Designed for use with :func:`cdpr.dynamics.iter_simulation` --- pull a
    step, ``.push()`` it, repeat. The animator keeps the figure alive,
    accumulates a fading trajectory ribbon, and redraws on every push.
    Headless use is fine: nothing tries to call ``plt.show`` for you, so
    :meth:`snapshot` works in CI just as in a desktop session.

    Accepts any object that exposes ``.time``, ``.pose`` (or ``.state.pose``),
    and ``.cable_tensions`` --- so both :class:`StreamStep` from the
    simulator and ad-hoc tuples-of-fields work.
    """

    def __init__(
        self,
        robot: "Robot",
        *,
        options: SceneOptions | None = None,
        trail: int = 200,
    ) -> None:
        self.robot = robot
        self.options = options or SceneOptions(tension_heatmap=True)
        self.trail = trail
        self._fig, self._ax = _make_axes()
        self._trajectory: list[NDArray[np.float64]] = []
        self._last_time: float | None = None

    @property
    def figure(self) -> "Figure":
        return self._fig

    def push(self, sample: object) -> None:
        pose = _coerce_pose(sample)
        tensions = getattr(sample, "cable_tensions", None)
        if tensions is None:
            tensions = getattr(sample, "tensions", None)
        time_val = float(getattr(sample, "time", 0.0))

        self._trajectory.append(np.asarray(pose.position, dtype=np.float64))
        if len(self._trajectory) > self.trail and self.trail > 0:
            self._trajectory = self._trajectory[-self.trail:]
        self._last_time = time_val

        self._ax.cla()
        render_scene(
            self.robot, pose,
            options=self.options,
            tensions=tensions,
            trajectory_positions=np.array(self._trajectory),
            ax=self._ax,
        )
        self._ax.set_title(f"t = {time_val:.3f} s")

    def snapshot(self, path: str | Path, **savefig_kwargs: object) -> Path:
        from cdpr.viz.export import save_figure
        return save_figure(self._fig, path, **savefig_kwargs)


def _coerce_pose(sample: object) -> "Pose":
    """Read a Pose out of either a StreamStep or a plain object with ``.pose``."""
    pose = getattr(sample, "pose", None)
    if pose is not None:
        return pose
    state = getattr(sample, "state", None)
    if state is not None and hasattr(state, "pose"):
        return state.pose
    raise TypeError(
        f"Sample {type(sample).__name__!r} has neither .pose nor .state.pose."
    )


# ---------------------------------------------------------------------------
# Adapter: turn a generator into a live animation file
# ---------------------------------------------------------------------------

def record_live_to_file(
    robot: "Robot",
    state_stream: "Iterator[StreamStep]",
    output_path: str | Path,
    *,
    fps: int = 30,
    options: SceneOptions | None = None,
) -> Path:
    """Drive a :class:`LiveAnimator` from a :class:`StreamStep` iterator and save the result.

    The generator pattern lets the caller integrate physics in their own
    loop (e.g. with a hardware-in-the-loop bridge) and still hand off the
    rendering / encoding to the visualisation layer.
    """
    require_matplotlib()
    import matplotlib.animation as manim
    samples = list(state_stream)
    if not samples:
        raise ValueError("state_stream yielded no samples")

    animator = LiveAnimator(robot, options=options, trail=len(samples))

    def draw_frame(k: int) -> tuple[object, ...]:
        animator.push(samples[k])
        return ()

    anim = manim.FuncAnimation(
        animator.figure, draw_frame, frames=len(samples), interval=1000.0 / fps, blit=False,
    )
    return save_animation(anim, output_path, fps=fps)


# ---------------------------------------------------------------------------
# Convenience iterator for SimulationResult -> StreamStep
# ---------------------------------------------------------------------------

def stream_from_result(result: "SimulationResult") -> "Iterator[StreamStep]":
    """Turn a batch :class:`SimulationResult` into a stream of :class:`StreamStep`.

    Used in tests, and as a reference example of how a custom integrator
    should yield samples to the visualisation layer.
    """
    from cdpr.core.frames import Twist
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.dynamics.simulator import StreamStep

    for k in range(len(result.time)):
        state = PlatformState(
            pose=_pose_at(result, k),
            velocity=Twist.from_parts(
                result.linear_velocities[k], result.angular_velocities[k]
            ),
        )
        yield StreamStep(
            step=k,
            time=float(result.time[k]),
            state=state,
            cable_tensions=result.cable_tensions[k].copy(),
            cable_lengths=result.cable_lengths[k].copy(),
            infeasible=(k in result.infeasible_steps),
        )
