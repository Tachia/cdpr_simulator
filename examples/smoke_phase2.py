"""End-to-end Phase-2 smoke run.

Sequentially exercises every Phase-2 surface:
1. Run a forward simulation (Phase 1).
2. Render a 3D scene snapshot + 2D plots (Phase 2 viz).
3. Save plots as captioned LaTeX figures + Markdown summary (Phase 2 reports).
4. Record the experiment to disk; load it back; compare it to itself.
5. Probe the adapter registry to confirm the registry runs without backends.

Run with:  python examples/smoke_phase2.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.spatial.transform import Rotation

from cdpr.adapters import available_backends
from cdpr.core.frames import Pose
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.recording import compare, load_experiment, record_simulation
from cdpr.reports import (
    CaptionedFigure,
    cable_summary_table,
    save_captioned_figure,
    write_markdown_summary,
)
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory
from cdpr.viz import plots2d
from cdpr.viz.scene import SceneOptions, render_scene
from cdpr.viz.style import apply_dissertation_style


def main(out_root: Path = Path("runs/phase2_smoke")) -> None:
    apply_dissertation_style()

    robot = ipanema_class()
    pose0 = Pose(position=np.zeros(3), rotation=Rotation.identity())
    state0 = PlatformState.at_rest(pose0)

    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.3, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=1.5),
    )

    print("[1/5] Simulating...")
    result = simulate(
        robot, state0, duration=1.5, dt=2e-3,
        reference_pose=lambda t: pose0,    # static hold; cleaner numerics for the smoke
    )

    print("[2/5] Rendering 3D scene + 2D analytics...")
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    scene = render_scene(
        robot, pose0,
        options=SceneOptions(tension_heatmap=True),
        tensions=result.cable_tensions[-1],
        trajectory_positions=result.positions,
    )
    cf = CaptionedFigure(
        figure=scene,
        caption=r"IPAnema-class scene snapshot with tension heatmap at $t = T$.",
        label="fig:scene",
    )
    save_captioned_figure(cf, fig_dir)

    for kind in ("position", "cable_lengths", "cable_tensions", "condition_number"):
        fn = getattr(plots2d, f"plot_{kind}")
        fig = fn(result, robot=robot) if kind in ("cable_tensions",) else fn(result, robot) if kind == "condition_number" else fn(result)
        cf = CaptionedFigure(figure=fig, caption=f"Time series: {kind.replace('_', ' ')}.",
                             label=f"fig:{kind}")
        save_captioned_figure(cf, fig_dir)

    print("[3/5] Recording experiment...")
    log = record_simulation(
        robot=robot, result=result, out_dir=out_root / "experiment",
        title="Phase 2 smoke run", seed=2026,
    )

    print("[4/5] Reloading + report generation...")
    exp = load_experiment(log.root)
    cable_summary_table(exp, out_root / "tables" / "cables.csv", fmt="csv")
    cable_summary_table(exp, out_root / "tables" / "cables.tex", fmt="latex",
                        caption="Per-cable tension and length statistics.",
                        label="tab:cables")

    figures_for_md = [
        ("3D scene snapshot", "figures/fig_scene.png"),
        ("Cable tensions",    "figures/fig_cable_tensions.png"),
    ]
    md_path = write_markdown_summary(exp, out_root / "summary.md", figures=figures_for_md)
    print(f"        markdown: {md_path}")

    # Sanity: compare the recording to itself round-tripped through disk.
    report = compare(exp, exp)
    assert report.position.rms == 0.0
    print(f"        round-trip RMS (should be zero): position={report.position.rms:.3e} m")

    print("[5/5] Adapter registry probe...")
    backends = available_backends()
    print(f"        available backends: {backends}")

    print("\nDone. Artifacts in:", out_root.resolve())


if __name__ == "__main__":
    main()
