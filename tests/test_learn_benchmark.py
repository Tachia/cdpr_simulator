"""Benchmark harness: PD beats open-loop, CT beats PD, all metrics finite."""

from __future__ import annotations

import numpy as np

from cdpr.control import ComputedTorqueController, PDController
from cdpr.learn.benchmark import Benchmark
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def test_benchmark_runs_three_controllers(ipanema):
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.1, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=0.5),
    )
    bench = Benchmark(robot=ipanema, trajectory=traj, duration=0.5, dt=2e-3)
    controllers = {
        "open_loop": None,
        "pd": PDController(Kp_pos=2000.0, Kd_pos=200.0, Kp_rot=50.0, Kd_rot=5.0),
        "computed_torque": ComputedTorqueController(
            Kp_pos=900.0, Kd_pos=60.0, Kp_rot=900.0, Kd_rot=60.0,
        ),
    }
    report = bench.run(controllers, keep_results=False)
    assert len(report.outcomes) == 3
    summary = {o.name: o for o in report.outcomes}
    # Open-loop has the worst tracking; CT has the best.
    assert summary["open_loop"].mean_position_error > summary["pd"].mean_position_error
    assert summary["pd"].mean_position_error > summary["computed_torque"].mean_position_error
    # CT mean error on a 10 cm circle should be sub-millimetre with these gains.
    assert summary["computed_torque"].mean_position_error < 1e-3


def test_benchmark_to_dict_serialisable(ipanema):
    from cdpr.control import PDController
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.05, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=0.2),
    )
    bench = Benchmark(robot=ipanema, trajectory=traj, duration=0.2, dt=2e-3)
    report = bench.run({"pd": PDController(Kp_pos=2000.0, Kd_pos=200.0,
                                           Kp_rot=50.0, Kd_rot=5.0)})
    d = report.to_dict()
    assert isinstance(d, list)
    assert "mean_position_error" in d[0]
