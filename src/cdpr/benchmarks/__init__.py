"""Closed-loop benchmark harness across cdpr core and external backends.

Differs from :mod:`cdpr.learn.benchmark` (which targets supervised /
RL controller comparison inside the cdpr core only). The benchmark
here is intended for *physics verification* and *backend comparison*:
the same controller is asked to track the same reference, but the
underlying physics integrator is swapped between cdpr core and a
:class:`PhysicsBackend` adapter (currently MuJoCo).

Three nouns describe the workflow:

* :class:`Scenario` -- the full description of one experiment (robot,
  trajectory, controller, duration, dt, seed, initial pose offset).
* :class:`BenchmarkRun` -- the per-(scenario, backend) result with
  tracking, control-effort, tension, feasibility, and runtime metrics.
* :class:`BenchmarkSuite` -- a collection of scenarios and backends to
  run pairwise; :meth:`run` produces a list of :class:`BenchmarkRun`.

Determinism: passing ``seed`` propagates through any RNG-dependent
controllers (currently none in the analytical set, but the harness
seeds NumPy for downstream consumers).
"""

from cdpr.benchmarks.metrics import BenchmarkMetrics
from cdpr.benchmarks.scenario import Scenario, scenario_hash
from cdpr.benchmarks.suite import BenchmarkRun, BenchmarkSuite, run_scenario

__all__ = [
    "Scenario",
    "scenario_hash",
    "BenchmarkRun",
    "BenchmarkSuite",
    "BenchmarkMetrics",
    "run_scenario",
]
