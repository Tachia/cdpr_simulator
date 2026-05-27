"""Quasi-static analysis: wrench feasibility and tension distribution.

The two top-level entry points are:

* :func:`tension_distribution` -- the workhorse: solve the bounded
  equilibrium QP for one pose, returning a feasible tension vector or
  raising :class:`~cdpr.core.exceptions.InfeasibleTensionError`.
* :func:`is_wrench_feasible` -- a boolean test used by workspace analysis,
  cheap because it short-circuits on the LP feasibility step.
"""

from cdpr.statics.tension import (
    TensionObjective,
    is_wrench_feasible,
    min_norm_tension,
    tension_distribution,
)

__all__ = [
    "tension_distribution",
    "is_wrench_feasible",
    "min_norm_tension",
    "TensionObjective",
]
