r"""Cross-mode cable-model diagnostics and comparison helpers.

Given the same robot, the same scenario, and a sequence of
:class:`CableModel` instances (one per mode), produce a directly
comparable diagnostic table over a trajectory. The Phase-7 directive's
"comparison artifact" requirement turns out to be exactly this: a small
DataFrame-shaped record that the dissertation appendix can quote in
prose ("at the apex, tension under SQCK exceeds Kelvin--Voigt by ...")
without re-running the constitutive evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:                                           # pragma: no cover
    from cdpr.cables.base import CableModel
    from cdpr.dynamics.rigid_body import PlatformState
    from cdpr.geometry.robot import Robot


@dataclass(slots=True)
class ModeDiagnostics:
    """Per-time-step trace of one mode's diagnostics."""

    mode_name: str
    time: NDArray[np.float64]
    tension_mean: NDArray[np.float64]
    tension_max: NDArray[np.float64]
    tension_min: NDArray[np.float64]
    n_slack: NDArray[np.int_]
    extras: dict[str, NDArray[np.float64]] = field(default_factory=dict)


@dataclass(slots=True)
class ComparisonReport:
    """Side-by-side per-mode summary across a trajectory."""

    modes: dict[str, ModeDiagnostics]

    def summary(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, d in self.modes.items():
            out[name] = {
                "tension_mean": float(d.tension_mean.mean()),
                "tension_peak": float(d.tension_max.max()),
                "tension_floor": float(d.tension_min.min()),
                "slack_steps": int(d.n_slack.sum()),
                "n_steps": int(d.time.shape[0]),
            }
        return out


def sweep_modes(
    models: dict[str, "CableModel"],
    robot: "Robot",
    states: list["PlatformState"],
    rest_lengths_history: NDArray[np.float64],     # (T, m)
    times: NDArray[np.float64],
) -> ComparisonReport:
    r"""Evaluate every model at every step; return a per-mode trace.

    ``models`` is a dict keyed by mode name (typically the three
    canonical names). ``states`` is the platform trajectory the
    comparison is computed *against* --- in the canonical workflow it
    comes from one run, so every mode reads the same state and only the
    *constitutive* output differs.

    Use this routine for cross-mode sensitivity analysis when the goal
    is "given that the platform moves like this, what tension does each
    constitutive law predict?". It does **not** re-simulate the
    dynamics per mode --- for that, run separate scenarios.
    """
    n_steps = len(states)
    if rest_lengths_history.shape != (n_steps, robot.n_cables):
        raise ValueError(
            f"rest_lengths_history must be shape ({n_steps}, {robot.n_cables}); "
            f"got {rest_lengths_history.shape}"
        )

    diagnostics: dict[str, ModeDiagnostics] = {}
    for name, model in models.items():
        mean_arr = np.zeros(n_steps)
        max_arr = np.zeros(n_steps)
        min_arr = np.zeros(n_steps)
        slack_arr = np.zeros(n_steps, dtype=int)
        extras: dict[str, list[float]] = {}
        for k in range(n_steps):
            d = model.diagnostics(robot, states[k], rest_lengths_history[k])
            mean_arr[k] = float(d["tension_mean"])
            max_arr[k] = float(d["tension_max"])
            min_arr[k] = float(d["tension_min"])
            slack_arr[k] = int(d["n_slack"])
            for key, value in d.items():
                if key in {"mode", "tension_mean", "tension_max", "tension_min",
                           "n_slack", "n_cables"}:
                    continue
                extras.setdefault(key, []).append(float(value))
        diagnostics[name] = ModeDiagnostics(
            mode_name=name,
            time=times,
            tension_mean=mean_arr,
            tension_max=max_arr,
            tension_min=min_arr,
            n_slack=slack_arr,
            extras={k: np.asarray(v, dtype=np.float64) for k, v in extras.items()},
        )

    return ComparisonReport(modes=diagnostics)
