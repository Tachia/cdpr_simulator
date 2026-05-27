"""Hybrid physics + data-driven learning layer.

Three workflows live here, intentionally kept compositional rather than
unified:

* **Reinforcement learning.** A Gymnasium-compatible :class:`CDPREnv` wraps
  the Phase-1 simulator and exposes the standard reset / step / reward
  contract; :mod:`cdpr.learn.rl` provides thin Stable-Baselines3 adapters
  for PPO / SAC / TD3 that share a logging callback recording
  per-step physics diagnostics (condition number, infeasibility) alongside
  the usual reward / loss curves.

* **Supervised learning.** :class:`cdpr.learn.datasets.TrajectoryDataset`
  turns a :class:`SimulationResult` or :class:`IngestedExperiment` into a
  PyTorch ``Dataset`` of state-target pairs, and :mod:`cdpr.learn.models`
  ships an MLP inverse-dynamics regressor as the canonical baseline.

* **Physics-informed neural networks.** :class:`cdpr.learn.models.pinn.InverseDynamicsPINN`
  layers a Newton-Euler residual on top of the supervised loss; the
  framework's structure matrix evaluates the physics term so the network
  is *always* anchored to the geometric model it is learning to amend.

Interpretability is a hard requirement. Every model exposes
``predict_with_components`` that returns the prediction *plus* its data /
physics residuals, and the training loop logs each loss component
separately. The :mod:`cdpr.learn.benchmark` harness then runs analytic,
PD, computed-torque, and learned controllers on the same task and emits
a side-by-side comparison.

Optional dependencies:
* ``pip install 'cdpr[learn]'`` -- PyTorch + Gymnasium for supervised / PINN /
  env construction.
* ``pip install 'cdpr[rl]'`` -- adds Stable-Baselines3 on top.
"""

from cdpr.learn._lazy import (
    require_gymnasium,
    require_stable_baselines3,
    require_torch,
)
from cdpr.learn.rewards import (
    ActionSmoothness,
    InfeasibilityPenalty,
    PoseTracking,
    RewardSum,
    TensionCost,
    VelocityTracking,
)

__all__ = [
    "require_torch",
    "require_gymnasium",
    "require_stable_baselines3",
    "PoseTracking",
    "VelocityTracking",
    "ActionSmoothness",
    "TensionCost",
    "InfeasibilityPenalty",
    "RewardSum",
]
