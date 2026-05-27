"""Reinforcement-learning wrappers around Stable-Baselines3.

Three thin factories (one per algorithm) build a configured SB3 agent
over a :class:`cdpr.learn.env.CDPREnv`. They share the
:class:`PhysicsLoggingCallback`, which records per-step physics
diagnostics (tracking error, condition number, infeasibility rate,
tension bounds usage) alongside the standard SB3 logger so a learning
curve is interpretable in CDPR terms, not just "episode reward went up".

Construction is lazy --- importing this package does not import SB3;
calling a factory does, and raises a helpful install hint if the extra
is missing.
"""

from cdpr.learn.rl.callbacks import PhysicsLoggingCallback
from cdpr.learn.rl.factories import make_ppo, make_sac, make_td3

__all__ = [
    "PhysicsLoggingCallback",
    "make_ppo",
    "make_sac",
    "make_td3",
]
