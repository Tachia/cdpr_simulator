r"""Algorithm factories for PPO, SAC, TD3 on the CDPR environment.

Each factory wraps the corresponding SB3 class with the hyperparameters
known to behave reasonably on the CDPR task (continuous, bounded action
space; horizons in the 200-1000 step range; reward sparsity dominated by
infeasibility events). The defaults are sane starting points, not
research-tuned final values --- override via ``**kwargs``.

Why three separate factories instead of an algorithm-string dispatcher?
The three algorithms have meaningfully different config surfaces
(PPO has ``n_steps`` and ``gae_lambda``; SAC has ``ent_coef``; TD3 has
``policy_delay``). Hiding that behind one function obscures the choices
the researcher actually needs to make.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cdpr.learn._lazy import require_stable_baselines3

if TYPE_CHECKING:                                           # pragma: no cover
    from stable_baselines3.common.base_class import BaseAlgorithm
    from cdpr.learn.env import CDPREnv


def _wrap_env(env: "CDPREnv"):
    """SB3 expects a Gymnasium env; our :class:`CDPREnv` already is one, but
    Stable-Baselines3 1.x style wraps it in a :class:`DummyVecEnv`.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([lambda: env])


def make_ppo(
    env: "CDPREnv",
    *,
    learning_rate: float = 3e-4,
    n_steps: int = 1024,
    batch_size: int = 64,
    gae_lambda: float = 0.95,
    gamma: float = 0.99,
    ent_coef: float = 0.0,
    seed: int | None = None,
    tensorboard_log: str | None = None,
    **extra,
) -> "BaseAlgorithm":
    require_stable_baselines3()
    from stable_baselines3 import PPO
    return PPO(
        "MlpPolicy",
        _wrap_env(env),
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gae_lambda=gae_lambda,
        gamma=gamma,
        ent_coef=ent_coef,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=0,
        **extra,
    )


def make_sac(
    env: "CDPREnv",
    *,
    learning_rate: float = 3e-4,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    tau: float = 5e-3,
    gamma: float = 0.99,
    ent_coef: str | float = "auto",
    train_freq: int = 1,
    seed: int | None = None,
    tensorboard_log: str | None = None,
    **extra,
) -> "BaseAlgorithm":
    require_stable_baselines3()
    from stable_baselines3 import SAC
    return SAC(
        "MlpPolicy",
        _wrap_env(env),
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        ent_coef=ent_coef,
        train_freq=train_freq,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=0,
        **extra,
    )


def make_td3(
    env: "CDPREnv",
    *,
    learning_rate: float = 1e-3,
    buffer_size: int = 100_000,
    batch_size: int = 256,
    tau: float = 5e-3,
    gamma: float = 0.99,
    policy_delay: int = 2,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    seed: int | None = None,
    tensorboard_log: str | None = None,
    **extra,
) -> "BaseAlgorithm":
    require_stable_baselines3()
    from stable_baselines3 import TD3
    return TD3(
        "MlpPolicy",
        _wrap_env(env),
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        policy_delay=policy_delay,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
        seed=seed,
        tensorboard_log=tensorboard_log,
        verbose=0,
        **extra,
    )
