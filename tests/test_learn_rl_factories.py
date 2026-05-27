"""SB3 factory smoke test: can we build a PPO over the CDPR env?

Avoids actually training (which would take minutes); we just verify the
construction path is sound, the environment is compatible, and a single
predict() call returns an action of the right shape.
"""

from __future__ import annotations

import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("gymnasium")

from cdpr.learn.env import CDPREnv, CDPREnvConfig
from cdpr.learn.rl import make_ppo


def test_make_ppo_builds_and_predicts(ipanema):
    env = CDPREnv(ipanema, config=CDPREnvConfig(horizon=8))
    agent = make_ppo(env, n_steps=8, batch_size=4, seed=0)
    obs, _ = env.reset(seed=0)
    action, _ = agent.predict(obs, deterministic=True)
    assert action.shape == env.action_space.shape
