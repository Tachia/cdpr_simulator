"""Gymnasium environment: space shapes, reset/step contract, reward decomposition."""

from __future__ import annotations

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium")

from cdpr.learn.env import CDPREnv, CDPREnvConfig
from cdpr.learn.rewards import RewardSum, PoseTracking, InfeasibilityPenalty


def test_env_obs_action_shapes(ipanema):
    env = CDPREnv(ipanema, config=CDPREnvConfig(action_mode="wrench", horizon=10))
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert env.action_space.shape == (6,)
    assert obs.dtype == np.float32
    assert "reference_pose" in info


def test_env_step_returns_5tuple_and_terminates_on_horizon(ipanema):
    env = CDPREnv(ipanema, config=CDPREnvConfig(horizon=5))
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    terminated = False
    truncated = False
    n = 0
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(a)
        n += 1
        assert obs.shape == env.observation_space.shape
        assert isinstance(r, float)
        assert "tracking_error" in info
        assert "reward_decomposition" in info
        assert n < 100, "env failed to terminate"
    assert truncated
    assert n == 5


def test_tension_action_mode_yields_per_cable_action(ipanema):
    env = CDPREnv(ipanema, config=CDPREnvConfig(action_mode="tension"))
    assert env.action_space.shape == (ipanema.n_cables,)
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, r, terminated, truncated, info = env.step(a)
    # Tension action: cables get the midpoint of [t_min, t_max] when action == 0.
    expected_mid = 0.5 * (ipanema.limits.t_min + ipanema.limits.t_max)
    assert np.allclose(info["tensions"], expected_mid)


def test_reward_decomposition_present_each_step(ipanema):
    reward = RewardSum(components=[
        PoseTracking(weight=1.0),
        InfeasibilityPenalty(weight=5.0),
    ])
    env = CDPREnv(ipanema, reward=reward, config=CDPREnvConfig(horizon=3))
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, dtype=np.float32)
    _, _, _, _, info = env.step(a)
    decomp = info["reward_decomposition"]
    assert "pose_tracking" in decomp
    assert "infeasibility_penalty" in decomp
    assert "total" in decomp
