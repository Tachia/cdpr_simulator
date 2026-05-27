"""Lazy-import helpers for the learning extras.

Each function returns the imported module if available; otherwise raises
``ImportError`` with the exact pip command needed to fix it. This keeps
the scientific core importable on machines without PyTorch / SB3 / Gym.
"""

from __future__ import annotations

from types import ModuleType


def require_torch() -> ModuleType:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "The learning layer needs PyTorch. Install with:  pip install 'cdpr[learn]'"
        ) from exc
    return torch


def require_gymnasium() -> ModuleType:
    try:
        import gymnasium
    except ImportError as exc:
        raise ImportError(
            "The RL environment needs Gymnasium. Install with:  pip install 'cdpr[learn]'"
        ) from exc
    return gymnasium


def require_stable_baselines3() -> ModuleType:
    try:
        import stable_baselines3
    except ImportError as exc:
        raise ImportError(
            "The SB3 wrappers need Stable-Baselines3. "
            "Install with:  pip install 'cdpr[rl]'"
        ) from exc
    return stable_baselines3
