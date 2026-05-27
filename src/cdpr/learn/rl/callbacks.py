r"""SB3 callbacks that surface CDPR-specific physics diagnostics.

Standard SB3 logging shows ``ep_rew_mean`` and ``ep_len_mean``, neither
of which tells you whether the agent is producing physically reasonable
behaviour. :class:`PhysicsLoggingCallback` adds:

* mean tracking error per rollout,
* fraction of steps with an infeasible tension QP,
* mean and peak cable tension utilisation (relative to ``t_max``),
* mean ``log10(condition_number)`` of the structure matrix.

These metrics make the learning curve interpretable in the same units
the dissertation chapters report. They're written to the same SB3
logger so TensorBoard / CSV consumers see them as ordinary scalars.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from cdpr.learn._lazy import require_stable_baselines3

if TYPE_CHECKING:                                           # pragma: no cover
    from stable_baselines3.common.callbacks import BaseCallback


def _base_callback_class():
    """Resolve ``BaseCallback`` lazily so SB3 isn't imported at module-load time."""
    sb3 = require_stable_baselines3()
    from stable_baselines3.common.callbacks import BaseCallback
    return BaseCallback


class PhysicsLoggingCallback(_base_callback_class() if False else object):
    """SB3 callback recording CDPR physics diagnostics per rollout.

    Created lazily: the constructor binds the real ``BaseCallback`` base
    class on first instantiation so importing the module never needs SB3.
    """

    def __init__(self, *, robot, window: int = 1000, verbose: int = 0) -> None:
        sb3_base = _base_callback_class()
        # Re-class self so SB3's training loop recognises it.
        self.__class__ = type(
            "PhysicsLoggingCallback", (sb3_base,), dict(self.__class__.__dict__)
        )
        sb3_base.__init__(self, verbose=verbose)
        self.robot = robot
        self.window = window
        self._tracking_err = deque(maxlen=window)
        self._infeasible_flag = deque(maxlen=window)
        self._tension_util = deque(maxlen=window)
        self._condnum_log10 = deque(maxlen=window)

    # SB3 interface: called every env step.
    def _on_step(self) -> bool:                              # pragma: no cover - exercised by SB3
        infos = self.locals.get("infos", [])
        for info in infos:
            err = info.get("tracking_error")
            tensions = info.get("tensions")
            infeasible = info.get("infeasible", False)

            if err is not None:
                self._tracking_err.append(float(err))
            self._infeasible_flag.append(1.0 if infeasible else 0.0)
            if tensions is not None and self.robot.limits is not None:
                util = float(np.max(tensions) / max(self.robot.limits.t_max.max(), 1e-9))
                self._tension_util.append(util)

        # Push aggregates every couple of hundred steps to avoid noise.
        if self.num_timesteps % 200 == 0 and self._tracking_err:
            self.logger.record("cdpr/tracking_err_mean", float(np.mean(self._tracking_err)))
            self.logger.record("cdpr/infeasible_rate", float(np.mean(self._infeasible_flag)))
            if self._tension_util:
                self.logger.record("cdpr/tension_utilisation_mean", float(np.mean(self._tension_util)))
                self.logger.record("cdpr/tension_utilisation_peak", float(np.max(self._tension_util)))
        return True
