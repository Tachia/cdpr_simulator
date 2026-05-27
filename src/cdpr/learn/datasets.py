r"""Supervised datasets built from Phase-1 simulations or Phase-3 ingested logs.

The canonical inverse-dynamics dataset packages each time step as

* **inputs** :math:`\mathbf{x} = [\mathbf{p}, \mathbf{q}, \mathbf{v},
  \boldsymbol\omega, \dot{\mathbf{v}}_\text{des}, \dot{\boldsymbol\omega}_\text{des}]
  \in \mathbb{R}^{19}`,
* **target** :math:`\mathbf{y} = \boldsymbol\tau \in \mathbb{R}^m`.

Desired accelerations are computed by finite-differencing the velocity
arrays of the source data; this is what a sensor-equipped lab would also
need to estimate, so the dataset realistically reflects what a deployed
inverse-dynamics model receives at inference time.

Normalisation is per-feature z-score, computed on the *training* split
only --- the validation / test splits use the same transform so a
publishable mean / std bias diagnostic remains comparable across folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from cdpr.learn._lazy import require_torch

if TYPE_CHECKING:                                           # pragma: no cover
    import torch
    from torch.utils.data import Dataset
    from cdpr.dynamics.simulator import SimulationResult
    from cdpr.ingest.containers import IngestedExperiment


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _finite_difference(arr: NDArray[np.float64], dt: float) -> NDArray[np.float64]:
    """Central-difference time derivative; one-sided at the endpoints."""
    out = np.empty_like(arr)
    out[1:-1] = (arr[2:] - arr[:-2]) / (2 * dt)
    out[0] = (arr[1] - arr[0]) / dt
    out[-1] = (arr[-1] - arr[-2]) / dt
    return out


def _stack_inputs(
    positions: NDArray[np.float64],
    quaternions: NDArray[np.float64],
    linear_velocities: NDArray[np.float64],
    angular_velocities: NDArray[np.float64],
    linear_accel: NDArray[np.float64],
    angular_accel: NDArray[np.float64],
) -> NDArray[np.float64]:
    return np.concatenate(
        [positions, quaternions, linear_velocities, angular_velocities,
         linear_accel, angular_accel],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Normalizer:
    """Per-feature z-score (fit once on training data, apply elsewhere)."""

    mean: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    std: NDArray[np.float64] = field(default_factory=lambda: np.ones(0))

    def fit(self, X: NDArray[np.float64]) -> "Normalizer":
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std < 1e-9] = 1.0           # avoid division by zero on constant channels
        self.std = std
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return (X - self.mean) / self.std

    def inverse_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return X * self.std + self.mean


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrajectoryDataset:
    """In-memory inverse-dynamics dataset.

    The PyTorch ``Dataset`` interface is built lazily via :meth:`as_torch`;
    holding the numpy arrays directly is useful for non-torch
    benchmarking and unit tests.
    """

    inputs: NDArray[np.float64]
    targets: NDArray[np.float64]
    time: NDArray[np.float64]
    feature_names: tuple[str, ...]
    normalizer: Normalizer | None = None

    @property
    def n_features(self) -> int:
        return int(self.inputs.shape[-1])

    @property
    def n_targets(self) -> int:
        return int(self.targets.shape[-1])

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def split(self, fraction: float = 0.8, shuffle: bool = True,
              seed: int = 0) -> tuple["TrajectoryDataset", "TrajectoryDataset"]:
        """Train / validation split with shared time-stamp ordering preserved."""
        n = len(self)
        idx = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        cut = int(round(fraction * n))
        train_idx, val_idx = np.sort(idx[:cut]), np.sort(idx[cut:])
        train = TrajectoryDataset(
            inputs=self.inputs[train_idx], targets=self.targets[train_idx],
            time=self.time[train_idx], feature_names=self.feature_names,
        )
        val = TrajectoryDataset(
            inputs=self.inputs[val_idx], targets=self.targets[val_idx],
            time=self.time[val_idx], feature_names=self.feature_names,
        )
        return train, val

    def fit_normalizer(self) -> "TrajectoryDataset":
        self.normalizer = Normalizer().fit(self.inputs)
        return self

    def transform_with(self, normalizer: Normalizer) -> "TrajectoryDataset":
        return TrajectoryDataset(
            inputs=normalizer.transform(self.inputs),
            targets=self.targets,
            time=self.time,
            feature_names=self.feature_names,
            normalizer=normalizer,
        )

    def as_torch(self):
        """Return a ``torch.utils.data.TensorDataset`` view of the arrays."""
        torch = require_torch()
        from torch.utils.data import TensorDataset
        return TensorDataset(
            torch.as_tensor(self.inputs, dtype=torch.float32),
            torch.as_tensor(self.targets, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _common_feature_names(n_cables: int) -> tuple[str, ...]:
    return (
        "p_x", "p_y", "p_z",
        "q_x", "q_y", "q_z", "q_w",
        "v_x", "v_y", "v_z",
        "w_x", "w_y", "w_z",
        "a_x", "a_y", "a_z",
        "alpha_x", "alpha_y", "alpha_z",
    )


def dataset_from_simulation(
    result: "SimulationResult", *, dt: float | None = None,
) -> TrajectoryDataset:
    """Build an inverse-dynamics dataset from a :class:`SimulationResult`."""
    if dt is None:
        if len(result.time) < 2:
            raise ValueError("SimulationResult is too short to estimate dt.")
        dt = float(result.time[1] - result.time[0])

    lin_accel = _finite_difference(result.linear_velocities, dt)
    ang_accel = _finite_difference(result.angular_velocities, dt)
    X = _stack_inputs(
        result.positions, result.quaternions_xyzw,
        result.linear_velocities, result.angular_velocities,
        lin_accel, ang_accel,
    )
    return TrajectoryDataset(
        inputs=X,
        targets=result.cable_tensions.astype(np.float64),
        time=result.time.astype(np.float64),
        feature_names=_common_feature_names(result.cable_tensions.shape[1]),
    )


def dataset_from_experiment(
    experiment: "IngestedExperiment",
) -> TrajectoryDataset:
    """Build an inverse-dynamics dataset from an :class:`IngestedExperiment`.

    Requires the experiment to carry positions, quaternions, velocities,
    and cable tensions. Velocities are differentiated to produce
    accelerations; if the experiment also recorded velocities directly,
    those are used as-is (the differentiator only fills in accelerations).
    """
    needed = ("positions", "quaternions_xyzw", "linear_velocities",
              "angular_velocities", "cable_tensions")
    missing = [n for n in needed if getattr(experiment, n) is None]
    if missing:
        raise ValueError(f"experiment is missing required channels: {missing}")

    time = experiment.time
    if len(time) < 2:
        raise ValueError("experiment is too short to estimate dt.")
    dt = float(np.median(np.diff(time)))

    lin_accel = _finite_difference(experiment.linear_velocities, dt)
    ang_accel = _finite_difference(experiment.angular_velocities, dt)
    X = _stack_inputs(
        experiment.positions, experiment.quaternions_xyzw,
        experiment.linear_velocities, experiment.angular_velocities,
        lin_accel, ang_accel,
    )
    return TrajectoryDataset(
        inputs=X,
        targets=experiment.cable_tensions.astype(np.float64),
        time=time.astype(np.float64),
        feature_names=_common_feature_names(experiment.cable_tensions.shape[1]),
    )
