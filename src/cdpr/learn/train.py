r"""Deterministic supervised / PINN training loop.

Both :class:`InverseDynamicsMLP` and :class:`InverseDynamicsPINN`
implement a ``training_loss(x, y) -> dict`` returning a dict with
``total`` plus per-component scalars (``data``, ``physics``). The loop
below treats either model identically, logs the full decomposition, and
returns a :class:`TrainingHistory` that contains epoch-by-epoch metrics
suitable for plotting in dissertation figures.

The loop is intentionally vanilla:
* a single fixed learning rate (Adam),
* full-dataset shuffling each epoch,
* optional early stopping on the validation loss,
* deterministic given a seed.

Anything fancier (LR schedulers, gradient clipping, mixed precision)
should be added by the user in a wrapper --- this loop is the reference
baseline used in the benchmarker and the dissertation reproducibility
manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cdpr.learn._lazy import require_torch

if TYPE_CHECKING:                                           # pragma: no cover
    import torch
    from cdpr.learn.datasets import TrajectoryDataset
    from cdpr.learn.models.mlp import InverseDynamicsMLP

torch = require_torch()


# ---------------------------------------------------------------------------
# History container
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TrainingHistory:
    """Epoch-by-epoch metric trace from :func:`train`."""

    train_total: list[float] = field(default_factory=list)
    train_data: list[float] = field(default_factory=list)
    train_physics: list[float] = field(default_factory=list)
    val_total: list[float] = field(default_factory=list)
    val_data: list[float] = field(default_factory=list)
    val_physics: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        if not self.epochs:
            return {}
        last = -1
        return {
            "epochs": int(self.epochs[last] + 1),
            "train_total_last": float(self.train_total[last]),
            "val_total_last": float(self.val_total[last]) if self.val_total else float("nan"),
            "train_data_last": float(self.train_data[last]),
            "val_data_last": float(self.val_data[last]) if self.val_data else float("nan"),
        }


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _eval_loss(model, dataset_torch, device, batch_size: int) -> dict[str, float]:
    from torch.utils.data import DataLoader

    if len(dataset_torch) == 0:
        return {}
    model.eval()
    loader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=False)
    running: dict[str, float] = {}
    n_seen = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            losses = model.training_loss(x, y)
            bsz = x.shape[0]
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach()) * bsz
            n_seen += bsz
    return {k: v / n_seen for k, v in running.items()}


def train(
    model: "InverseDynamicsMLP",
    train_dataset: "TrajectoryDataset",
    *,
    val_dataset: "TrajectoryDataset | None" = None,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
    early_stop_patience: int | None = None,
    log_every: int = 10,
    on_epoch_end=None,
) -> TrainingHistory:
    """Train ``model`` on ``train_dataset``; optionally validate.

    Parameters
    ----------
    early_stop_patience:
        When set, stop training if the validation total loss does not
        improve for this many epochs. Requires ``val_dataset``.
    log_every:
        Print a one-line summary every ``log_every`` epochs (0 to silence).
    on_epoch_end:
        Optional ``callable(epoch, history) -> None`` hook. Useful for
        custom logging / TensorBoard.
    """
    from torch.utils.data import DataLoader

    _seed_everything(seed)
    device_obj = torch.device(device)
    model.to(device_obj)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    train_torch = train_dataset.as_torch()
    val_torch = val_dataset.as_torch() if val_dataset is not None else None
    history = TrainingHistory()
    best_val = float("inf")
    patience_left = early_stop_patience

    for epoch in range(epochs):
        model.train()
        loader = DataLoader(train_torch, batch_size=batch_size, shuffle=True)
        running: dict[str, float] = {}
        n_seen = 0
        for x, y in loader:
            x = x.to(device_obj); y = y.to(device_obj)
            losses = model.training_loss(x, y)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            optimizer.step()
            bsz = x.shape[0]
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + float(v.detach()) * bsz
            n_seen += bsz
        train_metrics = {k: v / n_seen for k, v in running.items()}

        history.epochs.append(epoch)
        history.train_total.append(train_metrics.get("total", float("nan")))
        history.train_data.append(train_metrics.get("data", float("nan")))
        history.train_physics.append(train_metrics.get("physics", 0.0))

        if val_torch is not None:
            val_metrics = _eval_loss(model, val_torch, device_obj, batch_size)
            history.val_total.append(val_metrics.get("total", float("nan")))
            history.val_data.append(val_metrics.get("data", float("nan")))
            history.val_physics.append(val_metrics.get("physics", 0.0))

        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            msg = f"epoch {epoch:4d}  train_total={train_metrics['total']:.4e}"
            if val_torch is not None:
                msg += f"  val_total={val_metrics['total']:.4e}"
            if "physics" in train_metrics:
                msg += f"  train_phys={train_metrics['physics']:.4e}"
            print(msg)

        if on_epoch_end is not None:
            on_epoch_end(epoch, history)

        # Early stopping
        if val_torch is not None and early_stop_patience is not None:
            current_val = val_metrics.get("total", float("inf"))
            if current_val < best_val - 1e-9:
                best_val = current_val
                patience_left = early_stop_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    if log_every:
                        print(f"early stop at epoch {epoch}  (no val improvement for "
                              f"{early_stop_patience} epochs)")
                    break

    return history


# ---------------------------------------------------------------------------
# Convenience: save / load
# ---------------------------------------------------------------------------

def save_checkpoint(model: "InverseDynamicsMLP", path) -> None:
    """Persist model state-dict + config to ``path`` (``.pt``)."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": getattr(model, "config", None),
        "model_class": type(model).__name__,
    }, p)
