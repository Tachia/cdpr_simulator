r"""Plain MLP inverse-dynamics regressor.

Used as the data-only baseline against which the PINN counterpart in
:mod:`cdpr.learn.models.pinn` is evaluated. The architecture is
deliberately small (two hidden layers, configurable widths) --- a
publishable comparison wants a baseline a researcher could plausibly
have arrived at, not a state-of-the-art over-engineered model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from cdpr.learn._lazy import require_torch

if TYPE_CHECKING:                                           # pragma: no cover
    import torch
    from torch import Tensor, nn

torch = require_torch()
from torch import nn                                          # noqa: E402


@dataclass(slots=True)
class MLPConfig:
    in_features: int = 19
    out_features: int = 8
    hidden: tuple[int, ...] = (128, 128)
    dropout: float = 0.0
    activation: str = "relu"


def _make_activation(name: str) -> "nn.Module":
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name!r}")


class InverseDynamicsMLP(nn.Module):
    """Feed-forward regressor :math:`\\mathbb{R}^{n_\\text{in}} \\to \\mathbb{R}^{n_\\text{out}}`."""

    def __init__(self, config: MLPConfig | None = None) -> None:
        super().__init__()
        cfg = config or MLPConfig()
        self.config = cfg
        layers: list[nn.Module] = []
        prev = cfg.in_features
        for h in cfg.hidden:
            layers += [nn.Linear(prev, h), _make_activation(cfg.activation)]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
            prev = h
        layers.append(nn.Linear(prev, cfg.out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: "Tensor") -> "Tensor":
        return self.net(x)

    # --- inference helpers ---------------------------------------------

    def predict(self, x: "Tensor") -> "Tensor":
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict_with_components(self, x: "Tensor", y: "Tensor | None" = None) -> dict[str, "Tensor"]:
        """Return the prediction plus a data-loss component (if y given).

        The MLP has only a data loss; the PINN counterpart returns one
        extra entry for the physics residual. Keeping the same return
        shape across models lets the training loop and the benchmarker
        log the same channels regardless of model class.
        """
        pred = self.predict(x)
        out: dict[str, "Tensor"] = {"prediction": pred}
        if y is not None:
            out["data_residual"] = pred - y
        return out

    # --- training loss --------------------------------------------------

    def training_loss(self, x: "Tensor", y: "Tensor") -> dict[str, "Tensor"]:
        pred = self.forward(x)
        data = ((pred - y) ** 2).mean()
        return {"total": data, "data": data}
