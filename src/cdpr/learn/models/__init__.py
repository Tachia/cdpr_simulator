"""PyTorch models for inverse dynamics.

Two models in symmetric roles:

* :class:`InverseDynamicsMLP` -- plain feed-forward regressor; the data-
  only baseline that subsequent physics-informed work is benchmarked
  against.
* :class:`InverseDynamicsPINN` -- same architecture, with a Newton-Euler
  physics-residual term added to the loss. Exposes
  :meth:`predict_with_components` so the data and physics residuals can
  be plotted separately.

Both implement the same public surface (``predict``, ``training_loss``,
``predict_with_components``) so the training loop in
:mod:`cdpr.learn.train` doesn't branch on model type.
"""

from cdpr.learn.models.mlp import InverseDynamicsMLP
from cdpr.learn.models.pinn import InverseDynamicsPINN

__all__ = ["InverseDynamicsMLP", "InverseDynamicsPINN"]
