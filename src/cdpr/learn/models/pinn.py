r"""Physics-informed inverse-dynamics model.

The same MLP backbone as :class:`InverseDynamicsMLP`, with the loss

.. math::

    \mathcal{L} \;=\; \mathcal{L}_\text{data}
        \;+\; \lambda_\text{phys}\,\mathcal{L}_\text{phys},

where

.. math::

    \mathcal{L}_\text{data} \;=\;
        \mathbb{E}\bigl[\lVert \hat{\boldsymbol\tau} - \boldsymbol\tau \rVert^2\bigr],

.. math::

    \mathcal{L}_\text{phys} \;=\;
        \mathbb{E}\Bigl[\,\bigl\lVert \mathbf{W}(\mathbf{q})\,\hat{\boldsymbol\tau}
            \;+\; \mathbf{w}_\text{ext}(\mathbf{q}, \dot{\mathbf{x}}_\text{des})
            \bigr\rVert^2\Bigr],

and the external wrench :math:`\mathbf{w}_\text{ext}` is the Newton-Euler
target the cables must apply ::

    F_target = m * a_des_lin - m * g
    tau_target = I_W * a_des_ang + omega x I_W omega

i.e. the cable wrench needed to realise the desired acceleration in the
presence of gravity and inertial coupling. The physics loss penalises
the deviation of :math:`\mathbf{W}(\mathbf{q})\hat{\boldsymbol\tau}` from
:math:`-\mathbf{w}_\text{ext}` --- a network that ignores the geometric
model is punished even if its data fit is good, which keeps the
learned mapping interpretable as an inverse-dynamics solution.

The structure matrix :math:`\mathbf{W}(\mathbf{q})` is computed in NumPy
via the Phase-1 :func:`structure_matrix` per sample and converted to a
detached tensor; gradients flow only through :math:`\hat{\boldsymbol\tau}`,
which is the right thing because :math:`\mathbf{W}` is a fixed
geometric quantity at each pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cdpr.learn._lazy import require_torch
from cdpr.learn.models.mlp import InverseDynamicsMLP, MLPConfig

if TYPE_CHECKING:                                           # pragma: no cover
    import torch
    from torch import Tensor
    from cdpr.geometry.robot import Robot

torch = require_torch()


@dataclass(slots=True)
class FeatureLayout:
    """Indices into the input vector for each physical quantity.

    Defaults match :func:`cdpr.learn.datasets.dataset_from_simulation`.
    Override to point a PINN at a custom dataset layout.
    """

    position: slice = field(default_factory=lambda: slice(0, 3))
    quaternion: slice = field(default_factory=lambda: slice(3, 7))
    linear_velocity: slice = field(default_factory=lambda: slice(7, 10))
    angular_velocity: slice = field(default_factory=lambda: slice(10, 13))
    desired_linear_accel: slice = field(default_factory=lambda: slice(13, 16))
    desired_angular_accel: slice = field(default_factory=lambda: slice(16, 19))


@dataclass(slots=True)
class PINNConfig:
    backbone: MLPConfig = field(default_factory=MLPConfig)
    physics_weight: float = 1e-3
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    feature_layout: FeatureLayout = field(default_factory=FeatureLayout)


class InverseDynamicsPINN(InverseDynamicsMLP):
    """Physics-informed counterpart to :class:`InverseDynamicsMLP`."""

    def __init__(self, robot: "Robot", config: PINNConfig | None = None) -> None:
        cfg = config or PINNConfig()
        # Default backbone output width to the robot's cable count if the
        # caller did not override it.
        if cfg.backbone.out_features != robot.n_cables:
            cfg.backbone.out_features = robot.n_cables
        super().__init__(cfg.backbone)
        self.robot = robot
        self.cfg = cfg
        self.inertia = robot.require_inertia()
        self._gravity = np.asarray(cfg.gravity, dtype=np.float64)

    # --- physics computation -------------------------------------------

    def _wrench_target(self, x_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute the cable wrench required by Newton--Euler for each sample.

        Returns ``(W_stack, w_target_stack)`` --- the structure matrix
        and the *negative* target external wrench (so the residual is
        ``W tau + w_target``).
        """
        from scipy.spatial.transform import Rotation
        from cdpr.core.frames import Pose
        from cdpr.kinematics.jacobian import structure_matrix

        layout = self.cfg.feature_layout
        positions = x_np[:, layout.position]
        quats = x_np[:, layout.quaternion]
        omegas = x_np[:, layout.angular_velocity]
        a_lin_des = x_np[:, layout.desired_linear_accel]
        a_ang_des = x_np[:, layout.desired_angular_accel]

        n = x_np.shape[0]
        dof = self.robot.dof
        m = self.robot.n_cables
        W_stack = np.empty((n, dof, m), dtype=np.float64)
        w_target_stack = np.empty((n, dof), dtype=np.float64)

        mass = self.inertia.mass
        I_body = self.inertia.inertia

        for k in range(n):
            pose = Pose(position=positions[k], rotation=Rotation.from_quat(quats[k]))
            W_stack[k] = structure_matrix(pose, self.robot)

            R = pose.rotation.as_matrix()
            I_world = R @ I_body @ R.T
            F_target = mass * a_lin_des[k] - mass * self._gravity
            Tau_target = I_world @ a_ang_des[k] + np.cross(omegas[k], I_world @ omegas[k])

            # tension_distribution solves W tau = -w_ext, so the cable wrench
            # equals the *negative* of what we plug here:  W tau = -(-target) = target.
            # We store w_target = -[F_target ; Tau_target] (so residual = W tau + w_target).
            w_full = np.concatenate([F_target, Tau_target])
            w_target_stack[k] = -w_full[:dof]

        return W_stack, w_target_stack

    # --- losses --------------------------------------------------------

    def training_loss(self, x: "Tensor", y: "Tensor") -> dict[str, "Tensor"]:
        pred = self.forward(x)                                  # (B, m)
        data = ((pred - y) ** 2).mean()

        # Compute the per-sample W and w_target in NumPy (no gradients flow
        # through them; they're fixed geometric quantities at the input pose).
        x_np = x.detach().cpu().numpy()
        W_stack, w_target = self._wrench_target(x_np)
        W_t = torch.as_tensor(W_stack, dtype=pred.dtype, device=pred.device)
        wt_t = torch.as_tensor(w_target, dtype=pred.dtype, device=pred.device)

        # residual_k = W_k @ pred_k + w_target_k    (shape (B, dof))
        residual = torch.einsum("bij,bj->bi", W_t, pred) + wt_t
        phys = (residual ** 2).mean()

        total = data + self.cfg.physics_weight * phys
        return {"total": total, "data": data, "physics": phys}

    def predict_with_components(self, x: "Tensor", y: "Tensor | None" = None) -> dict[str, "Tensor"]:
        pred = self.predict(x)
        out: dict[str, "Tensor"] = {"prediction": pred}
        if y is not None:
            out["data_residual"] = pred - y
        x_np = x.detach().cpu().numpy()
        W_stack, w_target = self._wrench_target(x_np)
        W_t = torch.as_tensor(W_stack, dtype=pred.dtype, device=pred.device)
        wt_t = torch.as_tensor(w_target, dtype=pred.dtype, device=pred.device)
        out["physics_residual"] = torch.einsum("bij,bj->bi", W_t, pred) + wt_t
        return out
