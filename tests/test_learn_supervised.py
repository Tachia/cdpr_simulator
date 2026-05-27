"""Supervised + PINN training on synthetic CDPR data."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cdpr.control import PDController
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.learn.datasets import Normalizer, dataset_from_simulation
from cdpr.learn.models import InverseDynamicsMLP, InverseDynamicsPINN
from cdpr.learn.models.mlp import MLPConfig
from cdpr.learn.models.pinn import PINNConfig
from cdpr.learn.train import train
from cdpr.trajectory.paths import CircularPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def _short_dataset(robot, home_pose):
    """Dataset from a PD-tracked circle. Tensions vary across samples,
    so the supervised problem is non-trivial."""
    traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.1, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=0.5),
    )
    state0 = PlatformState.at_rest(traj.pose(0.0))
    pd = PDController(Kp_pos=2000.0, Kd_pos=200.0, Kp_rot=50.0, Kd_rot=5.0)
    result = simulate(
        robot=robot, state0=state0, duration=0.5, dt=2e-3,
        reference=traj, controller=pd,
    )
    return dataset_from_simulation(result)


def _normalised_dataset(robot, home_pose):
    """Returns dataset with both inputs and targets z-score normalised."""
    ds = _short_dataset(robot, home_pose).fit_normalizer()
    ds = ds.transform_with(ds.normalizer)
    target_scaler = Normalizer().fit(ds.targets)
    ds.targets = target_scaler.transform(ds.targets)
    return ds


def test_dataset_shapes(ipanema, home_pose):
    ds = _short_dataset(ipanema, home_pose)
    assert ds.n_features == 19
    assert ds.n_targets == ipanema.n_cables
    assert len(ds) == len(ds.time)


def test_dataset_split_preserves_total(ipanema, home_pose):
    ds = _short_dataset(ipanema, home_pose)
    train_ds, val_ds = ds.split(fraction=0.8, seed=0)
    assert len(train_ds) + len(val_ds) == len(ds)
    assert train_ds.n_features == val_ds.n_features


def test_mlp_training_reduces_loss(ipanema, home_pose):
    """MLP on a tracked-circle dataset should cut training loss substantially."""
    ds = _normalised_dataset(ipanema, home_pose)

    cfg = MLPConfig(in_features=ds.n_features, out_features=ds.n_targets, hidden=(64, 64))
    model = InverseDynamicsMLP(cfg)
    history = train(
        model, ds, epochs=100, batch_size=32, learning_rate=3e-3,
        seed=0, log_every=0,
    )
    assert history.train_total[-1] < 0.2 * history.train_total[0]


def test_pinn_data_loss_drops_and_physics_finite(ipanema, home_pose):
    """PINN training should leave the physics residual finite and improve
    the data loss substantially."""
    ds = _normalised_dataset(ipanema, home_pose)

    cfg = PINNConfig()
    cfg.backbone.in_features = ds.n_features
    cfg.backbone.hidden = (64, 64)
    # Physics residual is in raw (un-normalised target) units. The MLP backbone
    # outputs normalised tensions, so the physics term is dominated by scale.
    # A very small weight keeps training stable while still penalising
    # wildly non-physical solutions.
    cfg.physics_weight = 1e-9
    model = InverseDynamicsPINN(ipanema, cfg)
    history = train(
        model, ds, epochs=100, batch_size=32, learning_rate=3e-3,
        seed=0, log_every=0,
    )
    assert all(np.isfinite(history.train_physics))
    assert history.train_data[-1] < 0.5 * history.train_data[0]


def test_predict_with_components_returns_data_and_physics(ipanema, home_pose):
    ds = _normalised_dataset(ipanema, home_pose)

    cfg = PINNConfig()
    cfg.backbone.in_features = ds.n_features
    cfg.backbone.hidden = (32,)
    model = InverseDynamicsPINN(ipanema, cfg)

    x = torch.as_tensor(ds.inputs[:4], dtype=torch.float32)
    y = torch.as_tensor(ds.targets[:4], dtype=torch.float32)
    comps = model.predict_with_components(x, y)
    assert "prediction" in comps and comps["prediction"].shape == (4, ipanema.n_cables)
    assert "data_residual" in comps
    assert "physics_residual" in comps
    assert comps["physics_residual"].shape == (4, ipanema.dof)
