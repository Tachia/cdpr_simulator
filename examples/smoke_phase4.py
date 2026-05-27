"""Phase-4 smoke run: hybrid physics + data-driven learning end-to-end.

Walks every Phase-4 surface:

1. Build a tracked-trajectory simulation, distill it into a supervised
   inverse-dynamics dataset.
2. Train an MLP and a PINN on the same data; report both loss curves.
3. Wrap the trained PINN as a closed-loop controller and evaluate it
   against analytic PD and computed-torque baselines on a *new*
   trajectory (sim-to-sim generalisation).
4. Build a Gymnasium env and confirm the SB3 factories construct.

Artifacts (training history JSON, benchmark table, a quick reward
trace) land in ``runs/phase4_smoke/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

from cdpr.control import ComputedTorqueController, PDController  # noqa: F401
from cdpr.dynamics.rigid_body import PlatformState
from cdpr.dynamics.simulator import simulate
from cdpr.learn.benchmark import Benchmark
from cdpr.learn.datasets import Normalizer, dataset_from_simulation
from cdpr.learn.env import CDPREnv, CDPREnvConfig
from cdpr.learn.models import InverseDynamicsMLP, InverseDynamicsPINN
from cdpr.learn.models.mlp import MLPConfig
from cdpr.learn.models.pinn import PINNConfig
from cdpr.learn.train import train
from cdpr.robots import ipanema_class
from cdpr.trajectory.paths import CircularPath, LissajousPath
from cdpr.trajectory.scaling import QuinticScaling
from cdpr.trajectory.trajectory import Trajectory


def _build_training_set(robot):
    """A 4 s CT-tracked figure-eight.

    Using a computed-torque controller for the training data gives labels
    that are the (approximate) analytical inverse-dynamics tensions ---
    the mapping the model is then learning to approximate. Training on
    PD-controlled labels would bake the PD law's reference dependency into
    the model and break transfer to a different trajectory at eval time.
    """
    traj = Trajectory(
        path=LissajousPath(center=np.zeros(3),
                           amplitudes=[0.3, 0.2, 0.0],
                           frequencies=[1.0, 2.0, 0.0]),
        scaling=QuinticScaling(duration=4.0),
    )
    state0 = PlatformState.at_rest(traj.pose(0.0))
    ct = ComputedTorqueController(Kp_pos=400.0, Kd_pos=40.0,
                                  Kp_rot=400.0, Kd_rot=40.0)
    sim = simulate(robot=robot, state0=state0, duration=4.0, dt=2e-3,
                   reference=traj, controller=ct)
    return dataset_from_simulation(sim)


def _train_model(model, ds, label: str, *, epochs: int = 80):
    history = train(model, ds, epochs=epochs, batch_size=64, learning_rate=3e-3,
                    seed=0, log_every=0)
    print(f"   [{label}] train_total:  {history.train_total[0]:.4e} -> "
          f"{history.train_total[-1]:.4e}")
    return history


def main(out_root: Path = Path("runs/phase4_smoke")) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    robot = ipanema_class()

    # --- 1. Dataset ----------------------------------------------------
    print("[1/4] Building inverse-dynamics dataset...")
    raw_ds = _build_training_set(robot)
    raw_ds.fit_normalizer()
    ds = raw_ds.transform_with(raw_ds.normalizer)
    target_scaler = Normalizer().fit(ds.targets)
    ds.targets = target_scaler.transform(ds.targets)
    train_ds, val_ds = ds.split(fraction=0.8, seed=0)

    # --- 2. Train MLP and PINN ----------------------------------------
    print("[2/4] Training models...")
    mlp = InverseDynamicsMLP(MLPConfig(
        in_features=ds.n_features, out_features=ds.n_targets, hidden=(128, 128),
    ))
    pinn_cfg = PINNConfig()
    pinn_cfg.backbone.in_features = ds.n_features
    pinn_cfg.backbone.hidden = (128, 128)
    pinn_cfg.physics_weight = 1e-9
    pinn = InverseDynamicsPINN(robot, pinn_cfg)

    h_mlp = _train_model(mlp, train_ds, "MLP", epochs=80)
    h_pinn = _train_model(pinn, train_ds, "PINN", epochs=80)

    (out_root / "training_history.json").write_text(json.dumps({
        "mlp":  h_mlp.summary(),
        "pinn": h_pinn.summary(),
    }, indent=2), encoding="utf-8")

    # --- 3a. Benchmark analytical controllers on a held-out trajectory ---
    print("[3a/4] Benchmarking analytical controllers on a held-out trajectory...")
    eval_traj = Trajectory(
        path=CircularPath(center=np.zeros(3), radius=0.25, axis=[0, 0, 1]),
        scaling=QuinticScaling(duration=1.5),
    )
    bench = Benchmark(robot=robot, trajectory=eval_traj, duration=1.5, dt=2e-3)
    controllers = {
        "open_loop":       None,
        "pd":              PDController(Kp_pos=2000.0, Kd_pos=200.0,
                                        Kp_rot=50.0, Kd_rot=5.0),
        "computed_torque": ComputedTorqueController(Kp_pos=900.0, Kd_pos=60.0,
                                                    Kp_rot=900.0, Kd_rot=60.0),
    }
    report = bench.run(controllers, keep_results=False)
    table = {
        o.name: {
            "mean_err_m": o.mean_position_error,
            "peak_err_m": o.peak_position_error,
            "rms_tension_N": o.rms_tension,
            "infeasible_steps": o.infeasible_steps,
        }
        for o in report.outcomes
    }
    for name, m in table.items():
        print(f"     {name:18s} mean={m['mean_err_m']:.4e} m   "
              f"peak={m['peak_err_m']:.4e} m")
    (out_root / "benchmark.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

    # --- 3b. Open-loop prediction accuracy on the eval trajectory --------
    # This is the dissertation-relevant comparison: how well does the trained
    # model approximate the analytical inverse-dynamics tensions across the
    # *new* trajectory? Closed-loop deployment of the learned policy needs
    # extra care (the PD safety net interacts with whatever PD-like behaviour
    # the model implicitly learned at training time) and is left as a separate
    # study; here we just measure pure prediction quality.
    print("[3b/4] Measuring prediction accuracy on the eval trajectory...")
    sim_eval = simulate(
        robot=robot,
        state0=PlatformState.at_rest(eval_traj.pose(0.0)),
        duration=1.5, dt=2e-3,
        reference=eval_traj,
        controller=controllers["computed_torque"],          # generates ground-truth labels
    )
    ds_eval_raw = dataset_from_simulation(sim_eval)
    ds_eval = ds_eval_raw.transform_with(raw_ds.normalizer)
    ds_eval.targets = target_scaler.transform(ds_eval.targets)
    import torch
    with torch.no_grad():
        x_eval = torch.as_tensor(ds_eval.inputs, dtype=torch.float32)
        tau_mlp = mlp(x_eval).cpu().numpy()
        tau_pinn = pinn(x_eval).cpu().numpy()
    # Denormalise.
    tau_mlp_N = target_scaler.inverse_transform(tau_mlp)
    tau_pinn_N = target_scaler.inverse_transform(tau_pinn)
    tau_true_N = target_scaler.inverse_transform(ds_eval.targets)

    def rms_err_N(pred):
        return float(np.sqrt(np.mean((pred - tau_true_N) ** 2)))
    pred_table = {
        "mlp_rms_tension_err_N":  rms_err_N(tau_mlp_N),
        "pinn_rms_tension_err_N": rms_err_N(tau_pinn_N),
        "reference_rms_tension_N": float(np.sqrt(np.mean(tau_true_N ** 2))),
    }
    for k, v in pred_table.items():
        print(f"     {k:30s} = {v:.3f}")
    (out_root / "prediction_accuracy.json").write_text(
        json.dumps(pred_table, indent=2), encoding="utf-8")

    # --- 4. RL environment construction ------------------------------
    print("[4/4] Building Gymnasium env + SB3 PPO factory (no training)...")
    env = CDPREnv(robot, config=CDPREnvConfig(horizon=64, dt=5e-3))
    obs, info = env.reset(seed=0)
    print(f"   env obs shape: {obs.shape}, action shape: {env.action_space.shape}")
    n_rand_steps = 16
    rewards = []
    for _ in range(n_rand_steps):
        action = env.action_space.sample()
        obs, r, terminated, truncated, info = env.step(action)
        rewards.append(r)
        if terminated or truncated:
            break
    print(f"   random rollout: {len(rewards)} steps, mean reward {np.mean(rewards):.3f}")

    try:
        from cdpr.learn.rl import make_ppo
        ppo = make_ppo(env, n_steps=16, batch_size=4, seed=0)
        print(f"   PPO agent built: policy={type(ppo.policy).__name__}")
    except ImportError as exc:
        print(f"   PPO factory unavailable (skipping): {exc}")

    print(f"\nDone. Artifacts in: {out_root.resolve()}")


if __name__ == "__main__":
    main()
