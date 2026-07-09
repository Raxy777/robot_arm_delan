"""Week 10 — the benchmark that IS the project.

Three models (analytic upper bound, black-box MLP, DeLaN) go into the SAME
InverseDynamicsController and track a figure-eight on the TRUE plant. We sweep
four scenarios, each a step further from the training distribution:

    nominal        the speed/size the data was collected at         (in-dist)
    fast           shorter period + larger amplitude -> higher q̇,q̈  (OOD kinematics)
    fast+noise     fast, but the controller reads noisy q, q̇         (OOD + sensing)
    fast+payload   fast, plus an unseen 0.5 kg tip mass on the plant  (OOD + dynamics)

The controller only ever calls model.inverse_dynamics(q, qd, aq); nothing knows
about the payload or the noise. So every difference between columns is the model
coping (or not) with a world its training data never showed it — which is the
entire argument for baking physics structure into the model.

Outputs (to results/):
    bench_closed_loop.csv    tracking RMSE (mm), model x scenario
    bench_open_loop.csv      torque RMSE (N m), model x {test_id, test_ood}
    torque_scatter.npz       pred-vs-true torques on OOD split, per model (for plots)

    python benchmark.py                 # analytic-plant RK4 (no MuJoCo needed)
    python benchmark.py --seed 1        # different noise draw
"""

import argparse
import csv
import os

import numpy as np

import dynamics
from controllers import InverseDynamicsController
from kinematics import forward_kinematics
from mlp_model import MLPInverseDynamics, build_inputs
from delan_model import DeLaNInverseDynamics
from plant import PlantSim, with_payload
from trajectories import figure_eight, figure_eight_joint

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Week_8", "data")
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")

# scenario = trajectory regime + plant perturbations. `noise` is the std of the
# additive Gaussian on the controller's (q, q̇) reading; `payload` is tip mass kg.
SCENARIOS = {
    "nominal":      dict(period=6.0, A=0.5, B=0.35, noise=0.0,   payload=0.0),
    "fast":         dict(period=2.5, A=0.7, B=0.5,  noise=0.0,   payload=0.0),
    "fast+noise":   dict(period=2.5, A=0.7, B=0.5,  noise=0.01,  payload=0.0),
    "fast+payload": dict(period=2.5, A=0.7, B=0.5,  noise=0.0,   payload=0.5),
}
# split kinematic (trajectory) keys from perturbation keys
_TRAJ_KEYS = ("period", "A", "B")


def _load_models():
    models = {"analytic": dynamics}
    mlp = os.path.join(MODELS, "mlp.pt")
    delan = os.path.join(MODELS, "delan.pt")
    if os.path.exists(mlp):
        models["mlp"] = MLPInverseDynamics.load(mlp)
    if os.path.exists(delan):
        models["delan"] = DeLaNInverseDynamics.load(delan)
    return models


def closed_loop_rmse(model, scenario, seed=0, duration=8.0,
                     kp=400.0, kd=40.0, elbow_up=True):
    """RMS end-effector tracking error (metres) on the true plant.

    The controller sees `q_meas = q_true + noise`; the score is computed from the
    NOISE-FREE true state, so noise degrades control (bad commands) without
    flattering the metric.
    """
    traj = {k: scenario[k] for k in _TRAJ_KEYS}
    plant_params = with_payload(scenario["payload"])
    noise = scenario["noise"]

    sim = PlantSim(params=plant_params)
    ctrl = InverseDynamicsController(kp=kp, kd=kd, model=model)
    dt = sim.dt
    rng = np.random.default_rng(seed)

    q0, qd0, _ = figure_eight_joint(0.0, elbow_up=elbow_up, **traj)
    sim.reset(q0, qd0)

    ee, ee_des = [], []
    n = int(duration / dt)
    for i in range(n):
        t = i * dt
        q_true, qd_true = sim.q, sim.qd
        if noise > 0.0:
            q_meas = q_true + rng.normal(0.0, noise, 2)
            qd_meas = qd_true + rng.normal(0.0, noise * 10.0, 2)  # vel noisier
        else:
            q_meas, qd_meas = q_true, qd_true
        q_d, qd_d, qdd_d = figure_eight_joint(t, elbow_up=elbow_up, **traj)
        sim.set_torque(ctrl(q_meas, qd_meas, q_d, qd_d, qdd_d))
        x_des, _, _ = figure_eight(t, **traj)
        ee.append(forward_kinematics(q_true))
        ee_des.append(x_des)
        sim.step()

    ee, ee_des = np.array(ee), np.array(ee_des)
    return float(np.sqrt(np.mean(np.sum((ee - ee_des) ** 2, axis=1))))


def _torque_pred(model, q, qd, qdd):
    if model is dynamics:
        return np.array([dynamics.inverse_dynamics(q[i], qd[i], qdd[i])
                         for i in range(len(q))])
    if hasattr(model, "predict"):                       # MLP batched
        return model.predict(build_inputs(q, qd, qdd))
    return np.array([model.inverse_dynamics(q[i], qd[i], qdd[i])  # DeLaN
                     for i in range(len(q))])


def open_loop(models):
    """Torque RMSE on held-out splits + raw pred/true for the scatter plot."""
    rows, scatter = {}, {}
    for split in ["test_id", "test_ood"]:
        d = np.load(os.path.join(DATA, f"{split}.npz"))
        q, qd, qdd, tau = d["q"], d["qd"], d["qdd"], d["tau"]
        for name, m in models.items():
            pred = _torque_pred(m, q, qd, qdd)
            rows.setdefault(name, {})[split] = float(
                np.sqrt(np.mean((pred - tau) ** 2)))
            if split == "test_ood":
                scatter[name] = (tau.copy(), pred.copy())
    return rows, scatter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="sensor-noise RNG seed")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)

    models = _load_models()
    print(f"Models: {', '.join(models)}\n")

    # ---- A. open loop -------------------------------------------------------
    ol_rows, scatter = open_loop(models)
    print("A. Open-loop torque-prediction RMSE (N m)")
    hdr = f"{'model':10s}{'test_id':>12s}{'test_ood':>12s}"
    print(hdr); print("-" * len(hdr))
    for name in models:
        r = ol_rows[name]
        print(f"{name:10s}{r['test_id']:12.4f}{r['test_ood']:12.4f}")
    with open(os.path.join(RESULTS, "bench_open_loop.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "test_id", "test_ood"])
        for name in models:
            w.writerow([name, ol_rows[name]["test_id"], ol_rows[name]["test_ood"]])
    np.savez(os.path.join(RESULTS, "torque_scatter.npz"),
             **{f"{k}_true": v[0] for k, v in scatter.items()},
             **{f"{k}_pred": v[1] for k, v in scatter.items()})

    # ---- B. closed loop -----------------------------------------------------
    print("\nB. Closed-loop end-effector RMS tracking error (mm)")
    hdr = f"{'model':10s}" + "".join(f"{s:>14s}" for s in SCENARIOS)
    print(hdr); print("-" * len(hdr))
    cl = {}
    for name, m in models.items():
        row = f"{name:10s}"
        cl[name] = {}
        for sname, sc in SCENARIOS.items():
            rms = closed_loop_rmse(m, sc, seed=args.seed) * 1000.0
            cl[name][sname] = rms
            row += f"{rms:14.3f}"
        print(row)
    with open(os.path.join(RESULTS, "bench_closed_loop.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", *SCENARIOS])
        for name in models:
            w.writerow([name, *[cl[name][s] for s in SCENARIOS]])

    print(f"\nWrote CSVs + torque_scatter.npz to {RESULTS}")
    print("\nRead it: analytic is the upper bound. MLP should track under 'nominal'")
    print("but blow up as you move right (fast -> noise -> payload). DeLaN should")
    print("stay far closer to analytic across the OOD columns - that gap is the story.")


if __name__ == "__main__":
    main()
