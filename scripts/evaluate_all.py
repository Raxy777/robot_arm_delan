"""Three-way benchmark: analytic vs MLP vs DeLaN.

This is the headline table for Weeks 9-10. Two views:

  A. Torque-prediction RMSE on held-out data, in-distribution vs OOD (open loop).
  B. Closed-loop end-effector tracking error, nominal vs fast figure-eight.

Each model is plugged into the SAME InverseDynamicsController, so differences
are purely about model quality. Analytic is the upper bound; the story is that
DeLaN stays closer to analytic than the MLP as you move out of distribution.

    python evaluate_all.py --backend analytic   # no MuJoCo needed
    python evaluate_all.py --backend mujoco
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os

import numpy as np

import src.dynamics as dynamics
from evaluate_model import closed_loop_track, REGIMES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")


def _load_models():
    models = {"analytic": dynamics}
    mlp_path = os.path.join(MODELS, "mlp.pt")
    delan_path = os.path.join(MODELS, "delan.pt")
    if os.path.exists(mlp_path):
        from src.mlp_model import MLPInverseDynamics
        models["mlp"] = MLPInverseDynamics.load(mlp_path)
    if os.path.exists(delan_path):
        from src.delan_model import DeLaNInverseDynamics
        models["delan"] = DeLaNInverseDynamics.load(delan_path)
    return models


def _torque_rmse(model, split):
    d = np.load(os.path.join(DATA, f"{split}.npz"))
    q, qd, qdd, tau = d["q"], d["qd"], d["qdd"], d["tau"]
    if model is dynamics:
        pred = np.array([dynamics.inverse_dynamics(q[i], qd[i], qdd[i])
                         for i in range(len(q))])
    elif hasattr(model, "predict"):          # MLP: fast batched path
        from src.mlp_model import build_inputs
        pred = model.predict(build_inputs(q, qd, qdd))
    else:                                    # DeLaN: per-sample numpy wrapper
        pred = np.array([model.inverse_dynamics(q[i], qd[i], qdd[i])
                         for i in range(len(q))])
    return float(np.sqrt(np.mean((pred - tau) ** 2)))


def table_open_loop(models):
    splits = ["test_id", "test_ood"]
    have_data = all(os.path.exists(os.path.join(DATA, f"{s}.npz")) for s in splits)
    print("A. Torque-prediction RMSE (N m)  [open loop]\n")
    if not have_data:
        print("   [skip] data/ not found - run data_collection.py first.\n")
        return
    hdr = f"{'model':10s}" + "".join(f"{s:>16s}" for s in splits)
    print(hdr); print("-" * len(hdr))
    for name, m in models.items():
        row = f"{name:10s}" + "".join(f"{_torque_rmse(m, s):16.4f}" for s in splits)
        print(row)
    print()


def table_closed_loop(models, backend):
    print(f"B. End-effector RMS tracking error (mm)  [closed loop, {backend}]\n")
    hdr = f"{'model':10s}" + "".join(f"{r:>12s}" for r in REGIMES)
    print(hdr); print("-" * len(hdr))
    for name, m in models.items():
        row = f"{name:10s}"
        for regime in REGIMES.values():
            row += f"{closed_loop_track(m, backend, regime) * 1000:12.3f}"
        print(row)
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="analytic",
                    choices=["mujoco", "analytic"])
    args = ap.parse_args()

    models = _load_models()
    present = ", ".join(models)
    print(f"Models loaded: {present}\n")
    if "mlp" not in models or "delan" not in models:
        print("[note] train MLP (train_mlp.py) and DeLaN (train_delan.py) for the")
        print("       full three-way comparison. Showing whatever is available.\n")

    table_open_loop(models)
    table_closed_loop(models, args.backend)

    print("How to read it: analytic is the upper bound (tiny error everywhere).")
    print("A good MLP matches it in-distribution but degrades out-of-distribution.")
    print("DeLaN should sit between them out-of-distribution — closer to analytic")
    print("than the MLP — because its physics structure constrains extrapolation.")


if __name__ == "__main__":
    main()
