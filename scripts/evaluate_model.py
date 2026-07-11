"""Closed-loop evaluation: put a model inside the controller and track.

Runs the figure-eight with InverseDynamicsController for each model and reports
end-effector RMS tracking error. Two regimes:

  - nominal : the same speed/size the arm normally operates at (in-distribution)
  - fast    : shorter period + bigger amplitude -> higher q̇, q̈ than training
              (out-of-distribution; the MLP should visibly degrade here)

    python evaluate_model.py --backend analytic   # works without MuJoCo
    python evaluate_model.py --backend mujoco      # the real sim

The analytic model is the gold standard (upper bound). Compare the MLP to it.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os

import numpy as np

import src.dynamics as dynamics
from src.arm_sim import ArmSim
from src.controllers import InverseDynamicsController
from src.kinematics import forward_kinematics
from src.trajectories import figure_eight, figure_eight_joint

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGIMES = {
    "nominal": dict(period=6.0, A=0.5, B=0.35),
    "fast":    dict(period=2.5, A=0.7, B=0.5),   # faster + larger => OOD
}


def closed_loop_track(model, backend, regime, duration=8.0,
                      kp=400.0, kd=40.0, elbow_up=True):
    sim = ArmSim(backend=backend)
    ctrl = InverseDynamicsController(kp=kp, kd=kd, model=model)
    dt = sim.dt

    q0, qd0, _ = figure_eight_joint(0.0, elbow_up=elbow_up, **regime)
    sim.reset(q0, qd0)

    ee, ee_des = [], []
    n = int(duration / dt)
    for i in range(n):
        t = i * dt
        q, qd = sim.q, sim.qd
        q_d, qd_d, qdd_d = figure_eight_joint(t, elbow_up=elbow_up, **regime)
        sim.set_torque(ctrl(q, qd, q_d, qd_d, qdd_d))
        x_des, _, _ = figure_eight(t, **regime)
        ee.append(forward_kinematics(q)); ee_des.append(x_des)
        sim.step()

    ee, ee_des = np.array(ee), np.array(ee_des)
    rms = float(np.sqrt(np.mean(np.sum((ee - ee_des) ** 2, axis=1))))
    return rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="analytic",
                    choices=["mujoco", "analytic"])
    args = ap.parse_args()

    models = {"analytic": dynamics}
    mlp_path = os.path.join(HERE, "models", "mlp.pt")
    if os.path.exists(mlp_path):
        from src.mlp_model import MLPInverseDynamics
        models["mlp"] = MLPInverseDynamics.load(mlp_path)
    else:
        print("[note] models/mlp.pt not found - train it first with train_mlp.py.")
        print("       Showing analytic baseline only.\n")

    print(f"End-effector RMS tracking error (mm), backend={args.backend}\n")
    header = f"{'model':10s}" + "".join(f"{r:>12s}" for r in REGIMES)
    print(header)
    print("-" * len(header))
    for mname, model in models.items():
        row = f"{mname:10s}"
        for rname, regime in REGIMES.items():
            rms = closed_loop_track(model, args.backend, regime)
            row += f"{rms * 1000:12.3f}"
        print(row)

    print("\nReading it: analytic should be tiny in both columns. A good MLP is")
    print("close to analytic under 'nominal' but degrades under 'fast' — that")
    print("in/out-of-distribution gap is the headline result of Week 8.")


if __name__ == "__main__":
    main()
