"""Collect (q, q̇, q̈, τ) data to train an inverse-dynamics model.

Two excitation strategies (mix of both gives good coverage):

  1. multisine torque  - open-loop sum-of-sines torques; produces fast, varied
     motion that sweeps a wide range of velocities and accelerations.
  2. pd setpoints      - gravity-comp PD driving the arm to random reachable
     targets; covers the realistic operating region and near-static poses.

We generate three datasets:
  - train      : moderate speeds/torques (the training distribution)
  - test_id    : same distribution, held out (in-distribution generalization)
  - test_ood   : deliberately faster & higher-torque than training
                 (out-of-distribution — this is where the black-box MLP breaks)

q̈ is read directly from the simulator, never finite-differenced.

Usage:
    python data_collection.py --backend mujoco   # real deliverable
    python data_collection.py --backend analytic # fast, no MuJoCo needed
"""

import argparse
import os

import numpy as np

from arm_sim import ArmSim
import controllers

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")

WORKSPACE_R = (0.5, 1.8)  # reachable radii to sample targets from (l1+l2 = 2)


def _random_reachable_target(rng):
    r = rng.uniform(*WORKSPACE_R)
    th = rng.uniform(0, np.pi)  # upper half-plane, avoids the arm folding down
    return np.array([r * np.cos(th), r * np.sin(th)])


def collect_multisine(sim, rng, n_bursts, burst_len, tau_amp, freq_range):
    """Open-loop multisine torques. Re-randomizes and re-seeds state each burst
    so we don't just orbit one attractor."""
    dt = sim.dt
    Q, QD, QDD, TAU = [], [], [], []
    n_sines = 5
    for _ in range(n_bursts):
        sim.reset(rng.uniform(-np.pi, np.pi, 2), rng.uniform(-1, 1, 2))
        freqs = rng.uniform(*freq_range, size=(2, n_sines))
        phases = rng.uniform(0, 2 * np.pi, size=(2, n_sines))
        amps = rng.uniform(0.3, 1.0, size=(2, n_sines))
        amps *= tau_amp / amps.sum(axis=1, keepdims=True)  # cap total amplitude
        for k in range(burst_len):
            t = k * dt
            tau = np.array([
                np.sum(amps[j] * np.sin(2 * np.pi * freqs[j] * t + phases[j]))
                for j in range(2)
            ])
            sim.set_torque(tau)
            Q.append(sim.q); QD.append(sim.qd)
            QDD.append(sim.acceleration()); TAU.append(sim.joint_torque())
            sim.step()
    return map(np.array, (Q, QD, QDD, TAU))


def collect_pd_setpoints(sim, rng, n_targets, hold_len, kp=80.0, kd=20.0):
    """Gravity-comp PD to random reachable targets. Gains are kept modest so the
    approach is smooth and torques stay within the actuator limit — this fills
    in near-static and low-speed poses, complementing the multisine bursts."""
    from kinematics import inverse_kinematics
    ctrl = controllers.GravityCompPDController(kp=kp, kd=kd)
    dt = sim.dt
    Q, QD, QDD, TAU = [], [], [], []
    sim.reset(np.array([0.5, 0.5]), np.zeros(2))
    for _ in range(n_targets):
        try:
            q_target = inverse_kinematics(_random_reachable_target(rng))
        except ValueError:
            continue
        for _ in range(hold_len):
            q, qd = sim.q, sim.qd
            tau = ctrl(q, qd, q_target, np.zeros(2), np.zeros(2))
            sim.set_torque(tau)
            Q.append(q); QD.append(qd)
            QDD.append(sim.acceleration()); TAU.append(sim.joint_torque())
            sim.step()
    return map(np.array, (Q, QD, QDD, TAU))


def _combine(parts):
    mats = [tuple(p) for p in parts]
    Q = np.concatenate([m[0] for m in mats])
    QD = np.concatenate([m[1] for m in mats])
    QDD = np.concatenate([m[2] for m in mats])
    TAU = np.concatenate([m[3] for m in mats])
    return Q, QD, QDD, TAU


def make_all(backend="mujoco", seed=0):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)
    sim = ArmSim(backend=backend)

    # --- training distribution: moderate torques & frequencies ---
    train = _combine([
        collect_multisine(sim, rng, n_bursts=60, burst_len=400,
                          tau_amp=8.0, freq_range=(0.1, 1.5)),
        collect_pd_setpoints(sim, rng, n_targets=60, hold_len=250),
    ])
    # --- in-distribution held-out test ---
    test_id = _combine([
        collect_multisine(sim, rng, n_bursts=15, burst_len=400,
                          tau_amp=8.0, freq_range=(0.1, 1.5)),
        collect_pd_setpoints(sim, rng, n_targets=15, hold_len=250),
    ])
    # --- OUT-of-distribution: faster and stronger than anything in training ---
    test_ood = _combine([
        collect_multisine(sim, rng, n_bursts=20, burst_len=400,
                          tau_amp=16.0, freq_range=(1.5, 3.0)),
    ])

    for name, d in [("train", train), ("test_id", test_id), ("test_ood", test_ood)]:
        q, qd, qdd, tau = d
        np.savez(os.path.join(OUT, f"{name}.npz"), q=q, qd=qd, qdd=qdd, tau=tau)
        print(f"{name:9s}: {len(q):6d} samples | "
              f"|qd| max {np.abs(qd).max():5.2f} | "
              f"|qdd| max {np.abs(qdd).max():6.2f} | "
              f"|tau| max {np.abs(tau).max():6.2f}")
    print(f"\nSaved datasets to {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mujoco",
                    choices=["mujoco", "analytic"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_all(backend=args.backend, seed=args.seed)
