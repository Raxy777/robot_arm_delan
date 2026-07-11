"""Energy-conservation drift: the structural payoff of DeLaN, made visible.

A passive arm (zero applied torque) swinging under gravity is a conservative
system: total mechanical energy E = KE + PE is constant. Whether a *model*
respects that depends on its structure, not its accuracy on torque data.

We roll the arm forward from rest using each model's OWN forward dynamics and
watch E(t):

  * analytic  — conserves by construction (drift is pure RK4 truncation, ~1e-6).
  * DeLaN     — derives q̈ from a learned M(q) and V(q); because it integrates a
                genuine Lagrangian, E is (approximately) conserved too. We read
                KE = ½ q̇ᵀM(q)q̇ and PE = V(q) straight out of the SAME networks
                that produce the dynamics, so this is self-consistent.
  * MLP       — N/A. It regresses torque with no mass matrix and no potential;
                there is literally no energy function to evaluate. That absence
                is the point — a black box can be accurate yet has no physics to
                conserve, and nothing to inspect.

DeLaN's forward dynamics from its inverse model: with τ=0,
    bias(q,q̇) = inverse_dynamics(q, q̇, 0) = c(q,q̇) + g(q)
    q̈ = M(q)⁻¹ ( −bias )
Energy offset is unobservable (only ∂V/∂q affects torque), so we plot E(t)−E(0):
drift, not absolute level, is what conservation is about.

    python energy.py                 # writes results/energy_drift.csv
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import os

import numpy as np

import src.dynamics as dynamics
from src.delan_model import DeLaNInverseDynamics

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "results")


def _rk4_passive(qdd_fn, q0, qd0, dt, n):
    """Integrate a passive system s=[q,qd] with q̈ = qdd_fn(q,qd)."""
    def deriv(s):
        return np.concatenate([s[2:], qdd_fn(s[:2], s[2:])])
    s = np.concatenate([q0, qd0])
    traj = np.empty((n, 4))
    for i in range(n):
        traj[i] = s
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return traj


def analytic_energy_series(q0, qd0, dt, n):
    qdd_fn = lambda q, qd: dynamics.forward_dynamics(q, qd, np.zeros(2))
    traj = _rk4_passive(qdd_fn, q0, qd0, dt, n)
    E = np.array([dynamics.total_energy(traj[i, :2], traj[i, 2:]) for i in range(n)])
    return E


def delan_energy_series(model, q0, qd0, dt, n):
    """Passive rollout on DeLaN's own dynamics; energy read from its M and V."""
    net = model.net

    def qdd_fn(q, qd):
        M = model.mass_matrix(q)                              # (2,2)
        bias = model.inverse_dynamics(q, qd, np.zeros(2))     # c + g at qdd=0
        return np.linalg.solve(M, -bias)

    def energy(q, qd):
        M = model.mass_matrix(q)
        ke = 0.5 * qd @ (M @ qd)
        import torch
        with torch.no_grad():
            V = float(net.potential(torch.as_tensor(q[None], dtype=torch.float32)).item())
        return ke + V

    traj = _rk4_passive(qdd_fn, q0, qd0, dt, n)
    E = np.array([energy(traj[i, :2], traj[i, 2:]) for i in range(n)])
    return E


def main(duration=4.0, dt=0.002):
    os.makedirs(RESULTS, exist_ok=True)
    n = int(duration / dt)
    ts = np.arange(n) * dt

    # released from rest, off-axis so gravity does work and it swings
    q0 = np.array([0.4, 0.6])
    qd0 = np.array([0.0, 0.0])

    series = {"t": ts}
    E_an = analytic_energy_series(q0, qd0, dt, n)
    series["analytic"] = E_an

    delan_path = os.path.join(MODELS, "delan.pt")
    have_delan = os.path.exists(delan_path)
    if have_delan:
        model = DeLaNInverseDynamics.load(delan_path)
        series["delan"] = delan_energy_series(model, q0, qd0, dt, n)

    def drift(E):
        d = E - E[0]
        return float(np.max(np.abs(d))), float(np.std(E))

    print("Passive energy-conservation drift (released from rest at q0=%s)\n" % q0)
    print(f"{'model':10s}{'E(0) [J]':>12s}{'max|dE| [J]':>14s}{'std(E) [J]':>12s}")
    print("-" * 48)
    for name in ("analytic", "delan"):
        if name in series:
            mx, sd = drift(series[name])
            print(f"{name:10s}{series[name][0]:12.4f}{mx:14.6f}{sd:12.6f}")
    print(f"{'mlp':10s}{'-':>12s}{'N/A':>14s}{'N/A':>12s}   (no energy function)")

    # write tidy CSV
    cols = ["t"] + [k for k in ("analytic", "delan") if k in series]
    with open(os.path.join(RESULTS, "energy_drift.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for i in range(n):
            w.writerow([series[c][i] for c in cols])
    print(f"\nWrote {os.path.join(RESULTS, 'energy_drift.csv')}")
    if not have_delan:
        print("[note] models/delan.pt not found — analytic-only.")
    print("\nRead it: analytic drift is pure integrator noise. DeLaN's drift stays")
    print("small because it integrates a real learned Lagrangian; a torque-regressing")
    print("MLP has no energy to conserve or to plot at all.")


if __name__ == "__main__":
    main()
