"""Torch-free validation of the DeLaN Euler-Lagrange assembly.

DeLaN builds torque from a learned M(q) and V(q) via

    tau = M qdd + [ Mdot qd - 1/2 d/dq(qd^T M qd) ] + dV/dq

If that formula is correct, then feeding the *analytic* M and V through the
SAME assembly (with derivatives taken by finite differences instead of autograd)
must reproduce dynamics.inverse_dynamics exactly. This isolates the physics/
math from the neural network and from PyTorch — if this passes, any error in
the real model is in the network wiring, not the equations.

Run:  python _verify_week9_math.py
"""

import numpy as np

import dynamics
from dynamics import mass_matrix, potential_energy


def dV_dq(q, h=1e-6):
    g = np.zeros(2)
    for k in range(2):
        e = np.zeros(2); e[k] = h
        g[k] = (potential_energy(q + e) - potential_energy(q - e)) / (2 * h)
    return g


def dM_dqk(q, k, h=1e-6):
    e = np.zeros(2); e[k] = h
    return (mass_matrix(q + e) - mass_matrix(q - e)) / (2 * h)


def el_inverse_dynamics(q, qd, qdd):
    """Reassemble tau from M and V by the DeLaN formula, finite-diff derivs."""
    M = mass_matrix(q)

    # Mdot qd = sum_k (dM/dq_k) qd * qd_k
    Mdot = sum(dM_dqk(q, k) * qd[k] for k in range(2))
    Mdot_qd = Mdot @ qd

    # 1/2 d/dq (qd^T M qd), component i = 1/2 qd^T (dM/dq_i) qd
    dquad = np.array([0.5 * qd @ (dM_dqk(q, i) @ qd) for i in range(2)])

    g = dV_dq(q)

    return M @ qdd + (Mdot_qd - dquad) + g


def main():
    rng = np.random.default_rng(0)
    worst = 0.0
    worst_case = None
    for _ in range(2000):
        q = rng.uniform(-np.pi, np.pi, 2)
        qd = rng.uniform(-3, 3, 2)
        qdd = rng.uniform(-5, 5, 2)
        tau_ref = dynamics.inverse_dynamics(q, qd, qdd)
        tau_el = el_inverse_dynamics(q, qd, qdd)
        err = np.max(np.abs(tau_ref - tau_el))
        if err > worst:
            worst, worst_case = err, (q, qd, qdd)

    print("--- DeLaN Euler-Lagrange assembly vs analytic inverse dynamics ---")
    print(f"  samples        = 2000")
    print(f"  max |tau diff| = {worst:.3e} N m  (finite-diff limited, expect ~1e-6)")
    print(f"  {'PASS' if worst < 1e-4 else 'FAIL'} (threshold 1e-4)")
    if worst >= 1e-4:
        print("  worst case (q, qd, qdd):", worst_case)


if __name__ == "__main__":
    main()
