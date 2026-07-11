"""Correctness checks for the hand-derived dynamics.

Two independent checks:

  1. energy_conservation_test()  - pure python, no MuJoCo. Integrates the
     PASSIVE arm (tau = 0, zero damping) with the analytic forward dynamics.
     A correct M/C/g triple conserves total mechanical energy, so energy drift
     over a long rollout should be tiny. This validates that C is consistent
     with M (the skew-symmetry property) and that g matches the potential.

  2. compare_with_mujoco()  - requires mujoco. For random (q, qd, qdd) it
     checks that the analytic inverse dynamics equals MuJoCo's mj_inverse.
     If your XML and params.py agree, the max error is ~1e-9.

Run:  python verify_dynamics.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import src.dynamics as dynamics


def energy_conservation_test(duration=20.0, dt=1e-4, q0=(2.2, -1.3),
                             qd0=(0.0, 0.0), verbose=True):
    q = np.array(q0, dtype=float)
    qd = np.array(qd0, dtype=float)

    def deriv(state):
        q_, qd_ = state[:2], state[2:]
        qdd_ = dynamics.forward_dynamics(q_, qd_, np.zeros(2))
        return np.concatenate([qd_, qdd_])

    state = np.concatenate([q, qd])
    E0 = dynamics.total_energy(q, qd)
    max_dev = 0.0
    n = int(duration / dt)
    for _ in range(n):
        # RK4
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        E = dynamics.total_energy(state[:2], state[2:])
        max_dev = max(max_dev, abs(E - E0))

    rel = max_dev / max(abs(E0), 1e-9)
    if verbose:
        print("--- Energy conservation (passive arm, tau=0) ---")
        print(f"  E0              = {E0:+.6f} J")
        print(f"  max |E - E0|    = {max_dev:.3e} J")
        print(f"  relative drift  = {rel:.3e}")
        print(f"  {'PASS' if rel < 1e-3 else 'FAIL'} "
              f"(threshold 1e-3 relative over {duration:.0f}s)")
    return rel


def compare_with_mujoco(n_samples=500, seed=0, verbose=True):
    try:
        import os
        import mujoco
    except ImportError:
        print("[skip] mujoco not installed; skipping MuJoCo cross-check.")
        return None

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = mujoco.MjModel.from_xml_path(os.path.join(here, "model", "arm2.xml"))
    data = mujoco.MjData(model)

    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n_samples):
        q = rng.uniform(-np.pi, np.pi, size=2)
        qd = rng.uniform(-3, 3, size=2)
        qdd = rng.uniform(-5, 5, size=2)

        # MuJoCo inverse dynamics: set state + accel, call mj_inverse,
        # read qfrc_inverse (the joint-space force needed for that qacc).
        data.qpos[:] = q
        data.qvel[:] = qd
        data.qacc[:] = qdd
        mujoco.mj_inverse(model, data)
        tau_mj = data.qfrc_inverse.copy()

        tau_analytic = dynamics.inverse_dynamics(q, qd, qdd)
        max_err = max(max_err, np.max(np.abs(tau_mj - tau_analytic)))

    if verbose:
        print("--- Analytic vs MuJoCo inverse dynamics ---")
        print(f"  samples        = {n_samples}")
        print(f"  max |tau diff| = {max_err:.3e} N m")
        print(f"  {'PASS' if max_err < 1e-6 else 'FAIL'} (threshold 1e-6)")
    return max_err


if __name__ == "__main__":
    energy_conservation_test()
    print()
    compare_with_mujoco()
