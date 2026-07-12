"""Deterministic acceptance checks for the Phase 1.3 rate hierarchy."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.mppi import (MPPIConfig, MPPIController, MultiRateMPPIController,
                      TorchRigidBodyDynamics)
from src import dynamics as numpy_dynamics


def main():
    # Small test budget; production defaults remain locked at 1024 samples.
    cfg = MPPIConfig(samples=64, horizon=8, lambda_=2.0,
                     control_rate_hz=50, tracker_rate_hz=1000)
    assert MPPIConfig().samples >= 1000
    assert cfg.ticks_per_plan == 20
    dynamics = TorchRigidBodyDynamics()

    # Batched dynamics must remain finite and preserve shape.
    x = torch.zeros(7, 4)
    u = torch.zeros(7, 2)
    xn = dynamics(x, u, cfg.control_dt)
    assert xn.shape == x.shape and torch.isfinite(xn).all()
    # The batched model must reproduce the repository's nominal acceleration.
    rng = np.random.default_rng(5)
    states = rng.uniform([-1.0, -1.0, -2.0, -2.0], [1.0, 1.0, 2.0, 2.0], (12, 4))
    controls = rng.uniform(-10.0, 10.0, (12, 2))
    predicted = dynamics(torch.tensor(states, dtype=torch.float64),
                         torch.tensor(controls, dtype=torch.float64), 1e-5).numpy()
    qdd_torch = (predicted[:, 2:] - states[:, 2:]) / 1e-5
    qdd_numpy = np.stack([numpy_dynamics.forward_dynamics(s[:2], s[2:], u)
                           for s, u in zip(states, controls)])
    assert np.allclose(qdd_torch, qdd_numpy, atol=1e-8)

    reference = np.zeros((cfg.horizon+1, 4))
    a = MPPIController(dynamics, cfg, device="cpu", seed=7)
    b = MPPIController(dynamics, cfg, device="cpu", seed=7)
    ua = a.command(np.zeros(4), reference)
    ub = b.command(np.zeros(4), reference)
    assert np.array_equal(ua, ub), (ua, ub)
    assert np.all(np.isfinite(ua)) and np.max(np.abs(ua)) <= cfg.torque_limit

    bridge = MultiRateMPPIController(MPPIController(dynamics, cfg, device="cpu", seed=3))
    torques, flags = [], []
    for _ in range(45):
        torque, replanned = bridge.step(np.zeros(4), reference)
        torques.append(torque); flags.append(replanned)
    assert np.flatnonzero(flags).tolist() == [0, 20, 40]
    assert bridge.plan_count == 3
    assert np.isfinite(torques).all()
    # Interpolation starts continuously at zero rather than jumping to target.
    assert np.array_equal(torques[0], np.zeros(2))

    try:
        MPPIConfig(control_rate_hz=60, tracker_rate_hz=1000)
        raise AssertionError("non-integer rate ratio was accepted")
    except ValueError:
        pass
    print("PASS: Phase 1.3 MPPI is deterministic, batched, bounded, and 50 Hz / 1 kHz")


if __name__ == "__main__":
    main()
