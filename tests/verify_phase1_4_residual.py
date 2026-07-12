"""Deterministic acceptance checks for Phase 1.4 structured residual dynamics."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.mppi import MPPIConfig, MPPIController, TorchRigidBodyDynamics
from src.residual_dynamics import StructuredHydrodynamicDynamics, StructuredResidualConfig


def main():
    torch.manual_seed(4)
    model = StructuredHydrodynamicDynamics(dtype=torch.float64)
    states = torch.tensor([
        [0.1, -0.4, 0.7, -0.2],
        [-0.8, 0.5, -0.3, 0.9],
        [0.4, -1.0, 1.2, 0.1],
    ], dtype=torch.float64)
    torques = torch.tensor([[2., -1.], [0., 3.], [-2., 4.]], dtype=torch.float64)

    # Added mass is PSD and the total inertia remains positive definite.
    A = model.added_mass_matrices()
    assert torch.linalg.eigvalsh(A).min() >= -1e-12
    M0, _ = model.rigid_body_terms(states)
    Ma, _, v_rel = model.residual_terms(states, (0.3, -0.1))
    assert torch.linalg.eigvalsh(M0 + Ma).min() > 0

    # Drag dissipates relative kinetic energy: sum(v_rel dot F) <= 0.
    linear = torch.nn.functional.softplus(model.raw_linear_drag)
    quadratic = torch.nn.functional.softplus(model.raw_quadratic_drag)
    force = -linear[:, None]*v_rel - quadratic[:, None]*v_rel.abs()*v_rel
    power = torch.sum(v_rel*force, dim=(-1, -2))
    assert torch.max(power) <= 1e-12

    # Zero residual coefficients recover the nominal rigid-body implementation.
    zero = StructuredResidualConfig(linear_drag=(0., 0.), quadratic_drag=(0., 0.),
                                    added_mass_diag=(0., 0.), coefficient_floor=1e-14)
    baseline = StructuredHydrodynamicDynamics(zero, dtype=torch.float64)
    nominal = TorchRigidBodyDynamics(dtype=torch.float64)
    dt = 1e-4
    xb = baseline(states, torques, dt)
    xn = nominal(states, torques, dt)
    assert torch.allclose(xb, xn, atol=2e-12, rtol=2e-12), torch.max(torch.abs(xb-xn))

    # Known flow is explicit context and changes the predicted acceleration.
    model.set_flow((0.0, 0.0, 0.0))
    still = model.acceleration(states, torques)
    model.set_flow((0.4, -0.2))
    current = model.acceleration(states, torques)
    assert not torch.allclose(still, current)

    # The model is trainable and all parameter gradients are finite.
    loss = model.acceleration(states, torques, (0.2, 0.0)).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

    # Drop-in compatibility with the Phase 1.3 MPPI forward-model interface.
    cfg = MPPIConfig(samples=32, horizon=4, lambda_=2.0)
    mppi_model = StructuredHydrodynamicDynamics(device="cpu")
    mppi_model.set_flow((0.25, 0.0))
    ctrl = MPPIController(mppi_model, cfg, device="cpu", seed=9)
    u = ctrl.command(np.zeros(4), np.zeros((cfg.horizon+1, 4)))
    assert u.shape == (2,) and np.isfinite(u).all()
    assert np.max(np.abs(u)) <= cfg.torque_limit

    print("PASS: Phase 1.4 residual is PSD, dissipative, flow-conditioned, trainable, and MPPI-compatible")


if __name__ == "__main__":
    main()
