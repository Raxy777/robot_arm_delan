"""GPU-batched MPPI and a deterministic 50 Hz / 1 kHz rate bridge.

MPPI is model-agnostic: ``dynamics(state, torque, dt)`` must accept batched
PyTorch tensors and return the next ``[q, qdot]`` state.  This lets the nominal
model be replaced by the Phase-1.2 structured residual forward model without
changing the optimizer or rate hierarchy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch


@dataclass(frozen=True)
class MPPIConfig:
    samples: int = 1024
    horizon: int = 25
    lambda_: float = 1.0
    control_rate_hz: int = 50
    tracker_rate_hz: int = 1000
    noise_sigma: tuple[float, float] = (8.0, 8.0)
    torque_limit: float = 60.0
    state_cost: tuple[float, float, float, float] = (80.0, 80.0, 2.0, 2.0)
    terminal_scale: float = 5.0
    control_cost: float = 2.0e-3

    def __post_init__(self):
        if self.samples < 2 or self.horizon < 1:
            raise ValueError("samples must be >=2 and horizon must be >=1")
        if self.lambda_ <= 0 or self.control_rate_hz <= 0 or self.tracker_rate_hz <= 0:
            raise ValueError("temperature and rates must be positive")
        if self.tracker_rate_hz % self.control_rate_hz:
            raise ValueError("tracker rate must be an integer multiple of MPPI rate")
        if len(self.noise_sigma) != 2 or min(self.noise_sigma) <= 0:
            raise ValueError("noise_sigma must contain two positive values")

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_rate_hz

    @property
    def ticks_per_plan(self) -> int:
        return self.tracker_rate_hz // self.control_rate_hz


class TorchRigidBodyDynamics:
    """Batched nominal two-link forward dynamics used as the safe baseline."""

    def __init__(self, device: str | torch.device = "cpu", dtype=torch.float32):
        from src.params import PARAMS as p
        self.device, self.dtype = torch.device(device), dtype
        self.constants = tuple(float(x) for x in
            (p.m1, p.m2, p.l1, p.lc1, p.lc2, p.I1, p.I2, p.g))

    def __call__(self, state: torch.Tensor, torque: torch.Tensor, dt: float):
        m1, m2, l1, lc1, lc2, I1, I2, g = self.constants
        q1, q2, q1d, q2d = state.unbind(-1)
        c2, s2 = torch.cos(q2), torch.sin(q2)
        m11 = m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2) + I1 + I2
        m12 = m2*(lc2**2 + l1*lc2*c2) + I2
        m22 = torch.full_like(m11, m2*lc2**2 + I2)
        h = m2*l1*lc2*s2
        coriolis_1 = -h*q2d*q1d - h*(q1d + q2d)*q2d
        coriolis_2 = h*q1d*q1d
        gravity_1 = (m1*lc1 + m2*l1)*g*torch.cos(q1) + m2*lc2*g*torch.cos(q1+q2)
        gravity_2 = m2*lc2*g*torch.cos(q1+q2)
        rhs1 = torque[..., 0] - coriolis_1 - gravity_1
        rhs2 = torque[..., 1] - coriolis_2 - gravity_2
        det = m11*m22 - m12*m12
        qdd1 = (m22*rhs1 - m12*rhs2) / det
        qdd2 = (-m12*rhs1 + m11*rhs2) / det
        qd_next = state[..., 2:] + dt*torch.stack((qdd1, qdd2), -1)
        q_next = state[..., :2] + dt*qd_next  # semi-implicit Euler
        return torch.cat((q_next, qd_next), -1)


class MPPIController:
    """Path-integral optimizer with warm-started control sequence."""

    def __init__(self, dynamics: Callable, config=MPPIConfig(), device=None, seed=0):
        self.config = config
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dynamics = dynamics
        self.dtype = torch.float32
        self.u = torch.zeros(config.horizon, 2, device=self.device, dtype=self.dtype)
        self.sigma = torch.tensor(config.noise_sigma, device=self.device, dtype=self.dtype)
        self.q_state = torch.tensor(config.state_cost, device=self.device, dtype=self.dtype)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.last_cost = float("nan")

    @torch.no_grad()
    def command(self, state, reference):
        """Optimize and return the first torque. Reference shape is (H+1, 4)."""
        c = self.config
        x0 = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        ref = torch.as_tensor(reference, dtype=self.dtype, device=self.device)
        if x0.shape != (4,) or ref.shape != (c.horizon + 1, 4):
            raise ValueError(f"expected state (4,) and reference ({c.horizon+1},4)")

        eps = torch.randn(c.samples, c.horizon, 2, generator=self.generator,
                          device=self.device, dtype=self.dtype) * self.sigma
        controls = torch.clamp(self.u.unsqueeze(0) + eps, -c.torque_limit, c.torque_limit)
        x = x0.expand(c.samples, -1).clone()
        cost = torch.zeros(c.samples, device=self.device)
        for t in range(c.horizon):
            error = x - ref[t]
            cost += torch.sum(self.q_state * error.square(), dim=-1)
            cost += c.control_cost * torch.sum(controls[:, t].square(), dim=-1)
            x = self.dynamics(x, controls[:, t], c.control_dt)
        terminal_error = x - ref[-1]
        cost += c.terminal_scale * torch.sum(self.q_state * terminal_error.square(), dim=-1)

        rho = torch.min(cost)
        weights = torch.exp(-(cost-rho)/c.lambda_)
        weights /= torch.sum(weights).clamp_min(torch.finfo(weights.dtype).tiny)
        self.u += torch.sum(weights[:, None, None] * eps, dim=0)
        self.u.clamp_(-c.torque_limit, c.torque_limit)
        action = self.u[0].clone()
        self.u[:-1] = self.u[1:].clone()
        self.u[-1].zero_()
        self.last_cost = float(torch.sum(weights*cost).item())
        return action.cpu().numpy()


class InterpolatingTorqueTracker:
    """Linearly bridge successive MPPI torques at the 1 kHz inner-loop rate."""

    def __init__(self, config=MPPIConfig()):
        self.steps = config.ticks_per_plan
        self.torque_limit = config.torque_limit
        self.previous = np.zeros(2, dtype=float)
        self.target = np.zeros(2, dtype=float)
        self.tick = self.steps

    def set_target(self, torque):
        # Start from the currently emitted value, avoiding a command jump.
        self.previous = self.value()
        self.target = np.clip(np.asarray(torque, float),
                              -self.torque_limit, self.torque_limit).copy()
        self.tick = 0

    def value(self):
        alpha = min(1.0, self.tick / self.steps)
        return (1.0-alpha)*self.previous + alpha*self.target

    def step(self):
        value = self.value()
        self.tick = min(self.steps, self.tick+1)
        return value


class MultiRateMPPIController:
    """Run MPPI periodically and emit an interpolated command every inner tick."""

    def __init__(self, mppi: MPPIController):
        self.mppi = mppi
        self.config = mppi.config
        self.tracker = InterpolatingTorqueTracker(self.config)
        self.inner_tick = 0
        self.plan_count = 0

    def step(self, state, reference):
        replanned = self.inner_tick % self.config.ticks_per_plan == 0
        if replanned:
            self.tracker.set_target(self.mppi.command(state, reference))
            self.plan_count += 1
        torque = self.tracker.step()
        self.inner_tick += 1
        return torque, replanned
