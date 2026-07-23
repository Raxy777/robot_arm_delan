"""Flow-conditioned structured hydrodynamic forward dynamics.

This module is the Phase 1.4 bridge between the rigid-body baseline and MPPI.
The residual is deliberately small and interpretable:

* link drag uses ``v_rel = J(q) qdot - v_fluid``;
* drag forces map to joints through ``J.T @ F``;
* task-space added-mass matrices are ``L @ L.T`` (positive semidefinite);
* the rigid-body and residual inertia are summed before solving for qdd.

MuJoCo's fluid model remains a simplified force model, not CFD ground truth.
The known uniform-flow vector is explicit context so one state is not assigned
multiple incompatible accelerations under different currents.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.params import PARAMS as P


def _inverse_softplus(value: float) -> float:
    value = torch.as_tensor(float(value), dtype=torch.float64)
    return float(torch.log(torch.expm1(value)).item())


@dataclass(frozen=True)
class StructuredResidualConfig:
    """Initial physical coefficients for two planar links.

    Drag coefficients are non-negative. ``added_mass_diag`` initializes a
    diagonal task-space added-mass matrix for each link. Set all coefficients
    to zero for an exact rigid-body baseline (up to the numerical floor).
    """

    linear_drag: tuple[float, float] = (0.10, 0.10)
    quadratic_drag: tuple[float, float] = (0.25, 0.25)
    added_mass_diag: tuple[float, float] = (0.05, 0.05)
    coefficient_floor: float = 1.0e-8
    added_mass_spd_floor: float = 1.0e-12

    def __post_init__(self):
        values = self.linear_drag + self.quadratic_drag + self.added_mass_diag
        if len(self.linear_drag) != 2 or len(self.quadratic_drag) != 2:
            raise ValueError("drag coefficients must contain one value per link")
        if len(self.added_mass_diag) != 2:
            raise ValueError("added_mass_diag must contain one value per link")
        if (min(values) < 0 or self.coefficient_floor <= 0
                or self.added_mass_spd_floor <= 0):
            raise ValueError("physical coefficients must be non-negative and floors positive")


class StructuredHydrodynamicDynamics(nn.Module):
    """Batched two-link dynamics with trainable physical residual parameters.

    ``forward(state, torque, dt)`` matches the MPPI model interface. Call
    :meth:`set_flow` with the known world-frame uniform flow before planning.
    For training or mixed-flow batches, pass ``flow_velocity`` explicitly.
    """

    def __init__(self, config=StructuredResidualConfig(), device="cpu", dtype=torch.float32):
        super().__init__()
        self.config = config
        self.dtype = dtype
        floor = config.coefficient_floor

        def raw(values):
            return torch.tensor([_inverse_softplus(max(v, floor)) for v in values], dtype=dtype)

        self.raw_linear_drag = nn.Parameter(raw(config.linear_drag))
        self.raw_quadratic_drag = nn.Parameter(raw(config.quadratic_drag))
        # One lower-triangular 2x2 task-space factor per link. Diagonal entries
        # use softplus; the off-diagonal is unconstrained.
        diag = [_inverse_softplus(max(v, floor) ** 0.5) for v in config.added_mass_diag]
        factors = torch.zeros(2, 3, dtype=dtype)
        factors[:, 0] = torch.tensor(diag, dtype=dtype)
        factors[:, 2] = torch.tensor(diag, dtype=dtype)
        self.raw_added_mass_factor = nn.Parameter(factors)
        self.register_buffer("flow_velocity", torch.zeros(2, dtype=dtype))
        constants = (P.m1, P.m2, P.l1, P.lc1, P.lc2, P.I1, P.I2, P.g)
        self.register_buffer("constants", torch.tensor(constants, dtype=dtype))
        self.to(device=device, dtype=dtype)

    @property
    def device(self):
        return self.constants.device

    def set_flow(self, flow_velocity):
        """Set known uniform flow context (x/y; an optional z is ignored)."""
        flow = torch.as_tensor(flow_velocity, dtype=self.dtype, device=self.device).flatten()
        if flow.numel() not in (2, 3):
            raise ValueError("flow_velocity must have two or three components")
        if not torch.isfinite(flow).all():
            raise ValueError("flow_velocity must be finite")
        self.flow_velocity.copy_(flow[:2])
        return self

    def coefficients(self):
        return {
            "linear_drag": F.softplus(self.raw_linear_drag),
            "quadratic_drag": F.softplus(self.raw_quadratic_drag),
            "added_mass": self.added_mass_matrices(),
        }

    def added_mass_matrices(self):
        raw = self.raw_added_mass_factor
        L = raw.new_zeros(2, 2, 2)
        L[:, 0, 0] = F.softplus(raw[:, 0])
        L[:, 1, 0] = raw[:, 1]
        L[:, 1, 1] = F.softplus(raw[:, 2])
        identity = torch.eye(2, dtype=L.dtype, device=L.device).expand(2, -1, -1)
        return L @ L.transpose(-1, -2) + self.config.added_mass_spd_floor * identity

    def link_jacobians(self, q):
        """Link-COM Jacobians, shape ``(..., 2 links, xy, 2 joints)``."""
        l1, lc1, lc2 = self.constants[2], self.constants[3], self.constants[4]
        q1, q2 = q.unbind(-1)
        s1, c1 = torch.sin(q1), torch.cos(q1)
        s12, c12 = torch.sin(q1 + q2), torch.cos(q1 + q2)
        zero = torch.zeros_like(q1)
        J1 = torch.stack((torch.stack((-lc1*s1, zero), -1),
                          torch.stack(( lc1*c1, zero), -1)), -2)
        J2 = torch.stack((torch.stack((-l1*s1-lc2*s12, -lc2*s12), -1),
                          torch.stack(( l1*c1+lc2*c12,  lc2*c12), -1)), -2)
        return torch.stack((J1, J2), -3)

    def rigid_body_terms(self, state):
        """Return rigid mass matrix and bias ``C(q,qd)qd + g(q)``."""
        m1, m2, l1, lc1, lc2, I1, I2, gravity = self.constants
        q1, q2, q1d, q2d = state.unbind(-1)
        c2, s2 = torch.cos(q2), torch.sin(q2)
        m11 = m1*lc1.square() + m2*(l1.square()+lc2.square()+2*l1*lc2*c2) + I1 + I2
        m12 = m2*(lc2.square()+l1*lc2*c2) + I2
        m22 = torch.ones_like(m11)*(m2*lc2.square()+I2)
        M = torch.stack((torch.stack((m11, m12), -1),
                         torch.stack((m12, m22), -1)), -2)
        h = m2*l1*lc2*s2
        coriolis = torch.stack((-h*q2d*q1d-h*(q1d+q2d)*q2d, h*q1d.square()), -1)
        g = torch.stack(((m1*lc1+m2*l1)*gravity*torch.cos(q1)
                         + m2*lc2*gravity*torch.cos(q1+q2),
                         m2*lc2*gravity*torch.cos(q1+q2)), -1)
        return M, coriolis + g

    def residual_terms(self, state, flow_velocity=None):
        """Return added joint inertia, hydrodynamic torque, and relative velocity."""
        J = self.link_jacobians(state[..., :2])
        link_velocity = torch.einsum("...lij,...j->...li", J, state[..., 2:])
        if flow_velocity is None:
            flow = self.flow_velocity
        else:
            flow = torch.as_tensor(flow_velocity, dtype=state.dtype, device=state.device)
            if flow.shape[-1] == 3:
                flow = flow[..., :2]
            if flow.shape[-1] != 2:
                raise ValueError("flow_velocity must end in two or three components")
        while flow.ndim < link_velocity.ndim:
            flow = flow.unsqueeze(-2)
        v_rel = link_velocity - flow
        linear = F.softplus(self.raw_linear_drag)
        quadratic = F.softplus(self.raw_quadratic_drag)
        force = (-linear[..., None]*v_rel
                 - quadratic[..., None]*torch.abs(v_rel)*v_rel)
        tau_h = torch.einsum("...lij,...li->...j", J, force)
        A = self.added_mass_matrices()
        M_added = torch.einsum("...lai,lab,...lbj->...ij", J, A, J)
        return M_added, tau_h, v_rel

    def acceleration(self, state, torque, flow_velocity=None):
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        torque = torch.as_tensor(torque, dtype=self.dtype, device=self.device)
        M_rigid, bias = self.rigid_body_terms(state)
        M_added, tau_h, _ = self.residual_terms(state, flow_velocity)
        rhs = torque - bias + tau_h
        return torch.linalg.solve(M_rigid + M_added, rhs.unsqueeze(-1)).squeeze(-1)

    def forward(self, state, torque, dt, flow_velocity=None):
        state = torch.as_tensor(state, dtype=self.dtype, device=self.device)
        qdd = self.acceleration(state, torque, flow_velocity)
        qd_next = state[..., 2:] + float(dt)*qdd
        q_next = state[..., :2] + float(dt)*qd_next
        return torch.cat((q_next, qd_next), -1)
