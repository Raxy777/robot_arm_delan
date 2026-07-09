"""Deep Lagrangian Network (DeLaN) inverse-dynamics model.

Core idea (Lutter, Ritter, Peters, ICLR 2019): instead of regressing torque
directly like the MLP, we learn the *ingredients of the Lagrangian* and rebuild
the equations of motion from them. Two small networks:

    L_params(q) -> lower-triangular L(q)  ; then  M(q) = L L^T + eps I   (SPD!)
    V(q)        -> scalar potential energy

From these, torque comes from the Euler-Lagrange equation, NOT from a regressed
output. With M(q) and V(q) in hand:

    tau = M(q) qdd + c(q, qdot) + g(q)

where the velocity-product (Coriolis/centrifugal) and gravity terms are
*derived* from M and V by differentiation:

    g(q)         = dV/dq
    c(q, qdot)   = Mdot(q,qdot) qdot - 1/2 * d/dq ( qdot^T M(q) qdot )

with  Mdot qdot = sum_k (dM/dq_k) qdot * qdot_k .

Everything on the right is obtained with autograd, so we never hand-derive
Coriolis terms — the network structure guarantees they are consistent with the
learned M and V. That structural consistency is why DeLaN extrapolates where the
black-box MLP cannot.

Why this beats a plain MLP for the project story:
  * M(q) is guaranteed symmetric positive-definite (Cholesky parameterization),
    so the "inertia" is always physically valid.
  * Coriolis and gravity are not free parameters — they are forced to be the
    correct derivatives of M and V. Far fewer ways to be wrong => generalizes.
  * You can read M(q) and V(q) out of the trained model and compare to the
    analytic ones (interpretability). The MLP has nothing to inspect.

The numpy wrapper exposes inverse_dynamics(q, qd, qdd) -> tau, matching the
analytic model and the MLP, so it drops straight into InverseDynamicsController.
"""

import numpy as np
import torch
import torch.nn as nn

DOF = 2
# indices of the lower-triangular entries of a DOFxDOF matrix (row-major)
_TRIL_IDX = torch.tril_indices(DOF, DOF)
_N_TRIL = _TRIL_IDX.shape[1]  # = 3 for DOF=2
# which of those are on the diagonal (we softplus these to keep them positive)
_DIAG_MASK = (_TRIL_IDX[0] == _TRIL_IDX[1])


def _mlp(in_dim, out_dim, hidden, act):
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


class DeLaN(nn.Module):
    """Learns L(q) (Cholesky factor of M) and V(q) (potential). Torque is
    assembled from them via the Euler-Lagrange equation using autograd."""

    def __init__(self, hidden=(64, 64), eps=1e-3, act=nn.Softplus):
        super().__init__()
        # Softplus (smooth) activations: DeLaN needs continuous 2nd derivatives
        # through the network, so ReLU (zero curvature) is a poor choice here.
        self._hidden = tuple(hidden)
        self.l_net = _mlp(DOF, _N_TRIL, hidden, act)
        self.v_net = _mlp(DOF, 1, hidden, act)
        self.eps = eps  # ridge added to M for strict positive-definiteness

    # --- building blocks ---
    def cholesky_L(self, q):
        """Assemble lower-triangular L(q), shape (B, DOF, DOF)."""
        raw = self.l_net(q)                       # (B, _N_TRIL)
        diag = torch.zeros_like(raw)
        # softplus on diagonal entries -> strictly positive -> M strictly PD
        diag_vals = torch.nn.functional.softplus(raw[:, _DIAG_MASK])
        off_vals = raw[:, ~_DIAG_MASK]
        B = q.shape[0]
        L = q.new_zeros(B, DOF, DOF)
        di = 0
        oi = 0
        for k in range(_N_TRIL):
            i, j = int(_TRIL_IDX[0, k]), int(_TRIL_IDX[1, k])
            if i == j:
                L[:, i, j] = diag_vals[:, di]; di += 1
            else:
                L[:, i, j] = off_vals[:, oi]; oi += 1
        return L

    def mass_matrix(self, q):
        """M(q) = L L^T + eps I, shape (B, DOF, DOF). Guaranteed SPD."""
        L = self.cholesky_L(q)
        M = L @ L.transpose(-1, -2)
        M = M + self.eps * torch.eye(DOF, device=q.device, dtype=q.dtype)
        return M

    def potential(self, q):
        """Scalar potential energy V(q), shape (B, 1)."""
        return self.v_net(q)

    # --- Euler-Lagrange torque assembly ---
    def inverse_dynamics(self, q, qd, qdd):
        """tau = M qdd + c(q,qd) + g(q), all terms differentiated from M and V.

        q, qd, qdd: (B, DOF) tensors with requires_grad handled internally.
        Returns tau: (B, DOF).
        """
        B = q.shape[0]
        q = q.clone().requires_grad_(True)

        M = self.mass_matrix(q)                       # (B, DOF, DOF)

        # gravity: g = dV/dq
        V = self.potential(q).sum()
        g = torch.autograd.grad(V, q, create_graph=True)[0]   # (B, DOF)

        # We need dM/dq_k. Differentiate each entry of M w.r.t. q.
        # kinetic quadratic term T2 = 1/2 qd^T M qd ; its gradient wrt q gives
        # the 1/2 d/dq(qd^T M qd) piece directly.
        qd_ = qd
        Mqd = torch.einsum("bij,bj->bi", M, qd_)              # M qd
        quad = 0.5 * torch.einsum("bi,bi->b", qd_, Mqd).sum() # sum_B 1/2 qd^T M qd
        dquad_dq = torch.autograd.grad(quad, q, create_graph=True)[0]  # (B,DOF)

        # Mdot qd = sum_k (dM/dq_k) qd * qd_k. Build via d/dq of (M qd) contracted.
        # Trick: (Mdot qd)_i = sum_k dM_ij/dq_k qd_j qd_k. We get sum_k dM_ij/dq_k qd_k
        # = d/dt M_ij along qd, i.e. directional derivative of M in direction qd.
        # Compute Jacobian-vector product of vec(M) wrt q in direction qd.
        Mflat = M.reshape(B, -1)
        Mdot_flat = torch.zeros_like(Mflat)
        for col in range(Mflat.shape[1]):
            gk = torch.autograd.grad(Mflat[:, col].sum(), q,
                                     create_graph=True)[0]     # dM_col/dq (B,DOF)
            Mdot_flat[:, col] = torch.einsum("bk,bk->b", gk, qd_)
        Mdot = Mdot_flat.reshape(B, DOF, DOF)
        Mdot_qd = torch.einsum("bij,bj->bi", Mdot, qd_)        # (B, DOF)

        # Coriolis/centrifugal term c = Mdot qd - 1/2 d/dq(qd^T M qd)
        c = Mdot_qd - dquad_dq

        tau = torch.einsum("bij,bj->bi", M, qdd) + c + g
        return tau


class DeLaNInverseDynamics:
    """numpy wrapper matching the analytic/MLP interface for the controller."""

    def __init__(self, net, device="cpu"):
        self.net = net.to(device)
        self.device = device

    def _t(self, a):
        return torch.as_tensor(np.atleast_2d(a), dtype=torch.float32,
                               device=self.device)

    def inverse_dynamics(self, q, qd, qdd):
        tau = self.net.inverse_dynamics(self._t(q), self._t(qd), self._t(qdd))
        return tau.detach().cpu().numpy()[0]

    def mass_matrix(self, q):
        M = self.net.mass_matrix(self._t(q))
        return M.detach().cpu().numpy()[0]

    def gravity_vector(self, q):
        qt = self._t(q).clone().requires_grad_(True)
        V = self.net.potential(qt).sum()
        g = torch.autograd.grad(V, qt)[0]
        return g.detach().cpu().numpy()[0]

    def save(self, path):
        torch.save({"state_dict": self.net.state_dict(),
                    "hidden": self.net._hidden,
                    "eps": self.net.eps}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        net = DeLaN(hidden=tuple(ck["hidden"]), eps=ck["eps"])
        net.load_state_dict(ck["state_dict"])
        return cls(net, device=device)
