"""Joint-space controllers, all implemented from scratch (no control library).

Each controller is a callable: given the current state (q, qd) and the desired
state (q_d, qd_d, qdd_d), it returns a torque command tau.

Three controllers, in increasing order of "cheating with the model":

  1. PDController              - pure feedback, knows nothing about dynamics.
  2. GravityCompPDController   - PD + gravity feedforward g(q).
  3. ComputedTorqueController  - full feedback linearization using M, C, g.

In Week 8-10 you'll swap the *model* inside ComputedTorque (analytic -> MLP ->
DeLaN) and keep the controller structure identical. That's the experiment.
"""

import numpy as np

import dynamics


class PDController:
    """tau = Kp (q_d - q) + Kd (qd_d - qd)."""

    def __init__(self, kp, kd):
        self.Kp = np.diag(np.atleast_1d(kp).astype(float) * np.ones(2))
        self.Kd = np.diag(np.atleast_1d(kd).astype(float) * np.ones(2))

    def __call__(self, q, qd, q_d, qd_d, qdd_d):
        e = q_d - q
        ed = qd_d - qd
        return self.Kp @ e + self.Kd @ ed


class GravityCompPDController(PDController):
    """PD plus gravity feedforward. Removes steady-state droop under gravity."""

    def __call__(self, q, qd, q_d, qd_d, qdd_d):
        pd = super().__call__(q, qd, q_d, qd_d, qdd_d)
        return pd + dynamics.gravity_vector(q)


class ComputedTorqueController:
    """Feedback linearization / inverse-dynamics control:

        tau = M(q) [ qdd_d + Kd (qd_d - qd) + Kp (q_d - q) ]
              + C(q, qd) qd + g(q)

    With a perfect model this yields decoupled error dynamics
    e_ddot + Kd e_dot + Kp e = 0, so choose Kp = wn^2, Kd = 2 zeta wn.

    `model` must expose mass_matrix, coriolis_matrix, gravity_vector. Default
    is the analytic module; later pass a learned model with the same API.
    """

    def __init__(self, kp, kd, model=dynamics):
        self.Kp = np.diag(np.atleast_1d(kp).astype(float) * np.ones(2))
        self.Kd = np.diag(np.atleast_1d(kd).astype(float) * np.ones(2))
        self.model = model

    def __call__(self, q, qd, q_d, qd_d, qdd_d):
        e = q_d - q
        ed = qd_d - qd
        aq = qdd_d + self.Kd @ ed + self.Kp @ e
        M = self.model.mass_matrix(q)
        bias = self.model.coriolis_matrix(q, qd) @ qd + self.model.gravity_vector(q)
        return M @ aq + bias


class InverseDynamicsController:
    """Model-agnostic version of computed torque.

        aq  = qdd_d + Kd (qd_d - qd) + Kp (q_d - q)
        tau = model.inverse_dynamics(q, qd, aq)

    The ONLY requirement on `model` is an inverse_dynamics(q, qd, qdd) -> tau
    method. That is deliberately the single interface shared by:
      - the analytic model (dynamics.py)  -> reproduces computed torque exactly,
      - the learned MLP (Week 8),
      - DeLaN (Week 9).
    So the whole benchmark is "swap the model, keep this controller."

    Using the analytic model here gives IDENTICAL torques to
    ComputedTorqueController, because analytic inverse_dynamics(q, qd, aq) =
    M(q) aq + C(q,qd) qd + g(q). (verified in verify_control.py)
    """

    def __init__(self, kp, kd, model):
        self.Kp = np.diag(np.atleast_1d(kp).astype(float) * np.ones(2))
        self.Kd = np.diag(np.atleast_1d(kd).astype(float) * np.ones(2))
        self.model = model

    def __call__(self, q, qd, q_d, qd_d, qdd_d):
        e = q_d - q
        ed = qd_d - qd
        aq = qdd_d + self.Kd @ ed + self.Kp @ e
        return self.model.inverse_dynamics(q, qd, aq)
