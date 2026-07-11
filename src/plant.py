"""The *true* plant for the Week-10 robustness tests.

Weeks 8-9 trained and evaluated every model on the SAME arm the data came from.
Week 10 asks the harder question an interviewer will: what happens when the
world is not the world you trained on? Two perturbations, both classic sim-to-
real gaps:

  1. Payload.  A mass `mp` is bolted to the end-effector that was never present
     during data collection. This changes the true M(q), C(q,q̇) and g(q) — the
     inertia and gravity the controller must compensate are now wrong for every
     learned model, because none of them saw this mass.

  2. Sensor noise.  The controller reads q, q̇ through noisy encoders/tachos.
     (Injected in benchmark.py, where the controller's *measured* state is
     decoupled from the plant's *true* state used to score tracking.)

This module supplies the parametrized rigid-body dynamics (so we can build the
payload plant) and `PlantSim`, an RK4 integrator on those true dynamics. The
learned models are NOT told about any of this — that is the whole point.

Payload model
-------------
A point mass `mp` rigidly attached at the tip of link 2 (distance l2 from joint
2) is exactly equivalent to a modified link 2 with lumped parameters

    m2'  = m2 + mp
    lc2' = (m2 lc2 + mp l2) / (m2 + mp)                      # combined COM
    I2'  = I2 + m2 (lc2 - lc2')^2 + mp (l2 - lc2')^2         # parallel axis

so we can feed the *same* closed-form dynamics with these substituted values.
For mp = 0 this reduces to the nominal params (checked in the smoke test).
"""

from dataclasses import dataclass, replace

import numpy as np

from src.params import PARAMS as NOMINAL


@dataclass(frozen=True)
class PlantParams:
    """Same fields dynamics.py reads, but a plain carrier we can perturb."""
    m1: float
    m2: float
    l1: float
    l2: float
    lc1: float
    lc2: float
    I1: float
    I2: float
    g: float

    @classmethod
    def nominal(cls):
        P = NOMINAL
        return cls(P.m1, P.m2, P.l1, P.l2, P.lc1, P.lc2, P.I1, P.I2, P.g)


def with_payload(mp, params=None):
    """Return PlantParams with a point mass `mp` (kg) added at the link-2 tip."""
    P = params or PlantParams.nominal()
    if mp == 0.0:
        return P
    m2p = P.m2 + mp
    lc2p = (P.m2 * P.lc2 + mp * P.l2) / m2p
    I2p = P.I2 + P.m2 * (P.lc2 - lc2p) ** 2 + mp * (P.l2 - lc2p) ** 2
    return replace(P, m2=m2p, lc2=lc2p, I2=I2p)


# --- parametrized closed-form dynamics (mirrors dynamics.py, P-driven) --------

def mass_matrix(q, P):
    c2 = np.cos(q[1])
    m11 = (P.m1 * P.lc1**2
           + P.m2 * (P.l1**2 + P.lc2**2 + 2 * P.l1 * P.lc2 * c2)
           + P.I1 + P.I2)
    m12 = P.m2 * (P.lc2**2 + P.l1 * P.lc2 * c2) + P.I2
    m22 = P.m2 * P.lc2**2 + P.I2
    return np.array([[m11, m12], [m12, m22]])


def coriolis_matrix(q, qd, P):
    q1d, q2d = qd
    s2 = np.sin(q[1])
    h = P.m2 * P.l1 * P.lc2 * s2
    return np.array([[-h * q2d, -h * (q1d + q2d)],
                     [h * q1d, 0.0]])


def gravity_vector(q, P):
    q1, q2 = q
    c1 = np.cos(q1)
    c12 = np.cos(q1 + q2)
    g1 = (P.m1 * P.lc1 + P.m2 * P.l1) * P.g * c1 + P.m2 * P.lc2 * P.g * c12
    g2 = P.m2 * P.lc2 * P.g * c12
    return np.array([g1, g2])


def forward_dynamics(q, qd, tau, P):
    M = mass_matrix(q, P)
    bias = coriolis_matrix(q, qd, P) @ qd + gravity_vector(q, P)
    return np.linalg.solve(M, tau - bias)


def inverse_dynamics(q, qd, qdd, P):
    return mass_matrix(q, P) @ qdd + coriolis_matrix(q, qd, P) @ qd + gravity_vector(q, P)


# --- simulator on the true plant ---------------------------------------------

class PlantSim:
    """RK4 integrator on the true (possibly payload-loaded) dynamics.

    Deliberately mirrors Week-8's ArmSim analytic backend (same dt, same torque
    saturation) so tracking numbers are comparable, but the plant it integrates
    can differ from what any controller believes. `q`/`qd` return the TRUE state;
    sensor noise is applied by the caller to the controller's view only.
    """

    def __init__(self, params=None, dt=0.001, tau_limit=60.0):
        self.P = params or PlantParams.nominal()
        self.dt = float(dt)
        self.tau_limit = float(tau_limit)
        self._q = np.zeros(2)
        self._qd = np.zeros(2)
        self._tau = np.zeros(2)

    def reset(self, q, qd):
        self._q = np.asarray(q, float).copy()
        self._qd = np.asarray(qd, float).copy()
        self._tau = np.zeros(2)
        return self

    @property
    def q(self):
        return self._q.copy()

    @property
    def qd(self):
        return self._qd.copy()

    def set_torque(self, tau):
        self._tau = np.clip(np.asarray(tau, float), -self.tau_limit, self.tau_limit)

    def step(self):
        dt, tau = self.dt, self._tau

        def deriv(s):
            q, qd = s[:2], s[2:]
            return np.concatenate([qd, forward_dynamics(q, qd, tau, self.P)])

        s = np.concatenate([self._q, self._qd])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._q, self._qd = s[:2], s[2:]
