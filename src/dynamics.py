"""Analytic rigid-body dynamics of the 2-link planar arm.

Manipulator equation:

    M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau

Everything here is derived by hand from the Lagrangian (see README.md) and
implemented in closed form. This is your "ground-truth" model: it is the
upper bound in the Week 10 benchmark and the target the learned models must
match.

All functions take numpy arrays q = [q1, q2], qd = [q1d, q2d].
"""

import numpy as np

from src.params import PARAMS as P


def mass_matrix(q):
    """Inertia matrix M(q), 2x2, symmetric positive-definite."""
    q2 = q[1]
    c2 = np.cos(q2)

    m11 = (P.m1 * P.lc1**2
           + P.m2 * (P.l1**2 + P.lc2**2 + 2 * P.l1 * P.lc2 * c2)
           + P.I1 + P.I2)
    m12 = P.m2 * (P.lc2**2 + P.l1 * P.lc2 * c2) + P.I2
    m22 = P.m2 * P.lc2**2 + P.I2

    return np.array([[m11, m12],
                     [m12, m22]])


def coriolis_matrix(q, qd):
    """Coriolis/centrifugal matrix C(q, qd) such that the velocity-product
    torque is C @ qd. Chosen so that (M_dot - 2C) is skew-symmetric, which is
    what the energy check in verify relies on."""
    q2 = q[1]
    q1d, q2d = qd
    s2 = np.sin(q2)
    h = P.m2 * P.l1 * P.lc2 * s2

    c11 = -h * q2d
    c12 = -h * (q1d + q2d)
    c21 = h * q1d
    c22 = 0.0

    return np.array([[c11, c12],
                     [c21, c22]])


def gravity_vector(q):
    """Gravity torque g(q). Angles measured from +x, gravity along -y."""
    q1, q2 = q
    c1 = np.cos(q1)
    c12 = np.cos(q1 + q2)

    g1 = ((P.m1 * P.lc1 + P.m2 * P.l1) * P.g * c1
          + P.m2 * P.lc2 * P.g * c12)
    g2 = P.m2 * P.lc2 * P.g * c12

    return np.array([g1, g2])


def inverse_dynamics(q, qd, qdd):
    """tau required to produce acceleration qdd. This is what a learned
    inverse-dynamics model (MLP / DeLaN) will try to reproduce."""
    return mass_matrix(q) @ qdd + coriolis_matrix(q, qd) @ qd + gravity_vector(q)


def forward_dynamics(q, qd, tau):
    """qdd produced by torque tau. Used by the pure-python integrator in
    verify_dynamics.py (MuJoCo does this internally for the real sim)."""
    M = mass_matrix(q)
    bias = coriolis_matrix(q, qd) @ qd + gravity_vector(q)
    return np.linalg.solve(M, tau - bias)


def kinetic_energy(q, qd):
    return 0.5 * qd @ (mass_matrix(q) @ qd)


def potential_energy(q):
    """U = sum m_i g y_com_i, with y measured along +y (gravity is -y)."""
    q1, q2 = q
    y1 = P.lc1 * np.sin(q1)
    y2 = P.l1 * np.sin(q1) + P.lc2 * np.sin(q1 + q2)
    return P.m1 * P.g * y1 + P.m2 * P.g * y2


def total_energy(q, qd):
    return kinetic_energy(q, qd) + potential_energy(q)
