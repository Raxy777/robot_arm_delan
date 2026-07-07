"""Forward/inverse kinematics and Jacobians for the 2-link planar arm.

Used to turn a task-space (end-effector) figure-eight into a joint-space
reference (q_d, q_dot_d, q_ddot_d) that the controllers track.

Note on differentiating the reference: we do NOT finite-difference the joint
angles. Instead we map task-space velocity/acceleration through the analytic
Jacobian: q_dot = J^{-1} x_dot, q_ddot = J^{-1}(x_ddot - J_dot q_dot). This
keeps the reference clean (the plan's warning about noisy accelerations is
about *measured* data; a designed reference should be computed analytically).
"""

import numpy as np

from params import PARAMS as P


def forward_kinematics(q):
    """End-effector position (x, y)."""
    q1, q2 = q
    x = P.l1 * np.cos(q1) + P.l2 * np.cos(q1 + q2)
    y = P.l1 * np.sin(q1) + P.l2 * np.sin(q1 + q2)
    return np.array([x, y])


def inverse_kinematics(x, elbow_up=True):
    """Joint angles reaching task position x = (x, y).

    Returns q = [q1, q2]. Raises ValueError if the target is unreachable.
    """
    px, py = x
    r2 = px**2 + py**2
    cos_q2 = (r2 - P.l1**2 - P.l2**2) / (2 * P.l1 * P.l2)
    if abs(cos_q2) > 1.0 + 1e-9:
        raise ValueError(f"Target {x} is outside the workspace.")
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    sin_q2 = np.sqrt(1 - cos_q2**2)
    if elbow_up:
        sin_q2 = -sin_q2
    q2 = np.arctan2(sin_q2, cos_q2)
    q1 = np.arctan2(py, px) - np.arctan2(P.l2 * np.sin(q2),
                                         P.l1 + P.l2 * np.cos(q2))
    return np.array([q1, q2])


def jacobian(q):
    """Task-space Jacobian J such that x_dot = J q_dot."""
    q1, q2 = q
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    j11 = -P.l1 * s1 - P.l2 * s12
    j12 = -P.l2 * s12
    j21 = P.l1 * c1 + P.l2 * c12
    j22 = P.l2 * c12
    return np.array([[j11, j12],
                     [j21, j22]])


def jacobian_dot(q, qd):
    """Time derivative of the Jacobian, J_dot, given joint velocities."""
    q1, q2 = q
    q1d, q2d = qd
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1 + q2), np.cos(q1 + q2)
    d1 = q1d          # d/dt (q1)
    d12 = q1d + q2d   # d/dt (q1 + q2)

    j11 = -P.l1 * c1 * d1 - P.l2 * c12 * d12
    j12 = -P.l2 * c12 * d12
    j21 = -P.l1 * s1 * d1 - P.l2 * s12 * d12
    j22 = -P.l2 * s12 * d12
    return np.array([[j11, j12],
                     [j21, j22]])


def task_to_joint(x, xd, xdd, elbow_up=True):
    """Map a task-space point + derivatives to joint reference.

    Returns (q, qd, qdd).
    """
    q = inverse_kinematics(x, elbow_up=elbow_up)
    J = jacobian(q)
    Jinv = np.linalg.inv(J)
    qd = Jinv @ xd
    Jd = jacobian_dot(q, qd)
    qdd = Jinv @ (xdd - Jd @ qd)
    return q, qd, qdd
