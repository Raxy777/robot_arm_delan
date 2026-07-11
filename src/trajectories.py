"""Reference trajectories.

The headline one is a task-space figure-eight (a Lissajous curve): the
end-effector traces an "8". We return position + analytic velocity and
acceleration, then convert to a joint-space reference via kinematics.
"""

import numpy as np

from src.kinematics import task_to_joint


def figure_eight(t, center=(1.0, 0.6), A=0.5, B=0.35, period=6.0):
    """Task-space figure-eight at time t (seconds).

    x(t) = cx + A sin(w t)
    y(t) = cy + B sin(2 w t)     # doubled frequency -> the '8'

    Returns (x, xd, xdd), each a length-2 array [., .].
    Defaults are chosen to stay comfortably inside the workspace of the
    default 1 m + 1 m arm.
    """
    cx, cy = center
    w = 2 * np.pi / period

    x = np.array([cx + A * np.sin(w * t),
                  cy + B * np.sin(2 * w * t)])
    xd = np.array([A * w * np.cos(w * t),
                   B * 2 * w * np.cos(2 * w * t)])
    xdd = np.array([-A * w**2 * np.sin(w * t),
                    -B * (2 * w)**2 * np.sin(2 * w * t)])
    return x, xd, xdd


def figure_eight_joint(t, elbow_up=True, **kw):
    """Figure-eight expressed as a joint-space reference (q, qd, qdd)."""
    x, xd, xdd = figure_eight(t, **kw)
    return task_to_joint(x, xd, xdd, elbow_up=elbow_up)


def sample_reference(traj_fn, duration, dt, **kw):
    """Sample a joint-space reference over [0, duration).

    Returns dict of arrays: t, q, qd, qdd (each q* is (N, 2)).
    """
    ts = np.arange(0.0, duration, dt)
    qs, qds, qdds = [], [], []
    for t in ts:
        q, qd, qdd = traj_fn(t, **kw)
        qs.append(q)
        qds.append(qd)
        qdds.append(qdd)
    return {
        "t": ts,
        "q": np.array(qs),
        "qd": np.array(qds),
        "qdd": np.array(qdds),
    }
