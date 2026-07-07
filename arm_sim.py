"""Unified stepping interface for the 2-link arm, with two backends.

    ArmSim(backend="mujoco")   -> the real simulator (your deliverable)
    ArmSim(backend="analytic") -> pure-python RK4 on the hand-derived model

Both expose the same API, so data collection, control, and evaluation code is
written once and runs either way. For this contact-free rigid arm the two
backends agree to ~1e-9 (see verify_dynamics.py), so the analytic backend is a
faithful, fast stand-in when MuJoCo isn't handy — and it's what lets the whole
Week-8 pipeline be tested without a GPU or a display.

Key correctness point (the plan's Week-8 pitfall): acceleration q̈ is read
straight from the dynamics, never finite-differenced from velocities.
  - MuJoCo: data.qacc after mj_forward with the applied control.
  - analytic: forward_dynamics(q, q̇, tau) directly.
"""

import os

import numpy as np

import dynamics

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "arm2.xml")


class ArmSim:
    def __init__(self, backend="mujoco", dt=0.001, tau_limit=60.0):
        self.backend = backend
        self._tau = np.zeros(2)
        # Actuator torque saturation. Matches ctrlrange in model/arm2.xml so the
        # analytic backend clips exactly like MuJoCo does (otherwise the two
        # backends diverge whenever a controller commands a big torque).
        self.tau_limit = float(tau_limit)
        if backend == "mujoco":
            import mujoco  # imported lazily so analytic path needs no mujoco
            self._mj = mujoco
            self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
            self.data = mujoco.MjData(self.model)
            self.dt = float(self.model.opt.timestep)
        elif backend == "analytic":
            self.dt = float(dt)
            self._q = np.zeros(2)
            self._qd = np.zeros(2)
        else:
            raise ValueError(f"unknown backend: {backend}")

    # --- state ---
    def reset(self, q, qd):
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        if self.backend == "mujoco":
            self.data.qpos[:] = q
            self.data.qvel[:] = qd
            self._mj.mj_forward(self.model, self.data)
        else:
            self._q = q.copy()
            self._qd = qd.copy()
        self._tau = np.zeros(2)
        return self

    @property
    def q(self):
        return self.data.qpos.copy() if self.backend == "mujoco" else self._q.copy()

    @property
    def qd(self):
        return self.data.qvel.copy() if self.backend == "mujoco" else self._qd.copy()

    def set_torque(self, tau):
        self._tau = np.clip(np.asarray(tau, float),
                            -self.tau_limit, self.tau_limit)

    def acceleration(self):
        """q̈ consistent with the current state and the set torque."""
        if self.backend == "mujoco":
            self.data.ctrl[:] = self._tau
            self._mj.mj_forward(self.model, self.data)
            return self.data.qacc.copy()
        return dynamics.forward_dynamics(self._q, self._qd, self._tau)

    def joint_torque(self):
        """Generalized joint torque actually applied (== ctrl since gear=1)."""
        if self.backend == "mujoco":
            self.data.ctrl[:] = self._tau
            self._mj.mj_forward(self.model, self.data)
            return self.data.qfrc_actuator.copy()
        return self._tau.copy()

    # --- integration ---
    def step(self):
        if self.backend == "mujoco":
            self.data.ctrl[:] = self._tau
            self._mj.mj_step(self.model, self.data)
        else:
            self._rk4_step()

    def _rk4_step(self):
        dt = self.dt
        tau = self._tau

        def deriv(s):
            q, qd = s[:2], s[2:]
            return np.concatenate([qd, dynamics.forward_dynamics(q, qd, tau)])

        s = np.concatenate([self._q, self._qd])
        k1 = deriv(s)
        k2 = deriv(s + 0.5 * dt * k1)
        k3 = deriv(s + 0.5 * dt * k2)
        k4 = deriv(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        self._q, self._qd = s[:2], s[2:]
