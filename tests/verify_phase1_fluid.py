"""Small deterministic checks for the Phase-1 uniform-fluid configuration."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.arm_sim import ArmSim, FluidConfig
from src.kinematics import link_relative_velocities


def fluid_force(flow):
    sim = ArmSim(backend="mujoco", fluid=FluidConfig(flow_velocity=flow))
    sim.reset([0.7, -0.5], [0.0, 0.0])
    sim.set_torque([0.0, 0.0])
    return sim.passive_torque()


def main():
    still = fluid_force((0.0, 0.0, 0.0))
    plus = fluid_force((0.3, 0.0, 0.0))
    minus = fluid_force((-0.3, 0.0, 0.0))
    assert np.allclose(still, 0.0, atol=1e-12)
    assert np.linalg.norm(plus) > 1e-4
    assert np.allclose(plus, -minus, rtol=1e-6, atol=1e-9)

    rel = link_relative_velocities([0.0, 0.0], [0.0, 0.0], [0.3, -0.2, 0.0])
    assert np.allclose(rel, [[-0.3, 0.2], [-0.3, 0.2]])
    print("PASS: uniform flow and relative-velocity checks")


if __name__ == "__main__":
    main()
