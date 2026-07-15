"""Deterministic checks for causal latency-aware Phase-1 data collection."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.phase1_latency_data import EpisodeConfig, collect_episode
from src.arm_sim import ArmSim, FluidConfig
from src.latency import FixedDelayQueue


def verify_queue():
    q = FixedDelayQueue(2, (-1, np.array([-1.0])))
    out = [q.push((k, np.array([float(k)]))) for k in range(4)]
    assert [sample[0] for sample in out] == [-1, -1, 0, 1]
    assert [sample[1][0] for sample in out] == [-1.0, -1.0, 0.0, 1.0]
    zero = FixedDelayQueue(0, None)
    assert zero.push(7) == 7


def verify_episode():
    sim = ArmSim(
        backend="mujoco",
        fluid=FluidConfig(flow_velocity=(0.3, 0.0, 0.0)),
    )
    config = EpisodeConfig(duration=0.04, control_period_steps=4,
                           sensor_delay_steps=3, actuation_delay_steps=2)
    log = collect_episode(sim, np.random.default_rng(7), 11, config)
    n = int(round(config.duration / sim.dt))
    assert all(len(value) == n for value in log.values())
    assert np.all(log["episode_id"] == 11)
    assert np.all(log["trajectory_id"] == log["episode_id"])
    assert np.allclose(log["flow_velocity"], [0.3, 0.0, 0.0])

    # Once initialized, observations are exactly three samples old—never future.
    assert np.allclose(log["q_observed"][3:], log["q_true"][:-3])
    assert np.all(log["time_observed"] <= log["time_true"] + 1e-15)
    assert np.all(log["sensor_age_steps"][3:] == 3)
    # The actuator receives the command stream from exactly two plant steps ago.
    assert np.allclose(
        log["tau_applied"][2:],
        np.clip(log["tau_commanded"][:-2], -sim.tau_limit, sim.tau_limit),
    )
    assert np.all(log["command_age_steps"] == np.arange(n) % 4)
    assert np.isfinite(log["qdd_true"]).all()
    assert np.isfinite(log["fluid_torque"]).all()
    assert log["link_relative_velocity"].shape == (n, 2, 2)


def main():
    verify_queue()
    verify_episode()
    print("PASS: causal sensor/actuation latency and ZOH dataset checks")


if __name__ == "__main__":
    main()
