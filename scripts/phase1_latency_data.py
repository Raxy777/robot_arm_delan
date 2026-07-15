"""Collect Phase-1 latency-aware hydrodynamic identification data.

Each episode uses one known, constant world-frame flow. Sensor and actuator
latency are modeled by separate causal queues. The controller runs slower than
the 1 kHz plant, so its command is explicitly held with zero-order hold (ZOH).
Both true and controller-observed state are logged; no future sample is used.

Example:
    python scripts/phase1_latency_data.py --episodes 24 --duration 4 \
        --flow 0.3 0.0 0.0 --out data/phase1_latency_raw.npz
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src import dynamics
from src.arm_sim import ArmSim, FluidConfig
from src.kinematics import link_relative_velocities
from src.latency import FixedDelayQueue


@dataclass(frozen=True)
class EpisodeConfig:
    duration: float = 4.0
    control_period_steps: int = 10
    sensor_delay_steps: int = 10
    actuation_delay_steps: int = 10
    kp: float = 90.0
    kd: float = 18.0

    def __post_init__(self):
        if self.duration <= 0:
            raise ValueError("duration must be positive")
        for name in ("control_period_steps", "sensor_delay_steps", "actuation_delay_steps"):
            value = getattr(self, name)
            if int(value) != value or value < (1 if name == "control_period_steps" else 0):
                raise ValueError(f"{name} has an invalid value")


def excitation_reference(t: float, rng: np.random.Generator, params=None):
    """Smooth multi-frequency joint reference with analytic derivatives."""
    if params is None:
        # Frequencies reach 3 Hz to excite the velocity-squared drag block.
        params = {
            "offset": rng.uniform([-0.5, -0.8], [0.5, 0.2]),
            "amplitude": rng.uniform([0.25, 0.20], [0.55, 0.45], size=(3, 2)),
            "frequency": rng.uniform([0.6, 0.8], [3.0, 3.0], size=(3, 2)),
            "phase": rng.uniform(0.0, 2.0 * np.pi, size=(3, 2)),
        }
    omega = 2.0 * np.pi * params["frequency"]
    angle = omega * t + params["phase"]
    q = params["offset"] + np.sum(params["amplitude"] * np.sin(angle), axis=0)
    qd = np.sum(params["amplitude"] * omega * np.cos(angle), axis=0)
    qdd = -np.sum(params["amplitude"] * omega**2 * np.sin(angle), axis=0)
    return q, qd, qdd, params


def collect_episode(sim: ArmSim, rng: np.random.Generator, episode_id: int,
                    config: EpisodeConfig):
    """Run one episode and return arrays with causal latency/ZOH provenance."""
    q0, qd0, _, reference_params = excitation_reference(0.0, rng)
    # Start close to the reference but vary every trajectory independently.
    sim.reset(q0 + rng.normal(0.0, 0.05, 2), qd0 * 0.1)
    dt = sim.dt
    n_steps = int(round(config.duration / dt))
    if n_steps < 1:
        raise ValueError("duration is shorter than one simulation step")

    initial_sensor = (0, 0.0, sim.q, sim.qd)
    sensor_queue = FixedDelayQueue(config.sensor_delay_steps, initial_sensor)
    initial_command = (-config.actuation_delay_steps, -config.actuation_delay_steps * dt,
                       np.zeros(2))
    actuator_queue = FixedDelayQueue(config.actuation_delay_steps, initial_command)
    held_command = np.zeros(2)
    command_step = 0

    fields = (
        "time_true", "time_observed", "q_true", "qdot_true", "q_observed",
        "qdot_observed", "qdd_true", "tau_commanded", "tau_applied",
        "flow_velocity", "link_relative_velocity", "fluid_torque",
        "sensor_delay_steps", "sensor_age_steps", "actuation_delay_steps",
        "command_age_steps", "applied_command_age_steps", "episode_id",
        "trajectory_id", "reference_q", "reference_qdot", "reference_qddot",
    )
    log = {name: [] for name in fields}

    for step in range(n_steps):
        t = step * dt
        q_true, qdot_true = sim.q, sim.qd
        observed_step, observed_time, q_obs, qdot_obs = sensor_queue.push(
            (step, t, q_true, qdot_true))

        q_ref, qdot_ref, qddot_ref, _ = excitation_reference(t, rng, reference_params)
        if step % config.control_period_steps == 0:
            # The command depends only on the delayed observation available now.
            held_command = (config.kp * (q_ref - q_obs)
                            + config.kd * (qdot_ref - qdot_obs)
                            + dynamics.gravity_vector(q_obs))
            command_step = step
        command_age = step - command_step

        applied_origin_step, _, tau_applied = actuator_queue.push(
            (command_step, command_step * dt, held_command))
        sim.set_torque(tau_applied)
        # Log simulator truth under the torque that will advance this sample.
        actual_tau = sim.joint_torque()
        qdd_true = sim.acceleration()
        fluid_torque = sim.passive_torque()

        values = {
            "time_true": t,
            "time_observed": observed_time,
            "q_true": q_true,
            "qdot_true": qdot_true,
            "q_observed": q_obs,
            "qdot_observed": qdot_obs,
            "qdd_true": qdd_true,
            "tau_commanded": held_command,
            "tau_applied": actual_tau,
            "flow_velocity": sim.flow_velocity,
            "link_relative_velocity": link_relative_velocities(
                q_true, qdot_true, sim.flow_velocity),
            "fluid_torque": fluid_torque,
            "sensor_delay_steps": config.sensor_delay_steps,
            "sensor_age_steps": step - observed_step,
            "actuation_delay_steps": config.actuation_delay_steps,
            "command_age_steps": command_age,
            "applied_command_age_steps": step - applied_origin_step,
            "episode_id": episode_id,
            "trajectory_id": episode_id,
            "reference_q": q_ref,
            "reference_qdot": qdot_ref,
            "reference_qddot": qddot_ref,
        }
        for name in fields:
            log[name].append(values[name])
        sim.step()

    return {name: np.asarray(values) for name, values in log.items()}


def concatenate_episodes(episodes):
    if not episodes:
        raise ValueError("at least one episode is required")
    keys = tuple(episodes[0])
    if any(tuple(ep) != keys for ep in episodes):
        raise ValueError("episode schemas do not match")
    return {key: np.concatenate([ep[key] for ep in episodes], axis=0) for key in keys}


def collect_dataset(episodes: int, duration: float, flow, seed: int,
                    control_period_steps: int, sensor_delays, actuation_delays):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)
    sim = ArmSim(backend="mujoco", fluid=FluidConfig(flow_velocity=tuple(flow)))
    logs = []
    for episode_id in range(episodes):
        config = EpisodeConfig(
            duration=duration,
            control_period_steps=control_period_steps,
            sensor_delay_steps=sensor_delays[episode_id % len(sensor_delays)],
            actuation_delay_steps=actuation_delays[episode_id % len(actuation_delays)],
        )
        logs.append(collect_episode(sim, rng, episode_id, config))
    return concatenate_episodes(logs)


def _nonnegative_steps(values, name):
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} must contain non-negative integers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=24)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--flow", type=float, nargs=3, default=(0.3, 0.0, 0.0))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control-period-steps", type=int, default=10,
                        help="ZOH update interval in 1 kHz simulation steps")
    parser.add_argument("--sensor-delays", type=int, nargs="+", default=(0, 10, 20))
    parser.add_argument("--actuation-delays", type=int, nargs="+", default=(0, 5, 10))
    parser.add_argument("--out", default="data/phase1_latency_raw.npz")
    args = parser.parse_args()
    if args.duration <= 0 or args.control_period_steps < 1 or args.episodes < 1:
        parser.error("episodes, duration, and control-period-steps must be positive")
    _nonnegative_steps(args.sensor_delays, "sensor-delays")
    _nonnegative_steps(args.actuation_delays, "actuation-delays")

    dataset = collect_dataset(
        args.episodes, args.duration, args.flow, args.seed,
        args.control_period_steps, args.sensor_delays, args.actuation_delays,
    )
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **dataset)
    speed = np.linalg.norm(dataset["link_relative_velocity"], axis=-1)
    print(f"saved {len(dataset['time_true'])} samples from {args.episodes} episodes: {out}")
    print(f"relative-link speed: mean={speed.mean():.3f}, p95={np.percentile(speed, 95):.3f}, "
          f"max={speed.max():.3f} m/s")
    print("flow is constant and known:", dataset["flow_velocity"][0].tolist())


if __name__ == "__main__":
    main()
