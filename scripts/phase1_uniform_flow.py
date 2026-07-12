"""Phase 1.1 smoke experiment: still water versus one uniform current.

The original XML remains the rigid-body/no-fluid baseline. Fluid parameters are
set at runtime, making the experiment explicit and preventing old tests from
silently changing meaning.

Usage:
    python scripts/phase1_uniform_flow.py --duration 6 --flow-speed 0.3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.arm_sim import ArmSim, FluidConfig
from src.controllers import ComputedTorqueController
from src.kinematics import forward_kinematics, link_relative_velocities
from src.trajectories import figure_eight_joint

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(HERE, "outputs_phase1")


def run_case(flow_velocity, duration):
    fluid = FluidConfig(flow_velocity=tuple(flow_velocity))
    sim = ArmSim(backend="mujoco", fluid=fluid)
    ctrl = ComputedTorqueController(kp=20.0**2, kd=40.0)
    q0, qd0, _ = figure_eight_joint(0.0)
    sim.reset(q0, qd0)

    keys = ("time", "q", "qdot", "qddot", "commanded_torque",
            "applied_torque", "fluid_torque", "flow_velocity",
            "link_relative_velocity", "reference_q", "reference_qdot",
            "reference_qddot", "end_effector")
    log = {k: [] for k in keys}

    for k in range(int(duration / sim.dt)):
        t = k * sim.dt
        q, qd = sim.q, sim.qd
        q_ref, qd_ref, qdd_ref = figure_eight_joint(t)
        tau_cmd = ctrl(q, qd, q_ref, qd_ref, qdd_ref)
        sim.set_torque(tau_cmd)

        log["time"].append(t)
        log["q"].append(q)
        log["qdot"].append(qd)
        log["qddot"].append(sim.acceleration())
        log["commanded_torque"].append(tau_cmd)
        log["applied_torque"].append(sim.joint_torque())
        log["fluid_torque"].append(sim.passive_torque())
        log["flow_velocity"].append(sim.flow_velocity)
        log["link_relative_velocity"].append(
            link_relative_velocities(q, qd, sim.flow_velocity))
        log["reference_q"].append(q_ref)
        log["reference_qdot"].append(qd_ref)
        log["reference_qddot"].append(qdd_ref)
        log["end_effector"].append(forward_kinematics(q))
        sim.step()

    return {k: np.asarray(v) for k, v in log.items()}


def rmse(log):
    return float(np.sqrt(np.mean((log["q"] - log["reference_q"])**2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--flow-speed", type=float, default=0.3)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.duration <= 0 or args.flow_speed < 0:
        ap.error("duration must be positive and flow-speed non-negative")

    os.makedirs(args.out, exist_ok=True)
    still = run_case((0.0, 0.0, 0.0), args.duration)
    current = run_case((args.flow_speed, 0.0, 0.0), args.duration)
    np.savez(os.path.join(args.out, "still_water.npz"), **still)
    np.savez(os.path.join(args.out, "uniform_current.npz"), **current)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for name, log in (("still water", still), ("uniform current", current)):
        t = log["time"]
        err = np.linalg.norm(log["q"] - log["reference_q"], axis=1)
        ax[0, 0].plot(t, err, label=name)
        ax[0, 1].plot(t, log["fluid_torque"][:, 0], label=f"joint 1: {name}")
        ax[1, 0].plot(t, log["fluid_torque"][:, 1], label=f"joint 2: {name}")
        rel_speed = np.linalg.norm(log["link_relative_velocity"], axis=2)
        ax[1, 1].plot(t, rel_speed[:, 0], label=f"link 1: {name}")
        ax[1, 1].plot(t, rel_speed[:, 1], "--", label=f"link 2: {name}")
    ax[0, 0].set_title("Joint tracking error norm")
    ax[0, 1].set_title("Passive fluid torque — joint 1")
    ax[1, 0].set_title("Passive fluid torque — joint 2")
    ax[1, 1].set_title("Link COM relative speeds")
    for a in ax.flat:
        a.set_xlabel("time (s)"); a.grid(alpha=0.25); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "uniform_flow_smoke_test.png"), dpi=140)
    plt.close(fig)

    delta = np.linalg.norm(current["fluid_torque"] - still["fluid_torque"], axis=1)
    print(f"still-water joint RMSE: {rmse(still):.6f} rad")
    print(f"uniform-current joint RMSE: {rmse(current):.6f} rad")
    print(f"mean current-induced torque difference: {delta.mean():.6f} N m")
    print(f"outputs: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
