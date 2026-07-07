"""Week 7 deliverable: track a figure-eight with the KNOWN model.

Runs the 2-link arm in MuJoCo under one of the three controllers, tracking the
task-space figure-eight. Logs the data, renders a video, and produces tracking
plots.

Usage:
    python run_sim.py --controller computed_torque --video
    python run_sim.py --controller pd
    python run_sim.py --controller gravity_pd --duration 12

Outputs (in ./outputs_run/):
    figure_eight.mp4      - rendered video of the arm
    tracking.png          - joint tracking + end-effector trace + torques
    log.npz               - full time series (for reuse in later weeks)
"""

import argparse
import os

import numpy as np

import mujoco

import controllers
from kinematics import forward_kinematics
from trajectories import figure_eight_joint, figure_eight

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model", "arm2.xml")
OUT_DIR = os.path.join(HERE, "outputs_run")


def make_controller(name):
    # Gains: critically-damped second-order error dynamics, wn ~ 20 rad/s.
    wn, zeta = 20.0, 1.0
    kp = wn**2
    kd = 2 * zeta * wn
    if name == "pd":
        # A pure-PD stiffness comparable in scale (no model, so gains are ad hoc)
        return controllers.PDController(kp=400.0, kd=40.0)
    if name == "gravity_pd":
        return controllers.GravityCompPDController(kp=400.0, kd=40.0)
    if name == "computed_torque":
        return controllers.ComputedTorqueController(kp=kp, kd=kd)
    raise ValueError(f"unknown controller: {name}")


def simulate(controller_name="computed_torque", duration=12.0, render=False,
             elbow_up=True):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    ctrl = make_controller(controller_name)
    dt = model.opt.timestep

    # Start the arm exactly on the reference so we measure tracking, not a
    # transient from a bad initial condition.
    q0, qd0, _ = figure_eight_joint(0.0, elbow_up=elbow_up)
    data.qpos[:] = q0
    data.qvel[:] = qd0
    mujoco.mj_forward(model, data)

    renderer = None
    frames = []
    if render:
        renderer = mujoco.Renderer(model, height=480, width=640)

    log = {k: [] for k in ("t", "q", "qd", "q_des", "tau", "ee", "ee_des")}
    n_steps = int(duration / dt)
    render_every = max(1, int((1 / 30) / dt))  # ~30 fps video

    for i in range(n_steps):
        t = i * dt
        q = data.qpos.copy()
        qd = data.qvel.copy()

        q_d, qd_d, qdd_d = figure_eight_joint(t, elbow_up=elbow_up)
        tau = ctrl(q, qd, q_d, qd_d, qdd_d)

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        x_des, _, _ = figure_eight(t)
        log["t"].append(t)
        log["q"].append(q)
        log["qd"].append(qd)
        log["q_des"].append(q_d)
        log["tau"].append(tau)
        log["ee"].append(forward_kinematics(q))
        log["ee_des"].append(x_des)

        if render and i % render_every == 0:
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())

    if renderer is not None:
        renderer.close()

    log = {k: np.array(v) for k, v in log.items()}
    return log, frames


def rms_tracking_error(log):
    """End-effector RMS tracking error in meters."""
    err = log["ee"] - log["ee_des"]
    return float(np.sqrt(np.mean(np.sum(err**2, axis=1))))


def save_outputs(log, frames, controller_name):
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "log.npz"), **log)

    # --- plots ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))

    ax[0, 0].plot(log["t"], log["q"][:, 0], label="q1")
    ax[0, 0].plot(log["t"], log["q_des"][:, 0], "--", label="q1 des")
    ax[0, 0].plot(log["t"], log["q"][:, 1], label="q2")
    ax[0, 0].plot(log["t"], log["q_des"][:, 1], "--", label="q2 des")
    ax[0, 0].set_title("Joint tracking"); ax[0, 0].set_xlabel("t (s)")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(log["ee_des"][:, 0], log["ee_des"][:, 1], "--",
                  label="reference")
    ax[0, 1].plot(log["ee"][:, 0], log["ee"][:, 1], label="actual")
    ax[0, 1].set_title("End-effector figure-eight")
    ax[0, 1].set_aspect("equal"); ax[0, 1].legend(fontsize=8)

    err = np.linalg.norm(log["ee"] - log["ee_des"], axis=1)
    ax[1, 0].plot(log["t"], err * 1000)
    ax[1, 0].set_title("EE tracking error (mm)"); ax[1, 0].set_xlabel("t (s)")

    ax[1, 1].plot(log["t"], log["tau"][:, 0], label="tau1")
    ax[1, 1].plot(log["t"], log["tau"][:, 1], label="tau2")
    ax[1, 1].set_title("Joint torques (N m)"); ax[1, 1].set_xlabel("t (s)")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle(f"Controller: {controller_name}   "
                 f"RMS EE error = {rms_tracking_error(log)*1000:.2f} mm")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "tracking.png"), dpi=120)
    plt.close(fig)

    # --- video ---
    if frames:
        try:
            import imageio
            imageio.mimsave(os.path.join(OUT_DIR, "figure_eight.mp4"),
                            frames, fps=30)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] could not write mp4 ({e}); saving frames as gif")
            import imageio
            imageio.mimsave(os.path.join(OUT_DIR, "figure_eight.gif"),
                            frames, fps=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="computed_torque",
                    choices=["pd", "gravity_pd", "computed_torque"])
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--elbow-down", action="store_true",
                    help="use elbow-down IK branch")
    args = ap.parse_args()

    log, frames = simulate(args.controller, args.duration,
                           render=args.video, elbow_up=not args.elbow_down)
    save_outputs(log, frames, args.controller)
    print(f"[{args.controller}] RMS EE tracking error = "
          f"{rms_tracking_error(log)*1000:.3f} mm")
    print(f"Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
