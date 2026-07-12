"""Phase 1.2: multi-trajectory validation of MuJoCo uniform-fluid forcing.

This is a phase-gate experiment, not a CFD validation.  It exercises a family
of reachable figure-eight references under still water and uniform currents,
records the known flow vector with every trace, and reports tracking, forcing,
saturation, and compute-time distributions.

Example:
    python scripts/phase1_2_validate_uniform_flow.py \
        --out /tmp/phase1_2 --fail-on-gate
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.arm_sim import ArmSim, FluidConfig
from src.controllers import ComputedTorqueController
from src.kinematics import forward_kinematics, link_relative_velocities
from src.trajectories import figure_eight_joint


@dataclass(frozen=True)
class TrajectoryCase:
    name: str
    duration: float
    center: tuple[float, float]
    A: float
    B: float
    period: float


TRAJECTORIES = (
    TrajectoryCase("nominal", 12.0, (1.0, 0.6), 0.50, 0.35, 6.0),
    TrajectoryCase("slow_wide", 16.0, (0.9, 0.55), 0.55, 0.30, 8.0),
    TrajectoryCase("fast_compact", 8.0, (1.0, 0.65), 0.30, 0.20, 4.0),
)

FLOWS = {
    "still": (0.0, 0.0, 0.0),
    "x_pos_030": (0.30, 0.0, 0.0),
    "x_neg_030": (-0.30, 0.0, 0.0),
    "y_pos_030": (0.0, 0.30, 0.0),
    "diag_030": (0.30 / np.sqrt(2.0), 0.30 / np.sqrt(2.0), 0.0),
}

# Conservative gates for the existing nominal computed-torque controller.
RMSE_LIMIT_RAD = 0.020
MAX_ERROR_NORM_LIMIT_RAD = 0.080
SATURATION_FRACTION_LIMIT = 0.005
MIN_CURRENT_RESPONSE_NM = 0.010


def reference(case: TrajectoryCase, t: float):
    return figure_eight_joint(
        t, center=case.center, A=case.A, B=case.B, period=case.period
    )


def percentile_us(samples_ns, percentile):
    return float(np.percentile(np.asarray(samples_ns, dtype=float), percentile) / 1e3)


def run_case(case: TrajectoryCase, flow_name: str, flow, save_path: Path | None):
    sim = ArmSim(backend="mujoco", fluid=FluidConfig(flow_velocity=tuple(flow)))
    controller = ComputedTorqueController(kp=20.0**2, kd=40.0)
    q0, qd0, _ = reference(case, 0.0)
    sim.reset(q0, qd0)

    duration = case.duration
    steps = int(round(duration / sim.dt))
    keys = (
        "time", "q", "qdot", "qddot", "reference_q", "reference_qdot",
        "reference_qddot", "commanded_torque", "applied_torque",
        "fluid_torque", "flow_velocity", "link_relative_velocity",
        "end_effector",
    )
    log = {key: [] for key in keys}
    control_ns, simulation_ns = [], []
    saturated = []

    for k in range(steps):
        t = k * sim.dt
        q, qd = sim.q, sim.qd

        tic = time.perf_counter_ns()
        q_ref, qd_ref, qdd_ref = reference(case, t)
        tau_cmd = controller(q, qd, q_ref, qd_ref, qdd_ref)
        control_ns.append(time.perf_counter_ns() - tic)

        saturated.append(bool(np.any(np.abs(tau_cmd) > sim.tau_limit)))
        tic = time.perf_counter_ns()
        sim.set_torque(tau_cmd)
        qdd = sim.acceleration()
        tau_applied = sim.joint_torque()
        tau_fluid = sim.passive_torque()
        rel_velocity = link_relative_velocities(q, qd, sim.flow_velocity)
        sim.step()
        simulation_ns.append(time.perf_counter_ns() - tic)

        values = (
            t, q, qd, qdd, q_ref, qd_ref, qdd_ref, tau_cmd, tau_applied,
            tau_fluid, sim.flow_velocity, rel_velocity, forward_kinematics(q),
        )
        for key, value in zip(keys, values):
            log[key].append(value)

    arrays = {key: np.asarray(value) for key, value in log.items()}
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_path, **arrays)

    error = arrays["q"] - arrays["reference_q"]
    error_norm = np.linalg.norm(error, axis=1)
    fluid_norm = np.linalg.norm(arrays["fluid_torque"], axis=1)
    finite = all(np.all(np.isfinite(value)) for value in arrays.values())
    loop_ns = np.asarray(control_ns) + np.asarray(simulation_ns)
    return {
        "trajectory": case.name,
        "flow": flow_name,
        "flow_x_mps": float(flow[0]),
        "flow_y_mps": float(flow[1]),
        "duration_s": duration,
        "samples": steps,
        "finite": finite,
        "joint_rmse_rad": float(np.sqrt(np.mean(error**2))),
        "max_error_norm_rad": float(np.max(error_norm)),
        "mean_fluid_torque_norm_nm": float(np.mean(fluid_norm)),
        "max_fluid_torque_norm_nm": float(np.max(fluid_norm)),
        "saturation_fraction": float(np.mean(saturated)),
        "control_mean_us": float(np.mean(control_ns) / 1e3),
        "control_p95_us": percentile_us(control_ns, 95),
        "control_max_us": float(np.max(control_ns) / 1e3),
        "simulation_mean_us": float(np.mean(simulation_ns) / 1e3),
        "simulation_p95_us": percentile_us(simulation_ns, 95),
        "simulation_max_us": float(np.max(simulation_ns) / 1e3),
        "loop_mean_us": float(np.mean(loop_ns) / 1e3),
        "loop_p95_us": percentile_us(loop_ns, 95),
        "loop_max_us": float(np.max(loop_ns) / 1e3),
        "_fluid_torque": arrays["fluid_torque"],
        "_loop_ns": loop_ns,
    }


def apply_gates(rows):
    by_key = {(row["trajectory"], row["flow"]): row for row in rows}
    failures = []
    for row in rows:
        row["current_response_nm"] = 0.0
        if row["flow"] != "still":
            baseline = by_key[(row["trajectory"], "still")]["_fluid_torque"]
            row["current_response_nm"] = float(
                np.mean(np.linalg.norm(row["_fluid_torque"] - baseline, axis=1))
            )
        reasons = []
        if not row["finite"]:
            reasons.append("non-finite trace")
        if row["joint_rmse_rad"] > RMSE_LIMIT_RAD:
            reasons.append(f"RMSE>{RMSE_LIMIT_RAD:g} rad")
        if row["max_error_norm_rad"] > MAX_ERROR_NORM_LIMIT_RAD:
            reasons.append(f"max error>{MAX_ERROR_NORM_LIMIT_RAD:g} rad")
        if row["saturation_fraction"] > SATURATION_FRACTION_LIMIT:
            reasons.append(f"saturation>{SATURATION_FRACTION_LIMIT:g}")
        if row["flow"] != "still" and row["current_response_nm"] < MIN_CURRENT_RESPONSE_NM:
            reasons.append(f"current response<{MIN_CURRENT_RESPONSE_NM:g} N m")
        row["gate_pass"] = not reasons
        row["gate_reasons"] = "; ".join(reasons)
        if reasons:
            failures.append(f'{row["trajectory"]}/{row["flow"]}: {row["gate_reasons"]}')
    return failures


def public_row(row):
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_outputs(out: Path, rows, failures):
    public = [public_row(row) for row in rows]
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(public[0]))
        writer.writeheader()
        writer.writerows(public)
    with open(out / "summary.json", "w") as handle:
        json.dump(public, handle, indent=2)
    config = {
        "trajectories": [asdict(case) for case in TRAJECTORIES],
        "flows": FLOWS,
        "gates": {
            "rmse_limit_rad": RMSE_LIMIT_RAD,
            "max_error_norm_limit_rad": MAX_ERROR_NORM_LIMIT_RAD,
            "saturation_fraction_limit": SATURATION_FRACTION_LIMIT,
            "min_current_response_nm": MIN_CURRENT_RESPONSE_NM,
        },
        "model_scope": "MuJoCo simplified fluid-force model; not CFD ground truth",
    }
    with open(out / "experiment_config.json", "w") as handle:
        json.dump(config, handle, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    names = [f'{r["trajectory"]}\n{r["flow"]}' for r in rows]
    x = np.arange(len(rows))
    axes[0].bar(x, [r["joint_rmse_rad"] for r in rows])
    axes[0].axhline(RMSE_LIMIT_RAD, color="r", linestyle="--", label="gate")
    axes[0].set_ylabel("joint RMSE (rad)")
    axes[1].bar(x, [r["current_response_nm"] for r in rows])
    axes[1].set_ylabel("mean current response (N m)")
    axes[2].bar(x, [r["loop_p95_us"] for r in rows])
    axes[2].set_ylabel("measured loop p95 (us)")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(names, rotation=90, fontsize=7)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle("Phase 1.2 uniform-flow validation")
    fig.tight_layout()
    fig.savefig(out / "phase1_2_summary.png", dpi=160)
    plt.close(fig)

    verdict = "PASS" if not failures else "FAIL"
    worst_rmse = max(rows, key=lambda row: row["joint_rmse_rad"])
    worst_error = max(rows, key=lambda row: row["max_error_norm_rad"])
    worst_p95 = max(rows, key=lambda row: row["loop_p95_us"])
    all_loop_ns = np.concatenate([row["_loop_ns"] for row in rows])
    aggregate_loop_mean_us = float(np.mean(all_loop_ns) / 1e3)
    aggregate_loop_p95_us = percentile_us(all_loop_ns, 95)
    aggregate_loop_max_us = float(np.max(all_loop_ns) / 1e3)
    lines = [
        "# Phase 1.2 Uniform-Flow Validation", "",
        f"**Gate verdict: {verdict}**", "",
        "This validates the repository's MuJoCo simplified fluid-force model; it is not CFD validation.", "",
        f"- Cases: {len(rows)} ({len(TRAJECTORIES)} trajectories × {len(FLOWS)} flow regimes)",
        f'- Worst joint RMSE: `{worst_rmse["joint_rmse_rad"]:.6f} rad` ({worst_rmse["trajectory"]}/{worst_rmse["flow"]})',
        f'- Worst error norm: `{worst_error["max_error_norm_rad"]:.6f} rad` ({worst_error["trajectory"]}/{worst_error["flow"]})',
        f'- Measured loop timing (all samples): mean `{aggregate_loop_mean_us:.2f} us`, p95 `{aggregate_loop_p95_us:.2f} us`, worst `{aggregate_loop_max_us:.2f} us`',
        f'- Worst per-case loop p95: `{worst_p95["loop_p95_us"]:.2f} us` ({worst_p95["trajectory"]}/{worst_p95["flow"]})',
        f'- Maximum saturation fraction: `{max(row["saturation_fraction"] for row in rows):.6f}`',
        "",
        "## Gate failures", "",
    ]
    lines.extend([f"- {failure}" for failure in failures] or ["- None"])
    lines += ["", "Detailed case metrics are in `summary.csv`; full traces are in `traces/`.", ""]
    (out / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="output directory (prefer outside source tree)")
    parser.add_argument("--no-traces", action="store_true", help="do not save per-case NPZ traces")
    parser.add_argument("--fail-on-gate", action="store_true", help="exit nonzero if an acceptance gate fails")
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in TRAJECTORIES:
        for flow_name, flow in FLOWS.items():
            trace_path = None if args.no_traces else out / "traces" / f"{case.name}__{flow_name}.npz"
            print(f"running {case.name}/{flow_name} ({case.duration:.1f} s)", flush=True)
            rows.append(run_case(case, flow_name, flow, trace_path))

    failures = apply_gates(rows)
    write_outputs(out, rows, failures)
    print(f"gate: {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"  {failure}")
    print(f"outputs: {out}")
    if failures and args.fail_on_gate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
