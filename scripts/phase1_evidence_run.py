"""Run the reproducible Phase-1 collection, calibration, and held-out gates.

The default profile creates 70 three-second trajectories (210,000 samples) over
seven fixed uniform flows and three causal sensor/actuator delay pairs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from scripts.phase1_latency_data import collect_dataset
from src.dataset_splits import export_grouped_splits
from src.phase1_evidence import merge_dataset_parts, write_run_config, write_sha256_manifest

DEFAULT_FLOWS = (
    (0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.3, 0.0, 0.0),
    (0.45, 0.0, 0.0), (-0.3, 0.0, 0.0), (0.0, 0.3, 0.0),
    (0.212, 0.212, 0.0),
)


def run(command, log_path):
    print("+", " ".join(map(str, command)), flush=True)
    with Path(log_path).open("w", encoding="utf-8") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="phase1_evidence")
    parser.add_argument("--flow", type=float, nargs=3, action="append", dest="flows",
                        help="repeat to override the seven default world-frame flows")
    parser.add_argument("--episodes-per-flow", type=int, default=10)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--collection-seed", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--split-ratios", type=float, nargs=3, default=(0.6, 0.2, 0.2))
    parser.add_argument("--control-period-steps", type=int, default=10)
    parser.add_argument("--sensor-delays", type=int, nargs="+", default=(0, 10, 20))
    parser.add_argument("--actuation-delays", type=int, nargs="+", default=(0, 5, 10))
    parser.add_argument("--calibration-steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--stride-steps", type=int, default=100)
    parser.add_argument("--min-improvement", type=float, default=10.0)
    parser.add_argument("--max-position-rmse", type=float)
    parser.add_argument("--max-position-p95", type=float)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.episodes_per_flow < 1 or args.duration <= 0 or args.calibration_steps < 1:
        parser.error("episodes-per-flow, duration, and calibration-steps must be positive")
    if any(x < 0 for x in (*args.sensor_delays, *args.actuation_delays)):
        parser.error("delay steps must be non-negative")

    root = Path(args.out_dir).resolve()
    raw_dir, split_dir, result_dir = root / "raw", root / "splits", root / "results"
    for path in (raw_dir, split_dir, result_dir):
        path.mkdir(parents=True, exist_ok=True)
    flows = tuple(tuple(flow) for flow in (args.flows or DEFAULT_FLOWS))
    config = vars(args).copy(); config["flows"] = flows
    write_run_config(root / "run_config.json", config)

    parts, raw_paths = [], []
    for i, flow in enumerate(flows):
        data = collect_dataset(
            args.episodes_per_flow, args.duration, flow, args.collection_seed + i,
            args.control_period_steps, args.sensor_delays, args.actuation_delays)
        path = raw_dir / f"flow_{i:02d}.npz"
        np.savez_compressed(path, **data)
        raw_paths.append(path); parts.append(data)
        print(f"collected flow {flow}: {len(data['trajectory_id'])} samples")
    combined = merge_dataset_parts(parts)
    combined_path = root / "phase1_multiflow_raw.npz"
    np.savez_compressed(combined_path, **combined)
    manifest = export_grouped_splits(
        combined_path, split_dir, args.split_ratios, args.split_seed, "trajectory_id")

    model_path = result_dir / "phase1_global_hydrodynamics.json"
    calibration_log = result_dir / "calibration.log"
    run([sys.executable, "scripts/phase1_calibrate_hydrodynamics.py", str(split_dir),
         "--out", str(model_path), "--steps", str(args.calibration_steps),
         "--batch-size", str(args.batch_size), "--seed", str(args.split_seed),
         "--min-test-improvement", str(args.min_improvement)], calibration_log)

    gate_path, plot_path = result_dir / "heldout_gate.json", result_dir / "heldout_gate.png"
    gate_command = [
        sys.executable, "scripts/phase1_heldout_rollout_gate.py", str(model_path),
        str(split_dir / "test.npz"), "--out", str(gate_path), "--plot", str(plot_path),
        "--horizon-steps", str(args.horizon_steps), "--stride-steps", str(args.stride_steps),
        "--min-torque-improvement", str(args.min_improvement),
        "--min-rollout-improvement", str(args.min_improvement),
    ]
    if args.max_position_rmse is not None:
        gate_command += ["--max-position-rmse", str(args.max_position_rmse)]
    if args.max_position_p95 is not None:
        gate_command += ["--max-position-p95", str(args.max_position_p95)]
    run(gate_command, result_dir / "gate.log")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    summary = {
        "samples": int(len(combined["trajectory_id"])),
        "trajectories": int(len(np.unique(combined["trajectory_id"]))),
        "flows": [list(x) for x in flows], "split_manifest": manifest,
        "gate_passed": bool(gate["passed"]), "gates": gate["gates"],
        "qualification": ("Relative improvement gates do not establish absolute control accuracy "
                          "unless max-position-rmse and max-position-p95 are configured."),
    }
    summary_path = root / "evidence_summary.json"
    write_run_config(summary_path, summary)
    tracked = raw_paths + [combined_path, root / "run_config.json", summary_path,
        split_dir / "train.npz", split_dir / "validation.npz", split_dir / "test.npz",
        split_dir / "split_manifest.json", model_path, gate_path, plot_path,
        calibration_log, result_dir / "gate.log"]
    write_sha256_manifest(root, tracked)
    print(json.dumps(summary, indent=2))
    if args.fail_on_gate and not gate["passed"]:
        raise SystemExit("held-out gate failed")


if __name__ == "__main__":
    main()
