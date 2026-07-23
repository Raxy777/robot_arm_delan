"""Evaluate a calibrated model on untouched Phase-1 trajectories."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.dataset_splits import load_npz_dataset, sha256_file
from src.heldout_rollouts import RolloutGateConfig, evaluate_heldout, save_evaluation
from src.hydrodynamic_calibration import load_calibrated_model


def save_plot(path, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Fluid torque\n(one step)", "Position\n(rollout)", "Position\n(endpoint)"]
    rigid = [result["one_step"]["rigid_only"]["fluid_torque"]["rmse"],
             result["rollout"]["rigid_only"]["position"]["rmse"],
             result["rollout"]["rigid_only"]["endpoint_position"]["rmse"]]
    hydro = [result["one_step"]["hydrodynamic"]["fluid_torque"]["rmse"],
             result["rollout"]["hydrodynamic"]["position"]["rmse"],
             result["rollout"]["hydrodynamic"]["endpoint_position"]["rmse"]]
    x = np.arange(len(labels)); width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x - width / 2, rigid, width, label="Rigid only", color="#8c8c8c")
    ax.bar(x + width / 2, hydro, width, label="Calibrated hydrodynamics", color="#1877b9")
    ax.set_xticks(x, labels); ax.set_ylabel("RMSE (native units)")
    ax.set_title("Held-out Phase-1 gates — " + ("PASS" if result["passed"] else "FAIL"))
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Task-3 calibrated JSON artifact")
    parser.add_argument("test_dataset", help="untouched test.npz")
    parser.add_argument("--out", default="results/phase1_heldout_rollout_gate.json")
    parser.add_argument("--plot", default="results/phase1_heldout_rollout_gate.png")
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--stride-steps", type=int, default=100)
    parser.add_argument("--min-torque-improvement", type=float, default=10.0)
    parser.add_argument("--min-rollout-improvement", type=float, default=10.0)
    parser.add_argument("--max-position-rmse", type=float)
    parser.add_argument("--max-position-p95", type=float)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--allow-unregistered-test", action="store_true",
                        help="allow a test file whose checksum is absent from model provenance")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    model, artifact = load_calibrated_model(args.model, args.device, getattr(torch, args.dtype))
    test_checksum = sha256_file(args.test_dataset)
    registered = artifact.get("provenance", {}).get("split_sha256", {}).get("test")
    if registered != test_checksum and not args.allow_unregistered_test:
        parser.error("test checksum does not match the untouched test split recorded by calibration; "
                     "use --allow-unregistered-test only for a separately documented external holdout")
    config = RolloutGateConfig(
        horizon_steps=args.horizon_steps, stride_steps=args.stride_steps,
        min_torque_improvement_percent=args.min_torque_improvement,
        min_rollout_improvement_percent=args.min_rollout_improvement,
        max_position_rmse=args.max_position_rmse,
        max_position_error_p95=args.max_position_p95,
    )
    result = evaluate_heldout(model, load_npz_dataset(args.test_dataset), config)
    result["provenance"] = {
        "model_artifact": str(Path(args.model).resolve()),
        "model_artifact_sha256": sha256_file(args.model),
        "test_dataset": str(Path(args.test_dataset).resolve()),
        "test_dataset_sha256": test_checksum,
        "registered_calibration_test_sha256": registered,
        "registered_test_match": registered == test_checksum,
    }
    save_evaluation(args.out, result)
    save_plot(args.plot, result)
    print(json.dumps({"passed": result["passed"], "gates": result["gates"],
                      "result": str(Path(args.out).resolve()),
                      "plot": str(Path(args.plot).resolve())}, indent=2))
    if args.fail_on_gate and not result["passed"]:
        raise SystemExit("held-out rollout gate failed")


if __name__ == "__main__":
    main()
