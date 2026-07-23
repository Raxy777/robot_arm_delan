"""Fit and evaluate the Phase-1 global structured hydrodynamic model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.dataset_splits import load_npz_dataset, sha256_file
from src.hydrodynamic_calibration import (
    CalibrationConfig, evaluate_model, fit_global_hydrodynamics,
    save_calibrated_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_dir", help="directory containing train/validation/test.npz")
    parser.add_argument("--out", default="artifacts/phase1_global_hydrodynamics.json")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--min-test-improvement", type=float, default=None,
                        help="exit nonzero unless test RMSE improves by this percentage")
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    paths = {name: split_dir / f"{name}.npz"
             for name in ("train", "validation", "test")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        parser.error(f"missing split files: {missing}")
    datasets = {name: load_npz_dataset(path) for name, path in paths.items()}
    dtype = getattr(torch, args.dtype)
    config = CalibrationConfig(
        steps=args.steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_interval=args.validation_interval, seed=args.seed,
    )
    model, fit = fit_global_hydrodynamics(
        datasets["train"], datasets["validation"], config,
        device=args.device, dtype=dtype)
    metrics = {name: evaluate_model(model, data) for name, data in datasets.items()}
    provenance = {"split_sha256": {name: sha256_file(path) for name, path in paths.items()}}
    artifact = save_calibrated_model(args.out, model, fit, metrics, provenance)
    summary = {
        "artifact": str(Path(args.out).resolve()),
        "best_step": fit["best_step"],
        "coefficients": artifact["coefficients"],
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))
    if (args.min_test_improvement is not None
            and metrics["test"]["rmse_improvement_percent"] < args.min_test_improvement):
        raise SystemExit(
            f"test improvement {metrics['test']['rmse_improvement_percent']:.3f}% "
            f"is below required {args.min_test_improvement:.3f}%")


if __name__ == "__main__":
    main()
