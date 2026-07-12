"""Compute benchmark for MPPI using the Phase 1.4 structured residual model."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.mppi import MPPIConfig, MPPIController
from src.residual_dynamics import StructuredHydrodynamicDynamics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--flow", type=float, nargs=2, default=(0.3, 0.0))
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MPPIConfig()
    model = StructuredHydrodynamicDynamics(device=device).set_flow(args.flow)
    controller = MPPIController(model, cfg, device=device, seed=0)
    state = np.zeros(4)
    reference = np.zeros((cfg.horizon+1, 4))

    controller.command(state, reference)
    if controller.device.type == "cuda":
        torch.cuda.synchronize()
    samples_ms = []
    for _ in range(args.iterations):
        tic = time.perf_counter_ns()
        controller.command(state, reference)
        if controller.device.type == "cuda":
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter_ns()-tic)/1e6)

    deadline_ms = 1000.0/cfg.control_rate_hz
    result = {
        "model": "structured_hydrodynamic_residual",
        "model_status": "uncalibrated_initial_coefficients",
        "device": str(controller.device),
        "flow_xy_mps": list(args.flow),
        "samples": cfg.samples,
        "horizon": cfg.horizon,
        "iterations": args.iterations,
        "solve_mean_ms": float(np.mean(samples_ms)),
        "solve_p95_ms": float(np.percentile(samples_ms, 95)),
        "solve_worst_ms": float(np.max(samples_ms)),
        "deadline_ms": deadline_ms,
        "deadline_pass_p95": bool(np.percentile(samples_ms, 95) <= deadline_ms),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
