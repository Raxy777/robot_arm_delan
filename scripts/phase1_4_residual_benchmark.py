"""Representative-state latency sweep for structured-residual MPPI."""
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


def scenarios(horizon, count):
    """Deterministic moving states/references spanning benign and fast motion."""
    for k in range(count):
        phase = 2 * np.pi * k / max(count, 1)
        state = np.array([
            0.8 * np.sin(phase), 
            0.6 * np.cos(0.7 * phase),
            1.2 * np.cos(phase), 
            -0.9 * np.sin(0.7 * phase)
        ], np.float32)
        
        t = np.arange(horizon + 1) * 0.02
        ref = np.column_stack((
            0.9 * np.sin(phase + 1.1 * t), 
            0.7 * np.cos(0.7 * phase + 0.8 * t),
            0.99 * np.cos(phase + 1.1 * t), 
            -0.56 * np.sin(0.7 * phase + 0.8 * t)
        )).astype(np.float32)
        
        yield state, ref


def measure(samples, horizon, args, device):
    cfg = MPPIConfig(samples=samples, horizon=horizon)
    model = StructuredHydrodynamicDynamics(device=device, dtype=torch.float32).set_flow(args.flow)
    
    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires torch.compile")
        model = torch.compile(model, mode="reduce-overhead")
        
    controller = MPPIController(model, cfg, device=device, seed=0)
    cases = list(scenarios(horizon, max(args.warmup, args.iterations)))
    
    for state, ref in cases[:args.warmup]:
        controller.command(state, ref)
        
    if controller.device.type == "cuda": 
        torch.cuda.synchronize()
        
    times = []
    for state, ref in cases[:args.iterations]:
        tic = time.perf_counter_ns()
        controller.command(state, ref)
        
        if controller.device.type == "cuda": 
            torch.cuda.synchronize()
            
        times.append((time.perf_counter_ns() - tic) / 1e6)
        
    p95 = float(np.percentile(times, 95))
    deadline = 1000 / cfg.control_rate_hz
    
    return {
        "samples": samples, 
        "horizon": horizon, 
        "mean_ms": float(np.mean(times)),
        "p95_ms": p95, 
        "worst_ms": float(np.max(times)), 
        "deadline_ms": deadline,
        "deadline_pass_p95": bool(p95 <= deadline)
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--flow", type=float, nargs=2, default=(0.3, 0.0))
    p.add_argument("--profiles", nargs="+", default=("1024x25", "768x20", "512x20"))
    p.add_argument("--compile", action="store_true")
    
    a = p.parse_args()
    
    if a.iterations < 20 or a.warmup < 1: 
        p.error("use >=20 iterations and >=1 warm-up")
        
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    parsed = []
    
    for profile in a.profiles:
        try: 
            samples, horizon = map(int, profile.lower().split("x"))
        except Exception: 
            p.error(f"invalid profile {profile!r}; expected SAMPLESxHORIZON")
        parsed.append((samples, horizon))
        
    results = [measure(s, h, a, device) for s, h in parsed]
    passing = [r for r in results if r["deadline_pass_p95"]]
    selected = passing[0] if passing else None  # profiles are ordered quality-first
    
    result = {
        "model": "structured_hydrodynamic_residual", 
        "status": "uncalibrated",
        "device": str(device), 
        "dtype": "float32", 
        "flow_xy_mps": list(a.flow),
        "warmup": a.warmup, 
        "iterations": a.iterations,
        "scenario": "moving sinusoidal states and references", 
        "compiled": a.compile,
        "profiles": results, 
        "selected_50hz_profile": selected,
        "compute_gate_pass": selected is not None,
        "warning": "A reduced profile still requires the closed-loop ablation gate."
    }
    
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": 
    main()