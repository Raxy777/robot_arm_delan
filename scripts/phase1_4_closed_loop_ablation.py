"""Nominal-vs-residual MPPI ablation with flow error, noise, and delay stress.

The plant here is the structured MuJoCo-inspired Phase 1 model, not CFD or
hardware truth. Passing this test is necessary plumbing evidence, not learned
model validation; the held-out calibration gate remains separate.
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from src.mppi import MPPIConfig, MPPIController, TorchRigidBodyDynamics, InterpolatingTorqueTracker
from src.phase1_4_gates import ClosedLoopGate, evaluate_closed_loop
from src.residual_dynamics import StructuredHydrodynamicDynamics


def reference(t):
    return np.array([
        0.65 * np.sin(0.8 * t), 
        0.5 * np.cos(0.6 * t),
        0.52 * np.cos(0.8 * t), 
        -0.3 * np.sin(0.6 * t)
    ], dtype=np.float32)


def rollout(kind, flow, flow_error, noise, delay_ms, seed, args):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    
    cfg = MPPIConfig(samples=args.samples, horizon=args.horizon)
    plant = StructuredHydrodynamicDynamics(device=args.device).set_flow(flow)
    
    if kind == "residual":
        estimate = np.asarray(flow) * (1.0 + flow_error)
        model = StructuredHydrodynamicDynamics(device=args.device).set_flow(estimate)
    else: 
        model = TorchRigidBodyDynamics(device=args.device)
        
    ctrl = MPPIController(model, cfg, device=args.device, seed=seed)
    tracker = InterpolatingTorqueTracker(cfg)
    
    x = torch.zeros(4, device=args.device)
    history = deque([np.zeros(4, np.float32)])
    delay_ticks = int(delay_ms)
    errors = []
    solve = []
    ticks = int(args.duration * cfg.tracker_rate_hz)
    
    for tick in range(ticks):
        t = tick / cfg.tracker_rate_hz
        true = x.detach().cpu().numpy()
        history.append(true.copy())
        
        if tick % cfg.ticks_per_plan == 0:
            delayed = history[max(0, len(history) - 1 - delay_ticks)].copy()
            measured = delayed + rng.normal(0, noise, 4)
            times = t + np.arange(cfg.horizon + 1) * cfg.control_dt
            ref = np.stack([reference(z) for z in times])
            
            tic = time.perf_counter_ns()
            tracker.set_target(ctrl.command(measured, ref))
            
            if args.device.startswith("cuda"): 
                torch.cuda.synchronize()
                
            solve.append((time.perf_counter_ns() - tic) / 1e6)
            
        torque = torch.as_tensor(tracker.step(), device=args.device, dtype=torch.float32)
        x = plant(x, torque, 1 / cfg.tracker_rate_hz)
        errors.append(np.mean((true - reference(t))[:2]**2))
        
    return {
        "tracking_rmse": float(np.sqrt(np.mean(errors))),
        "solve_p95_ms": float(np.percentile(solve, 95)), 
        "solve_worst_ms": float(np.max(solve))
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--horizon", type=int, default=12)
    p.add_argument("--duration", type=float, default=0.6)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--flows", nargs="+", default=("0.2,0", "0.3,0.15", "-0.2,0.25"))
    p.add_argument("--quick", action="store_true", help="one seed; diagnostic only and cannot pass coverage")
    
    a = p.parse_args()
    
    flows = [tuple(map(float, x.split(','))) for x in a.flows]
    seeds = range(1 if a.quick else a.seeds)
    gate = ClosedLoopGate()
    
    # Stress one factor at a time so failures remain attributable.
    conditions = []
    conditions += [(e, 0.0, 0) for e in gate.required_flow_error_levels]
    conditions += [(0.0, n, 0) for n in gate.required_noise_std if n]
    conditions += [(0.0, 0.0, d) for d in gate.required_sensor_delays_ms if d]
    
    rows = []
    for flow in flows:
        for seed in seeds:
            for ferr, noise, delay in conditions:
                base = {
                    "flow": flow, 
                    "seed": seed, 
                    "flow_error": ferr, 
                    "noise_std": noise, 
                    "delay_ms": delay
                }
                for kind in ("nominal", "residual"):
                    rows.append({
                        **base, 
                        "model": kind, 
                        **rollout(kind, flow, ferr, noise, delay, seed, a)
                    })
                    
    core = [r for r in rows if r["flow_error"] == r["noise_std"] == r["delay_ms"] == 0]
    nom = [r["tracking_rmse"] for r in core if r["model"] == "nominal"]
    res = [r["tracking_rmse"] for r in core if r["model"] == "residual"]
    
    decision = evaluate_closed_loop(nom, res, len(flows), len(list(seeds)), gate)
    deadline = 20.0
    residual_rows = [r for r in rows if r["model"] == "residual"]
    
    decision["timing_p95_worst_case_ms"] = max(r["solve_p95_ms"] for r in residual_rows)
    decision["timing_gate_pass"] = decision["timing_p95_worst_case_ms"] <= deadline
    decision["passed"] = decision["passed"] and decision["timing_gate_pass"]
    
    result = {
        "status": "uncalibrated structured-model plant; not CFD/hardware validation",
        "config": vars(a),
        "decision": decision,
        "runs": rows
    }
    
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    
    print(json.dumps({
        "status": result["status"], 
        "decision": decision, 
        "run_count": len(rows)
    }, indent=2))


if __name__ == "__main__": 
    main()