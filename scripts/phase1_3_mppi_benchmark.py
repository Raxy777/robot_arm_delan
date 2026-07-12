"""Phase 1.3 compute benchmark for the production MPPI configuration."""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
from src.mppi import MPPIConfig, MPPIController, TorchRigidBodyDynamics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=25)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default=None)
    args = p.parse_args()
    cfg = MPPIConfig()
    model = TorchRigidBodyDynamics(device=args.device or "cpu")
    ctrl = MPPIController(model, cfg, device=args.device, seed=0)
    state = np.zeros(4)
    reference = np.zeros((cfg.horizon+1, 4))
    samples = []
    ctrl.command(state, reference)  # warm-up
    if ctrl.device.type == "cuda":
        import torch; torch.cuda.synchronize()
    for _ in range(args.iterations):
        tic = time.perf_counter_ns()
        ctrl.command(state, reference)
        if ctrl.device.type == "cuda":
            import torch; torch.cuda.synchronize()
        samples.append((time.perf_counter_ns()-tic)/1e6)
    result = {
        "device": str(ctrl.device), "samples": cfg.samples, "horizon": cfg.horizon,
        "lambda": cfg.lambda_, "control_rate_hz": cfg.control_rate_hz,
        "tracker_rate_hz": cfg.tracker_rate_hz, "iterations": args.iterations,
        "solve_mean_ms": float(np.mean(samples)),
        "solve_p95_ms": float(np.percentile(samples, 95)),
        "solve_worst_ms": float(np.max(samples)),
        "deadline_ms": 1000.0/cfg.control_rate_hz,
        "deadline_pass_p95": bool(np.percentile(samples, 95) <= 1000.0/cfg.control_rate_hz),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
