"""Apply the pre-registered calibration gate to held-out rollout results."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.phase1_4_gates import CalibrationGate, evaluate_calibration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", help="NPZ with nominal_rmse, residual_rmse, flow_id, seed")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    
    data = np.load(a.dataset)
    required = {"nominal_rmse", "residual_rmse", "flow_id", "seed"}
    missing = required - set(data.files)
    
    if missing: 
        p.error("missing NPZ arrays: " + ", ".join(sorted(missing)))
        
    n = len(data["nominal_rmse"])
    if any(len(data[x]) != n for x in required): 
        p.error("all arrays must have equal length")
        
    result = evaluate_calibration(
        data["nominal_rmse"], 
        data["residual_rmse"],
        len(np.unique(data["flow_id"])), 
        len(np.unique(data["seed"])),
        CalibrationGate()
    )
    
    result["dataset"] = str(Path(a.dataset))
    result["cases"] = n
    
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__": 
    main()