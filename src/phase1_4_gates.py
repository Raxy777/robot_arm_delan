"""Pre-registered Phase 1.4 calibration and closed-loop acceptance gates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class CalibrationGate:
    """Criteria fixed before fitting to prevent post-hoc success definitions."""
    
    rollout_steps: int = 25
    minimum_flow_regimes: int = 3
    minimum_seeds: int = 5
    maximum_relative_rollout_rmse: float = 0.75
    minimum_case_win_fraction: float = 0.80


@dataclass(frozen=True)
class ClosedLoopGate:
    minimum_tracking_improvement: float = 0.10
    minimum_flow_regimes: int = 3
    minimum_seeds: int = 5
    required_flow_error_levels: tuple[float, ...] = (0.0, 0.20, -0.20, 0.50, -0.50)
    required_sensor_delays_ms: tuple[int, ...] = (0, 10, 20)
    required_noise_std: tuple[float, ...] = (0.0, 0.01, 0.03)


def evaluate_calibration(
    nominal_rmse, 
    residual_rmse, 
    flow_regimes: int, 
    seeds: int,
    gate=CalibrationGate()
):
    """Evaluate held-out N-step errors; arrays contain one value per case."""
    nominal = np.asarray(nominal_rmse, dtype=float)
    residual = np.asarray(residual_rmse, dtype=float)
    
    if nominal.shape != residual.shape or nominal.size == 0:
        raise ValueError("nominal and residual RMSE must be equal, non-empty shapes")
        
    if not np.isfinite(nominal).all() or not np.isfinite(residual).all() or np.any(nominal <= 0):
        raise ValueError("RMSE values must be finite and nominal RMSE strictly positive")
        
    relative = float(np.mean(residual) / np.mean(nominal))
    wins = float(np.mean(residual < nominal))
    
    coverage = flow_regimes >= gate.minimum_flow_regimes and seeds >= gate.minimum_seeds
    passed = coverage and relative <= gate.maximum_relative_rollout_rmse and wins >= gate.minimum_case_win_fraction
    
    return {
        "gate": asdict(gate), 
        "flow_regimes": int(flow_regimes), 
        "seeds": int(seeds),
        "relative_rollout_rmse": relative, 
        "case_win_fraction": wins,
        "coverage_pass": bool(coverage), 
        "passed": bool(passed),
    }


def evaluate_closed_loop(
    nominal_rmse, 
    residual_rmse, 
    flow_regimes: int, 
    seeds: int,
    gate=ClosedLoopGate()
):
    nominal = np.asarray(nominal_rmse, dtype=float)
    residual = np.asarray(residual_rmse, dtype=float)
    
    if nominal.shape != residual.shape or nominal.size == 0 or np.any(nominal <= 0):
        raise ValueError("closed-loop RMSE arrays must be equal/non-empty and nominal > 0")
        
    improvement = float(1.0 - np.mean(residual) / np.mean(nominal))
    coverage = flow_regimes >= gate.minimum_flow_regimes and seeds >= gate.minimum_seeds
    
    return {
        "gate": asdict(gate), 
        "flow_regimes": int(flow_regimes), 
        "seeds": int(seeds),
        "tracking_improvement": improvement, 
        "coverage_pass": bool(coverage),
        "passed": bool(coverage and improvement >= gate.minimum_tracking_improvement),
    }