"""Held-out one-step and finite-horizon gates for Phase-1 fluid dynamics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
from typing import Mapping

import numpy as np
import torch

from src.hydrodynamic_calibration import validate_calibration_dataset

REQUIRED_FIELDS = (
    "q_true", "qdot_true", "qdd_true", "tau_applied", "flow_velocity",
    "fluid_torque", "trajectory_id", "time_true",
)


@dataclass(frozen=True)
class RolloutGateConfig:
    """Evaluation horizon and independently configurable acceptance gates."""
    horizon_steps: int = 100
    stride_steps: int = 100
    min_torque_improvement_percent: float = 10.0
    min_rollout_improvement_percent: float = 10.0
    max_position_rmse: float | None = None
    max_position_error_p95: float | None = None
    added_mass_eigenvalue_floor: float = 1.0e-12

    def __post_init__(self):
        if self.horizon_steps < 1 or self.stride_steps < 1:
            raise ValueError("horizon_steps and stride_steps must be positive")
        for name in ("min_torque_improvement_percent", "min_rollout_improvement_percent"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        for name in ("max_position_rmse", "max_position_error_p95"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive when set")
        if self.added_mass_eigenvalue_floor <= 0:
            raise ValueError("added_mass_eigenvalue_floor must be positive")


def validate_rollout_dataset(dataset: Mapping[str, np.ndarray]) -> int:
    """Check the held-out schema, alignment, finiteness, and time ordering."""
    validate_calibration_dataset(dataset)
    missing = [name for name in REQUIRED_FIELDS if name not in dataset]
    if missing:
        raise ValueError(f"rollout dataset is missing fields: {missing}")
    n = len(np.asarray(dataset["q_true"]))
    shapes = {"fluid_torque": (2,), "trajectory_id": (), "time_true": ()}
    for name, tail in shapes.items():
        values = np.asarray(dataset[name])
        if values.shape != (n,) + tail:
            raise ValueError(f"{name} must have shape {(n,) + tail}")
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite numeric values")
    trajectory = np.asarray(dataset["trajectory_id"])
    time = np.asarray(dataset["time_true"], dtype=float)
    for value in np.unique(trajectory):
        rows = np.flatnonzero(trajectory == value)
        if len(rows) > 1 and np.any(np.diff(time[rows]) <= 0):
            raise ValueError(f"trajectory {value!r} is not strictly time ordered")
    return n


def _tensor(values, model):
    return torch.as_tensor(np.asarray(values), device=model.device, dtype=model.dtype)


def _rigid_acceleration(model, state, torque):
    mass, bias = model.rigid_body_terms(state)
    return torch.linalg.solve(mass, (torque - bias).unsqueeze(-1)).squeeze(-1)


def _error_metrics(prediction, truth):
    error = prediction - truth
    row_norm = torch.linalg.vector_norm(error, dim=-1)
    return {
        "rmse": float(torch.sqrt(torch.mean(torch.square(error))).cpu()),
        "mae": float(torch.mean(torch.abs(error)).cpu()),
        "error_norm_p95": float(torch.quantile(row_norm.flatten(), 0.95).cpu()),
        "error_norm_worst": float(torch.max(row_norm).cpu()),
    }


def _improvement(candidate, baseline):
    return 100.0 * (1.0 - candidate / max(baseline, 1.0e-12))


def one_step_metrics(model, dataset, rows=None):
    """Compare fluid torque and acceleration against a rigid-only baseline."""
    validate_rollout_dataset(dataset)
    if rows is None:
        rows = np.arange(len(dataset["q_true"]))
    rows = np.asarray(rows, dtype=int)
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("one-step evaluation rows must be non-empty")
    state = _tensor(np.c_[dataset["q_true"][rows], dataset["qdot_true"][rows]], model)
    torque = _tensor(dataset["tau_applied"][rows], model)
    flow = _tensor(dataset["flow_velocity"][rows], model)
    qdd = _tensor(dataset["qdd_true"][rows], model)
    observed_fluid = _tensor(dataset["fluid_torque"][rows], model)
    with torch.no_grad():
        acceleration = model.acceleration(state, torque, flow)
        rigid_acceleration = _rigid_acceleration(model, state, torque)
        added_mass, drag_torque, _ = model.residual_terms(state, flow)
        # The equivalent generalized fluid force at the measured acceleration.
        predicted_fluid = drag_torque - torch.einsum("...ij,...j->...i", added_mass, qdd)
        zero_fluid = torch.zeros_like(observed_fluid)
    hyd_acc = _error_metrics(acceleration, qdd)
    rigid_acc = _error_metrics(rigid_acceleration, qdd)
    hyd_tau = _error_metrics(predicted_fluid, observed_fluid)
    rigid_tau = _error_metrics(zero_fluid, observed_fluid)
    return {
        "samples": int(len(rows)),
        "hydrodynamic": {"acceleration": hyd_acc, "fluid_torque": hyd_tau},
        "rigid_only": {"acceleration": rigid_acc, "fluid_torque": rigid_tau},
        "improvement_percent": {
            "acceleration_rmse": _improvement(hyd_acc["rmse"], rigid_acc["rmse"]),
            "fluid_torque_rmse": _improvement(hyd_tau["rmse"], rigid_tau["rmse"]),
        },
    }


def _trajectory_windows(dataset, horizon, stride):
    groups = np.asarray(dataset["trajectory_id"])
    windows = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        for start in range(0, len(rows) - horizon, stride):
            window = rows[start:start + horizon + 1]
            times = np.asarray(dataset["time_true"])[window]
            differences = np.diff(times)
            dt = float(np.median(differences))
            if not np.allclose(differences, dt, rtol=1e-5, atol=1e-9):
                raise ValueError(f"trajectory {group!r} has a nonuniform sample period")
            windows.append((group, window, dt))
    if not windows:
        raise ValueError("no complete rollout windows; shorten --horizon-steps")
    return windows


def rollout_metrics(model, dataset, horizon_steps=100, stride_steps=100):
    """Open-loop recorded-input rollouts, reset at every held-out window."""
    validate_rollout_dataset(dataset)
    windows = _trajectory_windows(dataset, horizon_steps, stride_steps)
    initial = np.stack([
        np.r_[dataset["q_true"][rows[0]], dataset["qdot_true"][rows[0]]]
        for _, rows, _ in windows
    ])
    hydro = _tensor(initial, model)
    rigid = hydro.clone()
    hydro_history, rigid_history, truth_history = [], [], []
    with torch.no_grad():
        for step in range(horizon_steps):
            torque = _tensor(np.stack([dataset["tau_applied"][rows[step]]
                                       for _, rows, _ in windows]), model)
            flow = _tensor(np.stack([dataset["flow_velocity"][rows[step]]
                                     for _, rows, _ in windows]), model)
            dt = _tensor(np.asarray([value for _, _, value in windows])[:, None], model)
            hydro_qdd = model.acceleration(hydro, torque, flow)
            rigid_qdd = _rigid_acceleration(model, rigid, torque)
            hydro_qd = hydro[:, 2:] + dt * hydro_qdd
            rigid_qd = rigid[:, 2:] + dt * rigid_qdd
            hydro = torch.cat((hydro[:, :2] + dt * hydro_qd, hydro_qd), dim=1)
            rigid = torch.cat((rigid[:, :2] + dt * rigid_qd, rigid_qd), dim=1)
            truth = _tensor(np.stack([
                np.r_[dataset["q_true"][rows[step + 1]],
                      dataset["qdot_true"][rows[step + 1]]]
                for _, rows, _ in windows
            ]), model)
            hydro_history.append(hydro.clone())
            rigid_history.append(rigid.clone())
            truth_history.append(truth)
    hydro_all = torch.stack(hydro_history, dim=1)
    rigid_all = torch.stack(rigid_history, dim=1)
    truth_all = torch.stack(truth_history, dim=1)
    hyd_position = _error_metrics(hydro_all[..., :2], truth_all[..., :2])
    rigid_position = _error_metrics(rigid_all[..., :2], truth_all[..., :2])
    hyd_velocity = _error_metrics(hydro_all[..., 2:], truth_all[..., 2:])
    rigid_velocity = _error_metrics(rigid_all[..., 2:], truth_all[..., 2:])
    hyd_endpoint = _error_metrics(hydro_all[:, -1, :2], truth_all[:, -1, :2])
    rigid_endpoint = _error_metrics(rigid_all[:, -1, :2], truth_all[:, -1, :2])
    return {
        "windows": len(windows),
        "horizon_steps": horizon_steps,
        "horizon_seconds": {
            "min": min(dt * horizon_steps for _, _, dt in windows),
            "max": max(dt * horizon_steps for _, _, dt in windows),
        },
        "hydrodynamic": {"position": hyd_position, "velocity": hyd_velocity,
                         "endpoint_position": hyd_endpoint},
        "rigid_only": {"position": rigid_position, "velocity": rigid_velocity,
                       "endpoint_position": rigid_endpoint},
        "improvement_percent": {
            "position_rmse": _improvement(hyd_position["rmse"], rigid_position["rmse"]),
            "endpoint_position_rmse": _improvement(hyd_endpoint["rmse"], rigid_endpoint["rmse"]),
        },
    }


def _stratum_key(flow, sensor=None, actuator=None):
    flow_text = ",".join(f"{float(value):.6g}" for value in flow)
    return f"flow=[{flow_text}];sensor={sensor};actuator={actuator}"


def _subset(dataset, rows):
    return {name: np.asarray(values)[rows] for name, values in dataset.items()}


def stratified_metrics(model, dataset, horizon_steps=100, stride_steps=100):
    """Report one-step and rollout metrics for every flow/delay condition."""
    n = validate_rollout_dataset(dataset)
    sensor = np.asarray(dataset.get("sensor_delay_steps", np.full(n, -1)))
    actuator = np.asarray(dataset.get("actuation_delay_steps", np.full(n, -1)))
    flow = np.asarray(dataset["flow_velocity"])
    conditions = np.c_[flow, sensor, actuator]
    result = {}
    for condition in np.unique(conditions, axis=0):
        rows = np.flatnonzero(np.all(np.isclose(
            conditions, condition, rtol=0, atol=1e-12), axis=1))
        selected = _subset(dataset, rows)
        key = _stratum_key(condition[:3], int(condition[3]), int(condition[4]))
        result[key] = {
            "one_step": one_step_metrics(model, selected),
            "rollout": rollout_metrics(model, selected, horizon_steps, stride_steps),
        }
    return result


def trajectory_metrics(model, dataset, horizon_steps=100, stride_steps=100):
    """Expose failures hidden by aggregate metrics, one held-out trajectory at a time."""
    groups = np.asarray(dataset["trajectory_id"])
    result = {}
    for group in np.unique(groups):
        selected = _subset(dataset, np.flatnonzero(groups == group))
        result[str(group.item() if isinstance(group, np.generic) else group)] = {
            "one_step": one_step_metrics(model, selected),
            "rollout": rollout_metrics(model, selected, horizon_steps, stride_steps),
        }
    return result

def evaluate_heldout(model, dataset, config=RolloutGateConfig()):
    """Produce metrics and explicit pass/fail gates for one untouched dataset."""
    one_step = one_step_metrics(model, dataset)
    rollout = rollout_metrics(model, dataset, config.horizon_steps, config.stride_steps)
    minimum_eigenvalue = float(torch.linalg.eigvalsh(
        model.added_mass_matrices().detach()).min().cpu())
    strata = stratified_metrics(model, dataset, config.horizon_steps, config.stride_steps)
    trajectories = trajectory_metrics(model, dataset, config.horizon_steps, config.stride_steps)
    gates = {
        "finite_metrics": bool(np.isfinite([
            one_step["hydrodynamic"]["fluid_torque"]["rmse"],
            rollout["hydrodynamic"]["position"]["rmse"],
        ]).all()),
        "added_mass_strictly_spd": minimum_eigenvalue >= config.added_mass_eigenvalue_floor,
        "fluid_torque_improvement": (
            one_step["improvement_percent"]["fluid_torque_rmse"]
            >= config.min_torque_improvement_percent),
        "rollout_position_improvement": (
            rollout["improvement_percent"]["position_rmse"]
            >= config.min_rollout_improvement_percent),
        "all_strata_fluid_torque_improvement": all(
            item["one_step"]["improvement_percent"]["fluid_torque_rmse"]
            >= config.min_torque_improvement_percent for item in strata.values()),
        "all_strata_rollout_position_improvement": all(
            item["rollout"]["improvement_percent"]["position_rmse"]
            >= config.min_rollout_improvement_percent for item in strata.values()),
    }
    if config.max_position_rmse is not None:
        gates["position_rmse_limit"] = (
            rollout["hydrodynamic"]["position"]["rmse"] <= config.max_position_rmse)
    if config.max_position_error_p95 is not None:
        gates["position_p95_limit"] = (
            rollout["hydrodynamic"]["position"]["error_norm_p95"]
            <= config.max_position_error_p95)
    return {
        "config": asdict(config),
        "physical": {"minimum_added_mass_eigenvalue": minimum_eigenvalue},
        "one_step": one_step,
        "rollout": rollout,
        "strata": strata,
        "trajectories": trajectories,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def save_evaluation(path, result):
    """Atomically save a machine-readable evaluation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
