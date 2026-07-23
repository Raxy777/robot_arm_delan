"""Deterministic verification of held-out one-step and rollout gates."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.heldout_rollouts import (
    RolloutGateConfig, evaluate_heldout, rollout_metrics, save_evaluation,
    validate_rollout_dataset,
)
from src.residual_dynamics import StructuredHydrodynamicDynamics, StructuredResidualConfig


def synthetic_holdout(model, trajectories=4, steps=81, dt=0.002):
    fields = {name: [] for name in (
        "q_true", "qdot_true", "qdd_true", "tau_applied", "flow_velocity",
        "fluid_torque", "trajectory_id", "time_true", "sensor_delay_steps",
        "actuation_delay_steps",
    )}
    for trajectory in range(trajectories):
        state = torch.tensor([0.15 + .03 * trajectory, -0.25, 0.35, -0.2],
                             dtype=model.dtype)
        flow = torch.tensor([0.15 + .05 * trajectory, -0.04, 0.0], dtype=model.dtype)
        for step in range(steps):
            torque = torch.tensor([
                7.0 + 1.3 * np.sin(.07 * step + trajectory),
                1.8 * np.cos(.11 * step - trajectory),
            ], dtype=model.dtype)
            with torch.no_grad():
                qdd = model.acceleration(state, torque, flow)
                added, drag, _ = model.residual_terms(state, flow)
                fluid = drag - added @ qdd
            values = {
                "q_true": state[:2].numpy().copy(),
                "qdot_true": state[2:].numpy().copy(),
                "qdd_true": qdd.numpy().copy(),
                "tau_applied": torque.numpy().copy(),
                "flow_velocity": flow.numpy().copy(),
                "fluid_torque": fluid.numpy().copy(),
                "trajectory_id": trajectory,
                "time_true": step * dt,
                "sensor_delay_steps": (trajectory % 2) * 5,
                "actuation_delay_steps": (trajectory % 3) * 2,
            }
            for name, value in values.items():
                fields[name].append(value)
            with torch.no_grad():
                state = model(state, torque, dt, flow)
    return {name: np.asarray(values) for name, values in fields.items()}


def main():
    torch.manual_seed(4)
    model = StructuredHydrodynamicDynamics(StructuredResidualConfig(
        linear_drag=(0.7, 0.5), quadratic_drag=(1.0, 0.8),
        added_mass_diag=(0.12, 0.08), added_mass_spd_floor=1e-9,
    ), dtype=torch.float64)
    dataset = synthetic_holdout(model)
    assert validate_rollout_dataset(dataset) == 324

    config = RolloutGateConfig(horizon_steps=20, stride_steps=20,
                               min_torque_improvement_percent=99.0,
                               min_rollout_improvement_percent=99.0,
                               added_mass_eigenvalue_floor=1e-9)
    result = evaluate_heldout(model, dataset, config)
    assert result["passed"], result["gates"]
    assert result["one_step"]["hydrodynamic"]["fluid_torque"]["rmse"] < 1e-10
    assert result["rollout"]["hydrodynamic"]["position"]["rmse"] < 1e-10
    assert result["one_step"]["rigid_only"]["fluid_torque"]["rmse"] > 1e-3
    assert result["rollout"]["rigid_only"]["position"]["rmse"] > 1e-7
    assert len(result["strata"]) == 4

    impossible = evaluate_heldout(model, dataset, RolloutGateConfig(
        horizon_steps=20, stride_steps=20,
        min_torque_improvement_percent=101.0,
        min_rollout_improvement_percent=101.0,
        added_mass_eigenvalue_floor=1e-9,
    ))
    assert not impossible["passed"]
    assert not impossible["gates"]["fluid_torque_improvement"]

    malformed = dict(dataset)
    malformed["time_true"] = malformed["time_true"].copy()
    malformed["time_true"][1] = malformed["time_true"][0]
    try:
        validate_rollout_dataset(malformed)
        raise AssertionError("non-increasing trajectory time was accepted")
    except ValueError as error:
        assert "time ordered" in str(error)
    try:
        rollout_metrics(model, dataset, horizon_steps=1000)
        raise AssertionError("an unavailable horizon was accepted")
    except ValueError as error:
        assert "no complete rollout windows" in str(error)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gate.json"
        save_evaluation(path, result)
        loaded = json.loads(path.read_text())
        assert loaded["passed"] is True and loaded["rollout"]["windows"] == 16

    print("PASS: held-out one-step, finite-horizon, physical, and failure gates")


if __name__ == "__main__":
    main()
