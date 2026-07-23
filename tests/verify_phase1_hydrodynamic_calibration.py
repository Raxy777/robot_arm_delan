"""Deterministic checks for constrained global hydrodynamic calibration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from src.hydrodynamic_calibration import (
    CalibrationConfig, evaluate_model, fit_global_hydrodynamics,
    load_calibrated_model, save_calibrated_model, validate_calibration_dataset,
)
from src.residual_dynamics import StructuredHydrodynamicDynamics, StructuredResidualConfig


def synthetic_dataset(n=900, seed=7):
    rng = np.random.default_rng(seed)
    q = rng.uniform([-1.2, -1.0], [1.2, 1.0], size=(n, 2))
    qdot = rng.uniform(-3.5, 3.5, size=(n, 2))
    torque = rng.uniform(-25.0, 25.0, size=(n, 2))
    flow = rng.uniform([-0.45, -0.35, 0.0], [0.45, 0.35, 0.0], size=(n, 3))
    truth = StructuredHydrodynamicDynamics(StructuredResidualConfig(
        linear_drag=(0.32, 0.17), quadratic_drag=(0.58, 0.36),
        added_mass_diag=(0.11, 0.07), added_mass_spd_floor=2e-5,
    ), dtype=torch.float64)
    state = np.concatenate((q, qdot), axis=1)
    with torch.no_grad():
        qdd = truth.acceleration(state, torque, flow).numpy()
        _, tau_drag, _ = truth.residual_terms(
            torch.as_tensor(state), torch.as_tensor(flow))
        M_added, _, _ = truth.residual_terms(
            torch.as_tensor(state), torch.as_tensor(flow))
        fluid = (tau_drag - torch.einsum(
            "...ij,...j->...i", M_added, torch.as_tensor(qdd))).numpy()
    return {
        "q_true": q, "qdot_true": qdot, "qdd_true": qdd,
        "tau_applied": torque, "flow_velocity": flow,
        "fluid_torque": fluid,
    }, truth


def subset(data, rows):
    return {name: values[rows] for name, values in data.items()}


def main():
    data, truth = synthetic_dataset()
    train = subset(data, slice(0, 650))
    validation = subset(data, slice(650, 775))
    test = subset(data, slice(775, None))
    assert validate_calibration_dataset(train) == 650

    broken = dict(train)
    broken["flow_velocity"] = broken["flow_velocity"][:, :2]
    try:
        validate_calibration_dataset(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid flow shape was accepted")

    model, fit = fit_global_hydrodynamics(
        train, validation,
        CalibrationConfig(steps=900, batch_size=650, learning_rate=0.025,
                          validation_interval=20, seed=3),
        device="cpu", dtype=torch.float64,
    )
    metrics = evaluate_model(model, test)
    assert fit["best_step"] > 0
    assert metrics["acceleration_rmse"] < 0.035, metrics
    assert metrics["rmse_improvement_percent"] > 85.0, metrics

    learned = model.coefficients()
    target = truth.coefficients()
    for name in ("linear_drag", "quadratic_drag"):
        assert torch.allclose(learned[name], target[name], rtol=0.18, atol=0.035), (
            name, learned[name], target[name])
    eigenvalues = torch.linalg.eigvalsh(learned["added_mass"])
    assert torch.all(eigenvalues >= model.config.added_mass_spd_floor * 0.999999)
    # Even a nominally zero Cholesky block is strictly PD because epsilon I is explicit.
    floor_model = StructuredHydrodynamicDynamics(StructuredResidualConfig(
        linear_drag=(0.0, 0.0), quadratic_drag=(0.0, 0.0),
        added_mass_diag=(0.0, 0.0), added_mass_spd_floor=1e-7,
    ), dtype=torch.float64)
    floor_eigenvalues = torch.linalg.eigvalsh(floor_model.added_mass_matrices())
    assert torch.all(floor_eigenvalues >= 1e-7)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.json"
        save_calibrated_model(path, model, fit, {"test": metrics}, {"fixture": True})
        loaded, artifact = load_calibrated_model(path, dtype=torch.float64)
        assert artifact["format_version"] == 1
        state = torch.as_tensor(np.concatenate((test["q_true"], test["qdot_true"]), 1))
        with torch.no_grad():
            before = model.acceleration(state, test["tau_applied"], test["flow_velocity"])
            after = loaded.acceleration(state, test["tau_applied"], test["flow_velocity"])
        assert torch.equal(before, after)
        # The artifact is plain JSON, finite, and records held-out evidence.
        parsed = json.loads(path.read_text())
        assert parsed["metrics"]["test"]["samples"] == len(test["q_true"])

    print("PASS: constrained global fit recovers synthetic fluid dynamics and round-trips")


if __name__ == "__main__":
    main()
