"""Calibration and serialization for the global structured fluid model."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.residual_dynamics import StructuredHydrodynamicDynamics, StructuredResidualConfig

REQUIRED_FIELDS = (
    "q_true", "qdot_true", "qdd_true", "tau_applied", "flow_velocity",
)


@dataclass(frozen=True)
class CalibrationConfig:
    """Deterministic optimizer settings for the small global parameter block."""
    steps: int = 1500
    batch_size: int = 4096
    learning_rate: float = 0.02
    validation_interval: int = 25
    seed: int = 0
    gradient_clip: float = 10.0

    def __post_init__(self):
        if self.steps < 1 or self.batch_size < 1 or self.validation_interval < 1:
            raise ValueError("steps, batch_size, and validation_interval must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("learning_rate and gradient_clip must be positive")


def validate_calibration_dataset(dataset: Mapping[str, np.ndarray]) -> int:
    """Validate fields used by the physical fit and return the row count."""
    missing = [name for name in REQUIRED_FIELDS if name not in dataset]
    if missing:
        raise ValueError(f"calibration dataset is missing fields: {missing}")
    expected_tail = {
        "q_true": (2,), "qdot_true": (2,), "qdd_true": (2,),
        "tau_applied": (2,), "flow_velocity": (3,),
    }
    n = None
    for name, tail in expected_tail.items():
        values = np.asarray(dataset[name])
        if values.ndim != 2 or values.shape[1:] != tail:
            raise ValueError(f"{name} must have shape (N, {tail[0]})")
        if n is None:
            n = len(values)
        if len(values) != n:
            raise ValueError("calibration fields are not sample-aligned")
        if not np.issubdtype(values.dtype, np.number) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain only finite numeric values")
    if not n:
        raise ValueError("calibration dataset is empty")
    return int(n)


def _tensors(dataset, device, dtype):
    validate_calibration_dataset(dataset)
    def tensor(name):
        return torch.as_tensor(np.asarray(dataset[name]), device=device, dtype=dtype)
    state = torch.cat((tensor("q_true"), tensor("qdot_true")), dim=1)
    return state, tensor("tau_applied"), tensor("flow_velocity"), tensor("qdd_true")


def _copy_trainable_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _normalized_acceleration_loss(model, tensors, scale):
    state, torque, flow, qdd = tensors
    prediction = model.acceleration(state, torque, flow)
    return torch.mean(torch.square((prediction - qdd) / scale))


def fit_global_hydrodynamics(train, validation, config=CalibrationConfig(),
                             model_config=StructuredResidualConfig(), device="cpu",
                             dtype=torch.float64):
    """Fit one global constrained parameter set, selecting by validation loss.

    Only true state, actually applied torque, known flow, and directly reported
    acceleration are used. Delayed observations and commanded torque are never
    substituted for these causally aligned identification fields.
    """
    torch.manual_seed(config.seed)
    train_tensors = _tensors(train, device, dtype)
    validation_tensors = _tensors(validation, device, dtype)
    model = StructuredHydrodynamicDynamics(model_config, device=device, dtype=dtype)
    qdd_scale = torch.std(train_tensors[3], dim=0).clamp_min(1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    n_train = len(train_tensors[0])
    best_loss = math.inf
    best_step = 0
    best_state = _copy_trainable_state(model)
    history = []

    for step in range(1, config.steps + 1):
        if config.batch_size >= n_train:
            rows = torch.arange(n_train, device=device)
        else:
            rows = torch.randperm(n_train, generator=generator)[:config.batch_size].to(device)
        batch = tuple(value[rows] for value in train_tensors)
        optimizer.zero_grad(set_to_none=True)
        loss = _normalized_acceleration_loss(model, batch, qdd_scale)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()

        if step == 1 or step % config.validation_interval == 0 or step == config.steps:
            with torch.no_grad():
                validation_loss = float(_normalized_acceleration_loss(
                    model, validation_tensors, qdd_scale).cpu())
            history.append({"step": step, "train_loss": float(loss.detach().cpu()),
                            "validation_loss": validation_loss})
            if validation_loss < best_loss:
                best_loss, best_step = validation_loss, step
                best_state = _copy_trainable_state(model)

    model.load_state_dict(best_state)
    return model, {
        "best_step": best_step,
        "best_validation_loss": best_loss,
        "qdd_scale": qdd_scale.detach().cpu().tolist(),
        "history": history,
        "optimizer": asdict(config),
    }


def evaluate_model(model, dataset):
    """Return acceleration and generalized-fluid-force metrics for one split."""
    state, torque, flow, qdd = _tensors(dataset, model.device, model.dtype)
    with torch.no_grad():
        predicted = model.acceleration(state, torque, flow)
        rigid_M, bias = model.rigid_body_terms(state)
        rigid = torch.linalg.solve(rigid_M, (torque - bias).unsqueeze(-1)).squeeze(-1)
        M_added, tau_drag, _ = model.residual_terms(state, flow)
        effective_fluid = tau_drag - torch.einsum("...ij,...j->...i", M_added, qdd)
        error = predicted - qdd
        rigid_error = rigid - qdd
        row_norm = torch.linalg.vector_norm(error, dim=1)
        rmse = torch.sqrt(torch.mean(torch.square(error)))
        rigid_rmse = torch.sqrt(torch.mean(torch.square(rigid_error)))
        result = {
            "samples": len(state),
            "acceleration_rmse": float(rmse.cpu()),
            "acceleration_mae": float(torch.mean(torch.abs(error)).cpu()),
            "acceleration_error_norm_p95": float(torch.quantile(row_norm, 0.95).cpu()),
            "rigid_baseline_acceleration_rmse": float(rigid_rmse.cpu()),
            "rmse_improvement_percent": float((1.0 - rmse / rigid_rmse.clamp_min(1e-12)).cpu() * 100),
        }
        if "fluid_torque" in dataset:
            observed = torch.as_tensor(np.asarray(dataset["fluid_torque"]),
                                       device=model.device, dtype=model.dtype)
            if observed.shape == effective_fluid.shape and torch.isfinite(observed).all():
                result["effective_fluid_torque_rmse"] = float(torch.sqrt(
                    torch.mean(torch.square(effective_fluid - observed))).cpu())
    return result


def coefficient_summary(model):
    coefficients = model.coefficients()
    return {name: value.detach().cpu().tolist() for name, value in coefficients.items()}


def save_calibrated_model(path, model, fit, metrics, provenance=None):
    """Save a portable, inspectable JSON model artifact atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "format_version": 1,
        "model_class": "StructuredHydrodynamicDynamics",
        "model_config": asdict(model.config),
        "dtype": str(model.dtype).replace("torch.", ""),
        "state_dict": {name: value.detach().cpu().tolist()
                       for name, value in model.state_dict().items()
                       if name.startswith("raw_")},
        "coefficients": coefficient_summary(model),
        "fit": fit,
        "metrics": metrics,
        "provenance": provenance or {},
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return artifact


def load_calibrated_model(path, device="cpu", dtype=torch.float32):
    """Load a JSON artifact without pickle/code execution."""
    artifact = json.loads(Path(path).read_text())
    if artifact.get("format_version") != 1 or artifact.get("model_class") != "StructuredHydrodynamicDynamics":
        raise ValueError("unsupported calibrated-model artifact")
    config_values = dict(artifact["model_config"])
    for name in ("linear_drag", "quadratic_drag", "added_mass_diag"):
        config_values[name] = tuple(config_values[name])
    model = StructuredHydrodynamicDynamics(
        StructuredResidualConfig(**config_values), device=device, dtype=dtype)
    state = model.state_dict()
    for name, values in artifact["state_dict"].items():
        if name not in state or not name.startswith("raw_"):
            raise ValueError(f"unexpected model state field {name!r}")
        tensor = torch.as_tensor(values, dtype=dtype, device=device)
        if tensor.shape != state[name].shape or not torch.isfinite(tensor).all():
            raise ValueError(f"invalid model state field {name!r}")
        state[name] = tensor
    model.load_state_dict(state)
    return model, artifact
