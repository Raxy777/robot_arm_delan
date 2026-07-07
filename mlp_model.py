"""Plain MLP inverse-dynamics model:  tau = f(q, q̇, q̈).

This is the black-box baseline. It has NO physics structure — it just maps a
6-vector (q, q̇, q̈) to a 2-vector torque. Inputs and outputs are standardized
(zero mean, unit std) using statistics from the training set, which matters a
lot for training stability.

The wrapper class exposes `inverse_dynamics(q, qd, qdd) -> tau` as numpy, i.e.
exactly the interface InverseDynamicsController expects, so the trained model
drops straight into the control loop next to the analytic model.
"""

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim=6, out_dim=2, hidden=(128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.Tanh()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Standardizer:
    """Stores mean/std and moves data to/from normalized space."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float64)
        self.std = np.asarray(std, np.float64)
        self.std[self.std < 1e-8] = 1e-8

    @classmethod
    def fit(cls, x):
        return cls(x.mean(axis=0), x.std(axis=0))

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, z):
        return z * self.std + self.mean

    def as_dict(self, prefix):
        return {f"{prefix}_mean": self.mean, f"{prefix}_std": self.std}


class MLPInverseDynamics:
    """Trainable wrapper: holds the net + input/output standardizers and offers
    a numpy `inverse_dynamics` for the controller."""

    def __init__(self, net, x_scaler, y_scaler, device="cpu"):
        self.net = net.to(device)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = device

    # ---- inference in numpy, single sample (for the control loop) ----
    def inverse_dynamics(self, q, qd, qdd):
        x = np.concatenate([q, qd, qdd])[None, :]
        return self.predict(x)[0]

    # ---- batched numpy inference (for evaluation on datasets) ----
    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            xn = torch.as_tensor(self.x_scaler.norm(X), dtype=torch.float32,
                                 device=self.device)
            yn = self.net(xn).cpu().numpy()
        return self.y_scaler.denorm(yn)

    # ---- persistence ----
    def save(self, path):
        torch.save({
            "state_dict": self.net.state_dict(),
            "hidden": [m.out_features for m in self.net.net
                       if isinstance(m, nn.Linear)][:-1],
            "x_mean": self.x_scaler.mean, "x_std": self.x_scaler.std,
            "y_mean": self.y_scaler.mean, "y_std": self.y_scaler.std,
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        net = MLP(hidden=tuple(ck["hidden"]))
        net.load_state_dict(ck["state_dict"])
        return cls(net,
                   Standardizer(ck["x_mean"], ck["x_std"]),
                   Standardizer(ck["y_mean"], ck["y_std"]),
                   device=device)


def build_inputs(q, qd, qdd):
    """Assemble the (N, 6) feature matrix from component arrays."""
    return np.concatenate([q, qd, qdd], axis=1)
