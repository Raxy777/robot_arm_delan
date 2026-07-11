"""Train the MLP inverse-dynamics model on the collected data.

    python train_mlp.py --epochs 300

Reports normalized train/val loss during training, then denormalized torque
RMSE on the in-distribution and OOD test sets so you can see the extrapolation
gap immediately. Saves the model to models/mlp.pt.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from src.mlp_model import MLP, MLPInverseDynamics, Standardizer, build_inputs

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")


def load_split(name):
    d = np.load(os.path.join(DATA, f"{name}.npz"))
    X = build_inputs(d["q"], d["qd"], d["qdd"])
    Y = d["tau"]
    return X, Y


def torque_rmse(model, X, Y):
    pred = model.predict(X)
    return np.sqrt(np.mean((pred - Y) ** 2, axis=0)), \
        np.sqrt(np.mean((pred - Y) ** 2))


def train(epochs=300, hidden=(128, 128), lr=1e-3, batch=256, val_frac=0.1,
          seed=0, device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    X, Y = load_split("train")
    # shuffle + split off a validation set
    idx = np.random.permutation(len(X))
    X, Y = X[idx], Y[idx]
    n_val = int(val_frac * len(X))
    Xtr, Ytr = X[n_val:], Y[n_val:]
    Xva, Yva = X[:n_val], Y[:n_val]

    x_scaler = Standardizer.fit(Xtr)
    y_scaler = Standardizer.fit(Ytr)

    def to_t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    Xtr_n = to_t(x_scaler.norm(Xtr)); Ytr_n = to_t(y_scaler.norm(Ytr))
    Xva_n = to_t(x_scaler.norm(Xva)); Yva_n = to_t(y_scaler.norm(Yva))

    net = MLP(hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = len(Xtr_n)
    best_val, best_state = np.inf, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(Xtr_n[b]), Ytr_n[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(Xva_n), Yva_n).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        if ep % 25 == 0 or ep == epochs - 1:
            with torch.no_grad():
                tl = loss_fn(net(Xtr_n), Ytr_n).item()
            print(f"epoch {ep:4d} | train {tl:.5f} | val {vl:.5f}")

    net.load_state_dict(best_state)  # restore best-val weights
    model = MLPInverseDynamics(net, x_scaler, y_scaler, device=device)

    os.makedirs(MODELS, exist_ok=True)
    model.save(os.path.join(MODELS, "mlp.pt"))

    print("\n--- Torque prediction RMSE (N m) ---")
    for split in ["test_id", "test_ood"]:
        Xs, Ys = load_split(split)
        per_joint, overall = torque_rmse(model, Xs, Ys)
        tag = "in-distribution" if split == "test_id" else "OUT-of-distribution"
        print(f"  {tag:20s}: joint1 {per_joint[0]:7.4f} | "
              f"joint2 {per_joint[1]:7.4f} | overall {overall:7.4f}")
    print("\nExpect the OOD error to be much larger than in-distribution — that")
    print("gap is the black-box model's Achilles heel, and the motivation for")
    print("the physics-structured model in Week 9.")
    print(f"\nSaved model to {os.path.join(MODELS, 'mlp.pt')}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch=args.batch, device=args.device)
