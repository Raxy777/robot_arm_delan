"""Train the DeLaN inverse-dynamics model on the Week-8 data.

Same data, same loss (torque MSE), same eval splits as the MLP — the ONLY thing
that changes is the model's internal structure. That controlled comparison is
the point: any difference in generalization comes from the physics prior, not
from the data or the training setup.

    python train_delan.py --epochs 600

Notes
-----
* DeLaN trains slower per step than the MLP because each forward pass builds
  M(q), V(q) and differentiates them. It also usually needs more epochs and a
  slightly smaller LR. Defaults here are a reasonable starting point.
* De-risking tip (from the plan): if DeLaN is fiddly, get the pipeline working
  with a plain-MLP-style target first, then switch models. The math check in
  _verify_week9_math.py confirms the assembly formula independently of torch.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from delan_model import DeLaN, DeLaNInverseDynamics
from mlp_model import build_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODELS = os.path.join(HERE, "models")


def load_split(name):
    d = np.load(os.path.join(DATA, f"{name}.npz"))
    return d["q"], d["qd"], d["qdd"], d["tau"]


def torque_rmse_batched(net, q, qd, qdd, tau, device, chunk=1024):
    """Torque RMSE over a dataset; chunked because DeLaN's per-sample graph is
    heavier than the MLP's."""
    preds = []
    for i in range(0, len(q), chunk):
        sl = slice(i, i + chunk)
        qb = torch.as_tensor(q[sl], dtype=torch.float32, device=device)
        qdb = torch.as_tensor(qd[sl], dtype=torch.float32, device=device)
        qddb = torch.as_tensor(qdd[sl], dtype=torch.float32, device=device)
        preds.append(net.inverse_dynamics(qb, qdb, qddb).detach().cpu().numpy())
    pred = np.concatenate(preds)
    per_joint = np.sqrt(np.mean((pred - tau) ** 2, axis=0))
    return per_joint, float(np.sqrt(np.mean((pred - tau) ** 2)))


def train(epochs=600, hidden=(64, 64), lr=5e-4, batch=128, val_frac=0.1,
          seed=0, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)

    q, qd, qdd, tau = load_split("train")
    idx = np.random.permutation(len(q))
    q, qd, qdd, tau = q[idx], qd[idx], qdd[idx], tau[idx]
    n_val = int(val_frac * len(q))

    def dev(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    tr = slice(n_val, None); va = slice(0, n_val)
    q_tr, qd_tr, qdd_tr, tau_tr = dev(q[tr]), dev(qd[tr]), dev(qdd[tr]), dev(tau[tr])
    q_va, qd_va, qdd_va, tau_va = dev(q[va]), dev(qd[va]), dev(qdd[va]), dev(tau[va])

    net = DeLaN(hidden=hidden).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = q_tr.shape[0]
    best_val, best_state = np.inf, None
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            pred = net.inverse_dynamics(q_tr[b], qd_tr[b], qdd_tr[b])
            loss = loss_fn(pred, tau_tr[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pass  # eval needs grad (autograd assembly), so no no_grad here
        vpred = net.inverse_dynamics(q_va, qd_va, qdd_va)
        vl = loss_fn(vpred, tau_va).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if ep % 25 == 0 or ep == epochs - 1:
            tpred = net.inverse_dynamics(q_tr[:2048], qd_tr[:2048], qdd_tr[:2048])
            tl = loss_fn(tpred, tau_tr[:2048]).item()
            print(f"epoch {ep:4d} | train {tl:.5f} | val {vl:.5f}")

    net.load_state_dict(best_state)
    model = DeLaNInverseDynamics(net, device=device)
    os.makedirs(MODELS, exist_ok=True)
    model.save(os.path.join(MODELS, "delan.pt"))

    print("\n--- DeLaN torque prediction RMSE (N m) ---")
    for split in ["test_id", "test_ood"]:
        qs, qds, qdds, taus = load_split(split)
        pj, ov = torque_rmse_batched(net, qs, qds, qdds, taus, device)
        tag = "in-distribution" if split == "test_id" else "OUT-of-distribution"
        print(f"  {tag:20s}: joint1 {pj[0]:7.4f} | joint2 {pj[1]:7.4f} | overall {ov:7.4f}")

    compare_mass_matrix(model)
    print(f"\nSaved model to {os.path.join(MODELS, 'delan.pt')}")
    return model


def compare_mass_matrix(model, n=200, seed=1):
    """Interpretability: how close is the LEARNED M(q) to the analytic one?
    The MLP has no such internal quantity to inspect — this is a DeLaN-only win.

    Note: M is only identifiable up to the terms that actually affect torque;
    a constant offset in V (and hence absolute energy) is unobservable. We
    therefore report the error in M directly (which IS identifiable)."""
    import dynamics
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n):
        q = rng.uniform(-np.pi, np.pi, 2)
        M_hat = model.mass_matrix(q)
        M_true = dynamics.mass_matrix(q)
        errs.append(np.linalg.norm(M_hat - M_true) / np.linalg.norm(M_true))
    errs = np.array(errs)
    print("\n--- Learned vs analytic mass matrix M(q) ---")
    print(f"  relative Frobenius error: mean {errs.mean():.3f} | "
          f"median {np.median(errs):.3f} | max {errs.max():.3f}")
    print("  (lower is better; this is interpretability the MLP cannot offer)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    train(epochs=args.epochs, lr=args.lr, batch=args.batch, device=args.device)
