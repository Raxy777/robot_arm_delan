"""Render the three headline figures for Week 10 from the saved results.

Run benchmark.py and energy.py first (they write to results/); then:

    python plots.py        # writes results/*.png

Figures
-------
  tracking_rmse.png   grouped bars: closed-loop RMSE (mm), model x scenario
                      (log y — errors span 0.3 mm to 40 mm)
  energy_drift.png    E(t) - E(0) for analytic vs DeLaN; MLP absent (no energy)
  torque_scatter.png  predicted vs true torque on the OOD split, per model;
                      tight diagonal = good extrapolation
"""

import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

COLORS = {"analytic": "#444444", "mlp": "#d1495b", "delan": "#2e86ab"}


def _read_csv(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def plot_tracking():
    rows = _read_csv("bench_closed_loop.csv")
    scenarios = [c for c in rows[0].keys() if c != "model"]
    models = [r["model"] for r in rows]
    x = np.arange(len(scenarios))
    w = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, r in enumerate(rows):
        vals = [float(r[s]) for s in scenarios]
        ax.bar(x + i * w, vals, w, label=r["model"],
               color=COLORS.get(r["model"], None))
    ax.set_yscale("log")
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("EE tracking RMSE (mm, log)")
    ax.set_title("Closed-loop tracking: analytic vs MLP vs DeLaN")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RESULTS, "tracking_rmse.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_energy():
    rows = _read_csv("energy_drift.csv")
    cols = list(rows[0].keys())
    t = np.array([float(r["t"]) for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in ("analytic", "delan"):
        if name in cols:
            E = np.array([float(r[name]) for r in rows])
            ax.plot(t, E - E[0], label=f"{name} (drift)", color=COLORS.get(name))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("E(t) - E(0)  (J)")
    ax.set_title("Passive energy drift  (MLP: no energy function to plot)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(RESULTS, "energy_drift.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def plot_scatter():
    d = np.load(os.path.join(RESULTS, "torque_scatter.npz"))
    names = sorted({k.rsplit("_", 1)[0] for k in d.files})
    order = [n for n in ("analytic", "mlp", "delan") if n in names]
    fig, axes = plt.subplots(1, len(order), figsize=(4.2 * len(order), 4.2),
                             squeeze=False)
    for ax, name in zip(axes[0], order):
        true = d[f"{name}_true"].ravel()
        pred = d[f"{name}_pred"].ravel()
        lo, hi = float(min(true.min(), pred.min())), float(max(true.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.6)
        ax.scatter(true, pred, s=3, alpha=0.25, color=COLORS.get(name))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        ax.set_title(f"{name}  (OOD RMSE {rmse:.3f} N m)")
        ax.set_xlabel("true torque (N m)")
        ax.set_ylabel("predicted torque (N m)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Open-loop torque prediction on out-of-distribution data")
    fig.tight_layout()
    p = os.path.join(RESULTS, "torque_scatter.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def main():
    made = [plot_tracking(), plot_energy(), plot_scatter()]
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
