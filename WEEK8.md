# Week 8 — Data + Black-Box Baseline

Goal: collect a rich `(q, q̇, q̈, τ)` dataset, train a plain MLP to predict
inverse dynamics `τ = f(q, q̇, q̈)`, put it inside the same computed-torque loop
from Week 7, and **measure where it fails** — that failure is the whole reason
DeLaN exists (Week 9).

## The pipeline

```bash
pip install -r requirements.txt          # now also needs torch

# 1. collect data (real deliverable uses MuJoCo)
python scripts/data_collection.py --backend mujoco
#    -> data/train.npz, data/test_id.npz, data/test_ood.npz

# 2. train the MLP
python scripts/train_mlp.py --epochs 300
#    -> models/mlp.pt ; prints in-dist vs OOD torque RMSE

# 3. closed-loop comparison (MLP vs analytic, nominal vs fast)
python scripts/evaluate_model.py --backend mujoco
```

Everything also runs with `--backend analytic` (no MuJoCo needed) — that's how
this whole pipeline was verified without a simulator or GPU. For the arm's
contact-free rigid dynamics the two backends agree to ~1e-9.

## Design decisions worth knowing (interview fuel)

**Reading q̈ correctly.** `q̈` is read straight from the simulator
(`data.qacc` in MuJoCo, `forward_dynamics` analytically), never
finite-differenced from velocities. Differencing amplifies noise and would
poison the regression targets. This is the concrete version of the plan's
Week-8 warning.

**One controller, swappable model.** `InverseDynamicsController` needs only a
`model.inverse_dynamics(q, q̇, q̈)` method. The analytic model, the MLP, and
next week's DeLaN all expose that same method, so the benchmark is literally
"swap the model, keep the controller." With the analytic model this controller
is provably identical to Week 7's `ComputedTorqueController` (verified: torque
diff ~1e-13).

**Torque saturation.** Actuators are capped at ±60 N·m (the arm only needs
~30 N·m for the fast figure-eight). The cap lives in both the XML `ctrlrange`
and `ArmSim`, so the analytic and MuJoCo backends clip identically instead of
diverging when a controller commands a large torque.

**Two excitation strategies.** Multisine open-loop torques (fast, varied
motion sweeping the velocity/acceleration space) plus gravity-comp PD to random
reachable targets (near-static and low-speed workspace coverage). Mixing them
gives coverage a single strategy misses.

**In-distribution vs out-of-distribution.** Training and `test_id` use moderate
torques and frequencies; `test_ood` is deliberately faster and stronger. The
closed-loop test does the same: a `nominal` figure-eight vs a `fast` one
(shorter period, bigger amplitude). The MLP should be close to analytic on
nominal and visibly worse on fast — that gap is the headline of Week 8.

## What you'll write up

The deliverable table from `evaluate_model.py` (RMS tracking error, mm):

| model | nominal | fast (OOD) |
|-------|---------|------------|
| analytic | tiny | tiny |
| MLP | close to analytic | **noticeably worse** |

Plus the torque-prediction RMSE table (in-dist vs OOD) from `train_mlp.py`.
Narrate the story: the black-box model interpolates well but cannot extrapolate,
because it has no notion of the physics — it just memorized a mapping over the
region it saw. That is exactly the weakness DeLaN's structure fixes.

## Verified in the sandbox (numpy-only parts)

| check | result |
|-------|--------|
| `InverseDynamicsController(analytic)` == `ComputedTorqueController` | torque diff **9e-13** ✓ |
| analytic closed-loop tracking (nominal / fast) | **0.007 / 0.075 mm** ✓ |
| logged tuple self-consistency `f(q,q̇,q̈) == τ` | **~1e-13** ✓ |
| torque saturation + data sanity | `|τ|≤60`, `|q̈|` bounded, no NaNs ✓ |

Not runnable in my sandbox (locked-down package proxy) — **run these locally**:
the MLP training (`train_mlp.py`, needs `torch`) and the MuJoCo-backend paths
(need `mujoco`). The code is standard PyTorch/MuJoCo and the numpy scaffolding
around it is verified above.

## New files this week

```
arm_sim.py           # unified mujoco/analytic simulator (correct q̈ read)
data_collection.py   # multisine + PD excitation -> train/test_id/test_ood
mlp_model.py         # MLP + standardizers + numpy inverse_dynamics wrapper
train_mlp.py         # training loop, saves models/mlp.pt, reports RMSE
evaluate_model.py    # closed-loop MLP-vs-analytic, nominal vs OOD
controllers.py       # + InverseDynamicsController (model-agnostic)
_verify_week8.py     # optional: no-deps sanity check of data + clipping
```
