# Week 9 — Deep Lagrangian Networks (the crux week)

Goal: replace the black-box MLP's *output* with physics *structure*. Instead of
regressing torque directly, DeLaN learns the ingredients of the Lagrangian —
the mass matrix `M(q)` and potential `V(q)` — and rebuilds the equations of
motion from them with autograd. Same data, same loss, same controller; only the
model's internal structure changes. That controlled swap is the experiment.

## Run order (on Colab — you have no local GPU)

```
colab/week9_train.ipynb  →  Runtime → Run all   (T4 GPU recommended)
```

The notebook: clones the repo → installs deps → **validates the EL math** →
(optionally trains the MLP) → trains DeLaN → prints the 3-way table → downloads
`delan.pt`. Then locally:

```
# drop delan.pt (and mlp.pt) into models/, then:
python evaluate_all.py --backend analytic     # or mujoco
```

## The method in one screen (interview-ready)

Learn two small nets of `q`:

```
L(q)  -> lower-triangular  =>  M(q) = L Lᵀ + εI     (always symmetric PD)
V(q)  -> scalar potential energy
```

Then torque is *derived*, not regressed:

```
τ = M(q) q̈  +  [ Ṁ(q,q̇) q̇ − ½ ∂/∂q (q̇ᵀ M q̇) ]  +  ∂V/∂q
        │                    │                          │
     inertia          Coriolis/centrifugal           gravity
```

with `Ṁ q̇ = Σₖ (∂M/∂qₖ) q̇ · q̇ₖ`. Every derivative is taken by autograd, so
the Coriolis and gravity terms are *forced* to be consistent with the learned
`M` and `V`. There are far fewer ways for this to be wrong than for a free-form
MLP — which is exactly why it extrapolates better.

**Why SPD matters.** Parameterizing `M = LLᵀ + εI` (Cholesky) means the learned
inertia is *always* a valid mass matrix — positive-definite by construction.
A plain MLP predicting `M` entries could produce a non-physical, even singular,
"inertia." This is the single most important structural prior in DeLaN.

**Why Softplus, not ReLU.** DeLaN differentiates the network twice (once for the
`∂M/∂q` terms, and gradients flow through them in training). ReLU has zero
curvature and kinks; Softplus is smooth, so the second derivatives are
well-behaved. This is a real gotcha — a ReLU DeLaN trains badly.

## De-risking (heed the plan's warning: this week is fiddly)

1. **Run `_verify_week9_math.py` first.** It's torch-free: it pushes the
   *analytic* `M` and `V` through DeLaN's assembly formula (finite-diff
   derivatives) and checks it reproduces `dynamics.inverse_dynamics`. If this
   passes, the equations are correct and any remaining problem is optimization,
   not math. This isolates the two failure modes cleanly.
2. **Match the MLP's torque accuracy in-distribution before expecting the OOD
   win.** If DeLaN can't even fit the training regime, fix that first (more
   epochs, lr 3e-4).
3. **LNN fallback.** If DeLaN stays fiddly, Cranmer's LNN is the simpler sibling
   (learn a single scalar Lagrangian `L(q,q̇)` and get `q̈` from one
   matrix solve). The plan lists it as the fallback; the assembly here is close
   enough that switching is a small edit. Ship *something* structured this week.

## What to look for in the results

- **Open-loop torque RMSE**: MLP ≈ DeLaN in-distribution; DeLaN noticeably lower
  OUT-of-distribution. That gap is the headline.
- **Closed-loop tracking**: on the fast (OOD) figure-eight, DeLaN should track
  closer to the analytic upper bound than the MLP.
- **Interpretability (DeLaN only)**: `train_delan.py` prints the relative error
  between the *learned* `M(q)` and the analytic one. The MLP has no internal
  `M` to inspect — being able to recover interpretable physics is a talking
  point the black box simply can't match.

## Honest caveats to state in the write-up

- `M` is identifiable from torque data, but the *absolute* value of `V` is not
  (only `∂V/∂q` affects torque) — so we compare `M` directly, not energy offset.
- On a 2-DOF arm with clean data, the MLP is a *strong* baseline; DeLaN's
  advantage shows up specifically under extrapolation and with less/noisier
  data. If you want to make the gap dramatic for the demo, shrink the training
  set or add the payload-mass shift (that's the Week-10 robustness test).

## Verification status

| check | status |
|-------|--------|
| DeLaN EL-assembly formula vs analytic (`_verify_week9_math.py`) | **run it — expect ~1e-6** (my sandbox was down this session) |
| Model wrappers match the `inverse_dynamics(q,q̇,q̈)` interface | by construction; drop into `InverseDynamicsController` |
| Training / OOD tables | produced on Colab by `train_delan.py` + `evaluate_all.py` |

> Transparency: the compute sandbox I normally use to smoke-test was
> unavailable this session (host out of disk), so unlike Weeks 7-8 I could not
> execute the DeLaN code here. The math-validation script exists precisely so
> you get an unambiguous pass/fail the moment you run the notebook. If the EL
> check passes and training loss falls, you're in good shape.

## New files this week

```
delan_model.py          # DeLaN: Cholesky M(q), potential V(q), EL torque assembly
train_delan.py          # train on Week-8 data; RMSE + learned-vs-analytic M
evaluate_all.py         # analytic vs MLP vs DeLaN, open- and closed-loop
_verify_week9_math.py   # torch-free check of the EL assembly formula
colab/week9_train.ipynb # Colab: validate math -> train DeLaN -> compare
```
