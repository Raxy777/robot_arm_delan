# Learned Robot Arm Dynamics + Control — Week 7

Foundation for the project: a 2-link planar arm ("double pendulum with
actuated joints") simulated in MuJoCo, with the dynamics **derived by hand**
and three from-scratch controllers. This is the classical-control baseline that
later weeks compare the learned models (MLP, DeLaN) against.

> On the pitch wording: an actuated, damped manipulator does **not** conserve
> mechanical energy — the motors add and remove it. What the Lagrangian
> structure buys us is a guaranteed positive-definite mass matrix and
> physically consistent Coriolis/gravity terms, which is why the structured
> model is data-efficient and generalizes. Say it that way in interviews.

---

## 1. The system

Two rigid links in the vertical `x–y` plane, revolute joints, gravity along
`−y`. Generalized coordinates `q = [q1, q2]`: `q1` from the `+x` axis to link 1,
`q2` the relative elbow angle of link 2 w.r.t. link 1.

Each link is a uniform thin rod. Default parameters (`src/params.py`):

| symbol | meaning | value |
|--------|---------|-------|
| `m1, m2` | link masses | 1 kg |
| `l1, l2` | link lengths | 1 m |
| `lc1, lc2` | joint → COM distance | 0.5 m |
| `I1, I2` | link inertia about COM (z) | `mL²/12 = 0.0833` kg·m² |
| `g` | gravity | 9.81 m/s² |

---

## 2. Hand-derived Lagrangian (put this in your interview notes)

**Positions of the link centers of mass**

```
x_c1 = lc1 cos q1                      y_c1 = lc1 sin q1
x_c2 = l1 cos q1 + lc2 cos(q1+q2)      y_c2 = l1 sin q1 + lc2 sin(q1+q2)
```

**Velocities** (differentiate, then square and add):

```
v_c1² = lc1² q̇1²
v_c2² = l1² q̇1² + lc2² (q̇1+q̇2)² + 2 l1 lc2 q̇1 (q̇1+q̇2) cos q2
```

**Kinetic energy** (translation of COM + rotation about COM):

```
T = ½ m1 v_c1² + ½ I1 q̇1²
  + ½ m2 v_c2² + ½ I2 (q̇1+q̇2)²
```

**Potential energy** (height = `y`, gravity along −y):

```
U = m1 g lc1 sin q1 + m2 g [ l1 sin q1 + lc2 sin(q1+q2) ]
```

**Lagrangian** `L = T − U`. Apply the Euler–Lagrange equation for each joint,

```
d/dt (∂L/∂q̇_i) − ∂L/∂q_i = τ_i
```

and collect terms into the standard **manipulator form**:

```
M(q) q̈ + C(q, q̇) q̇ + g(q) = τ
```

**Mass matrix** (let `c2 = cos q2`):

```
M11 = m1 lc1² + m2 (l1² + lc2² + 2 l1 lc2 c2) + I1 + I2
M12 = M21 = m2 (lc2² + l1 lc2 c2) + I2
M22 = m2 lc2² + I2
```

**Coriolis / centrifugal** (let `h = m2 l1 lc2 sin q2`):

```
C = [ −h q̇2   −h (q̇1 + q̇2) ]
    [  h q̇1        0         ]
```

**Gravity**:

```
g1 = (m1 lc1 + m2 l1) g cos q1 + m2 lc2 g cos(q1+q2)
g2 = m2 lc2 g cos(q1+q2)
```

This `C` is the specific choice that makes `Ṁ − 2C` skew-symmetric — the
energy-consistency property the verification below relies on. All of this is
implemented verbatim in `src/dynamics.py`.

---

## 3. Controllers (`src/controllers.py`)

All hand-written, no control library:

1. **PD** — `τ = Kp e + Kd ė`. Knows nothing about the dynamics; droops under
   gravity and lags on fast motion.
2. **Gravity-compensated PD** — `τ = Kp e + Kd ė + g(q)`. Kills the steady
   gravity droop.
3. **Computed torque** (feedback linearization) —
   `τ = M(q)[q̈_d + Kd ė + Kp e] + C(q,q̇)q̇ + g(q)`.
   With a perfect model the error obeys `ë + Kd ė + Kp e = 0`, so pick
   `Kp = ωₙ²`, `Kd = 2ζωₙ`.

**Why this matters for the project:** in Weeks 8–10 you keep the computed-torque
*structure* and swap the *model* (`M, C, g`) from analytic → MLP → DeLaN. The
controller code doesn't change; only the model does. That is the experiment.

---

## 4. Task-space figure-eight (`src/trajectories.py`, `src/kinematics.py`)

The end-effector traces a Lissajous "8":
`x(t) = cx + A sin(ωt)`, `y(t) = cy + B sin(2ωt)`.

The reference is converted to joint space **analytically** — inverse
kinematics for position, then the Jacobian for the rest:

```
q̇_d = J⁻¹ ẋ_d          q̈_d = J⁻¹ (ẍ_d − J̇ q̇_d)
```

We deliberately do **not** finite-difference the joint angles to get velocity
and acceleration. (The plan's warning about noisy accelerations is about
*measured* data in Week 8 — there you should log MuJoCo's `qacc` directly, not
differentiate. A *designed* reference like this one is computed in closed form.)

---

## 5. How to run

```bash
pip install -r requirements.txt

# verify the hand-derived model first
python tests/verify_dynamics.py

# track the figure-eight and render a video
python scripts/run_sim.py --controller computed_torque --video
python scripts/run_sim.py --controller pd            # see it lag/droop for contrast
python scripts/run_sim.py --controller gravity_pd
```

Outputs land in `outputs_run/`: `figure_eight.mp4`, `tracking.png`, `log.npz`.

---

## 6. Verification status

Run in a numpy-only sandbox (MuJoCo checks are in `verify_dynamics.py` for you
to run locally):

| check | result |
|-------|--------|
| Energy conservation, passive arm (τ=0), 6 s RK4 | relative drift **6.3e-11** ✓ |
| `FK(IK(x))` round-trip over the figure-eight | max err **3.5e-16** ✓ |
| Analytic `J̇` vs finite differences | max err **5.9e-10** ✓ |
| Velocity mapping `J q̇ == ẋ` | max err **2.3e-16** ✓ |
| Figure-eight max reach vs. arm length | **1.69 m** < 2.0 m ✓ |
| Analytic inverse dynamics vs MuJoCo `mj_inverse` | **run `verify_dynamics.py` locally** |

The energy-conservation check is the strong one: it confirms `M`, `C`, and `g`
are mutually consistent (correct Coriolis structure and matching potential).
The one remaining check needs MuJoCo installed — running it locally confirms
your XML matches the analytic parameters (expect max torque error ~1e-9).

---

## 7. Files

```
robot_arm_delan/
├── src/
│   ├── params.py           # single source of truth for physical constants
│   ├── dynamics.py         # M(q), C(q,q̇), g(q), forward/inverse dynamics, energy
│   ├── kinematics.py       # FK, IK, Jacobian, J̇, task→joint mapping
│   ├── trajectories.py     # figure-eight reference
│   └── controllers.py      # PD, gravity-comp PD, computed torque
├── scripts/
│   └── run_sim.py          # MuJoCo sim loop → video + tracking plots + log
├── tests/
│   └── verify_dynamics.py  # energy check + MuJoCo cross-check
├── model/arm2.xml      # MuJoCo model, built to match params.py
└── requirements.txt
```

## 8. Week 7 checklist (from the plan)

- [x] Build the 2-link arm in MuJoCo XML
- [x] Derive the analytic dynamics by hand (this README)
- [x] Implement PD and computed-torque control with the known model
- [ ] Record the deliverable video of the arm tracking a figure-eight
      → run `python scripts/run_sim.py --controller computed_torque --video`


## Phase 1.3: MPPI rate hierarchy

`src/mppi.py` adds a model-agnostic, PyTorch-batched MPPI optimizer and a
deterministic multi-rate bridge. Production defaults are locked to 1024
samples, a 25-step horizon, temperature `lambda=1.0`, 50 Hz replanning, and a
1000 Hz linearly interpolated torque loop. The included nominal batched model
is a fallback; pass the structured residual forward model through the same
`dynamics(state, torque, dt)` interface when available.

```bash
python tests/verify_phase1_3_mppi.py
python scripts/phase1_3_mppi_benchmark.py --iterations 50 --out /tmp/phase1_3.json
```

The benchmark records mean, p95, and worst solve latency. A p95 below the
20 ms deadline is the Phase 1.3 compute gate; retain the worst time in reports
even when the gate passes. CPU results are development measurements only—rerun
on the deployment GPU before making a real-time claim.

## Phase 1.4: structured hydrodynamic residual

`src/residual_dynamics.py` provides a flow-conditioned forward model that is a
drop-in replacement for MPPI's nominal model. It computes link relative
velocity as `J(q) qdot - v_fluid`, maps non-negative linear/quadratic drag
through `J.T`, and parameterizes each task-space added-mass matrix as `L L.T`.
The resulting joint-space added mass is added to the rigid-body mass matrix
before solving for acceleration. Call `set_flow(...)` with the known uniform
flow before each plan; do not hide changing flow regimes from the model.

```bash
python tests/verify_phase1_4_residual.py
```

This acceptance check covers physical matrix constraints, dissipative drag,
recovery of the nominal baseline, explicit flow conditioning, finite training
gradients, and the Phase 1.3 MPPI interface. It does **not** establish learned
accuracy. That requires training only on designated trajectories/flows and
reporting one-step and rollout errors on held-out trajectories, payloads, flow
regimes, noise/delay conditions, and random seeds. MuJoCo fluid forces remain a
simplified simulation target—not CFD ground truth.

A separate smoke benchmark measures the structured model inside the production
MPPI budget and reports mean, p95, and worst solve latency:

```bash
python scripts/phase1_4_residual_benchmark.py --iterations 50 --out /tmp/phase1_4_timing.json
```

Rerun it on the target GPU before making a real-time claim.

### Phase 1.4 hard gates (pre-registered)

The structured model remains **uncalibrated** until it passes all gates below.
The thresholds are fixed before training:

1. On held-out 25-step rollouts spanning at least 3 unseen flow regimes and 5
random seeds, mean residual-model RMSE must be at most 75% of nominal-model
RMSE, and the residual must beat nominal in at least 80% of cases.
2. In closed loop, residual MPPI must improve mean tracking RMSE by at least 10%
over nominal MPPI across at least 3 flows and 5 seeds.
3. Stress results must include flow-estimation errors of 0%, ±20%, and ±50%,
sensor delays of 0/10/20 ms, and state-noise standard deviations of
0/0.01/0.03. Stressors are varied one at a time for attributable failures.
4. The selected controller profile must meet a 20 ms p95 planning deadline.
A lower-cost profile is accepted only if it also passes the closed-loop gate.

Apply the calibration gate to held-out results (never training trajectories):

```bash
python scripts/phase1_4_calibration_gate.py held_out_rollouts.npz \
--out results/phase1_4_calibration_gate.json
```

The NPZ must contain equal-length `nominal_rmse`, `residual_rmse`, `flow_id`,
and `seed` arrays. A failing gate exits with status 2.

The latency benchmark now uses float32, warm starts (already built into MPPI),
100 timed iterations after warm-up, and moving states/references rather than a
single zero state. It tests profiles in quality-first order:

```bash
python scripts/phase1_4_residual_benchmark.py --iterations 100 --warmup 10 \
--profiles 1024x25 768x20 512x20 --out results/phase1_4_timing_sweep.json
```

On the development CPU, the analytic 2x2 dynamics solve plus the 768x20 profile
reduced measured p95 to 14.49 ms and passed the 20 ms compute gate. The 1024x25
profile still missed at 20.95 ms. These figures are machine-specific and do not
replace target-hardware measurements.

Run the early nominal-vs-residual ablation and robustness sweep with:

```bash
python scripts/phase1_4_closed_loop_ablation.py \
--out results/phase1_4_closed_loop_ablation.json
```

The initial uncalibrated coefficients improved tracking by only 1.50% in this
structured-model-plant smoke test, below the pre-registered 10% gate. Therefore
Phase 1.4 currently **fails the benefit gate** despite passing its reduced
compute profile. This simplified plant is neither CFD nor hardware truth; use
held-out MuJoCo traces for calibration and then repeat on deployment hardware.
