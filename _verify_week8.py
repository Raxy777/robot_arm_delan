"""Fresh-path verification of the Week-8 clip fix + data sanity.

Mirrors the exact logic in arm_sim.ArmSim(analytic) + data_collection, inlined
here so it doesn't depend on the (sandbox-cache-stale) edited modules.
"""
import numpy as np
import dynamics
from kinematics import inverse_kinematics
from controllers import GravityCompPDController  # dynamics-only, safe

DT = 0.001
TAU_LIMIT = 60.0


def clip(tau):
    return np.clip(np.asarray(tau, float), -TAU_LIMIT, TAU_LIMIT)


def rk4(q, qd, tau):
    def deriv(s):
        return np.concatenate([s[2:], dynamics.forward_dynamics(s[:2], s[2:], tau)])
    s = np.concatenate([q, qd])
    k1 = deriv(s); k2 = deriv(s + .5*DT*k1); k3 = deriv(s + .5*DT*k2); k4 = deriv(s + DT*k3)
    s = s + (DT/6)*(k1 + 2*k2 + 2*k3 + k4)
    return s[:2], s[2:]


def multisine(rng, n_bursts, burst_len, tau_amp, fr):
    Q=QD=QDD=TAU=None; out=[[],[],[],[]]
    for _ in range(n_bursts):
        q=rng.uniform(-np.pi,np.pi,2); qd=rng.uniform(-1,1,2)
        f=rng.uniform(*fr,size=(2,5)); ph=rng.uniform(0,2*np.pi,size=(2,5))
        a=rng.uniform(.3,1,size=(2,5)); a*=tau_amp/a.sum(1,keepdims=True)
        for k in range(burst_len):
            t=k*DT
            tau=clip([np.sum(a[j]*np.sin(2*np.pi*f[j]*t+ph[j])) for j in range(2)])
            out[0].append(q); out[1].append(qd); out[2].append(dynamics.forward_dynamics(q,qd,tau)); out[3].append(tau)
            q,qd=rk4(q,qd,tau)
    return [np.array(o) for o in out]


def pd_setpoints(rng, n_targets, hold, kp=80., kd=20.):
    ctrl=GravityCompPDController(kp=kp,kd=kd); out=[[],[],[],[]]
    q=np.array([0.5,0.5]); qd=np.zeros(2)
    for _ in range(n_targets):
        r=rng.uniform(0.5,1.8); th=rng.uniform(0,np.pi)
        try: qt=inverse_kinematics([r*np.cos(th), r*np.sin(th)])
        except ValueError: continue
        for _ in range(hold):
            tau=clip(ctrl(q,qd,qt,np.zeros(2),np.zeros(2)))
            out[0].append(q); out[1].append(qd); out[2].append(dynamics.forward_dynamics(q,qd,tau)); out[3].append(tau)
            q,qd=rk4(q,qd,tau)
    return [np.array(o) for o in out]


rng=np.random.default_rng(0)
for label, parts in [
    ("train-like", [multisine(rng,8,300,8.0,(0.1,1.5)), pd_setpoints(rng,8,200)]),
    ("ood-like",   [multisine(rng,6,300,16.0,(1.5,3.0))]),
]:
    q=np.concatenate([p[0] for p in parts]); qd=np.concatenate([p[1] for p in parts])
    qdd=np.concatenate([p[2] for p in parts]); tau=np.concatenate([p[3] for p in parts])
    err=max(np.max(np.abs(dynamics.inverse_dynamics(q[i],qd[i],qdd[i])-tau[i])) for i in range(len(q)))
    print(f"{label:11s} n={len(q):5d} |qd|max {np.abs(qd).max():6.2f} "
          f"|qdd|max {np.abs(qdd).max():7.1f} |tau|max {np.abs(tau).max():6.2f} "
          f"selfcons {err:.1e}")
