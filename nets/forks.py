"""Exact interventions: fork the city at a moment, change one AV action, replay with the same dice (shared by N2 and N7).

Every 4 s of a 5x-risk Oakland run (25 % AVs driven by the phase-3 policy, no safety filter), all 32 copies are forked into K + 1 = 6
branches that run 4 s as one vmapped batch:

  branches 0..K-1   every AV gets one of the K action perturbations for the first 2 s; which one is a Latin square over AVs
                    (AV i in branch b gets DELTAS[(b + h_i) mod K], h_i random), so each AV sees all K actions and the AVs around it
                    do not all brake together
  branch K          nobody is perturbed: what the policy really did (the "log"), and the state the run continues from

With Behaviour.crn = 1 every human keeps its dice in every branch, so for AV i the K outcomes differ only by actions (its own, and the
fixed-offset actions of the other AVs). Per AV and fork we keep: the observation and the policy's action at the fork, the outcome of
each action (crash, near miss, red run in the next 4 s, counted only while the same vehicle holds the slot), and the acceleration of
the vehicle directly behind and directly ahead over the 4 s in every branch (for N7).

    python -m nets.forks [cycles]          # ~10 min on 4 cores -> results/nets/forks.npz
"""
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from avlab import adversary as AD  # noqa: E402
from avrl import xppo as P  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import collect as C  # noqa: E402

OUT = "results/nets"
PRIOR = "results/unaided/phase3/policy.npz"
DELTAS = np.array([(0.0, 0.0), (-0.4, 0.0), (0.3, 0.0), (0.0, 0.8), (0.0, -0.8)], np.float32)  # accel command, gap-acceptance
K, PERT, WIN = len(DELTAS), 10, 20  # perturb 2 s, watch 4 s (0.2 s steps)
BAD_MARGIN = -1.0  # a moving AV more than 1 m inside its barrier (closer than it could stop for the obstacle ahead) counts as a bad outcome
NAMES = ("as is", "brake harder", "accelerate", "bigger gaps", "smaller gaps")


def make_policy(prior):
    def pol(obs, key, pc):
        u = P.mean(prior, obs)
        on = pc["_t"] < pc["pert"]
        return u + jnp.where(on, pc["d"], 0.0), jnp.zeros_like(u)
    return pol


def feat(s, tr, pc):
    return {"crash": tr["crash"], "near": tr["near"], "red": tr["red"], "uid": s.uid, "acc": s.acc.astype(jnp.float16),
            "bar": jnp.where(s.v > 2.0, tr["bar_h"], 99.0).astype(jnp.float16)}


def feat0(s, tr, pc):
    return {"obs": tr["obs"], "valid": tr["valid"], "uid": s.uid}


def neighbours(trk, s, length, alive):
    """Per slot: the slot directly behind on the same track (follower) and directly ahead (leader), -1 if none within 40 m."""
    n = len(s)
    fol, lead = np.full(n, -1), np.full(n, -1)
    idx = np.where(alive)[0]
    o = idx[np.lexsort((s[idx], trk[idx]))]
    for a, b in zip(o[:-1], o[1:]):                 # b is ahead of a on the same track
        if trk[a] == trk[b] and s[b] - length[b] - s[a] < 40.0:
            fol[b], lead[a] = a, b
    return fol, lead


def collect(cycles=120, n_envs=32, warm=120.0, seed=0):
    os.makedirs(OUT, exist_ok=True)
    X.SHIELD = "off"
    w = C.World()
    beh = AD.world(5.0)._replace(crn=1.0)
    prior = P.load(PRIOR)
    pol = make_policy(prior)
    rng = np.random.default_rng(seed)
    st = C.run(w, w.reset(n_envs, seed), beh, warm)
    N = st.v.shape[1]
    B = K + 1
    rows = {k: [] for k in ("margin", "obs", "env", "cycle", "slot", "bad", "crash", "near", "red", "fol_acc", "lead_acc", "has_fol", "has_lead", "u_prior", "fol_state")}
    t0 = time.time()
    for c in range(cycles):
        h = rng.integers(0, K, (n_envs, N))
        didx = np.stack([(b + h) % K for b in range(K)] + [np.full((n_envs, N), -1)], 1).reshape(n_envs * B, N)   # [E*B, N]
        d = np.where(didx[..., None] >= 0, DELTAS[np.maximum(didx, 0)], 0.0).astype(np.float32)
        stf = C.fork(st, B)
        host0 = jax.tree.map(np.asarray, (st.trk, st.s, st.length, st.alive, st.av, st.v, st.acc))
        pc = {"d": jnp.asarray(d), "pert": jnp.full((n_envs * B,), PERT, jnp.int32)}
        s1, f0 = C.record(w, stf, beh, X.DT, 1, feat0, policy=pol, per_copy=pc)
        pc1 = {"d": jnp.asarray(d), "pert": jnp.full((n_envs * B,), PERT - 1, jnp.int32)}
        s2, f1 = C.record(w, s1, beh, (WIN - 1) * X.DT, 1, feat, policy=pol, per_copy=pc1)
        f0, f1 = C.to_host(f0, np.float32), C.to_host(f1, np.float32)
        obs0 = f0["obs"][0].reshape(n_envs, B, N, -1)[:, K]             # the observation at the fork (same in every branch)
        valid0 = f0["valid"][0].reshape(n_envs, B, N)[:, K].astype(bool)
        uid0 = f0["uid"][0].reshape(n_envs, B, N)[:, K]
        same = (f1["uid"].reshape(WIN - 1, n_envs, B, N) == uid0[None, :, None, :])       # the vehicle still holds the slot
        ev = {k: (f1[k].reshape(WIN - 1, n_envs, B, N).astype(bool) & same).any(0) for k in ("crash", "near", "red")}   # [E, B, N]
        acc = f1["acc"].reshape(WIN - 1, n_envs, B, N)
        margin = np.where(same, f1["bar"].reshape(WIN - 1, n_envs, B, N), 99.0).min(0)        # the AV's worst barrier value while moving [E, B, N]
        for e in range(n_envs):
            trk, s, ln, al, av, vv, aa = (x[e] for x in host0)
            fol, lead = neighbours(trk, s, ln, al)
            for i in np.where(valid0[e] & av)[0]:
                # outcome per action k: the branch where this AV got action k, then the unperturbed branch (index K)
                br = [int(np.where(didx.reshape(n_envs, B, N)[e, :K, i] == k)[0][0]) for k in range(K)] + [K]
                for name in ("crash", "near", "red"):
                    rows[name].append(ev[name][e, br, i])
                rows["margin"].append(margin[e, br, i])
                rows["bad"].append(ev["crash"][e, br, i] | ev["near"][e, br, i] | (margin[e, br, i] < BAD_MARGIN))
                rows["obs"].append(obs0[e, i])
                rows["env"].append(e), rows["cycle"].append(c), rows["slot"].append(i)
                rows["has_fol"].append(fol[i] >= 0), rows["has_lead"].append(lead[i] >= 0)
                f_, l_ = max(fol[i], 0), max(lead[i], 0)
                # the follower's own situation at the fork: its gap to us, its speed and acceleration, and whether it is an AV
                rows["fol_state"].append(np.array([s[i] - ln[i] - s[f_], vv[f_], aa[f_], float(av[f_]), vv[i], aa[i]], np.float32) if fol[i] >= 0 else np.zeros(6, np.float32))
                rows["fol_acc"].append(acc[:, e, br, max(fol[i], 0)].T if fol[i] >= 0 else np.zeros((K + 1, WIN - 1), np.float32))
                rows["lead_acc"].append(acc[:, e, br, max(lead[i], 0)].T if lead[i] >= 0 else np.zeros((K + 1, WIN - 1), np.float32))
        st = jax.tree.map(lambda a: a.reshape((n_envs, B) + a.shape[1:])[:, K], s2)   # continue from the unperturbed branch
        if c % 10 == 0 or c == cycles - 1:
            bad = np.array(rows["bad"])
            print(f"cycle {c}: {len(bad)} AV forks, bad-outcome rate per action {bad.mean(0).round(4).tolist()} ({time.time() - t0:.0f}s)", flush=True)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["u_prior"] = np.asarray(P.mean(prior, jnp.asarray(out["obs"])))
    np.savez_compressed(f"{OUT}/forks.npz", **out, deltas=DELTAS)
    X.SHIELD = "idm"
    print(f"saved {len(out['bad'])} AV forks x {K + 1} branches in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    collect(*(int(x) for x in sys.argv[1:2]))
