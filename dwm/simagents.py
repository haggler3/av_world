"""Learned sim agents: a next-token behaviour model for every vehicle in the city, run in closed loop inside the simulator.

Current traffic simulation models agents with next-token prediction over discretised motion (MotionLM, arXiv 2309.16534;
SMART, arXiv 2405.15677) and scores them in closed loop by how likely the real logs are under the simulated distribution
(the Waymo Open Sim Agents Challenge, arXiv 2305.12032). This module does the same at city scale, on the simulator's own
human drivers:

    data   : run E copies of a city with calibrated human drivers only (diag=True); every 5th step, for every live,
             unwrecked vehicle, keep its 28 observation features o (the AV observation of xnet.sim.step, computed for every
             row) and the acceleration a it then applied.
    tokens : a -> one of K = 32 bins on [-8, 3] m/s^2.
    model  : MLP 28 -> 256 -> 256 -> K logits, trained by cross-entropy (Adam), i.e. p(token | o).
    rollout: the same city with every vehicle flagged as a policy-driven row (p_av = 1, safety envelope off); each step
             every vehicle samples a token from p(. | o) (or takes the arg-max), plus uniform jitter inside the bin.
             Routes, lane changes and junction entry stay the simulator's rules; the model sets only the acceleration.
    realism: per feature f in {speed, acceleration, jerk, gap to leader, time headway}, the WOSAC-style likelihood
             L_f = exp(mean_x log q_f(x)), x the held-out ground-truth samples, q_f the rollout's smoothed histogram;
             meta = mean_f L_f. Per-copy scores give intervals.

    python -m dwm.simagents --quick          # 2 copies, short runs, a 300-step model (smoke test, ~2 min)
    python -m dwm.simagents                  # data, train, closed-loop cells -> results/simagents/
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

K = 32
A_LO, A_HI = -8.0, 3.0
EDGES = np.linspace(A_LO, A_HI, K + 1)
CENTERS = 0.5 * (EDGES[1:] + EDGES[:-1])
WIDTH = (A_HI - A_LO) / K
U1_HUMAN = float(np.arctanh((1.25 - 1.15) / 0.35))  # policy output that sets the gap-acceptance scale to the humans' 1.25
OUT = "results/simagents"

# realism features: name -> (histogram edges)
FEATS = {"speed": np.linspace(0, 25, 51), "accel": np.linspace(-9, 3.5, 51), "jerk": np.linspace(-30, 30, 61),
         "gap": np.linspace(0, 150, 61), "headway": np.linspace(0, 10, 51)}


def tokenize(a):
    return np.clip(np.searchsorted(EDGES, np.asarray(a), side="right") - 1, 0, K - 1)


# ---------------------------------------------------------------- model (plain JAX)
def init(key, d_in=28, h=256):
    ks = jax.random.split(key, 3)
    g = lambda k, i, o: jax.random.normal(k, (i, o)) * np.sqrt(2.0 / i)  # noqa: E731
    return {"w1": g(ks[0], d_in, h), "b1": jnp.zeros(h), "w2": g(ks[1], h, h), "b2": jnp.zeros(h),
            "w3": g(ks[2], h, K) * 0.1, "b3": jnp.zeros(K)}


def logits(p, o):
    x = jax.nn.gelu(o @ p["w1"] + p["b1"])
    x = jax.nn.gelu(x @ p["w2"] + p["b2"])
    return x @ p["w3"] + p["b3"]


def train(obs, tok, steps=4000, batch=4096, lr=1e-3, seed=0, log=print):
    import optax
    obs, tok = jnp.asarray(obs, jnp.float32), jnp.asarray(tok, jnp.int32)
    p = init(jax.random.PRNGKey(seed), obs.shape[1])
    opt = optax.adamw(optax.cosine_decay_schedule(lr, steps), weight_decay=1e-4)
    s = opt.init(p)

    def loss(p, o, t):
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits(p, o)), t[:, None], 1))

    @jax.jit
    def upd(p, s, k):
        i = jax.random.randint(k, (batch,), 0, obs.shape[0])
        l, g = jax.value_and_grad(loss)(p, obs[i], tok[i])
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l
    key = jax.random.PRNGKey(seed + 1)
    for i in range(steps):
        p, s, l = upd(p, s, jax.random.fold_in(key, i))
        if i % 1000 == 0 or i == steps - 1:
            log(f"step {i} loss {float(l):.3f}")
    return p


def nll(p, obs, tok):
    lp = jax.nn.log_softmax(logits(p, jnp.asarray(obs, jnp.float32)))
    return float(-jnp.mean(jnp.take_along_axis(lp, jnp.asarray(tok, jnp.int32)[:, None], 1)))


def policy(p, mode="sample", temp=1.0):
    """xnet rl policy: obs [N, 28] -> (u [N, 2], aux). mode: 'sample' | 'argmax' | 'idm'."""
    def f(obs, key):
        if mode == "idm":
            from avlab import teb as T
            u = T.idm_policy_masked(obs, key, None)[0]
            return u.at[:, 1].set(U1_HUMAN), jnp.zeros((obs.shape[0], 2))
        lg = logits(p, obs)
        k1, k2 = jax.random.split(key)
        t = jnp.argmax(lg, 1) if mode == "argmax" else jax.random.categorical(k1, lg / temp, axis=1)
        jit = jax.random.uniform(k2, t.shape, minval=-0.5, maxval=0.5) * (mode == "sample")
        a = jnp.asarray(CENTERS)[t] + jit * WIDTH
        u0 = jnp.clip(jnp.where(a >= 0, a / 3.0, a / 8.0), -1.0, 1.0)
        return jnp.stack([u0, jnp.full_like(u0, U1_HUMAN)], 1), jnp.zeros((obs.shape[0], 2))
    return f


# ---------------------------------------------------------------- runs
def run(area="oakland", E=8, seed=0, warm_s=120.0, seconds=600.0, pol=None, keep_every=5, keep_obs=False, chunk_s=40.0, label_rule=False,
        n_slots=1024):
    """E copies; pol=None: calibrated humans only (ground truth); else every vehicle is driven by `pol`.
    Returns realism samples per copy, counters, and (keep_obs) training pairs. With label_rule (and a policy) the pairs
    are the states the policy visits, labelled with the acceleration the simulator's rule driver would apply there
    (DAgger, Ross et al. 2011, arXiv 1011.0686)."""
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    X.SHIELD = "off"
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated(p_av=0.0 if pol is None else 1.0)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for i in range(E)]), J, n_slots)

    def one(si, k):
        if pol is None:
            s2, tr = X.step(si, J, beh, nh, diag=True)
        else:
            s2, tr = X.step(si, J, beh, nh, rl=(pol, k))
        ok = si.alive & (si.wreck_t <= 0) & s2.alive & (s2.wreck_t <= 0)
        o = tr["obs"]
        has = o[:, 8] > 0.5
        gap = 50.0 * o[:, 9]
        f = dict(speed=s2.v, accel=s2.acc, jerk=(s2.acc - si.acc) / X.DT, gap=gap,
                 headway=gap / jnp.maximum(si.v, 0.1))
        m = dict(speed=ok, accel=ok, jerk=ok & (si.v > 0.1), gap=ok & has, headway=ok & has & (si.v > 1.0))
        lab = tr["a_rule"] if (pol is not None and label_rule) else s2.acc
        return s2, (f, m, o, lab, ok)
    n = int(round(chunk_s / X.DT))

    @jax.jit
    def chunk(st, key):
        def body(c, k):
            s2, out = jax.vmap(one)(c, jax.random.split(k, E))
            return s2, out
        return jax.lax.scan(body, st, jax.random.split(key, n))
    key = jax.random.PRNGKey(seed + 777)
    for i in range(int(round(warm_s / chunk_s))):
        st, _ = chunk(st, jax.random.fold_in(key, i))
    m0 = np.asarray(st.m)
    hist = {f: np.zeros((E, len(e) - 1)) for f, e in FEATS.items()}
    obs_l, tok_l = [], []
    for i in range(int(round(seconds / chunk_s))):
        st, (f, m, o, a, ok) = chunk(st, jax.random.fold_in(key, 10_000 + i))
        f = {k: np.asarray(v) for k, v in f.items()}
        m = {k: np.asarray(v) for k, v in m.items()}
        for k, e in FEATS.items():
            for c in range(E):
                hist[k][c] += np.histogram(np.clip(f[k][:, c][m[k][:, c]], e[0], e[-1] - 1e-6), e)[0]
        if keep_obs:
            sel = np.asarray(ok)[::keep_every]
            obs_l.append(np.asarray(o)[::keep_every][sel])
            tok_l.append(tokenize(np.asarray(a)[::keep_every][sel]))
    M = np.asarray(st.m) - m0
    from xnet import sim as X2
    idx = {k: j for j, k in enumerate(X2.METRICS)}
    crashes = M[:, [idx[k] for k in X2.CRASH_NAMES]].sum(1)
    res = dict(hist={k: v.tolist() for k, v in hist.items()}, veh_s=M[:, idx["veh_s"]].tolist(),
               veh_m=M[:, idx["veh_m"]].tolist(), finished=M[:, idx["finished"]].tolist(), crashes=crashes.tolist(),
               E=E, seed=seed, warm_s=warm_s, seconds=seconds)
    if keep_obs:
        res["obs"], res["tok"] = np.concatenate(obs_l), np.concatenate(tok_l)
    return res


# ---------------------------------------------------------------- realism
def likelihood(gt_counts, sim_counts, eps=1e-3):
    """WOSAC-style: exp(mean log q(x)) for ground-truth samples x (as bin counts) under the sim histogram q (probabilities
    per bin, additively smoothed so an empty bin is not -inf). 1 = identical, lower = less realistic."""
    gt, sim = np.asarray(gt_counts, float), np.asarray(sim_counts, float)
    q = sim / max(sim.sum(), 1.0)
    q = (q + eps) / (1 + eps * len(q))
    nz = gt > 0  # 0 log 0 = 0
    return float(np.exp((gt[nz] * np.log(q[nz])).sum() / max(gt.sum(), 1.0)))


def realism(gt, sim):
    """Per-feature and meta likelihood for the pooled sim and for each sim copy (pooled ground truth)."""
    g = {k: np.asarray(v).sum(0) for k, v in gt["hist"].items()}
    pooled = {k: likelihood(g[k], np.asarray(sim["hist"][k]).sum(0)) for k in FEATS}
    per_copy = [float(np.mean([likelihood(g[k], np.asarray(sim["hist"][k])[c]) for k in FEATS])) for c in range(sim["E"])]
    return {"features": pooled, "meta": float(np.mean(list(pooled.values()))), "per_copy_meta": per_copy}


def city_stats(r):
    h = sum(r["veh_s"]) / 3600
    return {"veh_hours": h, "speed_kmh": 3.6 * sum(r["veh_m"]) / sum(r["veh_s"]), "crashes": sum(r["crashes"]),
            "crashes_per_1000h": 1000 * sum(r["crashes"]) / h, "finished_per_h": sum(r["finished"]) / h}


def diff_ci(a, b, n=10000, seed=0):
    """Mean difference a - b of per-copy scores with a bootstrap 95 % interval and a two-sided permutation p."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    d = a.mean() - b.mean()
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    z = np.concatenate([a, b])
    perm = []
    for _ in range(n):
        rng.shuffle(z)
        perm.append(z[:len(a)].mean() - z[len(a):].mean())
    p = (np.sum(np.abs(perm) >= abs(d)) + 1) / (n + 1)
    return {"diff": float(d), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)), "p": float(p)}


def save_params(p, path):
    np.savez(path, **{k: np.asarray(v) for k, v in p.items()})


def load_params(path):
    z = np.load(path)
    return {k: jnp.asarray(z[k]) for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", default="all", help="data | train | cell:<name> | analyse | all")
    a = ap.parse_args()
    q = a.quick
    out = OUT + ("_quick" if q else "")
    os.makedirs(out, exist_ok=True)
    E, warm, secs = (2, 40.0, 40.0) if q else (8, 120.0, 600.0)
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)  # noqa: E731
    stages = ["data", "train", "cell:gt2", "cell:idm", "cell:argmax", "cell:sample", "analyse"] if a.stage == "all" else [a.stage]
    for stg in stages:
        if stg == "data":  # training data and the ground truth used for scoring come from different random streams
            r = run(E=E, seed=11, warm_s=warm, seconds=secs, keep_obs=True)
            np.savez_compressed(f"{out}/train_data.npz", obs=r.pop("obs"), tok=r.pop("tok"))
            json.dump(r, open(f"{out}/train_city.json", "w"))
            g = run(E=E, seed=12, warm_s=warm, seconds=secs)
            json.dump(g, open(f"{out}/gt.json", "w"))
            log("data", city_stats(r), city_stats(g))
        elif stg == "train":
            z = np.load(f"{out}/train_data.npz")
            obs, tok = z["obs"], z["tok"]
            rng = np.random.default_rng(0)
            idx = rng.permutation(len(tok))
            nv = min(50_000, len(tok) // 10)
            va, tr = idx[:nv], idx[nv:]
            t0 = time.time()
            p = train(obs[tr], tok[tr], steps=300 if q else 6000, log=log)
            marg = np.bincount(tok[tr], minlength=K) + 1.0
            info = {"n_train": int(len(tr)), "n_val": int(nv), "train_s": time.time() - t0,
                    "val_nll": nll(p, obs[va], tok[va]),
                    "val_nll_marginal": float(-np.mean(np.log(marg / marg.sum())[tok[va]])),
                    "val_acc_top1": float(np.mean(np.argmax(np.asarray(logits(p, jnp.asarray(obs[va]))), 1) == tok[va]))}
            save_params(p, f"{out}/model.npz")
            json.dump(info, open(f"{out}/train.json", "w"), indent=1)
            log("train", info)
        elif stg == "dagger":  # post hoc: one DAgger round on the states the sampled agents visit
            p = load_params(f"{out}/model.npz")
            r = run(E=E, seed=14, warm_s=warm, seconds=secs, pol=policy(p, "sample"), keep_obs=True, label_rule=True)
            z = np.load(f"{out}/train_data.npz")
            obs = np.concatenate([z["obs"], r["obs"]]); tok = np.concatenate([z["tok"], r["tok"]])
            t0 = time.time()
            p2 = train(obs, tok, steps=300 if q else 6000, log=log)
            save_params(p2, f"{out}/model_dagger.npz")
            json.dump({"n_added": int(len(r["tok"])), "n_total": int(len(tok)), "train_s": time.time() - t0},
                      open(f"{out}/dagger_train.json", "w"), indent=1)
            log("dagger", len(r["tok"]), "states added")
        elif stg.startswith("cell:"):
            c = stg[5:]
            if c == "gt2":
                r = run(E=E, seed=13, warm_s=warm, seconds=secs)
            elif c == "dagger":
                r = run(E=E, seed=13, warm_s=warm, seconds=secs, pol=policy(load_params(f"{out}/model_dagger.npz"), "sample"))
            else:
                p = load_params(f"{out}/model.npz")
                r = run(E=E, seed=13, warm_s=warm, seconds=secs, pol=policy(p, c))
            r["cell"] = c
            json.dump(r, open(f"{out}/{c}.json", "w"))
            log(c, city_stats(r))
        elif stg == "analyse":
            gt = json.load(open(f"{out}/gt.json"))
            cells = {c: json.load(open(f"{out}/{c}.json")) for c in ("gt2", "idm", "argmax", "sample", "dagger")
                     if os.path.exists(f"{out}/{c}.json")}
            res = {"gt": city_stats(gt), "cells": {c: {**realism(gt, r), **city_stats(r)} for c, r in cells.items()}}
            t = {}
            pc = lambda c: res["cells"][c]["per_copy_meta"]  # noqa: E731
            if {"sample", "idm"} <= set(cells):
                t["H1"] = {"cells": ["sample", "idm"], **diff_ci(pc("sample"), pc("idm"))}
            if {"sample", "argmax"} <= set(cells):
                t["S1"] = {"cells": ["sample", "argmax"], **diff_ci(pc("sample"), pc("argmax"))}
            if {"gt2", "sample"} <= set(cells):
                t["S2"] = {"cells": ["gt2", "sample"], **diff_ci(pc("gt2"), pc("sample"))}
            if {"dagger", "idm"} <= set(cells):  # post hoc, not pre-registered
                t["post_hoc_dagger_vs_idm"] = {"cells": ["dagger", "idm"], **diff_ci(pc("dagger"), pc("idm"))}
                t["post_hoc_dagger_vs_sample"] = {"cells": ["dagger", "sample"], **diff_ci(pc("dagger"), pc("sample"))}
            sec = [n for n in ("S1", "S2") if n in t]
            if sec:
                from avlab.whocrashes import holm
                for n_, ph in zip(sec, holm([t[n_]["p"] for n_ in sec])):
                    t[n_]["p_holm"] = ph
            res["tests"] = t
            json.dump(res, open(f"{out}/realism.json", "w"), indent=1)
            for c, v in res["cells"].items():
                log(f"{c:7s} meta {v['meta']:.3f}", {k: round(x, 3) for k, x in v["features"].items()},
                    f"{v['speed_kmh']:.1f} km/h, {v['crashes']} crashes, {v['finished_per_h']:.1f} trips/veh-h")
            log("gt", res["gt"])
            for n_, v in t.items():
                log(n_, v)


if __name__ == "__main__":
    main()
