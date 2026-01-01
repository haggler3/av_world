"""Latent world model and planner, v2: what the v1 planner (wmplan.model / wmplan.planner) got wrong, fixed in one pass.

On AVBench v1 ran red lights (about 5 per 100 km, its PPO prior none) and kept its prior's crash rate. The causes, and the fixes
here (after TD-MPC2, DreamerV3 and PETS / MOPO):

  horizon      v1 plans 2.4 s and trains on 3 s; stopping from 15 m/s takes about 4 s, so a red run is often already
               unavoidable inside the horizon  ->  4 s plans, 5 s training unrolls, a terminal value for what lies beyond
  stop target  the distance to the stop target carries a 300 m sentinel when there is none, and a target appears when a light
               turns: v1 regressed that jump (stop-gap error 36 m against 8.6 m for holding it)  ->  distance features
               are only scored while their flag says the thing exists
  exploitation one deterministic model searched over a wide action box  ->  an ensemble of dynamics and event heads, plans
               scored by mean - beta x spread, actions kept inside the range the data covers. (Hard-rejecting plans whose
               decoded stop or leader margin goes negative made it worse: 3.9 rear-end crashes per AV-hour against 0.3 without;
               the `hard` weight is kept at 0.)
  prior, data  the 234k-step PPO policy (0.64 crashes per AV-hour unaided) and 363 k decisions  ->  the barrier-teacher
               phase 3 policy as the prior, and 1.2 M decisions logged half under each policy
  network      plain ELU MLPs, 48-d latent  ->  LayerNorm + SiLU blocks, a SimNorm latent (groups of 8 on the simplex)

    python -m wmplan.v2 train          # results/wmplan/v2/model.pkl, model_eval.json (data: results/wmplan/v2/data_*.npz)
    python -m avrl.bench run-wm2       # the planner on every AVBench scenario
    python -m wmplan.v2 gifs           # docs/img/wm2_planner_junction.gif, wm2_planner_merge.gif
"""
import json
import os
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from avrl import xppo as P

OUT = "results/wmplan/v2"
PRIOR = "results/unaided/phase3/policy.npz"
DATA = (f"{OUT}/data_xnet3.npz", f"{OUT}/data_phase3.npz")
OBS, ACT, NMARG = 28, 2, 5
H, ZD, HID, HDYN, GROUP, E = 25, 64, 256, 160, 8, 3  # HDYN: the dynamics run 32 x 20 x 2 times per AV per step in planning
R_SCALE, POS_W, GAMMA, HAZ_K = 10.0, 30.0, 0.97, 10
HOLDOUT = (7, 15)  # one copy of the area under each logging policy
MASKED = {7: 6, 24: 6, 9: 8, 10: 8, 11: 8, 25: 8, 26: 8}  # distance feature -> the flag saying the stop target / leader exists
# the planner: every AV in a copy, 4 s ahead, 4 knots per action dimension
PLAN = dict(K=112, S=32, ITERS=2, HP=20, ELITE=6, KNOTS=4, beta=1.0, crash=30.0, near=3.0, red=40.0, haz=5.0, margin=0.5,
            hard=0.0, jerk=0.002, prior=0.05, sd0=0.5, value=1.0, gate=1.0)


# ---------------------------------------------------------------- data

def load_data(paths=DATA):
    """Both logs, rows sorted by (copy, slot, time); the second log's copies are numbered 8-15. Adds the continuation flag, the
    'crash / red run within the next HAZ_K steps' targets and the discounted return-to-go."""
    parts = []
    for i, p in enumerate(paths):
        d = dict(np.load(p))
        d["env"] = d["env"].astype(np.int16) + 8 * i
        parts.append(d)
    d = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
    n = len(d["t"])
    same = (d["env"][1:] == d["env"][:-1]) & (d["slot"][1:] == d["slot"][:-1]) & (d["t"][1:] == d["t"][:-1] + 1) & (d["done"][:-1] == 0)
    cont = np.concatenate([same, [False]])
    ev_c, ev_r = d["crash"].astype(bool), d["red"].astype(bool)
    hc, hr, alive = ev_c.copy(), ev_r.copy(), np.ones(n, bool)
    for k in range(1, HAZ_K):
        alive[:n - k] &= cont[k - 1:n - 1]
        alive[n - k:] = False
        hc[:n - k] |= alive[:n - k] & ev_c[k:]
        hr[:n - k] |= alive[:n - k] & ev_r[k:]
    r = (np.clip(d["reward"].astype(np.float32), -0.5, 0.5) - 0.5 * (ev_c | ev_r)).tolist()
    c = cont.tolist()
    g, nxt = [0.0] * n, 0.0
    for i in range(n - 1, -1, -1):
        nxt = r[i] + (GAMMA * nxt if c[i] else 0.0)
        g[i] = nxt
    d.update(cont=cont, hcrash=hc, hred=hr, G=np.asarray(g, np.float32), n=n)
    return d


def device_data(d):
    f = lambda k: jnp.asarray(np.asarray(d[k], np.float32))  # noqa: E731
    return {"obs": f("obs"), "u": f("u"), "marg": f("margins"), "rew": jnp.clip(f("reward"), -0.5, 0.5) * R_SCALE,
            "ev": jnp.stack([f("crash"), f("near"), f("red"), f("hcrash"), f("hred")], -1), "G": f("G"), "cont": f("cont")}


def windows(D, idx):
    n = D["obs"].shape[0]
    j = jnp.minimum(idx[:, None] + jnp.arange(H + 1)[None], n - 1)
    valid = jnp.cumprod(jnp.concatenate([jnp.ones((idx.shape[0], 1)), D["cont"][j[:, :-1]]], 1), 1)
    return {"obs": D["obs"][j], "u": D["u"][j[:, :H]], "marg": D["marg"][j], "rew": D["rew"][j[:, :H]], "ev": D["ev"][j[:, :H]],
            "G": D["G"][j], "valid": valid}


# ---------------------------------------------------------------- the model

def _mlp(key, sizes):
    ks = jax.random.split(key, len(sizes))
    out = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layer = {"w": jax.random.normal(ks[i], (a, b)) * np.sqrt(2.0 / a), "b": jnp.zeros(b)}
        if i < len(sizes) - 2:
            layer |= {"g": jnp.ones(b), "beta": jnp.zeros(b)}
        out.append(layer)
    return out


BF16 = True  # matmuls in bfloat16 with float32 accumulation; LayerNorm, softmax and losses stay in float32


def run(layers, x):
    for layer in layers:
        if BF16:
            x = jnp.matmul(x.astype(jnp.bfloat16), layer["w"].astype(jnp.bfloat16), preferred_element_type=jnp.float32) + layer["b"]
        else:
            x = x @ layer["w"] + layer["b"]
        if "g" in layer:  # LayerNorm + SiLU on every hidden layer
            mu, var = x.mean(-1, keepdims=True), x.var(-1, keepdims=True)
            x = jax.nn.silu((x - mu) * jax.lax.rsqrt(var + 1e-5) * layer["g"] + layer["beta"])
    return x


def simnorm(z):
    return jax.nn.softmax(z.reshape(z.shape[:-1] + (ZD // GROUP, GROUP)), -1).reshape(z.shape)


def init(key, mu, sd, act_lo, act_hi):
    k = jax.random.split(key, 8)

    def member(kk):
        return {"dyn": _mlp(jax.random.fold_in(kk, 0), [ZD + ACT, HDYN, HDYN, ZD]),
                "trans": _mlp(jax.random.fold_in(kk, 1), [ZD + ACT, 96, 6])}
    return {"enc": _mlp(k[0], [OBS, HID, HID, ZD]), "ens": jax.vmap(member)(jax.random.split(k[1], E)),
            "obs": _mlp(k[2], [ZD, HID, OBS]), "marg": _mlp(k[3], [ZD, 128, NMARG]), "value": _mlp(k[4], [ZD, HID, 1]),
            "stats": {"mu": jnp.asarray(mu), "sd": jnp.asarray(sd), "lo": jnp.asarray(act_lo), "hi": jnp.asarray(act_hi)}}


def encode(p, obs):
    s = p["stats"]
    return simnorm(run(p["enc"], (obs - s["mu"][:OBS]) / s["sd"][:OBS]))


def act_in(u):
    return jnp.stack([jnp.clip(u[..., 0], -1.0, 1.0), jnp.tanh(u[..., 1])], -1)  # what the simulator does with u


def unroll(p, z0, us):
    """z0 [..., ZD], us [..., T, 2] -> per ensemble member: latents [E, ..., T+1, ZD], transition heads [E, ..., T, 6]
    (reward, then logits of crash, near miss, red run, crash within HAZ_K steps, red run within HAZ_K steps)."""
    a = jnp.moveaxis(act_in(us), -2, 0)

    def member(m):
        def body(z, a_):
            x = jnp.concatenate([z, a_], -1)
            z1 = simnorm(run(m["dyn"], x))
            return z1, (z1, run(m["trans"], x))
        _, (zs, trs) = jax.lax.scan(body, z0, a)
        return jnp.moveaxis(jnp.concatenate([z0[None], zs], 0), 0, -2), jnp.moveaxis(trs, 0, -2)
    return jax.vmap(member)(p["ens"])


def decode_marg(p, z):
    s = p["stats"]
    return run(p["marg"], z) * s["sd"][OBS:] + s["mu"][OBS:]


def decode(p, z):
    """-> (obs, margins) in their original units."""
    s = p["stats"]
    obs = run(p["obs"], z) * s["sd"][:OBS] + s["mu"][:OBS]
    return obs.at[..., 0].set(jnp.maximum(obs[..., 0], 0.0)), decode_marg(p, z)


def value(p, z):
    return run(p["value"], z)[..., 0]


def loss_fn(p, b):
    s = p["stats"]
    v = b["valid"]
    vt = v[:, :H]

    def wsum(x, w):  # x [E, B, T] or [B, T], w [B, T]
        return (x * w).sum() / jnp.maximum(w.sum() * (x.size // w.size), 1.0)
    z0 = encode(p, b["obs"][:, 0])
    zs, trs = unroll(p, z0, b["u"])  # [E, B, H+1, ZD], [E, B, H, 6]
    on = (b["obs"] - s["mu"][:OBS]) / s["sd"][:OBS]
    m = jnp.ones_like(on)
    for j, f in MASKED.items():  # no target for the distance to a stop target / leader that is not there
        m = m.at[..., j].set(b["obs"][..., f])
    w0 = v.at[:, 0].multiply(4.0)  # reconstructing the present scene well is the base of every prediction
    l_obs = wsum((((run(p["obs"], zs) - on) ** 2) * m).mean(-1), w0)
    l_m = wsum(((run(p["marg"], zs) - (b["marg"] - s["mu"][OBS:]) / s["sd"][OBS:]) ** 2).mean(-1), v)
    l_r = wsum((trs[..., 0] - b["rew"]) ** 2, vt)
    y = b["ev"]
    l_c = wsum(-(POS_W * y * jax.nn.log_sigmoid(trs[..., 1:]) + (1 - y) * jax.nn.log_sigmoid(-trs[..., 1:])).sum(-1), vt)
    z_tgt = jax.lax.stop_gradient(encode(p, b["obs"][:, 1:]))
    l_z = wsum(((zs[:, :, 1:] - z_tgt) ** 2).sum(-1), v[:, 1:])
    l_v = wsum((value(p, encode(p, b["obs"])) - b["G"]) ** 2, v) + wsum((value(p, zs) - b["G"]) ** 2, v)
    return l_obs + 2.0 * l_m + l_r + 0.3 * l_c + l_z + 0.5 * l_v, (l_obs, l_m, l_r, l_c, l_z, l_v)


def save(p, path=f"{OUT}/model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, p), f)


def load(path=f"{OUT}/model.pkl"):
    with open(path, "rb") as f:
        return jax.tree.map(jnp.asarray, pickle.load(f))


def _auroc(s, y):
    from scipy.stats import rankdata
    y = y > 0
    if y.sum() == 0 or y.all():
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (len(y) - y.sum())))


def evaluate(p, D, idx):
    """v1's held-out metrics (wmplan.model.evaluate) from the ensemble mean, 0-4 s ahead, plus the hazard heads."""
    @jax.jit
    def f(b):
        zs, trs = unroll(p, encode(p, b["obs"][:, 0]), b["u"])
        return decode(p, zs.mean(0)), trs.mean(0)
    acc = {k: [] for k in ("obs", "oh", "mh", "marg", "valid", "tr", "ev")}
    for i in range(0, len(idx), 2048):
        b = windows(D, jnp.asarray(idx[i:i + 2048]))
        (o_hat, m_hat), trs = f(b)
        for k, x in (("obs", b["obs"]), ("oh", o_hat), ("mh", m_hat), ("marg", b["marg"]), ("valid", b["valid"]), ("tr", trs), ("ev", b["ev"])):
            acc[k].append(np.asarray(x))
    ob, oh, mh, mg, v, tr, ev = (np.concatenate(acc[k]) for k in ("obs", "oh", "mh", "marg", "valid", "tr", "ev"))
    res = {}
    for name, j, scale, present in (("speed m/s", 0, 15.0, None), ("leader gap m", 9, 50.0, 8), ("stop gap m", 7, 100.0, 6)):
        for k in (0, 5, 10, 15, 20):
            msk = v[:, k] * (1.0 if present is None else ob[:, 0, present] * ob[:, k, present])  # the leader / stop target exists
            r = lambda e: float(np.sqrt((e ** 2 * msk).sum() / max(msk.sum(), 1)))  # noqa: E731
            res[f"{name} @{k * 0.2:.0f}s"] = {"model": r(oh[:, k, j] - ob[:, k, j]) * scale, "hold": r(ob[:, 0, j] - ob[:, k, j]) * scale}
    for i, name in enumerate(("stop margin", "leader margin", "junction margin", "merge gap behind", "merge gap ahead")):
        w = v[:, 10]
        rms = lambda e: float(np.sqrt((e ** 2 * w).sum() / w.sum()))  # noqa: E731
        res[f"{name} @2s (RMS)"] = {"model": rms(mh[:, 10, i] - mg[:, 10, i]), "hold": rms(mg[:, 0, i] - mg[:, 10, i])}
    vt = v[:, :H].ravel() > 0
    for name, i in (("crash", 0), ("near miss", 1), ("red-light run", 2), ("crash within 2 s", 3), ("red-light run within 2 s", 4)):
        y, s_ = ev[..., i].ravel()[vt], tr[..., i + 1].ravel()[vt]
        res[f"{name} AUROC"] = _auroc(s_, y)
        res[f"held-out {name} events"] = int(y.sum())
    return res


def train(steps=15000, batch=1024, seed=0):
    t0 = time.time()
    d = load_data()
    starts = np.nonzero(d["cont"][: d["n"] - H])[0]
    held = np.isin(d["env"][starts], HOLDOUT)
    tr_idx, te_idx = starts[~held], starts[held]
    # oversample windows that contain a crash, near miss or red run (rare, and what the planner most needs to foresee)
    ev = np.convolve((d["crash"] + d["near"] + d["red"]).astype(float), np.ones(H), "full")[H - 1:H - 1 + d["n"]]
    hot = tr_idx[ev[tr_idx] > 0]
    rows = np.nonzero(~np.isin(d["env"], HOLDOUT))[0]
    x = np.concatenate([d["obs"][rows].astype(np.float32), d["margins"][rows].astype(np.float32)], 1)
    u = d["u"][rows].astype(np.float32)
    mu, sd = x.mean(0), np.maximum(x.std(0), 1e-2)
    for j, f in MASKED.items():  # a distance feature's statistics over the rows where the thing exists
        xs = x[x[:, f] > 0.5, j]
        mu[j], sd[j] = xs.mean(), max(float(xs.std()), 1e-2)
    lo, hi = np.percentile(u, 1, 0), np.percentile(u, 99, 0)
    print(f"{d['n']} decisions, {len(starts)} windows ({len(tr_idx)} train, {len(te_idx)} held out), {len(hot)} with an event; "
          f"action range u0 [{lo[0]:.2f}, {hi[0]:.2f}] u1 [{lo[1]:.2f}, {hi[1]:.2f}]; loaded in {time.time() - t0:.0f} s", flush=True)
    D = device_data(d)
    p = init(jax.random.PRNGKey(seed), mu, sd, lo, hi)
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(500, steps // 10), steps, 3e-5), weight_decay=1e-5))
    st = opt.init(p)

    @jax.jit
    def upd(p, st, idx):
        (l, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(p, windows(D, idx))
        g["stats"] = jax.tree.map(jnp.zeros_like, g["stats"])  # the statistics are fixed
        u_, st = opt.update(g, st, p)
        return optax.apply_updates(p, u_), st, l, parts

    rng = np.random.default_rng(seed)
    t1 = time.time()
    for s_ in range(steps):
        idx = rng.choice(tr_idx, batch)
        idx[: batch // 8] = rng.choice(hot, batch // 8)
        p, st, l, parts = upd(p, st, jnp.asarray(idx))
        if s_ % 1000 == 0 or s_ == steps - 1:
            print(f"step {s_}: loss {float(l):.4f} (obs {float(parts[0]):.3f} margins {float(parts[1]):.3f} reward {float(parts[2]):.3f} "
                  f"events {float(parts[3]):.3f} latent {float(parts[4]):.4f} value {float(parts[5]):.3f}) {time.time() - t1:.0f} s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    save(p)
    te = np.random.default_rng(1).choice(te_idx, min(30000, len(te_idx)), replace=False)
    res = evaluate(p, D, te) | {"train_s": round(time.time() - t1), "steps": steps, "batch": batch, "decisions": int(d["n"])}
    with open(f"{OUT}/model_eval.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res.items():
        print(k, v)


# ---------------------------------------------------------------- the planner

def _interp(hp, kn):
    return jnp.asarray(np.stack([np.interp(np.linspace(0, kn - 1, hp), np.arange(kn), np.eye(kn)[i]) for i in range(kn)], 1))


def accel(u):
    u0 = jnp.clip(u[..., 0], -1.0, 1.0)
    return jnp.where(u0 >= 0, 3.0 * u0, 8.0 * u0)


def score(p, z0, us, w=PLAN, a0=None, u_ref=None):
    """z0 [..., ZD], us [..., S, T, 2] -> plan scores [..., S] (ensemble mean - beta x spread) and the decoded futures
    (ensemble means: latents [..., S, T+1, ZD], transition heads [..., S, T, 6], margins [..., S, T, 5])."""
    z = jnp.broadcast_to(z0[..., None, :], us.shape[:-2] + z0.shape[-1:])
    zs, trs = unroll(p, z, us)
    marg = decode_marg(p, zs[..., 1:, :])
    T = us.shape[-2]
    sh = jnp.log(POS_W)  # the event heads were trained with positives weighted x POS_W

    def sg(i):
        return jax.nn.sigmoid(trs[..., i] - sh)
    step = trs[..., 0] / R_SCALE - w["crash"] * sg(1) - w["near"] * sg(2) - w["red"] * sg(3) \
        - w["margin"] * jax.nn.relu(-marg[..., :3]).sum(-1) - w["hard"] * (marg[..., :2].min(-1) < -0.05)
    a = accel(us)
    prev = jnp.concatenate([(a[..., :1] if a0 is None else jnp.broadcast_to(a0[..., None, None], a[..., :1].shape)), a[..., :-1]], -1)
    step = step - w["jerk"] * ((a - prev) / 0.2) ** 2
    if u_ref is not None:
        step = step - w["prior"] * ((us - u_ref[..., None, :, :]) ** 2).sum(-1)
    ret = (step * GAMMA ** jnp.arange(T)).sum(-1) + GAMMA ** T * (w["value"] * value(p, zs[..., -1, :])
                                                                 - w["haz"] * (sg(4)[..., -1] + sg(5)[..., -1]))
    return ret.mean(0) - w["beta"] * ret.std(0), (zs.mean(0), trs.mean(0), marg.mean(0))


def plan(p, z0, mu0, key, w=PLAN, a0=None):
    """CEM over KNOTS knots per action dimension around mu0 [K, HP, 2] (the prior's action, held), inside the action range
    the data covers -> (mean [K, HP, 2], the last samples, their scores)."""
    kn, hp = w["KNOTS"], mu0.shape[1]
    M = _interp(hp, kn)
    lo, hi = p["stats"]["lo"], p["stats"]["hi"]
    mu = jnp.clip(mu0[:, np.round(np.linspace(0, hp - 1, kn)).astype(int)], lo, hi)  # the prior plan at the knot times
    sd = jnp.full_like(mu, w["sd0"])
    us = sc = None
    for _ in range(w["ITERS"]):
        key, k = jax.random.split(key)
        eps = jax.random.normal(k, mu.shape[:1] + (w["S"],) + mu.shape[1:]).at[:, 0].set(0.0)
        knots = jnp.clip(mu[:, None] + sd[:, None] * eps, lo, hi)
        us = jnp.einsum("hk,bskd->bshd", M, knots)
        sc, _ = score(p, z0, us, w, a0, mu0)
        el = jnp.argsort(-sc, axis=1)[:, :w["ELITE"]]
        e = jnp.take_along_axis(knots, el[:, :, None, None], 1)
        mu, sd = e.mean(1), e.std(1) + 0.05
    return jnp.einsum("hk,bkd->bhd", M, mu), us, sc


def prior_plan(p, prior, z0, u0, hp):
    """The prior's own plan, imagined: the policy acts on the decoded latent at every step and the latent unfolds under its
    action (ensemble member 0) -> [K, hp, 2], starting with its real action u0. Holding its current action for 4 s instead
    (v1, and the first v2) makes the prior's plan look bad whenever it is braking, and the planner then drives on into reds."""
    m0 = jax.tree.map(lambda x: x[0], p["ens"])

    def body(z, _):
        u = P.mean(prior, decode(p, z)[0])
        return simnorm(run(m0["dyn"], jnp.concatenate([z, act_in(u)], -1))), u
    _, us = jax.lax.scan(body, z0, None, length=hp)
    return jnp.moveaxis(us, 0, 1).at[:, 0].set(u0)


def make(p, prior, w=PLAN):
    """A policy for xnet.sim.step(rl=(policy, key, True)): plans for up to K AV rows of `mask`; the prior drives any others.
    A trust region: an AV takes the planned action only where the plan's score beats the prior's own (imagined) plan by
    more than `gate`, i.e. where the model foresees the prior getting into trouble, and then only in the cautious direction
    (harder braking, a larger gap): a predictive safety layer on the policy. Free overrides ran red lights the prior never
    runs: the model's counterfactual 'go' is less reliable than its 'this ends badly'."""
    def policy(obs, key, mask):
        fb = P.mean(prior, obs)
        idx = jnp.nonzero(mask, size=w["K"], fill_value=0)[0]
        ok = mask[idx]
        z0, a0 = encode(p, obs[idx]), obs[idx, 1] * 3.0
        mu0 = prior_plan(p, prior, z0, fb[idx], w["HP"])
        mu, _, _ = plan(p, z0, mu0, key, w, a0=a0)
        sc, _ = score(p, z0, jnp.stack([mu0, mu], 1), w, a0, mu0)  # [K, 2]: the prior's plan, the planned one
        take = ok & (sc[:, 1] > sc[:, 0] + w["gate"])
        # only ever more cautious than the prior: brake at least as hard, demand at least as large a gap (u1 scales the gap the
        # AV needs at junctions and in merges)
        safe = jnp.stack([jnp.minimum(mu[:, 0, 0], fb[idx, 0]), jnp.maximum(mu[:, 0, 1], fb[idx, 1])], -1)
        u = fb.at[idx].set(jnp.where(take[:, None], safe, fb[idx]))
        return u, take.sum()[None].astype(jnp.float32) * jnp.ones((obs.shape[0], 2)) / jnp.maximum(ok.sum(), 1)
    return policy


class Viz:
    """wmplan.viz's view of a model and a planner (M.load / encode / decode, PL.make / plan / score / HP), for the v2 GIFs."""
    HP = PLAN["HP"]
    load, encode, decode = staticmethod(load), staticmethod(encode), staticmethod(decode)

    @staticmethod
    def make(p, prior):
        return make(p, prior)

    @staticmethod
    def plan(p, z0, mu0, key):
        return plan(p, z0, mu0, key)

    @staticmethod
    def score(p, z0, us):
        return score(p, z0, us)


def gifs():
    from wmplan import viz as V
    load0 = P.load
    V.P.load = lambda path: load0(PRIOR if "xnet3" in path else path)  # the v2 prior
    V.M, V.PL = Viz, Viz
    try:
        net, geo, info, frames, trs, wm, ppo = V.record()
        for kind, title in (("junction", "Latent planner v2 at a junction: every candidate future, unfolded and decoded"),
                            ("merge", "Latent planner v2 in a merge: every candidate future, unfolded and decoded")):
            sel = V.pick(frames, trs, kind, len(net["link_len"]))
            if sel is None:
                print("no clip for", kind)
                continue
            V.gif(net, geo, info, frames, trs, wm, ppo, *sel, f"docs/img/wm2_planner_{kind}.gif", title, kind)
    finally:
        V.P.load = load0
    busy_gif()


def busy_gif(scenario="red_runners", seed=21, out="docs/img/wm2_planner_busy_junction.gif"):
    """The planner where it has the most to get right: of the junction entries with a tight time-to-conflict margin, the one with
    the most road users within 35 m of the AV as it commits, in AVBench's red-runner traffic."""
    from avrl import bench as B
    from wmplan import viz as V
    from xnet import render as R
    load0, world0 = P.load, V.E.world
    V.P.load = lambda path: load0(PRIOR if "xnet3" in path else path)
    V.M, V.PL = Viz, Viz
    V.E.world = lambda: B.behaviour(B.SCENARIOS[scenario][1])
    try:
        net, geo, info, frames, trs, wm, ppo = V.record(seed)
        L, T, clip = len(net["link_len"]), len(trs), V.CLIP
        trk = np.stack([f.trk for f in frames])
        valid = np.stack([t["valid"] for t in trs])
        marg = np.stack([t["margins"] for t in trs])
        obs = np.stack([t["obs"] for t in trs])
        t0s = np.stack([f.t0 for f in frames])
        cand = []
        for slot in np.nonzero(valid.any(0))[0]:
            for c in range(clip // 2, T - clip // 2, 5):
                a, b = c - clip // 2, c + clip // 2
                if not valid[a:b, slot].all() or (t0s[a:b, slot] != t0s[a, slot]).any() or obs[a:b, slot, 0].mean() * 15 < 3.0:
                    continue
                tr_ = trk[c - 5:c + 15, slot]
                if (tr_[:5] < 2 * L).all() and (tr_ >= 2 * L).any():  # on the approach, enters the junction soon after
                    cand.append((float(-marg[c - 10:c, slot, 2].min()), int(slot), a, b, c))
        if not cand:
            print("no busy junction clip")
            return
        tracks = R.Tracks(net, geo)
        best, best_s = None, -1e9
        for tight, slot, a, b, c in sorted(cand, reverse=True)[:60]:
            f = frames[c]
            idx = np.nonzero(np.asarray(f.alive))[0]
            x, y, _ = tracks.pose(np.asarray(f.trk)[idx], np.asarray(f.s)[idx], np.asarray(f.length)[idx])
            me = int(np.nonzero(idx == slot)[0][0])
            near = int((np.hypot(x - x[me], y - y[me]) < 35.0).sum()) - 1
            if near + 2.0 * tight > best_s:
                best, best_s = (slot, a, b, near), near + 2.0 * tight
        slot, a, b, near = best
        print(f"busy junction clip: AV slot {slot}, {near} road users within 35 m", flush=True)
        V.gif(net, geo, info, frames, trs, wm, ppo, slot, a, b, out,
              f"Latent planner v2 at a busy junction ({near} road users within 35 m, red-light runners about): every candidate future",
              "junction")
    finally:
        V.P.load, V.E.world = load0, world0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        train(steps=int(sys.argv[2]) if len(sys.argv) > 2 else 15000)
    elif cmd == "gifs":
        gifs()
    elif cmd == "busy":
        busy_gif()
