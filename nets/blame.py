"""N5: but-for blame. Given a crash, which driver caused it?

The simulator can answer exactly. When two vehicles crash, rewind the city 6-8 s and replay it once per nearby driver with that driver's
mistakes switched off (red-light running, not looking, aggressive gap acceptance, signal ignoring, a distraction in progress), every
other driver keeping its dice (identity-keyed CRN). A driver is a but-for cause if the crash does not happen in its branch. Several
drivers can be causes; a crash may have none (it came from ordinary car-following and the geometry).

A network then learns to name the causes from what an investigator could see in the seconds before the crash: each candidate's speed
and acceleration, gaps, whether it was one of the two vehicles that collided, whether it crossed a line on red, its lane changes, its
distance to the crash point. The hidden mistakes themselves are not inputs. Set attention over the candidates, a logit per candidate.
Baselines: the faster of the two vehicles that collided ("the striker"), the vehicle that crossed on red, and (privileged) the
candidate that had a mistake switched on.

    python -m nets.blame [minutes]          # ~10 min -> results/nets/blame.json
"""
import json
import os
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from avlab import adversary as AD  # noqa: E402
from xnet import render as R  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import collect as C  # noqa: E402
from . import common as CM  # noqa: E402

OUT = "results/nets"
RISK = 10.0
from avlab.paired import STRESS  # noqa: E402  the stress world of avlab.paired: crashes are frequent enough to study one by one
CHUNK = 2.0          # seconds between snapshots
REWIND = 4           # snapshots kept: the fork starts 6-8 s before the crash
M = 8                # candidates per crash
BATCH = 36           # copies per branch batch (a multiple of 4 devices): 2 crashes x (1 base + 8 calmed + 8 careful) + 2 spare
WIN = 10.0           # seconds replayed per branch
F = 18               # features per candidate (see run_branches)


def new_wreck(s):
    return (s.wreck_t == 300.0) | (s.wreck_t == 120.0)          # the wreck timer is set to exactly this on the crash step


def feat_detect(s, tr, pc):
    return {"new": new_wreck(s)}


def feat_branch(s, tr, pc):
    return {"new": new_wreck(s), "uid": s.uid, "v": s.v.astype(jnp.float16), "acc": s.acc.astype(jnp.float16), "trk": s.trk,
            "s": s.s.astype(jnp.float16), "ran_red": s.ran_red}


def calm(st, slot):
    """This driver's mistakes off from now on (its dice and everything else unchanged). Works on numpy or jax state."""
    def off(a, v):
        a = np.array(a, copy=True)
        a[slot] = v
        return a
    return st._replace(runner=off(st.runner, False), nolook=off(st.nolook, False), aggr=off(st.aggr, False),
                       nosignal=off(st.nosignal, False), dist_t=off(st.dist_t, 0.0))


def careful(st, slot):
    """Mistakes off and a careful style: headway at least 1.8 s, standstill gap at least 2.5 m, braking at least 2.5 m/s^2 comfort."""
    st = calm(st, slot)

    def at_least(a, v):
        a = np.array(a, copy=True)
        a[slot] = max(float(a[slot]), v)
        return a
    return st._replace(T=at_least(st.T, 1.8), s0=at_least(st.s0, 2.5), b=at_least(st.b, 2.5))


def take(st, e):
    return jax.tree.map(lambda a: a[e], st)


def collect(minutes=20, n_envs=32, seed=3):
    w = C.World()
    tracks = R.Tracks(w.net_np, w.geo)
    beh = X.Behaviour(**STRESS, p_av=0.25, crn=1.0)
    st = C.run(w, w.reset(n_envs, seed), beh, 120.0)
    snaps = []
    events, pending = [], []
    t0 = time.time()
    n_chunks = int(minutes * 60 / CHUNK)
    for c in range(n_chunks):
        snaps.append(st)
        snaps = snaps[-REWIND:]
        st, f = C.record(w, st, beh, CHUNK, 1, feat_detect)
        new = np.asarray(f["new"])                                   # [steps, E, N]
        for e in range(n_envs):
            for t in np.unique(np.nonzero(new[:, e])[0]):
                slots = np.nonzero(new[t, e])[0]
                if len(slots) >= 2 and len(snaps) == REWIND:         # a crash between vehicles, with a snapshot 6-8 s before it
                    pending.append((e, int(c), int(t), slots[:2], jax.tree.map(np.asarray, take(snaps[0], e))))
        while len(pending) >= BATCH // (2 * M + 1):
            events += run_branches(w, tracks, beh, pending[:BATCH // (2 * M + 1)])
            pending = pending[BATCH // (2 * M + 1):]
        if c % 30 == 0:
            print(f"{c * CHUNK / 60:.0f} min: {len(events)} crashes resolved, {len(pending)} pending ({time.time() - t0:.0f}s)", flush=True)
    return events


def run_branches(w, tracks, beh, batch):
    """For each crash: the snapshot, its base branch and one branch per candidate with that candidate calmed, all in one batch."""
    copies, meta = [], []
    for (e, c, t, slots, snap) in batch:
        h = jax.tree.map(np.asarray, snap)
        uid_c = h.uid[slots]
        alive = h.alive & (h.kind != X.BIKE) | False
        x, y, _ = tracks.pose(h.trk, h.s, h.length)
        inv = np.array([np.nonzero((h.uid == u) & h.alive)[0][0] if ((h.uid == u) & h.alive).any() else -1 for u in uid_c])
        if (inv < 0).any():
            continue                                               # a crasher was not yet on the map at the snapshot
        cx, cy = x[inv].mean(), y[inv].mean()
        d = np.where(h.alive, np.hypot(x - cx, y - cy), np.inf)
        cand = list(inv) + [i for i in np.argsort(d) if i not in inv and np.isfinite(d[i])][:M - 2]
        cand = cand[:M]
        copies.append(snap)
        for i in cand:
            copies.append(calm(snap, int(i)))
        copies += [snap] * (M - len(cand))
        for i in cand:
            copies.append(careful(snap, int(i)))
        copies += [snap] * (M - len(cand))
        meta.append((e, c, t, uid_c, np.array(cand), np.array([h.runner[i] or h.nolook[i] or h.aggr[i] or h.nosignal[i] or h.dist_t[i] > 0 for i in cand]),
                     np.stack([x[cand] - cx, y[cand] - cy], 1), np.array([h.av[i] for i in cand]), np.array([h.kind[i] for i in cand])))
    while len(copies) < BATCH:
        copies.append(copies[0])
    stb = jax.tree.map(lambda *a: jnp.asarray(np.stack([np.asarray(x) for x in a])), *copies)
    _, f = C.record(w, stb, beh, WIN, 1, feat_branch)
    f = C.to_host(f, np.float32)
    out = []
    for j, (e, c, t, uid_c, cand, mistake, rel, av, kind) in enumerate(meta):
        b0 = j * (2 * M + 1)

        def crashed(b):
            u = f["uid"][:, b0 + b]
            hit = f["new"][:, b0 + b] & np.isin(u, uid_c)
            return bool(hit.any())
        base = crashed(0)
        if not base:
            continue                                              # the replay did not reproduce the crash: drop it
        cause = np.array([not crashed(1 + k) for k in range(len(cand))])
        could = np.array([not crashed(1 + M + k) for k in range(len(cand))])          # a careful driver in this seat avoids it
        # what an investigator sees in the base replay, up to the crash: per candidate, from its slot
        tc = int(np.argmax(f["new"][:, b0].any(1)))
        cols = []
        for k, i in enumerate(cand):
            sl = slice(0, max(tc, 1))
            v, a = f["v"][sl, b0, i], f["acc"][sl, b0, i]
            trk = f["trk"][sl, b0, i]
            lc = float(((trk[1:] != trk[:-1]) & (trk[1:] // 2 == trk[:-1] // 2)).any()) if len(trk) > 1 else 0.0
            rr = f["ran_red"][sl, b0, i]
            red_cross = float((rr[1:] & ~rr[:-1]).any()) if len(rr) > 1 else 0.0
            cols.append([float(k < 2), float(av[k]), *(np.eye(4)[min(int(kind[k]), 3)]), rel[k, 0] / 30.0, rel[k, 1] / 30.0,
                         np.hypot(*rel[k]) / 30.0, v.mean() / 15.0, v.max() / 15.0, v[-1] / 15.0, a.min() / 3.0, a.max() / 3.0, a[-5:].mean() / 3.0,
                         red_cross, lc, float(len(v)) / 50.0])
        x = np.zeros((M, F), np.float32)
        x[:len(cand)] = np.asarray(cols, np.float32)
        mask = np.zeros(M, bool)
        mask[:len(cand)] = True
        y = np.zeros(M, np.float32)
        y[:len(cand)] = cause
        mk = np.zeros(M, bool)
        mk[:len(cand)] = mistake
        y2 = np.zeros(M, np.float32)
        y2[:len(cand)] = could
        out.append({"x": x, "mask": mask, "y": y, "y2": y2, "mistake": mk, "env": e, "time": c * CHUNK + t * X.DT})
    return out


# ---------------------------------------------------------------- the set network
def init(key, d_in=None, d=64):
    d_in = F if d_in is None else d_in
    ks = jax.random.split(key, 5)
    return {"emb": CM.init_mlp(ks[0], [d_in, d]), "qkv": CM.init_mlp(ks[1], [d, 3 * d]), "ff": CM.init_mlp(ks[2], [2 * d, d, d]),
            "out": CM.init_mlp(ks[3], [d, 64, 1])}


def apply(p, x, mask):
    h = CM.mlp(p["emb"], x)                                   # [B, M, d]
    q, k, v = jnp.split(CM.mlp(p["qkv"], h), 3, -1)
    att = q @ jnp.swapaxes(k, 1, 2) / np.sqrt(q.shape[-1])
    att = jnp.where(mask[:, None, :], att, -1e9)
    ctx = jax.nn.softmax(att, -1) @ v
    h = h + CM.mlp(p["ff"], jnp.concatenate([h, ctx], -1))
    return CM.mlp(p["out"], h)[..., 0]


def loss(p, b):
    y = apply(p, b["x"], b["mask"])
    l = optax.sigmoid_binary_cross_entropy(y, b["y"]) * b["mask"]
    return l.sum() / b["mask"].sum()


def evaluate(score, y, mask, any_cause):
    """Top-1: the highest-scored candidate is a but-for cause (crashes with at least one cause). AUROC over all real candidates."""
    s = np.where(mask, score, -np.inf)
    top = s.argmax(1)
    t1 = y[np.arange(len(y)), top][any_cause]
    return {"top1": CM.boot(t1), "auroc": CM.auroc(score[mask], y[mask].astype(bool))}


def main(minutes=20, out_dir=OUT, quick=False):
    os.makedirs(out_dir, exist_ok=True)
    minutes, steps = (3, 800) if quick else (minutes, 3000)
    t0 = time.time()
    cache = f"{out_dir}/blame_data.npz"
    if os.path.exists(cache) and "y2" in np.load(cache).files:
        z = np.load(cache)
        x, mask, ys, mk, env = z["x"], z["mask"], {"mistake": z["y"], "careful": z["y2"]}, z["mistake"], z["env"]
    else:
        ev = collect(minutes)
        x = np.stack([e["x"] for e in ev]); mask = np.stack([e["mask"] for e in ev]); mk = np.stack([e["mistake"] for e in ev])
        ys = {"mistake": np.stack([e["y"] for e in ev]), "careful": np.stack([e["y2"] for e in ev])}
        env = np.array([e["env"] for e in ev])
        np.savez_compressed(cache, x=x, mask=mask, y=ys["mistake"], y2=ys["careful"], mistake=mk, env=env)
    res = {"n_crashes": int(len(x)), "questions": {}}
    te = env >= CM.split_env(env)
    tr = ~te
    for q, y in ys.items():
        any_cause = y.any(1)
        r = {"n_test": int(te.sum()), "share_with_cause": float(any_cause.mean()), "causes_per_crash": float(y.sum(1).mean()),
             "involved_is_cause": float(y[:, :2][mask[:, :2]].mean()), "bystander_is_cause": float(y[:, 2:][mask[:, 2:]].mean()),
             "test_crashes_with_cause": int(any_cause[te].sum())}
        print(f"{q}: {len(x)} crashes, {any_cause.mean():.3f} with a cause, {r['involved_is_cause']:.3f} of the collided pair and "
              f"{r['bystander_is_cause']:.3f} of bystanders are causes", flush=True)
        p = init(jax.random.PRNGKey(0))
        p, curve = CM.fit(loss, p, {"x": x[tr], "mask": mask[tr], "y": y[tr]}, steps=steps, batch=256)
        CM.save_params(f"{out_dir}/blame_{q}_params.pkl", p)
        score = np.asarray(apply(p, jnp.asarray(x[te]), jnp.asarray(mask[te])))
        xt, yt, mt, ac = x[te], y[te], mask[te], any_cause[te]
        striker = np.where(np.arange(M)[None] < 2, xt[..., 11] * 15.0, -1.0)            # final speed of the two that collided
        red = xt[..., 15] + 0.01 * (np.arange(M)[None] < 2)                               # crossed on red (ties to the collided pair)
        priv = mk[te].astype(float) + 0.01 * (np.arange(M)[None] < 2)
        r["methods"] = {"network": evaluate(score, yt, mt, ac), "striker (faster of the two)": evaluate(striker, yt, mt, ac),
                        "crossed on red": evaluate(red, yt, mt, ac), "privileged: had a mistake on": evaluate(priv, yt, mt, ac)}
        for k, v in r["methods"].items():
            print(" ", k, {a: (np.round(b, 3).tolist() if isinstance(b, tuple) else round(b, 3)) for a, b in v.items()}, flush=True)
        r["curve"] = curve[::30].tolist()
        res["questions"][q] = r
    with open(f"{out_dir}/blame.json", "w") as f:
        json.dump(res, f)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:2]))
