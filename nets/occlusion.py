"""N6: seeing through the bus. Where are the road users the AV cannot see?

In the simulator every vehicle's position is known, so for each AV we can compute exactly which nearby vehicles and cyclists are hidden
behind others (line of sight from the AV to a vehicle's centre blocked by another vehicle's footprint, 40 m range). The network sees only
what the AV sees (the visible road users, in the AV's frame: position, heading, speed, length, kind) and predicts an 8 x 8 grid over
80 x 80 m of where the hidden ones are, plus one number: is someone hidden within 25 m ahead of the AV, inside its lane corridor.

Baselines: the marginal frequency of each cell, and a geometric one (the cell's frequency given whether it lies in the shadow of a
visible vehicle).

    python -m nets.occlusion              # ~8 min -> results/nets/occlusion.json
"""
import json
import os
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
RANGE, G, KMAX, WID = 40.0, 8, 32, 2.0


def feat(s, tr, pc):
    return {"trk": s.trk, "s": s.s.astype(jnp.float32), "length": s.length.astype(jnp.float16), "kind": s.kind.astype(jnp.int8),
            "alive": s.alive, "v": s.v.astype(jnp.float16), "av": s.av}


def seg_hits_box(p0, p1, cx, cy, hd, L, W):
    """Does the segment p0 -> p1 (arrays [..., 2]) cross the rotated rectangle(s) (centre, heading, length, width)? Slab test in the box frame."""
    c, s = np.cos(hd), np.sin(hd)
    def to_box(p):
        dx, dy = p[..., 0] - cx, p[..., 1] - cy
        return dx * c + dy * s, -dx * s + dy * c
    ax, ay = to_box(p0)
    bx, by = to_box(p1)
    dx, dy = bx - ax, by - ay
    t0, t1 = np.zeros_like(ax), np.ones_like(ax)
    for a, d, h in ((ax, dx, L / 2), (ay, dy, W / 2)):
        with np.errstate(divide="ignore", invalid="ignore"):
            ta, tb = (-h - a) / d, (h - a) / d
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        par = np.abs(d) < 1e-9
        inside = np.abs(a) <= h
        lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
        hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
        t0, t1 = np.maximum(t0, lo), np.minimum(t1, hi)
    return t0 < t1


def visibility(ex, ey, x, y, hd, L):
    """Ego at (ex, ey); agents [A]. -> visible [A]: no other agent's box crosses the sight line to the agent's centre."""
    A = len(x)
    p0 = np.broadcast_to(np.array([ex, ey]), (A, A, 2))
    p1 = np.broadcast_to(np.stack([x, y], 1)[:, None, :], (A, A, 2))          # [target, occluder]
    hit = seg_hits_box(p0, p1, x[None, :], y[None, :], hd[None, :], L[None, :], np.full((1, A), WID))
    hit[np.arange(A), np.arange(A)] = False                                     # an agent does not hide itself
    return ~hit.any(1)


def samples(rec, tracks):
    T, E, N = rec["s"].shape
    toks, grids, path, shadow, envs = [], [], [], [], []
    for t in range(T):
        for e in range(E):
            al = rec["alive"][t, e]
            idx = np.nonzero(al)[0]
            if len(idx) < 3:
                continue
            x, y, hd = tracks.pose(rec["trk"][t, e, idx], rec["s"][t, e, idx], rec["length"][t, e, idx].astype(float))
            L = rec["length"][t, e, idx].astype(float)
            for a in np.nonzero(rec["av"][t, e, idx])[0]:
                d = np.hypot(x - x[a], y - y[a])
                near = np.nonzero((d < RANGE * 1.42) & (np.arange(len(idx)) != a))[0]
                if len(near) == 0:
                    continue
                vis = visibility(x[a], y[a], x[near], y[near], hd[near], L[near])
                c, s = np.cos(hd[a]), np.sin(hd[a])
                fx = (x[near] - x[a]) * c + (y[near] - y[a]) * s
                fy = -(x[near] - x[a]) * s + (y[near] - y[a]) * c
                inr = (np.abs(fx) < RANGE) & (np.abs(fy) < RANGE)
                v_i = near[vis & inr][:KMAX]
                tok = np.zeros((KMAX, 10), np.float32)
                kind = rec["kind"][t, e, idx][v_i]
                dh = hd[v_i] - hd[a]
                fxv, fyv = fx[vis & inr][:KMAX], fy[vis & inr][:KMAX]
                tok[:len(v_i)] = np.stack([fxv / RANGE, fyv / RANGE, np.cos(dh), np.sin(dh), rec["v"][t, e, idx][v_i].astype(float) / 15.0, L[v_i] / 12.0,
                                           kind == 0, kind == 1, kind == 2, kind == 3], 1)
                hid = ~vis & inr
                gx = np.clip(((fx[hid] + RANGE) / (2 * RANGE) * G).astype(int), 0, G - 1)
                gy = np.clip(((fy[hid] + RANGE) / (2 * RANGE) * G).astype(int), 0, G - 1)
                grid = np.zeros((G, G), np.float32)
                grid[gx, gy] = 1.0
                # cells in the shadow of a visible vehicle (cell centre's sight line crosses a visible box)
                cc = (np.arange(G) + 0.5) / G * 2 * RANGE - RANGE
                CX, CY = np.meshgrid(cc, cc, indexing="ij")
                wx, wy = x[a] + CX * c - CY * s, y[a] + CX * s + CY * c
                P1 = np.stack([wx, wy], -1).reshape(-1, 1, 2)
                P0 = np.broadcast_to(np.array([x[a], y[a]]), P1.shape)
                sh = seg_hits_box(np.broadcast_to(P0, (G * G, len(v_i), 2)), np.broadcast_to(P1, (G * G, len(v_i), 2)),
                                  x[v_i][None], y[v_i][None], hd[v_i][None], L[v_i][None], np.full((1, len(v_i)), WID)).any(1).reshape(G, G) if len(v_i) else np.zeros((G, G), bool)
                toks.append(tok), grids.append(grid), shadow.append(sh)
                path.append(float((hid & (fx > 0) & (fx < 25.0) & (np.abs(fy) < 2.5)).any()))
                envs.append(e)
    return np.stack(toks), np.stack(grids), np.array(path, np.float32), np.stack(shadow), np.array(envs)


def collect(n_envs=16, minutes=6, seed=5):
    w = C.World()
    tracks = R.Tracks(w.net_np, w.geo)
    beh = AD.world(1.0)._replace(crn=1.0)
    st = C.run(w, w.reset(n_envs, seed), beh, 120.0)
    st, f = C.record(w, st, beh, minutes * 60.0, 10, feat)                     # every 2 s
    return samples(C.to_host(f, np.float32), tracks)


def init(key, d=96):
    ks = jax.random.split(key, 4)
    return {"phi": CM.init_mlp(ks[0], [10, d, d]), "rho": CM.init_mlp(ks[1], [2 * d, 256, G * G + 1])}


def apply(p, tok):
    m = (np.abs(tok).sum(-1) > 0)[..., None] if isinstance(tok, np.ndarray) else (jnp.abs(tok).sum(-1) > 0)[..., None]
    h = CM.mlp(p["phi"], tok) * m
    z = jnp.concatenate([h.sum(1) / 10.0, jnp.where(m, h, -1e9).max(1)], -1)
    z = jnp.where(jnp.isfinite(z), z, 0.0)
    return CM.mlp(p["rho"], z)


def loss(p, b):
    y = apply(p, b["tok"])
    return optax.sigmoid_binary_cross_entropy(y[:, :-1], b["grid"]).mean() + optax.sigmoid_binary_cross_entropy(y[:, -1], b["path"]).mean()


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cache = f"{OUT}/occlusion_samples.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        tok, grid, path, shadow, env = z["tok"], z["grid"], z["path"], z["shadow"], z["env"]
    else:
        tok, grid, path, shadow, env = collect()
        np.savez_compressed(cache, tok=tok, grid=grid, path=path, shadow=shadow, env=env)
    print(f"{len(tok)} AV views, hidden-cell rate {grid.mean():.4f}, someone hidden ahead {path.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    tr, te = env < 12, env >= 12
    gflat = grid.reshape(len(grid), -1)
    p = init(jax.random.PRNGKey(0))
    p, curve = CM.fit(loss, p, {"tok": tok[tr], "grid": gflat[tr], "path": path[tr]}, steps=4000, batch=1024)
    out = np.concatenate([np.asarray(jax.jit(apply)(p, jnp.asarray(tok[te][i:i + 8192]))) for i in range(0, te.sum(), 8192)])
    pg = 1 / (1 + np.exp(-out[:, :-1]))
    pp = 1 / (1 + np.exp(-out[:, -1]))
    yt, st_ = gflat[te], shadow.reshape(len(shadow), -1)[te]
    prior = np.broadcast_to(gflat[tr].mean(0), yt.shape)
    sh_tr = shadow.reshape(len(shadow), -1)[tr]
    p_in = (gflat[tr] * sh_tr).sum(0) / np.maximum(sh_tr.sum(0), 1)
    p_out = (gflat[tr] * ~sh_tr).sum(0) / np.maximum((~sh_tr).sum(0), 1)
    geo = np.where(st_, p_in[None], p_out[None])
    brier = lambda q: float(((q - yt) ** 2).mean())
    res = {"n_views": int(len(tok)), "n_test": int(te.sum()), "hidden_cell_rate": float(grid.mean()), "hidden_ahead_rate": float(path.mean()),
           "grid": {"network": {"auroc": CM.auroc(pg.ravel(), yt.ravel()), "brier": brier(pg)},
                    "marginal prior": {"auroc": CM.auroc(prior.ravel(), yt.ravel()), "brier": brier(prior)},
                    "shadow geometry": {"auroc": CM.auroc(geo.ravel(), yt.ravel()), "brier": brier(geo)}},
           "ahead": {"network_auroc": CM.auroc(pp, path[te]), "shadow_ahead_auroc": CM.auroc(st_.reshape(-1, G, G)[:, G // 2:G // 2 + 3, 3:5].mean((1, 2)), path[te])}}
    print(json.dumps(res, indent=1), flush=True)
    res["curve"] = curve[::40].tolist()
    with open(f"{OUT}/occlusion.json", "w") as f:
        json.dump(res, f)
    import pickle
    with open(f"{OUT}/occlusion_params.pkl", "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, p), f)
    k = np.argsort(-path[te])[:400]
    np.savez_compressed(f"{OUT}/occlusion_preds.npz", tok=tok[te][k], grid=yt[k], pg=pg[k], shadow=st_[k], pp=pp[k], path=path[te][k])


if __name__ == "__main__":
    main()
