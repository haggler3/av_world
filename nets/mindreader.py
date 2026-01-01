"""N1: the mind reader. Infer a driver's hidden traits, and whether they are about to run a red light, from 10 s of their driving.

A real dataset never says who is a red-light runner, how close a driver likes to follow or how hard they are willing to brake; the
simulator draws all of it per driver and keeps it in the state. Recorded every 0.4 s for every human car in 16 copies of 5x-risk
Oakland (8 min each): speed, acceleration, gap and closing speed to the car ahead, distance to the stop line, the light ahead, the
speed limit, lane changes. Targets, at the end of each 10 s window:

  runner    will run a red that has just turned (p = 0.26 per driver in this world)           binary
  aggr      aggressive: no zipper courtesy, clips cyclists (p = 0.2)                          binary
  T, amax, b, v0f, s0   IDM headway, comfortable acceleration and braking, desired-speed factor, standstill gap    regression
  red_soon  crosses a stop line on red within the next 3 s                                   binary

Network: three causal dilated 1-D convolutions over the 25 steps (kernel 5, dilation 1, 2, 4), mean and last-step pooling, one MLP head
per target. Baseline: a linear / logistic model on 14 hand features of the same window. Split by copy.

    python -m nets.mindreader               # ~8 min -> results/nets/mindreader.json
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
from xnet import sim as X  # noqa: E402

from . import collect as C  # noqa: E402
from . import common as CM  # noqa: E402

OUT = "results/nets"
W, EVERY, FUT = 25, 2, 8          # 10 s windows of 0.4 s records; the hazard looks 8 records (3.2 s) ahead
REG = ("T", "amax", "b", "v0f", "s0")
CH = ("v", "acc", "gap", "vl", "dist", "sig", "vlim")


def make_feat(J):
    L = J["link_len"].shape[0]

    def feat1(s):
        N = s.v.shape[0]
        key = jnp.where(s.alive, s.trk * X.SCALE + jnp.clip((s.s * 8).astype(jnp.int32), 0, X.SCALE - 1), jnp.iinfo(jnp.int32).max)
        order = jnp.argsort(key)
        rank = jnp.zeros(N, jnp.int32).at[order].set(jnp.arange(N))
        nxt = order[jnp.minimum(rank + 1, N - 1)]
        same = s.alive & (rank < N - 1) & s.alive[nxt] & (s.trk[nxt] == s.trk)
        gap = jnp.where(same, s.s[nxt] - s.length[nxt] - s.s, 150.0)
        vl = jnp.where(same, s.v[nxt], s.v)
        on_link = s.trk < 2 * L
        link = jnp.where(on_link, s.trk // 2, 0)
        sig = jnp.where(on_link, X.signal_state(J, s.t)[link], 2)
        dist = jnp.where(on_link, J["link_len"][link] - s.s, 150.0)
        human = s.alive & ~s.av & (s.kind == X.CAR)
        f16 = lambda x: x.astype(jnp.float16)
        return {"v": f16(s.v), "acc": f16(s.acc), "gap": f16(jnp.minimum(gap, 150.0)), "vl": f16(vl), "dist": f16(jnp.clip(dist, -5.0, 150.0)),
                "sig": sig.astype(jnp.int8), "vlim": f16(jnp.where(on_link, J["link_v"][link], 8.0)), "trk": s.trk, "uid": s.uid, "human": human,
                "ran_red": s.ran_red, "runner": s.runner, "aggr": s.aggr, "T": f16(s.T), "amax": f16(s.amax), "b": f16(s.b), "v0f": f16(s.v0f), "s0": f16(s.s0)}

    def feat(s, tr, pc):  # s holds all copies: reduce each one
        return jax.vmap(feat1)(s)
    return feat


def collect(n_envs=16, minutes=8, seed=1):
    w = C.World()
    beh = AD.world(5.0)._replace(crn=1.0)
    st = C.run(w, w.reset(n_envs, seed), beh, 120.0)
    feat = make_feat(w.J)
    chunks = []
    t0 = time.time()
    for m in range(minutes):
        st, f = C.record(w, st, beh, 60.0, EVERY, feat)
        chunks.append(C.to_host(f))
        print(f"minute {m + 1}: {time.time() - t0:.0f}s", flush=True)
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}         # [T, E, N]


def windows(rec, stride=3):
    """Per (copy, vehicle) track: windows of W records of one human car, the traits at the window's end, and the hazard label."""
    T, E, N = rec["v"].shape
    feats, labels, envs = [], [], []
    for e in range(E):
        uid, hum = rec["uid"][:, e], rec["human"][:, e]
        for n in range(N):
            u, h = uid[:, n], hum[:, n]
            # runs of the same vehicle in this slot
            brk = np.concatenate([[True], (u[1:] != u[:-1]) | ~h[1:] | ~h[:-1]])
            starts = np.where(brk)[0]
            ends = np.concatenate([starts[1:], [T]])
            for a, b in zip(starts, ends):
                if not h[a] or b - a < W + FUT:
                    continue
                sl = slice(a, b)
                x = np.stack([rec[c][sl, e, n].astype(np.float32) for c in CH], 1)
                trk = rec["trk"][sl, e, n]
                lc = np.concatenate([[0], (trk[1:] != trk[:-1]) & (trk[1:] // 2 == trk[:-1] // 2)]).astype(np.float32)   # lane change on a link
                rr = rec["ran_red"][sl, e, n]
                cross_red = np.concatenate([[False], rr[1:] & ~rr[:-1]])
                for j in range(W - 1, b - a - FUT, stride):
                    feats.append(np.concatenate([x[j - W + 1:j + 1], lc[j - W + 1:j + 1, None]], 1))
                    labels.append([float(rec["runner"][a + j, e, n]), float(rec["aggr"][a + j, e, n]), *[float(rec[k][a + j, e, n]) for k in REG],
                                   float(cross_red[j + 1:j + 1 + FUT].any())])
                    envs.append(e)
    return np.asarray(feats, np.float32), np.asarray(labels, np.float32), np.asarray(envs)


def normalise(x):
    """Raw window channels (v, acc, gap, vl, dist, sig, vlim, lane change) -> network input [B, W, 12]."""
    v, acc, gap, vl, dist, sig, vlim, lc = (x[..., i] for i in range(8))
    has = (gap < 149.0).astype(np.float32)
    near = (dist < 100.0).astype(np.float32)
    return np.stack([v / 15.0, acc / 3.0, np.minimum(gap, 60.0) / 30.0 * has, has, (v - vl) / 10.0 * has, np.clip(dist, 0, 150) / 100.0,
                     (sig == 0) * near, (sig == 1) * near, (sig == 2) * near, v / np.maximum(vlim, 1.0), lc, (v < 0.5).astype(np.float32)], -1).astype(np.float32)


def hand_features(x):
    """14 summary features of the same window for the linear baseline."""
    v, acc, gap, vl, dist, sig, vlim = (x[..., i] for i in range(7))
    fol = (gap < 60.0) & (v > 3.0)
    thw = np.where(fol, gap / np.maximum(v, 1.0), np.nan)
    ratio = v / np.maximum(vlim, 1.0)
    with np.errstate(all="ignore"):
        f = [np.nanmean(ratio, 1), np.nanmax(ratio, 1), acc.min(1), acc.max(1), np.nanmean(np.where(fol, acc, np.nan), 1),
             np.nanmedian(thw, 1), np.nanmin(thw, 1), (v < 0.5).mean(1), fol.mean(1), v[:, -1] / 15.0, np.clip(dist[:, -1], 0, 150) / 100.0,
             (sig[:, -1] == 0) * 1.0, (sig[:, -1] == 1) * 1.0, (v[:, -1] ** 2 / np.maximum(2 * dist[:, -1], 1.0)) / 8.0]
    f = np.nan_to_num(np.stack(f, 1), nan=0.0, posinf=0.0, neginf=0.0)
    return f.astype(np.float32)


# ---------------------------------------------------------------- the network
def init(key, cin=12, ch=64, n_bin=3, n_reg=len(REG)):
    ks = jax.random.split(key, 6)
    conv = lambda k, i, o: {"w": jax.random.normal(k, (5, i, o)) * np.sqrt(2 / (5 * i)), "b": jnp.zeros(o)}
    return {"c": [conv(ks[0], cin, ch), conv(ks[1], ch, ch), conv(ks[2], ch, ch)],
            "head": CM.init_mlp(ks[3], [2 * ch, 128, n_bin + n_reg])}


def apply(p, x):
    """Causal dilated convolutions written as shifted copies + one matmul per layer (XLA's CPU convolution backward pass is ~100x slower)."""
    h = x
    for l, dil in zip(p["c"], (1, 2, 4)):
        T = h.shape[1]
        hp = jnp.pad(h, ((0, 0), (4 * dil, 0), (0, 0)))                     # causal: only the past
        taps = jnp.concatenate([hp[:, j * dil:j * dil + T] for j in range(5)], -1)   # [B, T, 5 * c_in]
        y = taps @ l["w"].reshape(-1, l["w"].shape[-1]) + l["b"]
        h = jax.nn.gelu(y) + (h if h.shape[-1] == y.shape[-1] else 0.0)
    z = jnp.concatenate([h.mean(1), h[:, -1]], -1)
    return CM.mlp(p["head"], z)


BIN = (0, 1, 7)  # runner, aggr, red_soon in the label vector
REGI = (2, 3, 4, 5, 6)


def loss(p, b, apply_fn):
    y = apply_fn(p, b["x"])
    lb = optax.sigmoid_binary_cross_entropy(y[:, :3], b["yb"]) * b["wb"]
    lr = (y[:, 3:] - b["yr"]) ** 2
    return lb.mean() + 0.5 * lr.mean()


def main(stride=9):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cache = f"{OUT}/mindreader_windows.npz"
    if os.path.exists(cache):
        z = np.load(cache)
        x, y, env = z["x"].astype(np.float32), z["y"], z["env"]
    else:
        rec = collect()
        x, y, env = windows(rec, stride)
        del rec
        np.savez(cache, x=x.astype(np.float16), y=y, env=env)
    print(f"{len(x)} windows from {len(np.unique(env))} copies ({time.time() - t0:.0f}s)", flush=True)
    tr, te = env < 12, env >= 12
    yb, yr = y[:, BIN], y[:, REGI]
    mu, sd = yr[tr].mean(0), yr[tr].std(0) + 1e-6
    yrn = (yr - mu) / sd
    # up-weight the rare hazard label so it is not ignored
    wb = np.ones_like(yb)
    wb[:, 2] = np.where(yb[:, 2] > 0.5, min(20.0, 0.5 / max(yb[tr, 2].mean(), 1e-4)), 1.0)
    xn = normalise(x)
    p = init(jax.random.PRNGKey(0))
    p, curve = CM.fit(lambda p, b: loss(p, b, apply), p, {"x": xn[tr], "yb": yb[tr], "yr": yrn[tr], "wb": wb[tr]}, steps=4000, batch=1024)
    out = np.concatenate([np.asarray(jax.jit(apply)(p, jnp.asarray(xn[te][i:i + 8192]))) for i in range(0, te.sum(), 8192)])
    hf = hand_features(x)
    hmu, hsd = hf[tr].mean(0), hf[tr].std(0) + 1e-6
    hfn = (hf - hmu) / hsd
    lin = [{"w": jnp.zeros((hf.shape[1], 3 + len(REGI))), "b": jnp.zeros(3 + len(REGI))}]
    lin, _ = CM.fit(lambda p, b: loss(p, b, lambda q, xx: xx @ q[0]["w"] + q[0]["b"]), lin, {"x": hfn[tr], "yb": yb[tr], "yr": yrn[tr], "wb": wb[tr]}, steps=3000, batch=2048)
    lout = np.asarray(hfn[te] @ np.asarray(lin[0]["w"]) + np.asarray(lin[0]["b"]))
    res = {"n_windows": int(len(x)), "n_train": int(tr.sum()), "n_test": int(te.sum()), "base_rates": yb[te].mean(0).tolist(), "targets": {}}
    names = ("runner", "aggr", "red_soon")
    for i, nm in enumerate(names):
        res["targets"][nm] = {"net_auroc": CM.auroc(out[:, i], yb[te, i]), "linear_auroc": CM.auroc(lout[:, i], yb[te, i]), "base_rate": float(yb[te, i].mean())}
    for j, nm in enumerate(REG):
        t = yrn[te, j]
        r2 = lambda pr: float(1 - ((t - pr) ** 2).sum() / ((t - t.mean()) ** 2).sum())
        res["targets"][nm] = {"net_r2": r2(out[:, 3 + j]), "linear_r2": r2(lout[:, 3 + j])}
    for k, v in res["targets"].items():
        print(k, {a: round(b, 3) for a, b in v.items()}, flush=True)
    res["curve"] = curve[::40].tolist()
    with open(f"{OUT}/mindreader.json", "w") as f:
        json.dump(res, f)
    np.savez_compressed(f"{OUT}/mindreader_preds.npz", net=out, lin=lout, yb=yb[te], yr=yrn[te], x_last=x[te][:, -1])
    import pickle
    with open(f"{OUT}/mindreader_params.pkl", "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, p), f)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
