"""Rare crash probabilities in minutes: interacting particles that follow a learned committor.

Plain Monte Carlo needs about 100 / p independent runs to see a crash of probability p ten times, which is days of
simulation for p = 1e-5. Splitting runs N copies of the city in parallel and, every 3 s, clones the copies that are heading
for trouble and drops the ones that are not. Done carelessly it is biased or noisy (see README, "negative result"); done as a
Feynman-Kac particle system it is unbiased for any choice of the guide, and low variance when the guide is the committor.

  event A     an AV is in a crash at some time in the next T seconds (K windows of 3 s), from a warmed-up state
  guide V(x)  log of the probability that A happens within the next H windows, built from a per-AV committor: every AV
              gets p_i from its own 28-feature observation (gap, closing speed, time to the junction conflict, signal ...),
              its barrier margin and the scene around it (red-runners, distracted drivers, stalled cars, pedestrians), and the
              copy gets 1 - prod(1 - p_i). The p_i come from a small network trained on a pilot run in a more dangerous world
              (so that crashes are common) and are checked by AUC on copies held out of the fit.
  weights     after each window every copy gets G = exp(lam (V_new - V_old)); copies are resampled in proportion to G
              (systematic resampling); a copy that has crashed has V = 0 (probability 1).
  estimate    p = prod_k mean(G_k) * mean_i [ G_K 1_A exp(lam V_0) ] over the final copies: the product of the average weights is the
              estimate of the normalising constant, and the telescoping of V makes it an unbiased estimate of P(A) for ANY V.

    python -m avlab.rare pilot            # ~3 min: learn the guide        -> results/rare/guide.json
    python -m avlab.rare validate         # ~10 min: plain vs particles at equal cost, a regime where plain can be checked
    python -m avlab.rare rare             # ~10 min: a regime where plain cannot, with its confidence interval
"""
import json
import os
import sys
import time

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xnet import network as NW  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import adversary as AD  # noqa: E402

OUT = "results/rare"
WINDOW, N_SLOTS = 15, 1024  # 3 s
H = 5  # the guide predicts a crash within the next H windows (15 s)

K_AV = 192  # AV rows kept per copy (an area holds ~100 AVs at 25 %); the overflow is counted
NAV = 28 + 3 + 4  # observation, barrier margin and the two vetoes, scene context


class Rare:
    """N copies of an area under one behaviour, run in windows of 3 s that also report every AV's risk features."""

    def __init__(self, beh, shield="idm", n=128, area="oakland", seed=0):
        self.n, self.shield, self.seed = n, shield, seed
        net_np, _, _ = NW.build(area)
        self.J = J = X.to_jnp(net_np)
        nh = jnp.asarray(net_np["nh"])

        def one(s, k):
            return X.step(s, J, beh, nh)

        def window_one(st, key):
            w0 = st.wreck_t > 0
            m0 = st.m
            ks = jax.random.split(key, WINDOW)
            st = jax.lax.scan(lambda s, k: (one(s, k), None), st, ks[:-1])[0]
            st, d = X.step(st, J, beh, nh, diag=True)
            av = st.alive & st.av
            idx = jnp.nonzero(av, size=K_AV, fill_value=0)[0]
            ok = jnp.arange(K_AV) < av.sum()
            cnt = lambda m: jnp.log1p(m.sum().astype(jnp.float32))  # noqa: E731
            ctx = jnp.stack([cnt(st.alive & st.ran_red), cnt(st.alive & (st.dist_t > 0)), cnt(st.alive & ((st.stop_t > 0) | (st.wreck_t > 0))),
                             cnt(st.ped_t > 0)])
            feat = jnp.concatenate([d["obs"][idx], jnp.clip(d["bar_h"][idx], -10.0, 40.0)[:, None] / 10.0,
                                    d["bar_veto_j"][idx][:, None].astype(jnp.float32), d["bar_veto_m"][idx][:, None].astype(jnp.float32),
                                    jnp.broadcast_to(ctx, (K_AV, 4))], 1)
            live = ok & (st.wreck_t[idx] <= 0)  # an AV that is still driving
            new = ok & (st.wreck_t[idx] > 0) & ~w0[idx]  # crashed in this window
            return st, (st.m - m0)[X.METRICS.index("av_crash")], feat, live, new, idx, st.t0[idx], jnp.maximum(av.sum() - K_AV, 0)

        self._win = jax.jit(jax.vmap(window_one))
        self._reset = jax.jit(jax.vmap(lambda k: X.reset(k, J, N_SLOTS)))
        self._run = jax.jit(jax.vmap(lambda st, key: jax.lax.scan(lambda s, k: (one(s, k), None), st, jax.random.split(key, 100))[0]))

    def window(self, st, key):
        """-> state, AV crashes [n], and numpy: feat [n, K, NAV], live, new, slot, t0, overflow."""
        X.SHIELD = self.shield  # read when traced
        out = self._win(st, jax.random.split(key, self.n))
        return (out[0],) + tuple(np.asarray(o) for o in out[1:])

    def init(self, warm_s=200.0, seed=None):
        X.SHIELD = self.shield
        key = jax.random.PRNGKey(self.seed if seed is None else seed)
        st = self._reset(jax.random.split(key, self.n))
        for i in range(int(warm_s / (X.DT * 100))):
            st = self._run(st, jax.random.split(jax.random.fold_in(key, 1000 + i), self.n))
        return st


# ----------------------------------------------------------------------------------------------------------- the guide
def fit_logistic(Xm, y, l2=3.0, iters=40):
    Xb = np.concatenate([np.ones((len(Xm), 1)), Xm], 1)
    b = np.zeros(Xb.shape[1])
    pen = l2 * np.eye(len(b))
    pen[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ b, -30, 30)))
        g = Xb.T @ (p - y) + pen @ b
        Hs = (Xb * (p * (1 - p))[:, None]).T @ Xb + pen + 1e-6 * np.eye(len(b))
        step = np.linalg.solve(Hs, g)
        b -= step
        if np.abs(step).max() < 1e-6:
            break
    return {"kind": "linear", "w": b[1:].tolist(), "b0": float(b[0])}


def fit_mlp(Z, y, hidden=16, steps=2000, l2=1e-2, seed=0):
    """A one-hidden-layer network (tanh) for the committor; Adam on the cross-entropy, weights back as lists."""
    import optax
    k = jax.random.split(jax.random.PRNGKey(seed), 2)
    p0 = {"w1": jax.random.normal(k[0], (Z.shape[1], hidden)) / np.sqrt(Z.shape[1]), "b1": jnp.zeros(hidden),
          "w2": jax.random.normal(k[1], (hidden,)) / np.sqrt(hidden), "b2": jnp.full((), np.log(y.mean() / (1 - y.mean())))}
    Zj, yj = jnp.asarray(Z, jnp.float32), jnp.asarray(y, jnp.float32)

    def loss(p):
        lp = jnp.tanh(Zj @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]
        ce = jnp.mean(jnp.logaddexp(0.0, lp) - yj * lp)
        return ce + l2 * (jnp.sum(p["w1"] ** 2) + jnp.sum(p["w2"] ** 2))

    opt = optax.adam(1e-2)

    @jax.jit
    def step(p, s):
        g = jax.grad(loss)(p)
        u, s = opt.update(g, s)
        return optax.apply_updates(p, u), s
    p, s = p0, opt.init(p0)
    for _ in range(steps):
        p, s = step(p, s)
    return {"kind": "mlp", **{k_: np.asarray(v).tolist() for k_, v in p.items()}}


def logit(m, Z):
    if m["kind"] == "linear":
        return Z @ np.asarray(m["w"]) + m["b0"]
    return np.tanh(Z @ np.asarray(m["w1"]) + np.asarray(m["b1"])) @ np.asarray(m["w2"]) + m["b2"]


def auc(score, y):
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


class Guide:
    """p_i = P(AV i crashes within the next H windows) from its features; V(copy) = log(1 - prod(1 - p_i)), floored at 1e-5."""

    def __init__(self, d):
        self.mean, self.sd, self.model = np.asarray(d["mean"]), np.asarray(d["sd"]), d["model"]
        self.auc = d.get("auc")

    def p(self, feat):
        return 1.0 / (1.0 + np.exp(-np.clip(logit(self.model, (feat - self.mean) / self.sd), -30, 30)))

    def V(self, feat, live):
        """feat [n, K, NAV], live [n, K] -> [n]"""
        n, k, f = feat.shape
        p = self.p(feat.reshape(-1, f)).reshape(n, k) * live
        return np.clip(np.log(np.maximum(-np.expm1(np.log1p(-np.minimum(p, 0.999)).sum(1)), 1e-30)), np.log(1e-5), 0.0)


def pilot(beh=None, shield="idm", n=128, windows=100, seed=900, out=f"{OUT}/guide.json"):
    """Run a dangerous world, label every AV in every window with 'crashes within the next H windows', fit the guide."""
    # crashes must be common in the pilot: risky humans, stalled cars, dart-outs and AV sensor faults (any world works as a guide)
    beh = beh or AD.world(15.0, breakdown_rate=1e-4, p_ped_violate=0.15, p_dart=0.3, p_blind=2e-4, p_ghost=1e-3)
    r = Rare(beh, shield, n, seed=seed)
    st = r.init(200.0)
    F, LV, NW_, SL, T0, CR, OV = ([] for _ in range(7))
    for k in range(windows):
        st, c, f, live, new, slot, t0, ov = r.window(st, jax.random.PRNGKey(seed * 1000 + k))
        for lst, v in zip((F, LV, NW_, SL, T0, CR, OV), (f, live, new, slot, t0, c, ov)):
            lst.append(v)
    F, LV, NW_, SL, T0, CR = (np.stack(a) for a in (F, LV, NW_, SL, T0, CR))  # [windows, n, ...]
    y = np.zeros(LV.shape, bool)  # [windows, n, K]: this AV crashes within the next H windows
    for j, c, i in zip(*np.nonzero(NW_)):
        for k in range(max(j - H, 0), j):
            y[k, c] |= LV[k, c] & (SL[k, c] == SL[j, c, i]) & (T0[k, c] == T0[j, c, i])
    cop = np.zeros(CR.shape, bool)  # [windows, n]: some AV crashes within the next H windows
    for k in range(windows - H):
        cop[k] = (CR[k + 1:k + 1 + H] > 0).any(0)
    keep = LV & (np.arange(windows)[:, None, None] < windows - H)
    cid = np.broadcast_to(np.arange(n)[None, :, None], LV.shape)
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/pilot.npz", F=F[keep].astype(np.float16), y=y[keep], cid=cid[keep], cop=cop, copF=F, copLV=LV, overflow=int(np.sum(OV)))
    print(f"pilot: {int(keep.sum())} AV-windows, {int(y[keep].sum())} followed by that AV's crash within {H * WINDOW * X.DT:.0f} s "
          f"({int(np.sum(NW_))} AV crashes, {int(np.sum(OV))} AV rows over the cap)", flush=True)
    return fit_guide(F[keep].astype(np.float32), y[keep].astype(float), cid[keep], out, cop=(cop, F, LV))


def fit_guide(F, y, cid, out=f"{OUT}/guide.json", cop=None):
    """Pick the best of a linear and a small neural committor by AUC on copies held out of the fit, refit on everything. Negatives are
    subsampled 10:1 (and the intercept corrected back)."""
    rng = np.random.default_rng(0)
    q = min(1.0, 20.0 * y.sum() / max((1 - y).sum(), 1))
    sub = (y > 0) | (rng.random(len(y)) < q)
    mean, sd_ = F.mean(0), F.std(0) + 1e-6
    Z = (F - mean) / sd_
    test = (cid % 4 == 0)  # held out by copy
    cands = {"linear": lambda Zt, yt: fit_logistic(Zt, yt, 30.0), "mlp16": lambda Zt, yt: fit_mlp(Zt, yt, 16),
             "mlp32": lambda Zt, yt: fit_mlp(Zt, yt, 32, l2=3e-3)}
    scores, models = {}, {}
    for name, f in cands.items():
        m = f(Z[~test & sub], y[~test & sub])
        scores[name], models[name] = auc(logit(m, Z[test]), y[test]), m
        print(f"  {name}: held-out AV-level AUC {scores[name]:.3f}", flush=True)
    best = max(scores, key=scores.get)
    model = cands[best](Z[sub], y[sub])
    model["b0" if model["kind"] == "linear" else "b2"] += float(np.log(q))  # undo the negative subsampling
    g = {"mean": mean.tolist(), "sd": sd_.tolist(), "model": model, "model_name": best, "auc": scores[best], "auc_by_model": scores,
         "av_windows": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()), "features": [f"obs{i}" for i in range(28)] + ["barrier margin", "veto junction", "veto merge", "red-runners", "distracted", "stalled", "pedestrians"]}
    if cop is not None:  # does the copy-level V rank the copies that crash soon?
        cp, Fw, LVw = cop
        gd = Guide(g)
        Vs, ys = [], []
        for k in range(0, Fw.shape[0] - H, 3):
            te = np.arange(Fw.shape[1]) % 4 == 0
            Vs.append(gd.V(Fw[k][te].astype(np.float32), LVw[k][te]))
            ys.append(cp[k][te])
        g["auc_copy_level"] = auc(np.concatenate(Vs), np.concatenate(ys).astype(float))
        print(f"  copy-level AUC of V (copies held out): {g['auc_copy_level']:.3f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(g, open(out, "w"), indent=1)
    print(f"guide: {best}, held-out AV-level AUC {scores[best]:.3f}", flush=True)
    return Guide(g)


# --------------------------------------------------------------------------------------------------- the estimators
def systematic(p, rng):
    cs = np.cumsum(p)
    cs[-1] = 1.0
    return np.searchsorted(cs, (rng.random() + np.arange(len(p))) / len(p))


def run(r, st0, guide, K=20, lam=1.0, seed=0, method="particles"):
    """One estimate of P(AV crash within K windows) from the warm states st0. -> dict(p, ess_min, crashed_copies)."""
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(10_000 + seed)
    n = r.n
    # the simulator draws from the key inside the state: give every replicate its own random numbers from the same warm states
    st0 = st0._replace(key=jax.vmap(jax.random.fold_in)(st0.key, jnp.arange(n) + 7919 * (seed + 1)))
    st, c0, f0, l0, *_ = r.window(st0, jax.random.fold_in(key, 999))  # x_0: the warm state, one window on
    hit = c0 > 0
    V0 = V_prev = np.where(hit, 0.0, guide.V(f0, l0)) if method == "particles" else None
    logZ, ess = 0.0, []
    for k in range(K):
        st, c, f, lv, *_ = r.window(st, jax.random.fold_in(key, k))
        hit = hit | (c > 0)
        if method == "plain":
            continue
        V = np.where(hit, 0.0, guide.V(f, lv))
        G = np.exp(lam * (V - V_prev))
        ess.append(float(G.sum() ** 2 / (G ** 2).sum()))
        if k < K - 1:
            logZ += np.log(G.mean())
            idx = systematic(G / G.sum(), rng)
            st = jax.tree.map(lambda a: a[idx], st)
            st = st._replace(key=jax.vmap(jax.random.fold_in)(st.key, jnp.arange(n) + 1000 * (k + 1)))
            hit, V_prev, V0 = hit[idx], V[idx], V0[idx]
        else:
            return {"p": float(np.exp(logZ) * np.mean(G * hit * np.exp(lam * V0))), "ess_min": float(min(ess)), "crashed_copies": int(hit.sum())}
    return {"p": float(hit.mean()), "crashed_copies": int(hit.sum())}


def compare(beh, shield, guide, replicates=12, n=128, K=20, lam=1.0, tag="", seed=1):
    """Plain vs particles at equal cost (n copies x K windows each), from the same warm states and the same replicate seeds."""
    r = Rare(beh, shield, n, seed=seed)
    res = {"plain": [], "particles": []}
    t0 = time.time()
    st = r.init(200.0, seed=seed * 100)  # one set of warm states (different traffic in every copy); the replicates differ in their random numbers
    for i in range(replicates):
        for m in res:
            res[m].append(run(r, st, guide, K, lam, seed=i, method=m))
        print(f"{tag} replicate {i}: plain {res['plain'][-1]['p']:.2e} ({res['plain'][-1]['crashed_copies']} crashed)  "
              f"particles {res['particles'][-1]['p']:.2e} (min ess {res['particles'][-1]['ess_min']:.0f})  {time.time() - t0:.0f} s", flush=True)
    out = {"n": n, "K": K, "seconds_per_run": K * WINDOW * X.DT, "lam": lam, "shield": shield, "replicates": replicates,
           "seconds_wall": time.time() - t0}
    for m in res:
        p = np.array([x["p"] for x in res[m]])
        out[m] = {"mean": float(p.mean()), "sd": float(p.std(ddof=1)), "se": float(p.std(ddof=1) / np.sqrt(len(p))), "values": p.tolist(),
                  "zero_runs": int((p == 0).sum())}
    out["variance_ratio"] = (out["plain"]["sd"] / max(out["particles"]["sd"], 1e-300)) ** 2
    return out


def validate():
    guide = Guide(json.load(open(f"{OUT}/guide.json")))
    beh = AD.world(5.0)  # the ladder's risky world; rule-based AV under the IDM envelope
    res = compare(beh, "idm", guide, replicates=12, tag="validate")
    json.dump(res, open(f"{OUT}/validate.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("plain", "particles", "variance_ratio")}, default=float), flush=True)


def rare():
    guide = Guide(json.load(open(f"{OUT}/guide.json")))
    beh = AD.world(1.0)  # calibrated humans; every AV under the control barrier function
    res = compare(beh, "cbf", guide, replicates=12, tag="rare")
    json.dump(res, open(f"{OUT}/rare.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("plain", "particles", "variance_ratio")}, default=float), flush=True)


if __name__ == "__main__":
    {"pilot": pilot, "validate": validate, "rare": rare}[sys.argv[1]]()
