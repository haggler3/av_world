"""Rare crash probabilities in minutes: interacting particles that follow a learned committor.

Plain Monte Carlo needs about 100 / p independent runs to see a crash of probability p ten times, which is days of
simulation for p = 1e-5. Splitting runs N copies of the city in parallel and, every 3 s, clones the copies that are heading
for trouble and drops the ones that are not. Done carelessly it is biased or noisy (see README, "negative result"); done as a
Feynman-Kac particle system it is unbiased for any choice of the guide, and low variance when the guide is the committor.

  event A     an AV is in a crash at some time in the next T seconds (K windows of 3 s), from a warmed-up state
  guide V(x)  log of the probability of A within the next H windows, predicted from 12 features of the whole scene (how
              close the AVs are to their barrier, red-runners and distracted drivers on the road, stalled cars, AVs in
              junctions, pedestrians crossing ...). A logistic regression with pairwise interactions, learned from a pilot
              run in a more dangerous world (so that crashes are common) and checked by held-out AUC.
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
FEATURES = ("closest AV to its barrier (m/10)", "AVs within 2 m of the barrier", "... 5 m", "... 10 m", "AVs the barrier would hold at a junction",
            "red-runners on the road", "distracted drivers", "stalled cars and wrecks", "AVs inside junctions", "AVs (log)",
            "pedestrians crossing", "share of AV time inside the barrier")
NF = len(FEATURES)


class Rare:
    """N copies of an area under one behaviour, run in windows of 3 s that also report the 12 scene features."""

    def __init__(self, beh, shield="idm", n=128, area="oakland", seed=0):
        self.n, self.shield, self.seed = n, shield, seed
        net_np, _, _ = NW.build(area)
        self.J = J = X.to_jnp(net_np)
        nh = jnp.asarray(net_np["nh"])
        L = net_np["link_len"].shape[0]

        def one(s, k):
            return X.step(s, J, beh, nh)

        def feats(st, d, dm):
            av = st.alive & st.av & (st.wreck_t <= 0)
            moving = av & (st.v > 1.0)
            h = jnp.clip(d["bar_h"], -10.0, 40.0)
            hmin = jnp.where(moving, h, 40.0).min()
            cnt = lambda m: jnp.log1p(m.sum().astype(jnp.float32))  # noqa: E731
            share = dm[X.METRICS.index("av_h_neg")] / jnp.maximum(dm[X.METRICS.index("av_veh_s")], 1.0)
            return jnp.stack([hmin / 10.0, cnt(moving & (h < 2.0)), cnt(moving & (h < 5.0)), cnt(moving & (h < 10.0)),
                              cnt(d["bar_veto_j"] & av), cnt(st.alive & st.ran_red), cnt(st.alive & (st.dist_t > 0)),
                              cnt(st.alive & ((st.stop_t > 0) | (st.wreck_t > 0))), cnt(av & (st.trk >= 2 * L)), cnt(av),
                              cnt(st.ped_t > 0), jnp.log1p(100.0 * share)])

        def window_one(st, key):
            m0 = st.m
            ks = jax.random.split(key, WINDOW)
            st = jax.lax.scan(lambda s, k: (one(s, k), None), st, ks[:-1])[0]
            st, d = X.step(st, J, beh, nh, diag=True)
            dm = st.m - m0
            return st, dm[X.METRICS.index("av_crash")], feats(st, d, dm)

        self._win = jax.jit(jax.vmap(window_one))
        self._reset = jax.jit(jax.vmap(lambda k: X.reset(k, J, N_SLOTS)))
        self._run = jax.jit(jax.vmap(lambda st, key: jax.lax.scan(lambda s, k: (one(s, k), None), st, jax.random.split(key, 100))[0]))

    def window(self, st, key):
        X.SHIELD = self.shield  # read when traced
        st, c, f = self._win(st, jax.random.split(key, self.n))
        return st, np.asarray(c), np.asarray(f)

    def init(self, warm_s=200.0, seed=None):
        X.SHIELD = self.shield
        key = jax.random.PRNGKey(self.seed if seed is None else seed)
        st = self._reset(jax.random.split(key, self.n))
        for i in range(int(warm_s / (X.DT * 100))):
            st = self._run(st, jax.random.split(jax.random.fold_in(key, 1000 + i), self.n))
        return st


# ----------------------------------------------------------------------------------------------------------- the guide
def design(F):
    """Features -> regressors: the 12 features and their pairwise products (a logistic regression with interactions)."""
    i, j = np.triu_indices(F.shape[1], 1)
    return np.concatenate([F, F[:, i] * F[:, j]], 1)


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
    return b


def auc(score, y):
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = y.sum(), (1 - y).sum()
    return float((r[y > 0].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


class Guide:
    def __init__(self, d):
        self.mean, self.sd, self.b = (np.asarray(d[k]) for k in ("mean", "sd", "b"))
        self.auc = d.get("auc")

    def V(self, F):
        """log P(crash within the next H windows | scene), floored at 1e-5."""
        z = design((F - self.mean[:NF]) / self.sd[:NF])
        z = (z - self.mean[NF:]) / self.sd[NF:] if len(self.mean) > NF else z
        lp = np.concatenate([np.ones((len(z), 1)), z], 1) @ self.b
        return np.clip(-np.logaddexp(0.0, -lp), np.log(1e-5), 0.0)  # log sigmoid


def pilot(beh=None, shield="idm", n=64, windows=100, seeds=(900,), out=f"{OUT}/guide.json"):
    """Run a dangerous world, label every window with 'a crash in the next H windows', fit the guide, report held-out AUC."""
    beh = beh or AD.world(15.0, breakdown_rate=1e-4, p_ped_violate=0.15)
    rows, labels, copy_id = [], [], []
    for sd in seeds:
        r = Rare(beh, shield, n, seed=sd)
        st = r.init(200.0)
        F, C = [], []
        for k in range(windows):
            st, c, f = r.window(st, jax.random.PRNGKey(sd * 1000 + k))
            F.append(f)
            C.append(c)
        F, C = np.stack(F), np.stack(C)  # [windows, n, ...]
        ever = np.cumsum(C > 0, 0) > 0
        for k in range(windows - H):
            fut = (C[k + 1:k + 1 + H] > 0).any(0)
            keep = ~ever[k]  # copies that have not crashed yet
            rows.append(F[k][keep])
            labels.append(fut[keep].astype(float))
            copy_id.append(np.nonzero(keep)[0] + n * sd)
    F, y, cid = np.concatenate(rows), np.concatenate(labels), np.concatenate(copy_id)
    print(f"pilot: {len(y)} windows, {int(y.sum())} followed by a crash within {H * WINDOW * X.DT:.0f} s", flush=True)
    mean, sd_ = F.mean(0), F.std(0) + 1e-6
    Z = design((F - mean) / sd_)
    zm, zs = Z.mean(0), Z.std(0) + 1e-6
    Z = (Z - zm) / zs
    test = (cid % 4 == 0)  # held out by copy
    b = fit_logistic(Z[~test], y[~test])
    s_te = np.concatenate([np.ones((test.sum(), 1)), Z[test]], 1) @ b
    a_te = auc(s_te, y[test])
    b = fit_logistic(Z, y)
    s_tr = np.concatenate([np.ones((len(Z), 1)), Z], 1) @ b
    # single features, for comparison: how well does the closest barrier margin alone predict?
    a_single = {FEATURES[i]: auc(F[test, i] * (-1 if i == 0 else 1), y[test]) for i in range(NF)}
    g = {"mean": np.concatenate([mean, zm]).tolist(), "sd": np.concatenate([sd_, zs]).tolist(), "b": b.tolist(), "auc": a_te,
         "auc_single": a_single, "windows": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()), "features": FEATURES}
    os.makedirs(OUT, exist_ok=True)
    json.dump(g, open(out, "w"), indent=1)
    print(f"held-out AUC {a_te:.3f} (in-sample {auc(s_tr, y):.3f}); best single feature "
          f"{max(a_single, key=a_single.get)} {max(a_single.values()):.3f}", flush=True)
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
    st, c0, f0 = r.window(st0, jax.random.fold_in(key, 999))  # x_0: the warm state, one window on
    hit = c0 > 0
    V0 = V_prev = np.where(hit, 0.0, guide.V(f0)) if method == "particles" else None
    logZ, ess = 0.0, []
    for k in range(K):
        st, c, f = r.window(st, jax.random.fold_in(key, k))
        hit = hit | (c > 0)
        if method == "plain":
            continue
        V = np.where(hit, 0.0, guide.V(f))
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
    """Plain vs particles at equal cost (n copies x K windows each), from the same warm states."""
    r = Rare(beh, shield, n, seed=seed)
    res = {"plain": [], "particles": []}
    t0 = time.time()
    for i in range(replicates):
        st = r.init(200.0, seed=seed * 100 + i)
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
