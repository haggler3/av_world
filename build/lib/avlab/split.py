"""Estimating an AV's crash rate with far fewer AV-hours: importance splitting on danger.

The obvious way to see rare crashes sooner is to raise the human risk rates and reweight by the likelihood ratio.
In a whole city it fails: one 5-minute run draws thousands of independent Bernoulli decisions (is this driver a red
runner, does that one look), and the likelihood ratio is their product. Tilting p_red from 0.05 to 0.10 over the ~3,000
drivers of one run gives log-weights with a standard deviation near 9, so one copy carries all the weight
(`tilt_ess`, below, computes it from the measured number of spawned drivers).

Splitting avoids the products. Every `WINDOW` steps (3 s; a crash is preceded by seconds, not minutes, of a scene that
looks dangerous), each copy of the area gets an importance g from what it
looks like now (the latent world model's crash probability summed over its AVs, how many moving AVs are near their
barrier, how long they were inside it, near misses), the copies are
resampled in proportion to weight x g (systematic resampling) whenever the copies' g differ enough to matter, and a copy
that is cloned k times carries 1/k of the weight. Clones get new random numbers, so their futures differ. For any g the weighted sum of crashes is unbiased:

    resampling picks copy i  N_i  times with E[N_i] = E pi_i,  pi_i = w_i g_i / sum_j w_j g_j
    each offspring of i gets   w'_i = w_i / (E pi_i) = mean_j(w_j g_j) / g_i

so E[sum_k w'_k f(X'_k)] = sum_i w_i f(X_i). A good g just puts the copies where crashes are about to happen: here
g is the fitted expected number of crashes in the next window (a Poisson regression on a pilot run).

    python -m avlab.split pilot results/xnet3/policy.npz oakland              # fit g (~4 min)
    python -m avlab.split validate results/unaided/phase2/policy.npz oakland  # plain vs splitting, 12 replicates (~8 min)
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from avrl import ladder as LD
from avrl import xppo as P
from wmplan import model as WM
from wmplan import planner as PL
from xnet import mutate as MU
from xnet import network as NW
from xnet import sim as X

from . import analyze as AN

OUT = "results/avlab/split"
IMP = "results/avlab/split_importance.json"
IDX = {k: i for i, k in enumerate(X.METRICS)}
WINDOW = 15  # steps: 3 s
WARM_STEPS, WINDOWS, E, N_SLOTS = 1200, 100, 16, 1024  # 240 s warm-up, 300 s per run
RESAMPLE_IF = 1.5  # resample when the largest g exceeds this many times the mean
SHARPEN = 1.0  # g = exp(SHARPEN * fitted log intensity): >1 sends more clones to the copies that look most dangerous
G_CLIP = (0.25, 4.0)
FEATURES = ("moving AVs within 2 m of the barrier", "... within 5 m", "... within 10 m", "AV time inside the barrier",
            "near misses", "crashes", "world-model crash risk, summed over AVs", "... of the riskiest AV")


def build_world(spec, n_envs=E, seed0=0):
    """-> (net, behaviour [E], per-copy cities or None). spec: oakland | squirrel_hill | downtown | mutated (held-out
    mutated Oakland cities, a different set for every seed0)."""
    beh = jax.tree.map(lambda x: jnp.full((n_envs,), x, jnp.float32), LD.test_world())
    if spec == "mutated":
        net, geo, _ = NW.build("oakland")
        return net, beh, MU.worlds(net, geo, [MU.HELD_OUT + 100 * seed0 + i for i in range(n_envs)])
    return NW.build(spec)[0], beh, None


class Runner:
    """E copies of one area (or E different mutated cities) in windows of WINDOW steps, driven by a policy's params
    (None: the rule-based AV), with a safety filter ("off", "idm" or "cbf")."""

    def __init__(self, net, beh, world, shield="off"):
        self.E = int(beh.demand.shape[0])
        self.shield = shield
        J, nh0, n_envs = X.to_jnp(net), jnp.asarray(net["nh"]), self.E
        wm = WM.load()
        ax = 0 if world is not None else None
        nw, hw = (world["net"], world["nh"]) if world is not None else (None, None)

        def stp(params, info):
            pol = None if params is None else P.deterministic(params)

            def f(s, b, k, n, h):
                nn = J if n is None else {**J, **n}
                hh = nh0 if h is None else h
                if pol is None:
                    return X.step(s, nn, b, hh, diag=True) if info else X.step(s, nn, b, hh)
                o, d = X.step(s, nn, b, hh, rl=(pol, k))
                return (o, d) if info else o
            return lambda st, keys: jax.vmap(f, in_axes=(0, 0, 0, ax, ax))(st, beh, keys, nw, hw)

        def window(params, st, key, n):
            m0 = st.m
            ks = jax.random.split(key, n)
            st, _ = jax.lax.scan(lambda s, k: (stp(params, False)(s, jax.random.split(k, n_envs)), None), st, ks[:-1])
            st, info = stp(params, True)(st, jax.random.split(ks[-1], n_envs))
            h = info["bar_h"]
            moving = st.alive & st.av & (st.v > 1.0)
            tight = jnp.stack([(moving & (h < r)).sum(1) for r in (2.0, 5.0, 10.0)], 1).astype(jnp.float32)
            if params is None:
                risk = jnp.zeros((n_envs, 2))
            else:  # the latent model's probability of a crash on the next step, from each AV's scene and chosen action
                z = WM.encode(wm, info["obs"])
                a = WM.act_in(info["u"])
                pc = jax.nn.sigmoid(WM.run(wm["trans"], jnp.concatenate([z, a], -1))[..., 1] - jnp.log(PL.POS_W)) * info["valid"]
                risk = jnp.stack([pc.sum(1), pc.max(1)], 1)
            return st, st.m - m0, jnp.concatenate([tight, risk], 1)

        self._window, self.J = jax.jit(window, static_argnums=3), J

    def window(self, params, st, key, n=WINDOW):
        X.SHIELD = self.shield  # read when the window is traced (its first call)
        try:
            return self._window(params, st, key, n)
        finally:
            X.SHIELD = "idm"

    def init(self, params, seed):
        keys = jax.random.split(jax.random.PRNGKey(seed), self.E)
        st = jax.vmap(lambda k: X.reset(k, self.J, N_SLOTS))(keys)
        for i in range(WARM_STEPS // 100):
            st, _, _ = self.window(params, st, jax.random.fold_in(jax.random.PRNGKey(seed + 1), i), 100)
        return st


def features(d, tight):
    """[E, 8] features of each copy after a window (see FEATURES)."""
    share = d[:, IDX["av_h_neg"]] / np.maximum(d[:, IDX["av_veh_s"]], 1.0)
    return np.stack([np.log1p(tight[:, 0]), np.log1p(tight[:, 1]), np.log1p(tight[:, 2]), np.log1p(100 * share),
                     (d[:, IDX["av_near"]] > 0).astype(float), (d[:, IDX["av_crash"]] > 0).astype(float),
                     np.log1p(100 * tight[:, 3]), np.log1p(100 * tight[:, 4])], 1)


def importance(f, imp):
    """g per copy: the fitted expected crashes in the next window, normalised to mean 1 and clipped to [1/4, 4]."""
    if imp is None:
        g = np.exp(1.0 * (f[:, 6] - f[:, 6].mean()))
    else:
        g = np.exp(SHARPEN * ((f - np.asarray(imp["mean"])) / np.asarray(imp["sd"])) @ np.asarray(imp["coef"]))
    return np.clip(g / g.mean(), *G_CLIP)


def systematic(p, rng):
    cs = np.cumsum(p)
    cs[-1] = 1.0
    return np.searchsorted(cs, (rng.random() + np.arange(len(p))) / len(p))


def estimate_once(runner, params, st0, method, seed, imp=None, windows=WINDOWS):
    """One run from the warm state st0: {"sum": weighted metric sums, "ess": per window, "raw_crashes": unweighted}."""
    st, w = st0, np.ones(runner.E)
    acc, raw, ess, resampled = np.zeros(X.NM), np.zeros(X.NM), [], 0
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    for k in range(windows):
        st, d, tight = runner.window(params, st, jax.random.fold_in(key, k))
        d, tight = np.asarray(d, float), np.asarray(tight, float)
        acc += (w[:, None] * d).sum(0)
        raw += d.sum(0)
        ess.append(float(w.sum() ** 2 / (w ** 2).sum()))
        g = importance(features(d, tight), imp) if method == "split" else None
        if method == "split" and k < windows - 1 and g.max() > RESAMPLE_IF:
            resampled += 1
            p = w * g
            idx = systematic(p / p.sum(), rng)
            w = p.mean() / g[idx]  # w'_i = mean_j(w_j g_j) / g_i
            st = jax.tree.map(lambda a: a[idx], st)
            st = st._replace(key=jax.vmap(jax.random.fold_in)(st.key, jnp.arange(runner.E) + 1000 * (k + 1)))
    return {"sum": acc, "raw": raw, "ess": ess, "resampled": resampled}


def rates(s):
    """Metric sums -> the usual rates."""
    d = dict(zip(X.METRICS, s))
    h = max(d["av_veh_s"] / 3600, 1e-9)
    out = {"crash_per_h": d["av_crash"] / h, "speed_kmh": 3.6 * d["av_veh_m"] / max(d["av_veh_s"], 1.0), "hard_per_h": d["av_hard"] / h,
           "near_per_h": d["av_near"] / h, "red_per_h": d["av_red"] / h, "av_hours": h}
    out["types_per_h"] = {k[3:]: d[k] / h for k in X.AV_CRASH}
    return out


_CACHE = {}  # (spec, replicate, shield) -> (runner, warm state): compiled once, and every policy starts from the same traffic


def estimate(params, spec, replicates=3, method="split", imp=None, shield="off"):
    """Crash rate (and speed, hard braking) of a policy in `spec`, with a 95 % interval from the replicates. Every policy
    is run from the same warm states (warmed up by the rule-based AV) and the same random numbers, so comparisons are paired."""
    imp = imp if imp is not None else (json.load(open(IMP)) if os.path.exists(IMP) else None)
    runs = []
    for r in range(replicates):
        if (spec, r, shield) not in _CACHE:
            net, beh, world = build_world(spec, seed0=r)
            runner = Runner(net, beh, world, shield)
            _CACHE[(spec, r, shield)] = (runner, runner.init(None, 100 + r))
        runner, st = _CACHE[(spec, r, shield)]
        runs.append(rates(estimate_once(runner, params, st, method, 200 + r, imp)["sum"]))
    out = {"replicates": replicates, "method": method}
    for k in ("crash_per_h", "speed_kmh", "hard_per_h", "near_per_h"):
        v = np.array([x[k] for x in runs])
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        out[k] = [float(v.mean()), float(v.mean() - 2 * se), float(v.mean() + 2 * se)]
    out["av_hours"] = float(np.mean([x["av_hours"] for x in runs]))
    return out


def tilt_ess(spawned_per_run, p=0.0514, factor=2.0):
    """Effective sample size (as a fraction of the copies) of tilting the red-runner probability p by `factor` over
    `spawned_per_run` independent drivers: the log likelihood ratio has variance n p (1-p) (ln(p'/p) - ln((1-p')/(1-p)))^2."""
    q = min(p * factor, 0.99)
    var = spawned_per_run * p * (1 - p) * (np.log(q / p) - np.log((1 - q) / (1 - p))) ** 2
    return {"factor": factor, "log_weight_sd": float(np.sqrt(var)), "ess_fraction": float(np.exp(-var))}


def pilot(policy, spec, replicates=6):
    """Fit g: Poisson regression of the crashes in the next window on the features after this one."""
    params = P.load(policy) if policy != "rule" else None
    X_, y, ex = [], [], []
    for r in range(replicates):
        net, beh, world = build_world(spec, seed0=900 + r)
        runner = Runner(net, beh, world)
        st = runner.init(params, 900 + r)
        prev, key = None, jax.random.PRNGKey(900 + r)
        for k in range(WINDOWS):
            st, d, tight = runner.window(params, st, jax.random.fold_in(key, k))
            d, tight = np.asarray(d, float), np.asarray(tight, float)
            if prev is not None:
                X_.append(prev)
                y.append(d[:, IDX["av_crash"]])
                ex.append(d[:, IDX["av_veh_s"]] / 3600)
            prev = features(d, tight)
    Xa, ya, ea = np.concatenate(X_), np.concatenate(y), np.concatenate(ex)
    mean, sd = Xa.mean(0), Xa.std(0) + 1e-6
    b, se = AN.glm(ya, ea, (Xa - mean) / sd)
    coef = np.clip(b[1:], -1.5, 1.5)
    imp = {"mean": mean.tolist(), "sd": sd.tolist(), "coef": coef.tolist(), "features": FEATURES, "crashes": float(ya.sum()),
           "windows": int(len(ya)), "z": (b[1:] / se[1:]).tolist()}
    os.makedirs(os.path.dirname(IMP), exist_ok=True)
    json.dump(imp, open(IMP, "w"), indent=1)
    print(json.dumps({k: (np.round(v, 3).tolist() if isinstance(v, list) else v) for k, v in imp.items()}), flush=True)


def validate(policy, spec, replicates=12):
    """Plain Monte Carlo vs splitting at equal simulation cost: the spread of the estimates over replicates."""
    params = P.load(policy) if policy != "rule" else None
    imp = json.load(open(IMP)) if os.path.exists(IMP) else None
    res = {"plain": [], "split": []}
    ess = []
    t0 = time.time()
    spawned = []
    for r in range(replicates):
        net, beh, world = build_world(spec, seed0=r)
        runner = Runner(net, beh, world)
        st = runner.init(params, 100 + r)
        for m in ("plain", "split"):
            o = estimate_once(runner, params, st, m, 200 + r, imp)
            res[m].append(rates(o["sum"]))
            if m == "split":
                ess.append(float(np.mean(o["ess"])))
            else:
                spawned.append(float(o["raw"][IDX["spawned"]]) / runner.E)
        print(f"replicate {r}: plain {res['plain'][-1]['crash_per_h']:.4f}  split {res['split'][-1]['crash_per_h']:.4f}  "
              f"{time.time() - t0:.0f} s", flush=True)
    out = {"policy": policy, "spec": spec, "replicates": replicates, "copies": E, "seconds_per_copy": WINDOWS * WINDOW * X.DT,
           "mean_ess_fraction_split": float(np.mean(ess) / E), "tilt_ess": tilt_ess(float(np.mean(spawned)))}
    for m in res:
        for k in ("crash_per_h", "speed_kmh", "hard_per_h"):
            v = np.array([x[k] for x in res[m]])
            out[f"{m}_{k}"] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "values": v.tolist()}
    sp, sl = out["plain_crash_per_h"]["sd"], out["split_crash_per_h"]["sd"]
    out["variance_ratio"] = (sp / max(sl, 1e-12)) ** 2
    os.makedirs(OUT, exist_ok=True)
    tag = os.path.basename(os.path.dirname(policy)) + "_" + spec
    json.dump(out, open(f"{OUT}/validate_{tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or k == "tilt_ess"}, default=float), flush=True)
    for k in ("plain_crash_per_h", "split_crash_per_h"):
        print(k, {a: round(b, 4) for a, b in out[k].items() if a != "values"}, flush=True)


def tune(policy, spec, replicates=8):
    """Splitting at several sharpness settings against plain Monte Carlo, from the same warm states (fresh seeds)."""
    global SHARPEN, G_CLIP
    params = P.load(policy) if policy != "rule" else None
    imp = json.load(open(IMP))
    settings = [("plain", 1.0, (0.25, 4.0)), ("split x1", 1.0, (0.25, 4.0)), ("split x3", 3.0, (0.1, 8.0)), ("split x6", 6.0, (0.05, 16.0))]
    out = {n: [] for n, _, _ in settings}
    for r in range(replicates):
        net, beh, world = build_world(spec, seed0=500 + r)
        runner = Runner(net, beh, world)
        st = runner.init(params, 500 + r)
        for n, sh, clip in settings:
            SHARPEN, G_CLIP = sh, clip
            o = estimate_once(runner, params, st, "plain" if n == "plain" else "split", 600 + r, imp)
            c = rates(o["sum"])["crash_per_h"]
            out[n].append((c, float(o["raw"][IDX["av_crash"]]), float(np.mean(o["ess"])) / runner.E, o["resampled"]))
        print(f"replicate {r}: " + "  ".join(f"{n} {out[n][-1][0]:.4f} (raw {out[n][-1][1]:.0f}, ess {out[n][-1][2]:.2f})" for n in out), flush=True)
    for n in out:
        v = np.array([x[0] for x in out[n]])
        print(f"{n:9s} mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  raw crashes/run {np.mean([x[1] for x in out[n]]):.1f}  "
              f"ess {np.mean([x[2] for x in out[n]]):.2f}  resamples {np.mean([x[3] for x in out[n]]):.0f}", flush=True)
    SHARPEN, G_CLIP = 1.0, (0.25, 4.0)
    json.dump(out, open(f"{OUT}/tune_{spec}.json", "w"), indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "tune":
        tune(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8)
    if cmd == "pilot":
        pilot(sys.argv[2], sys.argv[3])
    elif cmd == "validate":
        validate(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12)
