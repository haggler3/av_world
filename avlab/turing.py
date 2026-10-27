"""The AV Turing test: from how a car moves (speed and acceleration over its trip, nothing else), can a classifier tell an AV from a
human driver? An AUC of 0.5 means the AV drives indistinguishably from people; 1.0 means it is trivially recognisable.

    python -m avlab.turing           # rule-based AV, bundled PPO, IDM: ~6 min on 4 CPU cores -> results/turing/, docs/img/turing.png

Why it matters: human drivers predict each other. An AV that is safe but moves unlike anyone else (brakes earlier, never rolls, keeps
odd gaps) is a surprise to the people around it, and real AV fleets report being rear-ended for exactly that. This measures it.

Per trip (cars only, at least 30 s of driving) eight kinematic features: mean speed, speed SD, acceleration SD, the 5th and 95th
percentile of acceleration, RMS jerk, the share of time stopped, the share of time braking harder than 2 m/s^2. Then a logistic
regression (standardised features, L2) with 5-fold cross-validation, folds by copy so no trip is judged by a model that saw its
own traffic; AUC from the Mann-Whitney statistic and a bootstrap interval over trips. The coefficients say which features give the
AV away.
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from xnet import sim as X

OUT = "results/turing"
FEATURES = ("mean speed", "speed SD", "accel SD", "accel p5", "accel p95", "RMS jerk", "share stopped", "share braking > 2 m/s2")


def record(area="oakland", policy=None, copies=4, warm_s=90.0, seconds=360.0, av_share=0.3, seed=0, chunk_s=30.0):
    """Per step and copy: speed, applied acceleration, alive, AV, car, trip start. Arrays [T, E, N] (float16 where it can be)."""
    from av_world import easy
    net, _, _ = easy._build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = easy.world(av_share)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), copies), J, 1024)
    n = int(round(chunk_s / X.DT))

    def one(s, key):
        if policy is None:
            return X.step(s, J, beh, nh)
        return X.step(s, J, beh, nh, rl=(policy, key))[0]

    def run(st, key, keep):
        def body(s, kk):
            s = jax.vmap(one)(s, jax.random.split(kk, copies))
            rec = (s.v.astype(jnp.float16), s.acc.astype(jnp.float16), s.alive, s.av, s.kind == X.CAR, s.t0) if keep else None
            return s, rec
        return jax.lax.scan(body, st, jax.random.split(key, n))

    run_keep = jax.jit(lambda s, k: run(s, k, True))
    run_skip = jax.jit(lambda s, k: run(s, k, False))
    key = jax.random.PRNGKey(seed + 1)
    for i in range(int(warm_s / chunk_s)):
        st, _ = run_skip(st, jax.random.fold_in(key, i))
    recs = []
    for i in range(int(seconds / chunk_s)):
        st, rec = run_keep(st, jax.random.fold_in(key, 1000 + i))
        recs.append(jax.tree.map(np.asarray, rec))
    return tuple(np.concatenate([r[j] for r in recs]) for j in range(6))


def trips(rec, min_s=30.0):
    """Cut each slot's record into trips (a new trip starts when t0 changes) -> features [n, 8], is_av [n], copy [n]."""
    v, a, alive, av, car, t0 = rec
    T, E, N = v.shape
    out, lab, grp = [], [], []
    for e in range(E):
        for i in range(N):
            al = alive[:, e, i] & car[:, e, i]
            if al.sum() < min_s / X.DT:
                continue
            tt = t0[:, e, i]
            starts = np.nonzero(al & (np.r_[True, tt[1:] != tt[:-1]] | np.r_[True, ~al[:-1]]))[0]
            for s0 in starts:
                seg = np.arange(s0, T)
                seg = seg[np.cumprod(al[seg] & (tt[seg] == tt[s0])).astype(bool)]
                if len(seg) < min_s / X.DT:
                    continue
                vs, acc = v[seg, e, i].astype(float), a[seg, e, i].astype(float)
                jerk = np.diff(acc) / X.DT
                out.append([vs.mean(), vs.std(), acc.std(), np.percentile(acc, 5), np.percentile(acc, 95), np.sqrt((jerk ** 2).mean()),
                            (vs < 0.3).mean(), (acc < -2).mean()])
                lab.append(bool(av[seg[0], e, i]))
                grp.append(e)
    return np.array(out), np.array(lab), np.array(grp)


def auc(score, y):
    """Mann-Whitney AUC."""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def logistic(Xf, y, l2=1.0, iters=200):
    """Newton's method on the L2-penalised log-likelihood. -> weights (with intercept first)."""
    A = np.c_[np.ones(len(Xf)), Xf]
    w = np.zeros(A.shape[1])
    R = l2 * np.eye(A.shape[1])
    R[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + R @ w
        H = A.T @ (A * (p * (1 - p))[:, None]) + R
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def test(F, y, g, n_boot=2000, seed=0):
    """Cross-validated AUC (folds by copy), its bootstrap interval, and the standardised coefficients on all data."""
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Z = (F - mu) / sd
    score = np.zeros(len(y))
    for e in np.unique(g):
        tr, te = g != e, g == e
        w = logistic(Z[tr], y[tr].astype(float))
        score[te] = np.c_[np.ones(te.sum()), Z[te]] @ w
    a = auc(score, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].any() and (~y[i]).any():
            boots.append(auc(score[i], y[i]))
    w = logistic(Z, y.astype(float))
    return {"auc": a, "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))), "n_av": int(y.sum()), "n_human": int((~y).sum()),
            "coef": dict(zip(FEATURES, w[1:].round(3).tolist())),
            "av_mean": dict(zip(FEATURES, F[y].mean(0).round(3).tolist())), "human_mean": dict(zip(FEATURES, F[~y].mean(0).round(3).tolist()))}


def drivers():
    from av_world import easy
    return {"rule-based AV": None, "PPO (bundled)": easy.drivers.ppo(), "IDM (T = 1.5 s)": easy.drivers.idm()}


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
    names = list(res)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 4.2), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.4]})
    for ax in (a0, a1):
        ax.set_facecolor(BG)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
    y = np.arange(len(names))[::-1]
    a0.barh(y, [res[n]["auc"] - 0.5 for n in names], left=0.5, color="#3987e5", height=0.55,
            xerr=[[res[n]["auc"] - res[n]["ci"][0] for n in names], [res[n]["ci"][1] - res[n]["auc"] for n in names]], error_kw={"ecolor": INK2})
    for yy, n in zip(y, names):
        a0.text(res[n]["ci"][1] + 0.01, yy, f"AUC {res[n]['auc']:.2f}", va="center", color=INK, fontsize=9)
    a0.set_yticks(y, names, color=INK, fontsize=9)
    a0.set_xlim(0.45, 1.08)
    a0.axvline(0.5, color=INK2, ls=":")
    a0.text(0.505, y[0] + 0.42, "indistinguishable", color=INK2, fontsize=7.5)
    a0.set_title("Can a classifier tell the AV from people? (AUC, 95 % interval)", color=INK, fontsize=10, loc="left")
    W = np.array([[res[n]["coef"][f] for f in FEATURES] for n in names])
    lim = np.abs(W).max()
    a1.imshow(W, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    a1.set_xticks(range(len(FEATURES)), FEATURES, rotation=25, ha="right")
    a1.set_yticks(range(len(names)), names, color=INK, fontsize=9)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            a1.text(j, i, f"{W[i, j]:+.1f}", ha="center", va="center", color=INK if abs(W[i, j]) > lim / 2 else "#111", fontsize=8)
    a1.set_title("What gives it away (standardised logistic coefficients; red = more AV-like)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def main(quick=False):
    os.makedirs(OUT, exist_ok=True)
    kw = dict(copies=2, warm_s=30.0, seconds=90.0) if quick else {}
    res = {}
    for name, pol in drivers().items():
        t0 = time.time()
        F, y, g = trips(record(policy=pol, **kw))
        if quick and (y.sum() < 2 or (~y).sum() < 2):
            continue
        res[name] = test(F, y, g)
        print(f"{name:18s} AUC {res[name]['auc']:.3f} [{res[name]['ci'][0]:.3f}, {res[name]['ci'][1]:.3f}]  "
              f"({res[name]['n_av']} AV trips, {res[name]['n_human']} human; {time.time() - t0:.0f} s)", flush=True)
    json.dump(res, open(f"{OUT}/{'quick' if quick else 'turing'}.json", "w"), indent=1)
    if not quick:
        plot(res, "docs/img/turing.png")
    return res


def humanize(variants=None, updates=40, out="results/turing"):
    """The fix the test suggests: the bundled policy is recognisable because it is too smooth (low jerk) yet brakes moderately hard
    often. Fine-tune it with reward weights that stop rewarding smoothness (jerk, comfort -> 0), optionally penalising hard braking
    more, then re-run the Turing test and AVBench's nominal scenario on each. -> {variant: {auc, ci, av_score, crash unaided}}."""
    from av_world import data_path
    from avrl import bench
    from avrl import xppo as P
    variants = variants or {"no smoothness reward": {"jerk": 0.0, "comfort": 0.0},
                            "no smoothness reward, 3x hard-brake penalty": {"jerk": 0.0, "comfort": 0.0, "hard": 0.6}}
    res = {}
    for name, rw in variants.items():
        path = f"{out}/humanize_{len(res)}"
        P.train(out=path, updates=updates, init=data_path("policy.npz"), reward=rw, n_envs=8, warm=60.0)
        F, y, g = trips(record(policy=P.deterministic(P.load(f"{path}/policy.npz"))))
        t = test(F, y, g)
        sc = bench.evaluate(f"{path}/policy.npz", ["nominal"])["scenarios"]["nominal"]
        res[name] = {"reward": rw, "auc": t["auc"], "ci": t["ci"], "coef": t["coef"], "av_mean": t["av_mean"], "human_mean": t["human_mean"],
                     "av_score": sc["av_score"], "crash_per_h_unaided": sc["metrics"]["crash_per_h_unaided"], "comfort": sc["metrics"]["comfort_pass"],
                     "progress": sc["metrics"]["progress"], "policy": f"{path}/policy.npz"}
        print(f"{name}: AUC {t['auc']:.3f} [{t['ci'][0]:.3f}, {t['ci'][1]:.3f}], av_world score {sc['av_score']:.1f}, "
              f"unaided crashes/AV-h {sc['metrics']['crash_per_h_unaided']:.3f}", flush=True)
        json.dump(res, open(f"{out}/humanize.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--humanize", action="store_true", help="fine-tune the bundled policy toward human-like motion and re-test it")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from results/turing/turing.json")
    a = ap.parse_args()
    if a.humanize:
        humanize()
    elif a.replot:
        plot(json.load(open(f"{OUT}/turing.json")), "docs/img/turing.png")
    else:
        main(a.quick)
