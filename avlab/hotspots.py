"""Crash hotspots the way road agencies find them, and a test of why they do it that way.

    python -m avlab.hotspots          # 128 copies x 20 min of Oakland (~6 min, 4 CPU cores) -> results/hotspots/, docs/img/hotspots.png

Every crash is placed at a site: the junction it happened in, or the junction at the end of the link it happened on (its approach).
Each site's exposure is the vehicle-hours spent on its tracks. Then three ways to rank sites, all from the first half of the copies:

  naive    the crash count (what a list of "most crashes last year" does)
  rate     crashes per vehicle-hour
  EB       empirical Bayes (Hauer 1997, the Highway Safety Manual's method): a safety performance function mu_i = exp(a) E_i^b fitted
           across sites (Poisson regression), the negative-binomial overdispersion alpha by moments (Var = mu + alpha mu^2), and the
           site's estimate EB_i = w_i mu_i + (1 - w_i) y_i with w_i = 1 / (1 + alpha mu_i): a site with little data leans on what is
           normal for its traffic, a site with lots of data on its own record

and each is judged on the second, independent half: how many of its crashes the top-10 sites hold, and how far the predicted counts
are from what happened (regression to the mean: the naive top-10 always "improve" on their own).
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from xnet import sim as X

OUT = "results/hotspots"


def site_ids(net):
    """Track index -> site (junction node): link lanes map to the node at their end, moves to their junction."""
    link_site = np.repeat(np.asarray(net["link_end"]), 2)
    return np.concatenate([link_site, np.asarray(net["mov_node"])]).astype(np.int32), int(len(net["node_signal"]))


def measure(area="oakland", copies=128, warm_s=120.0, seconds=1200.0, seed=0, batch=64, chunk_s=60.0):
    """-> crashes [copies, sites], vehicle-hours [copies, sites] (measured after the warm-up)."""
    from examples import _common as C
    from av_world import easy
    net, geo, info = easy._build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    sid, S = site_ids(net)
    sid = jnp.asarray(sid)
    beh = C.calibrated(p_av=0.0)
    n = int(round(chunk_s / X.DT))

    def one(s):
        s2 = X.step(s, J, beh, nh)
        new = (s2.wreck_t > 0) & (s.wreck_t <= 0)
        site = sid[s2.trk]
        crashes = jnp.minimum(jax.ops.segment_sum(new.astype(jnp.float32), site, S), 1.0)  # a crash, not each vehicle in it
        return s2, (crashes, jax.ops.segment_sum(s2.alive.astype(jnp.float32) * X.DT, site, S))

    @jax.jit
    def chunk(st, acc):
        def body(c, _):
            s, (k, t) = c
            s, (dk, dt) = jax.vmap(one)(s)
            return (s, (k + dk, t + dt)), None
        return jax.lax.scan(body, (st, acc), None, length=n)[0]

    K, T = [], []
    for b in range(0, copies, batch):
        E = min(batch, copies - b)
        st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed + b), E), J, 1024)
        zero = (jnp.zeros((E, S)), jnp.zeros((E, S)))
        for _ in range(int(warm_s / chunk_s)):
            st, _ = chunk(st, zero)
        acc = zero
        for _ in range(int(seconds / chunk_s)):
            st, acc = chunk(st, acc)
        K.append(np.asarray(acc[0]))
        T.append(np.asarray(acc[1]) / 3600)
        print(f"  {b + E}/{copies} copies", flush=True)
    return np.concatenate(K), np.concatenate(T), (net, geo, info)


def site_features(net):
    """Site-type covariates for the safety performance function: signalised (0/1), log(1 + number of movements through it)."""
    S = int(len(net["node_signal"]))
    nmov = np.bincount(np.asarray(net["mov_node"]), minlength=S)
    return np.c_[(np.asarray(net["node_signal"]) > 0).astype(float), np.log1p(nmov)]


def eb(y, E, Z=None):
    """Empirical Bayes estimates of each site's expected crashes. The SPF is log mu = a + b log E (+ c' Z, site-type covariates,
    as the Highway Safety Manual fits one SPF per site type). -> (EB, mu, alpha, coefficients)."""
    from examples.hypotheses import poisson_glm
    ok = E > 0
    D = np.c_[np.ones(len(E)), np.log(np.maximum(E, 1e-9))] if Z is None else np.c_[np.ones(len(E)), np.log(np.maximum(E, 1e-9)), Z]
    try:
        b, _, _, _ = poisson_glm(y[ok], D[ok], np.zeros(ok.sum()))
    except np.linalg.LinAlgError:  # too few crashes to fit the SPF: a common rate per vehicle-hour
        D = np.c_[np.ones(len(E)), np.log(np.maximum(E, 1e-9))]
        b = np.array([np.log(max(y.sum(), 0.5) / max(E.sum(), 1e-9)), 1.0])
    mu = np.where(ok, np.exp(D @ b), 0.0)
    alpha = max(float(np.sum((y - mu) ** 2 - mu) / max(np.sum(mu ** 2), 1e-12)), 1e-6)
    w = 1 / (1 + alpha * mu)
    return w * mu + (1 - w) * y, mu, alpha, tuple(float(x) for x in b)


def evaluate(K, T, Z=None, top=10, splits=20, seed=0):
    """Random halves of the copies: rank on half A, judge on half B. -> per method: share of B's crashes in its top sites, mean
    predicted vs observed B counts for its top sites (scaled by exposure), and a Spearman of predicted vs observed over all sites."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    methods = ("naive", "rate", "EB") + (("EB + site type",) if Z is not None else ())
    out = {m: {"captured": [], "pred_top": [], "obs_top": [], "spearman": []} for m in methods}
    for _ in range(splits):
        p = rng.permutation(len(K))
        a, b = p[: len(K) // 2], p[len(K) // 2:]
        ya, Ea, yb, Eb = K[a].sum(0), T[a].sum(0), K[b].sum(0), T[b].sum(0)
        scale = np.where(Ea > 0, Eb / np.maximum(Ea, 1e-12), 0)
        est, _, _, _ = eb(ya, Ea)
        rate = np.where(Ea > 0.05, ya / np.maximum(Ea, 1e-12), 0)
        # (ranking key, predicted next-period count): the rate ranks by crashes per vehicle-hour, whatever the traffic
        preds = {"naive": (ya, ya * scale), "rate": (rate, rate * Eb), "EB": (est * scale, est * scale)}
        if Z is not None:
            est_z, _, _, _ = eb(ya, Ea, Z)
            preds["EB + site type"] = (est_z * scale, est_z * scale)
        for m, (key, pr) in preds.items():
            order = np.argsort(-key + 1e-9 * rng.random(len(key)))[:top]
            out[m]["captured"].append(yb[order].sum() / max(yb.sum(), 1))
            out[m]["pred_top"].append(pr[order].sum())
            out[m]["obs_top"].append(yb[order].sum())
            out[m]["spearman"].append(spearmanr(pr, yb).statistic)
    return {m: {k: (float(np.mean(v)), float(np.std(v))) for k, v in d.items()} for m, d in out.items()}


def plot(K, T, nets, ev, out, top=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from xnet import render as R
    net, geo, info = nets
    est, mu, alpha, _ = eb(K.sum(0), T.sum(0), site_features(net))
    S = len(est)
    xy = np.full((S, 2), np.nan)
    pts = {}
    for mv, node in enumerate(np.asarray(net["mov_node"])):
        pts.setdefault(int(node), []).append(np.asarray(geo.mov_pts[mv]).mean(0))
    for node, p in pts.items():
        xy[node] = np.mean(p, 0)
    BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
    fig = plt.figure(figsize=(15, 6.4), facecolor=BG)
    ax = fig.add_axes([0.015, 0.02, 0.49, 0.86])
    ax.set_facecolor(R.GROUND)
    R.Map(net, geo, info).draw(ax, detail=False, labels=False)
    ok = ~np.isnan(xy[:, 0]) & (est > 0)
    rate = est / np.maximum(T.sum(0), 1e-9) * 1000
    sc = ax.scatter(xy[ok, 0], xy[ok, 1], s=8 + 400 * est[ok] / est.max(), c=rate[ok], cmap="YlOrRd", vmin=0, vmax=np.percentile(rate[ok], 98),
                    edgecolor="#0d1117", linewidth=0.5, zorder=5)
    for r, i in enumerate(np.argsort(-est)[:top]):
        if not np.isnan(xy[i, 0]):
            ax.annotate(str(r + 1), xy[i], color=INK, fontsize=8, xytext=(5, 5), textcoords="offset points", zorder=6, weight="bold")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Oakland's crash hotspots by empirical Bayes\n(size: expected crashes; colour: per 1,000 vehicle-hours; top 10 numbered)",
                 color=INK, fontsize=9.5, loc="left")
    a1 = fig.add_axes([0.57, 0.14, 0.4, 0.72])
    a1.set_facecolor(BG)
    a1.tick_params(colors=INK2, labelsize=8)
    for sp in a1.spines.values():
        sp.set_color("#30363d")
    ms = list(ev)
    x = np.arange(len(ms))
    a1.bar(x - 0.2, [ev[m]["pred_top"][0] for m in ms], 0.38, color="#8b949e", label="predicted for the next period (top-10 sites)")
    a1.bar(x + 0.2, [ev[m]["obs_top"][0] for m in ms], 0.38, color="#3987e5", label="what happened")
    for i, m in enumerate(ms):
        a1.text(i, max(ev[m]["pred_top"][0], ev[m]["obs_top"][0]) * 1.03, f"holds {100 * ev[m]['captured'][0]:.0f} % of all crashes", ha="center",
                color=INK, fontsize=8.5)
    a1.set_xticks(x, [{"naive": "crash count", "rate": "crash rate", "EB": "empirical Bayes", "EB + site type": "EB + site type"}[m] for m in ms],
                  color=INK, fontsize=8.5)
    a1.set_ylabel("crashes at the 10 selected sites, second half", color=INK2, fontsize=9)
    a1.set_ylim(0, max(max(d["pred_top"][0], d["obs_top"][0]) for d in ev.values()) * 1.18)
    a1.legend(facecolor=BG, edgecolor="#30363d", labelcolor=INK, fontsize=8, loc="lower right")
    a1.set_title("Pick the top 10 on one half of the data, check on the other (20 random splits)", color=INK, fontsize=10, loc="left")
    fig.savefig(out, dpi=105, facecolor=BG)
    plt.close(fig)


def main(quick=False):
    os.makedirs(OUT, exist_ok=True)
    kw = dict(copies=4, warm_s=60.0, seconds=120.0, batch=4) if quick else {}
    t0 = time.time()
    K, T, nets = measure(**kw)
    np.savez_compressed(f"{OUT}/{'quick' if quick else 'hotspots'}_counts.npz", K=K, T=T)
    return analyse(K, T, nets, quick, t0)


def analyse(K, T, nets, quick=False, t0=None):
    t0 = t0 or time.time()
    Z = site_features(nets[0])
    ev = evaluate(K, T, Z)
    est, mu, alpha, spf = eb(K.sum(0), T.sum(0), Z)
    res = {"crashes": float(K.sum()), "sites": int(K.shape[1]), "sites_with_crashes": int((K.sum(0) > 0).sum()), "alpha": alpha, "spf": spf,
           "evaluation": ev, "top10": [{"site": int(i), "eb": float(est[i]), "observed": float(K.sum(0)[i]), "vehicle_hours": float(T.sum(0)[i])}
                                        for i in np.argsort(-est)[:10]], "wall_s": time.time() - t0}
    json.dump(res, open(f"{OUT}/{'quick' if quick else 'hotspots'}.json", "w"), indent=1)
    if not quick:
        plot(K, T, nets, ev, "docs/img/hotspots.png")
    print(f"{res['crashes']:.0f} crashes at {res['sites_with_crashes']} of {res['sites']} sites; alpha {alpha:.2f}; SPF {np.round(spf, 2).tolist()}")
    for m, d in ev.items():
        print(f"  {m:6s} top-10 hold {100 * d['captured'][0]:.1f} % of next-period crashes; predicted {d['pred_top'][0]:.1f} vs observed "
              f"{d['obs_top'][0]:.1f}; Spearman {d['spearman'][0]:.2f}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reanalyse", action="store_true", help="re-run the statistics on the saved counts")
    a = ap.parse_args()
    if a.reanalyse:
        from av_world import easy
        z = np.load(f"{OUT}/hotspots_counts.npz")
        analyse(z["K"], z["T"], easy._build("oakland"))
    else:
        main(a.quick)
