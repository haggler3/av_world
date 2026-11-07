"""What kind of crash carries an effect? Crash counts by type for a fine friction grid and for the speed limit x pedestrian violators.

    python -m avlab.crashmix --quick        # smoke run
    python -m avlab.crashmix                # both experiments -> results/crashmix/crashmix.json  (~10 min on 4 cores)
    python -m avlab.crashmix --plot         # figure -> docs/img/crashmix.png

    A  for mu in 1.0 .. 0.3 (9 values): 12 copies of Oakland, calibrated humans, friction=mu, warm 120 s, count 600 s
         per cell: crashes of each type, closing speed of rear-ends (imp_rear_closing / imp_rear_n), per 1000 vehicle-hours
         cliff test: single power law vs one-break power law (best break on the grid), Poisson deviance + 2 x parameters
    B  for scale in 1.0, 1.3 (net["link_v"] *= scale) and p_ped_violate in 0.03, 0: 24 copies, same protocol
         elasticity of the rate between the two limits, per violator setting (exact conditional interval)
Only the simulator's counters are read; xnet/ is untouched.
"""
import argparse
import json
import os

import numpy as np

OUT = "results/crashmix"
MUS = (1.0, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3)
TYPES = ("rear-end", "angle", "left-turn", "sideswipe", "merge", "pedestrian", "bicycle")


def type_counts(m):
    """[E, NM] counters -> dict of summed counts (types, vehicle-hours, impact sums)."""
    from xnet import sim as X
    m = np.asarray(m, float)
    g = lambda k: float(m[:, X.METRICS.index(k)].sum())
    d = {t: g(t) for t in TYPES}
    d["crashes"] = sum(d[t] for t in TYPES)
    d["hours"] = g("veh_s") / 3600
    d["rear_n"], d["rear_closing"] = g("imp_rear_n"), g("imp_rear_closing")
    d["ped_n"], d["ped_speed"] = g("imp_ped_n"), g("imp_ped_speed")
    return d


def rate(d, key="crashes"):
    return 1000 * d[key] / d["hours"]


def poisson_dev(y, t, X):
    """Plain Poisson regression by IRLS; returns (beta, deviance)."""
    y, t, X = np.asarray(y, float), np.asarray(t, float), np.asarray(X, float)
    beta = np.zeros(X.shape[1]); beta[0] = np.log(max(y.sum(), 1) / t.sum())
    for _ in range(200):
        mu = t * np.exp(X @ beta)
        z = X @ beta + (y - mu) / np.maximum(mu, 1e-9)
        new = np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (mu * z))
        done = np.abs(new - beta).max() < 1e-10
        beta = new
        if done:
            break
    mu = t * np.exp(X @ beta)
    dev = 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu), 0) - (y - mu))
    return beta, float(dev)


def break_fit(mus, y, t):
    """Single power law vs power law with one hinge (slope changes at mu*, continuous), mu* tried at each interior grid value.
    delta_aic = AIC(single) - AIC(broken) > 0 favours the break. The break is chosen on the same data and the search is not paid
    for, so the evidence is overstated."""
    mus = np.asarray(mus, float)
    lm = np.log(mus); c = lm.mean(); lm = lm - c
    _, d1 = poisson_dev(y, t, np.stack([np.ones_like(lm), lm], 1))
    best = None
    for mu_b in sorted(set(mus))[1:-1]:
        b = np.log(mu_b) - c
        _, d2 = poisson_dev(y, t, np.stack([np.ones_like(lm), lm, np.minimum(lm - b, 0.0)], 1))
        if best is None or d2 < best[1]:
            best = (float(mu_b), d2)
    return {"single_dev": d1, "broken_dev": best[1], "best_break": best[0], "delta_aic": (d1 + 4) - (best[1] + 6)}


def run_experiment_a(E=12, mus=MUS, warm_s=120.0, seconds=600.0, area="oakland"):
    from examples import _common as C
    behs, tags = [], []
    for mu in mus:
        behs += [C.calibrated(friction=mu) for _ in range(E)]
        tags += [mu] * E
    m = C.run_copies(area, behs, warm_s=warm_s, seconds=seconds)
    tags = np.array(tags)
    return {"E": E, "seconds": seconds, "cells": {str(mu): type_counts(m[tags == mu]) for mu in mus}}


def run_experiment_b(E=24, scales=(1.0, 1.3), viol=(0.03, 0.0), warm_s=120.0, seconds=600.0, area="oakland", seed=0):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    nh = jnp.asarray(net["nh"])
    behs = [C.calibrated(p_ped_violate=v) for v in viol for _ in range(E)]
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    n = int(round(60.0 / X.DT))
    out = {"E": E, "seconds": seconds, "cells": {}}
    for sc in scales:
        J = X.to_jnp(dict(net, link_v=(np.asarray(net["link_v"]) * sc).astype(np.float32)))

        @jax.jit
        def chunk(s):
            return jax.lax.scan(lambda c, _: (jax.vmap(lambda si, b: X.step(si, J, b, nh))(c, B), None), s, None, length=n)[0]
        st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), len(behs)), J, 1024)
        for _ in range(int(round(warm_s / 60))):
            st = chunk(st)
        m0 = np.asarray(st.m)
        for _ in range(int(round(seconds / 60))):
            st = chunk(st)
        m = np.asarray(st.m) - m0
        for i, v in enumerate(viol):
            out["cells"][f"{sc}|{v}"] = type_counts(m[i * E:(i + 1) * E])
    return out


def summarize_a(res):
    cells = sorted(res["cells"].items(), key=lambda kv: -float(kv[0]))
    mus = [float(k) for k, _ in cells]
    out = {"mus": mus, "rate": [rate(c) for _, c in cells]}
    for t in TYPES:
        out[t] = [rate(c, t) for _, c in cells]
    out["break"] = break_fit(mus, [c["crashes"] for _, c in cells], [c["hours"] for _, c in cells])
    i6, i45 = mus.index(0.6), mus.index(0.45)
    extra = {t: out[t][i45] - out[t][i6] for t in TYPES}
    tot = sum(extra.values())
    out["extra_06_to_045"] = extra | {"total": tot, "rear_share": extra["rear-end"] / tot if tot else None}
    from scipy import stats
    cl = np.array([(c["rear_closing"] / c["rear_n"]) if c["rear_n"] > 0 else np.nan for _, c in cells])
    ok = ~np.isnan(cl)
    out["rear_closing_ms"] = [None if np.isnan(v) else float(v) for v in cl]
    rho = stats.spearmanr(np.array(mus)[ok], cl[ok])
    out["rear_closing_spearman"] = [float(rho[0]), float(rho[1])]
    return out


def ratio_ci(k1, t1, k0, t0):
    """Rate ratio (1 over 0) with the exact conditional interval."""
    from scipy import stats
    n = int(k0 + k1)
    lo = stats.beta.ppf(0.025, k1, n - k1 + 1) if k1 > 0 else 0.0
    hi = stats.beta.ppf(0.975, k1 + 1, n - k1) if n - k1 > 0 else 1.0
    f = lambda q: q / (1 - q) * t0 / t1
    return (k1 / t1) / (k0 / t0), f(lo), f(hi)


def summarize_b(res):
    out = {}
    el = lambda r: float(np.log(r) / np.log(1.3))
    for v in ("0.03", "0.0"):
        a, b = res["cells"][f"1.0|{v}"], res["cells"][f"1.3|{v}"]
        rr, lo, hi = ratio_ci(b["crashes"], b["hours"], a["crashes"], a["hours"])
        out[v] = {"crashes_1.0": a["crashes"], "crashes_1.3": b["crashes"], "ped_1.0": a["pedestrian"], "ped_1.3": b["pedestrian"],
                  "rate_1.0": rate(a), "rate_1.3": rate(b), "rr": rr, "rr_lo": lo, "rr_hi": hi,
                  "elasticity": el(rr), "elasticity_lo": el(lo) if lo > 0 else None, "elasticity_hi": el(hi),
                  "by_type_1.3": {t: b[t] for t in TYPES}}
    p, q = res["cells"]["1.3|0.03"], res["cells"]["1.3|0.0"]
    rr, lo, hi = ratio_ci(q["pedestrian"], q["hours"], p["pedestrian"], p["hours"])
    out["ped_rr_at_1.3_novio_vs_viol"] = {"rr": rr, "lo": lo, "hi": hi}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--only", choices=["a", "b"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/crashmix.json"
    if a.plot:
        plot()
        return gif()
    res = json.load(open(path)) if os.path.exists(path) and a.only else {}
    if a.quick:
        r = run_experiment_a(E=2, mus=(1.0, 0.6, 0.3), seconds=60.0)
        print(json.dumps({k: v["crashes"] for k, v in r["cells"].items()}))
        r = run_experiment_b(E=2, seconds=60.0)
        print(json.dumps({k: v["crashes"] for k, v in r["cells"].items()}))
        return
    if a.only in (None, "a"):
        res["A"] = run_experiment_a(); res["A"]["summary"] = summarize_a(res["A"]); print(json.dumps(res["A"]["summary"], indent=1))
        json.dump(res, open(path, "w"), indent=1)
    if a.only in (None, "b"):
        res["B"] = run_experiment_b(); res["B"]["summary"] = summarize_b(res["B"]); print(json.dumps(res["B"]["summary"], indent=1))
        json.dump(res, open(path, "w"), indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(f"{OUT}/crashmix.json"))
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    pal = {"rear-end": "#e69f00", "pedestrian": "#56b4e9", "angle": "#009e73", "left-turn": "#cc79a7", "sideswipe": "#f0e442", "merge": "#d55e00", "bicycle": "#999999"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), facecolor=BG, gridspec_kw={"width_ratios": [1.5, 1]})
    for a_ in ax:
        a_.set_facecolor(BG); a_.tick_params(colors=FG); [s.set_color(GR) for s in a_.spines.values()]
        a_.xaxis.label.set_color(FG); a_.yaxis.label.set_color(FG); a_.title.set_color(FG)
    S = res["A"]["summary"]; x = S["mus"]
    for t in TYPES:
        y = np.array(S[t])
        if y.max() < 1.0:
            continue
        ax[0].plot(x, np.maximum(y, 0.3), "o-", color=pal[t], lw=2, ms=4)
        yy = {"pedestrian": 1.1, "merge": 1.0, "bicycle": 0.78, "rear-end": 0.52, "angle": 0.62, "left-turn": 1.0}[t]
        ax[0].text(x[-1] * 0.97, max(y[-1], 0.3) * yy, t, color=pal[t], fontsize=9, ha="right", va="bottom" if yy >= 1 else "top")
    ax[0].set_yscale("log"); ax[0].set_xscale("log"); ax[0].invert_xaxis()
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(v) for v in x], fontsize=8); ax[0].minorticks_off()
    ax[0].set_xlabel("road friction (1 = dry, 0.3 = snow)"); ax[0].set_ylabel("crashes per 1000 vehicle-hours")
    ax[0].set_title("A. Crash types along the friction cliff (humans only)", fontsize=10)
    cells = res["B"]["cells"]
    w = 0.38
    for j, v in enumerate(("0.03", "0.0")):
        for i, sc in enumerate(("1.0", "1.3")):
            c = cells[f"{sc}|{v}"]; bottom = 0
            for t in ("pedestrian", "rear-end", "angle", "left-turn", "sideswipe", "merge", "bicycle"):
                h = 1000 * c[t] / c["hours"]
                ax[1].bar(i + (j - 0.5) * w, h, w * 0.92, bottom=bottom, color=pal[t], edgecolor=BG)
                bottom += h
            ax[1].text(i + (j - 0.5) * w, bottom + 0.4, f"{bottom:.1f}", color=FG, ha="center", fontsize=9)
            ax[1].text(i + (j - 0.5) * w, -1.8, "viol." if v == "0.03" else "no viol.", color=FG, ha="center", fontsize=7)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["limit x1.0", "limit x1.3"], color=FG); ax[1].tick_params(axis="x", pad=14)
    ax[1].set_ylabel("crashes per 1000 vehicle-hours"); ax[1].set_title("B. Speed limit x pedestrian violators", fontsize=10)
    for j, t in enumerate(("pedestrian", "rear-end", "angle", "left-turn", "merge", "bicycle")):
        ax[1].text(0.03, 0.97 - 0.055 * j, t, color=pal[t], fontsize=8, transform=ax[1].transAxes, va="top")
    ax[0].text(0.02, 0.02, "zero crashes plotted at 0.3", color=FG, fontsize=7, transform=ax[0].transAxes)
    fig.tight_layout(); fig.savefig("docs/img/crashmix.png", dpi=130, facecolor=BG)


def gif(path="docs/img/crashmix.gif"):
    """The friction sweep as an animation: friction falls from 1.0 to 0.3 and the stacked bars (crashes per 1000 vehicle-hours by type) grow."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    res = json.load(open(f"{OUT}/crashmix.json"))["A"]
    S = res["summary"]; x = S["mus"]
    BG, FG = "#0d1117", "#e6edf3"
    pal = {"rear-end": "#e69f00", "pedestrian": "#56b4e9", "merge": "#d55e00", "bicycle": "#999999", "angle": "#009e73", "left-turn": "#cc79a7"}
    order = ["pedestrian", "merge", "bicycle", "rear-end", "angle", "left-turn"]
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    def draw(k):
        ax.clear(); ax.set_facecolor(BG); ax.tick_params(colors=FG)
        [s_.set_color("#30363d") for s_ in ax.spines.values()]
        for i in range(min(k + 1, len(x))):
            b = 0
            for t in order:
                h = S[t][i]; ax.bar(i, h, 0.8, bottom=b, color=pal[t]); b += h
            ax.text(i, b + 3, f"{b:.0f}", color=FG, ha="center", fontsize=8)
        ax.set_xlim(-0.6, len(x) - 0.4); ax.set_ylim(0, 215)
        ax.set_xticks(range(len(x))); ax.set_xticklabels([str(v) for v in x], fontsize=8)
        ax.set_xlabel("road friction (1 = dry, 0.3 = snow)", color=FG); ax.set_ylabel("crashes per 1000 vehicle-hours", color=FG)
        ax.set_title("Oakland, humans only: which crashes appear as the road gets slick", color=FG, fontsize=10)
        for j, t in enumerate(order):
            ax.text(0.02, 0.95 - 0.06 * j, t, color=pal[t], transform=ax.transAxes, fontsize=9, va="top")
    anim = FuncAnimation(fig, draw, frames=list(range(len(x))) + [len(x) - 1] * 6, interval=200)
    anim.save(path, writer=PillowWriter(fps=3), savefig_kwargs={"facecolor": BG})


if __name__ == "__main__":
    main()
