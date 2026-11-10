"""Who is struck? AVs against human drivers, per vehicle-hour, inside the same city.

    python -m avlab.whocrashes [--quick]    # half-AV Oakland at friction 1.0 / 0.6 / 0.4 and at speed limit x1.3 -> results/whocrashes/who.json
    python -m avlab.whocrashes --plot       # docs/img/whocrashes.png

Pedestrian and bicycle crashes are single-vehicle events: one car (human or AV) strikes one road user. In a city where half the cars are AVs
the counters say how many of those were by an AV (`av_pedestrian`, `av_bicycle`) and how many vehicle-hours AVs drove (`av_veh_s`), so
the two kinds of driver can be compared on the SAME roads, signals and pedestrians, without a second city as the control.

    for each cell (friction mu, speed-limit scale s): 48 copies of Oakland, calibrated humans, p_av = 0.5, warm 120 s, count 600 s
    AV rate = AV strikes / AV vehicle-hours;  human rate = (all strikes - AV strikes) / (vehicle-hours - AV vehicle-hours)
    ratio = AV rate / human rate; exact conditional (binomial) interval: given n strikes, the AV share is Binomial(n, t_av RR / (t_av RR + t_h))
"""
import argparse
import json
import os

import numpy as np

OUT = "results/whocrashes"
CELLS = ((1.0, 1.0), (0.6, 1.0), (0.4, 1.0), (1.0, 1.3))  # (friction, speed-limit scale)


def rate_ratio(k_av, t_av, k_h, t_h, alpha=0.05):
    """AV-to-human rate ratio from counts and exposures, with an exact conditional interval and a two-sided p-value for RR = 1."""
    from scipy import stats
    n = k_av + k_h
    if n == 0:
        return {"rr": float("nan"), "lo": 0.0, "hi": float("inf"), "p": 1.0, "n": 0}
    p_lo = stats.beta.ppf(alpha / 2, k_av, k_h + 1) if k_av > 0 else 0.0
    p_hi = stats.beta.ppf(1 - alpha / 2, k_av + 1, k_h) if k_h > 0 else 1.0
    to_rr = lambda p: float("inf") if p >= 1 else (p / (1 - p)) * (t_h / t_av)
    p0 = t_av / (t_av + t_h)
    return {"rr": (k_av / t_av) / (k_h / t_h) if k_h > 0 else float("inf"), "lo": to_rr(p_lo), "hi": to_rr(p_hi),
            "p": float(stats.binomtest(int(k_av), int(n), p0).pvalue), "n": int(n)}


def holm(p):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (m - i) * p[j])); adj[j] = run
    return adj.tolist()


def run_cell_group(scale, mus, E=40, warm_s=120.0, seconds=600.0, area="oakland", p_av=0.5):
    """All frictions of one speed-limit scale in one compiled run. Returns {mu: dict of per-copy arrays}."""
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    J = X.to_jnp(dict(net, link_v=(np.asarray(net["link_v"]) * scale).astype(np.float32)))
    nh = jnp.asarray(net["nh"])
    behs, tags = [], []
    for mu in mus:
        for _ in range(E):
            behs.append(C.calibrated(friction=mu, p_av=p_av)); tags.append(mu)
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(0), len(behs)), J, 1024)
    n = int(round(60.0 / X.DT))

    @jax.jit
    def chunk(s):
        return jax.lax.scan(lambda c, _: (jax.vmap(lambda si, b: X.step(si, J, b, nh))(c, B), None), s, None, length=n)[0]

    for _ in range(int(round(warm_s / 60.0))):
        st = chunk(st)
    m0 = np.asarray(st.m)
    for _ in range(int(round(seconds / 60.0))):
        st = chunk(st)
    m = np.asarray(st.m) - m0
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    tags = np.array(tags)
    out = {}
    for mu in mus:
        s = tags == mu
        out[mu] = {k: d[k][s].tolist() for k in ("veh_s", "av_veh_s", "av_crash", "imp_ped_n", "imp_ped_speed", *X.CRASH_NAMES, *X.AV_CRASH)}
    return out


def run(E=48, warm_s=120.0, seconds=600.0, cells=CELLS):
    res = {"copies": E, "seconds": seconds, "p_av": 0.5, "cells": {}}
    for scale in sorted({c[1] for c in cells}):
        mus = [c[0] for c in cells if c[1] == scale]
        g = run_cell_group(scale, mus, E, warm_s, seconds)
        for mu in mus:
            res["cells"][f"{mu}|{scale}"] = {"mu": mu, "scale": scale, **g[mu]}
    return res


def summarize(res):
    """Per cell: strikes of pedestrians and bicycles by AVs and by humans, rates per 1000 vehicle-hours, ratio with interval; Holm over cells."""
    rows = []
    for key, c in res["cells"].items():
        t_av = float(np.sum(c["av_veh_s"]) / 3600); t_all = float(np.sum(c["veh_s"]) / 3600); t_h = t_all - t_av
        k_all = float(sum(np.sum(c[t]) for t in ("pedestrian", "bicycle")))
        k_av = float(sum(np.sum(c["av_" + t]) for t in ("pedestrian", "bicycle")))
        r = rate_ratio(k_av, t_av, k_all - k_av, t_h)
        rows.append({"cell": key, "mu": c["mu"], "scale": c["scale"], "av_share_hours": t_av / t_all, "strikes": k_all, "av_strikes": k_av,
                     "av_rate": 1000 * k_av / t_av, "human_rate": 1000 * (k_all - k_av) / t_h, **r,
                     "all_crashes": float(sum(np.sum(c[t]) for t in ("rear-end", "angle", "left-turn", "sideswipe", "merge", "pedestrian", "bicycle"))),
                     "av_involved": float(np.sum(c["av_crash"])),
                     "ped_strike_speed": float(np.sum(c["imp_ped_speed"]) / max(np.sum(c["imp_ped_n"]), 1))})
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
    return {"rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        return plot()
    os.makedirs(OUT, exist_ok=True)
    res = run(E=4, warm_s=60.0, seconds=120.0, cells=CELLS[:2]) if a.quick else run()
    res["summary"] = summarize(res)
    json.dump(res, open(f"{OUT}/who{'_quick' if a.quick else ''}.json", "w"))
    for r in res["summary"]["rows"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    rows = json.load(open(f"{OUT}/who.json"))["summary"]["rows"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=BG)
    ax.set_facecolor(BG); ax.tick_params(colors=FG); [s.set_color(GR) for s in ax.spines.values()]
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        lo, hi = r["lo"], min(r["hi"], 30)
        ax.plot([i, i], [lo, hi], color="#56b4e9", lw=3, solid_capstyle="round")
        ax.plot(i, r["rr"], "o", color="#e69f00", ms=9, zorder=3)
        ax.text(i + 0.08, r["rr"], f"{r['rr']:.2f}\n({int(r['av_strikes'])} of {int(r['strikes'])})", color=FG, fontsize=9, va="center")
    ax.axhline(1, color="#8b949e", ls="--", lw=1)
    ax.text(-0.45, 1.05, "AV = human", color="#8b949e", fontsize=9, ha="left", va="bottom")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"friction {r['mu']}\nlimit x{r['scale']}" for r in rows], color=FG)
    ax.set_ylabel("AV / human pedestrian+bicycle strikes per vehicle-hour (log)", color=FG)
    ax.set_title("Same city, same pedestrians: does an AV strike fewer road users than a human driver?", color=FG, fontsize=11)
    ax.set_xlim(-0.5, len(rows) - 0.3)
    fig.tight_layout(); fig.savefig("docs/img/whocrashes.png", dpi=150, facecolor=BG)


if __name__ == "__main__":
    main()
