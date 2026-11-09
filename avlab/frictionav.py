"""Do AVs that read the road friction crash less on a slick road? (docs/DAILY.md, 2026-11-08 next step 10)

    python -m avlab.frictionav [--quick]   # -> results/frictionav/frictionav.json
    python -m avlab.frictionav plot        # -> docs/img/frictionav.png

The planner in xnet/sim.py only meets friction through the braking cap (a_min = -9 x friction); an AV keeps its 1.3 s headway and its
desired speed on any road. Here the AV is made friction-aware from OUTSIDE the simulator: after every step the AVs' headway and
desired-speed factor are clamped, T_av >= 1.3 / mu**a  and  v0f_av <= mu**b  (idempotent, so new AVs are fixed one step after they spawn).
Cells (half AVs, mu = 0.45, same initial states per copy index): base, headway, speed, both.  The count is crashes that involve an AV
(av_crash) per AV vehicle-hour, plus all crashes per vehicle-hour; ratios against the base cell with exact conditional (binomial) intervals.
"""
import argparse
import json
import os

import numpy as np

OUT = "results/frictionav"
MU = 0.45


def clamp_av(st, T_min, v0_max):
    import jax.numpy as jnp
    return st._replace(T=jnp.where(st.av, jnp.maximum(st.T, T_min), st.T), v0f=jnp.where(st.av, jnp.minimum(st.v0f, v0_max), st.v0f))


def run(E=16, warm_s=120.0, seconds=600.0, mu=MU, area="oakland", a=1.0, b=0.5, cells=("base", "headway", "speed", "both")):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated(friction=mu, p_av=0.5)
    Tm = {"base": 0.0, "headway": 1.3 / mu ** a, "speed": 0.0, "both": 1.3 / mu ** a}
    Vm = {"base": 9.0, "headway": 9.0, "speed": mu ** b, "both": mu ** b}
    Tv = jnp.asarray([Tm[c] for c in cells for _ in range(E)], jnp.float32)
    Vv = jnp.asarray([Vm[c] for c in cells for _ in range(E)], jnp.float32)
    keys = jnp.tile(jax.random.split(jax.random.PRNGKey(0), E), (len(cells), 1))
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
    n = int(round(20.0 / X.DT))

    @jax.jit
    def chunk(s):
        def body(c, _):
            c = jax.vmap(lambda si: X.step(si, J, beh, nh))(c)
            return jax.vmap(clamp_av)(c, Tv, Vv), None
        return jax.lax.scan(body, s, None, length=n)[0]

    for _ in range(int(round(warm_s / 20.0))):
        st = chunk(st)
    m0 = np.asarray(st.m)
    for _ in range(int(round(seconds / 20.0))):
        st = chunk(st)
    m = np.asarray(st.m) - m0
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    cr = sum(d[k] for k in X.CRASH_NAMES)
    res = {"area": area, "mu": mu, "copies": E, "seconds": seconds, "a": a, "b": b, "cells": {}}
    for j, c in enumerate(cells):
        s = slice(j * E, (j + 1) * E)
        res["cells"][c] = {"T_min": Tm[c], "v0_max": min(Vm[c], 9.0), "crashes": cr[s].tolist(), "av_crashes": d["av_crash"][s].tolist(),
                           "hours": (d["veh_s"][s] / 3600).tolist(), "av_hours": (d["av_veh_s"][s] / 3600).tolist(),
                           "speed_kmh": float(3.6 * d["veh_m"][s].sum() / d["veh_s"][s].sum()),
                           "av_speed_kmh": float(3.6 * d["av_veh_m"][s].sum() / max(d["av_veh_s"][s].sum(), 1e-9)),
                           "finished": float(d["finished"][s].sum()),
                           "types": {k: float(d[k][s].sum()) for k in X.CRASH_NAMES}}
    return res


def ratio(k1, t1, k0, t0):
    """Rate ratio of cell 1 to cell 0 with an exact conditional (binomial) 95 % interval and two-sided p for ratio = 1."""
    from scipy import stats
    n = int(k1 + k0)
    if n == 0:
        return {"ratio": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": 1.0}
    ci = stats.binomtest(int(k1), n, t1 / (t1 + t0)).proportion_ci(0.95, method="exact")
    f = lambda p: p / (1 - p) * t0 / t1 if p < 1 else float("inf")
    return {"ratio": (k1 / t1) / max(k0 / t0, 1e-12), "lo": f(ci.low), "hi": f(ci.high), "p": float(stats.binomtest(int(k1), n, t1 / (t1 + t0)).pvalue)}


def summarize(res):
    out = {"cells": {}, "ratios": {}}
    base = res["cells"]["base"]
    for c, v in res["cells"].items():
        out["cells"][c] = {"crashes": sum(v["crashes"]), "av_crashes": sum(v["av_crashes"]), "rate_per_1000_veh_h": 1000 * sum(v["crashes"]) / sum(v["hours"]),
                           "av_crash_per_1000_av_h": 1000 * sum(v["av_crashes"]) / max(sum(v["av_hours"]), 1e-9), "av_hours": sum(v["av_hours"]),
                           "speed_kmh": v["speed_kmh"], "av_speed_kmh": v["av_speed_kmh"], "finished": v["finished"]}
    names = [c for c in res["cells"] if c != "base"]
    ps = []
    for c in names:
        v = res["cells"][c]
        r = ratio(sum(v["av_crashes"]), sum(v["av_hours"]), sum(base["av_crashes"]), sum(base["av_hours"]))
        r2 = ratio(sum(v["crashes"]), sum(v["hours"]), sum(base["crashes"]), sum(base["hours"]))
        out["ratios"][c] = {"av_involved": r, "all": r2}; ps.append(r["p"])
    order = np.argsort(ps); m = len(ps); run_max = 0.0
    for rank, i in enumerate(order):  # Holm step-down on the AV-involved ratios
        run_max = max(run_max, min(1.0, (m - rank) * ps[i])); out["ratios"][names[i]]["av_involved"]["p_holm"] = run_max
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    S = json.load(open(f"{OUT}/frictionav.json"))["summary"]
    cols = {"base": "#8b949e", "headway": "#e69f00", "speed": "#56b4e9", "both": "#009e73"}
    lab = {"base": "as is", "headway": "longer headway", "speed": "lower speed", "both": "both"}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG); a.tick_params(colors=FG); [s.set_color(GR) for s in a.spines.values()]
        a.xaxis.label.set_color(FG); a.yaxis.label.set_color(FG); a.title.set_color(FG)
    for i, c in enumerate(S["cells"]):
        v = S["cells"][c]
        ax[0].bar(i, v["rate_per_1000_veh_h"], color=cols[c]); ax[0].text(i, v["rate_per_1000_veh_h"] + 1, f"{v['rate_per_1000_veh_h']:.0f}\n({v['crashes']:.0f})", color=FG, ha="center", fontsize=9)
        ax[1].scatter(v["av_speed_kmh"], v["rate_per_1000_veh_h"], s=90, color=cols[c]); ax[1].text(v["av_speed_kmh"], v["rate_per_1000_veh_h"] * 1.07, lab[c], color=cols[c], ha="center", fontsize=9)
    ax[0].set_xticks(range(len(S["cells"]))); ax[0].set_xticklabels([lab[c] for c in S["cells"]], color=FG, fontsize=9)
    ax[0].set_ylabel("crashes per 1000 vehicle-hours (count)"); ax[0].set_title("Friction 0.45, half AVs: what the AV is told")
    ax[0].set_ylim(0, 1.3 * max(v["rate_per_1000_veh_h"] for v in S["cells"].values()))
    ax[1].set_ylim(top=1.12 * max(v["rate_per_1000_veh_h"] for v in S["cells"].values())); ax[1].set_xlim(13.2, 17.2); ax[1].set_xlabel("AV mean speed (km/h)"); ax[1].set_ylabel("crashes per 1000 vehicle-hours"); ax[1].set_title("Safety against the speed it costs")
    fig.tight_layout(); fig.savefig("docs/img/frictionav.png", dpi=130, facecolor=BG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="run", choices=["run", "plot"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == "plot":
        return plot()
    res = run(**(dict(E=2, seconds=40.0, warm_s=40.0) if a.quick else {}))
    res["summary"] = summarize(res)
    print(json.dumps(res["summary"], indent=1)[:3000])
    if not a.quick:
        json.dump(res, open(f"{OUT}/frictionav.json", "w"), indent=1)


if __name__ == "__main__":
    main()
