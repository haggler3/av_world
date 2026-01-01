"""Does a city with AVs pay less for a higher speed limit? Compares the crash-rate elasticity of the speed limit between worlds
with different AV shares, overall and by crash type (docs/DAILY.md, 2026-11-05).

    python -m avlab.speedlimit                  # 0 % AVs (saved):  results/speedlimit/speedlimit.json
    python -m avlab.speedlimit --p-av 0.5       # 50 % AVs:         results/speedlimit/speedlimit_av50.json
    python -m avlab.speedshare                  # compare -> results/speedlimit/speedshare.json
    python -m avlab.speedshare --plot           # -> docs/img/speedshare.png

    per world: quasi-Poisson regression of crashes on log(limit scale) with offset log(vehicle-hours), per copy  -> b, SE
    per crash type: the same on the 5 scale totals (Poisson SE x sqrt(dispersion of the all-types fit); the types' per-copy counts are not stored)
    difference of elasticities: z = (b_a - b_b) / sqrt(se_a^2 + se_b^2), two-sided p; Holm correction over the tested family
"""
import argparse
import json
import os

import numpy as np

from avlab.speedlimit import OUT, poisson_fit

TYPES = ("rear-end", "angle", "left-turn", "sideswipe", "merge", "pedestrian", "bicycle")


def normal_p(z):
    from math import erf, sqrt
    return 1.0 - erf(abs(z) / sqrt(2.0))


def holm(p):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj.tolist()


def elasticity(res):
    """(b, se, dispersion) of crashes against log(limit) over all copies of a speedlimit result."""
    y, t, x = [], [], []
    for s, r in res["scales"].items():
        y += r["crashes"]; t += [v / 3600 for v in r["veh_s"]]; x += [np.log(float(s))] * len(r["crashes"])
    return poisson_fit(y, t, x)


def type_elasticity(res, typ, phi=1.0):
    """Elasticity of one crash type from the scale totals (Poisson SE x sqrt(phi)); None if fewer than 3 scales have a crash of it."""
    sc = sorted(res["scales"], key=float)
    y = np.array([res["scales"][s]["by_type"][typ] for s in sc])
    t = np.array([sum(res["scales"][s]["veh_s"]) / 3600 for s in sc])
    if (y > 0).sum() < 3:
        return None
    b, se, _ = poisson_fit(y, t, np.log([float(s) for s in sc]))
    return b, se * np.sqrt(phi)


def compare(a, b):
    """a, b: speedlimit results (e.g. 0 % and 50 % AVs). Returns the overall and per-type comparison with Holm-adjusted p."""
    ba, sa, pa = elasticity(a)
    bb, sb, pb = elasticity(b)
    rows = [{"what": "all crashes", "b_a": ba, "se_a": sa, "b_b": bb, "se_b": sb}]
    for typ in TYPES:
        ta, tb = type_elasticity(a, typ, pa), type_elasticity(b, typ, pb)
        if ta and tb:
            rows.append({"what": typ, "b_a": ta[0], "se_a": ta[1], "b_b": tb[0], "se_b": tb[1]})
    for r in rows:
        r["diff"] = r["b_a"] - r["b_b"]
        r["se_diff"] = float(np.hypot(r["se_a"], r["se_b"]))
        r["p"] = normal_p(r["diff"] / r["se_diff"])
    adj = holm([r["p"] for r in rows])
    for r, q in zip(rows, adj):
        r["p_holm"] = q
    return rows


def rates(res):
    return {s: 1000 * sum(r["crashes"]) / (sum(r["veh_s"]) / 3600) for s, r in sorted(res["scales"].items(), key=lambda kv: float(kv[0]))}


def plot(a, b, rows, out="docs/img/speedshare.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, FG, C0, C1 = "#0d1117", "#e6edf3", "#e69f00", "#56b4e9"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.1]})
    for a_ in ax:
        a_.set_facecolor(BG); a_.tick_params(colors=FG); [s.set_color("#30363d") for s in a_.spines.values()]
        a_.xaxis.label.set_color(FG); a_.yaxis.label.set_color(FG); a_.title.set_color(FG)
    for res, col, name, dy in ((a, C0, f"{round(100 * a.get('p_av', 0))} % AVs", 0.62), (b, C1, f"{round(100 * b['p_av'])} % AVs", 1.35)):
        r = rates(res)
        x = [float(s) for s in r]
        ax[0].plot(x, list(r.values()), "o-", color=col)
        ax[0].text(x[-1], list(r.values())[-1] * dy, name, color=col, ha="right", va="center", fontsize=10)
    ax[0].set_yscale("log"); ax[0].set_yticks([2, 5, 10, 20]); ax[0].set_yticklabels(["2", "5", "10", "20"]); ax[0].minorticks_off(); ax[0].set_xlabel("speed limits x"); ax[0].set_ylabel("crashes per 1000 vehicle-hours")
    ax[0].set_title("Crash rate against the limit")
    ax[0].grid(color="#21262d")
    for i, r in enumerate(rows):
        y = len(rows) - i
        for d, col, k in ((0.12, C0, "a"), (-0.12, C1, "b")):
            ax[1].errorbar(r["b_" + k], y + d, xerr=1.96 * r["se_" + k], fmt="o", color=col, capsize=3, ms=4)
    ax[1].set_yticks([len(rows) - i for i in range(len(rows))]); ax[1].set_yticklabels([r["what"] for r in rows], color=FG)
    ax[1].axvline(0, color="#30363d"); ax[1].set_xlabel("elasticity b (crash rate ~ limit^b), 95 % interval")
    ax[1].text(0.02, 0.98, "orange: 0 % AVs   blue: 50 % AVs", transform=ax[1].transAxes, ha="left", va="top", color=FG, fontsize=9)
    ax[1].set_title("By crash type"); ax[1].grid(color="#21262d")
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=BG)


def gif(scale=1.3, seed=0, budget=1200.0, out="docs/img/speedshare.gif"):
    """A zoomed black-box GIF of a pedestrian crash in Oakland with every speed limit x `scale` (the crash type that carries the effect)."""
    from avlab import blackbox as BB
    from examples import _common as C
    from xnet import network as NW
    from xnet import render as R
    net, geo, info = NW.build("oakland")
    net = dict(net, link_v=(np.asarray(net["link_v"]) * scale).astype(np.float32))
    frames, e = BB.hunt(net, C.calibrated(), seed=seed, budget=budget, types=("pedestrian",))
    if frames is None:
        print("no pedestrian crash found")
        return None
    rep = BB.investigate(net, geo, frames, C.calibrated())
    k = rep["frame"]
    R.animate(net, geo, info, frames[max(0, k - 40):k + 25], out, zoom=(tuple(rep["centre_xy"]), 45.0),
              title=f"Speed limits x{scale}: a pedestrian crash", fps=10)
    json.dump(rep, open(f"{OUT}/speedshare_crash.json", "w"), indent=1, default=float)
    print(BB.markdown(rep))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--other", default=f"{OUT}/speedlimit_av50.json")
    a_ = ap.parse_args()
    if a_.gif:
        return gif()
    a = json.load(open(f"{OUT}/speedlimit.json"))
    b = json.load(open(a_.other))
    rows = compare(a, b)
    out = {"a_p_av": a.get("p_av", 0.0), "b_p_av": b["p_av"], "rows": rows, "rates_a": rates(a), "rates_b": rates(b)}
    print(json.dumps(out, indent=1))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/speedshare.json", "w"), indent=1)
    if a_.plot:
        plot(a, b, rows)


if __name__ == "__main__":
    main()
