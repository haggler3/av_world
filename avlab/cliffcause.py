"""What is behind the friction cliff, and does an AV-heavy city pay less for a higher speed limit?

    python -m avlab.cliffcause cliff [--quick]   # humans-only Oakland at 6 frictions: crashes by TYPE -> results/cliffcause/cliff.json
    python -m avlab.cliffcause limit [--quick]   # speed limit x0.85/1.0/1.3 x {humans, half AVs}  -> results/cliffcause/limit.json
    python -m avlab.cliffcause plot              # docs/img/cliffcause.png

    cliff:  for mu in 0.7 0.6 0.55 0.5 0.45 0.4: 12 copies, calibrated humans, friction=mu (one compiled run); count crashes by type and
            the closing speed of rear-end impacts (imp_rear_closing / imp_rear_n); the share of rear-end crashes is the diagnostic.
    limit:  for scale in .85 1 1.3: net link_v *= scale (a copy of the net dict); 12 copies with p_av=0 and 12 with p_av=0.5; quasi-Poisson
            slope of log rate on log scale for each share and their difference.
"""
import argparse
import json
import os

import numpy as np

OUT = "results/cliffcause"
MUS = (0.7, 0.6, 0.55, 0.5, 0.45, 0.4)
SCALES = (0.85, 1.0, 1.3)


def by_type(m, X):
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    return d


def run_cliff(E=12, warm_s=120.0, seconds=600.0, mus=MUS, area="oakland"):
    from examples import _common as C
    from xnet import sim as X
    behs, tags = [], []
    for mu in mus:
        for _ in range(E):
            behs.append(C.calibrated(friction=mu)); tags.append(mu)
    m = C.run_copies(area, behs, warm_s=warm_s, seconds=seconds)
    d = by_type(m, X)
    tags = np.array(tags)
    res = {"area": area, "copies": E, "seconds": seconds, "cells": {}}
    for mu in mus:
        s = tags == mu
        types = {k: float(d[k][s].sum()) for k in X.CRASH_NAMES}
        res["cells"][str(mu)] = {"mu": mu, "types": types, "crashes": float(sum(types.values())), "hours": float(d["veh_s"][s].sum() / 3600),
                                 "per_copy_crashes": sum(d[k][s] for k in X.CRASH_NAMES).tolist(), "per_copy_hours": (d["veh_s"][s] / 3600).tolist(),
                                 "rear_n": float(d["imp_rear_n"][s].sum()), "rear_closing": float(d["imp_rear_closing"][s].sum()),
                                 "ped_n": float(d["imp_ped_n"][s].sum()), "ped_speed": float(d["imp_ped_speed"][s].sum())}
    return res


def run_limit(E=12, warm_s=120.0, seconds=600.0, scales=SCALES, area="oakland"):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    nh = jnp.asarray(net["nh"])
    behs = [C.calibrated(p_av=p) for p in (0.0, 0.5) for _ in range(E)]
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    n = int(round(20.0 / X.DT))

    @jax.jit
    def chunk(s, J):
        return jax.lax.scan(lambda c, _: (jax.vmap(lambda si, b: X.step(si, J, b, nh))(c, B), None), s, None, length=n)[0]

    res = {"area": area, "copies": E, "seconds": seconds, "cells": {}}
    for sc in scales:
        J = X.to_jnp(dict(net, link_v=(np.asarray(net["link_v"]) * sc).astype(np.float32)))
        st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(0), 2 * E), J, 1024)
        for _ in range(int(round(warm_s / 20.0))):
            st = chunk(st, J)
        m0 = np.asarray(st.m)
        for _ in range(int(round(seconds / 20.0))):
            st = chunk(st, J)
        d = by_type(np.asarray(st.m) - m0, X)
        cr = sum(d[k] for k in X.CRASH_NAMES)
        for j, p in enumerate((0.0, 0.5)):
            s = slice(j * E, (j + 1) * E)
            res["cells"][f"{sc}|{p}"] = {"scale": sc, "p_av": p, "crashes": cr[s].tolist(), "hours": (d["veh_s"][s] / 3600).tolist(),
                                         "ped": float(d["pedestrian"][s].sum()), "rear": float(d["rear-end"][s].sum())}
    return res


def summarize_cliff(res):
    """Rear-end share by friction, fine-grid slopes, and a piecewise break (grid search of a hinge in log mu minimising Poisson deviance)."""
    from avlab.friction import irls
    rows = []
    for c in sorted(res["cells"].values(), key=lambda c: -c["mu"]):
        k = c["crashes"]
        rows.append({"mu": c["mu"], "crashes": k, "rate": 1000 * k / c["hours"], "rear_share": c["types"]["rear-end"] / max(k, 1),
                     "ped_share": c["types"]["pedestrian"] / max(k, 1), "rear_closing_mean": c["rear_closing"] / max(c["rear_n"], 1)})
    y = np.concatenate([c["per_copy_crashes"] for c in res["cells"].values()])
    t = np.concatenate([c["per_copy_hours"] for c in res["cells"].values()])
    lm = np.concatenate([[np.log(c["mu"])] * len(c["per_copy_crashes"]) for c in res["cells"].values()])
    best = None
    for h in sorted({np.log(c["mu"]) for c in res["cells"].values()})[1:-1]:
        Xd = np.stack([np.ones_like(lm), lm - h, np.maximum(h - lm, 0.0)], 1)  # slope above the hinge, extra slope below it
        b, cov, phi = irls(y, t, Xd)
        mu_ = t * np.exp(Xd @ b)
        dev = 2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-9) / mu_), 0) - (y - mu_))
        if best is None or dev < best["deviance"]:
            best = {"hinge_mu": float(np.exp(h)), "slope_above": float(b[1]), "extra_slope_below": float(-b[2]), "extra_se": float(np.sqrt(cov[2, 2])), "deviance": float(dev)}
    lo = [r for r in rows if r["mu"] <= 0.5]; hi = [r for r in rows if r["mu"] >= 0.6]
    share = lambda rs: sum(r["rear_share"] * r["crashes"] for r in rs) / max(sum(r["crashes"] for r in rs), 1)
    return {"rows": rows, "hinge": best, "rear_share_mu_ge_0.6": share(hi), "rear_share_mu_le_0.5": share(lo)}


def summarize_limit(res):
    from avlab.friction import irls
    out = {"per_cell": {}}
    y, t, lsc, sh = [], [], [], []
    for k, c in res["cells"].items():
        out["per_cell"][k] = {"crashes": sum(c["crashes"]), "rate": 1000 * sum(c["crashes"]) / sum(c["hours"]), "ped": c["ped"], "rear": c["rear"]}
        y += c["crashes"]; t += c["hours"]; lsc += [np.log(c["scale"])] * len(c["crashes"]); sh += [c["p_av"] > 0] * len(c["crashes"])
    y, t, lsc, sh = map(np.array, (y, t, lsc, sh))
    for name, m in (("humans", ~sh), ("half_AVs", sh)):
        b, cov, phi = irls(y[m], t[m], np.stack([np.ones(m.sum()), lsc[m] - lsc[m].mean()], 1))
        out[name] = {"elasticity": float(b[1]), "se": float(np.sqrt(cov[1, 1])), "dispersion": phi}
    Xd = np.stack([np.ones_like(lsc), sh.astype(float), lsc - lsc.mean(), (lsc - lsc.mean()) * sh], 1)
    b, cov, phi = irls(y, t, Xd)
    out["interaction"] = {"coef": float(b[3]), "se": float(np.sqrt(cov[3, 3])), "dispersion": phi}
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    C = json.load(open(f"{OUT}/cliff.json"))["summary"]
    L = json.load(open(f"{OUT}/limit.json"))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG); a.tick_params(colors=FG); [s.set_color(GR) for s in a.spines.values()]
        a.xaxis.label.set_color(FG); a.yaxis.label.set_color(FG); a.title.set_color(FG)
    r = C["rows"]; x = [q["mu"] for q in r]
    C0 = json.load(open(f"{OUT}/cliff.json"))["cells"]
    bot = np.zeros(len(r))
    for typ, col in (("rear-end", "#e69f00"), ("pedestrian", "#56b4e9"), ("merge", "#009e73"), ("bicycle", "#cc79a7"), ("other", "#8b949e")):
        v = np.array([100 * (C0[str(q["mu"])]["types"][typ] if typ != "other" else C0[str(q["mu"])]["crashes"] - sum(C0[str(q["mu"])]["types"][t] for t in ("rear-end", "pedestrian", "merge", "bicycle"))) / max(q["crashes"], 1) for q in r])
        ax[0].bar(range(len(r)), v, bottom=bot, color=col, label=typ); bot += v
    ax[0].set_xticks(range(len(r))); ax[0].set_xticklabels([str(v) for v in x])
    for i, q in enumerate(r):
        ax[0].text(i, 101, f"{q['rate']:.0f}", color=FG, ha="center", fontsize=9)
    ax[0].text(len(r) / 2 - 0.5, 110, "crashes per 1000 vehicle-h above bars", color=FG, ha="center", fontsize=9)
    ax[0].set_ylim(0, 135); ax[0].set_xlabel("road friction"); ax[0].set_ylabel("share of crashes (%)"); ax[0].set_title("Which crashes make the cliff")
    ax[0].legend(facecolor=BG, labelcolor=FG, edgecolor=GR, loc="upper center", fontsize=8, ncol=5, bbox_to_anchor=(0.5, 1.0))
    for p, col, lab in ((0.0, "#56b4e9", "humans only"), (0.5, "#e69f00", "half AVs")):
        cs = sorted([c for c in L["cells"].values() if c["p_av"] == p], key=lambda c: c["scale"])
        xs = [c["scale"] for c in cs]; ys = [1000 * sum(c["crashes"]) / sum(c["hours"]) for c in cs]
        ax[1].plot(xs, ys, "o-", color=col); ax[1].text(xs[0], ys[0] * (1.25 if p == 0 else 0.8), lab, color=col, va="center", fontsize=10)
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xticks(SCALES); ax[1].set_xticklabels([str(s) for s in SCALES]); ax[1].minorticks_off()
    ax[1].set_xlim(0.8, 1.7); ax[1].set_xlabel("speed limit scale"); ax[1].set_ylabel("crashes per 1000 vehicle-hours"); ax[1].set_title("Speed limit, with and without AVs")
    fig.tight_layout(); fig.savefig("docs/img/cliffcause.png", dpi=130, facecolor=BG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cliff", "limit", "plot"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == "plot":
        return plot()
    q = dict(E=2, seconds=60.0) if a.quick else {}
    if a.what == "cliff":
        res = run_cliff(**q, **(dict(mus=(0.6, 0.4)) if a.quick else {})); res["summary"] = summarize_cliff(res) if not a.quick else None
    else:
        res = run_limit(**q, **(dict(scales=(1.0, 1.3)) if a.quick else {})); res["summary"] = summarize_limit(res)
    print(json.dumps(res["summary"], indent=1)[:2500])
    if not a.quick:
        json.dump(res, open(f"{OUT}/{a.what}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
