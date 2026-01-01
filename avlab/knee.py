"""Knee or power law? A search-corrected hinge test, and a pre-registered out-of-sample replication of the friction knee.

    python -m avlab.knee --replicate [--quick]  # NEW humans-only Oakland run on a fresh seed stream (seed 1): 36 copies at friction
                                                # 1.0 / 0.85 / 0.7 / 0.6 / 0.5 -> results/knee/replicate.json (~10 min on 4 cores)
    python -m avlab.knee                        # tests on saved data (seconds) -> results/knee/knee.json, docs/img/knee.png, docs/img/knee.gif

The pooled friction runs (2026-11-10) preferred a hinge at friction 0.7 (F-test p 0.003) but the hinge position was chosen from six
candidates on the same data, so the F p-value is too small. Two fixes:

  1. Search-corrected p-value (Davies' problem: the hinge is not identified under the null). Statistic: the largest drop in Poisson
     deviance over the grid of hinge positions. Null distribution: parametric bootstrap from the fitted power law (Poisson counts with the
     observed exposures), refitting every hinge in each draw.
  2. Out of sample: a new run with a different random stream, the hinge FIXED at 0.7 before the run. One-sided test that the slope below
     0.7 is steeper (no search, so an ordinary likelihood-ratio test applies).

The same corrected test is applied to the pooled speed-limit runs (results/speedlimit, results/cliffcause/limit.json), with the knee ABOVE
the hinge (x = limit scale): does the crash rate bend upwards at some limit, or is crash rate ~ limit^b enough?

    model   log E[y_i] = log t_i + a + r * run_i + b log x_i + c * max(+-(log h - log x_i), 0)
"""
import argparse
import json
import os

import numpy as np

OUT = "results/knee"
REP_MUS = (1.0, 0.85, 0.7, 0.6, 0.5)
FRICTION_GRID = (0.45, 0.5, 0.55, 0.6, 0.7, 0.8)   # the grid searched on 2026-11-10
LIMIT_GRID = (0.85, 1.0, 1.15)                      # interior scales of the speed-limit pool


def design(x, run, hinge=None, side="below"):
    """Columns: 1, run dummies, log x, [hinge term]. side='below': extra slope for x < hinge; 'above': for x > hinge."""
    lx = np.log(np.asarray(x, float))
    run = np.asarray(run)
    cols = [np.ones_like(lx)] + [(run == r).astype(float) for r in np.unique(run)[1:]] + [lx]
    if hinge is not None:
        h = np.log(hinge)
        cols.append(np.maximum(h - lx, 0.0) if side == "below" else np.maximum(lx - h, 0.0))
    return np.stack(cols, 1)


def pdev(y, m):
    """Poisson deviance."""
    y = np.asarray(y, float)
    return float(2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / m), 0.0) - (y - m)))


def fit(y, t, X):
    from avlab.friction import irls
    b, cov, phi = irls(y, t, X)
    m = t * np.exp(X @ b)
    return {"beta": b, "se": np.sqrt(np.diag(cov)), "phi": phi, "dev": pdev(y, m), "mean": m}


def sup_drop(y, t, x, run, grid, side):
    """Largest deviance drop of a hinge over the power law, over the hinge grid. Returns (drop, best hinge, drops per hinge)."""
    d0 = fit(y, t, design(x, run))["dev"]
    drops = []
    for h in grid:
        try:
            drops.append(d0 - fit(y, t, design(x, run, h, side))["dev"])
        except np.linalg.LinAlgError:
            drops.append(0.0)
    i = int(np.argmax(drops))
    return float(drops[i]), grid[i], [float(d) for d in drops]


def boot_sup_p(y, t, x, run, grid, side, B=499, seed=0):
    """Parametric-bootstrap p-value of the sup-deviance-drop statistic under the fitted power law (Poisson draws)."""
    rng = np.random.default_rng(seed)
    obs, h, drops = sup_drop(y, t, x, run, grid, side)
    m0 = fit(y, t, design(x, run))["mean"]
    null = np.array([sup_drop(rng.poisson(m0).astype(float), t, x, run, grid, side)[0] for _ in range(B)])
    return {"stat": obs, "hinge": h, "drops": drops, "p_boot": float((1 + np.sum(null >= obs)) / (B + 1)), "B": B,
            "null_q95": float(np.quantile(null, 0.95))}


def fixed_hinge(y, t, x, run, h, side="below"):
    """Pre-specified hinge: slopes above / below, one-sided LR test that the extra slope has the predicted sign (steeper toward the knee)."""
    from scipy import stats
    p0 = fit(y, t, design(x, run))
    p1 = fit(y, t, design(x, run, h, side))
    k = p1["beta"].shape[0]
    c, se_c = float(p1["beta"][-1]), float(p1["se"][-1])
    lr = max(p0["dev"] - p1["dev"], 0.0) / p1["phi"]
    # the predicted sign: below a friction hinge the rate rises faster as friction falls -> extra coefficient on (log h - log x) > 0
    z = np.sign(c) * np.sqrt(lr)
    b_main, se_main = float(p1["beta"][k - 2]), float(p1["se"][k - 2])
    return {"hinge": h, "slope_above": b_main if side == "below" else b_main + c, "slope_above_se": se_main,
            "extra": c, "extra_se": se_c, "extra_lo": c - 1.96 * se_c, "extra_hi": c + 1.96 * se_c,
            "lr": float(lr), "p_one_sided": float(stats.norm.sf(z)), "phi": p1["phi"],
            "power_b": float(p0["beta"][-1]), "power_b_se": float(p0["se"][-1])}


def load_friction_pool():
    from avlab.frictionpool import load
    y, t, mu, run = load()
    return y, t, mu, run


def load_limit_pool():
    """Per-copy humans-only crashes and hours: speedlimit.json (32 copies x 5 scales, run 0) + limit.json p_av=0 (12 x 3, run 1)."""
    a = json.load(open("results/speedlimit/speedlimit.json"))
    b = json.load(open("results/cliffcause/limit.json"))
    y, t, s, run = [], [], [], []
    for k, c in a["scales"].items():
        y += c["crashes"]; t += (np.asarray(c["veh_s"]) / 3600).tolist(); s += [float(k)] * len(c["crashes"]); run += [0] * len(c["crashes"])
    for c in b["cells"].values():
        if c["p_av"] == 0.0:
            y += c["crashes"]; t += c["hours"]; s += [c["scale"]] * len(c["crashes"]); run += [1] * len(c["crashes"])
    return np.array(y, float), np.array(t, float), np.array(s, float), np.array(run, float)


def load_replication(path=f"{OUT}/replicate.json"):
    r = json.load(open(path))
    y, t, mu = [], [], []
    for c in r["cells"].values():
        y += c["per_copy_crashes"]; t += c["per_copy_hours"]; mu += [c["mu"]] * len(c["per_copy_crashes"])
    return np.array(y, float), np.array(t, float), np.array(mu, float), np.zeros(len(y))


def replicate(E=36, warm_s=120.0, seconds=600.0, mus=REP_MUS, seed=1, area="oakland"):
    """The new humans-only run: same protocol as avlab.cliffcause.run_cliff, but a fresh random stream (seed 1)."""
    from examples import _common as C
    from xnet import sim as X
    behs, tags = [], []
    for mu in mus:
        behs += [C.calibrated(friction=mu) for _ in range(E)]; tags += [mu] * E
    m = C.run_copies(area, behs, warm_s=warm_s, seconds=seconds, seed=seed)
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    tags = np.array(tags)
    res = {"area": area, "copies": E, "seconds": seconds, "warm_s": warm_s, "seed": seed, "cells": {}}
    for mu in mus:
        s = tags == mu
        types = {k: float(d[k][s].sum()) for k in X.CRASH_NAMES}
        res["cells"][str(mu)] = {"mu": mu, "types": types, "crashes": float(sum(types.values())), "hours": float(d["veh_s"][s].sum() / 3600),
                                 "per_copy_crashes": sum(d[k][s] for k in X.CRASH_NAMES).tolist(), "per_copy_hours": (d["veh_s"][s] / 3600).tolist()}
    return res


def cells(y, t, x):
    return {f"{v:g}": {"crashes": float(y[x == v].sum()), "hours": float(t[x == v].sum()), "rate": float(1000 * y[x == v].sum() / t[x == v].sum())}
            for v in np.unique(x)}


def analyse(B=499):
    out = {}
    y, t, mu, run = load_friction_pool()
    out["friction_pool"] = {"copies": int(len(y)), "crashes": float(y.sum()), "sup": boot_sup_p(y, t, mu, run, FRICTION_GRID, "below", B),
                            "fixed_0.7": fixed_hinge(y, t, mu, run, 0.7), "cells": cells(y, t, mu)}
    y, t, s, run = load_limit_pool()
    out["limit_pool"] = {"copies": int(len(y)), "crashes": float(y.sum()), "sup": boot_sup_p(y, t, s, run, LIMIT_GRID, "above", B, seed=1),
                         "power_b": fixed_hinge(y, t, s, run, 1.0, "above")["power_b"], "power_b_se": fixed_hinge(y, t, s, run, 1.0, "above")["power_b_se"],
                         "fixed_1.0": fixed_hinge(y, t, s, run, 1.0, "above"), "cells": cells(y, t, s)}
    if os.path.exists(f"{OUT}/replicate.json"):
        y, t, mu, run = load_replication()
        out["replication"] = {"copies": int(len(y)), "crashes": float(y.sum()), "fixed_0.7": fixed_hinge(y, t, mu, run, 0.7),
                              "sup": boot_sup_p(y, t, mu, run, (0.6, 0.7, 0.85), "below", B, seed=2), "cells": cells(y, t, mu)}
    from avlab.whocrashes import holm
    keys = [k for k in ("friction_pool", "limit_pool", "replication") if k in out]
    ps = [out["replication"]["fixed_0.7"]["p_one_sided"] if k == "replication" else out[k]["sup"]["p_boot"] for k in keys]
    out["holm"] = dict(zip(keys, holm(ps)))
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "mean"}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicate", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.replicate:
        res = replicate(E=4, warm_s=60.0, seconds=120.0, mus=(1.0, 0.6)) if a.quick else replicate()
        json.dump(res, open(f"{OUT}/replicate{'_quick' if a.quick else ''}.json", "w"))
        print({k: (c["crashes"], round(c["hours"], 1)) for k, c in res["cells"].items()})
        return
    out = _clean(analyse(B=99 if a.quick else 1999))
    json.dump(out, open(f"{OUT}/knee.json", "w"), indent=1)
    print(json.dumps(out, indent=1)[:4000])
    from avlab.knee_plot import plot, gif
    plot(out)
    gif(out)


if __name__ == "__main__":
    main()
