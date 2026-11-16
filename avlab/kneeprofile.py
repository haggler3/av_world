"""Where is the friction knee, and is it a knee at all? A profile likelihood for the hinge position on four pooled streams.

2026-11-15 (`avlab.knee4`) pooled four random streams (524 humans-only Oakland copies, 880 crashes) and found an extra log-log slope
below friction 0.7 of 2.8 (95 % 1.6-3.9), but the searched hinge moved to 0.6. No simulation here, only the saved runs:

    profile     for each hinge h on a fine grid: deviance drop of  log rate = a_run + b log mu + c max(log h - log mu, 0)
                over the plain power law; the 'likelihood set' {h : drop(h) >= max drop - 3.84} (approximate: the hinge is not a
                regular parameter, so this is a descriptive set, not a calibrated interval)
    bootstrap   resample copies with replacement within each (run, friction) cell, B times; record the best hinge -> percentile interval
    knee/curve  is a kink needed, or is the curve just convex?  T = dev(quadratic in log mu) - dev(best hinge), both one extra shape
                parameter beyond the power law (the hinge also picks its position). p from a parametric bootstrap under the fitted
                quadratic (Poisson draws). Also: curvature at all, quadratic vs power law (LR chi^2, 1 df). Holm over the two.

All covariates are cell-level, so copies are summed into (run, friction) cells before fitting; Poisson deviance differences are unchanged.

    python -m avlab.kneeprofile           # -> results/knee/kneeprofile.json, docs/img/kneeprofile.png, docs/img/kneeprofile.gif
    python -m avlab.kneeprofile --quick   # B = 99 (seconds)
"""
import argparse
import json
import os

import numpy as np

from avlab import knee as K

OUT = "results/knee"
GRID = tuple(np.round(np.arange(0.42, 0.961, 0.01), 3))


def aggregate(y, t, mu, run):
    """Sum copies into (run, friction) cells. Returns (y, t, mu, run) of the cells."""
    keys = np.stack([np.asarray(run, float), np.round(np.asarray(mu, float), 4)], 1)
    u, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    return (np.bincount(inv, weights=y), np.bincount(inv, weights=t), u[:, 1], u[:, 0])


def design_quad(mu, run):
    X = K.design(mu, run)
    lx = np.log(mu) - np.log(0.7)
    return np.concatenate([X, (lx ** 2)[:, None]], 1)


def dev(y, t, X):
    return K.fit(y, t, X)["dev"]


def profile(y, t, mu, run, grid=GRID):
    d0 = dev(y, t, K.design(mu, run))
    drops = np.array([d0 - dev(y, t, K.design(mu, run, h)) for h in grid])
    return d0, drops


def likelihood_set(grid, drops, cut=3.84):
    keep = [h for h, d in zip(grid, drops) if d >= drops.max() - cut]
    return float(min(keep)), float(max(keep))


def knee_vs_curve(y, t, mu, run, grid=GRID):
    """T = dev(quadratic) - dev(best hinge); > 0 means the kink fits better."""
    d0, drops = profile(y, t, mu, run, grid)
    dq = dev(y, t, design_quad(mu, run))
    return float(dq - (d0 - drops.max())), float(d0 - dq), grid[int(np.argmax(drops))]


def analyse(B=999, seed=16):
    from scipy import stats
    from avlab.knee4 import load_all4
    from avlab.whocrashes import holm
    yc, tc, muc, runc, _ = load_all4()
    y, t, mu, run = aggregate(yc, tc, muc, runc)
    d0, drops = profile(y, t, mu, run)
    best = GRID[int(np.argmax(drops))]
    lo, hi = likelihood_set(GRID, drops)
    T, qdrop, _ = knee_vs_curve(y, t, mu, run)
    rng = np.random.default_rng(seed)
    # copy bootstrap of the best hinge (within cells)
    cell = np.unique(np.stack([runc, np.round(muc, 4)], 1), axis=0, return_inverse=True)[1].ravel()
    groups = [np.flatnonzero(cell == k) for k in np.unique(cell)]
    coarse = GRID[::2]
    hb = []
    for _ in range(B):
        ii = np.concatenate([rng.choice(g, len(g)) for g in groups])
        yb, tb, mb, rb = aggregate(yc[ii], tc[ii], muc[ii], runc[ii])
        _, db = profile(yb, tb, mb, rb, coarse)
        hb.append(coarse[int(np.argmax(db))])
    hb = np.array(hb)
    # parametric bootstrap of T under the quadratic (no kink) and under the best hinge (power check)
    mq = K.fit(y, t, design_quad(mu, run))["mean"]
    mh = K.fit(y, t, K.design(mu, run, best))["mean"]
    Tq = np.array([knee_vs_curve(rng.poisson(mq).astype(float), t, mu, run, coarse)[0] for _ in range(B)])
    Th = np.array([knee_vs_curve(rng.poisson(mh).astype(float), t, mu, run, coarse)[0] for _ in range(B)])
    p_knee = float((1 + np.sum(Tq >= T)) / (B + 1))
    p_curv = float(stats.chi2.sf(qdrop, 1))
    ph = holm([p_knee, p_curv])
    fits = {f"{h}": K.fixed_hinge(y, t, mu, run, h) for h in (0.6, 0.7, 0.8)}
    cells = {}
    for m_ in np.unique(mu):
        s = mu == m_
        cells[f"{m_:g}"] = {"crashes": float(y[s].sum()), "hours": float(t[s].sum()), "rate": float(1000 * y[s].sum() / t[s].sum())}
    return {"copies": int(len(yc)), "crashes": float(yc.sum()), "cells_rows": int(len(y)), "grid": list(map(float, GRID)),
            "drops": drops.tolist(), "best_hinge": float(best), "max_drop": float(drops.max()), "likelihood_set_3.84": [lo, hi],
            "boot_hinge": {"B": B, "q025": float(np.quantile(hb, 0.025)), "q975": float(np.quantile(hb, 0.975)),
                           "median": float(np.median(hb)), "share_in_0.6_0.8": float(np.mean((hb >= 0.6) & (hb <= 0.8))),
                           "hist": {f"{h:g}": int(np.sum(hb == h)) for h in coarse}},
            "quad_drop_vs_power": qdrop, "T_hinge_minus_quad": T,
            "T_null_quad_q95": float(np.quantile(Tq, 0.95)), "p_knee_vs_curve": p_knee,
            "T_under_hinge_median": float(np.median(Th)), "power_T_gt_q95": float(np.mean(Th > np.quantile(Tq, 0.95))),
            "p_curvature": p_curv, "holm": {"knee_vs_curve": ph[0], "curvature": ph[1]}, "fixed_hinges": fits, "pooled_cells": cells}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = K._clean(analyse(B=99 if a.quick else 999))
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/kneeprofile{'_quick' if a.quick else ''}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("grid", "drops", "fixed_hinges")}, indent=1))
    if not a.quick:
        from avlab.kneeprofile_plot import gif, plot
        plot(out); gif(out)


if __name__ == "__main__":
    main()
