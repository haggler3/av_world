"""The friction knee on all three random streams: is the extra slope below friction 0.7 the same on each stream?

No new simulation. It pools the per-copy humans-only crash counts already saved:

    stream 0  results/friction/friction.json (p_av 0) + results/cliffcause/cliff.json      (avlab.frictionpool, run dummies 0, 1)
    stream 1  results/knee/replicate.json                                                   (seed 1, mu 1.0 ... 0.5)
    stream 3  results/stopline/stopline.json                                                (seed 3, mu 1.0 ... 0.4, 2026-11-13)

and fits the Poisson model of avlab.knee: log rate = run dummy + b log mu + c max(log 0.7 - log mu, 0) (quasi-Poisson SEs).

    tests  stream 3 alone (all mu; and mu >= 0.5, the replication's range)  fixed hinge 0.7, one-sided
           all streams pooled                                               fixed hinge 0.7, one-sided; sup over the hinge grid (bootstrap)
           heterogeneity                                                    LR test of a separate c per stream (df = streams - 1)

    python -m avlab.kneepool [--quick]   -> results/knee/kneepool.json, docs/img/kneepool.png
"""
import argparse
import json
import os

import numpy as np

from avlab import knee as K

OUT = "results/knee"


def load_stream3(path="results/stopline/stopline.json", mu_min=0.0):
    r = json.load(open(path))
    y, t, mu = [], [], []
    for c in r["cells"].values():
        if c["mu"] >= mu_min - 1e-9:
            y += c["per_copy_crashes"]; t += c["per_copy_hours"]; mu += [c["mu"]] * len(c["per_copy_crashes"])
    return np.array(y, float), np.array(t, float), np.array(mu, float), np.zeros(len(y))


def load_all():
    """(y, t, mu, run, stream): run 0, 1 = stream 0's two files, 2 = stream 1, 3 = stream 3."""
    parts = [K.load_friction_pool(), K.load_replication(), load_stream3()]
    y = np.concatenate([p[0] for p in parts]); t = np.concatenate([p[1] for p in parts]); mu = np.concatenate([p[2] for p in parts])
    run = np.concatenate([parts[0][3], np.full(len(parts[1][0]), 2.0), np.full(len(parts[2][0]), 3.0)])
    stream = np.where(run <= 1, 0, np.where(run == 2, 1, 3)).astype(float)
    return y, t, mu, run, stream


def heterogeneity(y, t, mu, run, stream, h=0.7):
    """LR test (quasi-Poisson, phi from the larger model) of one hinge slope per stream against one common hinge slope."""
    from scipy import stats
    X1 = K.design(mu, run, h)
    hin = X1[:, -1]
    S = np.unique(stream)
    X2 = np.concatenate([X1[:, :-1]] + [(hin * (stream == s))[:, None] for s in S], 1)
    f1, f2 = K.fit(y, t, X1), K.fit(y, t, X2)
    lr = max(f1["dev"] - f2["dev"], 0.0) / max(f2["phi"], 1.0)
    df = len(S) - 1
    per = {f"{int(s)}": {"extra": float(f2["beta"][-len(S) + i]), "se": float(f2["se"][-len(S) + i])} for i, s in enumerate(S)}
    return {"lr": float(lr), "df": df, "p": float(stats.chi2.sf(lr, df)), "per_stream": per}


def analyse(B=1999):
    out = {}
    y, t, mu, run = load_stream3()
    out["stream3_all"] = {"copies": int(len(y)), "crashes": float(y.sum()), "fixed_0.7": K.fixed_hinge(y, t, mu, run, 0.7), "cells": K.cells(y, t, mu)}
    y5, t5, mu5, run5 = load_stream3(mu_min=0.5)
    out["stream3_ge0.5"] = {"copies": int(len(y5)), "crashes": float(y5.sum()), "fixed_0.7": K.fixed_hinge(y5, t5, mu5, run5, 0.7)}
    y, t, mu, run, stream = load_all()
    out["pooled"] = {"copies": int(len(y)), "crashes": float(y.sum()), "fixed_0.7": K.fixed_hinge(y, t, mu, run, 0.7),
                     "sup": K.boot_sup_p(y, t, mu, run, K.FRICTION_GRID, "below", B, seed=3),
                     "heterogeneity": heterogeneity(y, t, mu, run, stream)}
    # the same pooled tests without mu 0.4 (only stream 3 has it): is the pooled knee carried by the new low end?
    k = mu >= 0.45
    out["pooled_no0.4"] = {"copies": int(k.sum()), "fixed_0.7": K.fixed_hinge(y[k], t[k], mu[k], run[k], 0.7)}
    out["per_stream_cells"] = {f"{int(s)}": K.cells(y[stream == s], t[stream == s], mu[stream == s]) for s in np.unique(stream)}
    from avlab.whocrashes import holm
    ks = ["stream3_all", "pooled"]
    out["holm"] = dict(zip(ks, holm([out["stream3_all"]["fixed_0.7"]["p_one_sided"], out["pooled"]["fixed_0.7"]["p_one_sided"]])))
    return out


def plot(out, path="docs/img/kneepool.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    bg, fg, grid = "#0d1117", "#e6edf3", "#30363d"
    pal = {"0": "#4c9be8", "1": "#f2a541", "3": "#3fb68b"}
    names = {"0": "stream 0 (2 runs, 11-06/11-08)", "1": "stream 1 (11-11)", "3": "stream 3 (11-13, new here)"}
    fig, ax = plt.subplots(figsize=(8.2, 5.0), facecolor=bg)
    ax.set_facecolor(bg)
    for s, cells in out["per_stream_cells"].items():
        x = np.array([float(k) for k in cells]); r = np.array([c["rate"] for c in cells.values()])
        n = np.array([c["crashes"] for c in cells.values()]); h = np.array([c["hours"] for c in cells.values()])
        o = np.argsort(x); x, r, n, h = x[o], r[o], n[o], h[o]
        lo = 1000 * np.maximum(n - 1.96 * np.sqrt(n), 0.5) / h; hi = 1000 * (n + 1.96 * np.sqrt(n)) / h
        ax.fill_between(x, lo, hi, color=pal[s], alpha=0.13, lw=0)
        ax.plot(x, r, "o-", color=pal[s], lw=2, ms=5)
        ax.annotate(names[s], (x[-1], r[-1] * {"0": 1.25, "1": 0.8, "3": 1.0}[s]), xytext=(8, 0), textcoords="offset points", color=pal[s], fontsize=9, va="center")
    f = out["pooled"]["fixed_0.7"]
    ax.axvline(0.7, color=fg, lw=1, ls=":", alpha=0.6)
    ax.text(0.71, 150, "hinge 0.7", color=fg, fontsize=9, alpha=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0]); ax.set_xticklabels(["0.3", "0.4", "0.5", "0.6", "0.7", "0.85", "1.0"])
    ax.set_xlim(0.28, 2.2); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("road friction (log)", color=fg); ax.set_ylabel("crashes per 1000 vehicle-hours (log)", color=fg)
    het = out["pooled"]["heterogeneity"]
    ax.set_title(f"Friction knee on three random streams, humans-only Oakland ({out['pooled']['copies']} copies)\n"
                 f"pooled extra slope below 0.7: {f['extra']:.1f} (95% {f['extra_lo']:.1f} to {f['extra_hi']:.1f}), "
                 + ("one-sided p < 0.001" if f['p_one_sided'] < 0.001 else f"one-sided p {f['p_one_sided']:.3f}") + f"; streams differ? p {het['p']:.2f}", color=fg, fontsize=10, loc="left")
    ax.tick_params(colors=fg, which="both"); ax.grid(color=grid, lw=0.6, which="both")
    for sp in ax.spines.values():
        sp.set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=bg)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = K._clean(analyse(B=99 if a.quick else 1999))
    json.dump(out, open(f"{OUT}/kneepool{'_quick' if a.quick else ''}.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("stream3_all", "stream3_ge0.5", "pooled", "pooled_no0.4", "holm")}, indent=1)[:5000])
    if not a.quick:
        plot(out)


if __name__ == "__main__":
    main()
