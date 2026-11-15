"""The friction knee on a fourth random stream: was stream 3's high 0.7 cell a fluke?

2026-11-14 (`avlab.kneepool`) found the knee sharp when three random streams are pooled (extra slope 2.9 below friction 0.7, 95 %
1.7-4.1), but stream 3 alone did not show it: its 0.7 cell (15.6 crashes per 1000 vehicle-h) sat *above* its 0.6 cell (13.2).
This module runs a fourth stream (seed 4) concentrated where the question is, 32 copies at friction 0.85 / 0.7 / 0.6 (humans only,
calibrated drivers, the protocol of `avlab.knee.replicate`: warm 120 s, 600 s scored), then

    stream 4 alone    0.7 vs 0.6 exact conditional Poisson rate ratio (is 0.7 below 0.6?); fixed hinge 0.7 (one-sided)
    four streams      fixed hinge 0.7 (one-sided), sup over the hinge grid (parametric bootstrap), and heterogeneity
                      (one extra slope per stream vs one common, LR chi^2 with 3 df)

    python -m avlab.knee4 --quick     # 3 copies x 2 frictions, 120 s (smoke test) -> results/knee/stream4_quick.json
    python -m avlab.knee4 --run       # 32 copies x 3 frictions (~6 min on 4 cores) -> results/knee/stream4.json
    python -m avlab.knee4             # tests on saved data -> results/knee/knee4.json, docs/img/knee4.png
"""
import argparse
import json
import os

import numpy as np

from avlab import knee as K
from avlab import kneepool as KP

OUT = "results/knee"
MUS4 = (0.85, 0.7, 0.6)
SEED4 = 4


def load_all4(path=f"{OUT}/stream4.json"):
    """(y, t, mu, run, stream): kneepool's three streams (runs 0-3) plus stream 4 as run 4 / stream 4."""
    y, t, mu, run, stream = KP.load_all()
    y4, t4, mu4, _ = K.load_replication(path)
    return (np.concatenate([y, y4]), np.concatenate([t, t4]), np.concatenate([mu, mu4]),
            np.concatenate([run, np.full(len(y4), 4.0)]), np.concatenate([stream, np.full(len(y4), 4.0)]))


def analyse(path=f"{OUT}/stream4.json", B=1999):
    from avlab.whocrashes import holm, rate_ratio
    out = {}
    y4, t4, mu4, run4 = K.load_replication(path)
    c4 = K.cells(y4, t4, mu4)
    out["stream4"] = {"copies": int(len(y4)), "crashes": float(y4.sum()), "cells": c4,
                      "rr_0.7_vs_0.6": rate_ratio(c4["0.7"]["crashes"], c4["0.7"]["hours"], c4["0.6"]["crashes"], c4["0.6"]["hours"]),
                      "rr_0.85_vs_0.7": rate_ratio(c4["0.85"]["crashes"], c4["0.85"]["hours"], c4["0.7"]["crashes"], c4["0.7"]["hours"]),
                      "fixed_0.7": K.fixed_hinge(y4, t4, mu4, run4, 0.7)}
    # stream 3 restricted to the same three frictions, for a like-for-like comparison of the 0.7 cell
    y3, t3, mu3, _ = KP.load_stream3()
    c3 = K.cells(y3, t3, mu3)
    out["stream3_same_cells"] = {k: c3[k] for k in ("0.85", "0.7", "0.6")}
    out["stream3_rr_0.7_vs_0.6"] = rate_ratio(c3["0.7"]["crashes"], c3["0.7"]["hours"], c3["0.6"]["crashes"], c3["0.6"]["hours"])
    y, t, mu, run, stream = load_all4(path)
    out["pooled4"] = {"copies": int(len(y)), "crashes": float(y.sum()), "fixed_0.7": K.fixed_hinge(y, t, mu, run, 0.7),
                      "sup": K.boot_sup_p(y, t, mu, run, K.FRICTION_GRID, "below", B, seed=4),
                      "heterogeneity": KP.heterogeneity(y, t, mu, run, stream)}
    out["per_stream_cells"] = {f"{int(s)}": K.cells(y[stream == s], t[stream == s], mu[stream == s]) for s in np.unique(stream)}
    ks = ["stream4_0.7_below_0.6", "stream4_fixed", "pooled4_fixed"]
    # one-sided p for "0.7 below 0.6" from the two-sided exact test: halve it when the ratio points the predicted way
    rr = out["stream4"]["rr_0.7_vs_0.6"]
    p1 = rr["p"] / 2 if rr["rr"] < 1 else 1 - rr["p"] / 2
    out["stream4"]["p_one_sided_0.7_below_0.6"] = p1
    out["holm"] = dict(zip(ks, holm([p1, out["stream4"]["fixed_0.7"]["p_one_sided"], out["pooled4"]["fixed_0.7"]["p_one_sided"]])))
    return out


def plot(out, path="docs/img/knee4.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    bg, fg, grid = "#0d1117", "#e6edf3", "#30363d"
    pal = {"0": "#4c9be8", "1": "#f2a541", "3": "#3fb68b", "4": "#e86bb3"}
    names = {"0": "stream 0", "1": "stream 1", "3": "stream 3", "4": "stream 4 (new, 32 copies/cell)"}
    fig, ax = plt.subplots(figsize=(8.4, 5.0), facecolor=bg)
    ax.set_facecolor(bg)
    for s, cells in out["per_stream_cells"].items():
        x = np.array([float(k) for k in cells]); r = np.array([c["rate"] for c in cells.values()])
        n = np.array([c["crashes"] for c in cells.values()]); h = np.array([c["hours"] for c in cells.values()])
        o = np.argsort(x); x, r, n, h = x[o], r[o], n[o], h[o]
        new = s == "4"
        if new:
            lo = 1000 * np.maximum(n - 1.96 * np.sqrt(n), 0.5) / h; hi = 1000 * (n + 1.96 * np.sqrt(n)) / h
            ax.fill_between(x, lo, hi, color=pal[s], alpha=0.18, lw=0)
        ax.plot(x, r, "o-", color=pal[s], lw=2.6 if new else 1.4, ms=6 if new else 4, alpha=1.0 if new else 0.75)
        # label at the left end (the lowest friction of that stream)
        ax.annotate(names[s], (x[0], r[0]), xytext=(-6, 0), textcoords="offset points", color=pal[s], fontsize=9, va="center", ha="right")
    f = out["pooled4"]["fixed_0.7"]
    ax.axvline(0.7, color=fg, lw=1, ls=":", alpha=0.6)
    ax.text(0.705, 0.97, "hinge 0.7", color=fg, fontsize=9, alpha=0.8, transform=ax.get_xaxis_transform(), va="top")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.85, 1.0]); ax.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.85", "1.0"])
    ax.set_xlim(0.24, 1.08); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("road friction (log)", color=fg); ax.set_ylabel("crashes per 1000 vehicle-hours (log)", color=fg)
    het = out["pooled4"]["heterogeneity"]
    rr = out["stream4"]["rr_0.7_vs_0.6"]
    ax.set_title(f"Friction knee, four random streams ({out['pooled4']['copies']} copies, humans-only Oakland)\n"
                 f"stream 4: rate at 0.7 / at 0.6 = {rr['rr']:.2f} ({rr['lo']:.2f}-{rr['hi']:.2f}); pooled extra slope below 0.7 "
                 f"{f['extra']:.1f} ({f['extra_lo']:.1f} to {f['extra_hi']:.1f}); streams differ? p {het['p']:.2f}",
                 color=fg, fontsize=9.5, loc="left")
    ax.tick_params(colors=fg, which="both"); ax.grid(color=grid, lw=0.6, which="both")
    for sp in ax.spines.values():
        sp.set_color(grid)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=bg)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.run or a.quick:
        res = K.replicate(E=3, warm_s=60.0, seconds=120.0, mus=(0.7, 0.6), seed=SEED4) if a.quick else K.replicate(E=32, mus=MUS4, seed=SEED4)
        json.dump(res, open(f"{OUT}/stream4{'_quick' if a.quick else ''}.json", "w"))
        print({k: (c["crashes"], round(c["hours"], 1)) for k, c in res["cells"].items()})
        return
    out = K._clean(analyse())
    json.dump(out, open(f"{OUT}/knee4.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("stream4", "stream3_rr_0.7_vs_0.6", "pooled4", "holm")}, indent=1)[:5000])
    plot(out)


if __name__ == "__main__":
    main()
