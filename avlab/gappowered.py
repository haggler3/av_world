"""Gap caution on ice, a powered test: 120 copies per cell on a third random stream (seed 8).

2026-11-15 (seed 6, `avlab.frictionmerge`) and 2026-11-16 (seed 7, `avlab.gapconfirm`) both found fewer AV crashes for an AV that
accepts only longer gaps at friction 0.4 (rate ratios 0.54 and 0.63), but neither passed its bar after correction; 40 copies per
cell detect a 0.5 ratio, not 0.6. This module runs ONE pre-registered test with three times the copies
(results/daily/2026-11-17/predictions.md):

    friction 0.4   blind (gsc 1.15)  vs  gap (gsc 1.495)   120 copies each, seed 8   <- confirmatory, two-sided p < 0.05

and then, labelled post hoc, a fixed-effect pooling of the three streams (Mantel-Haenszel style: stratified exact test).
xnet/ is untouched: the run is `avlab.gapconfirm.run` with other cells and seed.

    python -m avlab.gappowered --quick   # 2 copies per cell, short (smoke test)
    python -m avlab.gappowered           # -> results/frictionmerge/gappowered.json (~9 min on 4 cores)
    python -m avlab.gappowered --plot    # -> docs/img/gappowered.png
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avlab import gapconfirm as G  # noqa: E402

OUT = "results/frictionmerge"
CELLS = (("blind@0.4", 0.4, False, 120), ("gap@0.4", 0.4, True, 120))
SEED = 8


def stratified_rr(strata):
    """Common rate ratio over strata [(k_a, t_a, k_b, t_b), ...] (exposure-weighted fixed effect, Mantel-Haenszel for rates),
    with an exact conditional test: given each stratum's total n_s, k_a,s ~ Bin(n_s, p_s(RR)); the sum of k_a is tested by
    convolution, and the interval inverts that test (two one-sided 2.5 % tails)."""
    from scipy import optimize, stats
    S = [(float(a), float(ta), float(b), float(tb)) for a, ta, b, tb in strata]
    num = sum(a * tb / (ta + tb) for a, ta, b, tb in S); den = sum(b * ta / (ta + tb) for a, ta, b, tb in S)
    rr = num / den if den > 0 else float("inf")
    kobs = int(round(sum(s[0] for s in S)))

    def pmf(r):
        d = np.array([1.0])
        for a, ta, b, tb in S:
            n = int(round(a + b)); p = r * ta / (r * ta + tb)
            d = np.convolve(d, stats.binom.pmf(np.arange(n + 1), n, p))
        return d

    def tails(r):
        d = pmf(r)
        return float(d[kobs:].sum()), float(d[:kobs + 1].sum())  # P(K >= k), P(K <= k)

    up, lo_t = tails(1.0)
    p = min(1.0, 2 * min(up, lo_t))
    lo = optimize.brentq(lambda r: tails(r)[0] - 0.025, 1e-4, 1e4) if kobs > 0 else 0.0
    hi = optimize.brentq(lambda r: tails(r)[1] - 0.025, 1e-4, 1e4)
    return {"rr": rr, "lo": float(lo), "hi": float(hi), "p": p, "n": int(round(sum(s[0] + s[2] for s in S)))}


def cochran_q(strata):
    """Do the strata's rate ratios differ? Cochran's Q on log rate ratios (variance 1/a + 1/b), chi^2 with S-1 df."""
    from scipy import stats
    lr = np.array([np.log((a / ta) / (b / tb)) for a, ta, b, tb in strata]); w = np.array([1 / (1 / a + 1 / b) for a, _, b, _ in strata])
    q = float(np.sum(w * (lr - np.sum(w * lr) / w.sum()) ** 2))
    return {"Q": q, "df": len(strata) - 1, "p": float(stats.chi2.sf(q, len(strata) - 1))}


def _cells_015(path="results/frictionmerge/frictionmerge.json"):
    c = json.load(open(path))["cells"]
    return {"blind@0.4": c["blind"], "gap@0.4": c["gap"]}


def pooled(res, keys=("av_crashes", "av_merges")):
    """Post hoc: seeds 6 (2026-11-15), 7 (2026-11-16) and 8 (today), friction 0.4, gap vs blind."""
    tot = lambda c, k: float(np.sum(c[k]))  # noqa: E731
    streams = {6: _cells_015(), 7: json.load(open(f"{OUT}/gapconfirm.json"))["cells"], 8: res["cells"]}
    out = {}
    for k in keys:
        S = [(tot(c["gap@0.4"], k), tot(c["gap@0.4"], "av_hours"), tot(c["blind@0.4"], k), tot(c["blind@0.4"], "av_hours"))
             for c in streams.values()]
        out[k] = {"per_stream": {str(s): {"gap": a, "gap_h": ta, "blind": b, "blind_h": tb} for s, (a, ta, b, tb) in zip(streams, S)},
                  "pooled": stratified_rr(S), "heterogeneity": cochran_q(S)}
    return out


def summarise(res):
    G.PAIRS, G.CONFIRMATORY = (("gap@0.4", "blind@0.4"),), "gap@0.4 vs blind@0.4"
    s = G.summarise(res)
    s["posthoc_pooled_3_streams"] = pooled(res) if res.get("seed") == SEED and os.path.exists(f"{OUT}/gapconfirm.json") else None
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        from avlab.gappowered_plot import plot
        plot(json.load(open(f"{OUT}/gappowered.json")))
        return
    res = G.run(cells=CELLS, scale=1 / 60, warm_s=40.0, seconds=60.0, seed=SEED) if a.quick else G.run(cells=CELLS, seed=SEED)
    res["summary"] = summarise(res)
    json.dump(res, open(f"{OUT}/gappowered{'_quick' if a.quick else ''}.json", "w"), indent=1)
    print(json.dumps(res["summary"], indent=1)[:4000])
    print("wall", res["wall_s"], "s")


if __name__ == "__main__":
    main()
