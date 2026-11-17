"""Place the friction knee with data, not models: a fifth random stream at the two empty frictions, 0.65 and 0.75.

2026-11-16 (`avlab.kneeprofile`) found the knee real as curvature but not placed: every hinge 0.56-0.84 fits within 3.84 deviance,
and no run had friction between 0.6 and 0.7 or between 0.7 and 0.8. This module runs stream 5 (seed 5, humans-only Oakland,
calibrated drivers, the protocol of `avlab.knee.replicate`: warm 120 s, 600 s scored) with 48 copies at 0.75 and 0.65, then

    stream 5 alone   rate ratio 0.65 / 0.75 against what each four-stream fit predicted (hinge 0.64: 1.16, 0.7: 1.33, 0.8: 1.59,
                     power law: 1.49); which model predicts stream 5 best (Poisson log-likelihood of its two cells, levels free)
    five streams     the profile, copy bootstrap and knee-vs-curve test of `avlab.kneeprofile` on all five streams

    python -m avlab.knee5 --quick   # 3 copies x 2 frictions, 120 s (smoke test)
    python -m avlab.knee5 --run     # 48 copies x 2 frictions (~5 min on 4 cores) -> results/knee/stream5.json
    python -m avlab.knee5           # analysis on saved data -> results/knee/knee5.json, docs/img/knee5.png
"""
import argparse
import json
import os

import numpy as np

from avlab import knee as K
from avlab import kneeprofile as P

OUT = "results/knee"
MUS5 = (0.75, 0.65)
SEED5 = 5
MODELS = {"power law": None, "hinge 0.64": 0.64, "hinge 0.7": 0.7, "hinge 0.8": 0.8}


def load_all5(path=f"{OUT}/stream5.json"):
    from avlab.knee4 import load_all4
    y, t, mu, run, stream = load_all4()
    y5, t5, mu5, _ = K.load_replication(path)
    return (np.concatenate([y, y5]), np.concatenate([t, t5]), np.concatenate([mu, mu5]),
            np.concatenate([run, np.full(len(y5), 5.0)]), np.concatenate([stream, np.full(len(y5), 5.0)]))


def predicted_ratio(y, t, mu, run, h, a=0.65, b=0.75):
    """Rate ratio at friction a over friction b implied by the power law (h None) or the hinge-h model fitted to (y, t, mu, run)."""
    beta = K.fit(y, t, K.design(mu, run, h))["beta"]
    lr = np.log(a) - np.log(b)
    if h is None:
        return float(np.exp(beta[-1] * lr))
    hl = max(np.log(h) - np.log(a), 0.0) - max(np.log(h) - np.log(b), 0.0)
    return float(np.exp(beta[-2] * lr + beta[-1] * hl))


def two_cell_loglik(k_a, t_a, k_b, t_b, ratio):
    """Poisson log-likelihood of two cells whose rate ratio is fixed at `ratio`, the common level profiled out (up to a constant)."""
    lam_b = (k_a + k_b) / (ratio * t_a + t_b)
    m_a, m_b = ratio * t_a * lam_b, t_b * lam_b
    return float(k_a * np.log(m_a) - m_a + k_b * np.log(m_b) - m_b)


def analyse(path=f"{OUT}/stream5.json", B=999):
    from avlab.knee4 import load_all4
    from avlab.whocrashes import rate_ratio
    y4, t4, mu4, run4, _ = load_all4()
    y4, t4, mu4, run4 = P.aggregate(y4, t4, mu4, run4)
    y5, t5, mu5, _ = K.load_replication(path)
    c5 = K.cells(y5, t5, mu5)
    a, b = c5["0.65"], c5["0.75"]
    obs = rate_ratio(a["crashes"], a["hours"], b["crashes"], b["hours"])
    pred = {m: predicted_ratio(y4, t4, mu4, run4, h) for m, h in MODELS.items()}
    ll = {m: two_cell_loglik(a["crashes"], a["hours"], b["crashes"], b["hours"], r) for m, r in pred.items()}
    best = max(ll.values())
    out = {"stream5": {"copies": int(len(y5)), "crashes": float(y5.sum()), "cells": c5, "rr_0.65_vs_0.75": obs},
           "predicted_rr_from_4_streams": pred, "loglik_stream5": ll,
           "rel_likelihood": {m: float(np.exp(v - best)) for m, v in ll.items()}}
    out["five_streams"] = P.analyse(B=B, load=load_all5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.run or a.quick:
        res = K.replicate(E=3, warm_s=60.0, seconds=120.0, mus=MUS5, seed=SEED5) if a.quick else K.replicate(E=48, mus=MUS5, seed=SEED5)
        json.dump(res, open(f"{OUT}/stream5{'_quick' if a.quick else ''}.json", "w"))
        print({k: (c["crashes"], round(c["hours"], 1)) for k, c in res["cells"].items()})
        return
    out = K._clean(analyse())
    json.dump(out, open(f"{OUT}/knee5.json", "w"), indent=1)
    f = out["five_streams"]
    print(json.dumps({k: out[k] for k in ("stream5", "predicted_rr_from_4_streams", "rel_likelihood")}, indent=1))
    print(json.dumps({k: v for k, v in f.items() if k not in ("grid", "drops", "fixed_hinges")}, indent=1))
    from avlab.knee5_plot import plot
    plot(out)


if __name__ == "__main__":
    main()
