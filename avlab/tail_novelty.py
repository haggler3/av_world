"""Idea 4: mine the simulator for scenes nobody designed, by novelty.

Run the simulator at scale and keep only the moments that look unlike ordinary traffic. Each AV at each 3-s window has a 35-number description
(the policy's 28-feature observation, the barrier margin and vetoes, and the scene around it). Novelty is the distance to the 10th nearest neighbour among
a reference set of such descriptions from ordinary traffic. The question this experiment answers is whether *novel* moments are also
*dangerous* ones: the crash rate in the next 6 s of the most novel 0.1 %, 1 % and 5 % of AV-windows, against the average, and against the
obvious alternative of ranking by the barrier margin.

Two reference sets: ordinary traffic (a different world from the one mined, so the world shift counts as novelty) and the first half of
the mined world itself (pure within-world rarity, tested on the second half).

    python -m avlab.tail_novelty      # ~8 min -> results/tail/novelty.json
"""
import json
import os
import time

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from xnet import sim as X  # noqa: E402

from . import adversary as AD  # noqa: E402
from . import rare as RA  # noqa: E402

OUT = "results/tail"
H = 2  # windows (6 s) over which a crash of that AV counts


def collect(beh, shield, n, windows, seed):
    r = RA.Rare(beh, shield, n, seed=seed)
    st = r.init(150.0)
    F, LV, NW, SL, T0 = ([] for _ in range(5))
    for k in range(windows):
        st, c, f, live, new, slot, t0, ov = r.window(st, jax.random.PRNGKey(seed * 1000 + k))
        for lst, v in zip((F, LV, NW, SL, T0), (f, live, new, slot, t0)):
            lst.append(v)
    F, LV, NW, SL, T0 = (np.stack(a) for a in (F, LV, NW, SL, T0))
    y = np.zeros(LV.shape, bool)
    for j, c, i in zip(*np.nonzero(NW)):
        for k in range(max(j - H, 0), j):
            y[k, c] |= LV[k, c] & (SL[k, c] == SL[j, c, i]) & (T0[k, c] == T0[j, c, i])
    keep = LV & (np.arange(windows)[:, None, None] < windows - H)
    return F[keep].astype(np.float32), y[keep], np.broadcast_to(np.arange(windows)[:, None, None], LV.shape)[keep], int(NW.sum())


def novelty(ref, rows, k=10):
    mu, sd = ref.mean(0), ref.std(0) + 1e-6
    tree = cKDTree((ref - mu) / sd)
    d, _ = tree.query((rows - mu) / sd, k=k)
    return d.mean(1)


def enrich(score, y, label):
    order = np.argsort(-score)
    base = y.mean()
    out = {"base_rate": float(base), "crashes": int(y.sum()), "rows": int(len(y))}
    for frac in (0.001, 0.01, 0.05):
        top = order[: max(int(len(y) * frac), 1)]
        out[f"top_{frac}"] = {"rate": float(y[top].mean()), "enrichment": float(y[top].mean() / max(base, 1e-12)), "crashes": int(y[top].sum()), "rows": len(top)}
    r = np.argsort(np.argsort(score)) + 1.0
    n1, n0 = y.sum(), (~y).sum()
    out["auc"] = float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
    return out


def main(n=128, windows=60):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    nominal, _, _, _ = collect(AD.world(1.0), "idm", n, 40, seed=21)
    mined_beh = AD.world(5.0, p_blind=2e-4, p_ghost=1e-3, p_dart=0.3, breakdown_rate=1e-4, adv_cut=3.0, p_tailgate=0.3)
    F, y, w, n_cr = collect(mined_beh, "idm", n, windows, seed=22)
    print(f"reference {len(nominal)} rows, mined {len(F)} rows, {int(y.sum())} AV-windows followed by that AV's crash ({n_cr} crashes)  {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(0)
    ref = nominal[rng.choice(len(nominal), min(len(nominal), 150_000), replace=False)]
    res = {"mined_world": "humans x5, sensor faults, dart-outs, stalled cars, tight cut-ins, tailgaters", "rows": int(len(F)), "av_crashes": n_cr}
    res["cross-world novelty (reference: ordinary traffic)"] = enrich(novelty(ref, F), y, "cross")
    half = w < w.max() // 2
    res["within-world rarity (reference: first half of the mined world)"] = enrich(novelty(F[half][rng.choice(half.sum(), min(half.sum(), 150_000), replace=False)], F[~half]), y[~half], "within")
    res["barrier margin (smaller is riskier), same rows"] = enrich(-F[:, 28], y, "barrier")
    res["random"] = enrich(rng.random(len(F)), y, "random")
    for k, v in res.items():
        if isinstance(v, dict) and "auc" in v:
            print(f"{k}: AUC {v['auc']:.3f}; top 1 % crash rate {v['top_0.01']['rate']:.4f} (x{v['top_0.01']['enrichment']:.1f}); top 0.1 % x{v['top_0.001']['enrichment']:.1f}", flush=True)
    json.dump(res, open(f"{OUT}/novelty.json", "w"), indent=1)


if __name__ == "__main__":
    main()
