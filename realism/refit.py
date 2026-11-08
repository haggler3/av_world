"""The refit: cross-entropy search over the new knobs, scored by the scoreboard on symmetrically processed scenes.

Every candidate drives its own copy of the city in one compiled run (behaviours are vmapped), with common random numbers, so candidates
differ by their knobs and not by their dice. Its scenes go through the real sensor model and then the same cleaning as the real logs
(clean(corrupt(sim)) against clean(real)), and the loss is the scoreboard's gap: mean |log(sim / real)| of the four tail rates (capped
at log 20) plus the mean Wasserstein-1 distance of the six distributions in real interquartile ranges.
"""
import numpy as np

from . import measure as M
from . import simrec as SR
from . import stats as ST

SPACE = {"demand": (0.8, 4.0), "v0_scale": (0.9, 1.35), "ou_sigma": (0.0, 1.2), "ou_tau": (0.3, 5.0), "pulse_rate": (0.0, 0.02),
         "bank": (0.5, 1.0), "s0_scale": (0.3, 1.2)}
FIXED = {"idm_plus": 1.0}


def to_knobs(z):
    return {k: float(lo + (hi - lo) * np.clip(zi, 0, 1)) for (k, (lo, hi)), zi in zip(SPACE.items(), z)}


def process(scenes, nm, seed=0):
    return [M.clean(M.corrupt(s, nm, seed + i), nm) for i, s in enumerate(scenes)]


def evaluate(knobs, real_P, nm, bank, minutes=3, seed=0):
    sc = SR.record([SR.behaviour(**FIXED, **k) for k in knobs], minutes=minutes, copies=1, seed=seed, bank=bank)
    out = []
    for i in range(len(knobs)):
        L = process([s for s in sc if s["truth"]["beh"] == i], nm, seed)
        if len(L) < 5:
            out.append((99.0, None))
            continue
        sb = ST.scoreboard(real_P, ST.pool([ST.measure(s) for s in L]), n_boot=0, where=False)
        out.append((sb["gap"], sb))
    return out


def cem(real_P, nm, bank, iters=3, pop=8, elite=3, minutes=3, seed=0, log=print):
    rng = np.random.default_rng(seed)
    d = len(SPACE)
    mu, sd = np.full(d, 0.5), np.full(d, 0.3)
    hist, best = [], (np.inf, None, None)
    for it in range(iters):
        Z = np.clip(mu + sd * rng.normal(size=(pop, d)), 0, 1)
        if it == 0:
            Z[0] = 0.5
        knobs = [to_knobs(z) for z in Z]
        res = evaluate(knobs, real_P, nm, bank, minutes, seed)
        loss = np.array([r[0] for r in res])
        for k, l in zip(knobs, loss):
            hist.append({"iter": it, "loss": float(l), **k})
        i = int(np.argmin(loss))
        if loss[i] < best[0]:
            best = (float(loss[i]), knobs[i], res[i][1])
        el = Z[np.argsort(loss)[:elite]]
        mu, sd = el.mean(0), np.maximum(el.std(0), 0.05)
        log(f"    refit {it}: best {loss.min():.3f} (overall {best[0]:.3f}) {best[1]}")
    return {"loss": best[0], "knobs": dict(FIXED, **best[1]), "scoreboard": best[2], "history": hist}
