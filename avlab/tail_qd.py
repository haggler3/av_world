"""Idea 3: a library of *different* failures, by quality-diversity search (MAP-Elites) over the scenario knobs.

A search for the worst scenario finds one kind of failure, over and over. MAP-Elites keeps, for every kind of failure, the scenario that
produces it most strongly. A scenario here is a point in a 15-dimensional box of simulator knobs (human risk, red-runners, drivers who do
not look, dart-outs, AV sensor blackouts and ghosts, breakdowns, tight cut-ins, friction, pedestrian and cyclist volume, tailgaters, driver
noise ...), the policy under test is fixed, and a scenario is described by two things measured in the run: **which crash type** dominates the AV's
crashes (rear-end, angle, left-turn, sideswipe, merge, pedestrian, bicycle) and **how severe** (mean impact or closing speed). Each cell
of that 7 x 4 grid holds the scenario with the most AV crashes per AV-hour that lands in it. Mutations of existing elites fill the grid.

    python -m avlab.tail_qd run        # ~15 min: MAP-Elites and, at the same budget, random search  -> results/tail/qd.json
    python -m avlab.tail_qd verify     # the elites again on fresh random numbers                       -> results/tail/qd_verify.json
"""
import json
import os
import sys
import time

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import numpy as np  # noqa: E402

from avrl import xppo as P  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import adversary as AD  # noqa: E402

OUT = "results/tail"
POLICY = "results/famous/policy_A.npz"  # a policy trained without a filter (crashes ~0.08 per AV-hour in ordinary traffic)
# knob: (low, high, log scale)
GENES = {"risk": (1.0, 15.0, True), "p_red": (0.05, 0.5, False), "red_window": (0.3, 2.5, False), "p_nolook": (1e-5, 5e-3, True),
         "p_dart": (0.0, 0.8, False), "p_blind": (0.0, 1e-3, False), "p_ghost": (0.0, 3e-3, False), "breakdown_rate": (2e-6, 2e-4, True),
         "adv_cut": (1.0, 4.0, False), "friction": (0.35, 1.0, False), "ped_scale": (0.5, 4.0, True), "p_ped_violate": (0.03, 0.6, True),
         "p_bike": (0.02, 0.2, False), "p_close_pass": (5e-4, 3e-2, True), "p_tailgate": (0.0, 0.5, False)}
KEYS = list(GENES)
TYPES = X.CRASH_NAMES
SEV_EDGES = (2.0, 5.0, 8.0)  # m/s


def decode(g):
    out = {}
    for x, (k, (lo, hi, lg)) in zip(g, GENES.items()):
        out[k] = float(np.exp(np.log(lo) + x * (np.log(hi) - np.log(lo)))) if lg else float(lo + x * (hi - lo))
    return out


def behaviour(g):
    kw = decode(g)
    return AD.world(kw.pop("risk"), **kw)


def describe(d):
    """(crash type with the most AV crashes, severity bucket) of a run, or None if the AV did not crash."""
    counts = np.array([d[a] for a in X.AV_CRASH])
    if counts.sum() == 0:
        return None
    t = int(counts.argmax())
    if t == 5 and d["imp_ped_n"]:
        sp = d["imp_ped_speed"] / d["imp_ped_n"]
    elif d["imp_rear_n"]:
        sp = d["imp_rear_closing"] / d["imp_rear_n"]
    else:
        sp = 0.0
    return t, int(np.searchsorted(SEV_EDGES, sp))


class Evaluator:
    def __init__(self, n=6, warm=40.0, seconds=100.0, seed=3):
        self.arena = AD.Arena(policy=P.deterministic(P.load(POLICY)), shield="off", n=n, seed=seed)
        self.warm_s, self.seconds, self.calls = warm, seconds, 0

    def __call__(self, g, seed=None):
        beh = behaviour(g)
        arena = self.arena
        if seed is not None:
            import jax
            arena.fresh = jax.vmap(lambda k: X.reset(k, arena.net, 1024))(jax.random.split(jax.random.PRNGKey(seed), arena.n))
        st = arena.warm(beh, self.warm_s)
        _, d = arena.measure(st, beh, self.seconds)
        self.calls += 1
        h = max(d["av_veh_s"] / 3600, 1e-9)
        return d["av_crash"] / h, d["av_crash"], describe(d), d


def run(budget=240, init=60, seed=0):
    os.makedirs(OUT, exist_ok=True)
    ev = Evaluator()
    rng = np.random.default_rng(seed)
    res = {}
    for name in ("map-elites", "random"):
        archive, trace, t0 = {}, [], time.time()
        for i in range(budget):
            if name == "random" or i < init or not archive:
                g = rng.random(len(KEYS))
            else:
                parent = archive[list(archive)[int(rng.integers(len(archive)))]]
                g = np.clip(np.array(parent["genome"]) + rng.normal(0, 0.12, len(KEYS)) * (rng.random(len(KEYS)) < 0.4), 0, 1)
            fit, n_cr, cell, d = ev(g)
            if cell is not None and n_cr >= 2 and (cell not in archive or fit > archive[cell]["fitness"]):
                archive[cell] = {"genome": g.tolist(), "fitness": fit, "crashes": n_cr, "knobs": decode(g)}
            trace.append({"i": i, "cells": len(archive), "qd_score": float(sum(a["fitness"] for a in archive.values())),
                          "best": float(max([a["fitness"] for a in archive.values()], default=0.0))})
            if i % 20 == 0 or i == budget - 1:
                print(f"{name} {i}: {len(archive)} cells, QD score {trace[-1]['qd_score']:.2f}, best {trace[-1]['best']:.2f}  {time.time() - t0:.0f} s", flush=True)
        res[name] = {"archive": {f"{TYPES[c[0]]}|{c[1]}": v for c, v in archive.items()}, "trace": trace}
        json.dump(res, open(f"{OUT}/qd.json", "w"), indent=1)


def verify(top=12):
    """Re-run the strongest elites on fresh random numbers: does the failure come back?"""
    res = json.load(open(f"{OUT}/qd.json"))
    ev = Evaluator(seed=77)
    out = {}
    for name, r in res.items():
        elites = sorted(r["archive"].items(), key=lambda kv: -kv[1]["fitness"])[:top]
        rows = []
        for cell, a in elites:
            fit, n_cr, c2, _ = ev(np.array(a["genome"]), seed=900)
            rows.append({"cell": cell, "found": a["fitness"], "again": fit, "again_crashes": n_cr, "same_type": c2 is not None and TYPES[c2[0]] == cell.split("|")[0]})
            print(name, cell, round(a["fitness"], 2), "->", round(fit, 2), flush=True)
        out[name] = rows
    json.dump(out, open(f"{OUT}/qd_verify.json", "w"), indent=1)


if __name__ == "__main__":
    {"run": run, "verify": verify}[sys.argv[1]]()
