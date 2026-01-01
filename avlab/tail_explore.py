"""Idea 5: go-explore from near-misses, as a generator of crash scenes.

The rare-event estimator (avlab/rare.py) failed because it had to stay unbiased. A generator does not: it only has to find many different crash
scenes per hour of simulation. Go-explore keeps an archive of the states the AVs got into trouble in (a moving AV inside its barrier, an
AV in a junction with a red-runner or a pedestrian about), restarts copies from them with new random numbers, and files the new states it
reaches. Plain Monte Carlo spends its copies on ordinary traffic; this spends them where crashes are about to happen.

    python -m avlab.tail_explore      # ~18 min -> results/tail/explore.json
"""
import json
import os
import sys
import time

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xnet import sim as X  # noqa: E402

from . import adversary as AD  # noqa: E402
from . import rare as RA  # noqa: E402

OUT = "results/tail"
WINDOWS = RA.WINDOW * X.DT  # 3 s


def danger_and_cell(feat, live, mode="barrier"):
    """Per copy: how close the AVs are to trouble (0 none ... 2: inside the barrier), and the archive cell.
    feat [n, K, 35]: obs[0] = speed / 15, obs[5] = on a link (0: inside a junction box), 28 = barrier margin / 10 (m), 31 = red-runners, 34 = pedestrians."""
    v = feat[..., 0] * 15.0
    h = feat[..., 28] * 10.0
    moving = live & (v > 1.0)
    risk = np.where(moving, np.clip((6.0 - h) / 6.0, 0.0, 2.0), 0.0)
    D = risk.max(1)
    hmin = np.where(moving, h, 99.0).min(1)
    in_box = (live & (feat[..., 5] < 0.5)).any(1)
    cell = (np.digitize(hmin, (0.0, 2.0, 5.0)) * 8 + in_box * 4 + (feat[:, 0, 31] > 0.05) * 2 + (feat[:, 0, 34] > 0.05)).astype(int)
    if mode == "mistakes":  # a human mistake is in progress: a red-runner, a distracted driver, a stalled car, a pedestrian crossing, with an AV in a junction
        ctx = feat[:, 0, 31:35] > 0.05
        D = ctx[:, 0] * 1.0 + ctx[:, 1] * 1.0 + ctx[:, 2] * 0.5 + ctx[:, 3] * 0.5 + in_box * 0.5 + np.clip((4.0 - np.minimum(hmin, 4.0)) / 4.0, 0, 1) * 0.5
        cell = (ctx[:, 0] * 8 + ctx[:, 1] * 4 + ctx[:, 2] * 2 + ctx[:, 3]) * 2 + in_box
        cell = cell.astype(int)
    return D, cell


def host_copy(st, i):
    return jax.tree.map(lambda a: np.asarray(a[i]), st)


def stack(entries):
    return jax.tree.map(lambda *xs: jnp.stack(xs), *entries)


def plain(r, st, windows, key):
    crashes, sig = 0, set()
    for k in range(windows):
        st0 = st
        st, c, f, lv, *_ = r.window(st, jax.random.fold_in(key, k))
        D, cell = danger_and_cell(f, lv)
        crashes += int((c > 0).sum())
        for i in np.nonzero(c > 0)[0]:
            sig.add((int(r.last_types[i].argmax()), int(cell[i])))
    return {"crash_windows": crashes, "signatures": len(sig)}


def explore(r, st, windows, key, rng, beta=2.0, mode="barrier"):
    n = r.n
    archive = {}  # cell -> dict(state, danger, visits)
    crashes, sig, new_cells = 0, set(), 0
    for k in range(windows):
        if len(archive) >= 3 and k > 2:
            cells = list(archive)
            w = np.array([(1.0 + archive[c]["danger"]) ** beta / np.sqrt(1.0 + archive[c]["visits"]) for c in cells])
            pick = rng.choice(len(cells), n, p=w / w.sum())
            start = stack([archive[cells[j]]["state"] for j in pick])
            start = start._replace(key=jax.vmap(jax.random.fold_in)(start.key, jnp.arange(n) + 7919 * (k + 1)))
            for j in pick:
                archive[cells[j]]["visits"] += 1
            st = start
        st, c, f, lv, *_ = r.window(st, jax.random.fold_in(key, k))
        D, cell = danger_and_cell(f, lv, mode)
        crashes += int((c > 0).sum())
        for i in np.nonzero(c > 0)[0]:
            sig.add((int(r.last_types[i].argmax()), int(cell[i])))
        for i in np.argsort(-D)[: max(4, n // 8)]:  # file the most dangerous new states
            if D[i] <= (0.2 if mode == "barrier" else 1.4):
                continue
            ce = int(cell[i])
            if ce not in archive or D[i] > archive[ce]["danger"] * 1.05:
                if ce not in archive:
                    new_cells += 1
                archive[ce] = {"state": host_copy(st, i), "danger": float(D[i]), "visits": archive.get(ce, {}).get("visits", 0)}
    return {"crash_windows": crashes, "signatures": len(sig), "cells": len(archive)}


def main(n=64, windows=40, reps=3):
    os.makedirs(OUT, exist_ok=True)
    worlds = {"humans x5, rule-based AVs (IDM envelope)": (AD.world(5.0), "idm"), "humans x1, every AV under the barrier function": (AD.world(1.0), "cbf")}
    res = {}
    for name, (beh, shield) in worlds.items():
        r = RA.Rare(beh, shield, n, seed=5)
        rows = {"plain": [], "go-explore (barrier)": [], "go-explore (mistakes)": []}
        t0 = time.time()
        for rep in range(reps):
            st = r.init(150.0, seed=100 + rep)
            key = jax.random.PRNGKey(200 + rep)
            rows["plain"].append(plain(r, st, windows, key))
            rows["go-explore (barrier)"].append(explore(r, st, windows, key, np.random.default_rng(rep), mode="barrier"))
            rows["go-explore (mistakes)"].append(explore(r, st, windows, key, np.random.default_rng(rep), mode="mistakes"))
            print(name, rep, rows["plain"][-1], rows["go-explore (barrier)"][-1], rows["go-explore (mistakes)"][-1], f"{time.time() - t0:.0f} s", flush=True)
        hours = n * windows * WINDOWS / 3600
        res[name] = {"area_hours_per_run": hours, **rows}
        json.dump(res, open(f"{OUT}/explore.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
