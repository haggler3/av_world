"""Gap caution on ice, confirmed? A fresh random stream with the bar fixed in advance, and what the caution costs on drier roads.

2026-11-15 (`avlab.frictionmerge`) found that an AV that accepts only longer gaps at friction 0.4 (xnet's gap scale gsc 1.15 -> 1.495,
the policy slot's maximum) had 0.54x the AV crashes of a blind AV (95 % 0.30-0.95, p 0.024), but after Holm over four tests p 0.097.
This module runs ONE pre-registered test on a new random stream (seed 7) plus cost cells (results/daily/2026-11-16/predictions.md):

    friction 0.4   blind (gsc 1.15)  vs  gap (gsc 1.495)     40 copies each   <- the confirmatory test
    friction 0.6   blind             vs  gap (gsc 1.495)     20 copies each   <- cost + effect where the 1/mu rule still saturates
    friction 1.0   blind             vs  max gap (gsc 1.495) 20 copies each   <- cost of caution that a 1/mu rule would switch off

All six cells in one compiled program; copy i of every cell on key (seed * 1000 + i). xnet/ is untouched (policy slot only).

    python -m avlab.gapconfirm --quick   # 1 copy per cell, 40 s warm, 60 s scored (smoke test)
    python -m avlab.gapconfirm           # -> results/frictionmerge/gapconfirm.json (~9 min on 4 cores)
    python -m avlab.gapconfirm --plot    # -> docs/img/gapconfirm.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avlab.frictionmerge import CHUNK_S, GSC0, GSC_MAX, GSC_SPAN, merge_idm  # noqa: E402

OUT = "results/frictionmerge"
U1_MAX = float(np.arctanh((GSC_MAX - GSC0) / GSC_SPAN))
# (name, friction, gsc-on?, copies at full size)
CELLS = (("blind@0.4", 0.4, False, 40), ("gap@0.4", 0.4, True, 40),
         ("blind@0.6", 0.6, False, 20), ("gap@0.6", 0.6, True, 20),
         ("blind@1.0", 1.0, False, 20), ("gap@1.0", 1.0, True, 20))
PAIRS = (("gap@0.4", "blind@0.4"), ("gap@0.6", "blind@0.6"), ("gap@1.0", "blind@1.0"))
CONFIRMATORY = "gap@0.4 vs blind@0.4"


def run(cells=CELLS, scale=1.0, warm_s=120.0, seconds=480.0, seed=7, p_av=0.5, area="oakland"):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X

    behs, Us, tags = [], [], []
    for name, mu, gap, E in cells:
        for _ in range(max(1, int(round(E * scale)))):
            behs.append(C.calibrated(friction=mu, p_av=p_av)); Us.append(U1_MAX if gap else 0.0); tags.append(name)
    idx_copy = [i for name, _, _, E in cells for i in range(max(1, int(round(E * scale))))]
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    Us = jnp.asarray(Us, jnp.float32)
    keys = jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for i in idx_copy])
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
    pk = jax.random.PRNGKey(0)

    def one(s, b, u1):
        pol = lambda obs, key: merge_idm(obs, 1.0, u1)  # noqa: E731
        return X.step(s, J, b, nh, rl=(pol, pk))[0]

    n = int(round(CHUNK_S / X.DT))

    @jax.jit
    def chunk(st):
        def body(s, _):
            return jax.vmap(one)(s, B, Us), None
        return jax.lax.scan(body, st, None, length=n)[0]

    t0 = time.time()
    for _ in range(int(round(warm_s / CHUNK_S))):
        st = chunk(st)
    m0 = np.asarray(st.m)
    for _ in range(int(round(seconds / CHUNK_S))):
        st = chunk(st)
    m = np.asarray(st.m) - m0
    idx = {k: i for i, k in enumerate(X.METRICS)}
    tags = np.array(tags)
    crash_cols = [idx[k] for k in X.CRASH_NAMES]
    out = {"area": area, "seed": seed, "p_av": p_av, "warm_s": warm_s, "seconds": seconds, "wall_s": round(time.time() - t0, 1), "cells": {}}
    for name, mu, gap, _ in cells:
        s = tags == name
        g = lambda k: m[s, idx[k]]  # noqa: E731
        out["cells"][name] = {"mu": mu, "gsc": GSC_MAX if gap else GSC0, "copies": int(s.sum()),
                              "av_crashes": g("av_crash").tolist(), "av_hours": (g("av_veh_s") / 3600).tolist(),
                              "av_km": (g("av_veh_m") / 1000).tolist(), "crashes": m[s][:, crash_cols].sum(1).tolist(),
                              "hours": (g("veh_s") / 3600).tolist(), "av_finished": g("av_finished").tolist(),
                              "finished": g("finished").tolist(), "av_merges": g("av_merge").tolist()}
    return out


def summarise(res):
    from avlab.whocrashes import rate_ratio
    c = res["cells"]
    tot = lambda tag, k: float(np.sum(c[tag][k]))  # noqa: E731
    out = {"cells": {}, "tests": {}}
    for tag in c:
        k, h, km = tot(tag, "av_crashes"), tot(tag, "av_hours"), tot(tag, "av_km")
        out["cells"][tag] = {"av_crashes": k, "av_merges": tot(tag, "av_merges"), "av_hours": h, "av_rate_per_1000h": 1000 * k / max(h, 1e-9),
                             "av_speed_kmh": km / max(h, 1e-9), "av_trips": tot(tag, "av_finished"),
                             "city_rate_per_1000h": 1000 * tot(tag, "crashes") / max(tot(tag, "hours"), 1e-9)}
    for a, b in PAIRS:
        if a not in c or b not in c:
            continue
        sp = (np.asarray(c[a]["av_km"]) / np.asarray(c[a]["av_hours"])) / (np.asarray(c[b]["av_km"]) / np.asarray(c[b]["av_hours"]))
        out["tests"][f"{a} vs {b}"] = {
            "av_rr": rate_ratio(tot(a, "av_crashes"), tot(a, "av_hours"), tot(b, "av_crashes"), tot(b, "av_hours")),
            "av_merge_rr": rate_ratio(tot(a, "av_merges"), tot(a, "av_hours"), tot(b, "av_merges"), tot(b, "av_hours")),
            "city_rr": rate_ratio(tot(a, "crashes"), tot(a, "hours"), tot(b, "crashes"), tot(b, "hours")),
            "av_speed_ratio": float(sp.mean()), "av_speed_ratio_sd": float(sp.std(ddof=1)) if len(sp) > 1 else 0.0,
            "av_trip_ratio": tot(a, "av_finished") / max(tot(b, "av_finished"), 1.0)}
    if CONFIRMATORY in out["tests"]:
        t = out["tests"][CONFIRMATORY]
        out["confirmed"] = bool(t["av_rr"]["rr"] < 1 and t["av_rr"]["p"] < 0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        from avlab.gapconfirm_plot import plot
        res = json.load(open(f"{OUT}/gapconfirm.json"))
        plot(res)
        return
    res = run(scale=0.025, warm_s=40.0, seconds=60.0) if a.quick else run()
    res["summary"] = summarise(res)
    name = "gapconfirm_quick" if a.quick else "gapconfirm"
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
    print(json.dumps(res["summary"], indent=1)[:4000])
    print("wall", res["wall_s"], "s")


if __name__ == "__main__":
    main()
