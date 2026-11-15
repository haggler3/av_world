"""The merge channel: does a friction-aware AV that is also more cautious at gaps crash less on a slick road?

2026-11-14 (`avlab.frictionaware`) gave the example IDM AV the road's friction mu and lengthened its time headway as 1/mu: AV crashes
at friction 0.4 fell 31 % (not significant), and 13 of the blind AVs' 21 crashes were **merges**, which car-following does not reach.
The policy slot has a second output, u[:, 1], the gap-acceptance scale used at junctions, zipper merges and lane changes
(`xnet/sim.py`: gsc = 1.15 + 0.35 tanh(u1), it multiplies the critical gap and the lane-change gaps; the rule-based AV uses 1.25).
The example policy sends u1 = 0, i.e. gsc 1.15: it accepts *shorter* gaps than the rule-based AV. This module adds a gap channel:

    blind         IDM T = 1.5 s, u1 = 0 (gsc 1.15)                           (mu ignored)
    headway       T = 1.5 s / mu, u1 = 0                                       (2026-11-14's 'headway')
    gap           T = 1.5 s, gsc = clip(1.15 / mu, 1.15, 1.495)               (gap acceptance only)
    headway+gap   both

All four cells run at friction 0.4 (where the merges were) in one compiled program, copy i of every cell on key (seed * 1000 + i);
a fresh random stream (seed 6). xnet/ is untouched; the AVs act through the policy slot under the usual envelope.

    python -m avlab.frictionmerge --quick   # 2 copies x 4 cells, 60 s scored (~1.5 min)
    python -m avlab.frictionmerge           # 40 copies x 4 cells, 480 s scored -> results/frictionmerge/frictionmerge.json
    python -m avlab.frictionmerge --plot    # docs/img/frictionmerge.png, docs/img/frictionmerge.gif from the saved file
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from avlab.frictionaware import aware_idm  # noqa: E402

OUT = "results/frictionmerge"
POLICIES = {"blind": (False, False), "headway": (True, False), "gap": (False, True), "headway+gap": (True, True)}
MU = 0.4
GSC0, GSC_SPAN, GSC_MAX = 1.15, 0.35, 1.495
CHUNK_S = 20.0


def gap_u1(mu, gap):
    """The policy's u[:, 1] that makes xnet's gap scale gsc = 1.15 + 0.35 tanh(u1) equal clip(1.15 / mu, 1.15, 1.495) (0 when off)."""
    if not gap:
        return 0.0
    target = float(np.clip(GSC0 / mu, GSC0, GSC_MAX))
    return float(np.arctanh((target - GSC0) / GSC_SPAN))


def scales(policy, mu):
    """(time-headway scale, u1) for a policy on a road of friction mu."""
    head, gap = POLICIES[policy]
    return (1.0 / mu if head else 1.0), gap_u1(mu, gap)


def merge_idm(obs, T_scale=1.0, u1=0.0):
    u, aux = aware_idm(obs, T_scale, 1.0)
    return u.at[:, 1].set(u1), aux


def run(E=40, policies=tuple(POLICIES), mu=MU, warm_s=120.0, seconds=480.0, seed=6, p_av=0.5, area="oakland"):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X

    behs, Ts, Us, tags = [], [], [], []
    for pol in policies:
        ts, u1 = scales(pol, mu)
        for _ in range(E):
            behs.append(C.calibrated(friction=mu, p_av=p_av)); Ts.append(ts); Us.append(u1); tags.append(pol)
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    Ts, Us = jnp.asarray(Ts, jnp.float32), jnp.asarray(Us, jnp.float32)
    keys = jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for _ in policies for i in range(E)])
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
    pk = jax.random.PRNGKey(0)

    def one(s, b, ts, u1):
        pol = lambda obs, key: merge_idm(obs, ts, u1)  # noqa: E731
        return X.step(s, J, b, nh, rl=(pol, pk))[0]

    n = int(round(CHUNK_S / X.DT))

    @jax.jit
    def chunk(st):
        def body(s, _):
            return jax.vmap(one)(s, B, Ts, Us), None
        return jax.lax.scan(body, st, None, length=n)[0]

    t0 = time.time()
    for _ in range(int(round(warm_s / CHUNK_S))):
        st = chunk(st)
    m0 = np.asarray(st.m)
    series = []
    for _ in range(int(round(seconds / CHUNK_S))):
        st = chunk(st)
        series.append(np.asarray(st.m) - m0)
    m = series[-1]
    idx = {k: i for i, k in enumerate(X.METRICS)}
    tags = np.array(tags)
    out = {"area": area, "copies": E, "mu": mu, "seconds": seconds, "warm_s": warm_s, "seed": seed, "p_av": p_av,
           "chunk_s": CHUNK_S, "wall_s": round(time.time() - t0, 1), "cells": {}}
    crash_cols = [idx[k] for k in X.CRASH_NAMES]
    for pol in policies:
        s = tags == pol
        ts, u1 = scales(pol, mu)
        g = lambda k, mm=m: mm[s, idx[k]]  # noqa: E731
        out["cells"][pol] = {
            "policy": pol, "mu": mu, "T_scale": ts, "u1": u1, "gsc": GSC0 + GSC_SPAN * float(np.tanh(u1)),
            "av_crashes": g("av_crash").tolist(), "av_hours": (g("av_veh_s") / 3600).tolist(), "av_km": (g("av_veh_m") / 1000).tolist(),
            "crashes": m[s][:, crash_cols].sum(1).tolist(), "hours": (g("veh_s") / 3600).tolist(),
            "finished": g("finished").tolist(), "av_finished": g("av_finished").tolist(),
            "av_types": {k: float(m[s, idx["av_" + k]].sum()) for k in X.CRASH_NAMES},
            "av_merges": m[s, idx["av_merge"]].tolist(),
            "series_av_crashes": [float(x[s, idx["av_crash"]].sum()) for x in series],
            "series_av_hours": [float(x[s, idx["av_veh_s"]].sum() / 3600) for x in series],
        }
    return out


def summarise(res):
    from avlab.whocrashes import holm, rate_ratio
    c = res["cells"]
    tot = lambda tag, k: float(np.sum(c[tag][k]))  # noqa: E731
    out = {"cells": {}, "tests": {}}
    for tag, d in c.items():
        k, h, km = tot(tag, "av_crashes"), tot(tag, "av_hours"), tot(tag, "av_km")
        out["cells"][tag] = {"av_crashes": k, "av_merges": d["av_types"]["merge"], "av_hours": h, "av_rate_per_1000h": 1000 * k / h,
                             "av_speed_kmh": km / h, "city_crashes": tot(tag, "crashes"),
                             "city_rate_per_1000h": 1000 * tot(tag, "crashes") / tot(tag, "hours"),
                             "av_trips": tot(tag, "av_finished"), "trips": tot(tag, "finished"), "av_types": d["av_types"]}
    # pre-registered comparisons (predictions.md): every aware policy vs blind, and the gap channel on top of headway
    pairs = [("headway", "blind"), ("gap", "blind"), ("headway+gap", "blind"), ("headway+gap", "headway")]
    keys, ps = [], []
    for a, b in pairs:
        if a not in c or b not in c:
            continue
        name = f"{a} vs {b}"
        rr = rate_ratio(tot(a, "av_crashes"), tot(a, "av_hours"), tot(b, "av_crashes"), tot(b, "av_hours"))
        rr_m = rate_ratio(c[a]["av_types"]["merge"], tot(a, "av_hours"), c[b]["av_types"]["merge"], tot(b, "av_hours"))
        sp = np.asarray(c[a]["av_km"]) / np.asarray(c[a]["av_hours"]) / (np.asarray(c[b]["av_km"]) / np.asarray(c[b]["av_hours"]))
        out["tests"][name] = {"av_rr": rr, "av_merge_rr": rr_m,
                              "city_rr": rate_ratio(tot(a, "crashes"), tot(a, "hours"), tot(b, "crashes"), tot(b, "hours")),
                              "av_speed_ratio": float(sp.mean()), "av_speed_ratio_sd": float(sp.std(ddof=1)) if len(sp) > 1 else 0.0,
                              "av_trip_ratio": tot(a, "av_finished") / max(tot(b, "av_finished"), 1.0)}
        keys.append(name); ps.append(rr["p"])
    for k, p in zip(keys, holm(ps) if ps else []):
        out["tests"][k]["av_p_holm"] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        from avlab.frictionmerge_plot import gif, plot
        res = json.load(open(f"{OUT}/frictionmerge.json"))
        plot(res); gif(res)
        return
    res = run(E=2, warm_s=40.0, seconds=60.0) if a.quick else run()
    res["summary"] = summarise(res)
    name = "frictionmerge_quick" if a.quick else "frictionmerge"
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
    print(json.dumps(res["summary"], indent=1)[:3000])
    print("wall", res["wall_s"], "s")


if __name__ == "__main__":
    main()
