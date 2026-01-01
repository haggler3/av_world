"""Does an AV that knows the road's friction crash less on a slick road? (ROADMAP item 1, hidden road friction.)

2026-11-13 showed that the road's friction mu can be read off the fleet's braking within 10-20 s (`avlab.stopline.tomography`),
and 2026-11-10 that the simulator's AVs do not use it (mu appears only in the braking cap). This module gives a learned-policy
slot AV (the IDM policy of examples/my_policy.py, run under the simulator's usual envelope) that estimate, assumed exact, and asks
what it buys in a half-AV Oakland:

    blind     IDM, time headway T = 1.5 s, desired speed v0 as the road gives it          (mu is ignored)
    headway   T = 1.5 s / mu                                                              (the reaction-plus-braking budget grows as 1/mu)
    slow      T = 1.5 s / mu and v0 x sqrt(mu)                                            (the dry road's stopping distance v^2 / 2 a is kept)

All cells run in one compiled program: copy e has its own friction and its own (T scale, v0 scale), so blind, headway and slow
share code, network and seeds (copy index -> key). The AVs read nothing else new; xnet/ is untouched.

    python -m avlab.frictionaware --quick   # 2 copies x 7 cells, 60 s (~1.5 min)
    python -m avlab.frictionaware           # 16 copies x 7 cells, 480 s scored -> results/frictionaware/frictionaware.json
    python -m avlab.frictionaware --plot    # docs/img/frictionaware.png, docs/img/frictionaware.gif from the saved file
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "results/frictionaware"
T0 = 1.5
POLICIES = {"blind": (False, False), "headway": (True, False), "slow": (True, True)}
CELLS = (("blind", 1.0), ("blind", 0.6), ("headway", 0.6), ("slow", 0.6), ("blind", 0.4), ("headway", 0.4), ("slow", 0.4))
CHUNK_S = 20.0


def scales(policy, mu):
    """(time-headway scale, desired-speed scale) for a policy on a road of friction mu."""
    head, slow = POLICIES[policy]
    return (1.0 / mu if head else 1.0), (float(np.sqrt(mu)) if slow else 1.0)


def aware_idm(obs, T_scale=1.0, v_scale=1.0, s0=2.5, a_max=1.5, b=2.0):
    """examples/my_policy.careful_idm with the time headway multiplied by T_scale and the desired speed by v_scale."""
    import jax.numpy as jnp
    T = T0 * T_scale
    v = 15 * obs[:, 0]
    v0 = jnp.maximum(v_scale * jnp.minimum(15 * obs[:, 3], 15 * obs[:, 16] + 0.5), 0.5)
    has_lead, gap_lead, v_lead = obs[:, 8] > 0.5, 50 * obs[:, 9], 15 * obs[:, 10]
    has_stop, gap_stop = obs[:, 6] > 0.5, 100 * obs[:, 7]

    def idm_term(gap, dv):
        s_star = s0 + jnp.maximum(v * T + v * dv / (2 * jnp.sqrt(a_max * b)), 0.0)
        return (s_star / jnp.maximum(gap, 0.1)) ** 2
    inter = jnp.maximum(jnp.where(has_lead, idm_term(gap_lead, v - v_lead), 0.0), jnp.where(has_stop, idm_term(gap_stop, v), 0.0))
    a = a_max * (1 - (v / v0) ** 4 - inter)
    u0 = jnp.clip(jnp.where(a >= 0, a / 3.0, a / 8.0), -1.0, 1.0)
    return jnp.stack([u0, jnp.zeros_like(u0)], 1), jnp.zeros((obs.shape[0], 2))


def run(E=24, cells=CELLS, warm_s=120.0, seconds=480.0, seed=5, p_av=0.5, area="oakland"):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X

    behs, Ts, Vs, tags = [], [], [], []
    for pol, mu in cells:
        ts, vs = scales(pol, mu)
        for _ in range(E):
            behs.append(C.calibrated(friction=mu, p_av=p_av)); Ts.append(ts); Vs.append(vs); tags.append(f"{pol}@{mu:g}")
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    Ts, Vs = jnp.asarray(Ts, jnp.float32), jnp.asarray(Vs, jnp.float32)
    # the same seed per copy index within a cell: cell c copy i uses key i, so cells differ only by friction and policy
    keys = jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for _ in cells for i in range(E)])
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
    pk = jax.random.PRNGKey(0)

    def one(s, b, ts, vs):
        pol = lambda obs, key: aware_idm(obs, ts, vs)  # noqa: E731
        return X.step(s, J, b, nh, rl=(pol, pk))[0]

    n = int(round(CHUNK_S / X.DT))

    @jax.jit
    def chunk(st):
        def body(s, _):
            return jax.vmap(one)(s, B, Ts, Vs), None
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
    out = {"area": area, "copies": E, "seconds": seconds, "warm_s": warm_s, "seed": seed, "p_av": p_av, "T0": T0,
           "chunk_s": CHUNK_S, "wall_s": round(time.time() - t0, 1), "cells": {}}
    crash_cols = [idx[k] for k in X.CRASH_NAMES]
    for pol, mu in cells:
        tag = f"{pol}@{mu:g}"
        s = tags == tag
        ts, vs = scales(pol, mu)
        g = lambda k, mm=m: mm[s, idx[k]]  # noqa: E731
        out["cells"][tag] = {
            "policy": pol, "mu": mu, "T_scale": ts, "v_scale": vs,
            "av_crashes": g("av_crash").tolist(), "av_hours": (g("av_veh_s") / 3600).tolist(), "av_km": (g("av_veh_m") / 1000).tolist(),
            "crashes": m[s][:, crash_cols].sum(1).tolist(), "hours": (g("veh_s") / 3600).tolist(),
            "finished": g("finished").tolist(), "av_finished": g("av_finished").tolist(),
            "av_types": {k: float(m[s, idx["av_" + k]].sum()) for k in X.CRASH_NAMES},
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
        out["cells"][tag] = {"av_crashes": k, "av_hours": h, "av_rate_per_1000h": 1000 * k / h, "av_speed_kmh": km / h,
                             "city_crashes": tot(tag, "crashes"), "city_rate_per_1000h": 1000 * tot(tag, "crashes") / tot(tag, "hours"),
                             "av_trips": tot(tag, "av_finished"), "trips": tot(tag, "finished")}
    keys, ps = [], []
    for mu in sorted({d["mu"] for d in c.values()}, reverse=True):
        b = f"blind@{mu:g}"
        for pol in ("headway", "slow"):
            a = f"{pol}@{mu:g}"
            if a not in c:
                continue
            rr = rate_ratio(tot(a, "av_crashes"), tot(a, "av_hours"), tot(b, "av_crashes"), tot(b, "av_hours"))
            # per-copy paired speed change (same keys per copy index)
            sp = np.asarray(c[a]["av_km"]) / np.asarray(c[a]["av_hours"]) / (np.asarray(c[b]["av_km"]) / np.asarray(c[b]["av_hours"]))
            rr_city = rate_ratio(tot(a, "crashes"), tot(a, "hours"), tot(b, "crashes"), tot(b, "hours"))
            out["tests"][a] = {"vs": b, "av_rr": rr, "city_rr": rr_city, "av_speed_ratio": float(sp.mean()),
                               "av_speed_ratio_sd": float(sp.std(ddof=1)) if len(sp) > 1 else 0.0}
            keys.append(a); ps.append(rr["p"])
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
        from avlab.frictionaware_plot import gif, plot
        res = json.load(open(f"{OUT}/frictionaware.json"))
        plot(res); gif(res)
        return
    res = run(E=2, warm_s=40.0, seconds=60.0) if a.quick else run()
    res["summary"] = summarise(res)
    name = "frictionaware_quick" if a.quick else "frictionaware"
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1)
    print(json.dumps(res["summary"], indent=1)[:3500])
    print("wall", res["wall_s"], "s")


if __name__ == "__main__":
    main()
