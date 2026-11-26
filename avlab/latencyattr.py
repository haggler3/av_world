"""Where do delayed-perception crashes come from? Per-step attribution of AV time and AV crashes to the state of the AV's view.

`avlab.latency` shows the AV its leader as it was D steps ago, predicted forward. This module runs the same model and, at every
0.2 s step, compares what the AV is shown with the truth the simulator holds (the undelayed observation). Each alive AV row is put
in one of five states; the AV-seconds and the AV crashes of that step are added to the state's tally. Read-only: `xnet/` is untouched,
and the AV drives exactly as in `avlab.latency.run` (the extra memory only records the state).

    true leader?  shown leader?  state
    no            no             none     (empty road, seen as empty)
    yes           yes, |e|<=TOL  ok       (e = shown gap - true gap [m])
    yes           no             new      (a leader arrived during the delay: cut-in, lane change, turn; the view is still empty)
    yes           yes, e > TOL   over     (gap over-estimated: the leader braked, or the shown car is not the current leader)
    otherwise                    other    (no leader now but one shown, or gap under-estimated)

A crash of AV row r in the step from t to t+1 is counted in r's state at t (the view the policy acted on).
The per-state crash total equals the simulator's `av_crash` counter (checked in tests/test_latencyattr.py).

    from avlab import latencyattr as LA
    r = LA.run(LA.Latency(delay_s=2.0, compensate=True), E=16, seconds=600)   # r["state_s"], r["state_crash"], r["av_crash"]

    python -m avlab.latencyattr --quick     # 2 copies, 80 s
    python -m avlab.latencyattr             # analysis of results/latencyattr/cv*.json -> results/latencyattr/attr.json
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from avlab import latency as LT
from avlab import tracker as TR

Latency = LT.Latency
STATES = ("none", "ok", "new", "over", "other")
TOL = 2.0  # [m] gap error counted as "ok"


class Mem(NamedTuple):
    lt: LT.Mem
    state: jnp.ndarray  # [N] int32, index into STATES of the view the policy acted on this step


def classify(has_true, g_true, shown, g_shown, tol=TOL):
    """State index per row (see STATES)."""
    e = g_shown - g_true
    return jnp.select([~has_true & ~shown, has_true & shown & (jnp.abs(e) <= tol), has_true & ~shown, has_true & shown & (e > tol)],
                      [0, 1, 2, 3], 4).astype(jnp.int32)


def init_mem_for(m):
    base = LT.init_mem_for(m)
    return lambda n: Mem(base(n), jnp.zeros(n, jnp.int32))


def perceive(obs, key, mem: Mem, m):
    out, lt2 = LT.perceive(obs, key, mem.lt, m)
    st = classify(obs[:, 8] > 0.5, 50.0 * obs[:, 9], out[:, 8] > 0.5, 50.0 * out[:, 9])
    return out, Mem(lt2, st)


def run(m, E=16, seed=43, warm_s=120.0, seconds=600.0, chunk_s=40.0, p_av=0.5, friction=1.0, area="oakland", n_slots=1024):
    """As `avlab.latency.run` (envelope off), plus per-state AV-seconds `state_s` and AV crashes `state_crash` ([E][5])."""
    from examples import _common as C
    from avlab import teb as T
    from xnet import network as NW
    from xnet import sim as X
    X.SHIELD = "off"
    net, _, _ = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated(friction=friction, p_av=p_av)
    init_fn = init_mem_for(m)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for i in range(E)]), J, n_slots)
    mem = jax.vmap(lambda _: init_fn(n_slots))(jnp.arange(E))
    K = len(STATES)

    def f(o, k, mask, mi):
        o2, mi2 = perceive(o, k, mi, m)
        return T.idm_policy_masked(o2, k, mask)[0], mi2

    def one(si, mi, acc_s, acc_c):
        s2, tr = X.step(si, J, beh, nh, rl=(lambda o, k, mask: f(o, k, mask, mi), si.key, True))
        state = tr["aux"].state
        av0 = si.alive & si.av & ~(si.wreck_t > 0)
        crash = av0 & (s2.wreck_t > 0)
        oh = jax.nn.one_hot(state, K)
        acc_s = acc_s + X.DT * (oh * av0[:, None]).sum(0)
        acc_c = acc_c + (oh * crash[:, None]).sum(0)
        keep = s2.alive & s2.av
        mi2 = jax.tree.map(lambda new, z: jnp.where(keep, new, z), tr["aux"], init_fn(n_slots))
        return s2, mi2, acc_s, acc_c

    n = int(round(chunk_s / X.DT))

    @jax.jit
    def chunk(c):
        return jax.lax.scan(lambda c, _: (jax.vmap(one)(*c), None), c, None, length=n)[0]

    z = jnp.zeros((E, K))
    try:
        c = (st, mem, z, z)
        for _ in range(int(round(warm_s / chunk_s))):
            c = chunk(c)
        m0 = np.asarray(c[0].m)
        c = (c[0], c[1], z, z)
        for _ in range(int(round(seconds / chunk_s))):
            c = chunk(c)
    finally:
        X.SHIELD = "idm"
    d = np.asarray(c[0].m) - m0
    keep = ("av_crash", "av_veh_s", "av_veh_m", "av_hard")
    return {"model": m._asdict(), "copies": E, "seed": seed, "warm_s": warm_s, "seconds": seconds, "p_av": p_av,
            "friction": friction, "states": list(STATES), "state_s": np.asarray(c[2]).tolist(),
            "state_crash": np.asarray(c[3]).round().astype(int).tolist(),
            **{k: d[:, X.METRICS.index(k)].tolist() for k in keep}}


def analyse(cells):
    """Per-state rates with 95 % intervals and the pre-registered tests (results/daily/2026-11-26/predictions.md).
    `cells` maps a delay label ("cv2.0", "cv1.4") to a `run` result."""
    from scipy import stats
    from avlab.sensorbench import _poisson_ci
    from avlab.whocrashes import holm, rate_ratio
    out = {"states": list(STATES), "tol_m": TOL, "cells": {}, "tests": {}}
    for c, r in cells.items():
        s = np.asarray(r["state_s"]).sum(0) / 3600
        k = np.asarray(r["state_crash"]).sum(0)
        out["cells"][c] = {"delay_s": r["model"]["delay_s"], "av_crash": float(sum(r["av_crash"])), "crash_check":
                           bool(np.array_equal(np.asarray(r["state_crash"]).sum(1), np.round(r["av_crash"]).astype(int))),
                           "per_state": {st: {"av_hours": float(s[i]), "time_share": float(s[i] / s.sum()), "crashes": int(k[i]),
                                              "crash_share": float(k[i] / max(k.sum(), 1)), "rate_per_h": float(k[i] / s[i]) if s[i] else 0.0,
                                              "ci": [x / s[i] if s[i] else 0.0 for x in _poisson_ci(int(k[i]))]}
                                         for i, st in enumerate(STATES)}}
    P = lambda c, st: out["cells"][c]["per_state"][st]  # noqa: E731
    rr = lambda c, a, b: rate_ratio(P(c, a)["crashes"], P(c, a)["av_hours"], P(c, b)["crashes"], P(c, b)["av_hours"])  # noqa: E731
    if "cv2.0" in cells:
        out["tests"]["H1"] = {"what": "lambda_new / lambda_ok, 2.0 s", **rr("cv2.0", "new", "ok")}
        kn, ko = P("cv2.0", "new")["crashes"], P("cv2.0", "over")["crashes"]
        b = stats.binomtest(kn, kn + ko, 0.5)
        ci = b.proportion_ci()
        out["tests"]["S1"] = {"what": "share new among new+over crashes, 2.0 s", "share": kn / max(kn + ko, 1),
                              "lo": float(ci.low), "hi": float(ci.high), "p": float(b.pvalue), "n": kn + ko}
        out["tests"]["S2"] = {"what": "lambda_over / lambda_ok, 2.0 s", **rr("cv2.0", "over", "ok")}
    if "cv1.4" in cells:
        out["tests"]["S3"] = {"what": "lambda_new / lambda_ok, 1.4 s", **rr("cv1.4", "new", "ok")}
    sec = [n for n in ("S1", "S2", "S3") if n in out["tests"]]
    for n, p in zip(sec, holm([out["tests"][n]["p"] for n in sec])):
        out["tests"][n]["p_holm"] = float(p)
    return out


if __name__ == "__main__":
    import json
    import os
    import sys
    if "--quick" in sys.argv:
        r = run(Latency(2.0, True), E=2, warm_s=40, seconds=80)
        print({st: (round(float(np.asarray(r["state_s"]).sum(0)[i])), int(np.asarray(r["state_crash"]).sum(0)[i]))
               for i, st in enumerate(STATES)}, "av_crash", sum(r["av_crash"]))
    else:
        D = "results/latencyattr"
        cells = {c: json.load(open(f"{D}/{c}.json")) for c in ("cv2.0", "cv1.4") if os.path.exists(f"{D}/{c}.json")}
        res = analyse(cells)
        json.dump(res, open(f"{D}/attr.json", "w"), indent=1)
        print(json.dumps(res["tests"], indent=1))
        for c, v in res["cells"].items():
            print(c, "check", v["crash_check"], {st: (round(x["time_share"], 3), x["crashes"], round(x["rate_per_h"], 3))
                                                  for st, x in v["per_state"].items()})
