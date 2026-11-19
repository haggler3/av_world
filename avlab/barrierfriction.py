"""The AV barrier's braking assumption against the road's friction: the simulator side of formal/AvFormal/Barrier.lean.

The Lean file proves that `xnet.sim.cbf_bound` (b = 6 m/s^2, tau = 0.5) can be obeyed on every state with h >= 0 iff the
road's braking 9 mu >= 6, i.e. mu >= 2/3, and that the fix b_eff = min(6, 9 mu) restores this at every friction. Here:

    E1  default barrier, friction 1.0 ... 0.4: share of moving AV steps inside the barrier (h < 0) and faster than v_feas(mu)
    E2  friction 0.4: b = 6 (default) vs b = 3.6 (friction-aware), AV crashes per AV-hour, exact conditional Poisson test

Every AV is filtered (X.SHIELD = "cbf"); b is set through the module dict X.CBF before tracing. xnet/ is not changed.
Pre-registration: results/daily/2026-11-19b/predictions.md

    python -m avlab.barrierfriction --quick      # tiny E1 + E2
    python -m avlab.barrierfriction --run        # E1 then E2 -> results/barrierfriction/run.json (~8 min)
    python -m avlab.barrierfriction              # analysis -> results/barrierfriction/barrierfriction.json
"""
import argparse
import json
import os

import numpy as np

OUT = "results/barrierfriction"
MUS = (1.0, 0.8, 0.7, 0.6, 0.5, 0.4)
B_DEFAULT, TAU = 6.0, 0.5


def v_feas(mu, b=B_DEFAULT, tau=TAU):
    """Speed above which the barrier, on its edge with a stopped obstacle, needs braking beyond 9 mu (inf if 9 mu >= b).
    The formula proved in formal/AvFormal/Barrier.lean (bound_infeasible)."""
    B = 9.0 * mu
    return float("inf") if B >= b else B * b * tau / (b - B)


def _run(mus, E, seed, warm_s, seconds, b=B_DEFAULT, chunk_s=60.0):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    old_shield, old_b = X.SHIELD, X.CBF["b"]
    X.SHIELD, X.CBF["b"] = "cbf", b
    try:
        net, _, _ = NW.build("oakland")
        J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
        tags = [m for m in mus for _ in range(E)]
        B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *[C.calibrated(friction=m, p_av=0.5) for m in tags])
        vf = jnp.asarray([min(v_feas(m, b), 1e6) for m in tags], jnp.float32)
        keys = jnp.stack([jax.random.PRNGKey(seed * 1000 + i % E) for i in range(len(tags))])
        st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
        n = int(round(chunk_s / X.DT))

        def one(s, c, b_, vf_):
            out, d = X.step(s, J, b_, nh, diag=True)
            mov = s.alive & s.av & (s.wreck_t <= 0) & (s.v > 2.0)
            c = c + jnp.stack([mov.sum(), (mov & (d["bar_h"] < 0)).sum(), (mov & (s.v > vf_)).sum()]).astype(jnp.float32)
            return out, c

        @jax.jit
        def chunk(st, c):
            def body(carry, _):
                s, c = carry
                return jax.vmap(one)(s, c, B, vf), None
            return jax.lax.scan(body, (st, c), None, length=n)[0]

        c = jnp.zeros((len(tags), 3), jnp.float32)
        for _ in range(int(round(warm_s / chunk_s))):
            st, c = chunk(st, c)
        m0, c0 = np.asarray(st.m), np.asarray(c)
        for _ in range(int(round(seconds / chunk_s))):
            st, c = chunk(st, c)
        m, c = np.asarray(st.m) - m0, np.asarray(c) - c0
        col = lambda k: m[:, X.METRICS.index(k)].tolist()  # noqa: E731
        return {"mus": tags, "b": b, "seed": seed, "copies": E, "seconds": seconds, "warm_s": warm_s,
                "moving_steps": c[:, 0].tolist(), "inside_steps": c[:, 1].tolist(), "over_vfeas_steps": c[:, 2].tolist(),
                "av_crash": col("av_crash"), "av_veh_s": col("av_veh_s"), "av_veh_m": col("av_veh_m"),
                "crashes": (sum(m[:, X.METRICS.index(k)] for k in X.CRASH_NAMES)).tolist(), "veh_s": col("veh_s")}
    finally:
        X.SHIELD, X.CBF["b"] = old_shield, old_b


def run(quick=False):
    if quick:
        return {"E1": _run((1.0, 0.4), 2, 21, 60.0, 60.0), "E2": [_run((0.4,), 2, 22, 60.0, 60.0, b=b) for b in (6.0, 3.6)]}
    e1 = _run(MUS, 12, 21, 60.0, 300.0)
    print("E1 done", flush=True)
    e2 = [_run((0.4,), 40, 22, 120.0, 480.0, b=b) for b in (6.0, 3.6)]
    return {"E1": e1, "E2": e2}


def _boot_ratio(num_a, den_a, num_b, den_b, B=4000, seed=0):
    """Copy bootstrap of (sum num_a / sum den_a) / (sum num_b / sum den_b)."""
    rng = np.random.default_rng(seed)
    num_a, den_a, num_b, den_b = map(np.asarray, (num_a, den_a, num_b, den_b))
    r = []
    for _ in range(B):
        i, j = rng.integers(0, len(num_a), len(num_a)), rng.integers(0, len(num_b), len(num_b))
        if num_b[j].sum() > 0:
            r.append((num_a[i].sum() / den_a[i].sum()) / (num_b[j].sum() / den_b[j].sum()))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def analyse(res):
    from avlab.whocrashes import rate_ratio
    e1 = {k: np.asarray(v) for k, v in res["E1"].items() if isinstance(v, list)}
    out = {"E1": {}}
    for mu in sorted(set(res["E1"]["mus"]), reverse=True):
        s = e1["mus"] == mu
        out["E1"][str(mu)] = {"v_feas": v_feas(mu), "moving_steps": float(e1["moving_steps"][s].sum()),
                              "inside_share": float(e1["inside_steps"][s].sum() / e1["moving_steps"][s].sum()),
                              "over_vfeas_share": float(e1["over_vfeas_steps"][s].sum() / e1["moving_steps"][s].sum()),
                              "av_crash": float(e1["av_crash"][s].sum()), "av_hours": float(e1["av_veh_s"][s].sum() / 3600),
                              "av_crash_per_1000h": float(1000 * e1["av_crash"][s].sum() / (e1["av_veh_s"][s].sum() / 3600))}
    mus = set(res["E1"]["mus"])
    if {0.4, 0.7}.issubset(mus):
        a, b = e1["mus"] == 0.4, e1["mus"] == 0.7
        out["P1_ratio_04_07"] = out["E1"]["0.4"]["inside_share"] / out["E1"]["0.7"]["inside_share"]
        out["P1_boot95"] = _boot_ratio(e1["inside_steps"][a], e1["moving_steps"][a], e1["inside_steps"][b], e1["moving_steps"][b])
        out["P1_ratio_07_10"] = out["E1"]["0.7"]["inside_share"] / out["E1"]["1.0"]["inside_share"]
    d, f = ({k: np.asarray(v) for k, v in r.items() if isinstance(v, list)} for r in res["E2"])
    kd, td, kf, tf = d["av_crash"].sum(), d["av_veh_s"].sum() / 3600, f["av_crash"].sum(), f["av_veh_s"].sum() / 3600
    out["E2"] = {"default_b6": {"av_crash": float(kd), "av_hours": float(td), "inside_share": float(d["inside_steps"].sum() / d["moving_steps"].sum()),
                                "av_speed_kmh": float(3.6 * d["av_veh_m"].sum() / d["av_veh_s"].sum()), "city_crashes": float(d["crashes"].sum())},
                 "aware_b36": {"av_crash": float(kf), "av_hours": float(tf), "inside_share": float(f["inside_steps"].sum() / f["moving_steps"].sum()),
                               "av_speed_kmh": float(3.6 * f["av_veh_m"].sum() / f["av_veh_s"].sum()), "city_crashes": float(f["crashes"].sum())}}
    out["H1"] = rate_ratio(kf, tf, kd, td)
    out["H1_pass"] = bool(out["H1"]["rr"] < 1 and out["H1"]["p"] < 0.05)
    out["S2_speed_cost"] = 1 - out["E2"]["aware_b36"]["av_speed_kmh"] / out["E2"]["default_b6"]["av_speed_kmh"]
    out["city_crash_ratio"] = rate_ratio(f["crashes"].sum(), f["veh_s"].sum() / 3600, d["crashes"].sum(), d["veh_s"].sum() / 3600)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.run or a.quick:
        res = run(quick=a.quick)
        if a.quick:
            print(json.dumps({k: v for k, v in analyse(res).items() if k != "E1"}, indent=1, default=float))
            return
        json.dump(res, open(f"{OUT}/run.json", "w"))
    res = json.load(open(f"{OUT}/run.json"))
    out = analyse(res)
    json.dump(out, open(f"{OUT}/barrierfriction.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
