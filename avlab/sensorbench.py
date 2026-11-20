"""How much does AV sensor noise hurt, and how much does the safety envelope take back? (avlab.sensornoise in the policy slot.)

Pre-registration: results/daily/2026-11-20b/predictions.md

    python -m avlab.sensorbench --quick              # 2 copies, 40 s, two cells (a smoke test)
    python -m avlab.sensorbench --cell off:severe    # one cell -> results/sensor/off_severe.json (~1.5 min)
    python -m avlab.sensorbench --all                # all cells, one after another
    python -m avlab.sensorbench                      # analysis of the saved cells -> results/sensor/sensor.json
"""
import argparse
import json
import os

import numpy as np

OUT = "results/sensor"
CELLS = [f"off:{lv}" for lv in ("clean", "mild", "moderate", "severe", "range", "speed", "miss")] + ["idm:clean", "idm:severe"]
KEEP = ("av_crash", "av_veh_s", "av_veh_m", "av_hard", "av_shield", "av_near", "veh_s")


def model(level):
    from avlab import sensornoise as SN
    s = SN.LEVELS["severe"]
    single = {"range": SN.SensorModel(s.sigma_range, 0.0, 0.0), "speed": SN.SensorModel(0.0, s.sigma_speed, 0.0),
              "miss": SN.SensorModel(0.0, 0.0, s.p_miss)}
    return SN.LEVELS.get(level) or single[level]


def run_cell(cell, E=16, seed=41, warm_s=120.0, seconds=600.0, chunk_s=40.0, p_av=0.5, friction=1.0):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from avlab import sensornoise as SN
    from avlab import teb as T
    from xnet import network as NW
    from xnet import sim as X
    shield, level = cell.split(":")
    X.SHIELD = shield  # read at trace time
    net, _, _ = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated(friction=friction, p_av=p_av)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for i in range(E)]), J, 1024)
    pol = SN.noisy(T.idm_policy_masked, model(level))
    n = int(round(chunk_s / X.DT))

    @jax.jit
    def chunk(st):
        def body(s, _):
            return jax.vmap(lambda si: X.step(si, J, beh, nh, rl=(pol, si.key, True))[0])(s), None
        return jax.lax.scan(body, st, None, length=n)[0]

    try:
        for _ in range(int(round(warm_s / chunk_s))):
            st = chunk(st)
        m0 = np.asarray(st.m)
        for _ in range(int(round(seconds / chunk_s))):
            st = chunk(st)
    finally:
        X.SHIELD = "idm"
    m = np.asarray(st.m) - m0
    out = {"cell": cell, "model": model(level)._asdict(), "copies": E, "seed": seed, "warm_s": warm_s, "seconds": seconds,
           "p_av": p_av, "friction": friction, **{k: m[:, X.METRICS.index(k)].tolist() for k in KEEP},
           "crashes": sum(m[:, X.METRICS.index(k)] for k in X.CRASH_NAMES).tolist(),
           "av_types": {k: float(m[:, X.METRICS.index("av_" + k)].sum()) for k in X.CRASH_NAMES if "av_" + k in X.METRICS}}
    return out


def path(cell):
    return f"{OUT}/{cell.replace(':', '_')}.json"


def analyse(cells):
    from avlab.whocrashes import holm, rate_ratio
    h = lambda c: sum(cells[c]["av_veh_s"]) / 3600  # noqa: E731
    k = lambda c: sum(cells[c]["av_crash"])  # noqa: E731
    rows = {}
    for c, d in cells.items():
        rows[c] = {"av_hours": h(c), "av_crash": k(c), "rate_per_h": k(c) / h(c), "ci": [_poisson_ci(k(c))[0] / h(c), _poisson_ci(k(c))[1] / h(c)],
                   "av_speed_kmh": 3.6 * sum(d["av_veh_m"]) / sum(d["av_veh_s"]), "hard_per_h": sum(d["av_hard"]) / h(c),
                   "envelope_share": sum(d["av_shield"]) * 0.2 / sum(d["av_veh_s"]), "city_crashes": sum(d["crashes"]),
                   "av_types": d["av_types"]}
    rr = lambda a, b: rate_ratio(k(a), h(a), k(b), h(b))  # noqa: E731
    out = {"cells": rows}
    if "idm:severe" in cells and "off:severe" in cells:
        out["H1_envelope_rr"] = rr("idm:severe", "off:severe")
        out["H1_pass"] = bool(out["H1_envelope_rr"]["rr"] < 1 and out["H1_envelope_rr"]["p"] < 0.05)
        out["success"] = out["H1_pass"]
    sec = [s for s in ("off:severe", "off:miss", "off:range") if s in cells and "off:clean" in cells]
    if sec:
        res = [rr(s, "off:clean") for s in sec]
        adj = holm([r["p"] for r in res])
        out["secondary"] = {s: {**r, "p_holm": float(a), "pass": bool(a < 0.05 and r["lo"] > 1)} for s, r, a in zip(sec, res, adj)}
    out["descriptive_vs_clean"] = {s: rr(s, "off:clean") for s in ("off:mild", "off:moderate", "off:speed") if s in cells and "off:clean" in cells}
    return out


def _poisson_ci(k, alpha=0.05):
    from scipy import stats
    lo = stats.chi2.ppf(alpha / 2, 2 * k) / 2 if k > 0 else 0.0
    return lo, stats.chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2


def load():
    return {c: json.load(open(path(c))) for c in CELLS if os.path.exists(path(c))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cell", action="append")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    if a.quick:
        for c in ("off:clean", "off:severe"):
            d = run_cell(c, E=2, warm_s=40.0, seconds=40.0)
            print(c, "AV crashes", sum(d["av_crash"]), "AV-h", round(sum(d["av_veh_s"]) / 3600, 2), flush=True)
        return
    os.makedirs(OUT, exist_ok=True)
    for c in (CELLS if a.all else a.cell or []):
        d = run_cell(c)
        json.dump(d, open(path(c), "w"))
        print(c, "AV crashes", sum(d["av_crash"]), "AV-h", round(sum(d["av_veh_s"]) / 3600, 1), flush=True)
    if not (a.all or a.cell):
        res = analyse(load())
        json.dump(res, open(f"{OUT}/sensor.json", "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))
        for c, r in res["cells"].items():
            print(f"{c:14s} AV-h {r['av_hours']:6.1f} crashes {r['av_crash']:4.0f} rate {r['rate_per_h']:.3f} speed {r['av_speed_kmh']:.2f} envelope {r['envelope_share']:.4f}")


if __name__ == "__main__":
    main()
