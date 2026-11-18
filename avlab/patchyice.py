"""Patchy ice: is ice on half the streets worse than the same mean friction everywhere?

Uses the per-street friction of `xnet.sim` (2026-11-18): `J["link_patch"]` marks links whose friction is
friction x patch_friction. Four cells in one compiled program (humans-only Oakland, calibrated drivers, the protocol of
`avlab.knee.replicate`: warm 120 s, 600 s scored, random stream `seed`):

    U1.0   friction 1.0 everywhere            U0.7   friction 0.7 everywhere
    U0.4   friction 0.4 everywhere            P      friction 1.0, x 0.4 on links covering 50 % of the network's length
                                                     (length-weighted mean friction 0.7)

    H1 (pre-registered)  crash rate ratio P / U0.7 > 1, exact conditional Poisson test, two-sided p < 0.05
    S1 (secondary)       P against the mixture of the uniform cells: E = T_P (w r_0.4 + (1 - w) r_1.0), with w the share of P's
                         vehicle-steps on patched links (from st.link_n); log(O / E) by the delta method
    S2 (secondary)       P / U0.4 < 1                                     (S1, S2 Holm-corrected)

    python -m avlab.patchyice --quick   # 3 copies per cell, 120 s
    python -m avlab.patchyice --run     # 40 copies per cell -> results/patchyice/patchyice.json
    python -m avlab.patchyice           # analysis + figure on the saved run
"""
import argparse
import json
import os

import numpy as np

OUT = "results/patchyice"
CELLS = ("U1.0", "U0.7", "U0.4", "P")
SHARE, ICE, SEED, MASK_SEED = 0.5, 0.4, 9, 0


def cell_behaviour(c):
    from examples import _common as C
    return {"U1.0": C.calibrated(friction=1.0), "U0.7": C.calibrated(friction=0.7), "U0.4": C.calibrated(friction=0.4),
            "P": C.calibrated(friction=1.0, patch_friction=ICE)}[c]


def run(E=40, warm_s=120.0, seconds=600.0, seed=SEED, area="oakland", n_slots=1024, chunk_s=60.0):
    import jax
    import jax.numpy as jnp
    from xnet import network as NW
    from xnet import sim as X
    net, _, _ = NW.build(area)
    mask = X.patch_links(net, SHARE, seed=MASK_SEED)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    J["link_patch"] = jnp.asarray(mask)
    behs, tags = [], []
    for c in CELLS:
        behs += [cell_behaviour(c)] * E; tags += [c] * E
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), len(behs)), J, n_slots)
    n = int(round(chunk_s / X.DT))

    @jax.jit
    def chunk(st):
        return jax.lax.scan(lambda s, _: (jax.vmap(lambda si, b: X.step(si, J, b, nh))(s, B), None), st, None, length=n)[0]

    for _ in range(int(round(warm_s / chunk_s))):
        st = chunk(st)
    m0, n0 = np.asarray(st.m), np.asarray(st.link_n)
    for _ in range(int(round(seconds / chunk_s))):
        st = chunk(st)
    m, ln = np.asarray(st.m) - m0, np.asarray(st.link_n) - n0
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    tags = np.array(tags)
    res = {"area": area, "copies": E, "seconds": seconds, "warm_s": warm_s, "seed": seed, "share": SHARE, "ice": ICE,
           "mask_seed": MASK_SEED, "patched_links": int(mask.sum()), "links": int(len(mask)),
           "patched_length_share": float(np.asarray(net["link_len"])[mask].sum() / np.asarray(net["link_len"]).sum()), "cells": {}}
    for c in CELLS:
        s = tags == c
        types = {k: float(d[k][s].sum()) for k in X.CRASH_NAMES}
        steps = ln[s].sum(0)
        res["cells"][c] = {"types": types, "crashes": float(sum(types.values())), "hours": float(d["veh_s"][s].sum() / 3600),
                           "speed_kmh": float(3.6 * d["veh_m"][s].sum() / d["veh_s"][s].sum()),
                           "patched_step_share": float(steps[mask].sum() / steps.sum()),
                           "per_copy_crashes": sum(d[k][s] for k in X.CRASH_NAMES).tolist(), "per_copy_hours": (d["veh_s"][s] / 3600).tolist()}
    return res


def mixture_test(kp, tp, w, k4, t4, k1, t1):
    """Observed crashes kp in exposure tp against the mixture expectation E = tp (w k4/t4 + (1-w) k1/t1).
    Delta method: var log(O/E) = 1/kp + tp^2 (w^2 k4/t4^2 + (1-w)^2 k1/t1^2) / E^2. Returns ratio, 95 % interval, two-sided p."""
    from scipy import stats
    E = tp * (w * k4 / t4 + (1 - w) * k1 / t1)
    var = 1 / kp + tp ** 2 * (w ** 2 * k4 / t4 ** 2 + (1 - w) ** 2 * k1 / t1 ** 2) / E ** 2
    lr, se = np.log(kp / E), np.sqrt(var)
    return {"observed": float(kp), "expected": float(E), "ratio": float(kp / E), "lo": float(np.exp(lr - 1.96 * se)),
            "hi": float(np.exp(lr + 1.96 * se)), "p": float(2 * stats.norm.sf(abs(lr / se)))}


def analyse(res):
    from avlab.whocrashes import holm, rate_ratio
    c = res["cells"]
    k = {n: c[n]["crashes"] for n in CELLS}
    t = {n: c[n]["hours"] for n in CELLS}
    out = {"rates_per_1000_veh_h": {n: 1000 * k[n] / t[n] for n in CELLS}}
    out["H1_P_vs_U0.7"] = rate_ratio(k["P"], t["P"], k["U0.7"], t["U0.7"])
    out["S1_P_vs_mixture"] = mixture_test(k["P"], t["P"], c["P"]["patched_step_share"], k["U0.4"], t["U0.4"], k["U1.0"], t["U1.0"])
    out["S2_P_vs_U0.4"] = rate_ratio(k["P"], t["P"], k["U0.4"], t["U0.4"])
    ph = holm([out["S1_P_vs_mixture"]["p"], out["S2_P_vs_U0.4"]["p"]])
    out["S1_P_vs_mixture"]["p_holm"], out["S2_P_vs_U0.4"]["p_holm"] = float(ph[0]), float(ph[1])
    out["H1_pass"] = bool(out["H1_P_vs_U0.7"]["rr"] > 1 and out["H1_P_vs_U0.7"]["p"] < 0.05)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.run or a.quick:
        res = run(E=3, warm_s=60.0, seconds=120.0) if a.quick else run()
        json.dump(res, open(f"{OUT}/run{'_quick' if a.quick else ''}.json", "w"))
        print({n: (cc["crashes"], round(cc["hours"], 1), round(cc["patched_step_share"], 3)) for n, cc in res["cells"].items()}, flush=True)
        if a.quick:
            return
    res = json.load(open(f"{OUT}/run.json"))
    out = {"run": {k: v for k, v in res.items() if k != "cells"},
           "cells": {n: {k: v for k, v in cc.items() if not k.startswith("per_copy")} for n, cc in res["cells"].items()}} | analyse(res)
    json.dump(out, open(f"{OUT}/patchyice.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1))
    from avlab.patchyice_plot import plot
    plot(out)


if __name__ == "__main__":
    main()
