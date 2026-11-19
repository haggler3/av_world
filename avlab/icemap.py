"""Where do patchy-ice crashes happen? The first use of the per-crash event log (`avlab.crashlog`).

Three cells in one compiled program (humans-only Oakland, calibrated drivers, random stream SEED, warm 120 s, 600 s logged),
with the patch mask of `avlab.patchyice` (links covering 50 % of length):

    P  friction 1.0, x 0.4 on the masked links      D  friction 1.0 everywhere      I  friction 0.4 everywhere

    H1  in P, crash rate on masked (icy) links / rate on unmasked (dry) links > 1, exact conditional Poisson, p < 0.05
    S1  masked links: P / I  (is the ice risk local?)       S2  unmasked links: P / D  (does ice spill over?)   Holm, m = 2

Pre-registration: results/daily/2026-11-19/predictions.md

    python -m avlab.icemap --quick   # 2 copies per cell, 60 s
    python -m avlab.icemap --run     # 32 copies per cell -> results/icemap/run.json
    python -m avlab.icemap           # analysis on the saved run -> results/icemap/icemap.json
"""
import argparse
import json
import os

import numpy as np

OUT = "results/icemap"
CELLS = ("P", "D", "I")
SEED, SHARE, ICE, MASK_SEED = 11, 0.5, 0.4, 0


def cell_behaviour(c):
    from examples import _common as C
    return {"P": C.calibrated(friction=1.0, patch_friction=ICE), "D": C.calibrated(friction=1.0),
            "I": C.calibrated(friction=0.4)}[c]


def run(E=32, warm_s=120.0, seconds=600.0, seed=SEED, area="oakland", chunk_s=60.0):
    import jax
    import jax.numpy as jnp
    from avlab import crashlog as CL
    from xnet import network as NW
    from xnet import sim as X
    net, geo, _ = NW.build(area)
    mask = X.patch_links(net, SHARE, seed=MASK_SEED)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    J["link_patch"] = jnp.asarray(mask)
    tags = [c for c in CELLS for _ in range(E)]
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *[cell_behaviour(c) for c in tags])
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), len(tags)), J, 1024)
    lg = CL.Logger(J, nh, B, cap=1024, chunk_s=chunk_s)
    st, _ = lg.run(st, warm_s)
    m0 = np.asarray(st.m)
    st, buf = lg.run(st, seconds)
    m = np.asarray(st.m) - m0
    rows = CL.table(buf)
    xy = CL.positions(net, geo, rows)
    ev = CL.events(rows, xy)
    counters = sum(m[:, X.METRICS.index(k)] for k in X.CRASH_NAMES)
    expo = np.asarray(buf.expo)
    return {"area": area, "copies": E, "seconds": seconds, "warm_s": warm_s, "seed": seed, "share": SHARE, "ice": ICE,
            "mask_seed": MASK_SEED, "tags": tags, "mask": mask.astype(int).tolist(),
            "rows": {k: np.asarray(v).tolist() for k, v in rows.items()} | {"x": xy[0].tolist(), "y": xy[1].tolist()},
            "events": {k: np.asarray(v).tolist() for k, v in ev.items()},
            "counter_crashes": counters.tolist(), "veh_s": m[:, X.METRICS.index("veh_s")].tolist(),
            "expo_masked_s": expo[:, mask].sum(1).tolist(), "expo_unmasked_s": expo[:, ~mask].sum(1).tolist()}


def analyse(res):
    from avlab.whocrashes import holm, rate_ratio
    tags = np.array(res["tags"])
    ev = {k: np.asarray(v) for k, v in res["events"].items()}
    ecell = tags[ev["copy"].astype(int)] if len(ev["copy"]) else np.array([])
    tm, tu = np.asarray(res["expo_masked_s"]) / 3600, np.asarray(res["expo_unmasked_s"]) / 3600
    per_copy_events = np.bincount(ev["copy"].astype(int), minlength=len(tags))
    out = {"checks": {"events_equal_counters_every_copy": bool(np.array_equal(per_copy_events, np.asarray(res["counter_crashes"]).astype(int))),
                      "events": int(len(ev["copy"])), "counter_crashes": float(np.sum(res["counter_crashes"])),
                      "expo_max_rel_err": float(np.max(np.abs((tm + tu) * 3600 - np.asarray(res["veh_s"])) / np.asarray(res["veh_s"])))}}
    cells = {}
    for c in CELLS:
        s = tags == c
        k_ice = int(((ecell == c) & (ev["patched"] > 0)).sum()); k_dry = int(((ecell == c) & (ev["patched"] == 0)).sum())
        cells[c] = {"k_masked": k_ice, "k_unmasked": k_dry, "t_masked_h": float(tm[s].sum()), "t_unmasked_h": float(tu[s].sum()),
                    "rate_masked_per_1000h": 1000 * k_ice / tm[s].sum(), "rate_unmasked_per_1000h": 1000 * k_dry / tu[s].sum()}
    out["cells"] = cells
    P, D, I_ = cells["P"], cells["D"], cells["I"]
    out["H1_P_masked_vs_unmasked"] = rate_ratio(P["k_masked"], P["t_masked_h"], P["k_unmasked"], P["t_unmasked_h"])
    out["S1_masked_P_vs_I"] = rate_ratio(P["k_masked"], P["t_masked_h"], I_["k_masked"], I_["t_masked_h"])
    out["S2_unmasked_P_vs_D"] = rate_ratio(P["k_unmasked"], P["t_unmasked_h"], D["k_unmasked"], D["t_unmasked_h"])
    # post hoc (not pre-registered): are the two link sets equally risky when the friction is the same everywhere?
    out["posthoc_D_masked_vs_unmasked"] = rate_ratio(D["k_masked"], D["t_masked_h"], D["k_unmasked"], D["t_unmasked_h"])
    out["posthoc_I_masked_vs_unmasked"] = rate_ratio(I_["k_masked"], I_["t_masked_h"], I_["k_unmasked"], I_["t_unmasked_h"])
    ph = holm([out["S1_masked_P_vs_I"]["p"], out["S2_unmasked_P_vs_D"]["p"]])
    out["S1_masked_P_vs_I"]["p_holm"], out["S2_unmasked_P_vs_D"]["p_holm"] = ph
    # cluster check: bootstrap over copies of P for the H1 ratio
    rng = np.random.default_rng(0)
    pc = np.where(tags == "P")[0]
    ki = np.array([((ev["copy"] == e) & (ev["patched"] > 0)).sum() for e in pc]); kd = np.array([((ev["copy"] == e) & (ev["patched"] == 0)).sum() for e in pc])
    bs = []
    for _ in range(4000):
        j = rng.integers(0, len(pc), len(pc))
        if kd[j].sum() > 0:
            bs.append((ki[j].sum() / tm[pc][j].sum()) / (kd[j].sum() / tu[pc][j].sum()))
    out["H1_copy_bootstrap_95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if bs else None
    h = out["H1_P_masked_vs_unmasked"]
    out["H1_pass"] = bool(h["rr"] > 1 and h["p"] < 0.05)
    # descriptive: impact speed (km/h) of logged vehicles by cell, surface and role
    r = {k: np.asarray(v) for k, v in res["rows"].items()}
    rcell = tags[r["copy"].astype(int)] if len(r["copy"]) else np.array([])
    from avlab.crashlog import ROLES
    out["impact_kmh_median"] = {c: {ROLES[q]: float(3.6 * np.median(r["v"][(rcell == c) & (r["role"] == q)]))
                                    for q in range(len(ROLES)) if ((rcell == c) & (r["role"] == q)).any()} for c in CELLS}
    out["vehicle_rows"] = {c: int((rcell == c).sum()) for c in CELLS}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.run or a.quick:
        res = run(E=2, warm_s=60.0, seconds=60.0) if a.quick else run()
        json.dump(res, open(f"{OUT}/run{'_quick' if a.quick else ''}.json", "w"))
        print(json.dumps(analyse(res)["checks"]), flush=True)
        if a.quick:
            return
    res = json.load(open(f"{OUT}/run.json"))
    out = analyse(res)
    json.dump(out, open(f"{OUT}/icemap.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
