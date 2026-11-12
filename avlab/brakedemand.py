"""Why is there a friction knee near 0.7? Log the braking human drivers actually do, cell by cell.

Friction enters the simulator only through the hardest braking the road allows, a_min = -9 mu m/s^2 (`xnet/sim.py:231`):
every acceleration is clipped to it, and it decides whether a car can still stop before a line (`stoppable`). If the knee
is mechanical, braking at that cap ("saturated braking") should be rare where the crash curve is flat (mu >= 0.7) and
start where the curve steepens, and the cars in crashes should have been braking at the cap just before.

This module only reads the simulator. Per friction cell it runs E copies of humans-only Oakland (one compiled, vmapped
run, the same protocol as avlab.knee.replicate) and, after the warm-up, accumulates per step:

    moving human  = alive & human & not wrecked & not halted & speed > 0.5 m/s   (state before the step)
    applied acc   = state.acc after the step (the clipped value the car used)
    histogram of braking (-acc) in 0.25 m/s^2 bins; saturated = acc <= a_min + 0.01
    sat_age       = seconds since the car last braked at the cap (per slot; reset when the slot is empty)
    crash-involved vehicles (diag 'crashed', not already a wreck): how many had sat_age <= 1 s

python -m avlab.brakedemand --quick         # 2 copies x 3 frictions, 60 s (a smoke test)
python -m avlab.brakedemand                 # 10 copies x 6 frictions, 480 s -> results/brakedemand/brake.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MUS = (1.0, 0.85, 0.7, 0.6, 0.5, 0.4)
BINS = np.arange(0.0, 10.01, 0.25)  # braking magnitude bins, m/s^2
RECENT_S = 1.0


def run(E=16, mus=MUS, warm_s=120.0, seconds=600.0, seed=2, area="oakland", chunk_s=60.0):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X

    behs, tags = [], []
    for mu in mus:
        behs += [C.calibrated(friction=mu) for _ in range(E)]
        tags += [mu] * E
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    keys = jax.random.split(jax.random.PRNGKey(seed), len(behs))
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
    nb = len(BINS) - 1
    edges = jnp.asarray(BINS)

    def one(st, b, sat_age):
        st2, d = X.step(st, J, b, nh, diag=True)
        a_min = -9.0 * b.friction
        moving = st.alive & ~st.av & (st.wreck_t <= 0) & (st.stop_t <= 0) & (st.v > 0.5)
        acc = st2.acc
        sat = moving & (acc <= a_min + 0.01)
        sat_age = jnp.where(sat, 0.0, jnp.where(st2.alive, sat_age + X.DT, 1e3))
        brk = moving & (acc < 0)
        idx = jnp.clip(jnp.searchsorted(edges, -acc, side="right") - 1, 0, nb - 1)
        hist = jnp.zeros(nb).at[jnp.where(brk, idx, nb)].add(1.0, mode="drop")
        inv = d["crashed"] & ~st.av & (st.wreck_t <= 0)
        recent = sat_age <= RECENT_S
        stats = jnp.stack([moving.sum(), sat.sum(), inv.sum(), (inv & recent).sum(), (moving & recent).sum()]).astype(jnp.float32)
        return st2, sat_age, hist, stats

    n = int(round(chunk_s / X.DT))

    @jax.jit
    def chunk(carry):
        def body(c, _):
            s, age, H, S = c
            s2, age2, h, sv = jax.vmap(one)(s, B, age)
            return (s2, age2, H + h, S + sv), None
        return jax.lax.scan(body, carry, None, length=n)[0]

    Ecop = len(behs)
    carry = (st, jnp.full((Ecop, 1024), 1e3), jnp.zeros((Ecop, nb)), jnp.zeros((Ecop, 5)))
    t0 = time.time()
    for _ in range(int(round(warm_s / chunk_s))):
        carry = chunk(carry)
    m0 = np.asarray(carry[0].m)
    carry = (carry[0], carry[1], jnp.zeros((Ecop, nb)), jnp.zeros((Ecop, 5)))
    for _ in range(int(round(seconds / chunk_s))):
        carry = chunk(carry)
    m = np.asarray(carry[0].m) - m0
    H, S = np.asarray(carry[2]), np.asarray(carry[3])
    tags = np.array(tags)
    dm = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    out = {"area": area, "copies": E, "seconds": seconds, "warm_s": warm_s, "seed": seed, "dt": float(X.DT), "bins": BINS.tolist(),
           "recent_s": RECENT_S, "wall_s": round(time.time() - t0, 1), "cells": {}}
    for mu in mus:
        s = tags == mu
        crashes = sum(dm[k][s] for k in X.CRASH_NAMES)
        out["cells"][f"{mu:g}"] = {
            "mu": mu, "cap": 9.0 * mu, "hist": H[s].sum(0).tolist(), "moving_steps": float(S[s, 0].sum()),
            "sat_steps": float(S[s, 1].sum()), "per_copy_sat": S[s, 1].tolist(), "per_copy_moving": S[s, 0].tolist(),
            "involved": float(S[s, 2].sum()), "involved_recent_sat": float(S[s, 3].sum()), "moving_recent_sat": float(S[s, 4].sum()),
            "crashes": float(crashes.sum()), "hours": float(dm["veh_s"][s].sum() / 3600), "per_copy_crashes": crashes.tolist(),
            "per_copy_hours": (dm["veh_s"][s] / 3600).tolist(), "types": {k: float(dm[k][s].sum()) for k in X.CRASH_NAMES}}
    return out


def exceed(hist, bins, a):
    """Share of braking steps with |deceleration| >= a (from a histogram)."""
    h = np.asarray(hist, float)
    lo = np.asarray(bins[:-1])
    return float(h[lo >= a - 1e-9].sum() / max(h.sum(), 1.0))


def ratio_ci(k1, n1, k0, n0, z=1.96):
    """Ratio of two proportions (k1/n1) / (k0/n0) with a log-scale Katz interval (0.5 added to empty cells)."""
    k1, k0 = max(k1, 0.5), max(k0, 0.5)
    r = (k1 / n1) / (k0 / n0)
    se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return float(r), float(r * np.exp(-z * se)), float(r * np.exp(z * se))


def summarise(res):
    """Per cell: saturated share (with a copy-level bootstrap interval), the dry-road exceedance prediction, crash rate, and
    the enrichment of recent saturated braking among crash-involved cars."""
    rng = np.random.default_rng(0)
    dry = res["cells"]["1"]
    rows = []
    for k, c in res["cells"].items():
        ps, pm = np.asarray(c["per_copy_sat"]), np.asarray(c["per_copy_moving"])
        bs = [ps[i].sum() / pm[i].sum() for i in (rng.integers(0, len(ps), len(ps)) for _ in range(999))]
        enr = ratio_ci(c["involved_recent_sat"], c["involved"], c["moving_recent_sat"], c["moving_steps"]) if c["involved"] else None
        rows.append({"mu": c["mu"], "cap": c["cap"], "sat_share": c["sat_steps"] / c["moving_steps"],
                     "sat_share_ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                     "dry_exceed_cap": exceed(dry["hist"], res["bins"], c["cap"]) * sum(dry["hist"]) / dry["moving_steps"],
                     "rate": 1000 * c["crashes"] / c["hours"], "crashes": c["crashes"], "involved": c["involved"],
                     "involved_recent_sat_share": c["involved_recent_sat"] / max(c["involved"], 1),
                     "moving_recent_sat_share": c["moving_recent_sat"] / c["moving_steps"], "enrichment": enr})
    rows.sort(key=lambda r: -r["mu"])
    q = {}
    h = np.asarray(dry["hist"], float)
    cum = np.cumsum(h) / h.sum()
    for p in (0.99, 0.999, 0.9999):
        q[str(p)] = float(res["bins"][int(np.searchsorted(cum, p)) + 1])
    pooled = {}
    for name, sel in (("mu>=0.7", lambda m: m >= 0.7), ("mu<=0.6", lambda m: m <= 0.6), ("all", lambda m: True)):
        cs = [c for c in res["cells"].values() if sel(c["mu"])]
        k1, n1 = sum(c["involved_recent_sat"] for c in cs), sum(c["involved"] for c in cs)
        k0, n0 = sum(c["moving_recent_sat"] for c in cs), sum(c["moving_steps"] for c in cs)
        pooled[name] = {"involved": n1, "involved_recent_sat": k1, "share": k1 / max(n1, 1), "base": k0 / n0,
                        "enrichment": ratio_ci(k1, n1, k0, n0) if n1 else None}
    from scipy.stats import spearmanr
    rho = spearmanr([r["sat_share"] for r in rows], [r["rate"] for r in rows])
    return {"rows": rows, "dry_brake_quantiles": q, "pooled_enrichment": pooled, "spearman_sat_rate": [float(rho[0]), float(rho[1])]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--copies", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=480.0)
    ap.add_argument("--out", default="results/brakedemand")
    a = ap.parse_args()
    if a.quick:
        res = run(E=2, mus=(1.0, 0.7, 0.4), warm_s=60.0, seconds=60.0)
    else:
        res = run(E=a.copies, seconds=a.seconds)
    res["summary"] = summarise(res)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "quick.json" if a.quick else "brake.json")
    json.dump(res, open(fn, "w"))
    for r in res["summary"]["rows"]:
        print(f"mu {r['mu']:.2f} cap {r['cap']:.2f}  sat {100 * r['sat_share']:.4f}%  dry-exceed {100 * r['dry_exceed_cap']:.4f}%  "
              f"rate {r['rate']:.1f}  involved {r['involved']:.0f}  recent-sat {100 * r['involved_recent_sat_share']:.1f}% "
              f"vs {100 * r['moving_recent_sat_share']:.3f}%  enrichment {r['enrichment']}")
    print("dry quantiles", res["summary"]["dry_brake_quantiles"], "wall", res["wall_s"], "->", fn)


if __name__ == "__main__":
    main()
