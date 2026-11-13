"""Which of friction's channels makes the knee near 0.7? And can the road's friction be read off braking alone?

Friction (`beh.friction`, mu) enters `xnet/sim.py` in three places, all through a_min = -9 mu m/s^2:

    1. the braking cap        acc = clip(acc, a_min, amax)                                  (every car, every step)
    2. the commitment rule     can_stop = v^2 / (2 min(4, 9 mu)) < dist  or  v < 2           (line ~374)
                               a car that cannot stop is 'committed' and enters without looking.
                               min(4, 9 mu) equals 4 for every mu >= 4/9, so this channel is OFF above mu = 0.444.
    3. the stop-line overrun   stoppable = v^2 / (2 * 9 mu) <= dist + 1  or  tlen < 20       (line ~528)
                               a car that may not enter but is not stoppable is not held at the line: it overruns.

This module only reads the simulator (X.step(..., diag=True)) and counts, per friction cell, human car-steps in each
channel, with the dry-road counterfactual computed on the same state (mu = 1 in the formulas, nothing re-simulated):

    moving        = alive & human & not wrecked & not halted & v > 0.5          (state before the step)
    told_stop     = moving & mv >= 0 & ~enter & ~distracted                      (at a link end, not allowed in)
    overrun       = told_stop & ~stoppable(mu)            fric_overrun = overrun & stoppable(1.0)
    blind_commit  = moving & mv >= 0 & enter & ~(gap_ok & ~cw_busy) & ~can_stop(mu)
    fric_commit   = blind_commit & can_stop(1.0)          (entered blind only because the road is slick)
    ages per slot: seconds since the car was last in fric_overrun / fric_commit / braking at the cap
    crash-involved (diag 'crashed', human, not already a wreck): how many were in each channel within 2 s

It also keeps, per copy and per 10 s, a fine histogram (0.05 m/s^2) of the braking of moving humans, for the
friction-tomography estimator in `tomography()`.

python -m avlab.stopline --quick     # 2 copies x 3 frictions, 60 s
python -m avlab.stopline             # 16 copies x 6 frictions, 480 s, seed 3 -> results/stopline/stopline.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MUS = (1.0, 0.85, 0.7, 0.6, 0.5, 0.4)
RECENT_S = 2.0
FINE = np.round(np.arange(0.0, 10.0001, 0.05), 4)  # fine braking bins for tomography
CHUNK_S = 10.0
STATS = ("moving", "told_stop", "overrun", "fric_overrun", "blind_commit", "fric_commit", "sat",
         "involved", "inv_fric_overrun", "inv_overrun", "inv_fric_commit", "inv_sat",
         "mov_fric_overrun", "mov_overrun", "mov_fric_commit", "mov_sat")


def stoppable(v, dist, tlen, mu):
    """The simulator's overrun rule (xnet/sim.py, `stoppable`), as a plain function of mu."""
    return (v ** 2 / (2 * 9.0 * mu) <= dist + 1.0) | (tlen < 20.0)


def can_stop(v, dist, mu):
    """The simulator's commitment rule (xnet/sim.py, `can_stop`): False = too fast to stop, so committed."""
    return (v ** 2 / (2 * np.minimum(4.0, 9.0 * mu)) < dist) | (v < 2.0)


def run(E=16, mus=MUS, warm_s=120.0, seconds=480.0, seed=3, area="oakland"):
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
    nb = len(FINE) - 1
    edges = jnp.asarray(FINE)

    def one(st, b, ages):
        st2, d = X.step(st, J, b, nh, diag=True)
        mu = b.friction
        a_min = -9.0 * mu
        moving = st.alive & ~st.av & (st.wreck_t <= 0) & (st.stop_t <= 0) & (st.v > 0.5)
        v, dist, tlen = d["v"], d["dist"], d["tlen"]
        at_end = moving & (d["mv"] >= 0)
        told = at_end & ~d["enter"] & ~d["distracted"]
        ovr = told & ~d["stoppable"]
        f_ovr = ovr & ((v ** 2 / 18.0 <= dist + 1.0) | (tlen < 20.0))
        cs_mu = (v ** 2 / (2 * jnp.minimum(4.0, -a_min)) < dist) | (v < 2.0)
        cs_dry = (v ** 2 / 8.0 < dist) | (v < 2.0)
        blind = at_end & d["enter"] & ~(d["gap_ok"] & ~d["cw_busy"]) & ~cs_mu
        f_com = blind & cs_dry
        acc = st2.acc
        sat = moving & (acc <= a_min + 0.01)
        alive2 = st2.alive
        new = []
        for flag, age in zip((f_ovr, ovr, f_com, sat), ages):
            new.append(jnp.where(flag, 0.0, jnp.where(alive2, age + X.DT, 1e3)))
        rec = [a <= RECENT_S for a in new]
        inv = d["crashed"] & ~st.av & (st.wreck_t <= 0)
        brk = moving & (acc < 0)
        idx = jnp.clip(jnp.searchsorted(edges, -acc, side="right") - 1, 0, nb - 1)
        hist = jnp.zeros(nb).at[jnp.where(brk, idx, nb)].add(1.0, mode="drop")
        s = [moving, told, ovr, f_ovr, blind, f_com, sat, inv,
             inv & rec[0], inv & rec[1], inv & rec[2], inv & rec[3],
             moving & rec[0], moving & rec[1], moving & rec[2], moving & rec[3]]
        return st2, tuple(new), hist, jnp.stack([x.sum() for x in s]).astype(jnp.float32)

    n = int(round(CHUNK_S / X.DT))

    @jax.jit
    def chunk(carry):
        def body(c, _):
            s, ages, H, S = c
            s2, ages2, h, sv = jax.vmap(one)(s, B, ages)
            return (s2, ages2, H + h, S + sv), None
        return jax.lax.scan(body, carry, None, length=n)[0]

    Ec = len(behs)
    ages0 = tuple(jnp.full((Ec, 1024), 1e3) for _ in range(4))
    zero = lambda: (jnp.zeros((Ec, nb)), jnp.zeros((Ec, len(STATS))))
    carry = (st, ages0) + zero()
    t0 = time.time()
    for _ in range(int(round(warm_s / CHUNK_S))):
        carry = chunk(carry)
    m0 = np.asarray(carry[0].m)
    S = np.zeros((Ec, len(STATS)))
    Hs = []
    for _ in range(int(round(seconds / CHUNK_S))):
        carry = chunk(carry[:2] + zero())
        Hs.append(np.asarray(carry[2]))
        S += np.asarray(carry[3])
    m = np.asarray(carry[0].m) - m0
    Hs = np.stack(Hs, 1)  # copies x chunks x bins
    tags = np.array(tags)
    dm = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    out = {"area": area, "copies": E, "seconds": seconds, "warm_s": warm_s, "seed": seed, "dt": float(X.DT), "recent_s": RECENT_S,
           "chunk_s": CHUNK_S, "fine_bins": FINE.tolist(), "stats_names": list(STATS), "wall_s": round(time.time() - t0, 1), "cells": {}}
    for mu in mus:
        s = tags == mu
        crashes = sum(dm[k][s] for k in X.CRASH_NAMES)
        out["cells"][f"{mu:g}"] = {
            "mu": mu, "stats": dict(zip(STATS, S[s].sum(0).tolist())), "per_copy_stats": S[s].tolist(),
            "crashes": float(crashes.sum()), "hours": float(dm["veh_s"][s].sum() / 3600), "per_copy_crashes": crashes.tolist(),
            "per_copy_hours": (dm["veh_s"][s] / 3600).tolist(), "types": {k: float(dm[k][s].sum()) for k in X.CRASH_NAMES},
            # sparse per-copy, per-chunk histograms: only bins >= 2 m/s^2 are kept (the estimator looks at the tail)
            "tail_hist": Hs[s][:, :, 40:].astype(int).tolist(), "light_brake": Hs[s][:, :, :40].sum(2).astype(int).tolist()}
    return out


# ---------------------------------------------------------------- friction tomography

def mu_hat_max(tail_hist, lo=2.0, width=0.05):
    """No-noise estimator: the hardest braking seen (top non-empty bin's upper edge) / 9. A lower bound on mu, exact
    once any car has braked at the cap. Returns nan if nothing was seen."""
    h = np.asarray(tail_hist)
    nz = np.nonzero(h)[0]
    if len(nz) == 0:
        return np.nan
    return float((lo + (nz[-1] + 1) * width) / 9.0)


def mu_hat_spike(tail_hist, lo=2.0, width=0.05, sigma=0.0, rng=None, min_count=3, search_from=3.0):
    """Spike estimator: braking at the cap piles up in one bin (a spike), so mu = (location of the largest bin beyond
    `search_from`) / 9. With sensor noise sigma (m/s^2) each braking sample is jittered before binning (a simulated sensor).
    The spike is found as the peak of log(narrow / wide) Gaussian smoothings, so the smooth background is ignored.
    Searching from 3 m/s^2 means it cannot report mu below 0.33. Returns nan with fewer than `min_count` tail samples."""
    h = np.asarray(tail_hist, int)
    if h.sum() < min_count:
        return np.nan
    centres = lo + (np.arange(len(h)) + 0.5) * width
    if sigma > 0:
        rng = rng or np.random.default_rng(0)
        x = np.repeat(centres, h) + rng.normal(0, sigma, h.sum())
        h = np.histogram(x, bins=np.append(centres - width / 2, centres[-1] + width / 2))[0]
    # difference of Gaussians: a narrow kernel (the spike's width under noise) minus a wide one (the smooth background)
    def g(s):
        x = np.arange(-int(4 * s / width) - 1, int(4 * s / width) + 2) * width
        k = np.exp(-0.5 * (x / s) ** 2)
        return k / k.sum()
    s_n = max(np.hypot(sigma, width), 2 * width)
    def smooth(k):  # the histogram is cut at `lo`: reflect it there so the cut does not look like a drop
        o = (len(k) - 1) // 2
        hp = np.concatenate([h[:o][::-1], h, np.zeros(o)])
        return np.convolve(hp, k, mode="valid")[:len(h)]
    score = np.log(smooth(g(s_n)) + 0.5) - np.log(smooth(g(min(max(4 * s_n, 0.75), 1.0))) + 0.5)  # log ratio: a spike over its background
    score[centres < search_from] = -np.inf  # the histogram is cut at `lo`; that edge is not a spike
    return float(centres[int(np.argmax(score))] / 9.0)


def tomography(res, windows=(10, 20, 40, 80, 160, 320), sigmas=(0.0, 0.3, 0.5), tol=0.05, seed=0):
    """For each cell, window W (seconds of one copy's city, from the start of counting) and sensor noise: share of copies
    whose estimate is within `tol` of the true mu, and the median absolute error. Car-seconds = moving-car steps x dt."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, c in res["cells"].items():
        H = np.asarray(c["tail_hist"])  # copies x chunks x bins
        mu = c["mu"]
        light = np.asarray(c["light_brake"])
        rows = []
        for W in windows:
            nchunks = max(1, int(round(W / res["chunk_s"])))
            if nchunks > H.shape[1]:
                continue
            for sg in sigmas:
                est = []
                for i in range(H.shape[0]):
                    h = H[i, :nchunks].sum(0)
                    est.append(mu_hat_max(h) if sg == 0 else mu_hat_spike(h, sigma=sg, rng=rng))
                est = np.asarray(est)
                err = np.abs(est - mu)
                rows.append({"window_s": W, "sigma": sg, "estimator": "max" if sg == 0 else "spike",
                             "within_tol": float(np.mean(err <= tol + 1e-9)), "median_abs_err": float(np.nanmedian(err)),
                             "braking_samples_per_copy": float((H[:, :nchunks].sum((1, 2)) + light[:, :nchunks].sum(1)).mean())})
        out[k] = rows
    return out


# ---------------------------------------------------------------- summary

def ratio_ci(k1, n1, k0, n0, z=1.96):
    """(k1/n1) / (k0/n0) with a Katz log interval (0.5 added to empty counts)."""
    k1, k0 = max(k1, 0.5), max(k0, 0.5)
    r = (k1 / n1) / (k0 / n0)
    se = np.sqrt(max(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0, 0.0))
    return float(r), float(r * np.exp(-z * se)), float(r * np.exp(z * se))


def loglog_slopes(mus, y, knee=0.7):
    """Slopes of log y on log mu above (mu >= knee) and below (mu <= knee) the knee; zeros are dropped."""
    mus, y = np.asarray(mus, float), np.asarray(y, float)
    out = {}
    for name, sel in (("above", mus >= knee - 1e-9), ("below", mus <= knee + 1e-9)):
        m = sel & (y > 0)
        out[name] = float(np.polyfit(np.log(mus[m]), np.log(y[m]), 1)[0]) if m.sum() >= 2 else float("nan")
    return out


def summarise(res):
    rows = []
    for k, c in res["cells"].items():
        s = c["stats"]
        mv = s["moving"]
        r = {"mu": c["mu"], "crashes": c["crashes"], "rate": 1000 * c["crashes"] / c["hours"], "involved": s["involved"]}
        for ch in ("overrun", "fric_overrun", "fric_commit", "sat"):
            r[f"{ch}_per_1e5"] = 1e5 * s[ch] / mv
            inv_k = s[f"inv_{ch}"]
            r[f"inv_{ch}_share"] = inv_k / max(s["involved"], 1)
            r[f"enrich_{ch}"] = ratio_ci(inv_k, s["involved"], s[f"mov_{ch}"], mv) if s["involved"] else None
        r["told_stop_per_1e5"] = 1e5 * s["told_stop"] / mv
        r["blind_commit_per_1e5"] = 1e5 * s["blind_commit"] / mv
        rows.append(r)
    rows.sort(key=lambda r: -r["mu"])
    mus = [r["mu"] for r in rows]
    slopes = {ch: loglog_slopes(mus, [r[f"{ch}_per_1e5"] for r in rows]) for ch in ("fric_overrun", "overrun", "sat")}
    slopes["crash_rate"] = loglog_slopes(mus, [r["rate"] for r in rows])
    from scipy.stats import spearmanr
    rho = {ch: [float(x) for x in spearmanr([r[f"{ch}_per_1e5"] for r in rows], [r["rate"] for r in rows])]
           for ch in ("fric_overrun", "sat")}
    pooled = {}
    for name, sel in (("mu>=0.7", lambda m: m >= 0.7), ("mu<=0.6", lambda m: m <= 0.6), ("all", lambda m: True)):
        cs = [c["stats"] for c in res["cells"].values() if sel(c["mu"])]
        n1 = sum(c["involved"] for c in cs)
        n0 = sum(c["moving"] for c in cs)
        pooled[name] = {"involved": n1}
        for ch in ("fric_overrun", "overrun", "fric_commit", "sat"):
            k1, k0 = sum(c[f"inv_{ch}"] for c in cs), sum(c[f"mov_{ch}"] for c in cs)
            pooled[name][ch] = {"k": k1, "share": k1 / max(n1, 1), "base": k0 / n0, "enrichment": ratio_ci(k1, n1, k0, n0) if n1 else None}
    return {"rows": rows, "loglog_slopes": slopes, "spearman_vs_rate": rho, "pooled": pooled}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--copies", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=480.0)
    ap.add_argument("--out", default="results/stopline")
    a = ap.parse_args()
    if a.quick:
        res = run(E=2, mus=(1.0, 0.7, 0.4), warm_s=60.0, seconds=60.0)
    else:
        res = run(E=a.copies, seconds=a.seconds)
    res["summary"] = summarise(res)
    res["tomography"] = tomography(res, windows=(10, 20, 40) if a.quick else (10, 20, 40, 80, 160, 320))
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, "quick.json" if a.quick else "stopline.json")
    json.dump(res, open(fn, "w"))
    for r in res["summary"]["rows"]:
        print(f"mu {r['mu']:.2f} rate {r['rate']:.1f} crashes {r['crashes']:.0f} inv {r['involved']:.0f} | per 1e5 moving: "
              f"told {r['told_stop_per_1e5']:.0f} ovr {r['overrun_per_1e5']:.1f} f_ovr {r['fric_overrun_per_1e5']:.1f} "
              f"f_com {r['fric_commit_per_1e5']:.2f} sat {r['sat_per_1e5']:.0f} | inv share f_ovr {r['inv_fric_overrun_share']:.2f} "
              f"ovr {r['inv_overrun_share']:.2f} sat {r['inv_sat_share']:.2f}")
    print("slopes", res["summary"]["loglog_slopes"])
    print("pooled", json.dumps(res["summary"]["pooled"], indent=0)[:1500])
    for k, rows in res["tomography"].items():
        print(k, [(r["window_s"], r["sigma"], round(r["within_tol"], 2), round(r["median_abs_err"], 3)) for r in rows])
    print("wall", res["wall_s"], "->", fn)


if __name__ == "__main__":
    main()
