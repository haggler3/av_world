"""Who rear-ends whom in a half-AV city (first use of the crash log's partner fields, 2026-11-25).

The crash log now names the other vehicle of every rear-end (`avlab.crashlog`, fields partner, p_av, p_kind, p_v,
p_wreck). This experiment counts, among rear-ends between an AV and a human-driven vehicle, how often the AV is the car that
was struck. Half the cars are AVs (IDM policy, envelope off), friction 0.4 everywhere (enough crashes in 10 minutes).

    python -m avlab.rearpartner --quick   # 4 copies, 60 s
    python -m avlab.rearpartner --run     # 96 copies (3 x 32), 120 s warm-up + 600 s -> results/rearpartner/run.json
"""
import argparse
import json
import os
import time

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import numpy as np  # noqa: E402

OUT = "results/rearpartner"
SEED, SHARE_AV, FRICTION = 51, 0.5, 0.4


def run(E=32, batches=3, warm_s=120.0, seconds=600.0, seed=SEED, area="oakland"):
    import jax
    import jax.numpy as jnp
    from avlab import crashlog as CL
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, _ = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *[C.calibrated(p_av=SHARE_AV, friction=FRICTION)] * E)
    lg = CL.Logger(J, nh, B, cap=1024, chunk_s=60.0)
    allrows, av_s, hum_s = [], 0.0, 0.0
    for b in range(batches):
        keys = jax.random.split(jax.random.PRNGKey(seed * 1000 + b), E)
        st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
        st, _ = lg.run(st, warm_s)
        t0 = time.time()
        # exposure by AV flag: count alive vehicle-steps per chunk end (a cheap sample, every 60 s)
        st, buf = lg.run(st, seconds)
        rows = CL.table(buf)
        rows["copy"] = rows["copy"] + b * E
        allrows.append(rows)
        alive, av = np.asarray(st.alive), np.asarray(st.av)
        av_s += float((alive & av).sum()); hum_s += float((alive & ~av).sum())
        print(f"batch {b}: {len(rows['t'])} rows, {time.time() - t0:.0f} s", flush=True)
    rows = {k: np.concatenate([r[k] for r in allrows]) for k in allrows[0]}
    return {"area": area, "copies": E * batches, "seconds": seconds, "warm_s": warm_s, "seed": seed, "p_av": SHARE_AV,
            "friction": FRICTION, "alive_av_end": av_s, "alive_human_end": hum_s,
            "rows": {k: v.tolist() for k, v in rows.items()}}


def binom_two_sided(k, n, p=0.5):
    from scipy.stats import binomtest
    r = binomtest(int(k), int(n), p)
    ci = r.proportion_ci(0.95, method="exact")
    return float(r.pvalue), float(ci.low), float(ci.high)


def analyse(res):
    from avlab import crashlog as CL
    rows = {k: np.asarray(v) for k, v in res["rows"].items()}
    rp = CL.rear_pairs(rows, fresh=True)
    sa, ka = rp["striker_av"] == 1, rp["struck_av"] == 1
    av_struck, av_striker = int((ka & ~sa).sum()), int((sa & ~ka).sum())
    n_mixed = av_struck + av_striker
    p1, lo1, hi1 = binom_two_sided(av_struck, n_mixed)
    # S1: human strikers hit AVs vs humans in proportion to the AV share of leaders? compare share of AV among struck cars
    # with a human striker against the AV share of alive vehicles at chunk ends
    hum_strk = ~sa
    k_s1, n_s1 = int((hum_strk & ka).sum()), int(hum_strk.sum())
    share_av = res["alive_av_end"] / (res["alive_av_end"] + res["alive_human_end"])
    p2, lo2, hi2 = binom_two_sided(k_s1, n_s1, share_av)
    # S2: impact speed difference (striker speed minus struck speed) for human strikers: AV leaders vs human leaders
    from scipy.stats import mannwhitneyu
    dv = rp["striker_v"] - rp["struck_v"]
    a, h = dv[hum_strk & ka], dv[hum_strk & ~ka]
    p3 = float(mannwhitneyu(a, h).pvalue) if len(a) and len(h) else float("nan")
    ps = sorted([("S1", p2), ("S2", p3)], key=lambda x: x[1])
    holm = {}
    run_max = 0.0
    for i, (name, p) in enumerate(ps):
        run_max = max(run_max, min(1.0, p * (len(ps) - i)))
        holm[name] = run_max
    roles = {CL.ROLES[r]: int((rows["role"] == r).sum()) for r in range(len(CL.ROLES))}
    rear_rows = rows["role"] == 1
    return {"rows": int(len(rows["t"])), "roles": roles,
            "rear_rows": int(rear_rows.sum()), "rear_paired": int((rear_rows & (rows["partner"] >= 0)).sum()),
            "rear_into_wreck": int((rear_rows & (rows["p_wreck"] == 1)).sum()), "fresh_pairs": int(len(sa)),
            "pairs": {"human_on_human": int((~sa & ~ka).sum()), "human_on_av": av_struck, "av_on_human": av_striker,
                      "av_on_av": int((sa & ka).sum())},
            "H1": {"k_av_struck": av_struck, "n_mixed": n_mixed, "share": av_struck / max(n_mixed, 1), "ci": [lo1, hi1], "p": p1},
            "S1": {"k": k_s1, "n": n_s1, "share": k_s1 / max(n_s1, 1), "ci": [lo2, hi2], "null_share_av": share_av, "p": p2, "holm": holm["S1"]},
            "S2": {"median_dv_av_leader": float(np.median(a)) if len(a) else None, "median_dv_human_leader": float(np.median(h)) if len(h) else None,
                   "n_av": int(len(a)), "n_human": int(len(h)), "p": p3, "holm": holm["S2"]}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    res = run(E=4, batches=1, warm_s=60.0, seconds=60.0) if a.quick else run()
    res["analysis"] = analyse(res)
    print(json.dumps(res["analysis"], indent=1))
    json.dump(res, open(f"{OUT}/{'quick' if a.quick else 'run'}.json", "w"))
