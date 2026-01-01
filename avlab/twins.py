"""Is the simulator chaotic? Twin runs: the same city twice, one of them nudged, and a measure of how far apart they drift.

    python -m avlab.twins            # A: 16 twins x 4 nudges x 10 simulated min (~6 min, 4 CPU cores) -> results/twins/twins.json
    python -m avlab.twins --quick    # a smoke run
    python -m avlab.twins --bench    # B: AVBench scores of the rule-based AV under the same nudges (~10 min) -> results/twins/bench.json
    python -m avlab.twins --plot     # figures and GIF from the saved results (docs/img/twins*.png|gif)

The leaderboard moved by up to 75 points per scenario between a GPU and a CPU (docs/FIXES.md, item 14), and the explanation offered
was "floating-point differences flip rare events". That is a claim about sensitive dependence on initial conditions, and it can be
measured: take a warmed-up city, make two copies with the same random numbers, nudge one, and compare vehicle by vehicle.

    for each of E seeds:  warm a city 120 s;  twin = copy of the state
    nudge (one of):   one car +1 mm/s | one car +3 m/s | every car +1 mm/s | a different random stream (new key), state untouched
    run both for 600 s in 20 s steps; displaced(t) = share of vehicles alive in both whose position differs by more than 1 m
    at the end: do the crash count and the trips finished agree?

Rounding differences between a CPU and a GPU are ~1e-7 relative, so a 1 mm/s nudge to every car (1e-4 relative at 10 m/s) is a
generous stand-in for them. A different random stream stands in for a different PRNG implementation (the repo's scripts set
`JAX_DEFAULT_PRNG_IMPL=unsafe_rbg`, whose numbers differ between backends).
"""
import argparse
import json
import os
import time

import numpy as np

import jax
import jax.numpy as jnp

OUT = "results/twins"
NUDGES = {  # name: (cars kicked or "all", m/s) or "stream"
    "1 car + 1 mm/s": (1, 1e-3),
    "1 car + 3 m/s": (1, 3.0),
    "every car + 1 mm/s": ("all", 1e-3),
    "new random stream": "stream",
}


def displaced(a, b, tol=1.0):
    """Share of vehicles alive in both states whose arc position differs by more than tol metres (same slot = same vehicle);
    a vehicle on a different track counts as displaced. Returns (share, number compared)."""
    both = np.asarray(a.alive & b.alive)
    far = (np.abs(np.asarray(a.s - b.s)) > tol) | (np.asarray(a.trk) != np.asarray(b.trk))
    n = both.sum(-1)
    return ((far & both).sum(-1) / np.maximum(n, 1)), n


def nudge(st, how, key):
    """A copy of the batched state with the nudge applied (the original is untouched: JAX arrays are immutable)."""
    if how == "stream":
        return st._replace(key=jax.vmap(lambda k: jax.random.fold_in(k, 1))(st.key))
    k, eps = how
    E = st.v.shape[0]
    if k == "all":
        return st._replace(v=st.v + eps * st.alive)
    idx = jnp.argmax(jnp.where(st.alive, st.v, -1.0), axis=1)  # the fastest car: it is moving, so a nudge has something to act on
    return st._replace(v=st.v.at[jnp.arange(E), idx].add(eps))


def run(area="oakland", E=16, warm_s=120.0, seconds=600.0, step_s=20.0, seed=0, keep_frames=False, nudges=NUDGES):
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated()
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), E), J, 1024)
    n = int(round(step_s / X.DT))

    @jax.jit
    def chunk(s):
        return jax.lax.scan(lambda c, _: (jax.vmap(lambda si: X.step(si, J, beh, nh))(c), None), s, None, length=n)[0]

    for _ in range(int(round(warm_s / step_s))):
        st = chunk(st)
    t0, m0 = time.time(), np.asarray(st.m)
    steps = int(round(seconds / step_s))
    base_frames, twin_frames = [], {}
    res = {"area": area, "copies": E, "warm_s": warm_s, "seconds": seconds, "step_s": step_s, "nudges": {}}
    base = [st]
    for _ in range(steps):
        base.append(chunk(base[-1]))
    crash = lambda m: np.asarray(m)[:, [X.METRICS.index(c) for c in X.CRASH_NAMES]].sum(1)
    fin = X.METRICS.index("finished")
    bm = np.asarray(base[-1].m) - m0
    res["base"] = {"crashes": crash(bm).tolist(), "finished": bm[:, fin].tolist(), "vehicles": float(np.asarray(st.alive).sum(1).mean())}
    for name, how in nudges.items():
        tw = [nudge(st, how, None)]
        for _ in range(steps):
            tw.append(chunk(tw[-1]))
        curve = np.stack([displaced(a, b)[0] for a, b in zip(base, tw)], 1)  # [E, steps+1]
        m = np.asarray(tw[-1].m) - m0
        res["nudges"][name] = {"displaced": curve.tolist(), "crashes": crash(m).tolist(), "finished": m[:, fin].tolist()}
        if keep_frames:
            twin_frames[name] = tw
        print(f"  {name:22s} displaced at 1/5/10 min: {curve[:, min(int(60 / step_s), steps)].mean():.3f} {curve[:, min(int(300 / step_s), steps)].mean():.3f} "
              f"{curve[:, -1].mean():.3f}   same crash count {np.mean(crash(m) == crash(bm)):.2f}  same trips {np.mean(m[:, fin] == bm[:, fin]):.2f}  "
              f"({time.time() - t0:.0f} s)", flush=True)
    res["wall_s"] = time.time() - t0
    return (res, base, twin_frames, (net, geo, info)) if keep_frames else res


def bench(scenarios=("nominal", "rush_hour", "red_runners", "wet_road"), part=None, only=None):
    """AVBench scores of the rule-based AV from one warm state, under four conditions: the leaderboard's, every car + 1 mm/s,
    every car + 1 m/s, and another random stream (the simulator's key re-keyed after the warm-up; `key_seed` would not do: only learned policies read it)."""
    from avrl import bench as B
    conds = {"as published": {}, "every car + 1 mm/s": dict(kick=1e-3), "every car + 1 m/s": dict(kick=1.0), "new random stream": dict(restream=1)}
    res, t0 = {}, time.time()
    for n in scenarios:
        res[n] = {}
        for c, kw in conds.items():
            if only and c not in only:
                continue
            res[n][c] = B.evaluate(None, [n], **kw)["scenarios"][n]["av_score"]
            print(f"  {n:12s} {c:20s} score {res[n][c]:6.1f}  ({time.time() - t0:.0f} s)", flush=True)
        if part:
            json.dump(res, open(f"{OUT}/bench_{part}.json", "w"), indent=1)
    return res


def summarize(res):
    """Numbers for the prose, from the saved twin results."""
    out = {}
    for name, r in res["nudges"].items():
        d = np.array(r["displaced"])
        out[name] = {"displaced_1min": float(d[:, min(int(60 / res["step_s"]), d.shape[1] - 1)].mean()), "displaced_5min": float(d[:, min(int(300 / res["step_s"]), d.shape[1] - 1)].mean()),
                     "displaced_10min": float(d[:, -1].mean()), "twins_with_any_displacement": float((d[:, -1] > 0).mean()),
                     "same_crashes": float(np.mean(np.array(r["crashes"]) == np.array(res["base"]["crashes"]))),
                     "same_trips": float(np.mean(np.array(r["finished"]) == np.array(res["base"]["finished"])))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--part", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="bench: only these conditions")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        from avlab import twins_plot
        return twins_plot.main()
    if a.bench:
        res = bench(tuple(a.scenarios) if a.scenarios else ("nominal", "rush_hour", "red_runners", "wet_road"), a.part or "all", a.only)
        if not a.scenarios:
            json.dump(res, open(f"{OUT}/bench.json", "w"), indent=1)
        return res
    res = run(E=4 if a.quick else 16, seconds=60.0 if a.quick else 600.0)
    res["summary"] = summarize(res)
    if not a.quick:
        json.dump(res, open(f"{OUT}/twins.json", "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
