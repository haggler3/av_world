"""Paired comparison of two worlds with common random numbers.

Two worlds that differ in one thing (a policy, the AV share, a human-risk knob) are run from the same seeds. With
`Behaviour.crn = 1` every per-vehicle dice roll is a hash of (step, vehicle id, stream), so a vehicle keeps its dice in both
worlds and the worlds differ by the fewest possible events (see docs/WORLD_MODEL_UPGRADES.md, section 1). The comparison is
then a paired difference over copies, and `variance_ratio` says what that bought: Var(A) + Var(B) over Var(A - B), measured on
the same copies.

    python -m avlab.paired            # distraction x2 and 25 % AVs, CRN off vs on -> results/crn/paired.json (~6 min on 4 cores)
"""
import json
import os
import sys
import time

import jax
import numpy as np

from xnet import network as NW
from xnet import sim as X

OUT = "results/crn"
STRESS = dict(p_red=0.3, p_nolook=0.01, distract_rate=1 / 90, p_aggressive=0.3, ped_scale=3.0, p_ped_violate=0.2)


def run_world(net, beh, n_envs, seconds, seed, warm=0.0, n_slots=768):
    """Per-copy metric rows (n_envs, len(X.METRICS)) after `seconds` from fresh copies with seeds PRNGKey(seed) split."""
    sim = X.Sim(net, n_slots=n_slots, n_envs=n_envs, beh=beh, seed=seed, reroute_s=0)
    sim.run(warm + seconds)
    return np.asarray(sim.st.m, float)


def outcomes(m):
    """Per-copy outcomes from the metric rows: crashes, mean speed (km/h), red runs, lane changes."""
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    crashes = sum(d[k] for k in X.CRASH_NAMES)
    return {"crashes": crashes, "speed_kmh": 3.6 * d["veh_m"] / np.maximum(d["veh_s"], 1), "red_runs": d["red_runs"],
            "lane_changes": d["lane_changes"], "peds": d["peds"]}


def variance_ratio(a, b):
    """Var(A) + Var(B) (what independent worlds would give) over Var(A - B) (what the pairing gives), per copy."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    v_pair = np.var(a - b, ddof=1)
    return float((np.var(a, ddof=1) + np.var(b, ddof=1)) / max(v_pair, 1e-12))


def paired_diff(a, b, n_boot=2000, seed=0):
    """Mean of B - A over copies with a bootstrap 95 % interval (resampling copies, so the pairing is kept)."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "se": float(d.std(ddof=1) / np.sqrt(len(d)))}


def compare(net, beh_a, beh_b, n_envs=16, seconds=300, seed=0):
    ma, mb = run_world(net, beh_a, n_envs, seconds, seed), run_world(net, beh_b, n_envs, seconds, seed)
    oa, ob = outcomes(ma), outcomes(mb)
    return {k: {"a": oa[k].tolist(), "b": ob[k].tolist(), "ratio": variance_ratio(oa[k], ob[k]), **paired_diff(oa[k], ob[k])} for k in oa}


def main(n_envs=16, seconds=300):
    os.makedirs(OUT, exist_ok=True)
    net, _, _ = NW.build("oakland")
    res = {}
    for name, extra in (("av_share_25", dict(p_av=0.25)), ("distraction_x2", dict(distract_rate=2 / 90))):
        for crn in (0.0, 1.0):
            t0 = time.time()
            a = X.Behaviour(**STRESS, crn=crn)
            b = X.Behaviour(**(STRESS | extra), crn=crn)
            res[f"{name}/crn={int(crn)}"] = compare(net, a, b, n_envs, seconds, seed=11)
            r = res[f"{name}/crn={int(crn)}"]
            print(name, "crn", int(crn), {k: round(r[k]["ratio"], 2) for k in r}, f"{time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(OUT, "paired.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:3]))
