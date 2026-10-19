"""Crash and traffic statistics for a whole area: same seeds, three worlds, each a vmapped batch of copies of the area.

  clean        no risky drivers, pedestrians always cross lawfully  -> the model's own residual crash rate (baseline)
  drivers      the calibrated risky drivers (xworld/calibrate), amplified A x like the calibration's verification run
               -> (crashes - baseline) / A = crashes caused by driver behaviour
  pedestrians  clean drivers, 3 % of pedestrians cross against the signal or without a gap -> crashes - baseline

    python -m xnet.experiment          # -> results/xnet_oakland.json  (~10 min on 4 CPU cores)
"""
import json
import sys
import time

import jax
import numpy as np

from xworld import calibrate as C

from . import network as NW
from . import sim as X


def run(net, beh, n_envs, seconds, warm, seed):
    sim = X.Sim(net, n_slots=2048, n_envs=n_envs, beh=beh, seed=seed)
    sim.run(warm)
    sim.st = sim.st._replace(m=jax.numpy.zeros_like(sim.st.m))
    t0 = time.time()
    sim.run(seconds)
    m = np.asarray(sim.st.m).sum(0)
    return m, time.time() - t0


def main(area="oakland", n_envs=8, seconds=3600.0, warm=600.0, demand=1.0, seed=0):
    net, geo, info = NW.build(area)
    knobs = json.load(open("results/calibration.json"))["calibrated"]
    A = C.amplification(knobs)
    off = dict(p_red=0.0, p_nolook=0.0, p_nosignal=0.0, distract_rate=0.0, p_ped_violate=0.0, demand=demand)
    worlds = {"clean": X.Behaviour(**off),
              "drivers": X.Behaviour(**(off | {k: v * A for k, v in knobs.items()})),
              "pedestrians": X.Behaviour(**(off | {"p_ped_violate": 0.03}))}
    res = {"area": area, "n_envs": n_envs, "seconds_each": seconds, "demand": demand, "amplification": A,
           "network": {"junctions": int(info["n_nodes"]), "links": len(net["link_len"]), "moves": len(net["mov_len"]),
                       "conflict_points": len(net["c_type"]), "signals": int(net["node_signal"].sum()),
                       "stop_approaches": int(net["link_stop"].sum()), "crosswalks": len(net["leg_node"]),
                       "road_km": float(net["link_len"].sum() / 1000)}}
    ms = {}
    for name, beh in worlds.items():
        m, wall = run(net, beh, n_envs, seconds, warm, seed)
        ms[name] = m
        s = X.summary(m)
        s["wall_s"] = round(wall, 1)
        res[name] = s
        print(name, json.dumps(s, default=float), flush=True)
    d = dict(zip(X.METRICS, ms["drivers"] - ms["clean"]))
    mev = ms["drivers"][0] / 1e6
    by = {k: d[k] / A / mev for k in X.CRASH_NAMES}
    res["driver_caused_per_M_entries"] = {"total": sum(by.values()), **by,
                                          "crashes_drivers_run": int(sum(ms["drivers"][X.METRICS.index(k)] for k in X.CRASH_NAMES)),
                                          "crashes_clean_run": int(sum(ms["clean"][X.METRICS.index(k)] for k in X.CRASH_NAMES))}
    p = dict(zip(X.METRICS, ms["pedestrians"] - ms["clean"]))
    res["pedestrian_violation_crashes_per_M_entries"] = p["pedestrian"] / (ms["pedestrians"][0] / 1e6)
    res["clean_baseline_per_M_entries"] = res["clean"]["crashes_per_M_junction_entries"]
    print(json.dumps({k: res[k] for k in ("driver_caused_per_M_entries", "pedestrian_violation_crashes_per_M_entries",
                                           "clean_baseline_per_M_entries")}, default=float), flush=True)
    with open(f"results/xnet_{area}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["oakland"]))
