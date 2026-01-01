"""The long tail, made on purpose: how often the named rare scenes happen on their own, how much each lever raises them, and
a dataset of the scenes (with a sensor model, so perception can be scored on the hard cases).

The simulator tags crashes by cause (`xnet.sim.TAILS`): a rear-end into a stalled car, into a wreck (pile-up), by a distracted
driver, a pedestrian struck while a bus or van blocked the driver's view, a cut-in rear-end, an AV hit after phantom
braking, an AV that crashed during a sensor blackout. `survey` counts them per 1,000 simulated hours of the city under five
conditions, from the calibrated humans to the tail stress; `scenes` records the moments around an event, exports each as a
CARLA scene (xnet.carla_export) and runs the sensor model (xnet.perception) from the nearest AV.

    python -m avlab.tails survey      # ~8 min -> results/tails/survey.json
    python -m avlab.tails scenes masked_ped 3     # 3 scenes of one event type -> results/tails/scenes/
"""
import json
import os
import sys
import time

import jax
import numpy as np

from . import adversary as AD
from xnet import carla_export as CE
from xnet import network as NW
from xnet import perception as PC
from xnet import sim as X

OUT = "results/tails"
KINDS = X.CRASH_NAMES + X.TAILS + ("av_crash", "red_runs", "rlr_crashes", "breakdowns")


def conditions():
    mult = {}
    p = f"{AD.OUT}/search.json"
    if os.path.exists(p):
        th = json.load(open(p))["log"][-1]["theta_log10"]
        mult = {k: float(10 ** t) for k, t in zip(AD.LEVERS, th)}
    return {
        "calibrated humans (x1)": AD.world(1.0),
        "risky humans (x5)": AD.world(5.0),
        "x5 + AV sensor faults": AD.world(5.0, p_blind=2e-4, p_ghost=1e-3),
        "x5 + aimed adversary": AD.world(5.0, adv_budget=0.0, **mult),
        "tail stress (x15, breakdowns x50, vans/buses x3, pedestrians x2, dart-outs)": AD.world(
            15.0, breakdown_rate=1e-4, p_van=0.15, p_bus=0.09, ped_scale=2.0, p_ped_violate=0.15, p_dart=0.3, p_blind=2e-4, p_ghost=1e-3),
    }


def survey(seconds=600.0, n=32):
    os.makedirs(OUT, exist_ok=True)
    arena = AD.Arena(policy=None, shield="idm", n=n, seed=11)
    res = {}
    for name, beh in conditions().items():
        t0 = time.time()
        st = arena.warm(beh, 200.0)
        _, d = arena.measure(st, beh, seconds)
        hours = n * seconds / 3600
        res[name] = {"area_hours": hours, "av_hours": d["av_veh_s"] / 3600, "per_1000_area_h": {k: 1000 * d[k] / hours for k in KINDS},
                     "counts": {k: d[k] for k in KINDS}}
        print(f"{name}: {hours:.1f} area-hours, {time.time() - t0:.0f} s", flush=True)
        print("   " + "  ".join(f"{k}={d[k]:.0f}" for k in KINDS if d[k] > 0), flush=True)
        json.dump(res, open(f"{OUT}/survey.json", "w"), indent=1)


def scenes(event="masked_ped", count=3, condition="tail stress (x15, breakdowns x50, vans/buses x3, pedestrians x2, dart-outs)", seconds=900.0,
           before=15.0, after=5.0):
    """Record `count` scenes around the named event (a metric: 'tail_<event>', or a crash type) as CARLA scenes + sensor data."""
    metric = event if event in X.METRICS else f"tail_{event}"
    j = X.METRICS.index(metric)
    beh = conditions()[condition]
    net, geo, _ = NW.build("oakland")
    sim = X.Sim(net, n_envs=8, beh=beh, seed=23, reroute_s=0)
    sim.run(200.0)
    ring = [[] for _ in range(sim.st.alive.shape[0])]
    nb, na = int(before / X.DT), int(after / X.DT)
    pending, out, last = {}, [], np.asarray(sim.st.m)[:, j]
    os.makedirs(f"{OUT}/scenes", exist_ok=True)
    for _ in range(int(seconds / X.DT)):
        sim.st = sim.one(sim.st, sim.nh)
        m = np.asarray(sim.st.m)[:, j]
        host = jax.tree.map(np.asarray, sim.st)
        for e in range(len(ring)):
            ring[e].append(jax.tree.map(lambda a, e=e: a[e], host))
            ring[e] = ring[e][-nb:]
            if m[e] > last[e] and e not in pending:
                pending[e] = na
        for e in list(pending):
            pending[e] -= 1
            if pending[e] <= 0:
                del pending[e]
                if len(ring[e]) >= nb // 2:
                    out.append(list(ring[e]))
        last = m
        if len(out) >= count:
            break
    done = []
    for i, frames in enumerate(out[:count]):
        tag = f"{event}_{i}"
        scene = CE.export_frames(net, geo, frames, f"{OUT}/scenes/{tag}")
        avs = [v for v in scene["vehicles"] if v["av"] and len(v["x"]) > len(frames) // 2]
        if avs:
            ego = max(avs, key=lambda v: len(v["x"]))["id"]
            d = PC.detect(scene, ego, PC.Sensor(dropout_rate=0.01, ghost_rate=0.01), seed=i)
            np.savez(f"{OUT}/scenes/{tag}_perception.npz", **d)
            done.append({"scene": tag, "ego": ego, **PC.summary(d)})
        else:
            done.append({"scene": tag, "ego": None})
        print(done[-1], flush=True)
    json.dump(done, open(f"{OUT}/scenes/{event}.json", "w"), indent=1)
    return done


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "survey":
        survey()
    else:
        scenes(a[1], int(a[2]) if len(a) > 2 else 3)
