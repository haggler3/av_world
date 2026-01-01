"""Does the long tail that matters come out of the simulator? A critical audit against real driving logs and public incident reports.

Two checks.

1. Rare-event rates, simulator against real logs. `scene_metrics` measures, on any scene (an Argoverse 2 log or a recorded simulator run,
   both in the xnet.carla_export format), the rates of the events that make up the dangerous tail of ordinary driving: hard braking, tight
   following, short time-to-collision, close passes of pedestrians. The same function runs on real Argoverse 2 scenarios and on simulator
   scenes from the nominal worlds and from the tail scenarios, so the rates are directly comparable.
2. Famous incidents, feature by feature. For each incident, the features that defined it (from public reports) are checked against what the
   simulator can represent (present, approximate, absent), and where it can be run the mechanism is measured (rate, impact speed) against the
   reported numbers.

    python -m avlab.incidents real  /path/to/av2/val   # rates in real logs         -> results/incidents/real.json
    python -m avlab.incidents sim                       # rates in the simulator    -> results/incidents/sim.json
    python -m avlab.incidents audit                     # incident mechanisms       -> results/incidents/audit.json
"""
import glob
import json
import math
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

OUT = "results/incidents"
HARD_BRAKE = -4.0  # m/s^2
REFRACTORY = 3.0  # s between two hard-brake events of one vehicle


def arrays(scene):
    """-> x, y, yaw, v, length, alive arrays [vehicles, frames] (NaN / False outside a vehicle's life)."""
    T, V = scene["frames"], len(scene["vehicles"])
    x, y, yaw, v = (np.full((V, T), np.nan) for _ in range(4))
    ln = np.zeros(V)
    for k, veh in enumerate(scene["vehicles"]):
        n = len(veh["x"])
        a, b = veh["start"], min(veh["start"] + n, T)
        x[k, a:b], y[k, a:b], yaw[k, a:b], v[k, a:b] = (np.asarray(veh[f])[: b - a] for f in ("x", "y", "yaw", "v"))
        ln[k] = veh.get("length", 4.5)
    return x, y, yaw, v, ln, ~np.isnan(x)


def scene_metrics(scene, ped_radius=2.5):
    """Counts and exposure of the tail events in one scene:
    moving_veh_s    vehicle-seconds spent moving (speed > 1 m/s): the exposure
    hard_brakes     speed drops of more than 4 m/s within a second (one per vehicle per 3 s), from above 3 m/s
    tight_s         vehicle-seconds following a vehicle ahead in the same lane at a time headway under 0.8 s (speed > 3 m/s)
    ttc2_s          vehicle-seconds with a time to collision with the vehicle ahead under 2 s (closing speed > 0.5 m/s, gap < 25 m)
    ttc1_s          the same under 1 s
    ped_close_s     vehicle-seconds moving faster than 3 m/s within `ped_radius` metres of a pedestrian
    """
    dt = scene["dt"]
    x, y, yaw, v, ln, alive = arrays(scene)
    V, T = x.shape
    out = dict(moving_veh_s=float(((v > 1.0) & alive).sum() * dt), hard_brakes=0, tight_s=0.0, ttc2_s=0.0, ttc1_s=0.0, ped_close_s=0.0)
    if V == 0 or T < 4:
        return out
    # hard braking: deceleration averaged over 1 s, so frame-to-frame noise in a log does not count
    k = max(int(round(1.0 / dt)), 1)
    acc = (v[:, k:] - v[:, :-k]) / (k * dt)
    cond = (acc < HARD_BRAKE) & (v[:, :-k] > 3.0)
    gap = int(REFRACTORY / dt)
    for i in range(V):
        last = -10 ** 9
        for t in np.nonzero(np.nan_to_num(cond[i]))[0]:
            if t - last > gap:
                out["hard_brakes"] += 1
            last = t
    peds = scene.get("pedestrians", [])
    for t in range(T):
        idx = np.nonzero(alive[:, t])[0]
        if len(idx) >= 2:
            P = np.stack([x[idx, t], y[idx, t]], 1)
            for a, b in cKDTree(P).query_pairs(25.0):
                for f, l in ((a, b), (b, a)):
                    i, j = idx[f], idx[l]
                    if v[i, t] <= 3.0:
                        continue
                    dx, dy = x[j, t] - x[i, t], y[j, t] - y[i, t]
                    c, s = math.cos(yaw[i, t]), math.sin(yaw[i, t])
                    lon, lat = dx * c + dy * s, -dx * s + dy * c
                    if not (0 < lon < 25 and abs(lat) < 1.8 and math.cos(yaw[j, t] - yaw[i, t]) > 0.9):
                        continue
                    g = lon - 0.5 * (ln[i] + ln[j])
                    if g <= 0:
                        continue
                    if g / v[i, t] < 0.8:
                        out["tight_s"] += dt
                    closing = v[i, t] - v[j, t] * math.cos(yaw[j, t] - yaw[i, t])
                    if closing > 0.5 and g < 25:
                        ttc = g / closing
                        out["ttc2_s"] += dt * (ttc < 2.0)
                        out["ttc1_s"] += dt * (ttc < 1.0)
        for p in peds:
            j = t - p.get("start", 0)
            if "x" in p and 0 <= j < len(p["x"]):
                px, py = p["x"][j], p["y"][j]
            elif "from" in p and 0 <= (t * dt - p["t"]) <= p["duration"]:
                f = (t * dt - p["t"]) / p["duration"]
                px, py = p["from"][0] + f * (p["to"][0] - p["from"][0]), p["from"][1] + f * (p["to"][1] - p["from"][1])
            else:
                continue
            d = np.hypot(x[:, t] - px, y[:, t] - py)
            out["ped_close_s"] += float(np.nansum((d < ped_radius) & (v[:, t] > 3.0)) * dt)
    return out


def rates(tot):
    h = max(tot["moving_veh_s"] / 3600, 1e-9)
    return {"moving_veh_hours": h, "hard_brakes_per_h": tot["hard_brakes"] / h, "tight_follow_share": tot["tight_s"] / max(tot["moving_veh_s"], 1e-9),
            "ttc_under_2s_share": tot["ttc2_s"] / max(tot["moving_veh_s"], 1e-9), "ttc_under_1s_share": tot["ttc1_s"] / max(tot["moving_veh_s"], 1e-9),
            "ped_close_s_per_h": tot["ped_close_s"] / h}


def real(root, limit=400):
    """Rates in real Argoverse 2 scenarios (every second frame: 0.2 s, like the simulator)."""
    from xnet import avdata as AV
    os.makedirs(OUT, exist_ok=True)
    tot, n = None, 0
    for d in sorted(glob.glob(os.path.join(root, "*")))[:limit]:
        sc = AV.read_av2(d)
        scene = AV.av2_to_scene(sc, 2)
        # parked / static-looking vehicles are exposure-free (speed never above 1 m/s) and stay out of the pair search as followers anyway
        m = scene_metrics(scene)
        tot = m if tot is None else {k: tot[k] + m[k] for k in m}
        n += 1
    res = {"scenarios": n, "totals": tot, **rates(tot)}
    json.dump(res, open(f"{OUT}/real.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def sim(reps=4, warm=120.0, seconds=90.0):
    """Rates in simulator scenes: nominal worlds and the famous-event scenarios (test intensity u = 1)."""
    from avlab import famous as FM
    from xnet import carla_export as CE
    from xnet import network as NW
    from xnet import sim as X
    os.makedirs(OUT, exist_ok=True)
    net, geo, _ = NW.build("oakland")
    conds = {n: kw for n, kw in FM.NOMINAL.items()}
    conds.update({n: FM.SCENARIOS[n][2](1.0) | {"risk": 1.0} for n in FM.SCENARIOS})
    res = {}
    for name, kw in conds.items():
        beh = FM.behaviour(dict(kw))
        tot = None
        t0 = time.time()
        for r in range(reps):
            s = X.Sim(net, n_envs=1, beh=beh, seed=40 + r, reroute_s=0)
            s.run(warm)
            frames = s.run(seconds, record=(0, 1))
            scene = CE.trajectories(net, geo, frames)
            m = scene_metrics(scene)
            tot = m if tot is None else {k: tot[k] + m[k] for k in m}
        res[name] = {"totals": tot, **rates(tot)}
        print(f"{name:26s} {res[name]['moving_veh_hours']:.1f} veh-h  hard {res[name]['hard_brakes_per_h']:.1f}/h  tight {res[name]['tight_follow_share']:.3f}  "
              f"ttc<2 {res[name]['ttc_under_2s_share']:.4f}  ped {res[name]['ped_close_s_per_h']:.1f} s/h  {time.time() - t0:.0f} s", flush=True)
        json.dump(res, open(f"{OUT}/sim.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------------ incident audit
LEVEL = {"present": 1.0, "approx": 0.5, "absent": 0.0}
INCIDENTS = [
    dict(name="Tempe, Arizona, 18 March 2018", what="an automated test vehicle struck a pedestrian walking a bicycle across a multi-lane road at night",
         source="NTSB highway accident report HAR-19/03: impact at about 43 mph (19 m/s); the system first detected her about 5.6 s before impact and "
                "repeatedly changed her classification; automatic emergency braking was disabled", scenario="jaywalker_misclassified",
         features=[("pedestrian crosses outside any crosswalk", "absent", "pedestrians only cross at the crosswalks of junctions"),
                   ("wide arterial at ~45 mph", "approx", "at most two lanes per direction; the speed limits come from the map"),
                   ("darkness", "absent", "no lighting model"),
                   ("object seen early but classified inconsistently, so no braking", "approx", "a blackout hides the pedestrian; there is no classification or prediction error"),
                   ("impact at about 43 mph", "approx", "measured below"),
                   ("emergency braking off, backup driver inattentive", "absent", "not modelled"),
                   ("an unusual object (a person pushing a loaded bicycle)", "absent", "pedestrians are one class")]),
    dict(name="Driver-assist crashes into stopped emergency vehicles (NHTSA investigation opened August 2021)", what="a car on a driver-assist system hit a stopped "
         "emergency vehicle in or beside the travel lane", source="NHTSA preliminary evaluation PE21-020, opened with 11 crashes", scenario="stalled_vehicle_missed",
         features=[("stationary vehicle in or beside the lane", "present", "breakdowns and delivery stops"),
                   ("flashing lights, cones, a scene around the vehicle", "absent", "a stopped car is just a stopped car"),
                   ("highway speeds", "absent", "an urban network: free-flow speeds are the limits (measured below)"),
                   ("stationary object filtered out or missed by the sensors", "approx", "blackouts hide it for 0.6-2 s"),
                   ("the car ahead changes lane and reveals the stopped one", "approx", "lane changes exist; not set up as a scenario")]),
    dict(name="San Francisco, 2 October 2023", what="a pedestrian struck by a human-driven car was thrown into the path of an automated vehicle, which braked "
         "but hit her and then pulled over", source="public reports of the California DMV suspension and the company's own review",
         scenario=None,
         features=[("a pedestrian hit by another vehicle lands in the AV's lane", "absent", "no pedestrian dynamics after an impact"),
                   ("the AV brakes but cannot stop in time", "approx", "pile-ups into wrecks exist (tail_pileup), not with pedestrians"),
                   ("the AV moves on after the impact", "absent", "wrecks stay put"),
                   ("multi-lane urban street at night", "approx", "urban, two lanes at most, no night")]),
    dict(name="Phantom braking (NHTSA preliminary evaluation PE22-002, February 2022)", what="cars on a driver-assist system braked hard for no visible reason, "
         "sometimes getting rear-ended", source="NHTSA evaluation opened after hundreds of owner complaints", scenario="phantom_braking",
         features=[("hard braking without an obstacle", "present", "p_ghost: a phantom obstacle 12 m ahead"),
                   ("a following vehicle collides", "present", "followers are real drivers with reaction times; measured below"),
                   ("highway speeds", "absent", "urban speeds"),
                   ("braking from a false detection of an overpass, shadow or sign", "approx", "ghosts appear at random, not at overpasses")]),
    dict(name="Williston, Florida, May 2016", what="a car on a driver-assist system drove under a tractor-trailer crossing its path on a divided highway; "
         "the report cites a trailer the camera did not distinguish from a bright sky", source="NTSB highway accident report HAR-17/02", scenario="unprotected_left",
         features=[("a large vehicle turns across the path of the car", "approx", "cars, vans and buses turn across traffic; no tractor-trailers"),
                   ("camera fails to separate the trailer from the sky", "approx", "blackouts hide any object"),
                   ("under-ride of a high trailer", "absent", "no vehicle heights"),
                   ("divided highway at high speed", "absent", "urban network")]),
]


def audit(n=32, seconds=300.0):
    """Feature checklist for every incident, and the mechanism measured in its scenario: rate per 1,000 simulated hours, and impact or closing
    speeds against the reported numbers."""
    from avlab import adversary as AD
    from avlab import famous as FM
    from xnet import sim as X
    os.makedirs(OUT, exist_ok=True)
    arena = AD.Arena(policy=None, shield="idm", n=n, seed=17)
    net_vmax = float(arena.net_np["link_v"].max())
    res = {"max_link_speed_mps": net_vmax, "max_lanes_per_direction": int(arena.net_np["link_lanes"].max()), "incidents": []}
    hours = n * seconds / 3600
    extra = {"phantom_braking": dict(p_ghost=3e-3)}
    for inc in INCIDENTS:
        lv = [LEVEL[l] for _, l, _ in inc["features"]]
        row = {"name": inc["name"], "features": [{"feature": f, "level": l, "note": nt} for f, l, nt in inc["features"]], "fidelity": float(np.mean(lv)),
               "present": sum(l == "present" for _, l, _ in inc["features"]), "approx": sum(l == "approx" for _, l, _ in inc["features"]),
               "absent": sum(l == "absent" for _, l, _ in inc["features"])}
        if inc["scenario"]:
            kw = dict(FM.SCENARIOS[inc["scenario"]][2](1.0), risk=1.0)
            beh = FM.behaviour(kw)
            st = arena.warm(beh, 100.0)
            _, d = arena.measure(st, beh, seconds)
            m = {k: d[k] for k in X.TAILS + ("imp_ped_n", "imp_ped_speed", "imp_rear_n", "imp_rear_closing", "imp_stalled_closing", "pedestrian", "rear-end", "left-turn", "angle",
                                              "av_crash", "av_pedestrian", "av_rear-end", "av_left-turn", "av_angle", "av_hard")}
            row["scenario"] = inc["scenario"]
            row["counts"] = m
            row["area_hours"] = hours
            if m["imp_ped_n"]:
                row["pedestrian_impact_speed_mph_mean"] = m["imp_ped_speed"] / m["imp_ped_n"] * 2.237
            if m["tail_stalled"]:
                row["stalled_closing_speed_mph_mean"] = m["imp_stalled_closing"] / m["tail_stalled"] * 2.237
            row["rear_end_after_phantom_braking"] = m["tail_phantom"]
            row["phantom_events"] = m["ghosts"]
        res["incidents"].append(row)
        print(row["name"], "fidelity", round(row["fidelity"], 2), {k: row.get(k) for k in ("pedestrian_impact_speed_mph_mean", "stalled_closing_speed_mph_mean", "rear_end_after_phantom_braking", "phantom_events")}, flush=True)
    json.dump(res, open(f"{OUT}/audit.json", "w"), indent=1)


if __name__ == "__main__":
    {"audit": audit, "real": lambda: real(sys.argv[2]), "sim": sim}[sys.argv[1]]()
