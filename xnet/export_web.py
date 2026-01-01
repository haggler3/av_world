"""Export simulations for the web viewer (docs/viewer/): the map once per area, and compact binary recordings.

Per area  docs/viewer/data/<area>/map.json     road polygons, lane markings, crosswalks, stop signs, signal heads,
                                                street labels (metres, area-centred)
Per run   docs/viewer/data/<run>.json           meta: scenario, frames, events (time, place, type, label)
          docs/viewer/data/<run>.bin            frames: per vehicle x, y (int16, 0.1 m), heading (uint8), speed (uint8,
                                                0.25 m/s), flags (uint8), kind (uint8: car, bus, van, bicycle; high bits: barrier
                                                interventions), id (uint16), barrier value (int8, 0.5 m), keep-clear
                                                zone (uint8, 0.5 m); signal states; pedestrians

    python -m xnet.export_web        # the scenarios in SCENARIOS (~1-2 min each)
"""
import json
import os
import struct

import jax
import numpy as np

from pgh import mine as PM

from . import network as NW
from . import render as R
from . import sim as X

OUT = "docs/viewer/data"
EVERY = 2  # record every 2nd step: 2.5 frames per second
F_AV, F_BUS, F_VAN, F_WRECK, F_STOP, F_BROKEN, F_RED, F_DIST = (1 << i for i in range(8))
K_CUT, K_VETO_J, K_VETO_M = 16, 32, 64  # high bits of the kind byte: the barrier function braked / held / blocked this AV
REC = 12  # bytes per vehicle record
MAX_BARRIER_EVENTS = 40
CRASH_TYPES = X.CRASH_NAMES


def r1(a):
    return [round(float(v), 1) for v in np.asarray(a).ravel()]


def export_map(net, geo, info, area, center):
    mp = R.Map(net, geo, info)
    polys = []
    for g in getattr(mp.road, "geoms", [mp.road]):
        g = g.simplify(0.4)
        polys.append({"outer": r1(np.asarray(g.exterior.coords) - center),
                      "holes": [r1(np.asarray(h.coords) - center) for h in g.interiors]})
    centre_lines, dividers = [], []
    for pts, fwd, bwd, rank, name in geo.chains:
        if fwd and bwd:
            centre_lines.append(r1(pts - center))
    for li, p in enumerate(geo.link_pts):
        if net["link_lanes"][li] == 2:
            dividers.append(r1(NW._offset(p, geo.link_off[li] + NW.W) - center))
    stops, heads = [], []
    for li in range(len(net["link_len"])):
        p = geo.link_pts[li]
        d = NW._dir(p, True)
        r = np.array([d[1], -d[0]])
        edge = p[-1] + r * (geo.link_off[li] + NW.W * net["link_lanes"][li])
        line = [p[-1] + r * (geo.link_off[li] + 0.2), p[-1] + r * (geo.link_off[li] + NW.W * net["link_lanes"][li] - 0.2)]
        if net["link_stop"][li]:
            stops.append({"xy": r1(edge + r * 1.6 - center), "line": r1(np.array(line) - center)})
        if net["link_axis"][li] >= 0:
            heads.append({"link": int(li), "xy": r1(edge + r * 1.4 - d - center), "line": r1(np.array(line) - center)})
    labels, done = [], set()
    for i in sorted(range(len(geo.chains)), key=lambda i: -NW._poly_len(geo.chains[i][0])[-1]):
        pts, fwd, bwd, rank, name = geo.chains[i]
        if not name or name in done or NW._poly_len(pts)[-1] < 60:
            continue
        done.add(name)
        s = NW._poly_len(pts)
        m = int(np.searchsorted(s, s[-1] / 2))
        d = pts[min(m + 1, len(pts) - 1)] - pts[max(m - 1, 0)]
        labels.append({"name": name, "xy": r1(pts[m] - center), "angle": round(float(np.arctan2(d[1], d[0])), 3), "rank": int(rank)})
    x0, y0 = NW.to_m(geo.bbox[0], geo.bbox[1])
    x1, y1 = NW.to_m(geo.bbox[2], geo.bbox[3])
    signals = np.nonzero(net["node_signal"])[0]
    m = {"area": area, "bounds": r1([x0 - center[0], y0 - center[1], x1 - center[0], y1 - center[1]]), "roads": polys,
         "centre_lines": centre_lines, "dividers": dividers, "crosswalks": [r1(c - center) for c in geo.leg_xy],
         "stops": stops, "signals": heads, "labels": labels, "lane_width": NW.W,
         "stats": {"junctions": int(info["n_nodes"]), "links": int(len(net["link_len"])), "signals": int(len(signals)),
                   "stop_approaches": int(net["link_stop"].sum()), "crosswalks": int(len(net["leg_node"])),
                   "km_of_lanes": round(float((net["link_len"] * net["link_lanes"]).sum() / 1000), 1)},
         "origin_lonlat": [round(float(PM.LON0 + center[0] / PM.KX), 6), round(float(PM.LAT0 + center[1] / PM.KY), 6)]}
    os.makedirs(f"{OUT}/{area}", exist_ok=True)
    with open(f"{OUT}/{area}/map.json", "w") as f:
        json.dump(m, f, separators=(",", ":"))
    return heads


def record(net, geo, sim, seconds, center, heads):
    tracks = R.Tracks(net, geo)
    sig_links = np.array([h["link"] for h in heads], int)
    J = {k: np.asarray(v) for k, v in net.items()}
    events = []
    seen = {}  # (type, vehicle) -> last event time, so a bus serving many stops is listed every 90 s at most
    chunks = []
    prev = None
    n = int(seconds / X.DT)
    t_start = float(np.asarray(sim.st.t)[0])
    pbar = None
    n_bar = 0
    for k in range(n):
        sim.st, info = sim.one_info(sim.st, sim.nh)
        st = jax.tree.map(lambda a: np.asarray(a[0]), sim.st)
        bar = {k_: np.asarray(v[0]) for k_, v in info.items()}
        t = float(st.t) - t_start
        if pbar is not None and n_bar < MAX_BARRIER_EVENTS:  # the barrier function stepping in for an AV
            for cond, label in ((bar["bar_veto_j"] & ~pbar["bar_veto_j"], "Barrier holds an AV at a junction"),
                                (bar["bar_veto_m"] & ~pbar["bar_veto_m"], "Barrier blocks an AV's merge"),
                                ((bar["bar_cut"] > 2.0) & (pbar["bar_cut"] <= 2.0) & st.alive, "Barrier brakes an AV")):
                for i in np.nonzero(cond & st.alive)[0][:2]:
                    if t - seen.get(("barrier", int(i)), -1e9) < 60 or t - seen.get("barrier", -1e9) < 6:  # spread over the run
                        continue
                    seen["barrier"] = t
                    seen[("barrier", int(i))] = t
                    n_bar += 1
                    x, y, _ = tracks.pose(st.trk[i:i + 1], st.s[i:i + 1], st.length[i:i + 1])
                    events.append({"t": round(t, 1), "type": "barrier", "label": label,
                                   "x": round(float(x[0] - center[0]), 1), "y": round(float(y[0] - center[1]), 1), "ids": [int(i)]})
        pbar = bar
        if prev is not None:  # events: compare with the previous step
            dm = dict(zip(X.METRICS, st.m - prev.m))
            new_wreck = st.alive & (st.wreck_t > 0) & ~((prev.wreck_t > 0) & prev.alive)
            if new_wreck.any():
                typ = next((c for c in CRASH_TYPES if dm[c] > 0), "crash")
                i = np.nonzero(new_wreck)[0]
                x, y, _ = tracks.pose(st.trk[i], st.s[i], st.length[i])
                events.append({"t": round(t, 1), "type": "crash", "label": f"Crash: {typ}" + (" (AV involved)" if st.av[i].any() else ""),
                               "x": round(float(x.mean() - center[0]), 1), "y": round(float(y.mean() - center[1]), 1),
                               "ids": [int(v) for v in i]})
            for cond, typ, label in (
                    (st.alive & st.ran_red & ~prev.ran_red & (st.trk >= 2 * len(J["link_len"])), "red", "Red-light run"),
                    (st.alive & st.broken & ~prev.broken, "breakdown", "Breakdown blocks a lane"),
                    (st.alive & (st.stop_t > 0) & (prev.stop_t <= 0) & (st.kind == X.BUS) & ~st.broken, "bus", "Bus at a stop"),
                    (st.alive & (st.stop_t > 0) & (prev.stop_t <= 0) & (st.kind == X.VAN) & ~st.broken, "van", "Delivery van double-parks")):
                for i in np.nonzero(cond)[0][:3]:
                    if t - seen.get((typ, int(i)), -1e9) < 90:
                        continue
                    seen[(typ, int(i))] = t
                    x, y, _ = tracks.pose(st.trk[i:i + 1], st.s[i:i + 1], st.length[i:i + 1])
                    events.append({"t": round(t, 1), "type": typ, "label": label + (" (AV)" if st.av[i] else ""),
                                   "x": round(float(x[0] - center[0]), 1), "y": round(float(y[0] - center[1]), 1), "ids": [int(i)]})
        prev = st
        if k % EVERY:
            continue
        a = np.nonzero(st.alive)[0]
        x, y, th = tracks.pose(st.trk[a], st.s[a], st.length[a])
        flags = (st.av[a] * F_AV | (st.kind[a] == X.BUS) * F_BUS | (st.kind[a] == X.VAN) * F_VAN | (st.wreck_t[a] > 0) * F_WRECK
                 | (st.stop_t[a] > 0) * F_STOP | st.broken[a] * F_BROKEN | st.ran_red[a] * F_RED | (st.dist_t[a] > 0) * F_DIST).astype(np.uint8)
        veh = np.zeros(len(a), dtype=[("x", "<i2"), ("y", "<i2"), ("h", "u1"), ("v", "u1"), ("f", "u1"), ("pad", "u1"), ("id", "<u2"),
                                      ("bh", "i1"), ("br", "u1")])
        veh["x"] = np.clip(np.round((x - center[0]) * 10), -32767, 32767)
        veh["y"] = np.clip(np.round((y - center[1]) * 10), -32767, 32767)
        veh["h"] = np.round((th % (2 * np.pi)) / (2 * np.pi) * 256) % 256
        veh["v"] = np.clip(np.round(st.v[a] * 4), 0, 255)
        veh["f"] = flags
        veh["pad"] = st.kind[a] | (bar["bar_cut"][a] > 0.5) * K_CUT | bar["bar_veto_j"][a] * K_VETO_J | bar["bar_veto_m"][a] * K_VETO_M
        veh["bh"] = np.clip(np.round(bar["bar_h"][a] * 2), -127, 127)  # barrier value, 0.5 m
        veh["br"] = np.clip(np.round(bar["bar_reach"][a] * 2), 0, 255)  # keep-clear zone ahead, 0.5 m
        veh["id"] = a
        sig = np.asarray(X.signal_state(J, float(st.t)))[sig_links].astype(np.uint8)
        legs = np.nonzero(st.ped_t > 0)[0]
        ped = np.zeros(len(legs), dtype=[("leg", "<u2"), ("p", "u1"), ("pad", "u1")])
        ped["leg"] = legs
        ped["p"] = np.clip(np.round((1 - st.ped_t[legs] / 11.0) * 255), 0, 255)
        chunks.append(struct.pack("<HHH", len(a), len(sig), len(legs)) + veh.tobytes() + sig.tobytes() + ped.tobytes())
    return b"".join(chunks), events, len(chunks)


SCENARIOS = [
    dict(run="oakland_learned", area="oakland", title="Oakland, morning: 30 % of vehicles are learned AVs",
         blurb="One shared PPO policy, trained with the barrier function as its teacher, drives every blue car among "
               "calibrated human drivers, pedestrians, cyclists, buses and vans. The band ahead of each AV is its barrier.",
         shield="cbf",
         beh=dict(p_av=0.3, demand=1.2, p_bike=0.06), risk=1.0, policy="results/unaided/phase2/policy.npz", seconds=300, seed=4),
    dict(run="oakland_planner", area="oakland", title="Oakland: AVs plan in a latent world model, under a barrier function",
         blurb="Every blue car searches 32 imagined 2.4 s futures each 0.2 s and drives the best, with a control barrier "
               "function on top; humans x5 riskier, cyclists in the lanes.",
         beh=dict(p_av=0.15, demand=1.2, p_bike=0.06), risk=5.0, policy="wmplan", shield="cbf", seconds=240, seed=9),
    dict(run="oakland_risky", area="oakland", title="Oakland, stress test: risky humans, breakdowns, rush-hour demand",
         blurb="Human risk amplified ~100x (red runners, drivers who do not look, distraction) plus frequent breakdowns: "
               "the corner cases an AV must survive, compressed into five minutes. Rule-based AVs under the barrier function.",
         shield="cbf",
         beh=dict(p_av=0.15, demand=1.3, p_red=0.6, red_window=3.0, p_nolook=1e-2, distract_rate=4e-5, breakdown_rate=1e-4), risk=1.0, policy=None, seconds=300, seed=7),
    dict(run="squirrel_hill_learned", area="squirrel_hill", title="Squirrel Hill: half of the vehicles are learned AVs",
         blurb="A neighbourhood the policy never trained in: Forbes, Murray and the residential grid. Barrier function on.",
         shield="cbf",
         beh=dict(p_av=0.5, demand=1.0, p_bike=0.05), risk=1.0, policy="results/xnet3/policy.npz", seconds=240, seed=5),
]


def main(only=None):
    knobs = json.load(open(os.path.join(os.path.dirname(NW.__file__), "..", "results", "calibration.json")))["calibrated"]
    ip = f"{OUT}/index.json"
    index = {r["run"]: r for r in (json.load(open(ip)) if os.path.exists(ip) else [])}
    maps = {}
    for sc in SCENARIOS:
        if only and sc["run"] != only:
            continue
        area = sc["area"]
        net, geo, info = NW.build(area)
        x0, y0 = NW.to_m(geo.bbox[0], geo.bbox[1])
        x1, y1 = NW.to_m(geo.bbox[2], geo.bbox[3])
        center = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        if area not in maps:
            maps[area] = export_map(net, geo, info, area, center)
        pol, masked = None, False
        if sc["policy"] == "wmplan":
            from avrl import xppo as P
            from wmplan import model as WM
            from wmplan import planner as PL
            pol, masked = PL.make(WM.load(), P.load("results/xnet3/policy.npz")), True
        elif sc["policy"]:
            from avrl import xppo as P
            pol = P.deterministic(P.load(sc["policy"]))
        beh = X.Behaviour(**({k: min(v * sc["risk"], 0.95) for k, v in knobs.items()} | sc["beh"]))
        X.SHIELD = sc.get("shield", "idm")  # read when the simulator is traced (its first step)
        sim = X.Sim(net, n_envs=1, beh=beh, policy=pol, seed=sc["seed"], masked=masked)
        sim.run(480 if not masked else 240)
        blob, events, nf = record(net, geo, sim, sc["seconds"], center, maps[area])
        X.SHIELD = "idm"
        with open(f"{OUT}/{sc['run']}.bin", "wb") as f:
            f.write(blob)
        meta = {k: sc[k] for k in ("run", "area", "title", "blurb")} | {"frames": nf, "dt": X.DT * EVERY, "events": events, "rec": REC,
                                                                         "barrier": sc.get("shield", "idm"),
                                                                         "learned": bool(sc["policy"]), "risk": sc["risk"], **sc["beh"]}
        with open(f"{OUT}/{sc['run']}.json", "w") as f:
            json.dump(meta, f, separators=(",", ":"))
        index[sc["run"]] = {k: meta[k] for k in ("run", "area", "title", "blurb", "frames", "dt")} | {"n_events": len(events)}
        print(sc["run"], f"{len(blob) / 1e6:.1f} MB, {nf} frames, {len(events)} events", flush=True)
        order = [s["run"] for s in SCENARIOS]
        with open(ip, "w") as f:
            json.dump(sorted(index.values(), key=lambda r: order.index(r["run"]) if r["run"] in order else 99), f, indent=1)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
