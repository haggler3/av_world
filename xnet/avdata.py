"""Real driving logs into the simulator: Argoverse 2 and Waymo Open Motion scenarios.

    from xnet import avdata as AV
    sc = AV.read_av2("val/<scenario_id>")                # scenario_<id>.parquet + log_map_archive_<id>.json
    scene = AV.av2_to_scene(sc)                          # trajectories in the format xnet.carla_export / xnet.perception use
    net, geo, info = AV.av2_network(sc)                  # the scenario's map as an xnet road network (intersections detected)
    st = AV.seed_state(net, geo, scene, frame=49)        # simulator state holding the logged vehicles at that moment
    # ... then run it with reactive drivers (xnet.sim) and compare with what the log did next: AV.compare_future(...)

Argoverse 2 motion-forecasting scenarios are read directly (pandas + json; no `av2` package needed): 11 s at 10 Hz, tracks of
vehicles, pedestrians, cyclists and buses with position, heading and velocity, plus a lane-level vector map. Waymo Open Motion
scenarios are read from the `Scenario` protobuf (needs `waymo-open-dataset` or any module exposing the same message) or from the
plain dict produced by `waymo_to_dict`; see docs/DATA_IMPORT.md for what was and was not tested against real files.
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree
from shapely import wkb
from shapely.geometry import LineString

from . import network as NW
from . import osm
from . import render as R
from . import sim as X

AV2_KIND = {"vehicle": "car", "bus": "bus", "cyclist": "bike", "motorcyclist": "bike", "riderless_bicycle": "bike", "pedestrian": "ped"}
LENGTH = {"car": 4.5, "bus": 12.0, "bike": 1.8, "van": 7.0}
STEP = 2.0  # m between sampled points when clustering intersection lanes
CLUSTER_GAP = 2.5  # m: intersection lane pieces closer than this belong to one junction


# ------------------------------------------------------------------------------------------------------------ Argoverse 2
def fetch_av2(out, n=20, split="val", seed=0):
    """Download `n` random Argoverse 2 motion-forecasting scenarios (about 0.2 MB each) from the public S3 bucket into `out`/<scenario id>/.
    No account is needed; the data is licensed CC BY-NC-SA 4.0 by Argoverse, so it is not redistributed with this package."""
    import random

    import pyarrow.fs as pafs
    from concurrent.futures import ThreadPoolExecutor
    fs = pafs.S3FileSystem(anonymous=True, region="us-east-1")
    base = f"argoverse/datasets/av2/motion-forecasting/{split}"
    ids = [o.path.split("/")[-1] for o in fs.get_file_info(pafs.FileSelector(base))]
    random.Random(seed).shuffle(ids)

    def get(i):
        d = os.path.join(out, i)
        os.makedirs(d, exist_ok=True)
        for f in (f"scenario_{i}.parquet", f"log_map_archive_{i}.json"):
            p = os.path.join(d, f)
            if not os.path.exists(p):
                with fs.open_input_stream(f"{base}/{i}/{f}") as src, open(p, "wb") as dst:
                    dst.write(src.read())
        return d
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(get, ids[:n]))


def read_av2(path):
    """A scenario directory (or its parquet file) -> dict(tracks=DataFrame, map=dict, id, city, focal)."""
    import pandas as pd
    if os.path.isfile(path):
        path = os.path.dirname(path)
    sid = os.path.basename(path.rstrip("/"))
    df = pd.read_parquet(os.path.join(path, f"scenario_{sid}.parquet"))
    mp = json.load(open(os.path.join(path, f"log_map_archive_{sid}.json")))
    return {"tracks": df, "map": mp, "id": sid, "city": str(df.city.iloc[0]), "focal": str(df.focal_track_id.iloc[0]), "dt": 0.1}


def av2_to_scene(sc, every=1):
    """Trajectories of an AV2 scenario in the scene format of xnet.carla_export (`every` > 1 keeps every k-th frame: 2 gives the
    simulator's 0.2 s). Vehicles, buses and cyclists become vehicles; the ego (track 'AV') is the AV; pedestrians keep their path
    ({"start", "x", "y"}); static objects and background are dropped. Positions are the log's city frame in metres."""
    df = sc["tracks"]
    T = int(df.timestep.max()) + 1
    frames = list(range(0, T, every))
    veh, ped = [], []
    for tid, g in df.groupby("track_id", sort=False):
        kind = AV2_KIND.get(g.object_type.iloc[0])
        if kind is None:
            continue
        g = g.sort_values("timestep")
        g = g[g.timestep % every == 0]
        start = int(g.timestep.iloc[0]) // every
        x, y = g.position_x.to_numpy(), g.position_y.to_numpy()
        if kind == "ped":
            ped.append({"id": str(tid), "start": start, "x": np.round(x, 2).tolist(), "y": np.round(y, 2).tolist()})
            continue
        veh.append({"id": len(veh), "track": str(tid), "kind": "av" if str(tid) == "AV" else kind, "av": str(tid) == "AV", "crashed": False,
                    "start": start, "length": LENGTH[kind], "x": np.round(x, 2).tolist(), "y": np.round(y, 2).tolist(),
                    "yaw": np.round(g.heading.to_numpy(), 3).tolist(), "v": np.round(np.hypot(g.velocity_x, g.velocity_y).to_numpy(), 2).tolist()})
    return {"dt": sc["dt"] * every, "frames": len(frames), "vehicles": veh, "pedestrians": ped, "source": f"argoverse2:{sc['id']}", "city": sc["city"],
            "focal": sc["focal"]}


def _pts(poly):
    return np.array([[p["x"], p["y"]] for p in poly])


def av2_junctions(mp):
    """Junctions of an AV2 map. Lane segments flagged `is_intersection` are the paths through junctions; pieces that touch (closer than
    CLUSTER_GAP) or continue one another are one junction. -> (cluster id per intersection lane {lane id: cluster}, number of clusters)."""
    lanes = {int(k): v for k, v in mp["lane_segments"].items() if v["lane_type"] in ("VEHICLE", "BUS")}
    inter = [i for i, v in lanes.items() if v["is_intersection"]]
    if not inter:
        return {}, 0
    pts, owner = [], []
    for i in inter:
        c = _pts(lanes[i]["centerline"])
        seg = np.hypot(*np.diff(c, axis=0).T)
        s = np.concatenate([[0], np.cumsum(seg)])
        t = np.arange(0, s[-1] + STEP, STEP)
        t = t[t <= s[-1]] if s[-1] > 0 else np.array([0.0])
        pts.append(np.stack([np.interp(t, s, c[:, 0]), np.interp(t, s, c[:, 1])], 1))
        owner += [i] * len(t)
    P, owner = np.vstack(pts), np.array(owner)
    parent = {i: i for i in inter}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in cKDTree(P).query_pairs(CLUSTER_GAP):
        if owner[a] != owner[b]:
            parent[find(owner[a])] = find(owner[b])
    for i in inter:  # a lane that continues another inside a big junction
        for s in lanes[i]["successors"] or []:
            if s in parent:
                parent[find(i)] = find(s)
    roots = {r: k for k, r in enumerate(sorted({find(i) for i in inter}))}
    return {i: roots[find(i)] for i in inter}, len(roots)


def av2_rows(mp):
    """AV2 map -> road segments in the schema xnet.network.build reads (metre coordinates stored as lon/lat at the equator, so the
    caller must build with the origin at (0, 0): see av2_network). Every non-intersection driving lane is a one-way segment;
    approach and exit lanes of one junction share one connector, so the builder makes one node (and its box and moves) there."""
    lanes = {int(k): v for k, v in mp["lane_segments"].items() if v["lane_type"] in ("VEHICLE", "BUS")}
    jc, nj = av2_junctions(mp)
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)
    plain = [i for i, v in lanes.items() if not v["is_intersection"]]
    for i in plain:
        for s in lanes[i]["successors"] or []:
            if s in lanes and not lanes[s]["is_intersection"]:
                union(("e", i), ("s", s))  # one lane ends where the next begins
            elif s in jc:  # approach lane: its end is the junction
                union(("e", i), ("j", jc[s]))
        for p in lanes[i]["predecessors"] or []:
            if p in jc:  # exit lane: its start is the junction
                union(("s", i), ("j", jc[p]))
    for i in jc:  # through a junction: the exits of one lane's successors meet its approaches
        for s in lanes[i]["successors"] or []:
            if s in lanes and not lanes[s]["is_intersection"]:
                union(("s", s), ("j", jc[i]))
        for p in lanes[i]["predecessors"] or []:
            if p in lanes and not lanes[p]["is_intersection"]:
                union(("e", p), ("j", jc[i]))
    rows, xs, ys = [], [], []
    for i in plain:
        c = _pts(lanes[i]["centerline"])
        if len(c) < 2 or np.hypot(*np.diff(c, axis=0).T).sum() < 3.0:
            continue
        ll = c / np.array([111320.0, 110540.0])
        xs += [c[:, 0].min(), c[:, 0].max()]
        ys += [c[:, 1].min(), c[:, 1].max()]
        rows.append({"id": f"lane{i}", "names": {"primary": ""}, "class": "residential", "subclass": None,
                     "connectors": [{"connector_id": str(hash(find(("s", i))) % 10**12), "at": 0.0},
                                    {"connector_id": str(hash(find(("e", i))) % 10**12), "at": 1.0}],
                     "access_restrictions": [{"access_type": "denied", "between": None, "when": {"heading": "backward"}}],
                     "speed_limits": None, "geometry": wkb.dumps(LineString(ll)),
                     "bbox": {"xmin": float(ll[:, 0].min()), "xmax": float(ll[:, 0].max()), "ymin": float(ll[:, 1].min()), "ymax": float(ll[:, 1].max())}})
    box = (min(xs) / 111320.0, min(ys) / 110540.0, max(xs) / 111320.0, max(ys) / 110540.0) if xs else (0, 0, 0, 0)
    return rows, box, {"lanes": len(plain), "intersection_lanes": len(jc), "junction_clusters": nj}


def av2_network(sc):
    """-> (net, geo, info) for the scenario's map; coordinates are the log's city frame (metres)."""
    from pgh import mine
    rows, box, stats = av2_rows(sc["map"])
    old = (mine.LAT0, mine.LON0)
    osm.set_origin(0.0, 0.0)
    try:
        net, geo, info = NW.build(box, rows=rows, hub=True)
    finally:
        osm.set_origin(*old)
    return net, geo, {**info, **stats, "origin_lat_lon": (0.0, 0.0)}


# ------------------------------------------------------------------------------------------------ logs -> simulator state
class Matcher:
    """Nearest-lane lookup on a network: every track (lane of a link, path through a junction box) sampled every metre."""

    def __init__(self, net, geo, step=1.0):
        tr = R.Tracks(net, geo)
        L = len(net["link_len"])
        self.tracks = tr
        pts, trk, arc, hd = [], [], [], []
        for t in range(2 * L + len(net["mov_len"])):
            nominal = float(net["link_len"][t // 2] if t < 2 * L else net["mov_len"][t - 2 * L])
            g = tr.off[t] + np.arange(0.0, nominal + step, step)
            g = np.minimum(g, tr.off[t] + nominal)
            x, y = np.interp(g, tr.s, tr.x), np.interp(g, tr.s, tr.y)
            x2, y2 = np.interp(np.minimum(g + 0.8, tr.off[t] + nominal), tr.s, tr.x), np.interp(np.minimum(g + 0.8, tr.off[t] + nominal), tr.s, tr.y)
            pts.append(np.stack([x, y], 1))
            trk.append(np.full(len(g), t))
            arc.append(g - tr.off[t])
            hd.append(np.arctan2(y2 - y, x2 - x))
        self.pts, self.trk, self.arc, self.hd = np.vstack(pts), np.concatenate(trk), np.concatenate(arc), np.concatenate(hd)
        self.tree = cKDTree(self.pts)

    def match(self, x, y, yaw, max_d=3.0, max_angle=1.0):
        """-> (track, arc length of the point, distance) per query; track -1 where nothing within max_d metres points the same way."""
        out_t, out_s, out_d = [], [], []
        for xi, yi, hi in zip(x, y, yaw):
            idx = self.tree.query_ball_point([xi, yi], max_d)
            best = None
            for j in idx:
                da = abs((self.hd[j] - hi + math.pi) % (2 * math.pi) - math.pi)
                if da > max_angle:
                    continue
                d = float(np.hypot(*(self.pts[j] - (xi, yi))))
                if best is None or d < best[0]:
                    best = (d, j)
            if best is None:
                out_t.append(-1), out_s.append(0.0), out_d.append(np.inf)
            else:
                out_t.append(int(self.trk[best[1]])), out_s.append(float(self.arc[best[1]])), out_d.append(best[0])
        return np.array(out_t), np.array(out_s), np.array(out_d)


def seed_state(net, geo, scene, frame, key=0, n_slots=1024, matcher=None, max_d=3.0, route_from_log=0.0, speed_from_log=False):
    """A simulator State (one copy) holding the scene's logged vehicles at `frame`: placed on the nearest lane of the network with their
    logged speed, the ego as the AV, everyone else a human driver with random parameters. `route_from_log` (seconds): head for the
    destination nearest to where the log shows the vehicle that many seconds later (otherwise a random destination); `speed_from_log`:
    desired speed set from the logged speed. Returns (state, info) where info lists, per
    scene vehicle, its slot (or -1 if no lane of the network was within `max_d` metres pointing the same way)."""
    import jax
    import jax.numpy as jnp
    matcher = matcher or Matcher(net, geo)
    J = X.to_jnp(net)
    st = X.reset(jax.random.PRNGKey(key), J, n_slots)
    rng = np.random.default_rng(key)
    nh, dest_link, dest_cdf = np.asarray(net["nh"]), np.asarray(net["dest_link"]), np.asarray(net["dest_cdf"])
    L = len(net["link_len"])
    dest_xy = np.array([matcher.pts[np.nonzero(matcher.trk == 2 * int(l))[0][-1]] for l in dest_link])  # the far end of each destination link
    live = [(v, frame - v["start"]) for v in scene["vehicles"] if 0 <= frame - v["start"] < len(v["x"])]
    xs = np.array([v["x"][j] for v, j in live])
    ys = np.array([v["y"][j] for v, j in live])
    hs = np.array([v["yaw"][j] for v, j in live])
    trk, arc, dist = matcher.match(xs, ys, hs, max_d=max_d)
    a = {f: np.array(getattr(st, f)) for f in st._fields if f not in ("key", "t", "m", "ped_t", "link_vs", "link_n")}
    slots = {}
    used = set()
    for (v, j), t, s, d in zip(live, trk, arc, dist):
        if t < 0:
            continue
        # one vehicle per place: skip a vehicle that lands on top of one already placed on the same track
        key_ = (int(t), int(s // 3))
        if key_ in used:
            continue
        used.add(key_)
        i = len(slots)
        slots[v["id"]] = i
        kind = {"car": X.CAR, "av": X.CAR, "bus": X.BUS, "van": X.VAN, "bike": X.BIKE}[v["kind"]]
        length = float(v.get("length", 4.5))
        a["alive"][i], a["trk"][i], a["kind"][i], a["length"][i] = True, t, kind, length
        a["s"][i] = min(s + length / 2, float(net["link_len"][t // 2] if t < 2 * L else net["mov_len"][t - 2 * L]) - 0.1)
        a["v"][i] = v["v"][j]
        dst = int(np.searchsorted(dest_cdf, rng.random()))
        if route_from_log:
            jf = min(j + int(round(route_from_log / scene["dt"])), len(v["x"]) - 1)
            dst = int(np.argmin(np.hypot(dest_xy[:, 0] - v["x"][jf], dest_xy[:, 1] - v["y"][jf])))
        a["dest"][i] = min(dst, len(dest_link) - 1)
        a["nxt"][i] = int(nh[t // 2, a["dest"][i]]) if t < 2 * L else 0
        a["av"][i] = bool(v["av"])
        a["v0f"][i] = 1.0 if v["av"] else 0.9 + 0.3 * rng.random()
        if speed_from_log:
            lim = float(net["link_v"][t // 2]) if t < 2 * L else 8.0
            a["v0f"][i] = float(np.clip(1.02 * max(v["v"][j], 1.0) / lim, 0.3, 1.6))
        a["T"][i] = 1.3 if v["av"] else 1.0 + 0.8 * rng.random()
        a["amax"][i] = 1.6 if v["av"] else 1.2 + 0.8 * rng.random()
        a["b"][i] = 2.0 if v["av"] else 1.5 + rng.random()
        a["s0"][i] = 1.5 + rng.random()
        a["react"][i] = 0.3 if v["av"] else 0.7 + 0.8 * rng.random()
        a["gsc"][i] = 1.25
    st = st._replace(**{f: jnp.asarray(x) for f, x in a.items()}, t=jnp.asarray(frame * scene["dt"], jnp.float32))
    return st, {"slots": slots, "matched": len(slots), "logged": len(live), "median_offset_m": float(np.median(dist[np.isfinite(dist)])) if np.isfinite(dist).any() else None}


def compare_future(net, geo, scene, frame, horizon_s=6.0, beh=None, seed=0, matcher=None, **seed_kw):
    """Seed the simulator with the logged vehicles at `frame`, run it with reactive drivers, and compare each vehicle's position with the log
    at 1-s steps. -> dict with average displacement error (ADE, m) of the simulator and of constant-velocity extrapolation, per horizon, over the
    vehicles that were matched and still logged at that time."""
    matcher = matcher or Matcher(net, geo)
    st, info = seed_state(net, geo, scene, frame, seed, matcher=matcher, **seed_kw)
    import jax
    beh = beh or X.Behaviour(demand=0.0, p_av=0.0)
    sim = X.Sim(net, n_envs=1, beh=beh, seed=seed, reroute_s=0, policy=None)
    sim.st = jax.tree.map(lambda a: a[None], st)
    frames = sim.run(horizon_s, record=(0, 1))
    dt_sim = X.DT
    rows = {s: [] for s in range(1, int(horizon_s) + 1)}
    cv = {s: [] for s in rows}
    keep_all = {s: [] for s in rows}  # the same, but only vehicles that stay on their lane for the whole horizon (pure car following)
    cv_keep = {s: [] for s in rows}
    tracks = matcher.tracks
    by_id = {v["id"]: v for v in scene["vehicles"]}
    for vid, i in info["slots"].items():
        v = by_id[vid]
        j0 = frame - v["start"]
        moving = v["v"][j0] > 2.0
        for sec in rows:
            k = int(round(sec / scene["dt"]))
            if j0 + k >= len(v["x"]):
                continue
            fr = frames[min(int(round(sec / dt_sim)) - 1, len(frames) - 1)]
            if not fr.alive[i]:
                continue
            x, y, _ = tracks.pose(np.array([fr.trk[i]]), np.array([fr.s[i]]), np.array([fr.length[i]]))
            gt = np.array([v["x"][j0 + k], v["y"][j0 + k]])
            e_sim = float(np.hypot(x[0] - gt[0], y[0] - gt[1]))
            p0 = np.array([v["x"][j0], v["y"][j0]])
            pcv = p0 + sec * v["v"][j0] * np.array([math.cos(v["yaw"][j0]), math.sin(v["yaw"][j0])])
            e_cv = float(np.hypot(*(pcv - gt)))
            rows[sec].append(e_sim)
            cv[sec].append(e_cv)
            if moving and frames[-1].alive[i] and int(frames[-1].trk[i]) == int(frames[0].trk[i]):
                keep_all[sec].append(e_sim)
                cv_keep[sec].append(e_cv)
    return {"match": info, "sim_error_m": {s: (float(np.mean(r)) if r else None) for s, r in rows.items()},
            "const_velocity_error_m": {s: (float(np.mean(r)) if r else None) for s, r in cv.items()}, "n": {s: len(r) for s, r in rows.items()},
            "moving_same_lane": {"sim_error_m": {s: (float(np.mean(r)) if r else None) for s, r in keep_all.items()},
                          "const_velocity_error_m": {s: (float(np.mean(r)) if r else None) for s, r in cv_keep.items()},
                          "n": {s: len(r) for s, r in keep_all.items()}}}


# ------------------------------------------------------------------------------------------------------------------- Waymo
WAYMO_TYPE = {1: "vehicle", 2: "pedestrian", 3: "cyclist"}
WAYMO_LANE = {1: "VEHICLE", 2: "VEHICLE", 3: "VEHICLE"}  # freeway, surface street, bike lane (treated as a driving lane only for 1, 2)


def waymo_to_dict(scenario):
    """A Waymo Open Motion `Scenario` protobuf (waymo_open_dataset.protos.scenario_pb2.Scenario, or anything with the same fields) as the plain
    dict that waymo_scene / waymo_map read. Tracks: center, size, heading, velocity and validity per 0.1 s step; map features: lane centre
    lines with their entry / exit lanes, crosswalk polygons, stop signs. Traffic-light states are kept per step."""
    tracks = []
    for t in scenario.tracks:
        tracks.append({"id": int(t.id), "type": int(t.object_type),
                       "states": [[s.center_x, s.center_y, s.length, s.width, s.heading, s.velocity_x, s.velocity_y, bool(s.valid)] for s in t.states]})
    lanes, crosswalks, stops = {}, [], []
    for f in scenario.map_features:
        kind = f.WhichOneof("feature_data")
        if kind == "lane":
            lanes[int(f.id)] = {"type": int(f.lane.type), "speed_limit_mph": float(f.lane.speed_limit_mph),
                                "polyline": [[p.x, p.y] for p in f.lane.polyline], "entry": [int(i) for i in f.lane.entry_lanes],
                                "exit": [int(i) for i in f.lane.exit_lanes]}
        elif kind == "crosswalk":
            crosswalks.append([[p.x, p.y] for p in f.crosswalk.polygon])
        elif kind == "stop_sign":
            stops.append([f.stop_sign.position.x, f.stop_sign.position.y])
    signals = [[{"lane": int(ls.lane), "state": int(ls.state), "x": ls.stop_point.x, "y": ls.stop_point.y} for ls in ds.lane_states]
               for ds in scenario.dynamic_map_states]
    return {"id": scenario.scenario_id, "dt": 0.1, "current": int(scenario.current_time_index), "sdc": int(scenario.sdc_track_index),
            "tracks": tracks, "lanes": lanes, "crosswalks": crosswalks, "stop_signs": stops, "signals": signals}


def read_waymo(path):
    """Scenarios of a Waymo Open Motion .tfrecord file (needs tensorflow and waymo-open-dataset; not tested against real files here)."""
    import tensorflow as tf
    from waymo_open_dataset.protos import scenario_pb2
    for raw in tf.data.TFRecordDataset(path):
        sc = scenario_pb2.Scenario()
        sc.ParseFromString(raw.numpy())
        yield waymo_to_dict(sc)


def waymo_scene(d, every=1):
    """Waymo dict -> scene (same format as av2_to_scene). The self-driving car (`sdc`) is the AV."""
    veh, ped = [], []
    for k, t in enumerate(d["tracks"]):
        kind = WAYMO_TYPE.get(t["type"])
        if kind is None:
            continue
        st = [(i, s) for i, s in enumerate(t["states"]) if s[7] and i % every == 0]
        if not st:
            continue
        kind = {"vehicle": "car", "pedestrian": "ped", "cyclist": "bike"}[kind]
        start = st[0][0] // every
        if kind == "ped":
            ped.append({"id": str(t["id"]), "start": start, "x": [round(s[0], 2) for _, s in st], "y": [round(s[1], 2) for _, s in st]})
            continue
        veh.append({"id": len(veh), "track": str(t["id"]), "kind": "av" if k == d["sdc"] else kind, "av": k == d["sdc"], "crashed": False,
                    "start": start, "length": round(float(np.median([s[2] for _, s in st])), 2), "x": [round(s[0], 2) for _, s in st],
                    "y": [round(s[1], 2) for _, s in st], "yaw": [round(s[4], 3) for _, s in st],
                    "v": [round(math.hypot(s[5], s[6]), 2) for _, s in st]})
    return {"dt": d["dt"] * every, "frames": len(d["tracks"][0]["states"]) // every, "vehicles": veh, "pedestrians": ped, "source": f"waymo:{d['id']}"}


def infer_intersection_lanes(lanes):
    """Waymo's map has no 'this lane is inside a junction' flag. A lane is inside a junction when its centre line crosses, merges with or
    splits from another lane that is not its own predecessor / successor chain: it has more than one entry or exit, or it crosses another
    lane's centre line. {lane id: [(x, y) ...]} with entry / exit lists -> set of lane ids."""
    from shapely.geometry import LineString
    geo = {i: LineString(l["polyline"]) for i, l in lanes.items() if len(l["polyline"]) >= 2}
    ids = list(geo)
    boxes = np.array([geo[i].bounds for i in ids])
    inter = set()
    for a in range(len(ids)):
        i = ids[a]
        near = np.nonzero((boxes[:, 0] <= boxes[a, 2]) & (boxes[:, 2] >= boxes[a, 0]) & (boxes[:, 1] <= boxes[a, 3]) & (boxes[:, 3] >= boxes[a, 1]))[0]
        for b in near:
            j = ids[b]
            if j <= i or j in lanes[i]["entry"] or j in lanes[i]["exit"]:
                continue
            x = geo[i].intersection(geo[j])
            if not x.is_empty and x.geom_type in ("Point", "MultiPoint"):  # a proper crossing, not a shared end
                if min(x.distance(p) for p in (geo[i].boundary, geo[j].boundary)) > 1.0:
                    inter.update((i, j))
    return inter


def waymo_map(d):
    """Waymo lanes as an AV2-style map dict (lane_segments with centerline, predecessors, successors, is_intersection), ready for av2_rows."""
    lanes = {i: l for i, l in d["lanes"].items() if l["type"] in (1, 2)}
    inter = infer_intersection_lanes(lanes)
    seg = {}
    for i, l in lanes.items():
        seg[str(i)] = {"id": i, "lane_type": "VEHICLE", "is_intersection": i in inter, "predecessors": [e for e in l["entry"] if e in lanes],
                       "successors": [e for e in l["exit"] if e in lanes], "centerline": [{"x": p[0], "y": p[1], "z": 0.0} for p in l["polyline"]]}
    return {"lane_segments": seg, "drivable_areas": {}, "pedestrian_crossings": {}}


def waymo_network(d):
    """-> (net, geo, info) for a Waymo scenario's map, via the same lane-to-network conversion as Argoverse 2."""
    return av2_network({"map": waymo_map(d)})


def validate(root, limit=100, frame=25, horizon_s=5.0, out="results/avdata/av2_validation.json"):
    """Import `limit` real Argoverse 2 scenarios, report how much of each imported cleanly (map built, junctions found, logged vehicles
    matched to a lane) and how well the simulator's reactive drivers predict what the logged vehicles did next, against constant-velocity
    extrapolation (the standard baseline). Vehicles that were standing at the start are left out of the error, as the simulator has no
    information about the signal or queue that holds them."""
    import glob
    import time
    rows, errs = [], {}
    t0 = time.time()
    for d in sorted(glob.glob(os.path.join(root, "*")))[:limit]:
        try:
            sc = read_av2(d)
            scene = av2_to_scene(sc, 2)
            net, geo, info = av2_network(sc)
            m = Matcher(net, geo)
            r = compare_future(net, geo, scene, frame, horizon_s, matcher=m, speed_from_log=True)
        except Exception as e:  # keep going; count the failures
            rows.append({"scenario": os.path.basename(d), "error": repr(e)[:120]})
            continue
        rows.append({"scenario": os.path.basename(d), "city": sc["city"], "lane_segments": len(sc["map"]["lane_segments"]), "junction_clusters": info["junction_clusters"],
                     "links": int(net["link_len"].shape[0]), "moves": int(net["mov_len"].shape[0]), "logged": r["match"]["logged"], "matched": r["match"]["matched"],
                     "median_offset_m": r["match"]["median_offset_m"]})
        ml = r["moving_same_lane"]
        for k in range(1, int(horizon_s) + 1):
            if ml["sim_error_m"][k] is not None:
                errs.setdefault(k, []).append((ml["sim_error_m"][k], ml["const_velocity_error_m"][k], ml["n"][k]))
    ok = [r for r in rows if "error" not in r]
    summ = {"scenarios": len(rows), "imported": len(ok), "failed": len(rows) - len(ok),
            "median_junction_clusters": float(np.median([r["junction_clusters"] for r in ok])) if ok else None,
            "match_rate": float(sum(r["matched"] for r in ok) / max(sum(r["logged"] for r in ok), 1)),
            "median_match_offset_m": float(np.median([r["median_offset_m"] for r in ok if r["median_offset_m"] is not None])) if ok else None,
            "seconds": time.time() - t0}
    for k, v in errs.items():
        a = np.array(v)
        w = a[:, 2]
        summ[f"error_{k}s"] = {"simulator_m": float((a[:, 0] * w).sum() / w.sum()), "constant_velocity_m": float((a[:, 1] * w).sum() / w.sum()), "vehicles": int(w.sum())}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summ, "scenarios": rows}, open(out, "w"), indent=1)
    print(json.dumps(summ, indent=1))
    return summ


if __name__ == "__main__":
    import sys
    validate(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 100)
