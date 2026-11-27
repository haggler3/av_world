"""Cross-comparable failure logs: every crash in a city run, cut into an 11 s scenario in one schema for every city.

Every city runs under the same conditions (calibrated human drivers, 25 % rule-based AVs with the safety envelope, the same
warm-up, copies and duration), so crash rates, crash types and impact speeds can be compared between cities with exposure
(vehicle-hours) as the denominator. Each crash becomes a scenario in the canonical schema below; `avlab.failure_export` turns
a canonical scenario into the formats of common AV tools (Waymo Open Motion, Argoverse 2, ScenarioNet / MetaDrive, GPUDrive,
CommonRoad, OpenSCENARIO + OpenDRIVE, CARLA replay, BEV raster tensors, tables).

    run(city)  : E copies; each 10 s chunk returns per-step states of every vehicle slot and pedestrian crossing (5 Hz);
                 the host keeps the last 30 s. A vehicle that becomes a wreck in a step is a crash row (role from the
                 simulator's crash masks: rear-end, junction, lane change, pedestrian). Rows in the same copy and step that
                 lie within 25 m of each other are one event.
    scenario() : for an event at time t_c, the window [t_c - 8 s, t_c + 3 s] at 10 Hz (linear interpolation of the 5 Hz
                 states; 111 steps, the crash at index 80). Tracks: every vehicle within 60 m of the crash at some step, plus
                 pedestrians crossing within 60 m. A slot that a new trip reuses is a new track (told apart by its start time).
                 Ego: the AV in the crash if there is one, else the first vehicle in it. Map: lane centre lines (link lanes and
                 paths through junctions) within 120 m, with successors, speed limits and boundaries; crosswalks;
                 signal states per step at the stop line of each signalised lane.

Canonical schema ("av_world.failure/1"), JSON (gzip):
    id, schema, city {key, name, origin_lat, origin_lon, projection}, condition {...}, dt, n_steps, event_index, t0_sim,
    event {type (xnet.sim.CRASH_NAMES: rear-end, angle, left-turn, sideswipe, merge, pedestrian, bicycle), type_source, t_sim, x, y, lat, lon, impact_speed_mps, av_involved, ego_track, involved_tracks, pedestrian_hit, n_vehicles},
    tracks [{id, type (vehicle|bus|cyclist|pedestrian), is_av, is_ego, involved, length, width, height,
             x[T], y[T], heading[T], vx[T], vy[T], valid[T]}],
    map {lanes [{id, polyline, left, right, is_intersection, speed_limit_mps, successors, predecessors}],
         crosswalks [{id, polygon}], road_edges [{id, polyline}], signals [{lane, stop_point, states[T] (0 red, 1 yellow, 2 green)}]}

    python -m avlab.failurelog --quick                       # one city, 2 copies, 60 s (smoke test)
    python -m avlab.failurelog --cities all --cond dry wet   # every city; -> data/failures/<cond>/<city>/*.json.gz + index
"""
import argparse
import gzip
import json
import math
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

SCHEMA = "av_world.failure/1"
SIM_DT = 0.2
OUT_DT = 0.1
PRE_S, POST_S = 8.0, 3.0
TRACK_R, MAP_R, EVENT_R = 60.0, 120.0, 25.0
RING_S = 30.0
ROLES = ("other", "rear_end", "junction", "lane_change", "pedestrian")
KIND = {0: "vehicle", 1: "bus", 2: "vehicle", 3: "cyclist"}  # CAR, BUS, VAN, BIKE
SIZE = {0: (1.85, 1.5), 1: (2.5, 3.2), 2: (2.1, 2.2), 3: (0.7, 1.7)}  # width, height [m]
PGH_AREAS = ("oakland", "squirrel_hill", "downtown")
CONDITIONS = {"dry": dict(friction=1.0), "wet": dict(friction=0.5)}
BASE = dict(p_av=0.25, warm_s=120.0, seconds=600.0, E=16, av_policy="rule-based AV with the safety envelope (xnet.sim)",
            humans="calibrated (avrl.ladder.knobs)")


def all_cities():
    from xnet import cities
    return list(PGH_AREAS) + list(cities.CITIES)


def build_city(key):
    """-> (net, geo, info, origin (lat, lon), name)."""
    from pgh import mine
    from xnet import cities
    from xnet import network as NW
    if key in cities.CITIES:
        net, geo, info = cities.build(key)
        return net, geo, info, tuple(info["origin_lat_lon"]), cities.CITIES[key]["name"]
    net, geo, info = NW.build(key)
    return net, geo, info, (mine.LAT0, mine.LON0), f"{key.replace('_', ' ').title()}, Pittsburgh"


def to_latlon(x, y, origin):
    lat0, lon0 = origin
    kx, ky = 111320 * math.cos(math.radians(lat0)), 110540.0
    return lat0 + np.asarray(y) / ky, lon0 + np.asarray(x) / kx


# ---------------------------------------------------------------- simulation with a host-side history ring
FRAME_KEYS = ("trk", "s", "v", "alive", "wreck", "kind", "av", "t0", "length", "new", "role", "lead", "ped_t", "t", "trk_pre", "lc")
CRASH_TYPES = ("rear-end", "angle", "left-turn", "sideswipe", "merge", "pedestrian", "bicycle")  # = xnet.sim.CRASH_NAMES



def run(city, cond="dry", E=None, seed=0, warm_s=None, seconds=None, chunk_s=10.0, log=print, on_event=None):
    """Simulate `city`; call on_event(event dict, frames, k) for each crash once its post-crash window is in the ring.
    Returns run totals (exposure and counts per copy)."""
    from examples import _common as C
    from xnet import sim as X
    E = E or BASE["E"]
    warm_s = BASE["warm_s"] if warm_s is None else warm_s
    seconds = BASE["seconds"] if seconds is None else seconds
    net, geo, info, origin, name = build_city(city)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated(p_av=BASE["p_av"], **CONDITIONS[cond])
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for i in range(E)]), J, 1024)

    def one(s):
        out, d = X.step(s, J, beh, nh, diag=True)
        new = d["crashed"] & (s.wreck_t <= 0)
        role = jnp.where(d["ped_hit"], 4, jnp.where(d["ss"], 3, jnp.where(d["in_hit"], 2, jnp.where(d["rear"], 1, 0))))
        fr = dict(trk=out.trk, s=out.s, v=jnp.where(new, s.v, out.v), alive=out.alive | new, wreck=out.wreck_t > 0,
                  kind=out.kind, av=out.av, t0=out.t0, length=out.length, new=new, role=role, ped_t=out.ped_t, t=out.t,
                  lead=jnp.where(new & d["rear"], d["lead"], -1), trk_pre=s.trk,
                  lc=(s.lc_t > 0) | (out.lc_t > 2.9))  # changed lanes within the last 3 s (xnet.sim's lc_recent)
        return out, fr
    n = int(round(chunk_s / X.DT))

    @jax.jit
    def chunk(s):
        return jax.lax.scan(lambda c, _: jax.vmap(one)(c), s, None, length=n)

    @jax.jit
    def warm(s):
        return jax.lax.scan(lambda c, _: (jax.vmap(lambda z: X.step(z, J, beh, nh))(c), None), s, None, length=n)[0]
    for _ in range(int(round(warm_s / chunk_s))):
        st = warm(st)
    m0 = np.asarray(st.m)
    ring = None
    ring_n = int(RING_S / X.DT)
    pending = []  # (copy, frame index in the absolute counter, event)
    k_abs = 0
    post = int(round(POST_S / X.DT))
    tracks = None
    for ci in range(int(round(seconds / chunk_s))):
        st, fr = chunk(st)
        fr = {k: np.asarray(v) for k, v in fr.items()}  # [n, E, ...]
        ring = fr if ring is None else {k: np.concatenate([ring[k], fr[k]])[-ring_n:] for k in fr}
        base = k_abs + n - ring["t"].shape[0]  # absolute index of ring frame 0
        if tracks is None:
            from xnet import render as R
            tracks = R.Tracks(net, geo)
        for j in range(n):  # find new crash rows in this chunk
            for c in np.nonzero(fr["new"][j].any(1))[0]:
                pending.append((int(c), k_abs + j))
        k_abs += n
        keep = []
        for c, k in pending:
            if k + post < k_abs:
                frames = {kk: ring[kk][:, c] for kk in ring}
                for ev in group_events(frames, k - base, tracks, net):
                    ev.update(copy=c, city=city, cond=cond, seed=seed)
                    if on_event:
                        on_event(ev, frames, k - base, (net, geo, info, origin, name, tracks))
            else:
                keep.append((c, k))
        pending = keep
        if ci % 10 == 9:
            log(f"{city}/{cond}: {(ci + 1) * chunk_s:.0f} s simulated")
    M = np.asarray(st.m) - m0
    from xnet import sim as X2
    idx = {k: j for j, k in enumerate(X2.METRICS)}
    tot = {k: M[:, idx[k]].tolist() for k in ("veh_s", "veh_m", "av_veh_s", "av_veh_m", "av_crash", "finished")}
    tot["crashes_by_type"] = {k: float(M[:, idx[k]].sum()) for k in X2.CRASH_NAMES}
    tot.update(city=city, cond=cond, E=E, seed=seed, warm_s=warm_s, seconds=seconds, name=name, origin=list(origin),
               dropped_tail=len(pending))
    return tot


def crash_type(fr, k, sl, roles, kinds, net):
    """The simulator's crash taxonomy (xnet.sim.CRASH_NAMES), rebuilt from the diag masks with xnet.sim's rules:
    pedestrian strikes; any cyclist involved (or struck from behind) -> bicycle; a rear-end where either car changed lanes in
    the last 3 s -> merge; a junction hit between two paths -> the network's conflict type for that pair (angle, left-turn,
    merge), else (one path known) left-turn if it turns left, otherwise angle; a lane-change contact -> sideswipe."""
    L = len(net["link_len"])
    if "pedestrian" in roles:
        return "pedestrian"
    lead = [int(fr["lead"][k][i]) for i in sl if fr["lead"][k][i] >= 0]
    if 3 in kinds or any(int(fr["kind"][k][j]) == 3 for j in lead):
        return "bicycle"
    if "rear_end" in roles:
        lc = any(bool(fr["lc"][k][i]) for i in sl) or any(bool(fr["lc"][k][j]) for j in lead)
        return "merge" if lc else "rear-end"
    if "junction" in roles:
        mv = [int(fr["trk_pre"][k][i]) - 2 * L for i in sl if fr["trk_pre"][k][i] >= 2 * L]
        if len(mv) >= 2:  # the network's own conflict type for this pair of paths (xnet.network: 1 angle, 2 left, 3 merge)
            cm, ct = net["c_mov"], net["c_type"]
            hit = [int(ct[c]) for c in range(len(ct)) if {int(cm[c][0]), int(cm[c][1])} == {mv[0], mv[1]}]
            if hit:
                return {1: "angle", 2: "left-turn", 3: "merge"}.get(hit[0], "angle")
            if int(net["mov_out"][mv[0]]) == int(net["mov_out"][mv[1]]):
                return "merge"
        if any(int(net["mov_turn"][m]) == 0 for m in mv):  # fallback: a left-turning path (TURN_L)
            return "left-turn"
        return "angle"
    if "lane_change" in roles:
        return "sideswipe"
    return "rear-end"


def group_events(fr, k, tracks, net):
    """Crash rows of one copy at ring frame k -> events (rows within EVENT_R chained)."""
    rows = np.nonzero(fr["new"][k])[0]
    x, y, _ = tracks.pose(fr["trk"][k][rows], fr["s"][k][rows], fr["length"][k][rows])
    lab = list(range(len(rows)))
    for a in range(len(rows)):
        for b in range(a + 1, len(rows)):
            if math.hypot(x[a] - x[b], y[a] - y[b]) < EVENT_R:
                la, lb = lab[a], lab[b]
                lab = [la if z == lb else z for z in lab]
    groups = []
    for g in sorted(set(lab)):
        m = [i for i in range(len(rows)) if lab[i] == g]
        sl = rows[m]
        roles = [ROLES[int(fr["role"][k][i])] for i in sl]
        kinds = [int(fr["kind"][k][i]) for i in sl]
        role_typ = "pedestrian" if "pedestrian" in roles else next((r for r in ("rear_end", "junction", "lane_change") if r in roles), "other")
        groups.append((m, sl, roles, kinds, role_typ))
    out = []
    for m, sl, roles, kinds, role_typ in groups:
        typ = crash_type(fr, k, sl, roles, kinds, net)
        struck = [int(fr["lead"][k][i]) for i in sl if fr["lead"][k][i] >= 0 and int(fr["lead"][k][i]) not in sl]
        out.append(dict(frame=k, slots=[int(i) for i in sl], struck=struck, roles=roles, type=typ, type_source="reconstructed", x=float(np.mean(x[m])), y=float(np.mean(y[m])),
                        impact_speed_mps=float(max(fr["v"][k][i] for i in sl)), av_involved=bool(any(fr["av"][k][i] for i in sl))))
    return out


# ---------------------------------------------------------------- one scenario in the canonical schema
def _interp_track(xs, ys, hs, vs, ok, n_out, ratio):
    """5 Hz samples -> 10 Hz (linear; heading unwrapped). A 10 Hz step is valid if both neighbours are."""
    t_in = np.arange(len(xs)) * ratio
    t = np.arange(n_out)
    lo = np.clip(t // ratio, 0, len(xs) - 1).astype(int)
    hi = np.clip(lo + (t % ratio > 0), 0, len(xs) - 1)
    valid = ok[lo] & ok[hi]
    hu = np.unwrap(np.where(ok, hs, 0.0))
    f = lambda a: np.interp(t, t_in, a)  # noqa: E731
    h = f(hu)
    return f(xs), f(ys), (h + np.pi) % (2 * np.pi) - np.pi, f(vs), valid


def scenario(ev, fr, k, ctx):
    """Canonical scenario dict for event `ev` at ring frame k of one copy's frames `fr` (dict of [F, ...] arrays)."""
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info, origin, name, tracks = ctx
    pre, post = int(round(PRE_S / SIM_DT)), int(round(POST_S / SIM_DT))
    lo, hi = k - pre, k + post  # inclusive
    if lo < 0:
        return None
    ratio = int(round(SIM_DT / OUT_DT))
    n_out = (hi - lo) * ratio + 1
    ex, ey = ev["x"], ev["y"]
    # vehicles: identity = (slot, t0)
    trk = []
    slots = np.nonzero(fr["alive"][lo:hi + 1].any(0))[0]
    for i in slots:
        ids = {}
        for j in range(lo, hi + 1):
            if fr["alive"][j][i]:
                ids.setdefault(round(float(fr["t0"][j][i]), 3), []).append(j)
        for t0, js in ids.items():
            js = np.array(js)
            x, y, h = tracks.pose(fr["trk"][js, i], fr["s"][js, i], fr["length"][js, i])
            if np.min(np.hypot(x - ex, y - ey)) > TRACK_R:
                continue
            ok = np.zeros(hi - lo + 1, bool)
            ok[js - lo] = True
            X_, Y_, H_, V_ = (np.zeros(hi - lo + 1) for _ in range(4))
            X_[js - lo], Y_[js - lo], H_[js - lo], V_[js - lo] = x, y, h, fr["v"][js, i]
            xs, ys, hs, vs, valid = _interp_track(X_, Y_, H_, V_, ok, n_out, ratio)
            kd = int(fr["kind"][js[-1], i])
            w, ht = SIZE.get(kd, (1.85, 1.5))
            involved = (int(i) in ev["slots"] or int(i) in ev.get("struck", [])) and (k in js)
            trk.append(dict(slot=int(i), type=KIND.get(kd, "vehicle"), is_av=bool(fr["av"][js[-1], i]), involved=bool(involved),
                            length=round(float(fr["length"][js[-1], i]), 2), width=w, height=ht,
                            x=np.round(xs, 2), y=np.round(ys, 2), heading=np.round(hs, 4),
                            vx=np.round(vs * np.cos(hs), 3), vy=np.round(vs * np.sin(hs), 3), valid=valid))
    # pedestrians: a crossing leg with time left; progress from its start value (read from the ring)
    leg = np.asarray(geo.leg_xy)
    pt = fr["ped_t"]
    hit_ped = None
    for g in np.nonzero((pt[lo:hi + 1] > 0).any(0))[0]:
        a, b = leg[g]
        if min(np.hypot(*(a - [ex, ey])), np.hypot(*(b - [ex, ey]))) > TRACK_R + 20:
            continue
        # split into crossings (a new crossing starts where ped_t jumps up)
        seq = pt[:, g]
        runs, cur = [], None
        for j in range(len(seq)):
            if seq[j] > 0 and (j == 0 or seq[j] > seq[j - 1] + 1e-6 or seq[j - 1] <= 0):
                cur = [j]
                runs.append(cur)
            elif seq[j] > 0 and cur is not None:
                cur.append(j)
            else:
                cur = None
        for r in runs:
            struck = ev["type"] == "pedestrian" and r[-1] in (k - 1, k) and \
                np.hypot(*(a + (b - a) * (1 - seq[r[-1]] / (seq[r[0]] + SIM_DT)) - [ex, ey])) < 15
            rr = r + [k] if struck and r[-1] == k - 1 else r  # the struck pedestrian is removed in the crash step: hold it there
            js = np.array([j for j in rr if lo <= j <= hi])
            if len(js) == 0:
                continue
            dur = seq[r[0]] + SIM_DT
            prog = np.clip(1 - np.where(js == k, seq[r[-1]], seq[js]) / dur, 0, 1) if struck else np.clip(1 - seq[js] / dur, 0, 1)
            P = a[None] + (b - a)[None] * prog[:, None]
            ok = np.zeros(hi - lo + 1, bool)
            ok[js - lo] = True
            X_, Y_ = np.zeros(hi - lo + 1), np.zeros(hi - lo + 1)
            X_[js - lo], Y_[js - lo] = P[:, 0], P[:, 1]
            hd = math.atan2(*(b - a)[::-1])
            spd = float(np.hypot(*(b - a)) / dur)
            xs, ys, hs, vs, valid = _interp_track(X_, Y_, np.full(hi - lo + 1, hd), np.where(ok, spd, 0.0), ok, n_out, ratio)
            trk.append(dict(slot=-1 - int(g), type="pedestrian", is_av=False, involved=bool(struck), length=0.5, width=0.5,
                            height=1.75, x=np.round(xs, 2), y=np.round(ys, 2), heading=np.round(hs, 4),
                            vx=np.round(vs * np.cos(hs), 3), vy=np.round(vs * np.sin(hs), 3), valid=valid))
            hit_ped = hit_ped or struck
    # ids, ego
    for n_, t in enumerate(trk):
        t["id"] = n_
    inv = [t for t in trk if t["involved"] and t["type"] != "pedestrian"]
    ego = next((t for t in inv if t["is_av"]), inv[0] if inv else trk[0])
    for t in trk:
        t["is_ego"] = t is ego
    ego_id = ego["id"]
    trk.sort(key=lambda t: (not t["is_ego"], not t["involved"], t["id"]))
    remap = {t["id"]: n_ for n_, t in enumerate(trk)}
    for t in trk:
        t["id"] = remap[t["id"]]
        for f in ("x", "y", "heading", "vx", "vy"):
            t[f] = t[f].tolist()
        t["valid"] = t["valid"].tolist()
        t.pop("slot")
    t_lo = float(fr["t"][lo]) if "t" in fr else 0.0
    mp = local_map(net, geo, tracks, ex, ey, t_lo, n_out)
    lat, lon = to_latlon(ex, ey, origin)
    sid = f"{ev['city']}-{ev['cond']}-s{ev['seed']}c{ev['copy']}-t{float(fr['t'][k]):.1f}-{min(ev['slots'])}"
    return {"schema": SCHEMA, "id": sid,
            "city": {"key": ev["city"], "name": name, "origin_lat": origin[0], "origin_lon": origin[1],
                     "projection": "local metres: x east, y north, equirectangular around the origin (xnet.osm.set_origin)"},
            "condition": {"name": ev["cond"], **CONDITIONS[ev["cond"]], **{k_: v for k_, v in BASE.items() if k_ in ("p_av", "av_policy", "humans")},
                          "copy": ev["copy"], "seed": ev["seed"]},
            "dt": OUT_DT, "n_steps": n_out, "event_index": pre * ratio, "t0_sim": round(t_lo, 2),
            "event": {"type": ev["type"], "t_sim": round(float(fr["t"][k]), 2), "x": round(ex, 2), "y": round(ey, 2),
                      "lat": round(float(lat), 7), "lon": round(float(lon), 7), "impact_speed_mps": round(ev["impact_speed_mps"], 2),
                      "av_involved": any(t["is_av"] and t["involved"] for t in trk), "ego_track": remap[ego_id],
                      "involved_tracks": [t["id"] for t in trk if t["involved"]], "pedestrian_hit": bool(hit_ped),
                      "n_vehicles": len(ev["slots"]) + len(ev.get("struck", [])), "roles": ev["roles"],
                      "type_source": ev["type_source"]},
            "tracks": trk, "map": mp}


def local_map(net, geo, tracks, ex, ey, t0, n_out):
    from xnet import network as NW
    from xnet import sim as X
    L = len(net["link_len"])
    polys = {}
    for li in range(L):
        for lane in range(int(net["link_lanes"][li])):
            polys[2 * li + lane] = NW._offset(geo.link_pts[li], geo.link_off[li] + NW.W * (lane + 0.5))
    for m, p in enumerate(geo.mov_pts):
        polys[2 * L + m] = np.asarray(p)
    near = {t for t, p in polys.items() if len(p) and np.min(np.hypot(p[:, 0] - ex, p[:, 1] - ey)) < MAP_R}
    succ, pred = {t: [] for t in near}, {t: [] for t in near}
    for m in range(len(net["mov_in"])):
        a, b, c = 2 * int(net["mov_in"][m]) + int(net["mov_lane"][m]), 2 * L + m, 2 * int(net["mov_out"][m]) + int(net["mov_out_lane"][m])
        for u, v in ((a, b), (b, c)):
            if u in near and v in near:
                succ[u].append(v)
                pred[v].append(u)
    lanes = []
    for t in sorted(near):
        p = polys[t]
        li = t // 2 if t < 2 * L else int(net["mov_out"][t - 2 * L])
        hw = NW.W / 2
        lanes.append(dict(id=int(t), polyline=np.round(p, 2).tolist(), left=np.round(NW._offset(p, -hw), 2).tolist(),
                          right=np.round(NW._offset(p, hw), 2).tolist(), is_intersection=t >= 2 * L,
                          speed_limit_mps=round(float(net["link_v"][li]), 2), successors=succ[t], predecessors=pred[t]))
    edges = []
    for li in range(L):
        p = geo.link_pts[li]
        if np.min(np.hypot(p[:, 0] - ex, p[:, 1] - ey)) < MAP_R:
            e = NW._offset(p, geo.link_off[li] + NW.W * int(net["link_lanes"][li]))
            edges.append(dict(id=int(li), polyline=np.round(e, 2).tolist()))
    cws = []
    for g, (a, b) in enumerate(np.asarray(geo.leg_xy)):
        if min(np.hypot(*(a - [ex, ey])), np.hypot(*(b - [ex, ey]))) < MAP_R:
            d = (b - a) / max(np.hypot(*(b - a)), 1e-6)
            nrm = np.array([-d[1], d[0]]) * 1.5
            cws.append(dict(id=int(g), polygon=np.round([a - nrm, b - nrm, b + nrm, a + nrm], 2).tolist()))
    sig = []
    npn = {k: np.asarray(v) for k, v in net.items()}
    states = np.stack([np.asarray(X.signal_state(npn, t0 + j * OUT_DT)) for j in range(n_out)])  # [T, L]
    for t in sorted(near):
        if t < 2 * L and net["link_axis"][t // 2] >= 0:
            sig.append(dict(lane=int(t), stop_point=np.round(polys[t][-1], 2).tolist(), states=states[:, t // 2].astype(int).tolist()))
    return dict(lanes=lanes, road_edges=edges, crosswalks=cws, signals=sig)


def save(sc, path):
    with gzip.open(path, "wt") as f:
        json.dump(sc, f, separators=(",", ":"))


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


def index_row(sc):
    e = sc["event"]
    ego = sc["tracks"][e["ego_track"]]
    return dict(id=sc["id"], city=sc["city"]["key"], condition=sc["condition"]["name"], type=e["type"], t_sim=e["t_sim"],
                x=e["x"], y=e["y"], lat=e["lat"], lon=e["lon"], impact_speed_mps=e["impact_speed_mps"], av_involved=e["av_involved"],
                type_source=e.get("type_source", "roles"), ego_is_av=ego["is_av"], ego_type=ego["type"], n_vehicles=e["n_vehicles"], pedestrian_hit=e["pedestrian_hit"],
                n_tracks=len(sc["tracks"]), n_lanes=len(sc["map"]["lanes"]))


def collect(cities, conds, root="data/failures", E=None, warm_s=None, seconds=None, log=print, cap=None):
    """Run every (city, condition); write canonical scenarios and run totals."""
    totals = []
    for cond in conds:
        for city in cities:
            d = f"{root}/{cond}/{city}"
            os.makedirs(d, exist_ok=True)
            n = [0]

            def on_event(ev, fr, k, ctx, d=d, n=n):
                if cap is not None and n[0] >= cap:
                    return
                sc = scenario(ev, fr, k, ctx)
                if sc is not None:
                    save(sc, f"{d}/{sc['id']}.json.gz")
                    n[0] += 1
            t0 = time.time()
            tot = run(city, cond, E=E, warm_s=warm_s, seconds=seconds, log=log, on_event=on_event)
            tot["scenarios"] = n[0]
            tot["wall_s"] = time.time() - t0
            json.dump(tot, open(f"{d}/_run.json", "w"), indent=1)
            totals.append(tot)
            log(f"{city}/{cond}: {n[0]} scenarios, {sum(tot['veh_s']) / 3600:.0f} veh-h, {tot['wall_s']:.0f} s")
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cities", nargs="+", default=["oakland"])
    ap.add_argument("--cond", nargs="+", default=["dry"])
    ap.add_argument("--root", default="data/failures")
    a = ap.parse_args()
    cities = all_cities() if a.cities == ["all"] else a.cities
    log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)  # noqa: E731
    if a.quick:
        collect(cities[:1], ["dry"], root=a.root + "_quick", E=2, warm_s=20.0, seconds=60.0, log=log)
    else:
        collect(cities, a.cond, root=a.root, log=log)


if __name__ == "__main__":
    main()
