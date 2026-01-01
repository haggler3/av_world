"""Finding the intersections in a map, fast and with few false ones.

A node where three or more road pieces meet is the obvious candidate, but in OpenStreetMap that rule alone is wrong in four common
ways: a divided road crossing a street is two T-junctions a few metres apart; a roundabout is a ring of junction nodes; an overpass
that shares a node with the road below (or two ways at different `layer`s) is no junction; a node with a traffic-signal or crossing
tag in the middle of a block is a pedestrian signal, not a junction. `detect` handles all four, in O(n log n):

    nodes, ways = xnet.osm.parse("map.osm")             # lon/lat and tags
    res = junctions.detect(nodes, ways)                 # res["junctions"], res["hints"], res["nodes"], res["ways"], res["stats"]

res["nodes"], res["ways"] are the map with roundabouts contracted to one node and grade-separated crossings split, ready for the road
network builder; res["hints"] carry the traffic-signal and stop-sign tags that `xnet.network.build(hints=...)` uses instead of guessing.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

from pgh import mine

JOIN = 30.0  # metres of road between two junction nodes that make them one junction (divided roads, offset T's)
NEAR = 12.0  # metres apart in a straight line: also one junction (a median between two carriageways)
LEG_M = 18.0  # a leg's direction is taken this far from the junction
MERGE_DEG = 30.0  # legs of one junction closer than this in bearing, on the same street, are one road (two carriageways)
SIGNAL_R = 30.0  # a signal or stop tag this close to a junction belongs to it


def level(tags):
    """Vertical level of a way: layer, +1 on a bridge, -1 in a tunnel (ways at different levels never meet at a shared node)."""
    try:
        lay = int(tags.get("layer", 0))
    except ValueError:
        lay = 0
    return lay + (1 if tags.get("bridge", "no") not in ("no", "") else 0) - (1 if tags.get("tunnel", "no") not in ("no", "") else 0)


def _project(nodes):
    ids = list(nodes)
    ll = np.array([nodes[i] for i in ids])
    lat0, lon0 = ll[:, 1].mean(), ll[:, 0].mean()
    x = (ll[:, 0] - lon0) * 111320.0 * math.cos(math.radians(lat0))
    y = (ll[:, 1] - lat0) * 110540.0
    return {i: (float(a), float(b)) for i, a, b in zip(ids, x, y)}, (lat0, lon0)


def _bearing(dx, dy):
    return math.degrees(math.atan2(dy, dx)) % 360


def _bdiff(a, b):
    return abs((a - b + 180) % 360 - 180)


def detect(nodes, ways, join=JOIN, near=NEAR, node_tags=None):
    """nodes {id: (lon, lat)}, ways [(id, [node ids], tags)] -> dict(junctions, hints, nodes, ways, stats). Junction: dict(x, y [local metres],
    lon, lat, kind 'T' | 'cross' | 'multi' | 'roundabout', legs [bearings], roads [names], members [node ids], signal (tagged), stop (tagged))."""
    xy, origin = _project(nodes)
    drive = [(i, list(ns), t) for i, ns, t in ways if t.get("highway") and all(n in xy for n in ns) and len(ns) >= 2]
    stats = {"ways": len(drive), "roundabouts": 0, "grade_separated_nodes": 0}

    # ---- 1. roundabouts: contract each ring to one node at its centre
    remap, ring_of = {}, {}
    for i, ns, t in drive:
        if t.get("junction") == "roundabout" or (ns[0] == ns[-1] and t.get("oneway") in ("yes", "true", "1") and len(ns) >= 4):
            c = np.mean([xy[n] for n in ns], axis=0)
            rid = f"rb{i}"
            xy[rid] = (float(c[0]), float(c[1]))
            ring_of[rid] = ns
            for n in ns:
                remap[n] = rid
            stats["roundabouts"] += 1
    new_ways = []
    for i, ns, t in drive:
        if any(n in remap for n in ns) and (t.get("junction") == "roundabout" or (ns[0] == ns[-1] and t.get("oneway") in ("yes", "true", "1") and len(ns) >= 4)):
            continue  # the ring itself disappears
        m = [remap.get(n, n) for n in ns]
        m = [n for k, n in enumerate(m) if k == 0 or n != m[k - 1]]  # a way that ran along the ring collapses
        if len(m) >= 2:
            new_ways.append((i, m, t))
    drive = new_ways

    # ---- 2. grade separation: a node shared by ways at different levels is one node per level
    lv = defaultdict(set)
    for i, ns, t in drive:
        for n in ns:
            lv[n].add(level(t))
    out_ways = []
    for i, ns, t in drive:
        L = level(t)
        m = []
        for n in ns:
            if len(lv[n]) > 1:
                nn = f"{n}@{L}"
                xy[nn] = xy[n]
                stats["grade_separated_nodes"] += 1
                m.append(nn)
            else:
                m.append(n)
        out_ways.append((i, m, t))
    drive = out_ways

    # ---- 3. degree of every node (number of road pieces meeting there) and the road graph
    adj = defaultdict(list)  # node -> [(neighbour, way index)]
    for k, (i, ns, t) in enumerate(drive):
        for a, b in zip(ns[:-1], ns[1:]):
            adj[a].append((b, k))
            adj[b].append((a, k))
    deg = {n: len(v) for n, v in adj.items()}
    cand = [n for n, d in deg.items() if d >= 3]
    stats["degree3_nodes"] = len(cand)
    if not cand:
        return {"junctions": [], "hints": {"signals": np.zeros((0, 2)), "stops": np.zeros((0, 2))}, "nodes": nodes, "ways": ways,
                "stats": stats, "origin": origin}

    # ---- 4. cluster candidates: joined by a short road (through degree-2 nodes) or very close in a straight line
    idx = {n: k for k, n in enumerate(cand)}
    parent = list(range(len(cand)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    P = np.array([xy[n] for n in cand])
    for a, b in cKDTree(P).query_pairs(near):
        parent[find(a)] = find(b)
    for n in cand:  # walk out along every road piece until the next candidate or `join` metres
        for nb, k in adj[n]:
            prev, cur, dist = n, nb, math.dist(xy[n], xy[nb])
            while dist < join and cur not in idx and deg.get(cur, 0) == 2:
                nxt = [q for q, kk in adj[cur] if q != prev]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                dist += math.dist(xy[prev], xy[cur])
            if cur in idx and dist < join:
                parent[find(idx[n])] = find(idx[cur])
    groups = defaultdict(list)
    for n in cand:
        groups[find(idx[n])].append(n)

    # ---- 5. legs of every cluster: roads leaving it, with the bearing LEG_M metres out; two carriageways of one street are one road
    juncs = []
    for members in groups.values():
        mset = set(members)
        c = np.mean([xy[n] for n in members], axis=0)
        legs = []  # (bearing, name, oneway?)
        for n in members:
            for nb, k in adj[n]:
                if nb in mset:
                    continue
                prev, cur, d = n, nb, math.dist(xy[n], xy[nb])
                while d < LEG_M and deg.get(cur, 0) == 2:
                    nxt = [q for q, kk in adj[cur] if q != prev]
                    if not nxt:
                        break
                    prev, cur = cur, nxt[0]
                    d += math.dist(xy[prev], xy[cur])
                pt = np.array(xy[cur])
                legs.append((_bearing(*(pt - c)), drive[k][2].get("name", ""), drive[k][2].get("oneway", "no") in ("yes", "true", "1", "-1")))
        roads = []  # merged legs
        for b, name, ow in sorted(legs):
            for r in roads:
                if _bdiff(r["b"], b) < MERGE_DEG and (r["name"] == name or not name or not r["name"] or (ow and r["ow"])):
                    r["n"] += 1
                    break
            else:
                roads.append({"b": b, "name": name, "ow": ow, "n": 1})
        if len(members) == 1 and len(roads) < 3:
            continue
        if len(roads) < 3 and not any(m.startswith("rb") for m in members):
            continue
        rb = any(isinstance(m, str) and m.startswith("rb") for m in members)
        kind = "roundabout" if rb else {3: "T", 4: "cross"}.get(len(roads), "multi")
        juncs.append({"x": float(c[0]), "y": float(c[1]), "kind": kind, "legs": [round(r["b"]) for r in roads],
                      "roads": sorted({r["name"] for r in roads if r["name"]}), "members": members, "signal": False, "stop": False})

    # ---- 6. tags: signals and stop signs belong to the nearest junction within SIGNAL_R; the rest are mid-block (pedestrian) signals
    node_tags = node_tags or {}
    sig_pts, stop_pts = [], []
    hints = {"signals": np.zeros((0, 2)), "stops": np.zeros((0, 2))}
    if juncs:
        tree = cKDTree([[j["x"], j["y"]] for j in juncs])
        for n, t in node_tags.items():
            if n not in xy:
                continue
            h = t.get("highway")
            if h in ("traffic_signals", "stop"):
                d, j = tree.query(xy[n])
                if d <= SIGNAL_R:
                    juncs[j]["signal" if h == "traffic_signals" else "stop"] = True
                (sig_pts if h == "traffic_signals" else stop_pts).append(xy[n])
        hints = {"signals": np.array(sig_pts).reshape(-1, 2), "stops": np.array(stop_pts).reshape(-1, 2)}
    stats.update(junctions=len(juncs), clusters_merged=sum(1 for j in juncs if len(j["members"]) > 1),
                 by_kind={k: sum(1 for j in juncs if j["kind"] == k) for k in ("T", "cross", "multi", "roundabout")},
                 signals_at_junctions=sum(j["signal"] for j in juncs), signal_tags=len(sig_pts))
    # node table of the cleaned map (local metres back to lon/lat), for the builder
    lat0, lon0 = origin
    kx = 111320.0 * math.cos(math.radians(lat0))
    used = {n for _, ns, _ in drive for n in ns}
    out_nodes = {n: (lon0 + xy[n][0] / kx, lat0 + xy[n][1] / 110540.0) for n in used}
    for j in juncs:
        j["lon"], j["lat"] = lon0 + j["x"] / kx, lat0 + j["y"] / 110540.0
    for j in juncs:
        j["members"] = [str(m) for m in j["members"]]
    hints["origin"] = origin
    return {"junctions": juncs, "hints": hints, "nodes": out_nodes, "ways": drive, "stats": stats, "origin": origin}


def naive(nodes, ways):
    """The baseline: every node shared by three or more road pieces is a junction."""
    cnt = defaultdict(int)
    for i, ns, t in ways:
        if t.get("highway"):
            for a, b in zip(ns[:-1], ns[1:]):
                cnt[a] += 1
                cnt[b] += 1
    return [nodes[n] for n, d in cnt.items() if d >= 3 and n in nodes]


def from_overture_rows(rows):
    """Overture road segments (the schema xnet.network reads) as OSM-style nodes and ways, so `detect` can run on them: the segment's connectors
    become shared nodes at their position along the line, the other vertices private nodes."""
    from shapely import wkb
    nodes, ways = {}, []
    for r in rows:
        line = wkb.loads(r["geometry"])
        c = np.asarray(line.coords)[:, :2]
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])
        tot = cum[-1]
        if tot <= 0:
            continue
        cons = sorted(r["connectors"] or [], key=lambda q: q["at"])
        ids, pos = [], []
        for q in cons:
            nodes.setdefault(q["connector_id"], tuple(np.array([np.interp(q["at"] * tot, cum, c[:, 0]), np.interp(q["at"] * tot, cum, c[:, 1])])))
            ids.append(q["connector_id"])
            pos.append(q["at"] * tot)
        for k, (x, y) in enumerate(c):
            if any(abs(cum[k] - q) < 2e-7 for q in pos[:len(cons)]):
                continue  # a vertex under a connector is that connector
            nodes[f"{r['id']}:{k}"] = (float(x), float(y))
            ids.append(f"{r['id']}:{k}")
            pos.append(cum[k])
        order = np.argsort(pos, kind="stable")
        seq = [ids[i] for i in order]
        seq = [n for k, n in enumerate(seq) if k == 0 or n != seq[k - 1]]
        ways.append((r["id"], seq, {"highway": r["class"], "name": (r["names"] or {}).get("primary") or ""}))
    return nodes, ways
