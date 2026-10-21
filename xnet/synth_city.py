"""A made-up OpenStreetMap city with every awkward case and the true list of junctions, to measure junction detection.

A grid of north-south streets i and east-west streets j. Each crossing is one of:
  plain / signal      the two streets share a node (a signal tag on it for `signal`)
  offset              the north-south street is split in two T-junctions 22 m apart on the east-west street
  roundabout          a ring of eight nodes; the streets enter it on the four sides
  overpass            the east-west street is a bridge over the north-south one and shares no node with it
  overpass_shared     the same, but the bridge (layer 1) and the street share a node
Whole east-west streets can be divided arterials: two one-way carriageways 14 m apart, so each crossing is two T-junction nodes.
Also: pedestrian crossing and signal tags in the middle of blocks, and dead-end streets at the border.
"""
import math

import numpy as np

KINDS = ["plain", "plain", "plain", "plain", "signal", "offset", "roundabout", "overpass", "overpass_shared"]


def make(n=8, spacing=130.0, seed=0, lat0=51.5, lon0=-0.12, divided_rows=(2,)):
    """-> nodes {id: (lon, lat)}, ways [(id, [nodes], tags)], node_tags {id: tags}, truth [(x, y)] in metres from the origin corner."""
    rng = np.random.default_rng(seed)
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    nodes, ways, ntags, truth = {}, [], {}, []
    cnt = [0, 0]
    pos = {}

    def node(x, y, tags=None):
        cnt[0] += 1
        nodes[cnt[0]] = (lon0 + x / kx, lat0 + y / ky)
        pos[cnt[0]] = (x, y)
        if tags:
            ntags[cnt[0]] = tags
        return cnt[0]

    def way(ns, **tags):
        cnt[1] += 1
        ways.append((cnt[1], list(ns), {"highway": "residential", **tags}))

    ns_nodes = {i: [] for i in range(n)}  # (y, node, layer) along north-south street i
    ew_nodes = {j: [] for j in range(n)}  # eastbound / only carriageway
    ew_back = {j: [] for j in range(n)}  # westbound carriageway of a divided street
    legs = {}
    for i in range(n):
        for j in range(n):
            x, y = i * spacing + rng.normal(0, 2.5), j * spacing + rng.normal(0, 2.5)
            border = i in (0, n - 1) or j in (0, n - 1)
            k = "plain" if (border or j in divided_rows) else str(rng.choice(KINDS))
            n_ns = 1 + (0 < j < n - 1) if i is not None else 0
            n_legs = (2 if 0 < j < n - 1 else 1) + (2 if 0 < i < n - 1 else 1)
            if j in divided_rows and 0 < i < n - 1:  # divided: two T nodes 14 m apart
                a, b = node(x, y - 7), node(x, y + 7)
                ns_nodes[i] += [(y - 7, a, 0), (y + 7, b, 0)]
                ew_nodes[j].append((x, a))
                ew_back[j].append((x, b))
                truth.append((x, y))
                continue
            if k in ("plain", "signal"):
                a = node(x, y, {"highway": "traffic_signals"} if k == "signal" else None)
                ns_nodes[i].append((y, a, 0))
                ew_nodes[j].append((x, a))
                if n_legs >= 3:
                    truth.append((x, y))
            elif k == "offset":
                a, b = node(x - 11, y), node(x + 11, y)
                ns_nodes[i] += [(y - 1, a, 0), (y + 1, b, 0)]  # the street jogs: south half meets the east-west street at a, north half at b
                ew_nodes[j] += [(x - 11, a), (x + 11, b)]
                truth.append((x, y))
            elif k == "roundabout":
                ring = [node(x + 17 * math.cos(t), y + 17 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 9)[:-1]]
                way(ring + [ring[0]], junction="roundabout", oneway="yes")
                ns_nodes[i] += [(y - 17, ring[6], 0), (y + 17, ring[2], 0)]
                ew_nodes[j] += [(x - 17, ring[4]), (x + 17, ring[0])]
                truth.append((x, y))
            elif k == "overpass":
                s0, s1 = node(x, y - 1.5), node(x, y + 1.5)
                e0, e1 = node(x - 1.5, y), node(x + 1.5, y)
                ns_nodes[i] += [(y - 1.5, s0, 0), (y + 1.5, s1, 0)]
                way([s0, s1], name=f"NS{i}")
                ew_nodes[j] += [(x - 1.5, e0), (x + 1.5, e1)]
                way([e0, e1], name=f"EW{j}", bridge="yes", layer="1")
            elif k == "overpass_shared":
                a = node(x, y)
                e0, e1 = node(x - 12, y), node(x + 12, y)
                ns_nodes[i].append((y, a, 0))
                ew_nodes[j] += [(x - 12, e0), (x + 12, e1)]
                way([e0, a], name=f"EW{j}", bridge="yes", layer="1")  # the bridge passes through the street's node, one layer up
                way([a, e1], name=f"EW{j}", bridge="yes", layer="1")
    done = {(w[1][0], w[1][-1]) for w in ways}
    for i in range(n):  # streets between consecutive nodes, with a mid-block tag now and then
        pts = sorted(ns_nodes[i], key=lambda t: t[0])
        for (y0, a, _), (y1, b, _) in zip(pts[:-1], pts[1:]):
            if (a, b) in done or (b, a) in done:
                continue
            done.add((a, b))
            if y1 - y0 > 60 and rng.random() < 0.35:
                m = node(*np.mean([pos[a], pos[b]], axis=0), {"highway": "crossing"} if rng.random() < 0.7 else {"highway": "traffic_signals"})
                way([a, m, b], name=f"NS{i}")
            else:
                way([a, b], name=f"NS{i}")
    for j in range(n):
        for pts, ow in ((sorted(ew_nodes[j]), "yes" if j in divided_rows else None), (sorted(ew_back[j], reverse=True), "yes")):
            for (x0, a), (x1, b) in zip(pts[:-1], pts[1:]):
                if (a, b) in done or (b, a) in done:
                    continue
                done.add((a, b))
                way([a, b], name=f"EW{j}", **({"oneway": ow} if ow else {}))
    return nodes, ways, ntags, truth


def score(found, truth, radius=25.0):
    """Precision, recall and F1 of detected junction points against the truth (greedy one-to-one matching within `radius` metres)."""
    from scipy.spatial import cKDTree
    f, t = np.asarray(found).reshape(-1, 2), np.asarray(truth).reshape(-1, 2)
    if not len(f) or not len(t):
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "found": len(f), "truth": len(t)}
    d, j = cKDTree(t).query(f, distance_upper_bound=radius)
    ok = np.isfinite(d)
    matched = len(set(j[ok].tolist()))
    p, r = matched / len(f), matched / len(t)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9), "found": len(f), "truth": len(t)}
