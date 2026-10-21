"""A real road network for a map area, built from the Overture (OpenStreetMap) segments in data/pgh_segments.parquet.

Everything here is NumPy, computed once per area:

  pieces   consecutive connectors along each Overture segment
  nodes    connectors where the number of pieces is not 2 (junctions, dead ends, the map edge)
  chains   pieces joined through plain 2-piece connectors, node to node, with their polyline
  links    directed chains (one-way roads give one link); lanes, speed limit and class per link
  boxes    every junction gets a box of radius H; links are trimmed by H at both ends
  moves    (in-link, in-lane, out-link) paths through a box (cubic Bezier), with turn type, length and speed cap
  conflicts  pairs of moves at one junction whose paths cross or merge into the same out-lane, with the arc position on each
  control  per junction: signal (2 phases by road axis, permissive lefts), two-way stop or all-way stop, inferred like
           pgh.mine (lanes and control are not in the open map data)
  crosswalks  one per junction leg, pedestrians by junction type
  routing  next-hop tables to D destination links, Dijkstra on link travel times (re-run with measured speeds)

`Net` (jnp arrays) is what the simulator uses; `Geo` keeps polylines for drawing.
"""
import math
import os
from collections import defaultdict
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely import wkb
from shapely.geometry import LineString
from shapely.geometry import box as shapely_box

from pgh.mine import DRIVE, oneway, speed_mph, to_m

W = 3.6  # lane width
MPH = 0.44704
AREAS = {  # (west, south, east, north)
    "oakland": (-79.968, 40.436, -79.940, 40.452),
    "squirrel_hill": (-79.935, 40.428, -79.915, 40.442),
    "downtown": (-80.004, 40.436, -79.990, 40.446),
}
TURN_L, TURN_T, TURN_R = 0, 1, 2
C_ANGLE, C_LEFT, C_MERGE = 1, 2, 3  # conflict types (match CRASH_NAMES in xnet.sim)
KC = 16  # conflicts kept per move
JOIN = 30.0  # junctions closer than this are merged
DEST_MAX = 64


class Net(NamedTuple):
    # links
    link_len: object
    link_lanes: object
    link_v: object  # speed limit [m/s]
    link_rank: object
    link_end: object  # node at the end
    link_axis: object  # signal axis of this approach at its end node (0/1), -1 if not signalised
    link_stop: object  # stop sign at the end
    link_major: object  # unsignalised end: the major road (does not stop)
    link_bus: object  # bus stop position along the link, -1 none
    link_edge_in: object  # starts at the map edge / a dead end
    link_exit: object  # ends at the map edge / a dead end (trips may end there)
    link_leg_in: object  # crosswalk leg at the end (crossed before the box)
    link_leg_out: object  # crosswalk leg at the start (crossed after the box)
    succ_mov: object  # [L, 2 lanes, 4 slots] move index, -1 if this lane cannot make it
    succ_link: object  # [L, 4] out-link per slot
    # moves
    mov_in: object
    mov_lane: object
    mov_out: object
    mov_out_lane: object
    mov_len: object
    mov_cap: object  # speed cap through the box [m/s]
    mov_turn: object
    mov_node: object
    mov_c: object  # [M, KC] conflict index, -1 pad
    mov_side: object  # [M, KC] which side of the conflict this move is
    mov_cs: object  # [M, KC] arc position of the conflict point on this move
    c_type: object  # [C]
    c_mov: object  # [C, 2]
    # junctions
    node_signal: object
    node_cycle: object
    node_g0: object  # axis 0 green ends (then yellow, all-red, axis 1 green, yellow, all-red)
    node_offset: object
    leg_node: object
    leg_axis: object  # signal axis of the leg's road (-1 unsignalised)
    leg_rate: object  # pedestrians per second wanting to cross
    # demand and routing
    src_cdf: object  # spawn link sampling
    src_rate: object  # total vehicles per second
    dest_link: object  # [D]
    dest_cdf: object
    nh: object  # [L, D] next slot (0..3), -1 when the link is the destination


class Geo(NamedTuple):
    link_pts: list  # trimmed centre-line polylines of each link's lane block (lane 0 = inner) offset applied per lane
    link_off: np.ndarray  # lateral offset of lane 0 centre from the road centre line (to the right)
    mov_pts: list  # [M] polylines
    node_xy: np.ndarray
    node_H: np.ndarray
    link_name: list
    link_cls: list
    chains: list  # (pts, lanes_fwd, lanes_bwd, rank, name) for drawing roads
    leg_xy: np.ndarray  # [legs, 2, 2] crosswalk ends
    bbox: tuple


def _poly_len(p):
    return np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])


def _cut(p, a, b):
    """Polyline between arc lengths a and b."""
    s = _poly_len(p)
    a, b = np.clip(a, 0, s[-1]), np.clip(b, 0, s[-1])
    if b - a < 1e-3:
        b = min(a + 1e-3, s[-1])
    inner = p[(s > a) & (s < b)]
    pa = np.array([np.interp(a, s, p[:, 0]), np.interp(a, s, p[:, 1])])
    pb = np.array([np.interp(b, s, p[:, 0]), np.interp(b, s, p[:, 1])])
    return np.vstack([pa, inner, pb])


def _offset(p, d):
    """Polyline shifted d metres to the right of its direction."""
    t = np.gradient(p, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return p + d * np.stack([t[:, 1], -t[:, 0]], 1)


def _bezier(p0, h0, p3, h3, n=24):
    c = np.linalg.norm(p3 - p0)
    th = np.arccos(np.clip(h0 @ h3, -1, 1))
    k = c / 3 if th < 1e-3 else c * (4 / 3) * np.tan(th / 4) / (2 * np.sin(th / 2))
    k = min(k, c)
    p1, p2 = p0 + h0 * k, p3 - h3 * k
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def _dir(p, at_end):
    """Unit direction over the last (or first) ~6 m of a polyline."""
    s = _poly_len(p)
    if at_end:
        j = min(int(np.searchsorted(s, s[-1] - 6.0)), len(p) - 2)
        q = p[-1] - p[j]
    else:
        j = max(min(int(np.searchsorted(s, 6.0)), len(p) - 1), 1)
        q = p[j] - p[0]
    return q / max(np.linalg.norm(q), 1e-9)


def load_rows(bbox):
    w, s, e, n = bbox
    path = "data/pgh_segments.parquet"
    if not os.path.exists(path):  # run from elsewhere: the repo's own data folder, else the trimmed copy in the package
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    if not os.path.exists(path):
        from av_world import data_path
        path = data_path("pgh_areas.parquet")
    rows = pq.read_table(path, columns=["id", "names", "class", "subclass", "connectors",
                                                               "access_restrictions", "speed_limits", "geometry", "bbox"]).to_pylist()
    return [r for r in rows if r["class"] in DRIVE and r["subclass"] != "link" and r["bbox"]["xmax"] >= w
            and r["bbox"]["xmin"] <= e and r["bbox"]["ymax"] >= s and r["bbox"]["ymin"] <= n]


def build(area="oakland", seed=0, ped_scale=1.0, demand_scale=1.0, rows=None):
    """`area`: a name in AREAS or a (west, south, east, north) box; `rows`: road segments in the Overture schema (default: the
    Pittsburgh file; xnet.osm.rows_from_osm makes them from any OpenStreetMap extract)."""
    bbox = AREAS[area] if isinstance(area, str) else area
    rng = np.random.default_rng(seed)
    rows = load_rows(bbox) if rows is None else rows
    x0, y0 = to_m(bbox[0], bbox[1])
    x1, y1 = to_m(bbox[2], bbox[3])
    box = (float(x0), float(y0), float(x1), float(y1))
    clip = shapely_box(*box)
    # ---------------------------------------------------------------- pieces
    pieces = []  # (ca, cb, pts, attrs)
    for r in rows:
        line = wkb.loads(r["geometry"])
        xs, ys = to_m(*np.asarray(line.coords).T)
        p = np.stack([xs, ys], 1)
        cum = _poly_len(p)
        if cum[-1] < 1:
            continue
        cons = sorted(r["connectors"] or [], key=lambda c: c["at"])
        fwd, bwd = oneway(r["access_restrictions"])
        mph, _ = speed_mph(r["speed_limits"])
        attrs = dict(cls=r["class"], rank=DRIVE[r["class"]], name=(r["names"] or {}).get("primary") or "", mph=mph)
        for k, (c0, c1) in enumerate(zip(cons[:-1], cons[1:])):
            a, b = c0["at"] * cum[-1], c1["at"] * cum[-1]
            if b - a < 0.3:
                continue
            q = _cut(p, a, b)
            inside = (q[:, 0] >= box[0]) & (q[:, 0] <= box[2]) & (q[:, 1] >= box[1]) & (q[:, 1] <= box[3])
            if inside.all():
                pieces.append((c0["connector_id"], c1["connector_id"], q, fwd, bwd, attrs))
                continue
            clipped = LineString(q).intersection(clip)  # cut at the area's edge: the cut points become map-edge nodes
            for j, part in enumerate(getattr(clipped, "geoms", [clipped])):
                if part.is_empty or part.geom_type != "LineString" or part.length < 0.3:
                    continue
                pq_ = np.asarray(part.coords)
                if np.linalg.norm(pq_[0] - q[-1]) < np.linalg.norm(pq_[0] - q[0]) - 1e-6 and len(pq_) > 1:
                    pq_ = pq_[::-1]  # keep the piece's direction (for one-way flags)
                ca = c0["connector_id"] if np.linalg.norm(pq_[0] - q[0]) < 0.05 else f"cut:{r['id']}:{k}:{j}:a"
                cb = c1["connector_id"] if np.linalg.norm(pq_[-1] - q[-1]) < 0.05 else f"cut:{r['id']}:{k}:{j}:b"
                pieces.append((ca, cb, pq_, fwd, bwd, attrs))
    inc = defaultdict(list)
    for i, pc in enumerate(pieces):
        inc[pc[0]].append(i)
        inc[pc[1]].append(i)
    is_node = {c: len(v) != 2 for c, v in inc.items()}
    # ---------------------------------------------------------------- chains
    used = set()
    chains = []  # dict(a, b, pts, fwd, bwd, attrs)
    for c, plist in inc.items():
        if not is_node[c]:
            continue
        for i0 in plist:
            if i0 in used:
                continue
            cur, i, pts, fwd, bwd, attrs = c, i0, [], True, True, pieces[i0][5]
            rank = attrs["rank"]
            while True:
                used.add(i)
                ca, cb, p, f, b_, at = pieces[i]
                if ca == cur:
                    nxt, seg, ff, bb = cb, p, f, b_
                else:
                    nxt, seg, ff, bb = ca, p[::-1], b_, f
                pts.append(seg if not pts else seg[1:])
                fwd, bwd = fwd and ff, bwd and bb
                if at["rank"] > rank:
                    attrs, rank = at, at["rank"]
                cur = nxt
                if is_node.get(cur, True):
                    break
                others = [j for j in inc[cur] if j != i]
                if not others or others[0] in used:
                    break
                i = others[0]
            if cur == c or not (fwd or bwd):
                continue
            P = np.vstack(pts)
            if _poly_len(P)[-1] < 3.0:
                continue
            chains.append(dict(a=c, b=cur, pts=P, fwd=fwd, bwd=bwd, attrs=attrs))
    # ---------------------------------------------------------------- nodes
    # junctions joined by a chain shorter than JOIN metres are one junction (divided roads, offset crossings): a link that
    # short cannot hold a car, and drivers treat the pair as one intersection
    cdeg = defaultdict(int)
    for ch in chains:
        cdeg[ch["a"]] += 1
        cdeg[ch["b"]] += 1
    parent = {}

    def find(c):
        parent.setdefault(c, c)
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    for ch in chains:
        if _poly_len(ch["pts"])[-1] < JOIN and cdeg[ch["a"]] >= 3 and cdeg[ch["b"]] >= 3:
            parent[find(ch["a"])] = find(ch["b"])
    chains = [ch for ch in chains if find(ch["a"]) != find(ch["b"])]
    node_id = {}
    for ch in chains:
        for c in (ch["a"], ch["b"]):
            node_id.setdefault(find(c), len(node_id))
    node_id = {c: node_id[find(c)] for ch in chains for c in (ch["a"], ch["b"])}
    n_nodes = max(node_id.values()) + 1
    acc, cnt_ = np.zeros((n_nodes, 2)), np.zeros(n_nodes)
    for ch in chains:
        for c, pt in ((ch["a"], ch["pts"][0]), (ch["b"], ch["pts"][-1])):
            acc[node_id[c]] += pt
            cnt_[node_id[c]] += 1
    node_xy = acc / np.maximum(cnt_, 1)[:, None]
    # ---------------------------------------------------------------- directed links
    links = []
    for ci, ch in enumerate(chains):
        r = ch["attrs"]["rank"]
        two = ch["fwd"] and ch["bwd"]
        n = 2 if (r >= 4 or (not two and r >= 3)) else 1
        for d, ok in ((+1, ch["fwd"]), (-1, ch["bwd"])):
            if not ok:
                continue
            pts = ch["pts"] if d > 0 else ch["pts"][::-1]
            links.append(dict(chain=ci, a=node_id[ch["a"] if d > 0 else ch["b"]], b=node_id[ch["b"] if d > 0 else ch["a"]],
                              pts=pts, lanes=n, two=two, rank=r, name=ch["attrs"]["name"], cls=ch["attrs"]["cls"],
                              v=(ch["attrs"]["mph"] or (40.0 if r == 5 else 25.0)) * MPH))
    # largest strongly connected part (dead ends / map edge: a vehicle may leave there and come back = U-turn allowed)
    deg = np.zeros(n_nodes, int)
    for ch in chains:
        deg[node_id[ch["a"]]] += 1
        deg[node_id[ch["b"]]] += 1
    out_of = defaultdict(list)
    for li, l in enumerate(links):
        out_of[l["a"]].append(li)
    rows_, cols_ = [], []
    for li, l in enumerate(links):
        for lj in out_of[l["b"]]:
            if links[lj]["chain"] != l["chain"] or deg[l["b"]] == 1:
                rows_.append(li)
                cols_.append(lj)
    g = csr_matrix((np.ones(len(rows_)), (rows_, cols_)), shape=(len(links), len(links)))
    _, lab = connected_components(g, directed=True, connection="strong")
    keep = lab == np.bincount(lab).argmax()
    remap = -np.ones(len(links), int)
    remap[keep] = np.arange(keep.sum())
    links = [l for l, k in zip(links, keep) if k]
    L = len(links)
    out_of = defaultdict(list)
    in_of = defaultdict(list)
    for li, l in enumerate(links):
        out_of[l["a"]].append(li)
        in_of[l["b"]].append(li)
    deg = np.array([len({links[j]["chain"] for j in out_of[n] + in_of[n]}) for n in range(n_nodes)])
    # ---------------------------------------------------------------- junction boxes
    H = np.zeros(n_nodes)
    for nd in range(n_nodes):
        legs = out_of[nd] + in_of[nd]
        if deg[nd] == 2:  # the road just continues (a split in the map data): a tiny box, no control
            H[nd] = 1.0
        if deg[nd] < 3:
            continue
        half = max(l["lanes"] * W * (1.0 if l["two"] else 0.5) for l in (links[j] for j in legs))
        bear = sorted({round(math.degrees(math.atan2(*(_dir(links[j]["pts"], False) if j in out_of[nd]
                                                         else -_dir(links[j]["pts"], True))[::-1])) % 360) for j in legs})
        gaps = np.diff(bear + [bear[0] + 360]) if len(bear) > 1 else [90]
        ang = math.radians(min(max(min(gaps), 30.0), 90.0))
        shortest = min(_poly_len(links[j]["pts"])[-1] for j in legs)
        H[nd] = float(np.clip(min(half / math.tan(ang / 2) + 1.5, 0.42 * shortest), 2.0, 18.0))
    off = np.zeros(L)
    lpts, llen = [], np.zeros(L)
    for li, l in enumerate(links):
        P = l["pts"]
        tot = _poly_len(P)[-1]
        a, b = H[l["a"]], tot - H[l["b"]]
        if b - a < 2.0:
            mid = 0.5 * (a + b)
            a, b = mid - 1.0, mid + 1.0
        off[li] = 0.0 if l["two"] else -l["lanes"] * W / 2
        lpts.append(_cut(P, a, b))
        llen[li] = _poly_len(lpts[-1])[-1]

    def lane_pt(li, lane, end):
        p = _offset(lpts[li], off[li] + W * (lane + 0.5))
        return (p[-1] if end else p[0]), _dir(lpts[li], end)

    # ---------------------------------------------------------------- moves (paths through the boxes)
    moves = []
    succ_link = -np.ones((L, 4), np.int32)
    succ_mov = -np.ones((L, 2, 4), np.int32)
    for li, l in enumerate(links):
        nd = l["b"]
        if deg[nd] < 2:
            continue
        h_in = _dir(lpts[li], True)
        outs = []
        for lo in out_of[nd]:
            if links[lo]["chain"] == l["chain"]:
                continue  # no U-turns at junctions
            h_out = _dir(lpts[lo], False)
            ang = math.degrees(math.atan2(h_in[0] * h_out[1] - h_in[1] * h_out[0], h_in @ h_out))  # + = left
            if abs(ang) > 150:  # U-turns (also over a divided road)
                continue
            outs.append((ang, lo))
        outs.sort(key=lambda x: -x[0])  # leftmost first
        outs = outs[:4]
        n_in = l["lanes"]
        for slot, (ang, lo) in enumerate(outs):
            turn = TURN_L if ang > 35 else (TURN_R if ang < -35 else TURN_T)
            succ_link[li, slot] = lo
            n_out = links[lo]["lanes"]
            if n_in == 1:
                lanes = [0]
            elif turn == TURN_L:
                lanes = [0]
            elif turn == TURN_R:
                lanes = [1]
            elif n_out >= 2:
                lanes = [0, 1]
            else:  # one lane out: one through lane, the other lane is for the right turn (or ends: merge)
                lanes = [0] if any(o[0] < -35 for o in outs) else [1]
            for lane in lanes:
                ol = 0 if turn == TURN_L else (n_out - 1 if turn == TURN_R else min(lane, n_out - 1))
                p0, d0 = lane_pt(li, lane, True)
                p3 = _offset(lpts[lo], off[lo] + W * (ol + 0.5))[0]
                d3 = _dir(lpts[lo], False)
                pts = _bezier(p0, d0, p3, d3)
                ln = _poly_len(pts)[-1]
                r_ = ln / max(math.radians(abs(ang)), 1e-3)
                cap = 99.0 if turn == TURN_T else float(np.clip(math.sqrt(2.5 * r_), 3.5, 9.0))
                succ_mov[li, lane, slot] = len(moves)
                moves.append(dict(i=li, lane=lane, o=lo, ol=ol, pts=pts, len=max(ln, 0.5), cap=cap, turn=turn, node=nd, ang=ang))
    # lanes that can make no move at all end at the junction: drivers must merge out of them (zipper)
    M = len(moves)
    # ---------------------------------------------------------------- conflicts
    by_node = defaultdict(list)
    for mi, m in enumerate(moves):
        by_node[m["node"]].append(mi)
    confs = []  # (m1, m2, s1, s2, type)
    for nd, ms in by_node.items():
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                A, B = moves[ms[x]], moves[ms[y]]
                if A["i"] == B["i"]:
                    continue
                if A["o"] == B["o"] and A["ol"] == B["ol"]:
                    confs.append((ms[x], ms[y], A["len"], B["len"], C_MERGE))
                    continue
                d = np.linalg.norm(A["pts"][:, None] - B["pts"][None], axis=2)
                i, j = np.unravel_index(np.argmin(d), d.shape)
                if d[i, j] < 1.6:
                    sa, sb = _poly_len(A["pts"])[i], _poly_len(B["pts"])[j]
                    hi, hj = _dir(lpts[A["i"]], True), _dir(lpts[B["i"]], True)
                    opposed = hi @ hj < -0.5
                    kind = C_LEFT if opposed and TURN_L in (A["turn"], B["turn"]) else C_ANGLE
                    confs.append((ms[x], ms[y], sa, sb, kind))
    C = len(confs)
    mov_c = -np.ones((M, KC), np.int32)
    mov_side = np.zeros((M, KC), np.int32)
    mov_cs = np.zeros((M, KC), np.float32)
    cnt = np.zeros(M, int)
    for ci, (a, b, sa, sb, _k) in enumerate(confs):
        for m, side, s_ in ((a, 0, sa), (b, 1, sb)):
            if cnt[m] < KC:
                mov_c[m, cnt[m]], mov_side[m, cnt[m]], mov_cs[m, cnt[m]] = ci, side, s_
                cnt[m] += 1
    kc_used = max(int(cnt.max()), 1)  # trim the per-move conflict list to the widest one actually used
    # ---------------------------------------------------------------- control (inferred)
    cross_pairs = set()  # (in-link, in-link) whose through / right-turn paths cross or merge
    for a_, b_, _sa, _sb, _k in confs:
        if moves[a_]["turn"] != TURN_L and moves[b_]["turn"] != TURN_L:
            cross_pairs.add((moves[a_]["i"], moves[b_]["i"]))
    node_signal = np.zeros(n_nodes, bool)
    link_axis = -np.ones(L, np.int32)
    link_stop = np.zeros(L, bool)
    link_major = np.zeros(L, bool)
    for nd in range(n_nodes):
        ins = in_of[nd]
        if deg[nd] < 3 or not ins:
            continue
        legs = ins + out_of[nd]
        top = max(legs, key=lambda j: (links[j]["rank"], links[j]["name"]))
        tname, trank = links[top]["name"], links[top]["rank"]
        cross = max([links[j]["rank"] for j in legs if links[j]["name"] != tname or not tname], default=-1)
        hd = _dir(lpts[top], top not in out_of[nd])
        ax = {j: int(abs(_dir(lpts[j], j in ins) @ hd) < 0.7) for j in legs}  # 0: along the top road, 1: across
        # phases by colouring: two approaches whose straight or right-turn paths cross never share a phase
        order = sorted(ins, key=lambda j: (ax[j], -links[j]["rank"]))
        colour = {}
        for j in order:
            bad = {colour[k] for k in colour if (j, k) in cross_pairs or (k, j) in cross_pairs}
            colour[j] = 0 if 0 not in bad else 1
        ax.update(colour)
        if trank >= 3 and cross >= 2:
            node_signal[nd] = True
            for j in ins:
                link_axis[j] = ax[j]
        elif cross < trank:
            for j in ins:
                link_major[j] = ax[j] == 0
                link_stop[j] = ax[j] != 0
        else:  # equal roads: all-way stop
            for j in ins:
                link_stop[j] = True
    cycle = np.where(node_signal, 70.0, 60.0).astype(np.float32)
    node_g0 = np.full(n_nodes, 36.0, np.float32)
    node_offset = rng.uniform(0, 70, n_nodes).astype(np.float32)
    # ---------------------------------------------------------------- crosswalk legs (one per chain end at a junction)
    leg_of = {}
    leg_node, leg_axis, leg_rate = [], [], []
    for li, l in enumerate(links):
        for nd, is_end in ((l["b"], True), (l["a"], False)):
            if deg[nd] < 3:
                continue
            key = (nd, l["chain"])
            if key not in leg_of:
                leg_of[key] = len(leg_node)
                leg_node.append(nd)
                leg_axis.append(-1)
                leg_rate.append((300.0 if node_signal[nd] else 40.0) / 3600 * ped_scale)
            if is_end and link_axis[li] >= 0:
                leg_axis[leg_of[key]] = link_axis[li]
    for li, l in enumerate(links):  # one-way legs out of a signal: the axis of their road
        k = (l["a"], l["chain"])
        if k in leg_of and leg_axis[leg_of[k]] < 0 and node_signal[l["a"]]:
            hd = _dir(lpts[li], False)
            ins = in_of[l["a"]]
            if ins:
                j = max(ins, key=lambda q: abs(_dir(lpts[q], True) @ hd))
                leg_axis[leg_of[k]] = link_axis[j]
    leg_xy = np.zeros((len(leg_node), 2, 2))
    for li, l in enumerate(links):  # crosswalk: across the road just outside the box (from this link's end)
        k = leg_of.get((l["b"], l["chain"]))
        if k is None:
            continue
        p = lpts[li]
        d = _dir(p, True)
        r = np.array([d[1], -d[0]])
        c = p[-1] + d * 0.0
        half = (l["lanes"] * W * (1.0 if l["two"] else 0.5)) + 0.5
        base = c  # the polyline is the road centre line
        leg_xy[k] = [base - r * (half if l["two"] else half) - d * 1.5, base + r * half - d * 1.5]
    for li, l in enumerate(links):  # legs that only have an outgoing link
        k = leg_of.get((l["a"], l["chain"]))
        if k is None or np.any(leg_xy[k]):
            continue
        p = lpts[li]
        d = _dir(p, False)
        r = np.array([d[1], -d[0]])
        half = (l["lanes"] * W * (1.0 if l["two"] else 0.5)) + 0.5
        base = p[0]
        leg_xy[k] = [base - r * half + d * 1.5, base + r * half + d * 1.5]
    link_leg_in = np.array([leg_of.get((l["b"], l["chain"]), -1) for l in links], np.int32)
    link_leg_out = np.array([leg_of.get((l["a"], l["chain"]), -1) for l in links], np.int32)
    # ---------------------------------------------------------------- buses, demand, destinations
    link_bus = np.array([0.6 * llen[i] if (l["rank"] >= 3 and llen[i] > 60) else -1.0 for i, l in enumerate(links)], np.float32)
    edge_in = [i for i, l in enumerate(links) if deg[l["a"]] == 1]  # entering from the map edge / a dead end
    edge_out = [i for i, l in enumerate(links) if deg[l["b"]] == 1]
    per_class = {5: 500, 4: 350, 3: 250, 2: 150, 1: 40, 0: 20}  # vehicles/h entering per edge link (inferred)
    src_w = np.array([per_class[l["rank"]] / 3600 if i in set(edge_in) else 0.0 for i, l in enumerate(links)])
    src_w += np.array([llen[i] / 1000 * (60 if l["rank"] <= 1 else 120) / 3600 for i, l in enumerate(links)])  # parking, garages
    src_w *= 0.5 * demand_scale  # a moderate hour: the class-based guess above saturates Oakland's junctions
    src_rate = float(src_w.sum())
    src_cdf = np.cumsum(src_w) / src_w.sum()
    n_edge = min(len(edge_out), DEST_MAX // 2)
    edge_pick = sorted(edge_out, key=lambda i: -links[i]["rank"])[:n_edge]
    r_, c_ = np.nonzero(succ_link >= 0)
    keep_ = succ_mov.max(1)[r_, c_] >= 0
    gm = csr_matrix((np.ones(keep_.sum()), (r_[keep_], succ_link[r_[keep_], c_[keep_]])), shape=(L, L))
    _, lab_m = connected_components(gm, directed=True, connection="strong")
    main = lab_m == np.bincount(lab_m).argmax()  # interior destinations only where every car can get to and leave
    interior = [i for i in range(L) if i not in set(edge_out) and llen[i] > 20 and main[i]]
    lw = np.array([llen[i] for i in interior])
    inner_pick = list(rng.choice(interior, size=min(DEST_MAX - n_edge, len(interior)), replace=False, p=lw / lw.sum()))
    dest_link = np.array(edge_pick + inner_pick, np.int32)
    dw = np.array([per_class[links[i]["rank"]] for i in edge_pick] + [80.0] * len(inner_pick))
    dest_cdf = np.cumsum(dw) / dw.sum()
    net = dict(
        link_len=llen.astype(np.float32), link_lanes=np.array([l["lanes"] for l in links], np.int32),
        link_v=np.array([l["v"] for l in links], np.float32), link_rank=np.array([l["rank"] for l in links], np.int32),
        link_end=np.array([l["b"] for l in links], np.int32), link_axis=link_axis, link_stop=link_stop, link_major=link_major,
        link_bus=link_bus, link_edge_in=np.isin(np.arange(L), edge_in), link_exit=np.isin(np.arange(L), edge_out), link_leg_in=link_leg_in, link_leg_out=link_leg_out, succ_mov=succ_mov, succ_link=succ_link,
        mov_in=np.array([m["i"] for m in moves], np.int32), mov_lane=np.array([m["lane"] for m in moves], np.int32), mov_out=np.array([m["o"] for m in moves], np.int32),
        mov_out_lane=np.array([m["ol"] for m in moves], np.int32), mov_len=np.array([m["len"] for m in moves], np.float32),
        mov_cap=np.array([m["cap"] for m in moves], np.float32), mov_turn=np.array([m["turn"] for m in moves], np.int32),
        mov_node=np.array([m["node"] for m in moves], np.int32), mov_c=mov_c[:, :kc_used], mov_side=mov_side[:, :kc_used], mov_cs=mov_cs[:, :kc_used],
        c_type=np.array([c[4] for c in confs] or [0], np.int32), c_mov=np.array([(c[0], c[1]) for c in confs] or [(0, 0)], np.int32),
        node_signal=node_signal, node_cycle=cycle, node_g0=node_g0, node_offset=node_offset,
        leg_node=np.array(leg_node, np.int32), leg_axis=np.array(leg_axis, np.int32), leg_rate=np.array(leg_rate, np.float32),
        src_cdf=src_cdf.astype(np.float32), src_rate=np.float32(src_rate), dest_link=dest_link, dest_cdf=dest_cdf.astype(np.float32),
        nh=np.zeros((L, len(dest_link)), np.int32))
    info = dict(links=links, moves=moves, confs=confs, deg=deg, edge_in=edge_in, edge_out=edge_out, n_nodes=n_nodes,
                chains=chains, node_id=node_id)
    net["nh"] = next_hops(net, link_tt(net))
    geo = Geo(link_pts=lpts, link_off=off, mov_pts=[m["pts"] for m in moves], node_xy=node_xy, node_H=H,
              link_name=[l["name"] for l in links], link_cls=[l["cls"] for l in links],
              chains=[(ch["pts"], ch["fwd"], ch["bwd"], ch["attrs"]["rank"], ch["attrs"]["name"]) for ch in chains],
              leg_xy=leg_xy, bbox=bbox)
    return net, geo, info


TURN_COST = (8.0, 0.0, 4.0)  # seconds added for a left, through, right turn


def link_tt(net, speed=None):
    """Link travel times [s]: free flow (speed limit), or from measured mean speeds."""
    v = net["link_v"] * 0.85 if speed is None else np.maximum(speed, 1.0)
    return net["link_len"] / v


def next_hops(net, tt):
    """next slot (0..3) at the end of each link toward each destination link; -1 = this is the destination."""
    L = len(tt)
    sl, sm = net["succ_link"], net["succ_mov"]
    turn = np.full((L, 4), 1, int)
    for li in range(L):
        for k in range(4):
            m = sm[li, :, k].max()
            if m >= 0:
                turn[li, k] = net["mov_turn"][m]
    rows, cols, w = [], [], []
    for li in range(L):
        for k in range(4):
            lo = sl[li, k]
            if lo >= 0 and sm[li, :, k].max() >= 0:
                rows.append(li)
                cols.append(lo)
                w.append(tt[lo] + TURN_COST[turn[li, k]])
    g = csr_matrix((w, (rows, cols)), shape=(L, L))
    dist = dijkstra(g.T, indices=net["dest_link"])  # [D, L]: from each link to each destination (through the reversed graph)
    # where a destination cannot be reached (one-way pockets, turn restrictions), head for the nearest exit from the area
    exits = np.nonzero(net["link_exit"])[0]
    d_exit = dijkstra(g.T, indices=exits, min_only=True) + 600.0
    dist = np.where(np.isinf(dist), d_exit[None, :], dist)
    for li in exits:
        dist[:, li] = np.where(np.isinf(dist[:, li]), 600.0, dist[:, li])
    cost = np.full((L, 4, len(net["dest_link"])), np.inf)
    for li in range(L):
        for k in range(4):
            lo = sl[li, k]
            if lo >= 0 and sm[li, :, k].max() >= 0:
                cost[li, k] = tt[lo] + TURN_COST[turn[li, k]] + dist[:, lo]
    nh = np.argmin(cost, axis=1).astype(np.int32)
    nh[np.isinf(cost.min(1))] = 0
    for d, dl in enumerate(net["dest_link"]):
        nh[dl, d] = -1
    return nh
