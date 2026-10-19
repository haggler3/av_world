"""Mine Pittsburgh's intersections from the Overture road network (data/pgh_segments.parquet, from pgh.fetch).

An intersection is a connector where 3 or 4 drivable roads meet (a segment passing through a connector gives two legs).
For each leg: bearing, street name, road class, speed limit, one-way direction and the distance to the next connector.
Kept: 3- and 4-leg intersections whose legs fit the simulator's four approach slots (each leg within 35 degrees of a
slot 90 degrees apart), with every leg at least 25 m long (shorter legs are usually a divided road or a complex junction).

Overture has no lane counts and no traffic-signal locations, so those are INFERRED from road class and marked as such:
  lanes      trunk / primary: 2 in + 2 out (+ a left-turn bay when signalised); everything else 1 + 1; one-way roads
             get their lanes in the one direction
  control    signal if the two crossing roads are both tertiary or above and one is secondary or above;
             otherwise a two-way stop where the lower-class road stops (a T-junction's stem stops)
  demand     vehicles per hour per approach by class, an average hour (trunk 600 .. residential 60)
  speed      the mapped speed limit if there is one, else Pittsburgh's 25 mph default (trunk 40 mph)

    python -m pgh.mine      # -> data/pgh_intersections.json, docs/pgh_map.png
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
from shapely import wkb

DRIVE = {"trunk": 5, "primary": 4, "secondary": 3, "tertiary": 2, "unclassified": 1, "residential": 1, "living_street": 0}
DEMAND = {5: 600, 4: 450, 3: 320, 2: 200, 1: 60, 0: 30}  # average hour, not peak
LAT0, LON0 = 40.44, -79.99
KX, KY = 111320 * math.cos(math.radians(LAT0)), 110540.0
MIN_LEG = 25.0
SLOT_TOL = 35.0


def to_m(lon, lat):
    return (np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY


def oneway(ar):
    """(forward ok, backward ok) from Overture access restrictions (whole-segment, all-mode denials only)."""
    fwd = bwd = True
    for r in ar or []:
        w = r.get("when") or {}
        if r.get("access_type") == "denied" and r.get("between") is None and w.get("heading") and not w.get("mode") \
                and not w.get("using") and not w.get("recognized") and not w.get("vehicle") and not w.get("during"):
            if w["heading"] == "backward":
                bwd = False
            elif w["heading"] == "forward":
                fwd = False
    return fwd, bwd


def speed_mph(sl):
    for r in sl or []:
        ms = r.get("max_speed")
        if ms and r.get("between") is None and not r.get("when"):
            return float(ms["value"]) * (1.0 if ms["unit"] == "mph" else 0.621371), "mapped"
    return None, None


def parse(rows):
    segs = {}
    for r in rows:
        cls = r["class"]
        if cls not in DRIVE or r["subclass"] == "link":
            continue
        line = wkb.loads(r["geometry"])
        xs, ys = to_m(*np.asarray(line.coords).T)
        cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs), np.diff(ys)))])
        if cum[-1] < 1:
            continue
        cons = sorted(r["connectors"] or [], key=lambda c: c["at"])
        segs[r["id"]] = dict(r=r, xs=xs, ys=ys, cum=cum, L=cum[-1], cons=[(c["connector_id"], c["at"] * cum[-1]) for c in cons])
    at = defaultdict(list)  # connector -> [(segment id, index in its connector list)]
    for sid, sg in segs.items():
        for i, (c, _d) in enumerate(sg["cons"]):
            at[c].append((sid, i))
    degree = defaultdict(int)
    for sid, sg in segs.items():
        for i, (c, d0) in enumerate(sg["cons"]):
            for direction in (+1, -1):
                j = i + direction
                d1 = sg["cons"][j][1] if 0 <= j < len(sg["cons"]) else (sg["L"] if direction > 0 else 0.0)
                degree[c] += abs(d1 - d0) >= 0.5
    return segs, at, degree


def leg_length(segs, at, degree, sid, i, direction, hops=6):
    """Distance along the road from connector i of segment sid to the next junction (3+ legs), following plain
    continuations through 2-leg connectors (crosswalks, segment splits)."""
    total = 0.0
    for _ in range(hops):
        sg = segs[sid]
        cons = sg["cons"]
        d0 = cons[i][1]
        j = i + direction
        while 0 <= j < len(cons) and degree[cons[j][0]] < 3:
            j += direction
        if 0 <= j < len(cons):
            return total + abs(cons[j][1] - d0)
        end = sg["L"] if direction > 0 else 0.0
        total += abs(end - d0)
        last = cons[-1] if direction > 0 else cons[0]
        if abs(last[1] - end) > 0.5 or degree[last[0]] != 2:
            return total
        nxt = [(s2, k) for s2, k in at[last[0]] if s2 != sid]
        if not nxt:
            return total
        sid, i = nxt[0]
        direction = +1 if segs[sid]["cons"][i][1] < 0.5 else -1
    return total


def legs_by_connector(rows):
    segs, at, degree = parse(rows)
    out = defaultdict(list)
    for sid, sg in segs.items():
        r, xs, ys, cum, L, cons = sg["r"], sg["xs"], sg["ys"], sg["cum"], sg["L"], sg["cons"]
        fwd, bwd = oneway(r["access_restrictions"])
        name = (r["names"] or {}).get("primary") or ""
        mph, src = speed_mph(r["speed_limits"])
        for i, (c, d0) in enumerate(cons):
            if degree[c] < 3:
                continue
            for direction in (+1, -1):  # +1: the leg runs toward the segment end
                room = (L - d0) if direction > 0 else d0
                if room < 0.5:
                    continue
                probe = d0 + direction * min(20.0, room)
                px, py = np.interp(probe, cum, xs), np.interp(probe, cum, ys)
                cx, cy = np.interp(d0, cum, xs), np.interp(d0, cum, ys)
                out[c].append(dict(
                    bearing=math.degrees(math.atan2(py - cy, px - cx)), name=name, cls=r["class"], rank=DRIVE[r["class"]],
                    mph=mph, mph_src=src, length=float(leg_length(segs, at, degree, sid, i, direction)), xy=(float(cx), float(cy)),
                    # outbound = travelling away from the intersection along this leg
                    out_ok=fwd if direction > 0 else bwd, in_ok=bwd if direction > 0 else fwd, seg=sid))
    return out


def assign_slots(bearings):
    """Map legs to slots a = 0..3 whose outward directions are rot - 90 + 90 a degrees. Returns (slot per leg, rot) or None."""
    best = None
    for rot in np.arange(0, 90, 1.0):
        slot_ang = np.array([rot - 90 + 90 * a for a in range(4)])
        dev = (np.asarray(bearings)[:, None] - slot_ang[None, :] + 180) % 360 - 180
        s = np.argmin(np.abs(dev), 1)
        if len(set(s)) < len(bearings):
            continue
        e = np.abs(dev[np.arange(len(s)), s])
        if e.max() > SLOT_TOL:
            continue
        cost = float((e ** 2).sum())
        if best is None or cost < best[2]:
            best = (s.tolist(), float(rot), cost)
    return best


def build(legs, cid):
    if not 3 <= len(legs) <= 4 or min(l["length"] for l in legs) < MIN_LEG:
        return None
    if not any(l["in_ok"] for l in legs) or sum(l["out_ok"] for l in legs) == 0:
        return None
    fit = assign_slots([l["bearing"] for l in legs])
    if fit is None:
        return None
    slots, rot, _ = fit
    by = {s: l for s, l in zip(slots, legs)}
    names = [n for n in dict.fromkeys(l["name"] for l in legs) if n]
    if len(names) < 2:
        return None  # one named street bending or splitting, not a crossing
    road_rank = {}
    for pair in ((0, 2), (1, 3)):
        present = [by[s] for s in pair if s in by]
        road_rank[pair] = max(l["rank"] for l in present) if present else -1
    (r02, r13) = road_rank[(0, 2)], road_rank[(1, 3)]
    signal = min(r02, r13) >= 2 and max(r02, r13) >= 3
    if len(legs) == 3:
        stem = [s for s in by if (s + 2) % 4 not in by][0]
        major = (stem + 1) % 4, (stem + 3) % 4
    else:
        major = (0, 2) if r02 >= r13 else (1, 3)
    out_legs = []
    for s in range(4):
        if s not in by:
            continue
        l = by[s]
        big = l["rank"] >= 4
        n = 2 if big else 1
        lanes_in = (n + (1 if big and signal and l["out_ok"] else 0)) if l["in_ok"] else 0
        lanes_out = n if l["out_ok"] else 0
        if l["in_ok"] != l["out_ok"] and l["rank"] >= 3:  # one-way arterials: two lanes in the one direction
            lanes_in, lanes_out = (max(lanes_in, 2), 0) if l["in_ok"] else (0, max(lanes_out, 2))
        mph = l["mph"] or (40.0 if l["rank"] == 5 else 25.0)
        out_legs.append(dict(slot=s, bearing=round(l["bearing"], 1), name=l["name"], cls=l["cls"],
                             speed_mph=round(mph), speed_src=l["mph_src"] or "default (inferred)",
                             lanes_in=lanes_in, lanes_out=lanes_out, oneway=l["in_ok"] != l["out_ok"],
                             demand_vph=DEMAND[l["rank"]] if lanes_in else 0, length_m=round(l["length"], 1)))
    x, y = legs[0]["xy"]
    return dict(id=cid, name=" & ".join(names[:3]), x=round(x, 1), y=round(y, 1),
                lon=round(LON0 + x / KX, 6), lat=round(LAT0 + y / KY, 6), n_legs=len(legs),
                control="signal" if signal else "two-way stop", major=list(major), legs=out_legs,
                inferred=["lanes", "control", "demand"] + (["speed"] if any(l["speed_src"] != "mapped" for l in out_legs) else []))


def main():
    rows = pq.read_table("data/pgh_segments.parquet", columns=["id", "names", "class", "subclass", "connectors",
                                                               "access_restrictions", "speed_limits", "geometry"]).to_pylist()
    by_c = legs_by_connector(rows)
    degrees = {c: len(l) for c, l in by_c.items()}
    out = [it for c, legs in by_c.items() if len(legs) >= 3 and (it := build(legs, c))]
    # drop intersections closer than MIN_LEG to another candidate junction (divided roads, offset T pairs)
    pts = np.array([(i["x"], i["y"]) for i in out])
    alljx = np.array([by_c[c][0]["xy"] for c, d in degrees.items() if d >= 3])
    keep = []
    for i, p in enumerate(pts):
        dd = np.hypot(*(alljx - p).T)
        if (dd[dd > 0.5] < MIN_LEG).sum() == 0:
            keep.append(out[i])
    stats = {"junctions_3plus": int(sum(d >= 3 for d in degrees.values())), "mined": len(out), "kept": len(keep),
             "signal": sum(i["control"] == "signal" for i in keep), "two_way_stop": sum(i["control"] != "signal" for i in keep),
             "4_leg": sum(i["n_legs"] == 4 for i in keep), "3_leg": sum(i["n_legs"] == 3 for i in keep)}
    print(stats)
    os.makedirs("data", exist_ok=True)
    with open("data/pgh_intersections.json", "w") as f:
        json.dump({"source": "Overture Maps transportation theme, release 2026-09-23.1 (OpenStreetMap-derived)",
                   "inferred_rules": __doc__.split("Overture has no")[1].split("python -m")[0].strip(),
                   "stats": stats, "intersections": keep}, f, separators=(",", ":"))
    draw_map(rows, keep)


def draw_map(rows, its, out="docs/pgh_map.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    lines, widths = [], []
    for r in rows:
        if r["class"] not in DRIVE and r["class"] != "motorway":
            continue
        xs, ys = to_m(*np.asarray(wkb.loads(r["geometry"]).coords).T)
        lines.append(np.stack([xs, ys], 1))
        widths.append({"motorway": 1.4, "trunk": 1.2}.get(r["class"], 0.9 if DRIVE.get(r["class"], 0) >= 3 else 0.35))
    fig, ax = plt.subplots(figsize=(11, 8), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.add_collection(LineCollection(lines, linewidths=widths, colors="#30363d"))
    for ctrl, col, z in (("two-way stop", "#8b949e", 3), ("signal", "#3ddc84", 4)):
        p = np.array([(i["x"], i["y"]) for i in its if i["control"] == ctrl])
        ax.scatter(p[:, 0], p[:, 1], s=4 if ctrl != "signal" else 7, c=col, zorder=z, lw=0,
                   label=f"{ctrl} (inferred): {len(p)}")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale()
    ax.legend(loc="lower left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3", fontsize=9, markerscale=3)
    ax.set_title(f"{len(its)} intersections auto-mined from Overture Maps (OpenStreetMap) in Pittsburgh",
                 color="#e6edf3", fontsize=12)
    os.makedirs("docs", exist_ok=True)
    fig.savefig(out, dpi=110, facecolor="#0d1117", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
