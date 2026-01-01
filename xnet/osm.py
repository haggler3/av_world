"""Any city from OpenStreetMap: turn an OSM extract into the road network the simulator runs on.

    python -m xnet.osm map.osm city.npz          # OSM XML (export from openstreetmap.org, JOSM, osmium) or Overpass JSON
    python -m xnet.osm --fetch "-0.13,51.50,-0.12,51.51" city.npz   # west,south,east,north through the Overpass API (needs internet)

    from xnet import osm
    net, geo, info = osm.import_osm("map.osm")           # same objects as xnet.network.build("oakland")
    sim = xnet.sim.Sim(net, n_envs=8)

Roads (`highway=` motorway/trunk .. living_street, not links), `oneway`, `maxspeed` and the shape of each way are used. Lanes,
signals and stop signs are not read: as for the built-in Pittsburgh areas, they are inferred from the road class and the
junction (`xnet.network`), and demand comes from the road class. The local metre grid is centred on the extract.
"""
import json
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
from shapely import wkb
from shapely.geometry import LineString

from pgh import mine
from . import network as NW

HIGHWAY = {"motorway": "trunk", "trunk": "trunk", "primary": "primary", "secondary": "secondary", "tertiary": "tertiary",
           "unclassified": "unclassified", "residential": "residential", "living_street": "living_street"}


def set_origin(lat0, lon0):
    """Centre the local metre grid (pgh.mine.to_m) on a point: the Pittsburgh default would put other cities thousands of km away."""
    mine.LAT0, mine.LON0 = lat0, lon0
    mine.KX, mine.KY = 111320 * math.cos(math.radians(lat0)), 110540.0


def _speed(tag):
    if not tag:
        return None
    try:
        v = float(tag.split()[0])
    except ValueError:
        return None
    return [{"max_speed": {"value": int(round(v)), "unit": "mph" if "mph" in tag else "km/h"}, "between": None, "when": None}]


def parse_full(path_or_text):
    """-> (nodes {id: (lon, lat)}, ways [(id, [node ids], tags)], node_tags {id: tags}) from OSM XML or Overpass JSON (a path or the text)."""
    text = path_or_text
    if "\n" not in text and "<" not in text and "{" not in text:
        text = open(path_or_text, encoding="utf-8").read()
    nodes, ways, ntags = {}, [], {}
    if text.lstrip().startswith("{"):
        for e in json.loads(text)["elements"]:
            if e["type"] == "node":
                nodes[e["id"]] = (e["lon"], e["lat"])
                if e.get("tags"):
                    ntags[e["id"]] = e["tags"]
            elif e["type"] == "way":
                ways.append((e["id"], e["nodes"], e.get("tags", {})))
        return nodes, ways, ntags
    root = ET.fromstring(text)
    for e in root.iter("node"):
        nodes[int(e.get("id"))] = (float(e.get("lon")), float(e.get("lat")))
        tg = {t.get("k"): t.get("v") for t in e.iter("tag")}
        if tg:
            ntags[int(e.get("id"))] = tg
    for w in root.iter("way"):
        ways.append((int(w.get("id")), [int(n.get("ref")) for n in w.iter("nd")], {t.get("k"): t.get("v") for t in w.iter("tag")}))
    return nodes, ways, ntags


def parse(path_or_text):
    """-> (nodes {id: (lon, lat)}, ways [(id, [node ids], tags)])."""
    return parse_full(path_or_text)[:2]


def rows_from_osm(nodes, ways):
    """Road segments in the Overture schema that xnet.network.build reads. Needs set_origin() to be called first."""
    use = [(i, ns, t) for i, ns, t in ways if t.get("highway") in HIGHWAY and all(n in nodes for n in ns) and len(ns) >= 2]
    count = {}
    for _, ns, _ in use:
        for n in set(ns):
            count[n] = count.get(n, 0) + 1
    rows = []
    for i, ns, t in use:
        ll = np.array([nodes[n] for n in ns])
        x, y = mine.to_m(ll[:, 0], ll[:, 1])
        cum = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        if cum[-1] < 1.0:
            continue
        cons = [{"connector_id": str(n), "at": float(c / cum[-1])} for k, (n, c) in enumerate(zip(ns, cum))
                if k in (0, len(ns) - 1) or count[n] > 1]
        ow = t.get("oneway", "no")
        deny = [] if ow not in ("yes", "true", "1", "-1") else [{"access_type": "denied", "between": None,
                                                                 "when": {"heading": "forward" if ow == "-1" else "backward"}}]
        rows.append({"id": str(i), "names": {"primary": t.get("name", "")}, "class": HIGHWAY[t["highway"]], "subclass": None,
                     "connectors": cons, "access_restrictions": deny, "speed_limits": _speed(t.get("maxspeed")),
                     "geometry": wkb.dumps(LineString(ll)),
                     "bbox": {"xmin": float(ll[:, 0].min()), "xmax": float(ll[:, 0].max()), "ymin": float(ll[:, 1].min()),
                              "ymax": float(ll[:, 1].max())}})
    return rows


def import_osm(source, bbox=None, seed=0, detect=True, **kw):
    """OSM file (or text) -> (net, geo, info), like xnet.network.build. `bbox` (west, south, east, north) crops the extract; default:
    everything in the file. With `detect` (default) xnet.junctions cleans the map first (roundabouts become one junction, divided roads and
    offset T's one junction, overpasses are not junctions) and the traffic-signal / stop-sign tags decide the junction control; info["junctions"]
    holds the detected junctions."""
    nodes, ways, ntags = parse_full(source)
    hints = None
    jres = None
    if detect:
        from . import junctions as JN
        jres = JN.detect(nodes, ways, node_tags=ntags)
        nodes, ways = jres["nodes"], jres["ways"]
        hints = jres["hints"]
    used = {n for _, ns, t in ways if t.get("highway") in HIGHWAY for n in ns if n in nodes}
    if not used:
        raise ValueError("no drivable roads (highway=residential, primary, ...) in the extract")
    lon = np.array([nodes[n][0] for n in used])
    lat = np.array([nodes[n][1] for n in used])
    bbox = bbox or (float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max()))
    old = (mine.LAT0, mine.LON0)
    origin = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
    set_origin(*origin)
    try:
        if hints is not None:  # signal / stop positions in this grid's metres
            h = {k: (np.stack(mine.to_m(*_ll(hints[k], hints["origin"])), 1) if len(hints[k]) else np.zeros((0, 2))) for k in ("signals", "stops")}
        else:
            h = None
        net, geo, info = NW.build(bbox, seed=seed, rows=rows_from_osm(nodes, ways), hints=h, **kw)
    finally:
        set_origin(*old)  # the built-in Pittsburgh areas keep their grid
    out = {**info, "origin_lat_lon": origin}
    if jres:
        out["junctions"], out["junction_stats"] = jres["junctions"], jres["stats"]
    return net, geo, out


def _ll(xy, origin):
    """Local metres of junctions.detect (centred on its own origin) back to lon / lat."""
    lat0, lon0 = origin
    xy = np.asarray(xy)
    return lon0 + xy[:, 0] / (111320.0 * math.cos(math.radians(lat0))), lat0 + xy[:, 1] / 110540.0


def fetch(bbox, url="https://overpass-api.de/api/interpreter"):
    """Overpass JSON for the drivable roads in bbox = (west, south, east, north). Needs internet."""
    import urllib.parse
    import urllib.request
    w, s, e, n = bbox
    q = f'[out:json][timeout:60];way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street)$"]({s},{w},{n},{e});(._;>;);out body;'
    with urllib.request.urlopen(urllib.request.Request(url, urllib.parse.urlencode({"data": q}).encode()), timeout=90) as r:
        return r.read().decode()


def save(net, path):
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in net.items()})


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--fetch":
        src = fetch(tuple(float(x) for x in a[1].split(",")))
        a = [src] + a[2:]
    net, geo, info = import_osm(a[0])
    save(net, a[1])
    print(f"{net['link_len'].shape[0]} links, {net['mov_len'].shape[0]} moves, {int(net['node_signal'].sum())} signals -> {a[1]}")
