"""A realistic 3-D world around a simulated (or logged) scene, fused from open map data (Overture Maps, built largely from OpenStreetMap).

The NVIDIA export (xnet.nvidia) used to be road strips and boxes: it looked like a test track. This module adds the city around them:

  buildings      Overture footprints extruded to their mapped height (`height`, else floors x 3.4 m, else 10 m), with their mapped facade
                 and roof colours where present
  road surface   the whole carriageway including junction boxes as one asphalt surface; sidewalks 3 m wide and 15 cm high with curb faces
  markings       lane dividers (dashed white), centre lines of two-way streets (double yellow), stop lines, zebra crosswalks
  ground         parks, grass and gardens (green), water (the Chicago River, fountains), the rest as concrete
  furniture      mapped trees, street lamps, benches, hydrants and bollards (Overture `base` theme), and a signal pole with heads at every
                 approach the simulator signalises and a stop sign where it puts one, so what you see is what the simulated drivers obey
  vehicles       body, cabin and wheels instead of a box; pedestrians as capsules

    python -m xnet.scene3d fetch chicago              # ~1-2 min, data/scene3d/chicago_*.parquet (Overture's public bucket, no account)
    w = scene3d.build_world("chicago", net, geo, info)
    nvidia.to_usda(net, geo, scene, "scene.usda", world=w)
    scene3d.preview(w, scene, "view.png", ...)         # offscreen render, needs pyvista + OSMesa (optional)

Google's photoreal 3D tiles would look better still, but need an API key and are licensed for display in Google's own viewers, so they are not
used. The heights, colours and furniture are as mapped; where the map is silent, the defaults above apply (each building records which).
"""
import math
import os
import sys
import zlib

import numpy as np
import shapely
from shapely import wkb
from shapely.geometry import LineString, Polygon, box

from . import network as NW

RELEASE = "overturemaps-us-west-2/release/2026-09-23.1"
DIR = "data/scene3d"
FLOOR_M, DEFAULT_H = 3.4, 10.0
SIDEWALK_M, CURB_M = 3.0, 0.15
FURNITURE = {"street_lamp", "bench", "fire_hydrant", "bollard", "waste_basket", "bicycle_parking", "post_box", "traffic_signals"}
GREEN = {"park", "grass", "garden", "flowerbed", "meadow", "playground", "village_green", "recreation_ground", "golf_course", "green", "pitch", "dog_park"}
MATERIALS = {  # name: (diffuse colour, roughness, metallic)
    "asphalt": ((0.11, 0.11, 0.12), 0.9, 0.0), "sidewalk": ((0.62, 0.61, 0.58), 0.85, 0.0), "curb": ((0.55, 0.54, 0.52), 0.8, 0.0),
    "ground": ((0.45, 0.44, 0.41), 0.95, 0.0), "grass": ((0.24, 0.42, 0.18), 0.95, 0.0), "water": ((0.12, 0.25, 0.35), 0.1, 0.0),
    "paint_white": ((0.9, 0.9, 0.88), 0.6, 0.0), "paint_yellow": ((0.93, 0.74, 0.16), 0.6, 0.0), "roof": ((0.32, 0.32, 0.33), 0.8, 0.0),
    "bark": ((0.3, 0.22, 0.15), 0.9, 0.0), "foliage": ((0.18, 0.36, 0.14), 0.9, 0.0), "metal": ((0.25, 0.27, 0.29), 0.4, 0.8),
    "signal_red": ((0.9, 0.1, 0.08), 0.3, 0.0), "sign_red": ((0.75, 0.08, 0.1), 0.5, 0.0), "wood": ((0.45, 0.32, 0.2), 0.8, 0.0),
    "glass": ((0.08, 0.1, 0.12), 0.05, 0.3), "tyre": ((0.05, 0.05, 0.05), 0.9, 0.0),
}
FACADES = [(0.72, 0.69, 0.63), (0.6, 0.55, 0.5), (0.55, 0.38, 0.3), (0.7, 0.72, 0.74), (0.45, 0.5, 0.56), (0.82, 0.8, 0.74), (0.5, 0.47, 0.44),
           (0.36, 0.42, 0.5)]
PAINT = [(0.85, 0.85, 0.87), (0.1, 0.1, 0.12), (0.55, 0.57, 0.6), (0.6, 0.1, 0.1), (0.15, 0.25, 0.5), (0.8, 0.8, 0.82), (0.3, 0.32, 0.35), (0.7, 0.6, 0.4)]


# ------------------------------------------------------------------------------------------------------------- data
def area_info(area, info=None):
    """(bbox, origin lat/lon of the scene's metre grid) for a named area, a city, or an imported map (info["origin_lat_lon"])."""
    from pgh import mine

    from . import cities
    if isinstance(area, str) and area in cities.CITIES:
        b = cities.CITIES[area]["bbox"]
        return b, ((b[1] + b[3]) / 2, (b[0] + b[2]) / 2)
    if isinstance(area, str) and area in NW.AREAS:
        return NW.AREAS[area], (mine.LAT0, mine.LON0)
    if info is not None and "origin_lat_lon" in info and isinstance(area, (tuple, list)):
        return tuple(area), tuple(info["origin_lat_lon"])
    raise ValueError(f"no geographic frame for {area!r}: use a named area, a city, or an OpenStreetMap import with its bbox")


def fetch(area, out=DIR):
    """Overture buildings and base layers for the area -> out/<area>_{buildings,infrastructure,land,land_use,water}.parquet."""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.fs as pafs
    import pyarrow.parquet as pq
    bbox, _ = area_info(area)
    w, s, e, n = bbox
    fs = pafs.S3FileSystem(anonymous=True, region="us-west-2")
    flt = (pc.field("bbox", "xmax") >= w) & (pc.field("bbox", "xmin") <= e) & (pc.field("bbox", "ymax") >= s) & (pc.field("bbox", "ymin") <= n)
    os.makedirs(out, exist_ok=True)
    parts = {"buildings": ("theme=buildings/type=building", ["id", "height", "num_floors", "min_height", "class", "facade_color", "roof_color", "geometry", "bbox"]),
             "infrastructure": ("theme=base/type=infrastructure", ["id", "subtype", "class", "geometry", "bbox"]),
             "land": ("theme=base/type=land", ["id", "subtype", "class", "geometry", "bbox"]),
             "land_use": ("theme=base/type=land_use", ["id", "subtype", "class", "geometry", "bbox"]),
             "water": ("theme=base/type=water", ["id", "subtype", "class", "geometry", "bbox"])}
    for name, (p, cols) in parts.items():
        t = ds.dataset(f"{RELEASE}/{p}/", filesystem=fs, format="parquet").to_table(columns=cols, filter=flt)
        pq.write_table(t, f"{out}/{area}_{name}.parquet")
        print(f"{area} {name}: {t.num_rows}", flush=True)


def _load(area, name, d=DIR):
    import pyarrow.parquet as pq
    p = f"{d}/{area}_{name}.parquet"
    return pq.read_table(p).to_pylist() if os.path.exists(p) else []


def stable_hash(x):
    """A hash that is the same in every process (str hashes are salted), so colours and sizes do not change between exports."""
    return zlib.crc32(str(x).encode())


def _hex(c):
    if not c or not isinstance(c, str):
        return None
    c = c.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return tuple(int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)) if len(c) == 6 else None
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------------------------- geometry
class Mesh:
    """Triangles in one material (or one colour)."""

    def __init__(self, material, color=None):
        self.material, self.color, self.V, self.F, self.n = material, color, [], [], 0

    def add(self, V, F):
        if len(V) and len(F):
            self.V.append(np.asarray(V, float))
            self.F.append(np.asarray(F, int) + self.n)
            self.n += len(V)

    def arrays(self):
        return (np.concatenate(self.V), np.concatenate(self.F)) if self.V else (np.zeros((0, 3)), np.zeros((0, 3), int))


def triangulate(poly, z):
    """A polygon (holes allowed) -> (vertices [n, 3] at height z, triangles)."""
    V, F = [], []
    for g in getattr(poly, "geoms", [poly]):
        if g.is_empty or g.area < 0.05:
            continue
        for t in shapely.constrained_delaunay_triangles(g).geoms:
            c = np.asarray(t.exterior.coords)[:3]
            k = len(V)
            V.extend([(x, y, z) for x, y in c])
            F.append((k, k + 1, k + 2))
    return np.array(V).reshape(-1, 3), np.array(F, int).reshape(-1, 3)


def walls(poly, z0, z1):
    """Vertical faces along every ring of a polygon."""
    V, F = [], []
    for g in getattr(poly, "geoms", [poly]):
        for ring in [g.exterior] + list(g.interiors):
            c = np.asarray(ring.coords)
            for a, b in zip(c[:-1], c[1:]):
                k = len(V)
                V += [(a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)]
                F += [(k, k + 1, k + 2), (k, k + 2, k + 3)]
    return np.array(V).reshape(-1, 3), np.array(F, int).reshape(-1, 3)


def strip(poly, width, z):
    """A polyline widened into a band of triangles."""
    poly = np.asarray(poly, float)
    if len(poly) < 2:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    lft, rgt = NW._offset(poly, -width / 2), NW._offset(poly, width / 2)
    n = len(poly)
    V = np.concatenate([np.c_[lft, np.full(n, z)], np.c_[rgt, np.full(n, z)]])
    F = [(i, i + 1, n + i + 1) for i in range(n - 1)] + [(i, n + i + 1, n + i) for i in range(n - 1)]
    return V, np.array(F, int)


def dashes(poly, on=3.0, off=6.0):
    p = np.asarray(poly, float)
    s = NW._poly_len(p)
    return [NW._cut(p, a, min(a + on, s[-1])) for a in np.arange(1.0, s[-1] - 1.0, on + off)]


# ------------------------------------------------------------------------------------------------------------- the world
class World:
    def __init__(self):
        self.meshes, self.instances, self.stats, self.signal_links = {}, {}, {}, np.zeros(0, int)

    def mesh(self, name, material, color=None):
        if name not in self.meshes:
            self.meshes[name] = Mesh(material, color)
        return self.meshes[name]

    def instance(self, kind, x, y, scale=1.0, yaw=0.0):
        self.instances.setdefault(kind, []).append((x, y, scale, yaw))


def build_world(area, net, geo, info, data_dir=DIR, margin=60.0):
    """Fuse the map layers with the simulator's road network into a World (meshes + furniture instances, in the scene's metres)."""
    from . import render as R
    try:
        bbox, (lat0, lon0) = area_info(area, info)
    except ValueError:  # no geographic frame (a log in its own coordinates): the roads, markings and signals only
        lat0 = lon0 = None
    if lat0 is None:
        def _load(*a):  # noqa: E306
            return []
        lat0 = lon0 = 0.0
    else:
        _load = globals()["_load"]
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0

    def proj(g):
        return shapely.transform(g, lambda c: np.c_[(c[:, 0] - lon0) * kx, (c[:, 1] - lat0) * ky])

    xs = np.concatenate([p[:, 0] for p in geo.link_pts])
    ys = np.concatenate([p[:, 1] for p in geo.link_pts])
    frame = box(xs.min() - margin, ys.min() - margin, xs.max() + margin, ys.max() + margin)
    W = World()
    # roads, sidewalks, curbs
    road = R.Map(net, geo, info).road
    shapely.prepare(road)
    W.mesh("road_surface", "asphalt").add(*triangulate(road, 0.0))
    walk = road.buffer(SIDEWALK_M, join_style=2).difference(road)
    W.mesh("sidewalks", "sidewalk").add(*triangulate(walk, CURB_M))
    W.mesh("curbs", "curb").add(*walls(road, 0.0, CURB_M))
    ground = frame.difference(road)  # under the sidewalks too: no seam at their outer edge
    # markings
    white, yellow = W.mesh("markings_white", "paint_white"), W.mesh("markings_yellow", "paint_yellow")
    for li, p in enumerate(geo.link_pts):
        if net["link_lanes"][li] == 2:
            for d in dashes(NW._offset(np.asarray(p), float(geo.link_off[li]) + NW.W)):
                white.add(*strip(d, 0.12, 0.02))
        if net["link_stop"][li] or net["link_axis"][li] >= 0:
            dd = NW._dir(p, True)
            r = np.array([dd[1], -dd[0]])
            a = p[-1] + r * (geo.link_off[li] + 0.2) - dd * 0.4
            b = p[-1] + r * (geo.link_off[li] + NW.W * net["link_lanes"][li] - 0.2) - dd * 0.4
            white.add(*strip(np.stack([a, b]), 0.4, 0.02))
    for pts, fwd, bwd, rank, name in geo.chains:
        if fwd and bwd:
            for off in (-0.12, 0.12):
                yellow.add(*strip(NW._offset(np.asarray(pts), off), 0.1, 0.021))
    for a, b in geo.leg_xy:
        if not np.any(a) and not np.any(b):
            continue
        v = b - a
        L = float(np.linalg.norm(v))
        if L < 1:
            continue
        u = v / L
        nrm = np.array([-u[1], u[0]])
        for t in np.arange(0.3, L - 0.3, 1.0):
            c = a + u * t
            white.add(*strip(np.stack([c - nrm * 1.4, c + nrm * 1.4]), 0.5, 0.022))
    # ground: parks, water, the rest
    green, water = [], []
    for r in _load(area, "land_use", data_dir):
        if r["class"] in GREEN or r["subtype"] in ("park", "recreation"):
            g = proj(wkb.loads(r["geometry"]))
            if g.geom_type in ("Polygon", "MultiPolygon"):
                green.append(g)
    for r in _load(area, "water", data_dir):
        g = proj(wkb.loads(r["geometry"]))
        if g.geom_type in ("Polygon", "MultiPolygon"):
            water.append(g)
    gr = shapely.union_all(green).intersection(ground) if green else Polygon()
    wa = shapely.union_all(water).intersection(frame) if water else Polygon()
    W.mesh("grass", "grass").add(*triangulate(gr.difference(wa), 0.03))
    W.mesh("water", "water").add(*triangulate(wa, -0.4))
    W.mesh("ground", "ground").add(*triangulate(ground.difference(wa), 0.0))  # under the grass too: no slivers between the two
    # buildings, grouped by facade colour
    nb, mapped = 0, 0
    for r in _load(area, "buildings", data_dir):
        g = proj(wkb.loads(r["geometry"]))
        if g.is_empty or not g.intersects(frame) or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        g = g.intersection(frame).buffer(0)
        if road.intersects(g):
            g = g.difference(road)  # a footprint never covers the carriageway
        if g.is_empty or g.area < 4:
            continue
        h = r["height"] or (r["num_floors"] * FLOOR_M if r["num_floors"] else None)
        mapped += h is not None
        h = float(h or DEFAULT_H)
        z0 = float(r["min_height"] or 0.0)
        col = _hex(r["facade_color"]) or FACADES[stable_hash(r["id"]) % len(FACADES)]
        q = tuple(round(c * 6) / 6 for c in col)  # a few dozen facade materials at most
        m = W.meshes.setdefault(f"facade_{int(q[0] * 6)}{int(q[1] * 6)}{int(q[2] * 6)}", Mesh("facade", q))
        m.add(*walls(g, z0, h))
        rq = _hex(r["roof_color"])
        roof = W.meshes.setdefault("roofs" if rq is None else f"roof_{int(rq[0] * 6)}{int(rq[1] * 6)}{int(rq[2] * 6)}", Mesh("roof", rq))
        roof.add(*triangulate(g, h))
        nb += 1
    # furniture: mapped trees and street furniture, plus the simulator's own signals and stop signs
    fx0, fy0, fx1, fy1 = frame.bounds
    pts = []  # (kind, x, y, scale)
    for r in _load(area, "land", data_dir):
        if r["class"] == "tree":
            g = proj(wkb.loads(r["geometry"]))
            if g.geom_type == "Point":
                pts.append(("tree", g.x, g.y, 0.8 + 0.4 * ((stable_hash(r["id"]) % 100) / 100)))
    for r in _load(area, "infrastructure", data_dir):
        if r["class"] in FURNITURE and r["class"] != "traffic_signals":
            g = proj(wkb.loads(r["geometry"]))
            if g.geom_type == "Point":
                pts.append((r["class"], g.x, g.y, 1.0))
    if pts:
        x, y = np.array([q[1] for q in pts]), np.array([q[2] for q in pts])
        ok = (x > fx0) & (x < fx1) & (y > fy0) & (y < fy1) & ~shapely.contains_xy(road, x, y)  # nothing stands on the carriageway
        for q, keep in zip(pts, ok):
            if keep:
                W.instance(q[0], q[1], q[2], q[3])
    ntree = len(W.instances.get("tree", []))
    hx, hy, hl = R.signal_heads(net, geo)
    W.signal_links = np.asarray(hl, int)  # the link each traffic_signal instance faces, in order (for its light state)
    for x, y, li in zip(hx, hy, hl):
        d = NW._dir(geo.link_pts[li], True)
        W.instance("traffic_signal", x, y, 1.0, math.atan2(d[1], d[0]))
    for li in np.nonzero(net["link_stop"])[0]:
        p = geo.link_pts[li]
        d = NW._dir(p, True)
        r = np.array([d[1], -d[0]])
        xy = p[-1] + r * (geo.link_off[li] + NW.W * net["link_lanes"][li] + 1.6)
        W.instance("stop_sign", xy[0], xy[1], 1.0, math.atan2(d[1], d[0]))
    W.stats = {"buildings": nb, "buildings_with_mapped_height": mapped, "trees": ntree,
               "furniture": {k: len(v) for k, v in W.instances.items()}, "triangles": int(sum(m.n and np.concatenate(m.F).shape[0] for m in W.meshes.values()))}
    W.frame = frame.bounds
    return W


# ------------------------------------------------------------------------------------------------------------- vehicles
def vehicle_parts(kind, length, seed):
    """Proxy geometry of an agent in its own frame: [(name, shape, centre (x, y, z), size (x, y, z) or (radius, height), material/colour)]."""
    paint = PAINT[seed % len(PAINT)]
    if kind == "ped":
        return [("body", "capsule", (0, 0, 0.9), (0.25, 1.3), ((0.3, 0.3, 0.45),))]
    if kind == "bike":
        return [("frame", "cube", (0, 0, 0.55), (1.7, 0.12, 0.5), (paint,)), ("rider", "capsule", (0, 0, 1.25), (0.22, 0.8), ((0.25, 0.25, 0.3),)),
                ("wheel_f", "wheel", (0.6, 0, 0.35), (0.35, 0.06), ("tyre",)), ("wheel_r", "wheel", (-0.6, 0, 0.35), (0.35, 0.06), ("tyre",))]
    W_ = {"bus": 2.5, "van": 2.05}.get(kind, 1.85)
    H = {"bus": 3.0, "van": 2.3}.get(kind, 1.45)
    col = {"bus": (0.85, 0.65, 0.1), "av": (0.95, 0.95, 0.97)}.get(kind, paint)
    r_w = 0.5 if kind == "bus" else 0.34
    parts = [("body", "cube", (0, 0, r_w + (H * 0.45) / 2), (length, W_, H * 0.45), (col,))]
    if kind in ("bus", "van"):
        parts.append(("cabin", "cube", (0, 0, r_w + H * 0.45 + H * 0.5 / 2 - 0.05), (length * 0.98, W_ * 0.98, H * 0.5), (col,)))
    else:
        parts.append(("cabin", "cube", (-0.1 * length, 0, r_w + H * 0.45 + 0.25), (length * 0.5, W_ * 0.88, 0.55), ("glass",)))
    if kind == "av":
        parts.append(("lidar", "wheel_z", (0.0, 0, r_w + H * 0.45 + 0.6), (0.18, 0.15), ("metal",)))
    for sx in (0.36, -0.36):
        for sy in (0.5, -0.5):
            parts.append((f"wheel_{'f' if sx > 0 else 'r'}{'l' if sy > 0 else 'r'}", "wheel", (sx * length, sy * (W_ - 0.2), r_w), (r_w, 0.25), ("tyre",)))
    return parts


# ------------------------------------------------------------------------------------------------------------- rendering
LIGHTS = [(0.95, 0.15, 0.12), (1.0, 0.7, 0.1), (0.2, 0.9, 0.4)]  # red, yellow, green
SKY = ((0.70, 0.80, 0.90), (0.36, 0.55, 0.80))


def _ego_pose(scene, k, ego, smooth=4):
    """(x, y, yaw) of the ego at frame k, the heading averaged over a few frames so a chase camera does not shake."""
    for v in scene["vehicles"]:
        j = k - v["start"]
        if v["id"] == ego and 0 <= j < len(v["x"]):
            w = smooth * scene.get("sub", 1)  # the same time window when frames were interpolated
            lo, hi = max(j - w, 0), min(j + w + 1, len(v["x"]))
            yaw = math.atan2(np.mean(np.sin(v["yaw"][lo:hi])), np.mean(np.cos(v["yaw"][lo:hi])))
            return v["x"][j], v["y"][j], yaw
    return None


def _agent_meshes(scene, k, near):
    """The agents at frame k as pyvista meshes, merged per colour."""
    import pyvista as pv
    parts = {}
    for v in scene["vehicles"]:
        j = k - v["start"]
        if not 0 <= j < len(v["x"]) or not near(v["x"][j], v["y"][j]):
            continue
        x, y, yaw = v["x"][j], v["y"][j], v["yaw"][j]
        c, s_ = math.cos(yaw), math.sin(yaw)
        for name, shape, (cx, cy, cz), size, mat in vehicle_parts(v["kind"], v.get("length", 4.5), stable_hash(v["id"]) % 97):
            wx, wy = x + cx * c - cy * s_, y + cx * s_ + cy * c
            colr = mat[0] if isinstance(mat[0], tuple) else MATERIALS[mat[0]][0]
            if shape == "cube":
                m = pv.Box(bounds=(-size[0] / 2, size[0] / 2, -size[1] / 2, size[1] / 2, -size[2] / 2, size[2] / 2)).rotate_z(math.degrees(yaw)).translate((wx, wy, cz))
            elif shape in ("capsule", "wheel_z"):
                m = pv.Cylinder(center=(wx, wy, cz), direction=(0, 0, 1), radius=size[0], height=size[1], resolution=10)
            else:
                m = pv.Cylinder(center=(wx, wy, cz), direction=(-s_, c, 0), radius=size[0], height=size[1], resolution=12)
            parts.setdefault(colr, []).append(m)
        cf = v.get("crash_frame")
        if v.get("wrecked") and (cf is None or j >= cf * scene.get("sub", 1)):  # a hazard marker over a wreck, as on the av_world side
            parts.setdefault((1.0, 0.45, 0.1), []).append(pv.Cone(center=(x, y, {"bus": 4.0, "van": 3.1}.get(v["kind"], 2.3)), direction=(0, 0, -1), height=0.7, radius=0.35, resolution=12))
    for pos in _ped_positions(scene, k):
        if near(*pos):
            parts.setdefault((0.3, 0.3, 0.45), []).append(pv.Cylinder(center=(pos[0], pos[1], 0.9), direction=(0, 0, 1), radius=0.25, height=1.7, resolution=10))
    return {c: pv.merge(ms) for c, ms in parts.items()}


def _ped_positions(scene, k):
    out = []
    for ped in scene.get("pedestrians", []):
        if "x" in ped:
            j = k - ped.get("start", 0)
            if 0 <= j < len(ped["x"]):
                out.append((ped["x"][j], ped["y"][j]))
        else:
            f = (k * scene["dt"] - ped["t"]) / ped["duration"]
            if 0 <= f <= 1:
                out.append(tuple(np.asarray(ped["from"]) + f * (np.asarray(ped["to"]) - np.asarray(ped["from"]))))
    return out


class Renderer:
    """An offscreen view of a World (pyvista + OSMesa; optional) that loads the static city once and redraws only the agents and signal
    lights per frame, so a whole clip renders in seconds. Geometry farther than `radius` from `focus` is left out."""

    def __init__(self, world, focus, radius=260.0, size=(1280, 720)):
        import pyvista as pv
        pv.global_theme.allow_empty_mesh = True
        self.world, self.fx, self.fy, self.R = world, focus[0], focus[1], radius
        self.p = p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
        p.set_background(SKY[0], top=SKY[1])
        for name, m in world.meshes.items():
            V, F = m.arrays()
            if not len(F):
                continue
            T = V[F][:, :, :2]  # keep a triangle if its bounding box meets the focus square (large ground triangles span it)
            F = F[(T[:, :, 0].max(1) > self.fx - self.R) & (T[:, :, 0].min(1) < self.fx + self.R)
                  & (T[:, :, 1].max(1) > self.fy - self.R) & (T[:, :, 1].min(1) < self.fy + self.R)]
            if len(F):
                col = m.color if m.color is not None else MATERIALS.get(m.material, ((0.5, 0.5, 0.5),))[0]
                p.add_mesh(pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel()), color=col, specular=0.15 if m.material == "water" else 0.0)
        for kind, geom, z, col in (("tree", pv.Cylinder(direction=(0, 0, 1), radius=0.18, height=3.0, resolution=8), 1.5, MATERIALS["bark"][0]),
                                   ("tree", pv.Sphere(radius=2.3, theta_resolution=10, phi_resolution=8), 4.3, MATERIALS["foliage"][0]),
                                   ("street_lamp", pv.Cylinder(direction=(0, 0, 1), radius=0.08, height=7.0, resolution=8), 3.5, MATERIALS["metal"][0]),
                                   ("street_lamp", pv.Sphere(radius=0.25), 7.0, (0.95, 0.92, 0.75)),
                                   ("traffic_signal", pv.Cylinder(direction=(0, 0, 1), radius=0.09, height=5.5, resolution=8), 2.75, MATERIALS["metal"][0]),
                                   ("traffic_signal", pv.Box(bounds=(-0.2, 0.2, -0.2, 0.2, -0.6, 0.6)), 4.9, (0.08, 0.08, 0.08)),
                                   ("stop_sign", pv.Cylinder(direction=(0, 0, 1), radius=0.05, height=2.3, resolution=6), 1.15, MATERIALS["metal"][0]),
                                   ("stop_sign", pv.Cylinder(direction=(1, 0, 0), radius=0.38, height=0.05, resolution=8), 2.3, MATERIALS["sign_red"][0]),
                                   ("bench", pv.Box(bounds=(-0.9, 0.9, -0.25, 0.25, -0.25, 0.25)), 0.4, MATERIALS["wood"][0])):
            q = np.array([(x, y, s) for x, y, s, _ in world.instances.get(kind, []) if self.near(x, y)]).reshape(-1, 3)
            if len(q):
                pd = pv.PolyData(np.c_[q[:, 0], q[:, 1], z * q[:, 2]])
                pd["s"] = q[:, 2]
                p.add_mesh(pd.glyph(geom=geom, scale="s", orient=False), color=col)
        sig = world.instances.get("traffic_signal", [])
        self.sig_xy = np.array([(x, y) for x, y, _, _ in sig]).reshape(-1, 2)
        self.sig_on = self.near(self.sig_xy[:, 0], self.sig_xy[:, 1]) if len(sig) else np.zeros(0, bool)
        self.actors = []

    def near(self, x, y):
        return (np.asarray(x) - self.fx) ** 2 + (np.asarray(y) - self.fy) ** 2 < self.R ** 2

    def frame(self, scene, k, ego=None, view="street", lights=None):
        """The image (h, w, 3) at frame k. `lights`: 0 red / 1 yellow / 2 green per link (xnet.sim.signal_state), else the heads stay dark."""
        import pyvista as pv
        for a in self.actors:
            self.p.remove_actor(a)
        self.actors = [self.p.add_mesh(m, color=c) for c, m in _agent_meshes(scene, k, self.near).items()]
        if lights is not None and self.sig_on.any():
            st = np.asarray(lights)[self.world.signal_links]
            for s in range(3):
                on = self.sig_on & (st == s)
                if on.any():
                    pts = pv.PolyData(np.c_[self.sig_xy[on], np.full(on.sum(), 4.9)])
                    self.actors.append(self.p.add_mesh(pts.glyph(geom=pv.Sphere(radius=0.32), orient=False, scale=False), color=LIGHTS[s]))
        pose = _ego_pose(scene, k, ego) if ego is not None else None
        if view == "street" and pose is not None:
            x, y, yaw = pose
            self.p.camera_position = [(x - 10 * math.cos(yaw), y - 10 * math.sin(yaw), 3.4), (x + 30 * math.cos(yaw), y + 30 * math.sin(yaw), 2.0), (0, 0, 1)]
            self.p.camera.view_angle = 65
        else:
            fx, fy = (pose[0], pose[1]) if pose is not None else (self.fx, self.fy)
            self.p.camera_position = [(fx - 230, fy - 280, 240), (fx, fy, 0), (0, 0, 1)]
            self.p.camera.view_angle = 40
        return self.p.screenshot(return_img=True)

    def close(self):
        self.p.close()


def preview(world, scene, out, frame=None, ego=None, view="street", size=(1280, 720), radius=None):
    """One offscreen render of the fused scene: view = "street" (behind the ego) or "aerial"; the default radius is 260 m street, 700 m aerial."""
    from PIL import Image
    k = scene["frames"] // 2 if frame is None else frame
    pose = _ego_pose(scene, k, ego) if ego is not None else None
    focus = pose[:2] if pose else ((world.frame[0] + world.frame[2]) / 2, (world.frame[1] + world.frame[3]) / 2)
    r = Renderer(world, focus, radius or (260.0 if view == "street" else 700.0), size)
    Image.fromarray(r.frame(scene, k, ego, view)).save(out)
    r.close()
    return out


def render_usd(path, out, time=None, eye=None, target=None, size=(1280, 720)):
    """Render a USD stage written by xnet.nvidia directly from the file (pxr + pyvista; optional), as an independent check that what
    Omniverse receives is the scene: meshes with their bound material colours, point instancers expanded, cubes / cylinders / spheres /
    capsules with their transforms at `time`."""
    import pyvista as pv
    from pxr import Gf, Usd, UsdGeom, UsdShade
    st = Usd.Stage.Open(path)
    t = Usd.TimeCode(time if time is not None else st.GetStartTimeCode())
    cache = UsdGeom.XformCache(t)
    p = pv.Plotter(off_screen=True, window_size=size, lighting="three lights")
    p.set_background((0.70, 0.80, 0.90), top=(0.36, 0.55, 0.80))

    def colour(prim):
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if mat:
            sh = UsdShade.Shader(st.GetPrimAtPath(mat.GetPath().AppendChild("surface")))
            c = sh.GetInput("diffuseColor").Get() if sh else None
            if c is not None:
                return tuple(c)
        return (0.6, 0.6, 0.6)

    def shape_mesh(prim):
        if prim.IsA(UsdGeom.Cube):
            return pv.Box(bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
        if prim.IsA(UsdGeom.Sphere):
            return pv.Sphere(radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get(), theta_resolution=10, phi_resolution=8)
        if prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
            g = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
            ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[g.GetAxisAttr().Get()]
            return pv.Cylinder(direction=ax, radius=g.GetRadiusAttr().Get(), height=g.GetHeightAttr().Get(), resolution=10)
        return None

    def transformed(m, M):
        M = np.array(M).T  # USD matrices are row-vector
        return m.transform(M, inplace=False)

    groups = {}
    inst_roots = []
    for prim in st.Traverse():
        if prim.IsA(UsdGeom.PointInstancer):
            inst_roots.append(prim)
            continue
        if any(str(prim.GetPath()).startswith(str(r.GetPath())) for r in inst_roots):
            continue
        if not UsdGeom.Imageable(prim) or UsdGeom.Imageable(prim).ComputeVisibility(t) == UsdGeom.Tokens.invisible:
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            V = np.array(mesh.GetPointsAttr().Get())
            F = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            groups.setdefault(colour(prim), []).append(pv.PolyData(V, np.c_[np.full(len(F), 3), F].ravel()))
        else:
            m = shape_mesh(prim)
            if m is not None:
                groups.setdefault(colour(prim), []).append(transformed(m, cache.GetLocalToWorldTransform(prim)))
    for prim in inst_roots:
        pi = UsdGeom.PointInstancer(prim)
        mats = pi.ComputeInstanceTransformsAtTime(t, t)
        proto = st.GetPrimAtPath(pi.GetPrototypesRel().GetTargets()[0])
        parts = [(shape_mesh(c), colour(c), cache.GetLocalToWorldTransform(c) * cache.GetLocalToWorldTransform(proto).GetInverse()) for c in proto.GetChildren()]
        for geom, col, local in parts:
            if geom is None:
                continue
            base = transformed(geom, local)
            pts = np.array([np.array(M.ExtractTranslation()) for M in mats])
            sc = np.array([Gf.Vec3d(M.GetRow3(0)).GetLength() for M in mats])
            pd = pv.PolyData(pts)
            pd["s"] = sc
            groups.setdefault(col, []).append(pd.glyph(geom=base, scale="s", orient=False))
    for col, ms in groups.items():
        p.add_mesh(pv.merge(ms), color=col)
    if eye is not None:
        p.camera_position = [eye, target, (0, 0, 1)]
    p.screenshot(out)
    p.close()
    return out


if __name__ == "__main__":
    for a in sys.argv[2:]:
        fetch(a)
