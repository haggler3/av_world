"""Hand scenes to NVIDIA's simulation stack: OpenUSD for Omniverse, and HD-map + 3D-box control videos for Cosmos-style world models.

    from xnet import nvidia
    nvidia.to_usda(net, geo, scene, "scene.usda")                          # roads + animated agents, metres, Z up, 1 time code per frame
    nvidia.control_video(net, geo, scene, ego_id, "control/")              # the ego's front camera: lane lines, crosswalks, 3-D boxes
    av-world export-nvidia out --area chicago --scenario bus_dartout       # both, for a simulated long-tail scene (or --av2 DIR for a real log)

`scene` is the trajectory format of xnet.carla_export (a recorded simulator run) or xnet.avdata (an Argoverse 2 / Waymo log).

* **OpenUSD** (`.usda`, the scene format of NVIDIA Omniverse): the road surface as meshes (every lane of every link and every path through a
  junction), lane boundaries as curves, crosswalks, and every vehicle, cyclist and pedestrian as a prim with time-sampled position, heading
  and visibility; per-agent `customData` carries the class, whether it is the AV, and whether it crashed. Stage: metres, Z up,
  `timeCodesPerSecond` = 1 / dt.
* **Control video**: what conditioning inputs for driving world models look like: a pinhole camera on the ego, rendering the map
  (lane boundaries, crosswalks) and the 3-D bounding boxes of the other agents, colour-coded by class, one PNG per frame (+ an MP4 when
  ffmpeg is installed), with the camera intrinsics and per-frame pose in `camera.json`.

Checked with OpenUSD's own library (`pip install usd-core`; tests/test_nvidia.py opens the stage and reads the samples back). Not run in
Omniverse, NuRec or Cosmos, which were not available here; the formats follow their public documentation. See docs/NVIDIA.md.
"""
import json
import math
import os
import shutil
import subprocess

import numpy as np

from . import network as NW

HEIGHT = {"car": 1.5, "av": 1.6, "bus": 3.2, "van": 2.4, "bike": 1.7, "ped": 1.75}
WIDTH = {"car": 1.9, "av": 1.9, "bus": 2.5, "van": 2.1, "bike": 0.7, "ped": 0.6}
COLOR = {"car": (0.2, 0.6, 1.0), "av": (0.1, 0.9, 0.4), "bus": (1.0, 0.8, 0.1), "van": (0.9, 0.9, 0.9), "bike": (1.0, 0.4, 0.8), "ped": (1.0, 0.3, 0.2)}


def _f(x):
    return f"{x:.3f}".rstrip("0").rstrip(".") if abs(x) < 1e7 else f"{x:.1f}"


def _tuple(p):
    return "(" + ", ".join(_f(float(v)) for v in p) + ")"


def _ident(s):
    return "".join(c if c.isalnum() else "_" for c in str(s))


def lane_polys(net, geo):
    """Lane centre lines of every link lane and junction path, as (polyline, width) pairs."""
    out = []
    for li, p in enumerate(geo.link_pts):
        for lane in range(int(net["link_lanes"][li])):
            out.append((NW._offset(np.asarray(p), float(geo.link_off[li]) + NW.W * (lane + 0.5)), NW.W))
    for m in geo.mov_pts:
        out.append((np.asarray(m), NW.W))
    return out


def lane_edges(net, geo):
    """Lane boundary polylines (both edges of every link lane)."""
    out = []
    for li, p in enumerate(geo.link_pts):
        base = float(geo.link_off[li])
        for k in range(int(net["link_lanes"][li]) + 1):
            out.append(NW._offset(np.asarray(p), base + NW.W * k))
    return out


def _strip(poly, width):
    """A polyline widened into a quad strip -> (points [2n, 3], faces of 4)."""
    left, right = NW._offset(poly, -width / 2), NW._offset(poly, width / 2)
    n = len(poly)
    pts = np.concatenate([np.c_[left, np.zeros(n)], np.c_[right, np.zeros(n)]])
    faces = [(i, i + 1, n + i + 1, n + i) for i in range(n - 1)]
    return pts, faces


def _mesh_lines(w, name, V, F, material, indent="        "):
    w(f'{indent}def Mesh "{name}" (')
    w(f'{indent}    prepend apiSchemas = ["MaterialBindingAPI"]')
    w(f"{indent})")
    w(f"{indent}{{")
    w(f"{indent}    point3f[] points = [" + ", ".join(_tuple(q) for q in V) + "]")
    w(f"{indent}    int[] faceVertexCounts = [" + ", ".join(["3"] * len(F)) + "]")
    w(f"{indent}    int[] faceVertexIndices = [" + ", ".join(str(int(i)) for i in np.asarray(F).ravel()) + "]")
    w(f'{indent}    uniform token subdivisionScheme = "none"')
    w(f"{indent}    uniform bool doubleSided = 1")
    w(f"{indent}    rel material:binding = </World/Looks/{material}>")
    w(f"{indent}}}")


class _Looks:
    """UsdPreviewSurface materials, created on demand: named ones from scene3d.MATERIALS and one per extra colour."""

    def __init__(self):
        from .scene3d import MATERIALS
        self.mats = {k: v for k, v in MATERIALS.items()}
        self.used = set()

    def name(self, m):
        if isinstance(m, str):
            self.used.add(m)
            return m
        n = "c_" + "_".join(str(int(round(255 * c))) for c in m)
        self.mats.setdefault(n, (tuple(m), 0.6, 0.0))
        self.used.add(n)
        return n

    def write(self, w):
        w('    def Scope "Looks"')
        w("    {")
        for n in sorted(self.used):
            col, rough, metal = self.mats[n]
            w(f'        def Material "{n}"')
            w("        {")
            w(f"            token outputs:surface.connect = </World/Looks/{n}/surface.outputs:surface>")
            w('            def Shader "surface"')
            w("            {")
            w('                uniform token info:id = "UsdPreviewSurface"')
            w(f"                color3f inputs:diffuseColor = {_tuple(col)}")
            w(f"                float inputs:roughness = {rough}")
            w(f"                float inputs:metallic = {metal}")
            if n == "water" or n == "glass":
                w("                float inputs:opacity = 0.85")
            w("                token outputs:surface")
            w("            }")
            w("        }")
        w("    }")


PROTOTYPES = {  # furniture: [(shape, centre z, size)], material
    "tree": [("Cylinder", 1.5, (0.18, 3.0), "bark"), ("Sphere", 4.3, (2.3,), "foliage")],
    "street_lamp": [("Cylinder", 3.5, (0.08, 7.0), "metal"), ("Sphere", 7.0, (0.25,), (0.95, 0.92, 0.75))],
    "traffic_signal": [("Cylinder", 2.75, (0.09, 5.5), "metal"), ("Cube", 4.9, (0.4, 0.4, 1.2), (0.08, 0.08, 0.08)), ("Sphere", 5.2, (0.12,), "signal_red")],
    "stop_sign": [("Cylinder", 1.15, (0.05, 2.3), "metal"), ("Cube", 2.3, (0.05, 0.75, 0.75), "sign_red")],
    "bench": [("Cube", 0.4, (1.8, 0.5, 0.1), "wood")],
    "bollard": [("Cylinder", 0.45, (0.1, 0.9), "metal")],
    "fire_hydrant": [("Cylinder", 0.4, (0.15, 0.8), "sign_red")],
    "waste_basket": [("Cylinder", 0.45, (0.25, 0.9), "metal")],
    "bicycle_parking": [("Cube", 0.4, (1.5, 0.05, 0.8), "metal")],
    "post_box": [("Cube", 0.6, (0.5, 0.5, 1.2), (0.15, 0.25, 0.6))],
}


def _shape_lines(w, name, shape, z, size, mat, looks, indent, xy=(0.0, 0.0), axis="Z"):
    w(f'{indent}def {shape} "{name}" (')
    w(f'{indent}    prepend apiSchemas = ["MaterialBindingAPI"]')
    w(f"{indent})")
    w(f"{indent}{{")
    if shape == "Cube":
        w(f"{indent}    double size = 1")
        w(f"{indent}    float3 xformOp:scale = {_tuple(size)}")
        w(f"{indent}    double3 xformOp:translate = {_tuple((xy[0], xy[1], z))}")
        w(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]')
    else:
        w(f"{indent}    double radius = {_f(size[0])}")
        if shape in ("Cylinder", "Capsule"):
            w(f"{indent}    double height = {_f(size[1])}")
            w(f'{indent}    uniform token axis = "{axis}"')
        w(f"{indent}    double3 xformOp:translate = {_tuple((xy[0], xy[1], z))}")
        w(f'{indent}    uniform token[] xformOpOrder = ["xformOp:translate"]')
    w(f"{indent}    rel material:binding = </World/Looks/{looks.name(mat)}>")
    w(f"{indent}}}")


def to_usda(net, geo, scene, path, name="av_world_scene", meta=None, world=None):
    dt = scene["dt"]
    T = scene["frames"]
    L = []
    w = L.append
    w("#usda 1.0")
    w("(")
    w('    defaultPrim = "World"')
    w("    metersPerUnit = 1")
    w('    upAxis = "Z"')
    w("    startTimeCode = 0")
    w(f"    endTimeCode = {T - 1}")
    w(f"    timeCodesPerSecond = {_f(1.0 / dt)}")
    w(f'    doc = "{name}: generated by av_world (xnet.nvidia)"')
    w(")")
    w("")
    w('def Xform "World" (')
    cd = {"generator": "av_world", "source": scene.get("source", "simulation"), **({k: str(v) for k, v in (meta or {}).items()})}
    w("    customData = {")
    w("\n".join(f'        string {_ident(k)} = "{str(v)[:200]}"' for k, v in cd.items()))
    w("    }")
    w(")")
    w("{")
    looks = _Looks()
    if world is not None:  # the fused city: road surface, sidewalks, markings, ground, buildings, furniture
        from . import scene3d as S3
        w('    def Scope "Environment"')
        w("    {")
        for mname, m in world.meshes.items():
            V, F = m.arrays()
            if len(F):
                _mesh_lines(w, _ident(mname), V, F, looks.name(m.color if m.color is not None else m.material))
        w("    }")
        w('    def Scope "Furniture"')
        w("    {")
        for kind, items in world.instances.items():
            proto = PROTOTYPES.get(kind)
            if not proto or not items:
                continue
            q = np.asarray(items, float)
            w(f'        def PointInstancer "{_ident(kind)}"')
            w("        {")
            w(f"            rel prototypes = [</World/Furniture/{_ident(kind)}/Prototypes/{_ident(kind)}>]")
            w("            int[] protoIndices = [" + ", ".join(["0"] * len(q)) + "]")
            w("            point3f[] positions = [" + ", ".join(_tuple((x, y, 0.0)) for x, y, _, _ in q) + "]")
            w("            float3[] scales = [" + ", ".join(_tuple((sc, sc, sc)) for _, _, sc, _ in q) + "]")
            w("            quath[] orientations = [" + ", ".join(_tuple((math.cos(yw / 2), 0.0, 0.0, math.sin(yw / 2))) for _, _, _, yw in q) + "]")
            w('            def Scope "Prototypes"')
            w("            {")
            w(f'                def Xform "{_ident(kind)}"')
            w("                {")
            for i, (shape, z, size, mat) in enumerate(proto):
                _shape_lines(w, f"part{i}", shape, z, size, mat, looks, "                    ")
            w("                }")
            w("            }")
            w("        }")
        w("    }")
    if world is None:
        # roads
        pts_all, counts, idx, off = [], [], [], 0
        for poly, width in lane_polys(net, geo):
            if len(poly) < 2:
                continue
            p, faces = _strip(poly, width)
            pts_all.append(p)
            for f in faces:
                counts.append(4)
                idx.extend(v + off for v in f)
            off += len(p)
        P = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
        w('    def Scope "Roads"')
        w("    {")
        w('        def Mesh "lanes"')
        w("        {")
        w("            point3f[] points = [" + ", ".join(_tuple(q) for q in P) + "]")
        w("            int[] faceVertexCounts = [" + ", ".join(map(str, counts)) + "]")
        w("            int[] faceVertexIndices = [" + ", ".join(map(str, idx)) + "]")
        w("            color3f[] primvars:displayColor = [(0.18, 0.18, 0.2)]")
        w('            uniform token subdivisionScheme = "none"')
        w("        }")
        edges = [e for e in lane_edges(net, geo) if len(e) >= 2]
        ep = np.concatenate([np.c_[e, np.full(len(e), 0.02)] for e in edges]) if edges else np.zeros((0, 3))
        w('        def BasisCurves "lane_boundaries"')
        w("        {")
        w('            uniform token type = "linear"')
        w("            int[] curveVertexCounts = [" + ", ".join(str(len(e)) for e in edges) + "]")
        w("            point3f[] points = [" + ", ".join(_tuple(q) for q in ep) + "]")
        w("            float[] widths = [0.15]")
        w("            color3f[] primvars:displayColor = [(0.9, 0.9, 0.9)]")
        w("        }")
        cw = [np.asarray(c) for c in getattr(geo, "leg_xy", []) if np.any(c)]
        w('        def Mesh "crosswalks"')
        w("        {")
        cp, cc, ci = [], [], []
        for c in cw:
            p, faces = _strip(c, 3.0)
            base = len(cp)
            cp.extend(p.tolist())
            for f in faces:
                cc.append(4)
                ci.extend(v + base for v in f)
        w("            point3f[] points = [" + ", ".join(_tuple(np.asarray(q) + [0, 0, 0.01]) for q in cp) + "]")
        w("            int[] faceVertexCounts = [" + ", ".join(map(str, cc)) + "]")
        w("            int[] faceVertexIndices = [" + ", ".join(map(str, ci)) + "]")
        w("            color3f[] primvars:displayColor = [(0.85, 0.85, 0.85)]")
        w("        }")
        w("    }")
    # agents
    w('    def Scope "Agents"')
    w("    {")
    agents = [(v, v["kind"]) for v in scene["vehicles"]]
    peds = []
    for i, p in enumerate(scene.get("pedestrians", [])):
        if "x" in p:
            peds.append(({"id": f"ped{i}", "start": p.get("start", 0), "x": p["x"], "y": p["y"], "yaw": None, "length": 0.6}, "ped"))
        elif "from" in p:  # a simulator crossing: straight line at constant speed
            k0, n = int(round(p["t"] / dt)), max(int(round(p["duration"] / dt)), 1)
            xs = np.linspace(p["from"][0], p["to"][0], n).tolist()
            ys = np.linspace(p["from"][1], p["to"][1], n).tolist()
            peds.append(({"id": f"ped{i}", "start": k0, "x": xs, "y": ys, "yaw": None, "length": 0.6}, "ped"))
    for v, kind in agents + peds:
        n = len(v["x"])
        a, b = v["start"], min(v["start"] + n, T)
        if b <= a:
            continue
        yaw = v.get("yaw")
        if yaw is None:  # heading from the path
            dx, dy = np.gradient(np.asarray(v["x"], float)), np.gradient(np.asarray(v["y"], float))
            yaw = np.arctan2(dy, dx).tolist() if n > 1 else [0.0]
        hgt = HEIGHT.get(kind, 1.5)
        w(f'        def Xform "{kind}_{_ident(v["id"])}" (')
        w("            customData = {")
        w(f'                string agent_class = "{kind}"')
        w(f"                bool is_av = {1 if v.get('av') else 0}")
        w(f"                bool crashed = {1 if v.get('crashed') else 0}")
        w("            }")
        w("        )")
        w("        {")
        z_c = 0.0 if world is not None else hgt / 2
        tr = ", ".join(f"{a + k}: {_tuple((v['x'][k], v['y'][k], z_c))}" for k in range(b - a))
        rz = ", ".join(f"{a + k}: {_f(math.degrees(yaw[k]))}" for k in range(b - a))
        vis = {0: "invisible"} if a > 0 else {}
        vis[a] = "inherited"
        if b < T:
            vis[b] = "invisible"
        w(f"            float3 xformOp:translate.timeSamples = {{ {tr} }}")
        w(f"            float xformOp:rotateZ.timeSamples = {{ {rz} }}")
        w('            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ"]')
        w("            token visibility.timeSamples = { " + ", ".join(f'{k}: "{s}"' for k, s in sorted(vis.items())) + " }")
        if world is not None:  # body, cabin, wheels
            from .scene3d import stable_hash, vehicle_parts
            for pname, shape, (cx, cy, cz), size, mat in vehicle_parts(kind, v.get("length", 4.5), stable_hash(v["id"]) % 97):
                sh = {"cube": "Cube", "capsule": "Capsule", "wheel": "Cylinder", "wheel_z": "Cylinder"}[shape]
                _shape_lines(w, pname, sh, cz, size, mat[0], looks, "            ", xy=(cx, cy), axis="Y" if shape == "wheel" else "Z")
        else:
            w('            def Cube "box"')
            w("            {")
            w("                double size = 1")
            w(f"                float3 xformOp:scale = {_tuple((v.get('length', 4.5), WIDTH.get(kind, 1.9), hgt))}")
            w('                uniform token[] xformOpOrder = ["xformOp:scale"]')
            w(f"                color3f[] primvars:displayColor = [{_tuple(COLOR.get(kind, (0.5, 0.5, 0.5)))}]")
            w("            }")
        w("        }")
    w("    }")
    if looks.used:
        looks.write(w)
    w("}")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


# ---------------------------------------------------------------------------------------------------------- control video
class Camera:
    """A pinhole camera on the ego: `height` m above the road, looking along the heading, `fov` degrees horizontal."""

    def __init__(self, width=640, height=360, fov=90.0, mount_height=1.6, forward=1.5):
        self.w, self.h = width, height
        self.f = (width / 2) / math.tan(math.radians(fov) / 2)
        self.cx, self.cy, self.zc, self.fwd = width / 2, height / 2, mount_height, forward

    def intrinsics(self):
        return [[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]]

    def project(self, P, pose):
        """World points [n, 3] -> pixel [n, 2] and depth [n] for a camera at pose (x, y, yaw)."""
        x0, y0, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        cam = np.array([x0 + self.fwd * c, y0 + self.fwd * s, self.zc])
        d = P - cam
        fwd = d[:, 0] * c + d[:, 1] * s          # along the heading
        left = -d[:, 0] * s + d[:, 1] * c        # to the left
        up = d[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx - self.f * left / fwd
            v = self.cy - self.f * up / fwd
        return np.stack([u, v], 1), fwd

    def segments(self, P, pose, near=0.5):
        """Draw a polyline: each segment clipped to the part in front of the near plane -> list of pixel segments."""
        out = []
        uv, z = self.project(P, pose)
        for i in range(len(P) - 1):
            a, b, za, zb = P[i], P[i + 1], z[i], z[i + 1]
            if za < near and zb < near:
                continue
            if za < near or zb < near:
                t = (near - za) / (zb - za)
                m = a + t * (b - a)
                if za < near:
                    a = m
                else:
                    b = m
                uv2, _ = self.project(np.stack([a, b]), pose)
                out.append(uv2)
            else:
                out.append(uv[i:i + 2])
        return out


def _densify(p, step=4.0):
    """Points every `step` m along a polyline (so range culling and near-plane clipping work on long straight links)."""
    p = np.asarray(p, float)
    if len(p) < 2:
        return p
    s = NW._poly_len(p)
    t = np.unique(np.concatenate([np.arange(0.0, s[-1], step), s]))
    return np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)


def box_corners(x, y, yaw, length, width, height):
    c, s = math.cos(yaw), math.sin(yaw)
    pts = []
    for dz in (0.0, height):
        for dx, dy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
            lx, ly = dx * length / 2, dy * width / 2
            pts.append((x + lx * c - ly * s, y + lx * s + ly * c, dz))
    return np.array(pts)


BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def control_video(net, geo, scene, ego_id, out_dir, camera=None, max_range=80.0, fps=None):
    """Render the ego's front-camera control frames (map + boxes) into out_dir/frame_XXXX.png, write camera.json and (if ffmpeg exists)
    control.mp4. -> dict(frames, camera)."""
    from PIL import Image, ImageDraw
    cam = camera or Camera()
    os.makedirs(out_dir, exist_ok=True)
    ego = next(v for v in scene["vehicles"] if v["id"] == ego_id)
    edges = [np.c_[e, np.zeros(len(e))] for e in (_densify(e) for e in lane_edges(net, geo)) if len(e) >= 2]
    cws = [np.c_[np.asarray(c), np.zeros(2)] for c in getattr(geo, "leg_xy", []) if np.any(c)]
    poses, n = [], 0
    for j in range(len(ego["x"])):
        k = ego["start"] + j
        pose = (ego["x"][j], ego["y"][j], ego["yaw"][j])
        img = Image.new("RGB", (cam.w, cam.h), (0, 0, 0))
        dr = ImageDraw.Draw(img)
        for e in edges:
            if np.min(np.hypot(e[:, 0] - pose[0], e[:, 1] - pose[1])) > max_range:
                continue
            for sgm in cam.segments(e, pose):
                dr.line([tuple(sgm[0]), tuple(sgm[1])], fill=(230, 230, 230), width=2)
        for c in cws:
            if np.hypot(c[0, 0] - pose[0], c[0, 1] - pose[1]) > max_range:
                continue
            for sgm in cam.segments(c, pose):
                dr.line([tuple(sgm[0]), tuple(sgm[1])], fill=(255, 210, 0), width=4)
        agents = [(v, v["kind"]) for v in scene["vehicles"] if v["id"] != ego_id]
        for v, kind in agents:
            i = k - v["start"]
            if not (0 <= i < len(v["x"])) or math.hypot(v["x"][i] - pose[0], v["y"][i] - pose[1]) > max_range:
                continue
            P = box_corners(v["x"][i], v["y"][i], v["yaw"][i], v.get("length", 4.5), WIDTH.get(kind, 1.9), HEIGHT.get(kind, 1.5))
            col = tuple(int(255 * q) for q in COLOR.get(kind, (0.5, 0.5, 0.5)))
            for a_, b_ in BOX_EDGES:
                for sgm in cam.segments(P[[a_, b_]], pose):
                    dr.line([tuple(sgm[0]), tuple(sgm[1])], fill=col, width=2)
        for p in scene.get("pedestrians", []):
            if "x" not in p:
                continue
            i = k - p.get("start", 0)
            if 0 <= i < len(p["x"]) and math.hypot(p["x"][i] - pose[0], p["y"][i] - pose[1]) < max_range:
                P = box_corners(p["x"][i], p["y"][i], 0.0, 0.6, 0.6, HEIGHT["ped"])
                for a_, b_ in BOX_EDGES:
                    for sgm in cam.segments(P[[a_, b_]], pose):
                        dr.line([tuple(sgm[0]), tuple(sgm[1])], fill=(255, 77, 51), width=2)
        img.save(os.path.join(out_dir, f"frame_{j:04d}.png"))
        poses.append({"frame": k, "x": pose[0], "y": pose[1], "z": cam.zc, "yaw": pose[2]})
        n += 1
    meta = {"intrinsics": cam.intrinsics(), "width": cam.w, "height": cam.h, "mount_height_m": cam.zc, "forward_offset_m": cam.fwd,
            "dt": scene["dt"], "ego": ego_id, "poses": poses,
            "legend": {"lane boundaries": "white", "crosswalks": "yellow", **{k: [int(255 * q) for q in c] for k, c in COLOR.items()}}}
    json.dump(meta, open(os.path.join(out_dir, "camera.json"), "w"), indent=1)
    if shutil.which("ffmpeg") and n:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps or round(1 / scene["dt"])), "-i", os.path.join(out_dir, "frame_%04d.png"),
                        "-pix_fmt", "yuv420p", os.path.join(out_dir, "control.mp4")], check=False)
    return {"frames": n, "camera": meta}


def _font(size):
    from PIL import ImageFont
    for f in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


class TopDown:
    """The av_world view of a scene (the simulator's own map style), following the ego: one image per frame."""

    def __init__(self, net, geo, info, size=(480, 360), span=120.0):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from . import render as R
        self.R, self.net, self.geo, self.span = R, net, geo, span
        self.fig = plt.figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor=R.BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1], facecolor=R.GROUND)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        R.Map(net, geo, info).draw(self.ax, labels=False)
        self.heads = R.signal_heads(net, geo)
        self.arts = []

    def frame(self, scene, k, ego=None, lights=None):
        from matplotlib.collections import PolyCollection
        R, ax = self.R, self.ax
        for a in self.arts:
            a.remove()
        self.arts = []
        P, C, E = [], [], []
        pose = None
        for v in scene["vehicles"]:
            j = k - v["start"]
            if not 0 <= j < len(v["x"]):
                continue
            kind = v["kind"]
            wd = {"bus": 2.5, "van": 2.1, "bike": 0.7}.get(kind, 1.85)
            P.append(R.car_polys(np.array([v["x"][j]]), np.array([v["y"][j]]), np.array([v["yaw"][j]]), np.array([v.get("length", 4.5) - 0.3]), np.array([wd]))[0])
            C.append({"av": R.AVC, "bus": R.BUS, "van": R.VAN, "bike": R.BIKE}.get(kind, R.CARS[v["id"] % len(R.CARS)]))
            wreck = v.get("wrecked") and (v.get("crash_frame") is None or j >= v["crash_frame"] * scene.get("sub", 1))
            E.append("white" if v["id"] == ego else (R.WRECK if wreck else "none"))
            if v["id"] == ego:
                pose = (v["x"][j], v["y"][j])
        if P:
            self.arts.append(ax.add_collection(PolyCollection(P, facecolors=C, edgecolors=E, linewidths=1.4, zorder=7)))
        peds = scene3d_peds(scene, k)
        if peds:
            q = np.array(peds)
            self.arts.append(ax.scatter(q[:, 0], q[:, 1], s=16, c="#f778ba", edgecolors="white", lw=0.5, zorder=10))
        hx, hy, li = self.heads
        if lights is not None and len(hx):
            self.arts.append(ax.scatter(hx, hy, s=14, c=R.LIGHT[np.asarray(lights)[li]], lw=0, zorder=9))
        if pose is not None:
            w, h = self.fig.get_size_inches()
            ax.set_xlim(pose[0] - self.span / 2, pose[0] + self.span / 2)
            ax.set_ylim(pose[1] - self.span * h / w / 2, pose[1] + self.span * h / w / 2)
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        import matplotlib.pyplot as plt
        plt.close(self.fig)


def scene3d_peds(scene, k):
    from .scene3d import _ped_positions
    return _ped_positions(scene, k)


def pick_clip(scene, n):
    """(AV id, start frame) of the most watchable n frames: an AV that covers ground with traffic around it."""
    best = (-1.0, None, None)
    for v in scene["vehicles"]:
        if not v["av"] or len(v["x"]) < min(n, scene["frames"]) // 2:
            continue
        sp = np.asarray(v["v"])
        m = min(n, len(sp))
        dist = np.convolve(sp, np.ones(m), "valid")
        j = int(np.argmax(dist))
        k = v["start"] + j + m // 2
        x, y = v["x"][j + m // 2], v["y"][j + m // 2]
        crowd = sum(1 for u in scene["vehicles"] if u is not v and 0 <= k - u["start"] < len(u["x"])
                    and (u["x"][k - u["start"]] - x) ** 2 + (u["y"][k - u["start"]] - y) ** 2 < 50 ** 2)
        score = dist[j] * scene["dt"] * (1 + 0.15 * min(crowd, 8))
        if score > best[0]:
            best = (score, v["id"], v["start"] + j)
    return best[1], best[2]


def upsample(scene, m):
    """The scene with m - 1 frames interpolated between each pair of simulator frames (positions linearly, headings the short way round),
    so clips play at m / dt frames per second instead of 1 / dt."""
    if m <= 1:
        return scene

    def interp(a, wrap=False):
        a = np.unwrap(np.asarray(a, float)) if wrap else np.asarray(a, float)
        t = np.arange((len(a) - 1) * m + 1) / m
        return np.interp(t, np.arange(len(a)), a).tolist()
    out = dict(scene, dt=scene["dt"] / m, frames=(scene["frames"] - 1) * m + 1, sub=scene.get("sub", 1) * m)
    out["vehicles"] = [dict(v, start=v["start"] * m, x=interp(v["x"]), y=interp(v["y"]), yaw=interp(v["yaw"], True), v=interp(v["v"]))
                       for v in scene["vehicles"]]
    out["pedestrians"] = [dict(q, start=q.get("start", 0) * m, x=interp(q["x"]), y=interp(q["y"])) if "x" in q else q
                          for q in scene.get("pedestrians", [])]
    return out


def _write_video(frames, out, fps):
    """frames (PIL images) -> out: an animated GIF, or an H.264 MP4 if out ends in .mp4 (needs ffmpeg)."""
    from PIL import Image
    if out.endswith(".mp4"):
        w, h = frames[0].size
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-preset", "slow",
               "-movflags", "+faststart", out]
        subprocess.run(cmd, input=b"".join(f.convert("RGB").tobytes() for f in frames), check=True)
        return out
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=round(1000 / fps), loop=0, optimize=True)
    return out


def compare_clip(net, geo, info, scene, world, ego_id, out, seconds=8.0, start=None, panel=(480, 360), smooth=4, t0=None,
                 labels=("av_world simulation", "OpenUSD stage for Omniverse")):
    """A side-by-side clip of one scene: the av_world view on the left, the 3-D stage (as exported) on the right, both following the ego
    (ego_id None: the AV that makes the most watchable clip), with the simulator's signal states when the scene's start time t0 is known.
    `smooth` frames are interpolated per simulator step (4: 20 fps for dt = 0.2 s). `out`: a .gif or .mp4 path, or a list of them (rendered
    once). Returns the path(s), or None if pyvista cannot render here."""
    from PIL import Image, ImageDraw

    from . import scene3d as S3
    from . import sim as X
    n = int(round(seconds / scene["dt"]))
    if ego_id is None:
        ego_id, start = pick_clip(scene, n)
    ego = next(v for v in scene["vehicles"] if v["id"] == ego_id)
    n = min(n, len(ego["x"]))
    if start is None and ego.get("crash_frame") is not None:  # a crash: about 70 % of the way into the clip
        j = ego["crash_frame"]
        start = ego["start"] + int(np.clip(j - int(0.7 * n), 0, max(len(ego["v"]) - n, 0)))
    if start is None:  # the stretch of the ego's life where it moves most
        sp = np.convolve(ego["v"], np.ones(n), "valid") if len(ego["v"]) >= n else np.array([0.0])
        start = ego["start"] + int(np.argmax(sp))
    ks0 = [k for k in range(start, start + n) if 0 <= k - ego["start"] < len(ego["x"])]
    xs = np.array([ego["x"][k - ego["start"]] for k in ks0])
    ys = np.array([ego["y"][k - ego["start"]] for k in ks0])
    focus = ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2)
    radius = 220.0 + 0.5 * float(np.hypot(np.ptp(xs), np.ptp(ys)))
    try:
        r3 = S3.Renderer(world, focus, radius, size=panel)
    except Exception:  # no pyvista / offscreen OpenGL: the export is complete without the clip
        return None
    m = max(int(smooth), 1)
    sc = upsample(scene, m)
    ks = list(range(ks0[0] * m, ks0[-1] * m + 1))
    td = TopDown(net, geo, info, size=panel)
    netn = {k: np.asarray(v) for k, v in net.items()} if t0 is not None else None
    bar = 30
    font = _font(15)
    frames = []
    for k in ks:
        lights = np.asarray(X.signal_state(netn, t0 + k * sc["dt"])) if t0 is not None else None
        a = td.frame(sc, k, ego_id, lights)
        b = r3.frame(sc, k, ego_id, "street", lights)
        img = Image.new("RGB", (2 * panel[0] + 4, panel[1] + bar), (13, 17, 23))
        img.paste(Image.fromarray(a).resize(panel), (0, bar))
        img.paste(Image.fromarray(b).resize(panel), (panel[0] + 4, bar))
        d = ImageDraw.Draw(img)
        for i, t in enumerate(labels):
            d.text((i * (panel[0] + 4) + 12, 7), t, fill=(230, 237, 243), font=font)
        t = f"{(k - ks[0]) * sc['dt']:.1f} s"
        d.text((img.width - 12 - d.textlength(t, font=font), 7), t, fill=(139, 148, 158), font=font)
        frames.append(img)
    td.close()
    r3.close()
    fps = 1.0 / sc["dt"]
    outs = [out] if isinstance(out, str) else list(out)
    for o in outs:
        _write_video(frames, o, fps)
    return out


def _hunt_crash(net, geo, beh, seed, warm, seconds, every, budget, copies=8):
    """Run `copies` of the city in `seconds` chunks until an AV crashes, then replay that copy's chunk from a snapshot with recording on
    (the replay steps one at a time, so it is checked to contain the crash). -> (frames, copy, snapshot, sim) or (None, ...) if none."""
    from . import carla_export as CE
    from . import sim as X
    i = X.METRICS.index("av_crash")
    sim = X.Sim(net, n_envs=copies, beh=beh, seed=seed, reroute_s=0)
    sim.run(warm)
    for _ in range(max(int(budget / seconds), 1)):
        snap = sim.st
        sim.run(seconds)
        hit = np.nonzero(np.asarray(sim.st.m)[:, i] > np.asarray(snap.m)[:, i])[0]
        after = sim.st
        for e in hit:
            sim.st = snap
            frames = sim.run(seconds, record=(int(e), every))
            sc = CE.trajectories(net, geo, frames, step_s=X.DT * every)
            if any(v["av"] and v["crashed"] for v in sc["vehicles"]):
                return frames, int(e), snap, sim
        sim.st = after
    return None, 0, None, sim


def export(out, area="chicago", scenario=None, intensity=1.0, seconds=20.0, warm=120.0, av2=None, seed=0, every=1, world3d=True, fetch=False, previews=True, clip=True,
           find_crash=0.0):
    """A long-tail scene, ready for NVIDIA tools: out/scene.usda, out/scene.json, out/control/ (frames, camera.json, control.mp4), out/meta.json.
    From the simulator (`area`, optionally a famous-event `scenario` of avlab.famous at `intensity`) or from a real log (`av2`: an Argoverse 2
    scenario directory). With `world3d`, the city around the roads is fused in from Overture (xnet.scene3d; `fetch` downloads it first if it is
    not in data/scene3d); with `previews`, two offscreen renders are written if pyvista can render here, and with `clip` out/compare.gif: the
    av_world view and the 3-D stage side by side, following the ego. `find_crash` > 0: simulate up to that many seconds in 8 copies of
    the city, and export the first `seconds` window in which an AV crashes."""
    import jax

    from . import carla_export as CE
    os.makedirs(out, exist_ok=True)
    meta = {"source": None, "area": area, "scenario": scenario, "intensity": intensity}
    if av2:
        from . import avdata as AV
        sc = AV.read_av2(av2)
        scene = AV.av2_to_scene(sc, 2)
        net, geo, info = AV.av2_network(sc)
        meta.update(source=f"argoverse2:{sc['id']}", area=sc["city"])
    else:
        from . import cities
        from . import sim as X
        net, geo, info = cities.build(area) if area in cities.CITIES else NW.build(area)
        if scenario:
            from avlab import famous as FM
            kw = dict(FM.SCENARIOS[scenario][2](intensity), risk=1.0)
            beh = FM.behaviour(kw)
            meta["knobs"] = kw
        else:
            beh = X.Behaviour(p_av=0.25)
        if find_crash:
            frames, e, snap, sim = _hunt_crash(net, geo, beh, seed, warm, seconds, every, find_crash)
            meta["crash_search"] = "found" if frames is not None else f"no AV crash in {find_crash:.0f} s x 8 copies: an ordinary run instead"
        if not find_crash or frames is None:
            sim = X.Sim(net, n_envs=1, beh=beh, seed=seed, reroute_s=0)
            sim.run(warm)
            e, snap = 0, sim.st
            frames = sim.run(seconds, record=(0, every))
        meta["t0"] = float(np.asarray(snap.t)[e])
        scene = CE.trajectories(net, geo, frames, step_s=X.DT * every)
        d = dict(zip(X.METRICS, np.asarray(sim.st.m)[e] - np.asarray(snap.m)[e]))
        meta["events"] = {k: float(d[k]) for k in X.CRASH_NAMES + X.TAILS + ("av_crash", "red_runs") if d[k]}
        meta["source"] = "av_world simulation"
    avs = [v for v in scene["vehicles"] if v["av"]]
    if not avs:
        raise ValueError("no AV in the scene")
    ego = max(avs, key=lambda v: (v["crashed"], len(v["x"])))  # the AV that crashed on camera, else the one seen longest
    meta.update(ego=ego["id"], ego_crashed=ego["crashed"], frames=scene["frames"], dt=scene["dt"], vehicles=len(scene["vehicles"]))
    json.dump(scene, open(os.path.join(out, "scene.json"), "w"), separators=(",", ":"))
    world = None
    if world3d:
        from . import scene3d as S3
        if fetch and not av2 and not os.path.exists(f"{S3.DIR}/{area}_buildings.parquet"):
            S3.fetch(area)
        world = S3.build_world(area if not av2 else None, net, geo, info)  # without map layers: roads, sidewalks, markings, signals
        meta["world3d"] = world.stats
        if not av2 and not os.path.exists(f"{S3.DIR}/{area}_buildings.parquet"):
            meta["world3d_note"] = f"no map layers for {area}: run `python -m xnet.scene3d fetch {area}` or pass fetch=True for buildings and trees"
    to_usda(net, geo, scene, os.path.join(out, "scene.usda"), meta={k: v for k, v in meta.items() if k not in ("knobs", "events", "world3d")}, world=world)
    if world is not None and previews:
        try:
            from . import scene3d as S3
            k = max(range(len(ego["v"])), key=lambda j: ego["v"][j]) + ego["start"]  # the moment the ego is fastest
            for view in ("street", "aerial"):
                S3.preview(world, scene, os.path.join(out, f"preview_{view}.png"), frame=k, ego=ego["id"], view=view)
            meta["previews"] = True
        except Exception as e:  # pyvista or an offscreen OpenGL is missing: the USD is still complete
            meta["previews"] = f"skipped ({type(e).__name__})"
    if world is not None and clip:  # the av_world view and the 3-D stage side by side
        try:
            outs = [os.path.join(out, "compare.gif")] + ([os.path.join(out, "compare.mp4")] if shutil.which("ffmpeg") else [])
            r = compare_clip(net, geo, info, scene, world, ego["id"] if ego["crashed"] else None, outs, t0=meta.get("t0"))
            meta["clip"] = [os.path.basename(o) for o in outs] if r else "skipped (no offscreen OpenGL)"
        except Exception as e:
            meta["clip"] = f"skipped ({type(e).__name__}: {e})"
    r = control_video(net, geo, scene, ego["id"], os.path.join(out, "control"))
    meta["control_frames"] = r["frames"]
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1, default=float)
    return meta
