"""Training samples for camera BEV perception, rendered from av_world: one run of the simulator in a city, a few AVs as egos,
and for each ego every `stride`-th frame:

  imgs      [T, n_cams, H, W, 3] uint8   the rig's views (scene3d.Renderer: the city from open map data + the agents)
  pose      [T, 3]                       the ego's world pose (x, y, yaw), for temporal alignment
  boxes     [T, N, 9]                    ego frame: x, y, z (centre), l, w, h, yaw, vx, vy (padded with 0)
  cls       [T, N] int                   CLASSES index, -1 for padding
  occluded  [T, N] bool                  another vehicle hides it from the ego (xnet.perception's 2-D line-of-sight test)
  ego2img   [n_cams, 4, 4]               BEVFormer's lidar2img

    python -m bev.data new_york 2        # -> data/bev/new_york_s2.npz (a few minutes; needs pyvista + OSMesa)
"""
import json
import math
import os
import sys
import time

import numpy as np

from .rig import Rig, world_to_ego

CLASSES = ("car", "truck", "bus", "cyclist", "pedestrian")
KIND = {"car": 0, "av": 0, "van": 1, "bus": 2, "bike": 3, "ped": 4}
OUT = "data/bev"
BEV_RANGE = 32.0  # metres each way: the BEV grid covers [-32, 32] x [-32, 32] around the ego


def dims(kind, length):
    """(l, w, h) of the box around scene3d's proxy geometry, so the labels match what is drawn."""
    from xnet.scene3d import vehicle_parts
    if kind == "ped":
        return 0.6, 0.6, 1.8
    top = 0.0
    width = 0.0
    for _, shape, (cx, cy, cz), size, _ in vehicle_parts(kind, length, 0):
        half = size[2] / 2 if shape == "cube" else (size[1] / 2 if shape in ("capsule", "wheel_z") else size[0])
        top = max(top, cz + half)
        if shape == "cube":
            width = max(width, size[1])
    if kind == "bike":
        width = 0.6  # the rider, not the frame tube
    return float(length), float(width or 0.7), float(top)


def labels(scene, k, ego_id, pose, max_objects=64):
    """Ground truth around the ego at frame k, in the ego frame (within BEV_RANGE + 4 m)."""
    from xnet.perception import _box, _frame_objects, visible_share
    objs = [o for o in _frame_objects(scene, k, ego_id) if o["id"] != ego_id]
    e = np.array(pose[:2])
    boxes = {o["id"]: _box(o["x"], o["y"], o["yaw"], o["length"], o["width"]) for o in objs if o["cls"] != "ped"}
    rows = []
    for o in objs:
        x, y = world_to_ego(np.array([o["x"], o["y"]]), pose)
        if max(abs(x), abs(y)) > BEV_RANGE + 4:
            continue
        l, w, h = dims(o["cls"], o["length"])
        yaw = math.remainder(o["yaw"] - pose[2], 2 * math.pi)
        d = float(np.hypot(o["x"] - e[0], o["y"] - e[1]))
        blockers = [b for i, b in boxes.items() if i != o["id"] and np.hypot(b[0][0] - e[0], b[0][1] - e[1]) < d]
        occ = visible_share(e, (o["x"], o["y"], o["yaw"], o["length"]), blockers) < 0.4
        rows.append(([x, y, h / 2, l, w, h, yaw, o["v"] * math.cos(yaw), o["v"] * math.sin(yaw)], KIND[o["cls"]], occ, o["id"]))
    rows = sorted(rows, key=lambda r: math.hypot(r[0][0], r[0][1]))[:max_objects]
    B = np.zeros((max_objects, 9), np.float32)
    C = np.full(max_objects, -1, np.int32)
    O = np.zeros(max_objects, bool)
    for i, (b, c, oc, _) in enumerate(rows):
        B[i], C[i], O[i] = b, c, oc
    return B, C, O


def pick_egos(scene, n, stride, min_frames=40):
    """The vehicles to carry the rig: any car (not only AVs) seen for at least min_frames and moving (> 2 m/s) at least 40 % of the
    time, ranked by (others within BEV range on average) x (share of time moving): crowded enough to learn detection from, and
    moving, so consecutive frames differ and temporal attention has something to align."""
    pos = {}
    for v in scene["vehicles"]:
        for j in range(0, len(v["x"]), stride):
            pos.setdefault(v["start"] + j, []).append((v["id"], v["x"][j], v["y"][j]))
    score = {}
    for v in scene["vehicles"]:
        if v["kind"] not in ("car", "av") or len(v["x"]) < min_frames:
            continue
        c = []
        for j in range(0, len(v["x"]), stride):
            others = pos.get(v["start"] + j, [])
            c.append(sum(1 for i, x, y in others if i != v["id"] and max(abs(x - v["x"][j]), abs(y - v["y"][j])) < BEV_RANGE))
        moving = float(np.mean(np.asarray(v["v"]) > 2.0))
        if moving >= 0.4:  # a car stuck in a queue the whole time gives crowded but static, useless data for the temporal stage
            score[v["id"]] = float(np.mean(c)) * moving
    best = sorted(score, key=lambda i: -score[i])[:n]
    return [v for v in scene["vehicles"] if v["id"] in best]


def generate(area="new_york", seconds=20.0, warm=200.0, egos=3, stride=2, seed=0, rig=None, knobs=None, out=OUT):
    """Simulate, render and label. -> path of the .npz (all egos' sequences concatenated, with `seq` marking each one)."""
    from xnet import carla_export as CE
    from xnet import cities
    from xnet import network as NW
    from xnet import scene3d as S3
    from xnet import sim as X
    rig = rig or Rig()
    t_start = time.time()
    net, geo, info = cities.build(area) if area in cities.CITIES else NW.build(area)
    world = S3.build_world(area, net, geo, info)
    # denser than the simulator's calibrated default: a detector needs objects in view (demand x2.5, pedestrians x4)
    beh = X.Behaviour(**({"p_av": 0.25, "ped_scale": 4.0, "p_bike": 0.05, "demand": 2.5} | (knobs or {})))
    sim = X.Sim(net, n_envs=1, beh=beh, seed=seed, reroute_s=0)
    sim.run(warm)
    t0 = float(np.asarray(sim.st.t).reshape(-1)[0])
    scene = CE.trajectories(net, geo, sim.run(seconds, record=(0, 1)))
    netn = {k: np.asarray(v) for k, v in net.items()}
    avs = pick_egos(scene, egos, stride)
    E2I = rig.ego2img().astype(np.float32)
    acc = {k: [] for k in ("imgs", "pose", "boxes", "cls", "occluded", "seq", "frame")}
    t_render = 0.0
    for si, ego in enumerate(avs):
        js = list(range(0, len(ego["x"]), stride))
        xs, ys = np.array(ego["x"])[js], np.array(ego["y"])[js]
        r = S3.Renderer(world, ((xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2),
                        90.0 + 0.5 * float(np.hypot(np.ptp(xs), np.ptp(ys))), size=(rig.width, rig.height))
        for j in js:
            k = ego["start"] + j
            pose = (ego["x"][j], ego["y"][j], ego["yaw"][j])
            t1 = time.time()
            r.set_agents(scene, k, np.asarray(X.signal_state(netn, t0 + k * scene["dt"])), hide=ego["id"])
            views = []
            for i in range(rig.n):
                eye, target = rig.world_eye_target(i, pose)
                views.append(r.shoot(eye, target, rig.vfov(i), fresh=True))
            t_render += time.time() - t1
            B, C, O = labels(scene, k, ego["id"], pose)
            for key, val in (("imgs", np.stack(views)), ("pose", pose), ("boxes", B), ("cls", C), ("occluded", O), ("seq", si), ("frame", k)):
                acc[key].append(val)
        r.close()
    os.makedirs(out, exist_ok=True)
    path = f"{out}/{area}_s{seed}.npz"
    arrays = {k: np.asarray(v) for k, v in acc.items()}
    arrays["pose"] = arrays["pose"].astype(np.float32)
    np.savez_compressed(path, ego2img=E2I, dt=np.float32(scene["dt"] * stride), **arrays)
    meta = {"area": area, "seed": seed, "samples": len(acc["seq"]), "egos": len(avs), "rig": {"width": rig.width, "height": rig.height,
            "cams": rig.cams}, "classes": CLASSES, "bev_range": BEV_RANGE, "render_s": round(t_render, 1),
            "render_s_per_sample": round(t_render / max(len(acc["seq"]), 1), 3), "total_s": round(time.time() - t_start, 1)}
    json.dump(meta, open(path.replace(".npz", ".json"), "w"), indent=1)
    return path


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    a = sys.argv[1:]
    p = generate(a[0] if a else "new_york", seed=int(a[1]) if len(a) > 1 else 0)
    print(p, json.load(open(p.replace(".npz", ".json"))))
