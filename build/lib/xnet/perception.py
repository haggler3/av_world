"""A sensor model on top of recorded scenes: what an AV's perception stack would report, and where it is wrong.

The simulator's AVs act on exact gaps (plus optional noise and blackouts, `Behaviour.perception_noise / p_blind / p_ghost`).
For perception research the interesting object is the detection list. Given a scene (xnet.carla_export.trajectories)
and an ego AV, `detect` returns for every frame the ground truth around the ego and what a lidar-like sensor reports:

  occlusion      a target is hidden by the bodies of other vehicles between it and the sensor (2-D ray against oriented
                 boxes; the visible share of the target's centre line decides, so a pedestrian behind a bus is not seen)
  range          detection probability falls with distance (logistic around `Sensor.range_50`), with weather (`fog` shrinks it)
  noise          position error grows with distance, heading error, class confusion between car and van
  dropouts       whole frames lost in bursts (a flaky sensor / a compute stall), which makes every object disappear for a while
  ghosts         false positives (a phantom obstacle that lasts a few frames)
  latency        the report is `latency` frames old

The ground truth carries tags for the long tail: `occluded` (a vehicle hides it), `vru` (pedestrian or cyclist),
`emerging` (appeared from behind an occluder within 2 s), so a perception model can be scored on exactly those.

    scene = json.load(open("scene.json"))
    d = detect(scene, ego_id=3, sensor=Sensor(fog=0.5), seed=0)      # dict of numpy arrays, see below
    np.savez("perception.npz", **d)
"""
from dataclasses import dataclass, field

import numpy as np

CLASSES = ("car", "av", "bus", "van", "bike", "ped")
WIDTH = {"car": 1.9, "av": 1.9, "bus": 2.5, "van": 2.1, "bike": 0.7, "ped": 0.6}
LEN = {"car": 4.5, "av": 4.6, "bus": 12.0, "van": 7.0, "bike": 1.8, "ped": 0.6}


@dataclass
class Sensor:
    max_range: float = 70.0
    range_50: float = 45.0  # distance at which half of the visible targets are detected
    range_slope: float = 0.08
    fog: float = 0.0  # 0 clear .. 1: range_50 shrinks by this share
    pos_sigma: float = 0.15  # m at the sensor; grows 0.5 % per metre of range
    yaw_sigma: float = 0.05
    min_visible: float = 0.3  # a target with less than this share of its centre line in view is not detected
    dropout_rate: float = 0.0  # per frame chance to start a dropout burst
    dropout_frames: tuple = (3, 12)
    ghost_rate: float = 0.0  # per frame chance of a phantom object
    ghost_frames: tuple = (3, 8)
    latency: int = 0
    classes: tuple = field(default=CLASSES)


def _box(x, y, yaw, length, width):
    c, s = np.cos(yaw), np.sin(yaw)
    ax, ay = np.array([c, s]), np.array([-s, c])
    return np.array([x, y]), ax, ay, length / 2, width / 2


def _seg_hits_box(p, q, box):
    """Does the segment p-q cross the oriented box? (slab test in the box frame)"""
    ctr, ax, ay, hl, hw = box
    a = np.array([(p - ctr) @ ax, (p - ctr) @ ay])
    b = np.array([(q - ctr) @ ax, (q - ctr) @ ay])
    d, lo, hi = b - a, 0.0, 1.0
    for k, h in ((0, hl), (1, hw)):
        if abs(d[k]) < 1e-9:
            if abs(a[k]) > h:
                return False
        else:
            t0, t1 = (-h - a[k]) / d[k], (h - a[k]) / d[k]
            lo, hi = max(lo, min(t0, t1)), min(hi, max(t0, t1))
            if lo > hi:
                return False
    return True


def visible_share(ego, target, blockers, samples=5):
    """Share of the target's centre line (a segment as long as its body) that the ego can see past the blockers."""
    (tx, ty, tyaw, tl), seen = target, 0
    for f in np.linspace(-0.4, 0.4, samples):
        pt = np.array([tx + np.cos(tyaw) * tl * f, ty + np.sin(tyaw) * tl * f])
        if not any(_seg_hits_box(ego, pt, b) for b in blockers):
            seen += 1
    return seen / samples


def _frame_objects(scene, k, ego_id):
    """Everything in frame k: list of dicts (id, cls, x, y, yaw, length, width, v)."""
    objs = []
    for v in scene["vehicles"]:
        j = k - v["start"]
        if 0 <= j < len(v["x"]):
            objs.append(dict(id=v["id"], cls=v["kind"], x=v["x"][j], y=v["y"][j], yaw=v["yaw"][j], length=v.get("length", LEN[v["kind"]]),
                             width=WIDTH[v["kind"]], v=v["v"][j]))
    t = k * scene["dt"]
    for i, p in enumerate(scene["pedestrians"]):
        f = (t - p["t"]) / p["duration"]
        if 0 <= f <= 1:
            a, b = np.array(p["from"]), np.array(p["to"])
            q = a + (b - a) * f
            objs.append(dict(id=100000 + i, cls="ped", x=float(q[0]), y=float(q[1]), yaw=float(np.arctan2(*(b - a)[::-1])), length=0.6,
                             width=0.6, v=float(np.hypot(*(b - a)) / p["duration"])))
    return objs


def detect(scene, ego_id, sensor=Sensor(), seed=0):
    """Run the sensor model along the ego's trip. -> dict of arrays, one row per (frame, ground-truth object in range):
    frame, id, cls, x, y, yaw (ego frame), dist, occluded, vru, emerging, detected, dx, dy (reported - true, NaN if missed),
    plus ghosts: g_frame, g_x, g_y; and `dropout` [frames] (1 where the whole frame was lost)."""
    rng = np.random.default_rng(seed)
    ego = next(v for v in scene["vehicles"] if v["id"] == ego_id)
    rows, ghosts, dropout = [], [], np.zeros(ego["start"] + len(ego["x"]), bool)
    was_hidden, reappeared, drop_left, ghost_left, ghost_pos = set(), {}, 0, 0, None
    r50 = sensor.range_50 * (1 - 0.6 * sensor.fog)
    for k in range(ego["start"], ego["start"] + len(ego["x"])):
        j = k - ego["start"]
        e = np.array([ego["x"][j], ego["y"][j]])
        eyaw = ego["yaw"][j]
        if drop_left == 0 and rng.random() < sensor.dropout_rate:
            drop_left = int(rng.integers(*sensor.dropout_frames))
        dropped = drop_left > 0
        drop_left = max(drop_left - 1, 0)
        dropout[k] = dropped
        if ghost_left == 0 and rng.random() < sensor.ghost_rate:
            ghost_left, r, th = int(rng.integers(*sensor.ghost_frames)), rng.uniform(8, 40), eyaw + rng.normal(0, 0.4)
            ghost_pos = e + r * np.array([np.cos(th), np.sin(th)])
        if ghost_left > 0:
            ghosts.append((k, *ghost_pos))
            ghost_left -= 1
        objs = [o for o in _frame_objects(scene, k, ego_id) if o["id"] != ego_id]
        boxes = {o["id"]: _box(o["x"], o["y"], o["yaw"], o["length"], o["width"]) for o in objs if o["cls"] != "ped"}
        for o in objs:
            d = float(np.hypot(o["x"] - e[0], o["y"] - e[1]))
            if d > sensor.max_range:
                continue
            blockers = [b for i, b in boxes.items() if i != o["id"] and np.hypot(b[0][0] - e[0], b[0][1] - e[1]) < d]
            vis = visible_share(e, (o["x"], o["y"], o["yaw"], o["length"]), blockers)
            occluded = vis < sensor.min_visible
            if occluded:
                was_hidden.add(o["id"])
            elif o["id"] in was_hidden:  # visible again after being hidden
                was_hidden.discard(o["id"])
                reappeared[o["id"]] = k
            emerging = (not occluded) and k - reappeared.get(o["id"], -99) <= 10  # reappeared less than 2 s ago
            p_det = 1.0 / (1.0 + np.exp(sensor.range_slope * (d - r50))) * (0.0 if occluded else min(1.0, vis + 0.3))
            det = (not dropped) and rng.random() < p_det
            noise = rng.normal(0, sensor.pos_sigma * (1 + 0.005 * d), 2) if det else np.array([np.nan, np.nan])
            rows.append((k, o["id"], CLASSES.index(o["cls"]), o["x"], o["y"], o["yaw"] - eyaw, d, occluded, o["cls"] in ("ped", "bike"),
                         bool(emerging), det, noise[0], noise[1]))
    cols = ("frame", "id", "cls", "x", "y", "yaw", "dist", "occluded", "vru", "emerging", "detected", "dx", "dy")
    arr = list(zip(*rows)) if rows else [[] for _ in cols]
    out = {c: np.asarray(a) for c, a in zip(cols, arr)}
    g = np.asarray(ghosts).reshape(-1, 3)
    out.update(g_frame=g[:, 0].astype(int), g_x=g[:, 1], g_y=g[:, 2], dropout=dropout, ego=np.array([ego_id]), classes=np.array(CLASSES))
    if sensor.latency:
        out["frame_reported"] = out["frame"] + sensor.latency
    return out


def summary(d):
    """Detection rates overall and on the long-tail subsets."""
    def rate(m):
        return float(d["detected"][m].mean()) if m.any() else float("nan")
    allm = np.ones(len(d["frame"]), bool)
    return {"objects": int(allm.sum()), "detected": rate(allm), "occluded_share": float(d["occluded"].mean()) if allm.any() else float("nan"),
            "vru_detected": rate(d["vru"]), "emerging_detected": rate(d["emerging"]), "far_detected": rate(d["dist"] > 50),
            "ghost_frames": int(len(d["g_frame"])), "dropout_share": float(d["dropout"].mean())}
