"""Instruction data for a driving VLM, built from av_world's rendered camera frames (bev/data.py) and their ground-truth boxes.

Each sample: one camera image + a question + an answer written from the labels (no model in the loop), so every answer is checkable.

    python -m vlm.data out_dir [max_frames_per_file]    # -> out_dir/{train,val}.jsonl + out_dir/img/*.png ; val = whole Boston files
"""
import glob
import json
import math
import os
import sys

import numpy as np
from PIL import Image

from bev.data import CLASSES
from bev.rig import NUSCENES_LIKE

NAMES = ("car", "truck", "bus", "cyclist", "pedestrian")


def sector(cam_yaw, fov, x, y):
    """Is the ego-frame point (x fwd, y left) inside this camera's horizontal field of view?"""
    a = math.degrees(math.atan2(y, x)) - cam_yaw
    a = (a + 180) % 360 - 180
    return abs(a) < fov / 2 - 3, a


def where(a):
    return "ahead" if abs(a) < 12 else ("to the left" if a > 0 else "to the right")


def visible(boxes, cls, cam):
    _, yaw, fov = cam
    out = []
    for b, c in zip(boxes, cls):
        if c < 0:
            continue
        ok, a = sector(yaw, fov, b[0], b[1])
        d = math.hypot(b[0], b[1])
        if ok and 2 < d < 32:
            out.append((NAMES[c], d, a, b))
    return sorted(out, key=lambda t: t[1])


def count_phrase(n, name):
    return f"{n} {name}{'' if n == 1 else 's'}"


def qa(vis):
    """(question, answer) pairs for one camera view."""
    qs = []
    counts = {n: sum(1 for v in vis if v[0] == n) for n in NAMES}
    listed = [count_phrase(c, n) for n, c in counts.items() if c]
    qs.append(("Describe what the camera sees.", ("I can see " + ", ".join(listed) + ".") if listed else "The road looks empty."))
    for n in ("car", "pedestrian", "cyclist"):
        qs.append((f"How many {n}s are visible?", str(counts[n])))
    if vis:
        _, d, a, _ = vis[0]
        qs.append(("What is the closest road user, and where?", f"A {vis[0][0]}, about {round(d)} m {where(a)}."))
    ahead = [v for v in vis if abs(v[2]) < 12]
    qs.append(("Is anything directly ahead within 15 m?", ("Yes, a %s %d m ahead." % (ahead[0][0], round(ahead[0][1]))) if ahead and ahead[0][1] < 15 else "No."))
    ped = [v for v in vis if v[0] == "pedestrian" and v[1] < 20]
    qs.append(("Should the ego vehicle be careful about pedestrians?", "Yes, a pedestrian is %d m %s." % (round(ped[0][1]), where(ped[0][2])) if ped else "No pedestrian is close."))
    return qs


def build(out, max_frames=150, every=7):
    os.makedirs(f"{out}/img", exist_ok=True)
    rows = {"train": [], "val": []}
    for f in sorted(glob.glob("data/bev/*.npz")):
        if "bevformer" in f:
            continue
        split = "val" if "boston" in f else "train"
        z = np.load(f)
        d = {k: z[k] for k in ("imgs", "boxes", "cls")}  # NpzFile re-reads the array on every access
        stem = os.path.basename(f)[:-4]
        for t in range(0, min(len(d["imgs"]), max_frames * every), every):
            for ci, cam in enumerate(NUSCENES_LIKE):
                vis = visible(d["boxes"][t], d["cls"][t], cam)
                path = f"img/{stem}_{t:04d}_{cam[0]}.png"
                Image.fromarray(d["imgs"][t, ci]).resize((352, 192), Image.BICUBIC).save(f"{out}/{path}")
                for q, a in qa(vis):
                    rows[split].append({"image": path, "camera": cam[0], "question": q, "answer": a})
    for s, r in rows.items():
        with open(f"{out}/{s}.jsonl", "w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in r)
        print(s, len(r), "QA pairs")


if __name__ == "__main__":
    build(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150)
