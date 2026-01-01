"""Perception data for any scenario: camera images, 3-D ground truth and a sensor model's detections, in a simple nuScenes-style
folder that any framework can read.

    av-world export-perception out --area chicago --scenario bus_dartout --seconds 20 --egos 2
    python -m bev.export out --area new_york --width 352 --height 192        # a larger rig

out/
  CAM_FRONT/000000.png ... (one folder per camera)   the rendered views (bev.rig: nuScenes' 6-camera layout)
  samples.json      one record per sample: token, timestamp, ego pose (world x, y, yaw), the image files, and
                    boxes: [{class, center (x, y, z in the ego frame, m), size (l, w, h), yaw, velocity (vx, vy), occluded}]
                    detections: what xnet.perception's sensor model reports for that frame (with misses, noise, ghosts):
                                {class, true_id, center_xy (world), center_ego_xy (ego frame)}
  calibration.json  per camera: K (3x3), ego2cam (4x4), ego2img (4x4, BEVFormer's lidar2img), width, height
  scene.json        the simulator's trajectories of the whole run (replayable, or exportable to CARLA / OpenUSD)
  data.npz          the same as arrays (bev.data format), for training with bev.train
Ego frame: x forward, y left, z up, origin on the ground under the ego; camera frame: OpenCV. The format is nuScenes-*style* (the
same quantities and conventions), not the nuScenes devkit schema.
"""
import argparse
import json
import os
import shutil

import numpy as np

from . import data as D
from .rig import Rig


def export(out, area="chicago", scenario=None, seconds=20.0, egos=2, seed=0, width=176, height=96, stride=2, warm=200.0, sensor=None):
    from PIL import Image

    from xnet import perception as PC
    rig = Rig(width=width, height=height)
    tmp = os.path.join(out, "_tmp")
    p = D.generate(area, seconds=seconds, warm=warm, egos=egos, stride=stride, seed=seed, rig=rig, out=tmp, scenario=scenario, keep_scene=True)
    d, meta = D.load(p), json.load(open(p.replace(".npz", ".json")))
    scene = json.load(open(p.replace(".npz", "_scene.json")))
    # the sensor model along each ego's trip, indexed by (ego, frame)
    sens = {}
    for ego in meta["ego_ids"]:
        r = PC.detect(scene, ego, sensor or PC.Sensor(), seed=seed)
        for i in range(len(r["frame"])):
            if r["detected"][i]:
                sens.setdefault((ego, int(r["frame"][i])), []).append({"class": PC.CLASSES[int(r["cls"][i])], "true_id": int(r["id"][i]),
                    "center_xy": [round(float(r["x"][i] + r["dx"][i]), 2), round(float(r["y"][i] + r["dy"][i]), 2)], "world_frame": True})
        for g in range(len(r["g_frame"])):
            sens.setdefault((ego, int(r["g_frame"][g])), []).append({"class": "ghost", "true_id": None,
                                                                    "center_xy": [round(float(r["g_x"][g]), 2), round(float(r["g_y"][g]), 2)], "world_frame": True})
    for name, _, _ in rig.cams:
        os.makedirs(os.path.join(out, name), exist_ok=True)
    samples = []
    for t in range(len(d["seq"])):
        ego = meta["ego_ids"][int(d["seq"][t])]
        k = int(d["frame"][t])
        files = {}
        for i, (name, _, _) in enumerate(rig.cams):
            f = f"{name}/{t:06d}.png"
            Image.fromarray(d["imgs"][t, i]).save(os.path.join(out, f))
            files[name] = f
        boxes = [{"class": D.CLASSES[int(c)], "center": [round(float(b[0]), 3), round(float(b[1]), 3), round(float(b[2]), 3)],
                  "size": [round(float(b[3]), 3), round(float(b[4]), 3), round(float(b[5]), 3)], "yaw": round(float(b[6]), 4),
                  "velocity": [round(float(b[7]), 3), round(float(b[8]), 3)], "occluded": bool(o)}
                 for b, c, o in zip(d["boxes"][t], d["cls"][t], d["occluded"][t]) if c >= 0]
        ex, ey, eyaw = (float(v) for v in d["pose"][t])
        dets = [dict(q, center_ego_xy=[round(float(np.cos(eyaw) * (q["center_xy"][0] - ex) + np.sin(eyaw) * (q["center_xy"][1] - ey)), 2),
                                       round(float(-np.sin(eyaw) * (q["center_xy"][0] - ex) + np.cos(eyaw) * (q["center_xy"][1] - ey)), 2)])
                for q in sens.get((ego, k), [])]
        samples.append({"token": f"{area}_{scenario or 'everyday'}_s{seed}_{t:06d}", "timestamp_s": round(k * 0.2, 1), "ego_id": ego,
                        "sequence": int(d["seq"][t]), "ego_pose_world": [round(float(v), 3) for v in d["pose"][t]], "images": files,
                        "boxes": boxes, "detections": dets})
    json.dump(samples, open(os.path.join(out, "samples.json"), "w"), indent=0)
    json.dump({name: {"K": rig.K(i).round(4).tolist(), "ego2cam": rig.ego2cam(i).round(5).tolist(), "ego2img": rig.ego2img()[i].round(4).tolist(),
                      "width": width, "height": height, "yaw_deg": yaw, "hfov_deg": fov} for i, (name, yaw, fov) in enumerate(rig.cams)},
              open(os.path.join(out, "calibration.json"), "w"), indent=1)
    shutil.move(p, os.path.join(out, "data.npz"))
    shutil.move(p.replace(".npz", "_scene.json"), os.path.join(out, "scene.json"))
    shutil.move(p.replace(".npz", ".json"), os.path.join(out, "meta.json"))
    shutil.rmtree(tmp, ignore_errors=True)
    n_box = sum(len(s["boxes"]) for s in samples)
    n_det = sum(len(s["detections"]) for s in samples)
    return {"out": out, "samples": len(samples), "boxes": n_box, "sensor_detections": n_det, "cameras": rig.n, "area": area, "scenario": scenario}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="av-world export-perception")
    ap.add_argument("out")
    ap.add_argument("--area", default="chicago")
    ap.add_argument("--scenario", default=None, help="a famous-incident archetype of avlab.famous, e.g. bus_dartout")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--egos", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=176)
    ap.add_argument("--height", type=int, default=96)
    a = ap.parse_args(argv)
    print(json.dumps(export(a.out, a.area, a.scenario, a.seconds, a.egos, a.seed, a.width, a.height), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
