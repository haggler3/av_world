"""Scenes from the simulator into CARLA (https://carla.org): the road network as an OpenDRIVE map, the traffic as trajectories,
and a replay script that drives CARLA's actors along them.

    from xnet import carla_export as CE
    CE.export(sim, env=0, seconds=60, out="scene")     # -> scene.xodr, scene.json, scene_replay.py

    python scene_replay.py --host localhost --port 2000          # on a machine with CARLA running (0.9.13+)

What carries over, and what does not
  * The map: every link is a road with its lanes, every move through a junction a connecting road, grouped per junction.
    Lane widths are 3.6 m, geometry is the simulator's polylines (so curves are sequences of straight pieces). No
    traffic lights or signs, elevation or lane markings are written; CARLA draws default markings.
  * The traffic: one trajectory per vehicle (x, y, heading, speed at the simulator's 0.2 s) with its kind (car, AV, bus,
    van, bicycle) and whether it crashed; pedestrians as straight crossings. The replay sets each actor's transform every
    tick with physics off, so what you see is exactly what the simulator did, crashes included.
  * Not tested against a running CARLA here (none was available): the OpenDRIVE and trajectory files are validated
    structurally in tests/test_carla_export.py, and the replay script against a stand-in of the CARLA API.
"""
import json
import math
import os
from xml.sax.saxutils import quoteattr

import jax
import numpy as np

from . import network as NW
from . import render as R
from . import sim as X

BLUEPRINT = {"car": "vehicle.audi.a2", "av": "vehicle.lincoln.mkz_2020", "bus": "vehicle.mitsubishi.fusorosa",
             "van": "vehicle.volkswagen.t2", "bike": "vehicle.diamondback.century"}


def _segments(pts):
    """Polyline -> [(s, x, y, heading, length)] straight pieces (zero-length ones dropped)."""
    out, s = [], 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        ln = float(np.hypot(*d))
        if ln < 0.05:
            continue
        out.append((s, float(a[0]), float(a[1]), math.atan2(d[1], d[0]), ln))
        s += ln
    return out, s


def _road(rid, name, pts, n_right, junction, pred, succ, lane_links=None):
    """One OpenDRIVE <road>. pred/succ: (kind, id, contact) or None. lane_links: {lane id: (pred lane, succ lane)}."""
    seg, total = _segments(pts)
    if not seg:
        return None
    geom = "".join(f'<geometry s="{s:.3f}" x="{x:.3f}" y="{y:.3f}" hdg="{h:.6f}" length="{ln:.3f}"><line/></geometry>' for s, x, y, h, ln in seg)
    link = ""
    for tag, ref in (("predecessor", pred), ("successor", succ)):
        if ref:
            cp = f' contactPoint="{ref[2]}"' if ref[0] == "road" else ""
            link += f'<{tag} elementType="{ref[0]}" elementId="{ref[1]}"{cp}/>'
    lanes = ""
    for k in range(1, n_right + 1):
        ll = ""
        if lane_links and -k in lane_links:
            p, s_ = lane_links[-k]
            ll = "<link>" + (f'<predecessor id="{p}"/>' if p else "") + (f'<successor id="{s_}"/>' if s_ else "") + "</link>"
        lanes += (f'<lane id="-{k}" type="driving" level="false">{ll}<width sOffset="0" a="{NW.W}" b="0" c="0" d="0"/>'
                  f'<roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.12"/></lane>')
    return (f'<road name={quoteattr(name)} length="{total:.3f}" id="{rid}" junction="{junction}">'
            f'<link>{link}</link><planView>{geom}</planView>'
            f'<lanes><laneSection s="0"><left/><center><lane id="0" type="none" level="false"/></center><right>{lanes}</right></laneSection></lanes>'
            f'</road>')


def to_opendrive(net, geo, name="av_world"):
    """OpenDRIVE 1.4 text for the network. Link li -> road li; move m -> connecting road L + m in junction (its node) + 1."""
    L, M = len(net["link_len"]), len(net["mov_len"])
    mov_in, mov_out, mov_node = (np.asarray(net[k]) for k in ("mov_in", "mov_out", "mov_node"))
    lane_in, lane_out = np.asarray(net["mov_lane"]), np.asarray(net["mov_out_lane"])
    link_end = np.asarray(net["link_end"])
    start_node = np.full(L, -1)
    for m in range(M):
        start_node[mov_out[m]] = mov_node[m]
    roads, junctions = [], {}
    for li in range(L):
        ref = NW._offset(np.asarray(geo.link_pts[li]), float(geo.link_off[li]))  # the inner edge of the lane block
        has_end = bool((mov_in == li).any())
        pred = ("junction", int(start_node[li]) + 1, None) if start_node[li] >= 0 else None
        succ = ("junction", int(link_end[li]) + 1, None) if has_end else None
        r = _road(li, geo.link_name[li] if geo.link_name[li] else f"link {li}", ref, int(net["link_lanes"][li]), -1, pred, succ)
        if r:
            roads.append(r)
    for m in range(M):
        pts = NW._offset(np.asarray(geo.mov_pts[m]), -0.5 * NW.W)  # reference line on the left edge of its single lane
        ll = {-1: (-(int(lane_in[m]) + 1), -(int(lane_out[m]) + 1))}
        r = _road(L + m, f"move {m}", pts, 1, int(mov_node[m]) + 1, ("road", int(mov_in[m]), "end"), ("road", int(mov_out[m]), "start"), ll)
        if r:
            roads.append(r)
            junctions.setdefault(int(mov_node[m]), []).append(m)
    jx = ""
    for node, ms in sorted(junctions.items()):
        conns = "".join(
            f'<connection id="{k}" incomingRoad="{int(mov_in[m])}" connectingRoad="{L + m}" contactPoint="start">'
            f'<laneLink from="-{int(lane_in[m]) + 1}" to="-1"/></connection>' for k, m in enumerate(ms))
        jx += f'<junction name="junction {node}" id="{node + 1}" type="default">{conns}</junction>'
    return ('<?xml version="1.0" encoding="UTF-8"?><OpenDRIVE><header revMajor="1" revMinor="4" '
            f'name={quoteattr(name)} version="1.00" date="" north="0" south="0" east="0" west="0" vendor="av_world"/>'
            + "".join(roads) + jx + "</OpenDRIVE>")


def trajectories(net, geo, frames, step_s=X.DT):
    """frames: list of per-step State (numpy, one env). -> {"dt", "vehicles": [{id, kind, av, crashed, crash_frame, wrecked, t0, x, y, yaw,
    v}], "pedestrians": [{t, duration, from, to}]}. A slot that is reused by a new trip is a new vehicle (told apart by t0). `crashed`: the
    crash happens in the recording, at index `crash_frame` of the vehicle's arrays; `wrecked`: it is a wreck at some point (perhaps from
    a crash before the recording began)."""
    tracks = R.Tracks(net, geo)
    veh, ped = {}, []
    prev_ped = None
    kinds = {X.CAR: "car", X.BUS: "bus", X.VAN: "van", X.BIKE: "bike"}
    for k, st in enumerate(frames):
        alive = np.nonzero(st.alive)[0]
        if len(alive):
            x, y, h = tracks.pose(st.trk[alive], st.s[alive], st.length[alive])
            for j, i in enumerate(alive):
                key = (int(i), round(float(st.t0[i]), 3))
                v = veh.setdefault(key, {"id": len(veh), "kind": "av" if st.av[i] else kinds[int(st.kind[i])], "av": bool(st.av[i]),
                                         "crashed": False, "crash_frame": None, "wrecked": False, "start": k,
                                         "length": round(float(st.length[i]), 2), "x": [], "y": [], "yaw": [], "v": []})
                v["x"].append(round(float(x[j]), 2))
                v["y"].append(round(float(y[j]), 2))
                v["yaw"].append(round(float(h[j]), 3))
                v["v"].append(round(float(st.v[i]), 2))
                w = bool(st.wreck_t[i] > 0)
                if w and v["x"][1:] and not v["wrecked"]:  # intact when first seen, a wreck now: the crash is on camera
                    v["crashed"], v["crash_frame"] = True, len(v["x"]) - 1
                v["wrecked"] |= w
        if prev_ped is not None:
            for g in np.nonzero((np.asarray(st.ped_t) > 0) & (prev_ped <= 0))[0]:
                a, b = np.asarray(geo.leg_xy)[g]
                ped.append({"t": round(k * step_s, 2), "duration": round(float(st.ped_t[g]), 1), "from": a.round(2).tolist(),
                            "to": b.round(2).tolist()})
        prev_ped = np.asarray(st.ped_t)
    return {"dt": step_s, "frames": len(frames), "vehicles": sorted(veh.values(), key=lambda v: v["id"]), "pedestrians": ped}


REPLAY = '''"""Replay a scene exported by xnet.carla_export in CARLA (0.9.13+). Usage: python {name}_replay.py --host localhost --port 2000"""
import argparse
import json
import math
import time

import carla

HERE = {here!r}


def tf(x, y, yaw, z=0.3):
    """OpenDRIVE (x east, y north, counter-clockwise heading) -> CARLA/Unreal (y flipped, yaw clockwise in degrees)."""
    return carla.Transform(carla.Location(x=x, y=-y, z=z), carla.Rotation(yaw=-math.degrees(yaw)))


def replay(client, scene, xodr, speed=1.0, load_map=True):
    world = client.generate_opendrive_world(xodr) if load_map else client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, scene["dt"] / speed
    world.apply_settings(settings)
    lib, actors = world.get_blueprint_library(), {{}}
    bp = {blueprints!r}
    try:
        for k in range(scene["frames"]):
            for v in scene["vehicles"]:
                j = k - v["start"]
                if j < 0 or j >= len(v["x"]):
                    if v["id"] in actors:
                        actors.pop(v["id"]).destroy()
                    continue
                if v["id"] not in actors:
                    a = world.try_spawn_actor(lib.find(bp[v["kind"]]), tf(v["x"][j], v["y"][j], v["yaw"][j]))
                    if a is None:
                        continue
                    a.set_simulate_physics(False)
                    actors[v["id"]] = a
                actors[v["id"]].set_transform(tf(v["x"][j], v["y"][j], v["yaw"][j]))
            world.tick()
    finally:
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for a in actors.values():
            a.destroy()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--keep-map", action="store_true", help="use the map already loaded in CARLA instead of the exported OpenDRIVE")
    a = ap.parse_args()
    scene = json.load(open(HERE + "{name}.json"))
    xodr = open(HERE + "{name}.xodr").read()
    c = carla.Client(a.host, a.port)
    c.set_timeout(30.0)
    replay(c, scene, xodr, a.speed, not a.keep_map)
'''


def export(sim, env=0, seconds=60.0, out="scene", net=None, geo=None):
    """Run `sim` (an xnet.sim.Sim) for `seconds`, recording copy `env`, and write out.xodr, out.json and out_replay.py."""
    net = net if net is not None else sim.net_np
    if geo is None:
        geo = sim.NW.build("oakland")[1] if sim.net_np is net else None
    frames = sim.run(seconds, record=(env, 1))
    return export_frames(net, geo, frames, out)


def export_frames(net, geo, frames, out="scene"):
    d = os.path.dirname(os.path.abspath(out)) + os.sep
    name = os.path.basename(out)
    open(f"{out}.xodr", "w").write(to_opendrive(net, geo, name))
    scene = trajectories(net, geo, frames)
    json.dump(scene, open(f"{out}.json", "w"), separators=(",", ":"))
    open(f"{out}_replay.py", "w").write(REPLAY.format(name=name, here=d, blueprints=BLUEPRINT))
    return scene
