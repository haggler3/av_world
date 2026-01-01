"""Replay a scene exported by xnet.carla_export in CARLA (0.9.13+). Usage: python cutin_0_replay.py --host localhost --port 2000"""
import argparse
import json
import math
import time

import carla

HERE = '/home/user/av_world/results/tails/scenes/'


def tf(x, y, yaw, z=0.3):
    """OpenDRIVE (x east, y north, counter-clockwise heading) -> CARLA/Unreal (y flipped, yaw clockwise in degrees)."""
    return carla.Transform(carla.Location(x=x, y=-y, z=z), carla.Rotation(yaw=-math.degrees(yaw)))


def replay(client, scene, xodr, speed=1.0, load_map=True):
    world = client.generate_opendrive_world(xodr) if load_map else client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode, settings.fixed_delta_seconds = True, scene["dt"] / speed
    world.apply_settings(settings)
    lib, actors = world.get_blueprint_library(), {}
    bp = {'car': 'vehicle.audi.a2', 'av': 'vehicle.lincoln.mkz_2020', 'bus': 'vehicle.mitsubishi.fusorosa', 'van': 'vehicle.volkswagen.t2', 'bike': 'vehicle.diamondback.century'}
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
    scene = json.load(open(HERE + "cutin_0.json"))
    xodr = open(HERE + "cutin_0.xodr").read()
    c = carla.Client(a.host, a.port)
    c.set_timeout(30.0)
    replay(c, scene, xodr, a.speed, not a.keep_map)
