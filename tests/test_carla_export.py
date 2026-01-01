"""CARLA export: the OpenDRIVE map is well formed and connected, the trajectories are complete, the replay drives a stand-in API."""
import json
import math
import sys
import types
import xml.etree.ElementTree as ET

import numpy as np

from xnet import carla_export as CE
from xnet import network as NW
from xnet import sim as X
from tests.test_osm import grid_osm
from xnet import osm


def city():
    return osm.import_osm(grid_osm())


def test_opendrive_is_well_formed_and_connected():
    net, geo, _ = city()
    root = ET.fromstring(CE.to_opendrive(net, geo))
    roads = {r.get("id"): r for r in root.iter("road")}
    assert len(roads) > 20
    for r in roads.values():
        segs = list(r.iter("geometry"))
        assert segs and float(r.get("length")) > 0
        assert abs(sum(float(g.get("length")) for g in segs) - float(r.get("length"))) < 0.05
        for a, b in zip(segs[:-1], segs[1:]):  # pieces follow each other (the polyline is continuous)
            x = float(a.get("x")) + math.cos(float(a.get("hdg"))) * float(a.get("length"))
            y = float(a.get("y")) + math.sin(float(a.get("hdg"))) * float(a.get("length"))
            assert math.hypot(x - float(b.get("x")), y - float(b.get("y"))) < 0.05
    juncs = list(root.iter("junction"))
    assert juncs
    for j in juncs:
        for c in j.iter("connection"):
            assert c.get("incomingRoad") in roads and c.get("connectingRoad") in roads
            assert roads[c.get("connectingRoad")].get("junction") == j.get("id")
    # every junction road starts where its incoming road ends, within a lane width
    L = len(net["link_len"])
    for j in juncs:
        for c in j.iter("connection"):
            inc, con = roads[c.get("incomingRoad")], roads[c.get("connectingRoad")]
            g = list(inc.iter("geometry"))[-1]
            end = (float(g.get("x")) + math.cos(float(g.get("hdg"))) * float(g.get("length")),
                   float(g.get("y")) + math.sin(float(g.get("hdg"))) * float(g.get("length")))
            s0 = list(con.iter("geometry"))[0]
            assert math.hypot(end[0] - float(s0.get("x")), end[1] - float(s0.get("y"))) < 2 * NW.W + 1


def test_trajectories_and_replay(tmp_path):
    net, geo, _ = city()
    sim = X.Sim(net, n_envs=1, seed=1, reroute_s=0, beh=X.Behaviour(p_av=0.5, demand=8.0))
    sim.run(90)
    scene = CE.export(sim, 0, 30, str(tmp_path / "s"), net=net, geo=geo)
    assert scene["frames"] == 150 and scene["vehicles"]
    assert any(v["av"] for v in scene["vehicles"])
    for v in scene["vehicles"]:
        assert len(v["x"]) == len(v["y"]) == len(v["yaw"]) == len(v["v"])
        assert np.isfinite(v["x"]).all() and max(v["v"]) < 40
    assert json.load(open(tmp_path / "s.json"))["dt"] == X.DT
    # stand-in for the carla module: the replay spawns kinematic actors and ticks once per frame
    calls = {"tick": 0, "spawn": 0, "transform": 0}

    class Actor:
        def set_simulate_physics(self, on):
            assert on is False

        def set_transform(self, t):
            calls["transform"] += 1

        def destroy(self):
            pass

    class World:
        def get_settings(self):
            return types.SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)

        def apply_settings(self, s):
            pass

        def get_blueprint_library(self):
            return types.SimpleNamespace(find=lambda n: n)

        def try_spawn_actor(self, bp, t):
            calls["spawn"] += 1
            return Actor()

        def tick(self):
            calls["tick"] += 1

    fake = types.ModuleType("carla")
    fake.Transform = lambda l, r: (l, r)
    fake.Location = lambda **k: k
    fake.Rotation = lambda **k: k
    sys.modules["carla"] = fake
    try:
        ns = {"__name__": "replay"}
        exec(compile(open(tmp_path / "s_replay.py").read(), "replay", "exec"), ns)
        client = types.SimpleNamespace(generate_opendrive_world=lambda x: World(), get_world=lambda: World())
        ns["replay"](client, scene, open(tmp_path / "s.xodr").read())
    finally:
        del sys.modules["carla"]
    assert calls["tick"] == scene["frames"] and calls["spawn"] >= len(scene["vehicles"]) and calls["transform"] > 0
