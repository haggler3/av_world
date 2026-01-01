"""Argoverse 2 / Waymo import: a made-up crossroads in both formats, and (when a real Argoverse 2 download is given) real scenarios."""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

from xnet import avdata as AV
from xnet import sim as X

# four one-way approaches / exits and the paths through the junction; lane id -> (points, predecessors, successors, is_intersection)
H = 1.75
LANES = {
    1: ([(-70, -H), (-8, -H)], [], [2], False), 2: ([(-8, -H), (8, -H)], [1], [3], True), 3: ([(8, -H), (70, -H)], [2], [], False),
    4: ([(70, H), (8, H)], [], [5], False), 5: ([(8, H), (-8, H)], [4], [6], True), 6: ([(-8, H), (-70, H)], [5], [], False),
    7: ([(H, -70), (H, -8)], [], [8], False), 8: ([(H, -8), (H, 8)], [7], [9], True), 9: ([(H, 8), (H, 70)], [8], [], False),
    10: ([(-H, 70), (-H, 8)], [], [11], False), 11: ([(-H, 8), (-H, -8)], [10], [12], True), 12: ([(-H, -8), (-H, -70)], [11], [], False),
}


def av2_dir(tmp_path):
    sid = "synthetic-0001"
    d = tmp_path / sid
    d.mkdir()
    lane_segments = {}
    for i, (pts, pre, suc, inter) in LANES.items():
        lane_segments[str(i)] = {"id": i, "is_intersection": inter, "lane_type": "VEHICLE", "predecessors": pre, "successors": suc,
                                 "left_neighbor_id": None, "right_neighbor_id": None,
                                 "centerline": [{"x": float(x), "y": float(y), "z": 0.0} for x, y in np.linspace(pts[0], pts[1], 12)]}
    json.dump({"lane_segments": lane_segments, "drivable_areas": {}, "pedestrian_crossings": {}}, open(d / f"log_map_archive_{sid}.json", "w"))
    rows = []
    for T in range(110):
        t = T * 0.1
        rows.append(("1", "vehicle", 0, T, -60 + 8.0 * t, -H, 0.0, 8.0, 0.0))      # eastbound, 8 m/s
        rows.append(("AV", "vehicle", 4, T, H, -60 + 6.0 * t, np.pi / 2, 0.0, 6.0))  # northbound, 6 m/s (the ego)
        rows.append(("2", "pedestrian", 2, T, 20.0, 5.0 - 0.1 * t, -np.pi / 2, 0.0, -0.1))
        rows.append(("3", "static", 1, T, 30.0, -4.0, 0.0, 0.0, 0.0))
    df = pd.DataFrame(rows, columns=["track_id", "object_type", "object_category", "timestep", "position_x", "position_y", "heading",
                                     "velocity_x", "velocity_y"])
    df["observed"] = df.timestep < 50
    df["scenario_id"], df["start_timestamp"], df["end_timestamp"], df["num_timestamps"] = sid, 0.0, 11.0, 110
    df["focal_track_id"], df["city"] = "1", "synthetic"
    df.to_parquet(d / f"scenario_{sid}.parquet")
    return str(d)


def test_av2_scene_junction_and_network(tmp_path):
    sc = AV.read_av2(av2_dir(tmp_path))
    scene = AV.av2_to_scene(sc, every=2)
    assert scene["frames"] == 55 and abs(scene["dt"] - 0.2) < 1e-9
    kinds = sorted(v["kind"] for v in scene["vehicles"])
    assert kinds == ["av", "car"] and len(scene["pedestrians"]) == 1  # the static object is dropped
    jc, n = AV.av2_junctions(sc["map"])
    assert n == 1 and set(jc) == {2, 5, 8, 11}  # four paths through the crossroads are one junction
    net, geo, info = AV.av2_network(sc)
    assert info["junction_clusters"] == 1 and net["mov_len"].shape[0] >= 4
    assert (np.asarray(net["link_len"]) > 20).all()


def test_log_seeded_simulation_runs_and_follows_the_log(tmp_path):
    sc = AV.read_av2(av2_dir(tmp_path))
    scene = AV.av2_to_scene(sc, every=2)
    net, geo, _ = AV.av2_network(sc)
    m = AV.Matcher(net, geo)
    st, info = AV.seed_state(net, geo, scene, frame=10, matcher=m, speed_from_log=True)
    assert info["matched"] == 2 and info["median_offset_m"] < 1.0
    assert int(np.asarray(st.alive).sum()) == 2 and bool(np.asarray(st.av).sum() == 1)
    r = AV.compare_future(net, geo, scene, frame=10, horizon_s=3.0, matcher=m, speed_from_log=True)
    assert r["n"][1] == 2
    assert r["sim_error_m"][2] < 5.0  # free-flowing cars at constant speed: the simulator stays close to the log


def waymo_dict():
    tracks = [{"id": 5, "type": 1, "states": [[-60 + 0.8 * i, -H, 4.5, 1.9, 0.0, 8.0, 0.0, True] for i in range(91)]},
              {"id": 6, "type": 1, "states": [[H, -60 + 0.6 * i, 4.5, 1.9, np.pi / 2, 0.0, 6.0, True] for i in range(91)]},
              {"id": 7, "type": 2, "states": [[20.0, 5.0 - 0.01 * i, 0.6, 0.6, 0.0, 0.0, -0.1, i > 10] for i in range(91)]}]
    lanes = {i: {"type": 2, "speed_limit_mph": 25.0, "polyline": [list(p) for p in np.linspace(pts[0], pts[1], 12)], "entry": pre, "exit": suc}
             for i, (pts, pre, suc, _) in LANES.items()}
    return {"id": "w1", "dt": 0.1, "current": 10, "sdc": 1, "tracks": tracks, "lanes": lanes, "crosswalks": [], "stop_signs": [], "signals": []}


def test_waymo_dict_scene_and_map():
    d = waymo_dict()
    scene = AV.waymo_scene(d, every=2)
    assert sorted(v["kind"] for v in scene["vehicles"]) == ["av", "car"] and len(scene["pedestrians"]) == 1
    assert scene["pedestrians"][0]["start"] == 6  # the pedestrian is only valid from step 11
    inter = AV.infer_intersection_lanes(d["lanes"])
    assert inter == {2, 5, 8, 11}  # the four paths that cross inside the junction
    net, geo, info = AV.waymo_network(d)
    assert info["junction_clusters"] == 1 and net["mov_len"].shape[0] >= 4


@pytest.mark.skipif(not os.environ.get("AV2_DIR"), reason="set AV2_DIR to a folder of downloaded Argoverse 2 scenarios")
def test_real_argoverse2_scenarios():
    dirs = sorted(glob.glob(os.path.join(os.environ["AV2_DIR"], "*")))[:5]
    for d in dirs:
        sc = AV.read_av2(d)
        scene = AV.av2_to_scene(sc, every=2)
        assert scene["vehicles"]
        net, geo, info = AV.av2_network(sc)
        assert net["link_len"].shape[0] > 0 and info["junction_clusters"] >= 0
        r = AV.compare_future(net, geo, scene, 25, 3.0)
        assert r["match"]["logged"] > 0
