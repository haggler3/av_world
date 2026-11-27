"""avlab.failurelog / failure_export: crash windows are cut right, and every export format reads back."""
import json
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from avlab import failure_export as FX
from avlab import failurelog as F


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("fl"))
    F.collect(["los_angeles"], ["dry"], root=root, E=4, warm_s=40.0, seconds=200.0, log=lambda *a: None, cap=3)
    ps = sorted(p for p in os.listdir(f"{root}/dry/los_angeles") if p.endswith(".json.gz"))
    assert ps, "no crash in 4 copies x 200 s of downtown Los Angeles"
    return [F.load(f"{root}/dry/los_angeles/{p}") for p in ps], root


def test_interpolation_doubles_the_rate_and_keeps_validity():
    xs = np.array([0.0, 1.0, 2.0, 0.0])
    ok = np.array([True, True, True, False])
    x, y, h, v, valid = F._interp_track(xs, xs, np.zeros(4), xs, ok, 7, 2)
    assert np.allclose(x[:5], [0, 0.5, 1, 1.5, 2]) and valid.tolist() == [True] * 5 + [False] * 2


def test_crash_window_and_involved_tracks(scenarios):
    for sc in scenarios[0]:
        assert sc["schema"] == F.SCHEMA and sc["n_steps"] == 111 and sc["event_index"] == 80
        k, ev = sc["event_index"], sc["event"]
        inv = [sc["tracks"][i] for i in ev["involved_tracks"]]
        assert inv and all(t["valid"][k] for t in inv)
        assert all(np.hypot(t["x"][k] - ev["x"], t["y"][k] - ev["y"]) < 15 for t in inv)
        assert sc["tracks"][ev["ego_track"]]["is_ego"] and (sc["tracks"][ev["ego_track"]]["is_av"] or not ev["av_involved"])
        for t in sc["tracks"]:
            assert len(t["x"]) == len(t["valid"]) == sc["n_steps"]
        assert sc["map"]["lanes"] and all(len(ln["left"]) == len(ln["right"]) == len(ln["polyline"]) for ln in sc["map"]["lanes"])


def test_every_format_reads_back(scenarios, tmp_path):
    from xnet import avdata as AV
    sc = scenarios[0][0]
    out = str(tmp_path)
    FX.export(sc, out)
    sid = sc["id"]
    a = AV.read_av2(f"{out}/av2/{sid}")
    assert AV.av2_to_scene(a)["frames"] == 110
    w = json.load(open(f"{out}/waymo/{sid}.waymo.json"))
    assert len(w["tracks"][0]["states"]) == 91 and w["currentTimeIndex"] == 10
    assert AV.waymo_scene(FX.waymo_json_to_dict(w))["frames"] == 91
    g = json.load(open(f"{out}/gpudrive/{sid}.gpudrive.json"))
    assert len(g["objects"][0]["position"]) == 91 and any(r["type"] == "road_edge" for r in g["roads"])
    sn = pickle.load(open(f"{out}/scenarionet/{sid}.scenarionet.pkl", "rb"))
    assert sn["length"] == 111 and sn["metadata"]["sdc_id"] in sn["tracks"]
    ET.parse(f"{out}/openscenario/{sid}.xosc")
    ET.parse(f"{out}/maps/los_angeles.xodr")
    assert np.load(f"{out}/bev/{sid}.bev.npz")["bev"].shape[1:] == (8, 128, 128)
    assert json.load(open(f"{out}/carla/{sid}.carla.json"))["frames"] == 111
    try:
        from commonroad.common.file_reader import CommonRoadFileReader
    except Exception:  # not installed, or a protobuf version it cannot load
        ET.parse(f"{out}/commonroad/{sid}.cr.xml")
    else:
        scn, pps = CommonRoadFileReader(f"{out}/commonroad/{sid}.cr.xml").open()
        assert len(scn.lanelet_network.lanelets) == len(sc["map"]["lanes"]) and len(pps.planning_problem_dict) == 1


def test_tfrecord_framing():
    assert FX.crc32c(b"123456789") == 0xE3069283
    import tempfile
    p = os.path.join(tempfile.mkdtemp(), "x.tfrecord")
    FX.write_tfrecord(p, [b"abc", bytes(range(256)) * 9])
    assert FX.read_tfrecord(p) == [b"abc", bytes(range(256)) * 9]
