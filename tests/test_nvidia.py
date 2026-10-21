"""Export for NVIDIA tools: the OpenUSD stage opens and animates (when usd-core is installed), the control camera projects correctly."""
import math

import numpy as np
import pytest

from tests.test_osm import grid_osm
from xnet import carla_export as CE
from xnet import nvidia as NV
from xnet import osm
from xnet import sim as X


def scene():
    net, geo, _ = osm.import_osm(grid_osm())
    sim = X.Sim(net, n_envs=1, beh=X.Behaviour(p_av=0.5, demand=8.0), seed=1, reroute_s=0)
    sim.run(60)
    return net, geo, CE.trajectories(net, geo, sim.run(10, record=(0, 1)))


def test_usda_is_well_formed_and_opens_in_openusd(tmp_path):
    net, geo, sc = scene()
    p = NV.to_usda(net, geo, sc, str(tmp_path / "s.usda"), meta={"area": "grid"})
    text = open(p).read()
    assert text.startswith("#usda 1.0") and text.count("{") == text.count("}")
    assert text.count("xformOp:translate.timeSamples") == len(sc["vehicles"]) + sum(1 for q in sc["pedestrians"])
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(p)
    assert UsdGeom.GetStageUpAxis(st) == "Z" and st.GetTimeCodesPerSecond() == pytest.approx(1 / X.DT)
    agents = [q for q in st.Traverse() if q.GetParent().GetName() == "Agents"]
    v = sc["vehicles"][0]
    q = st.GetPrimAtPath(f"/World/Agents/{v['kind']}_{v['id']}")
    pos = UsdGeom.Xformable(q).ComputeLocalToWorldTransform(v["start"]).ExtractTranslation()
    assert len(agents) >= len(sc["vehicles"]) and abs(pos[0] - v["x"][0]) < 0.01 and abs(pos[1] - v["y"][0]) < 0.01
    assert pxr is not None


def test_camera_projection_and_clipping():
    cam = NV.Camera(width=640, height=360, fov=90.0, mount_height=1.6, forward=0.0)
    uv, z = cam.project(np.array([[20.0, 0.0, 1.6], [20.0, 5.0, 1.6], [-5.0, 0.0, 0.0]]), (0.0, 0.0, 0.0))
    assert np.allclose(uv[0], [320, 180]) and z[0] == pytest.approx(20.0)
    assert uv[1][0] < 320  # a point to the left lands left of centre
    assert cam.segments(np.array([[-10.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]), (0.0, 0.0, 0.0)) == []  # behind the camera: nothing
    seg = cam.segments(np.array([[-10.0, 0.0, 0.0], [30.0, 0.0, 0.0]]), (0.0, 0.0, math.pi * 0))
    assert len(seg) == 1 and seg[0][0][1] > 180  # clipped at the near plane, below the horizon


def test_control_video_writes_frames_and_camera(tmp_path):
    net, geo, sc = scene()
    ego = next(v for v in sc["vehicles"] if v["av"])
    r = NV.control_video(net, geo, sc, ego["id"], str(tmp_path / "c"))
    assert r["frames"] == len(ego["x"]) and (tmp_path / "c" / "camera.json").exists() and (tmp_path / "c" / "frame_0000.png").exists()
    assert len(r["camera"]["poses"]) == r["frames"]
