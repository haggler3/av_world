"""The open-data 3-D world: geometry helpers, vehicle parts, and a fused world (synthetic Overture-shaped layers) that lands in the USD stage."""
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely import wkb
from shapely.geometry import Point, box

from tests.test_nvidia import scene
from xnet import nvidia as NV
from xnet import scene3d as S3


def test_triangulate_and_walls_cover_the_polygon():
    sq = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    V, F = S3.triangulate(sq, 2.0)
    e1, e2 = V[F[:, 1], :2] - V[F[:, 0], :2], V[F[:, 2], :2] - V[F[:, 0], :2]
    area = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
    assert area == pytest.approx(96.0) and np.all(V[:, 2] == 2.0)
    V, F = S3.walls(sq, 0.0, 5.0)
    assert len(F) == 2 * (4 + 4) and V[:, 2].max() == 5.0


def test_vehicle_parts():
    car = S3.vehicle_parts("car", 4.5, 3)
    assert sum(p[0].startswith("wheel") for p in car) == 4 and car[0][0] == "body"
    assert any(p[0] == "lidar" for p in S3.vehicle_parts("av", 4.8, 0))
    assert S3.vehicle_parts("ped", 0.5, 0)[0][1] == "capsule"


def fake_overture(d, area, info, net, geo):
    lat0, lon0 = info["origin_lat_lon"]
    kx, ky = 111320.0 * math.cos(math.radians(lat0)), 110540.0
    ll = lambda g: wkb.dumps(__import__("shapely").transform(g, lambda c: np.c_[c[:, 0] / kx + lon0, c[:, 1] / ky + lat0]))
    xs = np.concatenate([p[:, 0] for p in geo.link_pts])
    ys = np.concatenate([p[:, 1] for p in geo.link_pts])
    # a block between the first two streets, and one footprint that overlaps the road (it must be trimmed)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    rows = [dict(id="b1", height=30.0, num_floors=None, min_height=None, facade_color="#aa8866", roof_color=None, geometry=ll(box(cx + 15, cy + 15, cx + 35, cy + 35))),
            dict(id="b2", height=None, num_floors=4, min_height=None, facade_color=None, roof_color="#333", geometry=ll(box(cx - 40, cy - 40, cx - 20, cy - 20))),
            dict(id="b3", height=None, num_floors=None, min_height=None, facade_color=None, roof_color=None, geometry=ll(box(xs.min() - 5, ys.min() - 5, xs.max() + 5, ys.min() + 5)))]
    pq.write_table(pa.Table.from_pylist(rows), f"{d}/{area}_buildings.parquet")
    land = [dict(id="t1", subtype="tree", **{"class": "tree"}, geometry=ll(Point(cx + 10, cy + 10))),
            dict(id="t2", subtype="tree", **{"class": "tree"}, geometry=ll(Point(*geo.link_pts[0][len(geo.link_pts[0]) // 2])))]  # on the road
    pq.write_table(pa.Table.from_pylist(land), f"{d}/{area}_land.parquet")
    pq.write_table(pa.Table.from_pylist([dict(id="l1", subtype="park", **{"class": "park"}, geometry=ll(box(cx - 40, cy + 15, cx - 20, cy + 35)))]),
                   f"{d}/{area}_land_use.parquet")


def test_world_fuses_layers_and_opens_in_usd(tmp_path):
    from tests.test_osm import grid_osm
    from xnet import osm
    net, geo, info = osm.import_osm(grid_osm())
    area = info["bbox_lon_lat"]
    fake_overture(str(tmp_path), "grid", info, net, geo)
    import xnet.scene3d as M
    orig = M._load
    M._load = lambda a, name, d=None: orig("grid", name, str(tmp_path))
    try:
        w = S3.build_world(area, net, geo, info)
    finally:
        M._load = orig
    st = w.stats
    assert st["buildings"] == 3 and st["buildings_with_mapped_height"] == 2 and st["trees"] == 1  # the tree on the road is dropped
    assert st["furniture"].get("traffic_signal", 0) + st["furniture"].get("stop_sign", 0) > 0
    V = np.concatenate([m.arrays()[0] for k, m in w.meshes.items() if k.startswith("roof")])
    assert np.isclose(V[:, 2], 4 * S3.FLOOR_M).any() and np.isclose(V[:, 2], S3.DEFAULT_H).any()
    for name in ("road_surface", "sidewalks", "markings_white", "grass", "ground"):
        assert w.meshes[name].n > 0, name
    _, _, sc = scene()
    p = NV.to_usda(net, geo, sc, str(tmp_path / "w.usda"), world=w)
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(p)
    assert stage.GetPrimAtPath("/World/Environment").IsValid() and stage.GetPrimAtPath("/World/Furniture").IsValid()
    meshes = [q for q in stage.Traverse() if q.IsA(UsdGeom.Mesh)]
    assert len(meshes) >= 6 and pxr is not None


def test_clip_picks_a_moving_av_and_renders_both_views(tmp_path):
    from tests.test_osm import grid_osm
    from xnet import osm
    net, geo, info = osm.import_osm(grid_osm())
    _, _, sc = scene()
    ego, start = NV.pick_clip(sc, 10)
    v = next(u for u in sc["vehicles"] if u["id"] == ego)
    assert v["av"] and v["start"] <= start
    td = NV.TopDown(net, geo, info, size=(160, 120))
    img = td.frame(sc, start, ego)
    td.close()
    assert img.shape == (120, 160, 3) and img.std() > 0
    pytest.importorskip("pyvista")
    w = S3.build_world(None, net, geo, info)  # no geographic layers: roads, markings, signals only
    assert w.stats["buildings"] == 0 and w.meshes["road_surface"].n > 0
    out = NV.compare_clip(net, geo, info, sc, w, None, str(tmp_path / "c.gif"), seconds=1.0, panel=(160, 120), t0=0.0)
    if out is None:
        pytest.skip("no offscreen OpenGL here")
    from PIL import Image
    g = Image.open(out)
    assert g.n_frames == 5 and g.size == (2 * 160 + 4, 120 + 30)
