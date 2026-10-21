"""OpenStreetMap import: a made-up 4 x 4 street grid becomes a runnable network."""
import jax.numpy as jnp
import numpy as np

from xnet import network as NW
from xnet import osm
from xnet import sim as X


def grid_osm(n=4, step=0.0016, lat0=51.5, lon0=-0.12):
    nodes, ways, k = [], [], 1
    for i in range(n):
        for j in range(n):
            nodes.append(f'<node id="{i * n + j + 1}" lat="{lat0 + i * step * 0.7:.6f}" lon="{lon0 + j * step:.6f}"/>')
    for i in range(n):  # east-west streets: the first is one-way
        refs = "".join(f'<nd ref="{i * n + j + 1}"/>' for j in range(n))
        ow = '<tag k="oneway" v="yes"/>' if i == 0 else ""
        ways.append(f'<way id="{100 + i}">{refs}<tag k="highway" v="{"secondary" if i == 1 else "residential"}"/>{ow}'
                    f'<tag k="maxspeed" v="30 mph"/></way>')
    for j in range(n):
        refs = "".join(f'<nd ref="{i * n + j + 1}"/>' for i in range(n))
        ways.append(f'<way id="{200 + j}">{refs}<tag k="highway" v="residential"/></way>')
    ways.append('<way id="300"><nd ref="1"/><nd ref="2"/><tag k="highway" v="footway"/></way>')  # not drivable: ignored
    return "<osm version='0.6'>" + "".join(nodes) + "".join(ways) + "</osm>"


def test_import_builds_and_runs():
    before = (osm.mine.LAT0, osm.mine.LON0)
    net, geo, info = osm.import_osm(grid_osm())
    assert (osm.mine.LAT0, osm.mine.LON0) == before  # the Pittsburgh grid is restored
    assert abs(info["origin_lat_lon"][0] - 51.5) < 0.01
    L = net["link_len"].shape[0]
    assert L >= 20 and net["mov_len"].shape[0] > L
    assert np.isfinite(net["link_len"]).all() and net["link_len"].min() > 0
    sim = X.Sim(net, n_envs=2, seed=0, reroute_s=0)
    sim.run(120)
    m = dict(zip(X.METRICS, np.asarray(sim.st.m).sum(0)))
    assert m["entered"] > 0 and m["veh_s"] > 0


def test_oneway_and_speed():
    nodes, ways = osm.parse(grid_osm())
    rows = osm.rows_from_osm(nodes, ways) if osm.set_origin(51.5, -0.12) is None else []
    assert len(rows) == 8  # the footway is dropped
    assert rows[0]["access_restrictions"][0]["when"]["heading"] == "backward"
    assert rows[0]["speed_limits"][0]["max_speed"] == {"value": 30, "unit": "mph"}
    osm.set_origin(40.44, -79.99)


def test_save_load_roundtrip(tmp_path):
    net, _, _ = osm.import_osm(grid_osm())
    osm.save(net, tmp_path / "c.npz")
    back = osm.load(tmp_path / "c.npz")
    assert set(back) == set(net) and np.allclose(back["link_len"], net["link_len"])
    assert jnp.asarray(back["nh"]).shape == jnp.asarray(net["nh"]).shape
