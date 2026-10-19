import os

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("data/pgh_segments.parquet"), reason="needs python -m pgh.fetch")


@pytest.fixture(scope="module")
def oak():
    from xnet import network as NW

    return NW.build("oakland")


def test_network_is_consistent(oak):
    net, geo, info = oak
    L = len(net["link_len"])
    assert L > 400 and len(net["mov_len"]) > 800 and net["node_signal"].sum() > 10
    for m in range(len(net["mov_len"])):  # a move starts on its in-link and is listed there
        i, lane = net["mov_in"][m], net["mov_lane"][m]
        assert m in net["succ_mov"][i, lane]
        assert net["mov_out_lane"][m] < net["link_lanes"][net["mov_out"][m]]
    # every conflict a move is part of is listed on that move
    for c, (a, b) in enumerate(net["c_mov"]):
        assert c in net["mov_c"][a] and c in net["mov_c"][b]


def test_every_route_reaches_its_destination_or_an_exit(oak):
    net, geo, info = oak
    nh, sl, sm, dl = net["nh"], net["succ_link"], net["succ_mov"], net["dest_link"]
    for li in range(len(net["link_len"])):
        for d in range(len(dl)):
            cur, n = li, 0
            while cur != dl[d] and n < 300:
                k = nh[cur, d]
                if k < 0 or sl[cur, k] < 0 or sm[cur, :, k].max() < 0:
                    break
                cur, n = sl[cur, k], n + 1
            assert cur == dl[d] or net["link_exit"][cur], (li, d)


def test_clean_world_flows_with_few_crashes(oak):
    from xnet import sim as X

    net, geo, info = oak
    clean = X.Behaviour(p_red=0, p_nolook=0, p_nosignal=0, distract_rate=0, p_ped_violate=0)
    sim = X.Sim(net, n_slots=2048, n_envs=2, beh=clean, seed=1)
    sim.run(900)
    s = X.summary(sim.st.m)
    assert s["trips_finished"] > 1000 and s["mean_speed_kmh"] > 10
    assert s["teleports"] < 0.03 * s["trips_started"]
    assert s["crashes_per_M_junction_entries"] < 300  # residual model artefacts (see README); risky behaviour adds to this
    assert s["bus_and_delivery_stops"] > 0 and s["pedestrian_crossings"] > 1000 and s["lane_changes"] > 100
    alive = int(np.asarray(sim.st.alive).sum())
    assert alive <= s["trips_started"] - s["trips_finished"] - s["teleports"]


def test_distraction_causes_rear_ends(oak):
    from xnet import sim as X

    net, geo, info = oak
    clean = X.Behaviour(p_red=0, p_nolook=0, p_nosignal=0, distract_rate=0, p_ped_violate=0)
    a = X.Sim(net, n_envs=2, beh=clean, seed=2)
    b = X.Sim(net, n_envs=2, beh=clean._replace(distract_rate=1 / 60), seed=2)
    a.run(600)
    b.run(600)
    ra, rb = X.summary(a.st.m)["by_type"], X.summary(b.st.m)["by_type"]
    assert rb["rear-end"] > ra["rear-end"] + 5


def test_deterministic(oak):
    from xnet import sim as X

    net, geo, info = oak
    ms = []
    for _ in range(2):
        s = X.Sim(net, n_envs=2, seed=5, reroute_s=0)
        s.run(60)
        ms.append(np.asarray(jax.device_get(s.st.m)))
    assert np.array_equal(ms[0], ms[1])
