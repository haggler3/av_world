"""Junction detection: the awkward OpenStreetMap cases, against a naive degree-3 rule, and the cleaned map runs in the simulator."""
import json
import math
import time

import numpy as np

from xnet import junctions as JN
from xnet import osm
from xnet import sim as X
from xnet import synth_city as SC

LON0, LAT0 = -0.12, 51.5
KX, KY = 111320 * math.cos(math.radians(LAT0)), 110540


def local(ll):
    return [((a - LON0) * KX, (b - LAT0) * KY) for a, b in ll]


def test_detection_beats_the_degree_three_rule_on_every_awkward_case():
    for seed in range(3):
        nodes, ways, ntags, truth = SC.make(8, seed=seed)
        res = JN.detect(nodes, ways, node_tags=ntags)
        a = SC.score(local([(j["lon"], j["lat"]) for j in res["junctions"]]), truth)
        b = SC.score(local(JN.naive(nodes, ways)), truth)
        assert a["precision"] > 0.97 and a["recall"] > 0.97
        assert b["precision"] < 0.8  # divided roads, rings and overpasses fool the naive rule
        assert res["stats"]["roundabouts"] > 0 and res["stats"]["clusters_merged"] > 0


def test_roundabout_is_one_junction_and_overpass_is_none():
    # a ring with four streets, and a bridge over a street that shares a node with it
    nodes, ways = {}, []
    for k in range(8):
        t = 2 * math.pi * k / 8
        nodes[k] = (LON0 + 17 * math.cos(t) / KX, LAT0 + 17 * math.sin(t) / KY)
    ways.append((100, list(range(8)) + [0], {"highway": "residential", "junction": "roundabout", "oneway": "yes"}))
    for i, k in enumerate((0, 2, 4, 6)):
        nodes[10 + i] = (nodes[k][0] * 1 + (nodes[k][0] - LON0) * 4, nodes[k][1] + (nodes[k][1] - LAT0) * 4)
        ways.append((200 + i, [k, 10 + i], {"highway": "residential", "name": f"s{i}"}))
    res = JN.detect(nodes, ways)
    assert [j["kind"] for j in res["junctions"]] == ["roundabout"]
    assert len(res["junctions"][0]["legs"]) == 4
    # overpass sharing a node
    nodes = {1: (LON0, LAT0 - 0.001), 2: (LON0, LAT0), 3: (LON0, LAT0 + 0.001), 4: (LON0 - 0.001, LAT0), 5: (LON0 + 0.001, LAT0)}
    ways = [(1, [1, 2, 3], {"highway": "residential"}), (2, [4, 2, 5], {"highway": "residential", "bridge": "yes", "layer": "1"})]
    assert JN.detect(nodes, ways)["junctions"] == []
    assert len(JN.naive(nodes, ways)) == 1


def test_speed():
    nodes, ways, ntags, truth = SC.make(60, seed=1)  # ~10,000 nodes
    t = time.time()
    JN.detect(nodes, ways, node_tags=ntags)
    assert time.time() - t < 5.0


def overpass_json(nodes, ways, ntags):
    el = [{"type": "node", "id": i, "lon": ll[0], "lat": ll[1], **({"tags": ntags[i]} if i in ntags else {})} for i, ll in nodes.items()]
    el += [{"type": "way", "id": i, "nodes": ns, "tags": t} for i, ns, t in ways]
    return json.dumps({"elements": el})


def test_tagged_signals_decide_control_and_the_cleaned_city_runs():
    nodes, ways, ntags, truth = SC.make(5, seed=2)
    net, geo, info = osm.import_osm(overpass_json(nodes, ways, ntags))
    assert len(info["junctions"]) >= 10 and info["junction_stats"]["roundabouts"] >= 0
    tagged = info["junction_stats"]["signals_at_junctions"]
    assert abs(int(np.asarray(net["node_signal"]).sum()) - tagged) <= max(2, tagged)  # signals come from the tags, not from a guess
    sim = X.Sim(net, n_envs=2, seed=0, reroute_s=0)
    sim.run(60)
    assert float(np.asarray(sim.st.m)[:, X.METRICS.index("entered")].sum()) > 0
