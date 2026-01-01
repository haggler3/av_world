"""Software / hardware in the loop: an external controller drives the ego AVs over TCP, in lockstep and in real time with deadlines."""
import threading

import numpy as np

from av_world import sil
from av_world import sil_client as C
from tests.test_osm import grid_osm
from xnet import osm


def city():
    return osm.import_osm(grid_osm())[0]


def drive(session, policy, delay_s=0.0):
    out = {}
    t = threading.Thread(target=lambda: out.setdefault("client", C.serve(C.connect(port=session.port), policy, delay_s)), daemon=True)
    t.start()
    out["sim"] = session.run(seconds=20.0, warm_s=40.0, accept_timeout=30)
    t.join(10)
    return out


def test_lockstep_every_step_is_answered():
    s = sil.Session(city(), egos=2, mode="lockstep", beh=dict(p_av=0.5, demand=8.0))
    r = drive(s, C.example)
    assert r["sim"]["steps"] == 100 and r["sim"]["replies"] == 100 and r["sim"]["misses"] == 0
    assert r["client"]["steps"] == 100  # the controller got the summary
    assert r["sim"]["ego_distance_m"] > 20  # the controller's commands moved the egos


def test_the_controller_really_drives_the_ego():
    s = sil.Session(city(), egos=1, mode="lockstep", shield="off", beh=dict(p_av=0.5, demand=8.0))
    r = drive(s, C.brake)
    v = np.asarray(s.state.v)
    assert r["sim"]["misses"] == 0
    assert all(v[e] < 0.5 for e in s.ego)  # full brake for 20 s: the ego stands


def test_realtime_deadline_misses_fall_back():
    s = sil.Session(city(), egos=1, mode="realtime", deadline_ms=30.0, speed=2.0, fallback="brake", beh=dict(p_av=0.5, demand=8.0))
    r = drive(s, C.example, delay_s=0.08)  # slower than the deadline
    assert r["sim"]["misses"] > 50 and r["sim"]["late_discarded"] > 0
    assert r["sim"]["real_time_factor"] < 2.1  # never faster than 2x wall clock
    if r["sim"]["realtime_overruns"] < 10:
        assert r["sim"]["real_time_factor"] > 1.6  # and close to it when it keeps up
