"""Per-crash event log (2026-11-19): logging does not change the dynamics, its events match the simulator's crash
counters, its exposure matches veh_s, and the event grouping."""
import jax
import jax.numpy as jnp
import numpy as np

from avlab import crashlog as CL
from xnet import network as NW
from xnet import sim as X


def test_log_is_passive_and_matches_counters():
    net, geo, _ = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    J["link_patch"] = jnp.asarray(X.patch_links(net, 0.5, seed=0))
    behs = [X.Behaviour(friction=0.3), X.Behaviour(friction=1.0, patch_friction=0.3)]
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    st0 = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(4), 2), J, 512)
    lg = CL.Logger(J, nh, B, cap=128, chunk_s=60.0)
    st1, buf = lg.run(st0, 240.0)
    st2 = jax.lax.scan(lambda s, _: (jax.vmap(lambda si, b: X.step(si, J, b, nh))(s, B), None), st0, None,
                       length=int(round(240.0 / X.DT)))[0]
    for a, b in zip(jax.tree.leaves(st1), jax.tree.leaves(st2)):
        assert np.array_equal(np.asarray(a), np.asarray(b))
    m = np.asarray(st1.m)
    counters = sum(m[:, X.METRICS.index(k)] for k in X.CRASH_NAMES).astype(int)
    rows = CL.table(buf)
    ev = CL.events(rows, CL.positions(net, geo, rows))
    assert counters.sum() > 0
    assert np.array_equal(np.bincount(ev["copy"].astype(int), minlength=2), counters)
    assert np.allclose(np.asarray(buf.expo).sum(1), m[:, X.METRICS.index("veh_s")], rtol=1e-4)
    assert set(np.unique(rows["role"])) <= set(range(len(CL.ROLES)))


def test_events_grouping():
    rows = {"copy": np.array([0, 0, 0, 1]), "t": np.array([1.0, 1.0, 1.0, 1.0]), "link": np.array([3, 4, 9, 3]),
            "patched": np.array([0, 1, 0, 0]), "v": np.array([5.0, 2.0, 7.0, 1.0]), "av": np.array([0, 0, 1, 0]),
            "role": np.array([1, 0, 2, 4])}
    xy = (np.array([0.0, 10.0, 100.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0]))
    ev = CL.events(rows, xy)
    assert len(ev["copy"]) == 3  # rows 0 and 1 are 10 m apart (one event); row 2 is far; row 3 is another copy
    i = int(np.argmax(ev["size"]))
    assert ev["size"][i] == 2 and ev["patched"][i] == 1 and ev["max_v"][i] == 5.0
    assert len(CL.events(rows)["copy"]) == 4  # without positions: by link


def test_quick_returns_tables():
    crashes, exposure = CL.quick("oakland", copies=1, minutes=1.0, warm_minutes=1.0, friction=0.25)
    assert list(crashes.columns) == ["copy", "time_s", "x", "y", "link", "vehicles", "impact_kmh", "av_involved", "kind"]
    assert exposure.vehicle_hours.sum() > 0 and (crashes.time_s >= 0).all() and set(crashes.kind) <= set(CL.ROLES)


def test_rear_partners_are_crash_rows():
    """2026-11-25: every rear-end striker names its leader, and that leader is a new wreck of the same copy and step (or was
    already a wreck); partner AV flags match the partner's own row."""
    net, geo, _ = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *[X.Behaviour(friction=0.25, p_av=0.5)] * 2)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(7), 2), J, 512)
    st, buf = CL.Logger(J, nh, B, cap=256, chunk_s=60.0).run(st, 240.0)
    r = CL.table(buf)
    rear = np.where(r["role"] == 1)[0]
    assert len(rear) > 0 and (r["partner"][rear] >= 0).all()
    for i in rear:
        same = (r["copy"] == r["copy"][i]) & (np.round(r["t"] * 10) == np.round(r["t"][i] * 10)) & (r["slot"] == r["partner"][i])
        assert same.sum() == 1 or r["p_wreck"][i] == 1
        if same.sum() == 1:
            assert r["av"][same][0] == r["p_av"][i] and r["kind"][same][0] == r["p_kind"][i]
    assert (r["partner"][r["role"] == 4] == -2).all()


def test_pair_rest_and_rear_pairs():
    rows = {"copy": np.array([0, 0, 0, 0]), "t": np.array([1.0, 1.0, 1.0, 2.0]), "slot": np.array([5, 6, 7, 8]),
            "v": np.array([5.0, 2.0, 7.0, 1.0]), "av": np.array([1, 0, 1, 0]), "kind": np.array([0, 0, 3, 0]),
            "role": np.array([1, 0, 2, 3]), "partner": np.array([6, 5, -1, -1]), "p_av": np.array([0, 1, -1, -1]),
            "p_kind": np.array([0, 0, -1, -1]), "p_v": np.array([2.0, 5.0, -1.0, -1.0]), "p_wreck": np.array([0, 0, 0, 0])}
    xy = (np.array([0.0, 4.0, 10.0, 0.0]), np.zeros(4))
    out = CL.pair_rest(rows, xy)
    assert out["partner"][2] == 6 and out["paired_by"][2] == 1 and out["p_kind"][2] == 0  # nearest same-step row
    assert out["partner"][3] == -1 and out["paired_by"][3] == -1  # alone in its step
    rp = CL.rear_pairs(rows)
    assert rp["striker_av"].tolist() == [1] and rp["struck_av"].tolist() == [0] and rp["struck_v"].tolist() == [2.0]
