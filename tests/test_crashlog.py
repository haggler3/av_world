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
