import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.path.exists("data/pgh_segments.parquet"), reason="needs python -m pgh.fetch")


@pytest.fixture(scope="module")
def oak():
    from xnet import network as NW

    return NW.build("oakland")


def counts(sim):
    from xnet import sim as X

    return dict(zip(X.METRICS, np.asarray(sim.st.m).sum(0)))


def test_cyclists_ride_and_get_overtaken_without_gridlock(oak):
    from xnet import sim as X

    net, geo, info = oak
    sim = X.Sim(net, n_envs=2, beh=X.Behaviour(p_bike=0.1), seed=2)
    sim.run(400)
    d = counts(sim)
    st = jax.tree.map(np.asarray, sim.st)
    bikes = st.alive & (st.kind == X.BIKE)
    assert d["bikes"] > 50 and bikes.sum() > 10
    assert st.v[bikes].max() <= 5.5 + 0.3  # cyclists keep to ~20 km/h
    assert d["passes"] > 20 and d["teleports"] < 0.02 * d["spawned"]


def test_barrier_function_bounds():
    from xnet import sim as X

    # far from an obstacle: a positive bound; inside the barrier (gap below the margin): braking
    a_far, h_far = X.cbf_bound(jnp.array(80.0), jnp.array(10.0), jnp.array(10.0), jnp.array(0.0))
    a_in, h_in = X.cbf_bound(jnp.array(3.0), jnp.array(10.0), jnp.array(0.0), jnp.array(0.0))
    assert float(h_far) > 0 and float(a_far) > 0 and float(h_in) < 0 and float(a_in) < -3
    assert float(X.cbf_bound(jnp.array(1e9), jnp.array(10.0), jnp.array(0.0), jnp.array(0.0))[0]) == 99.0  # no obstacle


def test_barrier_function_keeps_a_reckless_policy_from_crashing(oak):
    from xnet import sim as X

    net, geo, info = oak
    reckless = lambda obs, key: (jnp.tile(jnp.array([[1.0, -2.0]]), (obs.shape[0], 1)), jnp.zeros((obs.shape[0], 2)))  # noqa: E731
    beh = X.Behaviour(p_av=0.3, p_red=0, p_nolook=0, p_nosignal=0, distract_rate=0, p_ped_violate=0, p_close_pass=0)
    out = {}
    for mode in ("off", "cbf"):
        X.SHIELD = mode
        try:
            sim = X.Sim(net, n_envs=2, beh=beh, seed=4, policy=reckless)
            sim.run(240)
        finally:
            X.SHIELD = "idm"
        out[mode] = counts(sim)
    assert out["off"]["av_crash"] > 20  # flat-out, accepting every gap: crashes of several kinds
    assert sum(out["off"][k] > 0 for k in X.AV_CRASH) >= 2
    assert out["cbf"]["av_crash"] <= 2 and out["cbf"]["av_veto_junction"] > 0
