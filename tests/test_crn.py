import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xnet import sim as X


def test_hash_uniform_is_uniform_and_stream_independent():
    seed = jnp.uint32(12345)
    uid = jnp.arange(200_000, dtype=jnp.int32)
    a, b = np.asarray(X.hash_uniform(seed, uid, 0)), np.asarray(X.hash_uniform(seed, uid, 1))
    assert 0.0 <= a.min() and a.max() < 1.0
    assert abs(a.mean() - 0.5) < 0.005 and abs(a.std() - 0.2887) < 0.005
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.01                      # streams are independent
    assert abs(np.corrcoef(a[:-1], a[1:])[0, 1]) < 0.01             # sequential ids are not correlated
    c = np.asarray(X.hash_uniform(jnp.uint32(54321), uid, 0))
    assert abs(np.corrcoef(a, c)[0, 1]) < 0.01                      # another step's seed gives other numbers


def test_hash_uniform_depends_only_on_the_identity():
    seed = jnp.uint32(7)
    uid = jnp.array([5, 9, 11], jnp.int32)
    full = X.hash_uniform(seed, uid, 3)
    permuted = X.hash_uniform(seed, uid[::-1], 3)[::-1]
    assert np.array_equal(np.asarray(full), np.asarray(permuted))   # slot order is irrelevant


def test_hash_normal_is_standard_normal():
    z = np.asarray(X.hash_normal(jnp.uint32(1), jnp.arange(100_000, dtype=jnp.int32), 2))
    assert abs(z.mean()) < 0.02 and abs(z.std() - 1.0) < 0.02


def test_thinning_is_monotone():
    """A vehicle that rolls under a small rate also rolls under a larger one: u < p1 implies u < p2 for p1 <= p2."""
    u = X.hash_uniform(jnp.uint32(3), jnp.arange(50_000, dtype=jnp.int32), 1)
    low, high = np.asarray(u < 0.01), np.asarray(u < 0.02)
    assert (low & ~high).sum() == 0 and 0.8 < high.sum() / max(low.sum(), 1) / 2 < 1.2


needs_data = pytest.mark.skipif(not os.path.exists("data/pgh_segments.parquet"), reason="needs python -m pgh.fetch")


@pytest.fixture(scope="module")
def oak():
    from xnet import network as NW

    return NW.build("oakland")[0]


@needs_data
def test_crn_off_is_the_old_simulator(oak):
    """The stored default (crn = 0) must keep every earlier result: same seeds, same counts as before the change."""
    sim = X.Sim(oak, n_slots=512, n_envs=4, seed=0, reroute_s=0)
    sim.run(20)
    s = X.summary(sim.st.m)
    assert (s["junction_entries"], s["trips_started"], s["pedestrian_crossings"], s["blocked_arrivals"]) == (42, 136, 501, 6)


@needs_data
def test_uids_are_unique_among_live_vehicles(oak):
    sim = X.Sim(oak, n_slots=512, n_envs=2, seed=1, reroute_s=0, beh=X.Behaviour(crn=1.0))
    sim.run(60)
    for e in range(2):
        alive = np.asarray(sim.st.alive[e])
        uid = np.asarray(sim.st.uid[e])[alive]
        assert len(uid) > 20 and len(np.unique(uid)) == len(uid)


@needs_data
def test_distracted_set_is_nested_when_the_rate_rises(oak):
    """With CRN a driver who is distracted at rate r1 is, nearly always, also distracted at r2 > r1 (the contrast with no CRN,
    which decays towards chance as the worlds drift apart, is measured by avlab.crn_decay)."""
    def ever_distracted(rate):
        sim = X.Sim(oak, n_slots=512, n_envs=2, seed=3, reroute_s=0, beh=X.Behaviour(crn=1.0, distract_rate=rate))
        seen = [set(), set()]
        for _ in range(120):
            sim.run(1.0)
            d, u = np.asarray(sim.st.dist_t > 0), np.asarray(sim.st.uid)
            for e in range(2):
                seen[e] |= set(u[e][d[e]].tolist())
        return seen

    lo, hi = ever_distracted(1 / 60), ever_distracted(1 / 20)
    n_lo = sum(len(s) for s in lo)
    assert n_lo > 20
    assert sum(len(a & b) for a, b in zip(lo, hi)) / n_lo > 0.85   # nested up to the drift a distracted car causes
