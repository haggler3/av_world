"""2026-11-14: friction-aware AV policy (avlab.frictionaware) and the three-stream knee pool (avlab.kneepool)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_scales():
    from avlab.frictionaware import scales
    assert scales("blind", 0.4) == (1.0, 1.0)
    assert scales("headway", 0.4) == pytest.approx((2.5, 1.0))
    assert scales("slow", 0.25) == pytest.approx((4.0, 0.5))
    # on a dry road every policy is the blind one
    assert scales("slow", 1.0) == scales("blind", 1.0)


def test_aware_idm_reduces_to_example_policy():
    jnp = pytest.importorskip("jax.numpy")
    from avlab.frictionaware import aware_idm
    from examples.my_policy import careful_idm
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.uniform(0, 1.5, (64, 28)), jnp.float32)
    u1, _ = aware_idm(obs)
    u0, _ = careful_idm(obs, None)
    np.testing.assert_allclose(np.asarray(u1), np.asarray(u0), atol=1e-6)
    # a longer headway or a lower desired speed never asks for more acceleration
    for ts, vs in ((2.5, 1.0), (2.5, 0.63)):
        u2, _ = aware_idm(obs, ts, vs)
        assert np.all(np.asarray(u2[:, 0]) <= np.asarray(u0[:, 0]) + 1e-6)


def test_heterogeneity_on_synthetic_hinge():
    from avlab.kneepool import heterogeneity
    from avlab import knee as K
    rng = np.random.default_rng(1)
    mu = np.repeat([1.0, 0.85, 0.7, 0.6, 0.5, 0.4], 60).astype(float)
    stream = np.tile(np.repeat([0.0, 1.0, 3.0], 20), 6)
    run = stream.copy()
    t = np.full(len(mu), 10.0)
    lam = 0.01 * mu ** -1.0 * np.exp(3.0 * np.maximum(np.log(0.7) - np.log(mu), 0.0))
    y = rng.poisson(lam * t * 100).astype(float)
    f = K.fixed_hinge(y, t, mu, run, 0.7)
    assert 1.5 < f["extra"] < 4.5 and f["p_one_sided"] < 0.01
    h = heterogeneity(y, t, mu, run, stream)
    assert h["df"] == 2 and h["p"] > 0.01
    # a stream with no knee at all is detected
    y2 = y.copy(); s3 = stream == 3
    y2[s3] = rng.poisson(0.01 * mu[s3] ** -1.0 * t[s3] * 100)
    assert heterogeneity(y2, t, mu, run, stream)["p"] < heterogeneity(y, t, mu, run, stream)["p"]


def test_stream3_loader():
    from avlab.kneepool import load_stream3
    if not os.path.exists("results/stopline/stopline.json"):
        pytest.skip("no saved run")
    y, t, mu, run = load_stream3()
    assert len(y) == 96 and y.sum() == 113 and set(np.unique(mu)) == {0.4, 0.5, 0.6, 0.7, 0.85, 1.0}
    assert len(load_stream3(mu_min=0.5)[0]) == 80
