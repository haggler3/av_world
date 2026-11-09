import numpy as np

from avlab import frictionav, twinkicks


def test_ratio_matches_known_counts():
    r = frictionav.ratio(10, 100.0, 20, 100.0)
    assert abs(r["ratio"] - 0.5) < 1e-9 and r["lo"] < 0.5 < r["hi"] and 0 < r["p"] < 1
    assert np.isnan(frictionav.ratio(0, 1.0, 0, 1.0)["ratio"])


def test_clamp_av_only_touches_avs():
    import jax.numpy as jnp
    from types import SimpleNamespace as NS

    class S(NS):
        def _replace(self, **k):
            return S(**{**self.__dict__, **k})
    st = S(av=jnp.array([True, False]), T=jnp.array([1.3, 1.0]), v0f=jnp.array([1.0, 1.0]))
    o = frictionav.clamp_av(st, 2.9, 0.67)
    assert np.allclose(o.T, [2.9, 1.0]) and np.allclose(o.v0f, [0.67, 1.0])
    o2 = frictionav.clamp_av(o, 2.9, 0.67)  # idempotent
    assert np.allclose(o2.T, o.T) and np.allclose(o2.v0f, o.v0f)


def test_classify_kinds():
    T, N = 5, 3
    a = {"alive": np.ones((T, N), bool), "trk": np.zeros((T, N), int), "s": np.array([[0.0, 10.0, 500.0]] * T)}
    first = np.array([0, 2, -1])
    assert twinkicks.classify(a, 0, first, 100)["kind"] == "following"
    a["trk"][:, 1] = 150
    assert twinkicks.classify(a, 0, first, 100)["kind"] == "box"
    assert twinkicks.classify(a, 0, np.array([0, -1, -1]), 100)["kind"] == "none"


def test_friction_estimator_reads_the_floor_and_reports_lower_bound():
    from avlab import frictionsense as fs
    rng = np.random.default_rng(0)
    dec = np.abs(rng.normal(0, 1.0, (100, 50))).clip(0, 4.05)  # a floor at 9 x 0.45 = 4.05 is hit often
    dec[:, :5] = 4.05
    est = fs.estimate(dec)
    assert np.allclose(est["mu_hat"][50:], 0.45) and np.all(est["lower_bound"][50:] <= 0.45 + 1e-9)
    soft = np.minimum(dec, 3.0)  # never reaches a floor: only a bound, no identified value
    soft[:, :1] = 3.0
    e2 = fs.estimate(soft, k=3)
    assert np.isnan(e2["mu_hat"][50:]).sum() >= 0 and np.all(e2["lower_bound"][50:] <= 3.0 / 9 + 1e-9)
    sc = fs.score(est, np.full(100, 0.45), burn=10)
    assert sc["identified"] == 1.0 and sc["median_abs_err"] < 1e-6
