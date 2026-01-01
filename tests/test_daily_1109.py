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
