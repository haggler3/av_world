"""The five avlab ideas (attribution, blackbox, turing, hotspots, greenwave): their math on known answers, and tiny runs."""
import itertools

import numpy as np

from avlab import attribution as A
from avlab import hotspots as H
from avlab import turing as T


def test_shapley_is_efficient_and_additive():
    n = 4
    rng = np.random.default_rng(0)
    v = {S: rng.normal() for S in A.coalitions(n)}
    c = A.coefficients(n)
    phi = [sum(w * v[S] for S, w in c[i].items()) for i in range(n)]
    assert np.isclose(sum(phi), v[frozenset(range(n))] - v[frozenset()])
    add = {S: sum([1.0, 2.0, 3.0, 4.0][i] for i in S) for S in A.coalitions(n)}
    assert np.allclose([sum(w * add[S] for S, w in c[i].items()) for i in range(n)], [1, 2, 3, 4])


def test_auc_and_logistic_separate_a_shifted_class():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200, bool), np.zeros(400, bool)]
    F = rng.normal(size=(600, 3)) + np.where(y[:, None], [1.5, 0, 0], 0)
    r = T.test(F, y, np.arange(600) % 5, n_boot=200)
    assert 0.8 < r["auc"] < 0.95 and r["ci"][0] < r["auc"] < r["ci"][1]
    assert np.isclose(T.auc(np.r_[np.ones(5), np.zeros(5)], np.r_[np.ones(5, bool), np.zeros(5, bool)]), 1.0)


def test_empirical_bayes_shrinks_small_sites_toward_the_spf():
    rng = np.random.default_rng(1)
    E = rng.uniform(0.5, 50, 400)
    lam = 0.05 * E * rng.gamma(2.0, 0.5, 400)  # overdispersed true rates
    y = rng.poisson(lam).astype(float)
    est, mu, alpha, (a, b) = H.eb(y, E)
    assert 0.7 < b < 1.3 and alpha > 0
    assert np.mean((est - lam) ** 2) < np.mean((y - lam) ** 2)  # EB beats the raw count
    small = E < 2
    assert np.all(np.abs(est[small] - mu[small]) <= np.abs(y[small] - mu[small]) + 1e-9)


def test_hotspot_sites_cover_every_track():
    from xnet import network as NW
    net, _, _ = NW.build("oakland")
    sid, S = H.site_ids(net)
    assert len(sid) == 2 * len(net["link_len"]) + len(net["mov_len"]) and sid.min() >= 0 and sid.max() < S


def test_blackbox_finds_and_explains_a_crash(tmp_path):
    from avlab import blackbox as B
    rep = B.run("oakland", scenario="red_light_runner", out=str(tmp_path), gif=False, budget=300.0)
    assert rep is not None and rep["types"] and rep["vehicles"]
    assert all(k in rep["vehicles"][0] for k in ("kind", "speed_at_impact_kmh", "light_on_approach", "contributing"))
    assert (tmp_path / f"oakland_red_light_runner_s0.md").exists()


def test_greenwave_runner_batches_signal_timings():
    import jax
    import jax.numpy as jnp
    from avlab import greenwave as G
    from examples import _common as C
    from xnet import network as NW
    net, _, _ = NW.build("oakland")
    chunk, reset = G.make_runner(net, C.calibrated(p_av=0.0), n_slots=512)
    st = reset(jax.random.split(jax.random.PRNGKey(0), 2))
    o = jnp.tile(jnp.asarray(net["node_offset"], jnp.float32), (2, 1))
    g = jnp.tile(jnp.asarray(net["node_g0"], jnp.float32), (2, 1))
    st = chunk(st, o, g.at[1].set(20.0))
    assert float(st.t[0]) > 59 and st.alive.shape == (2, 512)
