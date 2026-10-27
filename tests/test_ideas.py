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


def test_ridge_poisson_stays_finite_under_separation():
    rng = np.random.default_rng(2)
    E = rng.uniform(1, 30, 300)
    z = (rng.random(300) < 0.3).astype(float)  # one site type never crashes: the unpenalised fit separates
    y = np.where(z > 0, 0.0, rng.poisson(0.1 * E)).astype(float)
    D = np.c_[np.ones(300), np.log(E), z]
    b = H.poisson_ridge(y, D)
    assert np.all(np.isfinite(b)) and abs(b[2]) < 15 and b[2] < -1


def test_ml_dispersion_recovers_a_known_alpha():
    rng = np.random.default_rng(3)
    mu = rng.uniform(0.5, 5, 4000)
    alpha = 0.8
    y = rng.poisson(rng.gamma(1 / alpha, alpha * mu)).astype(float)
    assert abs(H.nb_alpha_ml(y, mu) - alpha) < 0.15


def test_site_structure_has_conflict_counts():
    from xnet import network as NW
    net, _, _ = NW.build("oakland")
    Z = H.site_structure(net)
    assert Z.shape == (len(net["node_signal"]), 6) and np.isfinite(Z).all() and Z[:, :3].sum() > 0


def test_shapley_reports_speed_and_per_km_from_per_copy_counters():
    from xnet import sim as X
    rng = np.random.default_rng(4)
    names = ("a", "b")
    col = {k: i for i, k in enumerate(X.METRICS)}
    per = {}
    for S in A.coalitions(2):
        m = np.zeros((3, X.NM))
        m[:, col["veh_s"]], m[:, col["veh_m"]] = 3.6e5, 3.6e5 * (5 + len(S)) + rng.normal(0, 1e3, 3)  # speed rises 1 m/s per mechanism
        m[:, col["rear-end"]] = 2 + len(S)
        per[S] = m
    r = A.shapley(per, names)
    assert np.allclose([r["speed_kmh"]["phi"][n] for n in names], 3.6, atol=0.05)
    assert set(r["per_vkm"]["phi"]) == set(names)


def test_comfortable_driver_brakes_no_harder_than_needed():
    import jax.numpy as jnp
    from av_world import easy
    obs = np.zeros((2, 28), np.float32)
    obs[:, 0], obs[:, 3], obs[:, 16] = 10 / 15, 1.0, 1.0
    obs[:, 6], obs[:, 7] = 1.0, 30 / 100   # a stop line 30 m ahead at 10 m/s: about 1.7 m/s^2 needed
    obs[0, 24], obs[1, 24] = 1.7 / 8, 6.0 / 8   # car 1: an emergency (6 m/s^2 needed)
    u_c, _ = easy.drivers.comfortable()(jnp.asarray(obs), None)
    u_i, _ = easy.drivers.idm()(jnp.asarray(obs), None)
    a_c, a_i = np.asarray(easy.accel_from_command(u_c[:, 0])), np.asarray(easy.accel_from_command(u_i[:, 0]))
    assert a_c[0] >= a_i[0] and a_c[0] >= -1.5 * 1.7 - 0.2 - 1e-5   # gentler, but still braking enough
    assert np.isclose(a_c[1], a_i[1])                               # an emergency is left to IDM


def test_summary_reports_every_exposure():
    from av_world import easy
    run = easy.simulate("oakland", minutes=0.2, copies=1, warm_minutes=0.1)
    s = run.summary()
    assert {"crashes_per_1000_vehicle_hours", "crashes_per_million_vehicle_km", "crashes_per_million_junction_entries"} <= set(s)
