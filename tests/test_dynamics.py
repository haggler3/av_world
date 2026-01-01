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


def test_mutated_cities_differ_keep_shapes_and_stay_routable(oak):
    from xnet import mutate as MU

    net, geo, info = oak
    a, closed_a = MU.mutate(net, geo, 1)
    b, closed_b = MU.mutate(net, geo, 2)
    z, closed_z = MU.mutate(net, geo, 3, strength=0.0)
    for k in MU.KEYS + ("nh",):
        assert np.shape(a[k]) == np.shape(net[k]) and np.shape(b[k]) == np.shape(net[k])
    assert not np.allclose(a["link_v"], b["link_v"]) and not np.allclose(a["node_cycle"], b["node_cycle"])
    assert len(closed_a) > 5 and len(closed_z) == 0
    assert np.allclose(z["link_v"], net["link_v"]) and np.allclose(z["src_cdf"], net["src_cdf"], atol=1e-5)  # strength 0: the real map
    # the signal plans stay valid (green of axis 0 ends before axis 1's starts), and no trip starts on a closed link
    sig = net["node_signal"]
    assert np.all((a["node_g0"] + 10.0 < a["node_cycle"])[sig])
    w = np.diff(np.concatenate([[0.0], a["src_cdf"]]))
    assert np.all(w[closed_a] < 1e-9) and abs(a["src_cdf"][-1] - 1.0) < 1e-5
    # routing avoids the closed links: no first hop leads into one, unless it is the only way
    sl = net["succ_link"]
    nh = a["nh"]
    into = sum(int(sl[li, nh[li, d]] in set(closed_a)) for li in range(0, len(nh), 7) for d in range(0, nh.shape[1], 5) if nh[li, d] >= 0)
    assert into < 0.05 * (len(nh) // 7 * nh.shape[1] // 5)


def test_a_batch_of_different_cities_runs_in_one_vmap(oak):
    from xnet import mutate as MU
    from xnet import sim as X

    net, geo, info = oak
    world = MU.worlds(net, geo, [MU.HELD_OUT, MU.HELD_OUT + 1])
    J = X.to_jnp(net)
    beh = jax.tree.map(lambda x: jnp.full((2,), x, jnp.float32), X.Behaviour(p_av=0.3))
    st = jax.vmap(lambda k: X.reset(k, J, 512))(jax.random.split(jax.random.PRNGKey(0), 2))
    f = jax.jit(lambda s: jax.vmap(lambda s, b, n, h: X.step(s, {**J, **n}, b, h), in_axes=(0, 0, 0, 0))(s, beh, world["net"], world["nh"]))
    for _ in range(300):
        st = f(st)
    assert int(np.asarray(st.alive).sum()) > 20 and np.all(np.isfinite(np.asarray(st.s)))


def test_shaping_telescopes_and_leaves_returns_unchanged():
    # r' = r + gamma phi(s') - phi(s) over a trajectory that ends with phi = 0 differs from r only by -phi(s_0)
    g = 0.99
    phi = np.array([-0.5, -2.0, -0.1, -3.0, 0.0])
    r = np.array([0.1, 0.2, 0.3, 0.4])
    shaped = r + g * phi[1:] - phi[:-1]
    ret = lambda x: sum(g ** t * v for t, v in enumerate(x))  # noqa: E731
    assert abs(ret(shaped) - (ret(r) - phi[0])) < 1e-9


def test_splitting_is_unbiased_for_any_importance():
    from avlab import split as SP

    # a toy "area": each of 16 copies crashes in a window with probability p_i that depends on a hidden danger state;
    # resampling by any g must keep E[weighted crashes] equal to the plain Monte Carlo mean
    rng = np.random.default_rng(0)
    E, windows, trials = 16, 6, 4000
    def run(split):
        danger = rng.random(E) * 0.2
        w, tot = np.ones(E), 0.0
        for k in range(windows):
            crash = rng.random(E) < 0.02 + 0.5 * danger
            tot += (w * crash).sum()
            danger = np.clip(danger + rng.normal(0, 0.05, E) + 0.3 * crash, 0, 1)
            if split and k < windows - 1:
                g = np.clip(np.exp(8 * (danger - danger.mean())), 0.25, 4.0)
                g /= g.mean()
                p = w * g
                idx = SP.systematic(p / p.sum(), rng)
                w = p.mean() / g[idx]
                danger = danger[idx]
        return tot
    plain = np.array([run(False) for _ in range(trials)])
    split = np.array([run(True) for _ in range(trials)])
    se = np.sqrt(plain.var() / trials + split.var() / trials)
    assert abs(plain.mean() - split.mean()) < 4 * se


def test_global_hazard_tilting_collapses_for_a_city():
    from avlab import split as SP

    e = SP.tilt_ess(3000, p=0.0514, factor=2.0)
    assert e["log_weight_sd"] > 5 and e["ess_fraction"] < 1e-6
