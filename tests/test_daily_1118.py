"""Per-street friction (2026-11-18): the mask helper, default-off identity, and the patchy-ice statistics."""
import jax
import jax.numpy as jnp
import numpy as np

from avlab.patchyice import mixture_test
from xnet import network as NW
from xnet import sim as X


def _net():
    return NW.build("oakland")[0]


def test_patch_links_covers_share_of_length():
    net = _net()
    L = np.asarray(net["link_len"], float)
    for share in (0.0, 0.3, 0.5, 1.0):
        m = X.patch_links(net, share, seed=1)
        assert m.dtype == bool and m.shape == L.shape
        got = L[m].sum() / L.sum()
        assert got <= share + 1e-9 and got >= share - L.max() / L.sum() - 1e-9
    assert np.array_equal(X.patch_links(net, 0.5, seed=2), X.patch_links(net, 0.5, seed=2))


def _run(J, nh, behs, steps=150):
    B = jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *behs)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(3), len(behs)), J, 512)
    st = jax.lax.scan(lambda s, _: (jax.vmap(lambda si, b: X.step(si, J, b, nh))(s, B), None), st, None, length=steps)[0]
    return np.asarray(st.m), np.asarray(st.v)


def test_mask_with_multiplier_one_changes_nothing_and_full_ice_equals_scalar():
    net = _net()
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    behs = [X.Behaviour(friction=0.6), X.Behaviour(friction=0.6, patch_friction=1.0)]
    m0, v0 = _run(J, nh, behs[:1])
    Jm = dict(J, link_patch=jnp.asarray(X.patch_links(net, 0.5, seed=0)))
    m1, v1 = _run(Jm, nh, behs[1:])
    assert np.array_equal(m0, m1) and np.array_equal(v0, v1)
    # every link patched at x0.5 on friction 1.0 equals scalar friction 0.5
    Jall = dict(J, link_patch=jnp.ones(len(net["link_len"]), bool))
    ma, va = _run(Jall, nh, [X.Behaviour(friction=1.0, patch_friction=0.5)])
    mb, vb = _run(J, nh, [X.Behaviour(friction=0.5)])
    assert np.allclose(ma, mb) and np.allclose(va, vb, atol=1e-5)


def test_mixture_test():
    # observed equal to the mixture expectation: ratio 1, p 1
    r = mixture_test(kp=40, tp=1000, w=0.5, k4=70, t4=1000, k1=10, t1=1000)
    assert abs(r["ratio"] - 1) < 1e-12 and r["p"] > 0.99 and r["lo"] < 1 < r["hi"]
    # far above the mixture: small p
    r = mixture_test(kp=120, tp=1000, w=0.5, k4=70, t4=1000, k1=10, t1=1000)
    assert r["ratio"] == 3 and r["p"] < 1e-4
