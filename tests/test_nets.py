import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nets import blame as BL
from nets import common as CM
from nets import mindreader as MR
from nets import occlusion as OC
from nets import sbi as SB

needs_data = pytest.mark.skipif(not os.path.exists("data/pgh_segments.parquet"), reason="needs python -m pgh.fetch")


def test_auroc_and_grouped_bootstrap():
    assert CM.auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert CM.auroc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    m, lo, hi = CM.boot(np.r_[np.zeros(50), np.ones(50)], groups=np.repeat(np.arange(10), 10))
    assert m == 0.5 and lo < 0.5 < hi


def test_occlusion_geometry_on_a_hand_built_scene():
    # ego at the origin; a 5 m car at x=10 hides the car at x=20 behind it, not the one offset sideways or the one behind the ego
    x, y = np.array([10.0, 20.0, 20.0, -15.0]), np.array([0.0, 0.0, 10.0, 0.0])
    vis = OC.visibility(0.0, 0.0, x, y, np.zeros(4), np.array([5.0, 4.5, 4.5, 4.5]))
    assert vis.tolist() == [True, False, True, True]
    # a box rotated 90 degrees across the sight line still blocks it; one parallel to it but off the line does not
    assert OC.seg_hits_box(np.array([0.0, 0.0]), np.array([20.0, 0.0]), 10.0, 0.0, np.pi / 2, 4.0, 2.0)
    assert not OC.seg_hits_box(np.array([0.0, 0.0]), np.array([20.0, 0.0]), 10.0, 3.0, 0.0, 4.0, 2.0)


def test_occlusion_deep_set_is_permutation_invariant():
    p = OC.init(jax.random.PRNGKey(0))
    tok = jnp.asarray(np.random.default_rng(0).normal(size=(3, OC.KMAX, 10)), jnp.float32).at[:, 20:].set(0.0)
    perm = np.r_[np.random.default_rng(1).permutation(20), np.arange(20, OC.KMAX)]
    assert np.allclose(np.asarray(OC.apply(p, tok)), np.asarray(OC.apply(p, tok[:, perm])), atol=1e-4)


def test_blame_set_network_is_permutation_equivariant_and_ignores_padding():
    p = BL.init(jax.random.PRNGKey(0))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, BL.M, BL.F)), jnp.float32)
    mask = jnp.asarray(np.r_[np.ones(6), np.zeros(2)].astype(bool))[None].repeat(2, 0)
    s = np.asarray(BL.apply(p, x, mask))
    perm = np.r_[np.random.default_rng(2).permutation(6), 6, 7]
    s2 = np.asarray(BL.apply(p, x[:, perm], mask[:, perm]))
    assert np.allclose(s[:, perm], s2, atol=1e-4)
    s3 = np.asarray(BL.apply(p, x.at[:, 6:].set(99.0), mask))      # what is in the padded slots must not matter
    assert np.allclose(s[:, :6], s3[:, :6], atol=1e-4)


def test_mindreader_convolution_is_causal():
    p = MR.init(jax.random.PRNGKey(0))
    x = jnp.asarray(np.random.default_rng(0).normal(size=(2, MR.W, 12)), jnp.float32)
    h1 = MR.apply(p, x)
    # the last-step output depends on the past only: changing nothing but... check that changing the FIRST step changes it and the
    # pooled features are finite; causality proper: the conv stack at step t must not see t+1
    def stack(p, x):
        h = x
        for l, dil in zip(p["c"], (1, 2, 4)):
            T = h.shape[1]
            hp = jnp.pad(h, ((0, 0), (4 * dil, 0), (0, 0)))
            taps = jnp.concatenate([hp[:, j * dil:j * dil + T] for j in range(5)], -1)
            y = taps @ l["w"].reshape(-1, l["w"].shape[-1]) + l["b"]
            h = jax.nn.gelu(y) + (h if h.shape[-1] == y.shape[-1] else 0.0)
        return h
    a = np.asarray(stack(p, x))
    b = np.asarray(stack(p, x.at[:, 15:].set(5.0)))
    assert np.allclose(a[:, :15], b[:, :15], atol=1e-5) and not np.allclose(a[:, 15:], b[:, 15:])
    assert np.isfinite(np.asarray(h1)).all()


def test_mindreader_window_normalisation_shapes_and_flags():
    x = np.zeros((4, MR.W, 8), np.float32)
    x[..., 2] = 150.0          # no leader
    x[..., 4] = 50.0           # 50 m to the line
    x[..., 5] = 0              # red ahead
    n = MR.normalise(x)
    assert n.shape == (4, MR.W, 12) and n[..., 3].max() == 0 and n[..., 6].min() == 1.0
    assert MR.hand_features(x).shape == (4, 14)


def test_sbi_prior_maps_the_unit_cube_onto_the_box():
    u = np.array([[0.0] * SB.D, [1.0] * SB.D, [0.5] * SB.D])
    th = SB.to_theta(u)
    for k, lo, hi, lg in SB.KNOBS:
        want = (10 ** lo, 10 ** hi) if lg else (lo, hi)
        assert np.isclose(th[k][0], want[0]) and np.isclose(th[k][1], want[1])


def test_sbi_mixture_density_learns_a_known_posterior():
    """theta ~ U(0,1)^9, stats = first two thetas + noise: the posterior of knob 0 must concentrate, knob 5 must stay at the prior."""
    rng = np.random.default_rng(0)
    U = rng.uniform(size=(4000, SB.D)).astype(np.float32)
    S = np.concatenate([U[:, :2] + 0.02 * rng.normal(size=(4000, 2)), rng.normal(size=(4000, 4))], 1).astype(np.float32)
    p = SB.init(jax.random.PRNGKey(0))
    p, _ = CM.fit(SB.nll, p, {"s": S, "z": SB.logit(U).astype(np.float32)}, steps=1500, batch=256, lr=1e-3)
    dr = SB.sample(p, S[0], 2000, jax.random.PRNGKey(1))
    assert dr.shape == (2000, SB.D) and ((dr > 0) & (dr < 1)).all()
    assert abs(np.median(dr[:, 0]) - U[0, 0]) < 0.1 and dr[:, 0].std() < 0.1
    assert dr[:, 5].std() > 0.2                                        # an unidentified knob keeps a wide posterior


@needs_data
def test_forked_branches_with_the_same_action_are_bit_identical():
    from nets import collect as C
    from xnet import sim as X
    w = C.World()
    beh = X.Behaviour(crn=1.0)
    st = C.run(w, w.reset(2, 0), beh, 20.0)
    stf = C.fork(st, 3)
    _, f = C.record(w, stf, beh, 6.0, 5, lambda s, tr, pc: {"s": s.s, "v": s.v})
    s = np.asarray(f["s"]).reshape(f["s"].shape[0], 2, 3, -1)
    assert (s[:, :, 0] == s[:, :, 1]).all() and (s[:, :, 0] == s[:, :, 2]).all()


@needs_data
def test_calming_a_driver_changes_only_that_driver():
    from nets import collect as C
    from xnet import sim as X
    w = C.World()
    st = C.run(w, w.reset(1, 0), X.Behaviour(p_red=0.9, crn=1.0), 30.0)
    one = jax.tree.map(lambda a: a[0], st)
    i = int(np.nonzero(np.asarray(one.alive & one.runner))[0][0])
    c = BL.calm(one, i)
    assert not bool(c.runner[i]) and float(c.dist_t[i]) == 0.0
    for name in ("runner", "nolook", "aggr", "nosignal", "dist_t"):
        a, b = np.asarray(getattr(one, name)), np.asarray(getattr(c, name))
        assert (np.delete(a, i) == np.delete(b, i)).all()
    assert (np.asarray(c.s) == np.asarray(one.s)).all() and (np.asarray(c.uid) == np.asarray(one.uid)).all()


@needs_data
def test_recorder_matches_a_step_by_step_loop():
    from nets import collect as C
    from xnet import sim as X
    w = C.World()
    beh = X.Behaviour(crn=1.0)
    st = C.run(w, w.reset(2, 1), beh, 10.0)
    _, f = C.record(w, st, beh, 4.0, 1, lambda s, tr, pc: {"v": s.v})
    step = jax.jit(jax.vmap(lambda s: X.step(s, w.J, beh, w.nh)))
    s = st
    for _ in range(20):
        s = step(s)
    assert np.allclose(np.asarray(f["v"][-1]), np.asarray(s.v), atol=1e-5)
