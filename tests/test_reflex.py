"""avlab.reflex: range 0 is the latency view, the reflex shows the true near leader, a farther true leader is ignored."""
import jax
import jax.numpy as jnp
import numpy as np

from avlab import latency as LT
from avlab import reflex as RF
from avlab import tracker as TR


def _obs(n, gap=20.0, v=10.0, vl=8.0, has=1.0):
    o = jnp.zeros((n, 28))
    return o.at[:, 0].set(v / 15).at[:, 8].set(has).at[:, 9].set(gap / 50 if has else 3.0).at[:, 10].set(vl / 15 * has)


def _feed(m, gaps, n=3):
    key, mem, shown = jax.random.PRNGKey(0), RF.init_mem_for(m)(n), []
    for g in gaps:
        out, mem = RF.perceive(_obs(n, gap=g), key, mem, m)
        shown.append(np.asarray(out))
    return shown


def test_range_zero_equals_latency():
    gaps = [30.0 - 0.4 * k for k in range(15)] + [6.0] * 5   # a cut-in to 6 m at step 15
    a = _feed(RF.Reflex(delay_s=1.0, range_m=0.0), gaps)
    lt = LT.Latency(1.0, True)
    key, mem, b = jax.random.PRNGKey(0), LT.init_mem_for(lt)(3), []
    for g in gaps:
        out, mem = LT.perceive(_obs(3, gap=g), key, mem, lt)
        b.append(np.asarray(out))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_cut_in_within_range_is_seen_at_once():
    gaps = [30.0] * 10 + [6.0] * 3
    far = _feed(RF.Reflex(delay_s=1.0, range_m=0.0), gaps)
    near = _feed(RF.Reflex(delay_s=1.0, range_m=10.0), gaps)
    assert 50 * far[10][0, 9] > 20.0                     # delayed view still shows the old leader
    assert np.isclose(50 * near[10][0, 9], 6.0)            # reflex shows the cut-in at once
    assert np.isclose(15 * near[10][0, 10], 8.0)
    # all leader columns match a zero-delay observation of the same scene
    c = TR.leader_cols(jnp.full(3, 10.0), jnp.ones(3, bool), jnp.full(3, 6.0), jnp.full(3, 8.0))
    for j, col in c.items():
        assert np.allclose(near[10][:, j], np.asarray(col), atol=1e-6), j


def test_farther_true_leader_and_other_columns_untouched():
    o_true, o_lr = _obs(2, gap=15.0), _obs(2, gap=9.0)
    out, use = RF.fuse(o_true, o_lr, 20.0)       # delayed view is already more cautious: keep it
    assert not bool(use.any()) and np.array_equal(np.asarray(out), np.asarray(o_lr))
    out, use = RF.fuse(_obs(2, gap=5.0), _obs(2, has=0.0), 10.0)  # delayed view shows no leader
    assert bool(use.all()) and float(out[0, 8]) == 1.0
    o = _obs(2, gap=5.0).at[:, 3].set(0.7)
    out, _ = RF.fuse(o, _obs(2, has=0.0).at[:, 3].set(0.7), 10.0)
    assert np.isclose(float(out[0, 3]), 0.7)
