"""avlab.reflexfault: no faults equals avlab.reflex; a stuck miss falls back to the delayed view; noise only on the reflex."""
import jax
import jax.numpy as jnp
import numpy as np

from avlab import reflex as RF
from avlab import reflexfault as RX


def _obs(n, gap=20.0, v=10.0, vl=8.0, has=1.0):
    o = jnp.zeros((n, 28))
    return o.at[:, 0].set(v / 15).at[:, 8].set(has).at[:, 9].set(gap / 50 if has else 3.0).at[:, 10].set(vl / 15 * has)


GAPS = [30.0] * 10 + [6.0] * 5          # a cut-in to 6 m at step 10


def _feed(m, perceive, init, n=4):
    mem, shown = init(n), []
    for t, g in enumerate(GAPS):
        out, mem = perceive(_obs(n, gap=g), jax.random.PRNGKey(t), mem, m)
        shown.append(np.asarray(out))
    return shown


def test_no_faults_equals_reflex():
    a = _feed(RX.Fault(delay_s=1.0), RX.perceive, RX.init_mem_for(RX.Fault(delay_s=1.0)))
    m = RF.Reflex(delay_s=1.0, range_m=10.0)
    b = _feed(m, RF.perceive, RF.init_mem_for(m))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_always_missing_reflex_is_the_delayed_view():
    m = RX.Fault(delay_s=1.0, p_miss=0.999, burst_s=1e6)
    mem = RX.init_mem_for(m)(4)
    mem = mem._replace(miss=jnp.ones(4, bool))            # start inside a burst that never ends
    off = RF.Reflex(delay_s=1.0, range_m=0.0)
    mem_off = RF.init_mem_for(off)(4)
    for t, g in enumerate(GAPS):
        o, mem = RX.perceive(_obs(4, gap=g), jax.random.PRNGKey(t), mem, m)
        o2, mem_off = RF.perceive(_obs(4, gap=g), jax.random.PRNGKey(t), mem_off, off)
        assert np.array_equal(np.asarray(o), np.asarray(o2))


def test_stationary_miss_share_and_noise():
    m = RX.Fault(p_miss=0.15, burst_s=1.0)
    mem, n, misses = RX.init_mem_for(m)(2000), 2000, []
    for t in range(200):
        _, mem = RX.perceive(_obs(n, gap=6.0), jax.random.PRNGKey(t), mem, m)
        misses.append(float(mem.miss.mean()))
    assert abs(np.mean(misses[50:]) - 0.15) < 0.02
    m = RX.Fault(delay_s=1.0, sigma=0.2)
    shown = _feed(m, RX.perceive, RX.init_mem_for(m), n=2000)
    g = 50 * shown[11][:, 9]
    assert abs(np.median(g) - 6.0) < 0.3 and 0.6 < np.std(g) < 1.4   # sd ~ 0.2 * 6 m; over-reads above the delayed gap drop out
