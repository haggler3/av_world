"""avlab.tracker: Markov misses, the hold-last tracker, and the clean model leaves the run unchanged."""
import jax
import jax.numpy as jnp
import numpy as np

from avlab import tracker as TR


def _obs(n, gap=20.0, v=10.0, vl=8.0):
    o = jnp.zeros((n, 28))
    return o.at[:, 0].set(v / 15).at[:, 8].set(1.0).at[:, 9].set(gap / 50).at[:, 10].set(vl / 15)


def test_clean_is_identity():
    o = _obs(64)
    out, _ = TR.perceive(o, jax.random.PRNGKey(0), TR.init_mem(64), TR.Perception())
    assert np.array_equal(np.asarray(out), np.asarray(o))


def test_markov_share_and_burst_length():
    m = TR.Perception(p_miss=0.15, burst_s=1.0)
    n, T = 4000, 200
    o, mem = _obs(n), TR.init_mem(n)
    ms = []
    for t in range(T):
        _, mem = TR.perceive(o, jax.random.PRNGKey(t), mem, m)
        ms.append(np.asarray(mem.miss))
    ms = np.array(ms[50:])
    assert abs(ms.mean() - 0.15) < 0.01
    stay = (ms[1:] & ms[:-1]).sum() / ms[:-1].sum()
    assert abs(1 / (1 - stay) - 5.0) < 0.4  # mean burst 1 s = 5 steps


def test_hold_predicts_gap():
    n = 8
    m = TR.Perception(p_miss=0.999, burst_s=0.2, hold_s=1.0)
    mem = TR.init_mem(n)._replace(g=jnp.full(n, 20.0), vl=jnp.full(n, 8.0), age=jnp.zeros(n, jnp.int32))
    out, mem2 = TR.perceive(_obs(n, gap=30.0), jax.random.PRNGKey(1), mem, m)
    # missed: show the tracked gap advanced by closing speed (10 - 8) x 0.2 s
    assert np.allclose(np.asarray(out[:, 9]) * 50, 20.0 - 0.4, atol=1e-4)
    assert np.allclose(np.asarray(out[:, 8]), 1.0)
    # without a tracker the leader disappears
    out0, _ = TR.perceive(_obs(n, gap=30.0), jax.random.PRNGKey(1), mem, m._replace(hold_s=0.0))
    assert np.allclose(np.asarray(out0[:, 8]), 0.0)


def test_clean_run_matches_plain_policy():
    from avlab import sensorbench as SB
    a = TR.run(TR.Perception(), E=1, seed=7, warm_s=0.0, seconds=8.0, chunk_s=8.0, n_slots=1024)
    b = SB.run_cell("off:clean", E=1, seed=7, warm_s=0.0, seconds=8.0, chunk_s=8.0)
    for k in ("av_veh_s", "av_veh_m", "veh_s", "crashes"):
        assert np.allclose(a[k], b[k]), k
