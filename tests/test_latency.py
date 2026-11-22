"""avlab.latency: the delayed leader, the forward prediction, and delay 0 leaves the run unchanged."""
import jax
import jax.numpy as jnp
import numpy as np

from avlab import latency as LT
from avlab import tracker as TR


def _obs(n, gap=20.0, v=10.0, vl=8.0):
    o = jnp.zeros((n, 28))
    return o.at[:, 0].set(v / 15).at[:, 8].set(1.0).at[:, 9].set(gap / 50).at[:, 10].set(vl / 15)


def test_zero_delay_is_identity():
    o = _obs(16)
    m = LT.Latency()
    out, _ = LT.perceive(o, None, LT.init_mem_for(m)(16), m)
    assert np.array_equal(np.asarray(out), np.asarray(o))


def test_delay_shows_old_gap_and_prediction_recovers_it():
    n, key = 4, jax.random.PRNGKey(0)
    for comp in (False, True):
        m = LT.Latency(delay_s=0.6, compensate=comp)  # 3 steps
        mem = LT.init_mem_for(m)(n)
        # follower at 10 m/s closing on a leader at 8 m/s: the true gap shrinks 0.4 m per step
        shown = []
        for k in range(8):
            out, mem = LT.perceive(_obs(n, gap=30.0 - 0.4 * k), key, mem, m)
            shown.append(50 * float(out[0, 9]))
        if comp:  # constant-velocity prediction is exact here
            assert np.allclose(shown[3:], [30.0 - 0.4 * k for k in range(3, 8)], atol=1e-4)
        else:     # shows the gap from 3 steps ago
            assert np.allclose(shown[3:], [30.0 - 0.4 * (k - 3) for k in range(3, 8)], atol=1e-4)
        # the first step of a fresh ring shows the current observation
        assert abs(shown[0] - 30.0) < 1e-4


def test_leader_cols_match_the_simulator_formulas():
    o = _obs(1, gap=12.0, v=10.0, vl=6.0)
    c = TR.leader_cols(jnp.array([10.0]), jnp.array([True]), jnp.array([12.0]), jnp.array([6.0]))
    assert np.isclose(float(c[11][0]), 0.4) and np.isclose(float(c[26][0]), 12 / 10 / 3)
    assert np.isclose(float(c[25][0]), 16 / (2 * 10) / 8) and np.isclose(float(c[9][0]), float(o[0, 9]))


def test_zero_delay_run_matches_clean_tracker():
    a = LT.run(LT.Latency(), E=1, seed=7, warm_s=0.0, seconds=8.0, chunk_s=8.0)
    b = TR.run(TR.Perception(), E=1, seed=7, warm_s=0.0, seconds=8.0, chunk_s=8.0)
    for k in ("av_veh_s", "av_veh_m", "veh_s", "crashes"):
        assert np.allclose(a[k], b[k]), k
