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


def test_constant_acceleration_prediction_is_exact_for_a_braking_leader():
    # follower at 10 m/s, leader at 9 m/s braking at 2 m/s^2 (0.4 m/s per step); delay 0.6 s = 3 steps
    n, key, dt = 2, jax.random.PRNGKey(0), 0.2
    m = LT.Latency(delay_s=0.6, compensate=True, accel=True)
    mem = LT.init_mem_for(m)(n)
    gap, vl, shown = 40.0, 9.0, []
    for k in range(10):
        out, mem = LT.perceive(_obs(n, gap=gap, v=10.0, vl=vl), key, mem, m)
        shown.append((50 * float(out[0, 9]), 15 * float(out[0, 10]), gap, vl))
        gap -= (10.0 - vl) * dt
        vl -= 2.0 * dt
    # from step 4 on, two delayed observations exist: the prediction recovers the true gap and leader speed
    for g_s, vl_s, g, v in shown[4:]:
        assert abs(g_s - g) < 1e-3 and abs(vl_s - v) < 1e-3
    # constant velocity (accel=False) overestimates the gap of a braking leader
    m0 = LT.Latency(delay_s=0.6, compensate=True)
    mem0, gap, vl = LT.init_mem_for(m0)(n), 40.0, 9.0
    for k in range(6):
        out, mem0 = LT.perceive(_obs(n, gap=gap, v=10.0, vl=vl), key, mem0, m0)
        gap -= (10.0 - vl) * dt
        vl -= 2.0 * dt
    assert 50 * float(out[0, 9]) > gap + (10.0 - vl - 2 * dt) * dt + 0.05  # shown gap > the true gap at that step


def test_accel_false_keeps_the_constant_velocity_path():
    o = _obs(4, gap=25.0)
    a = LT.Latency(delay_s=0.4, compensate=True)
    b = LT.Latency(delay_s=0.4, compensate=True, accel=False)
    ma, mb = LT.init_mem_for(a)(4), LT.init_mem_for(b)(4)
    for _ in range(4):
        oa, ma = LT.perceive(o, None, ma, a)
        ob, mb = LT.perceive(o, None, mb, b)
    assert np.array_equal(np.asarray(oa), np.asarray(ob))
