"""avlab.visibility: hiding a far leader, the phantom at the edge of sight, and clear sight leaves the run unchanged."""
import jax.numpy as jnp
import numpy as np

from avlab import tracker as TR
from avlab import visibility as VI


def _obs(gaps, v=10.0, vl=8.0):
    n = len(gaps)
    o = jnp.zeros((n, 28))
    return o.at[:, 0].set(v / 15).at[:, 8].set(1.0).at[:, 9].set(jnp.asarray(gaps) / 50).at[:, 10].set(vl / 15)


def test_clear_sight_is_identity():
    o = _obs([5.0, 30.0, 140.0])
    out, _ = VI.perceive(o, None, VI.init_mem(3), VI.Visibility())
    assert np.array_equal(np.asarray(out), np.asarray(o))


def test_far_leader_hidden_near_leader_kept():
    o = _obs([5.0, 30.0])
    out, _ = VI.perceive(o, None, VI.init_mem(2), VI.Visibility(sight_m=10.0))
    assert np.array_equal(np.asarray(out[0]), np.asarray(o[0]))  # visible leader unchanged
    assert float(out[1, 8]) == 0.0 and np.isclose(float(out[1, 9]), 3.0)  # hidden: "no leader"


def test_phantom_is_a_stopped_object_at_the_sight_edge():
    o = _obs([5.0, 30.0]).at[1, 8].set(0.0)  # row 1: no leader at all
    out, _ = VI.perceive(o, None, VI.init_mem(2), VI.Visibility(sight_m=10.0, phantom=True))
    assert np.array_equal(np.asarray(out[0]), np.asarray(o[0]))
    c = TR.leader_cols(jnp.array([10.0]), jnp.array([True]), jnp.array([10.0]), jnp.array([0.0]))
    for j, col in c.items():
        assert np.isclose(float(out[1, j]), float(col[0])), j
    assert float(out[1, 10]) == 0.0 and np.isclose(50 * float(out[1, 9]), 10.0)


def test_stopping_speed_stops_within_sight():
    for s in (6.0, 8.0, 14.0, 40.0):
        v = float(VI.stopping_speed(s))
        assert np.isclose(v * 0.2 + v * v / 8.0, s - 2.0, atol=1e-4)
    assert float(VI.stopping_speed(1.0)) == 0.0


def test_clear_sight_run_matches_clean_tracker():
    a = VI.run(VI.Visibility(), E=1, seed=7, warm_s=0.0, seconds=8.0, chunk_s=8.0)
    b = TR.run(TR.Perception(), E=1, seed=7, warm_s=0.0, seconds=8.0, chunk_s=8.0)
    for k in ("av_veh_s", "av_veh_m", "veh_s", "crashes"):
        assert np.allclose(a[k], b[k]), k
