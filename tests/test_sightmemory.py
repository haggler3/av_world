"""avlab.sightmemory: the phantom remembers the last seen leader speed and slows at b_ph."""
import jax.numpy as jnp
import numpy as np

from avlab import sightmemory as SM


def _obs(n, gap, v=8.0, vl=6.0, has=1.0):
    o = jnp.zeros((n, 28))
    return o.at[:, 0].set(v / 15).at[:, 8].set(has).at[:, 9].set(gap / 50).at[:, 10].set(vl / 15)


def test_phantom_speed_bound():
    v = SM.phantom_speed(jnp.array([6.0, 6.0, 6.0]), jnp.array([1, 5, SM.NEVER]), 9.0)
    assert np.allclose(np.asarray(v), [6.0 - 1.8, 0.0, 0.0])


def test_seen_leader_unchanged_then_phantom_slows():
    m = SM.SightMemory(sight_m=8.0, b_ph=4.0)
    mem = SM.init_mem(1)
    o = _obs(1, gap=6.0)
    out, mem = SM.perceive(o, None, mem, m)
    assert np.array_equal(np.asarray(out), np.asarray(o))
    speeds = []
    for _ in range(3):  # leader drifts beyond sight: phantom at 8 m, speed 6 - 0.8 k
        out, mem = SM.perceive(_obs(1, gap=12.0), None, mem, m)
        assert abs(50 * float(out[0, 9]) - 8.0) < 1e-4
        speeds.append(15 * float(out[0, 10]))
    assert np.allclose(speeds, [5.2, 4.4, 3.6], atol=1e-4)


def test_never_seen_is_a_stopped_phantom():
    out, _ = SM.perceive(_obs(2, gap=0.0, has=0.0), None, SM.init_mem(2), SM.SightMemory(10.0))
    assert np.allclose(np.asarray(out[:, 8]), 1.0) and np.allclose(50 * np.asarray(out[:, 9]), 10.0)
    assert np.allclose(np.asarray(out[:, 10]), 0.0)
