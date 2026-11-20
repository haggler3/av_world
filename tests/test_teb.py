"""Timed Elastic Band planner (avlab.teb): converges, respects stop targets and leaders, and plugs into the policy slot."""
import jax
import jax.numpy as jnp
import numpy as np

from avlab import teb as T


def _obs(v, v0=13.0, lead=None, stop=None):
    o = np.zeros(28, np.float32)
    o[0], o[3], o[16] = v / 15, v0 / 15, 1.0
    if lead is not None:
        o[8], o[9], o[10] = 1, lead[0] / 50, lead[1] / 15
    if stop is not None:
        o[6], o[7] = 1, stop / 100
    return o


def test_free_road_cruise_and_speed_up():
    f = jax.jit(T.teb_accel)
    assert abs(float(f(jnp.asarray(_obs(13.0))[None])[0])) < 0.05  # holds the desired speed
    v = 0.0
    for _ in range(100):  # 20 s from rest
        v = max(v + float(np.clip(f(jnp.asarray(_obs(v))[None])[0], -8, 3)) * 0.2, 0.0)
    assert 12.5 < v < 13.6


def test_stops_before_line_and_behind_stopped_car():
    f = jax.jit(T.teb_accel)
    for kind in ("stop", "lead"):
        v, x = 13.0, 0.0
        for _ in range(200):
            g = 80.0 - x
            o = _obs(v, stop=g) if kind == "stop" else _obs(v, lead=(g, 0.0))
            v = max(v + float(np.clip(f(jnp.asarray(o)[None])[0], -8, 3)) * 0.2, 0.0)
            x += v * 0.2
        assert v < 0.1 and 0.0 < 80.0 - x < 4.0


def test_lm_does_not_increase_cost_and_masked_slot():
    fe = T.features(jnp.asarray(np.stack([_obs(10, lead=(20, 0)), _obs(5), _obs(12, stop=40)])))
    for i in range(3):
        fi = jax.tree.map(lambda a: a[i], fe)
        c0 = float(T._cost(T._init(fi), fi))
        assert float(T.plan_one(fi)[3]) <= c0 + 1e-6
    obs = jnp.asarray(np.stack([_obs(10, lead=(20, 0)), _obs(5), _obs(12, stop=40)]))
    u, _ = T.teb_policy_masked(obs, None, jnp.array([True, False, True]), cap=2)
    assert u.shape == (3, 2) and float(u[0, 0]) < 0 and float(u[2, 0]) < 0
