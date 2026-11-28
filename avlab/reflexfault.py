"""An imperfect reflex sensor: the zero-delay short-range sensor of `avlab.reflex` with bursty missed detections and
relative range noise. `xnet/` and `avlab.reflex` are untouched.

Per AV row and per 0.2 s step (noise key `fold_in(step key, SALT)`, separate from the simulator and `avlab.latency`):

    obs_lr, mem_lr = latency.perceive(obs_true, mem_lr)            # delayed (and predicted) long-range view
    miss_t  two-state Markov chain as `avlab.tracker`: P(miss | miss before) = 1 - 1/L, P(miss | none before) = q,
            L = burst_s / DT, q = p_miss / (L (1 - p_miss)); burst_s = DT gives independent misses
    g_r     = g * (1 + sigma * z),  z ~ N(0, 1)                     # reflex range reading [m]
    det     = true leader exists and g <= range_m and not miss_t
    use     = det and (obs_lr shows no leader or g_r < shown gap)
    shown   = (g_r, true leader speed) where use, else obs_lr's leader

With p_miss = 0 and sigma = 0 the AV sees exactly what `avlab.reflex` shows (test).

    from avlab import reflexfault as RX
    out = RX.run(RX.Fault(p_miss=0.15, burst_s=1.0), E=16, seconds=600)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from avlab import latency as LT
from avlab import tracker as TR

SALT = 0x5EF1
DT = TR.DT


class Fault(NamedTuple):
    delay_s: float = 2.0      # delay of the long-range view [s]
    compensate: bool = True   # forward prediction of the delayed view (as avlab.latency)
    accel: bool = False
    range_m: float = 10.0     # reach of the reflex sensor [m]
    p_miss: float = 0.0       # stationary share of steps the reflex misses a leader in range
    burst_s: float = DT       # mean length of a run of reflex misses [s]
    sigma: float = 0.0        # relative standard deviation of the reflex range reading


class Mem(NamedTuple):
    lr: tuple           # avlab.latency memory
    miss: jnp.ndarray   # [N] bool, Markov state of the reflex


def latency_of(m: Fault):
    return LT.Latency(m.delay_s, m.compensate, m.accel)


def init_mem_for(m: Fault):
    lr0 = LT.init_mem_for(latency_of(m))
    return lambda n: Mem(lr0(n), jnp.zeros(n, bool))


def perceive(obs, key, mem: Mem, m: Fault):
    obs_lr, lr2 = LT.perceive(obs, key, mem.lr, latency_of(m))
    n = obs.shape[0]
    L = max(m.burst_s / DT, 1.0)
    q = TR.onset_prob(TR.Perception(m.p_miss, m.burst_s))
    k = jax.random.fold_in(key, SALT)
    u = jax.random.uniform(k, (n,))
    miss = jnp.where(mem.miss, u < 1.0 - 1.0 / L, u < q)
    if m.range_m <= 0:
        return obs_lr, Mem(lr2, miss)
    v = 15.0 * obs[:, 0]
    has, g, vl = obs[:, 8] > 0.5, 50.0 * obs[:, 9], 15.0 * obs[:, 10]
    g_r = g
    if m.sigma > 0:
        g_r = jnp.maximum(g * (1.0 + m.sigma * jax.random.normal(jax.random.fold_in(k, 1), (n,))), 0.0)
    has_lr, g_lr, vl_lr = obs_lr[:, 8] > 0.5, 50.0 * obs_lr[:, 9], 15.0 * obs_lr[:, 10]
    use = has & (g <= m.range_m) & ~miss & (~has_lr | (g_r < g_lr))
    cols = TR.leader_cols(v, has_lr | use, jnp.where(use, g_r, g_lr), jnp.where(use, vl, vl_lr), obs.dtype)
    out = obs_lr
    for j, col in cols.items():
        out = out.at[:, j].set(jnp.where(use, col, obs_lr[:, j]))
    return out, Mem(lr2, miss)


def run(m: Fault, **kw):
    """E copies with half AVs driven by the IDM baseline behind the fused view; see `avlab.tracker.run` for arguments."""
    return TR.run(m, perceive_fn=perceive, init_fn=init_mem_for(m), **kw)
