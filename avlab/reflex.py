"""A short-range, zero-delay "reflex" sensor fused with a delayed long-range view of the leader. `xnet/` is untouched.

Long-range view: `avlab.latency` (a ring of delayed leader observations, optional forward prediction).
Reflex: a short-range sensor with no delay (think ultrasonic or short-range radar) that sees the true leader when it
is within `range_m` metres. Fusion keeps the more cautious of the two gaps:

    obs_lr, mem = latency.perceive(obs_true, mem)              # the delayed (and predicted) view
    near        = true leader exists and true gap g <= range_m
    use_reflex  = near and (no leader shown by obs_lr or g < shown gap)
    shown       = (true gap g, true leader speed) where use_reflex, else obs_lr's leader

With range_m = 0 the AV sees exactly what `avlab.latency` shows.

    from avlab import reflex as RF
    out = RF.run(RF.Reflex(delay_s=2.0, compensate=True, range_m=10.0), E=16, seconds=600)
"""
from typing import NamedTuple

import jax.numpy as jnp

from avlab import latency as LT
from avlab import tracker as TR


class Reflex(NamedTuple):
    delay_s: float = 2.0      # delay of the long-range view [s]
    compensate: bool = True   # forward prediction of the delayed view (as avlab.latency)
    accel: bool = False       # constant-acceleration prediction (as avlab.latency)
    range_m: float = 10.0     # reach of the zero-delay reflex sensor [m]; 0 turns it off


def latency_of(m: Reflex):
    return LT.Latency(m.delay_s, m.compensate, m.accel)


def init_mem_for(m: Reflex):
    return LT.init_mem_for(latency_of(m))


def fuse(obs_true, obs_lr, range_m):
    """(fused observation, bool [N] rows where the reflex overrode the long-range view)."""
    v = 15.0 * obs_true[:, 0]
    has, g, vl = obs_true[:, 8] > 0.5, 50.0 * obs_true[:, 9], 15.0 * obs_true[:, 10]
    has_lr, g_lr, vl_lr = obs_lr[:, 8] > 0.5, 50.0 * obs_lr[:, 9], 15.0 * obs_lr[:, 10]
    use = has & (g <= range_m) & (~has_lr | (g < g_lr))
    cols = TR.leader_cols(v, has_lr | use, jnp.where(use, g, g_lr), jnp.where(use, vl, vl_lr), obs_true.dtype)
    out = obs_lr
    for j, col in cols.items():
        out = out.at[:, j].set(jnp.where(use, col, obs_lr[:, j]))
    return out, use


def perceive(obs, key, mem, m: Reflex):
    obs_lr, mem2 = LT.perceive(obs, key, mem, latency_of(m))
    if m.range_m <= 0:
        return obs_lr, mem2
    return fuse(obs, obs_lr, m.range_m)[0], mem2


def run(m: Reflex, **kw):
    """E copies with half AVs driven by the IDM baseline behind the fused view; see `avlab.tracker.run` for arguments."""
    return TR.run(m, perceive_fn=perceive, init_fn=init_mem_for(m), **kw)
