"""AV perception latency in the policy slot: the policy sees its leader as it was `delay_s` ago, optionally with a
constant-velocity forward prediction (a Smith-predictor-style compensation). `xnet/` is untouched.

Memory per AV row (passed through `aux`, as in `avlab.tracker`): a ring of the last D = delay_s / DT leader observations
(has_leader, gap g, leader speed v_l) and of the AV's own speed v. Own speed is taken as current (odometry has no delay).

    at step t, with the ring holding steps t-D .. t-1:
        shown leader   = (has, g, v_l) from step t-D
        compensate     : g_hat = g_{t-D} - DT * sum_{j=t-D}^{t-1} (v_j - v_l,{t-D})     (leader assumed at constant speed)
        else           : g_hat = g_{t-D}
        then push (has, g, v_l, v) of step t and drop step t-D
    a row whose memory is empty (a new AV, or a reused slot) fills its whole ring with the current observation

    from avlab import latency as LT
    out = LT.run(LT.Latency(delay_s=0.6, compensate=True), E=16, seconds=600)   # same counters as avlab.tracker.run
"""
from typing import NamedTuple

import jax.numpy as jnp

from avlab import tracker as TR

DT = TR.DT


class Latency(NamedTuple):
    delay_s: float = 0.0      # perception delay on the leader [s]; rounded to whole 0.2 s steps
    compensate: bool = False  # predict the delayed gap forward with own-speed history and the delayed leader speed


class Mem(NamedTuple):
    has: jnp.ndarray   # [D, N] bool
    g: jnp.ndarray     # [D, N] gap [m]
    vl: jnp.ndarray    # [D, N] leader speed [m/s]
    v: jnp.ndarray     # [D, N] own speed [m/s]
    full: jnp.ndarray  # [N] bool, ring holds real observations


def steps(m: Latency):
    return int(round(m.delay_s / DT))


def init_mem_for(m: Latency):
    D = max(steps(m), 1)

    def init(n):
        z = jnp.zeros((D, n))
        return Mem(jnp.zeros((D, n), bool), z, z, z, jnp.zeros(n, bool))
    return init


def perceive(obs, key, mem: Mem, m: Latency):
    """(observation with the delayed leader [N, 28], new memory). Delay 0 returns obs unchanged."""
    del key
    D = steps(m)
    if D == 0:
        return obs, mem
    v = 15.0 * obs[:, 0]
    has, g, vl = obs[:, 8] > 0.5, 50.0 * obs[:, 9], 15.0 * obs[:, 10]
    fill = ~mem.full
    rh = jnp.where(fill, has, mem.has)
    rg = jnp.where(fill, g, mem.g)
    rvl = jnp.where(fill, vl, mem.vl)
    rv = jnp.where(fill, v, mem.v)
    h0, g0, vl0 = rh[0], rg[0], rvl[0]
    g_show = g0 - DT * jnp.sum(rv - vl0[None], axis=0) if m.compensate else g0
    g_show = jnp.where(fill, g, g_show)  # a fresh ring holds only the current observation: show it as it is
    cols = TR.leader_cols(v, h0, g_show, vl0, obs.dtype)
    out = obs
    for j, col in cols.items():
        out = out.at[:, j].set(col)
    roll = lambda r, x: jnp.concatenate([r[1:], x[None]], axis=0)  # noqa: E731
    return out, Mem(roll(rh, has), roll(rg, g), roll(rvl, vl), roll(rv, v), jnp.ones_like(mem.full))


def run(m: Latency, **kw):
    """E copies with half AVs driven by the IDM baseline behind the latency model; see `avlab.tracker.run` for arguments."""
    r = TR.run(m, perceive_fn=perceive, init_fn=init_mem_for(m), **kw)
    return r
