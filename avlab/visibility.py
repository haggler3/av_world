"""Limited sight distance (night, fog, heavy rain) for the AV in the policy slot, with an optional phantom obstacle at the
edge of sight. `xnet/` is untouched; human drivers still see as before.

Model, per AV row and per 0.2 s step, with sight distance s [m], true leader gap g [m] and leader speed v_l [m/s]:

    seen      = has_leader and g <= s                        (a leader beyond s is invisible)
    no phantom: an unseen leader is shown as "no leader"
    phantom   : an unseen leader, or no leader at all, is shown as a stopped object at gap s
                (the "assured clear distance ahead" rule: never drive faster than you can stop within what you can see)

    from avlab import visibility as VI
    out = VI.run(VI.Visibility(sight_m=25.0, phantom=True), E=16, seconds=600)   # same counters as avlab.tracker.run
"""
from typing import NamedTuple

import jax.numpy as jnp

from avlab import tracker as TR


class Visibility(NamedTuple):
    sight_m: float = 1e9   # sight distance [m]; 1e9 = clear day
    phantom: bool = False  # treat the edge of sight as a stopped obstacle


def init_mem(n):
    return jnp.zeros(n)  # stateless; a placeholder for the policy-memory loop of avlab.tracker.run


def perceive(obs, key, mem, m: Visibility):
    """(observation the policy sees [N, 28], unchanged memory). Only the leader features (8-11, 25, 26) change."""
    del key
    v = 15.0 * obs[:, 0]
    has, g, vl = obs[:, 8] > 0.5, 50.0 * obs[:, 9], 15.0 * obs[:, 10]
    seen = has & (g <= m.sight_m)
    if m.phantom:
        show = jnp.ones_like(seen)
        g2 = jnp.where(seen, g, m.sight_m)
        vl2 = jnp.where(seen, vl, 0.0)
    else:
        show, g2, vl2 = seen, g, vl
    cols = TR.leader_cols(v, show, g2, vl2, obs.dtype)
    out = obs
    for j, col in cols.items():
        out = out.at[:, j].set(jnp.where(seen, obs[:, j], col))  # a visible leader is shown exactly as the simulator gives it
    return out, mem


def run(m: Visibility, **kw):
    """E copies with half AVs driven by the IDM baseline behind the sight limit; see `avlab.tracker.run` for arguments."""
    return TR.run(m, perceive_fn=perceive, init_fn=init_mem, **kw)


def stopping_speed(s, b=4.0, s0=2.0, t_react=0.2):
    """Largest speed [m/s] that can stop within sight s [m]: v t_react + v^2 / (2 b) = s - s0."""
    d = jnp.maximum(jnp.asarray(s, float) - s0, 0.0)
    return -b * t_react + jnp.sqrt((b * t_react) ** 2 + 2.0 * b * d)
