"""A phantom obstacle that remembers the leader it last saw (follow-up to `avlab.visibility`). `xnet/` is untouched.

`avlab.visibility` shows a *stopped* phantom at the edge of sight s whenever no leader is seen. That is safe but slow
(7.0 km/h at s = 8 m vs 14.3 km/h on a clear day). When the AV has just lost sight of a leader that was moving at v_last, a
stopped phantom is too pessimistic: a leader that brakes no harder than b_ph is still moving at least
v_last - b_ph * t after t seconds, and it is farther than s (else it would be seen). So, per AV row and per 0.2 s step:

    seen  = has_leader and g <= s
    if seen:   show the leader as it is;  memory <- (v_last = v_l, age = 0)
    else:      age += 1;  v_ph = max(0, v_last - b_ph * age * DT)   (v_ph = 0 if no leader was ever seen)
               show a phantom at gap s moving at v_ph

With b_ph = 9 m/s^2 (the hardest braking the simulator allows at friction 1, `xnet/sim.py` a_min) the phantom is a sound
lower bound on the hidden leader's speed: the real leader is farther and at least as fast. Smaller b_ph is an optimistic
guess. A row whose vehicle is not an alive AV has its memory cleared by `avlab.tracker.run`.

    from avlab import sightmemory as SM
    out = SM.run(SM.SightMemory(sight_m=8.0, b_ph=9.0), E=16, seconds=600)   # same counters as avlab.tracker.run
"""
from typing import NamedTuple

import jax.numpy as jnp

from avlab import tracker as TR

DT = TR.DT
NEVER = 10 ** 6


class SightMemory(NamedTuple):
    sight_m: float = 8.0  # sight distance [m]
    b_ph: float = 9.0     # assumed hardest braking of a hidden leader [m/s^2]; inf = stopped phantom (avlab.visibility)


class Mem(NamedTuple):
    vl: jnp.ndarray   # [N] last seen leader speed [m/s]
    age: jnp.ndarray  # [N] steps since that leader was last seen (NEVER = no leader seen)


def init_mem(n):
    return Mem(jnp.zeros(n), jnp.full(n, NEVER, jnp.int32))


def phantom_speed(vl_last, age, b_ph):
    """Lower bound on a hidden leader's speed [m/s] after `age` steps unseen, if it brakes no harder than b_ph."""
    return jnp.where(age >= NEVER, 0.0, jnp.maximum(vl_last - b_ph * age * DT, 0.0))


def perceive(obs, key, mem: Mem, m: SightMemory):
    """(observation the policy sees [N, 28], new memory). Only the leader features (8-11, 25, 26) change."""
    del key
    v = 15.0 * obs[:, 0]
    has, g, vl = obs[:, 8] > 0.5, 50.0 * obs[:, 9], 15.0 * obs[:, 10]
    seen = has & (g <= m.sight_m)
    age = jnp.where(seen, 0, jnp.minimum(mem.age + 1, NEVER))
    vlast = jnp.where(seen, vl, mem.vl)
    vph = phantom_speed(vlast, age, m.b_ph)
    cols = TR.leader_cols(v, jnp.ones_like(seen), jnp.where(seen, g, m.sight_m), jnp.where(seen, vl, vph), obs.dtype)
    out = obs
    for j, col in cols.items():
        out = out.at[:, j].set(jnp.where(seen, obs[:, j], col))
    return out, Mem(vlast, age.astype(jnp.int32))


def run(m: SightMemory, **kw):
    """E copies with half AVs driven by the IDM baseline behind the sight limit; see `avlab.tracker.run` for arguments."""
    return TR.run(m, perceive_fn=perceive, init_fn=init_mem, **kw)
