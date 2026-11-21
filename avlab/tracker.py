"""Stateful AV perception in the policy slot: bursty (Markov) missed detections and a hold-last-detection tracker.

`avlab.sensornoise` draws every error independently per 0.2 s step, because the policy slot of `xnet.sim.step` carries no
memory. This module adds memory without touching `xnet/`: the policy returns its new memory as `aux`, `step` hands `aux` back
in the transition, and the driving loop (`run`) feeds it into the next step's policy. Rows whose vehicle is not an alive AV
have their memory cleared, so a slot that is reused by a new vehicle starts empty.

Model, per AV row and per step (noise key `fold_in(step key, SALT)`, separate from the simulator's draws):

    miss_t   two-state Markov chain: P(miss_t | miss_{t-1}) = 1 - 1 / L,  P(miss_t | not miss_{t-1}) = q,
             L = burst_s / DT (mean burst length in steps), q chosen so the stationary miss share is p_miss:
             q = p_miss / (L (1 - p_miss)).   burst_s = DT gives independent misses (q = p_miss, as sensornoise).
    detected = has_leader and not miss_t
    tracker  on a detection: store (g, v_lead) and age 0. On a miss with age < hold_s / DT: predict the gap by constant
             velocity, g <- g - (v - v_lead) DT, age += 1, and show the prediction as a detection. Otherwise: no leader.

    from avlab import tracker as TR
    m = TR.Perception(p_miss=0.15, burst_s=1.0, hold_s=1.0)
    out = TR.run(m, E=16, seconds=600)          # per-copy AV crashes, AV-seconds, ... (envelope off by default)
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

SALT = 0x7AC4
DT = 0.2


class Perception(NamedTuple):
    p_miss: float = 0.0   # stationary share of steps with the leader missed
    burst_s: float = DT   # mean length of a run of misses [s]; DT = independent per step
    hold_s: float = 0.0   # tracker: keep predicting a lost leader for this long [s]; 0 = no tracker


class Mem(NamedTuple):
    miss: jnp.ndarray  # [N] bool, Markov state
    g: jnp.ndarray     # [N] tracked gap [m]
    vl: jnp.ndarray    # [N] tracked leader speed [m/s]
    age: jnp.ndarray   # [N] steps since the last detection (large = no track)


def init_mem(n):
    return Mem(jnp.zeros(n, bool), jnp.full(n, 150.0), jnp.zeros(n), jnp.full(n, 10 ** 6, jnp.int32))


def onset_prob(m: Perception):
    """q = P(miss | no miss before); the stationary share q / (q + 1/L) equals p_miss."""
    L = max(m.burst_s / DT, 1.0)
    if m.p_miss <= 0:
        return 0.0
    return min(m.p_miss / (L * (1.0 - m.p_miss)), 1.0)


def perceive(obs, key, mem: Mem, m: Perception):
    """(observation the policy sees [N, 28], new memory). Only the leader features (8-11, 25, 26) change."""
    n = obs.shape[0]
    L = max(m.burst_s / DT, 1.0)
    q = onset_prob(m)
    u = jax.random.uniform(jax.random.fold_in(key, SALT), (n,))
    miss = jnp.where(mem.miss, u < 1.0 - 1.0 / L, u < q)
    v = 15.0 * obs[:, 0]
    has = obs[:, 8] > 0.5
    gap, vl = 50.0 * obs[:, 9], 15.0 * obs[:, 10]
    det = has & ~miss
    hold_n = int(round(m.hold_s / DT))
    pred_g = mem.g - (v - mem.vl) * DT
    held = ~det & (mem.age < hold_n)
    g_new = jnp.where(det, gap, jnp.where(held, pred_g, 150.0))
    vl_new = jnp.where(det, vl, jnp.where(held, mem.vl, 0.0))
    age = jnp.where(det, 0, jnp.minimum(mem.age, 10 ** 6 - 1) + 1)
    seen = det | held
    f = seen.astype(obs.dtype)
    closing = jnp.clip((v - vl_new) / 10.0, -2.0, 2.0) * f
    dec = jnp.clip(jnp.maximum(v - vl_new, 0.0) ** 2 / (2.0 * jnp.maximum(g_new - 2.0, 0.5)) / 8.0, 0.0, 2.0) * f
    hw = jnp.clip(g_new / jnp.maximum(v, 1.0) / 3.0, 0.0, 2.0) * f
    new = {8: f, 9: jnp.where(seen, jnp.clip(g_new / 50.0, -0.1, 3.0), 3.0), 10: jnp.where(seen, vl_new / 15.0, 0.0),
           11: closing, 25: dec, 26: hw}
    out = obs
    for j, col in new.items():
        # a detected leader is shown as it is; a missed or held one replaces the features; with no tracker and no miss, unchanged
        out = out.at[:, j].set(jnp.where(det, obs[:, j], col))
    return out, Mem(miss, g_new, vl_new, age)


def stateful(policy, m: Perception):
    """policy(obs, key, mask) -> (u, aux) behind the perception model; returns f(obs, key, mask, mem) -> (u, new mem)."""
    def f(obs, key, mask, mem):
        o, mem2 = perceive(obs, key, mem, m)
        u, _ = policy(o, key, mask)
        return u, mem2
    return f


def run(m: Perception, E=16, seed=43, warm_s=120.0, seconds=600.0, chunk_s=40.0, p_av=0.5, friction=1.0, shield="off",
        area="oakland", n_slots=1024):
    """E copies of `area` with half AVs driven by the IDM baseline behind `m`. Returns per-copy counters after the warm-up."""
    from examples import _common as C
    from avlab import teb as T
    from xnet import network as NW
    from xnet import sim as X
    X.SHIELD = shield  # read at trace time
    net, _, _ = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated(friction=friction, p_av=p_av)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for i in range(E)]), J, n_slots)
    mem = jax.vmap(lambda _: init_mem(n_slots))(jnp.arange(E))
    f = stateful(T.idm_policy_masked, m)
    n = int(round(chunk_s / X.DT))

    def one(si, mi):
        s2, tr = X.step(si, J, beh, nh, rl=(lambda o, k, mask: f(o, k, mask, mi), si.key, True))
        keep = s2.alive & s2.av
        mi2 = jax.tree.map(lambda new, z: jnp.where(keep, new, z), tr["aux"], init_mem(n_slots))
        return s2, mi2

    @jax.jit
    def chunk(c):
        return jax.lax.scan(lambda c, _: (jax.vmap(one)(*c), None), c, None, length=n)[0]

    try:
        c = (st, mem)
        for _ in range(int(round(warm_s / chunk_s))):
            c = chunk(c)
        m0 = np.asarray(c[0].m)
        for _ in range(int(round(seconds / chunk_s))):
            c = chunk(c)
    finally:
        X.SHIELD = "idm"
    d = np.asarray(c[0].m) - m0
    keep = ("av_crash", "av_veh_s", "av_veh_m", "av_hard", "av_shield", "veh_s")
    return {"model": m._asdict(), "copies": E, "seed": seed, "warm_s": warm_s, "seconds": seconds, "p_av": p_av,
            "friction": friction, "shield": shield, **{k: d[:, X.METRICS.index(k)].tolist() for k in keep},
            "crashes": sum(d[:, X.METRICS.index(k)] for k in X.CRASH_NAMES).tolist()}
