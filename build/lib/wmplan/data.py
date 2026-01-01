"""Experience for the world model: every AV decision in the Oakland simulator, with what happened next.

The behaviour policy is the 234k-step PPO driver with exploration on top: each vehicle slot carries a persistent style
(an acceleration bias and a gap-acceptance bias, from timid to reckless) plus per-step noise. No safety filter, so the
logs contain the consequences of bad merges and junction entries, crashes included. Humans are x3 riskier than
calibrated, with cyclists, so conflicts are frequent.

Per AV and step (rows sorted by copy, slot, time): obs [28], action u [2], reward, the five safety margins (barrier
values: stop target, leader, junction time-to-conflict, merge gap behind, merge gap ahead), crash, near miss, red-light
run, done.

    python -m wmplan.data          # ~3 min -> results/wmplan/data.npz (~600 k decisions)
"""
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from avrl import ladder as LD
from avrl import xppo as P
from xnet import network as NW
from xnet import sim as X

OUT = "results/wmplan"
POLICY = "results/xnet3/policy.npz"
N_ENVS, WARM, SECONDS, CHUNK = 8, 120.0, 150.0, 25


def world():
    knobs = {k: min(v * 3.0, 0.95) for k, v in LD.knobs().items()}
    return X.Behaviour(**(knobs | dict(p_av=0.3, p_bike=0.08, demand=1.15)))


def explorer(params, noise=0.35):
    slot = jnp.arange(1024)
    style = jnp.stack([0.45 * jnp.sin(slot * 2.39), 0.9 * jnp.sin(slot * 1.71 + 0.5)], 1)  # accel bias, gap-acceptance bias

    def f(obs, key):
        mu = P.mean(params, obs)
        return mu + style[: obs.shape[0]] + noise * jax.random.normal(key, mu.shape), jnp.zeros((obs.shape[0], 2))
    return f


def collect(seed=0, seconds=SECONDS):
    os.makedirs(OUT, exist_ok=True)
    net, geo, info = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = world()
    sim = X.Sim(net, n_envs=N_ENVS, beh=beh, seed=seed, reroute_s=0)
    sim.run(WARM)
    pol = explorer(P.load(POLICY))
    X.SHIELD = "off"

    @jax.jit
    def chunk(st, key):
        def body(st, k):
            st, tr = jax.vmap(lambda s, kk: X.step(s, J, beh, nh, rl=(pol, kk)))(st, jax.random.split(k, N_ENVS))
            keep = {k_: tr[k_] for k_ in ("obs", "u", "reward", "margins", "crash", "near", "red", "done", "valid")}
            return st, jax.tree.map(lambda x: x.astype(jnp.float16) if x.dtype == jnp.float32 else x, keep)
        return jax.lax.scan(body, st, jax.random.split(key, CHUNK))

    st, rows, t0 = sim.st, [], time.time()
    key = jax.random.PRNGKey(seed + 100)
    for c in range(int(seconds / X.DT / CHUNK)):
        st, tr = chunk(st, jax.random.fold_in(key, c))
        tr = jax.tree.map(np.asarray, tr)
        t_idx, e_idx, s_idx = np.nonzero(tr["valid"])
        rows.append({k: v[t_idx, e_idx, s_idx] for k, v in tr.items() if k != "valid"} |
                    {"t": (t_idx + c * CHUNK).astype(np.int32), "env": e_idx.astype(np.int16), "slot": s_idx.astype(np.int16)})
    X.SHIELD = "idm"
    d = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    o = np.lexsort((d["t"], d["slot"], d["env"]))
    d = {k: v[o] for k, v in d.items()}
    np.savez_compressed(f"{OUT}/data.npz", **d)
    print(f"{len(d['t'])} AV decisions in {time.time() - t0:.0f} s; crashes {int(d['crash'].sum())}, near misses {int(d['near'].sum())}, "
          f"red runs {int(d['red'].sum())}, done {int(d['done'].sum())}", flush=True)


if __name__ == "__main__":
    collect(seconds=float(sys.argv[1]) if len(sys.argv) > 1 else SECONDS)
