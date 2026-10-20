"""Three ways to make the simulator teach harder, as options of one PPO trainer (no safety filter; the barrier function
is only the teacher, as in avrl.unaided):

  fork     Fork at the moment of danger. After every update, the copies that just lived through trouble (an AV inside its
           barrier, a near miss, a red run, a crash) have their *pre-update* state saved to a buffer, ranked by how bad it
           was. Every update, half of the copies (2 buffer states x 2 branches) are overwritten with buffer states and
           replayed with new random numbers: the same dangerous situation, two different futures under the current
           policy. State is one pytree, so a fork is a copy.
  shape    Crash-risk shaping. Potential-based shaping from the latent world model's crash head (wmplan.risk): a step that
           drives the AV into a scene the model rates dangerous is charged when it happens, not at the impact.
  mutate   Mutating cities. Every 10 updates each copy gets a different city (xnet.mutate: speed limits, signal plans,
           demand and pedestrian hot spots, closed streets with rerouting), from a pool of 48. Fork states carry their city.

    python -m avrl.fork train fork+shape 0     # ~5 min -> results/dyn/fork+shape_s0/policy_{25,50,100}.npz
    python -m avrl.fork train fork+shape 0 early   # the first 15 updates again, saving policy_{5,10,15}.npz (~1 min)
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from wmplan import risk as RK
from xnet import mutate as MU
from xnet import network as NW
from xnet import sim as X

from . import ladder as LD
from . import xppo as P

OUT = "results/dyn"
START = "results/xnet3/policy.npz"
REWARD = LD.REWARD | {"barrier": 0.3, "inside": 0.05, "veto": 1.0}  # the barrier function as the teacher
IDX = {k: i for i, k in enumerate(X.METRICS)}
UPDATES, CKPTS, EARLY = 100, (25, 50, 100), (5, 10, 15)  # EARLY: a 15-update rerun of the same seed (it is deterministic)
POOL, REDRAW = 48, 10
BUFFER, SNAPSHOTS, BRANCHES = 48, 2, 2
VARIANTS = {"base": {}, "shape": dict(shape=10.0), "fork": dict(fork=True), "fork+shape": dict(fork=True, shape=10.0),
            "mutate": dict(mutate=True), "all": dict(fork=True, shape=10.0, mutate=True)}


def danger(m):
    """How bad the last window was for each copy [E]: crashes, near misses, red runs, seconds of AV time inside the barrier."""
    return 20 * m[:, IDX["av_crash"]] + 10 * m[:, IDX["av_near"]] + 10 * m[:, IDX["av_red"]] + m[:, IDX["av_h_neg"]] / 5.0


def set_env(st_np, e, snap):
    for a, b in zip(jax.tree.leaves(st_np), jax.tree.leaves(snap)):
        a[e] = b


def train(name, seed=0, updates=UPDATES, ckpts=CKPTS):
    v = VARIANTS[name]
    out = f"{OUT}/{name}_s{seed}"
    os.makedirs(out, exist_ok=True)
    cfg = P.Config(n_envs=LD.N_ENVS, reward=REWARD, warm=150.0)
    E = cfg.n_envs
    X.R = X.Reward(**cfg.reward)
    X.SHIELD = "off"
    net, geo, info = NW.build("oakland")
    J = X.to_jnp(net)
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)
    params = P.load(START)
    beh = LD.world("5")
    cities = MU.pool(net, geo, POOL) if v.get("mutate") else None
    city = [None] * E if cities is None else [cities[i] for i in rng.choice(POOL, E, replace=False)]
    shape = RK.potential(lam=v["shape"]) if v.get("shape") else None
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(cfg.lr))
    warm, _, update = P.make_fns(cfg, net, beh, opt, shape)

    def inputs():
        return {"beh": beh, **MU.stack(city)} if cities is not None else beh

    fresh = jax.vmap(lambda k: X.reset(k, J, cfg.n_slots))
    st = fresh(jax.random.split(key, E))
    n_warm = int(cfg.warm / (X.DT * (cfg.n_steps + 1)))
    det = P.deterministic
    for i in range(n_warm):
        st = warm(params, st, jax.random.fold_in(key, 10_000 + i), inputs())
    opt_state = opt.init(params)
    fork_envs = list(range(E - SNAPSHOTS * BRANCHES, E)) if v.get("fork") else []
    normal = [e for e in range(E) if e not in fork_envs]
    buf, log, t0, n_forks = [], [], time.time(), 0
    for i in range(updates):
        if cities is not None and i > 0 and i % REDRAW == 0:  # new cities, fresh copies, a short warm-up
            for e in normal:
                city[e] = cities[int(rng.integers(POOL))]
            sel = jnp.asarray([e in normal for e in range(E)])
            new = fresh(jax.random.split(jax.random.fold_in(key, 40_000 + i), E))
            st = jax.tree.map(lambda a, b: jnp.where(sel.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), st, new)
            for j in range(5):
                st = warm(params, st, jax.random.fold_in(key, 50_000 + 10 * i + j), inputs())
        elif cities is None and i > 0 and i % cfg.reset_every == 0:  # restart a quarter of the normal copies
            sel = jnp.asarray([(e in normal) and (e % 4 == (i // cfg.reset_every) % 4) for e in range(E)])
            new = fresh(jax.random.split(jax.random.fold_in(key, 30_000 + i), E))
            st = jax.tree.map(lambda a, b: jnp.where(sel.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), st, new)
        if fork_envs and len(buf) >= 8:  # overwrite the fork copies with saved dangerous states, new random numbers
            sc = np.array([b["score"] / (1 + b["used"]) for b in buf])
            pick = rng.choice(len(buf), SNAPSHOTS, replace=False, p=sc / sc.sum())
            st_np = jax.tree.map(np.array, st)
            rows = {f: np.array(getattr(beh, f)) for f in beh._fields}
            for j, bi in enumerate(pick):
                buf[bi]["used"] += 1
                for b_ in range(BRANCHES):
                    e = fork_envs[j * BRANCHES + b_]
                    set_env(st_np, e, buf[bi]["state"])
                    st_np.key[e] = np.asarray(jax.random.fold_in(jnp.asarray(buf[bi]["state"].key), 7919 * i + e))
                    for f in beh._fields:
                        rows[f][e] = buf[bi]["beh"][f]
                    if cities is not None:
                        city[e] = buf[bi]["city"]
            st = jax.tree.map(jnp.asarray, st_np)
            beh = type(beh)(**{f: jnp.asarray(rows[f]) for f in beh._fields})
            n_forks += SNAPSHOTS * BRANCHES
        before = st
        params, opt_state, st, s = update(params, opt_state, st, jax.random.fold_in(key, i), inputs())
        if fork_envs:  # save the pre-update state of the copies that met the most trouble
            score = danger(np.asarray(s["m_env"]))
            for e in np.argsort(-score)[:2]:
                if score[e] <= 0:
                    continue
                entry = {"score": float(score[e]), "used": 0, "state": jax.tree.map(lambda a: np.asarray(a[e]), before),
                         "beh": {f: float(getattr(beh, f)[e]) for f in beh._fields}, "city": city[e]}
                if len(buf) < BUFFER:
                    buf.append(entry)
                else:
                    w = min(range(len(buf)), key=lambda q: buf[q]["score"])
                    if entry["score"] > buf[w]["score"]:
                        buf[w] = entry
        row = {"update": i, "time_s": round(time.time() - t0, 1), "reward_per_step": float(s["reward_per_step"]),
               "buffer": len(buf), "forked_copies": n_forks, **P.summarize(s["m"])}
        log.append(row)
        if i + 1 in ckpts:
            P.save(params, f"{out}/policy_{i + 1}.npz")
        if i % 10 == 0 or i == updates - 1:
            print(json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in row.items()}, default=float), flush=True)
    json.dump(log, open(f"{out}/train_log{'' if updates == UPDATES else '_' + str(updates)}.json", "w"), default=float)
    X.SHIELD = "idm"
    print(f"{name} seed {seed}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    early = len(sys.argv) > 4 and sys.argv[4] == "early"
    train(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0, 15 if early else UPDATES, EARLY if early else CKPTS)
