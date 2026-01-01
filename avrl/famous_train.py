"""Does training on famous long-tail events help? Three PPO runs from the same start policy, same compute, no safety filter
(the barrier function is only the teacher, as in avrl.unaided):

  A  nominal   every copy is the ladder's randomised world (humans x1 / x5 / x15, AV share, demand)
  B  famous    half of the copies are that world, half are famous-event scenarios (avlab.famous), redrawn every 10 updates
  C  stress    every copy has everything bad at once (the control: the stress without the named structure)

    python -m avrl.famous_train A      # ~5 min each -> results/famous/policy_A.npz
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from av_world import data_path
from avlab import famous as FM
from xnet import network as NW
from xnet import sim as X

from . import ladder as LD
from . import xppo as P

OUT = FM.OUT
START = data_path("policy.npz")
REWARD = LD.REWARD | {"barrier": 0.3, "inside": 0.05, "veto": 1.0}
E, UPDATES, REDRAW = 16, 100, 10


def stack(rows):
    return jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *rows)


def row(beh, e):
    return X.Behaviour(**{f: float(getattr(beh, f)[e]) for f in beh._fields})


def train(tag, seed=0, updates=UPDATES):
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)
    cfg = P.Config(n_envs=E, reward=REWARD, warm=100.0, compact=0.25)
    X.R = X.Reward(**REWARD)
    X.SHIELD = "off"
    net, _, _ = NW.build("oakland")
    J = X.to_jnp(net)
    key = jax.random.PRNGKey(seed)
    nominal = P.behaviours(E, seed, LD.knobs())
    n_fam = E // 2 if tag == "B" else 0
    log = []

    def current():
        if tag == "A":
            return nominal
        rows = [row(nominal, e) for e in range(E - n_fam)]
        if tag == "C":
            return stack([FM.behaviour(FM.GENERIC)] * E)
        return stack(rows + [FM.behaviour(kw) for kw in fam])

    fam = []
    if tag == "B":
        fam = [FM.draw(str(rng.choice(list(FM.SCENARIOS))), rng, "train")[0] for _ in range(n_fam)]
    beh = current()
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(cfg.lr))
    warm, _, update = P.make_fns(cfg, net, beh, opt)
    params = P.load(START)
    opt_state = opt.init(params)
    fresh = jax.vmap(lambda k: X.reset(k, J, cfg.n_slots))
    st = fresh(jax.random.split(key, E))
    for i in range(int(cfg.warm / (X.DT * (cfg.n_steps + 1))) + 1):
        st = warm(params, st, jax.random.fold_in(key, 10_000 + i), beh)
    t0 = time.time()
    for i in range(updates):
        if tag == "B" and i > 0 and i % REDRAW == 0:  # new famous scenarios in the second half, fresh copies, a short warm-up
            fam = [FM.draw(str(rng.choice(list(FM.SCENARIOS))), rng, "train")[0] for _ in range(n_fam)]
            beh = current()
            sel = jnp.asarray([e >= E - n_fam for e in range(E)])
            new = fresh(jax.random.split(jax.random.fold_in(key, 40_000 + i), E))
            st = jax.tree.map(lambda a, b: jnp.where(sel.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), st, new)
            for j in range(3):
                st = warm(params, st, jax.random.fold_in(key, 50_000 + 10 * i + j), beh)
        elif i > 0 and i % cfg.reset_every == 0:  # restart a quarter of the copies
            sel = jnp.asarray([(e % 4) == (i // cfg.reset_every) % 4 for e in range(E)])
            new = fresh(jax.random.split(jax.random.fold_in(key, 30_000 + i), E))
            st = jax.tree.map(lambda a, b: jnp.where(sel.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), st, new)
        params, opt_state, st, s = update(params, opt_state, st, jax.random.fold_in(key, i), beh)
        row_ = {"update": i, "time_s": round(time.time() - t0, 1), "reward_per_step": float(s["reward_per_step"]), **P.summarize(s["m"])}
        log.append(row_)
        if i % 10 == 0 or i == updates - 1:
            print(json.dumps({k: (round(float(x), 3) if isinstance(x, (float, np.floating)) else x) for k, x in row_.items()}, default=float), flush=True)
    P.save(params, f"{OUT}/policy_{tag}{'' if seed == 0 else seed}.npz")
    json.dump(log, open(f"{OUT}/train_log_{tag}{'' if seed == 0 else seed}.json", "w"), default=float)
    X.SHIELD = "idm"
    print(f"{tag}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    train(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
