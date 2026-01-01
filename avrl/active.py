"""Active scenario selection: simulate the data the driving policy learns most from, instead of sampling scenarios uniformly.

A grid of 54 scenarios over the same Oakland map (demand x human risk x pedestrian rate x AV share). Every 5 PPO
updates each of the 8 simulated copies is (re)assigned a scenario:

  uniform   uniformly at random (classic domain randomisation)
  active    prioritised level replay (Jiang et al., 2021): each scenario's score is the mean |GAE advantage| per AV
            decision the last time it was simulated (how much the policy still has to learn there); copies are assigned by
            rank-based priority (temperature 0.3), mixed 30 % with staleness so no scenario is forgotten
  critic    uniform, but the first 10 updates only fit the value function (the policy is held at the imitation
            weights), so PPO's first policy steps use meaningful advantages instead of an untrained critic's noise
  focus     failure-focused replay: the same machinery, but a scenario's score is how often the AVs got into trouble
            there (hard brakes, near misses, crashes, red runs per AV-hour), with a sharper temperature (0.1)

Both start from the same imitation policy, get the same number of simulation steps, and every checkpoint is scored in
the same held-out full-dynamics test world as the dynamics ladder.

    python -m avrl.active train uniform     # ~4 min
    python -m avrl.active train active      # ~4 min
    python -m avrl.active train focus       # ~4 min
    python -m avrl.active train critic      # ~4 min
    python -m avrl.active eval              # ~1 min
"""
import itertools
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xnet import network as NW
from xnet import sim as X

from . import ladder as LD
from . import xppo as P

OUT = "results/active"
PERIOD = 5
UPDATES = 120
CKPTS = (15, 30, 60, 120)
CRITIC_WARMUP = 10
GRID = list(itertools.product((0.7, 1.0, 1.35), (1.0, 5.0, 15.0), (0.3, 1.0, 2.0), (0.25, 0.6)))  # demand, risk, peds, AV share


def scenario(i, knobs):
    demand, risk, peds, share = GRID[i]
    return X.Behaviour(**{k: min(v * risk, 0.95) for k, v in knobs.items()}, p_av=share, demand=demand, ped_scale=peds)


def batch(ids, knobs):
    return jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *[scenario(int(i), knobs) for i in ids])


class PLR:
    def __init__(self, n, beta=0.3, rho=0.3, seed=0):
        self.score = np.full(n, np.nan)
        self.last = np.zeros(n)
        self.beta, self.rho = beta, rho
        self.rng = np.random.default_rng(seed)

    def probs(self, t, exclude=()):
        seen = ~np.isnan(self.score)
        rank = np.empty(len(self.score))
        sc = np.where(seen, self.score, -np.inf)
        rank[np.argsort(-sc)] = np.arange(1, len(self.score) + 1)
        h = np.where(seen, (1.0 / rank) ** (1.0 / self.beta), 0.0)
        stale = (t - self.last) * seen
        p = (1 - self.rho) * h / max(h.sum(), 1e-12) + self.rho * stale / max(stale.sum(), 1e-12)
        p[list(exclude)] = 0.0
        return p / p.sum()

    def pick(self, k, t):
        """Unseen scenarios first (each visited once), the rest by priority."""
        unseen = np.nonzero(np.isnan(self.score))[0]
        first = list(self.rng.permutation(unseen)[:k])
        rest = k - len(first)
        if rest:
            first += list(self.rng.choice(len(self.score), rest, replace=False, p=self.probs(t, exclude=first)))
        return np.array(first)

    def update(self, ids, signal, t):
        for i, s in zip(ids, signal):
            self.score[i] = s if np.isnan(self.score[i]) else 0.5 * self.score[i] + 0.5 * s
            self.last[i] = t


def failure(m):
    """Trouble per AV-hour in each copy: hard brakes + 10 x near misses + 50 x (crashes + red runs)."""
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    return (d["av_hard"] + 10 * d["av_near"] + 50 * (d["av_crash"] + d["av_red"])) / np.maximum(d["av_veh_s"] / 3600, 1e-3)


def train(mode, seed=0):
    out = f"{OUT}/{mode}"
    os.makedirs(out, exist_ok=True)
    cfg = P.Config(n_envs=LD.N_ENVS, reward=LD.REWARD)
    X.R = X.Reward(**cfg.reward)
    net, geo, info = NW.build("oakland")
    knobs = LD.knobs()
    rng = np.random.default_rng(seed)
    plr = PLR(len(GRID), beta=0.1 if mode == "focus" else 0.3, seed=seed)
    ids = rng.choice(len(GRID), cfg.n_envs, replace=False)
    beh = batch(ids, knobs)
    key = jax.random.PRNGKey(seed)
    params = P.load(f"{LD.OUT}/bc/policy_bc.npz")
    J = X.to_jnp(net)
    st = jax.vmap(lambda k: X.reset(k, J, cfg.n_slots))(jax.random.split(key, cfg.n_envs))
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(cfg.lr))
    warm, _, update = P.make_fns(cfg, net, beh, opt)
    for i in range(int(150.0 / (X.DT * (cfg.n_steps + 1)))):
        st = warm(params, st, jax.random.fold_in(key, 10_000 + i), beh)
    opt_state = opt.init(params)
    t0 = time.time()
    log, visits = [], np.zeros(len(GRID), int)
    sig_acc, n_acc = np.zeros(cfg.n_envs), 0
    m_acc = np.zeros((cfg.n_envs, X.NM))
    for u in range(UPDATES):
        frozen = mode == "critic" and u < CRITIC_WARMUP
        new, opt_state, st, s = update(params, opt_state, st, jax.random.fold_in(key, u), beh)
        params = {**new, "pi": params["pi"], "log_std": params["log_std"]} if frozen else new
        if mode == "critic" and u + 1 == CRITIC_WARMUP:
            opt_state = opt.init(params)  # drop the policy moments gathered while it was held
        sig_acc += np.asarray(s["signal"])
        m_acc += np.asarray(s["m_env"])
        n_acc += 1
        if (u + 1) % PERIOD == 0:
            plr.update(ids, failure(m_acc) if mode == "focus" else sig_acc / n_acc, u)
            visits[ids] += 1
            ids = plr.pick(cfg.n_envs, u) if mode in ("active", "focus") else rng.choice(len(GRID), cfg.n_envs, replace=False)
            beh = batch(ids, knobs)
            sig_acc, n_acc = np.zeros(cfg.n_envs), 0
            m_acc[:] = 0
        log.append({"update": u, "time_s": round(time.time() - t0, 1), "reward_per_step": float(s["reward_per_step"]),
                    "signal": float(np.mean(np.asarray(s["signal"]))), "scenarios": [int(i) for i in ids]})
        if u + 1 in CKPTS:
            P.save(params, f"{out}/policy_{u + 1}.npz")
            print(f"{mode} update {u + 1}: {time.time() - t0:.0f} s, reward/step {float(s['reward_per_step']):.4f}", flush=True)
    with open(f"{out}/log.json", "w") as f:
        json.dump({"log": log, "visits": visits.tolist(), "plr_score": np.nan_to_num(plr.score).tolist(),
                   "grid": GRID}, f)


def evaluate():
    f = LD.evaluator()
    rows = []
    sim_steps = LD.N_ENVS * (P.Config().n_steps + 1)
    r0, s0 = f(f"{LD.OUT}/bc/policy_bc.npz")
    for mode in ("uniform", "active", "focus", "critic"):
        rows.append({"mode": mode, "updates": 0, "sim_steps": 0, "score": r0, **s0})
        for c in CKPTS:
            r, s = f(f"{OUT}/{mode}/policy_{c}.npz")
            rows.append({"mode": mode, "updates": c, "sim_steps": c * sim_steps, "score": r, **s})
            print(json.dumps({k: rows[-1][k] for k in ("mode", "updates", "score", "av_hard_per_h", "av_shield_share", "av_speed_kmh")}, default=float), flush=True)
    with open(f"{OUT}/scores.json", "w") as fo:
        json.dump(rows, fo, indent=1, default=float)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2])
    else:
        evaluate()
