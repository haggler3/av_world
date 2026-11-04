"""Idea 2: stop the planner (or the actor) from exploiting the world model's mistakes.

v3 (wmplan/v3.py) trained a policy inside the learned model and "the actor-critic exploited the model". Any optimiser pointed at a
learned model finds the actions where the model is most wrong in its favour. Three remedies from the model-based RL literature,
written as pure functions of an ensemble's per-step disagreement s_t and the imagined reward r_t:

  naive      J = sum_t g^t r_t
  MOPO       J = sum_t g^t (r_t - lam s_t)                       pessimism: price the model's uncertainty (Yu et al. 2020)
  WIMLE      J = sum_t g^t w_t r_t,  w_t = exp(-sum_{j<=t} s_j / tau)   trust a step less the less the model knows (Aubret et al. 2026)
  truncate   J = sum_{t < T*} g^t r_t,  T* = first t with s_t > s_max     stop imagining where the ensemble disagrees

`exploitation_gap` is the number that matters: imagined return minus the return the same actions earn in the real environment.

The experiment (`python -m wmplan.conservative`, ~6 min on 4 cores): an ensemble is fitted to a cautious, noisy controller in the
corridor environment (avrl/env.py), then a cross-entropy planner picks 2 s open-loop action sequences inside the model from 256
real states. The plans are executed in the real environment with the safety shield OFF. Reward on both sides is the same function of
the observation (progress, headway, gap, collision), so the gap between imagined and real return is model error alone.
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from avrl import env as E
from avrl import ppo
from avrl import worldmodel as W

OUT = "results/conservative"
GAMMA = 0.97
H = 20
PROGRESS_SCALE = 25.0  # a greedy task: distance is worth 25x the env's own weight, so the planner is tempted by what the model gets wrong
N_TRAIN = 20_000  # transitions the model sees: few, from a cautious controller, so the planner's aggressive plans are off its data


# ---------------------------------------------------------------- the four objectives
def discount(h):
    return GAMMA ** jnp.arange(h)


def j_naive(r, s, **_):
    return (discount(r.shape[-1]) * r).sum(-1)


def j_mopo(r, s, lam=1.0, **_):
    return (discount(r.shape[-1]) * (r - lam * s)).sum(-1)


def trust_weights(s, tau=0.5):
    """w_t = exp(-cumulative disagreement / tau): 1 while the ensemble agrees, falling as it drifts."""
    return jnp.exp(-jnp.cumsum(s, -1) / tau)


def j_wimle(r, s, tau=0.5, **_):
    return (discount(r.shape[-1]) * trust_weights(s, tau) * r).sum(-1)


def keep_mask(s, s_max):
    """1 up to (not including) the first step whose disagreement exceeds s_max, 0 after."""
    return (jnp.cumsum((s > s_max).astype(jnp.float32), -1) == 0).astype(jnp.float32)


def j_truncate(r, s, s_max=0.5, **_):
    return (discount(r.shape[-1]) * keep_mask(s, s_max) * r).sum(-1)


OBJECTIVES = {"naive": j_naive, "mopo": j_mopo, "wimle": j_wimle, "truncate": j_truncate}


def exploitation_gap(imagined, real):
    """Mean over plans of (imagined return - real return): positive = the model flatters the plans it picked."""
    return float(np.mean(np.asarray(imagined) - np.asarray(real)))


# ---------------------------------------------------------------- reward on observations, shared by the model and the world
def rhat(o):
    """The env's reward terms that are functions of the observation: progress, headway, gap, collision. o [..., 22]."""
    v, seen = o[..., 0] * E.V_SCALE, o[..., 10] > 0.5
    gap, thw = o[..., 11] * 50.0, o[..., 20] * 3.0
    return (PROGRESS_SCALE * E.W_PROGRESS * v * E.DT - E.W_HEADWAY * jnp.maximum(0.8 - thw, 0.0) * (v > 1.0) * seen
            - E.W_GAP * jnp.maximum(2.5 - gap, 0.0) * seen + E.R_COLLISION * (seen & (gap < 0.0)))


# ---------------------------------------------------------------- the model, with its disagreement
def ensemble_step(model, norm, o, u):
    """Mean next observation (as worldmodel.assemble) and s = mean over continuous features of the members' SD of the
    predicted change, in units of one-step-change SD."""
    x = jnp.concatenate([o, u[..., None]], -1)
    dcs = jnp.stack([ppo.mlp(m, x)[..., :len(W.CONT)] for m in model])
    s = dcs.std(0).mean(-1)
    return W.assemble(o, *W.predict(model, norm, o, u)), s


def imagine(model, norm, o0, us):
    """o0 [22], us [H]: the imagined rewards r [H] and disagreements s [H] and the imagined observations."""
    def body(o, u):
        o2, s = ensemble_step(model, norm, o, u)
        return o2, (rhat(o2), s, o2)

    _, (r, s, traj) = jax.lax.scan(body, o0, us)
    return r, s, traj


# ---------------------------------------------------------------- a second uncertainty signal: distance from the training data
def make_support(data, n_ref=2000, seed=0):
    """Reference set R of (obs, action) inputs from the training data, scaled per feature, and q = the median nearest-neighbour
    distance of held-out training inputs to R. s_support(x) = relu(d(x, R) / q - 1): zero on the data, growing off it."""
    x = np.concatenate([data[0], data[1][:, None]], 1).astype(np.float32)
    sc = x.std(0) + 1e-3
    idx = np.random.default_rng(seed).permutation(len(x))
    ref, held = jnp.asarray(x[idx[:n_ref]] / sc), jnp.asarray(x[idx[n_ref:n_ref + 500]] / sc)
    d = jnp.sqrt(((held[:, None] - ref[None]) ** 2).sum(-1)).min(1)
    return ref, jnp.asarray(sc), jnp.median(d)


def support_signal(support, o_in, us):
    """o_in [H, 22] the observations the model is fed, us [H]: s_t [H]."""
    ref, sc, q = support
    x = jnp.concatenate([o_in, us[:, None]], 1) / sc
    d2 = (x ** 2).sum(-1)[:, None] + (ref ** 2).sum(-1)[None] - 2.0 * x @ ref.T
    return jnp.maximum(jnp.sqrt(jnp.maximum(d2.min(1), 0.0)) / q - 1.0, 0.0)


def imagine_signal(model, norm, o0, us, support=None, signal="ensemble"):
    r, s, traj = imagine(model, norm, o0, us)
    if signal == "support":
        s = support_signal(support, jnp.concatenate([o0[None], traj[:-1]], 0), us)
    return r, s, traj


# ---------------------------------------------------------------- planner
def cem(model, norm, o0, key, objective, kw, pop=128, iters=4, elite=16, support=None, signal="ensemble"):
    mu, sd = jnp.full(H, 0.2), jnp.full(H, 0.6)
    score = lambda us: jax.vmap(lambda u: objective(*imagine_signal(model, norm, o0, u, support, signal)[:2], **kw))(us)
    for k in jax.random.split(key, iters):
        us = jnp.clip(mu + sd * jax.random.normal(k, (pop, H)), -1.0, 1.0)
        top = jnp.argsort(-score(us))[:elite]
        mu, sd = us[top].mean(0), us[top].std(0) + 0.05
    r, s, _ = imagine_signal(model, norm, o0, mu, support, signal)
    return mu, {"own": objective(r, s, **kw), "plain": j_naive(r, s), "spread": s.mean(), "r": r}


# ---------------------------------------------------------------- the real environment
def behaviour_action(o, key, noise=0.3):
    """A cautious controller: the shield's own acceleration bound, capped, plus exploration noise."""
    a = jnp.minimum(E.shield_bound(o), 2.0)
    u = jnp.where(a >= 0, a / E.MAX_ACCEL, a / E.MAX_DECEL)
    return jnp.clip(u + noise * jax.random.normal(key), -1.0, 1.0)


def collect(n_envs=256, steps=1500, seed=0):
    """Rows in the Rust logger's format, from the shielded environment driven by the noisy cautious controller."""
    cfg = E.EnvConfig()
    st = jax.vmap(E.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed), n_envs), cfg)
    obs = jax.vmap(E.observe)(st)

    @jax.jit
    def run(st, obs, keys):
        def body(c, xs):
            st, obs, segc = c
            k, i = xs
            ks = jax.random.split(k, n_envs + 1)
            u = jax.vmap(behaviour_action)(obs, ks[:n_envs])
            sp = jax.vmap(E.make_spare, in_axes=(0, None))(jax.random.split(ks[n_envs], n_envs), cfg)
            st2, obs2, r, d, info = jax.vmap(E.step, in_axes=(0, 0, None, 0))(st, u, cfg, sp)
            row = jnp.concatenate([jnp.arange(n_envs, dtype=jnp.float32)[:, None], jnp.full((n_envs, 1), i * E.DT), segc[:, None], obs, u[:, None]], 1)
            return (st2, obs2, segc + info["s_segs"] + d), row

        (st, obs, _), rows = jax.lax.scan(body, (st, obs, jnp.zeros(n_envs)), (keys, jnp.arange(steps, dtype=jnp.float32)))
        return st, obs, rows

    st, obs, rows = run(st, obs, jax.random.split(jax.random.PRNGKey(seed + 1), steps))
    return np.asarray(rows).reshape(-1, W.ROW), st, obs


def execute(st, plans, seed=0):
    """Run each state's plan open-loop in the unshielded env: the return of rhat on the real observations, and collisions."""
    cfg = E.EnvConfig(shield=False)
    n = plans.shape[0]
    c0 = st.m_coll

    def body(c, xs):
        st = c
        u, k = xs
        sp = jax.vmap(E.make_spare, in_axes=(0, None))(jax.random.split(k, n), cfg)
        st2, obs2, r, d, info = jax.vmap(E.step, in_axes=(0, 0, None, 0))(st, u, cfg, sp)
        return st2, (rhat(obs2), r)

    st2, (rh, renv) = jax.lax.scan(body, st, (plans.T, jax.random.split(jax.random.PRNGKey(seed), plans.shape[1])))
    ret = (discount(plans.shape[1])[:, None] * rh).sum(0)
    return np.asarray(ret), np.asarray(st2.m_coll - c0), np.asarray(renv.sum(0))


def main(n_states=256, epochs=15):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows, st, obs = collect()
    data = tuple(x[:N_TRAIN] for x in W.transitions(rows))
    print(f"{len(data[0])} behaviour transitions ({time.time() - t0:.0f}s)", flush=True)
    model, norm, _ = W.fit(data, jax.random.PRNGKey(0), epochs=epochs)
    err = W.one_step_errors(model, norm, data)
    print("one-step error v, gap:", round(err["v"], 4), round(err["gap"], 4), f"({time.time() - t0:.0f}s)", flush=True)
    o0 = obs[:n_states]
    st0 = jax.tree.map(lambda a: a[:n_states], st)
    # how large is the ensemble's disagreement on the behaviour data vs on the planner's own imagined steps?
    sample = jnp.asarray(data[0][:5000]), jnp.asarray(data[1][:5000])
    s_data = float(jax.vmap(lambda o, u: ensemble_step(model, norm, o, u)[1])(*sample).mean())

    support = make_support(data)
    print("support scale q =", round(float(support[2]), 3), flush=True)
    settings = {"naive": ({}, "ensemble"), "mopo": ({"lam": 1.0}, "ensemble"), "mopo_x4": ({"lam": 4.0}, "ensemble"),
                "wimle": ({"tau": 0.5}, "ensemble"), "truncate": ({"s_max": 0.5}, "ensemble"),
                "mopo_support": ({"lam": 0.25}, "support"), "mopo_support_x4": ({"lam": 1.0}, "support"),
                "truncate_support": ({"s_max": 1.0}, "support")}
    res = {"s_on_behaviour_data": s_data, "n_states": n_states, "horizon": H, "methods": {}}
    for name, (kw, signal) in settings.items():
        obj = OBJECTIVES[name.split("_")[0]]
        keys = jax.random.split(jax.random.PRNGKey(1), n_states)
        plans, info = jax.jit(lambda o0, keys: jax.lax.map(lambda a: cem(model, norm, a[0], a[1], obj, kw, support=support, signal=signal), (o0, keys), batch_size=32))(o0, keys)
        real, coll, renv = execute(st0, plans)
        imag = np.asarray(info["plain"])
        res["methods"][name] = {
            "imagined_return": float(imag.mean()), "real_return": float(real.mean()), "gap": exploitation_gap(imag, real),
            "own_objective": float(np.asarray(info["own"]).mean()), "mean_spread": float(np.asarray(info["spread"]).mean()),
            "collision_rate": float((coll > 0).mean()), "env_reward": float(renv.mean()), "mean_action": float(np.asarray(plans).mean()),
            "real_returns": real.tolist(), "imagined_returns": imag.tolist()}
        print(name, {k: round(v, 3) for k, v in res["methods"][name].items() if not isinstance(v, list)}, f"({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main(*(int(x) for x in sys.argv[1:3]))
