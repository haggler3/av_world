"""Idea 8: let a short-horizon planner see past its horizon with a learned value function.

The CEM planner scores H-step action sequences by their imagined return. A long H compounds model error, a short one is short-sighted.
TD-MPC2 / Dreamer-style planners score a short plan and add a learned terminal value:   J_h(a) = sum_{t<h} g^t r_t + g^h V(o_h).

V is fitted by regression on discounted Monte-Carlo returns of the same observation-reward `conservative.rhat` over the next 30 logged
steps of the cautious controller. The test, in the greedy corridor task of wmplan/conservative.py: plan h steps in the learned model, run
them in the real environment (shield off), then let the cautious controller drive until step 20. The real 20-step return of
  h = 5 without V,  h = 5 with V,  and the full 20-step plan (what conservative.py does)
is compared on the same 256 start states.

    python -m wmplan.terminal_value          # ~3 min -> results/extras/terminal_value.json
"""
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from avrl import env as E
from avrl import ppo
from avrl import worldmodel as W

from . import conservative as C

LEN = 30  # steps of logged return the value is regressed on


def mc_returns(rows, length=LEN):
    """Rows [id, t, seg, obs.., u] -> (obs [n, 22], G [n]): G_i = sum_{k<length} g^k rhat(obs_{i+k+1}) where the chain is unbroken."""
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    ok = (rows[1:, 0] == rows[:-1, 0]) & (rows[1:, 2] == rows[:-1, 2]) & (np.abs(rows[1:, 1] - rows[:-1, 1] - 0.1) < 0.02)
    cs = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(rows) - length)
    keep = i[cs[i + length] - cs[i] == length]
    obs = rows[:, 3:3 + E.OBS_DIM].astype(np.float32)
    rh = np.asarray(C.rhat(jnp.asarray(obs)))
    g = np.zeros(len(keep))
    for k in range(length):
        g += C.GAMMA ** k * rh[keep + k + 1]
    return obs[keep], g.astype(np.float32)


def fit_value(obs, g, steps=3000, seed=0):
    n_tr = int(0.8 * len(obs))
    key = jax.random.PRNGKey(seed)
    p = ppo.init_mlp(key, [E.OBS_DIM, 64, 64, 1])
    o, y = jnp.asarray(obs[:n_tr]), jnp.asarray(g[:n_tr])
    opt = optax.adam(1e-3)

    @jax.jit
    def run(p, key):
        def body(c, k):
            p, st = c
            i = jax.random.randint(k, (512,), 0, n_tr)
            l, gr = jax.value_and_grad(lambda p: ((ppo.mlp(p, o[i])[:, 0] - y[i]) ** 2).mean())(p)
            upd, st = opt.update(gr, st, p)
            return (optax.apply_updates(p, upd), st), l

        return jax.lax.scan(body, (p, opt.init(p)), jax.random.split(key, steps))

    (p, _), _ = run(p, key)
    pred = np.asarray(ppo.mlp(p, jnp.asarray(obs[n_tr:]))[:, 0])
    t = g[n_tr:]
    r2 = 1 - ((t - pred) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    return p, float(r2)


def cem_h(model, norm, vp, o0, key, h, use_v, pop=128, iters=4, elite=16):
    mu, sd = jnp.full(h, 0.2), jnp.full(h, 0.6)

    def score(us):
        def one(u):
            r, _, traj = C.imagine(model, norm, o0, u)
            j = (C.GAMMA ** jnp.arange(h) * r).sum()
            return j + (C.GAMMA ** h * ppo.mlp(vp, traj[-1])[0] if use_v else 0.0)
        return jax.vmap(one)(us)

    for k in jax.random.split(key, iters):
        us = jnp.clip(mu + sd * jax.random.normal(k, (pop, h)), -1.0, 1.0)
        top = jnp.argsort(-score(us))[:elite]
        mu, sd = us[top].mean(0), us[top].std(0) + 0.05
    return mu


def execute_then_cautious(st, plans, total=20, seed=0):
    """Plan for the first h steps (shield off), then the cautious controller (deterministic) to `total`. -> discounted real rhat."""
    cfg = E.EnvConfig(shield=False)
    n, h = plans.shape
    obs = jax.vmap(E.observe)(st)

    def body(c, xs):
        st, obs = c
        t, k = xs
        u_plan = jnp.where(t < h, plans[:, jnp.minimum(t, h - 1)], 0.0)
        u_safe = jax.vmap(lambda o: C.behaviour_action(o, jax.random.PRNGKey(0), noise=0.0))(obs)
        u = jnp.where(t < h, u_plan, u_safe)
        sp = jax.vmap(E.make_spare, in_axes=(0, None))(jax.random.split(k, n), cfg)
        st2, obs2, r, d, info = jax.vmap(E.step, in_axes=(0, 0, None, 0))(st, u, cfg, sp)
        return (st2, obs2), C.rhat(obs2)

    _, rh = jax.lax.scan(body, (st, obs), (jnp.arange(total), jax.random.split(jax.random.PRNGKey(seed), total)))
    return np.asarray((C.discount(total)[:, None] * rh).sum(0))


def main(n_states=256, out="results/extras"):
    os.makedirs(out, exist_ok=True)
    rows, st, obs = C.collect()
    obs_v, g = mc_returns(rows)
    vp, r2 = fit_value(obs_v, g)
    print(f"value function: held-out R2 {r2:.3f} on {len(g)} states", flush=True)
    data = tuple(x[:C.N_TRAIN] for x in W.transitions(rows))
    model, norm, _ = W.fit(data, jax.random.PRNGKey(0), epochs=15)
    o0, st0 = obs[:n_states], jax.tree.map(lambda a: a[:n_states], st)
    keys = jax.random.split(jax.random.PRNGKey(1), n_states)
    res = {"value_r2": r2, "n_states": n_states, "methods": {}}
    for name, h, use_v in (("h5_no_value", 5, False), ("h5_value", 5, True), ("h10_no_value", 10, False), ("h10_value", 10, True), ("h20_plan_only", 20, False)):
        plans = jax.jit(jax.vmap(lambda o, k: cem_h(model, norm, vp, o, k, h, use_v)))(o0, keys)
        real = execute_then_cautious(st0, plans)
        res["methods"][name] = {"real_return": float(real.mean()), "real_returns": real.tolist()}
        print(name, round(float(real.mean()), 3), flush=True)
    with open(os.path.join(out, "terminal_value.json"), "w") as f:
        json.dump(res, f)


if __name__ == "__main__":
    main()
