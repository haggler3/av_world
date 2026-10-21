"""A small functional environment over the simulator: many copies of a city, every AV driven by one policy.

    env = av_world.make("oakland", n_envs=8, beh=dict(p_av=0.3))        # area name, (w, s, e, n) box, OSM file, or a net dict
    state = env.reset(seed=0, warm_s=120)                               # copies of the area, warmed up
    state, tr = env.step(state, policy)                                 # policy(obs [N, 28], key) -> (u [N, 2], aux) per copy
    state, traj = env.rollout(state, policy, steps=100)                 # lax.scan of steps; traj leaves are [steps, n_envs, ...]
    env.metrics(state)                                                  # crashes, AV crashes per type, speeds ... (dict)

`tr` holds, per copy and vehicle slot: obs, u, reward, valid (the slot is a live AV), done, crash, near, red and the barrier
values (bar_h ...). `rule_policy` is the rule-based AV (no learning: the simulator's own driver). A learned policy is any
jit-compatible function; avrl.xppo has the trained one (`av_world.load_policy()`).
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from av_world import data_path
from xnet import network as NW
from xnet import sim as X


rule_policy = None  # pass policy=None (or this) to use the rule-based AV: the simulator's own IDM driver


def load_policy(path=None, deterministic=True):
    from avrl import xppo as P
    params = P.load(path or data_path("policy.npz"))
    return (P.deterministic if deterministic else P.stochastic)(params)


class Env:
    def __init__(self, area="oakland", n_envs=8, beh=None, shield="idm", n_slots=1024, seed=0, **build_kw):
        if isinstance(area, dict):
            self.net_np, self.geo, self.info = area, None, {}
        elif isinstance(area, str) and os.path.exists(area) and area.lower().endswith((".osm", ".xml", ".json")):
            from xnet import osm
            self.net_np, self.geo, self.info = osm.import_osm(area, **build_kw)
        elif isinstance(area, str) and area.lower().endswith(".npz"):
            from xnet import osm
            self.net_np, self.geo, self.info = osm.load(area), None, {}
        else:
            self.net_np, self.geo, self.info = NW.build(area, **build_kw)
        self.net, self.nh = X.to_jnp(self.net_np), jnp.asarray(self.net_np["nh"])
        self.n_envs, self.n_slots, self.shield = n_envs, n_slots, shield
        b = beh if isinstance(beh, X.Behaviour) else X.Behaviour(**(beh or {}))
        self.beh = b
        self._jit = {}

    # -- step functions are traced with the safety filter that is current at the first call
    def _fn(self, name, policy):
        k = (name, policy)
        if k not in self._jit:
            net, nh, beh = self.net, self.nh, self.beh

            def one(s, key):
                if policy is None:
                    return X.step(s, net, beh, nh), None
                return X.step(s, net, beh, nh, rl=(policy, key))

            if name == "step":
                self._jit[k] = jax.jit(lambda st, key: jax.vmap(one)(st, jax.random.split(key, self.n_envs)))
            else:
                n = name[1]

                def roll(st, key):
                    def body(s, kk):
                        s, tr = jax.vmap(one)(s, jax.random.split(kk, self.n_envs))
                        return s, tr
                    return jax.lax.scan(body, st, jax.random.split(key, n))
                self._jit[k] = jax.jit(roll)
        return self._jit[k]

    def reset(self, seed=0, warm_s=0.0):
        X.SHIELD = self.shield
        st = jax.vmap(lambda k: X.reset(k, self.net, self.n_slots))(jax.random.split(jax.random.PRNGKey(seed), self.n_envs))
        if warm_s:
            st, _ = self.rollout(st, None, int(warm_s / X.DT), key=jax.random.PRNGKey(seed + 1))
        return st

    def step(self, state, policy=None, key=None):
        X.SHIELD = self.shield
        key = jax.random.fold_in(jax.random.PRNGKey(0), int(np.asarray(state.t)[0] / X.DT)) if key is None else key
        return self._fn("step", policy)(state, key)

    def rollout(self, state, policy=None, steps=100, key=None):
        X.SHIELD = self.shield
        key = jax.random.fold_in(jax.random.PRNGKey(1), int(np.asarray(state.t)[0] / X.DT)) if key is None else key
        return self._fn(("roll", steps), policy)(state, key)

    def metrics(self, state):
        d = dict(zip(X.METRICS, np.asarray(state.m).sum(0)))
        return {k: float(v) for k, v in d.items()}
