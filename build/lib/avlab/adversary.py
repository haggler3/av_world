"""Adversarial humans, as an optional feature of the simulator (`Behaviour.adv_*`, default off).

The calibrated human mistakes are rare and unaimed. An adversary does not invent new mistakes; it aims the ones that
exist at the AVs: the driver right behind an AV is distracted (`adv_dist`), a car right in front of an AV breaks down
(`adv_brk`), a car squeezes into a gap right in front of an AV (`adv_cut`: the accepted gap is divided by it). With
`adv_budget = 1` the hazard of everyone else is lowered so that the expected number of mistakes stays the same ("same
mistakes, aimed"); with 0 the aimed hazard is simply added ("more mistakes, aimed").

`search` finds the multipliers that hurt a given AV driver most, by evolution strategies on the AV crash rate (common random
numbers: every candidate sees the same cities and seeds), under a cap on the multipliers. `evaluate` scores them on held-out seeds.

    python -m avlab.adversary search           # ~8 min, writes results/adversary/search.json
    python -m avlab.adversary evaluate         # ~6 min, writes results/adversary/eval.json
"""
import json
import os
import sys
import time

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from avrl import ladder as LD  # noqa: E402
from avrl import xppo as P  # noqa: E402
from av_world import data_path  # noqa: E402
from xnet import network as NW  # noqa: E402
from xnet import sim as X  # noqa: E402

OUT = "results/adversary"
POLICY = data_path("policy.npz")
LEVERS = ("adv_dist", "adv_brk", "adv_cut")
CAP = 1000.0  # no multiplier above this: the plausibility bound on how much a mistake can be aimed
IDX = {k: i for i, k in enumerate(X.METRICS)}
CHUNK = 250


class Arena:
    """Many copies of an area under one Behaviour that is an *argument* (changing it does not recompile)."""

    def __init__(self, area="oakland", policy=None, shield="off", n=12, seed=0):
        self.shield = shield
        X.SHIELD = shield
        self.net_np, _, _ = NW.build(area)
        self.net = X.to_jnp(self.net_np)
        self.nh = jnp.asarray(self.net_np["nh"])
        self.n, self.seed = n, seed
        net, nh = self.net, self.nh

        def stp(s, b):
            if policy is None:
                return X.step(s, net, b, nh)
            return X.step(s, net, b, nh, rl=(policy, s.key))[0]

        @jax.jit
        def chunk(st, beh):
            f = jax.vmap(stp, in_axes=(0, None))
            return jax.lax.scan(lambda s, _: (f(s, beh), None), st, None, length=CHUNK)[0]

        self._chunk = chunk
        self.fresh = jax.vmap(lambda k: X.reset(k, self.net, 1024))(jax.random.split(jax.random.PRNGKey(seed), n))

    def run(self, st, beh, seconds):
        X.SHIELD = self.shield  # read when the step is traced (first call)
        for _ in range(int(round(seconds / (X.DT * CHUNK)))):
            st = self._chunk(st, beh)
        return st

    def warm(self, beh, seconds=200.0):
        return self.run(self.fresh, beh, seconds)

    def measure(self, st, beh, seconds):
        m0 = np.asarray(st.m).sum(0)
        st = self.run(st, beh, seconds)
        d = np.asarray(st.m).sum(0) - m0
        return st, {k: float(d[i]) for k, i in IDX.items()}


def world(risk=5.0, **kw):
    b = LD.test_world()  # humans x5 riskier, 25 % AVs
    base = LD.knobs()
    b = b._replace(**{k: min(v * risk, 0.95) if k != "distract_rate" else v * risk for k, v in base.items()})
    return b._replace(**kw)


def rate(d, key="av_crash"):
    return d[key] / max(d["av_veh_s"] / 3600, 1e-9)


def search(gens=6, pop=6, seconds=250.0, risk=5.0, budget=0.0, seed=0):
    os.makedirs(OUT, exist_ok=True)
    arena = Arena(policy=P.deterministic(P.load(POLICY)), shield="off", n=12, seed=seed)
    base = world(risk, adv_budget=budget)
    warm = arena.warm(base)
    rng = np.random.default_rng(seed)
    theta = np.zeros(3)  # log10 of the multipliers
    sigma, lr, log = 0.8, 0.8, []
    top = np.log10(CAP)

    def fit(th):
        beh = base._replace(**{k: float(10 ** np.clip(t, 0, top)) for k, t in zip(LEVERS, th)})
        _, d = arena.measure(warm, beh, seconds)
        return rate(d), d["av_crash"]

    f0, n0 = fit(theta)
    print(f"no adversary: {f0:.3f} crashes/AV-h ({n0:.0f} crashes)", flush=True)
    for g in range(gens):
        eps = rng.standard_normal((pop // 2, 3))
        eps = np.concatenate([eps, -eps])
        res = [fit(theta + sigma * e) for e in eps]
        f = np.array([r[0] for r in res])
        adv = (f - f.mean()) / (f.std() + 1e-9)
        theta = np.clip(theta + lr / (len(eps) * sigma) * (adv[:, None] * eps).sum(0) * sigma, 0, top)
        cur = fit(theta)
        log.append({"gen": g, "theta_log10": theta.tolist(), "crash_per_h": cur[0], "crashes": cur[1], "pop_mean": float(f.mean())})
        print(f"gen {g}: x{[round(10 ** t, 1) for t in theta]} -> {cur[0]:.3f} crashes/AV-h (pop mean {f.mean():.3f})", flush=True)
        json.dump({"base_rate": f0, "log": log, "risk": risk, "budget": budget}, open(f"{OUT}/search.json", "w"), indent=1)
    return theta


def evaluate(seconds=400.0, risk=5.0, n=24, seed=100):
    s = json.load(open(f"{OUT}/search.json"))
    theta = np.array(s["log"][-1]["theta_log10"])
    mult = {k: float(10 ** t) for k, t in zip(LEVERS, theta)}
    arena = Arena(policy=P.deterministic(P.load(POLICY)), shield="off", n=n, seed=seed)
    cases = {"no adversary": {}, "learned, same mistakes (budget 1)": dict(mult, adv_budget=1.0),
             "learned, extra mistakes (budget 0)": dict(mult, adv_budget=0.0)}
    for k in LEVERS:
        cases[f"{k} alone"] = {k: mult[k], "adv_budget": 0.0}
    res = {}
    warm = arena.warm(world(risk))
    for name, kw in cases.items():
        beh = world(risk, **kw)
        _, d = arena.measure(warm, beh, seconds)
        res[name] = {"crash_per_h": rate(d), "crashes": d["av_crash"], "av_hours": d["av_veh_s"] / 3600,
                     **{k: d[k] for k in ("av_near", "av_hard", "rear-end") + X.TAILS}}
        print(f"{name:38s} {rate(d):.3f} crashes/AV-h  ({d['av_crash']:.0f} crashes, {d['av_veh_s'] / 3600:.0f} AV-h)", flush=True)
    json.dump({"multipliers": mult, "cases": res}, open(f"{OUT}/eval.json", "w"), indent=1)


if __name__ == "__main__":
    {"search": search, "evaluate": evaluate}[sys.argv[1]]()
