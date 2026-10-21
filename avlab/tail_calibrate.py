"""Idea 1: fit the everyday human driver to real logs, so the continuous tail appears.

The audit (docs/INCIDENT_AUDIT.md) found simulated traffic 4-6x short of real Argoverse 2 traffic in hard braking, short time-to-collision and
tight following. Four knobs shape that tail (`Behaviour.T_med / T_sigma / accel_noise / pulse_rate`: lognormal headways, acceleration noise,
spontaneous brake pulses). Simulation-based calibration by the cross-entropy method: draw knob sets, simulate Oakland, measure the same four
statistics on the simulated scenes that `avlab.incidents.scene_metrics` measures on the real ones, and move the sampling distribution toward
the sets whose statistics are closest (in log ratio) to the real values.

    python -m avlab.tail_calibrate fit        # ~10 min -> results/tail/calibrated_drivers.json
    python -m avlab.tail_calibrate check      # the fitted drivers against the real rates, and what they do to the AV's crash rate
"""
import json
import os
import sys
import time

os.environ.setdefault("JAX_DEFAULT_PRNG_IMPL", "unsafe_rbg")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xnet import carla_export as CE  # noqa: E402
from xnet import network as NW  # noqa: E402
from xnet import sim as X  # noqa: E402

from . import adversary as AD  # noqa: E402
from . import incidents as IN  # noqa: E402

OUT = "results/tail"
REAL = "results/incidents/real.json"
BOUNDS = {"T_med": (0.6, 1.6), "T_sigma": (0.0, 0.7), "accel_noise": (0.0, 1.0), "pulse_rate": (0.0, 0.02)}
KEYS = list(BOUNDS)
STATS = ("hard_brakes_per_h", "tight_follow_share", "ttc_under_2s_share", "ttc_under_1s_share")
EPS = {"hard_brakes_per_h": 1.0, "tight_follow_share": 2e-4, "ttc_under_2s_share": 2e-4, "ttc_under_1s_share": 5e-5}


class Recorder:
    """One copy of the area; the behaviour is an argument, so a new knob set does not recompile. Frames come back as host States."""

    def __init__(self, area="oakland", seed=0):
        self.net_np, self.geo, _ = NW.build(area)
        self.net, self.nh = X.to_jnp(self.net_np), jnp.asarray(self.net_np["nh"])
        net, nh = self.net, self.nh
        self._step = jax.jit(lambda s, b: X.step(s, net, b, nh))
        self.seed = seed

    def run(self, beh, warm=100.0, seconds=60.0, seed=None):
        X.SHIELD = "idm"
        st = X.reset(jax.random.PRNGKey(self.seed if seed is None else seed), self.net, 1024)
        for _ in range(int(warm / X.DT)):
            st = self._step(st, beh)
        frames = []
        for _ in range(int(seconds / X.DT)):
            st = self._step(st, beh)
            frames.append(jax.tree.map(np.asarray, st))
        return frames

    def stats(self, beh, reps=2, warm=100.0, seconds=60.0, seed0=0):
        tot = None
        for r in range(reps):
            sc = CE.trajectories(self.net_np, self.geo, self.run(beh, warm, seconds, seed=seed0 + r))
            m = IN.scene_metrics(sc)
            tot = m if tot is None else {k: tot[k] + m[k] for k in m}
        return IN.rates(tot)


def loss(sim, real):
    return float(sum(abs(np.log((float(sim[k]) + EPS[k]) / (real[k] + EPS[k]))) for k in STATS))


def behaviour(theta):
    return AD.world(1.0, **{k: float(v) for k, v in zip(KEYS, theta)})


def fit(iters=5, pop=8, elite=3, seed=0):
    os.makedirs(OUT, exist_ok=True)
    real = json.load(open(REAL))
    rec = Recorder()
    lo, hi = np.array([BOUNDS[k][0] for k in KEYS]), np.array([BOUNDS[k][1] for k in KEYS])
    mu, sd = (lo + hi) / 2, (hi - lo) / 3
    rng = np.random.default_rng(seed)
    log, best = [], (1e9, None, None)
    base = rec.stats(AD.world(1.0))
    print("default drivers:", {k: round(base[k], 4) for k in STATS}, "loss", round(loss(base, real), 2), flush=True)
    for it in range(iters):
        cand = np.clip(mu + sd * rng.standard_normal((pop, len(KEYS))), lo, hi)
        res = []
        for th in cand:
            st = rec.stats(behaviour(th))
            res.append((loss(st, real), th, st))
        res.sort(key=lambda r: r[0])
        el = np.array([r[1] for r in res[:elite]])
        mu, sd = el.mean(0), np.maximum(el.std(0), 0.05 * (hi - lo))
        if res[0][0] < best[0]:
            best = res[0]
        log.append({"iter": it, "best_loss": res[0][0], "mean_loss": float(np.mean([r[0] for r in res])), "mu": mu.tolist()})
        print(f"iter {it}: best loss {res[0][0]:.2f} (mean {log[-1]['mean_loss']:.2f})  knobs {dict(zip(KEYS, np.round(res[0][1], 3)))}", flush=True)
        json.dump({"knobs": dict(zip(KEYS, map(float, best[1]))), "loss": best[0], "stats": {k: float(best[2][k]) for k in STATS},
                   "default_stats": {k: float(base[k]) for k in STATS}, "default_loss": loss(base, real), "real": {k: real[k] for k in STATS}, "log": log},
                  open(f"{OUT}/calibrated_drivers.json", "w"), indent=1)


def check():
    """The fitted drivers on fresh seeds: the four statistics against real, and the rule-based AV's crash rate with default and fitted humans."""
    cal = json.load(open(f"{OUT}/calibrated_drivers.json"))
    real = json.load(open(REAL))
    rec = Recorder()
    out = {"knobs": cal["knobs"]}
    for name, beh in (("default drivers", AD.world(1.0)), ("fitted drivers", AD.world(1.0, **cal["knobs"]))):
        st = rec.stats(beh, reps=3, seed0=50)
        out[name] = {k: float(st[k]) for k in STATS} | {"loss": loss(st, real)}
        print(name, {k: round(float(st[k]), 4) for k in STATS}, flush=True)
    arena_res = {}
    for name, beh in (("default drivers", AD.world(5.0)), ("fitted drivers", AD.world(5.0, **cal["knobs"]))):
        arena = AD.Arena(policy=None, shield="idm", n=32, seed=31)
        st = arena.warm(beh, 100.0)
        _, d = arena.measure(st, beh, 300.0)
        arena_res[name] = {"av_hours": d["av_veh_s"] / 3600, "av_crashes": d["av_crash"], "av_hard_per_h": d["av_hard"] / (d["av_veh_s"] / 3600),
                           "av_near": d["av_near"], "rear-end": d["rear-end"]}
        print(name, "rule-based AV:", arena_res[name], flush=True)
    out["rule_based_av"] = arena_res
    out["real"] = {k: real[k] for k in STATS}
    json.dump(out, open(f"{OUT}/calibrated_check.json", "w"), indent=1)


if __name__ == "__main__":
    {"fit": fit, "check": check}[sys.argv[1]]()
