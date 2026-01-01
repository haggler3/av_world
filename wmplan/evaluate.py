"""The latent planner against the other drivers, in a world built to make merging and junctions hard.

Oakland with humans x5 riskier than calibrated (red runners, drivers who pull out without looking, distraction),
rush-hour demand x1.2, 1.5x pedestrians, 5 % cyclists, 15 % AVs. Every driver starts from the same warm state and sees
the same random numbers:

  rule            the rule-based AV
  ppo raw         the 234k-step PPO policy, no safety filter
  ppo+cbf         the same under the control barrier function
  wm-mpc raw      the latent world-model planner, no safety filter
  wm-mpc+cbf      the same under the control barrier function

    python -m wmplan.evaluate          # ~12 min (the two planner runs ~5 min each) -> results/wmplan/eval.json
"""
import json
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

from . import model as M
from . import planner as PL

OUT = M.OUT
N_ENVS, WARM, SECONDS = 4, 240.0, 300.0
CONFIGS = ("rule", "ppo raw", "ppo+cbf", "wm-mpc raw", "wm-mpc+cbf")


def world():
    knobs = {k: min(v * 5.0, 0.95) for k, v in LD.knobs().items()}
    return X.Behaviour(**(knobs | dict(p_av=0.15, p_bike=0.05, demand=1.2, ped_scale=1.5)))


def main(configs=CONFIGS):
    net, geo, info = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = world()
    sim = X.Sim(net, n_envs=N_ENVS, beh=beh, seed=21, reroute_s=0)
    sim.run(WARM)
    st0 = sim.st
    ppo = P.load("results/xnet3/policy.npz")
    wm = M.load()
    n = int(SECONDS / X.DT)
    keys = jax.random.split(jax.random.PRNGKey(5), n)
    out = json.load(open(f"{OUT}/eval.json")) if len(configs) < len(CONFIGS) and os.path.exists(f"{OUT}/eval.json") else {}
    for c in configs:
        driver, _, filt = c.partition(" ") if " " in c else c.partition("+")
        shield = {"raw": "off", "cbf": "cbf"}.get(filt, "idm")
        if driver == "rule":
            pol = None
        elif driver.startswith("ppo"):
            base = P.deterministic(ppo)
            pol = (lambda o, k, m, f=base: f(o, k))
        else:
            pol = PL.make(wm, ppo)
        X.SHIELD = shield

        @jax.jit
        def go(st):
            def body(st, k):
                ks = jax.random.split(k, N_ENVS)
                if pol is None:
                    return jax.vmap(lambda s, kk: X.step(s, J, beh, nh))(st, ks), None
                return jax.vmap(lambda s, kk: X.step(s, J, beh, nh, rl=(pol, kk, True))[0])(st, ks), None
            m0 = st.m
            return jax.lax.scan(body, st, keys)[0].m - m0
        t0 = time.time()
        m = np.asarray(go(st0)).sum(0)
        X.SHIELD = "idm"
        d = dict(zip(X.METRICS, m))
        h = d["av_veh_s"] / 3600
        out[c] = {"av_hours": h, "crashes": float(d["av_crash"]), "crash_per_h": d["av_crash"] / h,
                  "types": {k[3:]: float(d[k]) for k in X.AV_CRASH}, "near_per_h": d["av_near"] / h, "red": float(d["av_red"]),
                  "speed_kmh": 3.6 * d["av_veh_m"] / d["av_veh_s"], "hard_per_h": d["av_hard"] / h,
                  "jerk_rms": float(np.sqrt(d["av_jerk2"] / d["av_veh_s"])), "h_neg_share": d["av_h_neg"] / d["av_veh_s"],
                  "veto_junction_s_per_h": d["av_veto_junction"] / h, "veto_merge_s_per_h": d["av_veto_merge"] / h,
                  "human_speed_kmh": 3.6 * (d["veh_m"] - d["av_veh_m"]) / (d["veh_s"] - d["av_veh_s"]), "wall_s": time.time() - t0}
        print(c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[c].items()}, default=float), flush=True)
    with open(f"{OUT}/eval.json", "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or CONFIGS)
