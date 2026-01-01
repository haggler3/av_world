"""The dynamics ladder: how much better a driving policy gets when the world it trains in is more dynamic.

Five training worlds on the same Oakland map, each adding one source of dynamics to the one before:

  1 AVs only        the AVs share empty streets (light demand, no humans, no pedestrians, no buses, no incidents)
  2 + human traffic careful human drivers around 25 % AVs
  3 + pedestrians   crosswalks in use
  4 + buses, vans, breakdowns   vehicles stop in the lane; others queue, change lanes or pass
  5 + risky humans  red runners, distraction, not looking; demand, AV share and risk randomised per copy

Every policy starts from the same behaviour-cloned driver and gets the same number of PPO updates (same simulation
steps). Every checkpoint is scored in one held-out *test world* with all dynamics switched on (humans x5 riskier than
calibrated, so events are frequent). The score is the training reward per AV decision: progress, comfort, jerk,
headway, speeding, the safety envelope's corrections, near misses, red runs and crashes.

    python -m avrl.ladder bc           # the shared starting policy (~2 min)
    python -m avrl.ladder train 3      # one world (~4 min each)
    python -m avrl.ladder eval         # every checkpoint in the test world (~1 min per world)
    python -m avrl.ladder plot         # docs/img/dynamics_ladder.png
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from xnet import network as NW
from xnet import sim as X

from . import xppo as P

OUT = "results/ladder"
UPDATES = 120
CKPTS = (15, 30, 60, 120)
N_ENVS = 8
REWARD = {"progress": 0.04}
CALM = dict(p_red=0.0, p_nolook=0.0, p_nosignal=0.0, distract_rate=0.0, p_ped_violate=0.0)
STILL = dict(p_bus=0.0, p_van=0.0, breakdown_rate=0.0)
WORLDS = [
    ("1", "AVs only", lambda e, k: X.Behaviour(**CALM, **STILL, p_av=1.0, demand=0.4, ped_scale=0.0)),
    ("2", "+ human traffic", lambda e, k: X.Behaviour(**CALM, **STILL, p_av=0.25, demand=1.0, ped_scale=0.0)),
    ("3", "+ pedestrians", lambda e, k: X.Behaviour(**CALM, **STILL, p_av=0.25, demand=1.0, ped_scale=1.0)),
    ("4", "+ buses, vans, breakdowns", lambda e, k: X.Behaviour(**CALM, p_av=0.25, demand=1.0, ped_scale=1.0)),
    ("5", "+ risky humans, randomised", None),  # xppo.behaviours: the full randomisation
]


def knobs():
    return json.load(open(os.path.join(os.path.dirname(NW.__file__), "..", "results", "calibration.json")))["calibrated"]


def world(level, n=N_ENVS):
    _, _, f = next(w for w in WORLDS if w[0] == level)
    if f is None:
        return P.behaviours(n, 0, knobs())
    return jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *[f(e, knobs()) for e in range(n)])


def test_world():
    k = knobs()
    return X.Behaviour(**{n: min(v * 5.0, 0.95) for n, v in k.items()}, p_av=0.25, demand=1.1, ped_scale=1.0)


def bc():
    P.train(f"{OUT}/bc", updates=0, beh=world("5"), n_envs=N_ENVS, bc_updates=40, reward=REWARD)


def train(level):
    t0 = time.time()
    P.train(f"{OUT}/w{level}", updates=UPDATES, init=f"{OUT}/bc/policy_bc.npz", beh=world(level), ckpt_at=CKPTS,
            n_envs=N_ENVS, warm=150.0, reward=REWARD)
    print(f"world {level}: {time.time() - t0:.0f} s")


def scratch(level):
    """The same budget without the behaviour-cloned start: the world is the only teacher."""
    P.train(f"{OUT}/s{level}", updates=UPDATES, init=None, beh=world(level), ckpt_at=CKPTS, n_envs=N_ENVS, warm=150.0,
            reward=REWARD, bc_updates=0)


def evaluator(n_envs=8, warm=300.0, seconds=180.0):
    """A jitted scorer: params -> (reward per AV decision, AV metrics) in the test world, from one fixed warm state."""
    net, geo, info = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    X.R = X.Reward(**REWARD)
    beh = test_world()
    sim = X.Sim(net, n_envs=n_envs, beh=beh, seed=123, reroute_s=0)
    sim.run(warm)
    st0 = sim.st
    n = int(seconds / X.DT)

    @jax.jit
    def score(params, st, key):
        def body(st, k):
            st, tr = jax.vmap(lambda s, kk: X.step(s, J, beh, nh, rl=(P.deterministic(params), kk)))(st, jax.random.split(k, n_envs))
            return st, (tr["reward"].sum(), tr["valid"].sum())
        m0 = st.m
        st, (r, v) = jax.lax.scan(body, st, jax.random.split(key, n))
        return r.sum() / v.sum(), (st.m - m0).sum(0)

    def f(path):
        r, m = score(P.load(path), st0, jax.random.PRNGKey(7))
        return float(r), P.summarize(np.asarray(m))
    return f


def evaluate():
    f = evaluator()
    rows = []
    sim_steps = N_ENVS * (P.Config().n_steps + 1)  # simulator steps (area-copies x 0.2 s) per PPO update
    r0, s0 = f(f"{OUT}/bc/policy_bc.npz")
    runs = [(f"w{lv}", lv, name, "imitation start") for lv, name, _ in WORLDS] + \
           [(f"s{lv}", lv, name, "from scratch") for lv, name, _ in WORLDS]
    for d, level, name, start in runs:
        if not os.path.isdir(f"{OUT}/{d}"):
            continue
        if start == "imitation start":
            rows.append({"run": d, "world": level, "name": name, "start": start, "updates": 0, "sim_steps": 0, "score": r0, **s0})
        for c in CKPTS:
            path = f"{OUT}/{d}/policy_{c}.npz"
            if not os.path.exists(path):
                continue
            r, s = f(path)
            rows.append({"run": d, "world": level, "name": name, "start": start, "updates": c, "sim_steps": c * sim_steps, "score": r, **s})
            print(json.dumps({k: rows[-1][k] for k in ("run", "updates", "score", "av_speed_kmh", "av_hard_per_h", "av_shield_share", "av_crashes")}, default=float), flush=True)
    with open(f"{OUT}/scores.json", "w") as fo:
        json.dump(rows, fo, indent=1, default=float)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "bc":
        bc()
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "scratch":
        scratch(sys.argv[2])
    elif cmd == "eval":
        evaluate()
    elif cmd == "plot":
        from .ladder_plot import main as plot

        plot()
