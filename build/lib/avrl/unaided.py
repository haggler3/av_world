"""Teach the policy to drive without a safety filter, with the barrier function as the teacher.

Starting from the 234k-step policy (AVBench: 0.64 crashes per AV-hour with no filter), two phases in the full-dynamics
training world, the reward charging what the barrier function would have done (xnet.sim.Reward.barrier / inside /
veto), whether or not it is filtering:

  phase 1  the barrier function filters (data stays clean) while every m/s^2 it takes off, every metre inside the
           barrier and every junction entry or merge it would have vetoed costs reward: learn not to need it
  phase 2  no filter: the same charges, and the policy now meets the real consequences of what it still gets wrong

    python -m avrl.unaided 1        # ~4 min -> results/unaided/phase1/policy.npz
    python -m avrl.unaided 2        # ~4 min -> results/unaided/phase2/policy.npz
    python -m avrl.unaided 3        # ~4 min -> results/unaided/phase3/policy.npz
    python -m avrl.unaided 4        # ~4 min -> results/unaided/phase4/policy.npz

  phase 4  no filter, from phase 3: plus a cruise term (per m/s below the speed limit on open road), the rule-based
           AV's habit the policy never had a reason to learn
  phase 3  no filter, from phase 2: progress weighted 0.05 (AVBench's *efficient* check wants >= 97 % of the rule-based
           AV's speed) and half the copies of the area full of jaywalkers (its weakest scenario)
"""
import sys
import time

import jax.numpy as jnp

from av_world import data_path
from xnet import sim as X

from . import ladder as LD
from . import xppo as P

OUT = "results/unaided"
START = data_path("policy.npz")
TEACH = LD.REWARD | {"barrier": 0.3, "inside": 0.05, "veto": 1.0}
UPDATES = 100


def world(n):
    beh = LD.world("5")
    if n < 3:
        return beh
    jay = jnp.arange(LD.N_ENVS) % 2 == 1  # every other copy: 2.5x pedestrians, 40 % crossing against the signal
    return beh._replace(ped_scale=jnp.where(jay, 2.5, beh.ped_scale), p_ped_violate=jnp.where(jay, 0.4, beh.p_ped_violate))


def phase(n):
    t0 = time.time()
    X.SHIELD = "cbf" if n == 1 else "off"  # read when the update is traced
    init = {1: START, 2: f"{OUT}/phase1/policy.npz", 3: f"{OUT}/phase2/policy.npz", 4: f"{OUT}/phase3/policy.npz"}[n]
    reward = TEACH | ({"progress": 0.05} if n >= 3 else {}) | ({"cruise": 0.02} if n == 4 else {})
    try:
        P.train(f"{OUT}/phase{n}", updates=UPDATES, init=init, beh=world(n), n_envs=LD.N_ENVS, warm=150.0, reward=reward,
                ckpt_at=(UPDATES,))
    finally:
        X.SHIELD = "idm"
    print(f"phase {n}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    phase(int(sys.argv[1]))
