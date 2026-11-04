"""How long do two coupled worlds stay coupled? Distraction at 1/60 per second against 1/20, same seeds, CRN off and on.

Every 10 s window: of the drivers distracted in the low-rate world, the share that are also distracted in the high-rate world
(1.0 = nested, as the thinning argument says for a driver present in both). Without CRN the overlap collapses as soon as the
worlds' slot assignments drift apart; with CRN it stays near 1, and what is left is the genuine divergence of the traffic.

    python -m avlab.crn_decay            # ~5 min on 4 cores -> results/crn/decay.json
"""
import json
import os

import numpy as np

from xnet import network as NW
from xnet import sim as X

OUT = "results/crn"


def run(net, crn, rates=(1 / 60, 1 / 20), seconds=600, n_envs=8, window=10, seed=5):
    sims = [X.Sim(net, n_slots=768, n_envs=n_envs, seed=seed, reroute_s=0, beh=X.Behaviour(crn=crn, distract_rate=r)) for r in rates]
    rows = []
    cur = [[set() for _ in range(n_envs)] for _ in rates]
    for sec in range(1, seconds + 1):
        for k, sim in enumerate(sims):
            sim.run(1.0)
            d, u = np.asarray(sim.st.dist_t > 0), np.asarray(sim.st.uid)
            for e in range(n_envs):
                cur[k][e] |= set(u[e][d[e]].tolist())
        if sec % window == 0:
            n_lo = sum(len(s) for s in cur[0])
            n_both = sum(len(a & b) for a, b in zip(*cur))
            # drift of the traffic itself: vehicles alive in both worlds / alive in the low-rate one
            u0, u1 = [np.asarray(s.st.uid) for s in sims], None
            a0, a1 = [np.asarray(s.st.alive) for s in sims], None
            shared = sum(len(set(u0[0][e][a0[0][e]].tolist()) & set(u0[1][e][a0[1][e]].tolist())) for e in range(n_envs))
            alive = sum(int(a0[0][e].sum()) for e in range(n_envs))
            rows.append({"t": sec, "events_low": n_lo, "overlap": n_both / max(n_lo, 1), "shared_vehicles": shared / max(alive, 1)})
            cur = [[set() for _ in range(n_envs)] for _ in rates]
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    net, _, _ = NW.build("oakland")
    res = {f"crn={int(c)}": run(net, c) for c in (0.0, 1.0)}
    for k, rows in res.items():
        print(k, [round(r["overlap"], 2) for r in rows][::6], flush=True)
    with open(os.path.join(OUT, "decay.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
