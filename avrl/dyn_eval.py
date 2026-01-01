"""Score the policies trained by avrl.fork with the splitting estimator (avlab.split), in worlds with no safety filter.

Worlds: `oakland` (the ladder's full-dynamics test world: humans 5x riskier, 25 % AVs), `mutated` (held-out mutated
Oakland cities, never used in training), `squirrel_hill` and `downtown` (the real maps).

    python -m avrl.dyn_eval start oakland mutated            # the starting policy
    python -m avrl.dyn_eval base fork shape ... -- oakland   # variants (all seeds and checkpoints found), then worlds
"""
import json
import os
import sys
import time

from avlab import split as SP

from . import fork as FK
from . import xppo as P

EVAL = f"{FK.OUT}/eval.json"


def evaluate(names, worlds, ckpts=FK.CKPTS, replicates=3):
    res = json.load(open(EVAL)) if os.path.exists(EVAL) else {}
    jobs = []
    for n in names:
        if n == "start":
            jobs.append(("start", FK.START, 0))
            continue
        if n == "rule":  # the rule-based AV: the bar
            jobs.append(("rule", None, 0))
            continue
        for d in sorted(os.listdir(FK.OUT)) if os.path.isdir(FK.OUT) else []:
            if d.startswith(n + "_s"):
                for c in ckpts:
                    p = f"{FK.OUT}/{d}/policy_{c}.npz"
                    if os.path.exists(p):
                        jobs.append((f"{d}_{c}", p, c))
    for tag, path, c in jobs:
        for w in worlds:
            k = f"{tag}|{w}"
            if k in res:
                continue
            t0 = time.time()
            res[k] = SP.estimate(P.load(path) if path else None, w, replicates, method="plain")  # splitting did not lower the variance (README)
            os.makedirs(FK.OUT, exist_ok=True)
            json.dump(res, open(EVAL, "w"), indent=1)
            r = res[k]["crash_per_h"]
            print(f"{k:34s} crashes/AV-h {r[0]:.4f} ({r[1]:.4f}-{r[2]:.4f})  {res[k]['speed_kmh'][0]:.1f} km/h  "
                  f"{res[k]['hard_per_h'][0]:.1f} hard/h  {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    names, worlds = (a[: a.index("--")], a[a.index("--") + 1:]) if "--" in a else ([a[0]], a[1:])
    evaluate(names, worlds)
