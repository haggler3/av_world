"""The simulation half of examples/hypotheses.py: one compiled run per world setting, many copies each, every counter kept per copy.

    python examples/hypotheses_sims.py           # ~25 min on 4 CPU cores -> results/examples/hypotheses_sims.json
    python examples/hypotheses_sims.py --quick   # a smoke version

Settings (calibrated humans, no AVs): human mistake rates x 1, 2, 4 as avlab.adversary.world sets them (which also raises demand
to 1.1 and the aggressive share), the three calibrated mistake rates alone x 3, 10, 30; demand x 0.5, 1, 1.5, 2; road friction 1,
0.6, 0.3. Groups already in the results file are skipped, so the script resumes. Each copy is warmed 2 min and measured 10 min while traffic still flows (see av_share_sweep.py).
"""
import argparse
import json
import time

import numpy as np

try:
    from examples import _common as C
    from examples.av_share_sweep import humans
except ImportError:
    import _common as C
    from av_share_sweep import humans
from xnet import sim as X  # noqa: E402

def risky(r):
    """Calibrated humans with the three calibrated mistake rates (red-light running, not looking, distraction) x r; nothing else changed."""
    from avrl import ladder as LD
    return C.calibrated(p_av=0.0, **{k: min(v * r, 0.95) for k, v in LD.knobs().items()})


GROUPS = {  # name: (behaviour, vehicle slots)
    "base": (lambda: humans(1, 0.0), 1024),
    "risk_x2": (lambda: humans(2, 0.0), 1024),
    "risk_x4": (lambda: humans(4, 0.0), 1024),
    "mistakes_x3": (lambda: risky(3), 1024),
    "mistakes_x10": (lambda: risky(10), 1024),
    "mistakes_x30": (lambda: risky(30), 1024),
    "demand_x0.5": (lambda: C.calibrated(p_av=0.0, demand=0.5), 1024),
    "demand_x1.5": (lambda: C.calibrated(p_av=0.0, demand=1.5), 2048),
    "demand_x2": (lambda: C.calibrated(p_av=0.0, demand=2.0), 2048),
    "friction_0.6": (lambda: C.calibrated(p_av=0.0, friction=0.6), 1024),
    "friction_0.3": (lambda: C.calibrated(p_av=0.0, friction=0.3), 1024),
}

SEEDS = {g: 7000 + 100 * i for i, g in enumerate(("base", "risk_x2", "risk_x4", "demand_x0.5", "demand_x1.5", "demand_x2", "friction_0.6",
                                                  "friction_0.3", "mistakes_x3", "mistakes_x10", "mistakes_x30"))}


def main(copies=48, warm_s=120.0, seconds=600.0, area="oakland", name="hypotheses_sims", only=None):
    t0 = time.time()
    try:
        out = json.load(open(f"results/examples/{name}.json"))  # resume: groups already run are kept
    except (FileNotFoundError, ValueError):
        out = {"area": area, "copies": copies, "warm_s": warm_s, "seconds": seconds, "metrics": list(X.METRICS), "groups": {}}
    for g, (beh, slots) in GROUPS.items():
        if g in out["groups"] or (only and g not in only):
            continue
        seed = SEEDS[g]
        m = C.run_copies(area, [beh()] * copies, warm_s=warm_s, seconds=seconds, seed=seed, n_slots=slots)
        out.setdefault("seeds", {})[g] = seed
        out["groups"][g] = np.asarray(m).round(3).tolist()
        d = dict(zip(X.METRICS, np.asarray(m).sum(0)))
        k = sum(d[c] for c in X.CRASH_NAMES)
        print(f"  {g:13s} {k:4.0f} crashes in {d['veh_s'] / 3600:7.0f} vehicle-hours  ({time.time() - t0:.0f} s)", flush=True)
        C.save(name, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(copies=2, warm_s=30, seconds=60, name="hypotheses_sims_quick") if a.quick else main()
