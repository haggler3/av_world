"""Crash survey over mined Pittsburgh intersections: every inferred-signal intersection plus a random sample of stop-
controlled ones, one simulated intersection each, with the calibrated human behaviour (results/calibration.json)
amplified as in the calibration's verification run (xworld.calibrate.amplification); crash rates are divided back.

    python -m pgh.survey      # -> results/pgh_survey.json
"""
import json
import time

import jax
import numpy as np

from xworld import calibrate as C
from xworld import geometry as G
from xworld import sim as S

from . import layouts as P

def main(n_total=2048, seconds=1800, seed=0, out="results/pgh_survey.json"):
    its = P.load()
    rng = np.random.default_rng(seed)
    sig = [i for i in its if i["control"] == "signal"]
    stop = [i for i in its if i["control"] != "signal"]
    pick = sig + list(rng.choice(stop, n_total - len(sig), replace=False))
    lay = S.stack([P.build(i)[1] for i in pick])
    knobs = json.load(open("results/calibration.json"))["calibrated"]
    A = C.amplification(knobs)
    beh = S.Behaviour(**{k: v * A for k, v in knobs.items()}, p_aggressive=0.2)
    t0 = time.time()
    m, _ = S.run_batch(jax.random.PRNGKey(seed), len(pick), seconds, beh, lay=lay, per_env=True)
    wall = time.time() - t0
    groups = {}
    for name, sel in (("signal, 4-leg", lambda i: i["control"] == "signal" and i["n_legs"] == 4),
                      ("signal, 3-leg", lambda i: i["control"] == "signal" and i["n_legs"] == 3),
                      ("two-way stop, 4-leg", lambda i: i["control"] != "signal" and i["n_legs"] == 4),
                      ("two-way stop, 3-leg", lambda i: i["control"] != "signal" and i["n_legs"] == 3)):
        idx = [k for k, i in enumerate(pick) if sel(i)]
        mm = m[idx].sum(0)
        mev = mm[0] / 1e6
        groups[name] = {"intersections": len(idx), "entered": int(mm[0]), "crashes": int(mm[2:7].sum()),
                        "crashes_per_MEV": float(mm[2:7].sum() / max(mev, 1e-9) / A),
                        "share": {n: float(c / max(mm[2:7].sum(), 1)) for n, c in zip(G.CRASH_NAMES, mm[2:7])},
                        "blocked_share": float(mm[11] / max(mm[11] + mm[0], 1))}
        print(name, groups[name], flush=True)
    per = [{"id": i["id"], "name": i["name"], "control": i["control"], "n_legs": i["n_legs"], "entered": int(m[k, 0]),
            "crashes": int(m[k, 2:7].sum()), "types": m[k, 2:7].astype(int).tolist()} for k, i in enumerate(pick)]
    per.sort(key=lambda r: -r["crashes"])
    res = {"amplification": A, "seconds_per_intersection": seconds, "wall_s": round(wall, 1), "groups": groups,
           "most_crashes": per[:25], "per_intersection": per}
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{len(pick)} intersections x {seconds} s in {wall:.0f} s -> {out}")


if __name__ == "__main__":
    main()
