"""Calibrate the human-behaviour knobs so simulated crash rates match published intersection crash data, then sweep the
AV share.

Crashes are rare (about 1 per million entering vehicles, MEV), so the calibration probes each behaviour on its own at an
amplified level, measures crashes/MEV per unit of that behaviour, and solves a non-negative least-squares problem for the
knob values that reproduce the targets. At these low rates crashes are additive and linear in each behaviour; the final
verification run checks that.

Targets (urban signalised 4-way intersections; see README for sources):
  total            ~1.0 crashes / MEV        (0.78 Massachusetts statewide signalised average; 1.26 in a generic table)
  rear-end         36 %   (HSIS Michigan + Minnesota, 4-leg signalised, multi-vehicle)
  angle            23 %   (HSIS: 20.3 % Minnesota, 25.8 % California)
  left-turn        15 %   (assumed: not found in the sources)
  sideswipe        10 %   (assumed)
  red-light running involved in 16-20 % (HSIS, four states)
  red-light runs   3.9 per 1000 entering vehicles (observed at 15 intersections in Melbourne; studies range ~1-19)
The simulator models these four multi-vehicle types, i.e. ~84 % of all crashes, so the modelled total target is 0.84 / MEV.

    python -m xworld.calibrate            # probes + fit + verification (~12 min on 4 CPU cores)
"""
import json
import os
import sys
import time

import jax
import numpy as np
from scipy.optimize import lsq_linear

from . import sim as S

KNOBS = ("p_red", "p_nolook", "distract_rate", "p_nosignal")  # aggressive gap acceptance had no measurable crash effect
PROBE = {"p_red": 1.0, "p_nolook": 0.1, "distract_rate": 1 / 120, "p_nosignal": 1.0}
PROBE_SECONDS = {"p_red": 1800}  # red running rarely ends in a crash: probe it longer
CLEAN = dict(p_red=0.0, p_nolook=0.0, distract_rate=0.0, p_nosignal=0.0, p_aggressive=0.2)
TYPES = ("rear-end", "angle", "left-turn", "sideswipe")
TARGET_TOTAL = 0.84
TARGET_SHARE = np.array([0.36, 0.23, 0.15, 0.10]) / 0.84  # of the modelled crashes
TARGET_RLR = 0.18
TARGET_RED_RUNS = 3.9e-3  # red-light runs per entering vehicle (Melbourne, 15 intersections: 3.9 per 1000 vehicles)


def run(beh, seconds=300, n_envs=1024, seed=0):
    t0 = time.time()
    m, _ = S.run_batch(jax.random.PRNGKey(seed), n_envs, seconds, beh)
    mev = m[0] / 1e6
    return {"mev": float(mev), "per_type": (m[2:6] / mev).tolist(), "rlr": float(m[7] / mev), "crashes": int(m[2:7].sum()),
            "red_runs_per_1000": float(1000 * m[8] / max(m[0], 1)), "wall_s": round(time.time() - t0, 1)}


BOUNDS = {"p_red": 1.0, "p_nolook": 1.0, "distract_rate": 1 / 60, "p_nosignal": 1.0}


def vec(r):
    return np.array([*r["per_type"], r["red_runs_per_1000"]])


def fit(probes, clean):
    """Per-type crash rates (per MEV) and red runs (per 1000 entering) as a linear function of the knobs, each probe's
    clean-world background removed. Bounded least squares on relative errors. The red-light-running share of crashes is
    left out of the fit and reported as a check."""
    base = vec(clean)
    A = np.array([(vec(probes[k]) - base) / PROBE[k] for k in KNOBS]).T
    A = np.maximum(A, 0.0)
    b = np.array([*(TARGET_TOTAL * TARGET_SHARE), TARGET_RED_RUNS * 1e3]) - base
    w = 1.0 / np.maximum(np.abs(b), 0.05)
    w[-1] *= 0.3  # crash rates first; the observed red-run frequency is a softer target (see README)
    r = lsq_linear(A * w[:, None], b * w, bounds=(0, [BOUNDS[k] for k in KNOBS]))
    return dict(zip(KNOBS, r.x)), A @ r.x + base, b + base


def main(out="results"):
    os.makedirs(out, exist_ok=True)
    res = {"probes": {}, "clean": run(S.Behaviour(**CLEAN), seed=99)}
    print("clean", res["clean"], flush=True)
    for k in KNOBS:
        beh = S.Behaviour(**(CLEAN | {k: PROBE[k]}))
        res["probes"][k] = run(beh, seconds=PROBE_SECONDS.get(k, 300), seed=sum(map(ord, k)))
        print(k, res["probes"][k], flush=True)
    knobs, pred, target = fit(res["probes"], res["clean"])
    res["calibrated"] = {k: float(v) for k, v in knobs.items()}
    names = [*TYPES, "red_runs_per_1000"]
    res["fit"] = {"predicted": dict(zip(names, pred.round(4).tolist())), "target": dict(zip(names, target.round(4).tolist()))}
    print("calibrated", res["calibrated"], "\nfit", res["fit"], flush=True)
    # verification: all calibrated knobs x A together (probabilities capped at 1), rates divided by A
    A = 25.0
    amp = {k: min(v * A, 1.0) for k, v in knobs.items()}
    eff = {k: amp[k] / max(knobs[k], 1e-12) for k in knobs}
    ver = run(S.Behaviour(**amp, p_aggressive=0.2), seconds=900, seed=7)
    tot = sum(ver["per_type"])
    res["verification"] = {"amplification": A, "effective_amplification": eff, "raw": ver,
                           "crashes_per_MEV_scaled": tot / A, "share": dict(zip(TYPES, (np.array(ver["per_type"]) / max(tot, 1e-9)).round(3).tolist())),
                           "rlr_share": ver["rlr"] / max(tot, 1e-9)}
    print("verification", json.dumps(res["verification"], default=float), flush=True)
    with open(os.path.join(out, "calibration.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)


def av_sweep(out="results", shares=(0.0, 0.25, 0.5, 1.0), A=25.0):
    with open(os.path.join(out, "calibration.json")) as f:
        knobs = json.load(f)["calibrated"]
    amp = {k: min(v * A, 1.0) for k, v in knobs.items()} | {"p_aggressive": 0.2}
    rows = []
    for p in shares:
        r = run(S.Behaviour(**amp, p_av=p), seconds=900, seed=11)
        tot = sum(r["per_type"])
        rows.append({"av_share": p, "crashes_per_MEV": tot / A, **{t: v / A for t, v in zip(TYPES, r["per_type"])}, "crashes": r["crashes"], "mev": r["mev"]})
        print(rows[-1], flush=True)
    with open(os.path.join(out, "av_sweep.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    (av_sweep if "av" in sys.argv[1:] else main)()
