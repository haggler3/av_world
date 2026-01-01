"""The sim-to-real DAG (cached with autotag.dag in results/realism/cache): every stage a pure function of its inputs and config.

real_logs -> tracker_noise -> real_clean ----------------------------------------------+
                  |                                                                    |
sim_worlds (default | tail_fit | idm_plus) ---------+                                  |
headway_bank (deconvolution, records sims) -> refit (CEM, records sims) -> sim_patched |
                                                    v                                  v
                                   processed (raw | sensor | clean per world) -> mix -> scoreboard -> following_check
                                                                               -> discriminator -> replay -> report
"""
import json
import os

import numpy as np

from autotag import dag as D
from autotag.dag import stage

from . import discriminator as DS
from . import following as FL
from . import measure as M
from . import mix as MX
from . import refit as RF
from . import simrec as SR
from . import stats as ST

OUT = "results/realism"
CACHE = f"{OUT}/cache"
QUICK = False
# the stored drivers, the earlier tail fit (avlab.tail_calibrate, docs/LONG_TAIL_METHODS.md) and the structural fix alone
WORLDS = {"default": {}, "tail_fit": {"T_med": 0.757, "T_sigma": 0.52, "accel_noise": 0.94, "pulse_rate": 0.0193}, "idm_plus": {"idm_plus": 1.0}}


def cfg(full, quick):
    return quick if QUICK else full


def log(msg):
    print(msg, flush=True)


@stage("real_logs", config={"limit": None})
def real_logs(limit=None):
    from autotag import scenes as AS
    return [M.crop(s) for s in AS.av2_scenes(limit=limit or cfg(400, 120))]


@stage("tracker_noise", deps=("real_logs",))
def tracker_noise(real_logs):
    return M.noise_model(real_logs)


@stage("real_clean", deps=("real_logs", "tracker_noise"))
def real_clean(real_logs, tracker_noise):
    return [M.clean(s, tracker_noise) for s in real_logs]


@stage("sim_worlds", config={"version": 2})
def sim_worlds(version=2):
    sc = SR.record([SR.behaviour(**k) for k in WORLDS.values()], minutes=cfg(4, 2), copies=cfg(3, 1), seed=0)
    return {n: [s for s in sc if s["truth"]["beh"] == i] for i, n in enumerate(WORLDS)}


@stage("headway_bank", deps=("real_clean", "tracker_noise"), config={"iters": 2})
def headway_bank(real_clean, tracker_noise, iters=2):
    real_p10 = FL.p10_headways(real_clean)

    def simulate(bank):
        sc = SR.record(SR.behaviour(idm_plus=1.0, bank=1.0, demand=2.0), minutes=cfg(3, 2), copies=cfg(2, 1), seed=1, bank=bank)
        return FL.p10_headways(RF.process(sc, tracker_noise))
    B, hist = FL.deconvolve(real_p10, simulate, iters=cfg(iters, 1), log=log)
    return {"bank": B, "history": hist, "real_p10": real_p10}


@stage("refit", deps=("real_clean", "tracker_noise", "headway_bank"), config={"iters": 4, "pop": 8, "version": 2})
def refit(real_clean, tracker_noise, headway_bank, iters=4, pop=8, version=2):
    """version 2: the scoreboard includes standstill gaps (version 1 bought tight following with s0 x 0.3) and amax_scale is a knob."""
    real_P = ST.pool([ST.measure(s) for s in real_clean])
    return RF.cem(real_P, tracker_noise, headway_bank["bank"], iters=cfg(iters, 2), pop=cfg(pop, 6), minutes=cfg(3, 2), log=log)


@stage("sim_patched", deps=("refit", "headway_bank"))
def sim_patched(refit, headway_bank):
    """The patched drivers, and the stored drivers in the refit's traffic density (demand only): the density step of the decomposition."""
    sc = SR.record([SR.behaviour(**refit["knobs"]), SR.behaviour(demand=refit["knobs"]["demand"])], minutes=cfg(4, 2), copies=cfg(3, 1),
                   seed=0, bank=headway_bank["bank"])
    return {"patched": [s for s in sc if s["truth"]["beh"] == 0], "dense": [s for s in sc if s["truth"]["beh"] == 1]}


@stage("processed", deps=("sim_worlds", "sim_patched", "tracker_noise"))
def processed(sim_worlds, sim_patched, tracker_noise):
    worlds = dict(sim_worlds, **sim_patched)
    out = {}
    for n, L in worlds.items():
        sens = [M.corrupt(s, tracker_noise, i) for i, s in enumerate(L)]
        out[n] = {"raw": L, "sensor": sens, "clean": [M.clean(s, tracker_noise) for s in sens]}
    return out


@stage("mix", deps=("real_logs", "processed"))
def mix(real_logs, processed):
    return {n: MX.weights(real_logs, d["raw"]) for n, d in processed.items()}


@stage("scoreboard", deps=("real_logs", "real_clean", "processed", "mix"), config={"n_boot": 200})
def scoreboard(real_logs, real_clean, processed, mix, n_boot=200):
    Pr, Prc = ST.pool([ST.measure(s) for s in real_logs]), ST.pool([ST.measure(s) for s in real_clean])
    out = {}
    for n, d in processed.items():
        Ms = {k: [ST.measure(s) for s in d[k]] for k in ("raw", "sensor", "clean")}
        w = mix[n][0]
        out[n] = {"raw": ST.scoreboard(Pr, ST.pool(Ms["raw"]), n_boot, where=False),          # as the logs come: real tracker vs perfect sim
                  "sensor": ST.scoreboard(Pr, ST.pool(Ms["sensor"]), n_boot, where=False),    # the sim seen through the real tracker
                  "clean": ST.scoreboard(Prc, ST.pool(Ms["clean"]), n_boot),                   # both cleaned the same way: driving only
                  "clean_mix": ST.scoreboard(Prc, ST.pool(Ms["clean"], w), n_boot, where=False)}
    return out


@stage("following_check", deps=("real_clean", "processed", "headway_bank"))
def following_check(real_clean, processed, headway_bank):
    out = {"real": {"p10_hw": FL.p10_headways(real_clean).tolist(), "population": FL.pop_summary(FL.population(real_clean))}}
    for n, d in processed.items():
        p = FL.population(d["raw"])
        out[n] = {"p10_hw": FL.p10_headways(d["clean"]).tolist(), "population_raw_vs_truth": FL.pop_summary(p, d["raw"]) if p else None}
    # the fitter's own check: per-episode IDM fits on perfect simulated scenes, where every driver's T is known
    out["episode_fit_check"] = FL.summary(FL.fit_all(processed["idm_plus"]["raw"][:300], truth=True))
    out["bank"] = {"rows": int(len(headway_bank["bank"])), "T_p10_50_90": np.percentile(headway_bank["bank"][:, 0], [10, 50, 90]).tolist(),
                   "history": headway_bank["history"]}
    return out


@stage("discriminator", deps=("real_clean", "processed"), config={"steps": 3000})
def discriminator(real_clean, processed, steps=3000):
    return {n: DS.test(real_clean, d["clean"], steps=cfg(steps, 1000)) for n, d in processed.items()}


@stage("replay", deps=("refit", "headway_bank"), config={"limit": 30})
def replay(refit, headway_bank, limit=30):
    from . import replay as RP
    root = os.environ.get("AV2_ROOT", "data/av2")
    behs = {"default": (SR.behaviour(p_av=0.0), False), "patched": (SR.behaviour(p_av=0.0, **refit["knobs"]), True)}
    return RP.run(root, behs, bank=headway_bank["bank"], limit=cfg(limit, 8))


@stage("report", deps=("tracker_noise", "mix", "scoreboard", "following_check", "discriminator", "replay", "refit", "headway_bank"))
def report(tracker_noise, mix, scoreboard, following_check, discriminator, replay, refit, headway_bank):
    os.makedirs(OUT, exist_ok=True)
    nm = {k: ({a: b for a, b in v.items() if a != "spike_sizes"} if isinstance(v, dict) else v) for k, v in tracker_noise.items()}
    res = {"tracker_noise": nm, "scoreboard": scoreboard, "mix": {n: m[1] for n, m in mix.items()},
           "following": {k: (v if k in ("bank", "episode_fit_check") else {a: b for a, b in v.items() if a != "p10_hw"}) for k, v in following_check.items()},
           "discriminator": discriminator, "replay": replay, "preset": refit["knobs"], "refit_loss": refit["loss"],
           "refit_history": refit["history"]}
    res["decomposition"] = decomposition(scoreboard)
    path = f"{OUT}/quick" if QUICK else OUT
    os.makedirs(path, exist_ok=True)
    json.dump(res, open(f"{path}/scoreboard.json", "w"), indent=1, default=float)
    np.save(f"{path}/bank.npy", headway_bank["bank"])
    json.dump(refit["knobs"], open(f"{path}/preset.json", "w"), indent=1)
    return res


def decomposition(sb):
    """The gap, step by step: as logged -> through the same sensor -> both cleaned -> real scene mix -> structural fix -> fitted patch."""
    steps = [("as logged, stored drivers", sb["default"]["raw"]), ("sim through the real tracker", sb["default"]["sensor"]),
             ("both cleaned (driving only)", sb["default"]["clean"]), ("+ reweighted to the real scene mix", sb["default"]["clean_mix"]),
             ("+ real traffic density (demand)", sb["dense"]["clean"]), ("+ IDM+, headway bank, OU, pulses", sb["patched"]["clean"])]
    return [{"step": n, "gap": s["gap"], **{k: s["tails"][k]["ratio"] for k in ST.TAILS}} for n, s in steps]


def run(target="report", force=(), quick=False):
    global QUICK
    QUICK = quick
    return D.run(target, force=force, log=log, cache=f"{OUT}/cache_quick" if quick else CACHE)
