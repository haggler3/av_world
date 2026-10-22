"""Quantitative metrics for an AV policy, from the simulator's counters (xnet.sim.METRICS), to report and to optimise.

AVBench's pass/fail checks say whether a policy clears a bar; these are the continuous numbers underneath, so that training and
model selection have something to climb. Some are the field's usual metrics, some only make sense in this simulator because of
how it works (typed crashes with the humans' own crash rate in the same run, a safety filter whose interventions are counted, a
control barrier function whose safe set can be left, paired runs on identical traffic).

    from avrl import scorecard as SC
    s = SC.scorecard(raw=m_no_filter, deployed=m_with_envelope, cbf=m_with_cbf)   # m: summed counters [NM] or [E, NM]
    s["av_score"], s["metrics"]["harm_per_h"], ...

Popular (with the benchmark each comes from):
  crash_per_h, crash_per_1000km   collision rate per hour / distance (every closed-loop benchmark)
  crash_ratio_vs_human            AV crash rate over the humans' in the same run (how AV operators report safety, e.g. Waymo's
                                  comparisons with human benchmarks in the same cities)
  progress                        AV mean speed over the humans' (nuPlan's ego progress, CARLA's route completion, as a rate)
  comfort_pass                    share of 20 s windows with no braking harder than -4 m/s^2 (nuPlan's comfort bound is -4.05)
  hard_brakes_per_h, jerk_rms     the usual comfort rates
  red_runs_per_100km              traffic-light violations (CARLA infraction)
  near_miss_per_h                 entering a junction just ahead of a car with priority: a right-of-way violation, the
                                  simulator's time-to-collision proxy
  driving_score                   CARLA Leaderboard 2.0 style: 100 x progress x product of penalty^count, per 10 km route
                                  (pedestrian or cyclist collision 0.5, vehicle 0.6, red light 0.7, right of way 0.8)
Unique to av_world:
  harm_per_h                      crashes weighted by who gets hurt (pedestrian and cyclist x10, angle and left-turn x3, the
                                  rest x1); the weights are an assumption, change HARM
  crash_ucb95_per_h               the 95 % upper confidence bound on the crash rate given the AV-hours driven: what the run has
                                  *demonstrated*, not just observed (zero crashes in 10 AV-hours demonstrates < 0.3 per hour)
  filter_share                    share of AV time the safety envelope had to cut the policy's command
  filter_dependence               crashes per hour unaided minus with the envelope: how much of the safety is the filter's
  barrier_exit_share              share of moving AV time outside the control barrier function's safe set, unaided
  barrier_veto_s_per_h            seconds per AV-hour the barrier blocks the AV from entering a junction or merge (needs a run
                                  under the CBF): low means the policy already drives as the barrier would
  human_crash_per_h, human_speed  what the AVs do to everyone else (externality), in the same run
The av_world score (0-100, higher is better, the one number to optimise) is the product of five factors in [0, 1]:
  efficiency min(progress, 1) x safety 1 / (1 + harm_per_h unaided / 0.1) x comfort comfort_pass
  x conduct 1 / (1 + (red runs + near misses) per hour unaided / 0.1) x self-reliance (1 - filter_share)
The penalties have CARLA Leaderboard 2.1's additive form, 1 / (1 + rate / scale), so a policy with some crashes still has a gradient
to climb instead of a score of zero. The scales (0.1 per hour) set the exchange rates; change SCALE to weigh things differently.
"""
import math

import numpy as np

from xnet import sim as X

HARM = {"pedestrian": 10.0, "bicycle": 10.0, "angle": 3.0, "left-turn": 3.0, "rear-end": 1.0, "sideswipe": 1.0, "merge": 1.0}
CARLA = {"pedestrian": 0.5, "bicycle": 0.5, "vehicle": 0.6, "red": 0.7, "near": 0.8}  # Leaderboard 2.0 coefficients
ROUTE_KM = 10.0
WINDOW_S = 20.0  # nuPlan scenarios are ~20 s
SCALE = {"harm": 0.1, "conduct": 0.1}


def counters(m, names=X.METRICS):
    m = np.asarray(m, float)
    m = m.sum(0) if m.ndim == 2 else m
    return {k: float(m[i]) for i, k in enumerate(names)}


def poisson_ucb(k, t, conf=0.95):
    """Upper confidence bound on a Poisson rate with k events in exposure t (exact, chi-square)."""
    from scipy.stats import chi2
    return float(chi2.ppf(conf, 2 * k + 2) / 2 / max(t, 1e-9))


def _rates(d):
    h = max(d["av_veh_s"] / 3600, 1e-9)
    km = max(d["av_veh_m"] / 1000, 1e-9)
    hum_s = max(d["veh_s"] - d["av_veh_s"], 1e-9)
    av_types = {c: d["av_" + c] for c in X.CRASH_NAMES}
    human_crashes = sum(d[c] for c in X.CRASH_NAMES) - d["av_crash"]
    return dict(h=h, km=km, crash=d["av_crash"], types=av_types, crash_per_h=d["av_crash"] / h, crash_per_1000km=1000 * d["av_crash"] / km,
                harm_per_h=sum(HARM[c] * n for c, n in av_types.items()) / h,
                human_crash_per_h=max(human_crashes, 0.0) / (hum_s / 3600), speed=3.6 * d["av_veh_m"] / max(d["av_veh_s"], 1e-9),
                human_speed=3.6 * (d["veh_m"] - d["av_veh_m"]) / hum_s, hard_per_h=d["av_hard"] / h,
                jerk_rms=math.sqrt(d["av_jerk2"] / max(d["av_veh_s"], 1e-9)), red=d["av_red"], red_per_h=d["av_red"] / h,
                near=d["av_near"], near_per_h=d["av_near"] / h, filter_share=d["av_shield"] * X.DT / max(d["av_veh_s"], 1e-9),
                barrier_exit_share=d["av_h_neg"] / max(d["av_veh_s"], 1e-9),
                veto_s_per_h=(d["av_veto_junction"] + d["av_veto_merge"]) / h)


def driving_score(r, progress):
    """CARLA Leaderboard 2.0 style: route completion x the product of penalty^count, with infractions counted per ROUTE_KM."""
    per = ROUTE_KM / r["km"]
    vru = r["types"]["pedestrian"] + r["types"]["bicycle"]
    veh = r["crash"] - vru
    pen = CARLA["pedestrian"] ** (vru * per) * CARLA["vehicle"] ** (veh * per) * CARLA["red"] ** (r["red"] * per) * CARLA["near"] ** (r["near"] * per)
    return 100.0 * min(progress, 1.0) * pen


def scorecard(raw, deployed=None, cbf=None, names=X.METRICS):
    """raw: counters of the policy with no filter; deployed: with the envelope (defaults to raw, e.g. for the rule-based AV);
    cbf: under the control barrier function (optional). -> {"metrics": {...}, "factors": {...}, "av_score": float}."""
    R = _rates(counters(raw, names))
    D = _rates(counters(deployed if deployed is not None else raw, names))
    progress = D["speed"] / max(D["human_speed"], 1e-9)
    comfort = math.exp(-D["hard_per_h"] * WINDOW_S / 3600)
    met = {
        # popular
        "crash_per_h": D["crash_per_h"], "crash_per_1000km": D["crash_per_1000km"],
        "crash_ratio_vs_human": D["crash_per_h"] / D["human_crash_per_h"] if D["human_crash_per_h"] > 0 else (0.0 if D["crash"] == 0 else math.inf),
        "progress": progress, "speed_kmh": D["speed"], "comfort_pass": comfort, "hard_brakes_per_h": D["hard_per_h"], "jerk_rms": D["jerk_rms"],
        "red_runs_per_100km": 100 * R["red"] / R["km"], "near_miss_per_h": R["near_per_h"], "driving_score": driving_score(D, progress),
        "crash_per_h_unaided": R["crash_per_h"],
        # unique
        "harm_per_h": D["harm_per_h"], "harm_per_h_unaided": R["harm_per_h"], "crash_ucb95_per_h": poisson_ucb(D["crash"], D["h"]),
        "filter_share": D["filter_share"], "filter_dependence": R["crash_per_h"] - D["crash_per_h"],
        "barrier_exit_share": R["barrier_exit_share"], "human_crash_per_h": D["human_crash_per_h"], "human_speed_kmh": D["human_speed"],
        "av_hours": D["h"],
    }
    if cbf is not None:
        met["barrier_veto_s_per_h"] = _rates(counters(cbf, names))["veto_s_per_h"]
    f = {"efficiency": min(progress, 1.0), "safety": 1.0 / (1.0 + R["harm_per_h"] / SCALE["harm"]), "comfort": comfort,
         "conduct": 1.0 / (1.0 + (R["red_per_h"] + R["near_per_h"]) / SCALE["conduct"]), "self_reliance": 1.0 - D["filter_share"]}
    return {"metrics": met, "factors": f, "av_score": 100.0 * float(np.prod(list(f.values())))}


# --------------------------------------------------------------------------------------------------- AVBench's absolute bars
# Fixed bars that do not depend on the rule-based AV, so it can fail them too. Each is (description, test on the scorecard).
ABSOLUTE = {
    "safer than people": ("crashes per AV-hour with the envelope at most half the humans' rate in the same run (operators report "
                          "55-90 % reductions against human benchmarks)",
                          lambda s: s["crash_ratio_vs_human"] <= 0.5),
    "comfortable": ("at least 95 % of 20 s windows free of braking harder than -4 m/s^2 (nuPlan's comfort bound, -4.05)",
                    lambda s: s["comfort_pass"] >= 0.95),
    "keeps up": ("AV mean speed at least the humans' in the same run", lambda s: s["progress"] >= 1.0),
    "light touch": ("the envelope cuts the policy's command in at most 2 % of AV time", lambda s: s["filter_share"] <= 0.02),
    "safer unaided": ("with no filter, crashes per AV-hour at most half the humans' rate",
                      lambda s: (s["crash_per_h_unaided"] <= 0.5 * s["human_crash_per_h"]) if s["human_crash_per_h"] > 0 else s["crash_per_h_unaided"] == 0),
    "lawful": ("no red-light runs at all, unaided", lambda s: s["red_runs_per_100km"] == 0),
    "right of way": ("no junction entries ahead of a car with priority, unaided", lambda s: s["near_miss_per_h"] == 0),
    "inside the barrier": ("unaided, at most 0.2 % of moving time outside the control barrier function's safe set",
                           lambda s: s["barrier_exit_share"] <= 0.002),
}


def absolute_checks(s):
    """s: scorecard(...)["metrics"] -> {check: passed}."""
    return {k: bool(t(s)) for k, (_, t) in ABSOLUTE.items()}
