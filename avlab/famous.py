"""Famous long-tail events as scenario families, and how hard they really are.

Each scenario is the simulator's version of a recognisable real-world failure pattern ("archetype"), drawn with a random
intensity u (training draws: u in [0.4, 1.0]; test draws: u in [0.8, 1.2], so the strongest test draws are harder than anything
seen in training). The simulator is not a reconstruction of the real incidents, only of the mechanism the public reports
describe, and every scenario says which of its knobs carries it and which counter shows it happening.

    python -m avlab.famous difficulty      # rule-based AV and the start policy in every scenario  (~8 min) -> results/famous/difficulty.json
    python -m avlab.famous evaluate A B C  # trained policies (avrl.famous_train) on test draws of every scenario (~5 min each)
"""
import json
import os
import sys
import time

import numpy as np

from . import adversary as AD
from xnet import sim as X

OUT = "results/famous"
TRAIN_U, TEST_U = (0.4, 1.0), (0.8, 1.2)

# name: (archetype, public case it echoes, knobs as a function of intensity u, the counter that shows it)
SCENARIOS = {
    "jaywalker_misclassified": (
        "a pedestrian crosses outside the signal phase and the AV's perception drops out while it approaches",
        "Tempe, 2018: a pedestrian crossing mid-block at night; the NTSB found the system repeatedly reclassified her",
        lambda u: dict(p_ped_violate=min(0.15 + 0.35 * u, 0.95), ped_scale=2.0 + 2.0 * u, p_blind=4e-4 * u), "pedestrian"),
    "bus_dartout": (
        "a pedestrian steps out from in front of a stopped bus or van where cars cannot see",
        "the child-darts-out-from-behind-a-bus pattern in school-zone safety guidance",
        lambda u: dict(p_dart=min(0.5 * u, 1.0), p_bus=0.03 + 0.1 * u, p_van=0.05 + 0.15 * u, ped_scale=3.0), "tail_masked_ped"),
    "stalled_vehicle_missed": (
        "a vehicle is stopped in the lane and the AV's sensor drops it from view while closing",
        "NHTSA's 2021 investigation of driver-assist crashes into stopped first-responder vehicles",
        lambda u: dict(breakdown_rate=1e-4 * u, p_blind=3e-4 * u), "tail_stalled"),
    "phantom_braking": (
        "the AV brakes hard for an obstacle that is not there while traffic follows",
        "NHTSA's 2022 evaluation of phantom-braking complaints",
        lambda u: dict(p_ghost=3e-3 * u), "av_hard"),
    "sun_glare_blackout": (
        "the AV's sensors are blinded for a second or two at a time",
        "low-sun camera blinding, a standard perception test case",
        lambda u: dict(p_blind=1e-3 * u), "tail_blind"),
    "aggressive_cut_in": (
        "drivers squeeze into a gap far smaller than normal right in front of the AV",
        "short-headway cut-ins, the commonest complaint of cautious AVs",
        lambda u: dict(adv_cut=1.0 + 3.0 * u, p_aggressive=0.6), "tail_cutin"),
    "red_light_runner": (
        "many drivers run the red and enter the junction at speed",
        "the T-bone at a signalised junction, the largest single class of urban crashes",
        lambda u: dict(p_red=min(0.05 * (1 + 10 * u), 0.95), red_window=0.5 + 2.5 * u, p_nolook=1e-3 * u), "rlr_crashes"),
    "unprotected_left": (
        "drivers turn across oncoming traffic without looking",
        "the left-turn-across-path crash, common to every junction type",
        lambda u: dict(p_nolook=2e-3 * u, p_aggressive=0.5, demand=1.2), "left-turn"),
    "cyclist_close_pass": (
        "many cyclists, and aggressive overtakers who clip them",
        "the close-pass pattern behind most urban cyclist injuries",
        lambda u: dict(p_bike=0.05 + 0.15 * u, p_close_pass=0.02 * u, p_aggressive=0.5), "bicycle"),
    "black_ice": (
        "the road is slippery: braking distances double",
        "winter-weather disengagements and crashes",
        lambda u: dict(friction=1.0 - 0.65 * u, demand=0.9), "rear-end"),
}
NOMINAL = {"nominal x1": dict(risk=1.0), "nominal x5": dict(risk=5.0)}
# everything bad at once, in every copy: the control that has the stress but not the named structure
GENERIC = dict(risk=15.0, breakdown_rate=1e-4, p_dart=0.3, p_blind=2e-4, p_ghost=1e-3, adv_cut=3.0, p_ped_violate=0.15)


def draw(name, rng, split="train"):
    """Behaviour overrides of one random draw of scenario `name`: (kwargs for avlab.adversary.world, intensity)."""
    lo, hi = TRAIN_U if split == "train" else TEST_U
    u = float(rng.uniform(lo, hi))
    kw = dict(risk=float(rng.choice([1.0, 5.0])))
    kw.update(SCENARIOS[name][2](u))
    return kw, u


def behaviour(kw):
    kw = dict(kw)
    return AD.world(kw.pop("risk", 1.0), **kw)


def run_condition(arena, kw_list, seconds=250.0, warm=100.0):
    """Sum the metrics of one arena over several behaviours (one warm-up each)."""
    tot = None
    for kw in kw_list:
        beh = behaviour(kw)
        st = arena.warm(beh, warm)
        _, d = arena.measure(st, beh, seconds)
        tot = d if tot is None else {k: tot[k] + d[k] for k in d}
    return tot


def summarise(d):
    h = max(d["av_veh_s"] / 3600, 1e-9)
    return {"av_hours": h, "crash_per_h": d["av_crash"] / h, "crashes": d["av_crash"], "speed_kmh": 3.6 * d["av_veh_m"] / max(d["av_veh_s"], 1),
            "hard_per_h": d["av_hard"] / h, **{k: d[k] for k in X.TAILS + ("rlr_crashes", "pedestrian", "bicycle", "left-turn", "rear-end", "av_hard")}}


def conditions(split="test", seed=0, per=2):
    """{name: [kwargs ...]}: every scenario at `per` random intensities, and the two nominal worlds."""
    rng = np.random.default_rng(seed)
    out = {n: [draw(n, rng, split)[0] for _ in range(per)] for n in SCENARIOS}
    out.update({n: [kw] * per for n, kw in NOMINAL.items()})
    return out


def difficulty():
    """Rule-based AV (IDM envelope) and the unfiltered start policy in every scenario: how often does its signature event happen, and
    how much more do AVs crash than in the nominal world?"""
    from avrl import xppo as P
    os.makedirs(OUT, exist_ok=True)
    res = {}
    conds = conditions("test", seed=1)
    for who, pol, shield in (("rule-based AV (IDM envelope)", None, "idm"), ("start policy, no filter", P.deterministic(P.load(AD.POLICY)), "off")):
        arena = AD.Arena(policy=pol, shield=shield, n=8, seed=5)
        res[who] = {}
        for name, kws in conds.items():
            t0 = time.time()
            res[who][name] = summarise(run_condition(arena, kws))
            r = res[who][name]
            print(f"{who:30s} {name:26s} {r['crash_per_h']:.3f} crashes/AV-h ({r['crashes']:.0f})  signature {SCENARIOS[name][3] if name in SCENARIOS else '-'}="
                  f"{r.get(SCENARIOS[name][3], 0) if name in SCENARIOS else 0:.0f}  {time.time() - t0:.0f} s", flush=True)
            json.dump(res, open(f"{OUT}/difficulty.json", "w"), indent=1)


def evaluate(tags):
    from avrl import xppo as P
    p = f"{OUT}/eval.json"
    res = json.load(open(p)) if os.path.exists(p) else {}
    conds = conditions("test", seed=2)
    for tag in tags:
        arena = AD.Arena(policy=P.deterministic(P.load(f"{OUT}/policy_{tag}.npz")), shield="off", n=8, seed=7)
        res[tag] = {}
        for name, kws in conds.items():
            res[tag][name] = summarise(run_condition(arena, kws))
            print(f"{tag} {name:26s} {res[tag][name]['crash_per_h']:.3f}  {res[tag][name]['speed_kmh']:.1f} km/h", flush=True)
            json.dump(res, open(p, "w"), indent=1)


def report():
    """Markdown tables for docs/FAMOUS_EVENTS.md from results/famous/*.json."""
    d = json.load(open(f"{OUT}/difficulty.json"))
    names = list(SCENARIOS) + list(NOMINAL)
    print("| scenario | signature counter | rule-based AV: crashes / AV-h | start policy, no filter: crashes / AV-h | x nominal x5 |")
    print("|---|---|---|---|---|")
    base = d["start policy, no filter"]["nominal x5"]["crash_per_h"]
    for n in names:
        r, s_ = d["rule-based AV (IDM envelope)"][n], d["start policy, no filter"][n]
        sig = SCENARIOS[n][3] if n in SCENARIOS else "-"
        print(f"| {n} | {sig} = {s_.get(sig, 0):.0f} | {r['crash_per_h']:.3f} ({r['crashes']:.0f}) | {s_['crash_per_h']:.3f} ({s_['crashes']:.0f}) | {s_['crash_per_h'] / base:.1f} |")
    e = json.load(open(f"{OUT}/eval.json"))
    tags = [t for t in ("S", "A", "B", "C", "A1", "B1", "C1") if t in e]
    print("\n| held-out scenario | " + " | ".join(tags) + " |\n|---|" + "---|" * len(tags))
    tot = {t: [0.0, 0.0] for t in tags}
    for n in names:
        print(f"| {n} | " + " | ".join(f"{e[t][n]['crash_per_h']:.3f} ({e[t][n]['crashes']:.0f})" for t in tags) + " |")
        for t in tags:
            tot[t][0] += e[t][n]["crashes"]
            tot[t][1] += e[t][n]["av_hours"]
    print("| **all conditions** | " + " | ".join(f"{tot[t][0] / tot[t][1]:.3f}" for t in tags) + " |")
    print("| speed, nominal x5 (km/h) | " + " | ".join(f"{e[t]['nominal x5']['speed_kmh']:.1f}" for t in tags) + " |")


if __name__ == "__main__":
    {"report": report, "difficulty": lambda: difficulty(), "evaluate": lambda: evaluate(sys.argv[2:])}[sys.argv[1]]()
