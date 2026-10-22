"""AVBench: the bar a learned driving policy has to clear, scenario by scenario.

The training reward says how well a policy optimises its own objective. AVBench asks the questions a deployment
review would ask, in eleven scenarios, against the rule-based AV driving the *same* traffic (same warm state, same
random numbers, so every difference is the policy's):

  nominal        Oakland, calibrated human drivers, 25 % AVs
  rush_hour      demand x1.45: queues spill back through junctions
  red_runners    cross traffic runs reds (60 % of humans willing, up to 3 s into the red)
  inattentive    humans who pull out without looking and zone out behind the wheel
  jaywalkers     2.5x pedestrians, 40 % of them cross against the signal or without a gap
  blockages      breakdowns and double-parked vans block lanes: the AV has to wait, merge and pass
  av_majority    80 % AVs: the policy mostly meets copies of itself
  cyclists       15 % of trips by bicycle: slow riders in the lane, overtaking, close passes
  wet_road       friction 0.5: the hardest possible braking halves
  squirrel_hill  an unseen neighbourhood (nominal traffic)
  downtown       an unseen street grid (nominal traffic)

Every learned policy is run three times: under the IDM safety envelope (as deployed), under the control barrier function
(xnet.sim.cbf_bound: braking-distance, junction time-to-conflict and merge-gap barriers), and with no filter (the raw
policy, which is what RL actually learned). Eight checks per scenario, each against the rule-based AV in the same scenario:

  as deployed (with the envelope)
    safe            AV crashes per AV-hour <= the rule-based AV's
    efficient       AV mean speed >= 97 % of the rule-based AV's
    smooth          hard brakes (< -4 m/s^2) per AV-hour <= the rule-based AV's
    good neighbour  human drivers' mean speed >= 99 % of theirs next to rule-based AVs
  the raw policy (no envelope)
    safe unaided    AV crashes per AV-hour <= the rule-based AV's
    lawful          AV red-light runs <= the rule-based AV's
    courteous       near misses (entering a junction just ahead of a car with priority) per AV-hour <= the rule-based AV's
    self-reliant    the envelope would cut the policy's command in < 5 % of AV time

A scenario counts as *overcome* when all eight pass. These relative checks have a flaw: the rule-based AV passes all 88 by
construction. So there is also an absolute tier (avrl.scorecard.ABSOLUTE: fixed bars from outside the simulator, e.g. half the
humans' crash rate, nuPlan's comfort bound) that the rule-based AV fails too, over the 11 core scenarios plus 7 long-tail ones
(TAIL: the famous-incident archetypes), and a scorecard of continuous metrics with one number to optimise (the av_world score). Jobs are split by scenario (~1-2 min each):

    python -m avrl.bench train                          # the self-reliant policy (~4 min)
    python -m avrl.bench run nominal rush_hour ...     # -> results/bench/<scenario>.json (only the missing policies)
    python -m avrl.bench run glare dartout ...         # the long-tail scenarios
    python -m avrl.bench run-wm                         # the latent world-model planner (3 scenarios x 2 modes, ~10 min each)
    python -m avrl.bench report                         # -> results/bench/leaderboard.json, leaderboard.md
    python -m avrl.bench plot                           # -> docs/img/avbench.png
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

from . import ladder as LD
from . import xppo as P

OUT = "results/bench"
N_ENVS, WARM, SECONDS = 8, 300.0, 180.0

SCENARIOS = {
    "nominal": ("oakland", {}),
    "rush_hour": ("oakland", dict(demand=1.45)),
    "red_runners": ("oakland", dict(p_red=0.6, red_window=3.0)),
    "inattentive": ("oakland", dict(p_nolook=1e-2, distract_rate=4e-5)),
    "jaywalkers": ("oakland", dict(ped_scale=2.5, p_ped_violate=0.4)),
    "blockages": ("oakland", dict(breakdown_rate=1e-4, p_van=0.25, p_park=0.9)),
    "av_majority": ("oakland", dict(p_av=0.8)),
    "cyclists": ("oakland", dict(p_bike=0.15)),
    "wet_road": ("oakland", dict(friction=0.5)),
    "squirrel_hill": ("squirrel_hill", {}),
    "downtown": ("downtown", {}),
}
# the long tail: the famous-incident archetypes (avlab.famous) at intensity 1, on top of nominal traffic. The perception faults
# (glare, misclassification, a missed stalled car) hit the AV's own sensing, which is where the rule-based AV is fragile.
TAIL = {"glare": "sun_glare_blackout", "phantom": "phantom_braking", "stalled_blind": "stalled_vehicle_missed",
        "jaywalker_blind": "jaywalker_misclassified", "dartout": "bus_dartout", "cut_in": "aggressive_cut_in", "black_ice": "black_ice"}


def _tail(name):
    from avlab import famous as FM
    return ("oakland", FM.SCENARIOS[TAIL[name]][2](1.0))


SCENARIOS.update({k: _tail(k) for k in TAIL})

POLICIES = {  # name -> checkpoint (None = the rule-based AV, the bar to clear)
    "rule-based AV": None,
    "imitation (BC)": "results/ladder/bc/policy_bc.npz",
    "PPO, AVs-only world": "results/ladder/w1/policy_120.npz",
    "PPO, full world": "results/ladder/w5/policy_120.npz",
    "PPO, full world, no imitation": "results/ladder/s5/policy_120.npz",
    "PPO, failure-focused replay": "results/active/focus/policy_120.npz",
    "PPO, full world, 10x envelope penalty": "results/bench/self_reliant/policy_120.npz",
    "PPO, 130k steps (xnet)": "results/xnet/policy.npz",
    "PPO, 234k steps (xnet3)": "results/xnet3/policy.npz",
    "PPO + barrier teacher (phase 1)": "results/unaided/phase1/policy.npz",
    "PPO + barrier teacher, then no filter (phase 2)": "results/unaided/phase2/policy.npz",
    "PPO + barrier teacher, speed + jaywalkers (phase 3)": "results/unaided/phase3/policy.npz",
    "latent world-model planner (wm-mpc)": "wmplan",
}
# the latent planner replans every AV every step (~10 min per run here), so it runs on the junction-heavy scenarios it was built
# for, unaided and with the envelope, in two halves per run
WM, WM_SCENARIOS, WM_MODES, WM_K = "latent world-model planner (wm-mpc)", ("nominal", "red_runners", "inattentive"), ("raw", "shielded"), 112
STEPS = {"imitation (BC)": 0, "PPO, full world": 62_400, "PPO, 130k steps (xnet)": 130_000, "PPO, 234k steps (xnet3)": 234_000}
RENAME = {"PPO, long run (xnet3)": "PPO, 234k steps (xnet3)"}

CHECKS = ("safe", "efficient", "smooth", "good neighbour", "safe unaided", "lawful", "courteous", "self-reliant")


def train_self_reliant():
    """The full-world ladder recipe, with the envelope penalty x10 (0.04 -> 0.4 per m/s^2 cut): the fix AVBench points to."""
    P.train(f"{OUT}/self_reliant", updates=LD.UPDATES, init=f"{LD.OUT}/bc/policy_bc.npz", beh=LD.world("5"), ckpt_at=(LD.UPDATES,),
            n_envs=LD.N_ENVS, warm=150.0, reward=LD.REWARD | {"shield": 0.4})


def behaviour(over):
    return X.Behaviour(**({k: v for k, v in LD.knobs().items()} | dict(p_av=0.25, demand=1.0) | over))


def rates(m):
    """Per-copy metrics from summed counters m[E, NM]."""
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    av_h = np.maximum(d["av_veh_s"] / 3600, 1e-9)
    hum_s = np.maximum(d["veh_s"] - d["av_veh_s"], 1e-9)
    crashes = sum(d[k] for k in X.CRASH_NAMES)
    return {**{k: d[k] for k in X.AV_CRASH}, "av_hours": d["av_veh_s"] / 3600, "av_crash": d["av_crash"], "av_red": d["av_red"], "av_near": d["av_near"],
            "av_hard": d["av_hard"], "av_crash_per_h": d["av_crash"] / av_h, "av_near_per_h": d["av_near"] / av_h,
            "av_hard_per_h": d["av_hard"] / av_h, "av_speed_kmh": 3.6 * d["av_veh_m"] / np.maximum(d["av_veh_s"], 1e-9),
            "av_jerk_rms": np.sqrt(d["av_jerk2"] / np.maximum(d["av_veh_s"], 1e-9)),
            "shield_share": d["av_shield"] * X.DT / np.maximum(d["av_veh_s"], 1e-9),
            "human_speed_kmh": 3.6 * (d["veh_m"] - d["av_veh_m"]) / hum_s,
            "human_crash_per_h": (crashes - d["av_crash"]) / (hum_s / 3600), "teleports": d["teleports"]}


def runner(area, beh, chunks=1, planner=None):
    """Jitted rollouts from one fixed warm state: rule-based, and a policy with / without the envelope (or under the CBF).
    planner: a policy function for xnet.sim.step(rl=(f, key, True)) instead of PPO parameters (the latent planner)."""
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    sim = X.Sim(net, n_envs=N_ENVS, beh=beh, seed=123, reroute_s=0)
    sim.run(WARM)
    st0 = sim.st
    n = int(SECONDS / X.DT)
    keys = jax.random.split(jax.random.PRNGKey(7), n)
    parts = np.array_split(np.arange(n), chunks)  # the same steps and keys, in pieces (identical results, shorter jobs)

    def make(shield):
        @jax.jit
        def go(params, st, kk):
            def body(st, k):
                ks = jax.random.split(k, N_ENVS)
                if planner is not None:
                    return jax.vmap(lambda s, k2: X.step(s, J, beh, nh, rl=(planner, k2, True))[0])(st, ks), None
                if params is None:
                    return jax.vmap(lambda s, k2: X.step(s, J, beh, nh))(st, ks), None
                return jax.vmap(lambda s, k2: X.step(s, J, beh, nh, rl=(P.deterministic(params), k2))[0])(st, ks), None
            return jax.lax.scan(body, st, kk)[0]

        def out(params):
            X.SHIELD = shield  # read when go is traced (its first call)
            try:
                st = st0
                for i, idx in enumerate(parts):
                    st = go(params, st, keys[idx])
                    if chunks > 1:
                        print(f"  part {i + 1}/{chunks}", flush=True)
                return np.asarray(st.m - st0.m)
            finally:
                X.SHIELD = "idm"
        return out
    return {"shielded": make("idm"), "raw": make("off"), "cbf": make("cbf")}


def evaluate(policy, scenarios=("nominal",)):
    """The scorecard of one policy (a checkpoint path, PPO parameters, or None for the rule-based AV) on bench scenarios, from the
    same warm states and random numbers as the leaderboard: {"av_score": mean, "scenarios": {name: scorecard}}. An objective for
    model selection or a hyperparameter search (~1 min per scenario)."""
    from . import scorecard as SC
    params = P.load(policy) if isinstance(policy, str) else policy
    out = {}
    for n in scenarios:
        area, over = SCENARIOS[n]
        r = runner(area, behaviour(over))
        raw = r["raw"](params)
        out[n] = SC.scorecard(raw, r["shielded"](params) if params is not None else raw)
    return {"av_score": float(np.mean([v["av_score"] for v in out.values()])), "scenarios": out}


def run_wm(name, mode):
    """One mode of the latent planner in one scenario (~10 min)."""
    from wmplan import model as M
    from wmplan import planner as PL
    path_out = f"{OUT}/{name}.json"
    d = json.load(open(path_out))
    row = d["runs"].setdefault(WM, {})
    if mode in row:
        return
    area, over = SCENARIOS[name]
    PL.K_PLAN = WM_K  # every AV in a copy (~75-105 here), not the 48 the planner was tuned with
    t0 = time.time()
    go = runner(area, behaviour(over), chunks=2, planner=PL.make(M.load(), P.load("results/xnet3/policy.npz")))[mode]
    row[mode] = go(None).tolist()
    with open(path_out, "w") as f:
        json.dump(d, f)
    print(f"{name} {WM} {mode}: {time.time() - t0:.0f} s", flush=True)


def run(name):
    area, over = SCENARIOS[name]
    t0 = time.time()
    path_out = f"{OUT}/{name}.json"
    rows = json.load(open(path_out))["runs"] if os.path.exists(path_out) else {}
    rows = {RENAME.get(k, k): v for k, v in rows.items()}
    todo = [p for p in POLICIES if p not in rows and p != WM]
    if not todo:
        return
    r_ = runner(area, behaviour(over))
    on, off, cbf = r_["shielded"], r_["raw"], r_["cbf"]
    for pol in todo:
        path = POLICIES[pol]
        if path is None:
            m = on(None)
            rows[pol] = {"shielded": m.tolist(), "raw": m.tolist(), "cbf": cbf(None).tolist()}
        else:
            params = P.load(path)
            rows[pol] = {"shielded": on(params).tolist(), "raw": off(params).tolist(), "cbf": cbf(params).tolist()}
        r = rates(np.array(rows[pol]["raw"]))
        print(f"{name:14s} {pol:30s} raw: crash/h {r['av_crash'].sum() / r['av_hours'].sum():.3f}  near {int(r['av_near'].sum()):4d}  "
              f"hard/h {r['av_hard'].sum() / r['av_hours'].sum():5.1f}  speed {r['av_speed_kmh'].mean():5.1f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(path_out, "w") as f:
        json.dump({"scenario": name, "area": area, "over": over, "n_envs": N_ENVS, "seconds": SECONDS, "metrics": X.METRICS,
                   "runs": rows}, f)
    print(f"{name}: {time.time() - t0:.0f} s", flush=True)


def pooled(m):
    """Totals over the copies -> one row of rates (pooled, so rare events are counted over every AV-hour)."""
    r = rates(m.sum(0, keepdims=True))
    return {k: float(v[0]) for k, v in r.items()}


def ci(a, b, key):
    """95 % bootstrap interval of the per-copy paired difference a - b (same traffic, so differences are the policy's)."""
    d = rates(a)[key] - rates(b)[key]
    rng = np.random.default_rng(0)
    bs = rng.choice(d, (2000, len(d))).mean(1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def checks(pol, base):
    """pol / base: {"raw": m, "shielded": m}. Returns {check: (passed, value, bar)}."""
    pr, br, ps, bs = pooled(pol["raw"]), pooled(base["raw"]), pooled(pol["shielded"]), pooled(base["shielded"])
    return {
        "safe": (ps["av_crash_per_h"] <= bs["av_crash_per_h"], ps["av_crash_per_h"], bs["av_crash_per_h"]),
        "efficient": (ps["av_speed_kmh"] >= 0.97 * bs["av_speed_kmh"], ps["av_speed_kmh"], bs["av_speed_kmh"]),
        "smooth": (ps["av_hard_per_h"] <= bs["av_hard_per_h"], ps["av_hard_per_h"], bs["av_hard_per_h"]),
        "good neighbour": (ps["human_speed_kmh"] >= 0.99 * bs["human_speed_kmh"], ps["human_speed_kmh"], bs["human_speed_kmh"]),
        "safe unaided": (pr["av_crash_per_h"] <= br["av_crash_per_h"], pr["av_crash_per_h"], br["av_crash_per_h"]),
        "lawful": (pr["av_red"] <= br["av_red"], pr["av_red"], br["av_red"]),
        "courteous": (pr["av_near_per_h"] <= br["av_near_per_h"], pr["av_near_per_h"], br["av_near_per_h"]),
        "self-reliant": (ps["shield_share"] < 0.05, ps["shield_share"], 0.05),
    }


def _align(m, names):
    """Counters saved under an older metric list -> the current X.METRICS order (missing counters are 0)."""
    m = np.asarray(m, float)
    out = np.zeros(m.shape[:-1] + (len(X.METRICS),))
    for i, k in enumerate(names):
        if k in X.METRICS:
            out[..., X.METRICS.index(k)] = m[..., i]
    return out


def _runs(n):
    d = json.load(open(f"{OUT}/{n}.json"))
    names = d.get("metrics", X.METRICS)
    return X.METRICS, {RENAME.get(p, p): {k: _align(v, names) for k, v in r.items()} for p, r in d["runs"].items()}


def _score(names_m, r):
    from . import scorecard as SC
    return SC.scorecard(r["raw"], r.get("shielded"), r.get("cbf"), names=names_m)


def report():
    """results/bench/leaderboard.{json,md}: the relative tier (vs the rule-based AV, the 11 core scenarios), the absolute tier (fixed
    bars, every scenario) and the scorecard (continuous metrics and the av_world score)."""
    from . import scorecard as SC
    core = [n for n in SCENARIOS if n not in TAIL and os.path.exists(f"{OUT}/{n}.json")]
    tail = [n for n in TAIL if os.path.exists(f"{OUT}/{n}.json")]
    names = core + tail
    data = {n: _runs(n) for n in names}
    rel_pols = [p for p in POLICIES if p != WM]
    # ---- relative tier: the original eight checks against the rule-based AV, core scenarios
    board = {p: {"passed": 0, "overcome": [], "scenarios": {}} for p in rel_pols}
    for n in core:
        _, runs = data[n]
        base = runs["rule-based AV"]
        for p in rel_pols:
            c = checks(runs[p], base)
            ok = sum(v[0] for v in c.values())
            board[p]["passed"] += int(ok)
            if ok == len(CHECKS):
                board[p]["overcome"].append(n)
            board[p]["scenarios"][n] = {"checks": {k: [bool(v[0]), v[1], v[2]] for k, v in c.items()}, "passed": int(ok),
                                        "raw": pooled(runs[p]["raw"]), "shielded": pooled(runs[p]["shielded"]),
                                        "speed_diff_ci": ci(runs[p]["shielded"], base["shielded"], "av_speed_kmh"),
                                        "hard_diff_ci": ci(runs[p]["shielded"], base["shielded"], "av_hard_per_h")}
    # ---- absolute tier and scorecard: every scenario a policy has complete runs for
    absolute, scores = {}, {}
    for n in names:
        mnames, runs = data[n]
        for p, r in runs.items():
            if p not in POLICIES or "raw" not in r or "shielded" not in r:
                continue
            sc = _score(mnames, r)
            ab = SC.absolute_checks(sc["metrics"])
            absolute.setdefault(p, {})[n] = ab
            scores.setdefault(p, {})[n] = sc
    with open(f"{OUT}/leaderboard.json", "w") as f:
        json.dump({"scenarios": core, "checks": CHECKS, "board": board, "tail_scenarios": tail, "absolute_checks": list(SC.ABSOLUTE),
                   "absolute": absolute, "scores": scores, "wm_scenarios": list(WM_SCENARIOS)}, f, indent=1, default=float)

    def mean(p, ns, key=None):
        v = [scores[p][n]["av_score"] if key is None else scores[p][n]["metrics"][key] for n in ns if n in scores.get(p, {})]
        return float(np.mean(v)) if v else float("nan")

    full = [p for p in rel_pols if all(n in absolute.get(p, {}) for n in names)]
    ncheck = len(SC.ABSOLUTE)
    L = [f"## Absolute tier: fixed bars, {len(names)} scenarios ({len(core)} core + {len(tail)} long-tail) x {ncheck} checks", "",
         f"| policy | checks passed (of {len(names) * ncheck}) | core | long-tail | scenarios cleared | most failed |", "|---|---|---|---|---|---|"]
    for p in sorted(full, key=lambda p: -sum(sum(absolute[p][n].values()) for n in names)):
        tot = sum(sum(absolute[p][n].values()) for n in names)
        fails = {}
        for n in names:
            for k, v in absolute[p][n].items():
                fails[k] = fails.get(k, 0) + (not v)
        worst = ", ".join(f"{k} ({v})" for k, v in sorted(fails.items(), key=lambda x: -x[1]) if v)[:90]
        L.append(f"| {p} | {tot} | {sum(sum(absolute[p][n].values()) for n in core)} | {sum(sum(absolute[p][n].values()) for n in tail)} | "
                 f"{sum(all(absolute[p][n].values()) for n in names)} | {worst} |")
    wm_ns = [n for n in WM_SCENARIOS if n in absolute.get(WM, {})]
    if wm_ns:
        L += ["", f"Latent world-model planner, on the scenarios it ran ({', '.join(wm_ns)}), against every policy on the same ones:", "",
              f"| policy | absolute checks (of {len(wm_ns) * ncheck}) | av_world score | crashes /AV-h unaided | harm /AV-h unaided | comfort pass | progress |",
              "|---|---|---|---|---|---|---|"]
        for p in sorted([q for q in POLICIES if all(n in absolute.get(q, {}) for n in wm_ns)], key=lambda q: -mean(q, wm_ns)):
            L.append(f"| {p} | {sum(sum(absolute[p][n].values()) for n in wm_ns)} | {mean(p, wm_ns):.1f} | {mean(p, wm_ns, 'crash_per_h_unaided'):.3f} | "
                     f"{mean(p, wm_ns, 'harm_per_h_unaided'):.3f} | {mean(p, wm_ns, 'comfort_pass'):.3f} | {mean(p, wm_ns, 'progress'):.3f} |")
    L += ["", "## Scorecard: the numbers to optimise (means over scenarios; worst = the lowest long-tail scenario)", "",
          "| policy | av_world score: core | long-tail | worst | driving score | crashes /AV-h | unaided | crash UCB95 | harm unaided | "
          "comfort pass | jerk rms | progress | filter share | barrier exits | veto s /AV-h |", "|" + "---|" * 15]
    for p in sorted(full, key=lambda p: -mean(p, names)):
        worst = min(scores[p][n]["av_score"] for n in tail) if tail else float("nan")
        L.append(f"| {p} | {mean(p, core):.1f} | {mean(p, tail):.1f} | {worst:.1f} | {mean(p, names, 'driving_score'):.1f} | "
                 f"{mean(p, names, 'crash_per_h'):.3f} | {mean(p, names, 'crash_per_h_unaided'):.3f} | {mean(p, names, 'crash_ucb95_per_h'):.3f} | "
                 f"{mean(p, names, 'harm_per_h_unaided'):.3f} | {mean(p, names, 'comfort_pass'):.3f} | {mean(p, names, 'jerk_rms'):.2f} | "
                 f"{mean(p, names, 'progress'):.3f} | {mean(p, names, 'filter_share'):.1%} | {mean(p, names, 'barrier_exit_share'):.2%} | "
                 f"{mean(p, names, 'barrier_veto_s_per_h'):.1f} |")
    # ---- the relative tier, as before
    L += ["", f"## Relative tier: against the rule-based AV on the same traffic, {len(core)} core scenarios x {len(CHECKS)} checks", "",
          f"| policy | checks passed (of {len(core) * len(CHECKS)}) | scenarios overcome (of {len(core)}) | " + " | ".join(core) + " |",
          "|---|---|---|" + "---|" * len(core)]
    for p in sorted(rel_pols, key=lambda p: -board[p]["passed"]):
        b = board[p]
        L.append(f"| {p} | {b['passed']} | {len(b['overcome'])} | " + " | ".join(f"{b['scenarios'][n]['passed']}/{len(CHECKS)}" for n in core) + " |")
    L += ["", "", "| scenario | policy | crashes /AV-h | raw crashes /AV-h | raw near misses /AV-h | hard brakes /AV-h | AV km/h | human km/h | envelope cuts |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in core:
        for p in rel_pols:
            s_ = board[p]["scenarios"][n]
            r, sh = s_["raw"], s_["shielded"]
            L.append(f"| {n} | {p} | {sh['av_crash_per_h']:.3f} | {r['av_crash_per_h']:.3f} | {r['av_near_per_h']:.2f} | {sh['av_hard_per_h']:.1f} | "
                     f"{sh['av_speed_kmh']:.1f} | {sh['human_speed_kmh']:.1f} | {sh['shield_share']:.1%} |")
    md = "\n".join(L) + "\n"
    with open(f"{OUT}/leaderboard.md", "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        for n in sys.argv[2:] or list(SCENARIOS):
            run(n)
    elif cmd == "run-wm":  # run-wm [scenario [mode]]: the latent planner, one (scenario, mode) per call
        for n in sys.argv[2:3] or WM_SCENARIOS:
            for mode in sys.argv[3:4] or WM_MODES:
                run_wm(n, mode)
    elif cmd == "train":
        train_self_reliant()
    elif cmd == "report":
        report()
    elif cmd == "plot":
        from .bench_plot import main as plot

        plot()
