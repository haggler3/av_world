"""Does a Timed Elastic Band plan better than IDM? (avlab.teb in the policy slot.)

    E1  Oakland, p_av 0.5, friction 1.0: TEB vs careful_idm, 24 paired copies -> AV speed (H1), AV crashes (H2), hard brakes, jerk
    E2  AVBench scorecards on six scenarios for TEB and careful_idm (the leaderboard's states and keys); rule-based AV for reference

Pre-registration: results/daily/2026-11-20/predictions.md

    python -m avlab.tebbench --quick     # tiny E1
    python -m avlab.tebbench --e1        # -> results/teb/e1.json   (~10 min)
    python -m avlab.tebbench --e2        # -> results/teb/e2.json   (~15 min)
    python -m avlab.tebbench             # analysis -> results/teb/teb.json
"""
import argparse
import json
import os

import numpy as np

OUT = "results/teb"
E2_SCEN = ("nominal", "rush_hour", "red_runners", "jaywalkers", "wet_road", "cut_in")


def _policy(name):
    from avlab import teb as T
    return {"teb": T.teb_policy_masked, "idm": T.idm_policy_masked}[name]


def e1_run(name, E=24, seed=31, warm_s=120.0, seconds=360.0, p_av=0.5, chunk_s=40.0):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, _, _ = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated(friction=1.0, p_av=p_av)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(jnp.stack([jax.random.PRNGKey(seed * 1000 + i) for i in range(E)]), J, 1024)
    pol = _policy(name)
    n = int(round(chunk_s / X.DT))

    @jax.jit
    def chunk(st):
        def body(s, _):
            return jax.vmap(lambda si: X.step(si, J, beh, nh, rl=(pol, si.key, True))[0])(s), None
        return jax.lax.scan(body, st, None, length=n)[0]

    for _ in range(int(round(warm_s / chunk_s))):
        st = chunk(st)
    m0 = np.asarray(st.m)
    for _ in range(int(round(seconds / chunk_s))):
        st = chunk(st)
    m = np.asarray(st.m) - m0
    col = lambda k: m[:, X.METRICS.index(k)].tolist()  # noqa: E731
    return {"policy": name, "copies": E, "seed": seed, "seconds": seconds, "p_av": p_av,
            **{k: col(k) for k in ("av_crash", "av_veh_s", "av_veh_m", "av_hard", "av_jerk2", "av_finished", "av_shield", "veh_s", "veh_m")},
            "crashes": sum(m[:, X.METRICS.index(k)] for k in X.CRASH_NAMES).tolist()}


def e2_run(name, scenarios=E2_SCEN):
    from avrl import bench as B
    from avrl import scorecard as SC
    pol = _policy(name)
    out = {}
    for n in scenarios:
        area, over = B.SCENARIOS[n]
        r = B.runner(area, B.behaviour(over), planner=pol)
        out[n] = SC.scorecard(r["raw"](None), r["shielded"](None))
        print(name, n, round(out[n]["av_score"], 1), flush=True)
    return out


def analyse(e1, e2=None):
    from scipy import stats
    from avlab.whocrashes import holm, rate_ratio
    t, i = ({k: np.asarray(v) for k, v in e1[p].items() if isinstance(v, list)} for p in ("teb", "idm"))
    sp = lambda d: 3.6 * d["av_veh_m"] / d["av_veh_s"]  # noqa: E731
    jr = lambda d: np.sqrt(d["av_jerk2"] / d["av_veh_s"])  # noqa: E731
    diff = sp(t) - sp(i)
    tt = stats.ttest_rel(sp(t), sp(i))
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=diff.std(ddof=1) / np.sqrt(len(diff)))
    h = lambda d: d["av_veh_s"].sum() / 3600  # noqa: E731
    out = {"E1": {p: {"av_speed_kmh": float(3.6 * d["av_veh_m"].sum() / d["av_veh_s"].sum()), "av_crash": float(d["av_crash"].sum()),
                      "av_hours": float(h(d)), "av_hard": float(d["av_hard"].sum()), "av_jerk_rms": float(np.sqrt(d["av_jerk2"].sum() / d["av_veh_s"].sum())),
                      "av_trips": float(d["av_finished"].sum()), "city_crashes": float(d["crashes"].sum()),
                      "human_speed_kmh": float(3.6 * (d["veh_m"] - d["av_veh_m"]).sum() / (d["veh_s"] - d["av_veh_s"]).sum()),
                      "shield_share": float(d["av_shield"].sum() * 0.2 / d["av_veh_s"].sum())}
                  for p, d in (("teb", t), ("idm", i))}}
    out["H1_speed_diff_kmh"] = {"mean": float(diff.mean()), "lo": float(ci[0]), "hi": float(ci[1]), "p": float(tt.pvalue),
                                "rel": float(diff.mean() / sp(i).mean())}
    out["H2_crash_rr"] = rate_ratio(t["av_crash"].sum(), h(t), i["av_crash"].sum(), h(i))
    out["S1_hard_rr"] = rate_ratio(t["av_hard"].sum(), h(t), i["av_hard"].sum(), h(i))
    jt = stats.ttest_rel(jr(t), jr(i))
    out["S2_jerk_diff"] = {"mean": float((jr(t) - jr(i)).mean()), "p": float(jt.pvalue)}
    ph = holm([out["S1_hard_rr"]["p"], out["S2_jerk_diff"]["p"]])
    out["S1_hard_rr"]["p_holm"], out["S2_jerk_diff"]["p_holm"] = ph
    out["H1_pass"] = bool(diff.mean() > 0 and tt.pvalue < 0.05)
    out["H2_pass"] = bool(out["H2_crash_rr"]["hi"] < 2)
    out["success"] = out["H1_pass"] and out["H2_pass"]
    out["city_crash_rr"] = rate_ratio(t["crashes"].sum(), t["veh_s"].sum() / 3600, i["crashes"].sum(), i["veh_s"].sum() / 3600)
    if e2:
        lb = json.load(open("results/bench/leaderboard.json"))["scores"]["rule-based AV"]
        keys = ("crash_per_h_unaided", "crash_per_h", "progress", "speed_kmh", "comfort_pass", "hard_brakes_per_h", "jerk_rms", "filter_share")
        out["E2"] = {n: {"teb": {"av_score": e2["teb"][n]["av_score"], **{k: e2["teb"][n]["metrics"][k] for k in keys}},
                         "idm": {"av_score": e2["idm"][n]["av_score"], **{k: e2["idm"][n]["metrics"][k] for k in keys}},
                         "rule": {"av_score": lb[n]["av_score"], **{k: lb[n]["metrics"].get(k) for k in keys}}} for n in e2["teb"]}
        out["E2_mean_score"] = {p: float(np.mean([out["E2"][n][p]["av_score"] for n in out["E2"]])) for p in ("teb", "idm", "rule")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--e1", action="store_true")
    ap.add_argument("--e2", action="store_true")
    ap.add_argument("--diag", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.quick:
        r = {p: e1_run(p, E=2, warm_s=40.0, seconds=40.0) for p in ("teb", "idm")}
        print(json.dumps({k: v for k, v in analyse(r).items()}, indent=1, default=float))
        return
    if a.e1:
        json.dump({p: e1_run(p) for p in ("teb", "idm")}, open(f"{OUT}/e1.json", "w"))
        return
    if a.diag:  # post hoc (not pre-registered): first interval fixed at the 0.2 s control step, obstacle / stop weights x10
        from avlab import teb as T
        T.CFG["dt1"] = 0.2
        T.W.update(o=600.0, s=600.0)
        json.dump({"teb_fixed": e2_run("teb", ("nominal",))}, open(f"{OUT}/diag.json", "w"), default=float)
        return
    if a.e2:
        json.dump({p: e2_run(p) for p in ("teb", "idm")}, open(f"{OUT}/e2.json", "w"), default=float)
        return
    e1 = json.load(open(f"{OUT}/e1.json"))
    e2 = json.load(open(f"{OUT}/e2.json")) if os.path.exists(f"{OUT}/e2.json") else None
    out = analyse(e1, e2)
    json.dump(out, open(f"{OUT}/teb.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
