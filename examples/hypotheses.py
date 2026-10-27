"""Eleven plausible claims about driving, AVs and AV research, each stated so it can fail, and tested on av_world's data.

    python examples/hypotheses_sims.py     # the simulations (~25 min, once) -> results/examples/hypotheses_sims.json
    python examples/hypotheses.py          # the tests (~2 min) -> results/examples/hypotheses.json, docs/img/hypotheses.png, the table

Sources: the simulations above (calibrated humans in Oakland, 48 copies x 10 min per setting), the AV-share sweep
(results/examples/av_share.json), the AVBench leaderboard (17 drivers x 18 scenarios) and a fresh sensor-model run.

The statistics, all written out here so they can be checked:
  * crash counts per copy k_i with exposure t_i (vehicle-hours): Poisson regression log E[k_i] = log t_i + x_i' b, fitted by IRLS.
    Overdispersion phi = Pearson chi2 / (n - p); standard errors are scaled by sqrt(max(phi, 1)) (quasi-Poisson).
  * nested models are compared by the likelihood-ratio statistic D0 - D1 ~ chi2(df), divided by phi when phi > 1 (an F-type test).
  * a rate ratio between two settings: exact, conditional on k1 + k2 (k1 ~ Binomial(k1 + k2, t1 RR / (t1 RR + t2))), Clopper-Pearson.
  * the minimum detectable effect at 80 % power and two-sided 5 %: for a log rate ratio, (z_.975 + z_.8) sqrt(1/k1 + 1/k2).
  * rank correlations: Spearman rho, two-sided permutation p (20,000 permutations), percentile bootstrap 95 % interval.
  * paired differences across drivers: Wilcoxon signed-rank (exact for small n) and a bootstrap interval of the mean.
  * all p-values are corrected together with Holm's step-down method (family-wise error 5 %).
A claim is REFUTED when its prediction is contradicted at the corrected 5 % level (or its predicted effect lies outside the 95 %
interval), SURVIVES when the prediction holds and the test had the power to see it fail, and is UNDECIDED otherwise.
"""
import json
import math
import time

import numpy as np
from scipy import stats

try:
    from examples import _common as C
except ImportError:
    import _common as C
from xnet import sim as X  # noqa: E402

RNG = np.random.default_rng(0)
Z975, Z80 = 1.959964, 0.841621


# ------------------------------------------------------------------------------------------------- statistics
def poisson_glm(y, X_, offset):
    """IRLS for log E[y] = offset + X b. -> b, covariance (quasi-Poisson scaled), deviance, phi."""
    y, X_, offset = np.asarray(y, float), np.asarray(X_, float), np.asarray(offset, float)
    b = np.zeros(X_.shape[1])
    b[0] = math.log(max(y.sum(), 0.5) / np.exp(offset).sum())
    for _ in range(100):
        mu = np.exp(offset + X_ @ b)
        W = mu
        z = X_ @ b + (y - mu) / mu
        A = X_.T @ (W[:, None] * X_)
        b_new = np.linalg.solve(A, X_.T @ (W * z))
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new
            break
        b = b_new
    mu = np.exp(offset + X_ @ b)
    with np.errstate(divide="ignore", invalid="ignore"):
        dev = 2 * np.sum(np.where(y > 0, y * np.log(y / mu), 0.0) - (y - mu))
    n, p = X_.shape
    phi = float(np.sum((y - mu) ** 2 / mu) / max(n - p, 1))
    cov = np.linalg.inv(X_.T @ (mu[:, None] * X_)) * max(phi, 1.0)
    return b, cov, float(dev), phi


def lrt(dev0, dev1, df, phi):
    """Likelihood-ratio test of nested Poisson models; F-type when overdispersed."""
    stat = (dev0 - dev1) / max(phi, 1.0)
    return float(stat), float(stats.chi2.sf(stat, df))


def rate_ratio(k1, t1, k2, t2):
    """(k1/t1) / (k2/t2) with an exact conditional 95 % interval and p-value for RR = 1."""
    n = k1 + k2
    rr = (k1 / t1) / max(k2 / t2, 1e-12)
    lo_p = stats.beta.ppf(0.025, k1, n - k1 + 1) if k1 > 0 else 0.0
    hi_p = stats.beta.ppf(0.975, k1 + 1, n - k1) if k1 < n else 1.0
    to_rr = lambda q: (q / (1 - q)) * (t2 / t1) if q < 1 else math.inf  # noqa: E731
    p = stats.binomtest(int(k1), int(n), t1 / (t1 + t2)).pvalue if n else 1.0
    return rr, (to_rr(lo_p), to_rr(hi_p)), float(p)


def mde_ratio(k1, k2):
    """The smallest rate ratio (>1) detectable with 80 % power at two-sided 5 % given the expected counts."""
    return float(math.exp((Z975 + Z80) * math.sqrt(1 / max(k1, 1) + 1 / max(k2, 1))))


def spearman(x, y, n_perm=20000, n_boot=5000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = stats.spearmanr(x, y).statistic
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    perm = np.array([np.corrcoef(rx, RNG.permutation(ry))[0, 1] for _ in range(n_perm)])
    p = float((np.sum(np.abs(perm) >= abs(rho) - 1e-12) + 1) / (n_perm + 1))
    idx = RNG.integers(0, len(x), (n_boot, len(x)))
    boots = [stats.spearmanr(x[i], y[i]).statistic for i in idx if len(set(x[i])) > 2 and len(set(y[i])) > 2]
    return float(rho), (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))), p


def boot_mean(d, n_boot=10000):
    d = np.asarray(d, float)
    m = d[RNG.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


# ------------------------------------------------------------------------------------------------- data
def copies(sims, g):
    """Per-copy (crashes, vehicle-hours, vehicles mean speed...) of one simulated setting."""
    M = np.asarray(sims["groups"][g], float)
    names = sims["metrics"]
    col = {n: M[:, i] for i, n in enumerate(names)}
    k = sum(col[c] for c in X.CRASH_NAMES)
    t = col["veh_s"] / 3600
    return {"k": k, "t": t, "speed": 3.6 * col["veh_m"] / np.maximum(col["veh_s"], 1), "col": col}


def h_mistakes(sims):
    """H1. Crash rate is proportional to the human mistake rate: elasticity b = 1 in log rate = a + b log(r)."""
    gs = [("base", 1)] + [(f"mistakes_x{r}", r) for r in (3, 10, 30) if f"mistakes_x{r}" in sims["groups"]]
    y, t, x, per = [], [], [], {}
    for g, r in gs:
        c = copies(sims, g)
        y += c["k"].tolist(); t += c["t"].tolist(); x += [math.log(r)] * len(c["k"])
        per[r] = {"crashes": float(c["k"].sum()), "vehicle_hours": float(c["t"].sum()), "per_1000_vh": 1000 * c["k"].sum() / c["t"].sum(),
                  "red_light_crashes": float(c["col"]["rlr_crashes"].sum()),
                  "by_type": {n: float(c["col"][n].sum()) for n in X.CRASH_NAMES}}
    Xm = np.c_[np.ones(len(x)), x]
    b, cov, dev1, phi = poisson_glm(y, Xm, np.log(t))
    se = math.sqrt(cov[1, 1])
    z = (b[1] - 1) / se
    p = float(2 * stats.norm.sf(abs(z)))
    return dict(id="H1", claim="Crash rate is proportional to how often humans make mistakes (elasticity 1)",
                prediction="b = 1 in log(crash rate) = a + b log(mistake multiplier)", test="Poisson regression over 1x-30x, Wald test of b = 1",
                effect=f"b = {b[1]:.2f} [{b[1] - Z975 * se:.2f}, {b[1] + Z975 * se:.2f}]", estimate=float(b[1]), ci=(b[1] - Z975 * se, b[1] + Z975 * se),
                p=p, refutes_if="p small and b != 1", per=per, phi=phi,
                impact_pct=100 * abs(1 - float(b[1])))


def h_density(sims):
    """H2. Denser traffic is more dangerous per vehicle-hour (b > 0 in log rate = a + b log demand)."""
    gs = [("demand_x0.5", 0.5), ("base", 1.0), ("demand_x1.5", 1.5), ("demand_x2", 2.0)]
    y, t, x, per, km = [], [], [], {}, []
    for g, d in gs:
        if g not in sims["groups"]:
            continue
        c = copies(sims, g)
        y += c["k"].tolist(); t += c["t"].tolist(); x += [math.log(d)] * len(c["k"]); km += (c["col"]["veh_m"] / 1000).tolist()
        per[d] = {"crashes": float(c["k"].sum()), "vehicle_hours": float(c["t"].sum()), "per_1000_vh": 1000 * c["k"].sum() / c["t"].sum(),
                  "speed_kmh": float(np.mean(c["speed"])), "per_1000_vkm": 1e6 * c["k"].sum() / c["col"]["veh_m"].sum()}
    b, cov, dev, phi = poisson_glm(y, np.c_[np.ones(len(x)), x], np.log(t))
    se = math.sqrt(cov[1, 1])
    bk, covk, _, _ = poisson_glm(y, np.c_[np.ones(len(x)), x], np.log(km))  # the same with vehicle-km as the exposure
    sek = math.sqrt(covk[1, 1])
    p_one = float(stats.norm.sf(b[1] / se))  # one-sided: the claim predicts b > 0
    return dict(id="H2", claim="Denser traffic is more dangerous per vehicle-hour",
                prediction="b > 0 in log(crash rate) = a + b log(demand), demand 0.5x-2x", test="Poisson regression, one-sided Wald test",
                effect=f"b = {b[1]:.2f} [{b[1] - Z975 * se:.2f}, {b[1] + Z975 * se:.2f}] per vehicle-hour; per vehicle-km b = {bk[1]:.2f} "
                       f"[{bk[1] - Z975 * sek:.2f}, {bk[1] + Z975 * sek:.2f}]", estimate=float(b[1]),
                ci=(b[1] - Z975 * se, b[1] + Z975 * se), p=p_one, refutes_if="b <= 0 inside the interval", per=per, phi=phi,
                per_km=(float(bk[1]), float(bk[1] - Z975 * sek), float(bk[1] + Z975 * sek)),
                impact_pct=100 * abs(math.exp(b[1] * math.log(2)) - 1))


def h_friction(sims):
    """H3. Wet roads (friction 0.6) raise the crash rate by at least 50 %."""
    def pooled(g):  # the first batch plus the extra independent batches (_b, _c, _d) where they exist
        cs = [copies(sims, x) for x in [g] + [f"{g}_{y}" for y in "bcd"] if x in sims["groups"]]
        return {"k": np.concatenate([c["k"] for c in cs]), "t": np.concatenate([c["t"] for c in cs]), "n": len(cs)}
    a, b = pooled("friction_0.6"), pooled("base")
    rr, ci, p = rate_ratio(a["k"].sum(), a["t"].sum(), b["k"].sum(), b["t"].sum())
    ice = copies(sims, "friction_0.3")
    rr3, ci3, p3 = rate_ratio(ice["k"].sum(), ice["t"].sum(), b["k"].sum(), b["t"].sum())
    p_claim = float(stats.binomtest(int(a["k"].sum()), int(a["k"].sum() + b["k"].sum()),
                                    1.5 * a["t"].sum() / (1.5 * a["t"].sum() + b["t"].sum()), alternative="less").pvalue)
    return dict(id="H3", claim="Wet roads (friction 0.6) raise the crash rate by at least 50 %",
                prediction="rate ratio wet/dry >= 1.5", test=f"{len(a['k'])} + {len(b['k'])} copies; exact conditional rate ratio; one-sided test of RR >= 1.5",
                effect=f"RR = {rr:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]; icy (0.3): {rr3:.2f} [{ci3[0]:.2f}, {ci3[1]:.2f}]",
                estimate=rr, ci=ci, p=p_claim, refutes_if="RR < 1.5 at the corrected level", mde=mde_ratio(a["k"].sum(), b["k"].sum()),
                icy=(rr3, ci3, p3), impact_pct=100 * abs(rr - 1))


def h_av_linear():
    """H4. Crashes fall linearly with the AV share (no curvature)."""
    import os
    f = "results/examples/av_share_fine.json" if os.path.exists("results/examples/av_share_fine.json") else "results/examples/av_share.json"
    r = json.load(open(f))["rows"]
    s = np.array([x["av_share"] for x in r]); k = np.array([x["crashes"] for x in r]); t = np.array([x["vehicle_hours"] for x in r])
    n = len(s)
    b1, _, d1, _ = poisson_glm(k, np.c_[np.ones(n), s], np.log(t))
    b2, cov2, d2, phi2 = poisson_glm(k, np.c_[np.ones(n), s, s ** 2], np.log(t))
    stat, p = lrt(d1, d2, 1, phi2 if n > 5 else 1.0)  # with 9 shares the quadratic model's residuals estimate the dispersion
    rr, ci, _ = rate_ratio(k[-1], t[-1], k[0], t[0])
    return dict(id="H4", claim="Crashes fall in a straight line as the AV share grows",
                prediction="no curvature: the quadratic term adds nothing (log-linear model fits)", test=f"LRT of a quadratic term in a Poisson regression on {n} AV shares, 0-100 %",
                effect=f"LRT chi2 = {stat:.2f}; 100 % vs 0 % AVs RR = {rr:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", estimate=stat, ci=ci, p=p,
                refutes_if="the quadratic term is significant", rr_full=rr, impact_pct=100 * (1 - rr), curvature=float(b2[2]),
                curvature_ci=(float(b2[2] - Z975 * math.sqrt(cov2[2, 2])), float(b2[2] + Z975 * math.sqrt(cov2[2, 2]))), shares=n)


def leaderboard():
    lb = json.load(open("results/bench/leaderboard.json"))
    full = [p for p in lb["scores"] if len(lb["scores"][p]) == 18]
    return lb, full


def h_core_tail(lb, full):
    core = [np.mean([lb["scores"][p][n]["av_score"] for n in lb["scenarios"]]) for p in full]
    tail = [np.mean([lb["scores"][p][n]["av_score"] for n in lb["tail_scenarios"]]) for p in full]
    rho, ci, p = spearman(core, tail)
    return dict(id="H5", claim="Doing well on everyday traffic and on long-tail incidents is the same skill",
                prediction="Spearman rho >= 0.8 between core and long-tail scores (15 drivers)", test="Spearman, permutation p, bootstrap CI; refuted if CI excludes 0.8",
                effect=f"rho = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", estimate=rho, ci=ci, p=p, threshold=0.8, impact_pct=100 * rho ** 2)


def h_checks_score(lb, full):
    checks = [sum(sum(v.values()) for v in lb["absolute"][p].values()) for p in full]
    score = [np.mean([lb["scores"][p][n]["av_score"] for n in lb["scenarios"]]) for p in full]
    rho, ci, p = spearman(checks, score)
    rb = int(np.argmax(score))
    return dict(id="H6", claim="A driver that passes more absolute safety checks gets a higher overall score",
                prediction="rho > 0 and the top scorer passes the most checks", test="Spearman over 15 drivers",
                effect=f"rho = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]; the top scorer ({full[rb]}, {score[rb]:.1f}) passes {checks[rb]} checks, the most passed is {max(checks)}",
                estimate=rho, ci=ci, p=p, impact_pct=100 * (1 - checks[rb] / max(checks)))


def h_replicate(lb, full):
    pairs = [(p, p[:-1] + ", GPU retrain)") for p in full if p.endswith(")") and "GPU" not in p and p[:-1] + ", GPU retrain)" in full]
    d = []
    for a, b in pairs:
        for n in lb["scenarios"] + lb["tail_scenarios"]:
            d.append(lb["scores"][b][n]["av_score"] - lb["scores"][a][n]["av_score"])
    per_policy = [np.mean([lb["scores"][b][n]["av_score"] - lb["scores"][a][n]["av_score"] for n in lb["scenarios"]]) for a, b in pairs]
    m, ci = boot_mean(d)
    sd = float(np.std(d, ddof=1))
    return dict(id="H7", claim="Retraining the same RL recipe reproduces its score within 5 points",
                prediction="|score(retrain) - score(original)| <= 5 per scenario, on average and typically", test=f"{len(pairs)} recipes x 18 scenarios, paired differences",
                effect=f"mean diff {m:+.1f} [{ci[0]:+.1f}, {ci[1]:+.1f}]; per-scenario sd {sd:.1f}; core means {', '.join(f'{x:+.1f}' for x in per_policy)}",
                estimate=m, ci=ci, p=float(stats.wilcoxon(d).pvalue), sd=sd, share_within5=float(np.mean(np.abs(d) <= 5)),
                impact_pct=100 * float(np.mean(np.abs(d) > 5)))


def h_transfer(lb, full):
    d = []
    for p in full:
        o = lb["scores"][p]["nominal"]["av_score"]
        d.append(np.mean([lb["scores"][p][n]["av_score"] for n in ("squirrel_hill", "downtown")]) - o)
    m, ci = boot_mean(d)
    w = stats.wilcoxon(d)
    return dict(id="H8", claim="Policies trained in Oakland transfer to other neighbourhoods unchanged",
                prediction="score(Squirrel Hill, Downtown) - score(Oakland) = 0", test="Wilcoxon signed-rank over 15 drivers, bootstrap CI of the mean",
                effect=f"mean change {m:+.1f} points [{ci[0]:+.1f}, {ci[1]:+.1f}]", estimate=m, ci=ci, p=float(w.pvalue), impact_pct=abs(m))


def h_speed_safety(lb, full):
    prog, crash = [], []
    for p in full:
        for n in lb["scenarios"]:
            m = lb["scores"][p][n]["metrics"]
            prog.append(m["progress"]); crash.append(m["crash_per_h_unaided"])
    rho, ci, p = spearman(prog, crash, n_perm=5000, n_boot=2000)
    return dict(id="H9", claim="Faster AVs crash more: there is a speed-safety trade-off",
                prediction="rho > 0 between progress and unaided crash rate (driver x scenario)", test=f"Spearman over {len(prog)} driver-scenario pairs",
                effect=f"rho = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", estimate=rho, ci=ci, p=p, impact_pct=100 * abs(rho))


def h_filter_hides(lb, full):
    fs, crash = [], []
    for p in full:
        for n in lb["scenarios"]:
            m = lb["scores"][p][n]["metrics"]
            fs.append(m["filter_share"]); crash.append(m["crash_per_h_unaided"])
    rho, ci, p = spearman(fs, crash, n_perm=5000, n_boot=2000)
    return dict(id="H10", claim="How often the safety filter steps in predicts how unsafe the policy is without it",
                prediction="rho > 0.5 between filter share and unaided crash rate", test=f"Spearman over {len(fs)} driver-scenario pairs",
                effect=f"rho = {rho:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]", estimate=rho, ci=ci, p=p, threshold=0.5, impact_pct=100 * rho ** 2)


def fog_rows(seeds=(11, 12, 13, 14, 15), per_run=12, cache="results/examples/hypotheses_fog.json"):
    """(ego, fog, detected VRU, VRU object-frames, detected vehicles, vehicle object-frames) per ego and fog level, over several
    independent runs (5 x 12 = 60 egos, 5x the first version's 12, which left H11 undecided). Cached: the sensor model is slow."""
    import os
    key = {"seeds": list(seeds), "per_run": per_run}
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get("key") == key:
            return np.array(c["rows"], float)
    from xnet import carla_export as CE
    from xnet import network as NW
    from xnet import perception as PC
    net, geo, info = NW.build("oakland")
    rows = []
    for sd in seeds:
        sim = X.Sim(net, n_envs=1, beh=C.calibrated(p_av=0.3, ped_scale=2.0, p_bike=0.08), seed=sd)
        sim.run(90.0)
        scene = CE.trajectories(net, geo, sim.run(90.0, record=(0, 1)))
        egos = [v["id"] for v in scene["vehicles"] if v["av"] and len(v["x"]) >= 200][:per_run]
        for e in egos:
            for fog in (0.0, 0.7):
                d = PC.detect(scene, e, PC.Sensor(fog=fog), seed=e)
                vru = d["vru"].astype(bool)
                rows.append((1000 * sd + e, fog, d["detected"][vru].sum(), vru.sum(), d["detected"][~vru].sum(), (~vru).sum()))
    json.dump({"key": key, "rows": np.asarray(rows, float).tolist()}, open(cache, "w"))
    return np.array(rows, float)


def h_fog():
    """H11. Fog hurts the detection of pedestrians and cyclists more (relatively) than of vehicles."""
    R = fog_rows()
    egos = sorted(set(R[:, 0].tolist()))

    def drop(sel):
        r = R[np.isin(R[:, 0], sel)]
        c, f = r[r[:, 1] == 0], r[r[:, 1] > 0]
        rel = lambda det, n: 1 - (f[:, det].sum() / f[:, n].sum()) / (c[:, det].sum() / c[:, n].sum())  # noqa: E731
        return rel(2, 3), rel(4, 5)
    v, nv = drop(egos)
    boots = []
    for _ in range(2000):  # cluster bootstrap over egos
        s = RNG.choice(egos, len(egos))
        try:
            a, b = drop(s)
            boots.append(a - b)
        except ZeroDivisionError:
            pass
    boots = np.asarray([b for b in boots if np.isfinite(b)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    p = float(2 * min(np.mean(boots <= 0), np.mean(boots >= 0)))
    n_vru = int(R[R[:, 1] == 0, 3].sum())
    return dict(id="H11", claim="Fog hurts the detection of pedestrians and cyclists more than of vehicles",
                prediction="relative drop in detection rate (fog 0.7 vs clear) larger for VRUs", test=f"sensor model on {len(egos)} AVs' trips, cluster bootstrap over AVs",
                effect=f"relative drop VRU {v:.0%} vs vehicles {nv:.0%}; difference {v - nv:+.0%} [{ci[0]:+.0%}, {ci[1]:+.0%}] ({n_vru} VRU object-frames)",
                estimate=v - nv, ci=ci, p=max(p, 1 / 2000), impact_pct=100 * abs(v - nv))


def verdict(h):
    """REFUTED / SURVIVES / UNDECIDED from the corrected p-value and the interval (rules per claim, see the module docstring)."""
    lo, hi = h["ci"]
    sig = h["p_holm"] < 0.05
    i = h["id"]
    if i == "H1":
        return "REFUTED" if sig else ("SURVIVES" if lo <= 1 <= hi and hi - lo < 1 else "UNDECIDED")
    if i == "H2":
        return "SURVIVES" if sig and lo > 0 else ("REFUTED" if hi < 0.1 else "UNDECIDED")
    if i == "H3":
        return "REFUTED" if sig else ("SURVIVES" if lo >= 1.5 else "UNDECIDED")
    if i == "H4":  # with 9 shares (rule fixed before the 9-share data came in): survives if no bend is detected and any bend is
        # small: |quadratic coefficient| < 1, i.e. it moves the predicted rate at 50 % AVs by at most e^0.25 - 1 = 28 %
        if sig:
            return "REFUTED"
        lo_c, hi_c = h.get("curvature_ci", (-9, 9))
        return "SURVIVES" if h.get("shares", 5) > 5 and -1 < lo_c and hi_c < 1 else "UNDECIDED"
    if i in ("H5", "H10"):
        return "REFUTED" if hi < h["threshold"] else ("SURVIVES" if lo >= h["threshold"] else "UNDECIDED")
    if i == "H6":
        return "REFUTED" if hi < 0 or h["impact_pct"] > 0 else ("SURVIVES" if sig and lo > 0 else "UNDECIDED")  # a counterexample refutes
    if i == "H7":
        return "REFUTED" if h["share_within5"] < 0.5 or abs(lo) > 5 or abs(hi) > 5 else "SURVIVES"
    if i == "H8":
        return "REFUTED" if sig else ("SURVIVES" if hi - lo < 10 else "UNDECIDED")
    if i == "H9":
        return "SURVIVES" if sig and lo > 0 else ("REFUTED" if hi < 0.1 else "UNDECIDED")
    if i == "H11":
        return "SURVIVES" if sig and lo > 0 else ("REFUTED" if hi < 0 else "UNDECIDED")
    return "UNDECIDED"


# reach R (share of the repository's research questions the answer bears on, a judgement) and the effort in days to turn the result
# into a paper-grade follow-up, with that follow-up; both stated here so the ranking can be argued with
FOLLOW_UP = {
    "H1": (1.0, 3, "attribute crashes to mechanisms (Shapley over the Behaviour knobs): which mistakes actually make crashes?"),
    "H2": (0.8, 2, "exposure matters: report every crash rate per hour, per km and per junction entry, and show where they disagree"),
    "H3": (0.6, 2, "a friction dose-response (1.0 .. 0.3) with 4x the copies: where does the cliff between wet and icy start?"),
    "H4": (0.9, 2, "the AV-share curve with 9 shares and 3x the copies: is there a threshold (convexity) or diminishing returns?"),
    "H5": (0.8, 3, "a 3-scenario 'core' subset that predicts the full 18-scenario score: a cheaper benchmark"),
    "H6": (0.7, 2, "a Pareto view of score vs checks: what the rule-based AV does that learned policies do not (comfort, speed)"),
    "H7": (1.0, 5, "seed variance of RL on AVBench: 10 seeds per recipe on a GPU, with confidence intervals on every leaderboard row"),
    "H8": (0.9, 4, "train on a mix of Oakland, Squirrel Hill and Downtown: does the transfer gap close, and at what cost at home?"),
    "H9": (0.7, 3, "the speed-safety frontier: a policy family indexed by one speed knob, with the frontier's slope and its kinks"),
    "H10": (0.9, 3, "filter dependence as a deployment metric: when does a filtered policy's record overstate its own safety?"),
    "H11": (0.5, 2, "per-class sensor curves under fog, rain and night, with 10x the egos to resolve a few-point difference"),
}


SHORT = {"H1": "crashes scale with human mistakes", "H2": "denser traffic is more dangerous", "H3": "wet roads: +50 % crashes",
         "H4": "crashes fall linearly with AV share", "H5": "everyday skill = long-tail skill", "H6": "more checks passed = higher score",
         "H7": "RL retrains reproduce within 5 points", "H8": "policies transfer across neighbourhoods", "H9": "faster AVs crash more",
         "H10": "filter use predicts unaided risk", "H11": "fog hurts pedestrians most"}


def impact(h):
    """Impact % = 100 x M x E x R. M (magnitude) = min(1, effect% / 50): a 50 % change (or rho^2 = 0.5, ...) counts fully.
    E (evidence) = 1 if the verdict is decided (refuted or survived at the corrected level, or by a counterexample), 0.5 if not.
    R (reach) from FOLLOW_UP. Priority = impact % / effort days."""
    reach, days, idea = FOLLOW_UP[h["id"]]
    M = min(1.0, h["impact_pct"] / 50)
    E = 1.0 if h["verdict"] in ("REFUTED", "SURVIVES") else 0.5
    score = 100 * M * E * reach
    return {"magnitude": M, "evidence": E, "reach": reach, "impact_score_pct": score, "effort_days": days,
            "impact_per_day": score / days, "follow_up": idea}


def plot(hs, out="docs/img/hypotheses.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, INK, INK2, GRID = "#0d1117", "#e6edf3", "#8b949e", "#21262d"
    col = {"REFUTED": "#d03b3b", "SURVIVES": "#0ca30c", "UNDECIDED": "#fab219"}  # status colours, always with the verdict in words
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(15, 6.2), facecolor=BG, gridspec_kw={"width_ratios": [1.35, 1]})
    for ax in (a0, a1):
        ax.set_facecolor(BG)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
    order = sorted(hs, key=lambda h: h["impact_score_pct"])
    for j, h in enumerate(order):
        a0.barh(j, h["impact_score_pct"], color=col[h["verdict"]], height=0.62)
        a0.text(h["impact_score_pct"] + 1.2, j, f"{h['verdict'].lower()}  ·  {h['effort_days']} d  ·  {h['impact_per_day']:.0f} %/d",
                va="center", color=INK2, fontsize=8)
    a0.set_yticks(range(len(order)), [f"{h['id']}  {SHORT[h['id']]}" for h in order], color=INK, fontsize=9)
    a0.set_xlim(0, 135)
    a0.set_axisbelow(True)
    a0.grid(axis="x", color=GRID)
    a0.set_xlabel("impact score, % = 100 x magnitude x evidence x reach", color=INK2, fontsize=9)
    a0.set_title("Eleven claims, tested to fail", color=INK, fontsize=11, loc="left")
    days = np.linspace(1.5, 5.5, 50)
    for r in (5, 10, 20, 30):
        a1.plot(days, r * days, color="#30363d", lw=1, zorder=1)
        x_end = min(5.5, 104 / r)
        a1.text(x_end - 0.05, r * x_end, f"{r} %/day", color=INK2, fontsize=7.5, ha="right", va="bottom")
    jit = {}
    for h in hs:
        k = (h["effort_days"], round(h["impact_score_pct"] / 6))
        dy = 4.5 * jit.get(k, 0)
        jit[k] = jit.get(k, 0) + 1
        a1.scatter(h["effort_days"], h["impact_score_pct"], s=70, color=col[h["verdict"]], edgecolor=BG, linewidth=2, zorder=3)
        a1.annotate(h["id"], (h["effective_x"] if "effective_x" in h else h["effort_days"], h["impact_score_pct"]), textcoords="offset points",
                    xytext=(8, -3 + dy), color=INK, fontsize=8.5)
    a1.set_xlim(1.4, 5.6)
    a1.set_ylim(0, 108)
    a1.set_axisbelow(True)
    a1.grid(color=GRID)
    a1.set_xlabel("effort for a paper-grade follow-up, days", color=INK2, fontsize=9)
    a1.set_ylabel("impact score, %", color=INK2, fontsize=9)
    a1.set_title("Impact over ease: up and to the left first", color=INK, fontsize=11, loc="left")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=c, label=v.lower()) for v, c in col.items()], facecolor=BG, edgecolor="#30363d", labelcolor=INK,
              fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def main():
    t0 = time.time()
    sims = json.load(open("results/examples/hypotheses_sims.json"))
    lb, full = leaderboard()
    hs = [h_mistakes(sims), h_density(sims), h_friction(sims), h_av_linear(), h_core_tail(lb, full), h_checks_score(lb, full),
          h_replicate(lb, full), h_transfer(lb, full), h_speed_safety(lb, full), h_filter_hides(lb, full), h_fog()]
    adj = holm([h["p"] for h in hs])
    for h, a in zip(hs, adj):
        h["p_holm"] = float(a)
        h["verdict"] = verdict(h)
        h.update(impact(h))
        print(f"{h['id']:4s} {h['verdict']:9s} p={h['p']:.3g} p_holm={a:.3g} impact {h['impact_score_pct']:.0f} % / {h['effort_days']} d  {h['claim']}\n      {h['effect']}")
    plot(hs)
    C.save("hypotheses", {"hypotheses": hs, "wall_s": time.time() - t0})
    print(f"{time.time() - t0:.0f} s")
    return hs


if __name__ == "__main__":
    main()
