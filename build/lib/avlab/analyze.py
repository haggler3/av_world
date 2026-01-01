"""Statistics over the condition sweeps: where, why and how the AV fails.

  pooled        per configuration: AV-hours, crash rate with an exact Poisson 95 % interval, crash-type mix, speed,
                hard braking, time outside the barrier's safe set, junction / merge vetoes
  sensitivity   a Poisson regression per configuration, crashes ~ AV-hours x exp(b . z), z = each factor scaled to
                [0, 1] over its range (log factors in log space): exp(b) is the rate ratio from the low to the high end
                of a factor with the others held fixed, with a 95 % interval
  dependence    pooled crash rate in quartile bins of every factor (what a reader can check without the model)
  vs humans     the human drivers' crash rate per vehicle-hour in the same conditions
  worst         the conditions the model rates most dangerous for the AV
  maps          Oakland (the training map) vs two unseen neighbourhoods
  odd           crash rate over a friction x perception-noise grid: where each configuration's operating domain ends

    python -m avlab.analyze        # -> results/avlab/report.md, summary.json
"""
import json
import os

import numpy as np
from scipy import stats

from xnet import sim as X

from . import sweep as S

OUT = S.OUT
ORDER = ("rule", "rule+cbf", "ppo+idm", "ppo+cbf", "ppo raw")


def fname(tag, config):
    return f"{OUT}/{tag}_{config.replace(' ', '_').replace('+', '_')}.npy"


def load(tag):
    conds = {k: np.array(v) for k, v in json.load(open(f"{OUT}/{tag}_conditions.json")).items()}
    runs = {c: np.load(fname(tag, c)) for c in ORDER if os.path.exists(fname(tag, c))}
    return conds, runs


def per_cond(m):
    d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
    hum_s = np.maximum(d["veh_s"] - d["av_veh_s"], 1e-9)
    crashes = sum(d[k] for k in X.CRASH_NAMES)
    av_inv = sum(d[k] for k in X.AV_CRASH)
    return {"crash": d["av_crash"], "hours": d["av_veh_s"] / 3600, "km": d["av_veh_m"] / 1000, "speed_m": d["av_veh_m"], "speed_s": d["av_veh_s"],
            "hard": d["av_hard"], "h_neg": d["av_h_neg"], "veto_j": d["av_veto_junction"], "veto_m": d["av_veto_merge"],
            "near": d["av_near"], "red": d["av_red"], "human_crash": crashes - av_inv, "human_hours": hum_s / 3600,
            "types": {k[3:]: d[k] for k in X.AV_CRASH}}


def poisson_ci(k, t):
    lo = stats.chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = stats.chi2.ppf(0.975, 2 * k + 2) / 2
    return k / t, lo / t, hi / t


def zscale(conds):
    Z = []
    for k, (lo, hi, lg) in S.FACTORS.items():
        v = np.asarray(conds[k], float)
        Z.append((np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) if lg else (v - lo) / (hi - lo))
    return np.stack(Z, 1)


def glm(y, expo, Z, ridge=1e-2, iters=60):
    """Poisson regression with log link and offset log(expo), by iteratively reweighted least squares (small ridge on the
    slopes so a factor with no crashes at one end stays finite). Returns (beta, standard errors); beta[0] = intercept."""
    A = np.concatenate([np.ones((len(y), 1)), Z], 1)
    b = np.zeros(A.shape[1])
    b[0] = np.log(max(y.sum(), 0.5) / expo.sum())
    R = ridge * np.eye(A.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):
        mu = expo * np.exp(np.clip(A @ b, -30, 30))
        H = A.T @ (A * mu[:, None]) + R
        g = A.T @ (y - mu) - R @ b
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-8:
            break
    mu = expo * np.exp(A @ b)
    cov = np.linalg.inv(A.T @ (A * mu[:, None]) + R)
    return b, np.sqrt(np.diag(cov))


def analyze(tag="oakland"):
    conds, runs = load(tag)
    Z = zscale(conds)
    res = {"tag": tag, "n_conditions": int(Z.shape[0]), "factors": list(S.FACTORS), "configs": {}}
    for c, m in runs.items():
        p = per_cond(m)
        k, t = float(p["crash"].sum()), float(p["hours"].sum())
        rate, lo, hi = poisson_ci(k, t)
        hk, ht = float(p["human_crash"].sum()), float(p["human_hours"].sum())
        b, se = glm(p["crash"], p["hours"], Z)
        dep = {}
        for j, f in enumerate(S.FACTORS):
            edges = np.quantile(Z[:, j], [0, 0.25, 0.5, 0.75, 1.0])
            bins = np.clip(np.searchsorted(edges, Z[:, j], side="right") - 1, 0, 3)
            vals = np.asarray(conds[f], float)
            dep[f] = [{"x": float(np.median(vals[bins == i])), "rate": poisson_ci(float(p["crash"][bins == i].sum()), float(p["hours"][bins == i].sum()))}
                      for i in range(4)]
        pred = np.exp(np.concatenate([np.ones((len(Z), 1)), Z], 1) @ b)
        worst = np.argsort(-pred)[:8]
        res["configs"][c] = {
            "av_hours": t, "crashes": k, "rate": [rate, lo, hi], "per_100k_km": 1e5 * k / max(float(p["km"].sum()), 1e-9),
            "speed_kmh": 3.6 * float(p["speed_m"].sum() / max(p["speed_s"].sum(), 1e-9)), "hard_per_h": float(p["hard"].sum()) / t,
            "h_neg_share": float(p["h_neg"].sum()) / (t * 3600), "veto_junction_s_per_h": float(p["veto_j"].sum()) / t,
            "veto_merge_s_per_h": float(p["veto_m"].sum()) / t, "near_per_h": float(p["near"].sum()) / t, "red": float(p["red"].sum()),
            "types": {k_: float(v.sum()) for k_, v in p["types"].items()},
            "human_rate": list(poisson_ci(hk, ht)),
            "rate_ratio": {f: [float(np.exp(b[j + 1])), float(np.exp(b[j + 1] - 1.96 * se[j + 1])), float(np.exp(b[j + 1] + 1.96 * se[j + 1]))]
                           for j, f in enumerate(S.FACTORS)},
            "dependence": dep,
            "worst": [{**{f: float(conds[f][i]) for f in S.FACTORS}, "predicted_rate": float(pred[i]),
                       "observed": [float(p["crash"][i]), float(p["hours"][i])]} for i in worst],
            "safer_than_humans_share": float(np.mean(pred < np.maximum(p["human_crash"], 0.5) / p["human_hours"])),
        }
    return res


def grid():
    if not os.path.exists(f"{OUT}/grid_conditions.json"):
        return None
    conds, runs = load("grid")
    out = {"friction": S.GRID_F.tolist(), "noise": S.GRID_N.tolist(), "configs": {}}
    for c, m in runs.items():
        p = per_cond(m)
        k = p["crash"].reshape(len(S.GRID_F), len(S.GRID_N), S.GRID_SEEDS).sum(-1)
        t = p["hours"].reshape(len(S.GRID_F), len(S.GRID_N), S.GRID_SEEDS).sum(-1)
        hn = p["h_neg"].reshape(len(S.GRID_F), len(S.GRID_N), S.GRID_SEEDS).sum(-1) / (t * 3600)
        out["configs"][c] = {"rate": (k / t).tolist(), "crashes": k.tolist(), "hours": t.tolist(), "h_neg_share": hn.tolist()}
    return out


def main():
    summary = {"sweeps": {}, "grid": grid()}
    for tag in ("oakland", "squirrel_hill", "downtown"):
        if os.path.exists(f"{OUT}/{tag}_conditions.json"):
            summary["sweeps"][tag] = analyze(tag)
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    L = []
    for tag, r in summary["sweeps"].items():
        L += [f"## {tag}: {r['n_conditions']} conditions", "",
              "| configuration | AV-hours | AV crashes /AV-h (95 % CI) | per 100k km | AV km/h | hard brakes /h | outside barrier safe set | human crashes /veh-h |",
              "|---|---|---|---|---|---|---|---|"]
        for c in ORDER:
            if c not in r["configs"]:
                continue
            s = r["configs"][c]
            L.append(f"| {c} | {s['av_hours']:.0f} | {s['rate'][0]:.3f} ({s['rate'][1]:.3f}-{s['rate'][2]:.3f}) | {s['per_100k_km']:.1f} | "
                     f"{s['speed_kmh']:.1f} | {s['hard_per_h']:.1f} | {100 * s['h_neg_share']:.2f} % | {s['human_rate'][0]:.4f} |")
        L += ["", "Rate ratios (low -> high end of each factor, others fixed; 95 % CI):", "",
              "| factor | " + " | ".join(c for c in ORDER if c in r["configs"]) + " |", "|---|" + "---|" * len(r["configs"])]
        for f in S.FACTORS:
            L.append(f"| {f} | " + " | ".join(f"{r['configs'][c]['rate_ratio'][f][0]:.2f} ({r['configs'][c]['rate_ratio'][f][1]:.2f}-{r['configs'][c]['rate_ratio'][f][2]:.2f})"
                                               for c in ORDER if c in r["configs"]) + " |")
        L += ["", "AV crash types:", "", "| configuration | " + " | ".join(X.CRASH_NAMES) + " |", "|---|" + "---|" * len(X.CRASH_NAMES)]
        for c in ORDER:
            if c in r["configs"]:
                L.append(f"| {c} | " + " | ".join(f"{r['configs'][c]['types'][k]:.0f}" for k in X.CRASH_NAMES) + " |")
        L.append("")
    with open(f"{OUT}/report.md", "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
