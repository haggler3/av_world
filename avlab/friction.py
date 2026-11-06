"""Where is the cliff? Crash rate against road friction, in a human-only city and in one where half the cars are AVs.

    python -m avlab.friction            # 5 frictions x 2 AV shares x 16 copies x 10 simulated min -> results/friction/friction.json
    python -m avlab.friction --quick    # a smoke run
    python -m avlab.friction --plot     # figure -> docs/img/friction.png

    for mu in 1.0 0.8 0.6 0.45 0.3, for p_av in 0 0.5:  Behaviour(calibrated humans, friction=mu, p_av=p_av)  (the hardest braking is 9 mu m/s^2)
        16 copies of Oakland in ONE compiled run (examples._common.run_copies), warm 120 s, count 600 s
    fit  log(rate) = a + b log(mu) per AV share (quasi-Poisson, offset log vehicle-hours); the interaction is the difference of the two slopes
    cliff test: add a (log mu)^2 term; a negative curvature of the log-rate would mean a rate that rises faster as the road gets worse
"""
import argparse
import json
import os

import numpy as np

OUT = "results/friction"
MUS = (1.0, 0.8, 0.6, 0.45, 0.3)
SHARES = (0.0, 0.5)


def irls(y, t, X):
    """Quasi-Poisson regression log E[y] = log t + X beta by IRLS. Returns (beta, covariance scaled by the dispersion, dispersion)."""
    y, t, X = np.asarray(y, float), np.asarray(t, float), np.asarray(X, float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.sum(), 1) / t.sum())
    for _ in range(100):
        mu = t * np.exp(X @ beta)
        z = X @ beta + (y - mu) / np.maximum(mu, 1e-9)
        new = np.linalg.solve(X.T @ (mu[:, None] * X), X.T @ (mu * z))
        done = np.abs(new - beta).max() < 1e-10
        beta = new
        if done:
            break
    mu = t * np.exp(X @ beta)
    phi = max(((y - mu) ** 2 / np.maximum(mu, 1e-9)).sum() / (len(y) - X.shape[1]), 1.0)
    return beta, np.linalg.inv(X.T @ (mu[:, None] * X)) * phi, float(phi)


def design(logmu, share, curvature=False, interaction=True):
    """Columns: 1, share, log mu (centred), [log mu x share], [log mu^2 (centred)]."""
    lm = np.asarray(logmu, float) - np.mean(logmu)
    sh = np.asarray(share, float)
    cols = [np.ones_like(lm), sh, lm]
    if interaction:
        cols.append(lm * sh)
    if curvature:
        cols.append(lm ** 2 - np.mean(lm ** 2))
    return np.stack(cols, 1)


def run(E=16, warm_s=120.0, seconds=600.0, mus=MUS, shares=SHARES, area="oakland"):
    from examples import _common as C
    behs, tags = [], []
    for mu in mus:
        for p in shares:
            for _ in range(E):
                behs.append(C.calibrated(friction=mu, p_av=p))
                tags.append((mu, p))
    m = C.run_copies(area, behs, warm_s=warm_s, seconds=seconds)
    rows = C.per_copy(m)
    res = {"area": area, "copies": E, "seconds": seconds, "cells": {}}
    for (mu, p), r in zip(tags, rows):
        c = res["cells"].setdefault(f"{mu}|{p}", {"mu": mu, "p_av": p, "crashes": [], "av_crashes": [], "hours": [], "speed": []})
        c["crashes"].append(float(r["crashes"])); c["av_crashes"].append(float(r["av_crashes"]))
        c["hours"].append(float(r["vehicle_hours"])); c["speed"].append(float(r["mean_speed_kmh"]))
    return res


def summarize(res):
    cells = sorted(res["cells"].values(), key=lambda c: (c["p_av"], -c["mu"]))
    y, t, lm, sh = [], [], [], []
    for c in cells:
        y += c["crashes"]; t += c["hours"]; lm += [np.log(c["mu"])] * len(c["crashes"]); sh += [c["p_av"]] * len(c["crashes"])
    y, t, lm, sh = map(np.array, (y, t, lm, sh))
    out = {"per_cell": {f"{c['mu']}|{c['p_av']}": {"crashes": sum(c["crashes"]), "vehicle_hours": sum(c["hours"]),
                                                    "rate_per_1000_vh": 1000 * sum(c["crashes"]) / sum(c["hours"]),
                                                    "mean_speed_kmh": float(np.mean(c["speed"]))} for c in cells}}
    for name, mask in (("humans_only", sh == 0), ("half_AVs", sh > 0)):
        b, cov, phi = irls(y[mask], t[mask], design(lm[mask], sh[mask], interaction=False)[:, [0, 2]])
        if len(set(lm[mask])) < 3:
            out[name] = {"elasticity": float(b[1]), "se": float(np.sqrt(cov[1, 1])), "dispersion": phi}
            continue
        bq, covq, _ = irls(y[mask], t[mask], np.stack([np.ones(mask.sum()), lm[mask] - lm[mask].mean(),
                                                       (lm[mask] - lm[mask].mean()) ** 2], 1))
        out[name] = {"elasticity": float(b[1]), "se": float(np.sqrt(cov[1, 1])), "dispersion": phi,
                     "curvature": float(bq[2]), "curvature_se": float(np.sqrt(covq[2, 2]))}
    b, cov, phi = irls(y, t, design(lm, (sh > 0).astype(float)))  # coefficient 3: slope(half-AV city) - slope(humans only)
    out["interaction"] = {"coef": float(b[3]), "se": float(np.sqrt(cov[3, 3])), "dispersion": phi}
    out["rate_ratio_half_AV_vs_humans"] = rate_ratios(res)
    return out


def rate_ratios(res):
    """Per friction: crash rate of the half-AV city over the human-only city, exact conditional (Clopper-Pearson) interval, Holm-adjusted p."""
    from scipy import stats
    rows = []
    for mu in sorted({c["mu"] for c in res["cells"].values()}, reverse=True):
        a, b = res["cells"][f"{mu}|0.0"], res["cells"][f"{mu}|0.5"]
        k0, t0, k1, t1 = sum(a["crashes"]), sum(a["hours"]), sum(b["crashes"]), sum(b["hours"])
        n = int(k0 + k1)
        lo, hi = stats.beta.ppf(0.025, k1, n - k1 + 1), stats.beta.ppf(0.975, k1 + 1, n - k1)
        f = lambda q: q / (1 - q) * t0 / t1
        rows.append({"mu": mu, "rr": (k1 / t1) / (k0 / t0), "lo": f(lo), "hi": f(hi), "p": float(stats.binomtest(int(k1), n, t1 / (t0 + t1)).pvalue)})
    order = np.argsort([r["p"] for r in rows])
    run_max = 0.0
    for rank_, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(rows) - rank_) * rows[i]["p"]))
        rows[i]["p_holm"] = run_max
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--copies", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        return plot()
    res = run(E=2 if a.quick else a.copies, seconds=60.0 if a.quick else 600.0, mus=(1.0, 0.3) if a.quick else MUS)
    res["summary"] = summarize(res)
    print(json.dumps(res["summary"], indent=1))
    if not a.quick:
        json.dump(res, open(f"{OUT}/friction.json", "w"), indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open(f"{OUT}/friction.json")); S = res["summary"]
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    cols = {0.0: "#56b4e9", 0.5: "#e69f00"}
    lab = {0.0: "humans only", 0.5: "half AVs"}
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=BG)
    for a_ in ax:
        a_.set_facecolor(BG); a_.tick_params(colors=FG); [s.set_color(GR) for s in a_.spines.values()]
        a_.xaxis.label.set_color(FG); a_.yaxis.label.set_color(FG); a_.title.set_color(FG)
    rng = np.random.default_rng(0)
    for p in (0.0, 0.5):
        cs = sorted([c for c in res["cells"].values() if c["p_av"] == p], key=lambda c: c["mu"])
        x = [c["mu"] for c in cs]
        rate, lo, hi = [], [], []
        for c in cs:
            k, h = np.array(c["crashes"]), np.array(c["hours"])
            r = 1000 * k.sum() / h.sum()
            bs = [1000 * k[i].sum() / h[i].sum() for i in (rng.integers(0, len(k), len(k)) for _ in range(800))]
            rate.append(r); lo.append(r - np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5) - r)
        ax[0].errorbar(x, rate, yerr=[lo, hi], fmt="o-", color=cols[p], capsize=3)
        ax[0].text(0.62 if p == 0 else 0.62, 90 if p == 0 else 55, f"{lab[p]}: rate ~ friction^{S['humans_only' if p == 0 else 'half_AVs']['elasticity']:.1f}" if False else f"{lab[p]} (power {S['humans_only' if p == 0 else 'half_AVs']['elasticity']:.1f})", color=cols[p], fontsize=10, ha="center", va="center")
        # crash type split for AV share 0.5: crashes involving an AV
        if p > 0:
            av = [1000 * sum(c["av_crashes"]) / sum(c["hours"]) for c in cs]
            ax[1].plot(x, av, "o-", color=cols[p])
            hu = [1000 * (sum(c["crashes"]) - sum(c["av_crashes"])) / sum(c["hours"]) for c in cs]
            ax[1].plot(x, hu, "s--", color="#009e73")
            ax[1].text(0.62, 55, "crashes involving an AV", color=cols[p], fontsize=10, ha="center")
            ax[1].text(0.62, 40, "crashes between humans only", color="#009e73", fontsize=10, ha="center")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_xticks(MUS); a_.set_xticklabels([str(m) for m in MUS]); a_.invert_xaxis(); a_.minorticks_off(); a_.set_xlabel("road friction (1 = dry, 0.6 wet, 0.3 snow)")
    ax[0].set_yscale("log"); ax[0].set_ylabel("crashes per 1000 vehicle-hours"); ax[0].set_title("Crash rate against friction")
    ax[1].set_ylabel("crashes per 1000 vehicle-hours (half-AV city)"); ax[1].set_title("Who crashes when the road gets slick")
    fig.tight_layout(); fig.savefig("docs/img/friction.png", dpi=130, facecolor=BG)


if __name__ == "__main__":
    main()
