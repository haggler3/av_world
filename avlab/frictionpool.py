"""Is there a friction cliff? Pool the two independent human-only friction runs and test shapes.

    python -m avlab.frictionpool          # reads results/friction/friction.json and results/cliffcause/cliff.json (no simulation, a few seconds)
                                          # -> results/frictionpool/pool.json, docs/img/frictionpool.png

Two earlier runs measured the humans-only crash rate against road friction: 16 copies at 1.0 / 0.8 / 0.6 / 0.45 / 0.3 (2026-11-06) and
12 copies at 0.7 / 0.6 / 0.55 / 0.5 / 0.45 / 0.4 (2026-11-08). The first reported a cliff (curvature of the log rate +1.25, interval 0.24-2.25),
the second did not. Pooled, 9 distinct frictions and 152 copies; per-copy counts are fitted by quasi-Poisson regression with offset
log vehicle-hours and a run indicator, and four shapes are compared:
    power    log rate = a + b log mu
    quad     power + c (log mu)^2
    hinge    power + extra slope below a hinge mu_h (grid search over mu_h in 0.45-0.8)
    expo     log rate = a + b mu             (rate rises exponentially as mu falls, no power law)
by deviance, AIC (quasi: deviance + 2p) and an F-type test of each against `power`. A leave-one-friction-out cross-validation (predict a held-out
friction's crash count from the others) gives an out-of-sample Poisson deviance per shape.
"""
import json
import os

import numpy as np

OUT = "results/frictionpool"


def load():
    """Per-copy (crashes, vehicle-hours, mu, run) for the humans-only cells of both runs."""
    a = json.load(open("results/friction/friction.json"))
    b = json.load(open("results/cliffcause/cliff.json"))
    y, t, mu, run = [], [], [], []
    for c in a["cells"].values():
        if c["p_av"] == 0.0:
            y += c["crashes"]; t += c["hours"]; mu += [c["mu"]] * len(c["crashes"]); run += [0] * len(c["crashes"])
    for c in b["cells"].values():
        y += c["per_copy_crashes"]; t += c["per_copy_hours"]; mu += [c["mu"]] * len(c["per_copy_crashes"]); run += [1] * len(c["per_copy_crashes"])
    return np.array(y, float), np.array(t, float), np.array(mu, float), np.array(run, float)


def design(kind, mu, run, hinge=None, force_run=False):
    lm = np.log(mu)
    cols = [np.ones_like(lm)] + ([run] if (force_run or np.ptp(run) > 0) else [])
    if kind == "power":
        cols += [lm]
    elif kind == "quad":
        cols += [lm, lm ** 2]
    elif kind == "hinge":
        cols += [lm, np.maximum(np.log(hinge) - lm, 0.0)]
    elif kind == "expo":
        cols += [mu]
    return np.stack(cols, 1)


def deviance(y, mu_hat):
    return float(2 * np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-12) / mu_hat), 0.0) - (y - mu_hat)))


def fit(kind, y, t, mu, run, hinge=None):
    from avlab.friction import irls
    X = design(kind, mu, run, hinge)
    b, cov, phi = irls(y, t, X)
    m = t * np.exp(X @ b)
    return {"kind": kind, "hinge": hinge, "beta": b.tolist(), "se": np.sqrt(np.diag(cov)).tolist(), "phi": phi, "deviance": deviance(y, m), "p": X.shape[1]}


def best_hinge(y, t, mu, run, grid=(0.45, 0.5, 0.55, 0.6, 0.7, 0.8)):
    fits = []
    for h in grid:
        if (mu < h).sum() >= 6:  # need data below the hinge, or the extra slope is not identified
            try:
                fits.append(fit("hinge", y, t, mu, run, h))
            except np.linalg.LinAlgError:
                pass
    return min(fits, key=lambda f: f["deviance"])


def loo_deviance(kind, y, t, mu, run):
    """Leave one friction out: fit on the others, predict the held-out cell's expected count; summed Poisson deviance of the cell totals."""
    tot = 0.0
    for m0 in np.unique(mu):
        tr, te = mu != m0, mu == m0
        if kind == "hinge":
            f = best_hinge(y[tr], t[tr], mu[tr], run[tr]); X = design("hinge", mu, run, f["hinge"])
        else:
            f = fit(kind, y[tr], t[tr], mu[tr], run[tr]); X = design(kind, mu, run)
        pred = t * np.exp(X @ np.array(f["beta"]))
        for r in np.unique(run[te]):  # one cell total per (friction, run)
            s = te & (run == r)
            tot += deviance(np.array([y[s].sum()]), np.array([max(pred[s].sum(), 1e-9)]))
    return float(tot)


def analyse():
    from scipy import stats
    y, t, mu, run = load()
    power = fit("power", y, t, mu, run)
    out = {"copies": int(len(y)), "crashes": float(y.sum()), "frictions": sorted(set(mu.tolist())), "power": power, "tests": {}}
    cand = {"quad": fit("quad", y, t, mu, run), "hinge": best_hinge(y, t, mu, run), "expo": fit("expo", y, t, mu, run)}
    phi = power["phi"]
    for k, f in cand.items():
        d = power["deviance"] - f["deviance"]
        dfp = f["p"] - power["p"]
        out["tests"][k] = {**f, "dev_drop_vs_power": d, "p_vs_power": float(stats.f.sf(d / max(dfp, 1) / phi, max(dfp, 1), len(y) - f["p"])) if dfp > 0 else None,
                           "qaic_gain_vs_power": power["deviance"] + 2 * power["p"] * phi - (f["deviance"] + 2 * f["p"] * phi)}
    out["loo_deviance"] = {k: loo_deviance(k, y, t, mu, run) for k in ("power", "quad", "hinge", "expo")}
    # per-run slopes: do the two runs agree on the elasticity?
    for r in (0, 1):
        s = run == r
        f = fit("power", y[s], t[s], mu[s], np.zeros(s.sum()))
        out[f"run{r}_elasticity"] = {"b": f["beta"][1], "se": f["se"][1], "n_copies": int(s.sum())}
    cells = {}
    for m0 in np.unique(mu):
        s = mu == m0
        cells[f"{m0:g}"] = {"crashes": float(y[s].sum()), "hours": float(t[s].sum()), "rate": float(1000 * y[s].sum() / t[s].sum())}
    out["cells"] = cells
    return out, (y, t, mu, run)


def plot(out, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    y, t, mu, run = data
    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor=BG)
    ax.set_facecolor(BG); ax.tick_params(colors=FG); [s.set_color(GR) for s in ax.spines.values()]
    for r, col, lab in ((0, "#56b4e9", "run of 2026-11-06 (16 copies)"), (1, "#e69f00", "run of 2026-11-08 (12 copies)")):
        for m0 in np.unique(mu[run == r]):
            s = (mu == m0) & (run == r)
            k, h = y[s].sum(), t[s].sum()
            lo, hi = (0.5 * 0 if k == 0 else 0), 0
            from scipy import stats
            lo = stats.gamma.ppf(0.025, k) / h * 1000 if k > 0 else 0
            hi = stats.gamma.ppf(0.975, k + 1) / h * 1000
            ax.plot([m0, m0], [max(lo, 0.3), hi], color=col, lw=2, alpha=0.8)
            ax.plot(m0, max(1000 * k / h, 0.3), "o", color=col, ms=7)
        ax.plot([], [], "o", color=col, label=lab)
    xs = np.linspace(0.28, 1.02, 100)
    P = out["power"]["beta"]; ref = 0.5 * (P[1]) if False else 0.0
    for kind, col, ls in (("power", "#e6edf3", "-"), ("expo", "#009e73", "--"), ("hinge", "#cc79a7", ":")):
        f = out["power"] if kind == "power" else out["tests"][kind]
        X = design(kind, xs, np.full_like(xs, 0.5), f.get("hinge"), force_run=True)  # the run effect at its mid-point
        ax.plot(xs, 1000 * np.exp(X @ np.array(f["beta"])), color=col, ls=ls, lw=2, label=f"{kind} fit")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.8, 1.0]); ax.set_xticklabels(["0.3", "0.4", "0.5", "0.6", "0.8", "1.0"])
    ax.invert_xaxis()
    ax.set_xlabel("road friction (log axis, slicker to the right)", color=FG); ax.set_ylabel("crashes per 1000 vehicle-hours (log)", color=FG)
    ax.set_title("Pooled human-only friction runs (152 copies): a bend near 0.7, not a cliff", color=FG, fontsize=11)
    ax.legend(facecolor=BG, edgecolor=GR, labelcolor=FG, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig("docs/img/frictionpool.png", dpi=150, facecolor=BG)


def main():
    os.makedirs(OUT, exist_ok=True)
    out, data = analyse()
    json.dump(out, open(f"{OUT}/pool.json", "w"), indent=1)
    plot(out, data)
    print(json.dumps({k: out[k] for k in ("copies", "crashes", "loo_deviance", "run0_elasticity", "run1_elasticity")}, indent=1))
    for k, v in out["tests"].items():
        print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items() if a not in ("beta", "se")})
    print("power", out["power"]["beta"], out["power"]["se"], out["power"]["phi"])


if __name__ == "__main__":
    main()
