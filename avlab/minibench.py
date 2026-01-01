"""A cheaper AVBench: which few scenarios predict the whole board? (follow-up of hypothesis H5; uses saved results only, no simulation)

    python -m avlab.minibench            # -> results/minibench/minibench.json
    python -m avlab.minibench --plot     # -> docs/img/minibench.png

    M[d, s] = av_score of driver d in scenario s   (15 drivers with all 18 scenarios, results/bench/leaderboard.json)
    target[d] = mean over the 18 scenarios
    for each subset S of k scenarios (k = 1..4, exhaustive):
        nested leave-one-driver-out: choose S on the other 14 drivers (smallest cross-validated RMSE of  target ~ a + b mean(M[:, S])),
        then predict the held-out driver from its S-scores with the line fitted on the other 14
    report: Spearman and RMSE between nested predictions and the true target; compare with a random subset of the same size
    (the same nested protocol cannot be applied to random subsets; a random subset has no selection, so its plain leave-one-out error is used)
"""
import argparse
import itertools
import json
import os

import numpy as np

OUT = "results/minibench"


def load(path="results/bench/leaderboard.json"):
    d = json.load(open(path))
    sc = d["scenarios"] + d["tail_scenarios"]
    names, rows = [], []
    for k, v in d["scores"].items():
        r = [v.get(s, {}).get("av_score", np.nan) for s in sc]
        if not np.isnan(r).any():
            names.append(k); rows.append(r)
    return names, sc, np.array(rows, float)


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def loo_predict(x, y):
    """Leave-one-out predictions of y from x by least squares (x, y: [n])."""
    n = len(y)
    out = np.zeros(n)
    for i in range(n):
        m = np.arange(n) != i
        A = np.stack([np.ones(n - 1), x[m]], 1)
        beta = np.linalg.lstsq(A, y[m], rcond=None)[0]
        out[i] = beta[0] + beta[1] * x[i]
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def best_subset(M, y, k, idx=None):
    """The k-subset (columns of M) whose mean has the smallest leave-one-out RMSE in predicting y, over the rows in idx."""
    idx = np.arange(len(y)) if idx is None else idx
    best, arg = np.inf, None
    for S in itertools.combinations(range(M.shape[1]), k):
        e = rmse(loo_predict(M[idx][:, S].mean(1), y[idx]), y[idx])
        if e < best:
            best, arg = e, S
    return arg, best


def nested(M, y, k):
    """Choose the subset without the held-out driver, predict that driver. Returns predictions [n] and the chosen subsets."""
    n = len(y)
    pred, chosen = np.zeros(n), []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        S, _ = best_subset(M, y, k, tr)
        chosen.append(S)
        A = np.stack([np.ones(n - 1), M[tr][:, S].mean(1)], 1)
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[i] = beta[0] + beta[1] * M[i, list(S)].mean()
    return pred, chosen


def random_baseline(M, y, k, reps=300, seed=0):
    rng = np.random.default_rng(seed)
    rs, es = [], []
    for _ in range(reps):
        S = rng.choice(M.shape[1], k, replace=False)
        p = loo_predict(M[:, S].mean(1), y)
        rs.append(spearman(p, y)); es.append(rmse(p, y))
    return {"spearman_median": float(np.median(rs)), "spearman_q10": float(np.percentile(rs, 10)), "rmse_median": float(np.median(es))}


def run(ks=(1, 2, 3, 4)):
    names, sc, M = load()
    y = M.mean(1)
    out = {"drivers": names, "scenarios": sc, "target_mean": y.tolist(), "sd_of_target": float(y.std()), "k": {}}
    for k in ks:
        pred, chosen = nested(M, y, k)
        S_all, e_all = best_subset(M, y, k)
        out["k"][str(k)] = {"nested_spearman": spearman(pred, y), "nested_rmse": rmse(pred, y),
                            "subset_all_drivers": [sc[i] for i in S_all], "in_sample_rmse": e_all,
                            "chosen_stability": float(np.mean([c == tuple(S_all) for c in chosen])),
                            "random": random_baseline(M, y, k), "pred": pred.tolist()}
        print(f"k={k}: nested rho {out['k'][str(k)]['nested_spearman']:.2f}, rmse {out['k'][str(k)]['nested_rmse']:.1f} "
              f"(random {out['k'][str(k)]['random']['rmse_median']:.1f}); subset {out['k'][str(k)]['subset_all_drivers']}", flush=True)
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.load(open(f"{OUT}/minibench.json"))
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.4), facecolor=BG)
    for a in ax:
        a.set_facecolor(BG); a.tick_params(colors=FG); [s.set_color(GR) for s in a.spines.values()]
        a.xaxis.label.set_color(FG); a.yaxis.label.set_color(FG); a.title.set_color(FG)
    ks = sorted(r["k"], key=int)
    x = [int(k) for k in ks]
    ax[0].plot(x, [r["k"][k]["nested_rmse"] for k in ks], "o-", color="#56b4e9")
    ax[0].plot(x, [r["k"][k]["random"]["rmse_median"] for k in ks], "s--", color="#e69f00")
    ax[0].text(x[-1], r["k"][ks[-1]]["nested_rmse"], " chosen subset\n (nested CV)", color="#56b4e9", fontsize=9, va="top")
    ax[0].text(x[-1], r["k"][ks[-1]]["random"]["rmse_median"], " random subset", color="#e69f00", fontsize=9, va="bottom")
    ax[0].set_xticks(x); ax[0].set_xlim(0.8, x[-1] + 1.6); ax[0].set_xlabel("scenarios in the mini-benchmark")
    ax[0].set_ylabel("error predicting the 18-scenario mean (points)"); ax[0].set_title("Fewer scenarios, how much error?")
    k = "3" if "3" in r["k"] else ks[-1]
    y = np.array(r["target_mean"]); p = np.array(r["k"][k]["pred"])
    ax[1].plot([y.min(), y.max()], [y.min(), y.max()], color=GR)
    ax[1].scatter(y, p, color="#56b4e9", s=28)
    ax[1].set_xlabel("full-board mean score"); ax[1].set_ylabel(f"predicted from {k} scenarios (held-out driver)")
    ax[1].set_title(f"{k} scenarios: rank correlation {r['k'][k]['nested_spearman']:.2f}")
    fig.tight_layout(); fig.savefig("docs/img/minibench.png", dpi=130, facecolor=BG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        return plot()
    json.dump(run(), open(f"{OUT}/minibench.json", "w"), indent=1)


if __name__ == "__main__":
    main()
