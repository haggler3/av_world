"""How long must you watch a car before you can tell it is an AV?

    python -m avlab.watchtime --quick     # ~1 min
    python -m avlab.watchtime             # rule-based AV, bundled PPO, IDM: ~10 min -> results/watchtime/, docs/img/watchtime.png

avlab.turing judges whole trips (30 s or more) and finds the bundled PPO policy recognisable (AUC 0.88). A person in the next lane
sees a few seconds, not a trip. This cuts every trip into non-overlapping windows of w seconds (w = 3 ... 60), computes the same
eight kinematic features on each window, and measures the cross-validated AUC (folds by simulation copy) as a function of w. The
curve says how much driving a human (or a classifier) needs to watch before the AV gives itself away. Confidence intervals resample
whole trips (windows of one trip are not independent).
"""
import argparse
import json
import os
import time

import numpy as np

from avlab import turing as T
from xnet import sim as X

OUT = "results/watchtime"
WINDOWS = (3, 5, 10, 20, 40, 60)


def window_features(vs, acc, w_s):
    """Features of every non-overlapping window of w_s seconds of one trip (speed and acceleration arrays) -> [n, 8]."""
    n = int(round(w_s / X.DT))
    out = []
    for a in range(0, len(vs) - n + 1, n):
        v, ac = vs[a:a + n].astype(float), acc[a:a + n].astype(float)
        jerk = np.diff(ac) / X.DT
        out.append([v.mean(), v.std(), ac.std(), np.percentile(ac, 5), np.percentile(ac, 95), np.sqrt((jerk ** 2).mean()),
                    (v < 0.3).mean(), (ac < -2).mean()])
    return np.array(out).reshape(-1, 8)


def segments(rec, min_s=60.0):
    """Trips of at least min_s seconds -> list of (speed array, accel array, is_av, copy, trip id)."""
    v, a, alive, av, car, t0 = rec
    Tn, E, N = v.shape
    segs = []
    for e in range(E):
        for i in range(N):
            al = alive[:, e, i] & car[:, e, i]
            tt = t0[:, e, i]
            starts = np.nonzero(al & (np.r_[True, tt[1:] != tt[:-1]] | np.r_[True, ~al[:-1]]))[0]
            for s0 in starts:
                seg = np.arange(s0, Tn)
                seg = seg[np.cumprod(al[seg] & (tt[seg] == tt[s0])).astype(bool)]
                if len(seg) * X.DT >= min_s:
                    segs.append((v[seg, e, i], a[seg, e, i], bool(av[seg[0], e, i]), e, len(segs)))
    return segs


def curve(segs, windows=WINDOWS, n_boot=1000, seed=0):
    """AUC per window length with a trip-level (cluster) bootstrap. Only trips long enough for the longest window are used, so every
    window length is judged on the same trips."""
    longest = max(windows)
    segs = [s for s in segs if len(s[0]) * X.DT >= longest]
    res = {}
    for w in windows:
        F, y, g, tid = [], [], [], []
        for vs, ac, is_av, e, k in segs:
            f = window_features(vs, ac, w)
            F.append(f)
            y += [is_av] * len(f)
            g += [e] * len(f)
            tid += [k] * len(f)
        F, y, g, tid = np.vstack(F), np.array(y), np.array(g), np.array(tid)
        mu, sd = F.mean(0), F.std(0) + 1e-9
        Z = (F - mu) / sd
        score = np.zeros(len(y))
        for e in np.unique(g):
            tr, te = g != e, g == e
            if y[tr].all() or (~y[tr]).all():
                continue
            wts = T.logistic(Z[tr], y[tr].astype(float))
            score[te] = np.c_[np.ones(te.sum()), Z[te]] @ wts
        a = T.auc(score, y)
        rng = np.random.default_rng(seed)
        ids = np.unique(tid)
        by = {k: np.nonzero(tid == k)[0] for k in ids}
        boots = []
        for _ in range(n_boot):
            pick = np.concatenate([by[k] for k in rng.choice(ids, len(ids))])
            if y[pick].any() and (~y[pick]).any():
                boots.append(T.auc(score[pick], y[pick]))
        res[w] = {"auc": a, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "windows": int(len(y)),
                  "av_windows": int(y.sum()), "trips": len(ids)}
    return res


def plot(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
    pal = ["#56b4e9", "#e69f00", "#009e73", "#cc79a7"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), facecolor=BG)
    for c, (name, r) in zip(pal, res.items()):
        ws = sorted(int(w) for w in r)
        a = np.array([r[str(w) if str(w) in r else w]["auc"] for w in ws])
        lo = np.array([r[str(w) if str(w) in r else w]["ci"][0] for w in ws])
        hi = np.array([r[str(w) if str(w) in r else w]["ci"][1] for w in ws])
        ax.plot(ws, a, color=c, lw=2, marker="o", ms=4)
        ax.fill_between(ws, lo, hi, color=c, alpha=0.18, lw=0)
        ax.text(ws[-1] * 1.05, a[-1], name, color=c, fontsize=9, va="center")
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.text(3, 0.505, "coin flip", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(list(WINDOWS), [f"{w} s" for w in WINDOWS])
    ax.minorticks_off()
    ax.set_xlim(2.6, 140)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("how long you watch one car", color=INK2)
    ax.set_ylabel("AUC: AV or human?", color=INK2)
    ax.set_title("How long must you watch a car to tell it is an AV?", color=INK, fontsize=11, loc="left")
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK2, labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#30363d")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    kw = dict(copies=2, warm_s=30.0, seconds=150.0) if a.quick else dict(copies=4, warm_s=90.0, seconds=480.0)
    wins = (3, 10, 30) if a.quick else WINDOWS
    res = {}
    for name, pol in T.drivers().items():
        t0 = time.time()
        segs = segments(T.record(policy=pol, **kw), min_s=max(wins))
        if sum(s[2] for s in segs) < 3 or sum(not s[2] for s in segs) < 3:
            print(name, "too few trips", flush=True)
            continue
        res[name] = curve(segs, wins, n_boot=200 if a.quick else 1000)
        print(name, {w: round(r["auc"], 3) for w, r in res[name].items()}, f"{time.time() - t0:.0f} s", flush=True)
    json.dump(res, open(f"{OUT}/{'quick' if a.quick else 'watchtime'}.json", "w"), indent=1)
    if not a.quick:
        plot(res, "docs/img/watchtime.png")


if __name__ == "__main__":
    main()
