"""IDM fitted directly to real car following: the trait distribution of real drivers, not a guess through tail rates.

An episode is a follower that keeps the same leader (same lane, heading within 35 degrees, within 60 m) for at least 4 s while moving.
With the follower's cleaned speed v, its acceleration a = dv/dt (central differences), the gap s and the leader's speed v_l, IDM says
  a = amax [1 - (v / v0)^4 - (s*(v, dv) / s)^2],   s* = s0 + max(0, v T + v (v - v_l) / (2 sqrt(amax b))).
Least squares on a over the episode gives (T, s0, amax, b, v0) per follower (v0 is bounded below by the fastest speed seen, since a driver
who never reaches their desired speed reveals only a lower bound). The well-identified fits (R^2 above a floor and a gap that varied)
form the trait bank: each row is one real driver's (T, s0, amax, b), so their correlations survive. The same fit on simulated scenes,
where each driver's T is known, checks the fitter.
"""
import numpy as np
from scipy.optimize import least_squares

from autotag.rules import leaders, relative

BOUNDS = ([0.1, 0.2, 0.3, 0.5, 3.0], [4.0, 8.0, 4.0, 6.0, 45.0])


def idm(p, v, s, vl):
    T, s0, amax, b, v0 = p
    ss = s0 + np.maximum(0.0, v * T + v * (v - vl) / (2 * np.sqrt(amax * b)))
    return amax * (1 - (v / v0) ** 4 - (ss / np.maximum(s, 0.1)) ** 2)


def episodes(sc, min_len=20, skip_ego=True):
    """-> list of dicts(v, a, s, vl, follower, leader) for every 4 s+ stretch of following one leader."""
    dt = sc["dt"]
    lead, gap = leaders(sc, relative(sc))
    v = sc["v"]
    V, T = v.shape
    out = []
    for i in range(V):
        if (skip_ego and i == sc["ego"]) or ("kind" in sc and sc["kind"][i] in ("bike", "bus")):
            continue
        ok = (lead[i] >= 0) & (gap[i] > 0) & (gap[i] < 60) & np.isfinite(v[i])
        t = 0
        while t < T:
            if not ok[t]:
                t += 1
                continue
            j, t1 = lead[i, t], t
            while t1 < T and ok[t1] and lead[i, t1] == j:
                t1 += 1
            if t1 - t >= min_len:
                vv = v[i, t:t1]
                vl = v[j, t:t1]
                if np.isfinite(vl).all() and vv.max() > 2.0:
                    a = np.gradient(vv, dt)
                    out.append({"v": vv, "a": a, "s": gap[i, t:t1], "vl": vl, "follower": i, "leader": int(j), "scene": sc["id"]})
            t = t1
    return out


def fit_episode(ep):
    v, a, s, vl = ep["v"], ep["a"], ep["s"], ep["vl"]
    lo, hi = list(BOUNDS[0]), list(BOUNDS[1])
    lo[4] = max(lo[4], float(v.max()))
    hi[4] = max(hi[4], lo[4] + 1)
    x0 = np.clip([1.2, 2.0, 1.5, 2.0, max(v.max() * 1.15, 10.0)], np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    r = least_squares(lambda p: idm(p, v, s, vl) - a, x0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0)
    res = idm(r.x, v, s, vl) - a
    r2 = 1 - np.var(res) / max(np.var(a), 1e-6)
    return {"T": float(r.x[0]), "s0": float(r.x[1]), "amax": float(r.x[2]), "b": float(r.x[3]), "v0": float(r.x[4]), "r2": float(r2),
            "n": len(v), "gap_range": float(s.max() - s.min()), "v_mean": float(v.mean()), "resid_sd": float(res.std()),
            "hw_median": float(np.median(s / np.maximum(v, 0.5)))}


def fit_all(scenes, truth=False):
    rows = []
    for sc in scenes:
        for ep in episodes(sc):
            f = fit_episode(ep)
            f["scene"] = ep["scene"]
            if truth and sc.get("truth"):
                f["T_true"] = float(sc["truth"]["T_head"][ep["follower"]])
            rows.append(f)
    return rows


def good(rows, r2=0.2, gap_range=2.0):
    return [r for r in rows if r["r2"] > r2 and r["gap_range"] > gap_range and r["v_mean"] > 3.0]


def bank(rows):
    """The trait table [M, 4] = (T, s0, amax, b) of the well-identified real drivers, clipped to the simulator's safe range."""
    g = good(rows)
    B = np.array([[r["T"], r["s0"], r["amax"], r["b"]] for r in g], np.float32)
    lo, hi = np.array([0.25, 0.3, 0.6, 1.0]), np.array([3.0, 6.0, 3.0, 5.0])
    return np.clip(B, lo, hi)


def summary(rows):
    g = good(rows)
    if not g:
        return {"episodes": len(rows), "good": 0}
    q = lambda k: [float(np.percentile([r[k] for r in g], p)) for p in (10, 50, 90)]
    out = {"episodes": len(rows), "good": len(g), "T_p10_50_90": q("T"), "s0_p10_50_90": q("s0"), "amax_p10_50_90": q("amax"),
           "b_p10_50_90": q("b"), "r2_median": float(np.median([r["r2"] for r in g])), "hw_median_p10_50_90": q("hw_median"),
           "resid_sd_median": float(np.median([r["resid_sd"] for r in g]))}
    if "T_true" in g[0]:
        from scipy.stats import spearmanr
        out["T_spearman_vs_truth"] = float(spearmanr([r["T"] for r in g], [r["T_true"] for r in g])[0])
        out["T_true_p10_50_90"] = q("T_true")
    return out


# ---------------------------------------------------------------- the identifiable version: steady following
def steady(ep, a_max=0.5, dv_max=1.0, v_min=3.0):
    """Frames of an episode in (near) equilibrium: small acceleration and nearly equal speeds, where IDM says s = s0 + v T."""
    m = (np.abs(ep["a"]) < a_max) & (np.abs(ep["v"] - ep["vl"]) < dv_max) & (ep["v"] > v_min)
    return ep["s"][m], ep["v"][m]


def population(scenes, skip_ego=True):
    """Two-stage fit. Stage 1, pooled over all steady frames: s = s0 + T v by least absolute deviations -> s0_pop, T_pop.
    Stage 2, per episode with >= 5 steady frames: the driver's time gap T_i = median((s - s0_pop) / v). Then amax and b, pooled, by least
    squares of IDM on every frame of every episode with each driver's T_i. -> dict(s0, T, amax, b, per-driver rows)."""
    eps = [e for sc in scenes for e in episodes(sc, skip_ego=skip_ego)]
    S, Vv = [], []
    for e in eps:
        s, v = steady(e)
        S.append(s), Vv.append(v)
    s_all, v_all = np.concatenate(S) if S else np.zeros(0), np.concatenate(Vv) if Vv else np.zeros(0)
    if len(s_all) < 20:
        return None
    r = least_squares(lambda p: p[0] + p[1] * v_all - s_all, [2.0, 1.0], bounds=([0.0, 0.05], [10.0, 5.0]), loss="soft_l1", f_scale=1.0)
    s0, T = (float(x) for x in r.x)
    rows = []
    for e, s, v in zip(eps, S, Vv):
        if len(s) >= 5:
            rows.append({"T": float(np.clip(np.median((s - s0) / v), 0.1, 4.0)), "episode": e, "n_steady": len(s)})
    # amax, b pooled with every driver's own T
    if rows:
        vv = np.concatenate([q["episode"]["v"] for q in rows])
        ss = np.concatenate([q["episode"]["s"] for q in rows])
        vl = np.concatenate([q["episode"]["vl"] for q in rows])
        aa = np.concatenate([q["episode"]["a"] for q in rows])
        TT = np.concatenate([np.full(len(q["episode"]["v"]), q["T"]) for q in rows])
        v0 = np.concatenate([np.full(len(q["episode"]["v"]), max(q["episode"]["v"].max() * 1.2, 12.0)) for q in rows])

        def res(p):
            ssr = s0 + np.maximum(0.0, vv * TT + vv * (vv - vl) / (2 * np.sqrt(p[0] * p[1])))
            return p[0] * (1 - (vv / v0) ** 4 - (ssr / np.maximum(ss, 0.1)) ** 2) - aa
        r2 = least_squares(res, [1.5, 2.0], bounds=([0.3, 0.5], [4.0, 6.0]), loss="soft_l1", f_scale=1.0)
        amax, b = (float(x) for x in r2.x)
    else:
        amax, b = 1.5, 2.0
    return {"s0": s0, "T": T, "amax": amax, "b": b, "drivers": [{"T": q["T"], "n_steady": q["n_steady"], "scene": q["episode"]["scene"],
                                                                   "follower": q["episode"]["follower"]} for q in rows],
            "steady_frames": int(len(s_all)), "episodes": len(eps)}


def pop_bank(pop, seed=0, s0_spread=0.25, a_spread=0.25):
    """Trait table from the two-stage fit: each row one real driver's T_i, the pooled s0, amax, b with a +-25 % spread (the simulator's own)."""
    rng = np.random.default_rng(seed)
    T = np.array([d["T"] for d in pop["drivers"]])
    n = len(T)
    B = np.stack([T, pop["s0"] * (1 + s0_spread * (2 * rng.random(n) - 1)), pop["amax"] * (1 + a_spread * (2 * rng.random(n) - 1)),
                  pop["b"] * (1 + a_spread * (2 * rng.random(n) - 1))], 1)
    return np.clip(B, [0.25, 0.3, 0.6, 1.0], [3.0, 6.0, 3.0, 5.0]).astype(np.float32)


def pop_summary(pop, scenes=None):
    T = np.array([d["T"] for d in pop["drivers"]])
    out = {k: pop[k] for k in ("s0", "T", "amax", "b", "steady_frames", "episodes")}
    out.update(drivers=len(T), T_i_p10_50_90=[float(np.percentile(T, p)) for p in (10, 50, 90)] if len(T) else None,
               T_i_share_below_0_8=float((T < 0.8).mean()) if len(T) else None)
    if scenes is not None and scenes and scenes[0].get("truth"):
        from scipy.stats import spearmanr
        by = {s["id"]: s for s in scenes}
        tt = np.array([float(by[d["scene"]]["truth"]["T_head"][d["follower"]]) for d in pop["drivers"]])
        out["T_i_spearman_vs_truth"] = float(spearmanr(T, tt)[0])
        out["T_true_p10_50_90"] = [float(np.percentile(tt, p)) for p in (10, 50, 90)]
    return out


# ---------------------------------------------------------------- the headway bank, by simulation-based deconvolution
def p10_headways(scenes, v_min=3.0, min_frames=5):
    """Per following episode: the 10th percentile of its time headway gap / v (frames above v_min). Observable on both sides; a driver's
    own T shows through it only blurred (gaps are often not constrained), which the deconvolution below undoes."""
    out = []
    for s in scenes:
        for e in episodes(s):
            m = e["v"] > v_min
            if m.sum() >= min_frames:
                out.append(np.percentile(e["s"][m] / e["v"][m], 10))
    return np.asarray(out)


QS = np.linspace(0.02, 0.98, 49)


def deconvolve(real_p10, simulate, iters=2, offset=0.8, seed=0, log=print):
    """Find a trait table whose *simulated* p10-headway distribution matches the real one. Start from T = real p10 - offset; then, per
    iteration, simulate with the bank, compare quantiles and move each bank quantile by the real-minus-simulated difference (a quantile-
    mapping fixed point, like iterative unfolding). simulate(bank) -> simulated p10 headways. -> (bank [M, 4], history)."""
    rng = np.random.default_rng(seed)
    qr = np.quantile(real_p10, QS)
    Tq = np.clip(qr - offset, 0.25, 3.0)
    hist = []
    def table(Tq, n=2000):
        T = np.interp(rng.random(n), QS, Tq)
        return np.stack([T, 1.5 + rng.random(n), 1.2 + 0.8 * rng.random(n), 1.5 + rng.random(n)], 1).astype(np.float32)
    B = table(Tq)
    for it in range(iters + 1):
        sp = simulate(B)
        qs = np.quantile(sp, QS)
        err = float(np.mean(np.abs(qs - qr)))
        hist.append({"iter": it, "mean_abs_quantile_err_s": err, "sim_p10_median": float(np.median(sp)), "real_p10_median": float(np.median(real_p10)),
                     "T_median": float(np.median(B[:, 0])), "sim_share_below_0_8": float((sp < 0.8).mean()), "real_share_below_0_8": float((real_p10 < 0.8).mean())})
        log(f"    deconvolve {it}: quantile error {err:.3f} s, T median {np.median(B[:, 0]):.2f}")
        if it == iters:
            break
        Tq = np.clip(Tq + (qr - qs), 0.25, 3.0)
        Tq = np.maximum.accumulate(Tq)
        B = table(Tq)
    return B, hist
