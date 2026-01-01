"""Scene-mix reweighting: is the gap the drivers, or just that the simulated city is not the same mix of streets and traffic?

Context of a scene (no driving style in it): median moving speed, how many vehicles move within 30 m of the ego, the share of standing
vehicle-frames, the number of moving vehicles, the share of turning frames. A logistic classifier p(real | context) gives the density ratio
w(c) = p(real | c) / p(sim | c) x n_sim / n_real for every simulated scene (clipped to [0.05, 20], normalised to mean 1). Re-scoring with
these weights compares the simulator with real logs *as if* it drove the real mix; the effective sample size says how much that costs.
"""
import numpy as np

NAMES = ("median_speed", "density", "standing_share", "moving_vehicles", "turning_share")


def context(sc):
    v, h = sc["v"], sc["yaw"]
    mv = v[np.isfinite(v) & (v > 1.0)]
    e = sc["ego"]
    d = np.hypot(sc["x"] - sc["x"][e], sc["y"] - sc["y"][e])
    ever = np.nanmax(np.where(np.isfinite(v), v, 0), 1) > 1.0
    dens = np.nanmean(((np.where(np.isfinite(d), d, np.inf) < 30.0) & ever[:, None]).sum(0))
    yr = np.abs(np.angle(np.exp(1j * np.diff(h, axis=1)))) / sc["dt"]
    alive = np.isfinite(v)
    return np.array([np.median(mv) if len(mv) else 0.0, dens, (v[alive] < 0.5).mean() if alive.any() else 0.0, ever.sum(),
                     np.nanmean(yr[np.isfinite(yr)] > 0.15) if np.isfinite(yr).any() else 0.0], float)


def logistic(X, y, l2=1e-2, iters=50):
    Xb = np.c_[X, np.ones(len(X))]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def weights(real, sim, clip=(0.05, 20.0)):
    Cr, Cs = np.stack([context(s) for s in real]), np.stack([context(s) for s in sim])
    mu, sd = np.r_[Cr, Cs].mean(0), np.r_[Cr, Cs].std(0) + 1e-9
    X = (np.r_[Cr, Cs] - mu) / sd
    y = np.r_[np.ones(len(Cr)), np.zeros(len(Cs))]
    w = logistic(X, y)
    p = 1 / (1 + np.exp(-(np.c_[(Cs - mu) / sd, np.ones(len(Cs))] @ w)))
    r = np.clip(p / (1 - p) * len(Cs) / len(Cr), *clip)
    r = r / r.mean()
    ess = r.sum() ** 2 / (r ** 2).sum()
    return r, {"ess": float(ess), "n_sim": len(Cs), "coef": dict(zip(NAMES, map(float, w[:-1]))),
               "real_context_mean": dict(zip(NAMES, map(float, Cr.mean(0)))), "sim_context_mean": dict(zip(NAMES, map(float, Cs.mean(0)))),
               "sim_context_mean_weighted": dict(zip(NAMES, map(float, (Cs * r[:, None]).sum(0) / r.sum())))}
