"""A classifier two-sample test: train a network to tell real vehicles from simulated ones. If it cannot (AUROC 0.5), the simulator's
drivers are, by these features, indistinguishable from real ones; what it relies on is what gives the simulator away.

Unit: one moving vehicle in one scene (at least 5 s moving, not the ego). Features (all from the shared scene format, after the same
measurement processing on both sides): speed level and spread, acceleration spread and tails, jerk, the short-lag share of the velocity
structure function (how jittery), time with a leader, headway level and minimum, the largest 1 / TTC, local density, turning share.
Split by scene (no scene in both train and test); permutation importance on the test half.
"""
import jax
import numpy as np

from autotag.rules import leaders, relative
from nets import common as NC

NAMES = ("v_mean", "v_sd", "a_sd", "a_p02", "a_p98", "jerk_rms", "D1_over_D5", "lead_share", "hw_median", "hw_min", "ttc_inv_max",
         "density", "turn_share")


def units(sc):
    dt = sc["dt"]
    v, x, y, h = sc["v"], sc["x"], sc["y"], sc["yaw"]
    lead, gap = leaders(sc, relative(sc))
    T = v.shape[1]
    vl = np.where(lead >= 0, v[np.maximum(lead, 0), np.arange(T)[None]], np.nan)
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    out = []
    for i in range(len(v)):
        if i == sc["ego"] or ("kind" in sc and sc["kind"][i] == "bike"):
            continue
        m = np.isfinite(v[i]) & (v[i] > 1.0)
        if m.sum() < 25:
            continue
        vi = v[i][np.isfinite(v[i])]
        a = np.diff(vi) / dt
        j = np.diff(a) / dt
        d1 = np.mean(np.diff(vi) ** 2)
        d5 = np.mean((vi[5:] - vi[:-5]) ** 2) if len(vi) > 5 else d1
        has = (lead[i] >= 0) & (gap[i] > 0) & (gap[i] < 60) & m
        hw = gap[i][has] / np.maximum(v[i][has], 0.5)
        ti = np.where(has & (gap[i] < 25), np.maximum(v[i] - vl[i], 0) / np.maximum(gap[i], 0.1), 0.0)
        yr = np.abs(np.angle(np.exp(1j * np.diff(h[i][np.isfinite(h[i])])))) / dt
        out.append([vi.mean(), vi.std(), np.clip(a, -10, 10).std(), np.percentile(np.clip(a, -10, 10), 2), np.percentile(np.clip(a, -10, 10), 98),
                    np.sqrt(np.mean(np.clip(j, -60, 60) ** 2)) if len(j) else 0.0, d1 / max(d5, 1e-6), has.mean(),
                    np.median(hw) if len(hw) else 6.0, hw.min() if len(hw) else 6.0, np.nanmax(ti) if np.isfinite(ti).any() else 0.0,
                    np.nanmean((np.where(np.isfinite(d[i]), d[i], np.inf) < 30).sum(0)[m]), (yr > 0.15).mean() if len(yr) else 0.0])
    return np.asarray(out, np.float32).reshape(-1, len(NAMES))


def test(real, sim, steps=3000, seed=0):
    """-> dict(auroc, auroc_ci, importance {feature: AUROC drop}, n)."""
    Ur = [units(s) for s in real]
    Us = [units(s) for s in sim]
    rng = np.random.default_rng(seed)
    def split(U):
        idx = rng.permutation(len(U))
        h = len(U) // 2
        return np.concatenate([U[i] for i in idx[:h]] or [np.zeros((0, len(NAMES)))]), np.concatenate([U[i] for i in idx[h:]] or [np.zeros((0, len(NAMES)))])
    r_tr, r_te = split(Ur)
    s_tr, s_te = split(Us)
    # balance the classes in training
    n = min(len(r_tr), len(s_tr))
    Xtr = np.r_[r_tr[rng.permutation(len(r_tr))[:n]], s_tr[rng.permutation(len(s_tr))[:n]]]
    ytr = np.r_[np.ones(n), np.zeros(n)].astype(np.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xte = np.r_[r_te, s_te]
    yte = np.r_[np.ones(len(r_te)), np.zeros(len(s_te))]
    p = NC.init_mlp(jax.random.PRNGKey(seed), [len(NAMES), 64, 64, 1])

    def loss(p, b):
        z = NC.mlp(p, b["x"])[:, 0]
        return jax.numpy.mean(jax.nn.softplus(z) - b["y"] * z)
    p, _ = NC.fit(loss, p, {"x": (Xtr - mu) / sd, "y": ytr}, steps=steps, batch=512, lr=2e-3, seed=seed)
    score = lambda X: np.asarray(NC.mlp(p, (X - mu) / sd)[:, 0])
    base = NC.auroc(score(Xte), yte)
    imp = {}
    for k, name in enumerate(NAMES):
        Xp = Xte.copy()
        Xp[:, k] = Xp[rng.permutation(len(Xp)), k]
        imp[name] = float(base - NC.auroc(score(Xp), yte))
    # bootstrap over test units
    s = score(Xte)
    bs = [NC.auroc(s[i], yte[i]) for i in (rng.integers(0, len(s), len(s)) for _ in range(200))]
    return {"auroc": float(base), "auroc_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])), "n_train": int(2 * n), "n_test": int(len(yte)),
            "real_mean": dict(zip(NAMES, map(float, np.concatenate(Ur).mean(0)))), "sim_mean": dict(zip(NAMES, map(float, np.concatenate(Us).mean(0))))}
