"""One extractor for real and simulated scenes (the autotag scene format: dense [V, T] arrays on a 0.2 s grid) and the scoreboard.

For every moving vehicle-frame (speed > 1 m/s; the ego, which is an AV in both sources, is left out) it records
  v        speed
  a        acceleration over one step, (v[t+1] - v[t]) / dt
  a1       acceleration over one second (the incident audit's hard-brake definition)
  jerk     change of a per step
  hw       time headway gap / v to the leader (same lane, within 60 m, v > 3 m/s)
  ttc_inv  closing speed / gap (1 / time to collision; 0 when opening)
  gap0     the bumper gap in queues: follower and leader both stopped (frames of their own)
and its context: speed bin, whether there is a leader, whether the vehicle is turning (|yaw rate| > 0.15 rad/s) and how many moving
vehicles are within 30 m. Tail counts follow avlab.incidents (hard brake: a 4 m/s drop within 1 s from above 3 m/s, one per vehicle per
3 s; tight following: headway < 0.8 s; TTC under 2 s and 1 s with closing > 0.5 m/s and gap < 25 m) so the rates are comparable.
"""
import numpy as np
from scipy.stats import wasserstein_distance

from autotag.rules import leaders, relative, wrap

DISTS = ("v", "a", "a1", "jerk", "hw", "ttc_inv", "gap0")
TAILS = ("hard_brakes_per_h", "tight_share", "ttc2_share", "ttc1_share")
SPEED_BINS = (1.0, 5.0, 10.0, 15.0, 99.0)
K_MAX = 10
LR_CAP = 3.0         # a tail the simulator never produces counts as a factor e^3 = 20 off, not infinitely


def measure(sc, skip_ego=True):
    dt = sc["dt"]
    v, x, y, h = sc["v"], sc["x"], sc["y"], sc["yaw"]
    V, T = v.shape
    lead, gap = leaders(sc, relative(sc))
    vl = np.where(lead >= 0, v[np.maximum(lead, 0), np.arange(T)[None]], np.nan)
    a = np.full((V, T), np.nan)
    a[:, :-1] = (v[:, 1:] - v[:, :-1]) / dt
    k1 = int(round(1.0 / dt))
    a1 = np.full((V, T), np.nan)
    a1[:, :-k1] = (v[:, k1:] - v[:, :-k1]) / (k1 * dt)
    jerk = np.full((V, T), np.nan)
    jerk[:, :-1] = (a[:, 1:] - a[:, :-1]) / dt
    yr = np.full((V, T), np.nan)
    yr[:, :-1] = wrap(h[:, 1:] - h[:, :-1]) / dt
    ever = np.nanmax(np.where(np.isfinite(v), v, 0.0), 1) > 1.0          # parked cars never count as traffic
    d = np.hypot(x[:, None] - x[None], y[:, None] - y[None])
    near = (np.where(np.isfinite(d), d, np.inf) < 30.0) & ever[None, :, None]
    dens = near.sum(1) - 1
    moving = np.isfinite(v) & (v > 1.0)
    if skip_ego:
        moving[sc["ego"]] = False
    if "kind" in sc:
        moving &= (np.asarray(sc["kind"]) != "bike")[:, None]
    has = (lead >= 0) & (gap > 0) & (gap < 60.0)
    hw = np.where(has & (v > 3.0), gap / np.maximum(v, 1e-3), np.nan)
    closing = v - vl
    ttc_inv = np.where(has & (gap < 25.0), np.maximum(closing, 0.0) / np.maximum(gap, 0.1), np.nan)
    m = moving
    # standstill gaps: follower and leader both stopped (queues at lights); keeps a fit from buying tight following with tiny s0
    still = np.isfinite(v) & (v < 0.3) & (lead >= 0) & (vl < 0.3) & (gap > -1.0) & (gap < 15.0)
    still[sc["ego"]] = False
    if "kind" in sc:
        still &= (np.asarray(sc["kind"]) != "bike")[:, None]
    gap0 = gap[still]
    out = {"v": v[m], "a": a[m], "a1": a1[m], "jerk": jerk[m], "hw": hw[m], "ttc_inv": ttc_inv[m],
           "ctx_speed": np.digitize(v[m], SPEED_BINS[1:-1]), "ctx_lead": has[m], "ctx_turn": np.abs(yr[m]) > 0.15, "ctx_dens": dens[m]}
    # tail counts (exposure: moving vehicle-seconds)
    hb = 0
    drop = np.zeros((V, T), bool)
    drop[:, :-k1] = (v[:, :-k1] - v[:, k1:] > 4.0) & (v[:, :-k1] > 3.0) & m[:, :-k1]
    gap_n = int(3.0 / dt)
    for i in np.nonzero(drop.any(1))[0]:
        last = -10 ** 9
        for t in np.nonzero(drop[i])[0]:
            if t - last > gap_n:
                hb += 1
            last = t
    fast = m & (v > 3.0)
    ttc = np.where((closing > 0.5) & has & (gap < 25.0), gap / np.maximum(closing, 1e-3), np.inf)
    out["gap0"] = gap0
    out["tail"] = np.array([m.sum() * dt, hb, ((hw < 0.8) & fast).sum() * dt, ((ttc < 2.0) & fast).sum() * dt, ((ttc < 1.0) & fast).sum() * dt])
    # velocity structure function D(k) = E[(v[t+k] - v[t])^2] of moving non-ego tracks: sums and counts, so scenes pool
    D = np.zeros((2, K_MAX))
    vv = np.where(m, v, np.nan)
    for k in range(1, K_MAX + 1):
        dv = (vv[:, k:] - vv[:, :-k]) ** 2
        ok = np.isfinite(dv)
        D[0, k - 1], D[1, k - 1] = dv[ok].sum(), ok.sum()
    out["D"] = D
    return out


def pool(ms, weights=None):
    """Concatenate per-scene measures; `weights` (one per scene) become per-sample weights."""
    w = np.ones(len(ms)) if weights is None else np.asarray(weights, float)
    P = {k: np.concatenate([m[k] for m in ms]) for k in ms[0] if k not in ("tail", "D", "gap0")}
    P["gap0"] = np.concatenate([m.get("gap0", np.zeros(0)) for m in ms])
    P["w"] = np.concatenate([np.full(len(m["v"]), wi) for m, wi in zip(ms, w)])
    P["scene"] = np.concatenate([np.full(len(m["v"]), i) for i, m in enumerate(ms)])
    P["tail_scene"] = np.stack([m["tail"] for m in ms])
    P["scene_w"] = w
    P["D"] = sum(m["D"] * wi for m, wi in zip(ms, w))
    return P


def tails(tail, w=None):
    t = (tail * (np.ones(len(tail)) if w is None else w)[:, None]).sum(0)
    e = max(t[0], 1e-9)
    return {"hard_brakes_per_h": t[1] / (e / 3600), "tight_share": t[2] / e, "ttc2_share": t[3] / e, "ttc1_share": t[4] / e}


def structure(P):
    return P["D"][0] / np.maximum(P["D"][1], 1)


def _w1(a, b, wa=None, wb=None):
    ok_a, ok_b = np.isfinite(a), np.isfinite(b)
    if ok_a.sum() < 20 or ok_b.sum() < 20:
        return np.nan
    return wasserstein_distance(a[ok_a], b[ok_b], None if wa is None else wa[ok_a], None if wb is None else wb[ok_b])


def _scale(a):
    a = a[np.isfinite(a)]
    q = np.percentile(a, [25, 75]) if len(a) else [0, 1]
    return max(q[1] - q[0], 1e-6)


CLIP = {"a": 10.0, "a1": 10.0, "jerk": 60.0, "hw": 6.0, "ttc_inv": 3.0, "gap0": 15.0}


def _arr(P, k):
    x = P[k].astype(float)
    return np.clip(x, -CLIP[k], CLIP[k]) if k in CLIP else x


def scoreboard(R, S, n_boot=200, seed=0, where=True):
    """Real pool R against simulated pool S. Tail rates: sim / real with a bootstrap CI over scenes (both sides resampled).
    Distributions: Wasserstein-1 distance in units of the real interquartile range. `gap` = mean |log ratio| of the tails + mean W1:
    the single number the refit minimises."""
    rng = np.random.default_rng(seed)
    tr, ts = tails(R["tail_scene"], R["scene_w"]), tails(S["tail_scene"], S["scene_w"])
    out = {"tails": {}, "dists": {}, "where": [], "n": {"real_samples": int(len(R["v"])), "sim_samples": int(len(S["v"])),
                                                         "real_scenes": int(len(R["scene_w"])), "sim_scenes": int(len(S["scene_w"]))}}
    nr, ns = len(R["scene_w"]), len(S["scene_w"])
    boots = []
    for _ in range(n_boot):
        ir, is_ = rng.integers(0, nr, nr), rng.integers(0, ns, ns)
        br = tails(R["tail_scene"][ir], R["scene_w"][ir])
        bs = tails(S["tail_scene"][is_], S["scene_w"][is_])
        boots.append([np.log((bs[k] + 1e-9) / (br[k] + 1e-9)) for k in TAILS])
    boots = np.array(boots).reshape(-1, len(TAILS))
    for j, k in enumerate(TAILS):
        lr = float(np.log((ts[k] + 1e-9) / (tr[k] + 1e-9)))
        lo, hi = np.percentile(boots[:, j], [2.5, 97.5]) if len(boots) else (lr, lr)
        out["tails"][k] = {"real": tr[k], "sim": ts[k], "ratio": float(np.exp(lr)), "ratio_ci": [float(np.exp(lo)), float(np.exp(hi))], "log_ratio": lr}
    for k in DISTS:
        a, b = _arr(R, k), _arr(S, k)
        sc = _scale(a)
        w1 = (_w1(a, b) if k == "gap0" else _w1(a, b, R["w"], S["w"])) / sc
        fa, fb = a[np.isfinite(a)], b[np.isfinite(b)]
        out["dists"][k] = {"w1_iqr": float(w1), "real_median": float(np.median(fa)) if len(fa) else None,
                           "sim_median": float(np.median(fb)) if len(fb) else None,
                           "real_sd": float(fa.std()) if len(fa) else None, "sim_sd": float(fb.std()) if len(fb) else None}
    Dr, Ds = structure(R), structure(S)
    out["structure"] = {"real": Dr.tolist(), "sim": Ds.tolist(), "log_rms": float(np.sqrt(np.mean(np.log((Ds + 1e-6) / (Dr + 1e-6)) ** 2)))}
    out["gap"] = float(np.mean([min(abs(out["tails"][k]["log_ratio"]), LR_CAP) for k in TAILS]) + np.nanmean([out["dists"][k]["w1_iqr"] for k in DISTS]))
    if where:
        out["where"] = where_table(R, S)
    return out


def where_table(R, S):
    """The gap per context cell: speed bin x leader x turning. Per cell: share of samples, W1 of the acceleration (in real IQR units),
    the tight-following share sim / real, and the headway W1."""
    rows = []
    names = ["1-5", "5-10", "10-15", "15+"]
    for sb in range(4):
        for ld in (False, True):
            for tu in (False, True):
                mr = (R["ctx_speed"] == sb) & (R["ctx_lead"] == ld) & (R["ctx_turn"] == tu)
                ms = (S["ctx_speed"] == sb) & (S["ctx_lead"] == ld) & (S["ctx_turn"] == tu)
                if mr.sum() < 50 or ms.sum() < 50:
                    continue
                ar, as_ = _arr(R, "a")[mr], _arr(S, "a")[ms]
                row = {"speed": names[sb], "leader": ld, "turning": tu, "real_share": float(mr.mean()), "sim_share": float(ms.mean()),
                       "w1_a": float(_w1(ar, as_) / _scale(_arr(R, "a"))), "real_a_sd": float(np.nanstd(ar)), "sim_a_sd": float(np.nanstd(as_))}
                if ld:
                    hr, hs = R["hw"][mr], S["hw"][ms]
                    tr, ts = np.nanmean(hr < 0.8), np.nanmean(hs < 0.8)
                    row.update(real_tight=float(tr), sim_tight=float(ts), w1_hw=float(_w1(np.clip(hr, 0, 6), np.clip(hs, 0, 6)) / _scale(R["hw"])))
                rows.append(row)
    return rows
