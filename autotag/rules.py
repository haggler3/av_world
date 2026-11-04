"""Kinematic taggers: rules on trajectories, the same code for real logs and simulated scenes.

Everything starts from one geometric fact per vehicle and frame: who is directly ahead. In vehicle i's frame at time t (x ahead,
y left), j is i's leader if it is the nearest vehicle with x > 0, |y| < 1.8 m and a heading within 35 degrees. Then
  gap = x - (L_i + L_j) / 2,   headway = gap / v_i,   TTC = gap / (v_i - v_j)   (when closing).
"""
import numpy as np

from .schema import Tag, intervals

LANE_HALF, HEAD_DIFF = 1.8, np.deg2rad(35)
# thresholds the rule-tuning stage searches over on the simulator golden set (defaults: before tuning)
DEFAULTS = {"lc_heading_deg": 25.0, "lc_lateral": 2.5, "lc_curve_deg": 90.0, "ci_heading_deg": 90.0, "ci_straight_deg": 90.0}


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def relative(sc):
    """Per frame: every vehicle's offsets in every other vehicle's frame. -> ax, ay, dh [V(i), V(j), T] (NaN where either is absent)."""
    x, y, h = sc["x"], sc["y"], sc["yaw"]
    dx, dy = x[None] - x[:, None], y[None] - y[:, None]
    c, s = np.cos(h)[:, None], np.sin(h)[:, None]
    return dx * c + dy * s, -dx * s + dy * c, wrap(h[None] - h[:, None])


def leaders(sc, rel=None):
    """-> lead [V, T] (index or -1), gap [V, T] (m, NaN without a leader)."""
    ax, ay, dh = rel if rel is not None else relative(sc)
    V, T = sc["x"].shape
    cand = (ax > 0) & (np.abs(ay) < LANE_HALF) & (np.abs(dh) < HEAD_DIFF)
    cand[np.arange(V), np.arange(V)] = False
    d = np.where(cand, ax, np.inf)
    lead = np.argmin(d, 1)                                    # [V, T]
    has = np.isfinite(np.take_along_axis(d, lead[:, None], 1)[:, 0])
    L = sc["length"]
    gap = np.where(has, np.take_along_axis(ax, lead[:, None], 1)[:, 0] - (L[:, None] + L[lead]) / 2, np.nan)
    return np.where(has, lead, -1), gap


def tag_scene(sc, tagger="rules", params=None):
    P = dict(DEFAULTS, **(params or {}))
    dt, V, T = sc["dt"], *sc["x"].shape
    rel = relative(sc)
    ax, ay, dh = rel
    lead, gap = leaders(sc, rel)
    v, h = sc["v"], sc["yaw"]
    vl = np.where(lead >= 0, v[np.maximum(lead, 0), np.arange(T)[None]], np.nan)               # the leader's speed [V, T]
    tags = []
    mk = lambda name, i, a, b, level="interval": Tag(sc["id"], sc["source"], name, level, int(i), float(a), float(b), 1.0, tagger)
    k1 = int(round(1.0 / dt))
    for i in range(V):
        if sc["kind"][i] in ("bike",):
            continue
        vi = v[i]
        # hard brake: speed drop > 4 m/s within 1 s, from above 3 m/s
        drop = np.zeros(T, bool)
        if T > k1:
            drop[:-k1] = (vi[:-k1] - vi[k1:] > 4.0) & (vi[:-k1] > 3.0)
        for a, b in intervals(drop, dt):
            tags.append(mk("hard_brake", i, a, b + 1.0))
        hw = gap[i] / np.maximum(vi, 1e-3)
        for a, b in intervals((hw < 0.8) & (vi > 3.0) & (gap[i] > 0), dt, min_len=k1):
            tags.append(mk("tailgating", i, a, b))
        closing = vi - vl[i]
        ttc = np.where((closing > 0.5) & (gap[i] < 25.0) & (gap[i] > 0), gap[i] / np.maximum(closing, 1e-3), np.inf)
        for a, b in intervals(ttc < 1.5, dt):
            tags.append(mk("low_ttc", i, a, b))
        # cut-in: j becomes i's leader within 15 m, and 2 s earlier j was ahead but in the next lane (|y| between 2.5 and 6 m)
        k2 = int(round(2.0 / dt))
        for t in range(k2, T):
            j = lead[i, t]
            if j < 0 or lead[i, t - 1] == j or gap[i, t] > 15.0:
                continue
            y0, x0 = ay[i, j, t - k2], ax[i, j, t - k2]
            par = abs(dh[i, j, t - k2]) < np.deg2rad(P["ci_heading_deg"])                         # j drove parallel to i: a lane, not a crossing road
            own = abs(wrap(h[i, t] - h[i, t - k2])) < np.deg2rad(P["ci_straight_deg"])           # and i itself was not turning or on a curve
            if np.isfinite(y0) and 2.5 < abs(y0) < 6.0 and x0 > -5.0 and par and own:
                tags.append(mk("cut_in", j, (t - k2) * dt, t * dt))
        # lane change: lateral move > 2.5 m relative to the heading 3 s earlier, heading change < 25 degrees
        k3 = int(round(3.0 / dt))
        lc = np.zeros(T, bool)
        for t in range(k3, T):
            a = t - k3
            if np.isfinite(sc["x"][i, a]) and np.isfinite(sc["x"][i, t]):
                c, s = np.cos(h[i, a]), np.sin(h[i, a])
                lat = -(sc["x"][i, t] - sc["x"][i, a]) * s + (sc["y"][i, t] - sc["y"][i, a]) * c
                seg = np.unwrap(np.nan_to_num(h[i, a:t + 1]))
                curve = seg.max() - seg.min()                                                        # a lane change swings out and back; a curve keeps turning
                lc[t] = (abs(lat) > P["lc_lateral"] and abs(wrap(h[i, t] - h[i, a])) < np.deg2rad(P["lc_heading_deg"])
                         and curve < np.deg2rad(P["lc_curve_deg"]) and vi[t] > 2.0)
        for a, b in intervals(lc, dt):
            tags.append(mk("lane_change", i, max(a - 3.0, 0.0), b))
        k4 = int(round(4.0 / dt))
        tu = np.zeros(T, bool)
        if T > k4:
            tu[k4:] = np.abs(wrap(h[i, k4:] - h[i, :-k4])) > np.deg2rad(60)
        for a, b in intervals(tu, dt):
            tags.append(mk("turning", i, max(a - 4.0, 0.0), b))
        q = (vi < 0.5) & (gap[i] < 10.0) & (vl[i] < 0.5)
        for a, b in intervals(q, dt, min_len=2 * k1):
            tags.append(mk("queued", i, a, b))
        if len(sc["px"]):
            dp = np.hypot(sc["px"][:, None, :] - sc["x"][i][None, None, :], sc["py"][:, None, :] - sc["y"][i][None, None, :])[:, 0]
            near = (np.nanmin(np.where(np.isfinite(dp), dp, np.inf), 0) < 4.0) & (vi > 3.0)
            for a, b in intervals(near, dt):
                tags.append(mk("ped_close", i, a, b))
    e = sc["ego"]
    d = np.hypot(sc["x"] - sc["x"][e], sc["y"] - sc["y"][e])
    dens = np.nanmean(np.sum(np.where(np.isfinite(d), d < 30.0, False), 0) - 1)
    if dens > 8:
        tags.append(Tag(sc["id"], sc["source"], "dense_traffic", "scene", -1, 0.0, T * dt, 1.0, tagger))
    return tags


def scene_features(sc, tags=None):
    """A fixed-length description of a scene (domain check, search, the scene-level learned tagger): kinematics + tag rates."""
    v = sc["v"][sc["alive"]]
    a = np.diff(sc["v"], axis=1)[sc["alive"][:, 1:] & sc["alive"][:, :-1]] / sc["dt"]
    a = a[np.isfinite(a) & (np.abs(a) < 10)]
    mv = v[v > 1.0]
    e = sc["ego"]
    d = np.hypot(sc["x"] - sc["x"][e], sc["y"] - sc["y"][e])
    tags = tag_scene(sc) if tags is None else tags
    names = ("hard_brake", "tailgating", "low_ttc", "cut_in", "lane_change", "turning", "queued", "ped_close")
    n_veh = max(int(sc["alive"].any(1).sum()), 1)
    rates = [sum(t.name == n for t in tags) / n_veh for n in names]
    f = [np.nanmean(np.sum(np.where(np.isfinite(d), d < 30.0, False), 0)), n_veh / 50.0,
         np.median(mv) / 10 if len(mv) else 0.0, np.percentile(mv, 90) / 10 if len(mv) else 0.0, (v < 0.5).mean() if len(v) else 0.0,
         a.std() if len(a) else 0.0, np.percentile(a, 2) / 5 if len(a) else 0.0, np.percentile(a, 98) / 5 if len(a) else 0.0,
         len(sc["px"]) / 10.0]
    return np.asarray(f + rates, np.float32)
