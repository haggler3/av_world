"""Measurement parity: the part of the gap that is the sensor, not the drivers.

A real log's speeds come from a tracker. Two facts from the logs themselves:
  * spikes: a share of tracks has speed jumps no car can make (|dv| / dt > 12 m/s^2 within one 0.2 s step);
  * correlated noise: the velocity structure function D(k) = E[(v[t+k] - v[t])^2] of real tracks rises almost linearly at short lags,
    where a smooth driver's D(k) rises like k^2.

Model (first principles): observed speed y = v + e. The driver's acceleration is an Ornstein-Uhlenbeck process (sd sa, time constant tau),
the tracker's error an AR(1) (sd sm, lag-one correlation rho per step). Then, for lag h = k dt,
  D(k) = 2 sa^2 tau^2 (h / tau - 1 + exp(-h / tau))  +  2 sm^2 (1 - rho^k).
Fitting the four numbers to D(1..15) of non-spiky real tracks splits each track's wiggle into driver and sensor. The fitted model then works
in both directions:
  clean(real)    spikes out (a Hampel filter), then a Rauch-Tung-Striebel smoother with exactly this state-space model: the best linear
                 estimate of the driver's speed;
  corrupt(sim)   the same AR(1) error and spikes, at the measured rates, added to simulated speeds (and position jitter): the simulator seen
                 through the real sensor.
A statistic that differs between real and sim after both is a difference of driving.
"""
import numpy as np
from scipy.optimize import least_squares

DT = 0.2
SPIKE = 12.0          # m/s^2 within one step: no car does this


def spiky(v, dt=DT):
    """Tracks [V, T] with at least one impossible speed jump."""
    dv = np.abs(np.diff(v, axis=1)) / dt
    return np.nanmax(np.where(np.isfinite(dv), dv, 0.0), 1) > SPIKE


def tracks(scenes, who="other", min_len=30):
    """Speed series of moving tracks: who = 'other' (tracked road users) or 'ego' (the AV's own localisation)."""
    out = []
    for s in scenes:
        for i in range(len(s["v"])):
            if (who == "ego") != (i == s["ego"]):
                continue
            if "kind" in s and s["kind"][i] == "bike":
                continue
            m = np.isfinite(s["v"][i])
            if m.sum() < min_len or np.nanmax(s["v"][i]) < 3.0:
                continue
            out.append(s["v"][i][m].astype(float))
    return out


def structure(series, K=15):
    D = np.zeros(K)
    for k in range(1, K + 1):
        d = np.concatenate([(v[k:] - v[:-k]) ** 2 for v in series if len(v) > k])
        D[k - 1] = d.mean()
    return D


def d_model(p, k, dt=DT):
    sm, rho, sa, tau = p
    h = k * dt
    return 2 * sa ** 2 * tau ** 2 * (h / tau - 1 + np.exp(-h / tau)) + 2 * sm ** 2 * (1 - rho ** k)


def fit_structure(D, dt=DT):
    k = np.arange(1, len(D) + 1)
    r = least_squares(lambda p: np.log(d_model(p, k, dt)) - np.log(D), [0.2, 0.5, 1.0, 2.0],
                      bounds=([0.0, 0.0, 0.05, 0.2], [2.0, 0.98, 5.0, 30.0]))
    sm, rho, sa, tau = (float(x) for x in r.x)
    return {"sm": sm, "rho": rho, "sa": sa, "tau": tau, "fit_log_rms": float(np.sqrt(np.mean(r.fun ** 2)))}


def noise_model(scenes, pos_scenes=None):
    """Fit the sensor model of real logs, separately for tracked road users and for the ego (localisation)."""
    out = {}
    for who in ("other", "ego"):
        ser = tracks(scenes, who)
        bad = np.array([spiky(v[None])[0] for v in ser]) if ser else np.zeros(0, bool)
        clean = [v for v, b in zip(ser, bad) if not b]
        D = structure(clean)
        f = fit_structure(D)
        # spikes: the per-step rate of an impossible jump and the jumps' size distribution
        jumps = np.concatenate([np.diff(v) for v in ser]) if ser else np.zeros(1)
        big = np.abs(jumps) / DT > SPIKE
        f.update(spiky_track_share=float(bad.mean()) if len(bad) else 0.0, spike_rate=float(big.mean()),
                 spike_sizes=np.round(jumps[big], 2).tolist()[:2000] or [3.0], D=D.tolist(), tracks=len(ser))
        out[who] = f
    # position jitter: parked vehicles (never above 0.5 m/s, at least 6 s logged)
    j = []
    for s in pos_scenes or scenes:
        for i in range(len(s["x"])):
            m = np.isfinite(s["x"][i])
            if m.sum() >= 30 and np.nanmax(s["v"][i]) < 0.5:
                j.append(np.hypot(s["x"][i][m] - s["x"][i][m].mean(), s["y"][i][m] - s["y"][i][m].mean()))
    out["pos_jitter"] = float(np.sqrt(np.mean(np.concatenate(j) ** 2) / 2)) if j else 0.0
    return out


def crop(sc, radius=80.0):
    """Field-of-view parity: keep the road users within `radius` of the ego at the middle frame, as simulated scenes are cut
    (autotag.scenes.RADIUS). An AV2 log tracks objects much further out; uncropped, a real scene has several times the vehicles."""
    T = sc["x"].shape[1]
    e, k = sc["ego"], T // 2
    ex, ey = sc["x"][e, k], sc["y"][e, k]
    if not np.isfinite(ex):
        ex, ey = np.nanmean(sc["x"][e]), np.nanmean(sc["y"][e])
    # a vehicle's distance at the middle frame, or its closest approach if it is not logged then
    d = np.nanmin(np.hypot(sc["x"] - ex, sc["y"] - ey), 1)
    dm = np.hypot(sc["x"][:, k] - ex, sc["y"][:, k] - ey)
    d = np.where(np.isfinite(dm), dm, d)
    keep = np.nonzero(np.isfinite(d) & (d < radius))[0]
    keep = np.r_[e, keep[keep != e]]
    s = dict(sc)
    for f in ("x", "y", "yaw", "v", "alive"):
        s[f] = sc[f][keep]
    for f in ("kind", "av", "length"):
        s[f] = np.asarray(sc[f])[keep]
    s["ego"] = 0
    if len(sc["px"]):
        pd = np.nanmin(np.hypot(sc["px"] - ex, sc["py"] - ey), 1)
        pk = np.nonzero(np.isfinite(pd) & (pd < radius))[0]
        s["px"], s["py"] = sc["px"][pk], sc["py"][pk]
    return s


# ---------------------------------------------------------------- real -> clean
def hampel(v, k=2, n_sigma=3.0, floor=0.5):
    """Replace isolated spikes: a sample further than n_sigma robust SDs (and `floor` m/s) from its window median."""
    out = v.copy()
    T = len(v)
    for t in range(T):
        w = v[max(0, t - k): t + k + 1]
        w = w[np.isfinite(w)]
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = 1.4826 * np.median(np.abs(w - med))
        if abs(v[t] - med) > max(n_sigma * mad, floor):
            out[t] = med
    return out


def rts(y, nm, dt=DT, r=1e-4):
    """Rauch-Tung-Striebel smoother for one speed series under the fitted model: state (v, a, e), y = v + e (+ a tiny white term)."""
    phi_a, phi_e = np.exp(-dt / nm["tau"]), nm["rho"]
    F = np.array([[1.0, dt, 0.0], [0.0, phi_a, 0.0], [0.0, 0.0, phi_e]])
    Q = np.diag([1e-6, nm["sa"] ** 2 * (1 - phi_a ** 2), nm["sm"] ** 2 * (1 - phi_e ** 2) + 1e-9])
    H = np.array([1.0, 0.0, 1.0])
    T = len(y)
    xs, Ps, xp, Pp = np.zeros((T, 3)), np.zeros((T, 3, 3)), np.zeros((T, 3)), np.zeros((T, 3, 3))
    x = np.array([y[np.isfinite(y)][0] if np.isfinite(y).any() else 0.0, 0.0, 0.0])
    P = np.diag([1.0, nm["sa"] ** 2, nm["sm"] ** 2])
    for t in range(T):
        if t:
            x, P = F @ x, F @ P @ F.T + Q
        xp[t], Pp[t] = x, P
        if np.isfinite(y[t]):
            S = H @ P @ H + r
            K = P @ H / S
            x = x + K * (y[t] - H @ x)
            P = P - np.outer(K, H @ P)
        xs[t], Ps[t] = x, P
    for t in range(T - 2, -1, -1):
        C = Ps[t] @ F.T @ np.linalg.inv(Pp[t + 1])
        xs[t] = xs[t] + C @ (xs[t + 1] - xp[t + 1])
        Ps[t] = Ps[t] + C @ (Ps[t + 1] - Pp[t + 1]) @ C.T
    return xs[:, 0]


def clean(sc, nm):
    """A real scene with every tracked speed despiked and smoothed (the ego with its own, much smaller, sensor model)."""
    s = dict(sc)
    v = sc["v"].copy()
    for i in range(len(v)):
        m = np.isfinite(v[i])
        if m.sum() < 5:
            continue
        model = nm["ego"] if i == sc["ego"] else nm["other"]
        y = hampel(v[i][m])
        v[i][m] = np.maximum(rts(y, model), 0.0)
    s["v"] = v
    return s


# ---------------------------------------------------------------- sim -> through the real sensor
def corrupt(sc, nm, seed=0):
    """A simulated scene seen through the real tracker: AR(1) speed error, spikes at the measured rate, position jitter."""
    rng = np.random.default_rng(seed)
    s = dict(sc)
    v, x, y = sc["v"].copy(), sc["x"].copy(), sc["y"].copy()
    V, T = v.shape
    for i in range(V):
        m = np.isfinite(v[i])
        if not m.any():
            continue
        md = nm["ego"] if i == sc["ego"] else nm["other"]
        n = int(m.sum())
        e = np.zeros(n)
        e[0] = rng.normal() * md["sm"]
        for t in range(1, n):
            e[t] = md["rho"] * e[t - 1] + rng.normal() * md["sm"] * np.sqrt(1 - md["rho"] ** 2)
        sp = rng.random(n) < md["spike_rate"]
        e = e + np.where(sp, rng.choice(np.asarray(md["spike_sizes"]), n), 0.0)
        moving = np.nanmax(v[i]) > 0.5
        v[i][m] = np.maximum(v[i][m] + (e if moving else 0.1 * e), 0.0)
        x[i][m] += rng.normal(size=n) * nm["pos_jitter"]
        y[i][m] += rng.normal(size=n) * nm["pos_jitter"]
    s["v"], s["x"], s["y"] = v, x, y
    return s
