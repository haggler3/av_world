"""Can an AV find out the road friction from its own braking? (docs/DAILY.md, 2026-11-09 follow-up)

    python -m avlab.frictionsense [--quick]   # run + estimate -> results/frictionsense/frictionsense.json
    python -m avlab.frictionsense plot        # figure and GIF  -> docs/img/frictionsense.png|gif

xnet has ONE friction value for the whole city; it only caps braking, a = max(a_cmd, -9 x friction).  So friction can be learnt from
the cap: when an AV brakes as hard as the road allows, its achieved deceleration is the cap, 9 x friction.  The 'patches' here are
patches in TIME, city-wide (the same friction for every road at a given second): a smooth profile 1.0 -> 0.45 -> 0.7 -> 0.3 -> 1.0.
(Spatial patches would need a per-road friction inside xnet/sim.py, which we do not change.)

run:      E copies, half AVs, friction = mu(t) (interpolated inside the scan); every second keep, per slot, the largest deceleration
          applied by an AV moving > 1 m/s (st.acc, 'acceleration applied last step').  No access to the truth.
estimate: over a trailing window W s, pool all (slot, second) decelerations of the fleet; f = the largest; n_sat = how many are within
          3 % of f.  If n_sat >= k the floor is 'seen' and mu_hat = f / 9, else the AV only knows mu >= f / 9 (a lower bound).
          Scored against the true mu(t): error where identified, share of time identified, delay after a step down.
"""
import argparse
import json
import os

import numpy as np

OUT = "results/frictionsense"
KNOTS_T = [0, 250, 300, 500, 550, 700, 750, 850, 900, 1000]
KNOTS_MU = [1.0, 1.0, 0.45, 0.45, 0.7, 0.7, 0.3, 0.3, 1.0, 1.0]


def mu_of(t):
    return np.interp(t, KNOTS_T, KNOTS_MU)


def estimate(dec, W=30, k=3, tol=0.03, a0=9.0, top=1, tol_abs=0.0):
    """dec [T, N] per-second largest deceleration (>= 0, 0 where no AV braked) -> dict of arrays [T]: floor f, count at the floor, mu_hat (nan = not seen), lower bound."""
    T = dec.shape[0]
    f = np.zeros(T); nsat = np.zeros(T, int)
    for t in range(T):
        w = dec[max(0, t - W + 1):t + 1]
        v = np.sort(w[w > 0])[::-1] if (w > 0).any() else np.zeros(1)
        f[t] = v[:top].mean()  # top=1: the largest; top>1: robust to measurement noise
        nsat[t] = int((w >= f[t] - max(tol * f[t], tol_abs)).sum()) if f[t] > 0 else 0
    mu_hat = np.where(nsat >= k, np.minimum(f / a0, 1.0), np.nan)
    return {"floor": f, "n_at_floor": nsat, "mu_hat": mu_hat, "lower_bound": np.minimum(f / a0, 1.0)}


def score(est, mu_true, burn=60, tol_mu=0.1):
    """Share of time identified, median |error| and share of identified times within tol_mu, after a burn-in of `burn` s."""
    m = est["mu_hat"][burn:]; tr = mu_true[burn:]
    ok = ~np.isnan(m)
    err = np.abs(m[ok] - tr[ok])
    return {"identified": float(ok.mean()), "median_abs_err": float(np.median(err)) if ok.any() else None,
            "within_tol": float((err <= tol_mu).mean()) if ok.any() else None,
            "lower_bound_valid": float((est["lower_bound"][burn:] <= tr + 1e-6).mean())}


def noise_table(dec, tt, rng=np.random.default_rng(1)):
    """Add Gaussian accelerometer noise to the recorded decelerations; naive (largest) vs robust (mean of the top 5, band 2 sigma) estimator."""
    mu_true = mu_of(tt); E = dec.shape[1]
    out = {}
    for sig in (0.0, 0.2, 0.5):
        nd = np.where(dec > 0, np.maximum(dec + sig * rng.standard_normal(dec.shape), 0), 0).astype(np.float32)
        for name, kw in (("naive", {}), ("robust", {"top": 5, "tol_abs": 2 * sig})):
            ests = [estimate(nd[:, e], **kw) for e in range(E)]
            row = {}
            for band, m in (("dry", mu_true >= 0.9), ("slick", mu_true < 0.6)):
                m = m.copy(); m[:60] = False
                ids = np.concatenate([~np.isnan(x["mu_hat"][m]) for x in ests]); er = np.concatenate([x["mu_hat"][m] - mu_true[m] for x in ests])
                row[band] = {"identified": float(ids.mean()), "median_abs_err": float(np.median(np.abs(er[ids]))) if ids.any() else None,
                             "median_bias": float(np.median(er[ids])) if ids.any() else None}
            out[f"sigma={sig}|{name}"] = row
    return out


def run(E=8, seconds=1000, area="oakland", p_av=0.5):
    import jax
    import jax.numpy as jnp
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    beh = C.calibrated(p_av=p_av)
    kt, km = jnp.asarray(KNOTS_T, jnp.float32), jnp.asarray(KNOTS_MU, jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(0), E)
    st = jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, 1024)
    n = int(round(1.0 / X.DT))

    @jax.jit
    def sec(s):
        def one(si):
            def body(c, _):
                s_, mn, mf = c
                b = beh._replace(friction=jnp.interp(s_.t, kt, km))
                s_ = X.step(s_, J, b, nh)
                ok = s_.alive & s_.av & (s_.v > 1.0) & (s_.wreck_t <= 0)
                free = ok & (s_.stop_t <= 0)  # not in the scripted 'halted' branch (stop_t > 0), where xnet sets acc = cap whatever the demand
                return (s_, jnp.maximum(mn, jnp.where(ok, -s_.acc, 0.0)), jnp.maximum(mf, jnp.where(free, -s_.acc, 0.0))), None
            (s2, mn, mf), _ = jax.lax.scan(body, (si, jnp.zeros_like(si.v), jnp.zeros_like(si.v)), None, length=n)
            return s2, jnp.stack([mn, mf])
        return jax.vmap(one)(s)
    dec = np.zeros((seconds, 2, E, st.v.shape[1]), np.float32)
    tt = np.zeros(seconds)
    for i in range(seconds):
        st, mn = sec(st)
        dec[i] = np.asarray(mn).transpose(1, 0, 2); tt[i] = float(st.t[0])
    return dec, tt


def analyse(dec, tt, rng=np.random.default_rng(0)):
    mu_true = mu_of(tt)
    E = dec.shape[1]
    out = {"t": tt.tolist(), "mu_true": mu_true.tolist(), "copies": []}
    for e in range(E):
        est = estimate(dec[:, e])
        out["copies"].append({"mu_hat": [None if np.isnan(v) else float(v) for v in est["mu_hat"]], "lower_bound": est["lower_bound"].tolist(),
                              "n_at_floor": est["n_at_floor"].tolist(), "score": score(est, mu_true)})
    # by true-friction band
    est0 = [estimate(dec[:, e]) for e in range(E)]
    bands = {"dry (mu >= 0.9)": mu_true >= 0.9, "0.6-0.9": (mu_true >= 0.6) & (mu_true < 0.9), "slick (mu < 0.6)": mu_true < 0.6}
    out["bands"] = {}
    for name, m in bands.items():
        m = m.copy(); m[:60] = False
        ids = np.concatenate([~np.isnan(x["mu_hat"][m]) for x in est0]); er = np.concatenate([np.abs(x["mu_hat"][m] - mu_true[m]) for x in est0])
        out["bands"][name] = {"seconds": int(m.sum()), "identified": float(ids.mean()), "median_abs_err": float(np.nanmedian(er)) if ids.any() else None,
                              "within_0.1": float((er[ids] <= 0.1).mean()) if ids.any() else None}
    # how many AVs are needed: subsample slots that were ever an AV braking
    slots = np.nonzero((dec > 0).any((0, 1)))[0]
    out["fleet"] = {}
    for k in (1, 3, 10, 30, len(slots)):
        ids, ers = [], []
        for e in range(E):
            for _ in range(3 if k < len(slots) else 1):
                sel = rng.choice(slots, size=min(k, len(slots)), replace=False)
                est = estimate(dec[:, e][:, sel])
                sc = score(est, mu_true); ids.append(sc["identified"]); ers.append(sc["median_abs_err"])
        out["fleet"][str(k)] = {"identified": float(np.mean(ids)), "median_abs_err": float(np.nanmedian([x for x in ers if x is not None])) if any(x is not None for x in ers) else None}
    # delay after the step down at t=250..300 and the step to 0.3 at 700..750 (first time identified within 0.1 of truth after the ramp ends)
    out["delays"] = {}
    for name, t_end, lo in (("to 0.45", 300, 0.45), ("to 0.3", 750, 0.3)):
        ds = []
        for e in range(E):
            m = np.array([np.nan if v is None else v for v in out["copies"][e]["mu_hat"]])
            idx = np.nonzero((tt >= t_end) & (np.abs(m - lo) <= 0.1))[0]
            ds.append(float(tt[idx[0]] - t_end) if len(idx) else None)
        out["delays"][name] = ds
    return out


def plot(gif=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    BG, FG, GR = "#0d1117", "#e6edf3", "#30363d"
    R = json.load(open(f"{OUT}/frictionsense.json"))
    z = np.load(f"{OUT}/events.npz")
    tt, mu = np.array(R["t"]), np.array(R["mu_true"])
    ev_t, ev_d = z["t"], z["d"]
    e0 = 0
    mh = np.array([np.nan if v is None else v for v in R["copies"][e0]["mu_hat"]], float)
    lb = np.array(R["copies"][e0]["lower_bound"])
    fig, ax = plt.subplots(2, 1, figsize=(8.4, 6.2), facecolor=BG, sharex=True, gridspec_kw={"height_ratios": [1.1, 1]})
    for a in ax:
        a.set_facecolor(BG); a.tick_params(colors=FG); [s.set_color(GR) for s in a.spines.values()]
        a.xaxis.label.set_color(FG); a.yaxis.label.set_color(FG); a.title.set_color(FG)
    sc = ax[1].scatter(ev_t, ev_d, s=5, color="#56b4e9", alpha=.5, lw=0)
    l_floor, = ax[1].plot(tt, 9 * mu, color="#e69f00", lw=2)
    ax[1].text(tt[-1], 9.6, "braking limit 9 x friction (hidden from the AVs)", color="#e69f00", ha="right", fontsize=9)
    ax[1].text(10, 9.75, "dots: hardest braking of each AV, each second", color="#56b4e9", fontsize=9)
    ax[1].set_ylim(0, 10.5); ax[1].set_ylabel("deceleration (m/s^2)"); ax[1].set_xlabel("time (s)")
    l_true, = ax[0].plot(tt, mu, color="#e69f00", lw=2); fill = ax[0].fill_between(tt, lb, 0, color="#8b949e", alpha=.25, lw=0)
    l_est, = ax[0].plot(tt, mh, color="#009e73", lw=2.6)
    ax[0].text(520, 0.97, "true friction", color="#e69f00", fontsize=10); ax[0].text(820, 0.12, "AVs' estimate (green) and lower bound (grey)", color="#009e73", fontsize=9, ha="center")
    ax[0].set_ylim(0, 1.1); ax[0].set_ylabel("friction"); ax[0].set_title("Can AVs learn the road friction from their own braking?")
    fig.tight_layout(); fig.savefig("docs/img/frictionsense.png", dpi=130, facecolor=BG)
    if not gif:
        return
    # animation: reveal left to right with a moving clip rectangle
    from matplotlib.patches import Rectangle
    cur = [ax[0].axvline(0, color=FG, lw=1, alpha=.6), ax[1].axvline(0, color=FG, lw=1, alpha=.6)]
    frames = list(range(0, len(tt), 20)) + [len(tt) - 1]
    groups = ((ax[0], [l_true, l_est, fill]), (ax[1], [sc, l_floor]))

    def upd(i):
        x = tt[i]
        for a, arts in groups:
            r = Rectangle((-50, -1), x + 50, 20, transform=a.transData)
            for art in arts:
                art.set_clip_path(r)
        for c in cur:
            c.set_xdata([x, x])
        return []
    ani = FuncAnimation(fig, upd, frames=frames, blit=False)
    ani.save("docs/img/frictionsense.gif", writer=PillowWriter(fps=10), dpi=70, savefig_kwargs={"facecolor": BG})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="?", default="run", choices=["run", "plot"])
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == "plot":
        return plot()
    dec, tt = run(**(dict(E=2, seconds=60) if a.quick else {}))
    if a.quick:
        print("max decel", dec.max(), "events", int((dec > 0).sum()), "free", int((dec[:, 1] > 0).sum())); return
    res = analyse(dec[:, 1], tt)  # main result: halted-branch decelerations excluded
    res["with_halted"] = analyse(dec[:, 0], tt)
    res["noise"] = noise_table(dec[:, 1], tt)
    res["share_of_halted_floor_events"] = float(((dec[:, 0] > 0.5) & (dec[:, 1] < 0.5)).sum() / max((dec[:, 0] > 0.5).sum(), 1))
    json.dump(res, open(f"{OUT}/frictionsense.json", "w"))
    e, s = np.nonzero(dec[:, 1, 0] > 0.5)
    np.savez_compressed(f"{OUT}/events.npz", t=tt[e], d=dec[e, 1, 0, s])
    for nm, r in (("excluding halted", res), ("including halted", res["with_halted"])):
        print(nm, json.dumps({k: r[k] for k in ("bands", "fleet")}))
    print(res["share_of_halted_floor_events"]); print(json.dumps(res["noise"], indent=0))


if __name__ == "__main__":
    main()
