"""Does the crash rate follow a power of speed? Scale every speed limit of Oakland and count crashes (Nilsson's power model, tested).

    python -m avlab.speedlimit            # 5 limits x 32 copies x 10 simulated min  -> results/speedlimit/speedlimit.json
    python -m avlab.speedlimit --quick    # a smoke run
    python -m avlab.speedlimit --plot     # figure -> docs/img/speedlimit.png

    for scale in 0.7 0.85 1.0 1.15 1.3:  net["link_v"] *= scale   (a post-hoc change to the net dict; xnet/ is not touched)
        32 copies, calibrated humans, warm 120 s, count 600 s;  record crashes, vehicle-hours, vehicle-km
    fit  log(rate) = a + b log(scale)  and  log(rate) = a + b log(mean speed)  (quasi-Poisson, offset log vehicle-hours)
"""
import argparse
import json
import os

import numpy as np

import jax
import jax.numpy as jnp

OUT = "results/speedlimit"
SCALES = (0.7, 0.85, 1.0, 1.15, 1.3)


def poisson_fit(y, t, x):
    """Quasi-Poisson regression log E[y] = log t + a + b x by IRLS. Returns (b, se_b, dispersion)."""
    y, t, x = map(lambda v: np.asarray(v, float), (y, t, x))
    X = np.stack([np.ones_like(x), x - x.mean()], 1)
    beta = np.array([np.log(max(y.sum(), 1) / t.sum()), 0.0])
    for _ in range(50):
        mu = t * np.exp(X @ beta)
        W = mu
        z = X @ beta + (y - mu) / np.maximum(mu, 1e-9)
        beta_new = np.linalg.solve(X.T @ (W[:, None] * X), X.T @ (W * z))
        if np.abs(beta_new - beta).max() < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    mu = t * np.exp(X @ beta)
    phi = max(((y - mu) ** 2 / np.maximum(mu, 1e-9)).sum() / (len(y) - 2), 1.0)
    cov = np.linalg.inv(X.T @ (mu[:, None] * X)) * phi
    return float(beta[1]), float(np.sqrt(cov[1, 1])), float(phi)


def run(E=32, warm_s=120.0, seconds=600.0, seed=0, scales=SCALES, area="oakland", p_av=0.0):
    from examples import _common as C
    from xnet import network as NW
    from xnet import sim as X
    net, geo, info = NW.build(area)
    nh = jnp.asarray(net["nh"])
    beh = C.calibrated(p_av=p_av) if p_av else C.calibrated()   # p_av=0 reproduces the saved run exactly
    n = int(round(20.0 / X.DT))

    @jax.jit
    def chunk(s, J):
        return jax.lax.scan(lambda c, _: (jax.vmap(lambda si: X.step(si, J, beh, nh))(c), None), s, None, length=n)[0]

    res = {"area": area, "copies": E, "seconds": seconds, "p_av": p_av, "scales": {}}
    for sc in scales:
        J = X.to_jnp(dict(net, link_v=(np.asarray(net["link_v"]) * sc).astype(np.float32)))
        st = jax.vmap(X.reset, in_axes=(0, None, None))(jax.random.split(jax.random.PRNGKey(seed), E), J, 1024)
        for _ in range(int(round(warm_s / 20.0))):
            st = chunk(st, J)
        m0 = np.asarray(st.m)
        for _ in range(int(round(seconds / 20.0))):
            st = chunk(st, J)
        m = np.asarray(st.m) - m0
        d = {k: m[:, i] for i, k in enumerate(X.METRICS)}
        crashes = sum(d[k] for k in X.CRASH_NAMES)
        res["scales"][str(sc)] = {"crashes": crashes.tolist(), "veh_s": d["veh_s"].tolist(), "veh_m": d["veh_m"].tolist(),
                                  "finished": d["finished"].tolist(), "by_type": {k: float(d[k].sum()) for k in X.CRASH_NAMES}}
        print(f"  limit x{sc}: {crashes.sum():.0f} crashes in {d['veh_s'].sum() / 3600:.0f} vehicle-h, "
              f"mean speed {3.6 * d['veh_m'].sum() / d['veh_s'].sum():.1f} km/h, trips {d['finished'].sum():.0f}", flush=True)
    return res


def summarize(res):
    sc = sorted(res["scales"], key=float)
    y, t, vm, x, sp = [], [], [], [], []
    for s in sc:
        r = res["scales"][s]
        y += r["crashes"]; t += [v / 3600 for v in r["veh_s"]]; vm += r["veh_m"]; x += [np.log(float(s))] * len(r["crashes"])
        sp += [m / max(v, 1e-9) for m, v in zip(r["veh_m"], r["veh_s"])]
    y, t, x, sp = map(np.array, (y, t, x, sp))
    b, se, phi = poisson_fit(y, t, x)
    b2, se2, phi2 = poisson_fit(y, t, np.log(np.maximum(sp, 1e-3)))
    per = {}
    for s in sc:
        r = res["scales"][s]
        k, h = sum(r["crashes"]), sum(r["veh_s"]) / 3600
        per[s] = {"crashes": k, "vehicle_hours": h, "rate_per_1000_vh": 1000 * k / h, "mean_speed_kmh": 3.6 * sum(r["veh_m"]) / sum(r["veh_s"]),
                  "trips": sum(r["finished"])}
    return {"elasticity_vs_limit": b, "se_limit": se, "dispersion_limit": phi, "elasticity_vs_speed": b2, "se_speed": se2, "per_scale": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--copies", type=int, default=32)
    ap.add_argument("--p-av", type=float, default=0.0, help="share of AVs (rule-based AV under its envelope); output goes to speedlimit_av<pct>.json")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.plot:
        return plot()
    res = run(E=4 if a.quick else a.copies, seconds=60.0 if a.quick else 600.0, scales=(0.7, 1.3) if a.quick else SCALES, p_av=a.p_av)
    res["summary"] = summarize(res)
    print(json.dumps(res["summary"], indent=1))
    if not a.quick:
        json.dump(res, open(f"{OUT}/speedlimit.json" if not a.p_av else f"{OUT}/speedlimit_av{round(100 * a.p_av)}.json", "w"), indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.load(open(f"{OUT}/speedlimit.json"))
    S = res["summary"]
    sc = sorted(S["per_scale"], key=float)
    BG, FG, C1, C2 = "#0d1117", "#e6edf3", "#56b4e9", "#e69f00"
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=BG)
    for a_ in ax:
        a_.set_facecolor(BG); a_.tick_params(colors=FG); [s.set_color("#30363d") for s in a_.spines.values()]
        a_.xaxis.label.set_color(FG); a_.yaxis.label.set_color(FG); a_.title.set_color(FG)
    x = np.array([float(s) for s in sc])
    for s in sc:
        r = res["scales"][s]
        k, h = np.array(r["crashes"]), np.array(r["veh_s"]) / 3600
        rate = 1000 * k.sum() / h.sum()
        # per-copy bootstrap interval
        rng = np.random.default_rng(0)
        bs = [1000 * k[i].sum() / h[i].sum() for i in (rng.integers(0, len(k), len(k)) for _ in range(500))]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        ax[0].errorbar(float(s), rate, yerr=[[rate - lo], [hi - rate]], fmt="o", color=C1, capsize=4)
    b, xs = S["elasticity_vs_limit"], np.linspace(0.65, 1.35, 50)
    r1 = S["per_scale"]["1.0"]["rate_per_1000_vh"]
    ax[0].plot(xs, r1 * xs ** b, color=C1, lw=1.5)
    ax[0].plot(xs, r1 * xs ** 2, color=C2, lw=1.2, ls="--")
    ax[0].text(1.3, r1 * 1.3 ** 2 * 1.02, "power 2", color=C2, ha="right", va="bottom", fontsize=9)
    ax[0].text(1.0, r1 * 3.2, f"fit: power {b:.1f}", color=C1, fontsize=9, ha="right")
    ax[0].set_xlabel("speed limits x"); ax[0].set_ylabel("crashes per 1000 vehicle-hours"); ax[0].set_title("Crash rate against the speed limit")
    ax[1].plot(x, [S["per_scale"][s]["mean_speed_kmh"] for s in sc], "o-", color=C2)
    ax[1].set_xlabel("speed limits x"); ax[1].set_ylabel("mean speed (km/h)"); ax[1].set_title("What the limit buys in speed")
    fig.tight_layout()
    fig.savefig("docs/img/speedlimit.png", dpi=130, facecolor=BG)


if __name__ == "__main__":
    main()
