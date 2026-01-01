"""Green wave: retime every traffic light in a city with the cross-entropy method, then check on fresh traffic that it really helps
(and that it is not bought with crashes).

    python -m avlab.greenwave           # Oakland's 34 signals, 12 generations x 24 timings (~12 min, 4 CPU cores) -> results/greenwave/

The parameters are each signalised junction's offset (where in the 70 s cycle its green starts, 0-70 s) and split (how long the first
axis is green, 15-50 s). A timing is scored by the network's mean speed (vehicle-km / vehicle-hours) over 4 min, from the same warmed
traffic and the same random numbers for every candidate (common random numbers: the comparison is of timings, not of luck). Every
candidate x copy runs in one vmapped batch: the signal arrays are batched, the rest of the network is shared, and the simulator itself
is unchanged.

Cross-entropy method: keep a Gaussian per parameter; each generation draw 24 timings, score them, refit the Gaussian to the best 6
(with smoothing), repeat. Then the honest part: the best timing and today's timing are run on 32 *new* copies (new seeds, 10 min) and
compared with paired statistics, so an optimiser that only learned its training traffic shows up as no gain.
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from xnet import sim as X

OUT = "results/greenwave"
KEYS = ("node_offset", "node_g0")


def make_runner(net, beh, n_slots=1024):
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    axes = {k: (0 if k in KEYS else None) for k in J}

    def one(s, jt):
        return X.step(s, jt, beh, nh)

    @jax.jit
    def chunk(st, offs, g0s):
        """st: [B, ...] states; offs, g0s: [B, nodes] timings. 60 s."""
        def body(s, _):
            def f(si, o, g):
                return one(si, {**J, "node_offset": o, "node_g0": g})
            return jax.vmap(f)(s, offs, g0s), None
        return jax.lax.scan(body, st, None, length=int(60 / X.DT))[0]

    def reset(keys):
        return jax.vmap(X.reset, in_axes=(0, None, None))(keys, J, n_slots)
    return chunk, reset


def score(m0, m1):
    """Per run: mean speed km/h, trips finished, crashes."""
    d = np.asarray(m1) - np.asarray(m0)
    c = {k: d[:, i] for i, k in enumerate(X.METRICS)}
    return 3.6 * c["veh_m"] / np.maximum(c["veh_s"], 1), c["finished"], sum(c[k] for k in X.CRASH_NAMES), c["veh_s"] / 3600


def optimise(area="oakland", gens=12, pop=24, elite=6, copies=4, warm_s=120.0, eval_s=240.0, seed=0, demand=1.2, splits=True):
    """splits=False: search the offsets only. Every green, yellow and red then lasts exactly as long as today; only the lights'
    coordination changes, so the search cannot buy speed with shorter clearance for one axis."""
    from av_world import easy
    from examples import _common as C
    net, geo, info = easy._build(area)
    beh = C.calibrated(p_av=0.0, demand=demand)
    chunk, reset = make_runner(net, beh)
    sig = np.nonzero(np.asarray(net["node_signal"]) > 0)[0]
    cyc = float(np.asarray(net["node_cycle"])[sig][0])
    off0, g00 = np.asarray(net["node_offset"], np.float32), np.asarray(net["node_g0"], np.float32)
    # one warmed start (today's timing), copied for every candidate: common random numbers
    st0 = reset(jax.random.split(jax.random.PRNGKey(seed), copies))
    base_o, base_g = jnp.tile(off0, (copies, 1)), jnp.tile(g00, (copies, 1))
    for _ in range(int(warm_s / 60)):
        st0 = chunk(st0, base_o, base_g)
    mu = np.r_[off0[sig], g00[sig]].astype(float)
    sd = np.r_[np.full(len(sig), cyc / 4), np.full(len(sig), 6.0 if splits else 0.0)]
    lo, hi = np.r_[np.full(len(sig), -1e9), np.full(len(sig), 15.0)], np.r_[np.full(len(sig), 1e9), np.full(len(sig), cyc - 20.0)]
    rng = np.random.default_rng(seed)
    hist, best = [], (-np.inf, mu.copy())

    def timings(cands):
        O, G = np.tile(off0, (len(cands), 1)), np.tile(g00, (len(cands), 1))
        O[:, sig] = np.mod(cands[:, :len(sig)], cyc)
        G[:, sig] = np.clip(cands[:, len(sig):], 15.0, cyc - 20.0)
        return O, G

    def evaluate(cands):
        O, G = timings(cands)
        B = len(cands) * copies
        st = jax.tree.map(lambda a: jnp.tile(a, (len(cands),) + (1,) * (a.ndim - 1)), st0)
        Ob, Gb = jnp.asarray(np.repeat(O, copies, 0)), jnp.asarray(np.repeat(G, copies, 0))
        m0 = st.m
        for _ in range(int(eval_s / 60)):
            st = chunk(st, Ob, Gb)
        sp, fin, cr, vh = score(m0, st.m)
        return sp.reshape(len(cands), copies).mean(1), cr.reshape(len(cands), copies).sum(1), B

    t0 = time.time()
    base_speed = float(evaluate(mu[None])[0][0])
    for g in range(gens):
        cands = np.clip(mu + sd * rng.normal(size=(pop, len(mu))), lo, hi)
        cands[0] = mu  # the current mean is always scored
        sp, cr, _ = evaluate(cands)
        order = np.argsort(-sp)
        el = cands[order[:elite]]
        mu = 0.7 * el.mean(0) + 0.3 * mu
        sd = np.maximum(0.7 * el.std(0) + 0.3 * sd, np.r_[np.full(len(sig), 1.0), np.full(len(sig), 0.5 if splits else 0.0)])
        if sp[order[0]] > best[0]:
            best = (float(sp[order[0]]), cands[order[0]].copy())
        hist.append({"gen": g, "best_kmh": float(sp[order[0]]), "mean_kmh": float(sp.mean()), "today_kmh": base_speed, "wall_s": round(time.time() - t0)})
        print(f"  gen {g:2d}: best {sp[order[0]]:.2f} km/h, population mean {sp.mean():.2f} (today's timing {base_speed:.2f}) ({time.time() - t0:.0f} s)", flush=True)
    return {"net": net, "beh": beh, "sig": sig, "timings": timings, "best": best, "history": hist, "chunk": chunk, "reset": reset, "mu": mu}


def validate(opt, copies=32, warm_s=120.0, seconds=600.0, seed=12345, O=None, G=None):
    """Today's timing vs the optimised one on new seeds, paired by copy (same seed, same demand)."""
    chunk, reset, net = opt["chunk"], opt["reset"], opt["net"]
    if O is None:
        O, G = opt["timings"](opt["best"][1][None])
    off0, g00 = np.asarray(net["node_offset"], np.float32), np.asarray(net["node_g0"], np.float32)
    res = {}
    for name, (o, g) in (("today", (off0, g00)), ("optimised", (O[0], G[0]))):
        st = reset(jax.random.split(jax.random.PRNGKey(seed), copies))
        ob, gb = jnp.tile(jnp.asarray(o), (copies, 1)), jnp.tile(jnp.asarray(g), (copies, 1))
        for _ in range(int(warm_s / 60)):
            st = chunk(st, ob, gb)
        m0 = st.m
        for _ in range(int(seconds / 60)):
            st = chunk(st, ob, gb)
        res[name] = score(m0, st.m)
    from scipy import stats
    d_speed = res["optimised"][0] - res["today"][0]
    d_trips = res["optimised"][1] - res["today"][1]
    k1, k0 = res["optimised"][2].sum(), res["today"][2].sum()
    t1, t0 = res["optimised"][3].sum(), res["today"][3].sum()
    from examples.hypotheses import rate_ratio
    rr, ci, p = rate_ratio(k1, t1, k0, t0)
    return {"speed_today_kmh": float(res["today"][0].mean()), "speed_optimised_kmh": float(res["optimised"][0].mean()),
            "speed_gain_pct": float(100 * d_speed.mean() / res["today"][0].mean()),
            "speed_gain_ci_kmh": [float(d_speed.mean() - 1.96 * d_speed.std(ddof=1) / np.sqrt(copies)), float(d_speed.mean() + 1.96 * d_speed.std(ddof=1) / np.sqrt(copies))],
            "speed_wilcoxon_p": float(stats.wilcoxon(d_speed).pvalue), "copies_better": int((d_speed > 0).sum()),
            "trips_per_copy_today": float(res["today"][1].mean()), "trips_gain_pct": float(100 * d_trips.mean() / res["today"][1].mean()),
            "crashes_today": float(k0), "crashes_optimised": float(k1), "crash_rate_ratio": rr, "crash_rr_ci": ci, "copies": copies, "seconds": seconds,
            "per_copy_speed_gain_kmh": d_speed.round(4).tolist()}


def plot(opt, val, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BG, INK, INK2 = "#0d1117", "#e6edf3", "#8b949e"
    h = opt["history"]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(14, 4.2), facecolor=BG)
    for ax in (a0, a1):
        ax.set_facecolor(BG)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color="#21262d")
        for sp in ax.spines.values():
            sp.set_color("#30363d")
    g = [r["gen"] for r in h]
    a0.plot(g, [r["best_kmh"] for r in h], color="#3987e5", marker="o", label="best timing of the generation")
    a0.plot(g, [r["mean_kmh"] for r in h], color=INK2, ls="--", label="population mean")
    a0.axhline(h[0]["today_kmh"], color="#f0883e", ls=":", label="today's timing")
    a0.set_xlabel("generation", color=INK2)
    a0.set_ylabel("network mean speed, km/h (training traffic)", color=INK2, fontsize=9)
    a0.legend(facecolor=BG, edgecolor="#30363d", labelcolor=INK, fontsize=8)
    a0.set_title(opt.get("title", "Cross-entropy search over 34 offsets and 34 splits"), color=INK, fontsize=10, loc="left")
    d = np.asarray(val.get("per_copy_speed_gain_kmh", []))
    lo, hi = val["speed_gain_ci_kmh"]
    a1.axhline(0, color=INK2, lw=1)
    if len(d):
        jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(d))
        a1.scatter(jit, d, s=28, color=np.where(d > 0, "#3987e5", "#f0883e"), edgecolor=BG, linewidth=1, zorder=3)
    a1.errorbar([0.35], [(lo + hi) / 2], yerr=[[(hi - lo) / 2]], fmt="o", color=INK, capsize=5, zorder=4)
    a1.text(0.42, (lo + hi) / 2, f"mean {val['speed_gain_pct']:+.1f} %\n[{lo:+.2f}, {hi:+.2f}] km/h", color=INK, fontsize=9, va="center")
    a1.set_xlim(-0.4, 0.9)
    a1.set_xticks([])
    a1.set_ylabel("speed gain on new traffic, km/h (one dot per copy)", color=INK2, fontsize=9)
    a1.set_title(f"Validated on {val['copies']} new copies x {val['seconds'] / 60:.0f} min: {val['copies_better']} faster; "
                 f"crashes {val['crashes_today']:.0f} -> {val['crashes_optimised']:.0f} (x{val['crash_rate_ratio']:.2f}, "
                 f"[{val['crash_rr_ci'][0]:.2f}, {val['crash_rr_ci'][1]:.2f}])", color=INK, fontsize=9.5, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def main(quick=False, splits=True, copies=64):
    os.makedirs(OUT, exist_ok=True)
    kw = dict(gens=2, pop=4, elite=2, copies=1, warm_s=60.0, eval_s=60.0) if quick else {}
    t0 = time.time()
    opt = optimise(splits=splits, **kw)
    val = validate(opt, **(dict(copies=2, warm_s=60.0, seconds=60.0) if quick else dict(copies=copies)))
    O, G = opt["timings"](opt["best"][1][None])
    name = "quick" if quick else ("greenwave" if splits else "greenwave_offsets")
    res = {"history": opt["history"], "validation": val, "signals": opt["sig"].tolist(), "offsets": O[0][opt["sig"]].tolist(),
           "splits": G[0][opt["sig"]].tolist(), "search": "offsets and splits" if splits else "offsets only", "wall_s": time.time() - t0}
    json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)
    if not quick:
        opt["title"] = "Cross-entropy search over 34 offsets" + (" and 34 splits" if splits else " (splits unchanged)")
        plot(opt, val, f"docs/img/{name}.png")
    print(json.dumps({k: v for k, v in val.items() if k != "per_copy_speed_gain_kmh"}, indent=1, default=float))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--offsets-only", action="store_true", help="keep every split as today; search the offsets (coordination) only")
    ap.add_argument("--copies", type=int, default=64, help="new copies for the validation")
    ap.add_argument("--revalidate", action="store_true", help="re-run the validation of the saved timing and redraw the figure")
    a = ap.parse_args()
    name = "greenwave_offsets" if a.offsets_only else "greenwave"
    if a.revalidate:
        res = json.load(open(f"{OUT}/{name}.json"))
        from av_world import easy
        from examples import _common as C
        net, _, _ = easy._build("oakland")
        chunk, reset = make_runner(net, C.calibrated(p_av=0.0, demand=1.2))
        sig = np.asarray(res["signals"])
        O, G = np.asarray(net["node_offset"], np.float32).copy(), np.asarray(net["node_g0"], np.float32).copy()
        O[sig], G[sig] = res["offsets"], res["splits"]
        res["validation"] = validate({"chunk": chunk, "reset": reset, "net": net}, copies=a.copies, O=O[None], G=G[None])
        json.dump(res, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)
        title = "Cross-entropy search over 34 offsets" + (" (splits unchanged)" if a.offsets_only else " and 34 splits")
        plot({"history": res["history"], "title": title}, res["validation"], f"docs/img/{name}.png")
        print(json.dumps({k: v for k, v in res["validation"].items() if k != "per_copy_speed_gain_kmh"}, default=float))
    else:
        main(a.quick, splits=not a.offsets_only, copies=a.copies)
