"""How many AVs until a city's crashes halve? Sweep the share of AVs from 0 to 100 % in Oakland and plot crashes, throughput and
speed against it, with 95 % intervals.

    python examples/av_share_sweep.py            # ~7 min on 4 CPU cores -> docs/img/example_av_share.png, results/examples/av_share.json
    python examples/av_share_sweep.py --quick    # a 20 s smoke version

The humans are calibrated to published crash rates by crash type, the AVs are the rule-based AV under its safety envelope.
Two things make crash curves expensive, and the design follows from them:
  * crashes are rare: ~15 per 1,000 vehicle-hours in flowing traffic, so a share needs ~2,500 vehicle-hours for ~40 crashes and a
    +-30 % interval;
  * with fixed routes the city fills up over half an hour (speed 22 -> 6 km/h) and congested traffic barely crashes.
So each share is many short copies (2 min warm-up, 5 min measured, while traffic still flows), one compiled run per share.
"""
import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from examples import _common as C  # noqa: E402
except ImportError:
    import _common as C  # noqa: E402

BG, INK, INK2, AV, HUM = "#0d1117", "#e6edf3", "#8b949e", "#2f81f7", "#f0883e"


def humans(risk, p_av):
    """Calibrated human drivers (risk 1), or the stress world with every human mistake rate x risk (avlab.adversary.world)."""
    if risk == 1:
        return C.calibrated(p_av=p_av)
    from avlab import adversary as AD
    return AD.world(risk)._replace(p_av=p_av)


def plot(res, area, risk, copies, seconds, headline, out):
    x = [r["av_share"] * 100 for r in res]
    base = res[0]["crashes_per_1000_vh"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=BG)
    for ax, title in zip(axs, ("crashes per 1,000 vehicle-hours (95 % interval)", "trips completed per hour (per copy)", "mean speed, km/h (sd over copies)")):
        ax.set_facecolor(BG)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(color="#21262d")
        ax.set_xlabel("share of AVs, %", color=INK2, fontsize=9)
        for sp in ax.spines.values():
            sp.set_color("#30363d")
    axs[0].fill_between(x, [r["ci"][0] for r in res], [r["ci"][1] for r in res], color=AV, alpha=0.2, lw=0)
    axs[0].plot(x, [r["crashes_per_1000_vh"] for r in res], color=AV, marker="o")
    for xi, r in zip(x, res):
        axs[0].annotate(f"{r['crashes']:.0f}", (xi, r["crashes_per_1000_vh"]), textcoords="offset points", xytext=(0, 8), ha="center", color=INK2, fontsize=8)
    axs[0].axhline(base / 2, color=INK2, ls=":", lw=1)
    axs[0].text(x[0], base / 2, " half of 0 %", color=INK2, fontsize=8, va="top", ha="left")
    axs[0].set_ylim(0, None)
    axs[0].text(0.01, 0.02, "labels: crashes counted", transform=axs[0].transAxes, color=INK2, fontsize=8)
    tp = [r["trips_per_hour"] for r in res]
    axs[1].plot(x, tp, color=AV, marker="o")
    axs[1].set_ylim(0, max(tp) * 1.2)
    axs[2].errorbar(x, [r["speed_kmh"] for r in res], yerr=[r["speed_sd"] for r in res], color=AV, marker="o", capsize=3)
    axs[2].set_ylim(0, max(r["speed_kmh"] + r["speed_sd"] for r in res) * 1.2)
    world = "calibrated humans" if risk == 1 else f"stress world: human mistakes x{risk:g}"
    fig.suptitle(f"{area}, {world}: {copies} copies x {seconds / 60:.0f} simulated min per AV share (traffic still flowing). {headline}.",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=BG)
    plt.close(fig)


def main(shares=(0.0, 0.25, 0.5, 0.75, 1.0), copies=80, warm_s=120.0, seconds=300.0, area="oakland", risk=1.0, out=None, name=None,
         seed=None):
    t0 = time.time()
    name = name or ("av_share" if risk == 1 else f"av_share_risk{risk:g}")
    out = out or f"docs/img/example_{name}.png"
    rows = []
    for i, s in enumerate(shares):  # one compiled run per share keeps memory bounded; every copy has its own random traffic
        # seed=None keeps the published results reproducible (seeded by position). Give a seed for a new, independent sweep: then
        # each share's seed depends on its value and the seed, so sweeps over different grids never silently reuse random numbers
        sd = 1000 * i if seed is None else 100_000 * (seed + 1) + int(round(1000 * s))
        rows += C.per_copy(C.run_copies(area, [humans(risk, s)] * copies, warm_s=warm_s, seconds=seconds, seed=sd))
        print(f"  share {s:.0%} done ({time.time() - t0:.0f} s)", flush=True)
    res = []
    for i, s in enumerate(shares):
        r = rows[i * copies:(i + 1) * copies]
        vh = sum(x["vehicle_hours"] for x in r)
        k = sum(x["crashes"] for x in r)
        hh = sum(x["human_hours"] for x in r)
        kh = sum(x["human_only_crashes"] for x in r)
        lo, hi = C.poisson_ci(k, vh / 1000)
        res.append({"av_share": s, "vehicle_hours": vh, "crashes": k, "crashes_per_1000_vh": 1000 * k / vh, "ci": (lo, hi),
                    "human_only_per_1000_human_h": 1000 * kh / hh if hh > 1 else float("nan"),
                    "trips_per_hour": sum(x["trips_finished"] for x in r) / (copies * seconds / 3600),
                    "speed_kmh": float(np.mean([x["mean_speed_kmh"] for x in r])),
                    "speed_sd": float(np.std([x["mean_speed_kmh"] for x in r]))})
    base = res[0]["crashes_per_1000_vh"]
    half = next((x["av_share"] for x in res if x["crashes_per_1000_vh"] <= base / 2), None)
    if half is not None and half > 0:  # interpolate between the last share above half and the first at or below
        j = next(i for i, x in enumerate(res) if x["av_share"] == half)
        a, b = res[j - 1], res[j]
        f = (a["crashes_per_1000_vh"] - base / 2) / max(a["crashes_per_1000_vh"] - b["crashes_per_1000_vh"], 1e-9)
        half = a["av_share"] + f * (b["av_share"] - a["av_share"])
    headline = (f"crashes halve at about {half:.0%} AVs" if half is not None else "crashes never halve in this sweep")
    plot(res, area, risk, copies, seconds, headline, out)
    C.save(name, {"area": area, "risk": risk, "copies": copies, "seconds": seconds, "rows": res, "headline": headline, "wall_s": time.time() - t0})
    print(headline)
    for r in res:
        print(f"  {r['av_share']:4.0%}  crashes/1000 vh {r['crashes_per_1000_vh']:5.2f} [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] ({r['crashes']:.0f} crashes, "
              f"{r['vehicle_hours']:.0f} vh)  trips/h {r['trips_per_hour']:6.0f}  {r['speed_kmh']:.1f} km/h")
    print(f"{time.time() - t0:.0f} s")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw the figure from results/examples/av_share.json")
    ap.add_argument("--risk", type=float, default=1.0, help="human mistake rates x this (avlab.adversary.world); 1 = calibrated")
    a = ap.parse_args()
    if a.replot:
        import json
        r = json.load(open("results/examples/av_share.json"))
        plot(r["rows"], r["area"], r["risk"], r["copies"], r["seconds"], r["headline"], "docs/img/example_av_share.png")
    elif a.quick:
        main(shares=(0.0, 1.0), copies=2, warm_s=30, seconds=60, risk=a.risk, out="/tmp/example_av_share_quick.png", name="av_share_quick")
    else:
        main(risk=a.risk)
