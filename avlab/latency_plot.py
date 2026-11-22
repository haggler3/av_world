"""GIF and figure for avlab.latency / avlab.latencybench.

    python -m avlab.latency_plot          # docs/img/latency_lane.gif: one scripted lane, clean vs delayed vs compensated
    python -m avlab.latency_plot --bench  # docs/img/latency_bench.png from results/latency/latency.json

The lane: a leader 30 m ahead at 12 m/s brakes at 6 m/s^2 to a stop at t = 4 s. Three AVs run the same IDM policy
(avlab.teb.idm_policy_masked), no envelope, each behind a different avlab.latency.Latency (the code the benchmark uses).
"""
import functools
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avlab.patchyice_plot import BG, FG, GRID, MUTED  # noqa: E402
from avlab.sensornoise_plot import _obs  # noqa: E402

DT, L = 0.2, 4.5
LEADC, REDC = "#8b949e", "#D55E00"
DELAY = 1.4
MODELS = {"clean sensors": ((0.0, False), "#009E73"), f"{DELAY} s delay": ((DELAY, False), "#E69F00"),
          f"{DELAY} s delay + prediction": ((DELAY, True), "#56B4E9")}


@functools.lru_cache(None)
def _fns():
    import jax
    from avlab import latency as LT
    from avlab import teb as T
    pol = jax.jit(lambda o, m: T.idm_policy_masked(o, None, m)[0])
    per = jax.jit(LT.perceive, static_argnums=3)
    return pol, per


def scripted(params=(DELAY, False), T_total=16.0):
    """Per-step rows for one follower behind Latency(*params); stops it at contact."""
    import jax.numpy as jnp
    from avlab import latency as LT
    pol, per = _fns()
    m = LT.Latency(*params)
    lx, lv, x, v, hit, rows = 30.0 + L, 12.0, 0.0, 12.0, None, []
    mem = LT.init_mem_for(m)(1)
    for k in range(int(T_total / DT)):
        lv = max(lv + (-6.0 if k * DT >= 4 else 0.0) * DT, 0.0); lx += lv * DT
        gap = lx - x - L
        o, mem = per(jnp.asarray(_obs(v, 13.0, gap, lv))[None], None, mem, m)
        seen = 50 * float(o[0, 9])
        u = float(pol(o, jnp.ones(1, bool))[0, 0])
        a = 3 * u if u >= 0 else 8 * u
        if hit is None and gap <= 0:
            hit = (k * DT, v - lv)
        if hit is not None:
            a, v = 0.0, lv
        rows.append(dict(t=k * DT, x=x, v=v, a=a, lead=lx, gap=gap, seen=seen, hit=hit))
        if hit is None:
            v = max(v + a * DT, 0.0); x += v * DT
    return rows


def lane_gif(path="docs/img/latency_lane.gif", fps=10, every=1):
    from matplotlib.animation import FuncAnimation, PillowWriter
    names = list(MODELS)
    d = {nm: scripted(MODELS[nm][0]) for nm in names}
    n = len(d[names[0]])
    frames = list(range(0, n, every)) + [n - 1] * 10
    fig = plt.figure(figsize=(7.2, 4.6), facecolor=BG)
    ax = fig.add_axes([0.03, 0.47, 0.94, 0.36]); ax2 = fig.add_axes([0.1, 0.09, 0.86, 0.24])
    fig.text(0.03, 0.94, f"Perception latency: the AV sees its leader {DELAY} s late", color=FG, fontsize=12, weight="bold")
    fig.text(0.03, 0.865, "Leader brakes at 6 m/s² from t = 4 s. Same IDM policy, no envelope. Dashed box = where the AV\n"
             "believes the leader is (delayed, or delayed then predicted forward with its own speed). Real time.",
             color=MUTED, fontsize=7.5)
    t_all = [r["t"] for r in d[names[0]]]

    def draw(k):
        ax.cla(); ax.set_facecolor(BG); ax.set_yticks([]); ax.set_ylim(-0.6, 4.0)
        x0 = d[names[0]][k]["x"] - 10
        ax.set_xlim(x0, x0 + 60)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("position (m)", color=MUTED, fontsize=8, labelpad=1)
        for row, nm in enumerate(reversed(names)):
            r, c, y = d[nm][k], MODELS[nm][1], row * 1.5
            ax.plot([x0, x0 + 60], [y - 0.45, y - 0.45], color=GRID, lw=0.8)
            ax.add_patch(plt.Rectangle((r["lead"] - L, y - 0.3), L, 0.6, color=LEADC, alpha=0.7))
            ax.add_patch(plt.Rectangle((r["x"] - L, y - 0.3), L, 0.6, color=c))
            ax.text(x0 + 1, y + 0.42, nm, color=c, fontsize=8, weight="bold")
            if r["hit"] is None:
                if MODELS[nm][0][0] > 0:
                    ax.add_patch(plt.Rectangle((r["x"] + r["seen"], y - 0.3), L, 0.6, fill=False, ec=c, ls="--", lw=1.2))
            else:
                ax.text(r["lead"] + 1, y + 0.42, f"CONTACT at {r['hit'][1] * 3.6:.0f} km/h", color=REDC, fontsize=8.5,
                        weight="bold")
            ax.text(x0 + 59, y + 0.42, f"gap {r['gap']:5.1f} m  v {r['v']:4.1f} m/s", color=FG, fontsize=7.5, ha="right")
        ax.text(x0 + 30, 3.6, f"t = {t_all[k]:4.1f} s", color=FG, fontsize=9, ha="center", weight="bold")
        ax2.cla(); ax2.set_facecolor(BG)
        for i, nm in enumerate(names):
            ax2.plot(t_all[:k + 1], [max(d[nm][j]["gap"], 0) for j in range(k + 1)], color=MODELS[nm][1], lw=1.6)
            ax2.text(t_all[-1] * 0.98, 27 - 7 * i, nm, color=MODELS[nm][1], fontsize=7.5, ha="right", weight="bold")
        ax2.set_xlim(0, t_all[-1]); ax2.set_ylim(0, 32)
        ax2.set_ylabel("true gap (m)", color=MUTED, fontsize=8); ax2.set_xlabel("time (s)", color=MUTED, fontsize=8)
        ax2.tick_params(colors=MUTED, labelsize=8)
        for s in list(ax.spines.values()) + list(ax2.spines.values()):
            s.set_color(GRID)
    FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72,
                                                                           savefig_kwargs={"facecolor": BG})
    return {nm: d[nm][-1]["hit"] for nm in names}


def bench_fig(src="results/latency/latency.json", path="docs/img/latency_bench.png"):
    res = json.load(open(src))
    cells = res["cells"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor=BG)
    ax.set_facecolor(BG)
    floor = 3e-4
    for comp, c, lab in ((False, "#E69F00", "delayed leader"), (True, "#56B4E9", "delayed + forward prediction")):
        rs = sorted((r for r in cells.values() if r["compensate"] == comp and r["delay_s"] > 0), key=lambda r: r["delay_s"])
        ds = np.array([r["delay_s"] for r in rs]) + (0.02 if comp else -0.02)
        rate = np.array([max(r["rate_per_h"], floor) for r in rs])
        ax.plot(ds, rate, "-", color=c, lw=1.6)
        for dd, r, y in zip(ds, rs, rate):
            ax.plot([dd, dd], [max(r["ci"][0], floor), r["ci"][1]], color=c, lw=2)
            ax.plot(dd, y, "o", ms=7, color=c, mfc=c if r["av_crash"] > 0 else BG, mew=2)
            ax.text(dd + 0.05, y / 2.6, f"{r['av_crash']:.0f}", color=c, fontsize=7.5, va="center")
        ax.text(ds[-1] + 0.1, rate[-1] * 1.6, lab, color=c, fontsize=8.5, weight="bold", va="center")
    if "clean" in cells:
        r = cells["clean"]
        ax.axhspan(max(r["ci"][0], floor), r["ci"][1], color="#8b949e", alpha=0.18, lw=0)
        ax.text(1.5, max(r["ci"][0], floor) / 1.8, f"clean sensors: {r['av_crash']:.0f} crashes, {r['rate_per_h']:.3f} / AV-h",
                color="#8b949e", fontsize=8)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 300); ax.set_xlim(0.4, 3.3)
    ax.set_xticks([0.6, 1.0, 1.4, 2.0])
    ax.set_xlabel("perception delay on the leader (s)", color=MUTED, fontsize=9)
    ax.set_ylabel("AV crashes per AV-hour (95 %)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    fig.text(0.02, 0.95, "Perception latency: crash rate against delay, with and without prediction", color=FG,
             fontsize=11, weight="bold")
    t = res["tests"].get("H1")
    if t:
        fig.text(0.02, 0.895, f"Oakland, half AVs, IDM, envelope off, 16 copies x 600 s per point; "
                 f"numbers = AV crashes. H1 at 1.0 s: RR {t['rr']:.4f} ({t['lo']:.4f}-{t['hi']:.4f})", color=MUTED,
                 fontsize=7.2)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.13)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    else:
        print(lane_gif())
