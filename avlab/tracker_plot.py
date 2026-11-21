"""GIF and figure for avlab.tracker / avlab.trackerbench.

    python -m avlab.tracker_plot          # docs/img/tracker_lane.gif: bursty misses with and without a 1 s hold, one scripted lane
    python -m avlab.tracker_plot --bench  # docs/img/tracker_bench.png from results/tracker/tracker.json
    python -m avlab.tracker_plot --creep  # results/tracker/creep.json: standing gap over 40 noise seeds per model

The lane: a leader 30 m ahead at 12 m/s brakes at 6 m/s^2 to a stop at t = 4 s. Three AVs run the same IDM policy
(avlab.teb.idm_policy_masked), no envelope, each behind a different avlab.tracker.Perception (the code the benchmark uses).
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
MODELS = {"independent misses": ((0.15, 0.2, 0.0), "#56B4E9"), "bursty misses": ((0.15, 1.0, 0.0), "#E69F00"),
          "bursty + 1 s hold": ((0.15, 1.0, 1.0), "#009E73")}


@functools.lru_cache(None)
def _fns():
    import jax
    from avlab import teb as T
    from avlab import tracker as TR
    pol = jax.jit(lambda o, m: T.idm_policy_masked(o, None, m)[0])
    per = jax.jit(TR.perceive, static_argnums=3)
    return pol, per


def scripted(seed=0, T_total=40.0, params=(0.15, 1.0, 0.0)):
    """Per-step rows for one follower behind Perception(*params); stops it at contact."""
    import jax
    import jax.numpy as jnp
    from avlab import tracker as TR
    pol, per = _fns()
    m = TR.Perception(*params)
    n = int(T_total / DT)
    lx, lv, x, v, hit, rows = 30.0 + L, 12.0, 0.0, 12.0, None, []
    mem = TR.init_mem(1)
    for k in range(n):
        lv = max(lv + (-6.0 if k * DT >= 4 else 0.0) * DT, 0.0); lx += lv * DT
        gap = lx - x - L
        o, mem = per(jnp.asarray(_obs(v, 13.0, gap, lv))[None], jax.random.PRNGKey(seed * 10000 + k), mem, m)
        seen = (50 * float(o[0, 9]), bool(o[0, 8] > 0.5), bool(mem.miss[0]))
        u = float(pol(o, jnp.ones(1, bool))[0, 0])
        a = 3 * u if u >= 0 else 8 * u
        if hit is None and gap <= 0:
            hit = k * DT
        if hit is not None:
            a, v = 0.0, lv
        rows.append(dict(t=k * DT, x=x, v=v, a=a, lead=lx, gap=gap, seen=seen, hit=hit))
        if hit is None:
            v = max(v + a * DT, 0.0); x += v * DT
    return rows


def creep(seeds=40):
    out = {"seeds": seeds, "t_s": [8, 16, 24, 32, 40], "models": {}}
    for name, (p, _) in MODELS.items():
        runs = [scripted(s, params=p) for s in range(seeds)]
        g = np.array([[r["gap"] for r in rs] for rs in runs])
        out["models"][name] = {"params": p, "mean_gap_m": g[:, [40, 80, 120, 160, 199]].mean(0).round(3).tolist(),
                               "contact_share": float(np.mean([rs[-1]["hit"] is not None for rs in runs]))}
    return out


def lane_gif(path="docs/img/tracker_lane.gif", seed=None, fps=10, every=2):
    from matplotlib.animation import FuncAnimation, PillowWriter
    names = list(MODELS)
    if seed is None:  # first seed where the bursty, untracked follower makes contact
        seed = next((s for s in range(60) if scripted(s, params=MODELS["bursty misses"][0])[-1]["hit"] is not None), 0)
    d = {nm: scripted(seed, params=MODELS[nm][0]) for nm in names}
    n = len(d[names[0]])
    frames = list(range(0, n, every)) + [n - 1] * 10
    fig = plt.figure(figsize=(7.2, 4.6), facecolor=BG)
    ax = fig.add_axes([0.03, 0.47, 0.94, 0.36]); ax2 = fig.add_axes([0.1, 0.09, 0.86, 0.24])
    fig.text(0.03, 0.94, "Bursty missed detections, and a tracker that holds the last one", color=FG, fontsize=12, weight="bold")
    fig.text(0.03, 0.865, "Leader stops at t = 4 s. Same IDM policy, no envelope, 15 % of steps missed in every lane.\n"
             "Dashed box = leader the AV perceives (red: MISSED). Played at 2x.", color=MUTED, fontsize=7.5)
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
            ax.add_patch(plt.Rectangle((r["lead"] - L, y - 0.3), L, 0.6, color=LEADC))
            ax.add_patch(plt.Rectangle((r["x"] - L, y - 0.3), L, 0.6, color=c))
            ax.text(x0 + 1, y + 0.42, nm, color=c, fontsize=8, weight="bold")
            g, det, miss = r["seen"]
            if r["hit"] is None:
                if det and miss:
                    ax.add_patch(plt.Rectangle((r["x"] + g, y - 0.3), L, 0.6, fill=False, ec=c, ls="--", lw=1.2))
                    ax.text(r["lead"] + 1, y + 0.42, "missed, tracked", color=c, fontsize=7.5)
                elif not det:
                    ax.text(r["lead"] + 1, y + 0.42, "leader MISSED", color=REDC, fontsize=8, weight="bold")
            else:
                ax.text(r["lead"] + 1, y + 0.42, f"CONTACT at t = {r['hit']:.1f} s", color=REDC, fontsize=8.5, weight="bold")
            ax.text(x0 + 59, y + 0.42, f"gap {r['gap']:5.2f} m  v {r['v']:4.1f} m/s", color=FG, fontsize=7.5, ha="right")
        ax.text(x0 + 30, 3.6, f"t = {t_all[k]:4.1f} s", color=FG, fontsize=9, ha="center", weight="bold")
        ax2.cla(); ax2.set_facecolor(BG)
        for nm in names:
            ax2.plot(t_all[:k + 1], [max(d[nm][j]["gap"], 0) for j in range(k + 1)], color=MODELS[nm][1], lw=1.4)
        ax2.set_xlim(0, t_all[-1]); ax2.set_ylim(0, 6)
        ax2.set_ylabel("true gap (m)", color=MUTED, fontsize=8); ax2.set_xlabel("time (s)", color=MUTED, fontsize=8)
        ax2.tick_params(colors=MUTED, labelsize=8)
        for s in list(ax.spines.values()) + list(ax2.spines.values()):
            s.set_color(GRID)
    FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72,
                                                                           savefig_kwargs={"facecolor": BG})
    return seed


LABELS = {"clean": "clean sensors", "iid": "independent misses", "iid_hold": "independent + 1 s hold",
          "burst": "bursty misses (1 s)", "burst_hold": "bursty + 1 s hold", "burst_hold2": "bursty + 2 s hold"}
COLORS = {"clean": "#8b949e", "iid": "#56B4E9", "iid_hold": "#56B4E9", "burst": "#E69F00", "burst_hold": "#E69F00",
          "burst_hold2": "#E69F00"}


def bench_fig(src="results/tracker/tracker.json", path="docs/img/tracker_bench.png"):
    res = json.load(open(src))
    cells = [c for c in LABELS if c in res["cells"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), facecolor=BG)
    ax.set_facecolor(BG)
    for i, c in enumerate(cells):
        r = res["cells"][c]
        lo, hi = max(r["ci"][0], 1e-4), r["ci"][1]
        y = len(cells) - 1 - i
        rate = max(r["rate_per_h"], 1e-4)
        hollow = "hold" in c
        ax.plot([lo, hi], [y, y], color=COLORS[c], lw=2)
        ax.plot(rate, y, "o", ms=8, color=COLORS[c], mfc=BG if hollow else COLORS[c], mew=2)
        ax.text(hi * 1.4, y, f"{r['av_crash']:.0f} crashes, {r['rate_per_h']:.3f} / AV-h", color=FG, fontsize=8, va="center")
        ax.text(1.2e-4, y + 0.32, LABELS[c], color=COLORS[c], fontsize=8.5, weight="bold")
    ax.set_xscale("log"); ax.set_xlim(1e-4, 5000); ax.set_ylim(-0.6, len(cells) - 0.3); ax.set_yticks([])
    ax.set_xlabel("AV crashes per AV-hour (95 % interval, log scale)", color=MUTED, fontsize=9)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    t = res["tests"].get("H1")
    ttl = "Missed detections: independent vs bursty, with and without a tracker"
    fig.text(0.02, 0.95, ttl, color=FG, fontsize=11, weight="bold")
    if t:
        fig.text(0.02, 0.895, f"Oakland, half AVs, IDM, envelope off, 16 copies x 600 s per row. H1 (hold vs none, independent): "
                 f"RR {t['rr']:.4f} ({t['lo']:.4f}-{t['hi']:.4f})", color=MUTED, fontsize=7.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.14)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    elif "--creep" in sys.argv:
        r = creep()
        json.dump(r, open("results/tracker/creep.json", "w"), indent=1)
        print(json.dumps(r, indent=1))
    else:
        print("seed", lane_gif())
