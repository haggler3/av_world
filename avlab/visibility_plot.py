"""GIF and figure for avlab.visibility / avlab.visibilitybench.

    python -m avlab.visibility_plot          # docs/img/visibility_lane.gif: one scripted lane, clear vs fog vs fog + phantom
    python -m avlab.visibility_plot --bench  # docs/img/visibility_bench.png from results/visibility/visibility.json

The lane: a stopped car 60 m ahead (a breakdown in fog). Three AVs start at 12 m/s and run the same IDM policy
(avlab.teb.idm_policy_masked), no envelope, each behind a different avlab.visibility.Visibility (the code the benchmark uses).
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
SIGHT = 8.0
MODELS = {"clear sight": ((1e9, False), "#009E73"), f"sight {SIGHT:.0f} m": ((SIGHT, False), "#E69F00"),
          f"sight {SIGHT:.0f} m + phantom": ((SIGHT, True), "#56B4E9")}


@functools.lru_cache(None)
def _fns():
    import jax
    from avlab import teb as T
    from avlab import visibility as VI
    pol = jax.jit(lambda o, m: T.idm_policy_masked(o, None, m)[0])
    per = jax.jit(VI.perceive, static_argnums=3)
    return pol, per


def scripted(params=(SIGHT, False), T_total=20.0, gap0=60.0, v0=12.0):
    """Per-step rows for one AV approaching a stopped car behind Visibility(*params); stops it at contact."""
    import jax.numpy as jnp
    from avlab import visibility as VI
    pol, per = _fns()
    m = VI.Visibility(*params)
    lx, x, v, hit, rows = gap0 + L, 0.0, v0, None, []
    mem = VI.init_mem(1)
    for k in range(int(T_total / DT)):
        gap = lx - x - L
        o, mem = per(jnp.asarray(_obs(v, 13.0, gap, 0.0))[None], None, mem, m)
        u = float(pol(o, jnp.ones(1, bool))[0, 0])
        a = 3 * u if u >= 0 else 8 * u
        if hit is None and gap <= 0:
            hit = (k * DT, v)
        if hit is not None:
            a, v = 0.0, 0.0
        rows.append(dict(t=k * DT, x=x, v=v, a=a, lead=lx, gap=gap, hit=hit))
        if hit is None:
            v = max(v + a * DT, 0.0); x += v * DT
    return rows


def lane_gif(path="docs/img/visibility_lane.gif", fps=10, every=1):
    from matplotlib.animation import FuncAnimation, PillowWriter
    names = list(MODELS)
    d = {nm: scripted(MODELS[nm][0]) for nm in names}
    n = len(d[names[0]])
    frames = list(range(0, n, every)) + [n - 1] * 10
    fig = plt.figure(figsize=(7.2, 4.6), facecolor=BG)
    ax = fig.add_axes([0.03, 0.47, 0.94, 0.36]); ax2 = fig.add_axes([0.1, 0.12, 0.86, 0.22])
    fig.text(0.03, 0.94, f"Fog: the AV sees only {SIGHT:.0f} m ahead", color=FG, fontsize=12, weight="bold")
    fig.text(0.03, 0.865, "A car is stopped 60 m ahead. Same IDM policy, no envelope. Shaded = what the AV can see.\n"
             "Phantom: the AV treats the edge of its sight as a stopped car (dashed box). Real time.",
             color=MUTED, fontsize=7.5)
    t_all = [r["t"] for r in d[names[0]]]

    def draw(k):
        ax.cla(); ax.set_facecolor(BG); ax.set_yticks([]); ax.set_ylim(-0.6, 4.0)
        x0 = min(min(d[nm][k]["x"] for nm in names) - 10, d[names[0]][0]["lead"] - 50)
        ax.set_xlim(x0, x0 + 75)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("position (m)", color=MUTED, fontsize=8, labelpad=1)
        for row, nm in enumerate(reversed(names)):
            r, (prm, c), y = d[nm][k], MODELS[nm], row * 1.5
            ax.plot([x0, x0 + 75], [y - 0.45, y - 0.45], color=GRID, lw=0.8)
            if prm[0] < 1e3 and r["hit"] is None:
                ax.add_patch(plt.Rectangle((r["x"], y - 0.4), prm[0], 0.8, color=c, alpha=0.18, lw=0))
            ax.add_patch(plt.Rectangle((r["lead"] - L, y - 0.3), L, 0.6, color=LEADC, alpha=0.7))
            ax.add_patch(plt.Rectangle((r["x"] - L, y - 0.3), L, 0.6, color=c))
            ax.text(x0 + 1, y + 0.42, nm, color=c, fontsize=8, weight="bold")
            if r["hit"] is None:
                if prm[1] and r["gap"] > prm[0]:
                    ax.add_patch(plt.Rectangle((r["x"] + prm[0], y - 0.3), L, 0.6, fill=False, ec=c, ls="--", lw=1.2))
            else:
                ax.text(x0 + 22, y + 0.42, f"CONTACT at {r['hit'][1] * 3.6:.0f} km/h", color=REDC, fontsize=8.5,
                        weight="bold")
            ax.text(x0 + 74, y + 0.42, f"gap {max(r['gap'], 0):5.1f} m  v {r['v']:4.1f} m/s", color=FG, fontsize=7.5,
                    ha="right")
        ax.text(x0 + 37, 3.6, f"t = {t_all[k]:4.1f} s", color=FG, fontsize=9, ha="center", weight="bold")
        ax2.cla(); ax2.set_facecolor(BG)
        for i, nm in enumerate(names):
            ax2.plot(t_all[:k + 1], [d[nm][j]["v"] * 3.6 for j in range(k + 1)], color=MODELS[nm][1], lw=1.6)
            ax2.text(t_all[-1] * 0.98, 46 - 9 * i, nm, color=MODELS[nm][1], fontsize=7.5, ha="right", weight="bold")
        ax2.set_xlim(0, t_all[-1]); ax2.set_ylim(0, 50)
        ax2.set_ylabel("AV speed (km/h)", color=MUTED, fontsize=8); ax2.set_xlabel("time (s)", color=MUTED, fontsize=8)
        ax2.tick_params(colors=MUTED, labelsize=8)
        for s in list(ax.spines.values()) + list(ax2.spines.values()):
            s.set_color(GRID)
    FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72,
                                                                           savefig_kwargs={"facecolor": BG})
    return {nm: d[nm][-1]["hit"] for nm in names}


def bench_fig(src="results/visibility/visibility.json", path="docs/img/visibility_bench.png"):
    res = json.load(open(src))
    cells = res["cells"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 4.0), facecolor=BG, gridspec_kw={"width_ratios": [1.6, 1]})
    floor = 5e-4
    for phantom, c, lab in ((False, "#E69F00", "AV blind beyond sight"), (True, "#56B4E9", "with phantom at sight edge")):
        rs = sorted((r for r in cells.values() if r["phantom"] == phantom and r["sight_m"] < 1e3), key=lambda r: r["sight_m"])
        if not rs:
            continue
        s = np.array([r["sight_m"] for r in rs]) + (0.15 if phantom else -0.15)
        rate = np.array([max(r["rate_per_h"], floor) for r in rs])
        ax.plot(s, rate, "-", color=c, lw=1.6)
        for ss, r, y in zip(s, rs, rate):
            ax.plot([ss, ss], [max(r["ci"][0], floor), r["ci"][1]], color=c, lw=2)
            ax.plot(ss, y, "o", ms=7, color=c, mfc=c if r["av_crash"] > 0 else BG, mew=2)
            ax.text(ss + 0.3, y, f"{r['av_crash']:.0f}", color=c, fontsize=7.5, va="center")
        ax.text(s[0] - 0.2, rate[0] * (2.2 if not phantom else 5.0), lab, color=c, fontsize=8.5, weight="bold")
        bx.plot(s, [r["av_speed_kmh"] for r in rs], "o-", color=c, lw=1.6)
    if "clear" in cells:
        r = cells["clear"]
        ax.axhspan(max(r["ci"][0], floor), r["ci"][1], color="#8b949e", alpha=0.18, lw=0)
        ax.text(9.5, floor * 1.3, f"clear sight: {r['av_crash']:.0f} crash{'es' * (r['av_crash'] != 1)}, {r['rate_per_h']:.3f} / AV-h",
                color="#8b949e", fontsize=8)
        bx.axhline(r["av_speed_kmh"], color="#8b949e", lw=1, ls="--")
        bx.text(14.3, r["av_speed_kmh"] - 0.5, "clear sight", color="#8b949e", fontsize=8, ha="right")
    ax.set_yscale("log"); ax.set_ylim(floor, 100); ax.set_xlim(5, 16)
    ax.set_xticks([6, 8, 10, 14])
    ax.set_xlabel("AV sight distance (m)", color=MUTED, fontsize=9)
    ax.set_ylabel("AV crashes per AV-hour (95 %)", color=MUTED, fontsize=9)
    bx.set_xticks([6, 8, 10, 14]); bx.set_xlim(5, 15)
    bx.set_xlabel("AV sight distance (m)", color=MUTED, fontsize=9)
    bx.set_ylabel("AV mean speed (km/h)", color=MUTED, fontsize=9)
    for a in (ax, bx):
        a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=8)
        for sp in a.spines.values():
            sp.set_color(GRID)
    fig.text(0.02, 0.95, "Limited sight: AV crash rate and speed, with and without a phantom obstacle", color=FG,
             fontsize=11, weight="bold")
    t = res["tests"].get("H1")
    if t:
        fig.text(0.02, 0.895, f"Oakland, half AVs, IDM, envelope off, 16 copies x 600 s per point; numbers = AV crashes. "
                 f"H1 at 8 m: RR {t['rr']:.4f} ({t['lo']:.4f}-{t['hi']:.4f})", color=MUTED, fontsize=7.2)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.85, bottom=0.13, wspace=0.3)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    else:
        print(lane_gif())
