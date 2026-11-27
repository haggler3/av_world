"""GIF and figure for avlab.reflex (a zero-delay short-range sensor fused with a 2.0 s delayed view).

    python -m avlab.reflex_plot          # docs/img/reflex_cutin.gif: the 2026-11-26 scripted cut-in, three AVs
    python -m avlab.reflex_plot --bench  # docs/img/reflex.png from results/reflex/reflex.json

The lane is the one of avlab.latencyattr_plot: the AV follows a leader 40 m ahead at 12 m/s; at t = 3 s a car cuts in
7 m ahead at 5 m/s. Three AVs run the same IDM policy, no envelope: clean sensors, 2.0 s delay with constant-velocity
prediction, and the same delayed view fused with a 10 m reflex sensor.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from avlab import latencyattr_plot as LAP  # noqa: E402
from avlab.patchyice_plot import BG, FG, GRID, MUTED  # noqa: E402
from avlab.sensornoise_plot import _obs  # noqa: E402

DT, L = LAP.DT, LAP.L
MODELS = {"clean sensors": ((2.0, True, False, -1.0), "#009E73"),
          "2.0 s delay + prediction": ((2.0, True, False, 0.0), "#E69F00"),
          "2.0 s delay + 10 m reflex": ((2.0, True, False, 10.0), "#56B4E9")}


def scripted(params, T_total=9.0):
    import jax
    import jax.numpy as jnp
    from avlab import reflex as RF
    from avlab import teb as T
    pol = jax.jit(lambda o, m: T.idm_policy_masked(o, None, m)[0])
    clean = params[3] < 0
    m = RF.Reflex(*params[:3], max(params[3], 0.0))
    per = jax.jit(RF.perceive, static_argnums=3)
    x, v, lx, cx, hit, rows = 0.0, 12.0, 40.0 + L, None, None, []
    mem = RF.init_mem_for(m)(1)
    for k in range(int(T_total / DT)):
        t = k * DT
        lx += 12.0 * DT
        if t >= LAP.CUT_T and cx is None:
            cx = x + L + LAP.CUT_GAP + L
        if cx is not None:
            cx += LAP.CUT_V * DT
        lead, lv = (cx, LAP.CUT_V) if cx is not None else (lx, 12.0)
        gap = lead - x - L
        o_true = jnp.asarray(_obs(v, 13.0, gap, lv))[None]
        o = o_true if clean else None
        if not clean:
            o, mem = per(o_true, None, mem, m)
        u = float(pol(o, jnp.ones(1, bool))[0, 0])
        a = 3 * u if u >= 0 else 8 * u
        if hit is None and gap <= 0:
            hit = (t, v - lv)
        if hit is not None:
            a, v = 0.0, lv
        rows.append(dict(t=t, x=x, v=v, lead=lx, cut=cx, gap=gap, seen=50 * float(o[0, 9]), shown=bool(o[0, 8] > 0.5),
                         reflex=(not clean) and m.range_m > 0 and 0 < gap <= m.range_m, hit=hit))
        v = max(v + a * DT, 0.0) if hit is None else v
        x += v * DT   # after contact the AV is carried along at the struck car's speed
    return rows


def cutin_gif(path="docs/img/reflex_cutin.gif", fps=10):
    from matplotlib.animation import FuncAnimation, PillowWriter
    names = list(MODELS)
    d = {nm: scripted(MODELS[nm][0]) for nm in names}
    n = len(d[names[0]])
    frames = list(range(n)) + [n - 1] * 12
    fig = plt.figure(figsize=(7.2, 4.4), facecolor=BG)
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.66])
    fig.text(0.03, 0.93, "A 10 m reflex sensor against a 2.0 s perception delay", color=FG, fontsize=12, weight="bold")
    fig.text(0.03, 0.84, f"At t = {LAP.CUT_T:.0f} s a car (red) cuts in {LAP.CUT_GAP:.0f} m ahead at {LAP.CUT_V:.0f} m/s. "
             "Dashed box = where the AV believes its leader is.\nBlue ring = reflex sensor in use. Same IDM policy, no envelope."
             " Real time.", color=MUTED, fontsize=7.5)

    def draw(k):
        ax.cla(); ax.set_facecolor(BG); ax.set_yticks([]); ax.set_ylim(-0.9, 4.5)
        x0 = d[names[0]][k]["x"] - 8
        ax.set_xlim(x0, x0 + 62)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("position (m)", color=MUTED, fontsize=8, labelpad=1)
        for row, nm in enumerate(reversed(names)):
            r, c, y = d[nm][k], MODELS[nm][1], row * 1.55
            ax.plot([x0, x0 + 62], [y - 0.45, y - 0.45], color=GRID, lw=0.8)
            ax.add_patch(plt.Rectangle((r["lead"] - L, y - 0.3), L, 0.6, color="#8b949e", alpha=0.7))
            if r["cut"] is not None:
                ax.add_patch(plt.Rectangle((r["cut"] - L, y - 0.3), L, 0.6, color="#D55E00"))
            ax.add_patch(plt.Rectangle((r["x"] - L, y - 0.3), L, 0.6, color=c))
            if r["reflex"] and r["hit"] is None:
                ax.add_patch(plt.Rectangle((r["x"], y - 0.36), 10.0, 0.72, fill=False, ec="#56B4E9", lw=1.0, alpha=0.8))
            ax.text(x0 + 1, y + 0.42, nm, color=c, fontsize=8, weight="bold")
            if r["hit"] is None and MODELS[nm][0][3] >= 0 and r["shown"]:
                ax.add_patch(plt.Rectangle((r["x"] + r["seen"], y - 0.3), L, 0.6, fill=False, ec=c, ls="--", lw=1.2))
            if r["hit"] is not None:
                ax.text(x0 + 31, y + 0.42, f"CONTACT at {r['hit'][1] * 3.6:.0f} km/h", color="#D55E00", fontsize=8.5,
                        weight="bold", ha="center")
            ax.text(x0 + 61, y + 0.42, f"gap {max(r['gap'], 0):5.1f} m  v {r['v']:4.1f} m/s", color=FG, fontsize=7.5, ha="right")
        ax.text(x0 + 31, 4.15, f"t = {d[names[0]][k]['t']:4.1f} s", color=FG, fontsize=9, ha="center", weight="bold")
        for s in ax.spines.values():
            s.set_color(GRID)
    FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72,
                                                                           savefig_kwargs={"facecolor": BG})
    return {nm: d[nm][-1]["hit"] for nm in names}


def bench_fig(src="results/reflex/reflex.json", path="docs/img/reflex.png"):
    res = json.load(open(src))
    cells = res["cells"]
    order = [("clean", "clean\nsensors", "#009E73"), ("comp2.0", "2.0 s delay\nno reflex", "#E69F00"),
             ("rf5", "+ 5 m\nreflex*", "#56B4E9"), ("rf10", "+ 10 m\nreflex", "#56B4E9"), ("rf20", "+ 20 m\nreflex", "#56B4E9")]
    order = [o for o in order if o[0] in cells]
    fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.9), facecolor=BG, gridspec_kw=dict(width_ratios=[1.5, 1]))
    ax, bx = axs
    for i, (c, lab, col) in enumerate(order):
        r = cells[c]
        y = max(r["rate_per_h"], 1e-3)
        ax.bar(i, y, width=0.6, color=col)
        ax.plot([i, i], [max(r["ci"][0], 1e-3), r["ci"][1]], color=FG, lw=1.2)
        ax.text(i, r["ci"][1] * 1.3, f"{r['av_crash']:.0f}", color=col, fontsize=8, ha="center")
        bx.plot(r["av_speed_kmh"], i, "o", color=col, ms=7)
        bx.text(r["av_speed_kmh"] + 0.03, i + 0.22, f"{r['av_speed_kmh']:.2f}", color=col, fontsize=7.5)
    ax.set_yscale("log"); ax.set_ylim(3e-3, 12)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o[1] for o in order], color=MUTED, fontsize=7.2)
    ax.set_ylabel("AV crashes per AV-hour (95 %)", color=MUTED, fontsize=8.5)
    ax.text(-0.4, 9.5, "numbers = AV crashes\n* 5 m added after the\n  pre-registered runs", color=MUTED, fontsize=7.5, va="top")
    bx.set_yticks(range(len(order))); bx.set_yticklabels([o[1].replace("\n", " ") for o in order], color=MUTED, fontsize=7)
    bx.invert_yaxis()
    sp = [cells[c]["av_speed_kmh"] for c, _, _ in order]
    bx.set_xlim(min(sp) - 0.3, max(sp) + 0.5)
    bx.set_xlabel("AV speed (km/h)", color=MUTED, fontsize=8)
    bx.set_title("AV speed", color=MUTED, fontsize=8.5)
    for a in axs:
        a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=8)
        for s in a.spines.values():
            s.set_color(GRID)
    t = res["tests"].get("H1", {})
    fig.text(0.02, 0.95, "A zero-delay short-range sensor against a 2.0 s perception delay", color=FG, fontsize=11,
             weight="bold")
    if t:
        fig.text(0.02, 0.895, f"Oakland, half AVs, IDM, envelope off, 16 copies x 600 s per cell.  H1 10 m reflex / no reflex:"
                 f" RR {t['rr']:.3f} ({t['lo']:.3f}-{t['hi']:.3f}), p {t['p']:.1g}", color=MUTED, fontsize=7.2)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.83, bottom=0.14, wspace=0.55)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    else:
        print(cutin_gif())
