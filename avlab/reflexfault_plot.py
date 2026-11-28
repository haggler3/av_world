"""GIF and figure for avlab.reflexfault (an imperfect 10 m reflex sensor fused with a 2.0 s delayed view).

    python -m avlab.reflexfault_plot          # docs/img/reflexfault_cutin.gif: the scripted cut-in with reflex outages
    python -m avlab.reflexfault_plot --bench  # docs/img/reflexfault.png from results/reflexfault/reflexfault.json

The lane is the one of avlab.reflex_plot: the AV follows a leader 40 m ahead at 12 m/s; at t = 3 s a car cuts in 7 m ahead
at 5 m/s. Three AVs run the same IDM policy behind the same 2.0 s delayed view and 10 m reflex; the reflex is blind for
0 s, 1.0 s or 1.6 s starting at the cut-in. The outages are scripted, not drawn at random.
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
MODELS = {"reflex works": (0.0, "#56B4E9"), "reflex blind 1.0 s": (1.0, "#009E73"),
          "reflex blind 1.6 s": (1.6, "#E69F00")}
OUTAGES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5]


def scripted(outage_s, T_total=9.0):
    import jax
    import jax.numpy as jnp
    from avlab import reflexfault as RX
    from avlab import teb as T
    pol = jax.jit(lambda o, m: T.idm_policy_masked(o, None, m)[0])
    on, off = RX.Fault(range_m=10.0), RX.Fault(range_m=0.0)   # same delayed view; off = reflex shows nothing
    per = jax.jit(RX.perceive, static_argnums=3)
    x, v, lx, cx, hit, rows = 0.0, 12.0, 40.0 + L, None, None, []
    mem = RX.init_mem_for(on)(1)
    for k in range(int(T_total / DT)):
        t = k * DT
        lx += 12.0 * DT
        if t >= LAP.CUT_T and cx is None:
            cx = x + L + LAP.CUT_GAP + L
        if cx is not None:
            cx += LAP.CUT_V * DT
        lead, lv = (cx, LAP.CUT_V) if cx is not None else (lx, 12.0)
        gap = lead - x - L
        blind = LAP.CUT_T - 1e-6 <= t < LAP.CUT_T + outage_s - 1e-6
        o, mem = per(jnp.asarray(_obs(v, 13.0, gap, lv))[None], jax.random.PRNGKey(k), mem, off if blind else on)
        u = float(pol(o, jnp.ones(1, bool))[0, 0])
        a = 3 * u if u >= 0 else 8 * u
        if hit is None and gap <= 0:
            hit = (t, v - lv)
        if hit is not None:
            a, v = 0.0, lv
        rows.append(dict(t=t, x=x, v=v, lead=lx, cut=cx, gap=gap, seen=50 * float(o[0, 9]), shown=bool(o[0, 8] > 0.5),
                         blind=blind, reflex=(not blind) and 0 < gap <= 10.0, hit=hit))
        v = max(v + a * DT, 0.0) if hit is None else v
        x += v * DT
    return rows


def cutin_gif(path="docs/img/reflexfault_cutin.gif", fps=10):
    from matplotlib.animation import FuncAnimation, PillowWriter
    names = list(MODELS)
    d = {nm: scripted(MODELS[nm][0]) for nm in names}
    n = len(d[names[0]])
    frames = list(range(n)) + [n - 1] * 12
    fig = plt.figure(figsize=(7.2, 4.4), facecolor=BG)
    ax = fig.add_axes([0.03, 0.12, 0.94, 0.66])
    fig.text(0.03, 0.93, "When the reflex sensor drops out at the wrong moment", color=FG, fontsize=12, weight="bold")
    fig.text(0.03, 0.84, f"2.0 s delayed view + 10 m reflex. At t = {LAP.CUT_T:.0f} s a car (red) cuts in {LAP.CUT_GAP:.0f} m"
             f" ahead at {LAP.CUT_V:.0f} m/s and the reflex goes blind.\nDashed box = where the AV believes its leader is."
             " Blue ring = reflex in use; grey ring = reflex blind. IDM, no envelope.", color=MUTED, fontsize=7.5)

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
            if r["hit"] is None and (r["reflex"] or r["blind"]):
                ax.add_patch(plt.Rectangle((r["x"], y - 0.36), 10.0, 0.72, fill=False, lw=1.0,
                                           ec="#8b949e" if r["blind"] else "#56B4E9", ls=":" if r["blind"] else "-"))
            ax.text(x0 + 1, y + 0.42, nm, color=c, fontsize=8, weight="bold")
            if r["hit"] is None and r["shown"]:
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


def bench_fig(src="results/reflexfault/reflexfault.json", path="docs/img/reflexfault.png"):
    res = json.load(open(src))
    cells = res["cells"]
    order = [("clean", "clean\nsensors", "#009E73"), ("comp2.0", "2.0 s delay\nno reflex", "#E69F00"),
             ("rf10", "+ perfect\n10 m reflex", "#56B4E9"), ("fi", "+ reflex,\n15 % misses", "#CC79A7"),
             ("fb", "+ reflex,\n15 % in 1 s\nbursts", "#CC79A7"), ("fn", "+ reflex,\n20 % range\nnoise", "#CC79A7")]
    order = [o for o in order if o[0] in cells]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.4, 4.0), facecolor=BG, gridspec_kw=dict(width_ratios=[1.6, 1]))
    for i, (c, lab, col) in enumerate(order):
        r = cells[c]
        y = max(r["rate_per_h"], 1e-3)
        ax.bar(i, y, width=0.6, color=col)
        ax.plot([i, i], [max(r["ci"][0], 1e-3), r["ci"][1]], color=FG, lw=1.2)
        ax.text(i, r["ci"][1] * 1.3, f"{r['av_crash']:.0f}", color=col, fontsize=8.5, ha="center")
    ax.set_yscale("log"); ax.set_ylim(3e-3, 12)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o[1] for o in order], color=MUTED, fontsize=7.2)
    ax.set_ylabel("AV crashes per AV-hour (95 %)", color=MUTED, fontsize=8.5)
    ax.text(-0.4, 9.5, "numbers = AV crashes", color=MUTED, fontsize=7.5, va="top")
    gaps = [min(r["gap"] for r in scripted(o)) for o in OUTAGES]
    bx.axhline(0, color=GRID, lw=0.8)
    bx.plot(OUTAGES, [max(g, -1.0) for g in gaps], "o-", color="#CC79A7", ms=4)
    first = next((o for o, g in zip(OUTAGES, gaps) if g <= 0), None)
    if first is not None:
        bx.axvline(first, color="#D55E00", lw=0.8, ls=":")
        bx.text(first + 0.05, 5.5, f"contact from\n{first:.1f} s blind", color="#D55E00", fontsize=7.5)
    bx.set_xlabel("reflex blind from the cut-in (s)", color=MUTED, fontsize=8)
    bx.set_ylabel("smallest gap (m)", color=MUTED, fontsize=8)
    bx.set_title("scripted cut-in (GIF)", color=MUTED, fontsize=8.5)
    for a in (ax, bx):
        a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=8)
        for sp in a.spines.values():
            sp.set_color(GRID)
    t = res["tests"].get("H1", {})
    fig.text(0.02, 0.95, "An imperfect reflex sensor against a 2.0 s perception delay", color=FG, fontsize=11, weight="bold")
    if t:
        fig.text(0.02, 0.895, f"Oakland, half AVs, IDM, envelope off, 16 copies x 600 s per cell.  H1 bursty misses / perfect"
                 f" reflex: RR {t['rr']:.2f} ({t['lo']:.2f}-{t['hi']:.2f}), p {t['p']:.2g}", color=MUTED, fontsize=7.2)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.17, wspace=0.3)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    else:
        print(cutin_gif())
