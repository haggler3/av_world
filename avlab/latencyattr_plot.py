"""GIF and figure for avlab.latencyattr (where delayed-perception crashes come from).

    python -m avlab.latencyattr_plot          # docs/img/latencyattr_cutin.gif: a scripted cut-in, clean vs 2.0 s delay
    python -m avlab.latencyattr_plot --bench  # docs/img/latencyattr.png from results/latencyattr/attr.json

The lane: the AV follows a leader 40 m ahead, both at 12 m/s. At t = 3 s a car cuts in 9 m ahead of the AV at 7 m/s and keeps
that speed. Two AVs run the same IDM policy (avlab.teb.idm_policy_masked), no envelope: clean sensors, and a 2.0 s delay with
constant-velocity prediction (avlab.latency, the code of the benchmark). The label under each AV is its avlab.latencyattr state.
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avlab import latency_plot as LP  # noqa: E402
from avlab.patchyice_plot import BG, FG, GRID, MUTED  # noqa: E402
from avlab.sensornoise_plot import _obs  # noqa: E402

DT, L = 0.2, 4.5
CUT_T, CUT_GAP, CUT_V = 3.0, 7.0, 5.0
MODELS = {"clean sensors": ((0.0, False), "#009E73"), "2.0 s delay + prediction": ((2.0, True), "#E69F00")}
SCOL = {"none": "#8b949e", "ok": "#009E73", "new": "#D55E00", "over": "#CC79A7", "other": "#56B4E9"}
SLAB = {"none": "empty road", "ok": "view right", "new": "NEW LEADER NOT YET SEEN", "over": "gap over-estimated",
        "other": "other"}


def scripted(params, T_total=9.0):
    import jax.numpy as jnp
    from avlab import latency as LT
    from avlab import latencyattr as LA
    pol, per = LP._fns()
    m = LT.Latency(*params)
    x, v, lx, cx, hit, rows = 0.0, 12.0, 40.0 + L, None, None, []
    mem = LT.init_mem_for(m)(1)
    for k in range(int(T_total / DT)):
        t = k * DT
        lx += 12.0 * DT
        if t >= CUT_T and cx is None:
            cx = x + L + CUT_GAP + L
        if cx is not None:
            cx += CUT_V * DT
        lead, lv = (cx, CUT_V) if cx is not None else (lx, 12.0)
        gap = lead - x - L
        o_true = jnp.asarray(_obs(v, 13.0, gap, lv))[None]
        o, mem = per(o_true, None, mem, m)
        st = LA.STATES[int(LA.classify(o_true[:, 8] > 0.5, 50 * o_true[:, 9], o[:, 8] > 0.5, 50 * o[:, 9])[0])]
        u = float(pol(o, jnp.ones(1, bool))[0, 0])
        a = 3 * u if u >= 0 else 8 * u
        if hit is None and gap <= 0:
            hit = (t, v - lv)
        if hit is not None:
            a, v = 0.0, lv
        rows.append(dict(t=t, x=x, v=v, lead=lx, cut=cx, gap=gap, seen=50 * float(o[0, 9]), shown=bool(o[0, 8] > 0.5),
                         state=st, hit=hit))
        if hit is None:
            v = max(v + a * DT, 0.0); x += v * DT
    return rows


def cutin_gif(path="docs/img/latencyattr_cutin.gif", fps=10):
    from matplotlib.animation import FuncAnimation, PillowWriter
    names = list(MODELS)
    d = {nm: scripted(MODELS[nm][0]) for nm in names}
    n = len(d[names[0]])
    frames = list(range(n)) + [n - 1] * 12
    fig = plt.figure(figsize=(7.2, 3.6), facecolor=BG)
    ax = fig.add_axes([0.03, 0.14, 0.94, 0.62])
    fig.text(0.03, 0.92, "A cut-in under a 2.0 s perception delay", color=FG, fontsize=12, weight="bold")
    fig.text(0.03, 0.83, f"At t = {CUT_T:.0f} s a car (red) cuts in {CUT_GAP:.0f} m ahead at {CUT_V:.0f} m/s. Dashed box = where"
             " the AV believes its leader is.\nLabel = the AV's avlab.latencyattr state. Same IDM policy, no envelope. Real time.",
             color=MUTED, fontsize=7.5)

    def draw(k):
        ax.cla(); ax.set_facecolor(BG); ax.set_yticks([]); ax.set_ylim(-0.9, 3.0)
        x0 = d[names[0]][k]["x"] - 8
        ax.set_xlim(x0, x0 + 62)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("position (m)", color=MUTED, fontsize=8, labelpad=1)
        for row, nm in enumerate(reversed(names)):
            r, c, y = d[nm][k], MODELS[nm][1], row * 1.6
            ax.plot([x0, x0 + 62], [y - 0.45, y - 0.45], color=GRID, lw=0.8)
            ax.add_patch(plt.Rectangle((r["lead"] - L, y - 0.3), L, 0.6, color="#8b949e", alpha=0.7))
            if r["cut"] is not None:
                ax.add_patch(plt.Rectangle((r["cut"] - L, y - 0.3), L, 0.6, color="#D55E00"))
            ax.add_patch(plt.Rectangle((r["x"] - L, y - 0.3), L, 0.6, color=c))
            ax.text(x0 + 1, y + 0.42, nm, color=c, fontsize=8, weight="bold")
            if r["hit"] is None and MODELS[nm][0][0] > 0 and r["shown"]:
                ax.add_patch(plt.Rectangle((r["x"] + r["seen"], y - 0.3), L, 0.6, fill=False, ec=c, ls="--", lw=1.2))
            if r["hit"] is not None:
                ax.text(r["x"] + 1, y - 0.75, f"CONTACT at {r['hit'][1] * 3.6:.0f} km/h", color="#D55E00", fontsize=8.5,
                        weight="bold")
            else:
                ax.text(r["x"] - L, y - 0.75, SLAB[r["state"]], color=SCOL[r["state"]], fontsize=7.5, weight="bold")
            ax.text(x0 + 61, y + 0.42, f"gap {max(r['gap'], 0):5.1f} m  v {r['v']:4.1f} m/s", color=FG, fontsize=7.5, ha="right")
        ax.text(x0 + 31, 2.65, f"t = {d[names[0]][k]['t']:4.1f} s", color=FG, fontsize=9, ha="center", weight="bold")
        for s in ax.spines.values():
            s.set_color(GRID)
    FuncAnimation(fig, lambda i: draw(frames[i]), frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72,
                                                                           savefig_kwargs={"facecolor": BG})
    return {nm: d[nm][-1]["hit"] for nm in names}


def bench_fig(src="results/latencyattr/attr.json", path="docs/img/latencyattr.png"):
    res = json.load(open(src))
    order = ["ok", "new", "over", "other"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.9), facecolor=BG, gridspec_kw=dict(width_ratios=[1.4, 1]))
    ax, bx = axs
    for i, c in enumerate(("cv1.4", "cv2.0")):
        if c not in res["cells"]:
            continue
        ps = res["cells"][c]["per_state"]
        for j, st in enumerate(order):
            r = ps[st]
            x = j + (i - 0.5) * 0.36
            y = max(r["rate_per_h"], 1e-3)
            ax.bar(x, y, width=0.32, color=SCOL[st], alpha=0.55 if c == "cv1.4" else 1.0)
            ax.plot([x, x], [max(r["ci"][0], 1e-3), r["ci"][1]], color=FG, lw=1.2)
            ax.text(x, r["ci"][1] * 1.25, f"{r['crashes']}", color=SCOL[st], fontsize=7.5, ha="center")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 200)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, color=MUTED)
    ax.set_ylabel("AV crashes per AV-hour in the state (95 %)", color=MUTED, fontsize=8.5)
    ax.text(-0.45, 70, "faint = 1.4 s, solid = 2.0 s delay;\nnumbers = crashes", color=MUTED, fontsize=7.5, va="top")
    c = "cv2.0" if "cv2.0" in res["cells"] else "cv1.4"
    ps = res["cells"][c]["per_state"]
    for j, (lab, key) in enumerate((("AV time", "time_share"), ("AV crashes", "crash_share"))):
        left = 0.0
        for st in ["none"] + order:
            w = ps[st][key]
            bx.barh(j, w, left=left, color=SCOL[st], height=0.55)
            if w > 0.07:
                bx.text(left + w / 2, j, f"{st}\n{w:.0%}", color=BG, fontsize=7, ha="center", va="center", weight="bold")
            left += w
    bx.set_yticks([0, 1]); bx.set_yticklabels(["AV time", "AV crashes"], color=MUTED, fontsize=8)
    bx.set_xlim(0, 1); bx.set_xticks([0, 0.5, 1]); bx.set_xticklabels(["0", "50 %", "100 %"])
    bx.set_title(f"share by state, {c[2:]} s delay", color=MUTED, fontsize=8.5)
    for a in axs:
        a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=8)
        for s in a.spines.values():
            s.set_color(GRID)
    t = res["tests"].get("H1", {})
    fig.text(0.02, 0.95, "Where delayed-perception crashes come from (constant-velocity prediction)", color=FG, fontsize=11,
             weight="bold")
    if t:
        fig.text(0.02, 0.895, f"Oakland, half AVs, IDM, envelope off, 16 copies x 600 s per delay.  H1 new / ok at 2.0 s: RR "
                 f"{t['rr']:.1f} ({t['lo']:.1f}-{t['hi']:.1f}), p {t['p']:.1g}", color=MUTED, fontsize=7.2)
    fig.subplots_adjust(left=0.1, right=0.96, top=0.83, bottom=0.1, wspace=0.3)
    fig.savefig(path, dpi=110, facecolor=BG)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_fig()
    else:
        print(cutin_gif())
