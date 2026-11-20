"""Figures and GIF for avlab.teb / avlab.tebbench.

    python -m avlab.teb_plot          # docs/img/teb_band.gif: the band re-planned every step, TEB vs IDM on one scripted lane
    python -m avlab.teb_plot --bench  # docs/img/teb_bench.png from results/teb/teb.json
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avlab.patchyice_plot import BG, FG, GRID, MUTED, shrink  # noqa: E402

TEBC, IDMC, LEADC, REDC, RULEC = "#56B4E9", "#E69F00", "#8b949e", "#D55E00", "#CC79A7"
DT = 0.2


def _obs(v, v0, lead=None, stop=None):
    o = np.zeros(28, np.float32)
    o[0], o[3], o[16] = v / 15, v0 / 15, 1.0
    if lead is not None:
        o[8], o[9], o[10] = 1, min(lead[0], 150) / 50, lead[1] / 15
    if stop is not None:
        o[6], o[7] = 1, stop / 100
    return o


def scripted(T_total=40.0):
    """One lane: a leader that brakes to a stop and leaves again, then a red light at 260 m that turns green at t = 30 s.
    Returns per-step dicts for TEB and IDM (position, speed, acceleration, TEB band)."""
    import jax
    import jax.numpy as jnp
    from avlab import teb as T
    plan = jax.jit(jax.vmap(lambda f: T.plan_one(f)[:3]))
    idm = jax.jit(T.idm_acc)
    n = int(T_total / DT)
    lead_x, lead_v = 45.0, 10.0
    lead = []
    for k in range(n):
        t = k * DT
        a = -2.5 if 6 <= t < 10 else (1.5 if 12 <= t < 22 else 0.0)
        lead_v = float(np.clip(lead_v + a * DT, 0, 11)); lead_x += lead_v * DT
        lead.append((lead_x, lead_v))
    out = {}
    for name in ("teb", "idm"):
        x, v, rows = 0.0, 12.0, []
        for k in range(n):
            t = k * DT
            lx, lv = lead[k]
            red = 260.0 if t < 30 else None
            # the nearest obstacle: the leader, unless it has passed the line while red (then the line)
            stop = (red - x) if (red is not None and red > x and lx > red) else None
            o = jnp.asarray(_obs(v, 13.0, lead=(lx - x - 4.5, lv) if stop is None else None, stop=stop))[None]
            f = T.features(o)
            if name == "teb":
                a, vb, dTb = (np.asarray(z)[0] for z in plan(f))
                band = (t + np.concatenate([[0], np.cumsum(dTb)]), x + np.concatenate([[0], np.cumsum(vb * dTb)]))
            else:
                a, band = float(idm(f["v"], f)[0]), None
            a = float(np.clip(a, -8, 3))
            rows.append(dict(t=t, x=x, v=v, a=a, band=band, lead=(lx, lv), red=red))
            v = max(v + a * DT, 0.0); x += v * DT
        out[name] = rows
    return out


def band_gif(path="docs/img/teb_band.gif", fps=10, every=2):
    from matplotlib.animation import FuncAnimation, PillowWriter
    sim = scripted()
    n = len(sim["teb"])
    t_all = np.array([r["t"] for r in sim["teb"]])
    fig = plt.figure(figsize=(9.0, 5.0), facecolor=BG)
    ax = fig.add_axes([0.07, 0.12, 0.52, 0.70]); ax2 = fig.add_axes([0.67, 0.55, 0.31, 0.27]); ax3 = fig.add_axes([0.67, 0.12, 0.31, 0.27])
    for a in (ax, ax2, ax3):
        a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=7.5)
        for s in a.spines.values():
            s.set_color(GRID)
    lead_x = np.array([r["lead"][0] for r in sim["teb"]])
    frames = list(range(0, n, every))

    def draw(fi):
        k = frames[fi]
        ax.clear(); ax2.clear(); ax3.clear()
        for a in (ax, ax2, ax3):
            a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=7.5)
            for s in a.spines.values():
                s.set_color(GRID)
        ax.plot(t_all, lead_x - 4.5, color=LEADC, lw=2)
        ax.text(t_all[-1], lead_x[-1] - 4.5, " leader's tail", color=LEADC, fontsize=7.5, ha="right", va="bottom")
        ax.plot([0, 30], [260, 260], color=REDC, lw=3)
        ax.text(0.5, 263, "red light (green at 30 s)", color=REDC, fontsize=7.5)
        for name, c in (("idm", IDMC), ("teb", TEBC)):
            xs = [r["x"] for r in sim[name][:k + 1]]
            ax.plot(t_all[:k + 1], xs, color=c, lw=2)
        tb, xb = sim["teb"][k]["band"]
        ax.plot(tb, xb, "-", color=TEBC, lw=1, alpha=0.7)
        ax.scatter(tb, xb, s=26, color=BG, edgecolors=TEBC, lw=1.4, zorder=5)
        ax.text(tb[-1] + 0.3, xb[-1], "planned band\n(poses + time gaps)", color=TEBC, fontsize=7.5, va="center")
        ax.set_xlim(0, t_all[-1]); ax.set_ylim(0, max(lead_x.max(), 330))
        ax.set_xlabel("time (s)", color=FG, fontsize=8.5); ax.set_ylabel("position along the lane (m)", color=FG, fontsize=8.5)
        for name, c, lab in (("teb", TEBC, "TEB"), ("idm", IDMC, "IDM")):
            vv = [r["v"] for r in sim[name][:k + 1]]
            aa = [r["a"] for r in sim[name][:k + 1]]
            ax2.plot(t_all[:k + 1], vv, color=c, lw=1.8); ax3.plot(t_all[:k + 1], aa, color=c, lw=1.4)
            ax2.text(t_all[k], vv[-1], f" {lab}", color=c, fontsize=7.5, va="bottom" if name == "teb" else "top")
        ax2.set_xlim(0, t_all[-1]); ax2.set_ylim(0, 14); ax2.set_title("speed (m/s)", color=FG, fontsize=8.5)
        ax3.set_xlim(0, t_all[-1]); ax3.set_ylim(-5, 2.5); ax3.set_title("acceleration (m/s²)", color=FG, fontsize=8.5)
        ax3.axhline(0, color=GRID, lw=0.8)
        return []

    fig.text(0.01, 0.95, "Timed Elastic Band: a planned band in space-time, re-optimised every 0.2 s", color=FG, fontsize=12, weight="bold")
    fig.text(0.01, 0.905, "blue: TEB AV and its band (8 poses, time gaps optimised by Levenberg-Marquardt)   orange: IDM on the same lane   "
             "grey: leader", color=MUTED, fontsize=7.5)
    fig.text(0.01, 0.015, "scripted single lane, not the city: the leader brakes to a stop at 6-10 s, leaves at 12 s and clears the light just before it turns green; avlab/teb.py",
             color=MUTED, fontsize=7)
    FuncAnimation(fig, draw, frames=len(frames)).save(path, writer=PillowWriter(fps=fps), dpi=72, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    shrink(path, fps)
    print("wrote", path)


def bench_png(out, path="docs/img/teb_bench.png"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.8), facecolor=BG, gridspec_kw={"width_ratios": [1, 1.5]})
    fig.subplots_adjust(0.13, 0.2, 0.98, 0.8, wspace=0.32)
    for a in (a1, a2):
        a.set_facecolor(BG); a.tick_params(colors=MUTED, labelsize=8)
        for s in a.spines.values():
            s.set_color(GRID)
    e1 = out["E1"]
    rows = [("AV speed\n(km/h)", "av_speed_kmh", 1), ("AV crashes per\n1000 AV-h", None, 1000), ("hard brakes\nper AV-h", "hard", 1),
            ("jerk RMS\n(m/s³)", "av_jerk_rms", 1)]
    vals = []
    for lab, k, sc in rows:
        if k is None:
            vals.append([sc * e1[p]["av_crash"] / e1[p]["av_hours"] for p in ("idm", "teb")])
        elif k == "hard":
            vals.append([e1[p]["av_hard"] / e1[p]["av_hours"] for p in ("idm", "teb")])
        else:
            vals.append([e1[p][k] for p in ("idm", "teb")])
    rel = [100 * (b / a - 1) for a, b in vals]
    y = np.arange(len(rows))[::-1]
    better = [rel[0] > 0, rel[1] < 0, rel[2] < 0, rel[3] < 0]  # faster is better; fewer crashes, hard brakes, less jerk
    cols = ["#009E73" if b else "#D55E00" for b in better]
    a1.barh(y, rel, color=cols, height=0.55)
    a1.axvline(0, color=FG, lw=0.8)
    for yi, r, (a, b) in zip(y, rel, vals):
        a1.text(r + (2 if r >= 0 else -2), yi, f"{r:+.0f}%\n{a:.1f} → {b:.1f}", color=FG, fontsize=7.5, va="center", ha="left" if r >= 0 else "right")
    a1.set_yticks(y, [r[0] for r in rows], color=FG, fontsize=8)
    lim = max(abs(min(rel)), abs(max(rel))) * 1.9 + 5
    a1.set_xlim(-lim, lim)
    a1.text(-0.95 * lim, y[1] - 0.45, f"{int(e1['teb']['av_crash'])} vs {int(e1['idm']['av_crash'])} crashes: not significant", color=MUTED, fontsize=7)
    a1.set_title("city run, 24 paired copies, with the envelope:\nTEB vs IDM (green better, red worse)", color=FG, fontsize=9)
    if "E2" in out:
        sc = list(out["E2"])
        x = np.arange(len(sc))
        for j, (p, c, lab) in enumerate((("rule", RULEC, "rule-based AV"), ("idm", IDMC, "IDM"), ("teb", TEBC, "TEB"))):
            v = [out["E2"][n][p]["av_score"] for n in sc]
            a2.bar(x + (j - 1) * 0.26, v, width=0.26, color=c)
            if p == "teb":
                for xi, vi in zip(x, v):
                    a2.text(xi + 0.26, vi + 1.5, f"{vi:.1f}", color=c, fontsize=7, ha="center")
            a2.text(len(sc) - 0.45, 92 - 7 * j, lab, color=c, fontsize=8, ha="right")
        a2.set_xticks(x, [n.replace("_", "\n") for n in sc], color=FG, fontsize=7.5)
        a2.set_ylim(0, 100); a2.set_ylabel("AVBench av_world score", color=FG, fontsize=8.5)
        a2.set_title("AVBench score (safety scored without the envelope),\n8 copies per scenario, descriptive", color=FG, fontsize=9)
    h = out["H1_speed_diff_kmh"]; c = out["H2_crash_rr"]
    fig.text(0.01, 0.94, "Does a Timed Elastic Band plan better than IDM?", color=FG, fontsize=12.5, weight="bold")
    fig.text(0.01, 0.885, f"H1 AV speed {h['mean']:+.2f} km/h (95 % {h['lo']:+.2f} to {h['hi']:+.2f}, p {h['p']:.2g});  "
             f"H2 AV crash ratio {c['rr']:.2f} (95 % {c['lo']:.2f}-{c['hi']:.2f})", color=MUTED, fontsize=8.5)
    fig.text(0.01, 0.015, "half-AV Oakland, friction 1.0, both policies in the same slot under the usual envelope; AVBench: the "
             "leaderboard's warm states and keys", color=MUTED, fontsize=7)
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench_png(json.load(open("results/teb/teb.json")))
    else:
        band_gif()
