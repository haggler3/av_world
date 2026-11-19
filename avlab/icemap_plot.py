"""Figure and GIF for avlab.icemap, drawn from the saved crash log (no simulation).

    python -m avlab.icemap_plot   # docs/img/icemap.png (rates by street type and cell), docs/img/icemap.gif (crash map over time)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from avlab.patchyice_plot import BG, FG, GRID, MUTED, poisson_ci, shrink  # noqa: E402

# Okabe-Ito
ICE, DRY, ROADDRY = "#56B4E9", "#E69F00", "#3a414a"
CELL_LABEL = {"D": "dry city\n(friction 1.0)", "P": "patchy city\n(0.4 on half)", "I": "icy city\n(friction 0.4)"}


def plot(out, path="docs/img/icemap.png"):
    cells = ["D", "P", "I"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=BG)
    ax.set_facecolor(BG)
    for j, c in enumerate(cells):
        cc = out["cells"][c]
        for dx, kind, col, lab in ((-0.14, "masked", ICE, "icy-mask links"), (0.14, "unmasked", DRY, "other links")):
            k, t = cc[f"k_{kind}"], cc[f"t_{kind}_h"]
            r = 1000 * k / t
            lo, hi = poisson_ci(k, t)
            left = dx < 0
            if k == 0:  # no crashes: draw only the upper bound
                ax.plot([j + dx] * 2, [0.6, hi], color=col, lw=2.2, ls=":")
                ax.plot(j + dx, hi, "v", color=col, ms=8, mfc=BG)
                ax.text(j + dx + (-0.05 if left else 0.05), hi, f"0 crashes\n(below {hi:.1f})", color=col, fontsize=8, va="center",
                        ha="right" if left else "left")
                continue
            ax.plot([j + dx] * 2, [lo, hi], color=col, lw=2.2)
            ax.plot(j + dx, r, "o", color=col, ms=8)
            ax.text(j + dx + (-0.05 if left else 0.05), r, f"{r:.1f}\n({k} crashes)", color=col, fontsize=8, va="center",
                    ha="right" if left else "left")
    ax.set_yscale("log")
    ax.set_xticks(range(3), [CELL_LABEL[c] for c in cells], color=FG, fontsize=9)
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(0.6, 150)
    ax.set_ylabel("crashes per 1000 vehicle-hours (95 % interval)", color=FG, fontsize=9)
    ax.tick_params(colors=MUTED)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6, which="major")
    h = out["H1_P_masked_vs_unmasked"]
    fig.text(0.01, 0.95, "Patchy ice: crashes happen where the ice is", color=FG, fontsize=12, weight="bold")
    fig.text(0.01, 0.905, f"blue: links in the ice mask   orange: the other links   patchy city, icy / dry rate "
             f"{h['rr']:.1f} (95 % {h['lo']:.1f} to {h['hi']:.1f})", color=MUTED, fontsize=8.5)
    fig.text(0.01, 0.015, "humans-only Oakland, 32 copies per city, 600 s each, random stream 11; same links in every city; "
             "from the per-crash log (avlab.crashlog)", color=MUTED, fontsize=7)
    fig.subplots_adjust(0.1, 0.16, 0.98, 0.86)
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)
    print("wrote", path)


def gif(res, path="docs/img/icemap.gif", every_s=10.0, fps=10):
    """Crash events of all 32 patchy-city copies on one map, appearing as they happen; right: running crash rate."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from xnet import network as NW
    net, geo, _ = NW.build(res["area"])
    tags = np.array(res["tags"])
    mask = np.asarray(res["mask"], bool)
    r = {k: np.asarray(v) for k, v in res["rows"].items()}
    ev = {k: np.asarray(v) for k, v in res["events"].items()}
    # event positions: the first row of each event (same copy and time, nearest in order)
    first = []
    for c, t in zip(ev["copy"], ev["t"]):
        first.append(np.where((r["copy"] == c) & (r["t"] == t))[0][0])
    first = np.array(first, int)
    sel = tags[ev["copy"].astype(int)] == "P"
    ex, ey, et, ep, ev_v = r["x"][first][sel], r["y"][first][sel], ev["t"][sel], ev["patched"][sel] > 0, ev["max_v"][sel]
    t0 = res["warm_s"]
    et = et - t0
    tm = np.asarray(res["expo_masked_s"])[tags == "P"].sum() / 3600
    tu = np.asarray(res["expo_unmasked_s"])[tags == "P"].sum() / 3600
    T = res["seconds"]
    segs = [np.asarray(p) for p in geo.link_pts]
    fig = plt.figure(figsize=(9.0, 5.0), facecolor=BG)
    ax = fig.add_axes([0.0, 0.06, 0.6, 0.8]); ax2 = fig.add_axes([0.68, 0.16, 0.29, 0.66])
    ax.set_facecolor(BG); ax2.set_facecolor(BG)
    ax.add_collection(LineCollection(segs, colors=[ICE if mask[i] else ROADDRY for i in range(len(segs))], linewidths=1.4))
    allp = np.concatenate(segs)
    ax.set_xlim(allp[:, 0].min(), allp[:, 0].max()); ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax.set_aspect("equal"); ax.axis("off")
    sc_i = ax.scatter([], [], s=[], c=ICE, edgecolors=FG, lw=0.5, zorder=5)
    sc_d = ax.scatter([], [], s=[], c=DRY, edgecolors=FG, lw=0.5, zorder=6, marker="D")
    hud = ax.text(0.02, 0.02, "", transform=ax.transAxes, color=FG, fontsize=9)
    ax2.set_xlim(0, T / 60); ymax = 1.15 * max(1000 * ep.sum() / tm, 1000 * (~ep).sum() / tu, 1)
    ax2.set_ylim(0, ymax)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for s in ax2.spines.values():
        s.set_color(GRID)
    ax2.set_xlabel("minutes", color=MUTED, fontsize=8)
    ax2.set_title("crashes per 1000 veh-h\nso far", color=FG, fontsize=9)
    li, = ax2.plot([], [], color=ICE, lw=2); ld, = ax2.plot([], [], color=DRY, lw=2)
    ti = ax2.text(0, 0, "", color=ICE, fontsize=8, ha="right", va="bottom"); td = ax2.text(0, 0, "", color=DRY, fontsize=8, ha="right", va="bottom")
    fig.text(0.01, 0.95, "Where patchy-ice crashes happen (32 copies of one city, overlaid)", color=FG, fontsize=12, weight="bold")
    fig.text(0.01, 0.905, "blue streets: ice (friction 0.4)   grey: dry (1.0)   blue dot: crash on ice   orange diamond: crash on a dry "
             "street   size: impact speed", color=MUTED, fontsize=7.5)
    fig.text(0.01, 0.015, "humans-only Oakland, random stream 11; from the per-crash log (avlab.crashlog)", color=MUTED, fontsize=7)
    times = np.arange(every_s, T + 1e-6, every_s)

    def draw(k):
        t = times[k]
        on = et <= t
        sz = 10 + 4 * 3.6 * ev_v
        sc_i.set_offsets(np.c_[ex[on & ep], ey[on & ep]] if (on & ep).any() else np.empty((0, 2))); sc_i.set_sizes(sz[on & ep])
        sc_d.set_offsets(np.c_[ex[on & ~ep], ey[on & ~ep]] if (on & ~ep).any() else np.empty((0, 2))); sc_d.set_sizes(sz[on & ~ep])
        hud.set_text(f"t = {t / 60:4.1f} min   crashes on ice {int((on & ep).sum())}, on dry streets {int((on & ~ep).sum())}")
        xs = times[:k + 1]
        ri = [1000 * (ep & (et <= x)).sum() / (tm * x / T) for x in xs]
        rd = [1000 * (~ep & (et <= x)).sum() / (tu * x / T) for x in xs]
        li.set_data(xs / 60, ri); ld.set_data(xs / 60, rd)
        ti.set_position((xs[-1] / 60, ri[-1])); ti.set_text(f"ice {ri[-1]:.0f}")
        td.set_position((xs[-1] / 60, rd[-1])); td.set_text(f"dry {rd[-1]:.0f}")
        return []

    FuncAnimation(fig, draw, frames=len(times)).save(path, writer=PillowWriter(fps=fps), dpi=72, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    shrink(path, fps)
    print("wrote", path)


if __name__ == "__main__":
    import sys
    plot(json.load(open("results/icemap/icemap.json")))
    if "--no-gif" not in sys.argv:
        gif(json.load(open("results/icemap/run.json")))
