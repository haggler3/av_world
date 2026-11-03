"""Figures and the GIF for avlab.twins, from the saved results (and a short re-run of one twin pair for the GIF)."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

BG, INK, INK2, GRID = "#0d1117", "#e6edf3", "#8b949e", "#21262d"
PAL = {"1 car + 1 mm/s": "#7ee787", "1 car + 3 m/s": "#d2a8ff", "every car + 1 mm/s": "#58a6ff", "new random stream": "#f0883e"}


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID)
    for sp in ax.spines.values():
        sp.set_color("#30363d")


def figure(res, out="docs/img/twins.png"):
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=BG)
    style(ax)
    t = np.arange(len(res["nudges"]["1 car + 1 mm/s"]["displaced"][0])) * res["step_s"] / 60
    for name, r in res["nudges"].items():
        d = 100 * np.array(r["displaced"])
        ax.fill_between(t, d.min(0), d.max(0), color=PAL[name], alpha=0.15, lw=0)
        ax.plot(t, d.mean(0), color=PAL[name], lw=2)
        dy = {"1 car + 1 mm/s": 3.5, "every car + 1 mm/s": -2.5, "1 car + 3 m/s": 9}.get(name, 0)
        ax.text(t[-1] + 0.1, d.mean(0)[-1] + dy, name, color=PAL[name], fontsize=9, va="center")
    ax.set_xlim(0, 13)
    ax.set_xlabel("minutes after the nudge", color=INK2, fontsize=9)
    ax.set_ylabel("% of vehicles displaced by more than 1 m", color=INK2, fontsize=9)
    ax.set_title(f"Millimetre nudges leave no trace; one 3 m/s kick grows in a few twins; a new random stream decorrelates the city\n"
                 f"{res['copies']} twin pairs of Oakland per line (mean; band = min to max); the 1 mm/s lines lie on 0",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


def bench_figure(b, out="docs/img/twins_bench.png"):
    conds = list(next(iter(b.values())))
    cols = {"as published": INK, "every car + 1 mm/s": PAL["every car + 1 mm/s"], "every car + 1 m/s": PAL["1 car + 3 m/s"], "new random stream": PAL["new random stream"]}
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=BG)
    style(ax)
    sc = list(b)
    for i, n in enumerate(sc):
        for j, c in enumerate(conds):
            ax.scatter(i + (j - 1.5) * 0.12, b[n][c], s=60, color=cols[c], zorder=3, label=c if i == 0 else None)
    ax.set_xticks(range(len(sc)), [s.replace("_", " ") for s in sc], color=INK2)
    ax.set_ylabel("AVBench av_world score, rule-based AV", color=INK2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(facecolor=BG, edgecolor="#30363d", labelcolor=INK, fontsize=8, loc="lower left", ncol=2)
    ax.set_title("Same warm city, same code: a rounding-sized nudge leaves the score alone; a new random stream moves it",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=BG)
    plt.close(fig)


def gif(out="docs/img/twins.gif", seconds=300.0, step_s=5.0):
    """One twin pair, two nudges side by side: vehicles that are where they would have been are grey, displaced ones orange."""
    from avlab import twins as T
    from xnet import render as R
    nud = {k: T.NUDGES[k] for k in ("1 car + 3 m/s", "new random stream")}
    res, base, tw, (net, geo, info) = T.run(E=1, seconds=seconds, step_s=step_s, keep_frames=True, nudges=nud)
    mp, tracks = R.Map(net, geo, info), R.Tracks(net, geo)
    from xnet import network as NW
    fig, axs = plt.subplots(1, 2, figsize=(10, 5.6), facecolor=BG)
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.84)
    x0, y0 = NW.to_m(geo.bbox[0], geo.bbox[1])
    x1, y1 = NW.to_m(geo.bbox[2], geo.bbox[3])
    for ax in axs:
        ax.set_facecolor("#141a1f")
        mp.draw(ax, detail=False, labels=False)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.015, 0.95, "Twin cities: the same Oakland twice, one of them nudged", color=INK, fontsize=13, weight="bold")
    fig.text(0.015, 0.905, "orange: vehicle more than 1 m from where its twin is      grey: same place      blue ring: the car given +3 m/s", color=INK2, fontsize=9)
    titles = {"1 car + 3 m/s": "one car +3 m/s", "new random stream": "same city, new random stream"}
    huds = [ax.set_title(titles[k], color=INK, fontsize=10, pad=6) for ax, k in zip(axs, nud)]
    kicked = None
    arts = []
    st0 = base[0]
    kicked = int(np.argmax(np.where(np.asarray(st0.alive[0]), np.asarray(st0.v[0]), -1)))

    def draw(k):
        nonlocal arts
        for a in arts:
            a.remove()
        arts = []
        for ax, name in zip(axs, nud):
            a, b = base[k], tw[name][k]
            al = np.asarray(b.alive[0])
            trk, s, ln = np.asarray(b.trk[0])[al], np.asarray(b.s[0])[al], np.asarray(b.length[0])[al]
            x, y, _ = tracks.pose(trk, s, ln)
            far = ((np.abs(np.asarray(a.s[0] - b.s[0])) > 1.0) | (np.asarray(a.trk[0]) != np.asarray(b.trk[0])) | ~np.asarray(a.alive[0]))[al]
            arts.append(ax.scatter(x[~far], y[~far], s=7, c="#8b949e", zorder=5))
            arts.append(ax.scatter(x[far], y[far], s=11, c="#f0883e", zorder=6))
            if name == "1 car + 3 m/s" and al[kicked]:
                i = int(np.sum(al[:kicked]))
                arts.append(ax.scatter([x[i]], [y[i]], s=90, facecolors="none", edgecolors="#58a6ff", linewidths=1.6, zorder=7))
            d, n = T.displaced(a, b)
            arts.append(ax.text(0.03, 0.03, f"t = {k * step_s:3.0f} s   displaced {100 * d[0]:4.1f} %", transform=ax.transAxes, color=INK, fontsize=10,
                                bbox=dict(facecolor=BG, edgecolor="none", alpha=0.8)))
        return arts

    FuncAnimation(fig, draw, frames=len(base)).save(out, writer=PillowWriter(fps=10), dpi=70, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out)


def main():
    res = json.load(open("results/twins/twins.json"))
    figure(res)
    try:
        bench_figure(json.load(open("results/twins/bench.json")))
    except FileNotFoundError:
        pass
    gif()


if __name__ == "__main__":
    main()
