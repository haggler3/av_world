"""Figure and GIF for avlab.twintrace: the kicked car, the vehicles it displaced, and the growth of the displaced set."""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from avlab import twintrace as T
from xnet import render as R

BG, FG, GREY, ORANGE, BLUE = "#0d1117", "#e6edf3", "#8b949e", "#e69f00", "#56b4e9"


def figure(res):
    nd, na = np.array(res["n_displaced_t"]), np.array(res["n_alive_t"])
    t = np.arange(len(nd))
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=BG)
    ax.set_facecolor(BG)
    ax.semilogy(t, np.maximum(nd, 0.8), color=ORANGE, lw=2)
    ax.semilogy(t, na, color=GREY, lw=1.2, ls="--")
    ax.set_ylim(0.7, 2000)
    ax.text(t[-1], na[-1] * 1.12, "vehicles alive in both", color=GREY, ha="right", fontsize=9)
    ax.text(t[-1] * 0.99, max(nd[-1], 1) * 0.3, "vehicles displaced > 1 m", color=ORANGE, ha="right", fontsize=9)
    s = res["summary"]
    ax.axvline(s["first_other_s"], color=BLUE, lw=1, ls=":")
    ax.text(s["first_other_s"] + 6, 0.9, f"first other car displaced: {s['first_other_s']} s", color=BLUE, fontsize=9)
    ax.set_xlabel("seconds after the kick (+3 m/s to the fastest car)", color=FG)
    ax.set_ylabel("vehicles", color=FG)
    ax.set_title(f"One kick, a city apart: doubling time about {s['doubling_time_s']:.0f} s", color=FG, loc="left")
    ax.tick_params(colors=FG)
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    fig.tight_layout()
    fig.savefig("docs/img/twintrace.png", dpi=130, facecolor=BG)
    plt.close(fig)


def gif(res, states, ctx):
    net, geo, info = ctx
    mp = R.Map(net, geo, info)
    tr = R.Tracks(net, geo)
    fig, ax = plt.subplots(figsize=(7.2, 7.4), facecolor=BG)
    fig.subplots_adjust(0.01, 0.06, 0.99, 0.9)
    ax.set_facecolor(R.GROUND)
    mp.draw(ax, detail=False, labels=False)
    from xnet import network as NW
    x0, y0 = NW.to_m(geo.bbox[0], geo.bbox[1])
    x1, y1 = NW.to_m(geo.bbox[2], geo.bbox[3])
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.015, 0.955, "Twin cities: one car kicked by 3 m/s", color=FG, fontsize=12, weight="bold")
    hud = fig.text(0.015, 0.918, "", color=GREY, fontsize=9)
    fig.text(0.015, 0.02, "grey: same place in both cities   orange: displaced by more than 1 m   white ring: the kicked car\n"
             "positions shown are the twin's; one frame every 10 simulated seconds", color=GREY, fontsize=7.5)
    kicked = res["kicked_slot"]
    arts = []

    def draw(k):
        nonlocal arts
        for a_ in arts:
            a_.remove()
        a, b = states[k]
        al = b.alive
        x, y, _ = tr.pose(b.trk[al], b.s[al], b.length[al])
        both = a.alive & b.alive
        far = ((np.abs(a.s - b.s) > 1.0) | (a.trk != b.trk)) & both
        far = far[al]
        arts = [ax.scatter(x[~far], y[~far], s=4, c=GREY, lw=0, zorder=7), ax.scatter(x[far], y[far], s=9, c=ORANGE, lw=0, zorder=8)]
        if b.alive[kicked]:
            xk, yk, _ = tr.pose(b.trk[[kicked]], b.s[[kicked]], b.length[[kicked]])
            arts.append(ax.scatter(xk, yk, s=70, facecolors="none", edgecolors="white", lw=1.4, zorder=9))
        hud.set_text(f"t = {10 * k:3d} s after the kick    displaced {int(far.sum())} of {int(both.sum())} vehicles")
    FuncAnimation(fig, draw, frames=len(states)).save("docs/img/twintrace.gif", writer=PillowWriter(fps=10), dpi=70, savefig_kwargs={"facecolor": BG})
    plt.close(fig)


def main():
    res, rec, states, ctx = T.run(keep_states=True)
    figure(res)
    gif(res, states, ctx)
    print("wrote docs/img/twintrace.png and .gif")
