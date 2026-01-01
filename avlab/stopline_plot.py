"""Figure and GIF for avlab.stopline: python -m avlab.stopline_plot -> docs/img/stopline.png, docs/img/tomography.gif"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avlab.stopline import mu_hat_max, mu_hat_spike  # noqa: E402

BG, FG, GRID = "#0d1117", "#e6edf3", "#30363d"
PAL = {"cap": "#4ea1ff", "ovr": "#ff9f43", "com": "#c77dff", "crash": "#e6edf3", "noise": "#2ecc71"}


def _style(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=FG, labelsize=9)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    ax.grid(color=GRID, lw=0.5)


def figure(res, fn):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["summary"]["rows"]
    mu = np.array([r["mu"] for r in rows])
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=BG)
    for ax in (a, b):
        _style(ax)
    lines = [("sat_per_1e5", "braking at the cap", PAL["cap"]), ("fric_overrun_per_1e5", "stop-line overrun (friction-caused)", PAL["ovr"]),
             ("fric_commit_per_1e5", "blind commitment (friction-caused)", PAL["com"])]
    for k, lab, c in lines:
        y = np.array([r[k] for r in rows])
        m = y > 0
        a.plot(mu[m], y[m], "o-", color=c, lw=2)
        a.annotate(lab, (mu[m][-1], y[m][-1]), xytext=(6, 0), textcoords="offset points", color=c, fontsize=8.5, va="center")
    rate = np.array([r["rate"] for r in rows])
    a.plot(mu, rate * 10, "s--", color=PAL["crash"], lw=1.2)
    a.annotate("crashes per 1000 veh-h (x10)", (mu[-1], rate[-1] * 10), xytext=(6, -12), textcoords="offset points", color=FG, fontsize=8.5)
    a.axvline(0.7, color=GRID, lw=1.5, ls=":")
    a.text(0.705, 0.25, "knee\n0.7", color=FG, fontsize=8)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xticks(mu); a.set_xticklabels([f"{m:g}" for m in mu]); a.minorticks_off()
    a.set_xlim(1.05, 0.22)
    a.set_xlabel("road friction mu (log, slick to the right)"); a.set_ylabel("per 100,000 moving car-steps")
    a.set_title("Each channel grows smoothly; none bends at 0.7", fontsize=10.5)
    pooled = res["summary"]["pooled"]
    groups = [("mu>=0.7", "mu >= 0.7"), ("mu<=0.6", "mu <= 0.6")]
    chans = [("sat", "at the cap", PAL["cap"]), ("overrun", "overrun (any)", PAL["ovr"]), ("fric_commit", "blind commit", PAL["com"])]
    w = 0.26
    for gi, (g, glab) in enumerate(groups):
        for ci, (c, clab, col) in enumerate(chans):
            p = pooled[g][c]
            x = gi + (ci - 1) * w
            b.bar(x, 100 * p["share"], w * 0.9, color=col)
            b.text(x, 100 * p["share"] + 1.5, f"{int(p['k'])}/{int(pooled[g]['involved'])}", ha="center", color=FG, fontsize=8)
    for ci, (c, clab, col) in enumerate(chans):
        b.text(1.55, 62 - 7 * ci, clab, color=col, fontsize=9)
    b.set_xticks([0, 1]); b.set_xticklabels([g[1] for g in groups])
    b.set_xlim(-0.5, 2.1); b.set_ylim(0, 72)
    b.set_ylabel("% of crash-involved human cars, channel within 2 s")
    b.set_title("Crashes follow the braking cap, not the stop line", fontsize=10.5)
    fig.suptitle(f"Friction's three channels, Oakland, {res['copies']} copies x 6 frictions, fresh stream (seed {res['seed']})", color=FG, fontsize=11)
    fig.tight_layout()
    fig.savefig(fn, dpi=110, facecolor=BG)
    plt.close(fig)


def gif(res, fn, copy=0, sigma=0.3, nframes=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    cells = list(res["cells"].values())
    lo, w = 2.0, 0.05
    centres = lo + (np.arange(160) + 0.5) * w
    rng = np.random.default_rng(7)
    frames = []
    for f in range(1, nframes + 1):
        fig, axs = plt.subplots(2, 3, figsize=(9, 5), facecolor=BG)
        for ax, c in zip(axs.flat, cells):
            _style(ax)
            h = np.asarray(c["tail_hist"])[copy, :f].sum(0)
            x = np.repeat(centres, h) + np.random.default_rng(f).normal(0, sigma, h.sum())
            hn = np.histogram(x, bins=np.append(centres - w / 2, centres[-1] + w / 2))[0]
            ax.bar(centres, np.maximum(hn, 0.3), width=w, color=PAL["noise"], lw=0)
            ax.set_yscale("log"); ax.set_ylim(0.8, 3000); ax.set_xlim(2, 10)
            est = mu_hat_spike(h, sigma=sigma, rng=np.random.default_rng(f))
            ax.axvline(9 * c["mu"], color=FG, lw=1, ls=":")
            if not np.isnan(est):
                ax.axvline(9 * est, color=PAL["ovr"], lw=2)
            ok = (not np.isnan(est)) and abs(est - c["mu"]) <= 0.05
            ax.set_title(f"true mu {c['mu']:g}   estimate {est:.2f} {'ok' if ok else ''}", fontsize=9.5,
                         color=PAL["ovr"] if ok else FG)
            ax.tick_params(labelsize=7)
        for ax in axs[1]:
            ax.set_xlabel("braking (m/s^2), sensor noise 0.3", fontsize=8)
        fig.suptitle(f"Friction tomography: one city's braking after {10 * f} s (orange = estimate, dotted = truth)", color=FG, fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("P", palette=Image.ADAPTIVE, colors=64))
        plt.close(fig)
    frames[0].save(fn, save_all=True, append_images=[x for x in frames for _ in range(3)][1:] + [frames[-1]] * 20, duration=100, loop=0, optimize=True)


if __name__ == "__main__":
    res = json.load(open("results/stopline/stopline.json"))
    figure(res, "docs/img/stopline.png")
    gif(res, "docs/img/tomography.gif")
    print("ok")
