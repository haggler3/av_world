"""Figure and GIF for avlab.brakedemand (dark theme, Okabe-Ito palette).

python -m avlab.brakedemand_plot    # results/brakedemand/brake.json -> docs/img/brakedemand.png, docs/img/brakedemand.gif
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avlab.knee_plot import BG, BLUE, FG, GR, MUTED, ORANGE, PINK, GREEN, _ax  # noqa: E402

COLS = {1.0: BLUE, 0.85: GREEN, 0.7: "#f0e442", 0.6: ORANGE, 0.5: "#d55e00", 0.4: PINK}


def _hist(ax, res, mus, lw=1.6, alpha=1.0):
    bins = np.asarray(res["bins"])
    for mu in mus:
        c = res["cells"][f"{mu:g}"]
        h = np.asarray(c["hist"]) / c["moving_steps"]
        x = bins[:-1] + 0.125
        ok = h > 0
        col = COLS.get(mu, FG)
        ax.step(x[ok], h[ok], where="mid", color=col, lw=lw, alpha=alpha)
        ax.axvline(9 * mu, color=col, lw=0.8, ls=":", alpha=alpha)
        top = h[ok][-1]
        ax.text(9 * mu, top * 2.2, f"cap {9 * mu:.1f}\n(mu {mu:g})", color=col, fontsize=7, ha="center", va="bottom", alpha=alpha)
    ax.set_yscale("log"); ax.set_xlim(0, 10); ax.set_ylim(3e-6, 0.5)
    ax.set_xlabel("braking of a moving human car (m/s$^2$, per 0.1 s step)", color=FG, fontsize=9)
    ax.set_ylabel("share of moving car-steps (log)", color=FG, fontsize=9)


def plot(res, path="docs/img/brakedemand.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = res["summary"]
    rows = S["rows"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.7), facecolor=BG)
    for ax in axs:
        _ax(ax)
    _hist(axs[0], res, [1.0, 0.7, 0.4])
    axs[0].set_title("(a) braking at three frictions: mass piles at the cap 9 mu", color=FG, fontsize=10, loc="left")
    # (b) saturated share vs friction, with the dry-road prediction, and crash rate
    ax = axs[1]
    mu = np.array([r["mu"] for r in rows])
    sat = np.array([r["sat_share"] for r in rows]) * 100
    lo = np.array([r["sat_share_ci"][0] for r in rows]) * 100
    hi = np.array([r["sat_share_ci"][1] for r in rows]) * 100
    pred = np.array([r["dry_exceed_cap"] for r in rows]) * 100
    ax.errorbar(mu, sat, yerr=[sat - lo, hi - sat], fmt="o-", color=ORANGE, lw=1.8, ms=5, capsize=2)
    ax.plot(mu, pred, "s--", color=MUTED, lw=1.2, ms=4)
    ax.text(mu[0], 0.75, "observed: braking\nat the cap (95% CI)", color=ORANGE, fontsize=7.5, ha="left", va="top")
    ax.text(mu[-2], pred[-2] * 0.55, "dry-road braking\nthat would exceed it", color=MUTED, fontsize=7.5, ha="center", va="top")
    ax2 = ax.twinx(); _ax(ax2)
    rate = np.array([r["rate"] for r in rows])
    ax2.plot(mu, rate, "D-", color=BLUE, lw=1.5, ms=4)
    ax2.text(mu[2], rate[2] * 1.15, "crashes / 1000 veh-h ", color=BLUE, fontsize=7.5, ha="right", va="bottom")
    ax2.set_yscale("log"); ax2.set_ylabel("crashes per 1000 vehicle-h (log)", color=BLUE, fontsize=9)
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_yscale("log"); ax.set_xlim(1.05, 0.35)
    for a_ in (ax, ax2):
        a_.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}")); a_.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(0.1, 1.3)
    ax2.set_yticks([3, 10, 30, 50])
    ax.axvline(0.7, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("road friction mu (slicker to the right)", color=FG, fontsize=9)
    ax.set_ylabel("% of moving car-steps braking at the cap", color=ORANGE, fontsize=9)
    ax.set_title("(b) when does braking hit the road's limit?", color=FG, fontsize=10, loc="left")
    # (c) enrichment among crash-involved cars
    ax = axs[2]
    pe = S["pooled_enrichment"]
    labs, xs = [], []
    for i, (k, lab) in enumerate((("mu>=0.7", "mu 1.0-0.7"), ("mu<=0.6", "mu 0.6-0.4"))):
        p = pe[k]
        ax.bar(i - 0.18, 100 * p["share"], 0.34, color=ORANGE)
        ax.bar(i + 0.18, 100 * p["base"], 0.34, color=MUTED)
        ax.text(i - 0.18, 100 * p["share"] + 1, f"{100 * p['share']:.0f}%\n({p['involved_recent_sat']:.0f}/{p['involved']:.0f})", color=ORANGE,
                fontsize=7.5, ha="center", va="bottom")
        ax.text(i + 0.18, 100 * p["base"] + 1, f"{100 * p['base']:.1f}%", color=MUTED, fontsize=7.5, ha="center", va="bottom")
        labs.append(lab); xs.append(i)
    ax.set_xticks(xs); ax.set_xticklabels(labs, color=FG)
    ymax = max(100 * pe[k]["share"] for k in ("mu>=0.7", "mu<=0.6"))
    ax.set_ylim(0, max(10, ymax * 1.45))
    ax.text(0.02, 0.97, "orange: cars in a crash that braked at the cap in the 1 s before\ngrey: any moving car, same window",
            color=FG, fontsize=7.5, transform=ax.transAxes, va="top")
    ax.set_ylabel("% braking at the cap within 1 s", color=FG, fontsize=9)
    ax.set_title("(c) were crashing cars at the limit?", color=FG, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def gif(res, path="docs/img/brakedemand.gif", fps=10):
    """The cap sweeps left as the road gets slicker; each friction's braking histogram is drawn, older ones fade."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    mus = [r["mu"] for r in res["summary"]["rows"]]
    rows = {r["mu"]: r for r in res["summary"]["rows"]}
    frames = []
    for i, mu in enumerate(mus):
        for j in range(12 if i < len(mus) - 1 else 25):
            fig, ax = plt.subplots(figsize=(6.4, 4.0), facecolor=BG, dpi=80)
            _ax(ax)
            if i:
                _hist(ax, res, mus[:i], lw=1.0, alpha=0.3)
            _hist(ax, res, [mu], lw=2.2)
            r = rows[mu]
            ax.set_title(f"Road friction {mu:g}: braking capped at {9 * mu:.1f} m/s$^2$", color=FG, fontsize=11, loc="left")
            ax.text(0.98, 0.97, f"at the cap: {100 * r['sat_share']:.2f}% of car-steps\ncrashes: {r['rate']:.0f} per 1000 vehicle-h",
                    color=FG, fontsize=9, ha="right", va="top", transform=ax.transAxes)
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).convert("P", palette=Image.ADAPTIVE, colors=64))
            plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)


if __name__ == "__main__":
    res = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/brakedemand/brake.json"))
    plot(res)
    gif(res)
    print("docs/img/brakedemand.png", os.path.getsize("docs/img/brakedemand.png"), "docs/img/brakedemand.gif", os.path.getsize("docs/img/brakedemand.gif"))
