"""Scaling-law chart (docs/img/scaling_laws.png): world-model error vs training steps and vs simulated data across
architectures, and the driving policy's score vs simulation steps, each with a fitted power law."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402

SURFACE, INK, INK2, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#383835"
COLORS = {"MLP 128x2": "#3987e5", "MLP 256x3": "#d95926", "transformer 64x2": "#199e70", "flow matching, MLP": "#c98500"}


def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=10)
    ax.tick_params(colors=INK2, labelsize=8, which="both")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(color=GRID, lw=0.6, which="both")
    ax.set_axisbelow(True)


def law(ax, data, x, xlabel, title, unit_scale=1.0):
    style(ax, title)
    rows, fits = data["rows"], data["fits"]
    for arch, c in COLORS.items():
        r = sorted([q for q in rows if q["arch"] == arch], key=lambda q: q[x])
        if not r:
            continue
        xs, ys = np.array([q[x] for q in r]) * unit_scale, np.array([q["skill"] for q in r])
        f = fits.get(arch)
        lab = arch + (f"   exponent b = {f[1]:.3f}  (R\u00b2 {f[2]:.2f})" if f else "")
        ax.plot(xs, ys, "o", color=c, ms=6, mec=SURFACE, mew=1.5, zorder=4, label=lab)
        if f:
            xx = np.geomspace(xs.min(), xs.max() * 1.6, 100)
            ax.plot(xx, f[0] * np.power(xx / unit_scale, -f[1]), color=c, lw=1.8, zorder=3)
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([1.0, 1.1, 1.2, 1.4, 1.6, 2.0]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(xlabel, color=INK2, fontsize=8.5)
    ax.set_ylabel("error vs physics model (CRPS skill; 1 = physics)", color=INK2, fontsize=8.5)
    leg = ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="upper right")


def main(out="docs/img/scaling_laws.png"):
    comp = json.load(open("results/scaling/compute.json"))
    dat = json.load(open("results/scaling/data.json"))
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), facecolor=SURFACE)
    law(axs[0], comp, "steps", "training steps (batch 512)", "A  World model: error vs training compute")
    axs[0].text(axs[0].get_xlim()[0] * 1.1, 1.03, "physics model", color=INK2, fontsize=8)
    law(axs[1], dat, "sim_steps", "simulated data: area-copy steps of 0.2 s behind the training set",
        "B  World model: error vs simulated data (1,500 steps each)")
    # RL: score vs simulation steps (from scratch), fitted score = s_inf - a x^-b
    ax = axs[2]
    style(ax, "C  Driving policy: score vs simulation steps")
    L = json.load(open("results/ladder/scores.json"))
    for run, c, name in (("s1", "#3987e5", "trained with AVs only"), ("s5", "#d95926", "trained in the full world")):
        r = sorted([q for q in L if q["run"] == run], key=lambda q: q["sim_steps"])
        xs, ys = np.array([q["sim_steps"] for q in r], float), np.array([q["score"] for q in r])
        f = lambda x, a, b, s: s - a * np.power(x, -b)  # noqa: E731
        p, _ = curve_fit(f, xs, ys, p0=(1.0, 0.5, 0.075), bounds=([0, 0.05, 0.0], [1e4, 3.0, 0.2]), maxfev=20000)
        xx = np.linspace(xs.min() * 0.9, xs.max(), 100)
        ax.plot(xx / 1000, f(xx, *p), color=c, lw=1.8, zorder=3)
        xp = np.linspace(xs.max(), 2.5e5, 100)
        ax.plot(xp / 1000, f(xp, *p), color=c, lw=1.4, ls=(0, (4, 3)), zorder=3)  # projection beyond the measured range
        ax.plot(xs / 1000, ys, "o", color=c, ms=6, mec=SURFACE, mew=1.5, zorder=4,
                label=f"{name}   exponent b = {p[1]:.2f}")
    ax.set_xlabel("simulator steps (thousands)", color=INK2, fontsize=8.5)
    ax.set_ylabel("reward per AV decision in the test world", color=INK2, fontsize=8.5)
    ax.axvspan(63, 250, color=GRID, alpha=0.35, lw=0)
    ax.text(70, 0.034, "dashed: power-law projection\nbeyond the 4 measured checkpoints;\nthe next experiment trains 4x longer\nto test it",
            color=INK2, fontsize=8)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK2, loc="lower right")
    fig.suptitle("Scaling laws: world-model error falls as a power law in compute and in simulated data; the driver's score rises "
                 "as one in simulation steps", color=INK, fontsize=12.5, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.005, "Fits: error = a * x^-b in log-log space (A, B); score = ceiling - a * x^-b (C). World models trained on AV decisions "
             "logged in the Oakland area simulator, scored on held-out vehicles (speed and gap at 1/2/3 s).", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, dpi=120, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
