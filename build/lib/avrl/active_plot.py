"""The active-learning chart (docs/img/active_learning.png) from results/active/."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .ladder_plot import INK, INK2, SURFACE, style  # noqa: E402

MODES = {"uniform": ("uniform scenarios (domain randomisation)", "#9da7b3"),
         "active": ("prioritised level replay, |advantage|", "#3987e5"),
         "focus": ("failure-focused replay", "#f0883e"),
         "critic": ("uniform + 10-update critic warm-up", "#7ee787")}


def main(out="docs/img/active_learning.png"):
    rows = json.load(open("results/active/scores.json"))
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    style(ax, "A  Test-world score vs simulation steps, same start, same budget")
    for m, (name, c) in MODES.items():
        r = sorted([x for x in rows if x["mode"] == m], key=lambda x: x["sim_steps"])
        if not r:
            continue
        ax.plot(np.array([x["sim_steps"] for x in r]) / 1000, [x["score"] for x in r], color=c, lw=2, marker="o", ms=5,
                mec=SURFACE, mew=1.5, label=name, zorder=3)
    ax.set_xlabel("simulator steps (thousands)", color=INK2, fontsize=8.5)
    ax.set_ylabel("reward per AV decision (held-out full-dynamics world)", color=INK2, fontsize=8.5)
    ax.legend(fontsize=8, loc="lower right", frameon=False, labelcolor=INK2)
    ax.annotate("the three scenario-selection rules finish within 2 %:\nwhich scenarios are simulated is not the bottleneck",
                xy=(61, 0.0697), xytext=(13, 0.0668), color=INK, fontsize=8.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    ax = axs[1]
    style(ax, "B  Why: every scenario still teaches about as much")
    act = json.load(open("results/active/active/log.json"))
    foc = json.load(open("results/active/focus/log.json"))
    a, f = np.array(act["plr_score"]), np.array(foc["plr_score"])
    o = np.argsort(-a)
    ax.bar(np.arange(len(a)), a[o] / a.max(), color=MODES["active"][1], width=0.8, label="learning signal (|advantage|)")
    ax.plot(np.arange(len(f)), f[o] / f.max(), color=MODES["focus"][1], lw=0, marker="o", ms=3.5,
            label="AV trouble per hour (failure score)")
    ax.set_title("B  Why: every scenario still teaches about as much\n    54 scenarios (demand x risk x pedestrians x AV share), sorted",
                 color=INK, fontsize=10.5, loc="left", pad=10)
    ax.set_ylabel("relative to the highest scenario", color=INK2, fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_xticks([])
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, labelcolor=INK2)
    ax.text(len(a) - 1, 1.06, f"lowest scenario still {100 * a.min() / a.max():.0f} % of the highest", color=INK, fontsize=8.5, ha="right")
    fig.tight_layout(pad=1.6)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
